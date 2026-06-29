"""Inspection report generator.

Takes the state of an ``Inspector`` run (waypoints, captured media,
detected defects) and emits a structured JSON report ready for storage,
delivery to an inspector, or upload to an insurance / asset-management
SaaS. JSON is intentionally schema-stable so it can be diffed run over
run for the same asset.

Optional add-on: write a sidecar Markdown summary that a human can
read without spinning up a viewer.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from applications.inspection.inspector import Inspector


_SEVERITY_RANK = {
    "none": 0, "minor": 1, "moderate": 2, "major": 3, "critical": 4,
}


@dataclass
class InspectionReport:
    report_id: str
    asset_name: str
    inspection_type: str
    summary: Dict[str, Any]
    waypoints: List[Dict[str, Any]]
    defects: List[Dict[str, Any]]
    media: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)
    json_path: Optional[str] = None
    markdown_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "asset_name": self.asset_name,
            "inspection_type": self.inspection_type,
            "generated_at": self.generated_at,
            "summary": dict(self.summary),
            "waypoints": [dict(w) for w in self.waypoints],
            "defects": [dict(d) for d in self.defects],
            "media": [dict(m) for m in self.media],
            "metadata": dict(self.metadata),
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


def _safe(value: Any) -> Any:
    """Make tuples/enums JSON-friendly."""
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


class InspectionReportGenerator:
    """Generates JSON (and optional Markdown) reports for an inspector run."""

    def __init__(
        self,
        asset_name: str = "asset",
        inspector_id: str = "inspector-1",
        operator_id: Optional[str] = None,
    ) -> None:
        self._asset_name = asset_name
        self._inspector_id = inspector_id
        self._operator_id = operator_id

    def generate(
        self,
        inspector: "Inspector",
        media_artifacts: Optional[List[Dict[str, Any]]] = None,
        output_dir: Optional[str] = None,
        include_markdown: bool = True,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> InspectionReport:
        defects = []
        for d in inspector.get_defects():
            defects.append({
                "defect_id": d.defect_id,
                "type": d.defect_type,
                "severity": _safe(d.severity),
                "confidence": d.confidence,
                "location": _safe(d.location),
                "description": d.description,
                "image_id": d.image_id,
                "timestamp": d.timestamp,
            })
        defects.sort(key=lambda d: _SEVERITY_RANK.get(d["severity"], 0), reverse=True)

        waypoints: List[Dict[str, Any]] = []
        inspection_points = getattr(inspector, "_inspection_points", []) or []
        for ip in inspection_points:
            waypoints.append({
                "point_id": ip.point_id,
                "position": _safe(ip.position),
                "look_at": _safe(ip.look_at),
                "distance_m": ip.distance,
                "gimbal_pitch_deg": ip.gimbal_pitch,
                "gimbal_yaw_deg": ip.gimbal_yaw,
                "dwell_time_s": ip.dwell_time,
            })

        media = list(media_artifacts or [])

        config = getattr(inspector, "_config", None)
        inspection_type = (
            _safe(config.inspection_type) if config is not None else "unknown"
        )

        sev_counts: Dict[str, int] = {k: 0 for k in _SEVERITY_RANK.keys()}
        for d in defects:
            sev_counts[d["severity"]] = sev_counts.get(d["severity"], 0) + 1
        highest = "none"
        for name in ("critical", "major", "moderate", "minor"):
            if sev_counts.get(name, 0) > 0:
                highest = name
                break

        summary = {
            "defect_count": len(defects),
            "highest_severity": highest,
            "severity_breakdown": sev_counts,
            "waypoint_count": len(waypoints),
            "media_count": len(media),
            "actionable": any(
                sev_counts.get(k, 0) > 0 for k in ("major", "critical")
            ),
        }

        report = InspectionReport(
            report_id=uuid.uuid4().hex[:12],
            asset_name=self._asset_name,
            inspection_type=str(inspection_type),
            summary=summary,
            waypoints=waypoints,
            defects=defects,
            media=media,
            metadata={
                "inspector_id": self._inspector_id,
                "operator_id": self._operator_id,
                **(custom_metadata or {}),
            },
        )

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"report_{report.report_id}.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(report.to_dict(), fh, indent=2)
            report.json_path = json_path
            if include_markdown:
                md_path = os.path.join(output_dir, f"report_{report.report_id}.md")
                with open(md_path, "w", encoding="utf-8") as fh:
                    fh.write(self._render_markdown(report))
                report.markdown_path = md_path

        return report

    def _render_markdown(self, report: InspectionReport) -> str:
        s = report.summary
        sev_lines = [
            f"  - {name}: {count}"
            for name, count in s.get("severity_breakdown", {}).items()
            if count > 0
        ]
        lines = [
            f"# Inspection Report — {report.asset_name}",
            "",
            f"- **Report ID:** `{report.report_id}`",
            f"- **Inspection type:** `{report.inspection_type}`",
            f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(report.generated_at))} UTC",
            f"- **Waypoints inspected:** {s['waypoint_count']}",
            f"- **Media captured:** {s['media_count']}",
            "",
            "## Findings",
            "",
            f"- **Defect count:** {s['defect_count']}",
            f"- **Highest severity:** `{s['highest_severity']}`",
            f"- **Actionable:** {'yes' if s['actionable'] else 'no'}",
        ]
        if sev_lines:
            lines.append("- **Severity breakdown:**")
            lines.extend(sev_lines)
        if report.defects:
            lines.extend(["", "## Defects"])
            for d in report.defects[:50]:
                loc = d.get("location") or []
                lines.append(
                    f"- `{d['severity']}` — {d['type']} "
                    f"(conf {d['confidence']:.2f}) @ {loc}"
                )
        if report.metadata:
            lines.extend(["", "## Metadata"])
            for k, v in report.metadata.items():
                lines.append(f"- **{k}:** {v}")
        return "\n".join(lines) + "\n"
