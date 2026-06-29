"""Ground safe-corridor planner.

Personnel-recovery use case: given a start and pickup point plus a set
of *threat zones* (polygon-approximated as circles for tractability)
and known obstacles, generate a piecewise-linear ground corridor that
stays outside every threat radius plus a safety margin.

Algorithm: deterministic visibility-graph search.
  - Nodes = start, goal, plus tangent points around every threat circle.
  - Edges connect any two nodes whose straight-line segment does not
    intersect any threat keep-out disk.
  - Dijkstra finds the shortest such path.

Pure-Python; no scipy/networkx dependency. Local distances are
computed in meters via equirectangular projection so the math is
straightforward and accurate within a few-km tile.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


def _to_xy(point: Tuple[float, float], origin: Tuple[float, float]) -> Tuple[float, float]:
    lat0 = math.radians(origin[0])
    dx = math.radians(point[1] - origin[1]) * 6371000.0 * math.cos(lat0)
    dy = math.radians(point[0] - origin[0]) * 6371000.0
    return dx, dy


def _to_latlon(xy: Tuple[float, float], origin: Tuple[float, float]) -> Tuple[float, float]:
    lat0 = math.radians(origin[0])
    dlat = (xy[1] / 6371000.0) * (180.0 / math.pi)
    dlon = (xy[0] / (6371000.0 * math.cos(lat0))) * (180.0 / math.pi)
    return origin[0] + dlat, origin[1] + dlon


def _segment_circle_intersects(
    p1: Tuple[float, float], p2: Tuple[float, float],
    center: Tuple[float, float], radius: float,
) -> bool:
    """True if the segment p1->p2 passes within ``radius`` of ``center``."""
    cx, cy = center
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(x1 - cx, y1 - cy) <= radius
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / L2))
    nx, ny = x1 + t * dx, y1 + t * dy
    return math.hypot(nx - cx, ny - cy) <= radius


@dataclass
class ThreatZone:
    latitude: float
    longitude: float
    radius_m: float
    name: str = "threat"
    severity: float = 1.0          # 0..1; cosmetic for now

    def to_dict(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "radius_m": self.radius_m,
            "name": self.name,
            "severity": self.severity,
        }


@dataclass
class CorridorSegment:
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    length_m: float

    def to_dict(self) -> dict:
        return {
            "start": {"latitude": self.start_lat, "longitude": self.start_lon},
            "end": {"latitude": self.end_lat, "longitude": self.end_lon},
            "length_m": self.length_m,
        }


@dataclass
class SafeCorridor:
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    segments: List[CorridorSegment] = field(default_factory=list)
    total_length_m: float = 0.0
    avoided_threats: int = 0
    safe: bool = False
    failure_reason: Optional[str] = None
    min_threat_distance_m: float = float("inf")

    def to_dict(self) -> dict:
        return {
            "waypoints": [list(wp) for wp in self.waypoints],
            "segments": [s.to_dict() for s in self.segments],
            "total_length_m": self.total_length_m,
            "avoided_threats": self.avoided_threats,
            "safe": self.safe,
            "failure_reason": self.failure_reason,
            "min_threat_distance_m": (
                None if self.min_threat_distance_m == float("inf")
                else self.min_threat_distance_m
            ),
        }


class SafeCorridorPlanner:
    """Generate a piecewise-linear corridor between two points."""

    DEFAULT_MARGIN_M = 25.0
    TANGENTS_PER_THREAT = 8

    def __init__(
        self,
        threats: Optional[Sequence[ThreatZone]] = None,
        margin_m: float = DEFAULT_MARGIN_M,
    ) -> None:
        self._threats = list(threats or [])
        self._margin = float(margin_m)

    @property
    def threats(self) -> List[ThreatZone]:
        return list(self._threats)

    def add_threat(self, threat: ThreatZone) -> None:
        self._threats.append(threat)

    def clear_threats(self) -> None:
        self._threats.clear()

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> SafeCorridor:
        if start == goal:
            return SafeCorridor(
                waypoints=[start, goal],
                segments=[],
                total_length_m=0.0,
                avoided_threats=0,
                safe=True,
            )
        origin = start
        start_xy = (0.0, 0.0)
        goal_xy = _to_xy(goal, origin)

        threats_xy: List[Tuple[Tuple[float, float], float]] = []
        for t in self._threats:
            cxy = _to_xy((t.latitude, t.longitude), origin)
            r = t.radius_m + self._margin
            threats_xy.append((cxy, r))

        for cxy, r in threats_xy:
            if math.hypot(start_xy[0] - cxy[0], start_xy[1] - cxy[1]) < r:
                return SafeCorridor(
                    waypoints=[start, goal],
                    segments=[],
                    safe=False,
                    failure_reason="start inside threat keep-out zone",
                    avoided_threats=0,
                )
            if math.hypot(goal_xy[0] - cxy[0], goal_xy[1] - cxy[1]) < r:
                return SafeCorridor(
                    waypoints=[start, goal],
                    segments=[],
                    safe=False,
                    failure_reason="goal inside threat keep-out zone",
                    avoided_threats=0,
                )

        nodes: List[Tuple[float, float]] = [start_xy, goal_xy]
        for cxy, r in threats_xy:
            for i in range(self.TANGENTS_PER_THREAT):
                angle = 2 * math.pi * i / self.TANGENTS_PER_THREAT
                nx = cxy[0] + (r * 1.05) * math.cos(angle)
                ny = cxy[1] + (r * 1.05) * math.sin(angle)
                nodes.append((nx, ny))

        n = len(nodes)
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._segment_collides(nodes[i], nodes[j], threats_xy):
                    continue
                dist = math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
                adj[i].append((j, dist))
                adj[j].append((i, dist))

        path = self._dijkstra(adj, 0, 1)
        if not path:
            return SafeCorridor(
                waypoints=[start, goal],
                segments=[],
                safe=False,
                failure_reason="no path found (over-constrained)",
                avoided_threats=len(self._threats),
            )

        latlon_path: List[Tuple[float, float]] = []
        for idx in path:
            xy = nodes[idx]
            latlon_path.append(_to_latlon(xy, origin))

        segments: List[CorridorSegment] = []
        total = 0.0
        min_threat_dist = float("inf")
        for a, b in zip(path[:-1], path[1:]):
            ax, ay = nodes[a]
            bx, by = nodes[b]
            d = math.hypot(bx - ax, by - ay)
            total += d
            ll_a = _to_latlon(nodes[a], origin)
            ll_b = _to_latlon(nodes[b], origin)
            segments.append(CorridorSegment(
                start_lat=ll_a[0], start_lon=ll_a[1],
                end_lat=ll_b[0], end_lon=ll_b[1], length_m=d,
            ))
            for cxy, r in threats_xy:
                dist_to_threat = self._segment_point_distance(nodes[a], nodes[b], cxy) - (r - self._margin)
                if dist_to_threat < min_threat_dist:
                    min_threat_dist = dist_to_threat

        return SafeCorridor(
            waypoints=latlon_path,
            segments=segments,
            total_length_m=total,
            avoided_threats=len(self._threats),
            safe=True,
            failure_reason=None,
            min_threat_distance_m=max(0.0, min_threat_dist),
        )

    # ------------------------- helpers -----------------------------------

    def _segment_collides(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        threats_xy: List[Tuple[Tuple[float, float], float]],
    ) -> bool:
        for cxy, r in threats_xy:
            if _segment_circle_intersects(a, b, cxy, r):
                return True
        return False

    def _segment_point_distance(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        p: Tuple[float, float],
    ) -> float:
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        nx, ny = ax + t * dx, ay + t * dy
        return math.hypot(nx - px, ny - py)

    def _dijkstra(
        self,
        adj: List[List[Tuple[int, float]]],
        src: int,
        dst: int,
    ) -> List[int]:
        n = len(adj)
        dist: List[float] = [float("inf")] * n
        prev: List[int] = [-1] * n
        dist[src] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == dst:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[dst] == float("inf"):
            return []
        path: List[int] = []
        cur = dst
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path
