"""Wind-corrected supply drop release-point planner.

Personnel-recovery use case: drop water / medical / radio resupply
packages on a stranded soldier's coordinates. The release point is
*upwind* of the impact target so the package, after fall + drag,
lands on the target.

Model:
  - Free-fall with drag coefficient ``k`` so vertical velocity:
        v(t) = v_term * (1 - exp(-k * t))  if k > 0 else g*t
  - Falltime by inversion of the altitude integral.
  - Lateral drift: package decelerates from drone's release velocity and
    then matches wind (an exponential decay toward the wind vector).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


GRAVITY_MS2 = 9.80665


@dataclass
class DropParameters:
    """Package + drop kinematic parameters."""
    package_mass_kg: float = 5.0
    drag_coefficient: float = 0.06       # exponential drag constant k in 1/s
    parachute_deploy_at_m: Optional[float] = None
    parachute_terminal_speed_mps: float = 6.0
    release_altitude_agl_m: float = 30.0
    release_speed_mps: float = 8.0       # drone ground speed at release
    target_impact_radius_m: float = 5.0


@dataclass
class SupplyDropPlan:
    target_lat: float
    target_lon: float
    release_lat: float
    release_lon: float
    release_heading_deg: float           # heading to fly INTO target on release
    release_altitude_agl_m: float
    fall_time_s: float
    horizontal_drift_m: float
    lateral_offset_m: float              # 0 if no crosswind
    expected_impact_radius_m: float
    wind_speed_mps: float
    wind_direction_deg: float
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "target": {"latitude": self.target_lat, "longitude": self.target_lon},
            "release_point": {
                "latitude": self.release_lat,
                "longitude": self.release_lon,
                "altitude_agl_m": self.release_altitude_agl_m,
                "heading_deg": self.release_heading_deg,
            },
            "fall_time_s": self.fall_time_s,
            "horizontal_drift_m": self.horizontal_drift_m,
            "lateral_offset_m": self.lateral_offset_m,
            "expected_impact_radius_m": self.expected_impact_radius_m,
            "wind": {
                "speed_mps": self.wind_speed_mps,
                "direction_deg": self.wind_direction_deg,
            },
            "notes": self.notes,
        }


def _wind_components(speed: float, direction_deg: float) -> Tuple[float, float]:
    """Return (east, north) wind components.

    ``direction_deg`` is the meteorological "wind from" direction in
    degrees (0 = from north, 90 = from east). The vector returned points
    in the direction the air is moving toward.
    """
    rad = math.radians(direction_deg)
    return -speed * math.sin(rad), -speed * math.cos(rad)


def _project_offset(
    origin: Tuple[float, float], east_m: float, north_m: float
) -> Tuple[float, float]:
    lat0 = origin[0]
    dlat = (north_m / 6371000.0) * (180.0 / math.pi)
    dlon = (east_m / (6371000.0 * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
    return origin[0] + dlat, origin[1] + dlon


class SupplyDropPlanner:
    def __init__(self, params: Optional[DropParameters] = None) -> None:
        self._params = params or DropParameters()

    @property
    def params(self) -> DropParameters:
        return self._params

    def plan(
        self,
        target_lat: float,
        target_lon: float,
        wind_speed_mps: float = 0.0,
        wind_direction_deg: float = 0.0,
        release_heading_deg: Optional[float] = None,
    ) -> SupplyDropPlan:
        if wind_speed_mps < 0:
            raise ValueError("wind_speed_mps must be >= 0")
        p = self._params
        fall_time = self._fall_time(p.release_altitude_agl_m)

        # Wind vector (east, north) — direction air moves toward
        wx, wy = _wind_components(wind_speed_mps, wind_direction_deg)

        # The package's average lateral velocity over fall time:
        # initially carrying drone's heading vector, then exponentially
        # decaying toward the wind vector. Average is a weighted mean.
        if release_heading_deg is None:
            heading = math.degrees(math.atan2(-wx, -wy)) % 360.0
        else:
            heading = release_heading_deg % 360.0
        hr = math.radians(heading)
        rel_e = p.release_speed_mps * math.sin(hr)
        rel_n = p.release_speed_mps * math.cos(hr)

        k = max(1e-3, p.drag_coefficient)
        avg_factor = (1.0 - math.exp(-k * fall_time)) / (k * fall_time) if fall_time > 0 else 1.0
        drift_e_release = rel_e * avg_factor * fall_time
        drift_n_release = rel_n * avg_factor * fall_time
        drift_e_wind = wx * (1.0 - avg_factor) * fall_time
        drift_n_wind = wy * (1.0 - avg_factor) * fall_time

        drift_e_total = drift_e_release + drift_e_wind
        drift_n_total = drift_n_release + drift_n_wind

        release_lat, release_lon = _project_offset(
            (target_lat, target_lon), -drift_e_total, -drift_n_total
        )

        horiz_drift = math.hypot(drift_e_total, drift_n_total)
        lateral_offset = abs(rel_e * (-wy) + rel_n * wx) / max(1e-3, p.release_speed_mps)

        expected_impact = max(
            p.target_impact_radius_m,
            wind_speed_mps * 0.3 + p.release_altitude_agl_m * 0.05,
        )

        notes = (
            "release upwind of target; verify package leaves cleanly. "
            f"falltime={fall_time:.1f}s, drift={horiz_drift:.1f}m"
        )

        return SupplyDropPlan(
            target_lat=target_lat,
            target_lon=target_lon,
            release_lat=release_lat,
            release_lon=release_lon,
            release_heading_deg=heading,
            release_altitude_agl_m=p.release_altitude_agl_m,
            fall_time_s=fall_time,
            horizontal_drift_m=horiz_drift,
            lateral_offset_m=lateral_offset,
            expected_impact_radius_m=expected_impact,
            wind_speed_mps=wind_speed_mps,
            wind_direction_deg=wind_direction_deg,
            notes=notes,
        )

    # ------------------------- physics ------------------------------------

    def _fall_time(self, height_m: float) -> float:
        if height_m <= 0:
            return 0.0
        p = self._params
        deploy_at = p.parachute_deploy_at_m
        if deploy_at is not None and deploy_at < height_m:
            t1 = self._free_fall_time(height_m - deploy_at)
            t2 = deploy_at / max(0.1, p.parachute_terminal_speed_mps)
            return t1 + t2
        return self._free_fall_time(height_m)

    def _free_fall_time(self, height_m: float) -> float:
        p = self._params
        k = p.drag_coefficient
        if k <= 0:
            return math.sqrt(2.0 * height_m / GRAVITY_MS2)
        # Integrate v(t) = v_term*(1 - e^{-k t}) where v_term = g/k
        v_term = GRAVITY_MS2 / k
        # Distance fallen: s(t) = v_term * (t + (e^{-k t}-1)/k)
        # Solve numerically (bisect).
        lo, hi = 0.0, 600.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            s = v_term * (mid + (math.exp(-k * mid) - 1.0) / k)
            if s < height_m:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
