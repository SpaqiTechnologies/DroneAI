"""
Battery-aware return-to-home path planner.

The planner exposes a single public method:
```
plan_return(home_position, current_state)
```
where ``current_state`` contains battery percentage and wind speed.
It returns a list of waypoints that safely bring the drone back to
``home_position`` while respecting the battery budget.  The algorithm is
simple: it first checks if the straight‑line distance can be covered;
if not, it inserts intermediate waypoints at a fixed step size until
the remaining distance fits within the available energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------
@dataclass
class Position:
    latitude: float
    longitude: float
    altitude: float = 0.0

@dataclass
class State:
    battery_percent: float  # 0‑100
    wind_speed_m_s: float   # meters per second

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0


def haversine(a: Position, b: Position) -> float:
    """Return distance in kilometers between two positions."""
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a_val = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a_val))
    return EARTH_RADIUS_KM * c

# ---------------------------------------------------------------------------
# Planner implementation
# ---------------------------------------------------------------------------
class PathPlanner:
    """Generate a battery‑aware return path.

    The planner assumes that the drone consumes 1 % battery per km of flight
    in still air.  Wind increases consumption proportionally to wind speed.
    For simplicity we use a linear model: ``consumption = base + k * wind``.
    """

    BASE_CONSUMPTION_PER_KM = 1.0  # % per km at zero wind
    WIND_FACTOR = 0.05            # additional % per m/s of wind
    MAX_WAYPOINTS = 20            # safety limit to avoid infinite loops

    def __init__(self, step_km: float = 0.5):
        self.step_km = step_km

    def _energy_needed(self, distance_km: float, wind_speed: float) -> float:
        return distance_km * (self.BASE_CONSUMPTION_PER_KM + self.WIND_FACTOR * wind_speed)

    def plan_return(
        self,
        home: Position,
        current: Position,
        state: State,
    ) -> List[Position]:
        """Return a list of waypoints from ``current`` to ``home``.

        Raises ``ValueError`` if the battery is insufficient for even the first step.
        """
        distance = haversine(current, home)
        energy_needed = self._energy_needed(distance, state.wind_speed_m_s)

        if state.battery_percent < energy_needed:
            # Try to insert intermediate waypoints until we fit the budget
            steps = max(1, int(math.ceil(distance / self.step_km)))
            if steps > self.MAX_WAYPOINTS:
                raise ValueError("Too many waypoints required for return path")
            waypoint_distance = distance / steps
            # Linear interpolation between current and home
            waypoints: List[Position] = []
            for i in range(1, steps + 1):
                frac = i / steps
                lat = current.latitude + (home.latitude - current.latitude) * frac
                lon = current.longitude + (home.longitude - current.longitude) * frac
                alt = current.altitude + (home.altitude - current.altitude) * frac
                wp = Position(lat, lon, alt)
                waypoints.append(wp)
            # Verify final segment fits battery budget
            total_energy = 0.0
            prev = current
            for wp in waypoints:
                seg_dist = haversine(prev, wp)
                total_energy += self._energy_needed(seg_dist, state.wind_speed_m_s)
                prev = wp
            if total_energy > state.battery_percent:
                raise ValueError("Battery insufficient even with intermediate waypoints")
            return waypoints
        else:
            # Straight line is fine
            return [home]

# ---------------------------------------------------------------------------
# Example usage (not executed in tests)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    planner = PathPlanner()
    home_pos = Position(52.2297, 21.0122)
    cur_pos = Position(52.4064, 16.9252)
    state = State(battery_percent=30.0, wind_speed_m_s=5.0)
    try:
        route = planner.plan_return(home_pos, cur_pos, state)
        print("Waypoints:")
        for wp in route:
            print(wp)
    except ValueError as e:
        print(e)
"""
