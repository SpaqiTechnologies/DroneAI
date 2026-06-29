"""
Landing system for Drone AI application.
Supports multiple landing modes: normal, precision, emergency, and terrain-following.
"""

import time
import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, List, Tuple

logger = logging.getLogger(__name__)


class LandingMode(Enum):
    """Available landing modes."""
    NORMAL = "normal"  # Standard controlled descent
    PRECISION = "precision"  # Marker-based precision landing
    EMERGENCY = "emergency"  # Immediate descent (motor failure, critical battery)
    TERRAIN_FOLLOW = "terrain_follow"  # Maintain AGL during descent
    SAFE_ZONE = "safe_zone"  # Find and land in safe zone


class LandingState(Enum):
    """Current landing state."""
    IDLE = "idle"
    DESCENDING = "descending"
    FINAL_APPROACH = "final_approach"
    TOUCHDOWN = "touchdown"
    LANDED = "landed"
    ABORTED = "aborted"


class LandingAbortReason(Enum):
    """Reasons for aborting landing."""
    OBSTACLE_DETECTED = auto()
    UNSAFE_TERRAIN = auto()
    HIGH_WIND = auto()
    USER_COMMAND = auto()
    POSITION_LOST = auto()
    MARKER_LOST = auto()


@dataclass
class LandingZone:
    """A potential landing zone."""
    latitude: float
    longitude: float
    altitude: float  # Ground altitude MSL
    size: float  # Radius in meters
    surface_type: str  # "flat", "grass", "concrete", "water", "unknown"
    is_safe: bool = True
    obstacles_nearby: bool = False
    slope_degrees: float = 0.0
    confidence: float = 1.0  # 0-1 confidence score


@dataclass
class LandingConfig:
    """Configuration for landing operations."""
    # Descent rates (m/s)
    normal_descent_rate: float = 0.5
    precision_descent_rate: float = 0.3
    emergency_descent_rate: float = 2.0
    final_approach_rate: float = 0.2

    # Altitude thresholds (meters AGL)
    final_approach_altitude: float = 5.0
    touchdown_altitude: float = 0.3

    # Precision landing
    marker_search_altitude: float = 10.0
    marker_lock_distance: float = 0.5  # meters from marker center

    # Safety limits
    max_wind_for_landing: float = 10.0  # m/s
    max_slope_degrees: float = 15.0
    min_safe_zone_radius: float = 2.0  # meters

    # Timeouts
    marker_search_timeout: float = 30.0  # seconds
    landing_timeout: float = 120.0  # seconds

    # Terrain following
    terrain_follow_height: float = 2.0  # meters AGL


@dataclass
class LandingStatus:
    """Current landing status."""
    mode: LandingMode
    state: LandingState
    target_position: Optional[Tuple[float, float, float]] = None  # lat, lon, alt
    current_altitude_agl: float = 0.0
    descent_rate: float = 0.0
    distance_to_target: float = 0.0
    marker_detected: bool = False
    marker_position: Optional[Tuple[float, float]] = None  # pixel x, y
    safe_zone: Optional[LandingZone] = None
    abort_reason: Optional[LandingAbortReason] = None
    start_time: Optional[float] = None
    elapsed_time: float = 0.0


class LandingManager:
    """
    Manages all landing operations with multiple modes and safety checks.
    """

    def __init__(self, config: LandingConfig = None):
        self._config = config or LandingConfig()
        self._mode = LandingMode.NORMAL
        self._state = LandingState.IDLE
        self._callbacks: List[Callable] = []

        # Landing tracking
        self._target_position: Optional[Tuple[float, float, float]] = None
        self._start_time: Optional[float] = None
        self._start_altitude: float = 0.0

        # Precision landing
        self._marker_detected = False
        self._marker_position: Optional[Tuple[float, float]] = None
        self._marker_search_start: Optional[float] = None

        # Safe zone detection
        self._detected_zones: List[LandingZone] = []
        self._selected_zone: Optional[LandingZone] = None

        # Abort handling
        self._abort_reason: Optional[LandingAbortReason] = None

    def add_callback(self, callback: Callable):
        """Add a callback for landing events."""
        self._callbacks.append(callback)

    def start_landing(
        self,
        mode: LandingMode,
        current_position: Tuple[float, float, float],
        target_position: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[bool, str]:
        """
        Start a landing operation.

        Args:
            mode: Landing mode to use
            current_position: Current (lat, lon, altitude_agl)
            target_position: Optional target landing position

        Returns:
            (success, message)
        """
        if self._state not in (LandingState.IDLE, LandingState.ABORTED, LandingState.LANDED):
            return False, f"Cannot start landing: already in {self._state.value} state"

        self._mode = mode
        self._state = LandingState.DESCENDING
        self._start_time = time.time()
        self._start_altitude = current_position[2]
        self._abort_reason = None

        # Set target position
        if target_position:
            self._target_position = target_position
        else:
            # Land at current horizontal position
            self._target_position = (current_position[0], current_position[1], 0.0)

        logger.info(f"Starting {mode.value} landing from {self._start_altitude:.1f}m AGL")
        self._notify_callbacks('landing_started', {'mode': mode.value})

        return True, f"Started {mode.value} landing"

    def update(
        self,
        current_position: Tuple[float, float, float],
        wind_speed: float = 0.0,
        obstacle_below: bool = False,
        terrain_altitude: Optional[float] = None,
        marker_detected: bool = False,
        marker_position: Optional[Tuple[float, float]] = None,
    ) -> Tuple[LandingState, float, Optional[Tuple[float, float, float]]]:
        """
        Update landing state and get next command.

        Args:
            current_position: Current (lat, lon, altitude_agl)
            wind_speed: Current wind speed in m/s
            obstacle_below: True if obstacle detected below drone
            terrain_altitude: Terrain altitude if known
            marker_detected: True if landing marker is detected
            marker_position: Marker position in image (x, y pixels)

        Returns:
            (state, descent_rate, target_position)
        """
        if self._state == LandingState.IDLE:
            return self._state, 0.0, None

        if self._state in (LandingState.LANDED, LandingState.ABORTED):
            return self._state, 0.0, None

        current_altitude = current_position[2]

        # Check for abort conditions
        abort_reason = self._check_abort_conditions(
            wind_speed, obstacle_below, current_position
        )
        if abort_reason and self._mode != LandingMode.EMERGENCY:
            self._abort_landing(abort_reason)
            return self._state, 0.0, None

        # Update marker tracking for precision landing
        if self._mode == LandingMode.PRECISION:
            self._marker_detected = marker_detected
            self._marker_position = marker_position

        # Calculate descent rate based on mode and altitude
        descent_rate = self._calculate_descent_rate(current_altitude)

        # Calculate target position
        target = self._calculate_target_position(
            current_position, marker_detected, marker_position, terrain_altitude
        )

        # Update state based on altitude
        self._update_state(current_altitude)

        # Check for landing timeout
        if self._start_time and (time.time() - self._start_time) > self._config.landing_timeout:
            if self._mode != LandingMode.EMERGENCY:
                logger.warning("Landing timeout reached")
                self._abort_landing(LandingAbortReason.POSITION_LOST)
                return self._state, 0.0, None

        return self._state, descent_rate, target

    def _calculate_descent_rate(self, current_altitude: float) -> float:
        """Calculate appropriate descent rate based on mode and altitude."""
        if self._mode == LandingMode.EMERGENCY:
            return self._config.emergency_descent_rate

        if current_altitude <= self._config.final_approach_altitude:
            return self._config.final_approach_rate

        if self._mode == LandingMode.PRECISION:
            return self._config.precision_descent_rate

        return self._config.normal_descent_rate

    def _calculate_target_position(
        self,
        current_position: Tuple[float, float, float],
        marker_detected: bool,
        marker_position: Optional[Tuple[float, float]],
        terrain_altitude: Optional[float],
    ) -> Tuple[float, float, float]:
        """Calculate target position for landing."""
        target_lat, target_lon, target_alt = self._target_position

        if self._mode == LandingMode.PRECISION and marker_detected and marker_position:
            # Adjust position based on marker offset
            # This is simplified - real implementation would use camera calibration
            marker_x, marker_y = marker_position
            # Assume marker should be at image center (320, 240 for 640x480)
            offset_x = (marker_x - 320) / 320 * 2  # Normalized offset
            offset_y = (marker_y - 240) / 240 * 2

            # Convert to position adjustment (simplified)
            lat_adj = offset_y * 0.00001 * current_position[2]
            lon_adj = offset_x * 0.00001 * current_position[2]

            target_lat = current_position[0] - lat_adj
            target_lon = current_position[1] - lon_adj

        elif self._mode == LandingMode.TERRAIN_FOLLOW and terrain_altitude is not None:
            # Maintain height above terrain
            target_alt = terrain_altitude + self._config.terrain_follow_height

        elif self._mode == LandingMode.SAFE_ZONE and self._selected_zone:
            target_lat = self._selected_zone.latitude
            target_lon = self._selected_zone.longitude
            target_alt = self._selected_zone.altitude

        return (target_lat, target_lon, target_alt)

    def _update_state(self, current_altitude: float):
        """Update landing state based on altitude."""
        previous_state = self._state

        if current_altitude <= self._config.touchdown_altitude:
            self._state = LandingState.TOUCHDOWN
            # Brief pause then landed
            self._state = LandingState.LANDED
            logger.info("Touchdown - landing complete")
            self._notify_callbacks('landing_complete', {
                'mode': self._mode.value,
                'duration': time.time() - self._start_time if self._start_time else 0
            })

        elif current_altitude <= self._config.final_approach_altitude:
            if self._state != LandingState.FINAL_APPROACH:
                self._state = LandingState.FINAL_APPROACH
                logger.debug("Entering final approach")

        if previous_state != self._state:
            self._notify_callbacks('state_changed', {
                'previous': previous_state.value,
                'current': self._state.value
            })

    def _check_abort_conditions(
        self,
        wind_speed: float,
        obstacle_below: bool,
        current_position: Tuple[float, float, float],
    ) -> Optional[LandingAbortReason]:
        """Check if landing should be aborted."""
        # Don't abort emergency landings
        if self._mode == LandingMode.EMERGENCY:
            return None

        if wind_speed > self._config.max_wind_for_landing:
            return LandingAbortReason.HIGH_WIND

        if obstacle_below and current_position[2] < self._config.final_approach_altitude:
            return LandingAbortReason.OBSTACLE_DETECTED

        if self._mode == LandingMode.PRECISION:
            if not self._marker_detected and self._state == LandingState.FINAL_APPROACH:
                if self._marker_search_start is None:
                    self._marker_search_start = time.time()
                elif (time.time() - self._marker_search_start) > self._config.marker_search_timeout:
                    return LandingAbortReason.MARKER_LOST

        return None

    def _abort_landing(self, reason: LandingAbortReason):
        """Abort the landing operation."""
        self._state = LandingState.ABORTED
        self._abort_reason = reason

        logger.warning(f"Landing aborted: {reason.name}")
        self._notify_callbacks('landing_aborted', {'reason': reason.name})

    def abort(self, reason: LandingAbortReason = LandingAbortReason.USER_COMMAND) -> Tuple[bool, str]:
        """Manually abort landing."""
        if self._state in (LandingState.IDLE, LandingState.LANDED, LandingState.ABORTED):
            return False, "No active landing to abort"

        self._abort_landing(reason)
        return True, f"Landing aborted: {reason.name}"

    def reset(self):
        """Reset landing manager to idle state."""
        self._state = LandingState.IDLE
        self._mode = LandingMode.NORMAL
        self._target_position = None
        self._start_time = None
        self._marker_detected = False
        self._marker_position = None
        self._marker_search_start = None
        self._abort_reason = None
        self._selected_zone = None

    def add_safe_zone(self, zone: LandingZone):
        """Add a detected safe landing zone."""
        self._detected_zones.append(zone)

    def select_best_safe_zone(
        self,
        current_position: Tuple[float, float, float],
    ) -> Optional[LandingZone]:
        """Select the best safe zone for landing."""
        if not self._detected_zones:
            return None

        def score_zone(zone: LandingZone) -> float:
            if not zone.is_safe:
                return -1

            score = zone.confidence * 100

            # Prefer larger zones
            score += min(zone.size / self._config.min_safe_zone_radius, 5) * 10

            # Penalize slopes
            score -= zone.slope_degrees * 2

            # Penalize nearby obstacles
            if zone.obstacles_nearby:
                score -= 20

            # Prefer closer zones
            distance = self._haversine_distance(
                current_position[0], current_position[1],
                zone.latitude, zone.longitude
            )
            score -= distance / 10  # 1 point per 10 meters

            # Surface preference
            surface_scores = {
                "concrete": 20,
                "flat": 15,
                "grass": 10,
                "unknown": 0,
                "water": -100,
            }
            score += surface_scores.get(zone.surface_type, 0)

            return score

        safe_zones = [z for z in self._detected_zones if z.is_safe]
        if not safe_zones:
            return None

        self._selected_zone = max(safe_zones, key=score_zone)
        return self._selected_zone

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000  # Earth radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _notify_callbacks(self, event: str, data: dict = None):
        """Notify callbacks of landing events."""
        for callback in self._callbacks:
            try:
                callback({
                    'event': event,
                    'mode': self._mode.value,
                    'state': self._state.value,
                    'timestamp': time.time(),
                    **(data or {}),
                })
            except Exception as e:
                logger.error(f"Landing callback error: {e}")

    def get_status(self) -> LandingStatus:
        """Get current landing status."""
        elapsed = time.time() - self._start_time if self._start_time else 0

        return LandingStatus(
            mode=self._mode,
            state=self._state,
            target_position=self._target_position,
            current_altitude_agl=0.0,  # Would be set by caller
            descent_rate=self._calculate_descent_rate(10.0),  # Estimate
            distance_to_target=0.0,  # Would be calculated
            marker_detected=self._marker_detected,
            marker_position=self._marker_position,
            safe_zone=self._selected_zone,
            abort_reason=self._abort_reason,
            start_time=self._start_time,
            elapsed_time=elapsed,
        )

    @property
    def state(self) -> LandingState:
        """Get current landing state."""
        return self._state

    @property
    def mode(self) -> LandingMode:
        """Get current landing mode."""
        return self._mode

    @property
    def is_landing(self) -> bool:
        """Check if landing is in progress."""
        return self._state in (
            LandingState.DESCENDING,
            LandingState.FINAL_APPROACH,
            LandingState.TOUCHDOWN,
        )

    @property
    def is_complete(self) -> bool:
        """Check if landing is complete."""
        return self._state == LandingState.LANDED

    def to_dict(self) -> dict:
        """Serialize landing manager state."""
        return {
            'mode': self._mode.value,
            'state': self._state.value,
            'target_position': self._target_position,
            'start_time': self._start_time,
            'marker_detected': self._marker_detected,
            'abort_reason': self._abort_reason.name if self._abort_reason else None,
            'selected_zone': {
                'lat': self._selected_zone.latitude,
                'lon': self._selected_zone.longitude,
                'size': self._selected_zone.size,
                'surface': self._selected_zone.surface_type,
            } if self._selected_zone else None,
            'detected_zones_count': len(self._detected_zones),
        }
