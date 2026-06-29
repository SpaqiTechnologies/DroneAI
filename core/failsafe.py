"""
Failsafe system for Drone AI application.
Manages automated safety responses for various failure conditions.
"""

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FailsafeType(Enum):
    """Types of failsafe conditions."""
    GPS_LOSS = auto()
    LOW_BATTERY = auto()
    CRITICAL_BATTERY = auto()
    SIGNAL_LOSS = auto()
    SENSOR_FAILURE = auto()
    GEOFENCE_BREACH = auto()
    OBSTACLE_COLLISION_IMMINENT = auto()
    MOTOR_FAILURE = auto()
    IMU_FAILURE = auto()


class FailsafeAction(Enum):
    """Actions that can be taken in response to failsafe conditions."""
    NONE = "none"
    HOVER = "hover"
    RETURN_TO_HOME = "return_to_home"
    LAND_IMMEDIATELY = "land_immediately"
    DESCEND_AND_LAND = "descend_and_land"
    HOLD_POSITION = "hold_position"
    EMERGENCY_STOP = "emergency_stop"


class FailsafeState(Enum):
    """Current state of the failsafe system."""
    NORMAL = "normal"
    WARNING = "warning"
    ACTIVE = "active"
    CRITICAL = "critical"


@dataclass
class FailsafeCondition:
    """Represents a failsafe condition configuration."""
    failsafe_type: FailsafeType
    action: FailsafeAction
    threshold: float  # Condition-specific threshold
    timeout: float  # Seconds before failsafe triggers
    priority: int  # Lower number = higher priority
    enabled: bool = True
    message: str = ""


@dataclass
class FailsafeEvent:
    """Records a failsafe event occurrence."""
    failsafe_type: FailsafeType
    action: FailsafeAction
    triggered_at: float
    cleared_at: Optional[float] = None
    reason: str = ""
    data: Optional[Dict] = None


class FailsafeManager:
    """
    Manages failsafe conditions and automated responses.
    Monitors system state and triggers appropriate safety actions.
    """

    # Default thresholds
    DEFAULT_GPS_LOSS_TIMEOUT = 5.0  # seconds
    DEFAULT_LOW_BATTERY_THRESHOLD = 25.0  # percent
    DEFAULT_CRITICAL_BATTERY_THRESHOLD = 10.0  # percent
    DEFAULT_SIGNAL_LOSS_TIMEOUT = 3.0  # seconds
    DEFAULT_MIN_OBSTACLE_DISTANCE = 1.5  # meters

    def __init__(self):
        self._conditions: Dict[FailsafeType, FailsafeCondition] = {}
        self._active_failsafes: Dict[FailsafeType, FailsafeEvent] = {}
        self._event_history: List[FailsafeEvent] = []
        self._state = FailsafeState.NORMAL
        self._callbacks: Dict[FailsafeType, List[Callable]] = {}

        # Condition timers (for timeout-based failsafes)
        self._condition_start_times: Dict[FailsafeType, float] = {}

        # Initialize default failsafe conditions
        self._setup_default_conditions()

    def _setup_default_conditions(self):
        """Setup default failsafe conditions."""
        defaults = [
            FailsafeCondition(
                failsafe_type=FailsafeType.GPS_LOSS,
                action=FailsafeAction.HOVER,
                threshold=self.DEFAULT_GPS_LOSS_TIMEOUT,
                timeout=self.DEFAULT_GPS_LOSS_TIMEOUT,
                priority=2,
                message="GPS signal lost - hovering in place",
            ),
            FailsafeCondition(
                failsafe_type=FailsafeType.LOW_BATTERY,
                action=FailsafeAction.RETURN_TO_HOME,
                threshold=self.DEFAULT_LOW_BATTERY_THRESHOLD,
                timeout=0,
                priority=3,
                message="Low battery - returning to home",
            ),
            FailsafeCondition(
                failsafe_type=FailsafeType.CRITICAL_BATTERY,
                action=FailsafeAction.LAND_IMMEDIATELY,
                threshold=self.DEFAULT_CRITICAL_BATTERY_THRESHOLD,
                timeout=0,
                priority=1,
                message="Critical battery - landing immediately",
            ),
            FailsafeCondition(
                failsafe_type=FailsafeType.SIGNAL_LOSS,
                action=FailsafeAction.RETURN_TO_HOME,
                threshold=self.DEFAULT_SIGNAL_LOSS_TIMEOUT,
                timeout=self.DEFAULT_SIGNAL_LOSS_TIMEOUT,
                priority=2,
                message="Control signal lost - returning to home",
            ),
            FailsafeCondition(
                failsafe_type=FailsafeType.SENSOR_FAILURE,
                action=FailsafeAction.LAND_IMMEDIATELY,
                threshold=0,
                timeout=0,
                priority=1,
                message="Critical sensor failure - landing immediately",
            ),
            FailsafeCondition(
                failsafe_type=FailsafeType.GEOFENCE_BREACH,
                action=FailsafeAction.RETURN_TO_HOME,
                threshold=0,
                timeout=0,
                priority=2,
                message="Geofence boundary breached - returning to home",
            ),
            FailsafeCondition(
                failsafe_type=FailsafeType.OBSTACLE_COLLISION_IMMINENT,
                action=FailsafeAction.HOVER,
                threshold=self.DEFAULT_MIN_OBSTACLE_DISTANCE,
                timeout=0,
                priority=1,
                message="Obstacle collision imminent - stopping",
            ),
            FailsafeCondition(
                failsafe_type=FailsafeType.MOTOR_FAILURE,
                action=FailsafeAction.LAND_IMMEDIATELY,
                threshold=0,
                timeout=0,
                priority=1,
                message="Motor failure - landing immediately",
            ),
            FailsafeCondition(
                failsafe_type=FailsafeType.IMU_FAILURE,
                action=FailsafeAction.LAND_IMMEDIATELY,
                threshold=0,
                timeout=0,
                priority=1,
                message="IMU failure - landing immediately",
            ),
        ]

        for condition in defaults:
            self._conditions[condition.failsafe_type] = condition

    def configure_failsafe(
        self,
        failsafe_type: FailsafeType,
        action: Optional[FailsafeAction] = None,
        threshold: Optional[float] = None,
        timeout: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        """Configure a failsafe condition."""
        if failsafe_type not in self._conditions:
            raise ValueError(f"Unknown failsafe type: {failsafe_type}")

        condition = self._conditions[failsafe_type]

        if action is not None:
            condition.action = action
        if threshold is not None:
            condition.threshold = threshold
        if timeout is not None:
            condition.timeout = timeout
        if enabled is not None:
            condition.enabled = enabled

        logger.info(f"Configured failsafe {failsafe_type.name}: {condition}")

    def register_callback(
        self,
        failsafe_type: FailsafeType,
        callback: Callable[[FailsafeEvent], None],
    ):
        """Register a callback to be called when a failsafe triggers."""
        if failsafe_type not in self._callbacks:
            self._callbacks[failsafe_type] = []
        self._callbacks[failsafe_type].append(callback)

    def check_gps(
        self,
        has_fix: bool,
        satellites: int,
        accuracy: float,
    ) -> Optional[FailsafeAction]:
        """Check GPS status and trigger failsafe if necessary."""
        condition = self._conditions[FailsafeType.GPS_LOSS]
        if not condition.enabled:
            return None

        # GPS is considered lost if no fix or poor quality
        gps_ok = has_fix and satellites >= 4 and accuracy < 50.0

        if not gps_ok:
            # Start or continue timing
            if FailsafeType.GPS_LOSS not in self._condition_start_times:
                self._condition_start_times[FailsafeType.GPS_LOSS] = time.time()
                logger.warning("GPS signal degraded - monitoring...")

            elapsed = time.time() - self._condition_start_times[FailsafeType.GPS_LOSS]

            if elapsed >= condition.timeout:
                return self._trigger_failsafe(
                    FailsafeType.GPS_LOSS,
                    f"GPS lost for {elapsed:.1f}s (satellites: {satellites}, accuracy: {accuracy:.1f}m)",
                    {'satellites': satellites, 'accuracy': accuracy},
                )
        else:
            # GPS recovered
            self._clear_condition(FailsafeType.GPS_LOSS)

        return None

    def check_battery(self, battery_percent: float) -> Optional[FailsafeAction]:
        """Check battery level and trigger failsafe if necessary."""
        # Check critical battery first (higher priority)
        critical_condition = self._conditions[FailsafeType.CRITICAL_BATTERY]
        if critical_condition.enabled and battery_percent <= critical_condition.threshold:
            return self._trigger_failsafe(
                FailsafeType.CRITICAL_BATTERY,
                f"Critical battery: {battery_percent:.1f}%",
                {'battery_percent': battery_percent},
            )

        # Check low battery
        low_condition = self._conditions[FailsafeType.LOW_BATTERY]
        if low_condition.enabled and battery_percent <= low_condition.threshold:
            return self._trigger_failsafe(
                FailsafeType.LOW_BATTERY,
                f"Low battery: {battery_percent:.1f}%",
                {'battery_percent': battery_percent},
            )

        # Battery OK - clear any battery-related failsafes
        self._clear_condition(FailsafeType.LOW_BATTERY)
        self._clear_condition(FailsafeType.CRITICAL_BATTERY)

        return None

    def check_signal(self, signal_strength: float, last_command_time: float) -> Optional[FailsafeAction]:
        """Check control signal and trigger failsafe if necessary."""
        condition = self._conditions[FailsafeType.SIGNAL_LOSS]
        if not condition.enabled:
            return None

        signal_ok = signal_strength > 0.3  # Minimum signal strength
        time_since_command = time.time() - last_command_time

        if not signal_ok or time_since_command > condition.timeout:
            if FailsafeType.SIGNAL_LOSS not in self._condition_start_times:
                self._condition_start_times[FailsafeType.SIGNAL_LOSS] = time.time()
                logger.warning("Control signal weak - monitoring...")

            elapsed = time.time() - self._condition_start_times[FailsafeType.SIGNAL_LOSS]

            if elapsed >= condition.timeout:
                return self._trigger_failsafe(
                    FailsafeType.SIGNAL_LOSS,
                    f"Signal lost for {elapsed:.1f}s",
                    {'signal_strength': signal_strength, 'time_since_command': time_since_command},
                )
        else:
            self._clear_condition(FailsafeType.SIGNAL_LOSS)

        return None

    def check_obstacle(self, distance: float) -> Optional[FailsafeAction]:
        """Check obstacle distance and trigger failsafe if necessary."""
        condition = self._conditions[FailsafeType.OBSTACLE_COLLISION_IMMINENT]
        if not condition.enabled:
            return None

        if distance < condition.threshold:
            return self._trigger_failsafe(
                FailsafeType.OBSTACLE_COLLISION_IMMINENT,
                f"Obstacle at {distance:.2f}m",
                {'distance': distance},
            )
        else:
            self._clear_condition(FailsafeType.OBSTACLE_COLLISION_IMMINENT)

        return None

    def check_geofence(
        self,
        position: Tuple[float, float],
        geofence_center: Tuple[float, float],
        geofence_radius: float,
    ) -> Optional[FailsafeAction]:
        """Check if drone is within geofence."""
        condition = self._conditions[FailsafeType.GEOFENCE_BREACH]
        if not condition.enabled:
            return None

        # Calculate distance from geofence center (simplified)
        from math import radians, sin, cos, sqrt, atan2

        lat1, lon1 = radians(position[0]), radians(position[1])
        lat2, lon2 = radians(geofence_center[0]), radians(geofence_center[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = 6371000 * c  # Earth radius in meters

        if distance > geofence_radius:
            return self._trigger_failsafe(
                FailsafeType.GEOFENCE_BREACH,
                f"Geofence breach - {distance:.0f}m from center (limit: {geofence_radius:.0f}m)",
                {'distance': distance, 'limit': geofence_radius},
            )
        else:
            self._clear_condition(FailsafeType.GEOFENCE_BREACH)

        return None

    def check_sensor(self, sensor_type: str, is_valid: bool) -> Optional[FailsafeAction]:
        """Check sensor health and trigger failsafe if critical sensor fails."""
        condition = self._conditions[FailsafeType.SENSOR_FAILURE]
        if not condition.enabled:
            return None

        critical_sensors = ['gps', 'imu', 'battery']

        if sensor_type in critical_sensors and not is_valid:
            return self._trigger_failsafe(
                FailsafeType.SENSOR_FAILURE,
                f"Critical sensor failure: {sensor_type}",
                {'sensor_type': sensor_type},
            )

        return None

    def _trigger_failsafe(
        self,
        failsafe_type: FailsafeType,
        reason: str,
        data: Optional[Dict] = None,
    ) -> FailsafeAction:
        """Trigger a failsafe condition."""
        condition = self._conditions[failsafe_type]

        # Check if already active
        if failsafe_type in self._active_failsafes:
            return condition.action

        event = FailsafeEvent(
            failsafe_type=failsafe_type,
            action=condition.action,
            triggered_at=time.time(),
            reason=reason,
            data=data,
        )

        self._active_failsafes[failsafe_type] = event
        self._event_history.append(event)

        # Update overall state
        self._update_state()

        logger.warning(f"FAILSAFE TRIGGERED: {failsafe_type.name} - {reason}")
        logger.warning(f"Action: {condition.action.value}")

        # Call registered callbacks
        if failsafe_type in self._callbacks:
            for callback in self._callbacks[failsafe_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Failsafe callback error: {e}")

        return condition.action

    def _clear_condition(self, failsafe_type: FailsafeType):
        """Clear a failsafe condition."""
        # Clear timing
        if failsafe_type in self._condition_start_times:
            del self._condition_start_times[failsafe_type]

        # Clear active failsafe
        if failsafe_type in self._active_failsafes:
            event = self._active_failsafes[failsafe_type]
            event.cleared_at = time.time()
            del self._active_failsafes[failsafe_type]

            self._update_state()
            logger.info(f"Failsafe cleared: {failsafe_type.name}")

    def _update_state(self):
        """Update overall failsafe state based on active conditions."""
        if not self._active_failsafes:
            self._state = FailsafeState.NORMAL
        else:
            # Find highest priority active failsafe
            min_priority = float('inf')
            for failsafe_type in self._active_failsafes:
                condition = self._conditions[failsafe_type]
                min_priority = min(min_priority, condition.priority)

            if min_priority == 1:
                self._state = FailsafeState.CRITICAL
            elif min_priority == 2:
                self._state = FailsafeState.ACTIVE
            else:
                self._state = FailsafeState.WARNING

    def get_active_failsafes(self) -> Dict[FailsafeType, FailsafeEvent]:
        """Get currently active failsafes."""
        return dict(self._active_failsafes)

    def get_highest_priority_action(self) -> FailsafeAction:
        """Get the action for the highest priority active failsafe."""
        if not self._active_failsafes:
            return FailsafeAction.NONE

        min_priority = float('inf')
        highest_action = FailsafeAction.NONE

        for failsafe_type in self._active_failsafes:
            condition = self._conditions[failsafe_type]
            if condition.priority < min_priority:
                min_priority = condition.priority
                highest_action = condition.action

        return highest_action

    def clear_all(self):
        """Clear all active failsafes (manual override)."""
        for failsafe_type in list(self._active_failsafes.keys()):
            self._clear_condition(failsafe_type)
        logger.info("All failsafes cleared manually")

    def is_failsafe_active(self, failsafe_type: Optional[FailsafeType] = None) -> bool:
        """Check if a specific or any failsafe is active."""
        if failsafe_type:
            return failsafe_type in self._active_failsafes
        return len(self._active_failsafes) > 0

    @property
    def state(self) -> FailsafeState:
        """Get current failsafe state."""
        return self._state

    @property
    def event_history(self) -> List[FailsafeEvent]:
        """Get failsafe event history."""
        return list(self._event_history)

    def get_status(self) -> Dict:
        """Get current failsafe system status."""
        return {
            'state': self._state.value,
            'active_failsafes': [
                {
                    'type': ft.name,
                    'action': event.action.value,
                    'reason': event.reason,
                    'triggered_at': event.triggered_at,
                }
                for ft, event in self._active_failsafes.items()
            ],
            'pending_conditions': list(self._condition_start_times.keys()),
            'total_events': len(self._event_history),
        }
