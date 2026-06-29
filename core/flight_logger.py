"""
Flight logger module for Drone AI application.
Records all sensor data, flight decisions, and events for post-flight analysis.
"""

import json
import time
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log entry severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Types of flight events that can be logged."""
    # Flight state events
    FLIGHT_START = "flight_start"
    FLIGHT_END = "flight_end"
    TAKEOFF = "takeoff"
    LANDING = "landing"
    HOVER = "hover"

    # Navigation events
    WAYPOINT_REACHED = "waypoint_reached"
    PATH_PLANNED = "path_planned"
    PATH_CHANGED = "path_changed"
    RETURN_TO_HOME_INITIATED = "return_to_home_initiated"
    HOME_REACHED = "home_reached"

    # Sensor events
    SENSOR_UPDATE = "sensor_update"
    SENSOR_ERROR = "sensor_error"
    SENSOR_CALIBRATION = "sensor_calibration"
    GPS_FIX_ACQUIRED = "gps_fix_acquired"
    GPS_FIX_LOST = "gps_fix_lost"

    # Safety events
    FAILSAFE_TRIGGERED = "failsafe_triggered"
    FAILSAFE_CLEARED = "failsafe_cleared"
    LOW_BATTERY_WARNING = "low_battery_warning"
    CRITICAL_BATTERY = "critical_battery"
    OBSTACLE_DETECTED = "obstacle_detected"
    OBSTACLE_AVOIDED = "obstacle_avoided"
    GEOFENCE_WARNING = "geofence_warning"
    GEOFENCE_BREACH = "geofence_breach"

    # Arming events
    ARMED = "armed"
    DISARMED = "disarmed"
    ARM_FAILED = "arm_failed"
    KILL_SWITCH = "kill_switch"

    # Landing events
    LANDED = "landed"
    LANDING_ABORTED = "landing_aborted"
    MARKER_DETECTED = "marker_detected"

    # System events
    SYSTEM_ERROR = "system_error"
    CONFIG_CHANGED = "config_changed"
    COMMAND_RECEIVED = "command_received"


@dataclass
class LogEntry:
    """Represents a single log entry."""
    timestamp: float
    event_type: str
    level: str
    message: str
    data: Optional[Dict[str, Any]] = None
    position: Optional[Tuple[float, float, float]] = None  # (lat, lon, alt)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type,
            'level': self.level,
            'message': self.message,
        }
        if self.data:
            result['data'] = self.data
        if self.position:
            result['position'] = {
                'latitude': self.position[0],
                'longitude': self.position[1],
                'altitude': self.position[2],
            }
        return result


@dataclass
class FlightSession:
    """Represents a complete flight session."""
    session_id: str
    start_time: float
    end_time: Optional[float]
    home_position: Tuple[float, float]
    entries: List[LogEntry]

    def duration(self) -> float:
        """Get flight duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class FlightLogger:
    """
    Records flight data for analysis and debugging.
    Supports real-time logging and post-flight export.
    """

    def __init__(self, log_dir: str = "flight_logs"):
        self._log_dir = log_dir
        self._current_session: Optional[FlightSession] = None
        self._entries: List[LogEntry] = []
        self._sensor_history: Dict[str, List[Tuple[float, Any]]] = {}
        self._max_sensor_history = 1000  # Max entries per sensor type
        self._auto_save_interval = 60.0  # Auto-save every 60 seconds
        self._last_auto_save = 0.0
        self._is_recording = False

        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def start_session(self, home_position: Tuple[float, float]) -> str:
        """Start a new flight session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._current_session = FlightSession(
            session_id=session_id,
            start_time=time.time(),
            end_time=None,
            home_position=home_position,
            entries=[],
        )

        self._entries = []
        self._sensor_history = {}
        self._is_recording = True
        self._last_auto_save = time.time()

        self.log_event(
            EventType.FLIGHT_START,
            "Flight session started",
            LogLevel.INFO,
            data={'home_position': home_position}
        )

        logger.info(f"Flight session started: {session_id}")
        return session_id

    def end_session(self) -> Optional[str]:
        """End current flight session and save log."""
        if not self._current_session:
            return None

        self.log_event(
            EventType.FLIGHT_END,
            "Flight session ended",
            LogLevel.INFO,
            data={'duration': self._current_session.duration()}
        )

        self._current_session.end_time = time.time()
        self._current_session.entries = self._entries.copy()
        self._is_recording = False

        # Save the session
        filepath = self._save_session()

        session_id = self._current_session.session_id
        self._current_session = None

        logger.info(f"Flight session ended: {session_id}")
        return filepath

    def log_event(
        self,
        event_type: EventType,
        message: str,
        level: LogLevel = LogLevel.INFO,
        data: Optional[Dict[str, Any]] = None,
        position: Optional[Tuple[float, float, float]] = None,
    ):
        """Log a flight event."""
        entry = LogEntry(
            timestamp=time.time(),
            event_type=event_type.value,
            level=level.value,
            message=message,
            data=data,
            position=position,
        )

        self._entries.append(entry)

        # Log to Python logger as well
        log_msg = f"[{event_type.value}] {message}"
        if level == LogLevel.DEBUG:
            logger.debug(log_msg)
        elif level == LogLevel.INFO:
            logger.info(log_msg)
        elif level == LogLevel.WARNING:
            logger.warning(log_msg)
        elif level == LogLevel.ERROR:
            logger.error(log_msg)
        elif level == LogLevel.CRITICAL:
            logger.critical(log_msg)

        # Auto-save check
        self._check_auto_save()

    def log_sensor_data(
        self,
        sensor_type: str,
        value: Any,
        is_valid: bool = True,
        position: Optional[Tuple[float, float, float]] = None,
    ):
        """Log sensor reading."""
        timestamp = time.time()

        # Store in sensor history
        if sensor_type not in self._sensor_history:
            self._sensor_history[sensor_type] = []

        history = self._sensor_history[sensor_type]
        history.append((timestamp, value))

        # Trim history if too large
        if len(history) > self._max_sensor_history:
            self._sensor_history[sensor_type] = history[-self._max_sensor_history:]

        # Log as event
        self.log_event(
            EventType.SENSOR_UPDATE,
            f"Sensor update: {sensor_type}",
            LogLevel.DEBUG,
            data={
                'sensor_type': sensor_type,
                'value': value if not hasattr(value, 'to_dict') else value.to_dict(),
                'is_valid': is_valid,
            },
            position=position,
        )

    def log_decision(
        self,
        decision: str,
        reason: str,
        data: Optional[Dict[str, Any]] = None,
        position: Optional[Tuple[float, float, float]] = None,
    ):
        """Log a flight decision made by the system."""
        self.log_event(
            EventType.PATH_CHANGED,
            f"Decision: {decision} - Reason: {reason}",
            LogLevel.INFO,
            data=data,
            position=position,
        )

    def log_failsafe(
        self,
        failsafe_type: str,
        reason: str,
        action: str,
        data: Optional[Dict[str, Any]] = None,
        position: Optional[Tuple[float, float, float]] = None,
    ):
        """Log a failsafe event."""
        self.log_event(
            EventType.FAILSAFE_TRIGGERED,
            f"Failsafe triggered: {failsafe_type} - {reason} - Action: {action}",
            LogLevel.WARNING,
            data={
                'failsafe_type': failsafe_type,
                'reason': reason,
                'action': action,
                **(data or {}),
            },
            position=position,
        )

    def log_error(
        self,
        error_type: str,
        message: str,
        exception: Optional[Exception] = None,
        data: Optional[Dict[str, Any]] = None,
        position: Optional[Tuple[float, float, float]] = None,
    ):
        """Log an error event."""
        error_data = {
            'error_type': error_type,
            **(data or {}),
        }
        if exception:
            error_data['exception'] = str(exception)
            error_data['exception_type'] = type(exception).__name__

        self.log_event(
            EventType.SYSTEM_ERROR,
            f"Error: {error_type} - {message}",
            LogLevel.ERROR,
            data=error_data,
            position=position,
        )

    def log_waypoint_reached(
        self,
        waypoint_index: int,
        waypoint: Tuple[float, float],
        remaining: int,
    ):
        """Log when a waypoint is reached."""
        self.log_event(
            EventType.WAYPOINT_REACHED,
            f"Waypoint {waypoint_index} reached, {remaining} remaining",
            LogLevel.INFO,
            data={
                'waypoint_index': waypoint_index,
                'waypoint': waypoint,
                'remaining_waypoints': remaining,
            },
            position=(waypoint[0], waypoint[1], 0.0),
        )

    def log_obstacle_detected(
        self,
        distance: float,
        direction: Optional[float] = None,
        position: Optional[Tuple[float, float, float]] = None,
    ):
        """Log obstacle detection."""
        self.log_event(
            EventType.OBSTACLE_DETECTED,
            f"Obstacle detected at {distance:.2f}m",
            LogLevel.WARNING,
            data={
                'distance': distance,
                'direction': direction,
            },
            position=position,
        )

    def get_entries(
        self,
        event_type: Optional[EventType] = None,
        level: Optional[LogLevel] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[LogEntry]:
        """Get filtered log entries."""
        entries = self._entries

        if event_type:
            entries = [e for e in entries if e.event_type == event_type.value]

        if level:
            entries = [e for e in entries if e.level == level.value]

        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]

        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        return entries

    def get_sensor_history(
        self,
        sensor_type: str,
        limit: Optional[int] = None,
    ) -> List[Tuple[float, Any]]:
        """Get historical sensor readings."""
        history = self._sensor_history.get(sensor_type, [])
        if limit:
            return history[-limit:]
        return history

    def get_statistics(self) -> Dict[str, Any]:
        """Get flight statistics."""
        if not self._current_session:
            return {}

        # Count events by type
        event_counts = {}
        for entry in self._entries:
            event_counts[entry.event_type] = event_counts.get(entry.event_type, 0) + 1

        # Count by level
        level_counts = {}
        for entry in self._entries:
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1

        return {
            'session_id': self._current_session.session_id,
            'duration': self._current_session.duration(),
            'total_entries': len(self._entries),
            'events_by_type': event_counts,
            'events_by_level': level_counts,
            'sensor_types_logged': list(self._sensor_history.keys()),
        }

    def _check_auto_save(self):
        """Check if auto-save should be triggered."""
        if not self._is_recording:
            return

        if time.time() - self._last_auto_save > self._auto_save_interval:
            self._auto_save()

    def _auto_save(self):
        """Perform auto-save of current session."""
        if not self._current_session:
            return

        self._current_session.entries = self._entries.copy()
        self._save_session(suffix="_autosave")
        self._last_auto_save = time.time()
        logger.debug("Flight log auto-saved")

    def _save_session(self, suffix: str = "") -> str:
        """Save session to file."""
        if not self._current_session:
            return ""

        filename = f"flight_{self._current_session.session_id}{suffix}.json"
        filepath = os.path.join(self._log_dir, filename)

        session_data = {
            'session_id': self._current_session.session_id,
            'start_time': self._current_session.start_time,
            'end_time': self._current_session.end_time,
            'home_position': self._current_session.home_position,
            'duration': self._current_session.duration(),
            'entries': [e.to_dict() for e in self._entries],
            'sensor_history': {
                sensor: [(t, v if not hasattr(v, 'to_dict') else v.to_dict())
                        for t, v in history]
                for sensor, history in self._sensor_history.items()
            },
            'statistics': self.get_statistics(),
        }

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        logger.info(f"Flight log saved to: {filepath}")
        return filepath

    def export_csv(self, filepath: Optional[str] = None) -> str:
        """Export sensor history to CSV format."""
        if not self._current_session:
            return ""

        if not filepath:
            filepath = os.path.join(
                self._log_dir,
                f"flight_{self._current_session.session_id}_sensors.csv"
            )

        # Collect all timestamps
        all_timestamps = set()
        for history in self._sensor_history.values():
            for t, _ in history:
                all_timestamps.add(t)

        sorted_timestamps = sorted(all_timestamps)
        sensor_types = list(self._sensor_history.keys())

        # Build CSV
        lines = ["timestamp,datetime," + ",".join(sensor_types)]

        # Create lookup for quick access
        sensor_lookup = {
            sensor: {t: v for t, v in history}
            for sensor, history in self._sensor_history.items()
        }

        for ts in sorted_timestamps:
            dt = datetime.fromtimestamp(ts).isoformat()
            values = [str(sensor_lookup.get(sensor, {}).get(ts, ""))
                     for sensor in sensor_types]
            lines.append(f"{ts},{dt}," + ",".join(values))

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))

        logger.info(f"Sensor data exported to: {filepath}")
        return filepath

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    @property
    def current_session_id(self) -> Optional[str]:
        """Get current session ID."""
        if self._current_session:
            return self._current_session.session_id
        return None
