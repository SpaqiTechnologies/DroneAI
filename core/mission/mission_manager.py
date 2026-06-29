"""
Mission Manager Module

Manages mission execution, including loading, starting, pausing, and monitoring.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Callable, Tuple, TYPE_CHECKING
import time

from .mission import Mission, MissionState
from .waypoint import Waypoint, WaypointAction
from .mission_validator import MissionValidator, ValidationResult
from .mission_storage import MissionStorage

if TYPE_CHECKING:
    from core.drone import Drone
    from core.flight_controller import FlightController


class MissionExecutionState(Enum):
    """State of mission execution."""
    IDLE = "idle"              # No mission loaded
    LOADED = "loaded"          # Mission loaded, not started
    RUNNING = "running"        # Executing mission
    PAUSED = "paused"          # Paused mid-execution
    COMPLETING = "completing"  # Finishing last waypoint
    COMPLETE = "complete"      # Mission completed
    ABORTED = "aborted"        # Aborted by user
    FAILED = "failed"          # Failed due to error


@dataclass
class MissionProgress:
    """Progress information for mission execution."""
    state: MissionExecutionState
    current_waypoint: int
    total_waypoints: int
    waypoints_completed: int
    distance_to_waypoint: float
    total_distance_remaining: float
    elapsed_time: float
    estimated_time_remaining: float
    battery_used: float
    mission_name: str

    @property
    def progress_percent(self) -> float:
        """Get completion percentage."""
        if self.total_waypoints == 0:
            return 0.0
        return (self.waypoints_completed / self.total_waypoints) * 100

    def to_dict(self) -> dict:
        return {
            'state': self.state.value,
            'current_waypoint': self.current_waypoint,
            'total_waypoints': self.total_waypoints,
            'waypoints_completed': self.waypoints_completed,
            'progress_percent': self.progress_percent,
            'distance_to_waypoint': self.distance_to_waypoint,
            'total_distance_remaining': self.total_distance_remaining,
            'elapsed_time': self.elapsed_time,
            'estimated_time_remaining': self.estimated_time_remaining,
            'battery_used': self.battery_used,
            'mission_name': self.mission_name,
        }


class MissionManager:
    """
    Manages mission execution on a drone.

    Handles:
    - Mission loading and validation
    - Mission execution control (start, pause, resume, abort)
    - Waypoint tracking and transitions
    - Progress monitoring
    - Action execution at waypoints
    """

    def __init__(
        self,
        drone: 'Drone',
        flight_controller: Optional['FlightController'] = None,
        storage_dir: Optional[str] = None
    ):
        """
        Initialize mission manager.

        Args:
            drone: Drone instance
            flight_controller: Optional flight controller (uses drone.flight_controller if not provided)
            storage_dir: Optional directory for mission storage
        """
        self._drone = drone
        self._flight_controller = flight_controller or getattr(drone, 'flight_controller', None)

        # Mission state
        self._mission: Optional[Mission] = None
        self._state = MissionExecutionState.IDLE
        self._current_waypoint_index = 0
        self._waypoints_completed = 0

        # Waypoint tracking
        self._at_waypoint = False
        self._waypoint_arrival_time = 0.0
        self._distance_to_waypoint = 0.0

        # Timing
        self._mission_start_time = 0.0
        self._pause_time = 0.0
        self._total_pause_duration = 0.0

        # Battery tracking
        self._start_battery = 100.0

        # Components
        self._validator = MissionValidator()
        self._storage = MissionStorage(storage_dir) if storage_dir else MissionStorage()

        # Callbacks
        self._waypoint_reached_callbacks: List[Callable[[int, Waypoint], None]] = []
        self._mission_complete_callbacks: List[Callable[[Mission], None]] = []
        self._state_change_callbacks: List[Callable[[MissionExecutionState], None]] = []

    @property
    def mission(self) -> Optional[Mission]:
        """Get current mission."""
        return self._mission

    @property
    def state(self) -> MissionExecutionState:
        """Get execution state."""
        return self._state

    @property
    def current_waypoint_index(self) -> int:
        """Get current waypoint index."""
        return self._current_waypoint_index

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        """Get current waypoint."""
        if self._mission and 0 <= self._current_waypoint_index < len(self._mission.waypoints):
            return self._mission.waypoints[self._current_waypoint_index]
        return None

    @property
    def is_running(self) -> bool:
        """Check if mission is running."""
        return self._state == MissionExecutionState.RUNNING

    @property
    def is_paused(self) -> bool:
        """Check if mission is paused."""
        return self._state == MissionExecutionState.PAUSED

    def load_mission(self, mission: Mission) -> Tuple[bool, str]:
        """
        Load a mission for execution.

        Args:
            mission: Mission to load

        Returns:
            Tuple of (success, message)
        """
        # Validate mission
        result = self._validator.validate(mission, self._drone)
        if not result.is_valid:
            errors = "; ".join(e.message for e in result.errors)
            return False, f"Mission validation failed: {errors}"

        self._mission = mission
        self._mission.state = MissionState.READY
        self._state = MissionExecutionState.LOADED
        self._current_waypoint_index = 0
        self._waypoints_completed = 0

        self._notify_state_change()
        return True, f"Mission '{mission.name}' loaded ({len(mission)} waypoints)"

    def load_mission_by_id(self, mission_id: str) -> Tuple[bool, str]:
        """
        Load a mission from storage by ID.

        Args:
            mission_id: Mission ID to load

        Returns:
            Tuple of (success, message)
        """
        mission = self._storage.load(mission_id)
        if mission is None:
            return False, f"Mission {mission_id} not found"
        return self.load_mission(mission)

    def validate_mission(self, mission: Optional[Mission] = None) -> ValidationResult:
        """
        Validate a mission (current or provided).

        Args:
            mission: Optional mission to validate (uses current if not provided)

        Returns:
            ValidationResult
        """
        target = mission or self._mission
        if target is None:
            result = ValidationResult(is_valid=False)
            return result

        return self._validator.validate(target, self._drone)

    def start(self) -> Tuple[bool, str]:
        """
        Start mission execution.

        Returns:
            Tuple of (success, message)
        """
        if self._mission is None:
            return False, "No mission loaded"

        if self._state not in [MissionExecutionState.LOADED, MissionExecutionState.PAUSED]:
            return False, f"Cannot start mission from {self._state.value} state"

        # Check drone is ready
        if not self._drone.is_armed:
            return False, "Drone must be armed to start mission"

        # Resume from pause
        if self._state == MissionExecutionState.PAUSED:
            self._total_pause_duration += time.time() - self._pause_time
            self._state = MissionExecutionState.RUNNING
            self._mission.state = MissionState.RUNNING
            self._notify_state_change()
            return True, "Mission resumed"

        # Start fresh
        self._state = MissionExecutionState.RUNNING
        self._mission.state = MissionState.RUNNING
        self._mission_start_time = time.time()
        self._total_pause_duration = 0.0
        self._start_battery = self._drone.battery_level
        self._at_waypoint = False

        # Navigate to first waypoint
        self._navigate_to_waypoint(self._current_waypoint_index)

        self._notify_state_change()
        return True, f"Mission '{self._mission.name}' started"

    def pause(self) -> Tuple[bool, str]:
        """
        Pause mission execution.

        Returns:
            Tuple of (success, message)
        """
        if self._state != MissionExecutionState.RUNNING:
            return False, "Mission is not running"

        self._state = MissionExecutionState.PAUSED
        self._mission.state = MissionState.PAUSED
        self._pause_time = time.time()

        # Tell flight controller to hold position
        if self._flight_controller:
            self._flight_controller.hold()

        self._notify_state_change()
        return True, "Mission paused"

    def resume(self) -> Tuple[bool, str]:
        """
        Resume paused mission.

        Returns:
            Tuple of (success, message)
        """
        return self.start()  # start() handles resume from paused state

    def abort(self) -> Tuple[bool, str]:
        """
        Abort mission execution.

        Returns:
            Tuple of (success, message)
        """
        if self._state not in [MissionExecutionState.RUNNING, MissionExecutionState.PAUSED]:
            return False, "No active mission to abort"

        self._state = MissionExecutionState.ABORTED
        if self._mission:
            self._mission.state = MissionState.ABORTED

        # Tell flight controller to hold
        if self._flight_controller:
            self._flight_controller.hold()

        self._notify_state_change()
        return True, "Mission aborted"

    def skip_waypoint(self) -> Tuple[bool, str]:
        """
        Skip to next waypoint.

        Returns:
            Tuple of (success, message)
        """
        if self._state != MissionExecutionState.RUNNING:
            return False, "Mission is not running"

        if not self._mission:
            return False, "No mission loaded"

        if self._current_waypoint_index >= len(self._mission.waypoints) - 1:
            return False, "Already at last waypoint"

        self._advance_to_next_waypoint()
        return True, f"Skipped to waypoint {self._current_waypoint_index + 1}"

    def goto_waypoint(self, index: int) -> Tuple[bool, str]:
        """
        Jump to a specific waypoint.

        Args:
            index: Waypoint index to jump to

        Returns:
            Tuple of (success, message)
        """
        if self._state != MissionExecutionState.RUNNING:
            return False, "Mission is not running"

        if not self._mission:
            return False, "No mission loaded"

        if index < 0 or index >= len(self._mission.waypoints):
            return False, f"Invalid waypoint index: {index}"

        self._current_waypoint_index = index
        self._at_waypoint = False
        self._navigate_to_waypoint(index)

        return True, f"Jumping to waypoint {index + 1}"

    def update(self, dt: float) -> None:
        """
        Update mission execution.

        Should be called every control loop iteration.

        Args:
            dt: Time delta since last update
        """
        if self._state != MissionExecutionState.RUNNING:
            return

        if not self._mission or not self._mission.waypoints:
            return

        # Check if we've completed all waypoints
        if self._current_waypoint_index >= len(self._mission.waypoints):
            self._complete_mission()
            return

        current_wp = self._mission.waypoints[self._current_waypoint_index]

        # Update distance to waypoint
        gps_pos = self._drone.state_estimator.get_gps_position()
        if gps_pos:
            self._distance_to_waypoint = current_wp.distance_to_position(
                gps_pos[0], gps_pos[1], gps_pos[2]
            )

            # Check if arrived at waypoint
            if self._distance_to_waypoint <= current_wp.acceptance_radius:
                self._handle_waypoint_arrival(current_wp)
            else:
                self._at_waypoint = False

    def _navigate_to_waypoint(self, index: int) -> None:
        """Navigate to a waypoint."""
        if not self._mission or index >= len(self._mission.waypoints):
            return

        wp = self._mission.waypoints[index]

        if self._flight_controller:
            self._flight_controller.goto(
                wp.latitude,
                wp.longitude,
                wp.altitude,
                speed=wp.speed,
                yaw=wp.yaw
            )

    def _handle_waypoint_arrival(self, waypoint: Waypoint) -> None:
        """Handle arrival at a waypoint."""
        if not self._at_waypoint:
            # Just arrived
            self._at_waypoint = True
            self._waypoint_arrival_time = time.time()

            # Execute waypoint actions
            self._execute_waypoint_actions(waypoint)

            # Notify callbacks
            self._notify_waypoint_reached(self._current_waypoint_index, waypoint)

        # Check if we should move on
        if waypoint.pass_through:
            # Don't wait, continue immediately
            self._advance_to_next_waypoint()
        else:
            # Check hold time
            hold_elapsed = time.time() - self._waypoint_arrival_time
            if hold_elapsed >= waypoint.hold_time:
                self._advance_to_next_waypoint()

    def _advance_to_next_waypoint(self) -> None:
        """Advance to the next waypoint."""
        self._waypoints_completed += 1
        self._current_waypoint_index += 1
        self._at_waypoint = False

        if self._current_waypoint_index < len(self._mission.waypoints):
            self._navigate_to_waypoint(self._current_waypoint_index)
        else:
            self._complete_mission()

    def _execute_waypoint_actions(self, waypoint: Waypoint) -> None:
        """Execute actions at a waypoint."""
        for action_item in waypoint.actions:
            self._execute_action(action_item)

    def _execute_action(self, action_item) -> None:
        """Execute a single waypoint action."""
        action = action_item.action

        if action == WaypointAction.TAKE_PHOTO:
            if hasattr(self._drone, 'camera_sensor'):
                self._drone.camera_sensor.take_snapshot()

        elif action == WaypointAction.START_VIDEO:
            if hasattr(self._drone, 'camera_sensor'):
                self._drone.camera_sensor.start_recording()

        elif action == WaypointAction.STOP_VIDEO:
            if hasattr(self._drone, 'camera_sensor'):
                self._drone.camera_sensor.stop_recording()

        elif action == WaypointAction.SET_GIMBAL:
            if hasattr(self._drone, 'camera_sensor'):
                self._drone.camera_sensor.set_gimbal(
                    pitch=action_item.param1,
                    roll=action_item.param2,
                    yaw=action_item.param3
                )

        elif action == WaypointAction.DELAY:
            # Delay is handled by hold_time, but could add extra delay here
            pass

    def _complete_mission(self) -> None:
        """Handle mission completion."""
        self._state = MissionExecutionState.COMPLETE
        if self._mission:
            self._mission.state = MissionState.COMPLETE

        # Check if RTL is requested
        if self._mission and self._mission.return_home:
            if self._flight_controller:
                self._flight_controller.return_to_home()

        self._notify_state_change()
        self._notify_mission_complete()

    def get_progress(self) -> MissionProgress:
        """Get current mission progress."""
        elapsed = 0.0
        if self._mission_start_time > 0:
            elapsed = time.time() - self._mission_start_time - self._total_pause_duration

        total_waypoints = len(self._mission.waypoints) if self._mission else 0

        # Calculate remaining distance
        remaining_distance = self._distance_to_waypoint
        if self._mission:
            for i in range(self._current_waypoint_index + 1, len(self._mission.waypoints)):
                if i > 0:
                    remaining_distance += self._mission.waypoints[i - 1].distance_to(
                        self._mission.waypoints[i]
                    )

        # Estimate remaining time
        avg_speed = 5.0  # Default
        if self._mission:
            avg_speed = self._mission.default_speed
        time_remaining = remaining_distance / avg_speed if avg_speed > 0 else 0

        return MissionProgress(
            state=self._state,
            current_waypoint=self._current_waypoint_index + 1,
            total_waypoints=total_waypoints,
            waypoints_completed=self._waypoints_completed,
            distance_to_waypoint=self._distance_to_waypoint,
            total_distance_remaining=remaining_distance,
            elapsed_time=elapsed,
            estimated_time_remaining=time_remaining,
            battery_used=self._start_battery - self._drone.battery_level,
            mission_name=self._mission.name if self._mission else "",
        )

    # Callback management
    def add_waypoint_reached_callback(self, callback: Callable[[int, Waypoint], None]) -> None:
        """Add callback for waypoint reached events."""
        self._waypoint_reached_callbacks.append(callback)

    def add_mission_complete_callback(self, callback: Callable[[Mission], None]) -> None:
        """Add callback for mission complete events."""
        self._mission_complete_callbacks.append(callback)

    def add_state_change_callback(self, callback: Callable[[MissionExecutionState], None]) -> None:
        """Add callback for state change events."""
        self._state_change_callbacks.append(callback)

    def _notify_waypoint_reached(self, index: int, waypoint: Waypoint) -> None:
        for callback in self._waypoint_reached_callbacks:
            try:
                callback(index, waypoint)
            except Exception:
                pass

    def _notify_mission_complete(self) -> None:
        if self._mission:
            for callback in self._mission_complete_callbacks:
                try:
                    callback(self._mission)
                except Exception:
                    pass

    def _notify_state_change(self) -> None:
        for callback in self._state_change_callbacks:
            try:
                callback(self._state)
            except Exception:
                pass

    # Storage access
    def save_mission(self, mission: Optional[Mission] = None) -> Tuple[bool, str]:
        """Save mission to storage."""
        target = mission or self._mission
        if target is None:
            return False, "No mission to save"

        path = self._storage.save(target)
        return True, f"Mission saved to {path}"

    def list_saved_missions(self) -> List[dict]:
        """List all saved missions."""
        return self._storage.list_missions()

    def delete_saved_mission(self, mission_id: str) -> Tuple[bool, str]:
        """Delete a saved mission."""
        if self._storage.delete(mission_id):
            return True, f"Mission {mission_id} deleted"
        return False, f"Mission {mission_id} not found"
