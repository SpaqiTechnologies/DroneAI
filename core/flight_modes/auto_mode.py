"""
Auto Mode Handler

Autonomous mission execution. The drone follows a pre-defined mission
consisting of waypoints and commands.
"""

from typing import Tuple, Optional, List, Callable, TYPE_CHECKING
import time
import math

from .base_mode import FlightModeHandler, FlightMode, FlightCommand

# Import from mission package
from core.mission import (
    Waypoint,
    WaypointAction,
    WaypointActionItem,
    Mission,
    MissionState,
)

if TYPE_CHECKING:
    from core.drone import Drone
    from core.trajectory.trajectory_generator import TrajectoryPoint


class AutoModeHandler(FlightModeHandler):
    """
    Auto mode - autonomous mission execution.

    In this mode:
    - Drone executes a pre-loaded mission
    - Navigates through waypoints sequentially
    - Executes actions at each waypoint
    - Supports pause/resume
    """

    def __init__(self, drone: 'Drone'):
        super().__init__(drone)
        self._mission: Optional[Mission] = None
        self._mission_state = MissionState.DRAFT
        self._current_waypoint_index = 0

        # Current waypoint tracking
        self._waypoint_entry_time = 0.0
        self._holding_at_waypoint = False
        self._distance_to_waypoint = 0.0

        # Trajectory
        self._trajectory: List['TrajectoryPoint'] = []
        self._trajectory_start_time = 0.0

        # Hold position (for pause)
        self._hold_position: Optional[Tuple[float, float, float]] = None

        # Statistics
        self._mission_start_time = 0.0
        self._waypoints_completed = 0

        # Callbacks
        self._waypoint_reached_callbacks: List[Callable[[int, Waypoint], None]] = []
        self._mission_complete_callbacks: List[Callable[[Mission], None]] = []

    @property
    def mode(self) -> FlightMode:
        return FlightMode.AUTO

    def can_enter(self) -> Tuple[bool, str]:
        """Check if auto mode can be entered."""
        # Require GPS
        if not self._drone.gps_sensor.is_valid():
            return False, "AUTO requires valid GPS fix"

        if self._drone.gps_sensor.get_satellites() < 4:
            return False, "AUTO requires at least 4 satellites"

        # Require armed
        if not self._drone.is_armed:
            return False, "AUTO requires armed state"

        # Require mission
        if not self._mission or len(self._mission.waypoints) == 0:
            return False, "AUTO requires a loaded mission"

        return True, ""

    def _on_enter(self) -> Tuple[bool, str]:
        """Initialize auto mode."""
        if self._mission_state == MissionState.PAUSED:
            # Resume from pause
            self._mission_state = MissionState.RUNNING
            return True, "Mission resumed"

        # Start fresh
        if self._mission_state not in [MissionState.READY, MissionState.DRAFT]:
            return False, "Mission not ready to start"

        if not self._mission:
            return False, "No mission loaded"

        self._mission_state = MissionState.RUNNING
        self._mission.state = MissionState.RUNNING
        self._mission_start_time = time.time()
        self._waypoints_completed = 0

        # Generate trajectory to first waypoint
        self._generate_waypoint_trajectory()

        return True, f"Mission '{self._mission.name}' started"

    def _on_exit(self) -> None:
        """Handle exit from auto mode."""
        if self._mission_state == MissionState.RUNNING:
            self._mission_state = MissionState.PAUSED
            if self._mission:
                self._mission.state = MissionState.PAUSED
            # Capture current position for resume
            gps_pos = self._drone.state_estimator.get_gps_position()
            if gps_pos:
                self._hold_position = gps_pos

    def update(self, dt: float) -> FlightCommand:
        """
        Update auto mode.

        Args:
            dt: Time delta since last update

        Returns:
            FlightCommand for mission execution
        """
        if self._mission_state != MissionState.RUNNING:
            return FlightCommand.hold()

        if not self._mission or len(self._mission.waypoints) == 0:
            return FlightCommand.hold()

        # Get current waypoint
        if self._current_waypoint_index >= len(self._mission.waypoints):
            # Mission complete
            self._complete_mission()
            return FlightCommand.hold()

        current_wp = self._mission.waypoints[self._current_waypoint_index]

        # Get current position
        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return FlightCommand.hold()

        # Calculate distance to waypoint
        wp_position = (current_wp.latitude, current_wp.longitude, current_wp.altitude)
        self._distance_to_waypoint = self._calculate_distance(gps_pos, wp_position)

        # Check if at waypoint
        if self._distance_to_waypoint <= current_wp.acceptance_radius:
            return self._handle_at_waypoint(current_wp, dt)

        # Navigate to waypoint
        return self._navigate_to_waypoint(current_wp, gps_pos, dt)

    def _navigate_to_waypoint(
        self,
        waypoint: Waypoint,
        current_pos: Tuple[float, float, float],
        dt: float
    ) -> FlightCommand:
        """Navigate to current waypoint."""
        # Reset holding flag
        self._holding_at_waypoint = False

        # Follow trajectory if available
        if self._trajectory:
            elapsed = time.time() - self._trajectory_start_time
            for point in self._trajectory:
                if point.time >= elapsed:
                    return FlightCommand(
                        position=point.position,
                        velocity=point.velocity,
                        yaw=point.yaw if waypoint.yaw is None else waypoint.yaw
                    )

        # Direct navigation
        return FlightCommand.position_command(
            waypoint.latitude,
            waypoint.longitude,
            waypoint.altitude,
            waypoint.yaw
        )

    def _handle_at_waypoint(
        self,
        waypoint: Waypoint,
        dt: float
    ) -> FlightCommand:
        """Handle arrival at waypoint."""
        # Mark entry time if just arrived
        if not self._holding_at_waypoint:
            self._holding_at_waypoint = True
            self._waypoint_entry_time = time.time()

            # Execute waypoint actions
            self._execute_waypoint_actions(waypoint)

            # Notify callbacks
            self._notify_waypoint_reached(self._current_waypoint_index, waypoint)

        # Check if pass-through
        if waypoint.pass_through:
            self._advance_to_next_waypoint()
            return self._get_next_waypoint_command()

        # Check hold time
        hold_elapsed = time.time() - self._waypoint_entry_time
        if hold_elapsed >= waypoint.hold_time:
            self._advance_to_next_waypoint()
            return self._get_next_waypoint_command()

        # Continue holding
        return FlightCommand.position_command(
            waypoint.latitude,
            waypoint.longitude,
            waypoint.altitude,
            waypoint.yaw
        )

    def _advance_to_next_waypoint(self) -> None:
        """Advance to next waypoint."""
        self._waypoints_completed += 1
        self._current_waypoint_index += 1
        self._holding_at_waypoint = False

        # Generate trajectory to next waypoint
        if self._mission and self._current_waypoint_index < len(self._mission.waypoints):
            self._generate_waypoint_trajectory()

    def _get_next_waypoint_command(self) -> FlightCommand:
        """Get command for next waypoint."""
        if not self._mission or self._current_waypoint_index >= len(self._mission.waypoints):
            return FlightCommand.hold()

        next_wp = self._mission.waypoints[self._current_waypoint_index]
        return FlightCommand.position_command(
            next_wp.latitude,
            next_wp.longitude,
            next_wp.altitude,
            next_wp.yaw
        )

    def _execute_waypoint_actions(self, waypoint: Waypoint) -> None:
        """Execute actions at waypoint."""
        for action_item in waypoint.actions:
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
            elif action == WaypointAction.SET_SPEED:
                # Speed is set via waypoint.speed
                pass
            elif action == WaypointAction.SET_GIMBAL:
                if hasattr(self._drone, 'camera_sensor'):
                    params = action_item.parameters
                    pitch = params.get('pitch', 0)
                    self._drone.camera_sensor.set_gimbal_pitch(pitch)
            elif action == WaypointAction.TRIGGER_SENSOR:
                # Custom sensor trigger
                pass

    def _complete_mission(self) -> None:
        """Handle mission completion."""
        self._mission_state = MissionState.COMPLETE
        if self._mission:
            self._mission.state = MissionState.COMPLETE
        self._notify_mission_complete()

        # RTL if configured
        if self._mission and self._mission.return_home:
            if hasattr(self._drone, 'flight_controller'):
                self._drone.flight_controller.return_to_home()

    def _generate_waypoint_trajectory(self) -> None:
        """Generate trajectory to current waypoint."""
        if not self._mission or self._current_waypoint_index >= len(self._mission.waypoints):
            return

        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return

        waypoint = self._mission.waypoints[self._current_waypoint_index]
        wp_position = (waypoint.latitude, waypoint.longitude, waypoint.altitude)

        # Use trajectory generator if available
        if hasattr(self._drone, 'flight_controller'):
            traj_gen = self._drone.flight_controller.trajectory_generator
            end_speed = 0.0 if not waypoint.pass_through else waypoint.speed
            self._trajectory = traj_gen.generate_trajectory(
                [gps_pos, wp_position],
                speeds=[waypoint.speed, end_speed],
                smooth=True
            )
            self._trajectory_start_time = time.time()

    def load_mission(self, mission: Mission) -> Tuple[bool, str]:
        """
        Load a mission for execution.

        Args:
            mission: Mission to load

        Returns:
            Tuple of (success, message)
        """
        if len(mission.waypoints) == 0:
            return False, "Mission has no waypoints"

        self._mission = mission
        self._mission_state = MissionState.READY
        self._mission.state = MissionState.READY
        self._current_waypoint_index = 0
        self._waypoints_completed = 0
        self._trajectory = []

        return True, f"Mission '{mission.name}' loaded with {len(mission.waypoints)} waypoints"

    def start_mission(self) -> Tuple[bool, str]:
        """Start the loaded mission."""
        if self._mission_state != MissionState.READY:
            return False, "Mission not ready to start"

        self._mission_state = MissionState.RUNNING
        if self._mission:
            self._mission.state = MissionState.RUNNING
        self._mission_start_time = time.time()
        self._generate_waypoint_trajectory()

        return True, "Mission started"

    def pause_mission(self) -> Tuple[bool, str]:
        """Pause current mission."""
        if self._mission_state != MissionState.RUNNING:
            return False, "Mission not running"

        self._mission_state = MissionState.PAUSED
        if self._mission:
            self._mission.state = MissionState.PAUSED
        gps_pos = self._drone.state_estimator.get_gps_position()
        if gps_pos:
            self._hold_position = gps_pos

        return True, "Mission paused"

    def resume_mission(self) -> Tuple[bool, str]:
        """Resume paused mission."""
        if self._mission_state != MissionState.PAUSED:
            return False, "Mission not paused"

        self._mission_state = MissionState.RUNNING
        if self._mission:
            self._mission.state = MissionState.RUNNING
        self._generate_waypoint_trajectory()

        return True, "Mission resumed"

    def abort_mission(self) -> Tuple[bool, str]:
        """Abort current mission."""
        self._mission_state = MissionState.ABORTED
        if self._mission:
            self._mission.state = MissionState.ABORTED
        self._trajectory = []
        return True, "Mission aborted"

    def skip_waypoint(self) -> Tuple[bool, str]:
        """Skip to next waypoint."""
        if self._mission_state != MissionState.RUNNING:
            return False, "Mission not running"

        self._advance_to_next_waypoint()
        return True, f"Skipped to waypoint {self._current_waypoint_index + 1}"

    def goto_waypoint(self, index: int) -> Tuple[bool, str]:
        """Jump to specific waypoint."""
        if not self._mission:
            return False, "No mission loaded"

        if index < 0 or index >= len(self._mission.waypoints):
            return False, f"Invalid waypoint index {index}"

        self._current_waypoint_index = index
        self._holding_at_waypoint = False
        self._generate_waypoint_trajectory()

        return True, f"Jumping to waypoint {index + 1}"

    def get_mission_state(self) -> MissionState:
        """Get current mission state."""
        return self._mission_state

    def get_current_waypoint_index(self) -> int:
        """Get current waypoint index."""
        return self._current_waypoint_index

    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Get current waypoint."""
        if not self._mission or self._current_waypoint_index >= len(self._mission.waypoints):
            return None
        return self._mission.waypoints[self._current_waypoint_index]

    def get_mission(self) -> Optional[Mission]:
        """Get loaded mission."""
        return self._mission

    def get_progress(self) -> float:
        """Get mission progress (0-1)."""
        if not self._mission or len(self._mission.waypoints) == 0:
            return 0.0
        return self._waypoints_completed / len(self._mission.waypoints)

    def get_mission_time(self) -> float:
        """Get elapsed mission time in seconds."""
        if self._mission_start_time == 0:
            return 0.0
        return time.time() - self._mission_start_time

    def add_waypoint_reached_callback(
        self,
        callback: Callable[[int, Waypoint], None]
    ) -> None:
        """Add callback for waypoint reached events."""
        self._waypoint_reached_callbacks.append(callback)

    def add_mission_complete_callback(
        self,
        callback: Callable[[Mission], None]
    ) -> None:
        """Add callback for mission complete events."""
        self._mission_complete_callbacks.append(callback)

    def _notify_waypoint_reached(self, index: int, waypoint: Waypoint) -> None:
        """Notify waypoint reached callbacks."""
        for callback in self._waypoint_reached_callbacks:
            try:
                callback(index, waypoint)
            except Exception:
                pass

    def _notify_mission_complete(self) -> None:
        """Notify mission complete callbacks."""
        if self._mission:
            for callback in self._mission_complete_callbacks:
                try:
                    callback(self._mission)
                except Exception:
                    pass

    def _calculate_distance(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float]
    ) -> float:
        """Calculate 3D distance between two GPS points."""
        lat1, lon1, alt1 = p1
        lat2, lon2, alt2 = p2

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        horizontal_dist = 6371000 * c

        vertical_dist = abs(alt2 - alt1)
        return math.sqrt(horizontal_dist ** 2 + vertical_dist ** 2)

    def get_status(self) -> dict:
        """Get mode status."""
        status = super().get_status()

        current_wp = self.get_current_waypoint()

        status.update({
            'mission_state': self._mission_state.value,
            'mission_name': self._mission.name if self._mission else None,
            'mission_id': self._mission.id if self._mission else None,
            'total_waypoints': len(self._mission.waypoints) if self._mission else 0,
            'current_waypoint': self._current_waypoint_index + 1,
            'current_waypoint_name': current_wp.name if current_wp else None,
            'waypoints_completed': self._waypoints_completed,
            'progress': self.get_progress(),
            'mission_time': self.get_mission_time(),
            'distance_to_waypoint': self._distance_to_waypoint,
            'holding_at_waypoint': self._holding_at_waypoint,
        })
        return status
