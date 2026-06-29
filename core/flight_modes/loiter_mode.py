"""
Loiter Mode Handler

GPS-based position hold. The drone maintains its position and altitude
using GPS and barometer feedback. Manual inputs nudge the position.
"""

from typing import Tuple, Optional, TYPE_CHECKING
import time
import math

from .base_mode import FlightModeHandler, FlightMode, FlightCommand

if TYPE_CHECKING:
    from core.drone import Drone


class LoiterModeHandler(FlightModeHandler):
    """
    Loiter mode - GPS position hold.

    In this mode:
    - Drone holds current GPS position and altitude
    - Manual inputs move the target position
    - Position is maintained using GPS feedback
    - Requires valid GPS fix
    """

    # Position hold parameters
    POSITION_GAIN = 0.5       # m/s per meter of error
    MAX_HORIZONTAL_SPEED = 5.0  # m/s
    MAX_VERTICAL_SPEED = 2.0    # m/s
    HOLD_RADIUS = 1.0           # meters

    # Manual control parameters
    NUDGE_RATE = 2.0            # m/s when stick is fully deflected

    def __init__(self, drone: 'Drone'):
        super().__init__(drone)
        self._hold_position: Optional[Tuple[float, float, float]] = None
        self._hold_yaw: Optional[float] = None

        # Manual input
        self._lat_input = 0.0    # -1 to 1
        self._lon_input = 0.0    # -1 to 1
        self._alt_input = 0.0    # -1 to 1
        self._yaw_input = 0.0    # -1 to 1

        # Position tracking
        self._last_update_time = time.time()

    @property
    def mode(self) -> FlightMode:
        return FlightMode.LOITER

    def can_enter(self) -> Tuple[bool, str]:
        """Check if loiter mode can be entered."""
        # Require GPS
        if not self._drone.gps_sensor.is_valid():
            return False, "LOITER requires valid GPS fix"

        if self._drone.gps_sensor.satellites < 4:
            return False, "LOITER requires at least 4 satellites"

        return True, ""

    def _on_enter(self) -> Tuple[bool, str]:
        """Capture current position as hold point."""
        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return False, "Could not get GPS position"

        self._hold_position = (gps_pos[0], gps_pos[1], gps_pos[2])

        # Get current yaw
        state = self._drone.state_estimator.get_state()
        if state:
            self._hold_yaw = state.yaw
        else:
            self._hold_yaw = 0.0

        # Reset inputs
        self._lat_input = 0.0
        self._lon_input = 0.0
        self._alt_input = 0.0
        self._yaw_input = 0.0

        self._last_update_time = time.time()

        return True, f"Holding at ({gps_pos[0]:.6f}, {gps_pos[1]:.6f}, {gps_pos[2]:.1f}m)"

    def update(self, dt: float) -> FlightCommand:
        """
        Update loiter mode.

        Args:
            dt: Time delta since last update

        Returns:
            FlightCommand for position hold
        """
        if not self._hold_position:
            return FlightCommand.hold()

        # Apply manual nudge to hold position
        if abs(self._lat_input) > 0.1 or abs(self._lon_input) > 0.1:
            self._apply_nudge(dt)

        # Apply altitude change
        if abs(self._alt_input) > 0.1:
            new_alt = self._hold_position[2] + self._alt_input * self.MAX_VERTICAL_SPEED * dt
            new_alt = max(0.0, new_alt)  # Don't go below ground
            self._hold_position = (
                self._hold_position[0],
                self._hold_position[1],
                new_alt
            )

        # Apply yaw change
        if abs(self._yaw_input) > 0.1 and self._hold_yaw is not None:
            self._hold_yaw += self._yaw_input * 45.0 * dt  # 45 deg/s max
            self._hold_yaw = self._hold_yaw % 360

        # Get current position
        gps_pos = self._drone.state_estimator.get_gps_position()
        if not gps_pos:
            return FlightCommand.hold()

        # Calculate position error
        lat_error = self._hold_position[0] - gps_pos[0]
        lon_error = self._hold_position[1] - gps_pos[1]
        alt_error = self._hold_position[2] - gps_pos[2]

        # Convert lat/lon error to meters (approximate)
        lat_error_m = lat_error * 111320  # ~111km per degree latitude
        lon_error_m = lon_error * 111320 * math.cos(math.radians(gps_pos[0]))

        # Calculate velocity commands (simple P controller)
        vn = lat_error_m * self.POSITION_GAIN
        ve = lon_error_m * self.POSITION_GAIN
        vd = -alt_error * self.POSITION_GAIN  # Down is positive in NED

        # Limit speeds
        horizontal_speed = math.sqrt(vn**2 + ve**2)
        if horizontal_speed > self.MAX_HORIZONTAL_SPEED:
            scale = self.MAX_HORIZONTAL_SPEED / horizontal_speed
            vn *= scale
            ve *= scale

        vd = max(-self.MAX_VERTICAL_SPEED, min(self.MAX_VERTICAL_SPEED, vd))

        return FlightCommand(
            position=self._hold_position,
            velocity=(vn, ve, vd),
            yaw=self._hold_yaw
        )

    def _apply_nudge(self, dt: float) -> None:
        """Apply manual nudge to hold position."""
        if not self._hold_position:
            return

        # Convert stick input to velocity (m/s)
        vn = self._lat_input * self.NUDGE_RATE
        ve = self._lon_input * self.NUDGE_RATE

        # Convert to lat/lon change
        dlat = (vn * dt) / 111320
        dlon = (ve * dt) / (111320 * math.cos(math.radians(self._hold_position[0])))

        self._hold_position = (
            self._hold_position[0] + dlat,
            self._hold_position[1] + dlon,
            self._hold_position[2]
        )

    def set_hold_position(
        self,
        lat: float,
        lon: float,
        alt: float
    ) -> Tuple[bool, str]:
        """Set a new hold position."""
        self._hold_position = (lat, lon, alt)
        return True, f"Hold position set to ({lat:.6f}, {lon:.6f}, {alt:.1f}m)"

    def set_inputs(
        self,
        lat: float = 0.0,
        lon: float = 0.0,
        alt: float = 0.0,
        yaw: float = 0.0
    ) -> None:
        """
        Set manual control inputs for nudging.

        Args:
            lat: Latitude nudge (-1.0 to 1.0)
            lon: Longitude nudge (-1.0 to 1.0)
            alt: Altitude nudge (-1.0 to 1.0)
            yaw: Yaw nudge (-1.0 to 1.0)
        """
        self._lat_input = max(-1.0, min(1.0, lat))
        self._lon_input = max(-1.0, min(1.0, lon))
        self._alt_input = max(-1.0, min(1.0, alt))
        self._yaw_input = max(-1.0, min(1.0, yaw))

    def get_hold_position(self) -> Optional[Tuple[float, float, float]]:
        """Get current hold position."""
        return self._hold_position

    def get_status(self) -> dict:
        """Get mode status."""
        status = super().get_status()
        status.update({
            'hold_position': self._hold_position,
            'hold_yaw': self._hold_yaw,
            'lat_input': self._lat_input,
            'lon_input': self._lon_input,
            'alt_input': self._alt_input,
            'yaw_input': self._yaw_input,
        })
        return status
