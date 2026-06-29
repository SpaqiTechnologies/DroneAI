"""
Stabilize Mode Handler

Manual attitude control with automatic leveling.
The pilot controls roll, pitch, yaw rate, and throttle.
When sticks are centered, the drone automatically levels.
"""

from typing import Tuple, TYPE_CHECKING

from .base_mode import FlightModeHandler, FlightMode, FlightCommand

if TYPE_CHECKING:
    from core.drone import Drone


class StabilizeModeHandler(FlightModeHandler):
    """
    Stabilize mode - manual control with auto-leveling.

    In this mode:
    - Roll/pitch inputs command target angles
    - Yaw input commands yaw rate
    - Throttle controls vertical speed
    - When inputs are zero, drone levels automatically
    """

    # Maximum angles in degrees
    MAX_ROLL = 35.0
    MAX_PITCH = 35.0
    MAX_YAW_RATE = 90.0  # deg/s

    def __init__(self, drone: 'Drone'):
        super().__init__(drone)
        self._roll_input = 0.0
        self._pitch_input = 0.0
        self._yaw_input = 0.0
        self._throttle_input = 0.5  # 0.0-1.0

    @property
    def mode(self) -> FlightMode:
        return FlightMode.STABILIZE

    def can_enter(self) -> Tuple[bool, str]:
        """Stabilize mode can always be entered."""
        return True, ""

    def _on_enter(self) -> Tuple[bool, str]:
        """Reset inputs on mode entry."""
        self._roll_input = 0.0
        self._pitch_input = 0.0
        self._yaw_input = 0.0
        self._throttle_input = 0.5
        return True, ""

    def update(self, dt: float) -> FlightCommand:
        """
        Update stabilize mode.

        Args:
            dt: Time delta since last update

        Returns:
            FlightCommand with attitude targets
        """
        # Calculate target angles from inputs
        target_roll = self._roll_input * self.MAX_ROLL
        target_pitch = self._pitch_input * self.MAX_PITCH
        target_yaw_rate = self._yaw_input * self.MAX_YAW_RATE

        # Calculate thrust from throttle
        # Map 0.0-1.0 to 0.0-1.0 thrust
        thrust = self._throttle_input

        return FlightCommand(
            roll=target_roll,
            pitch=target_pitch,
            yaw_rate=target_yaw_rate,
            thrust=thrust
        )

    def set_inputs(
        self,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        throttle: float = 0.5
    ) -> None:
        """
        Set manual control inputs.

        Args:
            roll: Roll input (-1.0 to 1.0)
            pitch: Pitch input (-1.0 to 1.0)
            yaw: Yaw input (-1.0 to 1.0)
            throttle: Throttle input (0.0 to 1.0)
        """
        self._roll_input = max(-1.0, min(1.0, roll))
        self._pitch_input = max(-1.0, min(1.0, pitch))
        self._yaw_input = max(-1.0, min(1.0, yaw))
        self._throttle_input = max(0.0, min(1.0, throttle))

    def get_status(self) -> dict:
        """Get mode status."""
        status = super().get_status()
        status.update({
            'roll_input': self._roll_input,
            'pitch_input': self._pitch_input,
            'yaw_input': self._yaw_input,
            'throttle_input': self._throttle_input,
            'max_roll': self.MAX_ROLL,
            'max_pitch': self.MAX_PITCH,
            'max_yaw_rate': self.MAX_YAW_RATE,
        })
        return status
