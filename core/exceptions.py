"""
Custom exception classes for Drone AI application.
"""


class SensorConnectionLostException(Exception):
    """Exception raised when sensor connection is lost."""

    pass


class LowBatteryException(Exception):
    """Exception raised when battery level is critically low."""

    pass


class GPSLostException(Exception):
    """Exception raised when GPS signal is lost."""

    pass


class SignalLostException(Exception):
    """Exception raised when control signal is lost."""

    pass


class FailsafeException(Exception):
    """Base exception for failsafe conditions."""

    def __init__(self, message: str, failsafe_type: str, action: str):
        super().__init__(message)
        self.failsafe_type = failsafe_type
        self.action = action


class GeofenceBreachException(Exception):
    """Exception raised when drone breaches geofence boundary."""

    pass


class ArmingException(Exception):
    """Exception raised when arming fails."""

    def __init__(self, message: str, failed_checks: list = None):
        super().__init__(message)
        self.failed_checks = failed_checks or []


class ObstacleDetectedException(Exception):
    """Exception raised when obstacle is detected in flight path."""

    def __init__(self, message: str, distance: float):
        super().__init__(message)
        self.distance = distance


class SensorReadingInvalidException(Exception):
    """Exception raised when sensor reading is outside valid range."""

    pass


class SensorSignalWeakException(Exception):
    """Exception raised when sensor signal quality is too low."""

    pass
