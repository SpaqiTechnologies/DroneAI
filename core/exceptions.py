"""
Custom exception classes for Drone AI application.
"""


class SensorConnectionLostException(Exception):
    """Exception raised when sensor connection is lost."""

    pass


class LowBatteryException(Exception):
    """Exception raised when battery level is critically low."""

    pass
