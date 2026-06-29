"""
Communication Module

Provides MAVLink protocol support for integration with flight controllers
like PX4 and ArduPilot.
"""

from .mavlink_protocol import (
    MAVLinkConnection,
    MAVLinkMessage,
    MAVLinkHeartbeat,
    MAVLinkCommand,
    MAVLinkTelemetry,
    ConnectionState,
    SystemStatus,
    FlightMode,
)

from .mavlink_commands import (
    MAVLinkCommandHandler,
    CommandResult,
    CommandType,
)

__all__ = [
    'MAVLinkConnection',
    'MAVLinkMessage',
    'MAVLinkHeartbeat',
    'MAVLinkCommand',
    'MAVLinkTelemetry',
    'ConnectionState',
    'SystemStatus',
    'FlightMode',
    'MAVLinkCommandHandler',
    'CommandResult',
    'CommandType',
]
