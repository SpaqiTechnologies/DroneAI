"""
Takeoff Package

Provides automated takeoff sequence management:
- Pre-arm safety checks
- Motor arming and spool-up
- Controlled liftoff and climb
- Hover stabilization
"""

from .takeoff_manager import (
    TakeoffManager,
    TakeoffState,
    TakeoffMode,
    TakeoffConfig,
    TakeoffResult,
)

__all__ = [
    'TakeoffManager',
    'TakeoffState',
    'TakeoffMode',
    'TakeoffConfig',
    'TakeoffResult',
]
