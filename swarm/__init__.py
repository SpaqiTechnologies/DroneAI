"""
Swarm coordination module for multi-drone operations.

Provides formation flying, inter-drone communication,
and coordinated mission execution capabilities.
"""

from .drone_registry import DroneRegistry, DroneInfo, DroneStatus
from .inter_drone_comm import (
    InterDroneComm,
    DroneMessage,
    MessageType,
)
from .formation_controller import (
    FormationController,
    FormationType,
    FormationSlot,
)
from .swarm_collision import (
    SwarmCollisionAvoidance,
    CollisionThreat,
)
from .swarm_coordinator import (
    SwarmCoordinator,
    SwarmState,
    SwarmMission,
)

__all__ = [
    # Registry
    'DroneRegistry',
    'DroneInfo',
    'DroneStatus',
    # Communication
    'InterDroneComm',
    'DroneMessage',
    'MessageType',
    # Formation
    'FormationController',
    'FormationType',
    'FormationSlot',
    # Collision
    'SwarmCollisionAvoidance',
    'CollisionThreat',
    # Coordinator
    'SwarmCoordinator',
    'SwarmState',
    'SwarmMission',
]
