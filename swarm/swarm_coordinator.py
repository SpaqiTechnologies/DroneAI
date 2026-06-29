"""
Swarm coordinator for multi-drone orchestration.

Manages drone registration, formation flying,
mission execution, and coordinated behaviors.
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Callable, Any, Tuple
)
import logging

from .drone_registry import (
    DroneRegistry,
    DroneInfo,
    DroneStatus,
    DronePosition,
    DroneVelocity,
)
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
    ThreatLevel,
)

logger = logging.getLogger(__name__)


class SwarmState(Enum):
    """State of the swarm."""
    IDLE = "idle"
    FORMING = "forming"
    READY = "ready"
    EXECUTING = "executing"
    PAUSED = "paused"
    RETURNING = "returning"
    LANDING = "landing"
    EMERGENCY = "emergency"


@dataclass
class SwarmMission:
    """Mission for the swarm to execute."""
    mission_id: str
    name: str = ""
    waypoints: List[Dict] = field(default_factory=list)
    formation: FormationType = FormationType.V_FORMATION
    spacing: float = 10.0
    altitude: float = 50.0
    speed: float = 5.0
    current_waypoint: int = 0
    completed: bool = False


@dataclass
class SwarmConfig:
    """Configuration for swarm operations."""
    update_rate: float = 10.0  # Hz
    formation_tolerance: float = 2.0  # meters
    waypoint_tolerance: float = 3.0  # meters
    min_battery: float = 20.0  # percent
    comm_timeout: float = 5.0  # seconds
    auto_collision_avoid: bool = True
    auto_return_on_low_battery: bool = True


class SwarmCoordinator:
    """
    Central coordinator for swarm operations.

    Responsibilities:
    - Manage drone registration and status
    - Coordinate formation flying
    - Execute swarm missions
    - Handle inter-drone communication
    - Manage collision avoidance
    """

    def __init__(
        self,
        swarm_id: str = "swarm_1",
        config: Optional[SwarmConfig] = None,
    ):
        """
        Initialize swarm coordinator.

        Args:
            swarm_id: Unique identifier for this swarm
            config: Swarm configuration
        """
        self.swarm_id = swarm_id
        self._config = config or SwarmConfig()

        # Components
        self._registry = DroneRegistry()
        self._formation = FormationController()
        self._collision = SwarmCollisionAvoidance()
        self._comm: Optional[InterDroneComm] = None

        # State
        self._state = SwarmState.IDLE
        self._leader_id: Optional[str] = None
        self._mission: Optional[SwarmMission] = None
        self._lock = threading.RLock()

        # Control loop
        self._running = False
        self._update_thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_state_changed: List[Callable] = []
        self._on_formation_reached: List[Callable] = []
        self._on_waypoint_reached: List[Callable] = []
        self._on_mission_complete: List[Callable] = []
        self._on_emergency: List[Callable] = []

        # Setup registry callbacks
        self._setup_registry_callbacks()

    def _setup_registry_callbacks(self) -> None:
        """Setup callbacks for drone registry events."""
        self._registry.on_drone_added(self._on_drone_added)
        self._registry.on_drone_removed(self._on_drone_removed)
        self._registry.on_status_changed(self._on_drone_status_changed)
        self._registry.on_drone_lost(self._on_drone_lost)

    def start(self, enable_comm: bool = True) -> bool:
        """
        Start swarm coordinator.

        Args:
            enable_comm: Enable inter-drone communication

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        try:
            # Start registry monitoring
            self._registry.start_monitoring()

            # Start communication if enabled
            if enable_comm:
                self._comm = InterDroneComm(
                    drone_id=f"{self.swarm_id}_coordinator",
                )
                if self._comm.start():
                    self._setup_comm_handlers()
                else:
                    logger.warning("Communication start failed")

            # Start update loop
            self._running = True
            self._update_thread = threading.Thread(
                target=self._update_loop,
                daemon=True,
            )
            self._update_thread.start()

            logger.info(f"Swarm coordinator {self.swarm_id} started")
            return True

        except Exception as e:
            logger.error(f"Failed to start coordinator: {e}")
            return False

    def stop(self) -> None:
        """Stop swarm coordinator."""
        self._running = False

        if self._update_thread:
            self._update_thread.join(timeout=2.0)

        if self._comm:
            self._comm.stop()

        self._registry.stop_monitoring()

        logger.info(f"Swarm coordinator {self.swarm_id} stopped")

    def _setup_comm_handlers(self) -> None:
        """Setup message handlers for communication."""
        if not self._comm:
            return

        self._comm.on_message(
            MessageType.HEARTBEAT,
            self._handle_heartbeat,
        )
        self._comm.on_message(
            MessageType.STATE_UPDATE,
            self._handle_state_update,
        )
        self._comm.on_message(
            MessageType.EMERGENCY,
            self._handle_emergency_message,
        )
        self._comm.on_message(
            MessageType.COLLISION_WARNING,
            self._handle_collision_warning,
        )

    def _handle_heartbeat(self, msg: DroneMessage) -> None:
        """Handle heartbeat from a drone."""
        drone_id = msg.sender_id
        payload = msg.payload

        # Update registry
        if drone_id not in [
            d.drone_id for d in self._registry.get_all_drones()
        ]:
            self._registry.register(drone_id)

        # Update state
        position = None
        velocity = None

        if 'position' in payload:
            pos = payload['position']
            position = DronePosition(
                lat=pos.get('lat', 0),
                lon=pos.get('lon', 0),
                alt=pos.get('alt', 0),
                heading=pos.get('heading', 0),
            )

        if 'velocity' in payload:
            vel = payload['velocity']
            velocity = DroneVelocity(
                vx=vel.get('vx', 0),
                vy=vel.get('vy', 0),
                vz=vel.get('vz', 0),
                speed=vel.get('speed', 0),
            )

        status = None
        if 'status' in payload:
            try:
                status = DroneStatus(payload['status'])
            except ValueError:
                pass

        self._registry.update_state(
            drone_id,
            position=position,
            velocity=velocity,
            status=status,
            battery_percent=payload.get('battery'),
            signal_strength=payload.get('signal'),
        )

    def _handle_state_update(self, msg: DroneMessage) -> None:
        """Handle state update from a drone."""
        self._handle_heartbeat(msg)  # Same processing

    def _handle_emergency_message(self, msg: DroneMessage) -> None:
        """Handle emergency message."""
        reason = msg.payload.get('reason', 'unknown')
        drone_id = msg.sender_id

        logger.error(f"Emergency from {drone_id}: {reason}")

        self._set_state(SwarmState.EMERGENCY)
        self._notify_emergency(drone_id, reason)

    def _handle_collision_warning(self, msg: DroneMessage) -> None:
        """Handle collision warning from another drone."""
        payload = msg.payload
        logger.warning(
            f"Collision warning: {msg.sender_id} -> "
            f"{payload.get('other_drone_id')}"
        )

    def _update_loop(self) -> None:
        """Main update loop."""
        interval = 1.0 / self._config.update_rate

        while self._running:
            start_time = time.time()

            try:
                self._update()
            except Exception as e:
                logger.error(f"Update error: {e}")

            # Maintain update rate
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _update(self) -> None:
        """Single update cycle."""
        with self._lock:
            # Check collision avoidance
            if self._config.auto_collision_avoid:
                self._check_collisions()

            # Check battery levels
            if self._config.auto_return_on_low_battery:
                self._check_battery_levels()

            # State-specific updates
            if self._state == SwarmState.FORMING:
                self._update_formation()
            elif self._state == SwarmState.EXECUTING:
                self._update_mission()

    def _check_collisions(self) -> None:
        """Check for inter-drone collisions."""
        drones = self._registry.get_flying_drones()

        for drone in drones:
            threats = self._collision.check_collisions(
                drone.drone_id,
                drone.position,
                drone.velocity,
                drones,
            )

            # Handle critical threats
            for threat in threats:
                if threat.threat_level == ThreatLevel.CRITICAL:
                    self._handle_critical_collision(
                        drone.drone_id,
                        threat,
                    )

    def _handle_critical_collision(
        self,
        drone_id: str,
        threat: CollisionThreat,
    ) -> None:
        """Handle critical collision threat."""
        logger.warning(
            f"Critical collision: {drone_id} <-> "
            f"{threat.other_drone_id}"
        )

        # Get avoidance command
        avoidance = self._collision.get_combined_avoidance(drone_id)

        # Send avoidance command via communication
        if self._comm:
            self._comm.send_command(
                drone_id,
                'avoid',
                {
                    'velocity': avoidance,
                    'duration': 2.0,
                },
            )

    def _check_battery_levels(self) -> None:
        """Check battery levels and trigger RTH if needed."""
        for drone in self._registry.get_flying_drones():
            if drone.battery_percent < self._config.min_battery:
                logger.warning(
                    f"Low battery on {drone.drone_id}: "
                    f"{drone.battery_percent:.0f}%"
                )
                self._send_return_command(drone.drone_id)

    def _send_return_command(self, drone_id: str) -> None:
        """Send return to home command."""
        if self._comm:
            self._comm.send_command(
                drone_id,
                'return_to_home',
                {'reason': 'low_battery'},
            )

    def _update_formation(self) -> None:
        """Update formation progress."""
        if not self._leader_id:
            return

        leader = self._registry.get_drone(self._leader_id)
        if not leader:
            return

        all_in_position = True
        tolerance = self._config.formation_tolerance

        for drone in self._registry.get_flying_drones():
            if drone.drone_id == self._leader_id:
                continue

            target = self._formation.get_target_position(
                drone.drone_id,
                leader.position.lat,
                leader.position.lon,
                leader.position.alt,
                leader.position.heading,
            )

            if target:
                dist = self._distance_to_target(
                    drone.position,
                    target,
                )
                if dist > tolerance:
                    all_in_position = False

        if all_in_position:
            self._set_state(SwarmState.READY)
            self._notify_formation_reached()

    def _update_mission(self) -> None:
        """Update mission progress."""
        if not self._mission or not self._leader_id:
            return

        leader = self._registry.get_drone(self._leader_id)
        if not leader:
            return

        # Check if at current waypoint
        if self._mission.current_waypoint >= len(self._mission.waypoints):
            self._mission.completed = True
            self._set_state(SwarmState.READY)
            self._notify_mission_complete()
            return

        waypoint = self._mission.waypoints[self._mission.current_waypoint]
        dist = self._distance_to_target(
            leader.position,
            (waypoint['lat'], waypoint['lon'], waypoint['alt']),
        )

        if dist < self._config.waypoint_tolerance:
            self._mission.current_waypoint += 1
            self._notify_waypoint_reached(
                self._mission.current_waypoint - 1
            )

    def _distance_to_target(
        self,
        pos: DronePosition,
        target: Tuple[float, float, float],
    ) -> float:
        """Calculate distance to target position."""
        import math

        lat1 = math.radians(pos.lat)
        lat2 = math.radians(target[0])
        dlon = math.radians(target[1] - pos.lon)
        dlat = lat2 - lat1

        # Haversine for horizontal
        a = (
            math.sin(dlat / 2) ** 2 +
            math.cos(lat1) * math.cos(lat2) *
            math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        horiz = 6371000 * c

        # Vertical
        vert = abs(pos.alt - target[2])

        return math.sqrt(horiz ** 2 + vert ** 2)

    # Public API

    def register_drone(
        self,
        drone_id: str,
        name: str = "",
    ) -> Optional[DroneInfo]:
        """Register a drone in the swarm."""
        return self._registry.register(drone_id, name)

    def unregister_drone(self, drone_id: str) -> bool:
        """Remove a drone from the swarm."""
        return self._registry.unregister(drone_id)

    def set_leader(self, drone_id: str) -> bool:
        """Set the swarm leader."""
        if self._registry.set_leader(drone_id):
            self._leader_id = drone_id
            return True
        return False

    def set_formation(
        self,
        formation_type: FormationType,
        spacing: float = 10.0,
        **kwargs,
    ) -> bool:
        """
        Set swarm formation.

        Args:
            formation_type: Type of formation
            spacing: Distance between drones
            **kwargs: Additional formation parameters
        """
        drones = self._registry.get_online_drones()
        if len(drones) < 2:
            logger.warning("Need at least 2 drones for formation")
            return False

        slots = self._formation.set_formation(
            formation_type,
            len(drones),
            spacing,
            **kwargs,
        )

        # Assign drones to slots
        for i, drone in enumerate(drones):
            self._formation.assign_drone_to_slot(
                drone.drone_id,
                slots[i].slot_id,
            )
            self._registry.set_formation_slot(
                drone.drone_id,
                slots[i].slot_id,
            )

        self._set_state(SwarmState.FORMING)

        # Broadcast formation update
        if self._comm:
            self._comm.broadcast(
                MessageType.FORMATION_UPDATE,
                {
                    'type': formation_type.value,
                    'spacing': spacing,
                    'slots': [
                        {
                            'slot_id': s.slot_id,
                            'drone_id': s.drone_id,
                            'offset': s.offset_vector,
                        }
                        for s in slots
                    ],
                },
            )

        return True

    def start_mission(self, mission: SwarmMission) -> bool:
        """
        Start a swarm mission.

        Args:
            mission: Mission to execute
        """
        if self._state not in [SwarmState.READY, SwarmState.IDLE]:
            logger.warning(
                f"Cannot start mission in state {self._state}"
            )
            return False

        if not self._leader_id:
            # Auto-select leader
            drones = self._registry.get_online_drones()
            if drones:
                self.set_leader(drones[0].drone_id)
            else:
                logger.error("No drones available")
                return False

        self._mission = mission
        self._mission.current_waypoint = 0
        self._mission.completed = False

        # Set formation for mission
        self.set_formation(
            mission.formation,
            mission.spacing,
        )

        self._set_state(SwarmState.EXECUTING)

        # Broadcast mission start
        if self._comm:
            self._comm.broadcast(
                MessageType.MISSION_SYNC,
                {
                    'action': 'start',
                    'mission_id': mission.mission_id,
                    'waypoints': mission.waypoints,
                },
            )

        logger.info(f"Started mission: {mission.mission_id}")
        return True

    def pause_mission(self) -> bool:
        """Pause current mission."""
        if self._state != SwarmState.EXECUTING:
            return False

        self._set_state(SwarmState.PAUSED)

        if self._comm:
            self._comm.broadcast(
                MessageType.MISSION_SYNC,
                {'action': 'pause'},
            )

        return True

    def resume_mission(self) -> bool:
        """Resume paused mission."""
        if self._state != SwarmState.PAUSED:
            return False

        self._set_state(SwarmState.EXECUTING)

        if self._comm:
            self._comm.broadcast(
                MessageType.MISSION_SYNC,
                {'action': 'resume'},
            )

        return True

    def abort_mission(self) -> None:
        """Abort current mission."""
        self._mission = None
        self._set_state(SwarmState.IDLE)

        if self._comm:
            self._comm.broadcast(
                MessageType.MISSION_SYNC,
                {'action': 'abort'},
            )

    def return_all(self) -> None:
        """Command all drones to return home."""
        self._set_state(SwarmState.RETURNING)

        if self._comm:
            self._comm.broadcast(
                MessageType.COMMAND,
                {
                    'command': 'return_to_home',
                    'params': {},
                },
            )

    def land_all(self) -> None:
        """Command all drones to land."""
        self._set_state(SwarmState.LANDING)

        if self._comm:
            self._comm.broadcast(
                MessageType.COMMAND,
                {
                    'command': 'land',
                    'params': {},
                },
            )

    def emergency_stop(self) -> None:
        """Emergency stop all drones."""
        self._set_state(SwarmState.EMERGENCY)

        if self._comm:
            self._comm.broadcast_emergency("coordinator_emergency_stop")
            self._comm.broadcast(
                MessageType.COMMAND,
                {
                    'command': 'emergency_stop',
                    'params': {},
                },
            )

    # State and info

    @property
    def state(self) -> SwarmState:
        """Get current swarm state."""
        return self._state

    @property
    def drone_count(self) -> int:
        """Get number of registered drones."""
        return self._registry.count

    @property
    def online_count(self) -> int:
        """Get number of online drones."""
        return self._registry.online_count

    def get_drones(self) -> List[DroneInfo]:
        """Get all registered drones."""
        return self._registry.get_all_drones()

    def get_drone(self, drone_id: str) -> Optional[DroneInfo]:
        """Get info for a specific drone."""
        return self._registry.get_drone(drone_id)

    def get_leader(self) -> Optional[DroneInfo]:
        """Get current swarm leader."""
        return self._registry.get_leader()

    def get_formation_positions(
        self,
    ) -> List[Tuple[str, float, float, float]]:
        """
        Get target positions for all drones.

        Returns:
            List of (drone_id, lat, lon, alt)
        """
        if not self._leader_id:
            return []

        leader = self._registry.get_drone(self._leader_id)
        if not leader:
            return []

        positions = []
        for slot in self._formation.slots:
            if slot.drone_id:
                lat, lon, alt = self._formation.get_absolute_position(
                    slot,
                    leader.position.lat,
                    leader.position.lon,
                    leader.position.alt,
                    leader.position.heading,
                )
                positions.append((slot.drone_id, lat, lon, alt))

        return positions

    # Internal state management

    def _set_state(self, new_state: SwarmState) -> None:
        """Set swarm state and notify."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            logger.info(f"Swarm state: {old_state} -> {new_state}")
            self._notify_state_changed(old_state, new_state)

    # Registry callbacks

    def _on_drone_added(
        self,
        drone_id: str,
        info: DroneInfo,
    ) -> None:
        """Handle drone registration."""
        logger.info(f"Drone joined swarm: {drone_id}")

    def _on_drone_removed(
        self,
        drone_id: str,
        info: DroneInfo,
    ) -> None:
        """Handle drone removal."""
        logger.info(f"Drone left swarm: {drone_id}")

        if drone_id == self._leader_id:
            self._elect_new_leader()

    def _on_drone_status_changed(
        self,
        drone_id: str,
        old_status: DroneStatus,
        new_status: DroneStatus,
    ) -> None:
        """Handle drone status change."""
        logger.debug(
            f"Drone {drone_id}: {old_status} -> {new_status}"
        )

    def _on_drone_lost(self, drone_id: str) -> None:
        """Handle lost drone."""
        logger.warning(f"Drone lost: {drone_id}")

        if drone_id == self._leader_id:
            self._elect_new_leader()

    def _elect_new_leader(self) -> None:
        """Elect a new swarm leader."""
        drones = self._registry.get_online_drones()
        if drones:
            # Simple: pick first online drone
            new_leader = drones[0].drone_id
            self.set_leader(new_leader)
            logger.info(f"New leader elected: {new_leader}")
        else:
            self._leader_id = None
            logger.warning("No drones available for leader")

    # Event callbacks

    def on_state_changed(self, callback: Callable) -> None:
        """Register state change callback."""
        self._on_state_changed.append(callback)

    def on_formation_reached(self, callback: Callable) -> None:
        """Register formation reached callback."""
        self._on_formation_reached.append(callback)

    def on_waypoint_reached(self, callback: Callable) -> None:
        """Register waypoint reached callback."""
        self._on_waypoint_reached.append(callback)

    def on_mission_complete(self, callback: Callable) -> None:
        """Register mission complete callback."""
        self._on_mission_complete.append(callback)

    def on_emergency(self, callback: Callable) -> None:
        """Register emergency callback."""
        self._on_emergency.append(callback)

    def _notify_state_changed(
        self,
        old_state: SwarmState,
        new_state: SwarmState,
    ) -> None:
        """Notify state change callbacks."""
        for cb in self._on_state_changed:
            try:
                cb(old_state, new_state)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_formation_reached(self) -> None:
        """Notify formation reached callbacks."""
        for cb in self._on_formation_reached:
            try:
                cb()
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_waypoint_reached(self, index: int) -> None:
        """Notify waypoint reached callbacks."""
        for cb in self._on_waypoint_reached:
            try:
                cb(index)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_mission_complete(self) -> None:
        """Notify mission complete callbacks."""
        for cb in self._on_mission_complete:
            try:
                cb()
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_emergency(
        self,
        drone_id: str,
        reason: str,
    ) -> None:
        """Notify emergency callbacks."""
        for cb in self._on_emergency:
            try:
                cb(drone_id, reason)
            except Exception as e:
                logger.error(f"Callback error: {e}")
