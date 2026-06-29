"""
Drone registry for tracking multiple drones in a swarm.

Maintains state information for all registered drones
and provides lookup/query capabilities.
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class DroneStatus(Enum):
    """Status of a drone in the swarm."""
    UNKNOWN = "unknown"
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    READY = "ready"
    ARMED = "armed"
    FLYING = "flying"
    LANDING = "landing"
    EMERGENCY = "emergency"
    LOST = "lost"


@dataclass
class DronePosition:
    """3D position of a drone."""
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    heading: float = 0.0


@dataclass
class DroneVelocity:
    """Velocity vector of a drone."""
    vx: float = 0.0  # m/s north
    vy: float = 0.0  # m/s east
    vz: float = 0.0  # m/s down
    speed: float = 0.0  # ground speed


@dataclass
class DroneInfo:
    """Complete information about a drone in the swarm."""
    drone_id: str
    name: str = ""
    status: DroneStatus = DroneStatus.UNKNOWN
    position: DronePosition = field(default_factory=DronePosition)
    velocity: DroneVelocity = field(default_factory=DroneVelocity)
    battery_percent: float = 100.0
    signal_strength: float = 100.0
    last_heartbeat: float = 0.0
    formation_slot: int = -1
    is_leader: bool = False
    capabilities: List[str] = field(default_factory=list)
    custom_data: Dict = field(default_factory=dict)

    @property
    def is_online(self) -> bool:
        """Check if drone is online (heartbeat within 5 seconds)."""
        return (time.time() - self.last_heartbeat) < 5.0

    @property
    def is_flying(self) -> bool:
        """Check if drone is currently in flight."""
        return self.status in [
            DroneStatus.ARMED,
            DroneStatus.FLYING,
            DroneStatus.LANDING,
        ]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'drone_id': self.drone_id,
            'name': self.name,
            'status': self.status.value,
            'position': {
                'lat': self.position.lat,
                'lon': self.position.lon,
                'alt': self.position.alt,
                'heading': self.position.heading,
            },
            'velocity': {
                'vx': self.velocity.vx,
                'vy': self.velocity.vy,
                'vz': self.velocity.vz,
                'speed': self.velocity.speed,
            },
            'battery_percent': self.battery_percent,
            'signal_strength': self.signal_strength,
            'last_heartbeat': self.last_heartbeat,
            'formation_slot': self.formation_slot,
            'is_leader': self.is_leader,
            'is_online': self.is_online,
        }


class DroneRegistry:
    """
    Registry for tracking multiple drones in a swarm.

    Provides:
    - Drone registration and deregistration
    - State updates and queries
    - Heartbeat monitoring
    - Event callbacks for state changes
    """

    def __init__(
        self,
        heartbeat_timeout: float = 5.0,
        lost_timeout: float = 30.0,
    ):
        """
        Initialize the drone registry.

        Args:
            heartbeat_timeout: Seconds before marking drone offline
            lost_timeout: Seconds before marking drone lost
        """
        self._drones: Dict[str, DroneInfo] = {}
        self._lock = threading.RLock()
        self._heartbeat_timeout = heartbeat_timeout
        self._lost_timeout = lost_timeout

        # Callbacks
        self._on_drone_added: List[Callable] = []
        self._on_drone_removed: List[Callable] = []
        self._on_status_changed: List[Callable] = []
        self._on_drone_lost: List[Callable] = []

        # Background monitor
        self._monitor_running = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self) -> None:
        """Start background heartbeat monitoring."""
        if self._monitor_running:
            return

        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Drone registry monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Drone registry monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background loop to check drone heartbeats."""
        while self._monitor_running:
            self._check_heartbeats()
            time.sleep(1.0)

    def _check_heartbeats(self) -> None:
        """Check all drones for heartbeat timeouts."""
        current_time = time.time()

        with self._lock:
            for drone_id, info in self._drones.items():
                elapsed = current_time - info.last_heartbeat

                if elapsed > self._lost_timeout:
                    if info.status != DroneStatus.LOST:
                        old_status = info.status
                        info.status = DroneStatus.LOST
                        self._notify_status_changed(
                            drone_id, old_status, DroneStatus.LOST
                        )
                        self._notify_drone_lost(drone_id)

                elif elapsed > self._heartbeat_timeout:
                    if info.status not in [
                        DroneStatus.OFFLINE,
                        DroneStatus.LOST,
                    ]:
                        old_status = info.status
                        info.status = DroneStatus.OFFLINE
                        self._notify_status_changed(
                            drone_id, old_status, DroneStatus.OFFLINE
                        )

    def register(
        self,
        drone_id: str,
        name: str = "",
        capabilities: Optional[List[str]] = None,
    ) -> DroneInfo:
        """
        Register a new drone in the swarm.

        Args:
            drone_id: Unique identifier for the drone
            name: Human-readable name
            capabilities: List of drone capabilities

        Returns:
            DroneInfo object for the registered drone
        """
        with self._lock:
            if drone_id in self._drones:
                logger.warning(f"Drone {drone_id} already registered")
                return self._drones[drone_id]

            info = DroneInfo(
                drone_id=drone_id,
                name=name or f"Drone-{drone_id[:8]}",
                status=DroneStatus.INITIALIZING,
                last_heartbeat=time.time(),
                capabilities=capabilities or [],
            )
            self._drones[drone_id] = info

            logger.info(f"Registered drone: {drone_id}")
            self._notify_drone_added(drone_id, info)

            return info

    def unregister(self, drone_id: str) -> bool:
        """
        Remove a drone from the registry.

        Args:
            drone_id: ID of drone to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if drone_id not in self._drones:
                return False

            info = self._drones.pop(drone_id)
            logger.info(f"Unregistered drone: {drone_id}")
            self._notify_drone_removed(drone_id, info)

            return True

    def update_heartbeat(self, drone_id: str) -> bool:
        """
        Update heartbeat timestamp for a drone.

        Args:
            drone_id: ID of drone

        Returns:
            True if updated, False if drone not found
        """
        with self._lock:
            if drone_id not in self._drones:
                return False

            info = self._drones[drone_id]
            info.last_heartbeat = time.time()

            # Restore from offline if applicable
            if info.status == DroneStatus.OFFLINE:
                info.status = DroneStatus.READY
                self._notify_status_changed(
                    drone_id, DroneStatus.OFFLINE, DroneStatus.READY
                )

            return True

    def update_state(
        self,
        drone_id: str,
        position: Optional[DronePosition] = None,
        velocity: Optional[DroneVelocity] = None,
        status: Optional[DroneStatus] = None,
        battery_percent: Optional[float] = None,
        signal_strength: Optional[float] = None,
        custom_data: Optional[Dict] = None,
    ) -> bool:
        """
        Update state information for a drone.

        Args:
            drone_id: ID of drone to update
            position: New position data
            velocity: New velocity data
            status: New status
            battery_percent: New battery level
            signal_strength: New signal strength
            custom_data: Additional custom data

        Returns:
            True if updated, False if drone not found
        """
        with self._lock:
            if drone_id not in self._drones:
                return False

            info = self._drones[drone_id]
            info.last_heartbeat = time.time()

            if position:
                info.position = position
            if velocity:
                info.velocity = velocity
            if battery_percent is not None:
                info.battery_percent = battery_percent
            if signal_strength is not None:
                info.signal_strength = signal_strength
            if custom_data:
                info.custom_data.update(custom_data)

            if status and status != info.status:
                old_status = info.status
                info.status = status
                self._notify_status_changed(drone_id, old_status, status)

            return True

    def get_drone(self, drone_id: str) -> Optional[DroneInfo]:
        """Get info for a specific drone."""
        with self._lock:
            return self._drones.get(drone_id)

    def get_all_drones(self) -> List[DroneInfo]:
        """Get info for all registered drones."""
        with self._lock:
            return list(self._drones.values())

    def get_online_drones(self) -> List[DroneInfo]:
        """Get all drones that are currently online."""
        with self._lock:
            return [d for d in self._drones.values() if d.is_online]

    def get_flying_drones(self) -> List[DroneInfo]:
        """Get all drones that are currently flying."""
        with self._lock:
            return [d for d in self._drones.values() if d.is_flying]

    def get_leader(self) -> Optional[DroneInfo]:
        """Get the current swarm leader."""
        with self._lock:
            for drone in self._drones.values():
                if drone.is_leader:
                    return drone
            return None

    def set_leader(self, drone_id: str) -> bool:
        """
        Set a drone as the swarm leader.

        Args:
            drone_id: ID of drone to make leader

        Returns:
            True if set, False if drone not found
        """
        with self._lock:
            if drone_id not in self._drones:
                return False

            # Clear existing leader
            for drone in self._drones.values():
                drone.is_leader = False

            self._drones[drone_id].is_leader = True
            logger.info(f"Swarm leader set to: {drone_id}")

            return True

    def set_formation_slot(
        self,
        drone_id: str,
        slot: int,
    ) -> bool:
        """
        Assign a formation slot to a drone.

        Args:
            drone_id: ID of drone
            slot: Formation slot number

        Returns:
            True if set, False if drone not found
        """
        with self._lock:
            if drone_id not in self._drones:
                return False

            self._drones[drone_id].formation_slot = slot
            return True

    @property
    def count(self) -> int:
        """Get total number of registered drones."""
        with self._lock:
            return len(self._drones)

    @property
    def online_count(self) -> int:
        """Get number of online drones."""
        return len(self.get_online_drones())

    # Event callbacks
    def on_drone_added(self, callback: Callable) -> None:
        """Register callback for drone registration."""
        self._on_drone_added.append(callback)

    def on_drone_removed(self, callback: Callable) -> None:
        """Register callback for drone removal."""
        self._on_drone_removed.append(callback)

    def on_status_changed(self, callback: Callable) -> None:
        """Register callback for status changes."""
        self._on_status_changed.append(callback)

    def on_drone_lost(self, callback: Callable) -> None:
        """Register callback for lost drones."""
        self._on_drone_lost.append(callback)

    def _notify_drone_added(
        self,
        drone_id: str,
        info: DroneInfo,
    ) -> None:
        """Notify callbacks of drone addition."""
        for cb in self._on_drone_added:
            try:
                cb(drone_id, info)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_drone_removed(
        self,
        drone_id: str,
        info: DroneInfo,
    ) -> None:
        """Notify callbacks of drone removal."""
        for cb in self._on_drone_removed:
            try:
                cb(drone_id, info)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_status_changed(
        self,
        drone_id: str,
        old_status: DroneStatus,
        new_status: DroneStatus,
    ) -> None:
        """Notify callbacks of status change."""
        for cb in self._on_status_changed:
            try:
                cb(drone_id, old_status, new_status)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_drone_lost(self, drone_id: str) -> None:
        """Notify callbacks of lost drone."""
        for cb in self._on_drone_lost:
            try:
                cb(drone_id)
            except Exception as e:
                logger.error(f"Callback error: {e}")
