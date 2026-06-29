"""
Maintenance tracking for component wear and service scheduling.

Tracks flight hours, cycles, and component health to
schedule preventive maintenance.
"""

import time
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component health status."""
    GOOD = "good"
    FAIR = "fair"
    WORN = "worn"
    REPLACE_SOON = "replace_soon"
    REPLACE_NOW = "replace_now"
    FAILED = "failed"


class ComponentType(Enum):
    """Types of tracked components."""
    MOTOR = "motor"
    ESC = "esc"
    PROPELLER = "propeller"
    BATTERY = "battery"
    FRAME = "frame"
    GIMBAL = "gimbal"
    CAMERA = "camera"
    GPS = "gps"
    IMU = "imu"
    FLIGHT_CONTROLLER = "flight_controller"
    OTHER = "other"


@dataclass
class MaintenanceItem:
    """A maintenance task or component service."""
    item_id: str
    component_type: ComponentType
    component_id: str
    description: str
    interval_hours: float  # Service interval
    interval_cycles: int  # Or cycle-based interval
    last_service: float  # Timestamp of last service
    last_service_hours: float
    last_service_cycles: int
    current_hours: float = 0.0
    current_cycles: int = 0
    status: ComponentStatus = ComponentStatus.GOOD
    notes: str = ""

    @property
    def hours_since_service(self) -> float:
        """Hours since last service."""
        return self.current_hours - self.last_service_hours

    @property
    def cycles_since_service(self) -> int:
        """Cycles since last service."""
        return self.current_cycles - self.last_service_cycles

    @property
    def hours_remaining(self) -> float:
        """Hours until service due."""
        return max(0, self.interval_hours - self.hours_since_service)

    @property
    def cycles_remaining(self) -> int:
        """Cycles until service due."""
        return max(0, self.interval_cycles - self.cycles_since_service)

    @property
    def percent_life(self) -> float:
        """Percentage of service interval used."""
        if self.interval_hours > 0:
            hour_pct = self.hours_since_service / self.interval_hours
        else:
            hour_pct = 0

        if self.interval_cycles > 0:
            cycle_pct = self.cycles_since_service / self.interval_cycles
        else:
            cycle_pct = 0

        return max(hour_pct, cycle_pct) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'item_id': self.item_id,
            'component_type': self.component_type.value,
            'component_id': self.component_id,
            'description': self.description,
            'interval_hours': self.interval_hours,
            'interval_cycles': self.interval_cycles,
            'current_hours': self.current_hours,
            'current_cycles': self.current_cycles,
            'hours_since_service': self.hours_since_service,
            'cycles_since_service': self.cycles_since_service,
            'hours_remaining': self.hours_remaining,
            'cycles_remaining': self.cycles_remaining,
            'percent_life': self.percent_life,
            'status': self.status.value,
            'last_service': self.last_service,
            'notes': self.notes,
        }


@dataclass
class DroneMaintenanceRecord:
    """Complete maintenance record for a drone."""
    drone_id: str
    total_flight_hours: float = 0.0
    total_flight_cycles: int = 0
    total_distance_km: float = 0.0
    items: Dict[str, MaintenanceItem] = field(default_factory=dict)
    service_history: List[Dict] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class MaintenanceTracker:
    """
    Maintenance tracker for drone components.

    Tracks usage hours, cycles, and schedules
    preventive maintenance.
    """

    # Default service intervals
    DEFAULT_INTERVALS = {
        ComponentType.MOTOR: (100, 500),  # hours, cycles
        ComponentType.ESC: (200, 1000),
        ComponentType.PROPELLER: (25, 100),
        ComponentType.BATTERY: (50, 200),
        ComponentType.GIMBAL: (100, 500),
        ComponentType.GPS: (500, 0),
        ComponentType.IMU: (500, 0),
        ComponentType.FLIGHT_CONTROLLER: (1000, 0),
    }

    def __init__(self, storage_path: str = "maintenance"):
        """
        Initialize maintenance tracker.

        Args:
            storage_path: Path for maintenance data
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._drones: Dict[str, DroneMaintenanceRecord] = {}

    def register_drone(
        self,
        drone_id: str,
        components: Optional[Dict[str, ComponentType]] = None,
    ) -> None:
        """
        Register a drone for maintenance tracking.

        Args:
            drone_id: Drone identifier
            components: Optional component mapping
        """
        if drone_id in self._drones:
            return

        record = DroneMaintenanceRecord(drone_id=drone_id)

        # Add default components if not specified
        if components is None:
            components = {
                'motor_1': ComponentType.MOTOR,
                'motor_2': ComponentType.MOTOR,
                'motor_3': ComponentType.MOTOR,
                'motor_4': ComponentType.MOTOR,
                'esc_1': ComponentType.ESC,
                'esc_2': ComponentType.ESC,
                'esc_3': ComponentType.ESC,
                'esc_4': ComponentType.ESC,
                'prop_1': ComponentType.PROPELLER,
                'prop_2': ComponentType.PROPELLER,
                'prop_3': ComponentType.PROPELLER,
                'prop_4': ComponentType.PROPELLER,
                'battery': ComponentType.BATTERY,
                'gps': ComponentType.GPS,
                'imu': ComponentType.IMU,
                'fc': ComponentType.FLIGHT_CONTROLLER,
            }

        # Create maintenance items
        for comp_id, comp_type in components.items():
            interval = self.DEFAULT_INTERVALS.get(
                comp_type, (100, 500)
            )

            item = MaintenanceItem(
                item_id=f"{drone_id}_{comp_id}",
                component_type=comp_type,
                component_id=comp_id,
                description=f"{comp_type.value} maintenance",
                interval_hours=interval[0],
                interval_cycles=interval[1],
                last_service=time.time(),
                last_service_hours=0.0,
                last_service_cycles=0,
            )
            record.items[comp_id] = item

        self._drones[drone_id] = record
        self._save_record(record)

        logger.info(
            f"Registered drone {drone_id} with "
            f"{len(record.items)} components"
        )

    def log_flight(
        self,
        drone_id: str,
        flight_hours: float,
        distance_km: float = 0.0,
    ) -> None:
        """
        Log a completed flight.

        Args:
            drone_id: Drone identifier
            flight_hours: Flight duration in hours
            distance_km: Distance flown in km
        """
        record = self._drones.get(drone_id)
        if record is None:
            logger.warning(f"Unknown drone: {drone_id}")
            return

        record.total_flight_hours += flight_hours
        record.total_flight_cycles += 1
        record.total_distance_km += distance_km
        record.last_updated = time.time()

        # Update all components
        for item in record.items.values():
            item.current_hours = record.total_flight_hours
            item.current_cycles = record.total_flight_cycles
            self._update_component_status(item)

        self._save_record(record)

    def _update_component_status(self, item: MaintenanceItem) -> None:
        """Update component status based on usage."""
        pct = item.percent_life

        if pct >= 100:
            item.status = ComponentStatus.REPLACE_NOW
        elif pct >= 90:
            item.status = ComponentStatus.REPLACE_SOON
        elif pct >= 75:
            item.status = ComponentStatus.WORN
        elif pct >= 50:
            item.status = ComponentStatus.FAIR
        else:
            item.status = ComponentStatus.GOOD

    def record_service(
        self,
        drone_id: str,
        component_id: str,
        description: str = "",
        replaced: bool = False,
    ) -> bool:
        """
        Record a maintenance service.

        Args:
            drone_id: Drone identifier
            component_id: Component serviced
            description: Service description
            replaced: Whether component was replaced

        Returns:
            True if recorded successfully
        """
        record = self._drones.get(drone_id)
        if record is None:
            return False

        item = record.items.get(component_id)
        if item is None:
            return False

        # Update service record
        item.last_service = time.time()
        item.last_service_hours = record.total_flight_hours
        item.last_service_cycles = record.total_flight_cycles

        if replaced:
            item.current_hours = record.total_flight_hours
            item.current_cycles = record.total_flight_cycles

        item.status = ComponentStatus.GOOD

        # Add to history
        service_entry = {
            'timestamp': time.time(),
            'component_id': component_id,
            'description': description,
            'replaced': replaced,
            'hours_at_service': record.total_flight_hours,
            'cycles_at_service': record.total_flight_cycles,
        }
        record.service_history.append(service_entry)

        self._save_record(record)

        logger.info(
            f"Service recorded: {drone_id}/{component_id}"
        )
        return True

    def get_due_maintenance(
        self,
        drone_id: str,
    ) -> List[MaintenanceItem]:
        """
        Get components due for maintenance.

        Args:
            drone_id: Drone identifier

        Returns:
            List of components needing service
        """
        record = self._drones.get(drone_id)
        if record is None:
            return []

        due_items = []
        for item in record.items.values():
            if item.status in [
                ComponentStatus.WORN,
                ComponentStatus.REPLACE_SOON,
                ComponentStatus.REPLACE_NOW,
            ]:
                due_items.append(item)

        # Sort by urgency
        status_priority = {
            ComponentStatus.REPLACE_NOW: 0,
            ComponentStatus.REPLACE_SOON: 1,
            ComponentStatus.WORN: 2,
        }

        due_items.sort(
            key=lambda x: status_priority.get(x.status, 99)
        )

        return due_items

    def get_component_status(
        self,
        drone_id: str,
        component_id: str,
    ) -> Optional[MaintenanceItem]:
        """Get status of a specific component."""
        record = self._drones.get(drone_id)
        if record is None:
            return None
        return record.items.get(component_id)

    def get_all_statuses(
        self,
        drone_id: str,
    ) -> Dict[str, MaintenanceItem]:
        """Get all component statuses for a drone."""
        record = self._drones.get(drone_id)
        if record is None:
            return {}
        return record.items.copy()

    def get_flight_summary(self, drone_id: str) -> Optional[Dict]:
        """Get flight time summary for a drone."""
        record = self._drones.get(drone_id)
        if record is None:
            return None

        return {
            'drone_id': drone_id,
            'total_hours': record.total_flight_hours,
            'total_cycles': record.total_flight_cycles,
            'total_distance_km': record.total_distance_km,
            'components': len(record.items),
            'due_maintenance': len(self.get_due_maintenance(drone_id)),
            'service_count': len(record.service_history),
        }

    def _save_record(self, record: DroneMaintenanceRecord) -> None:
        """Save maintenance record to file."""
        file_path = self._storage_path / f"{record.drone_id}.json"

        data = {
            'drone_id': record.drone_id,
            'total_flight_hours': record.total_flight_hours,
            'total_flight_cycles': record.total_flight_cycles,
            'total_distance_km': record.total_distance_km,
            'last_updated': record.last_updated,
            'items': {
                k: v.to_dict() for k, v in record.items.items()
            },
            'service_history': record.service_history,
        }

        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save record: {e}")

    def load_drone(self, drone_id: str) -> bool:
        """Load drone record from file."""
        file_path = self._storage_path / f"{drone_id}.json"

        if not file_path.exists():
            return False

        try:
            with open(file_path) as f:
                data = json.load(f)

            record = DroneMaintenanceRecord(
                drone_id=data['drone_id'],
                total_flight_hours=data['total_flight_hours'],
                total_flight_cycles=data['total_flight_cycles'],
                total_distance_km=data.get('total_distance_km', 0),
                service_history=data.get('service_history', []),
                last_updated=data.get('last_updated', time.time()),
            )

            # Reconstruct items
            for comp_id, item_data in data.get('items', {}).items():
                item = MaintenanceItem(
                    item_id=item_data['item_id'],
                    component_type=ComponentType(
                        item_data['component_type']
                    ),
                    component_id=item_data['component_id'],
                    description=item_data['description'],
                    interval_hours=item_data['interval_hours'],
                    interval_cycles=item_data['interval_cycles'],
                    last_service=item_data['last_service'],
                    last_service_hours=item_data.get(
                        'last_service_hours', 0
                    ),
                    last_service_cycles=item_data.get(
                        'last_service_cycles', 0
                    ),
                    current_hours=item_data.get('current_hours', 0),
                    current_cycles=item_data.get('current_cycles', 0),
                    status=ComponentStatus(item_data['status']),
                    notes=item_data.get('notes', ''),
                )
                record.items[comp_id] = item

            self._drones[drone_id] = record
            return True

        except Exception as e:
            logger.error(f"Failed to load record: {e}")
            return False
