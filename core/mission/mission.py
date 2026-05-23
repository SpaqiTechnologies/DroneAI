"""
Mission Module

Defines the Mission container class for holding waypoints and mission parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime
import uuid

from .waypoint import Waypoint, AltitudeFrame


class MissionType(Enum):
    """Type of mission."""
    WAYPOINT = "waypoint"       # Standard waypoint mission
    SURVEY = "survey"           # Survey/mapping mission
    ORBIT = "orbit"             # Orbit around point
    CORRIDOR = "corridor"       # Corridor survey
    SEARCH = "search"           # Search pattern
    CUSTOM = "custom"           # Custom mission


class MissionState(Enum):
    """State of mission execution."""
    DRAFT = "draft"              # Being edited
    READY = "ready"              # Validated and ready
    UPLOADING = "uploading"      # Being uploaded to drone
    UPLOADED = "uploaded"        # On drone, not started
    RUNNING = "running"          # Currently executing
    PAUSED = "paused"            # Paused mid-execution
    COMPLETE = "complete"        # Successfully completed
    ABORTED = "aborted"          # Aborted by user
    FAILED = "failed"            # Failed due to error


@dataclass
class MissionStatistics:
    """Statistics about a mission."""
    total_distance: float = 0.0      # meters
    total_waypoints: int = 0
    estimated_time: float = 0.0      # seconds
    max_altitude: float = 0.0        # meters
    min_altitude: float = 0.0        # meters
    estimated_battery: float = 0.0   # percentage needed

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_distance': self.total_distance,
            'total_waypoints': self.total_waypoints,
            'estimated_time': self.estimated_time,
            'max_altitude': self.max_altitude,
            'min_altitude': self.min_altitude,
            'estimated_battery': self.estimated_battery,
        }


@dataclass
class Mission:
    """
    A complete mission definition containing waypoints and parameters.

    Supports serialization, validation, and statistical analysis.
    """
    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unnamed Mission"
    description: str = ""

    # Mission type and state
    mission_type: MissionType = MissionType.WAYPOINT
    state: MissionState = MissionState.DRAFT

    # Waypoints
    waypoints: List[Waypoint] = field(default_factory=list)

    # Mission parameters
    default_speed: float = 5.0           # m/s
    default_altitude: float = 50.0       # meters
    altitude_frame: AltitudeFrame = AltitudeFrame.RELATIVE
    auto_continue: bool = True           # Auto continue between waypoints
    return_home: bool = False            # RTL after mission complete

    # Safety parameters
    max_distance_from_home: float = 1000.0  # meters (0 = unlimited)
    max_altitude: float = 120.0             # meters
    min_altitude: float = 10.0              # meters

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of waypoints."""
        return len(self.waypoints)

    def __bool__(self) -> bool:
        # A Mission instance is always truthy; without this, an empty mission
        # would be falsy via __len__ and break `if mission:` checks.
        return True

    def __iter__(self) -> Iterator[Waypoint]:
        """Iterate over waypoints."""
        return iter(self.waypoints)

    def __getitem__(self, index: int) -> Waypoint:
        """Get waypoint by index."""
        return self.waypoints[index]

    def add_waypoint(self, waypoint: Waypoint) -> int:
        """
        Add a waypoint to the mission.

        Args:
            waypoint: Waypoint to add

        Returns:
            Index of added waypoint
        """
        waypoint.sequence = len(self.waypoints)
        self.waypoints.append(waypoint)
        self.modified_at = datetime.now()
        self.state = MissionState.DRAFT
        return waypoint.sequence

    def insert_waypoint(self, index: int, waypoint: Waypoint) -> None:
        """
        Insert a waypoint at specific index.

        Args:
            index: Index to insert at
            waypoint: Waypoint to insert
        """
        self.waypoints.insert(index, waypoint)
        self._resequence_waypoints()
        self.modified_at = datetime.now()
        self.state = MissionState.DRAFT

    def remove_waypoint(self, index: int) -> Optional[Waypoint]:
        """
        Remove waypoint at index.

        Args:
            index: Index of waypoint to remove

        Returns:
            Removed waypoint or None
        """
        if 0 <= index < len(self.waypoints):
            wp = self.waypoints.pop(index)
            self._resequence_waypoints()
            self.modified_at = datetime.now()
            self.state = MissionState.DRAFT
            return wp
        return None

    def update_waypoint(self, index: int, waypoint: Waypoint) -> bool:
        """
        Update waypoint at index.

        Args:
            index: Index of waypoint to update
            waypoint: New waypoint data

        Returns:
            True if updated, False if index invalid
        """
        if 0 <= index < len(self.waypoints):
            waypoint.sequence = index
            self.waypoints[index] = waypoint
            self.modified_at = datetime.now()
            self.state = MissionState.DRAFT
            return True
        return False

    def move_waypoint(self, from_index: int, to_index: int) -> bool:
        """
        Move waypoint from one index to another.

        Args:
            from_index: Current index
            to_index: Target index

        Returns:
            True if moved, False if invalid indices
        """
        if (0 <= from_index < len(self.waypoints) and
            0 <= to_index < len(self.waypoints)):
            wp = self.waypoints.pop(from_index)
            self.waypoints.insert(to_index, wp)
            self._resequence_waypoints()
            self.modified_at = datetime.now()
            self.state = MissionState.DRAFT
            return True
        return False

    def clear_waypoints(self) -> None:
        """Remove all waypoints."""
        self.waypoints.clear()
        self.modified_at = datetime.now()
        self.state = MissionState.DRAFT

    def _resequence_waypoints(self) -> None:
        """Resequence waypoints after modification."""
        for i, wp in enumerate(self.waypoints):
            wp.sequence = i

    def get_statistics(self) -> MissionStatistics:
        """
        Calculate mission statistics.

        Returns:
            MissionStatistics with distance, time estimates, etc.
        """
        stats = MissionStatistics()
        stats.total_waypoints = len(self.waypoints)

        if not self.waypoints:
            return stats

        total_distance = 0.0
        max_alt = float('-inf')
        min_alt = float('inf')

        for i in range(len(self.waypoints)):
            wp = self.waypoints[i]

            # Track altitude extremes
            max_alt = max(max_alt, wp.altitude)
            min_alt = min(min_alt, wp.altitude)

            # Calculate distance to next waypoint
            if i < len(self.waypoints) - 1:
                next_wp = self.waypoints[i + 1]
                total_distance += wp.distance_to(next_wp)

        stats.total_distance = total_distance
        stats.max_altitude = max_alt if max_alt != float('-inf') else 0
        stats.min_altitude = min_alt if min_alt != float('inf') else 0

        # Estimate time (distance / average speed + hold times)
        avg_speed = self.default_speed
        travel_time = total_distance / avg_speed if avg_speed > 0 else 0
        hold_time = sum(wp.hold_time for wp in self.waypoints)
        stats.estimated_time = travel_time + hold_time

        # Estimate battery (rough: 1% per km + 0.5% per minute)
        distance_km = total_distance / 1000
        time_minutes = stats.estimated_time / 60
        stats.estimated_battery = distance_km * 1.0 + time_minutes * 0.5

        return stats

    def get_total_distance(self) -> float:
        """Get total mission distance in meters."""
        return self.get_statistics().total_distance

    def get_bounding_box(self) -> Optional[tuple]:
        """
        Get bounding box of mission waypoints.

        Returns:
            Tuple of (min_lat, min_lon, max_lat, max_lon) or None
        """
        if not self.waypoints:
            return None

        lats = [wp.latitude for wp in self.waypoints]
        lons = [wp.longitude for wp in self.waypoints]

        return (min(lats), min(lons), max(lats), max(lons))

    def get_center(self) -> Optional[tuple]:
        """
        Get center point of mission.

        Returns:
            Tuple of (lat, lon) or None
        """
        if not self.waypoints:
            return None

        avg_lat = sum(wp.latitude for wp in self.waypoints) / len(self.waypoints)
        avg_lon = sum(wp.longitude for wp in self.waypoints) / len(self.waypoints)

        return (avg_lat, avg_lon)

    def reverse(self) -> None:
        """Reverse the order of waypoints."""
        self.waypoints.reverse()
        self._resequence_waypoints()
        self.modified_at = datetime.now()
        self.state = MissionState.DRAFT

    def copy(self) -> 'Mission':
        """Create a deep copy of the mission."""
        new_mission = Mission(
            id=str(uuid.uuid4()),
            name=f"{self.name} (Copy)",
            description=self.description,
            mission_type=self.mission_type,
            state=MissionState.DRAFT,
            waypoints=[wp.copy() for wp in self.waypoints],
            default_speed=self.default_speed,
            default_altitude=self.default_altitude,
            altitude_frame=self.altitude_frame,
            auto_continue=self.auto_continue,
            return_home=self.return_home,
            max_distance_from_home=self.max_distance_from_home,
            max_altitude=self.max_altitude,
            min_altitude=self.min_altitude,
            tags=list(self.tags),
            metadata=dict(self.metadata),
        )
        return new_mission

    def to_dict(self) -> Dict[str, Any]:
        """Convert mission to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'mission_type': self.mission_type.value,
            'state': self.state.value,
            'waypoints': [wp.to_dict() for wp in self.waypoints],
            'default_speed': self.default_speed,
            'default_altitude': self.default_altitude,
            'altitude_frame': self.altitude_frame.value,
            'auto_continue': self.auto_continue,
            'return_home': self.return_home,
            'max_distance_from_home': self.max_distance_from_home,
            'max_altitude': self.max_altitude,
            'min_altitude': self.min_altitude,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata,
            'statistics': self.get_statistics().to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Mission':
        """Create mission from dictionary."""
        waypoints = [
            Waypoint.from_dict(wp)
            for wp in data.get('waypoints', [])
        ]

        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', 'Unnamed Mission'),
            description=data.get('description', ''),
            mission_type=MissionType(data.get('mission_type', 'waypoint')),
            state=MissionState(data.get('state', 'draft')),
            waypoints=waypoints,
            default_speed=data.get('default_speed', 5.0),
            default_altitude=data.get('default_altitude', 50.0),
            altitude_frame=AltitudeFrame(data.get('altitude_frame', 'relative')),
            auto_continue=data.get('auto_continue', True),
            return_home=data.get('return_home', False),
            max_distance_from_home=data.get('max_distance_from_home', 1000.0),
            max_altitude=data.get('max_altitude', 120.0),
            min_altitude=data.get('min_altitude', 10.0),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            modified_at=datetime.fromisoformat(data['modified_at']) if 'modified_at' in data else datetime.now(),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
        )

    def __str__(self) -> str:
        return f"Mission '{self.name}' ({len(self)} waypoints)"

    def __repr__(self) -> str:
        return f"Mission(id={self.id}, name={self.name}, waypoints={len(self)})"
