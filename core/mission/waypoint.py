"""
Waypoint Module

Defines waypoint data structures for mission planning with full parameter support.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import math


class AltitudeFrame(Enum):
    """Reference frame for altitude."""
    AGL = "agl"           # Above Ground Level (relative to terrain)
    MSL = "msl"           # Mean Sea Level (absolute)
    RELATIVE = "relative"  # Relative to home position


class WaypointType(Enum):
    """Type of waypoint."""
    NAVIGATE = "navigate"       # Standard navigation waypoint
    LOITER = "loiter"          # Loiter at position
    LAND = "land"              # Landing waypoint
    TAKEOFF = "takeoff"        # Takeoff waypoint
    ROI = "roi"                # Region of Interest (camera target)
    SPLINE = "spline"          # Spline waypoint for smooth curves


class WaypointAction(Enum):
    """Actions that can be executed at waypoints."""
    NONE = "none"
    TAKE_PHOTO = "take_photo"
    START_VIDEO = "start_video"
    STOP_VIDEO = "stop_video"
    DELAY = "delay"
    SET_SPEED = "set_speed"
    SET_YAW = "set_yaw"
    SET_GIMBAL = "set_gimbal"
    TRIGGER_SERVO = "trigger_servo"
    SET_RELAY = "set_relay"
    REPEAT_SERVO = "repeat_servo"


@dataclass
class WaypointActionItem:
    """A single action to execute at a waypoint."""
    action: WaypointAction
    param1: float = 0.0  # Action-specific parameter 1
    param2: float = 0.0  # Action-specific parameter 2
    param3: float = 0.0  # Action-specific parameter 3
    param4: float = 0.0  # Action-specific parameter 4

    @classmethod
    def take_photo(cls) -> 'WaypointActionItem':
        """Create a take photo action."""
        return cls(action=WaypointAction.TAKE_PHOTO)

    @classmethod
    def start_video(cls) -> 'WaypointActionItem':
        """Create a start video action."""
        return cls(action=WaypointAction.START_VIDEO)

    @classmethod
    def stop_video(cls) -> 'WaypointActionItem':
        """Create a stop video action."""
        return cls(action=WaypointAction.STOP_VIDEO)

    @classmethod
    def delay(cls, seconds: float) -> 'WaypointActionItem':
        """Create a delay action."""
        return cls(action=WaypointAction.DELAY, param1=seconds)

    @classmethod
    def set_speed(cls, speed_mps: float) -> 'WaypointActionItem':
        """Create a set speed action."""
        return cls(action=WaypointAction.SET_SPEED, param1=speed_mps)

    @classmethod
    def set_yaw(cls, yaw_deg: float, relative: bool = False) -> 'WaypointActionItem':
        """Create a set yaw action."""
        return cls(
            action=WaypointAction.SET_YAW,
            param1=yaw_deg,
            param2=1.0 if relative else 0.0
        )

    @classmethod
    def set_gimbal(cls, pitch: float, roll: float = 0.0, yaw: float = 0.0) -> 'WaypointActionItem':
        """Create a set gimbal action."""
        return cls(
            action=WaypointAction.SET_GIMBAL,
            param1=pitch,
            param2=roll,
            param3=yaw
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action': self.action.value,
            'param1': self.param1,
            'param2': self.param2,
            'param3': self.param3,
            'param4': self.param4,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WaypointActionItem':
        """Create from dictionary."""
        return cls(
            action=WaypointAction(data['action']),
            param1=data.get('param1', 0.0),
            param2=data.get('param2', 0.0),
            param3=data.get('param3', 0.0),
            param4=data.get('param4', 0.0),
        )


@dataclass
class Waypoint:
    """
    A waypoint in a mission with full parameter support.

    Supports various waypoint types, altitude frames, and actions.
    """
    # Required position
    latitude: float
    longitude: float
    altitude: float

    # Waypoint identification
    sequence: int = 0
    name: str = ""

    # Waypoint type and frame
    waypoint_type: WaypointType = WaypointType.NAVIGATE
    altitude_frame: AltitudeFrame = AltitudeFrame.RELATIVE

    # Navigation parameters
    speed: float = 5.0                    # m/s (0 = use default)
    acceptance_radius: float = 2.0        # meters
    pass_through: bool = False            # True = fly through, False = stop
    hold_time: float = 0.0                # seconds to hold at waypoint

    # Yaw control
    yaw: Optional[float] = None           # Target yaw (None = auto toward next wp)
    yaw_rate: float = 0.0                 # deg/s (0 = default)

    # Actions to execute at waypoint
    actions: List[WaypointActionItem] = field(default_factory=list)

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def position(self) -> tuple:
        """Get position as tuple (lat, lon, alt)."""
        return (self.latitude, self.longitude, self.altitude)

    @property
    def position_2d(self) -> tuple:
        """Get 2D position as tuple (lat, lon)."""
        return (self.latitude, self.longitude)

    def distance_to(self, other: 'Waypoint') -> float:
        """Calculate 3D distance to another waypoint in meters."""
        return self.distance_to_position(other.latitude, other.longitude, other.altitude)

    def distance_to_position(self, lat: float, lon: float, alt: float) -> float:
        """Calculate 3D distance to a position in meters."""
        # Haversine formula for horizontal distance
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(lat)
        dlat = math.radians(lat - self.latitude)
        dlon = math.radians(lon - self.longitude)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        horizontal_dist = 6371000 * c  # Earth radius in meters

        # 3D distance
        vertical_dist = abs(alt - self.altitude)
        return math.sqrt(horizontal_dist ** 2 + vertical_dist ** 2)

    def horizontal_distance_to(self, other: 'Waypoint') -> float:
        """Calculate horizontal distance to another waypoint in meters."""
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        dlat = math.radians(other.latitude - self.latitude)
        dlon = math.radians(other.longitude - self.longitude)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371000 * c

    def bearing_to(self, other: 'Waypoint') -> float:
        """Calculate bearing to another waypoint in degrees."""
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        dlon = math.radians(other.longitude - self.longitude)

        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

        bearing = math.atan2(x, y)
        return (math.degrees(bearing) + 360) % 360

    def copy(self) -> 'Waypoint':
        """Create a copy of this waypoint."""
        return Waypoint(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            sequence=self.sequence,
            name=self.name,
            waypoint_type=self.waypoint_type,
            altitude_frame=self.altitude_frame,
            speed=self.speed,
            acceptance_radius=self.acceptance_radius,
            pass_through=self.pass_through,
            hold_time=self.hold_time,
            yaw=self.yaw,
            yaw_rate=self.yaw_rate,
            actions=list(self.actions),
            description=self.description,
            tags=list(self.tags),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert waypoint to dictionary for serialization."""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'sequence': self.sequence,
            'name': self.name,
            'waypoint_type': self.waypoint_type.value,
            'altitude_frame': self.altitude_frame.value,
            'speed': self.speed,
            'acceptance_radius': self.acceptance_radius,
            'pass_through': self.pass_through,
            'hold_time': self.hold_time,
            'yaw': self.yaw,
            'yaw_rate': self.yaw_rate,
            'actions': [a.to_dict() for a in self.actions],
            'description': self.description,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Waypoint':
        """Create waypoint from dictionary."""
        actions = [
            WaypointActionItem.from_dict(a)
            for a in data.get('actions', [])
        ]

        return cls(
            latitude=data['latitude'],
            longitude=data['longitude'],
            altitude=data['altitude'],
            sequence=data.get('sequence', 0),
            name=data.get('name', ''),
            waypoint_type=WaypointType(data.get('waypoint_type', 'navigate')),
            altitude_frame=AltitudeFrame(data.get('altitude_frame', 'relative')),
            speed=data.get('speed', 5.0),
            acceptance_radius=data.get('acceptance_radius', 2.0),
            pass_through=data.get('pass_through', False),
            hold_time=data.get('hold_time', 0.0),
            yaw=data.get('yaw'),
            yaw_rate=data.get('yaw_rate', 0.0),
            actions=actions,
            description=data.get('description', ''),
            tags=data.get('tags', []),
        )

    @classmethod
    def from_position(
        cls,
        lat: float,
        lon: float,
        alt: float,
        **kwargs
    ) -> 'Waypoint':
        """Create waypoint from position coordinates."""
        return cls(latitude=lat, longitude=lon, altitude=alt, **kwargs)

    def __str__(self) -> str:
        return f"WP{self.sequence}: ({self.latitude:.6f}, {self.longitude:.6f}, {self.altitude:.1f}m)"

    def __repr__(self) -> str:
        return (f"Waypoint(lat={self.latitude}, lon={self.longitude}, "
                f"alt={self.altitude}, seq={self.sequence})")
