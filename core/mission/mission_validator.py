"""
Mission Validator Module

Validates missions for feasibility, safety, and compliance before execution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, TYPE_CHECKING
import math

from .mission import Mission, MissionState
from .waypoint import Waypoint

if TYPE_CHECKING:
    from core.drone import Drone


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    INFO = "info"           # Informational, not blocking
    WARNING = "warning"     # Warning, mission can still run
    ERROR = "error"         # Error, mission cannot run


class ValidationType(Enum):
    """Type of validation check."""
    WAYPOINT_COUNT = "waypoint_count"
    ALTITUDE_LIMITS = "altitude_limits"
    DISTANCE_LIMITS = "distance_limits"
    BATTERY_FEASIBILITY = "battery_feasibility"
    GEOFENCE_COMPLIANCE = "geofence_compliance"
    SPEED_LIMITS = "speed_limits"
    WAYPOINT_SPACING = "waypoint_spacing"
    GPS_REQUIREMENT = "gps_requirement"
    SENSOR_REQUIREMENTS = "sensor_requirements"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    validation_type: ValidationType
    severity: ValidationSeverity
    message: str
    waypoint_index: Optional[int] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'type': self.validation_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'waypoint_index': self.waypoint_index,
            'details': self.details,
        }


@dataclass
class ValidationResult:
    """Result of mission validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    mission_feasibility: float = 1.0  # 0-1 feasibility score

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.errors.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def to_dict(self) -> dict:
        return {
            'is_valid': self.is_valid,
            'issues': [i.to_dict() for i in self.issues],
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'feasibility': self.mission_feasibility,
        }


class MissionValidator:
    """
    Validates missions for safety and feasibility.

    Performs various checks including:
    - Altitude limits
    - Distance from home
    - Battery feasibility
    - Geofence compliance
    - Speed limits
    - Waypoint spacing
    """

    # Default limits
    DEFAULT_MAX_ALTITUDE = 120.0      # meters
    DEFAULT_MIN_ALTITUDE = 5.0        # meters
    DEFAULT_MAX_DISTANCE = 2000.0     # meters from home
    DEFAULT_MIN_WAYPOINT_SPACING = 1.0  # meters
    DEFAULT_MAX_SPEED = 15.0          # m/s
    DEFAULT_MIN_SPEED = 0.5           # m/s

    # Battery consumption estimates
    BATTERY_PER_KM = 1.0              # % per km
    BATTERY_PER_MINUTE = 0.5          # % per minute
    BATTERY_RESERVE = 20.0            # % reserve required

    def __init__(self):
        """Initialize validator with default settings."""
        self.max_altitude = self.DEFAULT_MAX_ALTITUDE
        self.min_altitude = self.DEFAULT_MIN_ALTITUDE
        self.max_distance = self.DEFAULT_MAX_DISTANCE
        self.min_waypoint_spacing = self.DEFAULT_MIN_WAYPOINT_SPACING
        self.max_speed = self.DEFAULT_MAX_SPEED
        self.min_speed = self.DEFAULT_MIN_SPEED

    def validate(
        self,
        mission: Mission,
        drone: Optional['Drone'] = None,
        home_position: Optional[Tuple[float, float]] = None
    ) -> ValidationResult:
        """
        Validate a mission.

        Args:
            mission: Mission to validate
            drone: Optional drone instance for state checks
            home_position: Optional home position (lat, lon)

        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult(is_valid=True)

        # Basic checks
        self._check_waypoint_count(mission, result)

        if not mission.waypoints:
            return result

        # Get home position from drone or parameter
        home = home_position
        if home is None and drone is not None:
            home = drone.home_position

        # Waypoint-level checks
        for i, waypoint in enumerate(mission.waypoints):
            self._check_altitude(waypoint, i, result)
            self._check_speed(waypoint, i, result)
            if home:
                self._check_distance_from_home(waypoint, i, home, result)

        # Inter-waypoint checks
        self._check_waypoint_spacing(mission, result)

        # Mission-level checks
        self._check_battery_feasibility(mission, drone, result)

        # Geofence check if drone available
        if drone is not None:
            self._check_geofence_compliance(mission, drone, result)
            self._check_sensor_requirements(mission, drone, result)

        # Calculate feasibility score
        result.mission_feasibility = self._calculate_feasibility(result)

        return result

    def _check_waypoint_count(
        self,
        mission: Mission,
        result: ValidationResult
    ) -> None:
        """Check if mission has waypoints."""
        if not mission.waypoints:
            result.add_issue(ValidationIssue(
                validation_type=ValidationType.WAYPOINT_COUNT,
                severity=ValidationSeverity.ERROR,
                message="Mission has no waypoints",
            ))
        elif len(mission.waypoints) == 1:
            result.add_issue(ValidationIssue(
                validation_type=ValidationType.WAYPOINT_COUNT,
                severity=ValidationSeverity.WARNING,
                message="Mission has only one waypoint",
            ))

    def _check_altitude(
        self,
        waypoint: Waypoint,
        index: int,
        result: ValidationResult
    ) -> None:
        """Check waypoint altitude limits."""
        if waypoint.altitude > self.max_altitude:
            result.add_issue(ValidationIssue(
                validation_type=ValidationType.ALTITUDE_LIMITS,
                severity=ValidationSeverity.ERROR,
                message=f"Waypoint {index + 1} altitude ({waypoint.altitude:.1f}m) exceeds maximum ({self.max_altitude}m)",
                waypoint_index=index,
                details={'altitude': waypoint.altitude, 'max': self.max_altitude},
            ))
        elif waypoint.altitude < self.min_altitude:
            result.add_issue(ValidationIssue(
                validation_type=ValidationType.ALTITUDE_LIMITS,
                severity=ValidationSeverity.WARNING,
                message=f"Waypoint {index + 1} altitude ({waypoint.altitude:.1f}m) is below recommended minimum ({self.min_altitude}m)",
                waypoint_index=index,
                details={'altitude': waypoint.altitude, 'min': self.min_altitude},
            ))

    def _check_speed(
        self,
        waypoint: Waypoint,
        index: int,
        result: ValidationResult
    ) -> None:
        """Check waypoint speed limits."""
        if waypoint.speed > self.max_speed:
            result.add_issue(ValidationIssue(
                validation_type=ValidationType.SPEED_LIMITS,
                severity=ValidationSeverity.WARNING,
                message=f"Waypoint {index + 1} speed ({waypoint.speed:.1f}m/s) exceeds recommended maximum ({self.max_speed}m/s)",
                waypoint_index=index,
                details={'speed': waypoint.speed, 'max': self.max_speed},
            ))
        elif waypoint.speed < self.min_speed and waypoint.speed > 0:
            result.add_issue(ValidationIssue(
                validation_type=ValidationType.SPEED_LIMITS,
                severity=ValidationSeverity.INFO,
                message=f"Waypoint {index + 1} speed ({waypoint.speed:.1f}m/s) is very slow",
                waypoint_index=index,
            ))

    def _check_distance_from_home(
        self,
        waypoint: Waypoint,
        index: int,
        home: Tuple[float, float],
        result: ValidationResult
    ) -> None:
        """Check waypoint distance from home."""
        distance = self._haversine_distance(
            home[0], home[1],
            waypoint.latitude, waypoint.longitude
        )

        if distance > self.max_distance:
            result.add_issue(ValidationIssue(
                validation_type=ValidationType.DISTANCE_LIMITS,
                severity=ValidationSeverity.ERROR,
                message=f"Waypoint {index + 1} is {distance:.0f}m from home, exceeds maximum ({self.max_distance}m)",
                waypoint_index=index,
                details={'distance': distance, 'max': self.max_distance},
            ))
        elif distance > self.max_distance * 0.8:
            result.add_issue(ValidationIssue(
                validation_type=ValidationType.DISTANCE_LIMITS,
                severity=ValidationSeverity.WARNING,
                message=f"Waypoint {index + 1} is {distance:.0f}m from home, approaching maximum ({self.max_distance}m)",
                waypoint_index=index,
                details={'distance': distance, 'max': self.max_distance},
            ))

    def _check_waypoint_spacing(
        self,
        mission: Mission,
        result: ValidationResult
    ) -> None:
        """Check spacing between waypoints."""
        for i in range(len(mission.waypoints) - 1):
            wp1 = mission.waypoints[i]
            wp2 = mission.waypoints[i + 1]
            distance = wp1.distance_to(wp2)

            if distance < self.min_waypoint_spacing:
                result.add_issue(ValidationIssue(
                    validation_type=ValidationType.WAYPOINT_SPACING,
                    severity=ValidationSeverity.WARNING,
                    message=f"Waypoints {i + 1} and {i + 2} are very close ({distance:.2f}m)",
                    waypoint_index=i,
                    details={'distance': distance, 'min': self.min_waypoint_spacing},
                ))

    def _check_battery_feasibility(
        self,
        mission: Mission,
        drone: Optional['Drone'],
        result: ValidationResult
    ) -> None:
        """Check if battery is sufficient for mission."""
        stats = mission.get_statistics()

        # Calculate required battery
        distance_km = stats.total_distance / 1000
        time_minutes = stats.estimated_time / 60
        required_battery = (
            distance_km * self.BATTERY_PER_KM +
            time_minutes * self.BATTERY_PER_MINUTE +
            self.BATTERY_RESERVE
        )

        # Get current battery level
        current_battery = 100.0
        if drone is not None:
            current_battery = drone.battery_level

        if required_battery > current_battery:
            result.add_issue(ValidationIssue(
                validation_type=ValidationType.BATTERY_FEASIBILITY,
                severity=ValidationSeverity.ERROR,
                message=f"Insufficient battery: need ~{required_battery:.0f}%, have {current_battery:.0f}%",
                details={
                    'required': required_battery,
                    'available': current_battery,
                    'distance_km': distance_km,
                    'time_minutes': time_minutes,
                },
            ))
        elif required_battery > current_battery * 0.7:
            result.add_issue(ValidationIssue(
                validation_type=ValidationType.BATTERY_FEASIBILITY,
                severity=ValidationSeverity.WARNING,
                message=f"Battery may be marginal: need ~{required_battery:.0f}%, have {current_battery:.0f}%",
                details={
                    'required': required_battery,
                    'available': current_battery,
                },
            ))

    def _check_geofence_compliance(
        self,
        mission: Mission,
        drone: 'Drone',
        result: ValidationResult
    ) -> None:
        """Check if all waypoints are within geofence."""
        if not hasattr(drone, 'geofence_manager'):
            return

        from core.geofence import GeofenceZone

        for i, waypoint in enumerate(mission.waypoints):
            status = drone.geofence_manager.check_position(
                waypoint.latitude,
                waypoint.longitude,
                waypoint.altitude
            )

            if status.zone == GeofenceZone.OUTSIDE:
                fence_name = status.breached_fences[0] if status.breached_fences else 'geofence'
                result.add_issue(ValidationIssue(
                    validation_type=ValidationType.GEOFENCE_COMPLIANCE,
                    severity=ValidationSeverity.ERROR,
                    message=f"Waypoint {i + 1} is outside geofence '{fence_name}'",
                    waypoint_index=i,
                    details={'fence': fence_name, 'breached': status.breached_fences},
                ))

    def _check_sensor_requirements(
        self,
        mission: Mission,
        drone: 'Drone',
        result: ValidationResult
    ) -> None:
        """Check if required sensors are available."""
        # Check GPS
        if hasattr(drone, 'gps_sensor'):
            if not drone.gps_sensor.is_valid():
                result.add_issue(ValidationIssue(
                    validation_type=ValidationType.GPS_REQUIREMENT,
                    severity=ValidationSeverity.ERROR,
                    message="GPS is not available or has no fix",
                ))
            elif drone.gps_sensor.satellites < 6:
                result.add_issue(ValidationIssue(
                    validation_type=ValidationType.GPS_REQUIREMENT,
                    severity=ValidationSeverity.WARNING,
                    message=f"Low satellite count ({drone.gps_sensor.satellites}), GPS accuracy may be reduced",
                ))

    def _calculate_feasibility(self, result: ValidationResult) -> float:
        """Calculate overall feasibility score (0-1)."""
        if result.has_errors:
            return 0.0

        # Start at 1.0 and reduce for warnings
        score = 1.0
        for warning in result.warnings:
            if warning.validation_type == ValidationType.BATTERY_FEASIBILITY:
                score -= 0.2
            elif warning.validation_type == ValidationType.DISTANCE_LIMITS:
                score -= 0.1
            else:
                score -= 0.05

        return max(0.0, score)

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in meters."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return 6371000 * c

    def set_limits(
        self,
        max_altitude: Optional[float] = None,
        min_altitude: Optional[float] = None,
        max_distance: Optional[float] = None,
        max_speed: Optional[float] = None
    ) -> None:
        """
        Update validation limits.

        Args:
            max_altitude: Maximum altitude in meters
            min_altitude: Minimum altitude in meters
            max_distance: Maximum distance from home in meters
            max_speed: Maximum speed in m/s
        """
        if max_altitude is not None:
            self.max_altitude = max_altitude
        if min_altitude is not None:
            self.min_altitude = min_altitude
        if max_distance is not None:
            self.max_distance = max_distance
        if max_speed is not None:
            self.max_speed = max_speed
