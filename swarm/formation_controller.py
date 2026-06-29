"""
Formation controller for swarm flying patterns.

Computes relative positions for drones in various
formation types (line, V, grid, circle, etc.).
"""

import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FormationType(Enum):
    """Available formation patterns."""
    LINE = "line"
    V_FORMATION = "v_formation"
    ECHELON_LEFT = "echelon_left"
    ECHELON_RIGHT = "echelon_right"
    DIAMOND = "diamond"
    GRID = "grid"
    CIRCLE = "circle"
    COLUMN = "column"
    WEDGE = "wedge"
    CUSTOM = "custom"


@dataclass
class FormationSlot:
    """A slot position within a formation."""
    slot_id: int
    offset_x: float  # meters, relative to leader (forward)
    offset_y: float  # meters, relative to leader (right)
    offset_z: float  # meters, relative to leader (down)
    drone_id: Optional[str] = None

    @property
    def offset_vector(self) -> Tuple[float, float, float]:
        """Get offset as (x, y, z) tuple."""
        return (self.offset_x, self.offset_y, self.offset_z)


@dataclass
class FormationConfig:
    """Configuration for a formation."""
    formation_type: FormationType
    spacing: float = 10.0  # meters between drones
    altitude_offset: float = 0.0  # stagger altitude
    angle: float = 45.0  # degrees for V/echelon
    rows: int = 2  # for grid formation
    cols: int = 2  # for grid formation
    radius: float = 20.0  # for circle formation


class FormationController:
    """
    Computes and manages drone formation positions.

    Generates slot positions relative to the leader
    and converts to absolute GPS coordinates.
    """

    # Earth radius for coordinate calculations
    EARTH_RADIUS = 6371000.0  # meters

    def __init__(self):
        """Initialize formation controller."""
        self._current_formation: Optional[FormationType] = None
        self._config = FormationConfig(FormationType.LINE)
        self._slots: List[FormationSlot] = []
        self._custom_offsets: Dict[int, Tuple[float, float, float]] = {}

    def set_formation(
        self,
        formation_type: FormationType,
        num_drones: int,
        spacing: float = 10.0,
        **kwargs,
    ) -> List[FormationSlot]:
        """
        Set formation type and compute slot positions.

        Args:
            formation_type: Type of formation
            num_drones: Number of drones in formation
            spacing: Distance between drones in meters
            **kwargs: Additional config (angle, rows, cols, etc.)

        Returns:
            List of FormationSlot with positions
        """
        self._config = FormationConfig(
            formation_type=formation_type,
            spacing=spacing,
            angle=kwargs.get('angle', 45.0),
            rows=kwargs.get('rows', 2),
            cols=kwargs.get('cols', 2),
            radius=kwargs.get('radius', 20.0),
            altitude_offset=kwargs.get('altitude_offset', 0.0),
        )

        self._current_formation = formation_type
        self._slots = self._compute_slots(num_drones)

        logger.info(
            f"Formation set: {formation_type.value} "
            f"with {num_drones} drones"
        )

        return self._slots

    def _compute_slots(self, num_drones: int) -> List[FormationSlot]:
        """Compute slot positions based on formation type."""
        if self._current_formation == FormationType.LINE:
            return self._compute_line(num_drones)
        elif self._current_formation == FormationType.V_FORMATION:
            return self._compute_v_formation(num_drones)
        elif self._current_formation == FormationType.ECHELON_LEFT:
            return self._compute_echelon(num_drones, left=True)
        elif self._current_formation == FormationType.ECHELON_RIGHT:
            return self._compute_echelon(num_drones, left=False)
        elif self._current_formation == FormationType.DIAMOND:
            return self._compute_diamond(num_drones)
        elif self._current_formation == FormationType.GRID:
            return self._compute_grid(num_drones)
        elif self._current_formation == FormationType.CIRCLE:
            return self._compute_circle(num_drones)
        elif self._current_formation == FormationType.COLUMN:
            return self._compute_column(num_drones)
        elif self._current_formation == FormationType.WEDGE:
            return self._compute_wedge(num_drones)
        elif self._current_formation == FormationType.CUSTOM:
            return self._compute_custom(num_drones)
        else:
            return self._compute_line(num_drones)

    def _compute_line(self, n: int) -> List[FormationSlot]:
        """Line formation (side by side)."""
        slots = []
        spacing = self._config.spacing

        for i in range(n):
            # Center the line around leader
            offset = (i - (n - 1) / 2) * spacing
            slots.append(FormationSlot(
                slot_id=i,
                offset_x=0,
                offset_y=offset,
                offset_z=i * self._config.altitude_offset,
            ))

        return slots

    def _compute_v_formation(self, n: int) -> List[FormationSlot]:
        """V formation (like flying geese)."""
        slots = []
        spacing = self._config.spacing
        angle_rad = math.radians(self._config.angle)

        # Leader at front
        slots.append(FormationSlot(
            slot_id=0,
            offset_x=0,
            offset_y=0,
            offset_z=0,
        ))

        # Alternating left and right
        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2

            offset_x = -row * spacing * math.cos(angle_rad)
            offset_y = side * row * spacing * math.sin(angle_rad)
            offset_z = row * self._config.altitude_offset

            slots.append(FormationSlot(
                slot_id=i,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_z=offset_z,
            ))

        return slots

    def _compute_echelon(
        self,
        n: int,
        left: bool = True,
    ) -> List[FormationSlot]:
        """Echelon formation (diagonal line)."""
        slots = []
        spacing = self._config.spacing
        angle_rad = math.radians(self._config.angle)
        side = -1 if left else 1

        for i in range(n):
            offset_x = -i * spacing * math.cos(angle_rad)
            offset_y = side * i * spacing * math.sin(angle_rad)
            offset_z = i * self._config.altitude_offset

            slots.append(FormationSlot(
                slot_id=i,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_z=offset_z,
            ))

        return slots

    def _compute_diamond(self, n: int) -> List[FormationSlot]:
        """Diamond formation."""
        slots = []
        spacing = self._config.spacing

        # Predefined diamond positions for common sizes
        if n >= 1:
            slots.append(FormationSlot(0, spacing, 0, 0))
        if n >= 2:
            slots.append(FormationSlot(1, 0, -spacing, 0))
        if n >= 3:
            slots.append(FormationSlot(2, 0, spacing, 0))
        if n >= 4:
            slots.append(FormationSlot(3, -spacing, 0, 0))

        # Additional drones fill in between
        for i in range(4, n):
            angle = (i - 4) * (2 * math.pi / (n - 4))
            r = spacing * 1.5
            slots.append(FormationSlot(
                slot_id=i,
                offset_x=r * math.cos(angle),
                offset_y=r * math.sin(angle),
                offset_z=self._config.altitude_offset,
            ))

        return slots

    def _compute_grid(self, n: int) -> List[FormationSlot]:
        """Grid formation (rows and columns)."""
        slots = []
        spacing = self._config.spacing
        rows = self._config.rows
        cols = self._config.cols

        # Adjust grid size if needed
        while rows * cols < n:
            if cols <= rows:
                cols += 1
            else:
                rows += 1

        slot_id = 0
        for row in range(rows):
            for col in range(cols):
                if slot_id >= n:
                    break

                # Center grid around leader
                offset_x = -(row - (rows - 1) / 2) * spacing
                offset_y = (col - (cols - 1) / 2) * spacing
                offset_z = 0

                slots.append(FormationSlot(
                    slot_id=slot_id,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    offset_z=offset_z,
                ))
                slot_id += 1

        return slots

    def _compute_circle(self, n: int) -> List[FormationSlot]:
        """Circle formation around center point."""
        slots = []
        radius = self._config.radius

        for i in range(n):
            angle = i * (2 * math.pi / n)
            slots.append(FormationSlot(
                slot_id=i,
                offset_x=radius * math.cos(angle),
                offset_y=radius * math.sin(angle),
                offset_z=i * self._config.altitude_offset,
            ))

        return slots

    def _compute_column(self, n: int) -> List[FormationSlot]:
        """Column formation (single file)."""
        slots = []
        spacing = self._config.spacing

        for i in range(n):
            slots.append(FormationSlot(
                slot_id=i,
                offset_x=-i * spacing,
                offset_y=0,
                offset_z=i * self._config.altitude_offset,
            ))

        return slots

    def _compute_wedge(self, n: int) -> List[FormationSlot]:
        """Wedge formation (inverted V)."""
        slots = []
        spacing = self._config.spacing
        angle_rad = math.radians(self._config.angle)

        # Leader at back
        slots.append(FormationSlot(
            slot_id=0,
            offset_x=0,
            offset_y=0,
            offset_z=0,
        ))

        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2

            # Wedge points forward (positive x)
            offset_x = row * spacing * math.cos(angle_rad)
            offset_y = side * row * spacing * math.sin(angle_rad)
            offset_z = 0

            slots.append(FormationSlot(
                slot_id=i,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_z=offset_z,
            ))

        return slots

    def _compute_custom(self, n: int) -> List[FormationSlot]:
        """Custom formation from user-defined offsets."""
        slots = []

        for i in range(n):
            if i in self._custom_offsets:
                ox, oy, oz = self._custom_offsets[i]
            else:
                ox, oy, oz = 0, 0, 0

            slots.append(FormationSlot(
                slot_id=i,
                offset_x=ox,
                offset_y=oy,
                offset_z=oz,
            ))

        return slots

    def set_custom_offset(
        self,
        slot_id: int,
        offset_x: float,
        offset_y: float,
        offset_z: float,
    ) -> None:
        """Set custom offset for a slot."""
        self._custom_offsets[slot_id] = (offset_x, offset_y, offset_z)

    def get_absolute_position(
        self,
        slot: FormationSlot,
        leader_lat: float,
        leader_lon: float,
        leader_alt: float,
        leader_heading: float,
    ) -> Tuple[float, float, float]:
        """
        Convert slot offset to absolute GPS coordinates.

        Args:
            slot: Formation slot with offsets
            leader_lat: Leader latitude
            leader_lon: Leader longitude
            leader_alt: Leader altitude
            leader_heading: Leader heading (degrees)

        Returns:
            Tuple of (lat, lon, alt) for the slot
        """
        # Rotate offsets by leader heading
        heading_rad = math.radians(leader_heading)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)

        # Rotated offsets (x=north, y=east)
        north = slot.offset_x * cos_h - slot.offset_y * sin_h
        east = slot.offset_x * sin_h + slot.offset_y * cos_h

        # Convert to lat/lon delta
        lat_offset = north / self.EARTH_RADIUS
        lon_offset = east / (
            self.EARTH_RADIUS * math.cos(math.radians(leader_lat))
        )

        lat = leader_lat + math.degrees(lat_offset)
        lon = leader_lon + math.degrees(lon_offset)
        alt = leader_alt - slot.offset_z

        return (lat, lon, alt)

    def get_all_positions(
        self,
        leader_lat: float,
        leader_lon: float,
        leader_alt: float,
        leader_heading: float,
    ) -> List[Tuple[int, float, float, float]]:
        """
        Get absolute positions for all slots.

        Returns:
            List of (slot_id, lat, lon, alt) tuples
        """
        positions = []

        for slot in self._slots:
            lat, lon, alt = self.get_absolute_position(
                slot,
                leader_lat,
                leader_lon,
                leader_alt,
                leader_heading,
            )
            positions.append((slot.slot_id, lat, lon, alt))

        return positions

    def assign_drone_to_slot(
        self,
        drone_id: str,
        slot_id: int,
    ) -> bool:
        """Assign a drone to a formation slot."""
        for slot in self._slots:
            if slot.slot_id == slot_id:
                slot.drone_id = drone_id
                return True
        return False

    def get_slot_for_drone(
        self,
        drone_id: str,
    ) -> Optional[FormationSlot]:
        """Get the slot assigned to a drone."""
        for slot in self._slots:
            if slot.drone_id == drone_id:
                return slot
        return None

    def get_target_position(
        self,
        drone_id: str,
        leader_lat: float,
        leader_lon: float,
        leader_alt: float,
        leader_heading: float,
    ) -> Optional[Tuple[float, float, float]]:
        """Get target position for a specific drone."""
        slot = self.get_slot_for_drone(drone_id)
        if not slot:
            return None

        return self.get_absolute_position(
            slot,
            leader_lat,
            leader_lon,
            leader_alt,
            leader_heading,
        )

    @property
    def slots(self) -> List[FormationSlot]:
        """Get current formation slots."""
        return self._slots.copy()

    @property
    def formation_type(self) -> Optional[FormationType]:
        """Get current formation type."""
        return self._current_formation
