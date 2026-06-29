"""
Natural Language Command Interpreter

Converts natural language commands into structured drone operations.
Supports mission planning, waypoint generation, and real-time commands.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

from .llm_backends import (
    LLMBackend,
    LLMConfig,
    LLMResponse,
    get_available_backend,
)


class CommandType(Enum):
    """Types of drone commands."""
    TAKEOFF = "takeoff"
    LAND = "land"
    GOTO = "goto"
    RTL = "rtl"
    HOVER = "hover"
    FOLLOW = "follow"
    ORBIT = "orbit"
    SURVEY = "survey"
    MISSION = "mission"
    SET_ALTITUDE = "set_altitude"
    SET_SPEED = "set_speed"
    EMERGENCY_STOP = "emergency_stop"
    ARM = "arm"
    DISARM = "disarm"
    UNKNOWN = "unknown"


@dataclass
class WaypointDef:
    """Waypoint definition from natural language."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: float = 30.0
    description: str = ""
    hold_time: float = 0.0


@dataclass
class DroneCommand:
    """Parsed drone command."""
    command_type: CommandType
    parameters: Dict[str, Any] = field(default_factory=dict)
    waypoints: List[WaypointDef] = field(default_factory=list)
    confidence: float = 0.0
    raw_text: str = ""
    interpretation: str = ""
    executable: bool = True
    requires_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'command_type': self.command_type.value,
            'parameters': self.parameters,
            'waypoints': [
                {
                    'lat': wp.latitude,
                    'lon': wp.longitude,
                    'alt': wp.altitude,
                    'description': wp.description,
                }
                for wp in self.waypoints
            ],
            'confidence': self.confidence,
            'interpretation': self.interpretation,
            'executable': self.executable,
            'requires_confirmation': self.requires_confirmation,
        }

    def to_mission(
        self,
        home_position: Optional[Tuple[float, float]] = None,
        default_altitude: float = 30.0,
        default_speed: float = 5.0,
        survey_width_m: float = 200.0,
        survey_height_m: float = 200.0,
        survey_spacing_m: float = 25.0,
        orbit_radius_m: float = 30.0,
        orbit_points: int = 12,
        mission_name: Optional[str] = None,
    ):
        """Materialize this command into a real ``core.mission.Mission``.

        Returns ``None`` if the command does not describe a multi-waypoint
        mission (e.g. ``LAND``, ``ARM``, ``HOVER``). For ``GOTO``/``MISSION``
        the waypoints in ``self.waypoints`` are used directly. ``SURVEY`` and
        ``ORBIT`` generate their respective patterns via
        ``SurveyPatternGenerator``. Falls back to ``home_position`` if a
        waypoint coordinate is missing.
        """
        # Local imports keep import-time cost low and avoid cycles.
        from core.mission.mission import Mission, MissionType, AltitudeFrame
        from core.mission.waypoint import Waypoint, WaypointType
        from core.mission.survey_patterns import SurveyPatternGenerator

        name = mission_name or (
            f"{self.command_type.value}: {self.interpretation[:60]}"
            if self.interpretation else f"LLM mission ({self.command_type.value})"
        )
        params = self.parameters or {}
        speed = float(params.get("speed", default_speed))
        altitude = float(params.get("altitude", default_altitude))

        def _coord(wp: WaypointDef) -> Tuple[float, float, float]:
            lat = wp.latitude if wp.latitude is not None else (home_position[0] if home_position else 0.0)
            lon = wp.longitude if wp.longitude is not None else (home_position[1] if home_position else 0.0)
            return lat, lon, wp.altitude or altitude

        if self.command_type in (CommandType.MISSION, CommandType.GOTO):
            if not self.waypoints:
                return None
            mission = Mission(
                name=name,
                description=self.raw_text or self.interpretation,
                mission_type=MissionType.WAYPOINT,
                default_speed=speed,
                default_altitude=altitude,
                altitude_frame=AltitudeFrame.RELATIVE,
                metadata={"source": "llm", "confidence": self.confidence},
            )
            for i, wp in enumerate(self.waypoints):
                lat, lon, alt = _coord(wp)
                mission.waypoints.append(Waypoint(
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    sequence=i,
                    name=wp.description or f"wp_{i}",
                    waypoint_type=WaypointType.NAVIGATE,
                    speed=speed,
                    hold_time=float(wp.hold_time or 0.0),
                ))
            return mission

        if self.command_type == CommandType.SURVEY:
            gen = SurveyPatternGenerator()
            center = (
                float(params.get("lat", home_position[0] if home_position else 0.0)),
                float(params.get("lon", home_position[1] if home_position else 0.0)),
            )
            width = float(params.get("width", survey_width_m))
            height = float(params.get("height", survey_height_m))
            spacing = float(params.get("spacing", params.get("line_spacing", survey_spacing_m)))
            pattern = params.get("pattern", "grid")
            if pattern == "orbit":
                radius = float(params.get("radius", orbit_radius_m))
                points = int(params.get("points", orbit_points))
                m = gen.generate_orbit(
                    center=center,
                    radius=radius,
                    altitude=altitude,
                    num_points=points,
                    speed=speed,
                    name=name,
                )
            else:
                m = gen.generate_grid(
                    center=center,
                    width=width,
                    height=height,
                    altitude=altitude,
                    line_spacing=spacing,
                    speed=speed,
                    name=name,
                )
            m.metadata.update({"source": "llm", "confidence": self.confidence})
            return m

        if self.command_type == CommandType.ORBIT:
            gen = SurveyPatternGenerator()
            center = (
                float(params.get("lat", home_position[0] if home_position else 0.0)),
                float(params.get("lon", home_position[1] if home_position else 0.0)),
            )
            radius = float(params.get("radius", orbit_radius_m))
            points = int(params.get("points", orbit_points))
            m = gen.generate_orbit(
                center=center,
                radius=radius,
                altitude=altitude,
                num_points=points,
                speed=speed,
                name=name,
            )
            m.metadata.update({"source": "llm", "confidence": self.confidence})
            return m

        return None


@dataclass
class InterpretedMission:
    """Complete mission interpreted from natural language."""
    name: str = "Mission"
    description: str = ""
    commands: List[DroneCommand] = field(default_factory=list)
    total_distance: float = 0.0
    estimated_duration: float = 0.0
    waypoint_count: int = 0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'commands': [c.to_dict() for c in self.commands],
            'total_distance': self.total_distance,
            'estimated_duration': self.estimated_duration,
            'waypoint_count': self.waypoint_count,
            'confidence': self.confidence,
        }


class CommandInterpreter:
    """
    Interprets natural language commands for drone operations.

    Uses LLM for understanding with rule-based fallback.
    """

    # Known location keywords for simple spatial reasoning
    LOCATION_KEYWORDS = {
        "here": (0, 0),
        "home": "home_position",
        "start": "home_position",
        "north": (0, 100),
        "south": (0, -100),
        "east": (100, 0),
        "west": (-100, 0),
    }

    # Distance patterns
    DISTANCE_PATTERN = re.compile(
        r'(\d+(?:\.\d+)?)\s*(m|meters?|km|kilometers?|ft|feet|miles?|mi)?',
        re.IGNORECASE
    )

    # Altitude patterns
    ALTITUDE_PATTERN = re.compile(
        r'(?:at|to|altitude)\s*(\d+(?:\.\d+)?)\s*(m|meters?|ft|feet)?',
        re.IGNORECASE
    )

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        config: Optional[LLMConfig] = None,
    ):
        """
        Initialize command interpreter.

        Args:
            backend: LLM backend to use (auto-detected if None)
            config: LLM configuration
        """
        self._config = config or LLMConfig()
        self._backend = backend or get_available_backend(self._config)
        self._context: Dict[str, Any] = {}

        # Location memory for contextual commands
        self._known_locations: Dict[str, Tuple[float, float]] = {}
        self._home_position: Optional[Tuple[float, float]] = None
        self._current_position: Optional[Tuple[float, float, float]] = None

    @property
    def backend_name(self) -> str:
        """Get current backend name."""
        return self._backend.name

    def set_home_position(self, lat: float, lon: float) -> None:
        """Set home position for 'home' references."""
        self._home_position = (lat, lon)

    def set_current_position(self, lat: float, lon: float, alt: float) -> None:
        """Set current drone position for relative commands."""
        self._current_position = (lat, lon, alt)

    def add_known_location(self, name: str, lat: float, lon: float) -> None:
        """Add a named location for reference."""
        self._known_locations[name.lower()] = (lat, lon)

    def interpret_to_mission(
        self,
        command_text: str,
        context: Optional[Dict[str, Any]] = None,
        home_position: Optional[Tuple[float, float]] = None,
        **mission_kwargs,
    ):
        """One-shot: NL → DroneCommand → ``Mission``.

        Returns a tuple ``(command, mission)``. ``mission`` is ``None`` if
        the command type doesn't materialize into a multi-waypoint mission
        (e.g. ``LAND``, ``HOVER``, ``ARM``); callers should fall back to
        their direct-command path in that case.
        """
        cmd = self.interpret(command_text, context=context)
        mission = cmd.to_mission(home_position=home_position, **mission_kwargs)
        return cmd, mission

    def interpret(
        self,
        command_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DroneCommand:
        """
        Interpret a natural language command.

        Args:
            command_text: Natural language command
            context: Optional context (current position, etc.)

        Returns:
            Parsed DroneCommand
        """
        if context:
            self._context.update(context)

        # Get system prompt
        system_prompt = self._backend.get_drone_system_prompt()

        # Add context to prompt
        enhanced_prompt = self._enhance_prompt(command_text)

        # Get LLM response
        response = self._backend.generate(enhanced_prompt, system_prompt)

        if not response.success:
            # Fall back to rule-based parsing
            return self._rule_based_parse(command_text)

        # Parse LLM response
        try:
            parsed = self._parse_llm_response(response.text, command_text)
            return parsed
        except Exception:
            return self._rule_based_parse(command_text)

    def interpret_mission(self, description: str) -> InterpretedMission:
        """
        Interpret a mission description into a complete mission.

        Args:
            description: Natural language mission description

        Returns:
            InterpretedMission with waypoints and commands
        """
        mission = InterpretedMission(description=description)

        # Use LLM to break down mission
        system_prompt = """You are a drone mission planner. Convert the mission description into a sequence of commands.

Output JSON array of commands:
[
    {"command": "takeoff", "altitude": 30},
    {"command": "goto", "description": "location name"},
    {"command": "survey", "pattern": "grid", "area": "description"},
    {"command": "land"}
]

Be specific about waypoints and actions."""

        response = self._backend.generate(description, system_prompt)

        if response.success:
            try:
                commands_data = json.loads(response.text)
                if isinstance(commands_data, list):
                    for cmd_data in commands_data:
                        cmd = self._convert_to_command(cmd_data)
                        if cmd:
                            mission.commands.append(cmd)
            except json.JSONDecodeError:
                # Fall back to single command interpretation
                cmd = self.interpret(description)
                mission.commands.append(cmd)

        mission.waypoint_count = sum(len(c.waypoints) for c in mission.commands)
        mission.confidence = (
            sum(c.confidence for c in mission.commands) / len(mission.commands)
            if mission.commands else 0
        )

        return mission

    def _enhance_prompt(self, command_text: str) -> str:
        """Add context to the prompt."""
        context_str = ""

        if self._current_position:
            lat, lon, alt = self._current_position
            context_str += f"\nCurrent position: ({lat:.6f}, {lon:.6f}) at {alt}m altitude."

        if self._home_position:
            lat, lon = self._home_position
            context_str += f"\nHome position: ({lat:.6f}, {lon:.6f})."

        if self._known_locations:
            locations = ", ".join(
                f"{name}: ({lat:.6f}, {lon:.6f})"
                for name, (lat, lon) in list(self._known_locations.items())[:5]
            )
            context_str += f"\nKnown locations: {locations}"

        return f"{command_text}{context_str}"

    def _parse_llm_response(self, response_text: str, original_command: str) -> DroneCommand:
        """Parse LLM JSON response into DroneCommand."""
        # Clean response (remove markdown code blocks if present)
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON in response")

        # Convert to DroneCommand
        cmd_type_str = data.get('command_type', 'unknown').upper()
        try:
            cmd_type = CommandType[cmd_type_str]
        except KeyError:
            cmd_type = CommandType.UNKNOWN

        # Parse waypoints
        waypoints = []
        for wp_data in data.get('waypoints', []):
            waypoints.append(WaypointDef(
                latitude=wp_data.get('lat'),
                longitude=wp_data.get('lon'),
                altitude=wp_data.get('alt', 30.0),
                description=wp_data.get('description', ''),
            ))

        return DroneCommand(
            command_type=cmd_type,
            parameters=data.get('parameters', {}),
            waypoints=waypoints,
            confidence=data.get('confidence', 0.7),
            raw_text=original_command,
            interpretation=f"Interpreted as {cmd_type.value}",
            executable=cmd_type != CommandType.UNKNOWN,
            requires_confirmation=cmd_type in [
                CommandType.EMERGENCY_STOP,
                CommandType.MISSION,
            ],
        )

    def _rule_based_parse(self, command_text: str) -> DroneCommand:
        """Fall back to rule-based command parsing."""
        text_lower = command_text.lower()

        # Emergency commands (highest priority)
        if any(word in text_lower for word in ["stop", "emergency", "abort", "kill"]):
            if "emergency" in text_lower or "kill" in text_lower:
                return DroneCommand(
                    command_type=CommandType.EMERGENCY_STOP,
                    confidence=0.95,
                    raw_text=command_text,
                    interpretation="Emergency stop command",
                    requires_confirmation=True,
                )
            return DroneCommand(
                command_type=CommandType.HOVER,
                confidence=0.9,
                raw_text=command_text,
                interpretation="Stop and hover",
            )

        # Takeoff
        if any(word in text_lower for word in ["take off", "takeoff", "launch", "lift"]):
            alt_match = self.ALTITUDE_PATTERN.search(command_text)
            altitude = 10.0
            if alt_match:
                altitude = float(alt_match.group(1))
                if alt_match.group(2) in ['ft', 'feet']:
                    altitude *= 0.3048

            return DroneCommand(
                command_type=CommandType.TAKEOFF,
                parameters={"altitude": altitude},
                confidence=0.9,
                raw_text=command_text,
                interpretation=f"Take off to {altitude}m",
            )

        # Land
        if any(word in text_lower for word in ["land", "touch down", "set down"]):
            return DroneCommand(
                command_type=CommandType.LAND,
                confidence=0.95,
                raw_text=command_text,
                interpretation="Land at current position",
            )

        # Return to launch
        if any(phrase in text_lower for phrase in ["return", "come back", "go home", "rtl", "rth"]):
            return DroneCommand(
                command_type=CommandType.RTL,
                confidence=0.9,
                raw_text=command_text,
                interpretation="Return to home position",
            )

        # Follow mode
        if any(word in text_lower for word in ["follow", "track", "chase"]):
            target = "person"
            if any(word in text_lower for word in ["car", "vehicle", "truck"]):
                target = "vehicle"
            return DroneCommand(
                command_type=CommandType.FOLLOW,
                parameters={"target": target},
                confidence=0.85,
                raw_text=command_text,
                interpretation=f"Follow {target}",
            )

        # Survey/scan
        if any(word in text_lower for word in ["survey", "scan", "map", "inspect"]):
            pattern = "grid"
            if "circle" in text_lower or "orbit" in text_lower:
                pattern = "orbit"
            elif "corridor" in text_lower or "line" in text_lower:
                pattern = "corridor"

            return DroneCommand(
                command_type=CommandType.SURVEY,
                parameters={"pattern": pattern},
                confidence=0.8,
                raw_text=command_text,
                interpretation=f"Survey with {pattern} pattern",
            )

        # Orbit
        if any(word in text_lower for word in ["orbit", "circle", "around"]):
            radius = 20.0
            dist_match = self.DISTANCE_PATTERN.search(command_text)
            if dist_match:
                radius = float(dist_match.group(1))
                unit = dist_match.group(2) or 'm'
                if unit.lower() in ['km', 'kilometers', 'kilometer']:
                    radius *= 1000
                elif unit.lower() in ['ft', 'feet']:
                    radius *= 0.3048

            return DroneCommand(
                command_type=CommandType.ORBIT,
                parameters={"radius": radius},
                confidence=0.85,
                raw_text=command_text,
                interpretation=f"Orbit at {radius}m radius",
            )

        # Go to / fly to
        if any(phrase in text_lower for phrase in ["go to", "fly to", "move to", "navigate to"]):
            # Check for known locations
            for location_name, coords in self._known_locations.items():
                if location_name in text_lower:
                    return DroneCommand(
                        command_type=CommandType.GOTO,
                        parameters={"location": location_name},
                        waypoints=[WaypointDef(
                            latitude=coords[0],
                            longitude=coords[1],
                            description=location_name,
                        )],
                        confidence=0.9,
                        raw_text=command_text,
                        interpretation=f"Fly to {location_name}",
                    )

            return DroneCommand(
                command_type=CommandType.GOTO,
                parameters={"description": command_text},
                confidence=0.6,
                raw_text=command_text,
                interpretation="Fly to described location (coordinates needed)",
                executable=False,
            )

        # Arm/disarm
        if "arm" in text_lower and "disarm" not in text_lower:
            return DroneCommand(
                command_type=CommandType.ARM,
                confidence=0.9,
                raw_text=command_text,
                interpretation="Arm the drone",
            )
        if "disarm" in text_lower:
            return DroneCommand(
                command_type=CommandType.DISARM,
                confidence=0.9,
                raw_text=command_text,
                interpretation="Disarm the drone",
            )

        # Unknown command
        return DroneCommand(
            command_type=CommandType.UNKNOWN,
            confidence=0.1,
            raw_text=command_text,
            interpretation="Could not understand command",
            executable=False,
        )

    def _convert_to_command(self, cmd_data: Dict[str, Any]) -> Optional[DroneCommand]:
        """Convert command dictionary to DroneCommand."""
        cmd_str = cmd_data.get('command', '').upper()

        try:
            cmd_type = CommandType[cmd_str]
        except KeyError:
            return None

        params = {k: v for k, v in cmd_data.items() if k != 'command'}

        return DroneCommand(
            command_type=cmd_type,
            parameters=params,
            confidence=0.8,
        )

    def get_suggestions(self, partial_command: str) -> List[str]:
        """Get command suggestions for autocomplete."""
        suggestions = []
        partial_lower = partial_command.lower()

        command_templates = [
            "take off to {altitude}m",
            "land",
            "return to home",
            "fly to {location}",
            "follow me",
            "survey the area",
            "orbit at {radius}m",
            "hold position",
            "emergency stop",
        ]

        for template in command_templates:
            if partial_lower in template.lower():
                suggestions.append(template)

        return suggestions[:5]

    def get_status(self) -> Dict[str, Any]:
        """Get interpreter status."""
        return {
            'backend': self._backend.name,
            'backend_available': self._backend.is_available(),
            'home_position': self._home_position,
            'known_locations': len(self._known_locations),
        }
