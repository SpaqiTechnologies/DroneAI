"""
MAVLink Command Handler

High-level command interface for MAVLink operations including
mission management, parameter handling, and command acknowledgment.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Callable, Any, Tuple
import struct

from .mavlink_protocol import (
    MAVLinkConnection,
    MAVLinkMessage,
    MAVLinkCommand,
    MAVCmd,
    MAVMsgId,
    MAVResult,
    FlightMode,
)


class CommandType(Enum):
    """High-level command types."""
    ARM = "arm"
    DISARM = "disarm"
    TAKEOFF = "takeoff"
    LAND = "land"
    RTL = "rtl"
    GOTO = "goto"
    SET_MODE = "set_mode"
    SET_SPEED = "set_speed"
    SET_HOME = "set_home"
    PAUSE = "pause"
    CONTINUE = "continue"
    MISSION_START = "mission_start"
    MISSION_PAUSE = "mission_pause"
    MISSION_CLEAR = "mission_clear"
    REBOOT = "reboot"
    CALIBRATE = "calibrate"


@dataclass
class CommandResult:
    """Result of a command execution."""
    success: bool = False
    command_type: Optional[CommandType] = None
    result_code: int = MAVResult.FAILED
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'command_type': self.command_type.value if self.command_type else None,
            'result_code': self.result_code,
            'message': self.message,
            'timestamp': self.timestamp,
        }


@dataclass
class MissionItem:
    """Mission waypoint item."""
    seq: int = 0
    frame: int = 3  # MAV_FRAME_GLOBAL_RELATIVE_ALT
    command: int = MAVCmd.NAV_WAYPOINT
    current: int = 0
    autocontinue: int = 1
    param1: float = 0.0  # Hold time
    param2: float = 0.0  # Acceptance radius
    param3: float = 0.0  # Pass radius
    param4: float = 0.0  # Yaw
    x: float = 0.0       # Latitude
    y: float = 0.0       # Longitude
    z: float = 0.0       # Altitude

    def to_bytes(self, target_system: int, target_component: int) -> bytes:
        """Encode to MISSION_ITEM_INT format."""
        return struct.pack(
            '<BBBBBHHBBBfffffffI',
            target_system,
            target_component,
            self.seq,
            self.frame,
            self.command,
            self.current,
            self.autocontinue,
            self.param1,
            self.param2,
            self.param3,
            self.param4,
            int(self.x * 1e7),  # Latitude in 1e7 degrees
            int(self.y * 1e7),  # Longitude in 1e7 degrees
            self.z,
            0,  # Mission type
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], seq: int = 0) -> 'MissionItem':
        """Create from dictionary."""
        return cls(
            seq=seq,
            command=data.get('command', MAVCmd.NAV_WAYPOINT),
            x=data.get('latitude', data.get('x', 0.0)),
            y=data.get('longitude', data.get('y', 0.0)),
            z=data.get('altitude', data.get('z', 0.0)),
            param1=data.get('hold_time', 0.0),
            param2=data.get('acceptance_radius', 2.0),
            param4=data.get('yaw', 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'seq': self.seq,
            'command': self.command,
            'latitude': self.x,
            'longitude': self.y,
            'altitude': self.z,
            'hold_time': self.param1,
            'acceptance_radius': self.param2,
            'yaw': self.param4,
        }


class MissionState(Enum):
    """Mission upload/download state."""
    IDLE = "idle"
    UPLOADING = "uploading"
    DOWNLOADING = "downloading"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class MissionStatus:
    """Mission management status."""
    state: MissionState = MissionState.IDLE
    total_items: int = 0
    current_item: int = 0
    items: List[MissionItem] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'state': self.state.value,
            'total_items': self.total_items,
            'current_item': self.current_item,
            'items': [item.to_dict() for item in self.items],
            'error': self.error,
        }


class MAVLinkCommandHandler:
    """
    High-level MAVLink command handler.

    Provides command execution with acknowledgment handling,
    mission management, and parameter operations.
    """

    # Timeouts
    COMMAND_TIMEOUT = 5.0   # seconds
    MISSION_TIMEOUT = 30.0  # seconds

    def __init__(self, connection: MAVLinkConnection):
        """
        Initialize command handler.

        Args:
            connection: MAVLink connection instance
        """
        self._connection = connection

        # Command tracking
        self._pending_commands: Dict[int, Tuple[CommandType, float, Callable]] = {}
        self._command_results: List[CommandResult] = []
        self._command_lock = threading.Lock()

        # Mission management
        self._mission_status = MissionStatus()
        self._mission_lock = threading.Lock()

        # Parameter cache
        self._parameters: Dict[str, float] = {}

        # Callbacks
        self._result_callbacks: List[Callable[[CommandResult], None]] = []

        # Register message handlers
        self._connection.register_callback(MAVMsgId.COMMAND_ACK, self._handle_command_ack)
        self._connection.register_callback(MAVMsgId.MISSION_REQUEST, self._handle_mission_request)
        self._connection.register_callback(MAVMsgId.MISSION_ACK, self._handle_mission_ack)
        self._connection.register_callback(MAVMsgId.MISSION_COUNT, self._handle_mission_count)
        self._connection.register_callback(MAVMsgId.MISSION_ITEM, self._handle_mission_item)
        self._connection.register_callback(MAVMsgId.PARAM_VALUE, self._handle_param_value)

    def register_result_callback(
        self,
        callback: Callable[[CommandResult], None],
    ) -> None:
        """Register callback for command results."""
        self._result_callbacks.append(callback)

    def arm(self, force: bool = False) -> bytes:
        """
        Command the vehicle to arm.

        Args:
            force: Force arm even if pre-arm checks fail

        Returns:
            Encoded command bytes
        """
        return self._send_command(
            CommandType.ARM,
            self._connection.arm(force),
            MAVCmd.COMPONENT_ARM_DISARM,
        )

    def disarm(self, force: bool = False) -> bytes:
        """
        Command the vehicle to disarm.

        Args:
            force: Force disarm even while flying (dangerous!)

        Returns:
            Encoded command bytes
        """
        return self._send_command(
            CommandType.DISARM,
            self._connection.disarm(force),
            MAVCmd.COMPONENT_ARM_DISARM,
        )

    def takeoff(self, altitude: float) -> bytes:
        """
        Command takeoff to specified altitude.

        Args:
            altitude: Target altitude in meters

        Returns:
            Encoded command bytes
        """
        return self._send_command(
            CommandType.TAKEOFF,
            self._connection.takeoff(altitude),
            MAVCmd.NAV_TAKEOFF,
        )

    def land(self) -> bytes:
        """
        Command immediate landing.

        Returns:
            Encoded command bytes
        """
        return self._send_command(
            CommandType.LAND,
            self._connection.land(),
            MAVCmd.NAV_LAND,
        )

    def rtl(self) -> bytes:
        """
        Command return-to-launch.

        Returns:
            Encoded command bytes
        """
        return self._send_command(
            CommandType.RTL,
            self._connection.rtl(),
            MAVCmd.NAV_RETURN_TO_LAUNCH,
        )

    def goto(self, latitude: float, longitude: float, altitude: float) -> bytes:
        """
        Command vehicle to fly to position.

        Args:
            latitude: Target latitude in degrees
            longitude: Target longitude in degrees
            altitude: Target altitude in meters

        Returns:
            Encoded command bytes
        """
        return self._send_command(
            CommandType.GOTO,
            self._connection.goto(latitude, longitude, altitude),
            MAVCmd.DO_REPOSITION,
        )

    def set_mode(self, mode: FlightMode) -> bytes:
        """
        Set flight mode.

        Args:
            mode: Target flight mode

        Returns:
            Encoded command bytes
        """
        return self._send_command(
            CommandType.SET_MODE,
            self._connection.set_mode(mode),
            None,  # SET_MODE doesn't use COMMAND_ACK
        )

    def set_speed(self, speed: float, speed_type: int = 0) -> bytes:
        """
        Set target speed.

        Args:
            speed: Speed in m/s
            speed_type: 0=airspeed, 1=groundspeed, 2=climb, 3=descent

        Returns:
            Encoded command bytes
        """
        cmd = MAVLinkCommand(
            command=MAVCmd.DO_CHANGE_SPEED,
            param1=float(speed_type),
            param2=speed,
            param3=-1,  # Throttle (-1 = no change)
        )
        return self._send_command(
            CommandType.SET_SPEED,
            self._connection.send_command(cmd),
            MAVCmd.DO_CHANGE_SPEED,
        )

    def pause_mission(self) -> bytes:
        """
        Pause current mission.

        Returns:
            Encoded command bytes
        """
        cmd = MAVLinkCommand(
            command=MAVCmd.DO_PAUSE_CONTINUE,
            param1=0,  # 0 = pause
        )
        return self._send_command(
            CommandType.PAUSE,
            self._connection.send_command(cmd),
            MAVCmd.DO_PAUSE_CONTINUE,
        )

    def continue_mission(self) -> bytes:
        """
        Continue paused mission.

        Returns:
            Encoded command bytes
        """
        cmd = MAVLinkCommand(
            command=MAVCmd.DO_PAUSE_CONTINUE,
            param1=1,  # 1 = continue
        )
        return self._send_command(
            CommandType.CONTINUE,
            self._connection.send_command(cmd),
            MAVCmd.DO_PAUSE_CONTINUE,
        )

    def set_home(
        self,
        latitude: float = 0,
        longitude: float = 0,
        altitude: float = 0,
        use_current: bool = True,
    ) -> bytes:
        """
        Set home position.

        Args:
            latitude: Home latitude (ignored if use_current)
            longitude: Home longitude (ignored if use_current)
            altitude: Home altitude (ignored if use_current)
            use_current: Use current position as home

        Returns:
            Encoded command bytes
        """
        cmd = MAVLinkCommand(
            command=MAVCmd.DO_SET_HOME,
            param1=1 if use_current else 0,
            param5=latitude,
            param6=longitude,
            param7=altitude,
        )
        return self._send_command(
            CommandType.SET_HOME,
            self._connection.send_command(cmd),
            MAVCmd.DO_SET_HOME,
        )

    def upload_mission(self, items: List[MissionItem]) -> None:
        """
        Upload mission to vehicle.

        Args:
            items: List of mission items
        """
        with self._mission_lock:
            self._mission_status = MissionStatus(
                state=MissionState.UPLOADING,
                total_items=len(items),
                current_item=0,
                items=items,
            )

        # Send mission count to start upload
        count_msg = struct.pack(
            '<HBB',
            len(items),
            self._connection.target_system_id,
            1,  # Component ID
        )

        # Connection will handle the rest via callbacks

    def download_mission(self) -> None:
        """Start mission download from vehicle."""
        with self._mission_lock:
            self._mission_status = MissionStatus(
                state=MissionState.DOWNLOADING,
            )

        # Send mission request list
        request_msg = struct.pack(
            '<BB',
            self._connection.target_system_id,
            1,
        )

    def clear_mission(self) -> bytes:
        """
        Clear all mission items.

        Returns:
            Encoded command bytes
        """
        # MISSION_CLEAR_ALL
        payload = struct.pack(
            '<BB',
            self._connection.target_system_id,
            1,
        )
        msg = self._connection.encode_message(MAVMsgId.MISSION_CLEAR_ALL, payload)

        with self._mission_lock:
            self._mission_status = MissionStatus(state=MissionState.IDLE)

        return msg

    def get_mission_status(self) -> MissionStatus:
        """Get current mission status."""
        with self._mission_lock:
            return self._mission_status

    def request_parameter(self, param_id: str) -> bytes:
        """
        Request a parameter value.

        Args:
            param_id: Parameter name (max 16 chars)

        Returns:
            Encoded request bytes
        """
        param_id_bytes = param_id.encode('utf-8')[:16].ljust(16, b'\x00')
        payload = struct.pack(
            '<BBh16s',
            self._connection.target_system_id,
            1,  # Component ID
            -1,  # param_index (-1 to use param_id)
            param_id_bytes,
        )
        return self._connection.encode_message(MAVMsgId.PARAM_REQUEST_READ, payload)

    def set_parameter(self, param_id: str, value: float, param_type: int = 9) -> bytes:
        """
        Set a parameter value.

        Args:
            param_id: Parameter name
            value: Parameter value
            param_type: Parameter type (9 = float)

        Returns:
            Encoded message bytes
        """
        param_id_bytes = param_id.encode('utf-8')[:16].ljust(16, b'\x00')
        payload = struct.pack(
            '<BB16sfB',
            self._connection.target_system_id,
            1,
            param_id_bytes,
            value,
            param_type,
        )
        return self._connection.encode_message(MAVMsgId.PARAM_SET, payload)

    def get_parameter(self, param_id: str) -> Optional[float]:
        """Get cached parameter value."""
        return self._parameters.get(param_id)

    def get_last_result(self) -> Optional[CommandResult]:
        """Get most recent command result."""
        with self._command_lock:
            if self._command_results:
                return self._command_results[-1]
        return None

    def get_command_history(self, limit: int = 10) -> List[CommandResult]:
        """Get recent command results."""
        with self._command_lock:
            return list(self._command_results[-limit:])

    def _send_command(
        self,
        cmd_type: CommandType,
        msg_bytes: bytes,
        expected_cmd: Optional[int],
    ) -> bytes:
        """Send command and track for acknowledgment."""
        if expected_cmd is not None:
            with self._command_lock:
                self._pending_commands[expected_cmd] = (
                    cmd_type,
                    time.time(),
                    None,
                )
        return msg_bytes

    def _handle_command_ack(self, msg: MAVLinkMessage) -> None:
        """Handle COMMAND_ACK message."""
        if len(msg.payload) < 3:
            return

        command, result = struct.unpack('<HB', msg.payload[:3])

        with self._command_lock:
            if command in self._pending_commands:
                cmd_type, start_time, _ = self._pending_commands.pop(command)

                result_obj = CommandResult(
                    success=(result == MAVResult.ACCEPTED),
                    command_type=cmd_type,
                    result_code=result,
                    message=self._result_to_string(result),
                )

                self._command_results.append(result_obj)

                # Limit history size
                if len(self._command_results) > 100:
                    self._command_results = self._command_results[-50:]

                # Notify callbacks
                for callback in self._result_callbacks:
                    try:
                        callback(result_obj)
                    except Exception:
                        pass

    def _handle_mission_request(self, msg: MAVLinkMessage) -> None:
        """Handle MISSION_REQUEST message."""
        if len(msg.payload) < 2:
            return

        seq = struct.unpack('<H', msg.payload[:2])[0]

        with self._mission_lock:
            if self._mission_status.state == MissionState.UPLOADING:
                if seq < len(self._mission_status.items):
                    self._mission_status.current_item = seq
                    # Send mission item
                    item = self._mission_status.items[seq]
                    # Encode and send via connection

    def _handle_mission_ack(self, msg: MAVLinkMessage) -> None:
        """Handle MISSION_ACK message."""
        if len(msg.payload) < 3:
            return

        target_system, target_component, ack_type = struct.unpack('<BBB', msg.payload[:3])

        with self._mission_lock:
            if ack_type == 0:  # MAV_MISSION_ACCEPTED
                self._mission_status.state = MissionState.COMPLETE
            else:
                self._mission_status.state = MissionState.FAILED
                self._mission_status.error = f"Mission rejected: {ack_type}"

    def _handle_mission_count(self, msg: MAVLinkMessage) -> None:
        """Handle MISSION_COUNT message."""
        if len(msg.payload) < 4:
            return

        count = struct.unpack('<H', msg.payload[:2])[0]

        with self._mission_lock:
            if self._mission_status.state == MissionState.DOWNLOADING:
                self._mission_status.total_items = count
                self._mission_status.items = []
                # Request first item

    def _handle_mission_item(self, msg: MAVLinkMessage) -> None:
        """Handle MISSION_ITEM message."""
        if len(msg.payload) < 37:
            return

        # Parse mission item
        target_system, target_component, seq, frame, command, current, autocontinue = struct.unpack(
            '<BBHBHBB', msg.payload[:8]
        )
        param1, param2, param3, param4, x, y, z = struct.unpack(
            '<ffffffI', msg.payload[8:36]
        )

        item = MissionItem(
            seq=seq,
            frame=frame,
            command=command,
            current=current,
            autocontinue=autocontinue,
            param1=param1,
            param2=param2,
            param3=param3,
            param4=param4,
            x=x / 1e7,
            y=y / 1e7,
            z=z,
        )

        with self._mission_lock:
            if self._mission_status.state == MissionState.DOWNLOADING:
                self._mission_status.items.append(item)
                self._mission_status.current_item = seq

                if len(self._mission_status.items) >= self._mission_status.total_items:
                    self._mission_status.state = MissionState.COMPLETE
                # Request next item

    def _handle_param_value(self, msg: MAVLinkMessage) -> None:
        """Handle PARAM_VALUE message."""
        if len(msg.payload) < 25:
            return

        param_id = msg.payload[:16].rstrip(b'\x00').decode('utf-8', errors='ignore')
        value = struct.unpack('<f', msg.payload[16:20])[0]

        self._parameters[param_id] = value

    def _result_to_string(self, result: int) -> str:
        """Convert result code to string."""
        result_strings = {
            MAVResult.ACCEPTED: "Command accepted",
            MAVResult.TEMPORARILY_REJECTED: "Temporarily rejected",
            MAVResult.DENIED: "Denied",
            MAVResult.UNSUPPORTED: "Unsupported",
            MAVResult.FAILED: "Failed",
            MAVResult.IN_PROGRESS: "In progress",
            MAVResult.CANCELLED: "Cancelled",
        }
        return result_strings.get(result, f"Unknown result: {result}")

    def check_timeouts(self) -> None:
        """Check for command timeouts."""
        current_time = time.time()

        with self._command_lock:
            expired = []
            for cmd_id, (cmd_type, start_time, _) in self._pending_commands.items():
                if current_time - start_time > self.COMMAND_TIMEOUT:
                    expired.append(cmd_id)

            for cmd_id in expired:
                cmd_type, start_time, _ = self._pending_commands.pop(cmd_id)
                result = CommandResult(
                    success=False,
                    command_type=cmd_type,
                    result_code=MAVResult.FAILED,
                    message="Command timed out",
                )
                self._command_results.append(result)

                for callback in self._result_callbacks:
                    try:
                        callback(result)
                    except Exception:
                        pass

    def get_status(self) -> Dict[str, Any]:
        """Get handler status."""
        with self._command_lock:
            pending = {
                cmd_id: cmd_type.value
                for cmd_id, (cmd_type, _, _) in self._pending_commands.items()
            }

        return {
            'pending_commands': pending,
            'mission': self.get_mission_status().to_dict(),
            'parameter_count': len(self._parameters),
            'history_size': len(self._command_results),
        }
