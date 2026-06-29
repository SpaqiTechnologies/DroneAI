"""
MAVLink Protocol Implementation

Provides MAVLink v2.0 protocol support for communication with PX4/ArduPilot
flight controllers. This module implements message encoding/decoding,
heartbeat management, and telemetry streaming.

References:
- MAVLink 2.0 Specification: https://mavlink.io/en/
- Common Message Set: https://mavlink.io/en/messages/common.html
"""

from __future__ import annotations

import struct
import time
import threading
from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from abc import ABC, abstractmethod
import queue


# ---------------------------------------------------------------------------
# MAVLink Constants
# ---------------------------------------------------------------------------

MAVLINK_STX_V2 = 0xFD  # MAVLink 2.0 start byte
MAVLINK_MAX_PAYLOAD = 255
MAVLINK_HEADER_LEN = 10
MAVLINK_CHECKSUM_LEN = 2
MAVLINK_SIGNATURE_LEN = 13


class MAVComponent(IntEnum):
    """MAVLink component IDs."""
    AUTOPILOT = 1
    CAMERA = 100
    GIMBAL = 154
    GPS = 220
    ONBOARD_COMPUTER = 191
    ALL = 0


class MAVType(IntEnum):
    """MAVLink system types."""
    GENERIC = 0
    FIXED_WING = 1
    QUADROTOR = 2
    COAXIAL = 3
    HELICOPTER = 4
    ANTENNA_TRACKER = 5
    GCS = 6
    AIRSHIP = 7
    FREE_BALLOON = 8
    ROCKET = 9
    GROUND_ROVER = 10
    SURFACE_BOAT = 11
    SUBMARINE = 12
    HEXAROTOR = 13
    OCTOROTOR = 14
    TRICOPTER = 15
    FLAPPING_WING = 16
    KITE = 17
    ONBOARD_CONTROLLER = 18
    VTOL_DUOROTOR = 19
    VTOL_QUADROTOR = 20


class MAVAutopilot(IntEnum):
    """MAVLink autopilot types."""
    GENERIC = 0
    RESERVED = 1
    SLUGS = 2
    ARDUPILOTMEGA = 3
    OPENPILOT = 4
    GENERIC_WAYPOINTS_ONLY = 5
    GENERIC_WAYPOINTS_AND_SIMPLE = 6
    GENERIC_MISSION_FULL = 7
    INVALID = 8
    PPZ = 9
    UDB = 10
    FP = 11
    PX4 = 12
    SMACCMPILOT = 13
    AUTOQUAD = 14
    ARMAZILA = 15
    AEROB = 16
    ASLUAV = 17
    SMARTAP = 18


class MAVState(IntEnum):
    """MAVLink system state."""
    UNINIT = 0
    BOOT = 1
    CALIBRATING = 2
    STANDBY = 3
    ACTIVE = 4
    CRITICAL = 5
    EMERGENCY = 6
    POWEROFF = 7
    FLIGHT_TERMINATION = 8


class MAVModeFlag(IntEnum):
    """MAVLink mode flags (bitmask)."""
    CUSTOM_MODE_ENABLED = 1
    TEST_ENABLED = 2
    AUTO_ENABLED = 4
    GUIDED_ENABLED = 8
    STABILIZE_ENABLED = 16
    HIL_ENABLED = 32
    MANUAL_INPUT_ENABLED = 64
    SAFETY_ARMED = 128


class MAVResult(IntEnum):
    """Command acknowledgment result."""
    ACCEPTED = 0
    TEMPORARILY_REJECTED = 1
    DENIED = 2
    UNSUPPORTED = 3
    FAILED = 4
    IN_PROGRESS = 5
    CANCELLED = 6


class MAVCmd(IntEnum):
    """MAVLink command IDs (subset of common commands)."""
    NAV_WAYPOINT = 16
    NAV_LOITER_UNLIM = 17
    NAV_LOITER_TURNS = 18
    NAV_LOITER_TIME = 19
    NAV_RETURN_TO_LAUNCH = 20
    NAV_LAND = 21
    NAV_TAKEOFF = 22
    NAV_LAND_LOCAL = 23
    NAV_TAKEOFF_LOCAL = 24
    NAV_FOLLOW = 25
    NAV_CONTINUE_AND_CHANGE_ALT = 30
    NAV_LOITER_TO_ALT = 31
    DO_CHANGE_SPEED = 178
    DO_SET_HOME = 179
    DO_SET_SERVO = 183
    DO_SET_RELAY = 181
    DO_REPEAT_SERVO = 184
    DO_REPEAT_RELAY = 182
    DO_FLIGHTTERMINATION = 185
    DO_CHANGE_ALTITUDE = 186
    DO_SET_ACTUATOR = 187
    DO_LAND_START = 189
    DO_RALLY_LAND = 190
    DO_GO_AROUND = 191
    DO_REPOSITION = 192
    DO_PAUSE_CONTINUE = 193
    DO_SET_REVERSE = 194
    DO_SET_ROI_LOCATION = 195
    DO_SET_ROI_WPNEXT_OFFSET = 196
    DO_SET_ROI_NONE = 197
    DO_SET_ROI_SYSID = 198
    DO_CONTROL_VIDEO = 200
    DO_SET_ROI = 201
    DO_DIGICAM_CONFIGURE = 202
    DO_DIGICAM_CONTROL = 203
    PREFLIGHT_CALIBRATION = 241
    PREFLIGHT_SET_SENSOR_OFFSETS = 242
    PREFLIGHT_UAVCAN = 243
    PREFLIGHT_STORAGE = 245
    PREFLIGHT_REBOOT_SHUTDOWN = 246
    REQUEST_MESSAGE = 512
    SET_MESSAGE_INTERVAL = 511
    COMPONENT_ARM_DISARM = 400
    GET_HOME_POSITION = 410
    START_RX_PAIR = 500
    GET_MESSAGE_INTERVAL = 510
    REQUEST_AUTOPILOT_CAPABILITIES = 520


class MAVMsgId(IntEnum):
    """MAVLink message IDs (subset of common messages)."""
    HEARTBEAT = 0
    SYS_STATUS = 1
    SYSTEM_TIME = 2
    PING = 4
    CHANGE_OPERATOR_CONTROL = 5
    CHANGE_OPERATOR_CONTROL_ACK = 6
    AUTH_KEY = 7
    SET_MODE = 11
    PARAM_REQUEST_READ = 20
    PARAM_REQUEST_LIST = 21
    PARAM_VALUE = 22
    PARAM_SET = 23
    GPS_RAW_INT = 24
    GPS_STATUS = 25
    SCALED_IMU = 26
    RAW_IMU = 27
    RAW_PRESSURE = 28
    SCALED_PRESSURE = 29
    ATTITUDE = 30
    ATTITUDE_QUATERNION = 31
    LOCAL_POSITION_NED = 32
    GLOBAL_POSITION_INT = 33
    RC_CHANNELS_SCALED = 34
    RC_CHANNELS_RAW = 35
    SERVO_OUTPUT_RAW = 36
    MISSION_REQUEST_PARTIAL_LIST = 37
    MISSION_WRITE_PARTIAL_LIST = 38
    MISSION_ITEM = 39
    MISSION_REQUEST = 40
    MISSION_SET_CURRENT = 41
    MISSION_CURRENT = 42
    MISSION_REQUEST_LIST = 43
    MISSION_COUNT = 44
    MISSION_CLEAR_ALL = 45
    MISSION_ITEM_REACHED = 46
    MISSION_ACK = 47
    SET_GPS_GLOBAL_ORIGIN = 48
    GPS_GLOBAL_ORIGIN = 49
    PARAM_MAP_RC = 50
    MISSION_REQUEST_INT = 51
    SAFETY_SET_ALLOWED_AREA = 54
    SAFETY_ALLOWED_AREA = 55
    ATTITUDE_QUATERNION_COV = 61
    NAV_CONTROLLER_OUTPUT = 62
    GLOBAL_POSITION_INT_COV = 63
    LOCAL_POSITION_NED_COV = 64
    RC_CHANNELS = 65
    REQUEST_DATA_STREAM = 66
    DATA_STREAM = 67
    MANUAL_CONTROL = 69
    RC_CHANNELS_OVERRIDE = 70
    MISSION_ITEM_INT = 73
    VFR_HUD = 74
    COMMAND_INT = 75
    COMMAND_LONG = 76
    COMMAND_ACK = 77
    ATTITUDE_TARGET = 83
    POSITION_TARGET_LOCAL_NED = 85
    POSITION_TARGET_GLOBAL_INT = 87
    LOCAL_POSITION_NED_SYSTEM_GLOBAL_OFFSET = 89
    HIL_STATE = 90
    HIL_CONTROLS = 91
    HIL_RC_INPUTS_RAW = 92
    HIL_ACTUATOR_CONTROLS = 93
    OPTICAL_FLOW = 100
    GLOBAL_VISION_POSITION_ESTIMATE = 101
    VISION_POSITION_ESTIMATE = 102
    VISION_SPEED_ESTIMATE = 103
    VICON_POSITION_ESTIMATE = 104
    HIGHRES_IMU = 105
    OPTICAL_FLOW_RAD = 106
    HIL_SENSOR = 107
    SIM_STATE = 108
    RADIO_STATUS = 109
    FILE_TRANSFER_PROTOCOL = 110
    TIMESYNC = 111
    CAMERA_TRIGGER = 112
    HIL_GPS = 113
    HIL_OPTICAL_FLOW = 114
    HIL_STATE_QUATERNION = 115
    SCALED_IMU2 = 116
    LOG_REQUEST_LIST = 117
    LOG_ENTRY = 118
    LOG_REQUEST_DATA = 119
    LOG_DATA = 120
    LOG_ERASE = 121
    LOG_REQUEST_END = 122
    GPS_INJECT_DATA = 123
    GPS2_RAW = 124
    POWER_STATUS = 125
    SERIAL_CONTROL = 126
    GPS_RTK = 127
    GPS2_RTK = 128
    SCALED_IMU3 = 129
    DATA_TRANSMISSION_HANDSHAKE = 130
    ENCAPSULATED_DATA = 131
    DISTANCE_SENSOR = 132
    TERRAIN_REQUEST = 133
    TERRAIN_DATA = 134
    TERRAIN_CHECK = 135
    TERRAIN_REPORT = 136
    SCALED_PRESSURE2 = 137
    ATT_POS_MOCAP = 138
    SET_ACTUATOR_CONTROL_TARGET = 139
    ACTUATOR_CONTROL_TARGET = 140
    ALTITUDE = 141
    RESOURCE_REQUEST = 142
    SCALED_PRESSURE3 = 143
    FOLLOW_TARGET = 144
    CONTROL_SYSTEM_STATE = 146
    BATTERY_STATUS = 147
    AUTOPILOT_VERSION = 148
    LANDING_TARGET = 149
    ESTIMATOR_STATUS = 230
    WIND_COV = 231
    GPS_INPUT = 232
    GPS_RTCM_DATA = 233
    HIGH_LATENCY = 234
    HIGH_LATENCY2 = 235
    VIBRATION = 241
    HOME_POSITION = 242
    SET_HOME_POSITION = 243
    MESSAGE_INTERVAL = 244
    EXTENDED_SYS_STATE = 245
    ADSB_VEHICLE = 246
    COLLISION = 247
    V2_EXTENSION = 248
    MEMORY_VECT = 249
    DEBUG_VECT = 250
    NAMED_VALUE_FLOAT = 251
    NAMED_VALUE_INT = 252
    STATUSTEXT = 253
    DEBUG = 254


# ---------------------------------------------------------------------------
# Enums for Connection State
# ---------------------------------------------------------------------------

class ConnectionState(Enum):
    """MAVLink connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LOST = "lost"
    ERROR = "error"


class SystemStatus(Enum):
    """System status derived from MAVLink."""
    UNKNOWN = "unknown"
    BOOTING = "booting"
    CALIBRATING = "calibrating"
    STANDBY = "standby"
    ACTIVE = "active"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    POWEROFF = "poweroff"


class FlightMode(Enum):
    """Common flight modes (mapped from PX4/ArduPilot)."""
    UNKNOWN = "unknown"
    MANUAL = "manual"
    STABILIZE = "stabilize"
    ACRO = "acro"
    ALT_HOLD = "alt_hold"
    LOITER = "loiter"
    RTL = "rtl"
    AUTO = "auto"
    GUIDED = "guided"
    LAND = "land"
    TAKEOFF = "takeoff"
    OFFBOARD = "offboard"
    POSHOLD = "poshold"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MAVLinkMessage:
    """Base MAVLink message container."""
    msg_id: int
    system_id: int = 1
    component_id: int = 1
    sequence: int = 0
    payload: bytes = b''
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'msg_id': self.msg_id,
            'system_id': self.system_id,
            'component_id': self.component_id,
            'sequence': self.sequence,
            'payload_len': len(self.payload),
            'timestamp': self.timestamp,
        }


@dataclass
class MAVLinkHeartbeat:
    """Heartbeat message data."""
    system_id: int = 1
    component_id: int = 1
    type: int = MAVType.QUADROTOR
    autopilot: int = MAVAutopilot.PX4
    base_mode: int = 0
    custom_mode: int = 0
    system_status: int = MAVState.STANDBY
    mavlink_version: int = 3

    @property
    def is_armed(self) -> bool:
        """Check if system is armed."""
        return bool(self.base_mode & MAVModeFlag.SAFETY_ARMED)

    def to_bytes(self) -> bytes:
        """Encode heartbeat to bytes."""
        return struct.pack(
            '<BBBIBB',
            self.type,
            self.autopilot,
            self.base_mode,
            self.custom_mode,
            self.system_status,
            self.mavlink_version,
        )

    @classmethod
    def from_bytes(cls, data: bytes, sys_id: int = 1, comp_id: int = 1) -> 'MAVLinkHeartbeat':
        """Decode heartbeat from bytes."""
        if len(data) < 9:
            raise ValueError("Heartbeat payload too short")

        type_, autopilot, base_mode, custom_mode, status, version = struct.unpack(
            '<BBBIBB', data[:9]
        )

        return cls(
            system_id=sys_id,
            component_id=comp_id,
            type=type_,
            autopilot=autopilot,
            base_mode=base_mode,
            custom_mode=custom_mode,
            system_status=status,
            mavlink_version=version,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'system_id': self.system_id,
            'component_id': self.component_id,
            'type': self.type,
            'autopilot': self.autopilot,
            'base_mode': self.base_mode,
            'custom_mode': self.custom_mode,
            'system_status': self.system_status,
            'is_armed': self.is_armed,
        }


@dataclass
class MAVLinkTelemetry:
    """Aggregated telemetry data from MAVLink messages."""
    # Position
    latitude: float = 0.0       # degrees
    longitude: float = 0.0      # degrees
    altitude_msl: float = 0.0   # meters
    altitude_rel: float = 0.0   # meters above home

    # Velocity
    vx: float = 0.0             # m/s (NED)
    vy: float = 0.0
    vz: float = 0.0
    groundspeed: float = 0.0    # m/s

    # Attitude
    roll: float = 0.0           # radians
    pitch: float = 0.0
    yaw: float = 0.0
    heading: float = 0.0        # degrees

    # GPS
    gps_fix_type: int = 0
    satellites_visible: int = 0
    gps_hdop: float = 99.99
    gps_vdop: float = 99.99

    # Battery
    battery_voltage: float = 0.0     # V
    battery_current: float = 0.0     # A
    battery_remaining: int = 100     # %

    # Status
    armed: bool = False
    flight_mode: FlightMode = FlightMode.UNKNOWN
    system_status: SystemStatus = SystemStatus.UNKNOWN

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position': {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'altitude_msl': self.altitude_msl,
                'altitude_rel': self.altitude_rel,
            },
            'velocity': {
                'vx': self.vx,
                'vy': self.vy,
                'vz': self.vz,
                'groundspeed': self.groundspeed,
            },
            'attitude': {
                'roll': self.roll,
                'pitch': self.pitch,
                'yaw': self.yaw,
                'heading': self.heading,
            },
            'gps': {
                'fix_type': self.gps_fix_type,
                'satellites': self.satellites_visible,
                'hdop': self.gps_hdop,
                'vdop': self.gps_vdop,
            },
            'battery': {
                'voltage': self.battery_voltage,
                'current': self.battery_current,
                'remaining': self.battery_remaining,
            },
            'status': {
                'armed': self.armed,
                'flight_mode': self.flight_mode.value,
                'system_status': self.system_status.value,
            },
            'timestamp': self.timestamp,
        }


@dataclass
class MAVLinkCommand:
    """MAVLink command (COMMAND_LONG)."""
    command: int
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    param5: float = 0.0
    param6: float = 0.0
    param7: float = 0.0
    target_system: int = 1
    target_component: int = 1
    confirmation: int = 0

    def to_bytes(self) -> bytes:
        """Encode command to bytes."""
        return struct.pack(
            '<fffffffHBBB',
            self.param1, self.param2, self.param3, self.param4,
            self.param5, self.param6, self.param7,
            self.command,
            self.target_system, self.target_component,
            self.confirmation,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'command': self.command,
            'params': [
                self.param1, self.param2, self.param3, self.param4,
                self.param5, self.param6, self.param7,
            ],
            'target_system': self.target_system,
            'target_component': self.target_component,
        }


# ---------------------------------------------------------------------------
# CRC Implementation
# ---------------------------------------------------------------------------

# MAVLink CRC lookup table (X.25)
CRC_LOOKUP = [
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
    0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
    0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
    0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
    0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
    0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
    0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
    0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
    0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
    0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
    0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
    0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
    0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
    0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
    0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
    0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
    0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
    0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
    0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
    0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
    0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
    0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
    0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0,
]

# CRC extra bytes for each message (subset)
CRC_EXTRA = {
    MAVMsgId.HEARTBEAT: 50,
    MAVMsgId.SYS_STATUS: 124,
    MAVMsgId.GPS_RAW_INT: 24,
    MAVMsgId.ATTITUDE: 39,
    MAVMsgId.LOCAL_POSITION_NED: 185,
    MAVMsgId.GLOBAL_POSITION_INT: 104,
    MAVMsgId.MISSION_ITEM: 254,
    MAVMsgId.MISSION_REQUEST: 230,
    MAVMsgId.MISSION_COUNT: 221,
    MAVMsgId.MISSION_ACK: 153,
    MAVMsgId.COMMAND_LONG: 152,
    MAVMsgId.COMMAND_ACK: 143,
    MAVMsgId.VFR_HUD: 20,
    MAVMsgId.BATTERY_STATUS: 154,
    MAVMsgId.HOME_POSITION: 104,
    MAVMsgId.STATUSTEXT: 83,
}


def crc_accumulate(data: int, crc: int) -> int:
    """Accumulate one byte into CRC."""
    tmp = (data ^ (crc & 0xFF)) & 0xFF
    return (crc >> 8) ^ CRC_LOOKUP[tmp]


def crc_calculate(data: bytes, crc_extra: int = 0) -> int:
    """Calculate MAVLink CRC for data."""
    crc = 0xFFFF
    for byte in data:
        crc = crc_accumulate(byte, crc)
    if crc_extra:
        crc = crc_accumulate(crc_extra, crc)
    return crc


# ---------------------------------------------------------------------------
# MAVLink Connection
# ---------------------------------------------------------------------------

class MAVLinkConnection:
    """
    MAVLink connection manager for serial/UDP communication.

    Handles message encoding/decoding, heartbeat monitoring, and
    telemetry aggregation.
    """

    # Heartbeat timeout
    HEARTBEAT_TIMEOUT = 5.0  # seconds
    HEARTBEAT_INTERVAL = 1.0  # seconds

    def __init__(
        self,
        system_id: int = 255,
        component_id: int = MAVComponent.ONBOARD_COMPUTER,
        autopilot_type: int = MAVAutopilot.GENERIC,
    ):
        """
        Initialize MAVLink connection.

        Args:
            system_id: Our system ID (255 for GCS)
            component_id: Our component ID
            autopilot_type: Autopilot type we're compatible with
        """
        self._system_id = system_id
        self._component_id = component_id
        self._autopilot_type = autopilot_type

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._last_heartbeat = 0.0
        self._sequence = 0

        # Target system (discovered from heartbeat)
        self._target_system_id = 1
        self._target_component_id = 1

        # Telemetry
        self._telemetry = MAVLinkTelemetry()

        # Message queues
        self._tx_queue: queue.Queue = queue.Queue(maxsize=100)
        self._rx_queue: queue.Queue = queue.Queue(maxsize=100)

        # Callbacks
        self._message_callbacks: Dict[int, List[Callable]] = {}
        self._state_callbacks: List[Callable] = []

        # Threading
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            'tx_count': 0,
            'rx_count': 0,
            'errors': 0,
            'heartbeats_sent': 0,
            'heartbeats_received': 0,
        }

    @property
    def state(self) -> ConnectionState:
        """Get connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def telemetry(self) -> MAVLinkTelemetry:
        """Get current telemetry."""
        return self._telemetry

    @property
    def system_id(self) -> int:
        """Get our system ID."""
        return self._system_id

    @property
    def target_system_id(self) -> int:
        """Get target system ID."""
        return self._target_system_id

    def start(self) -> None:
        """Start the connection manager."""
        if self._running:
            return

        self._running = True
        self._state = ConnectionState.CONNECTING

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        """Stop the connection manager."""
        self._running = False
        self._state = ConnectionState.DISCONNECTED

        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)

    def register_callback(
        self,
        msg_id: int,
        callback: Callable[[MAVLinkMessage], None],
    ) -> None:
        """Register callback for specific message type."""
        if msg_id not in self._message_callbacks:
            self._message_callbacks[msg_id] = []
        self._message_callbacks[msg_id].append(callback)

    def register_state_callback(
        self,
        callback: Callable[[ConnectionState], None],
    ) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)

    def encode_message(self, msg_id: int, payload: bytes) -> bytes:
        """
        Encode a MAVLink 2.0 message.

        Args:
            msg_id: Message ID
            payload: Payload bytes

        Returns:
            Complete encoded message
        """
        with self._lock:
            seq = self._sequence
            self._sequence = (self._sequence + 1) % 256

        # Build header
        header = struct.pack(
            '<BBBBBBBBBH',
            MAVLINK_STX_V2,          # Start byte
            len(payload),            # Payload length
            0,                       # Incompatibility flags
            0,                       # Compatibility flags
            seq,                     # Sequence
            self._system_id,         # System ID
            self._component_id,      # Component ID
            msg_id & 0xFF,           # Message ID (low byte)
            (msg_id >> 8) & 0xFF,    # Message ID (mid byte)
            (msg_id >> 16) & 0xFF,   # Message ID (high byte)
        )

        # Calculate CRC
        crc_data = header[1:] + payload  # Exclude STX
        crc_extra = CRC_EXTRA.get(msg_id, 0)
        crc = crc_calculate(crc_data, crc_extra)

        # Build complete message
        return header + payload + struct.pack('<H', crc)

    def decode_message(self, data: bytes) -> Optional[MAVLinkMessage]:
        """
        Decode a MAVLink 2.0 message.

        Args:
            data: Raw message bytes

        Returns:
            Decoded message or None if invalid
        """
        if len(data) < MAVLINK_HEADER_LEN + MAVLINK_CHECKSUM_LEN:
            return None

        if data[0] != MAVLINK_STX_V2:
            return None

        payload_len = data[1]
        if len(data) < MAVLINK_HEADER_LEN + payload_len + MAVLINK_CHECKSUM_LEN:
            return None

        # Extract header fields
        _, length, incompat, compat, seq, sys_id, comp_id, msg_id_low, msg_id_mid, msg_id_high = struct.unpack(
            '<BBBBBBBBBH', data[:10]
        )
        # Reconstruct message ID
        msg_id = msg_id_low | (msg_id_mid << 8) | (msg_id_high << 16)

        # Extract payload
        payload = data[MAVLINK_HEADER_LEN:MAVLINK_HEADER_LEN + payload_len]

        # Verify CRC
        crc_data = data[1:MAVLINK_HEADER_LEN + payload_len]
        crc_extra = CRC_EXTRA.get(msg_id, 0)
        expected_crc = crc_calculate(crc_data, crc_extra)
        actual_crc = struct.unpack('<H', data[MAVLINK_HEADER_LEN + payload_len:MAVLINK_HEADER_LEN + payload_len + 2])[0]

        if expected_crc != actual_crc:
            self._stats['errors'] += 1
            return None

        self._stats['rx_count'] += 1

        return MAVLinkMessage(
            msg_id=msg_id,
            system_id=sys_id,
            component_id=comp_id,
            sequence=seq,
            payload=payload,
        )

    def process_message(self, msg: MAVLinkMessage) -> None:
        """
        Process a received message.

        Args:
            msg: Decoded message
        """
        # Update target system from first heartbeat
        if msg.msg_id == MAVMsgId.HEARTBEAT:
            self._process_heartbeat(msg)
        elif msg.msg_id == MAVMsgId.GLOBAL_POSITION_INT:
            self._process_global_position(msg)
        elif msg.msg_id == MAVMsgId.ATTITUDE:
            self._process_attitude(msg)
        elif msg.msg_id == MAVMsgId.GPS_RAW_INT:
            self._process_gps_raw(msg)
        elif msg.msg_id == MAVMsgId.BATTERY_STATUS:
            self._process_battery_status(msg)
        elif msg.msg_id == MAVMsgId.VFR_HUD:
            self._process_vfr_hud(msg)
        elif msg.msg_id == MAVMsgId.SYS_STATUS:
            self._process_sys_status(msg)

        # Call registered callbacks
        if msg.msg_id in self._message_callbacks:
            for callback in self._message_callbacks[msg.msg_id]:
                try:
                    callback(msg)
                except Exception:
                    pass

    def send_heartbeat(self) -> bytes:
        """Generate and return heartbeat message."""
        heartbeat = MAVLinkHeartbeat(
            system_id=self._system_id,
            component_id=self._component_id,
            type=MAVType.ONBOARD_CONTROLLER,
            autopilot=self._autopilot_type,
            base_mode=0,
            custom_mode=0,
            system_status=MAVState.ACTIVE,
        )

        msg = self.encode_message(MAVMsgId.HEARTBEAT, heartbeat.to_bytes())
        self._stats['heartbeats_sent'] += 1
        self._stats['tx_count'] += 1

        return msg

    def send_command(self, cmd: MAVLinkCommand) -> bytes:
        """
        Send a command (COMMAND_LONG).

        Args:
            cmd: Command to send

        Returns:
            Encoded message bytes
        """
        cmd.target_system = self._target_system_id
        cmd.target_component = self._target_component_id

        msg = self.encode_message(MAVMsgId.COMMAND_LONG, cmd.to_bytes())
        self._stats['tx_count'] += 1

        return msg

    def arm(self, force: bool = False) -> bytes:
        """Generate arm command."""
        return self.send_command(MAVLinkCommand(
            command=MAVCmd.COMPONENT_ARM_DISARM,
            param1=1.0,  # Arm
            param2=21196.0 if force else 0.0,  # Force arm magic number
        ))

    def disarm(self, force: bool = False) -> bytes:
        """Generate disarm command."""
        return self.send_command(MAVLinkCommand(
            command=MAVCmd.COMPONENT_ARM_DISARM,
            param1=0.0,  # Disarm
            param2=21196.0 if force else 0.0,
        ))

    def takeoff(self, altitude: float) -> bytes:
        """Generate takeoff command."""
        return self.send_command(MAVLinkCommand(
            command=MAVCmd.NAV_TAKEOFF,
            param7=altitude,  # Altitude
        ))

    def land(self) -> bytes:
        """Generate land command."""
        return self.send_command(MAVLinkCommand(
            command=MAVCmd.NAV_LAND,
        ))

    def rtl(self) -> bytes:
        """Generate RTL command."""
        return self.send_command(MAVLinkCommand(
            command=MAVCmd.NAV_RETURN_TO_LAUNCH,
        ))

    def goto(self, lat: float, lon: float, alt: float) -> bytes:
        """Generate goto command."""
        return self.send_command(MAVLinkCommand(
            command=MAVCmd.DO_REPOSITION,
            param5=lat,
            param6=lon,
            param7=alt,
        ))

    def set_mode(self, mode: FlightMode) -> bytes:
        """Generate set mode command."""
        # Mode mapping depends on autopilot type
        # This is a simplified version - real implementation needs
        # autopilot-specific mode numbers
        mode_map = {
            FlightMode.MANUAL: 0,
            FlightMode.STABILIZE: 1,
            FlightMode.ALT_HOLD: 2,
            FlightMode.LOITER: 3,
            FlightMode.AUTO: 4,
            FlightMode.GUIDED: 5,
            FlightMode.RTL: 6,
            FlightMode.LAND: 7,
        }

        custom_mode = mode_map.get(mode, 0)

        # SET_MODE message
        payload = struct.pack(
            '<IBB',
            custom_mode,
            self._target_system_id,
            MAVModeFlag.CUSTOM_MODE_ENABLED,
        )

        msg = self.encode_message(MAVMsgId.SET_MODE, payload)
        self._stats['tx_count'] += 1

        return msg

    def _process_heartbeat(self, msg: MAVLinkMessage) -> None:
        """Process heartbeat message."""
        try:
            heartbeat = MAVLinkHeartbeat.from_bytes(
                msg.payload, msg.system_id, msg.component_id
            )

            self._target_system_id = msg.system_id
            self._target_component_id = msg.component_id
            self._last_heartbeat = time.time()

            # Update telemetry
            self._telemetry.armed = heartbeat.is_armed
            self._telemetry.system_status = SystemStatus(
                MAVState(heartbeat.system_status).name.lower()
            ) if heartbeat.system_status in [s.value for s in MAVState] else SystemStatus.UNKNOWN

            # Update connection state
            if self._state != ConnectionState.CONNECTED:
                self._set_state(ConnectionState.CONNECTED)

            self._stats['heartbeats_received'] += 1

        except Exception:
            self._stats['errors'] += 1

    def _process_global_position(self, msg: MAVLinkMessage) -> None:
        """Process GLOBAL_POSITION_INT message."""
        if len(msg.payload) < 28:
            return

        lat, lon, alt, rel_alt, vx, vy, vz, hdg = struct.unpack(
            '<iiiihhhhH', msg.payload[:28]
        )

        self._telemetry.latitude = lat / 1e7
        self._telemetry.longitude = lon / 1e7
        self._telemetry.altitude_msl = alt / 1000.0
        self._telemetry.altitude_rel = rel_alt / 1000.0
        self._telemetry.vx = vx / 100.0
        self._telemetry.vy = vy / 100.0
        self._telemetry.vz = vz / 100.0
        self._telemetry.heading = hdg / 100.0
        self._telemetry.timestamp = time.time()

    def _process_attitude(self, msg: MAVLinkMessage) -> None:
        """Process ATTITUDE message."""
        if len(msg.payload) < 28:
            return

        _, roll, pitch, yaw, _, _, _ = struct.unpack(
            '<Iffffff', msg.payload[:28]
        )

        self._telemetry.roll = roll
        self._telemetry.pitch = pitch
        self._telemetry.yaw = yaw

    def _process_gps_raw(self, msg: MAVLinkMessage) -> None:
        """Process GPS_RAW_INT message."""
        if len(msg.payload) < 30:
            return

        _, fix_type, lat, lon, alt, eph, epv, vel, cog, satellites = struct.unpack(
            '<QBiiIHHHHB', msg.payload[:30]
        )

        self._telemetry.gps_fix_type = fix_type
        self._telemetry.satellites_visible = satellites
        self._telemetry.gps_hdop = eph / 100.0
        self._telemetry.gps_vdop = epv / 100.0

    def _process_battery_status(self, msg: MAVLinkMessage) -> None:
        """Process BATTERY_STATUS message."""
        if len(msg.payload) < 36:
            return

        # Battery status has complex structure, extract key fields
        voltages = struct.unpack('<10H', msg.payload[:20])
        current = struct.unpack('<h', msg.payload[20:22])[0]
        remaining = struct.unpack('<b', msg.payload[35:36])[0]

        # Use first non-zero voltage cell
        voltage = 0
        for v in voltages:
            if v != 0xFFFF:
                voltage = v
                break

        self._telemetry.battery_voltage = voltage / 1000.0
        self._telemetry.battery_current = current / 100.0
        self._telemetry.battery_remaining = max(0, remaining)

    def _process_vfr_hud(self, msg: MAVLinkMessage) -> None:
        """Process VFR_HUD message."""
        if len(msg.payload) < 20:
            return

        airspeed, groundspeed, heading, throttle, alt, climb = struct.unpack(
            '<ffffhh', msg.payload[:20]
        )

        self._telemetry.groundspeed = groundspeed

    def _process_sys_status(self, msg: MAVLinkMessage) -> None:
        """Process SYS_STATUS message."""
        if len(msg.payload) < 31:
            return

        # Extract battery info from SYS_STATUS
        voltage = struct.unpack('<H', msg.payload[14:16])[0]
        current = struct.unpack('<h', msg.payload[16:18])[0]
        remaining = struct.unpack('<b', msg.payload[30:31])[0]

        if voltage != 0xFFFF:
            self._telemetry.battery_voltage = voltage / 1000.0
        if current != -1:
            self._telemetry.battery_current = current / 100.0
        if remaining != -1:
            self._telemetry.battery_remaining = remaining

    def _set_state(self, new_state: ConnectionState) -> None:
        """Set connection state and notify callbacks."""
        if self._state != new_state:
            self._state = new_state
            for callback in self._state_callbacks:
                try:
                    callback(new_state)
                except Exception:
                    pass

    def _heartbeat_loop(self) -> None:
        """Heartbeat monitoring thread."""
        while self._running:
            # Check for heartbeat timeout
            if self._state == ConnectionState.CONNECTED:
                if time.time() - self._last_heartbeat > self.HEARTBEAT_TIMEOUT:
                    self._set_state(ConnectionState.LOST)

            time.sleep(0.5)

    def get_statistics(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            **self._stats,
            'state': self._state.value,
            'target_system': self._target_system_id,
            'last_heartbeat': self._last_heartbeat,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get connection status."""
        return {
            'state': self._state.value,
            'connected': self.is_connected,
            'target_system': self._target_system_id,
            'target_component': self._target_component_id,
            'telemetry': self._telemetry.to_dict(),
            'statistics': self._stats,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.get_status()
