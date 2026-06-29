"""
Inter-drone communication for swarm coordination.

Provides UDP broadcast for state sharing and TCP
for reliable command delivery between drones.
"""

import json
import socket
import struct
import threading
import time
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-drone messages."""
    HEARTBEAT = "heartbeat"
    STATE_UPDATE = "state_update"
    COMMAND = "command"
    COMMAND_ACK = "command_ack"
    FORMATION_UPDATE = "formation_update"
    EMERGENCY = "emergency"
    LEADER_ELECTION = "leader_election"
    COLLISION_WARNING = "collision_warning"
    MISSION_SYNC = "mission_sync"


@dataclass
class DroneMessage:
    """Message structure for inter-drone communication."""
    msg_type: MessageType
    sender_id: str
    timestamp: float
    sequence: int
    payload: Dict[str, Any]
    target_id: Optional[str] = None  # None = broadcast

    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        data = {
            'msg_type': self.msg_type.value,
            'sender_id': self.sender_id,
            'timestamp': self.timestamp,
            'sequence': self.sequence,
            'payload': self.payload,
            'target_id': self.target_id,
        }
        json_str = json.dumps(data)
        return json_str.encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> 'DroneMessage':
        """Deserialize message from bytes."""
        json_str = data.decode('utf-8')
        obj = json.loads(json_str)
        return cls(
            msg_type=MessageType(obj['msg_type']),
            sender_id=obj['sender_id'],
            timestamp=obj['timestamp'],
            sequence=obj['sequence'],
            payload=obj['payload'],
            target_id=obj.get('target_id'),
        )


class InterDroneComm:
    """
    Inter-drone communication handler.

    Uses UDP for broadcast messages (heartbeats, state)
    and TCP for reliable command delivery.
    """

    # Protocol constants
    UDP_PORT = 14550
    TCP_PORT = 14551
    MULTICAST_GROUP = "239.255.42.42"
    MAX_MESSAGE_SIZE = 65535
    HEARTBEAT_INTERVAL = 0.1  # 10 Hz

    def __init__(
        self,
        drone_id: str,
        broadcast_port: int = UDP_PORT,
        command_port: int = TCP_PORT,
        use_multicast: bool = True,
    ):
        """
        Initialize inter-drone communication.

        Args:
            drone_id: This drone's unique identifier
            broadcast_port: UDP port for broadcasts
            command_port: TCP port for commands
            use_multicast: Use multicast instead of broadcast
        """
        self.drone_id = drone_id
        self.broadcast_port = broadcast_port
        self.command_port = command_port
        self.use_multicast = use_multicast

        # Sockets
        self._udp_socket: Optional[socket.socket] = None
        self._tcp_socket: Optional[socket.socket] = None
        self._tcp_connections: Dict[str, socket.socket] = {}

        # State
        self._sequence = 0
        self._running = False
        self._lock = threading.Lock()

        # Callbacks
        self._message_handlers: Dict[
            MessageType, List[Callable]
        ] = {mt: [] for mt in MessageType}

        # Threads
        self._udp_recv_thread: Optional[threading.Thread] = None
        self._tcp_accept_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Heartbeat state to broadcast
        self._heartbeat_payload: Dict = {}

    def start(self) -> bool:
        """
        Start communication services.

        Returns:
            True if started successfully
        """
        try:
            self._setup_udp_socket()
            self._setup_tcp_socket()
            self._running = True

            # Start receiver threads
            self._udp_recv_thread = threading.Thread(
                target=self._udp_receive_loop,
                daemon=True,
            )
            self._udp_recv_thread.start()

            self._tcp_accept_thread = threading.Thread(
                target=self._tcp_accept_loop,
                daemon=True,
            )
            self._tcp_accept_thread.start()

            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
            )
            self._heartbeat_thread.start()

            logger.info(
                f"InterDroneComm started for {self.drone_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start communication: {e}")
            return False

    def stop(self) -> None:
        """Stop communication services."""
        self._running = False

        # Close sockets
        if self._udp_socket:
            try:
                self._udp_socket.close()
            except Exception:
                pass

        if self._tcp_socket:
            try:
                self._tcp_socket.close()
            except Exception:
                pass

        for conn in self._tcp_connections.values():
            try:
                conn.close()
            except Exception:
                pass

        self._tcp_connections.clear()
        logger.info("InterDroneComm stopped")

    def _setup_udp_socket(self) -> None:
        """Setup UDP socket for broadcast/multicast."""
        self._udp_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
            socket.IPPROTO_UDP,
        )
        self._udp_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        if self.use_multicast:
            # Join multicast group
            self._udp_socket.bind(('', self.broadcast_port))
            mreq = struct.pack(
                "4sl",
                socket.inet_aton(self.MULTICAST_GROUP),
                socket.INADDR_ANY,
            )
            self._udp_socket.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                mreq,
            )
            # Set multicast TTL
            self._udp_socket.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_TTL,
                2,
            )
        else:
            # Enable broadcast
            self._udp_socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_BROADCAST,
                1,
            )
            self._udp_socket.bind(('', self.broadcast_port))

        self._udp_socket.settimeout(1.0)

    def _setup_tcp_socket(self) -> None:
        """Setup TCP socket for commands."""
        self._tcp_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        self._tcp_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        self._tcp_socket.bind(('', self.command_port))
        self._tcp_socket.listen(10)
        self._tcp_socket.settimeout(1.0)

    def _udp_receive_loop(self) -> None:
        """Receive UDP messages in background."""
        while self._running:
            try:
                data, addr = self._udp_socket.recvfrom(
                    self.MAX_MESSAGE_SIZE
                )
                self._handle_message(data, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"UDP receive error: {e}")

    def _tcp_accept_loop(self) -> None:
        """Accept TCP connections in background."""
        while self._running:
            try:
                conn, addr = self._tcp_socket.accept()
                threading.Thread(
                    target=self._handle_tcp_connection,
                    args=(conn, addr),
                    daemon=True,
                ).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"TCP accept error: {e}")

    def _handle_tcp_connection(
        self,
        conn: socket.socket,
        addr: tuple,
    ) -> None:
        """Handle incoming TCP connection."""
        conn.settimeout(5.0)
        try:
            while self._running:
                # Read message length (4 bytes)
                length_data = conn.recv(4)
                if not length_data:
                    break

                msg_len = struct.unpack('!I', length_data)[0]
                if msg_len > self.MAX_MESSAGE_SIZE:
                    logger.warning(f"Message too large: {msg_len}")
                    break

                # Read message data
                data = b''
                while len(data) < msg_len:
                    chunk = conn.recv(msg_len - len(data))
                    if not chunk:
                        break
                    data += chunk

                if len(data) == msg_len:
                    self._handle_message(data, addr)

        except socket.timeout:
            pass
        except Exception as e:
            logger.debug(f"TCP connection closed: {e}")
        finally:
            conn.close()

    def _handle_message(
        self,
        data: bytes,
        addr: tuple,
    ) -> None:
        """Process received message."""
        try:
            msg = DroneMessage.from_bytes(data)

            # Ignore own messages
            if msg.sender_id == self.drone_id:
                return

            # Check if message is for us or broadcast
            if msg.target_id and msg.target_id != self.drone_id:
                return

            # Invoke handlers
            handlers = self._message_handlers.get(msg.msg_type, [])
            for handler in handlers:
                try:
                    handler(msg)
                except Exception as e:
                    logger.error(f"Handler error: {e}")

        except Exception as e:
            logger.debug(f"Failed to parse message: {e}")

    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            self.broadcast_heartbeat()
            time.sleep(self.HEARTBEAT_INTERVAL)

    def set_heartbeat_payload(self, payload: Dict) -> None:
        """
        Set the payload to include in heartbeats.

        Args:
            payload: Dict with position, velocity, status, etc.
        """
        with self._lock:
            self._heartbeat_payload = payload.copy()

    def broadcast_heartbeat(self) -> None:
        """Send heartbeat to all drones."""
        with self._lock:
            payload = self._heartbeat_payload.copy()

        self.broadcast(MessageType.HEARTBEAT, payload)

    def broadcast(
        self,
        msg_type: MessageType,
        payload: Dict,
    ) -> bool:
        """
        Broadcast message to all drones.

        Args:
            msg_type: Type of message
            payload: Message payload

        Returns:
            True if sent successfully
        """
        msg = self._create_message(msg_type, payload)
        data = msg.to_bytes()

        try:
            if self.use_multicast:
                addr = (self.MULTICAST_GROUP, self.broadcast_port)
            else:
                addr = ('<broadcast>', self.broadcast_port)

            self._udp_socket.sendto(data, addr)
            return True

        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
            return False

    def send_to(
        self,
        target_id: str,
        msg_type: MessageType,
        payload: Dict,
        reliable: bool = False,
    ) -> bool:
        """
        Send message to specific drone.

        Args:
            target_id: Target drone ID
            msg_type: Type of message
            payload: Message payload
            reliable: Use TCP for guaranteed delivery

        Returns:
            True if sent successfully
        """
        msg = self._create_message(msg_type, payload, target_id)

        if reliable:
            return self._send_tcp(target_id, msg)
        else:
            return self._send_udp(msg)

    def _send_udp(self, msg: DroneMessage) -> bool:
        """Send message via UDP."""
        data = msg.to_bytes()
        try:
            if self.use_multicast:
                addr = (self.MULTICAST_GROUP, self.broadcast_port)
            else:
                addr = ('<broadcast>', self.broadcast_port)

            self._udp_socket.sendto(data, addr)
            return True
        except Exception as e:
            logger.error(f"UDP send failed: {e}")
            return False

    def _send_tcp(
        self,
        target_id: str,
        msg: DroneMessage,
    ) -> bool:
        """Send message via TCP (reliable)."""
        # Note: In real implementation, we'd need address lookup
        # For simulation, we use broadcast with target filtering
        logger.warning(
            f"TCP send to {target_id} not fully implemented"
        )
        return self._send_udp(msg)

    def _create_message(
        self,
        msg_type: MessageType,
        payload: Dict,
        target_id: Optional[str] = None,
    ) -> DroneMessage:
        """Create a new message with sequence number."""
        with self._lock:
            self._sequence += 1
            seq = self._sequence

        return DroneMessage(
            msg_type=msg_type,
            sender_id=self.drone_id,
            timestamp=time.time(),
            sequence=seq,
            payload=payload,
            target_id=target_id,
        )

    def on_message(
        self,
        msg_type: MessageType,
        handler: Callable[[DroneMessage], None],
    ) -> None:
        """
        Register handler for message type.

        Args:
            msg_type: Type of message to handle
            handler: Callback function(message)
        """
        self._message_handlers[msg_type].append(handler)

    def send_command(
        self,
        target_id: str,
        command: str,
        params: Optional[Dict] = None,
    ) -> bool:
        """
        Send command to another drone.

        Args:
            target_id: Target drone ID
            command: Command name
            params: Command parameters

        Returns:
            True if sent
        """
        payload = {
            'command': command,
            'params': params or {},
        }
        return self.send_to(
            target_id,
            MessageType.COMMAND,
            payload,
            reliable=True,
        )

    def broadcast_emergency(self, reason: str) -> None:
        """Broadcast emergency to all drones."""
        self.broadcast(
            MessageType.EMERGENCY,
            {'reason': reason, 'drone_id': self.drone_id},
        )

    def broadcast_collision_warning(
        self,
        other_drone_id: str,
        time_to_collision: float,
        position: Dict,
    ) -> None:
        """Broadcast collision warning."""
        self.broadcast(
            MessageType.COLLISION_WARNING,
            {
                'other_drone_id': other_drone_id,
                'time_to_collision': time_to_collision,
                'position': position,
            },
        )
