"""
Flight data recorder (black box) implementation.

Records all flight data for post-flight analysis
and regulatory compliance.
"""

import time
import json
import gzip
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FlightRecord:
    """A single flight data record."""
    timestamp: float
    position: Dict[str, float]  # lat, lon, alt
    velocity: Dict[str, float]  # vx, vy, vz
    attitude: Dict[str, float]  # roll, pitch, yaw
    battery: Dict[str, float]  # voltage, current, percent
    motors: List[Dict[str, float]]  # rpm, current per motor
    sensors: Dict[str, Any]  # sensor readings
    commands: List[str]  # active commands
    warnings: List[str]  # active warnings
    mode: str  # flight mode

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FlightSession:
    """A complete flight session."""
    session_id: str
    drone_id: str
    pilot_id: str
    start_time: float
    end_time: Optional[float] = None
    takeoff_location: Optional[Dict[str, float]] = None
    landing_location: Optional[Dict[str, float]] = None
    max_altitude: float = 0.0
    max_speed: float = 0.0
    max_distance: float = 0.0
    total_distance: float = 0.0
    flight_time: float = 0.0
    record_count: int = 0
    warnings_count: int = 0
    errors_count: int = 0


@dataclass
class RecorderConfig:
    """Configuration for flight recorder."""
    record_rate: float = 10.0  # Hz
    storage_path: str = "flight_logs"
    compress: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    buffer_size: int = 1000  # Records before flush
    include_raw_sensors: bool = False


class FlightRecorder:
    """
    Flight data recorder for compliance and analysis.

    Records all flight parameters at configurable rate
    with automatic file management.
    """

    def __init__(self, config: Optional[RecorderConfig] = None):
        """
        Initialize flight recorder.

        Args:
            config: Recorder configuration
        """
        self._config = config or RecorderConfig()
        self._session: Optional[FlightSession] = None
        self._buffer: List[FlightRecord] = []
        self._file_path: Optional[Path] = None
        self._file_handle: Optional[Any] = None
        self._running = False
        self._lock = threading.Lock()

        # Statistics
        self._total_records = 0
        self._bytes_written = 0

        # Ensure storage directory exists
        Path(self._config.storage_path).mkdir(
            parents=True, exist_ok=True
        )

    def start_session(
        self,
        drone_id: str,
        pilot_id: str = "unknown",
    ) -> str:
        """
        Start a new recording session.

        Args:
            drone_id: Drone identifier
            pilot_id: Pilot identifier

        Returns:
            Session ID
        """
        if self._session is not None:
            self.end_session()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_id = f"{drone_id}_{timestamp}"

        self._session = FlightSession(
            session_id=session_id,
            drone_id=drone_id,
            pilot_id=pilot_id,
            start_time=time.time(),
        )

        # Create log file
        ext = ".json.gz" if self._config.compress else ".json"
        filename = f"flight_{session_id}{ext}"
        self._file_path = Path(self._config.storage_path) / filename

        self._running = True
        logger.info(f"Recording started: {session_id}")

        return session_id

    def end_session(self) -> Optional[FlightSession]:
        """
        End current recording session.

        Returns:
            Completed FlightSession or None
        """
        if self._session is None:
            return None

        self._running = False
        self._session.end_time = time.time()
        self._session.flight_time = (
            self._session.end_time - self._session.start_time
        )
        self._session.record_count = self._total_records

        # Flush remaining buffer
        self._flush_buffer()

        # Close file
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

        # Write session metadata
        self._write_session_metadata()

        session = self._session
        self._session = None
        self._total_records = 0

        logger.info(
            f"Recording ended: {session.session_id}, "
            f"{session.record_count} records"
        )

        return session

    def record(self, record: FlightRecord) -> None:
        """
        Record a flight data point.

        Args:
            record: Flight record to store
        """
        if not self._running or self._session is None:
            return

        with self._lock:
            self._buffer.append(record)
            self._total_records += 1

            # Update session statistics
            self._update_session_stats(record)

            # Flush if buffer full
            if len(self._buffer) >= self._config.buffer_size:
                self._flush_buffer()

    def record_state(
        self,
        position: Dict[str, float],
        velocity: Dict[str, float],
        attitude: Dict[str, float],
        battery: Dict[str, float],
        motors: List[Dict[str, float]],
        mode: str,
        sensors: Optional[Dict] = None,
        commands: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ) -> None:
        """
        Record current state (convenience method).

        Args:
            position: {lat, lon, alt}
            velocity: {vx, vy, vz}
            attitude: {roll, pitch, yaw}
            battery: {voltage, current, percent}
            motors: List of {rpm, current}
            mode: Current flight mode
            sensors: Optional sensor readings
            commands: Optional active commands
            warnings: Optional active warnings
        """
        record = FlightRecord(
            timestamp=time.time(),
            position=position,
            velocity=velocity,
            attitude=attitude,
            battery=battery,
            motors=motors,
            sensors=sensors or {},
            commands=commands or [],
            warnings=warnings or [],
            mode=mode,
        )
        self.record(record)

    def _update_session_stats(self, record: FlightRecord) -> None:
        """Update session statistics from record."""
        if self._session is None:
            return

        # Max altitude
        alt = record.position.get('alt', 0)
        self._session.max_altitude = max(
            self._session.max_altitude, alt
        )

        # Max speed
        vx = record.velocity.get('vx', 0)
        vy = record.velocity.get('vy', 0)
        vz = record.velocity.get('vz', 0)
        speed = (vx**2 + vy**2 + vz**2) ** 0.5
        self._session.max_speed = max(self._session.max_speed, speed)

        # Warnings count
        if record.warnings:
            self._session.warnings_count += len(record.warnings)

        # Takeoff location
        if self._session.takeoff_location is None:
            self._session.takeoff_location = record.position.copy()

        # Update landing location (last known)
        self._session.landing_location = record.position.copy()

    def _flush_buffer(self) -> None:
        """Flush buffer to file."""
        if not self._buffer:
            return

        with self._lock:
            records = self._buffer.copy()
            self._buffer.clear()

        try:
            # Open file if needed
            if self._file_handle is None:
                if self._config.compress:
                    self._file_handle = gzip.open(
                        self._file_path, 'at', encoding='utf-8'
                    )
                else:
                    self._file_handle = open(
                        self._file_path, 'a', encoding='utf-8'
                    )

            # Write records
            for record in records:
                line = json.dumps(record.to_dict()) + '\n'
                self._file_handle.write(line)
                self._bytes_written += len(line)

            self._file_handle.flush()

        except Exception as e:
            logger.error(f"Failed to write records: {e}")
            # Put records back in buffer
            with self._lock:
                self._buffer = records + self._buffer

    def _write_session_metadata(self) -> None:
        """Write session metadata to separate file."""
        if self._session is None:
            return

        meta_path = Path(self._config.storage_path) / (
            f"meta_{self._session.session_id}.json"
        )

        try:
            with open(meta_path, 'w') as f:
                json.dump(asdict(self._session), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write metadata: {e}")

    def get_session(self) -> Optional[FlightSession]:
        """Get current session info."""
        return self._session

    def list_sessions(self) -> List[str]:
        """List all recorded sessions."""
        path = Path(self._config.storage_path)
        sessions = []

        for f in path.glob("meta_*.json"):
            session_id = f.stem.replace("meta_", "")
            sessions.append(session_id)

        return sorted(sessions, reverse=True)

    def load_session(self, session_id: str) -> Optional[FlightSession]:
        """Load session metadata."""
        meta_path = Path(self._config.storage_path) / f"meta_{session_id}.json"

        if not meta_path.exists():
            return None

        try:
            with open(meta_path) as f:
                data = json.load(f)
                return FlightSession(**data)
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def load_records(
        self,
        session_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[FlightRecord]:
        """
        Load records from a session.

        Args:
            session_id: Session to load
            start_time: Optional start filter
            end_time: Optional end filter

        Returns:
            List of FlightRecord
        """
        # Find log file
        path = Path(self._config.storage_path)
        log_file = None

        for ext in ['.json.gz', '.json']:
            candidate = path / f"flight_{session_id}{ext}"
            if candidate.exists():
                log_file = candidate
                break

        if log_file is None:
            return []

        records = []

        try:
            if log_file.suffix == '.gz':
                opener = gzip.open
            else:
                opener = open

            with opener(log_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    record = FlightRecord(**data)

                    # Apply time filter
                    if start_time and record.timestamp < start_time:
                        continue
                    if end_time and record.timestamp > end_time:
                        continue

                    records.append(record)

        except Exception as e:
            logger.error(f"Failed to load records: {e}")

        return records

    @property
    def is_recording(self) -> bool:
        """Check if recording."""
        return self._running

    @property
    def record_count(self) -> int:
        """Get current record count."""
        return self._total_records

    def get_stats(self) -> Dict:
        """Get recorder statistics."""
        return {
            'is_recording': self._running,
            'session_id': (
                self._session.session_id if self._session else None
            ),
            'record_count': self._total_records,
            'bytes_written': self._bytes_written,
            'buffer_size': len(self._buffer),
        }
