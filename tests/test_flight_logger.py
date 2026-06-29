"""
Unit tests for flight logger module.
Tests flight logging, session management, and data export.
"""

import unittest
import os
import json
import tempfile
import shutil
import time
from core.flight_logger import (
    FlightLogger,
    EventType,
    LogLevel,
    LogEntry,
)


class TestFlightLogger(unittest.TestCase):
    """Test cases for the FlightLogger class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.logger = FlightLogger(log_dir=self.test_dir)

    def tearDown(self):
        """Clean up after each test method."""
        if self.logger.is_recording:
            self.logger.end_session()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_start_session(self):
        """Test starting a flight session."""
        home_position = (52.52, 13.405)
        session_id = self.logger.start_session(home_position)

        self.assertIsNotNone(session_id)
        self.assertTrue(self.logger.is_recording)
        self.assertEqual(self.logger.current_session_id, session_id)

    def test_end_session_saves_file(self):
        """Test that ending a session saves the log file."""
        self.logger.start_session((52.52, 13.405))
        filepath = self.logger.end_session()

        self.assertIsNotNone(filepath)
        self.assertTrue(os.path.exists(filepath))
        self.assertFalse(self.logger.is_recording)

    def test_log_event(self):
        """Test logging a flight event."""
        self.logger.start_session((52.52, 13.405))

        self.logger.log_event(
            EventType.TAKEOFF,
            "Drone taking off",
            LogLevel.INFO,
            data={'altitude': 0},
        )

        entries = self.logger.get_entries(event_type=EventType.TAKEOFF)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].message, "Drone taking off")

    def test_log_sensor_data(self):
        """Test logging sensor data."""
        self.logger.start_session((52.52, 13.405))

        self.logger.log_sensor_data("battery", 85.0)
        self.logger.log_sensor_data("battery", 84.5)
        self.logger.log_sensor_data("battery", 84.0)

        history = self.logger.get_sensor_history("battery")
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0][1], 85.0)

    def test_log_failsafe(self):
        """Test logging a failsafe event."""
        self.logger.start_session((52.52, 13.405))

        self.logger.log_failsafe(
            failsafe_type="LOW_BATTERY",
            reason="Battery at 20%",
            action="return_to_home",
        )

        entries = self.logger.get_entries(event_type=EventType.FAILSAFE_TRIGGERED)
        self.assertEqual(len(entries), 1)
        self.assertIn("LOW_BATTERY", entries[0].message)

    def test_log_waypoint_reached(self):
        """Test logging waypoint reached event."""
        self.logger.start_session((52.52, 13.405))

        self.logger.log_waypoint_reached(
            waypoint_index=1,
            waypoint=(52.525, 13.410),
            remaining=3,
        )

        entries = self.logger.get_entries(event_type=EventType.WAYPOINT_REACHED)
        self.assertEqual(len(entries), 1)

    def test_log_obstacle_detected(self):
        """Test logging obstacle detection."""
        self.logger.start_session((52.52, 13.405))

        self.logger.log_obstacle_detected(
            distance=1.2,
            direction=45.0,
            position=(52.52, 13.405, 50.0),
        )

        entries = self.logger.get_entries(event_type=EventType.OBSTACLE_DETECTED)
        self.assertEqual(len(entries), 1)
        self.assertIn("1.2", entries[0].message)

    def test_log_error(self):
        """Test logging an error event."""
        self.logger.start_session((52.52, 13.405))

        try:
            raise ValueError("Test error")
        except ValueError as e:
            self.logger.log_error("test_error", "A test error occurred", e)

        entries = self.logger.get_entries(event_type=EventType.SYSTEM_ERROR)
        self.assertEqual(len(entries), 1)
        self.assertIn("ValueError", str(entries[0].data))

    def test_filter_entries_by_level(self):
        """Test filtering entries by log level."""
        self.logger.start_session((52.52, 13.405))

        self.logger.log_event(EventType.TAKEOFF, "Taking off", LogLevel.INFO)
        self.logger.log_event(EventType.SYSTEM_ERROR, "Error!", LogLevel.ERROR)
        self.logger.log_event(EventType.HOVER, "Hovering", LogLevel.WARNING)

        error_entries = self.logger.get_entries(level=LogLevel.ERROR)
        self.assertEqual(len(error_entries), 1)

    def test_filter_entries_by_time(self):
        """Test filtering entries by timestamp."""
        self.logger.start_session((52.52, 13.405))

        start_time = time.time()
        self.logger.log_event(EventType.TAKEOFF, "Taking off", LogLevel.INFO)

        time.sleep(0.1)
        mid_time = time.time()

        self.logger.log_event(EventType.HOVER, "Hovering", LogLevel.INFO)

        entries_before_mid = self.logger.get_entries(end_time=mid_time)
        # Should have flight_start and takeoff
        self.assertGreaterEqual(len(entries_before_mid), 2)

    def test_get_statistics(self):
        """Test getting flight statistics."""
        self.logger.start_session((52.52, 13.405))

        self.logger.log_event(EventType.TAKEOFF, "Taking off", LogLevel.INFO)
        self.logger.log_event(EventType.HOVER, "Hovering", LogLevel.INFO)
        self.logger.log_sensor_data("battery", 85.0)

        stats = self.logger.get_statistics()

        self.assertIn('session_id', stats)
        self.assertIn('total_entries', stats)
        self.assertIn('events_by_type', stats)
        self.assertGreater(stats['total_entries'], 0)

    def test_sensor_history_limit(self):
        """Test that sensor history is limited to max entries."""
        self.logger._max_sensor_history = 5
        self.logger.start_session((52.52, 13.405))

        # Log more than max entries
        for i in range(10):
            self.logger.log_sensor_data("battery", 100 - i)

        history = self.logger.get_sensor_history("battery")
        self.assertEqual(len(history), 5)
        # Should have the last 5 entries
        self.assertEqual(history[0][1], 95)

    def test_get_sensor_history_with_limit(self):
        """Test getting limited sensor history."""
        self.logger.start_session((52.52, 13.405))

        for i in range(10):
            self.logger.log_sensor_data("battery", 100 - i)

        history = self.logger.get_sensor_history("battery", limit=3)
        self.assertEqual(len(history), 3)

    def test_json_file_format(self):
        """Test that saved JSON file has correct format."""
        self.logger.start_session((52.52, 13.405))
        self.logger.log_event(EventType.TAKEOFF, "Taking off", LogLevel.INFO)
        filepath = self.logger.end_session()

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.assertIn('session_id', data)
        self.assertIn('start_time', data)
        self.assertIn('entries', data)
        self.assertIn('statistics', data)

    def test_export_csv(self):
        """Test CSV export functionality."""
        self.logger.start_session((52.52, 13.405))

        self.logger.log_sensor_data("battery", 85.0)
        self.logger.log_sensor_data("wind_speed", 5.0)
        self.logger.log_sensor_data("battery", 84.0)

        csv_path = self.logger.export_csv()

        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, 'r') as f:
            content = f.read()
            self.assertIn("timestamp", content)
            self.assertIn("battery", content)
            self.assertIn("wind_speed", content)

    def test_log_entry_to_dict(self):
        """Test LogEntry serialization."""
        entry = LogEntry(
            timestamp=time.time(),
            event_type="takeoff",
            level="INFO",
            message="Test message",
            data={'key': 'value'},
            position=(52.52, 13.405, 100.0),
        )

        data = entry.to_dict()

        self.assertIn('timestamp', data)
        self.assertIn('datetime', data)
        self.assertIn('event_type', data)
        self.assertIn('data', data)
        self.assertIn('position', data)
        self.assertEqual(data['position']['latitude'], 52.52)

    def test_no_session_returns_empty(self):
        """Test that operations without a session return empty/None."""
        stats = self.logger.get_statistics()
        self.assertEqual(stats, {})

        csv_path = self.logger.export_csv()
        self.assertEqual(csv_path, "")


if __name__ == "__main__":
    unittest.main()
