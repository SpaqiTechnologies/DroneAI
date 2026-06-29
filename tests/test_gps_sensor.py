"""
Unit tests for GPS sensor module.
Tests GPS position tracking, accuracy metrics, and signal handling.
"""

import unittest
import time
from sensors.gps_sensor import GPSSensor, GPSReading


class TestGPSSensor(unittest.TestCase):
    """Test cases for the GPSSensor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.gps = GPSSensor()
        self.gps.start()

    def tearDown(self):
        """Clean up after each test method."""
        self.gps.stop()

    def test_initial_state(self):
        """Test GPS sensor initial state."""
        self.assertEqual(self.gps.type, "gps")
        self.assertTrue(self.gps.has_3d_fix())
        self.assertEqual(self.gps.get_satellites(), 8)

    def test_set_position(self):
        """Test setting GPS position."""
        self.gps.set_position(52.52, 13.405, 100.0)

        position = self.gps.get_position()
        self.assertEqual(position[0], 52.52)
        self.assertEqual(position[1], 13.405)
        self.assertEqual(self.gps.get_altitude(), 100.0)

    def test_update_returns_reading(self):
        """Test that update returns a valid GPSReading."""
        self.gps.set_position(52.52, 13.405, 100.0)
        reading = self.gps.update()

        self.assertIsInstance(reading, GPSReading)
        self.assertTrue(reading.is_valid())

    def test_accuracy_metrics(self):
        """Test GPS accuracy metrics."""
        horizontal, vertical = self.gps.get_accuracy()

        self.assertGreater(horizontal, 0)
        self.assertGreater(vertical, 0)
        self.assertGreater(vertical, horizontal)  # Vertical typically less accurate

    def test_signal_loss_simulation(self):
        """Test GPS signal loss simulation."""
        self.gps.simulate_signal_loss()

        self.assertFalse(self.gps.has_fix())
        self.assertEqual(self.gps.get_satellites(), 0)
        self.assertEqual(self.gps.get_signal_quality(), 0.0)

    def test_signal_recovery(self):
        """Test GPS signal recovery after loss."""
        self.gps.simulate_signal_loss()
        self.assertFalse(self.gps.has_fix())

        self.gps.simulate_signal_recovery()
        self.assertTrue(self.gps.has_fix())
        self.assertTrue(self.gps.has_3d_fix())

    def test_satellite_count_affects_fix_type(self):
        """Test that satellite count affects fix type."""
        # No fix with less than 3 satellites
        self.gps.set_satellites(2)
        self.assertEqual(self.gps._fix_type, GPSSensor.NO_FIX)

        # 2D fix with 3 satellites
        self.gps.set_satellites(3)
        self.assertEqual(self.gps._fix_type, GPSSensor.FIX_2D)

        # 3D fix with 4+ satellites
        self.gps.set_satellites(6)
        self.assertEqual(self.gps._fix_type, GPSSensor.FIX_3D)

    def test_speed_and_heading(self):
        """Test speed and heading values."""
        self.gps.set_speed(15.5)
        self.gps.set_heading(270.0)

        self.assertEqual(self.gps.get_speed(), 15.5)
        self.assertEqual(self.gps.get_heading(), 270.0)

    def test_heading_wraps_around(self):
        """Test that heading wraps around 360 degrees."""
        self.gps.set_heading(450.0)
        self.assertEqual(self.gps.get_heading(), 90.0)

    def test_to_dict(self):
        """Test GPS data serialization."""
        self.gps.set_position(52.52, 13.405, 100.0)
        data = self.gps.to_dict()

        self.assertEqual(data['type'], 'gps')
        self.assertEqual(data['latitude'], 52.52)
        self.assertEqual(data['longitude'], 13.405)
        self.assertEqual(data['altitude'], 100.0)
        self.assertIn('satellites_used', data)
        self.assertIn('fix_type', data)

    def test_invalid_reading_when_disconnected(self):
        """Test that disconnected GPS returns invalid reading."""
        self.gps.stop()
        reading = self.gps.update()

        self.assertFalse(reading.is_valid())

    def test_gps_reading_validity(self):
        """Test GPSReading validity checks."""
        # Valid reading
        valid_reading = GPSReading(
            latitude=52.52,
            longitude=13.405,
            altitude=100.0,
            accuracy_horizontal=5.0,
            accuracy_vertical=10.0,
            satellites_used=8,
            fix_type=2,
            timestamp=time.time(),
            speed=0.0,
            heading=0.0,
        )
        self.assertTrue(valid_reading.is_valid())

        # Invalid - no fix
        invalid_reading = GPSReading(
            latitude=52.52,
            longitude=13.405,
            altitude=100.0,
            accuracy_horizontal=5.0,
            accuracy_vertical=10.0,
            satellites_used=8,
            fix_type=0,  # No fix
            timestamp=time.time(),
            speed=0.0,
            heading=0.0,
        )
        self.assertFalse(invalid_reading.is_valid())

        # Invalid - poor accuracy
        poor_accuracy_reading = GPSReading(
            latitude=52.52,
            longitude=13.405,
            altitude=100.0,
            accuracy_horizontal=100.0,  # > 50m
            accuracy_vertical=150.0,
            satellites_used=8,
            fix_type=2,
            timestamp=time.time(),
            speed=0.0,
            heading=0.0,
        )
        self.assertFalse(poor_accuracy_reading.is_valid())


if __name__ == "__main__":
    unittest.main()
