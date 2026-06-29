"""
Unit tests for failsafe system.
Tests failsafe conditions, triggers, and actions.
"""

import unittest
import time
from core.failsafe import (
    FailsafeManager,
    FailsafeType,
    FailsafeAction,
    FailsafeState,
)


class TestFailsafeManager(unittest.TestCase):
    """Test cases for the FailsafeManager class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.failsafe = FailsafeManager()

    def test_initial_state(self):
        """Test failsafe manager initial state."""
        self.assertEqual(self.failsafe.state, FailsafeState.NORMAL)
        self.assertFalse(self.failsafe.is_failsafe_active())
        self.assertEqual(len(self.failsafe.get_active_failsafes()), 0)

    def test_low_battery_triggers_return_home(self):
        """Test that low battery triggers return to home."""
        action = self.failsafe.check_battery(20.0)  # Below 25% threshold

        self.assertEqual(action, FailsafeAction.RETURN_TO_HOME)
        self.assertTrue(self.failsafe.is_failsafe_active(FailsafeType.LOW_BATTERY))

    def test_critical_battery_triggers_land_immediately(self):
        """Test that critical battery triggers immediate landing."""
        action = self.failsafe.check_battery(5.0)  # Below 10% threshold

        self.assertEqual(action, FailsafeAction.LAND_IMMEDIATELY)
        self.assertTrue(self.failsafe.is_failsafe_active(FailsafeType.CRITICAL_BATTERY))

    def test_normal_battery_no_action(self):
        """Test that normal battery level triggers no action."""
        action = self.failsafe.check_battery(80.0)

        self.assertIsNone(action)
        self.assertFalse(self.failsafe.is_failsafe_active(FailsafeType.LOW_BATTERY))

    def test_gps_loss_triggers_hover(self):
        """Test that GPS loss triggers hover action after timeout."""
        # Set a very short timeout for testing
        self.failsafe.configure_failsafe(
            FailsafeType.GPS_LOSS,
            timeout=0.1,
        )

        # First check - starts timing
        action = self.failsafe.check_gps(
            has_fix=False,
            satellites=2,
            accuracy=100.0,
        )
        self.assertIsNone(action)  # Not yet triggered

        # Wait for timeout
        time.sleep(0.15)

        # Second check - should trigger
        action = self.failsafe.check_gps(
            has_fix=False,
            satellites=2,
            accuracy=100.0,
        )
        self.assertEqual(action, FailsafeAction.HOVER)

    def test_gps_recovery_clears_failsafe(self):
        """Test that GPS recovery clears the failsafe."""
        # Trigger GPS loss
        self.failsafe.configure_failsafe(FailsafeType.GPS_LOSS, timeout=0.0)
        self.failsafe.check_gps(has_fix=False, satellites=2, accuracy=100.0)

        # Recover GPS
        self.failsafe.check_gps(has_fix=True, satellites=8, accuracy=5.0)

        self.assertFalse(self.failsafe.is_failsafe_active(FailsafeType.GPS_LOSS))

    def test_obstacle_collision_triggers_hover(self):
        """Test that imminent obstacle collision triggers hover."""
        action = self.failsafe.check_obstacle(0.5)  # 0.5m < 1.5m threshold

        self.assertEqual(action, FailsafeAction.HOVER)
        self.assertTrue(self.failsafe.is_failsafe_active(FailsafeType.OBSTACLE_COLLISION_IMMINENT))

    def test_obstacle_cleared_clears_failsafe(self):
        """Test that obstacle clearing clears the failsafe."""
        # Trigger obstacle failsafe
        self.failsafe.check_obstacle(0.5)
        self.assertTrue(self.failsafe.is_failsafe_active(FailsafeType.OBSTACLE_COLLISION_IMMINENT))

        # Clear obstacle
        self.failsafe.check_obstacle(3.0)  # 3m > 1.5m threshold
        self.assertFalse(self.failsafe.is_failsafe_active(FailsafeType.OBSTACLE_COLLISION_IMMINENT))

    def test_signal_loss_triggers_return_home(self):
        """Test that signal loss triggers return to home."""
        self.failsafe.configure_failsafe(FailsafeType.SIGNAL_LOSS, timeout=0.1)

        # Simulate signal loss
        old_time = time.time() - 10  # 10 seconds ago
        self.failsafe.check_signal(0.0, old_time)

        time.sleep(0.15)

        action = self.failsafe.check_signal(0.0, old_time)
        self.assertEqual(action, FailsafeAction.RETURN_TO_HOME)

    def test_highest_priority_action(self):
        """Test that highest priority failsafe action is returned."""
        # Trigger multiple failsafes
        self.failsafe.check_battery(5.0)  # Critical battery - priority 1
        self.failsafe.check_battery(20.0)  # Low battery - priority 3

        # Critical battery should have highest priority
        action = self.failsafe.get_highest_priority_action()
        self.assertEqual(action, FailsafeAction.LAND_IMMEDIATELY)

    def test_configure_failsafe(self):
        """Test failsafe configuration."""
        self.failsafe.configure_failsafe(
            FailsafeType.LOW_BATTERY,
            threshold=30.0,
            action=FailsafeAction.HOVER,
        )

        # Now 25% should not trigger (threshold is 30%)
        action = self.failsafe.check_battery(28.0)
        self.assertEqual(action, FailsafeAction.HOVER)

    def test_disable_failsafe(self):
        """Test disabling a failsafe."""
        self.failsafe.configure_failsafe(
            FailsafeType.LOW_BATTERY,
            enabled=False,
        )

        # Use 20% which is below LOW_BATTERY (25%) but above CRITICAL_BATTERY (10%)
        action = self.failsafe.check_battery(20.0)
        self.assertIsNone(action)

    def test_callback_registration(self):
        """Test failsafe callback registration and execution."""
        callback_triggered = []

        def on_failsafe(event):
            callback_triggered.append(event)

        self.failsafe.register_callback(FailsafeType.LOW_BATTERY, on_failsafe)
        self.failsafe.check_battery(20.0)

        self.assertEqual(len(callback_triggered), 1)
        self.assertEqual(callback_triggered[0].failsafe_type, FailsafeType.LOW_BATTERY)

    def test_clear_all_failsafes(self):
        """Test clearing all active failsafes."""
        # Trigger multiple failsafes
        self.failsafe.check_battery(20.0)
        self.failsafe.check_obstacle(0.5)

        self.assertTrue(self.failsafe.is_failsafe_active())

        # Clear all
        self.failsafe.clear_all()

        self.assertFalse(self.failsafe.is_failsafe_active())
        self.assertEqual(self.failsafe.state, FailsafeState.NORMAL)

    def test_event_history(self):
        """Test failsafe event history tracking."""
        self.failsafe.check_battery(20.0)
        self.failsafe.check_battery(80.0)  # Clears the failsafe

        history = self.failsafe.event_history
        self.assertEqual(len(history), 1)
        self.assertIsNotNone(history[0].cleared_at)

    def test_get_status(self):
        """Test getting failsafe system status."""
        self.failsafe.check_battery(20.0)
        status = self.failsafe.get_status()

        self.assertIn('state', status)
        self.assertIn('active_failsafes', status)
        self.assertEqual(len(status['active_failsafes']), 1)

    def test_geofence_breach(self):
        """Test geofence breach detection."""
        center = (52.52, 13.405)
        radius = 100.0  # 100 meters

        # Position outside geofence
        action = self.failsafe.check_geofence(
            position=(52.53, 13.415),  # More than 100m away
            geofence_center=center,
            geofence_radius=radius,
        )

        self.assertEqual(action, FailsafeAction.RETURN_TO_HOME)

    def test_geofence_within_bounds(self):
        """Test geofence when within bounds."""
        center = (52.52, 13.405)
        radius = 10000.0  # 10km

        action = self.failsafe.check_geofence(
            position=(52.521, 13.406),  # Very close
            geofence_center=center,
            geofence_radius=radius,
        )

        self.assertIsNone(action)


if __name__ == "__main__":
    unittest.main()
