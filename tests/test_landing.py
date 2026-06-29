"""
Tests for the landing system.
"""

import pytest
import time
from core.landing import (
    LandingManager,
    LandingMode,
    LandingState,
    LandingConfig,
    LandingZone,
    LandingAbortReason,
)


class TestLandingManager:
    """Tests for the LandingManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = LandingManager()
        self.current_position = (37.7749, -122.4194, 50.0)  # lat, lon, alt_agl

    def test_initial_state(self):
        """Test initial state is idle."""
        assert self.manager.state == LandingState.IDLE
        assert not self.manager.is_landing
        assert not self.manager.is_complete

    def test_start_normal_landing(self):
        """Test starting a normal landing."""
        success, message = self.manager.start_landing(
            LandingMode.NORMAL,
            self.current_position,
        )

        assert success
        assert self.manager.state == LandingState.DESCENDING
        assert self.manager.mode == LandingMode.NORMAL
        assert self.manager.is_landing

    def test_start_precision_landing(self):
        """Test starting a precision landing."""
        success, message = self.manager.start_landing(
            LandingMode.PRECISION,
            self.current_position,
        )

        assert success
        assert self.manager.mode == LandingMode.PRECISION

    def test_start_emergency_landing(self):
        """Test starting an emergency landing."""
        success, message = self.manager.start_landing(
            LandingMode.EMERGENCY,
            self.current_position,
        )

        assert success
        assert self.manager.mode == LandingMode.EMERGENCY

    def test_cannot_start_landing_while_landing(self):
        """Test that starting a landing while already landing fails."""
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)

        success, message = self.manager.start_landing(
            LandingMode.NORMAL,
            self.current_position,
        )

        assert not success
        assert "already" in message.lower()

    def test_update_returns_descent_rate(self):
        """Test that update returns appropriate descent rate."""
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)

        state, descent_rate, target = self.manager.update(
            self.current_position,
            wind_speed=2.0,
        )

        assert descent_rate > 0
        assert target is not None

    def test_emergency_landing_faster_descent(self):
        """Test that emergency landing has faster descent rate."""
        config = LandingConfig()

        self.manager.start_landing(LandingMode.EMERGENCY, self.current_position)
        _, emergency_rate, _ = self.manager.update(self.current_position)

        self.manager.reset()
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)
        _, normal_rate, _ = self.manager.update(self.current_position)

        assert emergency_rate > normal_rate

    def test_abort_landing(self):
        """Test aborting a landing."""
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)

        success, message = self.manager.abort()

        assert success
        assert self.manager.state == LandingState.ABORTED
        assert not self.manager.is_landing

    def test_abort_when_not_landing(self):
        """Test that aborting when not landing fails."""
        success, message = self.manager.abort()

        assert not success

    def test_high_wind_aborts_landing(self):
        """Test that high wind aborts normal landing."""
        config = LandingConfig(max_wind_for_landing=5.0)
        self.manager = LandingManager(config)
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)

        state, _, _ = self.manager.update(
            self.current_position,
            wind_speed=10.0,  # Above threshold
        )

        assert state == LandingState.ABORTED

    def test_emergency_landing_ignores_high_wind(self):
        """Test that emergency landing doesn't abort for high wind."""
        config = LandingConfig(max_wind_for_landing=5.0)
        self.manager = LandingManager(config)
        self.manager.start_landing(LandingMode.EMERGENCY, self.current_position)

        state, _, _ = self.manager.update(
            self.current_position,
            wind_speed=10.0,
        )

        assert state != LandingState.ABORTED

    def test_obstacle_aborts_landing(self):
        """Test that obstacle below aborts landing during final approach."""
        config = LandingConfig(final_approach_altitude=10.0)
        self.manager = LandingManager(config)

        # Start landing at low altitude (in final approach)
        low_position = (37.7749, -122.4194, 5.0)
        self.manager.start_landing(LandingMode.NORMAL, low_position)

        # Force state to final approach
        self.manager._state = LandingState.FINAL_APPROACH

        state, _, _ = self.manager.update(
            low_position,
            obstacle_below=True,
        )

        assert state == LandingState.ABORTED

    def test_landing_complete_at_touchdown(self):
        """Test that landing completes at touchdown altitude."""
        config = LandingConfig(touchdown_altitude=0.3)
        self.manager = LandingManager(config)
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)

        # Update with very low altitude
        touchdown_position = (37.7749, -122.4194, 0.2)
        state, _, _ = self.manager.update(touchdown_position)

        assert state == LandingState.LANDED
        assert self.manager.is_complete

    def test_reset_manager(self):
        """Test resetting the landing manager."""
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)
        self.manager.reset()

        assert self.manager.state == LandingState.IDLE
        assert self.manager.mode == LandingMode.NORMAL
        assert not self.manager.is_landing

    def test_callback_on_landing_start(self):
        """Test callback is called on landing start."""
        callback_data = []

        def on_event(event):
            callback_data.append(event)

        self.manager.add_callback(on_event)
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)

        assert len(callback_data) > 0
        assert callback_data[0]['event'] == 'landing_started'

    def test_callback_on_landing_abort(self):
        """Test callback is called on landing abort."""
        callback_data = []

        def on_event(event):
            callback_data.append(event)

        self.manager.add_callback(on_event)
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)
        self.manager.abort()

        events = [e['event'] for e in callback_data]
        assert 'landing_aborted' in events

    def test_get_status(self):
        """Test getting landing status."""
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)

        status = self.manager.get_status()

        assert status.mode == LandingMode.NORMAL
        assert status.state == LandingState.DESCENDING
        assert status.start_time is not None

    def test_to_dict(self):
        """Test serialization."""
        self.manager.start_landing(LandingMode.NORMAL, self.current_position)

        data = self.manager.to_dict()

        assert data['mode'] == 'normal'
        assert data['state'] == 'descending'
        assert data['start_time'] is not None


class TestLandingZone:
    """Tests for the LandingZone class."""

    def test_landing_zone_creation(self):
        """Test creating a landing zone."""
        zone = LandingZone(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=0.0,
            size=5.0,
            surface_type="concrete",
            is_safe=True,
            slope_degrees=2.0,
            confidence=0.95,
        )

        assert zone.latitude == 37.7749
        assert zone.surface_type == "concrete"
        assert zone.is_safe
        assert zone.confidence == 0.95


class TestSafeZoneSelection:
    """Tests for safe zone selection."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = LandingManager()
        self.current_position = (37.7749, -122.4194, 50.0)

    def test_add_safe_zone(self):
        """Test adding a safe zone."""
        zone = LandingZone(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=0.0,
            size=5.0,
            surface_type="flat",
        )
        self.manager.add_safe_zone(zone)

        # Should be able to select it
        selected = self.manager.select_best_safe_zone(self.current_position)
        assert selected is not None

    def test_select_best_zone_prefers_larger(self):
        """Test that larger zones are preferred."""
        small_zone = LandingZone(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=0.0,
            size=2.0,
            surface_type="flat",
        )
        large_zone = LandingZone(
            latitude=37.7750,
            longitude=-122.4194,
            altitude=0.0,
            size=10.0,
            surface_type="flat",
        )

        self.manager.add_safe_zone(small_zone)
        self.manager.add_safe_zone(large_zone)

        selected = self.manager.select_best_safe_zone(self.current_position)
        assert selected.size == 10.0

    def test_select_best_zone_avoids_unsafe(self):
        """Test that unsafe zones are not selected."""
        unsafe_zone = LandingZone(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=0.0,
            size=10.0,
            surface_type="water",
            is_safe=False,
        )
        safe_zone = LandingZone(
            latitude=37.7750,
            longitude=-122.4194,
            altitude=0.0,
            size=3.0,
            surface_type="flat",
            is_safe=True,
        )

        self.manager.add_safe_zone(unsafe_zone)
        self.manager.add_safe_zone(safe_zone)

        selected = self.manager.select_best_safe_zone(self.current_position)
        assert selected.is_safe

    def test_select_best_zone_prefers_concrete(self):
        """Test that concrete surfaces are preferred."""
        grass_zone = LandingZone(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=0.0,
            size=5.0,
            surface_type="grass",
        )
        concrete_zone = LandingZone(
            latitude=37.7750,
            longitude=-122.4194,
            altitude=0.0,
            size=5.0,
            surface_type="concrete",
        )

        self.manager.add_safe_zone(grass_zone)
        self.manager.add_safe_zone(concrete_zone)

        selected = self.manager.select_best_safe_zone(self.current_position)
        assert selected.surface_type == "concrete"

    def test_no_safe_zones_returns_none(self):
        """Test that no safe zones returns None."""
        selected = self.manager.select_best_safe_zone(self.current_position)
        assert selected is None
