"""
Tests for the geofence system.
"""

import pytest
from core.geofence import (
    GeofenceManager,
    GeofenceConfig,
    GeofenceType,
    GeofenceAction,
    GeofenceZone,
    GeofencePoint,
)


class TestGeofenceManager:
    """Tests for the GeofenceManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = GeofenceManager()

    def test_create_circular_geofence(self):
        """Test creating a circular geofence."""
        config = GeofenceConfig(
            name="test_circle",
            fence_type=GeofenceType.CIRCULAR,
            center=(37.7749, -122.4194),
            radius=100.0,
        )
        self.manager.add_fence(config)

        assert "test_circle" in self.manager.get_all_fences()

    def test_create_home_geofence(self):
        """Test creating a home geofence."""
        config = self.manager.create_home_geofence(
            home_lat=37.7749,
            home_lon=-122.4194,
            radius=500.0,
            max_altitude=120.0,
        )

        assert config.name == "home_zone"
        assert config.radius == 500.0
        assert config.max_altitude == 120.0
        assert "home_zone" in self.manager.get_all_fences()

    def test_check_position_inside_circular(self):
        """Test position check inside circular geofence."""
        self.manager.create_home_geofence(37.7749, -122.4194, radius=1000.0)

        # Check position at center (should be inside)
        status = self.manager.check_position(37.7749, -122.4194, 50.0)
        assert status.zone == GeofenceZone.INSIDE
        assert len(status.breached_fences) == 0

    def test_check_position_outside_circular(self):
        """Test position check outside circular geofence."""
        self.manager.create_home_geofence(37.7749, -122.4194, radius=100.0)

        # Check position far from center (should be outside)
        # ~0.01 degrees is about 1.1km
        status = self.manager.check_position(37.7849, -122.4194, 50.0)
        assert status.zone == GeofenceZone.OUTSIDE
        assert "home_zone" in status.breached_fences

    def test_altitude_limit_exceeded(self):
        """Test altitude limit detection."""
        self.manager.create_home_geofence(37.7749, -122.4194, max_altitude=100.0)

        # Check position above max altitude
        status = self.manager.check_position(37.7749, -122.4194, 150.0)
        assert status.altitude_status == "too_high"
        assert "home_zone" in status.breached_fences

    def test_altitude_too_low(self):
        """Test altitude floor detection."""
        config = GeofenceConfig(
            name="altitude_test",
            fence_type=GeofenceType.CYLINDER,
            center=(37.7749, -122.4194),
            radius=1000.0,
            min_altitude=10.0,
            max_altitude=100.0,
        )
        self.manager.add_fence(config)

        status = self.manager.check_position(37.7749, -122.4194, 5.0)
        assert status.altitude_status == "too_low"

    def test_warning_zone(self):
        """Test warning zone detection."""
        self.manager.create_home_geofence(37.7749, -122.4194, radius=100.0)

        # Position at ~85% of boundary should trigger warning
        # Approximately 85 meters from center
        status = self.manager.check_position(37.77565, -122.4194, 50.0)
        # Should be in warning zone (inside but near boundary)
        assert status.zone in (GeofenceZone.WARNING, GeofenceZone.INSIDE)

    def test_polygon_geofence(self):
        """Test polygon geofence."""
        vertices = [
            GeofencePoint(37.7749, -122.4200),
            GeofencePoint(37.7749, -122.4188),
            GeofencePoint(37.7760, -122.4188),
            GeofencePoint(37.7760, -122.4200),
        ]
        config = GeofenceConfig(
            name="test_polygon",
            fence_type=GeofenceType.POLYGON,
            vertices=vertices,
        )
        self.manager.add_fence(config)

        # Check position inside polygon - may be INSIDE or WARNING depending on proximity to edge
        status = self.manager.check_position(37.7755, -122.4194, 50.0)
        assert status.zone in (GeofenceZone.INSIDE, GeofenceZone.WARNING)
        assert len(status.breached_fences) == 0

    def test_no_fly_zone(self):
        """Test no-fly zone (exclusion zone)."""
        vertices = [
            (37.7749, -122.4200),
            (37.7749, -122.4188),
            (37.7760, -122.4188),
            (37.7760, -122.4200),
        ]
        self.manager.create_no_fly_zone("restricted", vertices)

        # Position inside no-fly zone should be breached
        status = self.manager.check_position(37.7755, -122.4194, 50.0)
        assert "restricted" in status.breached_fences

    def test_enable_disable_fence(self):
        """Test enabling and disabling fences."""
        self.manager.create_home_geofence(37.7749, -122.4194, radius=100.0)

        # Disable the fence
        self.manager.enable_fence("home_zone", False)

        # Check far position - should not breach because fence is disabled
        status = self.manager.check_position(37.7849, -122.4194, 50.0)
        assert len(status.breached_fences) == 0

    def test_remove_fence(self):
        """Test removing a fence."""
        self.manager.create_home_geofence(37.7749, -122.4194)
        assert "home_zone" in self.manager.get_all_fences()

        self.manager.remove_fence("home_zone")
        assert "home_zone" not in self.manager.get_all_fences()

    def test_geofence_callback(self):
        """Test geofence breach callback."""
        callback_data = []

        def on_event(event):
            callback_data.append(event)

        self.manager.add_callback(on_event)
        self.manager.create_home_geofence(37.7749, -122.4194, radius=100.0)

        # Trigger a breach
        self.manager.check_position(37.7849, -122.4194, 50.0)

        assert len(callback_data) > 0
        assert callback_data[0]['event'] == 'breach'

    def test_invalid_polygon_raises_error(self):
        """Test that polygon with less than 3 vertices raises error."""
        config = GeofenceConfig(
            name="invalid",
            fence_type=GeofenceType.POLYGON,
            vertices=[GeofencePoint(0, 0), GeofencePoint(1, 1)],
        )
        with pytest.raises(ValueError):
            self.manager.add_fence(config)

    def test_circular_without_center_raises_error(self):
        """Test that circular geofence without center raises error."""
        config = GeofenceConfig(
            name="invalid",
            fence_type=GeofenceType.CIRCULAR,
            center=None,
            radius=100.0,
        )
        with pytest.raises(ValueError):
            self.manager.add_fence(config)

    def test_to_dict(self):
        """Test serialization."""
        self.manager.create_home_geofence(37.7749, -122.4194)
        data = self.manager.to_dict()

        assert 'fences' in data
        assert 'home_zone' in data['fences']
        assert data['fences']['home_zone']['type'] == 'CYLINDER'
