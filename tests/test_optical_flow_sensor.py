"""
Tests for Optical Flow sensor module.
"""

import pytest
import math
import time
from sensors.optical_flow_sensor import (
    OpticalFlowSensor,
    OpticalFlowReading,
    OpticalFlowQuality,
    OpticalFlowState,
)


class TestOpticalFlowReading:
    """Tests for OpticalFlowReading class."""

    def test_reading_creation(self):
        """Test OpticalFlowReading creation."""
        reading = OpticalFlowReading(
            flow_x=0.1,
            flow_y=0.2,
            integrated_x=0.001,
            integrated_y=0.002,
            velocity_x=0.5,
            velocity_y=1.0,
            quality=OpticalFlowQuality.HIGH,
            quality_percent=75,
            height_agl=2.0,
            timestamp=time.time(),
            integration_time=0.01,
            gyro_compensated=True,
            surface_valid=True,
        )
        assert reading.flow_x == 0.1
        assert reading.velocity_x == 0.5
        assert reading.quality == OpticalFlowQuality.HIGH

    def test_reading_is_valid(self):
        """Test validity check."""
        valid_reading = OpticalFlowReading(
            flow_x=0.1, flow_y=0.2,
            integrated_x=0.001, integrated_y=0.002,
            velocity_x=0.5, velocity_y=1.0,
            quality=OpticalFlowQuality.MEDIUM,
            quality_percent=50,
            height_agl=2.0,
            timestamp=time.time(),
            integration_time=0.01,
            gyro_compensated=False,
            surface_valid=True,
        )
        assert valid_reading.is_valid()

        invalid_reading = OpticalFlowReading(
            flow_x=0.1, flow_y=0.2,
            integrated_x=0.001, integrated_y=0.002,
            velocity_x=0.5, velocity_y=1.0,
            quality=OpticalFlowQuality.INVALID,
            quality_percent=10,
            height_agl=2.0,
            timestamp=time.time(),
            integration_time=0.01,
            gyro_compensated=False,
            surface_valid=False,
        )
        assert not invalid_reading.is_valid()

    def test_reading_get_velocity(self):
        """Test velocity getter."""
        reading = OpticalFlowReading(
            flow_x=0.1, flow_y=0.2,
            integrated_x=0.001, integrated_y=0.002,
            velocity_x=3.0, velocity_y=4.0,
            quality=OpticalFlowQuality.HIGH,
            quality_percent=75,
            height_agl=2.0,
            timestamp=time.time(),
            integration_time=0.01,
            gyro_compensated=False,
            surface_valid=True,
        )
        vx, vy = reading.get_velocity()
        assert vx == 3.0
        assert vy == 4.0

    def test_reading_get_speed(self):
        """Test speed calculation."""
        reading = OpticalFlowReading(
            flow_x=0.1, flow_y=0.2,
            integrated_x=0.001, integrated_y=0.002,
            velocity_x=3.0, velocity_y=4.0,
            quality=OpticalFlowQuality.HIGH,
            quality_percent=75,
            height_agl=2.0,
            timestamp=time.time(),
            integration_time=0.01,
            gyro_compensated=False,
            surface_valid=True,
        )
        assert reading.get_speed() == 5.0  # 3-4-5 triangle

    def test_reading_to_dict(self):
        """Test serialization."""
        reading = OpticalFlowReading(
            flow_x=0.1, flow_y=0.2,
            integrated_x=0.001, integrated_y=0.002,
            velocity_x=0.5, velocity_y=1.0,
            quality=OpticalFlowQuality.HIGH,
            quality_percent=75,
            height_agl=2.0,
            timestamp=time.time(),
            integration_time=0.01,
            gyro_compensated=True,
            surface_valid=True,
        )
        d = reading.to_dict()
        assert 'flow_x' in d
        assert 'velocity_x' in d
        assert 'quality' in d
        assert d['quality'] == 'HIGH'


class TestOpticalFlowSensor:
    """Tests for OpticalFlowSensor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sensor = OpticalFlowSensor()

    def test_sensor_creation(self):
        """Test sensor creation."""
        assert self.sensor.type == "optical_flow"
        assert self.sensor._state == OpticalFlowState.OFF

    def test_sensor_start_stop(self):
        """Test starting and stopping sensor."""
        self.sensor.start()
        assert self.sensor._is_connected
        assert self.sensor._state == OpticalFlowState.READY

        self.sensor.stop()
        assert not self.sensor._is_connected
        assert self.sensor._state == OpticalFlowState.OFF

    def test_sensor_update_when_connected(self):
        """Test update returns reading when connected."""
        self.sensor.start()
        self.sensor.set_height(2.0)
        reading = self.sensor.update()

        assert reading is not None
        assert isinstance(reading, OpticalFlowReading)

    def test_sensor_update_when_disconnected(self):
        """Test update returns invalid reading when disconnected."""
        reading = self.sensor.update()
        assert reading.quality == OpticalFlowQuality.INVALID
        assert not reading.is_valid()

    def test_velocity_from_flow_and_height(self):
        """Test velocity calculation: v = flow * height."""
        self.sensor.start()
        self.sensor.set_height(2.0)
        self.sensor.set_true_velocity(2.0, 0.0)  # 2 m/s forward

        # Expected flow = velocity / height = 2.0 / 2.0 = 1.0 rad/s
        reading = self.sensor.update()

        # Velocity should be approximately 2.0 (with noise)
        assert abs(reading.velocity_x - 2.0) < 0.5  # Allow for noise

    def test_flow_increases_with_velocity(self):
        """Test that flow rate increases with velocity."""
        self.sensor.start()
        self.sensor.set_height(1.0)

        # Low velocity
        self.sensor.set_true_velocity(1.0, 0.0)
        reading_slow = self.sensor.update()

        time.sleep(0.02)

        # High velocity
        self.sensor.set_true_velocity(5.0, 0.0)
        reading_fast = self.sensor.update()

        assert abs(reading_fast.flow_x) > abs(reading_slow.flow_x)

    def test_flow_decreases_with_height(self):
        """Test that flow rate decreases with height (same velocity)."""
        self.sensor.start()
        self.sensor.set_true_velocity(2.0, 0.0)

        # Low height
        self.sensor.set_height(1.0)
        reading_low = self.sensor.update()

        time.sleep(0.02)

        # High height
        self.sensor.set_height(5.0)
        reading_high = self.sensor.update()

        # Flow should be smaller at higher altitude
        assert abs(reading_high.flow_x) < abs(reading_low.flow_x)

    def test_quality_at_optimal_height(self):
        """Test quality is good at optimal height range."""
        self.sensor.start()
        self.sensor.set_height(2.0)  # Good height
        self.sensor.set_surface_texture(1.0)  # Good surface
        self.sensor.set_true_velocity(0.5, 0.0)  # Low velocity

        # Take multiple readings to average out noise
        qualities = []
        for _ in range(5):
            reading = self.sensor.update()
            qualities.append(reading.quality_percent)
            time.sleep(0.01)

        avg_quality = sum(qualities) / len(qualities)
        assert avg_quality > 50  # Should have decent quality

    def test_quality_degrades_at_high_altitude(self):
        """Test quality degrades at high altitude."""
        self.sensor.start()
        self.sensor.set_surface_texture(1.0)
        self.sensor.set_true_velocity(0.0, 0.0)

        # Optimal height
        self.sensor.set_height(2.0)
        reading_low = self.sensor.update()

        time.sleep(0.02)

        # Very high
        self.sensor.set_height(15.0)
        reading_high = self.sensor.update()

        assert reading_high.quality_percent <= reading_low.quality_percent

    def test_surface_texture_affects_quality(self):
        """Test that poor surface texture degrades quality."""
        self.sensor.start()
        self.sensor.set_height(2.0)

        # Good texture
        self.sensor.set_surface_texture(1.0)
        reading_good = self.sensor.update()

        time.sleep(0.02)

        # Poor texture
        self.sensor.set_surface_texture(0.3)
        reading_poor = self.sensor.update()

        assert reading_poor.quality_percent < reading_good.quality_percent

    def test_surface_loss_invalidates_reading(self):
        """Test that losing surface texture invalidates reading."""
        self.sensor.start()
        self.sensor.set_height(2.0)

        self.sensor.simulate_surface_loss()
        reading = self.sensor.update()

        assert not reading.surface_valid
        assert not reading.is_valid()

    def test_surface_recovery(self):
        """Test surface recovery after loss."""
        self.sensor.start()
        self.sensor.set_height(2.0)

        self.sensor.simulate_surface_loss()
        reading_lost = self.sensor.update()
        assert not reading_lost.surface_valid

        time.sleep(0.02)

        self.sensor.simulate_surface_recovery()
        reading_recovered = self.sensor.update()
        assert reading_recovered.surface_valid

    def test_minimum_height_requirement(self):
        """Test that readings are invalid below minimum height."""
        self.sensor.start()
        self.sensor.set_height(0.05)  # Below MIN_HEIGHT (0.1m)

        reading = self.sensor.update()
        assert not reading.surface_valid

    def test_gyro_compensation(self):
        """Test gyro compensation can be enabled."""
        self.sensor.start()
        self.sensor.set_height(2.0)
        self.sensor.set_gyro_rates(5.0, 0.0)  # 5 deg/s roll rate

        reading = self.sensor.update()
        assert reading.gyro_compensated

    def test_max_flow_rate_clamping(self):
        """Test that flow rate is clamped to sensor limits."""
        self.sensor.start()
        self.sensor.set_height(0.5)
        self.sensor.set_true_velocity(50.0, 0.0)  # Very high velocity

        reading = self.sensor.update()

        # Flow should be clamped to MAX_FLOW_RATE
        assert abs(reading.flow_x) <= self.sensor.MAX_FLOW_RATE + 0.1

    def test_getters(self):
        """Test getter methods."""
        self.sensor.start()
        self.sensor.set_height(2.0)
        self.sensor.set_true_velocity(1.0, 2.0)
        self.sensor.update()

        flow = self.sensor.get_flow()
        assert len(flow) == 2

        velocity = self.sensor.get_velocity()
        assert len(velocity) == 2

        speed = self.sensor.get_speed()
        assert speed >= 0

        quality = self.sensor.get_quality()
        assert isinstance(quality, OpticalFlowQuality)

        quality_pct = self.sensor.get_quality_percent()
        assert 0 <= quality_pct <= 100

    def test_to_dict(self):
        """Test serialization to dictionary."""
        self.sensor.start()
        self.sensor.set_height(2.0)
        self.sensor.update()

        d = self.sensor.to_dict()

        assert 'type' in d
        assert d['type'] == 'optical_flow'
        assert 'state' in d
        assert 'flow' in d
        assert 'velocity' in d
        assert 'quality' in d
        assert 'height_agl' in d

    def test_measure_alias(self):
        """Test measure() is alias for update()."""
        self.sensor.start()
        self.sensor.set_height(2.0)

        reading = self.sensor.measure()
        assert isinstance(reading, OpticalFlowReading)

    def test_custom_update_rate(self):
        """Test custom update rate."""
        sensor_fast = OpticalFlowSensor(update_rate=200)
        sensor_slow = OpticalFlowSensor(update_rate=50)

        assert sensor_fast._update_rate == 200
        assert sensor_slow._update_rate == 50
        assert sensor_fast._update_interval < sensor_slow._update_interval

    def test_state_transitions(self):
        """Test sensor state transitions."""
        assert self.sensor.get_state() == OpticalFlowState.OFF

        self.sensor.start()
        assert self.sensor.get_state() == OpticalFlowState.READY

        self.sensor.set_height(2.0)
        self.sensor.set_surface_texture(1.0)
        reading = self.sensor.update()

        if reading.is_valid():
            assert self.sensor.get_state() == OpticalFlowState.ACTIVE
        else:
            assert self.sensor.get_state() == OpticalFlowState.DEGRADED

        self.sensor.stop()
        assert self.sensor.get_state() == OpticalFlowState.OFF


class TestOpticalFlowSensorIntegration:
    """Integration tests for optical flow sensor."""

    def test_continuous_velocity_tracking(self):
        """Test tracking velocity over time."""
        sensor = OpticalFlowSensor()
        sensor.start()
        sensor.set_height(1.5)
        sensor.set_surface_texture(1.0)

        velocities = []
        true_velocity = 2.0

        for _ in range(20):
            sensor.set_true_velocity(true_velocity, 0.0)
            reading = sensor.update()
            if reading.is_valid():
                velocities.append(reading.velocity_x)
            time.sleep(0.01)

        # Average measured velocity should be close to true velocity
        if velocities:
            avg_velocity = sum(velocities) / len(velocities)
            assert abs(avg_velocity - true_velocity) < 0.5

    def test_hover_detection(self):
        """Test that hover (zero velocity) is detected."""
        sensor = OpticalFlowSensor()
        sensor.start()
        sensor.set_height(2.0)
        sensor.set_true_velocity(0.0, 0.0)

        speeds = []
        for _ in range(10):
            reading = sensor.update()
            speeds.append(reading.get_speed())
            time.sleep(0.01)

        # Average speed should be near zero
        avg_speed = sum(speeds) / len(speeds)
        assert avg_speed < 0.5  # Small due to noise
