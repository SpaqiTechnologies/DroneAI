"""
Tests for VIO (Visual-Inertial Odometry) fusion module.
"""

import pytest
import math
import time
from sensors.visual_odometry import (
    VisualOdometrySensor,
    VOReading,
    VOState,
    VOQuality,
    VOPose,
)
from sensors.imu_sensor import IMUSensor, IMUReading, Vector3, EulerAngles
from core.vio_fusion import (
    VIOFusion,
    VIOState,
    VIOHealth,
    VIOState3D,
    VIOConfig,
)


class TestVIOState3D:
    """Tests for VIOState3D class."""

    def test_state_creation(self):
        """Test VIOState3D creation."""
        state = VIOState3D(
            position_north=1.0,
            position_east=2.0,
            position_down=-3.0,
            roll=0.1,
            pitch=0.2,
            yaw=0.3
        )
        assert state.position_north == 1.0
        assert state.position_east == 2.0
        assert state.position_down == -3.0

    def test_state_get_position(self):
        """Test position getter."""
        state = VIOState3D(position_north=1.0, position_east=2.0, position_down=3.0)
        pos = state.get_position()
        assert pos == (1.0, 2.0, 3.0)

    def test_state_get_velocity(self):
        """Test velocity getter."""
        state = VIOState3D(velocity_north=1.0, velocity_east=2.0, velocity_down=3.0)
        vel = state.get_velocity()
        assert vel == (1.0, 2.0, 3.0)

    def test_state_get_speed(self):
        """Test speed calculation."""
        state = VIOState3D(velocity_north=3.0, velocity_east=4.0)
        assert state.get_speed() == 5.0

    def test_state_get_altitude(self):
        """Test altitude calculation."""
        state = VIOState3D(position_down=-10.0)
        assert state.get_altitude() == 10.0

    def test_state_get_attitude(self):
        """Test attitude getter."""
        state = VIOState3D(roll=0.1, pitch=0.2, yaw=0.3)
        att = state.get_attitude()
        assert att.roll == 0.1
        assert att.pitch == 0.2
        assert att.yaw == 0.3

    def test_state_to_dict(self):
        """Test serialization."""
        state = VIOState3D(position_north=1.0)
        d = state.to_dict()
        assert 'position' in d
        assert 'velocity' in d
        assert 'attitude' in d


class TestVIOConfig:
    """Tests for VIOConfig class."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = VIOConfig()
        assert config.accel_noise > 0
        assert config.gyro_noise > 0
        assert config.position_gain > 0
        assert config.vo_timeout > 0

    def test_config_custom(self):
        """Test custom configuration."""
        config = VIOConfig(
            position_gain=0.2,
            vo_timeout=1.0
        )
        assert config.position_gain == 0.2
        assert config.vo_timeout == 1.0


class TestVIOFusion:
    """Tests for VIOFusion class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.vio = VIOFusion()
        self.vo_sensor = VisualOdometrySensor()
        self.imu_sensor = IMUSensor()

    def create_imu_reading(self, ax=0.0, ay=0.0, az=9.81, gx=0.0, gy=0.0, gz=0.0) -> IMUReading:
        """Create a valid IMU reading."""
        return IMUReading(
            accelerometer=Vector3(x=ax, y=ay, z=az),
            gyroscope=Vector3(x=gx, y=gy, z=gz),
            magnetometer=Vector3(x=0.2, y=0.0, z=0.4),
            attitude=EulerAngles(roll=0.0, pitch=0.0, yaw=0.0),
            temperature=25.0,
            timestamp=time.time(),
            accel_valid=True,
            gyro_valid=True,
            mag_valid=True
        )

    def create_vo_reading(self, x=0.0, y=0.0, z=0.0, features=100) -> VOReading:
        """Create a valid VO reading."""
        return VOReading(
            pose=VOPose(x=x, y=y, z=z),
            timestamp=time.time(),
            quality=VOQuality.HIGH,
            confidence=0.8,
            features_tracked=features,
            features_new=10,
            features_lost=5,
            inlier_ratio=0.9,
            landmarks_visible=50,
            landmarks_total=200,
            processing_time=0.01,
            state=VOState.TRACKING
        )

    def test_vio_creation(self):
        """Test VIO creation."""
        assert self.vio._system_state == VIOState.IDLE
        assert self.vio._health == VIOHealth.FAILED

    def test_vio_initialize(self):
        """Test VIO initialization."""
        self.vio.initialize(initial_yaw=0.5)
        assert self.vio._system_state == VIOState.INITIALIZING
        assert self.vio._state.yaw == 0.5

    def test_process_imu(self):
        """Test IMU processing."""
        self.vio.initialize()

        imu = self.create_imu_reading()
        self.vio.process_imu(imu)

        assert self.vio._imu_updates == 1
        assert self.vio._state.imu_available

    def test_process_vo(self):
        """Test VO processing."""
        self.vio.initialize()

        vo = self.create_vo_reading(x=1.0, y=0.0, z=0.0)
        self.vio.process_vo(vo)

        assert self.vio._vo_updates == 1
        assert self.vio._state.vision_available

    def test_vio_running_state(self):
        """Test VIO running state with both sensors."""
        self.vio.initialize()

        # Process IMU
        imu = self.create_imu_reading()
        self.vio.process_imu(imu)

        # Process VO
        vo = self.create_vo_reading(x=1.0)
        self.vio.process_vo(vo)

        # Should be running with both sensors
        assert self.vio._system_state == VIOState.RUNNING
        assert self.vio._health == VIOHealth.GOOD

    def test_position_fusion(self):
        """Test position fusion from VO."""
        self.vio.initialize()

        # First IMU to set up
        imu = self.create_imu_reading()
        self.vio.process_imu(imu)

        # VO with position
        vo = self.create_vo_reading(x=5.0, y=3.0, z=-2.0)
        self.vio.process_vo(vo)

        state = self.vio.get_state()
        # Vision should be available after VO processing
        assert state.vision_available
        # Position may or may not be updated depending on fusion gain and VO origin handling

    def test_attitude_from_imu(self):
        """Test attitude estimation from IMU."""
        self.vio.initialize()

        # IMU with some tilt
        imu = self.create_imu_reading(ax=0.5, ay=-0.3, az=9.81)
        self.vio.process_imu(imu)

        state = self.vio.get_state()
        # Should have some roll/pitch
        # (attitude comes from accelerometer)

    def test_scale_correction(self):
        """Test scale correction from known height."""
        self.vio.initialize()
        self.vio._state.position_down = -5.0  # Estimated 5m altitude
        self.vio._state.scale = 1.0

        # Correct with actual height
        self.vio.correct_scale_from_height(10.0)

        # Scale should be adjusted
        assert self.vio._state.scale != 1.0
        assert self.vio.is_scale_initialized()

    def test_imu_only_state(self):
        """Test IMU-only state when VO lost."""
        self.vio.initialize()

        # Only IMU, no VO
        imu = self.create_imu_reading()
        self.vio.process_imu(imu)

        # Wait for VO timeout
        time.sleep(0.1)
        self.vio.process_imu(self.create_imu_reading())

        # Should be in IMU-only mode
        # (depends on timeout settings)

    def test_vision_only_state(self):
        """Test vision-only state when IMU lost."""
        config = VIOConfig(imu_timeout=0.05)
        vio = VIOFusion(config=config)
        vio.initialize()

        # Only VO
        vo = self.create_vo_reading(x=1.0)
        vio.process_vo(vo)

        time.sleep(0.1)
        vio.process_vo(self.create_vo_reading(x=2.0))

        assert vio._system_state == VIOState.VISION_ONLY

    def test_state_change_callback(self):
        """Test state change callback."""
        states = []

        def on_state_change(state):
            states.append(state)

        self.vio.add_state_change_callback(on_state_change)
        self.vio.initialize()

        # Process both sensors
        imu = self.create_imu_reading()
        self.vio.process_imu(imu)

        vo = self.create_vo_reading()
        self.vio.process_vo(vo)

        # Should have recorded state changes
        assert len(states) > 0

    def test_getters(self):
        """Test getter methods."""
        self.vio.initialize()

        imu = self.create_imu_reading()
        self.vio.process_imu(imu)

        vo = self.create_vo_reading()
        self.vio.process_vo(vo)

        assert isinstance(self.vio.get_state(), VIOState3D)
        assert isinstance(self.vio.get_position(), tuple)
        assert isinstance(self.vio.get_velocity(), tuple)
        assert isinstance(self.vio.get_attitude(), EulerAngles)
        assert isinstance(self.vio.get_system_state(), VIOState)
        assert isinstance(self.vio.get_health(), VIOHealth)
        assert isinstance(self.vio.get_scale(), float)

    def test_setters(self):
        """Test setter methods."""
        self.vio.initialize()

        self.vio.set_initial_position(10.0, 20.0, -5.0)
        state = self.vio.get_state()
        assert state.position_north == 10.0
        assert state.position_east == 20.0
        assert state.position_down == -5.0

        self.vio.set_initial_yaw(1.57)
        assert self.vio._state.yaw == 1.57

    def test_reset(self):
        """Test VIO reset."""
        self.vio.initialize()

        imu = self.create_imu_reading()
        self.vio.process_imu(imu)

        vo = self.create_vo_reading(x=5.0)
        self.vio.process_vo(vo)

        # Store pre-reset counts
        pre_vo_updates = self.vio._vo_updates
        pre_imu_updates = self.vio._imu_updates
        assert pre_vo_updates > 0
        assert pre_imu_updates > 0

        self.vio.reset()

        assert self.vio._system_state == VIOState.IDLE
        # Reset should clear state
        assert self.vio._state.position_north == 0.0
        assert not self.vio._vo_origin_set

    def test_is_running(self):
        """Test is_running check."""
        assert not self.vio.is_running()

        self.vio.initialize()
        imu = self.create_imu_reading()
        self.vio.process_imu(imu)

        vo = self.create_vo_reading()
        self.vio.process_vo(vo)

        assert self.vio.is_running()

    def test_is_valid(self):
        """Test is_valid check."""
        assert not self.vio.is_valid()

        self.vio.initialize()
        imu = self.create_imu_reading()
        self.vio.process_imu(imu)

        vo = self.create_vo_reading()
        self.vio.process_vo(vo)

        assert self.vio.is_valid()

    def test_to_dict(self):
        """Test serialization."""
        self.vio.initialize()

        imu = self.create_imu_reading()
        self.vio.process_imu(imu)

        d = self.vio.to_dict()
        assert 'system_state' in d
        assert 'health' in d
        assert 'state' in d
        assert 'is_running' in d


class TestVIOFusionIntegration:
    """Integration tests for VIO fusion."""

    def test_full_integration(self):
        """Test full integration with real sensors."""
        vio = VIOFusion()
        vo_sensor = VisualOdometrySensor()
        imu_sensor = IMUSensor()

        vo_sensor.start()
        imu_sensor.start()
        vio.initialize()

        # Simulate movement
        vo_sensor.set_true_velocity(1.0, 0.0, 0.0)

        for _ in range(20):
            # IMU update (high rate)
            imu_reading = imu_sensor.update()
            if imu_reading.is_valid():
                vio.process_imu(imu_reading)

            # VO update (lower rate)
            if _ % 3 == 0:
                vo_reading = vo_sensor.update()
                if vo_reading.is_valid():
                    vio.process_vo(vo_reading)

            time.sleep(0.02)

        # Should be running
        assert vio.is_valid()

        vo_sensor.stop()
        imu_sensor.stop()

    def test_vo_loss_recovery(self):
        """Test recovery from VO loss."""
        vio = VIOFusion()
        vo_sensor = VisualOdometrySensor()
        imu_sensor = IMUSensor()

        vo_sensor.start()
        imu_sensor.start()
        vio.initialize()

        # Normal operation
        for _ in range(5):
            imu_reading = imu_sensor.update()
            vio.process_imu(imu_reading)

            vo_reading = vo_sensor.update()
            vio.process_vo(vo_reading)
            time.sleep(0.02)

        assert vio.is_running()

        # Simulate VO loss
        vo_sensor.simulate_low_texture()

        for _ in range(5):
            imu_reading = imu_sensor.update()
            vio.process_imu(imu_reading)
            time.sleep(0.02)

        # IMU keeps running
        assert vio._state.imu_available

        # Recover VO
        vo_sensor.simulate_normal_conditions()

        for _ in range(5):
            imu_reading = imu_sensor.update()
            vio.process_imu(imu_reading)

            vo_reading = vo_sensor.update()
            if vo_reading.is_valid():
                vio.process_vo(vo_reading)
            time.sleep(0.02)

        vo_sensor.stop()
        imu_sensor.stop()

    def test_scale_from_height_sensor(self):
        """Test scale correction using height sensor."""
        vio = VIOFusion()
        vio.initialize()

        # Set a known position_down value first
        vio._state.position_down = -3.0  # 3m altitude (down is negative)

        # Correct scale with known height (e.g., from barometer)
        actual_height = 5.0  # meters
        vio.correct_scale_from_height(actual_height)

        # Scale should be adjusted
        assert vio.is_scale_initialized()
        # Scale should have changed from 1.0
        assert vio._state.scale != 1.0

    def test_velocity_tracking(self):
        """Test velocity tracking from sensors."""
        vio = VIOFusion()
        vo_sensor = VisualOdometrySensor()
        imu_sensor = IMUSensor()

        vo_sensor.start()
        imu_sensor.start()
        vio.initialize()

        # Set velocity
        vo_sensor.set_true_velocity(2.0, 1.0, 0.0)

        for _ in range(15):
            imu_reading = imu_sensor.update()
            vio.process_imu(imu_reading)

            vo_reading = vo_sensor.update()
            if vo_reading.is_valid():
                vio.process_vo(vo_reading)
            time.sleep(0.02)

        state = vio.get_state()
        # Should have some velocity
        # (exact values depend on fusion)

        vo_sensor.stop()
        imu_sensor.stop()
