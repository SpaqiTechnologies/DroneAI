"""
Tests for State Estimator (EKF) module.
"""

import pytest
import math
import time
from core.state_estimator import (
    StateEstimator,
    NavigationState,
    NavigationMode,
    EstimatorHealth,
    EKFConfig,
    Matrix,
)
from sensors.imu_sensor import IMUReading, Vector3, EulerAngles
from sensors.gps_sensor import GPSReading


class TestMatrix:
    """Tests for Matrix class."""

    def test_matrix_creation(self):
        """Test matrix creation."""
        m = Matrix(3, 3)
        assert m.rows == 3
        assert m.cols == 3
        assert m[0, 0] == 0.0

    def test_matrix_identity(self):
        """Test identity matrix creation."""
        m = Matrix.identity(3)
        assert m[0, 0] == 1.0
        assert m[1, 1] == 1.0
        assert m[2, 2] == 1.0
        assert m[0, 1] == 0.0

    def test_matrix_diagonal(self):
        """Test diagonal matrix creation."""
        m = Matrix.diagonal([1.0, 2.0, 3.0])
        assert m[0, 0] == 1.0
        assert m[1, 1] == 2.0
        assert m[2, 2] == 3.0
        assert m[0, 1] == 0.0

    def test_matrix_addition(self):
        """Test matrix addition."""
        m1 = Matrix.identity(2)
        m2 = Matrix.identity(2)
        result = m1 + m2
        assert result[0, 0] == 2.0
        assert result[1, 1] == 2.0

    def test_matrix_subtraction(self):
        """Test matrix subtraction."""
        m1 = Matrix.diagonal([2.0, 4.0])
        m2 = Matrix.identity(2)
        result = m1 - m2
        assert result[0, 0] == 1.0
        assert result[1, 1] == 3.0

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 2, [[5, 6], [7, 8]])
        result = m1 * m2
        assert result[0, 0] == 19  # 1*5 + 2*7
        assert result[0, 1] == 22  # 1*6 + 2*8
        assert result[1, 0] == 43  # 3*5 + 4*7
        assert result[1, 1] == 50  # 3*6 + 4*8

    def test_matrix_scalar_multiplication(self):
        """Test scalar multiplication."""
        m = Matrix.identity(2)
        result = m * 3.0
        assert result[0, 0] == 3.0
        assert result[1, 1] == 3.0

    def test_matrix_transpose(self):
        """Test matrix transpose."""
        m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        t = m.transpose()
        assert t.rows == 3
        assert t.cols == 2
        assert t[0, 0] == 1
        assert t[2, 1] == 6

    def test_matrix_inverse_3x3(self):
        """Test 3x3 matrix inverse."""
        m = Matrix(3, 3, [[1, 0, 0], [0, 2, 0], [0, 0, 4]])
        inv = m.inverse_3x3()
        assert abs(inv[0, 0] - 1.0) < 0.0001
        assert abs(inv[1, 1] - 0.5) < 0.0001
        assert abs(inv[2, 2] - 0.25) < 0.0001


class TestNavigationState:
    """Tests for NavigationState class."""

    def test_navigation_state_creation(self):
        """Test NavigationState creation."""
        state = NavigationState()
        assert state.position_north == 0.0
        assert state.navigation_mode == NavigationMode.FAILED

    def test_get_position_ned(self):
        """Test getting position as NED tuple."""
        state = NavigationState(
            position_north=100.0,
            position_east=50.0,
            position_down=-30.0
        )
        n, e, d = state.get_position_ned()
        assert n == 100.0
        assert e == 50.0
        assert d == -30.0

    def test_get_altitude_agl(self):
        """Test getting altitude AGL."""
        state = NavigationState(position_down=-50.0)
        assert state.get_altitude_agl() == 50.0

    def test_get_speed(self):
        """Test getting horizontal speed."""
        state = NavigationState(velocity_north=3.0, velocity_east=4.0)
        assert state.get_speed() == 5.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = NavigationState(
            position_north=10.0,
            velocity_north=2.0,
            roll=5.0
        )
        d = state.to_dict()
        assert 'position' in d
        assert 'velocity' in d
        assert 'attitude' in d
        assert d['position']['north'] == 10.0


class TestStateEstimator:
    """Tests for StateEstimator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = StateEstimator()
        # Set home position (Berlin)
        self.estimator.set_home_position(52.52, 13.405, 50.0)

    def create_imu_reading(self, accel_z=9.81, gyro_x=0, gyro_y=0, gyro_z=0) -> IMUReading:
        """Helper to create IMU reading."""
        return IMUReading(
            accelerometer=Vector3(0, 0, accel_z),
            gyroscope=Vector3(gyro_x, gyro_y, gyro_z),
            magnetometer=Vector3(20, 0, 45),
            attitude=EulerAngles(0, 0, 0),
            temperature=25.0,
            timestamp=time.time()
        )

    def create_gps_reading(self, lat=52.52, lon=13.405, alt=50.0,
                           satellites=8, accuracy=3.0) -> GPSReading:
        """Helper to create GPS reading."""
        return GPSReading(
            latitude=lat,
            longitude=lon,
            altitude=alt,
            accuracy_horizontal=accuracy,
            accuracy_vertical=accuracy * 1.5,
            satellites_used=satellites,
            fix_type=2,  # 3D fix
            timestamp=time.time(),
            speed=0.0,
            heading=0.0
        )

    def test_estimator_creation(self):
        """Test estimator creation."""
        est = StateEstimator()
        assert not est.is_initialized()
        assert est.get_navigation_mode() == NavigationMode.FAILED

    def test_set_home_position(self):
        """Test setting home position."""
        est = StateEstimator()
        est.set_home_position(52.52, 13.405, 50.0)
        assert est.is_initialized()
        assert est._home_set

    def test_predict_updates_state(self):
        """Test that predict step updates state."""
        imu = self.create_imu_reading()
        self.estimator.predict(imu)

        # After one IMU update, should still be at origin (no dt)
        state = self.estimator.get_state()
        assert abs(state.position_north) < 1.0

    def test_predict_with_acceleration(self):
        """Test prediction with acceleration."""
        # First reading to initialize timing
        imu1 = self.create_imu_reading()
        self.estimator.predict(imu1)

        # Wait and apply acceleration
        time.sleep(0.05)
        imu2 = IMUReading(
            accelerometer=Vector3(1.0, 0, 9.81),  # 1 m/s² forward
            gyroscope=Vector3(0, 0, 0),
            magnetometer=Vector3(20, 0, 45),
            attitude=EulerAngles(0, 0, 0),
            temperature=25.0,
            timestamp=time.time()
        )
        self.estimator.predict(imu2)

        state = self.estimator.get_state()
        # Should have moved north slightly due to acceleration
        assert state.velocity_north != 0  # Velocity changed

    def test_gps_update_corrects_position(self):
        """Test that GPS update corrects position."""
        # Use larger innovation gate for this test
        config = EKFConfig(innovation_gate=50.0)  # Allow large innovations
        estimator = StateEstimator(config=config)
        estimator.set_home_position(52.52, 13.405, 50.0)

        # First establish timing with IMU
        imu1 = self.create_imu_reading()
        estimator.predict(imu1)
        time.sleep(0.02)

        # Second IMU to establish dt
        imu2 = self.create_imu_reading()
        imu2.timestamp = time.time()
        estimator.predict(imu2)

        # GPS at offset position (~111m north for 0.001 degree)
        gps = self.create_gps_reading(lat=52.521, lon=13.405)
        estimator.update_gps(gps)

        state = estimator.get_state()
        # Position should be corrected toward GPS
        assert state.position_north > 50  # Moved north toward GPS position
        assert state.gps_available

    def test_navigation_mode_gps_imu_fusion(self):
        """Test navigation mode with both GPS and IMU."""
        # First IMU to initialize
        imu1 = self.create_imu_reading()
        self.estimator.predict(imu1)
        time.sleep(0.02)

        # Second IMU to establish dt and imu_available
        imu2 = self.create_imu_reading()
        imu2.timestamp = time.time()
        self.estimator.predict(imu2)

        # GPS update
        gps = self.create_gps_reading()
        self.estimator.update_gps(gps)

        # Third IMU after GPS to trigger mode update with both available
        time.sleep(0.02)
        imu3 = self.create_imu_reading()
        imu3.timestamp = time.time()
        self.estimator.predict(imu3)

        assert self.estimator.get_navigation_mode() == NavigationMode.GPS_IMU_FUSION
        assert self.estimator.is_gps_available()

    def test_navigation_mode_dead_reckoning(self):
        """Test dead reckoning mode when GPS lost."""
        # Initialize with IMU
        imu1 = self.create_imu_reading()
        self.estimator.predict(imu1)
        time.sleep(0.02)

        imu2 = self.create_imu_reading()
        imu2.timestamp = time.time()
        self.estimator.predict(imu2)

        # Add GPS
        gps = self.create_gps_reading()
        self.estimator.update_gps(gps)

        # IMU after GPS to get fusion mode
        time.sleep(0.02)
        imu3 = self.create_imu_reading()
        imu3.timestamp = time.time()
        self.estimator.predict(imu3)

        assert self.estimator.get_navigation_mode() == NavigationMode.GPS_IMU_FUSION

        # Now lose GPS (invalid reading)
        time.sleep(0.02)
        imu4 = self.create_imu_reading()
        imu4.timestamp = time.time()
        self.estimator.predict(imu4)

        bad_gps = GPSReading(
            latitude=0, longitude=0, altitude=0,
            accuracy_horizontal=100, accuracy_vertical=150,
            satellites_used=1,  # Too few satellites
            fix_type=0,  # No fix
            timestamp=time.time(),
            speed=0, heading=0
        )
        self.estimator.update_gps(bad_gps)

        # IMU after bad GPS to update mode
        time.sleep(0.02)
        imu5 = self.create_imu_reading()
        imu5.timestamp = time.time()
        self.estimator.predict(imu5)

        assert self.estimator.get_navigation_mode() == NavigationMode.DEAD_RECKONING
        assert not self.estimator.is_gps_available()

    def test_dead_reckoning_time_tracking(self):
        """Test dead reckoning time is tracked."""
        # Setup with GPS then lose it
        imu = self.create_imu_reading()
        self.estimator.predict(imu)
        gps = self.create_gps_reading()
        self.estimator.update_gps(gps)

        # Initially no DR time
        assert self.estimator.get_dead_reckoning_time() == 0.0

        # Lose GPS
        time.sleep(0.05)
        imu2 = self.create_imu_reading()
        imu2.timestamp = time.time()
        self.estimator.predict(imu2)

        bad_gps = GPSReading(
            latitude=0, longitude=0, altitude=0,
            accuracy_horizontal=100, accuracy_vertical=150,
            satellites_used=0, fix_type=0,
            timestamp=time.time(), speed=0, heading=0
        )
        self.estimator.update_gps(bad_gps)

        # Now in DR mode
        time.sleep(0.1)
        dr_time = self.estimator.get_dead_reckoning_time()
        assert dr_time > 0

    def test_gps_to_ned_conversion(self):
        """Test GPS to NED coordinate conversion."""
        # At home position
        n, e = self.estimator._gps_to_ned(52.52, 13.405)
        assert abs(n) < 1.0
        assert abs(e) < 1.0

        # 1 degree north (~111km)
        n, e = self.estimator._gps_to_ned(53.52, 13.405)
        assert n > 100000  # Should be >100km north

    def test_ned_to_gps_conversion(self):
        """Test NED to GPS coordinate conversion."""
        lat, lon = self.estimator._ned_to_gps(0, 0)
        assert abs(lat - 52.52) < 0.0001
        assert abs(lon - 13.405) < 0.0001

    def test_get_gps_position(self):
        """Test getting current position as GPS coordinates."""
        lat, lon, alt = self.estimator.get_gps_position()
        assert abs(lat - 52.52) < 0.01
        assert abs(lon - 13.405) < 0.01

    def test_estimator_reset(self):
        """Test estimator reset."""
        # Do some updates
        imu = self.create_imu_reading()
        self.estimator.predict(imu)
        gps = self.create_gps_reading(lat=52.521)
        self.estimator.update_gps(gps)

        # Reset
        self.estimator.reset()

        state = self.estimator.get_state()
        assert state.position_north == 0.0
        assert self.estimator.is_initialized()  # Home still set

    def test_mode_change_callback(self):
        """Test navigation mode change callback."""
        mode_changes = []

        def on_mode_change(mode):
            mode_changes.append(mode)

        self.estimator.add_mode_change_callback(on_mode_change)

        # Initialize IMU
        imu1 = self.create_imu_reading()
        self.estimator.predict(imu1)
        time.sleep(0.02)

        imu2 = self.create_imu_reading()
        imu2.timestamp = time.time()
        self.estimator.predict(imu2)

        # Add GPS
        gps = self.create_gps_reading()
        self.estimator.update_gps(gps)

        # IMU after GPS triggers fusion mode
        time.sleep(0.02)
        imu3 = self.create_imu_reading()
        imu3.timestamp = time.time()
        self.estimator.predict(imu3)

        assert len(mode_changes) > 0
        assert NavigationMode.GPS_IMU_FUSION in mode_changes

    def test_innovation_gate_rejects_outliers(self):
        """Test that innovation gate rejects GPS outliers."""
        # Get stable state
        for _ in range(5):
            imu = self.create_imu_reading()
            self.estimator.predict(imu)
            time.sleep(0.01)

        gps = self.create_gps_reading()
        self.estimator.update_gps(gps)

        state_before = self.estimator.get_state()

        # Apply GPS with huge jump (should be rejected)
        time.sleep(0.05)
        imu2 = self.create_imu_reading()
        imu2.timestamp = time.time()
        self.estimator.predict(imu2)

        outlier_gps = self.create_gps_reading(lat=60.0)  # Way off!
        self.estimator.update_gps(outlier_gps)

        state_after = self.estimator.get_state()

        # Position should not have jumped to the outlier
        assert abs(state_after.position_north - state_before.position_north) < 1000

    def test_health_status(self):
        """Test health status reporting."""
        # Initialize with both sensors for good health
        imu1 = self.create_imu_reading()
        self.estimator.predict(imu1)
        time.sleep(0.02)

        imu2 = self.create_imu_reading()
        imu2.timestamp = time.time()
        self.estimator.predict(imu2)

        gps = self.create_gps_reading()
        self.estimator.update_gps(gps)

        # IMU after GPS to update mode
        time.sleep(0.02)
        imu3 = self.create_imu_reading()
        imu3.timestamp = time.time()
        self.estimator.predict(imu3)

        # With GPS+IMU should be good
        assert self.estimator.get_health() == EstimatorHealth.GOOD

    def test_to_dict(self):
        """Test serialization to dictionary."""
        imu = self.create_imu_reading()
        self.estimator.predict(imu)

        d = self.estimator.to_dict()

        assert 'state' in d
        assert 'home' in d
        assert 'health' in d
        assert d['home']['lat'] == 52.52

    def test_custom_config(self):
        """Test custom EKF configuration."""
        config = EKFConfig(
            gps_position_noise=5.0,
            max_dead_reckoning_time=60.0,
            min_gps_satellites=6
        )
        est = StateEstimator(config=config)

        assert est.config.gps_position_noise == 5.0
        assert est.config.max_dead_reckoning_time == 60.0
        assert est.config.min_gps_satellites == 6


class TestStateEstimatorIntegration:
    """Integration tests for state estimator."""

    def test_continuous_updates(self):
        """Test continuous IMU updates with periodic GPS."""
        est = StateEstimator()
        est.set_home_position(52.52, 13.405, 50.0)

        # Simulate 1 second of updates at 100Hz IMU, 5Hz GPS
        for i in range(100):
            imu = IMUReading(
                accelerometer=Vector3(0.1, 0, 9.81),  # Slight forward accel
                gyroscope=Vector3(0, 0, 0),
                magnetometer=Vector3(20, 0, 45),
                attitude=EulerAngles(0, 0, 0),
                temperature=25.0,
                timestamp=time.time()
            )
            est.predict(imu)

            if i % 20 == 0:  # GPS at 5Hz
                gps = GPSReading(
                    latitude=52.52 + i * 0.00001,  # Moving north
                    longitude=13.405,
                    altitude=50.0,
                    accuracy_horizontal=3.0,
                    accuracy_vertical=4.5,
                    satellites_used=8,
                    fix_type=2,
                    timestamp=time.time(),
                    speed=1.0,
                    heading=0.0
                )
                est.update_gps(gps)

            time.sleep(0.01)

        state = est.get_state()
        assert state.navigation_mode == NavigationMode.GPS_IMU_FUSION
        assert state.position_north > 0  # Should have moved north
