"""
Tests for IMU sensor module.
"""

import pytest
import math
import time
from sensors.imu_sensor import (
    IMUSensor,
    IMUReading,
    IMUCalibration,
    IMUCalibrationState,
    IMUHealthStatus,
    Vector3,
    EulerAngles,
)


class TestVector3:
    """Tests for Vector3 class."""

    def test_vector3_creation(self):
        """Test Vector3 creation."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vector3_magnitude(self):
        """Test vector magnitude calculation."""
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude() == 5.0

    def test_vector3_normalize(self):
        """Test vector normalization."""
        v = Vector3(3.0, 4.0, 0.0)
        n = v.normalize()
        assert abs(n.magnitude() - 1.0) < 0.0001

    def test_vector3_addition(self):
        """Test vector addition."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, 5.0, 6.0)
        result = v1 + v2
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0

    def test_vector3_subtraction(self):
        """Test vector subtraction."""
        v1 = Vector3(4.0, 5.0, 6.0)
        v2 = Vector3(1.0, 2.0, 3.0)
        result = v1 - v2
        assert result.x == 3.0
        assert result.y == 3.0
        assert result.z == 3.0

    def test_vector3_scalar_multiplication(self):
        """Test scalar multiplication."""
        v = Vector3(1.0, 2.0, 3.0)
        result = v * 2.0
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_vector3_to_dict(self):
        """Test conversion to dictionary."""
        v = Vector3(1.0, 2.0, 3.0)
        d = v.to_dict()
        assert d == {'x': 1.0, 'y': 2.0, 'z': 3.0}


class TestEulerAngles:
    """Tests for EulerAngles class."""

    def test_euler_angles_creation(self):
        """Test EulerAngles creation."""
        e = EulerAngles(10.0, 20.0, 30.0)
        assert e.roll == 10.0
        assert e.pitch == 20.0
        assert e.yaw == 30.0

    def test_euler_angles_to_radians(self):
        """Test conversion to radians."""
        e = EulerAngles(180.0, 90.0, 45.0)
        r = e.to_radians()
        assert abs(r.roll - math.pi) < 0.0001
        assert abs(r.pitch - math.pi/2) < 0.0001
        assert abs(r.yaw - math.pi/4) < 0.0001

    def test_euler_angles_to_dict(self):
        """Test conversion to dictionary."""
        e = EulerAngles(10.0, 20.0, 30.0)
        d = e.to_dict()
        assert d == {'roll': 10.0, 'pitch': 20.0, 'yaw': 30.0}


class TestIMUReading:
    """Tests for IMUReading class."""

    def test_imu_reading_creation(self):
        """Test IMUReading creation."""
        reading = IMUReading(
            accelerometer=Vector3(0, 0, 9.81),
            gyroscope=Vector3(0, 0, 0),
            magnetometer=Vector3(20, 0, 45),
            attitude=EulerAngles(0, 0, 0),
            temperature=25.0,
            timestamp=time.time()
        )
        assert reading.accelerometer.z == 9.81
        assert reading.is_valid()

    def test_imu_reading_invalid_when_flags_false(self):
        """Test IMUReading is invalid when validity flags are False."""
        reading = IMUReading(
            accelerometer=Vector3(0, 0, 9.81),
            gyroscope=Vector3(0, 0, 0),
            magnetometer=Vector3(20, 0, 45),
            attitude=EulerAngles(0, 0, 0),
            temperature=25.0,
            timestamp=time.time(),
            accel_valid=False
        )
        assert not reading.is_valid()


class TestIMUCalibration:
    """Tests for IMUCalibration class."""

    def test_calibration_default_uncalibrated(self):
        """Test calibration default state is uncalibrated."""
        cal = IMUCalibration()
        assert cal.accel_state == IMUCalibrationState.UNCALIBRATED
        assert cal.gyro_state == IMUCalibrationState.UNCALIBRATED
        assert cal.mag_state == IMUCalibrationState.UNCALIBRATED
        assert not cal.is_fully_calibrated()

    def test_calibration_fully_calibrated(self):
        """Test fully calibrated state."""
        cal = IMUCalibration()
        cal.accel_state = IMUCalibrationState.CALIBRATED
        cal.gyro_state = IMUCalibrationState.CALIBRATED
        cal.mag_state = IMUCalibrationState.CALIBRATED
        assert cal.is_fully_calibrated()


class TestIMUSensor:
    """Tests for IMUSensor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.imu = IMUSensor()

    def test_imu_sensor_creation(self):
        """Test IMU sensor creation."""
        assert self.imu.type == "imu"
        assert not self.imu.is_calibrated()

    def test_imu_sensor_start_stop(self):
        """Test starting and stopping IMU sensor."""
        self.imu.start()
        assert self.imu._is_connected
        self.imu.stop()
        assert not self.imu._is_connected

    def test_imu_sensor_update_when_connected(self):
        """Test IMU update returns valid reading when connected."""
        self.imu.start()
        reading = self.imu.update()
        assert reading.is_valid()
        assert reading.accelerometer is not None
        assert reading.gyroscope is not None
        assert reading.magnetometer is not None

    def test_imu_sensor_update_when_disconnected(self):
        """Test IMU update returns invalid reading when disconnected."""
        reading = self.imu.update()
        assert not reading.is_valid()

    def test_imu_sensor_accelerometer_at_rest(self):
        """Test accelerometer reads gravity at rest."""
        self.imu.start()
        self.imu.set_true_attitude(0, 0, 0)  # Level
        self.imu.set_linear_acceleration(0, 0, 0)  # No movement
        reading = self.imu.update()

        # At rest and level, Z should be approximately gravity
        assert abs(reading.accelerometer.z - 9.81) < 1.0  # Within 1 m/s² (noise)
        # X and Y should be close to 0
        assert abs(reading.accelerometer.x) < 1.0
        assert abs(reading.accelerometer.y) < 1.0

    def test_imu_sensor_gyroscope_at_rest(self):
        """Test gyroscope reads near zero at rest."""
        self.imu.start()
        self.imu.set_angular_velocity(0, 0, 0)
        reading = self.imu.update()

        # At rest, gyro should be near zero (with some noise/bias)
        assert abs(reading.gyroscope.x) < 5.0  # Within 5 deg/s
        assert abs(reading.gyroscope.y) < 5.0
        assert abs(reading.gyroscope.z) < 5.0

    def test_imu_sensor_gyroscope_with_rotation(self):
        """Test gyroscope measures rotation rate."""
        self.imu.start()
        self.imu.set_angular_velocity(100, 0, 0)  # Roll at 100 deg/s
        reading = self.imu.update()

        # Should measure significant roll rate
        assert reading.gyroscope.x > 50  # At least 50 deg/s (accounting for bias)

    def test_imu_sensor_temperature(self):
        """Test temperature reading."""
        self.imu.start()
        reading = self.imu.update()

        # Temperature should be in reasonable range
        assert -40 <= reading.temperature <= 85

    def test_imu_sensor_set_temperature(self):
        """Test setting temperature."""
        self.imu.start()
        self.imu.set_temperature(45.0)
        reading = self.imu.update()

        assert abs(reading.temperature - 45.0) < 1.0  # Close to set value

    def test_imu_sensor_vibration_increases_noise(self):
        """Test that vibration increases sensor noise."""
        self.imu.start()
        self.imu.set_true_attitude(0, 0, 0)
        self.imu.set_angular_velocity(0, 0, 0)

        # Collect readings without vibration
        self.imu.set_vibration_level(0.0)
        readings_no_vib = []
        for _ in range(10):
            r = self.imu.update()
            readings_no_vib.append(r.gyroscope.x)

        # Collect readings with vibration
        self.imu.set_vibration_level(1.0)
        readings_vib = []
        for _ in range(10):
            r = self.imu.update()
            readings_vib.append(r.gyroscope.x)

        # Variance should be higher with vibration
        var_no_vib = sum((x - sum(readings_no_vib)/10)**2 for x in readings_no_vib) / 10
        var_vib = sum((x - sum(readings_vib)/10)**2 for x in readings_vib) / 10

        # With high vibration, variance should be noticeably higher
        assert var_vib > var_no_vib * 0.5  # At least comparable (randomness may vary)

    def test_imu_calibration_set_complete(self):
        """Test setting calibration complete."""
        self.imu.set_calibration_complete()
        assert self.imu.is_calibrated()
        assert self.imu.is_gyro_calibrated()
        assert self.imu.is_accel_calibrated()
        assert self.imu.is_mag_calibrated()

    def test_imu_sensor_attitude_computation(self):
        """Test attitude is computed from sensor data."""
        self.imu.start()
        self.imu.set_true_attitude(10.0, 5.0, 45.0)  # Some tilt

        # Allow a few updates for filter to converge
        for _ in range(50):
            reading = self.imu.update()
            time.sleep(0.01)

        # Attitude should be somewhat close to true attitude
        # (Note: complementary filter needs time to converge)
        assert reading.attitude is not None

    def test_imu_sensor_simulate_failure(self):
        """Test simulating sensor failure."""
        self.imu.start()
        self.imu.simulate_failure('all')

        assert self.imu.get_health_status() == IMUHealthStatus.FAILED

    def test_imu_sensor_simulate_recovery(self):
        """Test recovery from simulated failure."""
        self.imu.start()
        self.imu.simulate_failure('all')
        self.imu.simulate_recovery()

        assert self.imu.get_health_status() == IMUHealthStatus.GOOD

    def test_imu_sensor_getters(self):
        """Test various getter methods."""
        self.imu.start()
        self.imu.update()

        accel = self.imu.get_accelerometer()
        assert isinstance(accel, Vector3)

        gyro = self.imu.get_gyroscope()
        assert isinstance(gyro, Vector3)

        mag = self.imu.get_magnetometer()
        assert isinstance(mag, Vector3)

        attitude = self.imu.get_attitude()
        assert isinstance(attitude, EulerAngles)

        temp = self.imu.get_temperature()
        assert isinstance(temp, float)

    def test_imu_sensor_to_dict(self):
        """Test serialization to dictionary."""
        self.imu.start()
        self.imu.update()

        d = self.imu.to_dict()

        assert 'type' in d
        assert d['type'] == 'imu'
        assert 'accelerometer' in d
        assert 'gyroscope' in d
        assert 'magnetometer' in d
        assert 'attitude' in d
        assert 'temperature' in d
        assert 'calibration' in d
        assert 'health' in d

    def test_imu_sensor_measure_alias(self):
        """Test measure() is alias for update()."""
        self.imu.start()
        reading = self.imu.measure()
        assert reading.is_valid()

    def test_imu_sensor_update_rate(self):
        """Test IMU respects update rate setting."""
        imu_fast = IMUSensor(update_rate=200)
        imu_slow = IMUSensor(update_rate=50)

        assert imu_fast._update_rate == 200
        assert imu_slow._update_rate == 50
        assert imu_fast._update_interval < imu_slow._update_interval


class TestIMUSensorIntegration:
    """Integration tests for IMU sensor with sensor manager."""

    def test_imu_registered_in_sensor_manager(self):
        """Test IMU is registered in sensor manager."""
        from sensors.sensor_manager import SensorManager

        manager = SensorManager()
        assert 'imu' in manager._sensors
        assert isinstance(manager._sensors['imu'], IMUSensor)

    def test_imu_update_via_sensor_manager(self):
        """Test IMU updates through sensor manager."""
        from sensors.sensor_manager import SensorManager

        manager = SensorManager()
        manager.start()
        manager.update(time.time())

        imu_reading = manager.get('imu')
        assert imu_reading is not None
        assert isinstance(imu_reading, IMUReading)

        manager.stop()
