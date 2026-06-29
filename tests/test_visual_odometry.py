"""
Tests for Visual Odometry sensor module.
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
    Feature2D,
    Feature3D,
)


class TestFeature2D:
    """Tests for Feature2D class."""

    def test_feature_creation(self):
        """Test Feature2D creation."""
        feature = Feature2D(x=100.0, y=200.0, id=1, response=0.8, age=5)
        assert feature.x == 100.0
        assert feature.y == 200.0
        assert feature.id == 1
        assert feature.response == 0.8
        assert feature.age == 5

    def test_feature_to_dict(self):
        """Test serialization."""
        feature = Feature2D(x=100.0, y=200.0, id=1)
        d = feature.to_dict()
        assert 'x' in d
        assert 'y' in d
        assert 'id' in d


class TestFeature3D:
    """Tests for Feature3D class."""

    def test_landmark_creation(self):
        """Test Feature3D creation."""
        landmark = Feature3D(x=1.0, y=2.0, z=3.0, id=1, observations=10)
        assert landmark.x == 1.0
        assert landmark.y == 2.0
        assert landmark.z == 3.0
        assert landmark.observations == 10

    def test_landmark_distance(self):
        """Test distance calculation."""
        landmark = Feature3D(x=3.0, y=4.0, z=0.0)
        assert landmark.distance_from_origin() == 5.0

    def test_landmark_to_dict(self):
        """Test serialization."""
        landmark = Feature3D(x=1.0, y=2.0, z=3.0)
        d = landmark.to_dict()
        assert 'x' in d
        assert 'uncertainty' in d


class TestVOPose:
    """Tests for VOPose class."""

    def test_pose_creation(self):
        """Test VOPose creation."""
        pose = VOPose(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0

    def test_pose_get_position(self):
        """Test position getter."""
        pose = VOPose(x=1.0, y=2.0, z=3.0)
        pos = pose.get_position()
        assert pos == (1.0, 2.0, 3.0)

    def test_pose_get_velocity(self):
        """Test velocity getter."""
        pose = VOPose(vx=1.0, vy=2.0, vz=3.0)
        vel = pose.get_velocity()
        assert vel == (1.0, 2.0, 3.0)

    def test_pose_get_speed(self):
        """Test speed calculation."""
        pose = VOPose(vx=3.0, vy=4.0, vz=0.0)
        assert pose.get_speed() == 5.0

    def test_pose_to_dict(self):
        """Test serialization."""
        pose = VOPose(x=1.0, y=2.0, z=3.0)
        d = pose.to_dict()
        assert 'position' in d
        assert 'velocity' in d
        assert 'orientation' in d


class TestVOReading:
    """Tests for VOReading class."""

    def test_reading_is_valid(self):
        """Test validity check."""
        valid_reading = VOReading(
            pose=VOPose(),
            timestamp=time.time(),
            quality=VOQuality.HIGH,
            confidence=0.8,
            features_tracked=100,
            features_new=10,
            features_lost=5,
            inlier_ratio=0.9,
            landmarks_visible=50,
            landmarks_total=200,
            processing_time=0.01,
            state=VOState.TRACKING
        )
        assert valid_reading.is_valid()

        invalid_reading = VOReading(
            pose=VOPose(),
            timestamp=time.time(),
            quality=VOQuality.INVALID,
            confidence=0.1,
            features_tracked=5,
            features_new=0,
            features_lost=50,
            inlier_ratio=0.2,
            landmarks_visible=0,
            landmarks_total=0,
            processing_time=0.01,
            state=VOState.LOST
        )
        assert not invalid_reading.is_valid()

    def test_reading_to_dict(self):
        """Test serialization."""
        reading = VOReading(
            pose=VOPose(x=1.0),
            timestamp=time.time(),
            quality=VOQuality.HIGH,
            confidence=0.8,
            features_tracked=100,
            features_new=10,
            features_lost=5,
            inlier_ratio=0.9,
            landmarks_visible=50,
            landmarks_total=200,
            processing_time=0.01,
            state=VOState.TRACKING
        )
        d = reading.to_dict()
        assert 'pose' in d
        assert 'quality' in d
        assert 'is_valid' in d


class TestVisualOdometrySensor:
    """Tests for VisualOdometrySensor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sensor = VisualOdometrySensor()

    def test_sensor_creation(self):
        """Test sensor creation."""
        assert self.sensor.type == "visual_odometry"
        assert self.sensor._state == VOState.OFF

    def test_sensor_custom_params(self):
        """Test sensor with custom parameters."""
        sensor = VisualOdometrySensor(
            frame_rate=60,
            resolution=(1920, 1080),
            fov=120.0,
            stereo=True,
            baseline=0.12
        )
        assert sensor._frame_rate == 60
        assert sensor._resolution == (1920, 1080)
        assert sensor._stereo == True
        assert sensor._baseline == 0.12

    def test_sensor_start_stop(self):
        """Test starting and stopping sensor."""
        self.sensor.start()
        assert self.sensor._is_connected
        assert self.sensor._state == VOState.TRACKING
        assert len(self.sensor._features_2d) > 0

        self.sensor.stop()
        assert not self.sensor._is_connected
        assert self.sensor._state == VOState.OFF
        assert len(self.sensor._features_2d) == 0

    def test_update_when_disconnected(self):
        """Test update returns invalid reading when disconnected."""
        reading = self.sensor.update()
        assert not reading.is_valid()
        assert reading.state == VOState.OFF

    def test_update_when_connected(self):
        """Test update returns valid reading when connected."""
        self.sensor.start()
        reading = self.sensor.update()

        assert reading is not None
        assert isinstance(reading, VOReading)
        assert reading.features_tracked > 0

    def test_feature_tracking(self):
        """Test that features are tracked across frames."""
        self.sensor.start()

        reading1 = self.sensor.update()
        initial_count = reading1.features_tracked

        time.sleep(0.02)

        reading2 = self.sensor.update()

        # Should have similar feature count (some tracked, some new)
        assert reading2.features_tracked > 0
        assert reading2.features_new >= 0

    def test_pose_update_with_velocity(self):
        """Test pose updates with velocity."""
        self.sensor.start()
        self.sensor.set_true_velocity(1.0, 0.0, 0.0)  # 1 m/s forward

        # Process several frames
        for _ in range(10):
            self.sensor.update()
            time.sleep(0.02)

        pose = self.sensor.get_pose()
        # Should have moved forward
        assert pose.x > 0

    def test_quality_assessment(self):
        """Test quality assessment based on features."""
        self.sensor.start()
        self.sensor.set_texture_quality(1.0)
        self.sensor.set_lighting_level(1.0)

        reading = self.sensor.update()

        # Good conditions should give decent quality
        assert reading.quality != VOQuality.INVALID
        assert reading.confidence > 0

    def test_low_texture_degrades_quality(self):
        """Test that low texture degrades tracking."""
        self.sensor.start()
        self.sensor.set_texture_quality(1.0)
        reading_good = self.sensor.update()

        time.sleep(0.02)

        self.sensor.simulate_low_texture()
        reading_poor = self.sensor.update()

        assert reading_poor.features_tracked <= reading_good.features_tracked

    def test_low_light_degrades_quality(self):
        """Test that low light degrades tracking."""
        self.sensor.start()
        reading_bright = self.sensor.update()

        time.sleep(0.02)

        self.sensor.simulate_low_light()
        reading_dark = self.sensor.update()

        assert reading_dark.confidence <= reading_bright.confidence

    def test_motion_blur_degrades_quality(self):
        """Test that motion blur degrades quality."""
        self.sensor.start()
        reading_clear = self.sensor.update()

        time.sleep(0.02)

        self.sensor.simulate_fast_motion()
        reading_blurred = self.sensor.update()

        assert reading_blurred.quality.value <= reading_clear.quality.value or reading_blurred.confidence <= reading_clear.confidence

    def test_normal_conditions_reset(self):
        """Test resetting to normal conditions."""
        self.sensor.start()
        self.sensor.simulate_low_texture()
        self.sensor.simulate_low_light()

        self.sensor.simulate_normal_conditions()
        assert self.sensor._texture_quality == 1.0
        assert self.sensor._lighting_level == 1.0
        assert self.sensor._motion_blur == 0.0

    def test_tracking_loss_detection(self):
        """Test detection of tracking loss."""
        self.sensor.start()
        self.sensor.set_texture_quality(0.01)  # Very low texture

        # Process frames until tracking is lost
        for _ in range(20):
            reading = self.sensor.update()
            time.sleep(0.02)
            if reading.state == VOState.LOST:
                break

        # Should eventually lose tracking
        assert self.sensor._lost_count > 0 or self.sensor._features_tracked < self.sensor.MIN_FEATURES

    def test_measure_alias(self):
        """Test measure() is alias for update()."""
        self.sensor.start()
        reading = self.sensor.measure()
        assert isinstance(reading, VOReading)

    def test_getters(self):
        """Test getter methods."""
        self.sensor.start()
        self.sensor.set_true_velocity(1.0, 0.5, 0.0)
        self.sensor.update()

        assert isinstance(self.sensor.get_pose(), VOPose)
        assert isinstance(self.sensor.get_position(), tuple)
        assert isinstance(self.sensor.get_velocity(), tuple)
        assert isinstance(self.sensor.get_speed(), float)
        assert isinstance(self.sensor.get_state(), VOState)
        assert isinstance(self.sensor.get_quality(), VOQuality)
        assert isinstance(self.sensor.get_confidence(), float)

    def test_frame_count(self):
        """Test frame counter."""
        self.sensor.start()
        assert self.sensor.get_frame_count() == 0

        self.sensor.update()
        assert self.sensor.get_frame_count() == 1

        self.sensor.update()
        assert self.sensor.get_frame_count() == 2

    def test_reset_pose(self):
        """Test pose reset."""
        self.sensor.start()
        self.sensor.set_true_velocity(1.0, 0.0, 0.0)

        for _ in range(5):
            self.sensor.update()
            time.sleep(0.02)

        self.sensor.reset_pose()
        pose = self.sensor.get_pose()
        assert pose.x == 0.0
        assert pose.y == 0.0
        assert pose.z == 0.0

    def test_scale_correction(self):
        """Test scale correction from height."""
        self.sensor.start()
        self.sensor._pose.z = 2.0  # Estimated 2m

        self.sensor.correct_scale_from_height(3.0)  # Actual 3m

        assert self.sensor._scale_estimate != 1.0
        assert abs(self.sensor._pose.z - 3.0) < 0.1

    def test_to_dict(self):
        """Test serialization."""
        self.sensor.start()
        self.sensor.update()

        d = self.sensor.to_dict()
        assert d['type'] == 'visual_odometry'
        assert 'state' in d
        assert 'pose' in d
        assert 'quality' in d
        assert 'features_tracked' in d


class TestVisualOdometryIntegration:
    """Integration tests for visual odometry."""

    def test_continuous_tracking(self):
        """Test continuous tracking over time."""
        sensor = VisualOdometrySensor()
        sensor.start()
        sensor.set_true_velocity(0.5, 0.0, 0.0)

        positions = []
        for _ in range(20):
            reading = sensor.update()
            if reading.is_valid():
                positions.append(reading.pose.x)
            time.sleep(0.02)

        # Position should increase over time
        if len(positions) > 5:
            assert positions[-1] > positions[0]

    def test_hover_detection(self):
        """Test that hover (zero velocity) is detected."""
        sensor = VisualOdometrySensor()
        sensor.start()
        sensor.set_true_velocity(0.0, 0.0, 0.0)

        speeds = []
        for _ in range(10):
            reading = sensor.update()
            speeds.append(reading.pose.get_speed())
            time.sleep(0.02)

        # Average speed should be near zero (allow for simulation noise)
        avg_speed = sum(speeds) / len(speeds)
        assert avg_speed < 1.0

    def test_rotation_tracking(self):
        """Test rotation is tracked."""
        sensor = VisualOdometrySensor()
        sensor.start()
        sensor.set_true_attitude(0.0, 0.0, 0.5)  # Yaw rate

        for _ in range(10):
            sensor.update()
            time.sleep(0.02)

        pose = sensor.get_pose()
        assert pose.yaw != 0.0

    def test_stereo_mode(self):
        """Test stereo mode has proper scale."""
        sensor = VisualOdometrySensor(stereo=True, baseline=0.1)
        sensor.start()
        sensor.set_true_velocity(1.0, 0.0, 0.0)

        for _ in range(10):
            sensor.update()
            time.sleep(0.02)

        # Stereo should have scale = 1.0
        assert sensor._scale_estimate == 1.0
