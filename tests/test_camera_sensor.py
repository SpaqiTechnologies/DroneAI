"""
Tests for the camera sensor module.
"""

import pytest
import time
from sensors.camera_sensor import (
    CameraSensor,
    VisionMode,
    CameraState,
    DetectionType,
    BoundingBox,
    Detection,
    Frame,
    GimbalPosition,
)


class TestCameraSensor:
    """Tests for the CameraSensor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.camera = CameraSensor()

    def teardown_method(self):
        """Cleanup after tests."""
        if self.camera._state != CameraState.OFF:
            self.camera.stop()

    def test_initial_state(self):
        """Test camera initial state."""
        assert self.camera._state == CameraState.OFF
        assert self.camera._vision_mode == VisionMode.NORMAL
        assert not self.camera._streaming
        assert not self.camera._is_recording

    def test_start_stop(self):
        """Test starting and stopping the camera."""
        success, message = self.camera.start()
        assert success
        assert self.camera._state == CameraState.READY

        success, message = self.camera.stop()
        assert success
        assert self.camera._state == CameraState.OFF

    def test_cannot_start_twice(self):
        """Test that starting twice returns error."""
        self.camera.start()
        success, message = self.camera.start()
        assert not success
        assert "already started" in message.lower()

    def test_capture_frame(self):
        """Test frame capture."""
        self.camera.start()
        frame = self.camera.capture_frame()

        assert frame is not None
        assert frame.width == 1280
        assert frame.height == 720
        assert frame.mode == VisionMode.NORMAL

    def test_capture_frame_when_off(self):
        """Test that capture fails when camera is off."""
        frame = self.camera.capture_frame()
        assert frame is None

    def test_streaming(self):
        """Test streaming functionality."""
        self.camera.start()

        success, message = self.camera.start_streaming()
        assert success
        assert self.camera._streaming
        assert self.camera._state == CameraState.STREAMING

        success, message = self.camera.stop_streaming()
        assert success
        assert not self.camera._streaming

    def test_streaming_requires_camera_on(self):
        """Test that streaming requires camera to be started."""
        success, message = self.camera.start_streaming()
        assert not success

    def test_vision_modes(self):
        """Test setting vision modes."""
        self.camera.start()

        for mode in VisionMode:
            success, message = self.camera.set_vision_mode(mode)
            assert success
            assert self.camera._vision_mode == mode

    def test_gimbal_position(self):
        """Test gimbal position control."""
        self.camera.start()

        success, message = self.camera.set_gimbal_position(pitch=-45, roll=10, yaw=90)
        assert success
        assert self.camera._gimbal.pitch == -45
        assert self.camera._gimbal.roll == 10
        assert self.camera._gimbal.yaw == 90

    def test_gimbal_limits(self):
        """Test gimbal position limits."""
        self.camera.start()

        # Test pitch limits (-90 to +30)
        self.camera.set_gimbal_position(pitch=-100)
        assert self.camera._gimbal.pitch == -90  # Should clamp to -90

        self.camera.set_gimbal_position(pitch=50)
        assert self.camera._gimbal.pitch == 30  # Should clamp to +30

        # Test roll limits (-45 to +45)
        self.camera.set_gimbal_position(roll=-60)
        assert self.camera._gimbal.roll == -45

        self.camera.set_gimbal_position(roll=60)
        assert self.camera._gimbal.roll == 45

        # Test yaw limits (-180 to +180)
        self.camera.set_gimbal_position(yaw=-200)
        assert self.camera._gimbal.yaw == -180

        self.camera.set_gimbal_position(yaw=200)
        assert self.camera._gimbal.yaw == 180

    def test_point_down(self):
        """Test pointing camera down."""
        self.camera.start()
        success, message = self.camera.point_down()
        assert success
        assert self.camera._gimbal.pitch == -90
        assert self.camera._gimbal.roll == 0
        assert self.camera._gimbal.yaw == 0

    def test_point_forward(self):
        """Test pointing camera forward."""
        self.camera.start()
        self.camera.set_gimbal_position(pitch=-45, roll=10, yaw=90)

        success, message = self.camera.point_forward()
        assert success
        assert self.camera._gimbal.pitch == 0
        assert self.camera._gimbal.roll == 0
        assert self.camera._gimbal.yaw == 0

    def test_recording(self):
        """Test video recording."""
        self.camera.start()

        success, message = self.camera.start_recording()
        assert success
        assert self.camera._is_recording
        assert self.camera._recording_start_time is not None

        time.sleep(0.1)

        success, message = self.camera.stop_recording()
        assert success
        assert not self.camera._is_recording
        assert "saved" in message.lower()

    def test_snapshot(self):
        """Test taking snapshot."""
        self.camera.start()
        success, message = self.camera.take_snapshot()
        assert success
        assert "snapshot" in message.lower()

    def test_simulate_marker(self):
        """Test marker simulation."""
        self.camera.start()

        # Simulate marker
        self.camera.simulate_marker(visible=True, distance=5.0, marker_id="test_marker")

        # Capture frame to trigger detection
        frame = self.camera.capture_frame()

        assert self.camera.is_marker_detected()
        assert self.camera._marker_position is not None
        assert len(frame.detections) > 0

        marker_detection = next(
            (d for d in frame.detections if d.detection_type == DetectionType.LANDING_MARKER),
            None
        )
        assert marker_detection is not None
        assert marker_detection.distance == 5.0

    def test_simulate_marker_hidden(self):
        """Test hiding simulated marker."""
        self.camera.start()

        self.camera.simulate_marker(visible=True)
        self.camera.capture_frame()
        assert self.camera.is_marker_detected()

        self.camera.simulate_marker(visible=False)
        self.camera.capture_frame()
        assert not self.camera.is_marker_detected()

    def test_marker_offset(self):
        """Test marker offset calculation."""
        self.camera.start()

        # Simulate marker at center (no offset)
        center_x = self.camera._resolution[0] // 2 - 50
        center_y = self.camera._resolution[1] // 2 - 50
        self.camera.simulate_marker(visible=True, x=center_x, y=center_y)
        self.camera.capture_frame()

        offset = self.camera.get_marker_offset()
        assert offset is not None
        # Offset should be close to (0, 0) when marker is centered
        assert abs(offset[0]) < 0.2
        assert abs(offset[1]) < 0.2

    def test_simulate_obstacle(self):
        """Test obstacle simulation."""
        self.camera.start()

        idx = self.camera.simulate_obstacle(
            obstacle_type=DetectionType.TREE,
            x=100,
            y=100,
            width=50,
            height=100,
            distance=15.0,
        )
        assert idx == 0

        frame = self.camera.capture_frame()
        obstacles = self.camera.get_obstacles()

        assert len(obstacles) > 0
        assert obstacles[0].detection_type == DetectionType.TREE
        assert obstacles[0].distance == 15.0

    def test_closest_obstacle(self):
        """Test getting closest obstacle."""
        self.camera.start()

        # Add multiple obstacles at different distances
        self.camera.simulate_obstacle(distance=20.0)
        self.camera.simulate_obstacle(distance=5.0)
        self.camera.simulate_obstacle(distance=15.0)

        self.camera.capture_frame()
        closest = self.camera.get_closest_obstacle()

        assert closest is not None
        assert closest.distance == 5.0

    def test_clear_simulated_obstacles(self):
        """Test clearing simulated obstacles."""
        self.camera.start()

        self.camera.simulate_obstacle()
        self.camera.simulate_obstacle()
        self.camera.capture_frame()
        assert len(self.camera.get_obstacles()) == 2

        self.camera.clear_simulated_obstacles()
        self.camera.capture_frame()
        assert len(self.camera.get_obstacles()) == 0

    def test_detection_callbacks(self):
        """Test detection callbacks."""
        self.camera.start()
        detected_objects = []

        def on_detection(detection):
            detected_objects.append(detection)

        self.camera.add_detection_callback(on_detection)
        self.camera.simulate_marker(visible=True)
        self.camera.capture_frame()

        assert len(detected_objects) > 0
        assert detected_objects[0].detection_type == DetectionType.LANDING_MARKER

    def test_frame_callbacks(self):
        """Test frame callbacks."""
        self.camera.start()
        frames_received = []

        def on_frame(frame):
            frames_received.append(frame)

        self.camera.add_frame_callback(on_frame)
        self.camera.start_streaming()

        time.sleep(0.2)  # Wait for a few frames
        self.camera.stop_streaming()

        assert len(frames_received) > 0

    def test_is_valid(self):
        """Test is_valid method."""
        assert not self.camera.is_valid()  # Off

        self.camera.start()
        assert self.camera.is_valid()

        self.camera.stop()
        assert not self.camera.is_valid()

    def test_is_healthy(self):
        """Test is_healthy method."""
        self.camera.start()
        assert self.camera.is_healthy()

    def test_get_status(self):
        """Test getting camera status."""
        self.camera.start()
        status = self.camera.get_status()

        assert status['state'] == 'READY'
        assert status['vision_mode'] == 'normal'
        assert status['resolution'] == (1280, 720)
        assert status['fps'] == 30
        assert not status['is_streaming']
        assert not status['is_recording']

    def test_to_dict(self):
        """Test serialization."""
        self.camera.start()
        self.camera.simulate_marker(visible=True)
        self.camera.capture_frame()

        data = self.camera.to_dict()

        assert 'state' in data
        assert 'vision_mode' in data
        assert 'gimbal' in data
        assert 'detections' in data
        assert 'last_frame' in data


class TestBoundingBox:
    """Tests for the BoundingBox class."""

    def test_bounding_box_center(self):
        """Test bounding box center calculation."""
        box = BoundingBox(x=100, y=100, width=50, height=100)
        assert box.center == (125, 150)

    def test_bounding_box_area(self):
        """Test bounding box area calculation."""
        box = BoundingBox(x=0, y=0, width=50, height=100)
        assert box.area == 5000

    def test_bounding_box_to_dict(self):
        """Test bounding box serialization."""
        box = BoundingBox(x=10, y=20, width=30, height=40)
        data = box.to_dict()

        assert data['x'] == 10
        assert data['y'] == 20
        assert data['width'] == 30
        assert data['height'] == 40
        assert data['center'] == (25, 40)


class TestDetection:
    """Tests for the Detection class."""

    def test_detection_to_dict(self):
        """Test detection serialization."""
        box = BoundingBox(x=100, y=100, width=50, height=50)
        detection = Detection(
            detection_type=DetectionType.OBSTACLE,
            confidence=0.85,
            bounding_box=box,
            distance=10.5,
        )

        data = detection.to_dict()

        assert data['type'] == 'obstacle'
        assert data['confidence'] == 0.85
        assert data['distance'] == 10.5
        assert 'bounding_box' in data
        assert 'timestamp' in data


class TestGimbalPosition:
    """Tests for the GimbalPosition class."""

    def test_gimbal_default_values(self):
        """Test gimbal default values."""
        gimbal = GimbalPosition()
        assert gimbal.pitch == 0.0
        assert gimbal.roll == 0.0
        assert gimbal.yaw == 0.0

    def test_gimbal_to_dict(self):
        """Test gimbal serialization."""
        gimbal = GimbalPosition(pitch=-45, roll=10, yaw=90)
        data = gimbal.to_dict()

        assert data['pitch'] == -45
        assert data['roll'] == 10
        assert data['yaw'] == 90


class TestFrame:
    """Tests for the Frame class."""

    def test_frame_to_dict(self):
        """Test frame serialization."""
        frame = Frame(
            width=1280,
            height=720,
            timestamp=time.time(),
            mode=VisionMode.THERMAL,
        )

        data = frame.to_dict()

        assert data['width'] == 1280
        assert data['height'] == 720
        assert data['mode'] == 'thermal'
        assert data['has_data'] is False
        assert data['detections'] == []
