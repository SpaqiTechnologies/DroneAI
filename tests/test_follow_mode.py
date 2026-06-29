"""
Tests for Follow-Me flight mode.
"""

import pytest
import time
import math
from core.flight_modes.follow_mode import (
    FollowState,
    FollowPosition,
    TargetType,
    Target,
    FollowConfig,
    FollowCommand,
    MotionPredictor,
    FollowController,
    FollowModeHandler,
)


class TestTarget:
    """Tests for Target class."""

    def test_creation(self):
        target = Target(
            target_type=TargetType.PERSON,
            x=10.0,
            y=20.0,
            z=0.0,
            confidence=0.95,
        )

        assert target.target_type == TargetType.PERSON
        assert target.x == 10.0
        assert target.confidence == 0.95

    def test_get_position(self):
        target = Target(x=1, y=2, z=3)
        assert target.get_position() == (1, 2, 3)

    def test_get_velocity(self):
        target = Target(vx=1, vy=2, vz=3)
        assert target.get_velocity() == (1, 2, 3)

    def test_get_speed(self):
        target = Target(vx=3, vy=4, vz=0)
        assert abs(target.get_speed() - 5.0) < 0.001

    def test_predict_position(self):
        target = Target(x=0, y=0, z=0, vx=10, vy=0, vz=0)
        predicted = target.predict_position(1.0)
        assert predicted == (10, 0, 0)

    def test_to_dict(self):
        target = Target(
            target_type=TargetType.VEHICLE,
            x=5,
            y=10,
            z=-2,
        )
        d = target.to_dict()

        assert d['type'] == 'vehicle'
        assert d['position']['x'] == 5


class TestFollowConfig:
    """Tests for FollowConfig."""

    def test_defaults(self):
        config = FollowConfig()

        assert config.follow_distance == 10.0
        assert config.follow_altitude == 15.0
        assert config.max_speed == 15.0
        assert config.follow_position == FollowPosition.BEHIND

    def test_custom_config(self):
        config = FollowConfig(
            follow_distance=20.0,
            follow_position=FollowPosition.LEFT,
            max_speed=10.0,
        )

        assert config.follow_distance == 20.0
        assert config.follow_position == FollowPosition.LEFT

    def test_to_dict(self):
        config = FollowConfig()
        d = config.to_dict()

        assert 'follow_distance' in d
        assert 'follow_position' in d


class TestMotionPredictor:
    """Tests for MotionPredictor."""

    def test_creation(self):
        predictor = MotionPredictor()
        assert predictor.get_velocity() == (0, 0, 0)

    def test_update(self):
        predictor = MotionPredictor()

        predictor.update(0, 0, 0, 0)
        predictor.update(1, 0, 0, 0.1)

        vel = predictor.get_velocity()
        assert vel[0] > 0  # Moving in x direction

    def test_predict(self):
        predictor = MotionPredictor()

        predictor.update(0, 0, 0, 0)
        predictor.update(10, 0, 0, 1.0)

        predicted = predictor.predict(1.0)
        assert predicted[0] > 10  # Should predict ahead

    def test_reset(self):
        predictor = MotionPredictor()

        predictor.update(10, 10, 10, 0)
        predictor.reset()

        assert predictor.get_velocity() == (0, 0, 0)


class TestFollowController:
    """Tests for FollowController."""

    def test_initial_state(self):
        controller = FollowController()

        assert controller.state == FollowState.IDLE
        assert controller.target is None

    def test_start(self):
        controller = FollowController()
        controller.start()

        assert controller.state == FollowState.ACQUIRING

    def test_stop(self):
        controller = FollowController()
        controller.start()
        controller.stop()

        assert controller.state == FollowState.IDLE

    def test_pause_resume(self):
        controller = FollowController()
        controller.start()

        # Set target to enable tracking
        target = Target(
            x=10, y=0, z=0,
            confidence=0.9,
            last_seen=time.time(),
        )
        controller.set_target(target)

        controller.pause()
        assert controller.state == FollowState.PAUSED

        controller.resume()
        assert controller.state == FollowState.TRACKING

    def test_set_target(self):
        controller = FollowController()
        controller.start()

        target = Target(
            x=10, y=20, z=0,
            confidence=0.95,
            last_seen=time.time(),
        )
        controller.set_target(target)

        assert controller.target is not None
        assert controller.state == FollowState.TRACKING

    def test_compute_no_target(self):
        controller = FollowController()
        controller.start()

        command = controller.compute()

        assert not command.valid
        assert "No target" in command.reason

    def test_compute_with_target(self):
        controller = FollowController()
        controller.start()

        # Set target and drone state
        target = Target(
            x=10, y=0, z=0,
            vx=0, vy=0, vz=0,
            confidence=0.9,
            last_seen=time.time(),
        )
        controller.set_target(target)
        controller.update_drone_state((0, 0, -15), (0, 0, 0), 0)

        command = controller.compute()

        assert command.valid
        # Should command movement toward target offset position
        assert command.vx != 0 or command.vy != 0 or command.vz != 0

    def test_set_follow_distance(self):
        controller = FollowController()

        controller.set_follow_distance(25.0)
        assert controller.config.follow_distance == 25.0

        # Test clamping
        controller.set_follow_distance(1.0)
        assert controller.config.follow_distance == controller.config.min_distance

    def test_set_follow_position(self):
        controller = FollowController()

        controller.set_follow_position(FollowPosition.LEFT)
        assert controller.config.follow_position == FollowPosition.LEFT

        controller.set_follow_position(FollowPosition.CUSTOM, custom_angle=45.0)
        assert controller.config.follow_position == FollowPosition.CUSTOM
        assert controller.config.custom_angle == 45.0

    def test_state_callback(self):
        controller = FollowController()
        states_received = []

        controller.register_state_callback(lambda s: states_received.append(s))

        controller.start()

        assert FollowState.ACQUIRING in states_received

    def test_get_statistics(self):
        controller = FollowController()
        stats = controller.get_statistics()

        assert 'track_time' in stats
        assert 'lost_count' in stats

    def test_get_status(self):
        controller = FollowController()
        status = controller.get_status()

        assert 'state' in status
        assert 'config' in status


class TestFollowModeHandler:
    """Tests for FollowModeHandler."""

    def test_activate(self):
        handler = FollowModeHandler()

        result = handler.activate()

        assert result is True
        assert handler.is_active

    def test_deactivate(self):
        handler = FollowModeHandler()
        handler.activate()
        handler.deactivate()

        assert not handler.is_active

    def test_update_inactive(self):
        handler = FollowModeHandler()

        command = handler.update((0, 0, 0), (0, 0, 0), 0)

        assert not command.valid

    def test_process_detection(self):
        handler = FollowModeHandler()
        handler.activate()

        detection = {
            'type': 'person',
            'x': 10,
            'y': 5,
            'z': 0,
            'confidence': 0.9,
        }

        handler.process_detection(detection)

        assert handler.get_state() == FollowState.TRACKING

    def test_configure(self):
        handler = FollowModeHandler()

        handler.configure(
            distance=25.0,
            altitude=20.0,
            max_speed=12.0,
        )

        status = handler.get_status()
        assert status['config']['follow_distance'] == 25.0

    def test_get_status(self):
        handler = FollowModeHandler()
        status = handler.get_status()

        assert 'active' in status
        assert 'state' in status


class TestFollowCommand:
    """Tests for FollowCommand."""

    def test_creation(self):
        command = FollowCommand(vx=1, vy=2, vz=3)

        assert command.vx == 1
        assert command.vy == 2
        assert command.vz == 3
        assert command.valid is True

    def test_to_dict(self):
        command = FollowCommand(
            vx=1, vy=2, vz=3,
            yaw_rate=0.5,
            gimbal_pitch=-30,
            gimbal_yaw=10,
        )

        d = command.to_dict()

        assert d['velocity']['vx'] == 1
        assert d['yaw_rate'] == 0.5
        assert d['gimbal']['pitch'] == -30
