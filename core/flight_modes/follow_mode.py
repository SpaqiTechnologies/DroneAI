"""
Follow-Me Flight Mode

Autonomous subject tracking mode that maintains position relative to
a moving target (person, vehicle, or beacon). Features motion prediction,
smooth pursuit, and automatic obstacle avoidance.

Comparable to Skydio Shadow or DJI ActiveTrack.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any, Callable
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Enums and Data Structures
# ---------------------------------------------------------------------------

class FollowState(Enum):
    """Follow mode operational state."""
    IDLE = "idle"
    ACQUIRING = "acquiring"        # Searching for target
    TRACKING = "tracking"          # Actively following
    LOST = "lost"                  # Target lost, searching
    REACQUIRING = "reacquiring"    # Attempting to reacquire
    PAUSED = "paused"              # Tracking paused by user
    AVOIDING = "avoiding"          # Avoiding obstacle


class FollowPosition(Enum):
    """Relative position to maintain."""
    BEHIND = "behind"              # Follow from behind
    FRONT = "front"                # Lead from front
    LEFT = "left"                  # Follow from left side
    RIGHT = "right"                # Follow from right side
    ORBIT = "orbit"                # Orbit around target
    CUSTOM = "custom"              # Custom angle


class TargetType(Enum):
    """Type of tracking target."""
    PERSON = "person"
    VEHICLE = "vehicle"
    BEACON = "beacon"              # GPS/RF beacon
    VISUAL_MARKER = "visual_marker"
    THERMAL = "thermal"


@dataclass
class Target:
    """Tracking target information."""
    target_type: TargetType = TargetType.PERSON

    # Position (NED frame)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Velocity
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Heading (radians)
    heading: float = 0.0

    # Visual tracking info
    bbox_x: float = 0.0           # Bounding box center (normalized 0-1)
    bbox_y: float = 0.0
    bbox_width: float = 0.0
    bbox_height: float = 0.0

    # Tracking quality
    confidence: float = 0.0       # 0-1
    track_id: int = 0             # Persistent track ID
    last_seen: float = 0.0        # Timestamp

    def get_position(self) -> Tuple[float, float, float]:
        """Get position as tuple."""
        return (self.x, self.y, self.z)

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get velocity as tuple."""
        return (self.vx, self.vy, self.vz)

    def get_speed(self) -> float:
        """Get speed magnitude."""
        return math.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)

    def predict_position(self, dt: float) -> Tuple[float, float, float]:
        """Predict position after dt seconds."""
        return (
            self.x + self.vx * dt,
            self.y + self.vy * dt,
            self.z + self.vz * dt,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.target_type.value,
            'position': {'x': self.x, 'y': self.y, 'z': self.z},
            'velocity': {'vx': self.vx, 'vy': self.vy, 'vz': self.vz},
            'heading': self.heading,
            'confidence': self.confidence,
            'track_id': self.track_id,
        }


@dataclass
class FollowConfig:
    """Configuration for follow mode."""
    # Distance parameters
    follow_distance: float = 10.0      # meters behind/ahead
    follow_altitude: float = 15.0      # meters above target
    min_distance: float = 5.0          # minimum distance
    max_distance: float = 50.0         # maximum distance

    # Speed limits
    max_speed: float = 15.0            # m/s
    max_vertical_speed: float = 3.0    # m/s
    max_acceleration: float = 5.0      # m/s^2

    # Tracking parameters
    lead_time: float = 1.0             # seconds to predict ahead
    smoothing_factor: float = 0.3      # velocity smoothing (0-1)
    reacquire_timeout: float = 10.0    # seconds before giving up

    # Position
    follow_position: FollowPosition = FollowPosition.BEHIND
    custom_angle: float = 0.0          # degrees (for CUSTOM position)

    # Features
    auto_altitude: bool = True         # Adjust altitude based on terrain
    obstacle_avoidance: bool = True    # Avoid obstacles while following
    gimbal_tracking: bool = True       # Keep gimbal pointed at target

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'follow_distance': self.follow_distance,
            'follow_altitude': self.follow_altitude,
            'min_distance': self.min_distance,
            'max_distance': self.max_distance,
            'max_speed': self.max_speed,
            'follow_position': self.follow_position.value,
            'lead_time': self.lead_time,
        }


@dataclass
class FollowCommand:
    """Command output from follow controller."""
    # Velocity command (NED)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Yaw rate command
    yaw_rate: float = 0.0

    # Gimbal command (optional)
    gimbal_pitch: Optional[float] = None
    gimbal_yaw: Optional[float] = None

    # Status
    valid: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'velocity': {'vx': self.vx, 'vy': self.vy, 'vz': self.vz},
            'yaw_rate': self.yaw_rate,
            'gimbal': {
                'pitch': self.gimbal_pitch,
                'yaw': self.gimbal_yaw,
            } if self.gimbal_pitch is not None else None,
            'valid': self.valid,
        }


# ---------------------------------------------------------------------------
# Motion Predictor
# ---------------------------------------------------------------------------

class MotionPredictor:
    """
    Predicts target motion using Kalman filter-like estimation.

    Features:
    - Velocity estimation from position history
    - Motion model prediction
    - Confidence estimation
    """

    def __init__(self, history_size: int = 10):
        """Initialize predictor."""
        self._history: List[Tuple[float, float, float, float]] = []  # (t, x, y, z)
        self._history_size = history_size

        # Estimated state
        self._velocity = (0.0, 0.0, 0.0)
        self._acceleration = (0.0, 0.0, 0.0)

        # Process noise
        self._position_noise = 0.1
        self._velocity_noise = 0.5

    def update(self, x: float, y: float, z: float, timestamp: float) -> None:
        """Update with new position observation."""
        self._history.append((timestamp, x, y, z))

        # Keep history bounded
        if len(self._history) > self._history_size:
            self._history.pop(0)

        # Estimate velocity from recent history
        if len(self._history) >= 2:
            self._estimate_velocity()

    def _estimate_velocity(self) -> None:
        """Estimate velocity from position history."""
        if len(self._history) < 2:
            return

        # Use last few points for velocity estimation
        n = min(5, len(self._history))
        recent = self._history[-n:]

        # Linear regression for velocity
        sum_dt = 0.0
        sum_dx = 0.0
        sum_dy = 0.0
        sum_dz = 0.0

        t0, x0, y0, z0 = recent[0]
        for t, x, y, z in recent[1:]:
            dt = t - t0
            if dt > 0:
                sum_dt += dt
                sum_dx += (x - x0)
                sum_dy += (y - y0)
                sum_dz += (z - z0)
                t0, x0, y0, z0 = t, x, y, z

        if sum_dt > 0:
            vx = sum_dx / sum_dt
            vy = sum_dy / sum_dt
            vz = sum_dz / sum_dt

            # Smooth with previous estimate
            alpha = 0.3
            self._velocity = (
                alpha * vx + (1 - alpha) * self._velocity[0],
                alpha * vy + (1 - alpha) * self._velocity[1],
                alpha * vz + (1 - alpha) * self._velocity[2],
            )

    def predict(self, dt: float) -> Tuple[float, float, float]:
        """Predict position after dt seconds."""
        if not self._history:
            return (0.0, 0.0, 0.0)

        _, x, y, z = self._history[-1]

        # Constant velocity model
        px = x + self._velocity[0] * dt
        py = y + self._velocity[1] * dt
        pz = z + self._velocity[2] * dt

        return (px, py, pz)

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get estimated velocity."""
        return self._velocity

    def reset(self) -> None:
        """Reset predictor state."""
        self._history.clear()
        self._velocity = (0.0, 0.0, 0.0)
        self._acceleration = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Follow Controller
# ---------------------------------------------------------------------------

class FollowController:
    """
    Main controller for follow-me mode.

    Generates velocity commands to maintain desired position
    relative to a moving target.
    """

    def __init__(self, config: Optional[FollowConfig] = None):
        """Initialize controller."""
        self.config = config or FollowConfig()
        self._predictor = MotionPredictor()

        # State
        self._state = FollowState.IDLE
        self._target: Optional[Target] = None
        self._last_update = 0.0
        self._track_start_time = 0.0
        self._lost_time = 0.0

        # Drone state
        self._drone_position = (0.0, 0.0, 0.0)
        self._drone_velocity = (0.0, 0.0, 0.0)
        self._drone_heading = 0.0

        # Smoothed command
        self._last_command = FollowCommand()

        # Callbacks
        self._state_callbacks: List[Callable[[FollowState], None]] = []
        self._target_callbacks: List[Callable[[Target], None]] = []

        # Statistics
        self._stats = {
            'track_time': 0.0,
            'lost_count': 0,
            'distance_traveled': 0.0,
        }

    @property
    def state(self) -> FollowState:
        """Get current state."""
        return self._state

    @property
    def target(self) -> Optional[Target]:
        """Get current target."""
        return self._target

    def start(self, target_type: TargetType = TargetType.PERSON) -> None:
        """Start follow mode."""
        self._set_state(FollowState.ACQUIRING)
        self._predictor.reset()
        self._track_start_time = time.time()

    def stop(self) -> None:
        """Stop follow mode."""
        self._set_state(FollowState.IDLE)
        self._target = None
        self._predictor.reset()

    def pause(self) -> None:
        """Pause tracking."""
        if self._state == FollowState.TRACKING:
            self._set_state(FollowState.PAUSED)

    def resume(self) -> None:
        """Resume tracking."""
        if self._state == FollowState.PAUSED:
            self._set_state(FollowState.TRACKING)

    def set_target(self, target: Target) -> None:
        """Set or update target."""
        self._target = target
        self._predictor.update(
            target.x, target.y, target.z,
            time.time(),
        )

        if self._state in [FollowState.ACQUIRING, FollowState.LOST, FollowState.REACQUIRING]:
            self._set_state(FollowState.TRACKING)

        # Notify callbacks
        for callback in self._target_callbacks:
            try:
                callback(target)
            except Exception:
                pass

    def update_drone_state(
        self,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
        heading: float,
    ) -> None:
        """Update drone state."""
        self._drone_position = position
        self._drone_velocity = velocity
        self._drone_heading = heading

    def compute(self, dt: float = 0.1) -> FollowCommand:
        """
        Compute follow command.

        Args:
            dt: Time step

        Returns:
            FollowCommand with velocity commands
        """
        if self._state == FollowState.IDLE:
            return FollowCommand(valid=False, reason="Follow mode not active")

        if self._state == FollowState.PAUSED:
            return FollowCommand(vx=0, vy=0, vz=0, valid=True, reason="Paused")

        if self._target is None or self._state == FollowState.ACQUIRING:
            return FollowCommand(valid=False, reason="No target")

        # Check target freshness
        target_age = time.time() - self._target.last_seen
        if target_age > 2.0:  # Target not seen for 2 seconds
            if self._state != FollowState.LOST:
                self._set_state(FollowState.LOST)
                self._lost_time = time.time()
                self._stats['lost_count'] += 1

            # Check reacquire timeout
            if time.time() - self._lost_time > self.config.reacquire_timeout:
                self._set_state(FollowState.IDLE)
                return FollowCommand(valid=False, reason="Target lost timeout")

        # Predict target position
        predicted_pos = self._predictor.predict(self.config.lead_time)
        predicted_vel = self._predictor.get_velocity()

        # Calculate desired drone position
        desired_pos = self._calculate_desired_position(
            predicted_pos,
            predicted_vel,
        )

        # Calculate velocity command
        command = self._calculate_velocity_command(desired_pos, dt)

        # Calculate gimbal command if enabled
        if self.config.gimbal_tracking:
            command.gimbal_pitch, command.gimbal_yaw = self._calculate_gimbal_angles()

        # Smooth command
        command = self._smooth_command(command, dt)

        self._last_command = command
        self._last_update = time.time()

        # Update statistics
        self._stats['track_time'] += dt

        return command

    def _calculate_desired_position(
        self,
        target_pos: Tuple[float, float, float],
        target_vel: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Calculate desired drone position relative to target."""
        tx, ty, tz = target_pos
        tvx, tvy, _ = target_vel

        # Calculate target heading from velocity
        if abs(tvx) > 0.1 or abs(tvy) > 0.1:
            target_heading = math.atan2(tvy, tvx)
        else:
            target_heading = self._target.heading if self._target else 0

        # Calculate offset based on follow position
        if self.config.follow_position == FollowPosition.BEHIND:
            offset_angle = target_heading + math.pi  # Behind
        elif self.config.follow_position == FollowPosition.FRONT:
            offset_angle = target_heading  # In front
        elif self.config.follow_position == FollowPosition.LEFT:
            offset_angle = target_heading + math.pi / 2
        elif self.config.follow_position == FollowPosition.RIGHT:
            offset_angle = target_heading - math.pi / 2
        elif self.config.follow_position == FollowPosition.CUSTOM:
            offset_angle = target_heading + math.radians(self.config.custom_angle)
        else:
            offset_angle = target_heading + math.pi

        # Calculate offset position
        dx = math.cos(offset_angle) * self.config.follow_distance
        dy = math.sin(offset_angle) * self.config.follow_distance
        dz = -self.config.follow_altitude  # NED: negative = up

        return (tx + dx, ty + dy, tz + dz)

    def _calculate_velocity_command(
        self,
        desired_pos: Tuple[float, float, float],
        dt: float,
    ) -> FollowCommand:
        """Calculate velocity command to reach desired position."""
        dx, dy, dz = self._drone_position

        # Position error
        ex = desired_pos[0] - dx
        ey = desired_pos[1] - dy
        ez = desired_pos[2] - dz

        # Distance to desired position
        distance = math.sqrt(ex ** 2 + ey ** 2 + ez ** 2)

        # P controller with speed limit
        kp = 1.0  # Proportional gain

        # Calculate raw velocity
        vx = kp * ex
        vy = kp * ey
        vz = kp * ez

        # Limit horizontal speed
        h_speed = math.sqrt(vx ** 2 + vy ** 2)
        if h_speed > self.config.max_speed:
            scale = self.config.max_speed / h_speed
            vx *= scale
            vy *= scale

        # Limit vertical speed
        if abs(vz) > self.config.max_vertical_speed:
            vz = math.copysign(self.config.max_vertical_speed, vz)

        # Calculate yaw rate to face target
        target_yaw = math.atan2(
            self._target.y - dy if self._target else 0,
            self._target.x - dx if self._target else 0,
        )
        yaw_error = self._normalize_angle(target_yaw - self._drone_heading)
        yaw_rate = 2.0 * yaw_error  # Simple P control

        return FollowCommand(
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rate=yaw_rate,
            valid=True,
        )

    def _calculate_gimbal_angles(self) -> Tuple[float, float]:
        """Calculate gimbal angles to point at target."""
        if self._target is None:
            return (0.0, 0.0)

        dx, dy, dz = self._drone_position
        tx, ty, tz = self._target.x, self._target.y, self._target.z

        # Vector to target
        vx = tx - dx
        vy = ty - dy
        vz = tz - dz

        # Horizontal distance
        h_dist = math.sqrt(vx ** 2 + vy ** 2)

        # Pitch (looking down at target)
        pitch = math.atan2(-vz, h_dist)  # Negative because NED

        # Yaw (relative to drone heading)
        absolute_yaw = math.atan2(vy, vx)
        yaw = self._normalize_angle(absolute_yaw - self._drone_heading)

        return (math.degrees(pitch), math.degrees(yaw))

    def _smooth_command(self, command: FollowCommand, dt: float) -> FollowCommand:
        """Smooth command with acceleration limiting."""
        alpha = self.config.smoothing_factor

        # Smooth velocity
        vx = alpha * command.vx + (1 - alpha) * self._last_command.vx
        vy = alpha * command.vy + (1 - alpha) * self._last_command.vy
        vz = alpha * command.vz + (1 - alpha) * self._last_command.vz

        # Limit acceleration
        dvx = vx - self._last_command.vx
        dvy = vy - self._last_command.vy
        accel = math.sqrt(dvx ** 2 + dvy ** 2) / dt if dt > 0 else 0

        if accel > self.config.max_acceleration:
            scale = self.config.max_acceleration * dt / math.sqrt(dvx ** 2 + dvy ** 2)
            vx = self._last_command.vx + dvx * scale
            vy = self._last_command.vy + dvy * scale

        return FollowCommand(
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rate=alpha * command.yaw_rate + (1 - alpha) * self._last_command.yaw_rate,
            gimbal_pitch=command.gimbal_pitch,
            gimbal_yaw=command.gimbal_yaw,
            valid=command.valid,
            reason=command.reason,
        )

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _set_state(self, new_state: FollowState) -> None:
        """Set state and notify callbacks."""
        if self._state != new_state:
            self._state = new_state
            for callback in self._state_callbacks:
                try:
                    callback(new_state)
                except Exception:
                    pass

    def register_state_callback(
        self,
        callback: Callable[[FollowState], None],
    ) -> None:
        """Register state change callback."""
        self._state_callbacks.append(callback)

    def register_target_callback(
        self,
        callback: Callable[[Target], None],
    ) -> None:
        """Register target update callback."""
        self._target_callbacks.append(callback)

    def set_follow_distance(self, distance: float) -> None:
        """Set follow distance."""
        self.config.follow_distance = max(
            self.config.min_distance,
            min(self.config.max_distance, distance),
        )

    def set_follow_altitude(self, altitude: float) -> None:
        """Set follow altitude."""
        self.config.follow_altitude = max(5.0, min(120.0, altitude))

    def set_follow_position(
        self,
        position: FollowPosition,
        custom_angle: float = 0.0,
    ) -> None:
        """Set follow position."""
        self.config.follow_position = position
        if position == FollowPosition.CUSTOM:
            self.config.custom_angle = custom_angle

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            **self._stats,
            'state': self._state.value,
            'target_confidence': self._target.confidence if self._target else 0,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            'state': self._state.value,
            'target': self._target.to_dict() if self._target else None,
            'config': self.config.to_dict(),
            'statistics': self._stats,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.get_status()


# ---------------------------------------------------------------------------
# Follow Mode Handler (integrates with flight mode system)
# ---------------------------------------------------------------------------

class FollowModeHandler:
    """
    Flight mode handler for follow mode.

    Integrates with the main flight controller and detection systems.
    """

    def __init__(self):
        """Initialize handler."""
        self._controller = FollowController()
        self._active = False
        self._detector_callback: Optional[Callable] = None

    def activate(self, target_type: TargetType = TargetType.PERSON) -> bool:
        """Activate follow mode."""
        if self._active:
            return True

        self._controller.start(target_type)
        self._active = True
        return True

    def deactivate(self) -> None:
        """Deactivate follow mode."""
        self._controller.stop()
        self._active = False

    @property
    def is_active(self) -> bool:
        """Check if mode is active."""
        return self._active

    def update(
        self,
        drone_position: Tuple[float, float, float],
        drone_velocity: Tuple[float, float, float],
        drone_heading: float,
        dt: float = 0.1,
    ) -> FollowCommand:
        """
        Update follow mode and get command.

        Args:
            drone_position: Current drone position (NED)
            drone_velocity: Current drone velocity (NED)
            drone_heading: Current drone heading (radians)
            dt: Time step

        Returns:
            FollowCommand with velocity commands
        """
        if not self._active:
            return FollowCommand(valid=False, reason="Mode not active")

        self._controller.update_drone_state(
            drone_position,
            drone_velocity,
            drone_heading,
        )

        return self._controller.compute(dt)

    def process_detection(self, detection: Dict[str, Any]) -> None:
        """
        Process detection from camera/AI.

        Args:
            detection: Detection dict with position, confidence, etc.
        """
        if not self._active:
            return

        target = Target(
            target_type=TargetType(detection.get('type', 'person')),
            x=detection.get('x', 0),
            y=detection.get('y', 0),
            z=detection.get('z', 0),
            vx=detection.get('vx', 0),
            vy=detection.get('vy', 0),
            vz=detection.get('vz', 0),
            confidence=detection.get('confidence', 0),
            track_id=detection.get('track_id', 0),
            last_seen=time.time(),
        )

        self._controller.set_target(target)

    def configure(self, **kwargs) -> None:
        """Configure follow mode."""
        config = self._controller.config

        if 'distance' in kwargs:
            config.follow_distance = kwargs['distance']
        if 'altitude' in kwargs:
            config.follow_altitude = kwargs['altitude']
        if 'position' in kwargs:
            config.follow_position = FollowPosition(kwargs['position'])
        if 'max_speed' in kwargs:
            config.max_speed = kwargs['max_speed']

    def get_state(self) -> FollowState:
        """Get current state."""
        return self._controller.state

    def get_status(self) -> Dict[str, Any]:
        """Get mode status."""
        return {
            'active': self._active,
            **self._controller.get_status(),
        }
