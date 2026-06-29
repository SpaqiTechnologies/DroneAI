"""
Main Drone class for managing drone operations and path planning.
Integrates all safety systems: failsafe, geofence, health monitoring, arming, and landing.
"""

import math
import time
from typing import Tuple, List, Optional
from sensors.sensor_manager import SensorManager
from sensors.wind_sensor import WindSensor
from sensors.battery_sensor import BatterySensor
from sensors.gps_sensor import GPSSensor
from sensors.camera_sensor import CameraSensor, VisionMode, CameraState, DetectionType
from core.exceptions import (
    SensorConnectionLostException,
    LowBatteryException,
    GeofenceBreachException,
    ArmingException,
)
from core.path_planner import PathPlanner, Position, State
from core.flight_logger import FlightLogger, EventType, LogLevel
from core.failsafe import FailsafeManager, FailsafeAction, FailsafeType
from core.geofence import GeofenceManager, GeofenceAction, GeofenceZone
from core.health_monitor import HealthMonitor, HealthStatus, ComponentType
from core.arming import ArmingManager, ArmingState, PreArmCheck
from core.landing import LandingManager, LandingMode, LandingState
from core.state_estimator import StateEstimator, NavigationMode, NavigationState
from core.lidar_slam import LiDARSLAM, SLAMState, SLAMQuality
from core.vio_fusion import VIOFusion, VIOState, VIOHealth
from core.flight_controller import FlightController
from core.anomaly_failsafe import AnomalyFailsafeBridge
from ai.anomaly.anomaly_detector import AnomalyDetector, AnomalySeverity
from core.flight_modes import (
    FlightMode, FlightCommand,
    StabilizeModeHandler, LoiterModeHandler, GuidedModeHandler,
    RTLModeHandler, AutoModeHandler, Mission, Waypoint
)
from sensors.imu_sensor import IMUSensor
from sensors.optical_flow_sensor import OpticalFlowSensor
from sensors.lidar_sensor import LiDARSensor, LiDARState
from sensors.visual_odometry import VisualOdometrySensor, VOState, VOQuality
from core.communication.mavlink_backend import MAVLinkBackend


class Drone:
    """
    Main drone class that manages flight operations and path planning.
    Integrates GPS tracking, flight logging, failsafe, geofence, health monitoring,
    arming, and landing systems.
    """

    def __init__(self, enable_logging: bool = True):
        self.sensor_manager = SensorManager()
        self.wind_sensor = WindSensor()
        self.battery_sensor = BatterySensor()
        self.gps_sensor = GPSSensor()
        self.camera_sensor = CameraSensor()
        self.current_position = (0.0, 0.0)  # (latitude, longitude)
        self.current_altitude = 0.0  # meters
        self.home_position = (0.0, 0.0)  # (latitude, longitude)
        self.battery_level = 100.0  # Percentage
        self._battery_level_override = None  # For testing purposes

        # Flight logger
        self._enable_logging = enable_logging
        self.flight_logger = FlightLogger() if enable_logging else None

        # Safety systems
        self.failsafe_manager = FailsafeManager()
        self.geofence_manager = GeofenceManager()
        self.health_monitor = HealthMonitor()
        self.arming_manager = ArmingManager()
        self.landing_manager = LandingManager()

        # Navigation state estimator (GPS + IMU fusion)
        self.state_estimator = StateEstimator()
        self.imu_sensor = IMUSensor()
        self.optical_flow_sensor = OpticalFlowSensor()
        self._navigation_mode = NavigationMode.FAILED

        # LiDAR sensor and SLAM (for GPS-denied navigation)
        self.lidar_sensor = LiDARSensor()
        self.lidar_slam = LiDARSLAM()
        self._lidar_slam_enabled = False  # Disabled by default, enable when needed

        # Visual Odometry and VIO (Visual-Inertial Odometry) for indoor/GPS-denied
        self.visual_odometry = VisualOdometrySensor()
        self.vio_fusion = VIOFusion()
        self._vio_enabled = False  # Disabled by default, enable for indoor flight

        # Flight Controller (autonomous navigation)
        self.flight_controller = FlightController(self)
        self._setup_flight_controller()

        # Anomaly detection wired into failsafe: critical/emergency
        # anomalies trigger the matching FailsafeType automatically.
        self.anomaly_detector = AnomalyDetector()
        self.anomaly_failsafe_bridge = AnomalyFailsafeBridge(
            detector=self.anomaly_detector,
            failsafe=self.failsafe_manager,
            min_severity=AnomalySeverity.CRITICAL,
        )
        self.anomaly_failsafe_bridge.attach()

        # Setup callbacks for all safety systems
        self._setup_failsafe_callbacks()
        self._setup_geofence_callbacks()
        self._setup_arming_callbacks()
        self._setup_landing_callbacks()
        self._setup_camera_callbacks()
        self._setup_navigation_callbacks()
        self._setup_lidar_callbacks()
        self._setup_vio_callbacks()

        # Control signal tracking
        self._last_command_time = time.time()
        self._signal_strength = 1.0

        # MAVLink backend (None = simulation only; set via connect_mavlink()
        # to draw position/attitude/battery from a real autopilot or SITL).
        self.mavlink_backend: Optional[MAVLinkBackend] = None
        # Heartbeats older than this are treated as stale and the fake
        # sensors stay authoritative until the link recovers.
        self._mavlink_heartbeat_max_age_s = 3.0

        # Flight state
        self._is_flying = False
        self._is_returning_home = False
        self._is_landing = False

        # Sensor health data (simulated for now)
        self._compass_healthy = True
        self._compass_calibrated = True
        self._accel_calibrated = True
        self._gyro_calibrated = True
        self._baro_healthy = True
        self._rc_connected = True
        self._rc_calibrated = True
        self._motors_healthy = True
        self._ekf_healthy = True
        self._vibration_ok = True
        self._failsafe_configured = True
        self._safety_switch_off = False  # Must be manually turned off

    def _setup_failsafe_callbacks(self):
        """Setup callbacks for failsafe events."""
        def on_failsafe(event):
            if self.flight_logger:
                self.flight_logger.log_failsafe(
                    failsafe_type=event.failsafe_type.name,
                    reason=event.reason,
                    action=event.action.value,
                    data=event.data,
                    position=(self.current_position[0], self.current_position[1], self.current_altitude),
                )

        for failsafe_type in FailsafeType:
            self.failsafe_manager.register_callback(failsafe_type, on_failsafe)

    def _setup_geofence_callbacks(self):
        """Setup callbacks for geofence events."""
        def on_geofence(event):
            if self.flight_logger:
                position = (self.current_position[0], self.current_position[1], self.current_altitude)
                if event['event'] == 'breach':
                    self.flight_logger.log_event(
                        EventType.GEOFENCE_BREACH,
                        f"Geofence breach: {event['fence_name']}",
                        LogLevel.CRITICAL,
                        position=position,
                    )
                    # Trigger appropriate action
                    action = event.get('action')
                    if action == GeofenceAction.RTH.value:
                        self._initiate_return_to_home("geofence_breach")
                    elif action == GeofenceAction.LAND.value:
                        self._initiate_landing(LandingMode.EMERGENCY, "geofence_breach")
                elif event['event'] == 'warning':
                    self.flight_logger.log_event(
                        EventType.GEOFENCE_WARNING,
                        f"Geofence warning: {event['fence_name']}",
                        LogLevel.WARNING,
                        position=position,
                    )

        self.geofence_manager.add_callback(on_geofence)

    def _setup_arming_callbacks(self):
        """Setup callbacks for arming events."""
        def on_arming(event):
            if self.flight_logger:
                if event['event'] == 'armed':
                    self.flight_logger.log_event(
                        EventType.ARMED,
                        "Drone armed",
                        LogLevel.INFO,
                    )
                elif event['event'] == 'disarmed':
                    self.flight_logger.log_event(
                        EventType.DISARMED,
                        f"Drone disarmed: {event.get('reason', 'unknown')}",
                        LogLevel.INFO,
                    )
                elif event['event'] == 'killed':
                    self.flight_logger.log_event(
                        EventType.KILL_SWITCH,
                        f"Kill switch activated: {event.get('reason', 'emergency')}",
                        LogLevel.CRITICAL,
                    )

        self.arming_manager.add_callback(on_arming)

    def _setup_landing_callbacks(self):
        """Setup callbacks for landing events."""
        def on_landing(event):
            if self.flight_logger:
                position = (self.current_position[0], self.current_position[1], self.current_altitude)
                if event['event'] == 'landing_started':
                    self.flight_logger.log_event(
                        EventType.LANDING,
                        f"Landing started: {event['mode']}",
                        LogLevel.INFO,
                        position=position,
                    )
                elif event['event'] == 'landing_complete':
                    self.flight_logger.log_event(
                        EventType.LANDED,
                        f"Landing complete after {event.get('duration', 0):.1f}s",
                        LogLevel.INFO,
                        position=position,
                    )
                    self._is_landing = False
                    self._is_flying = False
                elif event['event'] == 'landing_aborted':
                    self.flight_logger.log_event(
                        EventType.LANDING_ABORTED,
                        f"Landing aborted: {event.get('reason', 'unknown')}",
                        LogLevel.WARNING,
                        position=position,
                    )
                    self._is_landing = False

        self.landing_manager.add_callback(on_landing)

    def _setup_camera_callbacks(self):
        """Setup callbacks for camera events."""
        def on_detection(detection):
            if self.flight_logger:
                position = (self.current_position[0], self.current_position[1], self.current_altitude)
                if detection.detection_type == DetectionType.OBSTACLE:
                    self.flight_logger.log_obstacle_detected(
                        distance=detection.distance or 0.0,
                        position=position,
                    )
                elif detection.detection_type == DetectionType.LANDING_MARKER:
                    self.flight_logger.log_event(
                        EventType.MARKER_DETECTED,
                        f"Landing marker detected at distance {detection.distance:.1f}m",
                        LogLevel.INFO,
                        position=position,
                    )

        self.camera_sensor.add_detection_callback(on_detection)

    def _setup_navigation_callbacks(self):
        """Setup callbacks for navigation state estimator events."""
        def on_navigation_mode_change(mode: NavigationMode):
            old_mode = self._navigation_mode
            self._navigation_mode = mode

            if self.flight_logger:
                position = (self.current_position[0], self.current_position[1], self.current_altitude)

                if mode == NavigationMode.DEAD_RECKONING:
                    self.flight_logger.log_event(
                        EventType.GPS_LOSS,
                        f"Navigation degraded: switched to dead reckoning from {old_mode.value}",
                        LogLevel.WARNING,
                        position=position,
                    )
                    # Check GPS failsafe after mode change
                    self.failsafe_manager.check_gps(
                        has_fix=False,
                        satellites=0,
                        accuracy=float('inf'),
                    )
                elif mode == NavigationMode.GPS_IMU_FUSION and old_mode == NavigationMode.DEAD_RECKONING:
                    self.flight_logger.log_event(
                        EventType.GPS_FIX,
                        "Navigation restored: GPS + IMU fusion active",
                        LogLevel.INFO,
                        position=position,
                    )
                elif mode == NavigationMode.FAILED:
                    self.flight_logger.log_event(
                        EventType.SENSOR_FAILURE,
                        "Navigation FAILED: no valid position source",
                        LogLevel.CRITICAL,
                        position=position,
                    )

        self.state_estimator.add_mode_change_callback(on_navigation_mode_change)

    def _setup_lidar_callbacks(self):
        """Setup callbacks for LiDAR SLAM events."""
        # LiDAR SLAM doesn't have built-in callbacks, so we monitor state changes
        # in the update loop instead
        pass

    def _setup_vio_callbacks(self):
        """Setup callbacks for VIO state changes."""
        def on_vio_state_change(vio_state: VIOState):
            if self.flight_logger:
                position = (self.current_position[0], self.current_position[1], self.current_altitude)

                if vio_state == VIOState.RUNNING:
                    self.flight_logger.log_event(
                        EventType.SENSOR_FIX if hasattr(EventType, 'SENSOR_FIX') else EventType.GPS_FIX,
                        "VIO running: Visual-Inertial fusion active",
                        LogLevel.INFO,
                        position=position,
                    )
                elif vio_state == VIOState.IMU_ONLY:
                    self.flight_logger.log_event(
                        EventType.SENSOR_FAILURE,
                        "VIO degraded: Vision lost, IMU only",
                        LogLevel.WARNING,
                        position=position,
                    )
                elif vio_state == VIOState.VISION_ONLY:
                    self.flight_logger.log_event(
                        EventType.SENSOR_FAILURE,
                        "VIO degraded: IMU lost, vision only",
                        LogLevel.WARNING,
                        position=position,
                    )
                elif vio_state == VIOState.FAILED:
                    self.flight_logger.log_event(
                        EventType.SENSOR_FAILURE,
                        "VIO FAILED: Both vision and IMU lost",
                        LogLevel.CRITICAL,
                        position=position,
                    )

        self.vio_fusion.add_state_change_callback(on_vio_state_change)

    def _setup_flight_controller(self):
        """Setup flight controller with mode handlers."""
        # Register mode handlers
        self.flight_controller.register_mode_handler(
            FlightMode.STABILIZE,
            StabilizeModeHandler(self)
        )
        self.flight_controller.register_mode_handler(
            FlightMode.LOITER,
            LoiterModeHandler(self)
        )
        self.flight_controller.register_mode_handler(
            FlightMode.GUIDED,
            GuidedModeHandler(self)
        )
        self.flight_controller.register_mode_handler(
            FlightMode.RTL,
            RTLModeHandler(self)
        )
        self.flight_controller.register_mode_handler(
            FlightMode.AUTO,
            AutoModeHandler(self)
        )

        # Setup mode change callback for logging
        def on_mode_change(old_mode: FlightMode, new_mode: FlightMode):
            if self.flight_logger:
                position = (self.current_position[0], self.current_position[1], self.current_altitude)
                self.flight_logger.log_event(
                    EventType.STATE_CHANGE,
                    f"Flight mode changed: {old_mode.value} -> {new_mode.value}",
                    LogLevel.INFO,
                    position=position,
                )

        self.flight_controller.add_mode_change_callback(on_mode_change)

        # Setup target reached callback
        def on_target_reached(position: tuple):
            if self.flight_logger:
                self.flight_logger.log_event(
                    EventType.WAYPOINT_REACHED if hasattr(EventType, 'WAYPOINT_REACHED') else EventType.STATE_CHANGE,
                    f"Target reached: ({position[0]:.6f}, {position[1]:.6f}, {position[2]:.1f}m)",
                    LogLevel.INFO,
                    position=position,
                )

        self.flight_controller.add_target_reached_callback(on_target_reached)

    def start_flight(self) -> str:
        """Start a new flight session."""
        self._is_flying = True
        self._last_command_time = time.time()

        # Initialize GPS with home position
        self.gps_sensor.start()
        self.gps_sensor.set_position(self.home_position[0], self.home_position[1])

        # Initialize IMU sensor
        self.imu_sensor.start()
        self.imu_sensor.set_calibration_complete()  # Assume calibrated for simulation

        # Initialize optical flow sensor
        self.optical_flow_sensor.start()

        # Initialize LiDAR sensor (always start, SLAM is optional)
        self.lidar_sensor.start()
        if self._lidar_slam_enabled:
            self.lidar_slam.initialize()

        # Initialize Visual Odometry and VIO (for indoor/GPS-denied navigation)
        self.visual_odometry.start()
        if self._vio_enabled:
            self.vio_fusion.initialize()

        # Initialize state estimator with home position
        self.state_estimator.set_home_position(
            self.home_position[0],
            self.home_position[1],
            self.current_altitude
        )

        # Start logging session
        session_id = None
        if self.flight_logger:
            session_id = self.flight_logger.start_session(self.home_position)

        return session_id or ""

    def end_flight(self) -> Optional[str]:
        """End the current flight session."""
        self._is_flying = False
        self._is_returning_home = False

        self.gps_sensor.stop()
        self.imu_sensor.stop()
        self.optical_flow_sensor.stop()
        self.lidar_sensor.stop()
        self.visual_odometry.stop()

        # End logging session and get log file path
        log_path = None
        if self.flight_logger:
            log_path = self.flight_logger.end_session()

        return log_path

    def update_sensors(self, dt: Optional[float] = None):
        """Update all sensor readings."""
        # Check if sensors are connected
        if not self.sensor_manager.check_sensor_connection():
            self._log_error("sensor_connection", "Sensor connection lost")
            raise SensorConnectionLostException("Sensor connection lost")

        try:
            self.wind_speed = self.wind_sensor.get_wind_speed()
            self.wind_direction = self.wind_sensor.get_wind_direction()

            # Use overridden battery level for testing, otherwise get from sensor
            if self._battery_level_override is not None:
                self.battery_level = self._battery_level_override
            else:
                self.battery_level = self.battery_sensor.get_battery_level()

            # Update IMU and feed to state estimator (high rate - 100Hz typical)
            imu_reading = self.imu_sensor.update()
            if imu_reading.is_valid():
                self.state_estimator.predict(imu_reading)

            # Update GPS and feed to state estimator (lower rate - 5-10Hz typical)
            gps_reading = self.gps_sensor.update()
            self.state_estimator.update_gps(gps_reading)

            # Update optical flow sensor (provides velocity at low altitude)
            # Uses ultrasonic/altitude for height, provides ground-relative velocity
            optical_flow_reading = self.optical_flow_sensor.update(height_agl=self.current_altitude)

            # Feed gyro rates to optical flow for compensation
            if imu_reading.is_valid():
                self.optical_flow_sensor.set_gyro_rates(
                    imu_reading.gyroscope.x,
                    imu_reading.gyroscope.y
                )

            # Update LiDAR sensor and process SLAM
            self.lidar_sensor.set_drone_pose(
                self.current_position[0],  # x (using lat as proxy)
                self.current_position[1],  # y (using lon as proxy)
                self.current_altitude,
                imu_reading.get_attitude().yaw if imu_reading.is_valid() else 0.0
            )
            lidar_reading = self.lidar_sensor.update()

            # Process LiDAR SLAM if enabled and reading is valid
            if self._lidar_slam_enabled and lidar_reading.is_valid():
                # Update SLAM with current attitude from IMU
                if imu_reading.is_valid():
                    attitude = imu_reading.get_attitude()
                    self.lidar_slam.set_attitude(attitude.roll, attitude.pitch, attitude.yaw)
                self.lidar_slam.set_height(self.current_altitude)

                # Process the scan
                slam_estimate = self.lidar_slam.process_scan(lidar_reading)

                # Log SLAM state changes
                if slam_estimate.is_valid() and self.flight_logger:
                    if self.lidar_slam.get_state() == SLAMState.LOST:
                        self.flight_logger.log_event(
                            EventType.SENSOR_FAILURE,
                            "LiDAR SLAM lost tracking",
                            LogLevel.WARNING,
                            position=(self.current_position[0], self.current_position[1], self.current_altitude),
                        )

            # Update Visual Odometry and process VIO if enabled
            self.visual_odometry.set_true_velocity(
                self.state_estimator.get_state().velocity_north,
                self.state_estimator.get_state().velocity_east,
                -self.state_estimator.get_state().velocity_down  # Convert NED to VO frame
            )
            vo_reading = self.visual_odometry.update()

            # Process VIO fusion if enabled
            if self._vio_enabled:
                # Feed IMU to VIO (high rate)
                if imu_reading.is_valid():
                    self.vio_fusion.process_imu(imu_reading)

                # Feed VO to VIO (lower rate, only when valid)
                if vo_reading.is_valid():
                    self.vio_fusion.process_vo(vo_reading)

                    # Correct scale using barometer/altitude
                    if self.current_altitude > 0.5:
                        self.vio_fusion.correct_scale_from_height(self.current_altitude)

            # Get fused position from state estimator
            nav_state = self.state_estimator.get_state()
            if nav_state.navigation_mode != NavigationMode.FAILED:
                # Use fused GPS position from state estimator
                fused_lat, fused_lon, fused_alt = self.state_estimator.get_gps_position()
                self.current_position = (fused_lat, fused_lon)
                self.current_altitude = fused_alt
                self._navigation_mode = nav_state.navigation_mode
            elif gps_reading.is_valid():
                # Fallback to raw GPS if estimator failed but GPS valid
                self.current_position = (gps_reading.latitude, gps_reading.longitude)
                self.current_altitude = gps_reading.altitude

            # MAVLink override: when a real autopilot/SITL is connected and
            # its heartbeat is fresh, prefer its position/altitude/battery
            # over the simulated stack. Logging and failsafes below run on
            # the overridden values.
            if self._mavlink_telemetry_fresh():
                self._apply_mavlink_telemetry()

            # Log sensor data
            self._log_sensor_data()

            # Check failsafe conditions
            self._check_failsafes()

        except Exception as e:
            self._log_error("sensor_update", f"Error updating sensors: {e}", e)
            raise SensorConnectionLostException("Failed to connect to sensors")

    def _log_sensor_data(self):
        """Log current sensor readings."""
        if not self.flight_logger:
            return

        position = (self.current_position[0], self.current_position[1], self.current_altitude)

        self.flight_logger.log_sensor_data("battery", self.battery_level, position=position)
        self.flight_logger.log_sensor_data("wind_speed", self.wind_speed, position=position)
        self.flight_logger.log_sensor_data("wind_direction", self.wind_direction, position=position)
        self.flight_logger.log_sensor_data("gps", self.gps_sensor.to_dict(), position=position)
        self.flight_logger.log_sensor_data("imu", self.imu_sensor.to_dict(), position=position)
        self.flight_logger.log_sensor_data("optical_flow", self.optical_flow_sensor.to_dict(), position=position)
        self.flight_logger.log_sensor_data("navigation", self.state_estimator.to_dict(), position=position)
        self.flight_logger.log_sensor_data("lidar", self.lidar_sensor.to_dict(), position=position)
        if self._lidar_slam_enabled:
            self.flight_logger.log_sensor_data("lidar_slam", self.lidar_slam.to_dict(), position=position)
        self.flight_logger.log_sensor_data("visual_odometry", self.visual_odometry.to_dict(), position=position)
        if self._vio_enabled:
            self.flight_logger.log_sensor_data("vio", self.vio_fusion.to_dict(), position=position)

    def _log_error(self, error_type: str, message: str, exception: Exception = None):
        """Log an error event."""
        if self.flight_logger:
            position = (self.current_position[0], self.current_position[1], self.current_altitude)
            self.flight_logger.log_error(error_type, message, exception, position=position)

    def _check_failsafes(self):
        """Check all failsafe conditions."""
        # Check battery
        battery_action = self.failsafe_manager.check_battery(self.battery_level)
        if battery_action:
            self._handle_failsafe_action(battery_action)

        # Check GPS
        gps_action = self.failsafe_manager.check_gps(
            has_fix=self.gps_sensor.has_fix(),
            satellites=self.gps_sensor.get_satellites(),
            accuracy=self.gps_sensor.get_accuracy()[0],
        )
        if gps_action:
            self._handle_failsafe_action(gps_action)

        # Check signal
        signal_action = self.failsafe_manager.check_signal(
            self._signal_strength,
            self._last_command_time,
        )
        if signal_action:
            self._handle_failsafe_action(signal_action)

    def _handle_failsafe_action(self, action: FailsafeAction):
        """Handle a failsafe action."""
        if action == FailsafeAction.RETURN_TO_HOME:
            if not self._is_returning_home:
                self._is_returning_home = True
                if self.flight_logger:
                    self.flight_logger.log_event(
                        EventType.RETURN_TO_HOME_INITIATED,
                        "Failsafe triggered return to home",
                        LogLevel.WARNING,
                    )
        elif action == FailsafeAction.LAND_IMMEDIATELY:
            if self.flight_logger:
                self.flight_logger.log_event(
                    EventType.LANDING,
                    "Emergency landing initiated",
                    LogLevel.CRITICAL,
                )
        elif action == FailsafeAction.HOVER:
            if self.flight_logger:
                self.flight_logger.log_event(
                    EventType.HOVER,
                    "Hovering due to failsafe condition",
                    LogLevel.WARNING,
                )

    def receive_command(self, command: str):
        """Process a received command (updates signal tracking)."""
        self._last_command_time = time.time()
        if self.flight_logger:
            self.flight_logger.log_event(
                EventType.COMMAND_RECEIVED,
                f"Command received: {command}",
                LogLevel.DEBUG,
            )

    def set_signal_strength(self, strength: float):
        """Set current signal strength (0.0 to 1.0) for testing."""
        self._signal_strength = max(0.0, min(1.0, strength))

    # ------------------------------------------------------------------ MAVLink

    @property
    def is_mavlink_connected(self) -> bool:
        return self.mavlink_backend is not None and self.mavlink_backend.is_connected

    def connect_mavlink(self, endpoint: str, baud: int = 57600,
                        heartbeat_timeout: float = 10.0) -> dict:
        """
        Connect to a real autopilot or SITL endpoint.

        Once connected, update_sensors() overrides the simulated
        position/altitude/battery with values from the autopilot whenever
        the heartbeat is fresh. The simulated subsystems keep running so
        downstream code (failsafes, logging, etc.) still sees consistent
        state on the legacy fields.

        Endpoint examples: "udpout:127.0.0.1:14550" (ArduPilot SITL),
        "udpin:0.0.0.0:14550" (bind & wait), "COM3" (serial).
        """
        if self.mavlink_backend is not None:
            raise RuntimeError("MAVLink already connected; call disconnect_mavlink() first")

        backend = MAVLinkBackend()
        peer = backend.connect(endpoint, baud=baud, heartbeat_timeout=heartbeat_timeout)
        self.mavlink_backend = backend
        if self.flight_logger:
            self.flight_logger.log_event(
                EventType.COMMAND_RECEIVED,
                f"MAVLink connected: {endpoint} (sys={peer.system}, autopilot={peer.autopilot})",
                LogLevel.INFO,
            )
        return {
            "endpoint": endpoint,
            "system": peer.system,
            "component": peer.component,
            "type": peer.type,
            "autopilot": peer.autopilot,
        }

    def disconnect_mavlink(self) -> None:
        """Close the MAVLink link and revert to simulated telemetry."""
        if self.mavlink_backend is None:
            return
        try:
            self.mavlink_backend.close()
        finally:
            self.mavlink_backend = None
            if self.flight_logger:
                self.flight_logger.log_event(
                    EventType.COMMAND_RECEIVED,
                    "MAVLink disconnected",
                    LogLevel.INFO,
                )

    def _mavlink_telemetry_fresh(self) -> bool:
        """True if a backend is connected and heartbeat age is within bounds."""
        if not self.is_mavlink_connected:
            return False
        age = self.mavlink_backend.heartbeat_age_s
        if age is None:
            return False
        return age <= self._mavlink_heartbeat_max_age_s

    def _apply_mavlink_telemetry(self) -> None:
        """
        Override public state from autopilot telemetry. Only called from
        update_sensors() after the simulated path runs, so failsafe checks
        and logging downstream see the real values.
        """
        tel = self.mavlink_backend.telemetry
        if tel.position is not None:
            self.current_position = (tel.position["lat"], tel.position["lon"])
            # Prefer altitude relative to home over MSL — matches how the
            # rest of the codebase uses current_altitude (AGL-ish).
            self.current_altitude = tel.position["alt_rel_m"]
        # battery_pct is -1 when the autopilot can't estimate; ignore in that case
        if tel.battery_pct is not None and tel.battery_pct >= 0:
            self.battery_level = float(tel.battery_pct)

    def get_mavlink_status(self) -> dict:
        """Return current MAVLink link state for the dashboard."""
        if not self.is_mavlink_connected:
            return {"connected": False}
        tel = self.mavlink_backend.telemetry
        return {
            "connected": True,
            "heartbeat_age_s": tel.heartbeat_age_s,
            "fresh": self._mavlink_telemetry_fresh(),
            "armed": tel.armed,
            "mode": tel.mode,
            "position": tel.position,
            "attitude": tel.attitude,
            "velocity_ned": tel.velocity_ned,
            "battery_pct": tel.battery_pct,
        }

    def set_home_position(self, latitude: float, longitude: float):
        """Set the home position for the drone."""
        self.home_position = (latitude, longitude)

    def set_current_position(self, latitude: float, longitude: float):
        """Set the current position of the drone."""
        self.current_position = (latitude, longitude)

    def set_battery_level_override(self, level: float):
        """Set battery level override for testing purposes."""
        self._battery_level_override = level

    @property
    def flight_mode(self) -> FlightMode:
        """Get the current flight mode."""
        return self.flight_controller.current_mode

    def set_flight_mode(self, mode: FlightMode) -> Tuple[bool, str]:
        """Set the current flight mode through the flight controller."""
        return self.flight_controller.set_mode(mode)

    def calculate_return_to_home(self) -> List[Tuple[float, float]]:
        """
        Calculate the safest return-to-home route based on current wind speed and battery percentage.

        Returns:
            List of waypoints (latitude, longitude) for the return path
        """
        # Update sensor data first
        self.update_sensors()

        # Check for low battery condition after updating sensors
        if self.battery_level < 10:
            raise LowBatteryException("Battery level critically low for safe flight")

        # Use PathPlanner to calculate safe route
        planner = PathPlanner()
        current_pos = Position(self.current_position[0], self.current_position[1])
        home_pos = Position(self.home_position[0], self.home_position[1])
        state = State(battery_percent=self.battery_level, wind_speed_m_s=self.wind_speed)

        try:
            path_positions = planner.plan_return(home_pos, current_pos, state)
        except ValueError as e:
            raise LowBatteryException(f"Unable to plan return path: {e}")

        # Convert Position objects to (lat, lon) tuples
        waypoints = [(p.latitude, p.longitude) for p in path_positions]

        # Add ultrasonic obstacle checks
        # Safety: if ultrasonic is valid and distance < min_clearance, trigger avoidance
        try:
            ultra_dist_m = self.sensor_manager.get("ultrasonic").get_distance()
            if not math.isnan(ultra_dist_m) and ultra_dist_m < 1.5:  # 1.5m clearance
                waypoints = self._add_obstacle_avoid_waypoint(
                    waypoints, self.current_position, ultra_dist_m
                )
        except Exception:
            pass

        return waypoints

    def _add_obstacle_avoid_waypoint(
        self, waypoints: List[Tuple[float, float]], current_pos: Tuple[float, float], distance: float
    ) -> List[Tuple[float, float]]:
        """Add a waypoint to avoid an obstacle detected by sensors."""
        # Log obstacle detection
        if self.flight_logger:
            self.flight_logger.log_obstacle_detected(
                distance=distance,
                position=(current_pos[0], current_pos[1], self.current_altitude),
            )

        # Check failsafe for imminent collision
        self.failsafe_manager.check_obstacle(distance)

        # Simple avoidance strategy: offset current position slightly
        lat, lon = current_pos
        avoid_point = (lat + 0.0001, lon + 0.0001)
        return [avoid_point] + waypoints

    def get_gps_status(self) -> dict:
        """Get current GPS status."""
        return {
            'has_fix': self.gps_sensor.has_fix(),
            'has_3d_fix': self.gps_sensor.has_3d_fix(),
            'satellites': self.gps_sensor.get_satellites(),
            'accuracy': self.gps_sensor.get_accuracy(),
            'position': self.gps_sensor.get_position(),
            'altitude': self.gps_sensor.get_altitude(),
            'speed': self.gps_sensor.get_speed(),
            'heading': self.gps_sensor.get_heading(),
            'signal_quality': self.gps_sensor.get_signal_quality(),
        }

    def get_navigation_status(self) -> dict:
        """Get current navigation state estimator status."""
        nav_state = self.state_estimator.get_state()
        return {
            'mode': self._navigation_mode.value,
            'gps_available': nav_state.gps_available,
            'imu_available': nav_state.imu_available,
            'position_ned': nav_state.get_position_ned(),
            'velocity_ned': nav_state.get_velocity_ned(),
            'attitude': nav_state.get_attitude().to_dict(),
            'speed': nav_state.get_speed(),
            'altitude_agl': nav_state.get_altitude_agl(),
            'uncertainty': {
                'position': nav_state.position_uncertainty,
                'velocity': nav_state.velocity_uncertainty,
                'attitude': nav_state.attitude_uncertainty,
            },
            'dead_reckoning_time': self.state_estimator.get_dead_reckoning_time(),
            'health': self.state_estimator.get_health().value,
        }

    def get_optical_flow_status(self) -> dict:
        """Get current optical flow sensor status."""
        return {
            'state': self.optical_flow_sensor.get_state().value,
            'velocity': self.optical_flow_sensor.get_velocity(),
            'speed': self.optical_flow_sensor.get_speed(),
            'quality': self.optical_flow_sensor.get_quality().name,
            'quality_percent': self.optical_flow_sensor.get_quality_percent(),
            'is_valid': self.optical_flow_sensor.is_valid(),
        }

    def get_lidar_status(self) -> dict:
        """Get current LiDAR sensor status."""
        is_connected = self.lidar_sensor._is_connected if hasattr(self.lidar_sensor, '_is_connected') else False
        obstacles = self.lidar_sensor.detect_obstacles() if is_connected else []
        return {
            'enabled': is_connected,
            'state': self.lidar_sensor.get_state().value,
            'frame_count': self.lidar_sensor.get_frame_count(),
            'point_count': self.lidar_sensor.get_point_count() if hasattr(self.lidar_sensor, 'get_point_count') else 0,
            'obstacle_count': len(obstacles),
            'nearest_obstacle': self.lidar_sensor.get_nearest_obstacle_distance(),
            'is_valid': self.lidar_sensor.is_valid(),
        }

    def get_lidar_slam_status(self) -> dict:
        """Get current LiDAR SLAM status."""
        if not self._lidar_slam_enabled:
            return {'enabled': False}

        pos = self.lidar_slam.get_position()
        return {
            'enabled': True,
            'state': self.lidar_slam.get_state().value,
            'is_tracking': self.lidar_slam.is_tracking(),
            'quality': self.lidar_slam.get_quality().name,
            'position': {'x': pos[0], 'y': pos[1]} if pos else None,
            'velocity': self.lidar_slam.get_velocity(),
            'scan_count': self.lidar_slam._scan_count if hasattr(self.lidar_slam, '_scan_count') else 0,
            'map_cells': self.lidar_slam.get_map().to_dict()['cells_updated'],
        }

    def enable_lidar_slam(self, enabled: bool = True):
        """Enable or disable LiDAR SLAM processing."""
        self._lidar_slam_enabled = enabled
        if enabled and self._is_flying:
            self.lidar_slam.initialize()

    def is_lidar_slam_enabled(self) -> bool:
        """Check if LiDAR SLAM is enabled."""
        return self._lidar_slam_enabled

    def get_lidar_obstacles(self) -> List[dict]:
        """Get obstacles detected by LiDAR."""
        obstacles = self.lidar_sensor.detect_obstacles()
        return [o.to_dict() for o in obstacles]

    def check_lidar_obstacle_ahead(self, max_distance: float = 10.0) -> Optional[float]:
        """Check for obstacles directly ahead using LiDAR."""
        return self.lidar_sensor.check_obstacle_in_direction(0.0, cone_angle=30.0, max_distance=max_distance)

    def get_visual_odometry_status(self) -> dict:
        """Get current Visual Odometry sensor status."""
        is_connected = self.visual_odometry._is_connected if hasattr(self.visual_odometry, '_is_connected') else False
        return {
            'enabled': is_connected,
            'state': self.visual_odometry.get_state().value,
            'quality': self.visual_odometry.get_quality().name,
            'confidence': self.visual_odometry.get_confidence(),
            'features_tracked': self.visual_odometry.get_feature_count(),
            'frame_count': self.visual_odometry.get_frame_count(),
            'is_tracking': self.visual_odometry.is_tracking(),
            'is_valid': self.visual_odometry.is_valid(),
            'position': self.visual_odometry.get_position(),
            'velocity': self.visual_odometry.get_velocity(),
        }

    def get_vio_status(self) -> dict:
        """Get current VIO (Visual-Inertial Odometry) status."""
        if not self._vio_enabled:
            return {'enabled': False}

        state = self.vio_fusion.get_state()
        return {
            'enabled': True,
            'system_state': self.vio_fusion.get_system_state().value,
            'health': self.vio_fusion.get_health().value,
            'is_running': self.vio_fusion.is_running(),
            'is_valid': self.vio_fusion.is_valid(),
            'position': state.get_position(),
            'velocity': state.get_velocity(),
            'attitude': state.get_attitude().to_dict(),
            'scale': self.vio_fusion.get_scale(),
            'scale_initialized': self.vio_fusion.is_scale_initialized(),
            'vision_available': state.vision_available,
            'imu_available': state.imu_available,
        }

    def enable_vio(self, enabled: bool = True):
        """Enable or disable VIO (Visual-Inertial Odometry) processing."""
        self._vio_enabled = enabled
        if enabled and self._is_flying:
            self.vio_fusion.initialize()

    def is_vio_enabled(self) -> bool:
        """Check if VIO is enabled."""
        return self._vio_enabled

    def get_failsafe_status(self) -> dict:
        """Get current failsafe system status."""
        return self.failsafe_manager.get_status()

    def get_flight_statistics(self) -> dict:
        """Get flight statistics from the logger."""
        if self.flight_logger:
            return self.flight_logger.get_statistics()
        return {}

    def is_failsafe_active(self) -> bool:
        """Check if any failsafe is currently active."""
        return self.failsafe_manager.is_failsafe_active()

    @property
    def is_flying(self) -> bool:
        """Check if drone is currently in flight."""
        return self._is_flying

    @property
    def is_returning_home(self) -> bool:
        """Check if drone is returning to home."""
        return self._is_returning_home

    @property
    def is_landing(self) -> bool:
        """Check if drone is currently landing."""
        return self._is_landing

    @property
    def is_armed(self) -> bool:
        """Check if drone is armed."""
        return self.arming_manager.is_armed

    # ==================== Arming System ====================

    def run_pre_arm_checks(self) -> List:
        """Run all pre-arm checks and return results."""
        return self.arming_manager.run_pre_arm_checks(
            gps_satellites=self.gps_sensor.get_satellites(),
            gps_accuracy=self.gps_sensor.get_accuracy()[0],
            gps_fix=self.gps_sensor.has_fix(),
            compass_healthy=self._compass_healthy,
            compass_calibrated=self._compass_calibrated,
            accel_calibrated=self._accel_calibrated,
            gyro_calibrated=self._gyro_calibrated,
            baro_healthy=self._baro_healthy,
            battery_voltage=self.battery_sensor.get_voltage() if hasattr(self.battery_sensor, 'get_voltage') else 12.6,
            battery_percentage=self.battery_level,
            battery_cells=3,  # Default 3S battery
            rc_connected=self._rc_connected,
            rc_calibrated=self._rc_calibrated,
            rc_signal=self._signal_strength,
            home_set=self.home_position != (0.0, 0.0),
            motors_healthy=self._motors_healthy,
            ekf_healthy=self._ekf_healthy,
            vibration_ok=self._vibration_ok,
            failsafe_configured=self._failsafe_configured,
            geofence_configured=len(self.geofence_manager.get_all_fences()) > 0,
            safety_switch_off=self._safety_switch_off,
        )

    def arm(self, force: bool = False) -> Tuple[bool, str]:
        """
        Arm the drone.

        Args:
            force: If True, skip non-critical checks (dangerous!)

        Returns:
            (success, message)
        """
        # Run pre-arm checks first
        self.run_pre_arm_checks()

        success, message = self.arming_manager.arm(force=force)

        if not success and self.flight_logger:
            self.flight_logger.log_event(
                EventType.ARM_FAILED,
                message,
                LogLevel.WARNING,
            )

        return success, message

    def disarm(self, reason: str = "user_request") -> Tuple[bool, str]:
        """Disarm the drone."""
        return self.arming_manager.disarm(reason)

    def kill(self, reason: str = "emergency") -> Tuple[bool, str]:
        """
        Emergency kill switch - immediately stop all motors.

        This is a one-way operation that requires a reset to clear.
        """
        self._is_flying = False
        self._is_returning_home = False
        self._is_landing = False
        return self.arming_manager.kill(reason)

    def reset_kill_switch(self) -> Tuple[bool, str]:
        """Reset the kill switch after emergency."""
        return self.arming_manager.reset_kill_switch()

    def set_safety_switch(self, off: bool):
        """Set the safety switch state (must be off to arm)."""
        self._safety_switch_off = off

    def set_props_off_mode(self, enabled: bool):
        """Enable/disable props-off mode for safe testing."""
        self.arming_manager.set_props_off_mode(enabled)

    def confirm_manual_check(self, check: PreArmCheck, passed: bool):
        """Confirm a manual pre-arm check (e.g., props installed)."""
        self.arming_manager.confirm_manual_check(check, passed)

    def get_arming_status(self) -> dict:
        """Get current arming status."""
        status = self.arming_manager.get_status()
        return {
            'state': status.state.value,
            'can_arm': status.can_arm,
            'kill_switch_active': status.kill_switch_active,
            'failed_required': [c.message for c in status.failed_required],
            'failed_optional': [c.message for c in status.failed_optional],
            'armed_time': status.armed_time,
            'flight_time': self.arming_manager.get_flight_time(),
        }

    # ==================== Landing System ====================

    def _initiate_landing(self, mode: LandingMode, reason: str = ""):
        """Internal method to initiate landing."""
        if self._is_landing:
            return

        self._is_landing = True
        current_pos = (self.current_position[0], self.current_position[1], self.current_altitude)

        success, _ = self.landing_manager.start_landing(mode, current_pos)

        if self.flight_logger and success:
            self.flight_logger.log_event(
                EventType.LANDING,
                f"Landing initiated: {mode.value} - {reason}",
                LogLevel.INFO,
            )

    def start_landing(self, mode: LandingMode = LandingMode.NORMAL) -> Tuple[bool, str]:
        """
        Start a landing operation.

        Args:
            mode: Landing mode to use

        Returns:
            (success, message)
        """
        if not self._is_flying:
            return False, "Drone is not flying"

        current_pos = (self.current_position[0], self.current_position[1], self.current_altitude)
        self._is_landing = True

        return self.landing_manager.start_landing(mode, current_pos)

    def abort_landing(self) -> Tuple[bool, str]:
        """Abort the current landing operation."""
        result = self.landing_manager.abort()
        if result[0]:
            self._is_landing = False
        return result

    def get_landing_status(self) -> dict:
        """Get current landing status."""
        status = self.landing_manager.get_status()
        return {
            'mode': status.mode.value,
            'state': status.state.value,
            'target_position': status.target_position,
            'elapsed_time': status.elapsed_time,
            'marker_detected': status.marker_detected,
            'abort_reason': status.abort_reason.name if status.abort_reason else None,
        }

    # ==================== Geofence System ====================

    def setup_home_geofence(self, radius: float = 500.0, max_altitude: float = 120.0):
        """Setup a geofence around the home position."""
        if self.home_position == (0.0, 0.0):
            return None

        return self.geofence_manager.create_home_geofence(
            self.home_position[0],
            self.home_position[1],
            radius=radius,
            max_altitude=max_altitude,
        )

    def add_no_fly_zone(self, name: str, vertices: List[Tuple[float, float]]):
        """Add a no-fly zone polygon."""
        return self.geofence_manager.create_no_fly_zone(name, vertices)

    def check_geofence(self) -> dict:
        """Check current position against all geofences."""
        status = self.geofence_manager.check_position(
            self.current_position[0],
            self.current_position[1],
            self.current_altitude,
        )
        return {
            'zone': status.zone.value,
            'distance_to_boundary': status.distance_to_boundary,
            'altitude_status': status.altitude_status,
            'breached_fences': status.breached_fences,
            'warning_fences': status.warning_fences,
        }

    def get_geofence_status(self) -> dict:
        """Get geofence manager status."""
        return self.geofence_manager.to_dict()

    # ==================== Health Monitoring ====================

    def update_health_status(self):
        """Update all health monitoring data."""
        # Update CPU (simulated)
        self.health_monitor.update_component(
            ComponentType.CPU,
            temperature=45.0,  # Would come from actual CPU temp sensor
            usage=30.0,
        )

        # Update memory (simulated)
        self.health_monitor.update_component(
            ComponentType.MEMORY,
            usage=50.0,
        )

        # Update battery from sensor
        self.health_monitor.update_component(
            ComponentType.BATTERY,
            voltage=self.battery_sensor.get_voltage() if hasattr(self.battery_sensor, 'get_voltage') else 12.0,
            percentage=self.battery_level,
            temperature=25.0,
        )

        # Update GPS
        self.health_monitor.update_component(
            ComponentType.GPS,
            satellites=self.gps_sensor.get_satellites(),
            accuracy=self.gps_sensor.get_accuracy()[0],
            has_fix=self.gps_sensor.has_fix(),
        )

        # Update compass (simulated)
        self.health_monitor.update_component(
            ComponentType.COMPASS,
            healthy=self._compass_healthy,
            calibrated=self._compass_calibrated,
        )

        # Update IMU (simulated)
        self.health_monitor.update_component(
            ComponentType.IMU,
            healthy=self._accel_calibrated and self._gyro_calibrated,
            vibration=0.5,  # m/s^2
        )

        # Update motors (simulated)
        for i in range(4):
            self.health_monitor.update_component(
                ComponentType.MOTOR,
                motor_id=i,
                healthy=self._motors_healthy,
                current=5.0,
                rpm=5000,
            )

        # Update camera
        self.health_monitor.update_component(
            ComponentType.CAMERA,
            healthy=self.camera_sensor.is_healthy(),
            streaming=self.camera_sensor._streaming,
        )

    def get_health_status(self) -> dict:
        """Get overall system health status."""
        return self.health_monitor.to_dict()

    def get_health_summary(self) -> dict:
        """Get a summary of system health."""
        summary = self.health_monitor.get_overall_status()
        return {
            'overall_status': summary.value,
            'components': {
                name: status.value
                for name, status in self.health_monitor.get_all_component_status().items()
            },
            'critical_issues': self.health_monitor.get_critical_issues(),
            'warnings': self.health_monitor.get_warnings(),
        }

    # ==================== Return to Home ====================

    def _initiate_return_to_home(self, reason: str = ""):
        """Internal method to initiate RTH."""
        if self._is_returning_home:
            return

        self._is_returning_home = True

        if self.flight_logger:
            self.flight_logger.log_event(
                EventType.RETURN_TO_HOME_INITIATED,
                f"RTH initiated: {reason}",
                LogLevel.WARNING,
            )

    def return_to_home(self) -> List[Tuple[float, float]]:
        """
        Initiate return to home and get the planned path.

        Returns:
            List of waypoints for the return path
        """
        self._initiate_return_to_home("user_request")
        return self.calculate_return_to_home()

    # ==================== Camera System ====================

    def start_camera(self) -> Tuple[bool, str]:
        """Start the camera sensor."""
        return self.camera_sensor.start()

    def stop_camera(self) -> Tuple[bool, str]:
        """Stop the camera sensor."""
        return self.camera_sensor.stop()

    def start_camera_streaming(self) -> Tuple[bool, str]:
        """Start camera frame streaming."""
        return self.camera_sensor.start_streaming()

    def stop_camera_streaming(self) -> Tuple[bool, str]:
        """Stop camera frame streaming."""
        return self.camera_sensor.stop_streaming()

    def set_camera_vision_mode(self, mode: VisionMode) -> Tuple[bool, str]:
        """Set the camera vision mode."""
        return self.camera_sensor.set_vision_mode(mode)

    def set_gimbal_position(
        self,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        yaw: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Set gimbal position."""
        return self.camera_sensor.set_gimbal_position(pitch, roll, yaw)

    def point_camera_down(self) -> Tuple[bool, str]:
        """Point camera straight down (for landing)."""
        return self.camera_sensor.point_down()

    def point_camera_forward(self) -> Tuple[bool, str]:
        """Point camera forward."""
        return self.camera_sensor.point_forward()

    def start_recording(self, path: Optional[str] = None) -> Tuple[bool, str]:
        """Start video recording."""
        return self.camera_sensor.start_recording(path)

    def stop_recording(self) -> Tuple[bool, str]:
        """Stop video recording."""
        return self.camera_sensor.stop_recording()

    def take_snapshot(self, path: Optional[str] = None) -> Tuple[bool, str]:
        """Take a camera snapshot."""
        return self.camera_sensor.take_snapshot(path)

    def is_marker_detected(self) -> bool:
        """Check if landing marker is detected."""
        return self.camera_sensor.is_marker_detected()

    def get_marker_offset(self) -> Optional[Tuple[float, float]]:
        """Get marker offset from frame center for precision landing."""
        return self.camera_sensor.get_marker_offset()

    def get_camera_status(self) -> dict:
        """Get camera sensor status."""
        return self.camera_sensor.get_status()

    def get_camera_detections(self) -> List[dict]:
        """Get current camera detections."""
        return [d.to_dict() for d in self.camera_sensor.get_detections()]

    def get_closest_obstacle(self) -> Optional[dict]:
        """Get the closest detected obstacle."""
        obstacle = self.camera_sensor.get_closest_obstacle()
        return obstacle.to_dict() if obstacle else None

    # ==================== Full Status ====================

    def get_full_status(self) -> dict:
        """Get complete drone status including all systems."""
        return {
            'position': {
                'latitude': self.current_position[0],
                'longitude': self.current_position[1],
                'altitude': self.current_altitude,
            },
            'home_position': {
                'latitude': self.home_position[0],
                'longitude': self.home_position[1],
            },
            'battery': {
                'level': self.battery_level,
            },
            'state': {
                'is_flying': self._is_flying,
                'is_armed': self.is_armed,
                'is_returning_home': self._is_returning_home,
                'is_landing': self._is_landing,
            },
            'gps': self.get_gps_status(),
            'navigation': self.get_navigation_status(),
            'optical_flow': self.get_optical_flow_status(),
            'lidar': self.get_lidar_status(),
            'lidar_slam': self.get_lidar_slam_status(),
            'visual_odometry': self.get_visual_odometry_status(),
            'vio': self.get_vio_status(),
            'arming': self.get_arming_status(),
            'landing': self.get_landing_status(),
            'geofence': self.check_geofence() if len(self.geofence_manager.get_all_fences()) > 0 else None,
            'health': self.get_health_summary(),
            'failsafe': self.get_failsafe_status(),
            'camera': self.get_camera_status(),
        }
