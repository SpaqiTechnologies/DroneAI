"""
Flask web server for drone simulation dashboard.
Provides real-time telemetry and map visualization.
"""

import os
import sys
import json
import time
import threading
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.drone import Drone
from core.failsafe import FailsafeAction
from core.landing import LandingMode
from core.arming import PreArmCheck
from sensors.camera_sensor import VisionMode

# Mission planning imports
from core.mission import (
    Mission,
    MissionType,
    MissionState,
    Waypoint,
    WaypointType,
    WaypointAction,
    WaypointActionItem,
    AltitudeFrame,
    MissionValidator,
    MissionStorage,
    SurveyPatternGenerator,
)

# Takeoff imports
from core.takeoff import TakeoffManager, TakeoffState, TakeoffMode, TakeoffConfig

# Swarm imports
from swarm import (
    SwarmCoordinator,
    SwarmState,
    SwarmMission,
    DroneRegistry,
    FormationType,
)

# AI/ML imports
from ai.detection import YOLODetector, DetectionTracker
from ai.terrain import TerrainClassifier
from ai.anomaly import AnomalyDetector, AnomalyType

# Applications imports
from applications import (
    AerialMapper,
    MappingConfig,
    Inspector,
    InspectionConfig,
    InspectionType,
    SearchPatternGenerator,
    SearchType,
)

# Compliance imports
from compliance import (
    RemoteID,
    RemoteIDConfig,
    FlightRecorder,
    MaintenanceTracker,
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'drone-simulation-secret'

# Use threading async mode (eventlet is deprecated)
# Threading mode works well for simulation/development
async_mode = 'threading'

# SocketIO - configure with threading mode
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode=async_mode,
)

# Global simulation state
simulation = {
    'drone': None,
    'running': False,
    'thread': None,
    'speed': 1.0,  # Simulation speed multiplier
    'scenario': 'normal',
}

# Mission planning components
mission_storage = MissionStorage()
survey_generator = SurveyPatternGenerator()
mission_validator = MissionValidator()
current_mission: Mission = None

# Takeoff manager (initialized when drone is created)
takeoff_manager: TakeoffManager = None

# Advanced feature instances
swarm_coordinator: SwarmCoordinator = None
ai_detector: YOLODetector = None
ai_tracker: DetectionTracker = None
terrain_classifier: TerrainClassifier = None
anomaly_detector: AnomalyDetector = None
aerial_mapper: AerialMapper = None
inspector: Inspector = None
search_generator: SearchPatternGenerator = None
remote_id: RemoteID = None
flight_recorder: FlightRecorder = None
maintenance_tracker: MaintenanceTracker = None

def init_advanced_features():
    """Initialize advanced feature modules."""
    global swarm_coordinator, ai_detector, ai_tracker
    global terrain_classifier, anomaly_detector
    global aerial_mapper, inspector, search_generator
    global remote_id, flight_recorder, maintenance_tracker

    # Swarm
    swarm_coordinator = SwarmCoordinator(swarm_id="sim_swarm")

    # AI/ML
    ai_detector = YOLODetector()
    ai_tracker = DetectionTracker()
    terrain_classifier = TerrainClassifier()
    anomaly_detector = AnomalyDetector()

    # Applications
    search_generator = SearchPatternGenerator()

    # Compliance
    remote_id = RemoteID(RemoteIDConfig(ua_id="SIM-DRONE-001"))
    flight_recorder = FlightRecorder()
    maintenance_tracker = MaintenanceTracker()
    maintenance_tracker.register_drone("sim_drone_1")

    print("[init] Advanced features initialized")


def create_drone():
    """Create and initialize a new drone instance."""
    print("[create_drone] Starting drone creation...", flush=True)

    try:
        drone = Drone(enable_logging=True)
        print("[create_drone] Drone instance created", flush=True)
    except Exception as e:
        print(f"[create_drone] ERROR creating Drone: {e}", flush=True)
        raise

    try:
        # Set initial positions (Berlin area) - drone starts at home
        drone.set_home_position(52.5200, 13.4050)
        drone.set_current_position(52.5200, 13.4050)  # Start at home to avoid geofence breach
        drone.gps_sensor.set_position(52.5200, 13.4050, 50.0)
        drone.gps_sensor.set_speed(0.0)
        drone.gps_sensor.set_heading(180.0)
        print("[create_drone] Positions set", flush=True)
    except Exception as e:
        print(f"[create_drone] ERROR setting positions: {e}", flush=True)
        raise

    try:
        # Set initial sensor values
        drone.battery_sensor.battery_level = 85.0
        drone.wind_sensor.wind_speed = 5.0
        drone.wind_sensor.wind_direction = 270.0
        print("[create_drone] Sensor values set", flush=True)
    except Exception as e:
        print(f"[create_drone] ERROR setting sensor values: {e}", flush=True)
        raise

    try:
        # Setup geofence around home position
        drone.setup_home_geofence(radius=1000.0, max_altitude=120.0)
        print("[create_drone] Geofence set", flush=True)
    except Exception as e:
        print(f"[create_drone] ERROR setting geofence: {e}", flush=True)
        raise

    try:
        # Enable props-off mode for simulation
        drone.set_props_off_mode(True)
        print("[create_drone] Props-off mode enabled", flush=True)
    except Exception as e:
        print(f"[create_drone] ERROR setting props-off: {e}", flush=True)
        raise

    try:
        # Confirm manual checks for simulation
        drone.confirm_manual_check(PreArmCheck.PROP_CHECK, True)
        drone.confirm_manual_check(PreArmCheck.AIRSPACE_CLEAR, True)
        print("[create_drone] Manual checks confirmed", flush=True)
    except Exception as e:
        print(f"[create_drone] ERROR confirming manual checks: {e}", flush=True)
        raise

    try:
        # Start core sensors
        drone.gps_sensor.start()
        drone.imu_sensor.start()
        drone.imu_sensor.set_calibration_complete()  # Assume calibrated for simulation
        print("[create_drone] Core sensors started", flush=True)
    except Exception as e:
        print(f"[create_drone] ERROR starting core sensors: {e}", flush=True)
        raise

    try:
        # Start navigation sensors
        drone.lidar_sensor.start()
        drone.visual_odometry.start()
        print("[create_drone] Navigation sensors started", flush=True)
    except Exception as e:
        print(f"[create_drone] ERROR starting navigation sensors: {e}", flush=True)
        raise

    try:
        # Enable navigation systems
        drone.enable_lidar_slam(True)
        drone.enable_vio(True)
        print("[create_drone] Navigation systems enabled", flush=True)
    except Exception as e:
        print(f"[create_drone] ERROR enabling navigation systems: {e}", flush=True)
        raise

    try:
        # Initialize navigation systems
        drone.lidar_slam.initialize()
        drone.vio_fusion.initialize()
        print("[create_drone] Navigation systems initialized", flush=True)
    except Exception as e:
        print(f"[create_drone] ERROR initializing navigation systems: {e}", flush=True)
        raise

    print("[create_drone] Drone creation complete!", flush=True)
    return drone


def get_drone_state():
    """Get current drone state as dictionary."""
    drone = simulation['drone']
    if not drone:
        return {}

    # Get failsafe status
    failsafe_status = drone.get_failsafe_status()
    active_failsafes = failsafe_status.get('active_failsafes', [])

    # Get arming status
    arming_status = drone.get_arming_status()

    # Get landing status
    landing_status = drone.get_landing_status()

    # Get geofence status (if configured)
    geofence_status = None
    if len(drone.geofence_manager.get_all_fences()) > 0:
        geofence_status = drone.check_geofence()

    # Get health summary
    health_summary = drone.get_health_summary()

    # Get mission progress if available
    mission_progress = None
    if hasattr(drone, 'flight_controller'):
        auto_handler = drone.flight_controller._mode_handlers.get('auto')
        if auto_handler:
            mission_status = auto_handler.get_status()
            mission_progress = {
                'active': mission_status.get('mission_state') == 'running',
                'state': mission_status.get('mission_state'),
                'name': mission_status.get('mission_name'),
                'progress': mission_status.get('progress', 0),
                'current_waypoint': mission_status.get('current_waypoint', 0),
                'total_waypoints': mission_status.get('total_waypoints', 0),
            }

    return {
        'position': {
            'lat': drone.current_position[0],
            'lon': drone.current_position[1],
            'alt': drone.current_altitude,
        },
        'home': {
            'lat': drone.home_position[0],
            'lon': drone.home_position[1],
        },
        'gps': {
            'satellites': drone.gps_sensor.get_satellites(),
            'fix_type': drone.gps_sensor._fix_type,
            'accuracy': drone.gps_sensor.get_accuracy()[0],
            'speed': drone.gps_sensor.get_speed(),
            'heading': drone.gps_sensor.get_heading(),
        },
        'battery': drone.battery_level,
        'wind': {
            'speed': drone.wind_sensor.wind_speed,
            'direction': drone.wind_sensor.wind_direction,
        },
        'failsafe': {
            'state': failsafe_status.get('state', 'normal'),
            'active': [f['type'] for f in active_failsafes],
        },
        'arming': arming_status,
        'landing': landing_status,
        'geofence': geofence_status,
        'health': health_summary,
        'camera': drone.get_camera_status(),
        'lidar': drone.get_lidar_status() if hasattr(drone, 'get_lidar_status') else None,
        'lidar_slam': drone.get_lidar_slam_status() if hasattr(drone, 'get_lidar_slam_status') else None,
        'visual_odometry': drone.get_visual_odometry_status() if hasattr(drone, 'get_visual_odometry_status') else None,
        'vio': drone.get_vio_status() if hasattr(drone, 'get_vio_status') else None,
        'is_flying': drone.is_flying,
        'is_armed': drone.is_armed,
        'is_returning_home': drone.is_returning_home,
        'is_landing': drone.is_landing,
        'mission': mission_progress,
        'timestamp': time.time(),
    }


def simulation_loop():
    """Main simulation loop running in background thread."""
    drone = simulation['drone']

    # Simulation parameters
    update_interval = 0.1  # 10 Hz base rate
    battery_drain_rate = 0.02  # % per second

    # Flight path (if returning home)
    waypoints = []
    current_waypoint_idx = 0

    while simulation['running']:
        speed = simulation['speed']
        scenario = simulation['scenario']

        # Apply scenario effects
        if scenario == 'low_battery':
            battery_drain_rate = 0.5
        elif scenario == 'gps_loss':
            if drone.gps_sensor.get_satellites() > 2:
                drone.gps_sensor.set_satellites(drone.gps_sensor.get_satellites() - 1)
        elif scenario == 'high_wind':
            drone.wind_sensor.wind_speed = min(20, drone.wind_sensor.wind_speed + 0.1)
        elif scenario == 'obstacle':
            # Simulate obstacle at random intervals
            pass

        # Drain battery
        drone.battery_sensor.battery_level = max(0, drone.battery_sensor.battery_level - battery_drain_rate * update_interval * speed)
        drone.battery_level = drone.battery_sensor.battery_level

        # Simulate movement if flying
        if drone.is_flying and waypoints and current_waypoint_idx < len(waypoints):
            target = waypoints[current_waypoint_idx]
            current_lat, current_lon = drone.current_position

            # Simple linear interpolation towards waypoint
            move_speed = 0.0001 * speed  # degrees per update

            dlat = target[0] - current_lat
            dlon = target[1] - current_lon
            dist = (dlat**2 + dlon**2)**0.5

            if dist < move_speed:
                # Reached waypoint
                drone.set_current_position(target[0], target[1])
                drone.gps_sensor.set_position(target[0], target[1], drone.current_altitude)
                current_waypoint_idx += 1

                # Emit waypoint reached event
                socketio.emit('waypoint_reached', {
                    'index': current_waypoint_idx,
                    'position': {'lat': target[0], 'lon': target[1]},
                    'remaining': len(waypoints) - current_waypoint_idx,
                })
            else:
                # Move towards waypoint
                new_lat = current_lat + (dlat / dist) * move_speed
                new_lon = current_lon + (dlon / dist) * move_speed
                drone.set_current_position(new_lat, new_lon)
                drone.gps_sensor.set_position(new_lat, new_lon, drone.current_altitude)

                # Update heading
                import math
                heading = math.degrees(math.atan2(dlon, dlat))
                if heading < 0:
                    heading += 360
                drone.gps_sensor.set_heading(heading)
                drone.gps_sensor.set_speed(12.0 * speed)

        # Update navigation sensors
        try:
            # Update LiDAR
            if drone.lidar_sensor._is_connected:
                lidar_reading = drone.lidar_sensor.update()
                if drone._lidar_slam_enabled and lidar_reading.is_valid():
                    drone.lidar_slam.process_scan(lidar_reading)

            # Update Visual Odometry and VIO
            if drone.visual_odometry._is_connected:
                vo_reading = drone.visual_odometry.update()
                if drone._vio_enabled and vo_reading.is_valid():
                    drone.vio_fusion.process_vo(vo_reading)

            # Update IMU for VIO
            if drone._vio_enabled:
                imu_reading = drone.imu_sensor.update()
                if imu_reading.is_valid():
                    drone.vio_fusion.process_imu(imu_reading)
        except Exception as e:
            pass  # Silently handle sensor update errors during simulation

        # Check failsafes
        drone.failsafe_manager.check_battery(drone.battery_level)
        drone.failsafe_manager.check_gps(
            drone.gps_sensor.has_fix(),
            drone.gps_sensor.get_satellites(),
            drone.gps_sensor.get_accuracy()[0],
        )

        # Emit state update
        try:
            state = get_drone_state()
            if state:
                socketio.emit('drone_state', state)
        except Exception as e:
            print(f"Error getting drone state: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(update_interval)

    # Simulation stopped
    socketio.emit('simulation_stopped', {})


# Routes
@app.route('/')
def index():
    """Render main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/state')
def api_state():
    """Get current drone state."""
    return jsonify(get_drone_state())


@app.route('/api/simulation/status')
def api_simulation_status():
    """Get simulation status."""
    return jsonify({
        'running': simulation['running'],
        'speed': simulation['speed'],
        'scenario': simulation['scenario'],
    })


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connected', {'status': 'ok'})
    if simulation['drone']:
        emit('drone_state', get_drone_state())


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


@socketio.on('start_simulation')
def handle_start_simulation(data=None):
    """Start the simulation."""
    print(f"Received start_simulation event with data: {data}")
    if simulation['running']:
        emit('error', {'message': 'Simulation already running'})
        return

    # Create new drone
    try:
        print("Creating drone...")
        simulation['drone'] = create_drone()
        print("Drone created successfully")
    except Exception as e:
        print(f"Error creating drone: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': f'Failed to create drone: {str(e)}'})
        return
    simulation['running'] = True
    simulation['speed'] = data.get('speed', 1.0) if data else 1.0
    simulation['scenario'] = data.get('scenario', 'normal') if data else 'normal'

    # Start simulation thread
    simulation['thread'] = threading.Thread(target=simulation_loop, daemon=True)
    simulation['thread'].start()

    emit('simulation_started', {
        'speed': simulation['speed'],
        'scenario': simulation['scenario'],
    })
    print(f"Simulation started: speed={simulation['speed']}, scenario={simulation['scenario']}")


@socketio.on('stop_simulation')
def handle_stop_simulation(data=None):
    """Stop the simulation."""
    print(f"[stop_simulation] Received stop request, current running state: {simulation['running']}", flush=True)

    was_running = simulation['running']
    simulation['running'] = False
    simulation['thread'] = None

    # Broadcast to all clients
    socketio.emit('simulation_stopped', {})
    print(f"[stop_simulation] Simulation stopped (was running: {was_running})", flush=True)

    # Return acknowledgment for callback
    return {'success': True, 'was_running': was_running}


@socketio.on('set_speed')
def handle_set_speed(data):
    """Set simulation speed."""
    simulation['speed'] = float(data.get('speed', 1.0))
    emit('speed_changed', {'speed': simulation['speed']})


@socketio.on('set_scenario')
def handle_set_scenario(data):
    """Set simulation scenario."""
    simulation['scenario'] = data.get('scenario', 'normal')
    emit('scenario_changed', {'scenario': simulation['scenario']})


@socketio.on('start_flight')
def handle_start_flight(data=None):
    """Start drone flight."""
    drone = simulation['drone']
    if drone:
        drone._is_flying = True
        drone.start_flight()
        emit('flight_started', {})


@socketio.on('return_to_home')
def handle_return_to_home(data=None):
    """Command drone to return to home."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    try:
        # Calculate path
        drone.wind_sensor.wind_speed = drone.wind_sensor.wind_speed  # Use current value
        drone._battery_level_override = drone.battery_level

        # Get waypoints
        from core.path_planner import PathPlanner, Position, State
        planner = PathPlanner()
        current_pos = Position(drone.current_position[0], drone.current_position[1])
        home_pos = Position(drone.home_position[0], drone.home_position[1])
        state = State(battery_percent=drone.battery_level, wind_speed_m_s=drone.wind_sensor.wind_speed)

        path = planner.plan_return(home_pos, current_pos, state)
        waypoints = [(p.latitude, p.longitude) for p in path]

        drone._is_returning_home = True

        emit('return_to_home_started', {
            'waypoints': [{'lat': w[0], 'lon': w[1]} for w in waypoints],
        })

    except Exception as e:
        emit('error', {'message': str(e)})


@socketio.on('trigger_failsafe')
def handle_trigger_failsafe(data):
    """Manually trigger a failsafe for testing."""
    drone = simulation['drone']
    if not drone:
        return

    failsafe_type = data.get('type')

    if failsafe_type == 'low_battery':
        drone.battery_sensor.battery_level = 20.0
        drone.battery_level = 20.0
    elif failsafe_type == 'critical_battery':
        drone.battery_sensor.battery_level = 5.0
        drone.battery_level = 5.0
    elif failsafe_type == 'gps_loss':
        drone.gps_sensor.simulate_signal_loss()
    elif failsafe_type == 'signal_loss':
        drone._signal_strength = 0.0
        drone._last_command_time = time.time() - 10

    emit('failsafe_triggered', {'type': failsafe_type})


@socketio.on('set_position')
def handle_set_position(data):
    """Set drone position manually."""
    drone = simulation['drone']
    if drone:
        lat = float(data.get('lat', drone.current_position[0]))
        lon = float(data.get('lon', drone.current_position[1]))
        drone.set_current_position(lat, lon)
        drone.gps_sensor.set_position(lat, lon, drone.current_altitude)
        emit('position_updated', {'lat': lat, 'lon': lon})


@socketio.on('arm_drone')
def handle_arm_drone(data=None):
    """Arm the drone."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    force = data.get('force', False) if data else False
    drone.set_safety_switch(True)  # Turn off safety switch

    success, message = drone.arm(force=force)
    emit('arm_result', {'success': success, 'message': message})
    if success:
        emit('armed', {})


@socketio.on('disarm_drone')
def handle_disarm_drone(data=None):
    """Disarm the drone."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    reason = data.get('reason', 'user_request') if data else 'user_request'
    success, message = drone.disarm(reason)
    emit('disarm_result', {'success': success, 'message': message})
    if success:
        emit('disarmed', {'reason': reason})


@socketio.on('kill_switch')
def handle_kill_switch(data=None):
    """Activate emergency kill switch."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    reason = data.get('reason', 'emergency') if data else 'emergency'
    success, message = drone.kill(reason)
    emit('kill_switch_result', {'success': success, 'message': message})
    if success:
        emit('killed', {'reason': reason})


@socketio.on('reset_kill_switch')
def handle_reset_kill_switch(data=None):
    """Reset the kill switch."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    success, message = drone.reset_kill_switch()
    emit('reset_kill_switch_result', {'success': success, 'message': message})
    if success:
        emit('kill_reset', {})


@socketio.on('start_landing')
def handle_start_landing(data=None):
    """Start landing sequence."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    mode_name = data.get('mode', 'normal') if data else 'normal'
    mode_map = {
        'normal': LandingMode.NORMAL,
        'precision': LandingMode.PRECISION,
        'emergency': LandingMode.EMERGENCY,
        'terrain_follow': LandingMode.TERRAIN_FOLLOW,
        'safe_zone': LandingMode.SAFE_ZONE,
    }
    mode = mode_map.get(mode_name, LandingMode.NORMAL)

    success, message = drone.start_landing(mode)
    emit('landing_result', {'success': success, 'message': message, 'mode': mode_name})


@socketio.on('abort_landing')
def handle_abort_landing(data=None):
    """Abort current landing."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    success, message = drone.abort_landing()
    emit('abort_landing_result', {'success': success, 'message': message})


@socketio.on('run_pre_arm_checks')
def handle_pre_arm_checks(data=None):
    """Run pre-arm checks and return results."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    checks = drone.run_pre_arm_checks()
    results = [
        {
            'check': c.check.name,
            'passed': c.passed,
            'message': c.message,
            'required': c.required,
        }
        for c in checks
    ]
    emit('pre_arm_checks_result', {
        'checks': results,
        'can_arm': drone.arming_manager.can_arm(),
    })


@socketio.on('get_health_status')
def handle_get_health_status(data=None):
    """Get detailed health status."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    drone.update_health_status()
    health = drone.get_health_status()
    emit('health_status', health)


# ==================== Camera Handlers ====================

@socketio.on('start_camera')
def handle_start_camera(data=None):
    """Start the camera."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    success, message = drone.start_camera()
    emit('camera_result', {'success': success, 'message': message, 'action': 'start'})

    if success:
        emit('camera_status', drone.get_camera_status())


@socketio.on('stop_camera')
def handle_stop_camera(data=None):
    """Stop the camera."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    success, message = drone.stop_camera()
    emit('camera_result', {'success': success, 'message': message, 'action': 'stop'})


@socketio.on('start_camera_streaming')
def handle_start_camera_streaming(data=None):
    """Start camera streaming."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    success, message = drone.start_camera_streaming()
    emit('camera_result', {'success': success, 'message': message, 'action': 'start_streaming'})

    if success:
        # Setup frame callback to emit frames via socket
        def on_frame(frame):
            socketio.emit('camera_frame', frame.to_dict())

        drone.camera_sensor.add_frame_callback(on_frame)


@socketio.on('stop_camera_streaming')
def handle_stop_camera_streaming(data=None):
    """Stop camera streaming."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    success, message = drone.stop_camera_streaming()
    emit('camera_result', {'success': success, 'message': message, 'action': 'stop_streaming'})


@socketio.on('set_vision_mode')
def handle_set_vision_mode(data):
    """Set camera vision mode."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    mode_name = data.get('mode', 'normal')
    mode_map = {
        'normal': VisionMode.NORMAL,
        'thermal': VisionMode.THERMAL,
        'night_vision': VisionMode.NIGHT_VISION,
        'obstacle_detection': VisionMode.OBSTACLE_DETECTION,
        'marker_detection': VisionMode.MARKER_DETECTION,
    }
    mode = mode_map.get(mode_name, VisionMode.NORMAL)

    success, message = drone.set_camera_vision_mode(mode)
    emit('camera_result', {'success': success, 'message': message, 'action': 'set_vision_mode', 'mode': mode_name})


@socketio.on('set_gimbal')
def handle_set_gimbal(data):
    """Set gimbal position."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    pitch = data.get('pitch')
    roll = data.get('roll')
    yaw = data.get('yaw')

    success, message = drone.set_gimbal_position(pitch, roll, yaw)
    emit('gimbal_result', {'success': success, 'message': message})


@socketio.on('point_camera_down')
def handle_point_camera_down(data=None):
    """Point camera down for landing."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    success, message = drone.point_camera_down()
    emit('gimbal_result', {'success': success, 'message': message, 'action': 'point_down'})


@socketio.on('point_camera_forward')
def handle_point_camera_forward(data=None):
    """Point camera forward."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    success, message = drone.point_camera_forward()
    emit('gimbal_result', {'success': success, 'message': message, 'action': 'point_forward'})


@socketio.on('start_recording')
def handle_start_recording(data=None):
    """Start video recording."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    path = data.get('path') if data else None
    success, message = drone.start_recording(path)
    emit('recording_result', {'success': success, 'message': message, 'action': 'start'})


@socketio.on('stop_recording')
def handle_stop_recording(data=None):
    """Stop video recording."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    success, message = drone.stop_recording()
    emit('recording_result', {'success': success, 'message': message, 'action': 'stop'})


@socketio.on('take_snapshot')
def handle_take_snapshot(data=None):
    """Take a camera snapshot."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    path = data.get('path') if data else None
    success, message = drone.take_snapshot(path)
    emit('snapshot_result', {'success': success, 'message': message})


@socketio.on('get_camera_status')
def handle_get_camera_status(data=None):
    """Get camera status."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    status = drone.get_camera_status()
    emit('camera_status', status)


@socketio.on('get_detections')
def handle_get_detections(data=None):
    """Get current camera detections."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    detections = drone.get_camera_detections()
    emit('detections', {'detections': detections})


@socketio.on('simulate_marker')
def handle_simulate_marker(data):
    """Simulate a landing marker (for testing)."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    visible = data.get('visible', True)
    x = data.get('x')
    y = data.get('y')
    distance = data.get('distance', 5.0)
    marker_id = data.get('marker_id', 'landing_pad')

    drone.camera_sensor.simulate_marker(visible, x, y, distance, marker_id)
    emit('marker_simulated', {'visible': visible, 'marker_id': marker_id})


@socketio.on('simulate_obstacle')
def handle_simulate_obstacle(data):
    """Simulate an obstacle (for testing)."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    from sensors.camera_sensor import DetectionType

    obstacle_type = DetectionType(data.get('type', 'obstacle'))
    x = data.get('x', 100)
    y = data.get('y', 100)
    width = data.get('width', 50)
    height = data.get('height', 50)
    distance = data.get('distance', 10.0)

    idx = drone.camera_sensor.simulate_obstacle(obstacle_type, x, y, width, height, distance)
    emit('obstacle_simulated', {'index': idx, 'type': obstacle_type.value})


# ==================== Mission Planning Handlers ====================

@socketio.on('create_mission')
def handle_create_mission(data):
    """Create a new mission."""
    global current_mission

    name = data.get('name', 'New Mission')
    description = data.get('description', '')
    mission_type = MissionType(data.get('type', 'waypoint'))

    current_mission = Mission(
        name=name,
        description=description,
        mission_type=mission_type,
        default_speed=data.get('default_speed', 5.0),
        default_altitude=data.get('default_altitude', 50.0),
        return_home=data.get('return_home', False),
    )

    emit('mission_created', {
        'id': current_mission.id,
        'name': current_mission.name,
        'mission': current_mission.to_dict(),
    })


@socketio.on('get_current_mission')
def handle_get_current_mission(data=None):
    """Get the current mission."""
    global current_mission

    if current_mission is not None:
        emit('current_mission', current_mission.to_dict())
    else:
        emit('current_mission', None)


@socketio.on('add_waypoint')
def handle_add_waypoint(data):
    """Add a waypoint to the current mission."""
    global current_mission

    if not current_mission:
        emit('error', {'message': 'No mission created'})
        return

    # Create waypoint from data
    waypoint = Waypoint(
        latitude=data.get('lat', 0.0),
        longitude=data.get('lon', 0.0),
        altitude=data.get('alt', current_mission.default_altitude),
        name=data.get('name', f'WP{len(current_mission) + 1}'),
        waypoint_type=WaypointType(data.get('type', 'navigate')),
        speed=data.get('speed', current_mission.default_speed),
        hold_time=data.get('hold_time', 0.0),
        acceptance_radius=data.get('radius', 2.0),
        pass_through=data.get('pass_through', False),
        yaw=data.get('yaw'),
    )

    # Add actions if specified
    actions = data.get('actions', [])
    for action_data in actions:
        action_item = WaypointActionItem(
            action=WaypointAction(action_data.get('action', 'none')),
            parameters=action_data.get('parameters', {}),
        )
        waypoint.add_action(action_item)

    index = current_mission.add_waypoint(waypoint)

    emit('waypoint_added', {
        'index': index,
        'waypoint': waypoint.to_dict(),
        'mission': current_mission.to_dict(),
    })


@socketio.on('update_waypoint')
def handle_update_waypoint(data):
    """Update a waypoint in the current mission."""
    global current_mission

    if not current_mission:
        emit('error', {'message': 'No mission created'})
        return

    index = data.get('index')
    if index is None or index < 0 or index >= len(current_mission):
        emit('error', {'message': 'Invalid waypoint index'})
        return

    # Get existing waypoint and update
    existing = current_mission.waypoints[index]

    waypoint = Waypoint(
        latitude=data.get('lat', existing.latitude),
        longitude=data.get('lon', existing.longitude),
        altitude=data.get('alt', existing.altitude),
        name=data.get('name', existing.name),
        waypoint_type=WaypointType(data.get('type', existing.waypoint_type.value)),
        speed=data.get('speed', existing.speed),
        hold_time=data.get('hold_time', existing.hold_time),
        acceptance_radius=data.get('radius', existing.acceptance_radius),
        pass_through=data.get('pass_through', existing.pass_through),
        yaw=data.get('yaw', existing.yaw),
    )

    current_mission.update_waypoint(index, waypoint)

    emit('waypoint_updated', {
        'index': index,
        'waypoint': waypoint.to_dict(),
        'mission': current_mission.to_dict(),
    })


@socketio.on('remove_waypoint')
def handle_remove_waypoint(data):
    """Remove a waypoint from the current mission."""
    global current_mission

    if not current_mission:
        emit('error', {'message': 'No mission created'})
        return

    index = data.get('index')
    removed = current_mission.remove_waypoint(index)

    if removed:
        emit('waypoint_removed', {
            'index': index,
            'mission': current_mission.to_dict(),
        })
    else:
        emit('error', {'message': 'Invalid waypoint index'})


@socketio.on('reorder_waypoints')
def handle_reorder_waypoints(data):
    """Reorder waypoints in the current mission."""
    global current_mission

    if not current_mission:
        emit('error', {'message': 'No mission created'})
        return

    from_index = data.get('from_index')
    to_index = data.get('to_index')

    success = current_mission.move_waypoint(from_index, to_index)

    if success:
        emit('waypoints_reordered', {
            'from': from_index,
            'to': to_index,
            'mission': current_mission.to_dict(),
        })
    else:
        emit('error', {'message': 'Invalid indices'})


@socketio.on('clear_waypoints')
def handle_clear_waypoints(data=None):
    """Clear all waypoints from current mission."""
    global current_mission

    if not current_mission:
        emit('error', {'message': 'No mission created'})
        return

    current_mission.clear_waypoints()
    emit('waypoints_cleared', {'mission': current_mission.to_dict()})


@socketio.on('reverse_mission')
def handle_reverse_mission(data=None):
    """Reverse the waypoint order."""
    global current_mission

    if not current_mission:
        emit('error', {'message': 'No mission created'})
        return

    current_mission.reverse()
    emit('mission_reversed', {'mission': current_mission.to_dict()})


@socketio.on('validate_mission')
def handle_validate_mission(data=None):
    """Validate the current mission."""
    global current_mission

    if not current_mission:
        emit('error', {'message': 'No mission created'})
        return

    drone = simulation['drone']
    home_position = None
    if drone:
        home_position = drone.home_position

    result = mission_validator.validate(
        current_mission,
        drone=drone,
        home_position=home_position,
    )

    emit('mission_validation', {
        'is_valid': result.is_valid,
        'issues': [
            {
                'severity': issue.severity.value,
                'type': issue.validation_type.value,
                'message': issue.message,
                'waypoint_index': issue.waypoint_index,
            }
            for issue in result.issues
        ],
    })


@socketio.on('save_mission')
def handle_save_mission(data=None):
    """Save the current mission to storage."""
    global current_mission

    if not current_mission:
        emit('error', {'message': 'No mission created'})
        return

    try:
        filepath = mission_storage.save(current_mission)
        emit('mission_saved', {
            'id': current_mission.id,
            'name': current_mission.name,
            'filepath': filepath,
        })
    except Exception as e:
        emit('error', {'message': f'Failed to save mission: {str(e)}'})


@socketio.on('load_mission')
def handle_load_mission(data):
    """Load a mission from storage."""
    global current_mission

    mission_id = data.get('id')
    if not mission_id:
        emit('error', {'message': 'Mission ID required'})
        return

    loaded = mission_storage.load(mission_id)
    if loaded:
        current_mission = loaded
        emit('mission_loaded', {'mission': current_mission.to_dict()})
    else:
        emit('error', {'message': 'Mission not found'})


@socketio.on('delete_mission')
def handle_delete_mission(data):
    """Delete a mission from storage."""
    mission_id = data.get('id')
    if not mission_id:
        emit('error', {'message': 'Mission ID required'})
        return

    success = mission_storage.delete(mission_id)
    emit('mission_deleted', {'success': success, 'id': mission_id})


@socketio.on('list_missions')
def handle_list_missions(data=None):
    """List all saved missions."""
    missions = mission_storage.list_missions()
    emit('missions_list', {'missions': missions})


@socketio.on('export_mission')
def handle_export_mission(data):
    """Export mission to file."""
    global current_mission

    if not current_mission:
        emit('error', {'message': 'No mission created'})
        return

    format_type = data.get('format', 'json')

    try:
        if format_type == 'kml':
            filepath = f"missions/{current_mission.id}.kml"
            success = mission_storage.export_to_kml(current_mission, filepath)
        elif format_type == 'csv':
            filepath = f"missions/{current_mission.id}.csv"
            success = mission_storage.export_to_csv(current_mission, filepath)
        else:
            filepath = mission_storage.save(current_mission)
            success = True

        if success:
            emit('mission_exported', {'format': format_type, 'filepath': filepath})
        else:
            emit('error', {'message': f'Failed to export as {format_type}'})
    except Exception as e:
        emit('error', {'message': f'Export failed: {str(e)}'})


@socketio.on('generate_survey')
def handle_generate_survey(data):
    """Generate a survey pattern mission."""
    global current_mission

    pattern_type = data.get('pattern', 'grid')
    center = (data.get('center_lat', 52.52), data.get('center_lon', 13.405))
    altitude = data.get('altitude', 50.0)

    try:
        if pattern_type == 'grid':
            current_mission = survey_generator.generate_grid(
                center=center,
                width=data.get('width', 100.0),
                height=data.get('height', 100.0),
                altitude=altitude,
                line_spacing=data.get('line_spacing', 10.0),
                angle=data.get('angle', 0.0),
                overshoot=data.get('overshoot', 5.0),
            )
        elif pattern_type == 'orbit':
            current_mission = survey_generator.generate_orbit(
                center=center,
                radius=data.get('radius', 50.0),
                altitude=altitude,
                num_points=data.get('num_points', 12),
                clockwise=data.get('clockwise', True),
            )
        elif pattern_type == 'corridor':
            path = data.get('path', [center, (center[0] + 0.001, center[1])])
            current_mission = survey_generator.generate_corridor(
                path=path,
                altitude=altitude,
                corridor_width=data.get('corridor_width', 20.0),
                line_spacing=data.get('line_spacing', 10.0),
            )
        elif pattern_type == 'expanding_square':
            current_mission = survey_generator.generate_expanding_square(
                center=center,
                altitude=altitude,
                initial_leg=data.get('initial_leg', 20.0),
                expansion=data.get('expansion', 20.0),
                max_legs=data.get('max_legs', 8),
            )
        elif pattern_type == 'sector_search':
            current_mission = survey_generator.generate_sector_search(
                center=center,
                altitude=altitude,
                radius=data.get('radius', 100.0),
                num_sectors=data.get('num_sectors', 6),
            )
        elif pattern_type == 'crosshatch':
            current_mission = survey_generator.generate_crosshatch(
                center=center,
                width=data.get('width', 100.0),
                height=data.get('height', 100.0),
                altitude=altitude,
                line_spacing=data.get('line_spacing', 15.0),
            )
        else:
            emit('error', {'message': f'Unknown pattern type: {pattern_type}'})
            return

        emit('survey_generated', {
            'pattern': pattern_type,
            'mission': current_mission.to_dict(),
        })
    except Exception as e:
        emit('error', {'message': f'Failed to generate survey: {str(e)}'})


@socketio.on('upload_mission')
def handle_upload_mission(data=None):
    """Upload mission to drone for execution."""
    global current_mission

    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    if not current_mission:
        emit('error', {'message': 'No mission created'})
        return

    if len(current_mission.waypoints) == 0:
        emit('error', {'message': 'Mission has no waypoints'})
        return

    # Load mission into flight controller auto mode
    if hasattr(drone, 'flight_controller'):
        auto_handler = drone.flight_controller._mode_handlers.get('auto')
        if auto_handler:
            success, message = auto_handler.load_mission(current_mission)
            if success:
                current_mission.state = MissionState.UPLOADED
                emit('mission_uploaded', {
                    'success': True,
                    'message': message,
                    'mission': current_mission.to_dict(),
                })
            else:
                emit('error', {'message': message})
        else:
            emit('error', {'message': 'Auto mode handler not available'})
    else:
        emit('error', {'message': 'Flight controller not available'})


@socketio.on('start_mission')
def handle_start_mission(data=None):
    """Start mission execution."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    if hasattr(drone, 'flight_controller'):
        from core.flight_modes import FlightMode

        # Switch to AUTO mode
        success, message = drone.flight_controller.set_mode(FlightMode.AUTO)

        if success:
            emit('mission_started', {'success': True, 'message': message})
        else:
            emit('error', {'message': message})
    else:
        emit('error', {'message': 'Flight controller not available'})


@socketio.on('pause_mission')
def handle_pause_mission(data=None):
    """Pause mission execution."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    if hasattr(drone, 'flight_controller'):
        auto_handler = drone.flight_controller._mode_handlers.get('auto')
        if auto_handler:
            success, message = auto_handler.pause_mission()
            emit('mission_paused', {'success': success, 'message': message})
        else:
            emit('error', {'message': 'Auto mode handler not available'})
    else:
        emit('error', {'message': 'Flight controller not available'})


@socketio.on('resume_mission')
def handle_resume_mission(data=None):
    """Resume mission execution."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    if hasattr(drone, 'flight_controller'):
        auto_handler = drone.flight_controller._mode_handlers.get('auto')
        if auto_handler:
            success, message = auto_handler.resume_mission()
            emit('mission_resumed', {'success': success, 'message': message})
        else:
            emit('error', {'message': 'Auto mode handler not available'})
    else:
        emit('error', {'message': 'Flight controller not available'})


@socketio.on('abort_mission')
def handle_abort_mission(data=None):
    """Abort mission execution."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    if hasattr(drone, 'flight_controller'):
        auto_handler = drone.flight_controller._mode_handlers.get('auto')
        if auto_handler:
            success, message = auto_handler.abort_mission()
            # Switch to LOITER mode
            from core.flight_modes import FlightMode
            drone.flight_controller.set_mode(FlightMode.LOITER)
            emit('mission_aborted', {'success': success, 'message': message})
        else:
            emit('error', {'message': 'Auto mode handler not available'})
    else:
        emit('error', {'message': 'Flight controller not available'})


@socketio.on('skip_waypoint')
def handle_skip_waypoint(data=None):
    """Skip to next waypoint in mission."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    if hasattr(drone, 'flight_controller'):
        auto_handler = drone.flight_controller._mode_handlers.get('auto')
        if auto_handler:
            success, message = auto_handler.skip_waypoint()
            emit('waypoint_skipped', {'success': success, 'message': message})
        else:
            emit('error', {'message': 'Auto mode handler not available'})
    else:
        emit('error', {'message': 'Flight controller not available'})


@socketio.on('goto_waypoint')
def handle_goto_waypoint(data):
    """Jump to specific waypoint in mission."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    index = data.get('index', 0)

    if hasattr(drone, 'flight_controller'):
        auto_handler = drone.flight_controller._mode_handlers.get('auto')
        if auto_handler:
            success, message = auto_handler.goto_waypoint(index)
            emit('waypoint_goto', {'success': success, 'message': message, 'index': index})
        else:
            emit('error', {'message': 'Auto mode handler not available'})
    else:
        emit('error', {'message': 'Flight controller not available'})


@socketio.on('get_mission_progress')
def handle_get_mission_progress(data=None):
    """Get current mission progress."""
    drone = simulation['drone']
    if not drone:
        emit('mission_progress', {'active': False})
        return

    if hasattr(drone, 'flight_controller'):
        auto_handler = drone.flight_controller._mode_handlers.get('auto')
        if auto_handler:
            status = auto_handler.get_status()
            emit('mission_progress', {
                'active': status.get('mission_state') == 'running',
                'state': status.get('mission_state'),
                'progress': status.get('progress', 0),
                'current_waypoint': status.get('current_waypoint', 0),
                'total_waypoints': status.get('total_waypoints', 0),
                'waypoints_completed': status.get('waypoints_completed', 0),
                'distance_to_waypoint': status.get('distance_to_waypoint', 0),
                'mission_time': status.get('mission_time', 0),
            })
        else:
            emit('mission_progress', {'active': False})
    else:
        emit('mission_progress', {'active': False})


# ==================== Geofence Handlers ====================

@socketio.on('upload_geofences')
def handle_upload_geofences(data):
    """Upload geofence definitions."""
    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    geofences = data.get('geofences', [])

    try:
        from core.geofence import (
            GeofenceConfig,
            GeofenceType,
            GeofencePoint,
        )

        # Add new geofences
        for i, gf in enumerate(geofences):
            if gf['type'] == 'circle':
                config = GeofenceConfig(
                    name=f"user_circle_{i}",
                    fence_type=GeofenceType.CIRCULAR,
                    center=(gf['center_lat'], gf['center_lon']),
                    radius=gf['radius'],
                    is_inclusion=gf.get('is_inclusion', True),
                )
                drone.geofence_manager.add_fence(config)
            elif gf['type'] == 'polygon':
                # Handle both {lat, lon} objects and [lat, lon] arrays
                vertices = []
                for v in gf['vertices']:
                    if isinstance(v, dict):
                        vertices.append(GeofencePoint(v['lat'], v['lon']))
                    else:
                        vertices.append(GeofencePoint(v[0], v[1]))
                config = GeofenceConfig(
                    name=f"user_polygon_{i}",
                    fence_type=GeofenceType.POLYGON,
                    vertices=vertices,
                    is_inclusion=gf.get('is_inclusion', True),
                )
                drone.geofence_manager.add_fence(config)

        emit('geofences_uploaded', {
            'count': len(geofences),
            'success': True
        })

    except Exception as e:
        emit('error', {'message': f'Failed to upload geofences: {str(e)}'})


@socketio.on('get_geofences')
def handle_get_geofences(data=None):
    """Get current geofence configuration."""
    drone = simulation['drone']
    if not drone:
        emit('geofence_list', {'fences': []})
        return

    from core.geofence import GeofenceType

    fences = drone.geofence_manager.get_all_fences()
    fence_data = []
    for name, fence in fences.items():
        fence_info = {
            'id': name,
            'is_inclusion': fence.is_inclusion,
            'enabled': fence.enabled,
        }
        if fence.fence_type == GeofenceType.CIRCULAR:
            fence_info['type'] = 'circle'
            fence_info['center_lat'] = fence.center[0]
            fence_info['center_lon'] = fence.center[1]
            fence_info['radius'] = fence.radius
        elif fence.fence_type == GeofenceType.POLYGON:
            fence_info['type'] = 'polygon'
            fence_info['vertices'] = [
                (v.latitude, v.longitude) for v in fence.vertices
            ]
        fence_data.append(fence_info)

    emit('geofence_list', {'fences': fence_data})


# ==================== Takeoff Handlers ====================

@socketio.on('start_takeoff')
def handle_start_takeoff(data=None):
    """Start automated takeoff sequence."""
    global takeoff_manager

    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    # Initialize takeoff manager if needed
    if takeoff_manager is None:
        takeoff_manager = TakeoffManager(drone)

    # Get parameters
    target_altitude = 10.0
    mode = TakeoffMode.NORMAL

    if data:
        target_altitude = data.get('altitude', 10.0)
        mode_name = data.get('mode', 'normal')
        mode_map = {
            'normal': TakeoffMode.NORMAL,
            'quick': TakeoffMode.QUICK,
            'precision': TakeoffMode.PRECISION,
            'vertical': TakeoffMode.VERTICAL,
        }
        mode = mode_map.get(mode_name, TakeoffMode.NORMAL)

    success, message = takeoff_manager.start_takeoff(target_altitude, mode)

    emit('takeoff_result', {
        'success': success,
        'message': message,
        'target_altitude': target_altitude,
        'mode': mode.value,
    })


@socketio.on('abort_takeoff')
def handle_abort_takeoff(data=None):
    """Abort takeoff sequence."""
    global takeoff_manager

    if takeoff_manager is None:
        emit('error', {'message': 'No takeoff in progress'})
        return

    reason = data.get('reason', 'User abort') if data else 'User abort'
    success, message = takeoff_manager.abort(reason)

    emit('takeoff_aborted', {
        'success': success,
        'message': message,
    })


@socketio.on('get_takeoff_status')
def handle_get_takeoff_status(data=None):
    """Get current takeoff status."""
    global takeoff_manager

    if takeoff_manager is None:
        emit('takeoff_status', {'active': False})
        return

    status = takeoff_manager.get_status()
    emit('takeoff_status', status)


@socketio.on('configure_takeoff')
def handle_configure_takeoff(data):
    """Configure takeoff parameters."""
    global takeoff_manager

    drone = simulation['drone']
    if not drone:
        emit('error', {'message': 'No drone initialized'})
        return

    if takeoff_manager is None:
        takeoff_manager = TakeoffManager(drone)

    config = TakeoffConfig(
        target_altitude=data.get('target_altitude', 10.0),
        climb_rate=data.get('climb_rate', 2.0),
        max_climb_rate=data.get('max_climb_rate', 5.0),
        motor_spool_time=data.get('motor_spool_time', 2.0),
        hover_check_time=data.get('hover_check_time', 3.0),
        min_battery=data.get('min_battery', 25.0),
        max_wind=data.get('max_wind', 10.0),
        require_gps=data.get('require_gps', True),
        min_satellites=data.get('min_satellites', 6),
    )

    takeoff_manager.configure(config)

    emit('takeoff_configured', {
        'success': True,
        'config': {
            'target_altitude': config.target_altitude,
            'climb_rate': config.climb_rate,
            'min_battery': config.min_battery,
            'max_wind': config.max_wind,
        }
    })


# ==================== Swarm Handlers ====================

@socketio.on('swarm_register_drone')
def handle_swarm_register(data):
    """Register a drone in the swarm."""
    drone_id = data.get('drone_id', f"drone_{time.time():.0f}")
    name = data.get('name', '')

    if swarm_coordinator:
        info = swarm_coordinator.register_drone(drone_id, name)
        emit('swarm_drone_registered', {
            'drone_id': drone_id,
            'name': name,
            'success': info is not None
        })


@socketio.on('swarm_set_formation')
def handle_swarm_formation(data):
    """Set swarm formation."""
    formation = data.get('formation', 'v_formation')
    spacing = data.get('spacing', 10.0)

    if swarm_coordinator:
        try:
            formation_type = FormationType(formation)
            success = swarm_coordinator.set_formation(
                formation_type, spacing
            )
            emit('swarm_formation_set', {
                'formation': formation,
                'success': success
            })
        except ValueError:
            emit('error', {'message': f'Invalid formation: {formation}'})


@socketio.on('swarm_start_mission')
def handle_swarm_mission(data):
    """Start a swarm mission."""
    if swarm_coordinator:
        mission = SwarmMission(
            mission_id=f"mission_{time.time():.0f}",
            name=data.get('name', 'Swarm Mission'),
            waypoints=data.get('waypoints', []),
            formation=FormationType(
                data.get('formation', 'v_formation')
            ),
            spacing=data.get('spacing', 10.0),
        )
        success = swarm_coordinator.start_mission(mission)
        emit('swarm_mission_started', {'success': success})


@socketio.on('get_swarm_status')
def handle_get_swarm_status(data=None):
    """Get current swarm status."""
    if swarm_coordinator:
        drones = swarm_coordinator.get_drones()
        emit('swarm_status', {
            'state': swarm_coordinator.state.value,
            'drone_count': swarm_coordinator.drone_count,
            'online_count': swarm_coordinator.online_count,
            'drones': [d.to_dict() for d in drones],
        })


# ==================== AI Detection Handlers ====================

@socketio.on('ai_detect_objects')
def handle_ai_detect(data):
    """Run object detection on current camera frame."""
    if ai_detector and simulation['drone']:
        # In real impl, would get camera frame
        # For simulation, generate mock detections
        detections, _ = ai_detector.detect(None)

        # Update tracker
        if ai_tracker:
            tracks = ai_tracker.update(detections)

        emit('ai_detections', {
            'detections': [d.to_dict() for d in detections],
            'tracks': [t.to_dict() for t in tracks] if ai_tracker else [],
            'inference_ms': ai_detector.last_inference_time,
        })


@socketio.on('ai_analyze_terrain')
def handle_ai_terrain(data):
    """Analyze terrain for landing zones."""
    if terrain_classifier:
        zones = terrain_classifier.find_landing_zones(
            None,  # Would be camera image
            drone_altitude=data.get('altitude', 30.0),
        )
        emit('ai_terrain_analysis', {
            'zones': [z.to_dict() for z in zones],
            'best_zone': zones[0].to_dict() if zones else None,
        })


@socketio.on('ai_check_anomalies')
def handle_ai_anomalies(data):
    """Check for system anomalies."""
    if anomaly_detector and simulation['drone']:
        drone = simulation['drone']

        # Feed sensor data to anomaly detector
        anomalies = anomaly_detector.update_batch({
            'battery_voltage': drone.battery_sensor.voltage,
            'battery_temp': 35.0,  # Simulated
            'motor_1_rpm': 5000,
            'motor_1_current': 10.0,
        })

        emit('ai_anomalies', {
            'anomalies': [a.to_dict() for a in anomalies],
            'health_score': anomaly_detector.get_health_score(),
            'stats': anomaly_detector.get_stats(),
        })


@socketio.on('get_ai_stats')
def handle_get_ai_stats(data=None):
    """Get AI module statistics."""
    stats = {
        'detector': ai_detector.get_stats() if ai_detector else {},
        'tracker': ai_tracker.get_stats() if ai_tracker else {},
        'terrain': terrain_classifier.get_stats() if terrain_classifier else {},
        'anomaly': anomaly_detector.get_stats() if anomaly_detector else {},
    }
    emit('ai_stats', stats)


# Simple frontend-friendly AI handlers
@socketio.on('detect_objects')
def handle_simple_detect(data=None):
    """Run detection from dashboard."""
    if ai_detector:
        detections, _ = ai_detector.detect(None)
        emit('detection_results', {
            'count': len(detections),
            'detections': [
                {'class_name': d.class_name, 'confidence': d.confidence}
                for d in detections[:10]
            ],
        })


@socketio.on('analyze_terrain')
def handle_simple_terrain(data=None):
    """Analyze terrain from dashboard."""
    if terrain_classifier:
        # Use classify_image with simulated data
        result = terrain_classifier.classify_image(None)
        if result:
            # Find landing zones and determine quality
            zones = terrain_classifier.find_landing_zones(result)
            best_zone = terrain_classifier.get_best_landing_zone(zones) if zones else None
            quality = best_zone.quality.value if best_zone else 'Poor'
        else:
            quality = 'Good'  # Simulated default
        emit('terrain_analysis', {'landing_quality': quality})
    else:
        emit('terrain_analysis', {'landing_quality': 'Unknown'})


# ==================== Application Handlers ====================

@socketio.on('app_start_mapping')
def handle_start_mapping(data):
    """Start aerial mapping application."""
    global aerial_mapper

    boundary = data.get('boundary', [])
    if len(boundary) < 3:
        emit('error', {'message': 'Mapping requires at least 3 boundary points'})
        return

    config = MappingConfig(
        altitude=data.get('altitude', 50.0),
        overlap_front=data.get('overlap_front', 75.0),
        overlap_side=data.get('overlap_side', 65.0),
    )
    aerial_mapper = AerialMapper(config)
    aerial_mapper.set_boundary(boundary)

    if aerial_mapper.initialize():
        aerial_mapper.start()
        emit('app_mapping_started', {
            'waypoints': aerial_mapper.get_all_waypoints(),
            'area_sqm': aerial_mapper.area,
        })
    else:
        emit('error', {'message': 'Failed to initialize mapping'})


@socketio.on('app_start_inspection')
def handle_start_inspection(data):
    """Start inspection application."""
    global inspector

    target = data.get('target')
    if not target:
        emit('error', {'message': 'Inspection requires a target'})
        return

    config = InspectionConfig(
        inspection_type=InspectionType(
            data.get('type', 'tower')
        ),
        standoff_distance=data.get('distance', 10.0),
        orbit_points=data.get('orbit_points', 8),
        vertical_levels=data.get('levels', 3),
    )
    inspector = Inspector(config)
    inspector.set_target(
        (target['lat'], target['lon'], target.get('alt', 0)),
        height=data.get('height', 30.0)
    )

    if inspector.initialize():
        inspector.start()
        emit('app_inspection_started', {
            'points': len(inspector._inspection_points),
            'type': config.inspection_type.value,
        })


@socketio.on('app_generate_search')
def handle_generate_search(data):
    """Generate search pattern."""
    if search_generator:
        center = (data.get('lat', 52.52), data.get('lon', 13.405))
        pattern_type = SearchType(data.get('type', 'expanding_square'))

        pattern = search_generator.generate(
            pattern_type,
            center,
            radius=data.get('radius', 500),
        )

        emit('app_search_pattern', {
            'type': pattern.pattern_type.value,
            'waypoints': pattern.waypoints,
            'total_distance': pattern.total_distance,
            'estimated_time': pattern.estimated_time,
            'coverage': pattern.coverage_probability,
        })


@socketio.on('get_app_status')
def handle_get_app_status(data=None):
    """Get application statuses."""
    status = {
        'mapping': aerial_mapper.get_stats() if aerial_mapper else None,
        'inspection': inspector.get_stats() if inspector else None,
    }
    emit('app_status', status)


# Simple frontend-friendly handlers for applications
@socketio.on('start_mapping')
def handle_simple_mapping(data):
    """Start mapping from dashboard."""
    if aerial_mapper:
        center_lat = data.get('center_lat', 52.52)
        center_lon = data.get('center_lon', 13.405)
        width = data.get('width', 200)
        height = data.get('height', 150)
        alt = data.get('altitude', 40)

        # Create simple rectangular boundary
        dlat = (height / 2) / 111000
        dlon = (width / 2) / 111000

        boundary = [
            (center_lat - dlat, center_lon - dlon),
            (center_lat - dlat, center_lon + dlon),
            (center_lat + dlat, center_lon + dlon),
            (center_lat + dlat, center_lon - dlon),
        ]

        waypoints = aerial_mapper.generate_survey(
            boundary, altitude=alt
        )
        emit('mapping_started', {'waypoints': len(waypoints)})


@socketio.on('start_inspection')
def handle_simple_inspection(data):
    """Start inspection from dashboard."""
    if inspector:
        target = (
            data.get('target_lat', 52.52),
            data.get('target_lon', 13.405),
            data.get('target_height', 50),
        )
        insp_type = InspectionType.TOWER

        waypoints = inspector.generate_inspection_path(
            target, insp_type
        )
        emit('inspection_path', {'waypoints': len(waypoints)})


@socketio.on('generate_search_pattern')
def handle_simple_search(data):
    """Generate search pattern from dashboard."""
    if search_generator:
        center = (
            data.get('center_lat', 52.52),
            data.get('center_lon', 13.405)
        )
        pattern_name = data.get('pattern', 'expanding_square')

        try:
            pattern_type = SearchType(pattern_name)
        except ValueError:
            pattern_type = SearchType.EXPANDING_SQUARE

        pattern = search_generator.generate(
            pattern_type,
            center,
            radius=data.get('radius', 300),
        )
        emit('search_pattern', {
            'waypoints': len(pattern.waypoints),
            'distance': pattern.total_distance,
        })


# ==================== Compliance Handlers ====================

@socketio.on('compliance_start_remote_id')
def handle_start_remote_id(data):
    """Start Remote ID broadcasting."""
    if remote_id:
        # Set operator position
        remote_id.set_operator_position(
            data.get('operator_lat', 52.52),
            data.get('operator_lon', 13.405),
        )
        remote_id.start()
        emit('compliance_remote_id_started', {'success': True})


@socketio.on('compliance_stop_remote_id')
def handle_stop_remote_id(data=None):
    """Stop Remote ID broadcasting."""
    if remote_id:
        remote_id.stop()
        emit('compliance_remote_id_stopped', {'success': True})


@socketio.on('compliance_start_recording')
def handle_start_flight_recording(data):
    """Start flight data recording."""
    if flight_recorder:
        session_id = flight_recorder.start_session(
            drone_id=data.get('drone_id', 'sim_drone'),
            pilot_id=data.get('pilot_id', 'sim_pilot'),
        )
        emit('compliance_recording_started', {'session_id': session_id})


@socketio.on('compliance_stop_recording')
def handle_stop_flight_recording(data=None):
    """Stop flight data recording."""
    if flight_recorder:
        session = flight_recorder.end_session()
        if session:
            emit('compliance_recording_stopped', {
                'session_id': session.session_id,
                'duration': session.flight_time,
                'records': session.record_count,
            })


# Simple frontend-friendly handlers for compliance
@socketio.on('start_remote_id')
def handle_simple_start_rid(data):
    """Start Remote ID from dashboard."""
    if remote_id:
        remote_id.start()
        emit('remote_id_started', {'success': True})


@socketio.on('stop_remote_id')
def handle_simple_stop_rid(data=None):
    """Stop Remote ID from dashboard."""
    if remote_id:
        remote_id.stop()
        emit('remote_id_stopped', {'success': True})


@socketio.on('flight_start_recording')
def handle_simple_start_rec(data):
    """Start flight recording from dashboard."""
    if flight_recorder:
        drone_id = data.get('drone_id', 'sim_drone')
        flight_recorder.start_session(drone_id=drone_id)
        emit('recording_started', {'success': True})


@socketio.on('flight_stop_recording')
def handle_simple_stop_rec(data=None):
    """Stop flight recording from dashboard."""
    if flight_recorder:
        session = flight_recorder.end_session()
        if session:
            emit('recording_stopped', {
                'duration': session.flight_time,
            })


@socketio.on('compliance_get_maintenance')
def handle_get_maintenance(data):
    """Get maintenance status."""
    if maintenance_tracker:
        drone_id = data.get('drone_id', 'sim_drone_1')
        due = maintenance_tracker.get_due_maintenance(drone_id)
        summary = maintenance_tracker.get_flight_summary(drone_id)

        emit('compliance_maintenance', {
            'summary': summary,
            'due_items': [item.to_dict() for item in due],
        })


@socketio.on('compliance_log_flight')
def handle_log_flight(data):
    """Log a flight for maintenance tracking."""
    if maintenance_tracker:
        maintenance_tracker.log_flight(
            drone_id=data.get('drone_id', 'sim_drone_1'),
            flight_hours=data.get('hours', 0.5),
            distance_km=data.get('distance_km', 5.0),
        )
        emit('compliance_flight_logged', {'success': True})


@socketio.on('get_compliance_status')
def handle_get_compliance_status(data=None):
    """Get overall compliance status."""
    status = {
        'remote_id': {
            'running': remote_id.is_running if remote_id else False,
            'messages': remote_id.message_count if remote_id else 0,
        },
        'recording': {
            'active': flight_recorder.is_recording if flight_recorder else False,
            'records': flight_recorder.record_count if flight_recorder else 0,
        },
        'maintenance': maintenance_tracker.get_flight_summary(
            'sim_drone_1'
        ) if maintenance_tracker else None,
    }
    emit('compliance_status', status)


@socketio.on('get_maintenance_status')
def handle_get_maintenance_status(data):
    """Get maintenance alerts for dashboard."""
    if maintenance_tracker:
        drone_id = data.get('drone_id', 'sim_drone_1')
        due = maintenance_tracker.get_due_maintenance(drone_id)
        alerts = []
        for item in due:
            critical = item.status.value in ['replace_now', 'failed']
            alerts.append({
                'component': item.component_id,
                'message': f"{item.percent_life:.0f}% used",
                'critical': critical
            })
        emit('maintenance_alerts', {'alerts': alerts})


# ==================== LLM Command Interpreter ====================

# LLM components (lazy loaded)
llm_interpreter = None

def get_llm_interpreter():
    """Get or create LLM interpreter."""
    global llm_interpreter
    if llm_interpreter is None:
        try:
            from ai.llm import CommandInterpreter, get_available_backend
            backend = get_available_backend()
            llm_interpreter = CommandInterpreter(backend=backend)
            print(f"[LLM] Interpreter initialized with {backend.name} backend")
        except Exception as e:
            print(f"[LLM] Failed to initialize interpreter: {e}")
            return None
    return llm_interpreter


@socketio.on('llm_interpret')
def handle_llm_interpret(data):
    """Interpret natural language command using LLM."""
    interpreter = get_llm_interpreter()
    if not interpreter:
        emit('error', {'message': 'LLM interpreter not available'})
        return

    command_text = data.get('command', '')
    if not command_text:
        emit('error', {'message': 'No command provided'})
        return

    # Add context from drone state
    drone = simulation['drone']
    context = {}
    if drone:
        context['current_position'] = drone.current_position
        context['altitude'] = drone.current_altitude
        context['is_flying'] = drone.is_flying
        interpreter.set_current_position(
            drone.current_position[0],
            drone.current_position[1],
            drone.current_altitude
        )
        interpreter.set_home_position(
            drone.home_position[0],
            drone.home_position[1]
        )

    try:
        result = interpreter.interpret(command_text, context)
        emit('llm_result', {
            'success': True,
            'command': result.to_dict(),
            'backend': interpreter.backend_name,
        })
    except Exception as e:
        emit('llm_result', {
            'success': False,
            'error': str(e),
        })


@socketio.on('llm_interpret_mission')
def handle_llm_interpret_mission(data):
    """Interpret natural language mission description."""
    interpreter = get_llm_interpreter()
    if not interpreter:
        emit('error', {'message': 'LLM interpreter not available'})
        return

    description = data.get('description', '')
    if not description:
        emit('error', {'message': 'No description provided'})
        return

    try:
        mission = interpreter.interpret_mission(description)
        emit('llm_mission_result', {
            'success': True,
            'mission': mission.to_dict(),
        })
    except Exception as e:
        emit('llm_mission_result', {
            'success': False,
            'error': str(e),
        })


@socketio.on('llm_get_suggestions')
def handle_llm_suggestions(data):
    """Get command suggestions for autocomplete."""
    interpreter = get_llm_interpreter()
    if not interpreter:
        emit('llm_suggestions', {'suggestions': []})
        return

    partial = data.get('partial', '')
    suggestions = interpreter.get_suggestions(partial)
    emit('llm_suggestions', {'suggestions': suggestions})


@socketio.on('llm_add_location')
def handle_llm_add_location(data):
    """Add a named location for LLM reference."""
    interpreter = get_llm_interpreter()
    if not interpreter:
        emit('error', {'message': 'LLM interpreter not available'})
        return

    name = data.get('name', '')
    lat = data.get('lat', 0)
    lon = data.get('lon', 0)

    if name:
        interpreter.add_known_location(name, lat, lon)
        emit('llm_location_added', {'name': name, 'lat': lat, 'lon': lon})


@socketio.on('get_llm_status')
def handle_get_llm_status(data=None):
    """Get LLM interpreter status."""
    interpreter = get_llm_interpreter()
    if interpreter:
        emit('llm_status', interpreter.get_status())
    else:
        emit('llm_status', {'available': False})


# ==================== Path Planning Algorithms ====================

@socketio.on('plan_path')
def handle_plan_path(data):
    """Plan a path using advanced algorithms."""
    try:
        from core.path_algorithms import (
            UnifiedPathPlanner,
            PathPlannerType,
            Point3D,
            Obstacle,
        )

        # Get parameters
        start = data.get('start', {})
        goal = data.get('goal', {})
        algorithm = data.get('algorithm', 'a_star')
        smooth = data.get('smooth', True)

        start_point = Point3D(
            start.get('x', 0),
            start.get('y', 0),
            start.get('z', 30)
        )
        goal_point = Point3D(
            goal.get('x', 100),
            goal.get('y', 100),
            goal.get('z', 30)
        )

        # Map algorithm name
        algo_map = {
            'a_star': PathPlannerType.A_STAR,
            'rrt': PathPlannerType.RRT,
            'rrt_star': PathPlannerType.RRT_STAR,
            'potential_field': PathPlannerType.POTENTIAL_FIELD,
        }
        planner_type = algo_map.get(algorithm, PathPlannerType.A_STAR)

        # Create planner and add obstacles
        planner = UnifiedPathPlanner()

        obstacles = data.get('obstacles', [])
        for obs in obstacles:
            planner.add_obstacle(Obstacle(
                center=Point3D(obs['x'], obs['y'], obs.get('z', 30)),
                radius=obs.get('radius', 5.0)
            ))

        # Plan path
        result = planner.plan(start_point, goal_point, planner_type, smooth=smooth)

        emit('path_result', {
            'success': result.success,
            'algorithm': result.algorithm,
            'path': [p.to_dict() for p in result.path],
            'cost': result.cost,
            'computation_time': result.computation_time,
            'nodes_explored': result.nodes_explored,
            'smoothed': result.smoothed,
        })

    except Exception as e:
        emit('path_result', {
            'success': False,
            'error': str(e),
        })


@socketio.on('plan_path_with_fallback')
def handle_plan_path_fallback(data):
    """Plan path with automatic algorithm fallback."""
    try:
        from core.path_algorithms import UnifiedPathPlanner, Point3D, Obstacle

        start = data.get('start', {})
        goal = data.get('goal', {})

        start_point = Point3D(start.get('x', 0), start.get('y', 0), start.get('z', 30))
        goal_point = Point3D(goal.get('x', 100), goal.get('y', 100), goal.get('z', 30))

        planner = UnifiedPathPlanner()

        obstacles = data.get('obstacles', [])
        for obs in obstacles:
            planner.add_obstacle(Obstacle(
                center=Point3D(obs['x'], obs['y'], obs.get('z', 30)),
                radius=obs.get('radius', 5.0)
            ))

        result = planner.plan_with_fallback(start_point, goal_point)

        emit('path_result', {
            'success': result.success,
            'algorithm': result.algorithm,
            'path': [p.to_dict() for p in result.path],
            'cost': result.cost,
            'computation_time': result.computation_time,
        })

    except Exception as e:
        emit('path_result', {'success': False, 'error': str(e)})


# ==================== Follow Mode ====================

follow_handler = None

def get_follow_handler():
    """Get or create follow mode handler."""
    global follow_handler
    if follow_handler is None:
        try:
            from core.flight_modes.follow_mode import FollowModeHandler
            follow_handler = FollowModeHandler()
            print("[Follow] Handler initialized")
        except Exception as e:
            print(f"[Follow] Failed to initialize: {e}")
            return None
    return follow_handler


@socketio.on('follow_start')
def handle_follow_start(data=None):
    """Start follow mode."""
    handler = get_follow_handler()
    if not handler:
        emit('error', {'message': 'Follow mode not available'})
        return

    # Configure if provided
    if data:
        handler.configure(
            distance=data.get('distance', 10.0),
            altitude=data.get('altitude', 15.0),
            max_speed=data.get('max_speed', 15.0),
        )

    success = handler.activate()
    emit('follow_started', {'success': success})


@socketio.on('follow_stop')
def handle_follow_stop(data=None):
    """Stop follow mode."""
    handler = get_follow_handler()
    if handler:
        handler.deactivate()
        emit('follow_stopped', {'success': True})


@socketio.on('follow_update_target')
def handle_follow_update_target(data):
    """Update target position for follow mode."""
    handler = get_follow_handler()
    if not handler:
        return

    detection = {
        'type': data.get('type', 'person'),
        'x': data.get('x', 0),
        'y': data.get('y', 0),
        'z': data.get('z', 0),
        'confidence': data.get('confidence', 0.9),
    }
    handler.process_detection(detection)
    emit('follow_target_updated', {'state': handler.get_state().value})


@socketio.on('follow_configure')
def handle_follow_configure(data):
    """Configure follow mode parameters."""
    handler = get_follow_handler()
    if not handler:
        emit('error', {'message': 'Follow mode not available'})
        return

    handler.configure(
        distance=data.get('distance', 10.0),
        altitude=data.get('altitude', 15.0),
        max_speed=data.get('max_speed', 15.0),
    )
    emit('follow_configured', handler.get_status())


@socketio.on('follow_set_position')
def handle_follow_set_position(data):
    """Set follow position (behind, left, right, etc)."""
    handler = get_follow_handler()
    if not handler:
        return

    from core.flight_modes.follow_mode import FollowPosition

    position_map = {
        'behind': FollowPosition.BEHIND,
        'front': FollowPosition.FRONT,
        'left': FollowPosition.LEFT,
        'right': FollowPosition.RIGHT,
        'above': FollowPosition.ABOVE,
        'orbit': FollowPosition.ORBIT,
    }

    pos_name = data.get('position', 'behind')
    position = position_map.get(pos_name, FollowPosition.BEHIND)

    handler._controller.set_follow_position(position)
    emit('follow_position_set', {'position': pos_name})


@socketio.on('get_follow_status')
def handle_get_follow_status(data=None):
    """Get follow mode status."""
    handler = get_follow_handler()
    if handler:
        emit('follow_status', handler.get_status())
    else:
        emit('follow_status', {'active': False, 'available': False})


# ==================== Predictive Maintenance ====================

predictive_maintenance = None

def get_predictive_maintenance():
    """Get or create predictive maintenance system."""
    global predictive_maintenance
    if predictive_maintenance is None:
        try:
            from maintenance.predictive_maintenance import PredictiveMaintenanceSystem
            predictive_maintenance = PredictiveMaintenanceSystem()
            print("[Maintenance] Predictive system initialized")
        except Exception as e:
            print(f"[Maintenance] Failed to initialize: {e}")
            return None
    return predictive_maintenance


@socketio.on('maintenance_analyze')
def handle_maintenance_analyze(data=None):
    """Run maintenance analysis."""
    system = get_predictive_maintenance()
    if not system:
        emit('error', {'message': 'Predictive maintenance not available'})
        return

    # Update with current drone data if available
    drone = simulation['drone']
    if drone:
        # Feed IMU data
        imu = drone.imu_sensor.get_raw_data()
        if imu:
            system.update_imu(imu.get('ax', 0), imu.get('ay', 0), imu.get('az', 9.81))

        # Feed battery data
        system.update_battery(
            voltage=drone.battery_sensor.voltage,
            current=drone.battery_sensor.current,
            temperature=35.0  # Simulated
        )

    result = system.analyze()
    emit('maintenance_analysis', result)


@socketio.on('maintenance_get_alerts')
def handle_maintenance_get_alerts(data=None):
    """Get maintenance alerts."""
    system = get_predictive_maintenance()
    if not system:
        emit('maintenance_alerts_list', {'alerts': []})
        return

    urgency = None
    if data and 'urgency' in data:
        from maintenance.predictive_maintenance import MaintenanceUrgency
        urgency_map = {
            'required': MaintenanceUrgency.REQUIRED,
            'recommended': MaintenanceUrgency.RECOMMENDED,
            'scheduled': MaintenanceUrgency.SCHEDULED,
        }
        urgency = urgency_map.get(data['urgency'])

    alerts = system.get_alerts(urgency=urgency)
    emit('maintenance_alerts_list', {
        'alerts': [a.to_dict() for a in alerts],
    })


@socketio.on('maintenance_acknowledge_alert')
def handle_maintenance_acknowledge(data):
    """Acknowledge a maintenance alert."""
    system = get_predictive_maintenance()
    if not system:
        return

    from maintenance.predictive_maintenance import ComponentType, MaintenanceAction

    component_name = data.get('component', '')
    action_name = data.get('action', '')

    try:
        component = ComponentType(component_name)
        action = MaintenanceAction(action_name)
        success = system.acknowledge_alert(component, action)
        emit('maintenance_alert_acknowledged', {'success': success})
    except ValueError:
        emit('error', {'message': 'Invalid component or action'})


@socketio.on('maintenance_record')
def handle_maintenance_record(data):
    """Record maintenance performed."""
    system = get_predictive_maintenance()
    if not system:
        return

    from maintenance.predictive_maintenance import ComponentType, MaintenanceAction

    try:
        component = ComponentType(data.get('component', ''))
        action = MaintenanceAction(data.get('action', ''))
        notes = data.get('notes', '')

        system.record_maintenance(component, action, notes)
        emit('maintenance_recorded', {'success': True})
    except ValueError:
        emit('error', {'message': 'Invalid component or action'})


@socketio.on('maintenance_flight_ready')
def handle_maintenance_flight_ready(data=None):
    """Check if drone is flight-ready based on maintenance."""
    system = get_predictive_maintenance()
    if not system:
        emit('maintenance_flight_ready_result', {'ready': True, 'issues': []})
        return

    ready, issues = system.is_flight_ready()
    emit('maintenance_flight_ready_result', {
        'ready': ready,
        'issues': issues,
    })


@socketio.on('maintenance_start_flight')
def handle_maintenance_start_flight(data=None):
    """Signal flight start to maintenance system."""
    system = get_predictive_maintenance()
    if system:
        system.start_flight()
        emit('maintenance_flight_started', {'success': True})


@socketio.on('maintenance_end_flight')
def handle_maintenance_end_flight(data=None):
    """Signal flight end to maintenance system."""
    system = get_predictive_maintenance()
    if system:
        system.end_flight()
        emit('maintenance_flight_ended', {'success': True})


@socketio.on('get_predictive_maintenance_status')
def handle_get_predictive_status(data=None):
    """Get predictive maintenance system status."""
    system = get_predictive_maintenance()
    if system:
        emit('predictive_maintenance_status', system.get_status())
    else:
        emit('predictive_maintenance_status', {'available': False})


# ==================== Flight Log Playback Handlers ====================

@socketio.on('get_flight_logs')
def handle_get_flight_logs(data=None):
    """Get list of available flight logs."""
    if flight_recorder:
        try:
            sessions = flight_recorder.list_sessions()
            logs = []
            for session in sessions:
                logs.append({
                    'session_id': session.session_id,
                    'start_time': session.start_time,
                    'duration': session.flight_time,
                    'records': session.record_count,
                })
            emit('flight_logs_list', {'logs': logs})
        except Exception as e:
            emit('flight_logs_list', {'logs': [], 'error': str(e)})
    else:
        emit('flight_logs_list', {'logs': []})


@socketio.on('load_flight_log')
def handle_load_flight_log(data):
    """Load a specific flight log for playback."""
    session_id = data.get('session_id')
    if not session_id:
        emit('error', {'message': 'No session ID provided'})
        return

    if flight_recorder:
        try:
            records = flight_recorder.load_session(session_id)
            if records:
                # Convert records to serializable format
                record_data = []
                for r in records:
                    record_data.append({
                        'timestamp': r.timestamp,
                        'lat': r.latitude,
                        'lon': r.longitude,
                        'alt': r.altitude,
                        'heading': r.heading,
                        'speed': r.speed,
                        'battery': r.battery_percent,
                    })
                emit('flight_log_data', {
                    'session_id': session_id,
                    'records': record_data
                })
            else:
                emit('error', {'message': 'No records found'})
        except Exception as e:
            emit('error', {'message': f'Failed to load log: {str(e)}'})
    else:
        emit('error', {'message': 'Flight recorder not available'})


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    # Initialize advanced features
    init_advanced_features()

    # Mirror the new FastAPI routes (SAR, survival, dock, scan, LLM, etc.)
    # onto this Flask app so the existing dashboard can call them too.
    # The advanced endpoints lazily spin up a Drone if the dashboard hasn't
    # clicked Start yet, so the operator can use the drawer without first
    # starting the existing simulation.
    def _get_or_create_drone():
        d = simulation.get('drone')
        if d is not None:
            return d
        try:
            d = create_drone()
            simulation['drone'] = d
            print("[advops] Lazy-created drone for advanced ops")
        except Exception as exc:
            print(f"[advops] Lazy drone creation failed: {exc}")
            return None
        return d
    try:
        from simulation.flask_advanced_routes import register_advanced_routes
        register_advanced_routes(app, get_drone=_get_or_create_drone)
        print("[init] Advanced routes registered (dock, SAR, survival, scan, LLM, anomalies)")
    except Exception as exc:
        print(f"[init] Failed to register advanced routes: {exc}")

    print(f"\n{'='*60}")
    print(f"  Drone Simulation Dashboard")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"{'='*60}\n")
    socketio.run(
        app, host=host, port=port,
        debug=debug, allow_unsafe_werkzeug=True
    )


if __name__ == '__main__':
    run_server(debug=True)
