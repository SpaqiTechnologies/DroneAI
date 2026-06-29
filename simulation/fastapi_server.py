"""
FastAPI web server for drone simulation and control.

FastAPI provides:
- Async support for better performance
- Automatic OpenAPI documentation at /docs
- WebSocket support for real-time telemetry
- Better type validation with Pydantic

Run with: uvicorn simulation.fastapi_server:app --reload --port 5000
"""

from __future__ import annotations

import os
import sys
import json
import time
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.drone import Drone
from core.failsafe import FailsafeAction, FailsafeType
from core.landing import LandingMode
from core.arming import PreArmCheck
from core.flight_modes.base_mode import FlightMode
from sensors.camera_sensor import VisionMode

# Mission planning imports
from core.mission import (
    Mission,
    MissionType,
    MissionState,
    Waypoint,
    WaypointType,
    MissionValidator,
    MissionStorage,
    SurveyPatternGenerator,
)

# Takeoff imports
from core.takeoff import TakeoffManager, TakeoffState, TakeoffMode, TakeoffConfig

# Swarm imports
from swarm import SwarmCoordinator, SwarmState, DroneRegistry, FormationType

# AI/ML imports
from ai.detection import YOLODetector, DetectionTracker
from ai.terrain import TerrainClassifier
from ai.anomaly import AnomalyDetector

# Maintenance imports
from maintenance import PredictiveMaintenanceSystem

# Path planning imports
from core.path_algorithms import UnifiedPathPlanner, Point3D, PathPlannerType, Obstacle

# Follow mode imports
from core.flight_modes.follow_mode import FollowModeHandler, FollowPosition, TargetType

# Security imports
from security import (
    init_security,
    get_api_key_manager,
    get_token_manager,
    get_access_control,
    create_fastapi_auth_dependency,
    FastAPISecurityMiddleware,
    generate_api_key,
    create_session_token,
    validate_credentials,
    Permission,
    CommandSigner,
    AuthenticationError,
    TokenExpiredError,
)


# ---------------------------------------------------------------------------
# Pydantic Models (Request/Response schemas)
# ---------------------------------------------------------------------------

class PositionModel(BaseModel):
    """GPS position model."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude: float = Field(default=30.0, ge=0, le=500)


class WaypointModel(BaseModel):
    """Waypoint definition."""
    latitude: float
    longitude: float
    altitude: float = 30.0
    hold_time: float = 0.0
    acceptance_radius: float = 2.0


class MissionModel(BaseModel):
    """Mission definition."""
    name: str = "Mission"
    mission_type: str = "waypoint"
    waypoints: List[WaypointModel]


class CommandResponse(BaseModel):
    """Standard command response."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class TelemetryModel(BaseModel):
    """Telemetry data model."""
    position: Dict[str, float]
    attitude: Dict[str, float]
    velocity: Dict[str, float]
    battery: Dict[str, float]
    gps: Dict[str, Any]
    flight_mode: str
    armed: bool
    timestamp: float


class PathPlanRequest(BaseModel):
    """Path planning request."""
    start: PositionModel
    goal: PositionModel
    algorithm: str = "a_star"
    obstacles: List[Dict[str, Any]] = []


class FollowConfigModel(BaseModel):
    """Follow mode configuration."""
    distance: float = 10.0
    altitude: float = 15.0
    position: str = "behind"
    max_speed: float = 15.0


class NaturalLanguageCommand(BaseModel):
    """Natural language command input."""
    command: str
    context: Optional[Dict[str, Any]] = None


class AuthRequest(BaseModel):
    """Authentication request."""
    name: str = "API Key"
    roles: List[str] = ["operator"]


class TokenRequest(BaseModel):
    """Token request."""
    api_key: str


class SignedCommandRequest(BaseModel):
    """Signed command request."""
    command_type: str
    parameters: Dict[str, Any] = {}
    signature: str
    timestamp: float
    nonce: str
    signer_id: str


class MAVLinkConnectRequest(BaseModel):
    """Open a MAVLink link to a real autopilot or SITL endpoint."""
    endpoint: str = "udpout:127.0.0.1:14550"
    baud: int = 57600
    heartbeat_timeout: float = 10.0


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

class SimulationState:
    """Global simulation state."""

    def __init__(self):
        self.drone: Optional[Drone] = None
        self.running: bool = False
        self.task: Optional[asyncio.Task] = None
        self.speed: float = 1.0

        # Components
        self.mission_storage = MissionStorage()
        self.survey_generator = SurveyPatternGenerator()
        self.mission_validator = MissionValidator()
        self.current_mission: Optional[Mission] = None
        self.takeoff_manager: Optional[TakeoffManager] = None
        self.path_planner = UnifiedPathPlanner()
        self.follow_handler = FollowModeHandler()
        self.maintenance = PredictiveMaintenanceSystem()

        # Security
        self.command_signer = CommandSigner()

        # Swarm
        self.swarm_coordinator: Optional[SwarmCoordinator] = None

        # AI
        self.ai_detector: Optional[YOLODetector] = None
        self.ai_tracker: Optional[DetectionTracker] = None

        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []

    def initialize_drone(self) -> Drone:
        """Initialize drone instance."""
        if self.drone is None:
            self.drone = Drone()
            self.takeoff_manager = TakeoffManager(self.drone)
        return self.drone

    async def broadcast_telemetry(self, data: Dict[str, Any]):
        """Broadcast telemetry to all connected clients."""
        disconnected = []
        for ws in self.websocket_connections:
            try:
                await ws.send_json({"type": "telemetry", "data": data})
            except Exception:
                disconnected.append(ws)

        # Clean up disconnected
        for ws in disconnected:
            self.websocket_connections.remove(ws)


state = SimulationState()


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Starting Drone AI FastAPI Server...")

    # Initialize security
    init_security()
    print("Security initialized")

    state.initialize_drone()

    # Initialize AI components with graceful fallback
    try:
        state.ai_detector = YOLODetector()
        state.ai_tracker = DetectionTracker()
    except Exception as e:
        print(f"AI components not available: {e}")

    yield

    # Shutdown
    print("Shutting down...")
    state.running = False
    if state.task:
        state.task.cancel()
    if state.drone and state.drone.is_mavlink_connected:
        state.drone.disconnect_mavlink()


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Drone AI Control API",
    description="REST API and WebSocket interface for autonomous drone control",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Survival / SAR routes
from simulation.survival_routes import build_router as _build_survival_router
app.include_router(_build_survival_router(lambda: state))

# Autonomy runtime routes
from simulation.autonomy_routes import build_router as _build_autonomy_router
app.include_router(_build_autonomy_router(lambda: state))

# LLM → Mission routes
from simulation.llm_routes import build_router as _build_llm_router
app.include_router(_build_llm_router(lambda: state))

# Enterprise routes (docking, patrols, adaptive scan, highlight reel, inspection)
from simulation.enterprise_routes import build_router as _build_enterprise_router
app.include_router(_build_enterprise_router(lambda: state))


# ---------------------------------------------------------------------------
# WebSocket Endpoints
# ---------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry."""
    await websocket.accept()
    state.websocket_connections.append(websocket)

    try:
        while True:
            # Send telemetry every 100ms
            if state.drone:
                telemetry = get_telemetry_data()
                await websocket.send_json({"type": "telemetry", "data": telemetry})

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        state.websocket_connections.remove(websocket)
    except Exception:
        if websocket in state.websocket_connections:
            state.websocket_connections.remove(websocket)


@app.websocket("/ws/commands")
async def websocket_commands(websocket: WebSocket):
    """WebSocket endpoint for bidirectional commands."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")
            params = data.get("params", {})

            result = await process_command(command, params)
            await websocket.send_json({"type": "response", "data": result})
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# REST API Endpoints - Drone Control
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Browser dashboard: live camera + dock + patrols + scan + LLM."""
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "static", "drone_dashboard.html",
    )
    try:
        with open(dashboard_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"dashboard missing: {exc}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation link."""
    return """
    <html>
        <head><title>Drone AI API</title></head>
        <body>
            <h1>Drone AI Control API</h1>
            <p>Live dashboard: <a href="/dashboard">/dashboard</a></p>
            <p>API Documentation: <a href="/docs">/docs</a></p>
            <p>WebSocket Telemetry: ws://localhost:5000/ws/telemetry</p>
        </body>
    </html>
    """


# ---------------------------------------------------------------------------
# REST API Endpoints - Authentication
# ---------------------------------------------------------------------------

@app.post("/api/auth/key", response_model=CommandResponse)
async def create_api_key(request: AuthRequest):
    """Generate a new API key."""
    plain_key, key_info = generate_api_key(request.name, request.roles)

    return CommandResponse(
        success=True,
        message="API key generated",
        data={
            'api_key': plain_key,
            'key_id': key_info['key_id'],
            'name': key_info['name'],
            'roles': key_info['roles'],
        }
    )


@app.post("/api/auth/token", response_model=CommandResponse)
async def create_token(request: TokenRequest):
    """Create a session token from an API key."""
    api_key_mgr = get_api_key_manager()
    validated = api_key_mgr.validate_key(request.api_key)

    if not validated:
        raise HTTPException(status_code=401, detail="Invalid API key")

    token = create_session_token(
        f"apikey:{validated.key_id}",
        validated.roles
    )

    return CommandResponse(
        success=True,
        message="Token created",
        data={
            'token': token,
            'roles': validated.roles,
        }
    )


@app.post("/api/auth/validate", response_model=CommandResponse)
async def validate_auth(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    """Validate authentication credentials."""
    result = validate_credentials(
        api_key=x_api_key,
        token=authorization[7:] if authorization and authorization.startswith('Bearer ') else None
    )

    if result:
        return CommandResponse(
            success=True,
            message="Valid credentials",
            data=result
        )
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/api/command/signed", response_model=CommandResponse)
async def execute_signed_command(
    request: SignedCommandRequest,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    """Execute a signed command (for critical operations)."""
    # First validate authentication
    auth = validate_credentials(
        api_key=x_api_key,
        token=authorization[7:] if authorization and authorization.startswith('Bearer ') else None
    )

    if not auth:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Verify command signature
    from security.crypto import SignedCommand, SignatureError

    command = SignedCommand(
        command_type=request.command_type,
        parameters=request.parameters,
        timestamp=request.timestamp,
        nonce=request.nonce,
        signature=request.signature,
        signer_id=request.signer_id,
    )

    try:
        state.command_signer.verify_command(command)
    except SignatureError as e:
        raise HTTPException(status_code=400, detail=f"Invalid command signature: {e}")

    # Execute command
    result = await process_command(request.command_type, request.parameters)

    return CommandResponse(
        success=result.get('success', False),
        message=f"Signed command executed: {request.command_type}",
        data=result
    )


@app.get("/api/auth/permissions")
async def get_permissions(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    """Get permissions for current user."""
    auth = validate_credentials(
        api_key=x_api_key,
        token=authorization[7:] if authorization and authorization.startswith('Bearer ') else None
    )

    if not auth:
        raise HTTPException(status_code=401, detail="Authentication required")

    ac = get_access_control()
    allowed_endpoints = ac.get_allowed_endpoints(auth['roles'])
    allowed_events = ac.get_allowed_events(auth['roles'])

    return {
        'user_id': auth['user_id'],
        'roles': auth['roles'],
        'allowed_endpoints': allowed_endpoints,
        'allowed_events': allowed_events,
    }


# ---------------------------------------------------------------------------
# REST API Endpoints - Status & Telemetry
# ---------------------------------------------------------------------------

@app.get("/api/status", response_model=CommandResponse)
async def get_status():
    """Get overall system status."""
    if not state.drone:
        return CommandResponse(success=False, message="Drone not initialized")

    return CommandResponse(
        success=True,
        message="System operational",
        data={
            "simulation_running": state.running,
            "drone_armed": state.drone.arming_manager.is_armed if state.drone.arming_manager else False,
            "flight_mode": state.drone.flight_mode.value if state.drone.flight_mode else "unknown",
            "gps_fix": state.drone.gps_sensor.has_fix() if state.drone.gps_sensor else False,
            "battery_percent": state.drone.battery_sensor.get_percent() if state.drone.battery_sensor else 0,
            "maintenance_health": state.maintenance.get_overall_health() if hasattr(state.maintenance, "get_overall_health") else state.maintenance.get_status(),
        }
    )


@app.get("/api/telemetry")
async def get_telemetry():
    """Get current telemetry data."""
    return get_telemetry_data()


# ---------------------------------------------------------------------------
# REST API Endpoints - MAVLink / SITL link
# ---------------------------------------------------------------------------

@app.post("/api/mavlink/connect", response_model=CommandResponse)
async def mavlink_connect(req: MAVLinkConnectRequest):
    """Open a MAVLink link. Once connected, telemetry comes from the autopilot."""
    if not state.drone:
        raise HTTPException(status_code=400, detail="Drone not initialized")
    try:
        peer = state.drone.connect_mavlink(
            req.endpoint,
            baud=req.baud,
            heartbeat_timeout=req.heartbeat_timeout,
        )
    except RuntimeError as e:
        return CommandResponse(success=False, message=str(e))
    except TimeoutError as e:
        return CommandResponse(success=False, message=str(e))
    return CommandResponse(success=True, message=f"connected to {req.endpoint}", data=peer)


@app.post("/api/mavlink/disconnect", response_model=CommandResponse)
async def mavlink_disconnect():
    """Close the MAVLink link and revert to simulated telemetry."""
    if not state.drone:
        raise HTTPException(status_code=400, detail="Drone not initialized")
    state.drone.disconnect_mavlink()
    return CommandResponse(success=True, message="disconnected")


@app.get("/api/mavlink/status")
async def mavlink_status():
    """Current state of the MAVLink link (connected? fresh? armed? mode?)."""
    if not state.drone:
        return {"connected": False}
    return state.drone.get_mavlink_status()


@app.post("/api/arm", response_model=CommandResponse)
async def arm_drone(force: bool = False):
    """Arm the drone."""
    if not state.drone:
        raise HTTPException(status_code=400, detail="Drone not initialized")

    success, message = state.drone.arm(force=force)
    return CommandResponse(
        success=success,
        message=message
    )


@app.post("/api/disarm", response_model=CommandResponse)
async def disarm_drone(reason: str = "user_request"):
    """Disarm the drone."""
    if not state.drone:
        raise HTTPException(status_code=400, detail="Drone not initialized")

    success, message = state.drone.disarm(reason=reason)
    return CommandResponse(
        success=success,
        message=message
    )


@app.post("/api/takeoff", response_model=CommandResponse)
async def takeoff(altitude: float = 10.0):
    """Command takeoff to specified altitude."""
    if not state.drone:
        raise HTTPException(status_code=400, detail="Drone not initialized")

    if not state.drone.is_armed:
        return CommandResponse(success=False, message="Drone must be armed first")

    if state.takeoff_manager:
        config = TakeoffConfig(target_altitude=altitude)
        state.takeoff_manager.configure(config)
        success, message = state.takeoff_manager.start_takeoff(altitude)
        return CommandResponse(
            success=success,
            message=message if message else f"Takeoff initiated to {altitude}m"
        )

    return CommandResponse(success=False, message="Takeoff manager not available")


@app.post("/api/land", response_model=CommandResponse)
async def land(mode: str = "normal"):
    """Command landing."""
    if not state.drone:
        raise HTTPException(status_code=400, detail="Drone not initialized")

    try:
        landing_mode = LandingMode[mode.upper()]
    except KeyError:
        landing_mode = LandingMode.NORMAL

    success, message = state.drone.start_landing(landing_mode)
    return CommandResponse(
        success=success,
        message=message
    )


@app.post("/api/rtl", response_model=CommandResponse)
async def return_to_launch():
    """Return to launch position."""
    if not state.drone:
        raise HTTPException(status_code=400, detail="Drone not initialized")

    success, message = state.drone.set_flight_mode(FlightMode.RTL)
    return CommandResponse(success=success, message=message)


@app.post("/api/goto", response_model=CommandResponse)
async def goto_position(position: PositionModel):
    """Fly to specified position."""
    if not state.drone:
        raise HTTPException(status_code=400, detail="Drone not initialized")

    # Set guided mode and target
    success, message = state.drone.flight_controller.goto(
        position.latitude,
        position.longitude,
        position.altitude,
    )

    return CommandResponse(
        success=success,
        message=message,
    )


@app.post("/api/mode", response_model=CommandResponse)
async def set_flight_mode(mode: str):
    """Set flight mode."""
    if not state.drone:
        raise HTTPException(status_code=400, detail="Drone not initialized")

    try:
        flight_mode = FlightMode[mode.upper()]
        success, message = state.drone.set_flight_mode(flight_mode)
        return CommandResponse(success=success, message=message)
    except KeyError:
        return CommandResponse(success=False, message=f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# REST API Endpoints - Mission
# ---------------------------------------------------------------------------

@app.get("/api/missions")
async def list_missions():
    """List all saved missions."""
    missions = state.mission_storage.list_missions()
    return {"missions": missions}


@app.post("/api/mission", response_model=CommandResponse)
async def create_mission(mission: MissionModel):
    """Create and upload a mission."""
    try:
        mission_type = MissionType[mission.mission_type.upper()]
    except KeyError:
        mission_type = MissionType.WAYPOINT

    # Create mission object
    new_mission = Mission(
        name=mission.name,
        mission_type=mission_type,
    )

    # Add waypoints
    for i, wp in enumerate(mission.waypoints):
        waypoint = Waypoint(
            sequence=i,
            latitude=wp.latitude,
            longitude=wp.longitude,
            altitude=wp.altitude,
        )
        new_mission.add_waypoint(waypoint)

    # Validate
    is_valid, errors = state.mission_validator.validate(new_mission)
    if not is_valid:
        return CommandResponse(success=False, message=f"Validation failed: {errors}")

    state.current_mission = new_mission
    return CommandResponse(
        success=True,
        message=f"Mission '{mission.name}' created with {len(mission.waypoints)} waypoints"
    )


@app.post("/api/mission/start", response_model=CommandResponse)
async def start_mission():
    """Start the current mission."""
    if not state.current_mission:
        return CommandResponse(success=False, message="No mission loaded")

    if not state.drone:
        return CommandResponse(success=False, message="Drone not initialized")

    state.current_mission.start()
    success, message = state.drone.set_flight_mode(FlightMode.AUTO)
    if not success:
        return CommandResponse(success=False, message=message)

    return CommandResponse(success=True, message="Mission started")


@app.post("/api/mission/pause", response_model=CommandResponse)
async def pause_mission():
    """Pause current mission."""
    if state.current_mission:
        state.current_mission.pause()
    return CommandResponse(success=True, message="Mission paused")


@app.post("/api/mission/resume", response_model=CommandResponse)
async def resume_mission():
    """Resume paused mission."""
    if state.current_mission:
        state.current_mission.resume()
    return CommandResponse(success=True, message="Mission resumed")


# ---------------------------------------------------------------------------
# REST API Endpoints - Path Planning
# ---------------------------------------------------------------------------

@app.post("/api/path/plan")
async def plan_path(request: PathPlanRequest):
    """Plan a path between two points."""
    start = Point3D(request.start.latitude, request.start.longitude, request.start.altitude)
    goal = Point3D(request.goal.latitude, request.goal.longitude, request.goal.altitude)

    # Convert obstacles
    obstacles = []
    for obs in request.obstacles:
        obstacles.append(Obstacle(
            center=Point3D(obs['x'], obs['y'], obs['z']),
            radius=obs.get('radius', 5.0),
        ))

    state.path_planner.set_obstacles(obstacles)

    try:
        algorithm = PathPlannerType[request.algorithm.upper()]
    except KeyError:
        algorithm = PathPlannerType.A_STAR

    result = state.path_planner.plan(start, goal, algorithm)

    return result.to_dict()


# ---------------------------------------------------------------------------
# REST API Endpoints - Follow Mode
# ---------------------------------------------------------------------------

@app.post("/api/follow/start", response_model=CommandResponse)
async def start_follow_mode(target_type: str = "person"):
    """Start follow-me mode."""
    try:
        target = TargetType[target_type.upper()]
    except KeyError:
        target = TargetType.PERSON

    state.follow_handler.activate(target)
    return CommandResponse(success=True, message=f"Follow mode started, tracking {target_type}")


@app.post("/api/follow/stop", response_model=CommandResponse)
async def stop_follow_mode():
    """Stop follow-me mode."""
    state.follow_handler.deactivate()
    return CommandResponse(success=True, message="Follow mode stopped")


@app.post("/api/follow/config", response_model=CommandResponse)
async def configure_follow(config: FollowConfigModel):
    """Configure follow mode parameters."""
    state.follow_handler.configure(
        distance=config.distance,
        altitude=config.altitude,
        position=config.position,
        max_speed=config.max_speed,
    )
    return CommandResponse(success=True, message="Follow mode configured")


@app.get("/api/follow/status")
async def get_follow_status():
    """Get follow mode status."""
    return state.follow_handler.get_status()


# ---------------------------------------------------------------------------
# REST API Endpoints - Maintenance
# ---------------------------------------------------------------------------

@app.get("/api/maintenance/status")
async def get_maintenance_status():
    """Get predictive maintenance status."""
    return state.maintenance.get_status()


@app.get("/api/maintenance/alerts")
async def get_maintenance_alerts(urgency: Optional[str] = None):
    """Get maintenance alerts."""
    if urgency:
        from maintenance import MaintenanceUrgency
        try:
            urg = MaintenanceUrgency[urgency.upper()]
            alerts = state.maintenance.get_alerts(urgency=urg)
        except KeyError:
            alerts = state.maintenance.get_alerts()
    else:
        alerts = state.maintenance.get_alerts()

    return {"alerts": [a.to_dict() for a in alerts]}


@app.get("/api/maintenance/flight-ready")
async def check_flight_ready():
    """Check if system is flight-ready."""
    ready, issues = state.maintenance.is_flight_ready()
    return {"ready": ready, "issues": issues}


# ---------------------------------------------------------------------------
# REST API Endpoints - Simulation Control
# ---------------------------------------------------------------------------

@app.post("/api/simulation/start", response_model=CommandResponse)
async def start_simulation():
    """Start the simulation loop."""
    if state.running:
        return CommandResponse(success=False, message="Simulation already running")

    state.running = True
    state.task = asyncio.create_task(simulation_loop())

    return CommandResponse(success=True, message="Simulation started")


@app.post("/api/simulation/stop", response_model=CommandResponse)
async def stop_simulation():
    """Stop the simulation loop."""
    state.running = False
    if state.task:
        state.task.cancel()

    return CommandResponse(success=True, message="Simulation stopped")


@app.post("/api/simulation/speed", response_model=CommandResponse)
async def set_simulation_speed(speed: float = Query(ge=0.1, le=10.0)):
    """Set simulation speed multiplier."""
    state.speed = speed
    return CommandResponse(success=True, message=f"Simulation speed set to {speed}x")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_telemetry_data() -> Dict[str, Any]:
    """Get current telemetry as dictionary."""
    if not state.drone:
        return {}

    drone = state.drone

    # When a real autopilot/SITL is connected and its heartbeat is fresh,
    # let it drive position/attitude/velocity/battery/mode/armed. Falls
    # back to the simulated sensors otherwise.
    mavlink_fresh = drone._mavlink_telemetry_fresh()
    tel = drone.mavlink_backend.telemetry if mavlink_fresh else None

    if tel is not None and tel.position is not None:
        latitude = tel.position["lat"]
        longitude = tel.position["lon"]
        altitude = tel.position["alt_rel_m"]
    else:
        latitude, longitude = drone.gps_sensor.get_position() if drone.gps_sensor else (0, 0)
        altitude = drone.gps_sensor.get_altitude() if drone.gps_sensor else 0

    if tel is not None and tel.attitude is not None:
        roll, pitch, yaw = tel.attitude["roll"], tel.attitude["pitch"], tel.attitude["yaw"]
    else:
        att = drone.imu_sensor.get_attitude() if drone.imu_sensor else None
        roll = att.roll if att else 0
        pitch = att.pitch if att else 0
        yaw = att.yaw if att else 0

    if tel is not None and tel.velocity_ned is not None:
        vx, vy, vz = tel.velocity_ned["vx"], tel.velocity_ned["vy"], tel.velocity_ned["vz"]
    else:
        vx = drone.gps_sensor.get_speed() if drone.gps_sensor else 0
        vy = 0
        vz = 0

    if tel is not None and tel.battery_pct is not None and tel.battery_pct >= 0:
        battery_percent = float(tel.battery_pct)
        battery_voltage = drone.battery_sensor.get_voltage() if drone.battery_sensor else 0
    else:
        battery_voltage = drone.battery_sensor.get_voltage() if drone.battery_sensor else 0
        battery_percent = drone.battery_sensor.get_percent() if drone.battery_sensor else 0

    return {
        "position": {"latitude": latitude, "longitude": longitude, "altitude": altitude},
        "attitude": {"roll": roll, "pitch": pitch, "yaw": yaw},
        "velocity": {"vx": vx, "vy": vy, "vz": vz},
        "battery": {"voltage": battery_voltage, "percent": battery_percent},
        "gps": {
            "fix": drone.gps_sensor.has_fix() if drone.gps_sensor else False,
            "satellites": drone.gps_sensor.get_satellites() if drone.gps_sensor else 0,
        },
        "flight_mode": (tel.mode if tel is not None and tel.mode else
                        (drone.flight_mode.value if drone.flight_mode else "unknown")),
        "armed": tel.armed if tel is not None and tel.armed is not None else drone.is_armed,
        "mavlink": {
            "connected": drone.is_mavlink_connected,
            "fresh": mavlink_fresh,
            "heartbeat_age_s": tel.heartbeat_age_s if tel is not None else None,
        },
        "timestamp": time.time(),
    }


async def process_command(command: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Process a command received via WebSocket."""
    commands = {
        "arm": lambda: state.drone.arm(),
        "disarm": lambda: state.drone.disarm(),
        "takeoff": lambda: state.takeoff_manager.start_takeoff() if state.takeoff_manager else (False, "Takeoff manager not available"),
        "land": lambda: state.drone.start_landing(),
        "rtl": lambda: state.drone.set_flight_mode(FlightMode.RTL),
    }

    if command in commands:
        try:
            if not state.drone:
                return {"success": False, "error": "Drone not initialized"}
            result = commands[command]()
            if isinstance(result, tuple) and len(result) >= 2:
                return {"success": bool(result[0]), "message": result[1], "result": result}
            return {"success": bool(result), "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {"success": False, "error": f"Unknown command: {command}"}


async def simulation_loop():
    """Main simulation update loop."""
    dt = 0.1

    while state.running:
        try:
            if state.drone:
                # Update sensors
                state.drone.update_sensors(dt * state.speed)

                # Update maintenance monitoring
                if state.drone.imu_sensor:
                    accel = state.drone.imu_sensor.get_accelerometer()
                    state.maintenance.update_imu(accel.x, accel.y, accel.z)

                if state.drone.battery_sensor:
                    state.maintenance.update_battery(
                        voltage=state.drone.battery_sensor.get_voltage(),
                        current=state.drone.battery_sensor.get_current(),
                    )

                # Broadcast telemetry
                telemetry = get_telemetry_data()
                await state.broadcast_telemetry(telemetry)

            await asyncio.sleep(dt)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Simulation error: {e}")
            await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
