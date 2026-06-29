# DroneAI — Feature Reference

Comprehensive map of every shipped capability, where it lives in the
codebase, which API endpoint exposes it, and whether it has UI in the
dashboard.

**Tests:** 806+ passing. **Servers:** Flask (`web_server.py`) on `/` is the
primary UI; FastAPI (`fastapi_server.py` via `--fastapi`) exposes the same
business logic with OpenAPI docs at `/docs`.

---

## Quick start

```bash
# Default (Flask, full UI):
python run_simulation.py
# → http://localhost:5000

# FastAPI variant (REST + OpenAPI):
python run_simulation.py --fastapi
# → http://localhost:5000  +  http://localhost:5000/docs
```

The Flask dashboard combines the original telemetry/map/mission UI with
a new **🛰 Advanced Ops** drawer (top-right pill button) for the newer
SAR / survival / docking / scan / media features.

---

## 1. Flight control & safety

| Feature | Module | UI |
|---|---|---|
| State estimator (EKF, GPS+IMU fusion) | [core/state_estimator.py](core/state_estimator.py) | Telemetry overlay |
| Visual-inertial odometry (VIO) | [core/vio_fusion.py](core/vio_fusion.py) | VIO status card |
| Visual odometry | [sensors/visual_odometry.py](sensors/visual_odometry.py) | VO status card |
| LiDAR SLAM | [core/lidar_slam.py](core/lidar_slam.py) | SLAM status card |
| Flight controller (mode dispatch + trajectory) | [core/flight_controller.py](core/flight_controller.py) | Mode pill |
| Flight modes: STABILIZE, LOITER, GUIDED, AUTO, RTL, FOLLOW | [core/flight_modes/](core/flight_modes/) | Mode buttons |
| Takeoff manager (6-phase: pre-arm → spool → liftoff → climb → hover-check) | [core/takeoff/takeoff_manager.py](core/takeoff/takeoff_manager.py) | Start button |
| Landing manager (NORMAL / PRECISION / EMERGENCY / TERRAIN_FOLLOW / SAFE_ZONE) | [core/landing.py](core/landing.py) | Landing controls |
| Arming + pre-arm checks (GPS lock, compass, battery, props, airspace, manual confirms) | [core/arming.py](core/arming.py) | ARM/DISARM buttons |
| Failsafe manager + 9 triggers (GPS loss, low/critical battery, signal loss, sensor/IMU/motor failure, geofence breach, obstacle collision) | [core/failsafe.py](core/failsafe.py) | Failsafe trigger panel |
| Anomaly → failsafe bridge (auto-triggers failsafe on critical/emergency anomaly) | [core/anomaly_failsafe.py](core/anomaly_failsafe.py) | Anomaly history table |
| Health monitor (CPU temp, memory, sensor health, EKF status, vibration, motor current/RPM, link quality) | [core/health_monitor.py](core/health_monitor.py) | Health card |
| Geofencing (circular/polygon, soft warning + hard stop, altitude ceiling/floor) | [core/geofence.py](core/geofence.py) | Geofence editor |
| Obstacle avoidance (VFH+ replanner, collision predictor, tracker, evasion strategies) | [core/obstacle_avoidance/](core/obstacle_avoidance/) | Obstacle overlay |
| Path planning (A\*, RRT, RRT\*, potential fields) | [core/path_algorithms.py](core/path_algorithms.py) | Mission planner |
| Trajectory generator (smoothed multi-waypoint) | [core/trajectory/trajectory_generator.py](core/trajectory/trajectory_generator.py) | — |
| Unified autonomy runtime (takeoff → cruise → land background loop) | [core/autonomy_runtime.py](core/autonomy_runtime.py) | Dock adapter |
| Flight logger (events, failsafes, geofence breaches, full JSON logs) | [core/flight_logger.py](core/flight_logger.py) | Event log |

## 2. Sensors

| Sensor | Module |
|---|---|
| GPS (position, speed, heading, satellites, HDOP) | [sensors/gps_sensor.py](sensors/gps_sensor.py) |
| IMU (accel + gyro + magnetometer/compass + baro) | [sensors/imu_sensor.py](sensors/imu_sensor.py) |
| Battery (level, voltage, current, temperature) | [sensors/battery_sensor.py](sensors/battery_sensor.py) |
| Wind (speed, direction) | [sensors/wind_sensor.py](sensors/wind_sensor.py) |
| Ultrasonic (proximity + obstacle avoidance) | [sensors/ultrasonic_sensor.py](sensors/ultrasonic_sensor.py) |
| LiDAR (point cloud + scan) | [sensors/lidar_sensor.py](sensors/lidar_sensor.py) |
| Optical flow | [sensors/optical_flow_sensor.py](sensors/optical_flow_sensor.py) |
| Visual odometry | [sensors/visual_odometry.py](sensors/visual_odometry.py) |
| Camera (single) | [sensors/camera_sensor.py](sensors/camera_sensor.py) |
| **Multi-camera array** (wide / tele / thermal coordinator) | [sensors/camera_array.py](sensors/camera_array.py) |

### Camera presets and modes

- **Resolutions:** 480P, 720P, 1080P, 2.7K, 4K, **6K, 8K** — `CameraSensor.RESOLUTION_*`
- **Vision modes:** NORMAL, THERMAL, NIGHT_VISION, OBSTACLE_DETECTION, MARKER_DETECTION
- **HDR + color profiles:** standard, Rec709, HLG, HDR10, DLog-M (`set_color_profile`, `enable_hdr`)
- **Gimbal control:** pitch (-90 down → +30 up), roll (±45), yaw (±180); convenience `point_down()` / `point_forward()`
- **Real photo + video I/O:** PPM + PNG (stdlib) + JPEG (Pillow when installed) — see [sensors/media/](sensors/media/)
- **Detection callbacks:** frame + per-detection subscribe API

## 3. Media

| Feature | Module | Endpoint |
|---|---|---|
| Pure-stdlib PNG encoder (`zlib`-deflate + manual chunks) | [sensors/media/encoder.py](sensors/media/encoder.py) | — |
| PPM (P6) encoder | [sensors/media/encoder.py](sensors/media/encoder.py) | — |
| JPEG encoder (optional, requires Pillow) | [sensors/media/encoder.py](sensors/media/encoder.py) | — |
| Disk storage + artifact index | [sensors/media/storage.py](sensors/media/storage.py) | `/api/media` |
| Video recorder (per-frame PNG bundle + `manifest.json`) | [sensors/media/recorder.py](sensors/media/recorder.py) | `/api/media/recording/{start,stop}` |
| Snapshot capture | [sensors/camera_sensor.py](sensors/camera_sensor.py) | `/api/media/snapshot` |
| Live frame stream (PNG over HTTP) | — | `/api/media/latest.png` |
| AI highlight reel (event-driven auto-editor) | [applications/media/highlight_reel.py](applications/media/highlight_reel.py) | `/api/media/highlight-reel` |

## 4. AI

| Feature | Module | Real model? |
|---|---|---|
| YOLOv8 object detection (person, vehicle, animal, landing pad, …) | [ai/detection/yolo_detector.py](ai/detection/yolo_detector.py) | Yes (ultralytics) + synthetic fallback |
| Multi-object tracking (IOU + velocity smoothing) | [ai/detection/detection_tracker.py](ai/detection/detection_tracker.py) | Classical |
| Anomaly detector (z-score over sliding window, 11 anomaly types) | [ai/anomaly/anomaly_detector.py](ai/anomaly/anomaly_detector.py) | Statistical + callback API |
| Terrain classifier (landing zone scoring) | [ai/terrain/terrain_classifier.py](ai/terrain/terrain_classifier.py) | Heuristic |
| LLM command interpreter (natural language → `DroneCommand`) | [ai/llm/command_interpreter.py](ai/llm/command_interpreter.py) | Ollama / Transformers / OpenAI + regex fallback |
| LLM → Mission materializer (`interpret_to_mission`) | [ai/llm/command_interpreter.py](ai/llm/command_interpreter.py) | — |

## 5. Mission

| Feature | Module | Endpoint |
|---|---|---|
| Mission + Waypoint + actions | [core/mission/](core/mission/) | Mission planner UI |
| Mission manager (load, validate, run, pause, resume, abort) | [core/mission/mission_manager.py](core/mission/mission_manager.py) | — |
| Mission storage (JSON files in `missions/`) | [core/mission/mission_storage.py](core/mission/mission_storage.py) | — |
| Survey pattern generator (grid, orbit) | [core/mission/survey_patterns.py](core/mission/survey_patterns.py) | Mission planner |
| LLM mission plan / load | [simulation/llm_routes.py](simulation/llm_routes.py) + [flask_advanced_routes.py](simulation/flask_advanced_routes.py) | `/api/llm/mission/{plan,load}` |

## 6. Search & Rescue applications

| Feature | Module | Endpoint |
|---|---|---|
| 6 search patterns (expanding square, sector, parallel, creeping line, spiral, grid) | [applications/search_rescue/search_pattern.py](applications/search_rescue/search_pattern.py) | `/api/sar/plan` |
| Single-drone SAR mission with target dedupe | [applications/search_rescue/sar_mission.py](applications/search_rescue/sar_mission.py) | `/api/sar/run` |
| **Investigate-on-detection** (descend, multi-photo close-range, resume pattern) | [applications/search_rescue/sar_mission.py](applications/search_rescue/sar_mission.py) | reported in `/api/sar/{id}/report` |
| Swarm SAR (N drones, area split, cross-drone target dedupe) | [applications/search_rescue/swarm_sar.py](applications/search_rescue/swarm_sar.py) | `/api/swarm-sar/run` |

## 7. Survival / personnel-recovery applications

| Feature | Module | Endpoint |
|---|---|---|
| Emergency beacon locator (RSSI log-distance + linearised least-squares trilateration, centroid fallback) | [applications/survival/beacon_locator.py](applications/survival/beacon_locator.py) | `/api/survival/beacon/{sample,fix,reset}` |
| Wind-corrected supply drop planner (free-fall + drag + parachute model) | [applications/survival/supply_drop.py](applications/survival/supply_drop.py) | `/api/survival/supply-drop/plan` |
| Safe corridor planner (visibility-graph + Dijkstra around threat keep-outs) | [applications/survival/safe_corridor.py](applications/survival/safe_corridor.py) | `/api/survival/corridor/plan` |

Scope-bounded: defensive / personnel-recovery only. **No weapons, targeting, or strike features.**

## 8. Mapping & inspection applications

| Feature | Module | Endpoint |
|---|---|---|
| Aerial mapper (lawnmower survey, ground coverage from FOV + altitude) | [applications/mapping/aerial_mapper.py](applications/mapping/aerial_mapper.py) | — |
| **Adaptive 3D scan** (coverage-aware multi-shell orbit, per-bin coverage matrix, auto-refinement) | [applications/mapping/adaptive_scan.py](applications/mapping/adaptive_scan.py) | `/api/scan3d/{plan,capture,report}` |
| Infrastructure inspector (8 types: power line, tower, building façade, bridge, solar panel, wind turbine, pipeline, custom) | [applications/inspection/inspector.py](applications/inspection/inspector.py) | — |
| **Inspection report generator** (JSON + Markdown sidecar, severity-ranked defects) | [applications/inspection/report.py](applications/inspection/report.py) | `/api/inspection/report` |

## 9. Enterprise: drone-in-a-box

| Feature | Module | Endpoint |
|---|---|---|
| Docking station state machine (EMPTY → DOCKED → CHARGING → READY → DEPLOYED → RECALLING) | [core/docking/docking_station.py](core/docking/docking_station.py) | `/api/dock/{setup,status,deploy,recall}` |
| Charge profile (configurable rate, target %, launch min %, auto-recall %) | [core/docking/docking_station.py](core/docking/docking_station.py) | — |
| Patrol scheduler (recurring missions, configurable period + flight duration) | [core/docking/patrol_scheduler.py](core/docking/patrol_scheduler.py) | `/api/patrols/{add,list,remove,enable}` |
| **Dock → Autonomy adapter** (auto-runs takeoff→orbit→land on each deploy) | [core/docking/autonomy_adapter.py](core/docking/autonomy_adapter.py) | implicit via `/api/dock/deploy` |

## 10. Swarm coordination

| Feature | Module |
|---|---|
| Swarm coordinator (state machine, leader election, mission sync) | [swarm/swarm_coordinator.py](swarm/swarm_coordinator.py) |
| Drone registry (heartbeat, position tracking, status callbacks) | [swarm/drone_registry.py](swarm/drone_registry.py) |
| Inter-drone communication (UDP multicast + TCP messaging) | [swarm/inter_drone_comm.py](swarm/inter_drone_comm.py) |
| Formation controller (9 formations: LINE, V, ECHELON, DIAMOND, GRID, CIRCLE, COLUMN, WEDGE, CUSTOM) | [swarm/formation_controller.py](swarm/formation_controller.py) |
| Swarm collision avoidance (closest-point-of-approach predictor) | [swarm/swarm_collision.py](swarm/swarm_collision.py) |

## 11. Communication

| Feature | Module |
|---|---|
| MAVLink protocol (pymavlink) | [core/communication/mavlink_protocol.py](core/communication/mavlink_protocol.py) |
| MAVLink command layer (ack-waited COMMAND_LONG, goto position, mission upload/download) | [core/communication/mavlink_commands.py](core/communication/mavlink_commands.py) |
| MAVLink backend (SITL or real autopilot link) | core/communication/mavlink_backend.py |

## 12. Compliance

| Feature | Module |
|---|---|
| Remote ID broadcaster (FAA ASD-STAN format) | [compliance/remote_id.py](compliance/remote_id.py) |
| Flight recorder (telemetry + events to JSON) | [compliance/flight_recorder.py](compliance/flight_recorder.py) |
| Maintenance tracker (cycles, intervals, alerts) | [compliance/maintenance_tracker.py](compliance/maintenance_tracker.py) |
| Predictive maintenance (motor wear, battery degradation, anomaly history) | [maintenance/predictive_maintenance.py](maintenance/predictive_maintenance.py) |

## 13. Security

| Feature | Module |
|---|---|
| **Authenticated message encryption** (encrypt-then-MAC over PBKDF2 keystream + HMAC-SHA256 tag) | [security/crypto.py](security/crypto.py) |
| API key manager + token issuance | [security/authentication.py](security/authentication.py) |
| Role-based access control | [security/authorization.py](security/authorization.py) |
| Command signer (replay-protected, nonce + timestamp) | [security/crypto.py](security/crypto.py) |
| FastAPI security middleware | [security/middleware.py](security/middleware.py) |
| Secure config store | [security/secure_config.py](security/secure_config.py) |

## 14. Dashboards

### Flask dashboard (default) — `http://localhost:5000/`

The original **Drone AI Command Center** with everything you'd expect:

- Leaflet map (drone marker, flight path, geofence polygons)
- Waypoint editor with click-to-add, drag, delete
- Geofence drawing tools (polygon, circle)
- Mission planner tabs (Waypoints, Geofence)
- Live telemetry overlay (altitude, battery, speed, heading, GPS, satellites, wind arrow)
- Chart.js telemetry graphs (altitude, battery, speed over time)
- Three.js 3D visualization
- Socket.IO real-time state stream
- Camera viewport with vision-mode buttons (Normal / Obstacle / Thermal) + gimbal controls
- Recording / Snapshot / Streaming buttons
- Arming/disarming, Kill switch, RTL
- Failsafe trigger panel (GPS loss, low battery, signal loss, critical battery)
- Swarm panel (formation selector, drone list)
- AI detection + Anomaly + Remote ID compliance cards
- VIO / VO / SLAM status sections
- Flight playback

#### 🛰 Advanced Ops drawer (top-right pill button)

Six tabs covering the newer features:

| Tab | Cards |
|---|---|
| 🏠 **Dock** | Drone-in-a-box state, battery, autonomy phase, setup/deploy/recall, scheduled patrols list |
| 🚨 **SAR** | Single-drone SAR (6 patterns, live target table), swarm SAR (N drones, aggregated targets) |
| 🆘 **Survival** | Beacon trilateration, wind-corrected supply drop, safe corridor with threat keep-outs |
| 🏗 **Scan** | Adaptive 3D scan with per-bin coverage heatmap |
| 🎬 **Media** | Highlight reel builder, LLM mission planner, inspection report generator |
| ⚙ **System** | Anomaly → failsafe history table + inject-test, event log |

### FastAPI dashboard — `http://localhost:5000/dashboard` (when run with `--fastapi`)

Tabbed alternative dashboard exposing the same advanced features without
the legacy chart/map/3D widgets. Auto-generated API docs at `/docs`.

---

## API summary

| Group | Endpoints |
|---|---|
| Telemetry | `GET /api/state`, `GET /api/simulation/status` |
| Socket.IO events | `start_simulation`, `stop_simulation`, `arm_drone`, `disarm_drone`, `kill_switch`, `start_flight`, `return_to_home`, `start_landing`, `abort_landing`, `trigger_failsafe`, `set_position`, `run_pre_arm_checks`, `get_health_status`, etc. (see `simulation/web_server.py`) |
| Dock | `POST /api/dock/{setup,deploy,recall}`, `GET /api/dock/status` |
| Patrols | `POST /api/patrols/add`, `GET /api/patrols`, `DELETE /api/patrols/{id}`, `POST /api/patrols/{id}/enabled/{bool}` |
| SAR | `POST /api/sar/{plan,run}`, `GET /api/sar/{id}/report`, `POST /api/sar/{id}/abort`, `POST /api/swarm-sar/run`, `GET /api/swarm-sar/{id}/report` |
| Survival | `POST /api/survival/beacon/{sample,reset}`, `GET /api/survival/beacon/fix`, `POST /api/survival/supply-drop/plan`, `POST /api/survival/corridor/plan` |
| 3D scan | `POST /api/scan3d/plan`, `POST /api/scan3d/{id}/capture`, `GET /api/scan3d/{id}/report` |
| Media | `POST /api/media/snapshot`, `POST /api/media/recording/{start,stop}`, `GET /api/media`, `GET /api/media/latest.png` |
| Highlight reel | `POST /api/media/highlight-reel` |
| Inspection | `POST /api/inspection/report` |
| LLM | `POST /api/llm/mission/{plan,load}` |
| Anomalies | `GET /api/anomalies/history`, `POST /api/anomalies/inject` |
| Autonomy (FastAPI-only) | `POST /api/autonomy/run`, `GET /api/autonomy/{id}/status`, `POST /api/autonomy/{id}/abort` |

---

## Testing

```bash
pytest tests/ -q
```

- **806+ tests** covering every module above
- Pure-Python; the only optional runtime deps are `ultralytics` (real YOLO),
  `Pillow` (JPEG encoder), and a real LLM backend (Ollama/OpenAI/Transformers).
  Everything else uses synthetic fallbacks so the suite runs offline.

---

## Project layout

```
ai/               YOLO detector, anomaly detector, terrain classifier, LLM
applications/     SAR, swarm SAR, survival, mapping, inspection, media
compliance/       Remote ID, flight recorder, maintenance tracker
core/             Drone, flight controller, modes, failsafes, missions,
                  docking, autonomy runtime, path planning, geofence
maintenance/      Predictive maintenance
scripts/          Smoke scripts for media / SAR / patrol / survival
security/         Auth, encryption, command signing
sensors/          GPS, IMU, battery, wind, camera, lidar, optical flow,
                  visual odometry, multi-camera array, media I/O
simulation/       Flask web_server + FastAPI fastapi_server + dashboard
                  templates + static JS modules
swarm/            Coordinator, registry, comms, formation, collision
tests/            806+ tests covering all of the above
```

---

## Notes on scope

DroneAI is built for **defensive and search-and-rescue** scenarios:
inspection, survey, mapping, infrastructure monitoring, security patrols,
personnel recovery (downed-soldier locate, supply drop, safe-corridor
mapping). It explicitly **does not** implement weapons, targeting,
strike, or anti-personnel features. Threat zones in the corridor planner
are treated as keep-outs to *avoid*, not engage.

---

*Last updated automatically — feature additions land here as they're
committed.*
