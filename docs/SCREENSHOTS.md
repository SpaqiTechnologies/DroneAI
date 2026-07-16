# DroneAI — Sample Screenshots

Live captures of the running Flask dashboard at
`http://localhost:5000/` after the default `python run_simulation.py`.

Map centered on **Gjakova, Kosovo** (42.3803°N, 20.4308°E) — the
default home position for a fresh simulation.

---

## 1. Idle dashboard (just booted)

Right after `python run_simulation.py`. The map is dark-themed, the
drone hasn't been created yet, so telemetry fields show `--` and the
arming card shows **DISARMED**.

![Idle dashboard](screenshots/01_dashboard_idle.png)

Visible panels (left → right):
- **Left sidebar:** Position (lat/lon/alt/satellites), Flight Data
  (speed/heading), Battery, Telemetry Graphs (altitude/battery/speed
  over time), Wind
- **Center map:** Leaflet with dark CARTO tiles, altitude overlay
  top-left, vision-mode + theme + 3D view buttons top-right
- **Right sidebar:** Arming System (ARM / DISARM / KILL SWITCH),
  Geofence status, Landing Control (Normal / Precision / Emergency /
  Abort), Camera preview

Top-right: **🛰 Advanced Ops** pill button — opens the enterprise
features drawer.

---

## 2. Mission in flight

After Start → 3 waypoints uploaded → ARM → mission Start. The drone
is now flying east/north-east from Gjakova at 60 m/s and 30 m
altitude.

![Dashboard with drone flying a mission](screenshots/02_dashboard_flying.png)

What changed:
- **Position:** `42.381000, 20.435000` (moved from home)
- **Altitude:** 30 m — climbed to first waypoint's target
- **Speed:** 60 m/s, **Heading:** 153°
- **Satellites:** 10 (stable 3D fix)
- **Battery:** 84% (draining ~2%/minute)
- **Arming System:** big green **ARMED** badge
- **Geofence:** *Inside zone* — distance to boundary 646 m
- **Map:** purple ✈ drone marker beside the green 🏠 home marker
- **Telemetry graphs** (bottom-left) show live altitude / battery /
  speed history

---

## 3. Advanced Ops drawer

Click the **🛰 Advanced Ops** pill at the top-right — a drawer
slides in from the right with tabs for the newer features (docking,
SAR, survival tooling, adaptive 3D scan, media, system).

![Advanced Ops drawer](screenshots/03_advanced_ops_drawer.png)

Six tabs:
- **🏠 Dock** — Drone-in-a-box state machine, scheduled patrols
- **🚨 SAR** — Single & swarm search-and-rescue
- **🆘 Survival** — Beacon trilateration, supply-drop planner, safe
  corridor
- **🏗 Scan** — Adaptive 3D scan with per-bin coverage heatmap
- **🎬 Media** — Highlight reel builder, LLM mission planner,
  inspection report
- **⚙ System** — Anomaly-to-failsafe history + inject-test button

The drawer overlays the existing dashboard — map, telemetry, and
mission remain active underneath.

---

## How these were captured

Headless Chrome, 1600×1000 viewport:

```bash
chrome --headless --disable-gpu --window-size=1600,1000 \
  --virtual-time-budget=5000 \
  --screenshot=<path>.png \
  http://127.0.0.1:5000/
```

For the flight screenshot, a socket.io client fired
`start_simulation → upload_mission → arm_drone → start_mission`
before capture. For the drawer screenshot, a temporary URL-param
hook was used to auto-open the drawer, then removed before
committing.

Full step-by-step flight instructions live in
[../README.md](../README.md#how-to-run-a-simulation-flight).
