/**
 * Mission Planner Module
 *
 * Provides mission planning functionality:
 * - Drag-drop waypoints on map
 * - Waypoint editing
 * - Survey pattern generation
 * - Mission upload/download
 */

class MissionPlanner {
    constructor(map, socket) {
        this.map = map;
        this.socket = socket;

        // Mission state
        this.mission = null;
        this.waypoints = [];
        this.waypointMarkers = [];
        this.pathLine = null;

        // Editing state
        this.editMode = false;
        this.selectedWaypoint = null;
        this.draggedMarker = null;

        // Default settings
        this.defaults = {
            altitude: 50,
            speed: 5,
            holdTime: 0,
            acceptanceRadius: 2
        };

        // Colors
        this.colors = {
            waypoint: '#00ff88',
            waypointSelected: '#ffaa00',
            waypointHover: '#00ffff',
            path: '#00ff88',
            pathActive: '#ffaa00'
        };

        this.init();
    }

    init() {
        this.setupSocketListeners();
        this.setupMapListeners();
    }

    setupSocketListeners() {
        // Mission events
        this.socket.on('mission_created', (data) => {
            this.mission = data.mission;
            this.updateMissionDisplay();
            this.showNotification('Mission created: ' + data.name, 'success');
        });

        this.socket.on('current_mission', (data) => {
            if (data) {
                this.mission = data;
                this.loadMissionToMap();
                this.updateMissionDisplay();
            }
        });

        this.socket.on('waypoint_added', (data) => {
            this.mission = data.mission;
            this.addWaypointMarker(data.waypoint, data.index);
            this.updatePath();
            this.updateWaypointList();
        });

        this.socket.on('waypoint_updated', (data) => {
            this.mission = data.mission;
            this.updateWaypointMarker(data.index, data.waypoint);
            this.updatePath();
            this.updateWaypointList();
        });

        this.socket.on('waypoint_removed', (data) => {
            this.mission = data.mission;
            this.removeWaypointMarker(data.index);
            this.updatePath();
            this.updateWaypointList();
        });

        this.socket.on('waypoints_cleared', (data) => {
            this.mission = data.mission;
            this.clearWaypointMarkers();
            this.updateWaypointList();
        });

        this.socket.on('mission_validation', (data) => {
            this.showValidationResults(data);
        });

        this.socket.on('survey_generated', (data) => {
            this.mission = data.mission;
            this.loadMissionToMap();
            this.updateMissionDisplay();
            this.showNotification('Survey pattern generated: ' + data.pattern, 'success');
        });

        this.socket.on('mission_uploaded', (data) => {
            this.showNotification('Mission uploaded to drone', 'success');
        });

        this.socket.on('mission_started', (data) => {
            this.showNotification('Mission started', 'success');
        });

        this.socket.on('mission_progress', (data) => {
            this.updateMissionProgress(data);
        });
    }

    setupMapListeners() {
        // Click to add waypoint in edit mode
        this.map.on('click', (e) => {
            if (this.editMode && this.mission) {
                this.addWaypoint(e.latlng.lat, e.latlng.lng);
            }
        });
    }

    // ==================== Mission Management ====================

    createMission(name, options = {}) {
        this.socket.emit('create_mission', {
            name: name || 'New Mission',
            description: options.description || '',
            type: options.type || 'waypoint',
            default_speed: options.speed || this.defaults.speed,
            default_altitude: options.altitude || this.defaults.altitude,
            return_home: options.returnHome || false
        });
    }

    loadMission() {
        this.socket.emit('get_current_mission');
    }

    saveMission() {
        this.socket.emit('save_mission');
    }

    validateMission() {
        this.socket.emit('validate_mission');
    }

    uploadMission() {
        this.socket.emit('upload_mission');
    }

    startMission() {
        this.socket.emit('start_mission');
    }

    pauseMission() {
        this.socket.emit('pause_mission');
    }

    resumeMission() {
        this.socket.emit('resume_mission');
    }

    abortMission() {
        this.socket.emit('abort_mission');
    }

    // ==================== Waypoint Management ====================

    addWaypoint(lat, lon, options = {}) {
        this.socket.emit('add_waypoint', {
            lat: lat,
            lon: lon,
            alt: options.altitude || this.defaults.altitude,
            name: options.name || `WP${this.waypoints.length + 1}`,
            type: options.type || 'navigate',
            speed: options.speed || this.defaults.speed,
            hold_time: options.holdTime || this.defaults.holdTime,
            radius: options.radius || this.defaults.acceptanceRadius,
            pass_through: options.passThrough || false,
            yaw: options.yaw || null,
            actions: options.actions || []
        });
    }

    updateWaypoint(index, updates) {
        this.socket.emit('update_waypoint', {
            index: index,
            ...updates
        });
    }

    removeWaypoint(index) {
        this.socket.emit('remove_waypoint', { index: index });
    }

    clearWaypoints() {
        this.socket.emit('clear_waypoints');
    }

    reorderWaypoints(fromIndex, toIndex) {
        this.socket.emit('reorder_waypoints', {
            from_index: fromIndex,
            to_index: toIndex
        });
    }

    // ==================== Map Display ====================

    loadMissionToMap() {
        this.clearWaypointMarkers();

        if (!this.mission || !this.mission.waypoints) return;

        this.mission.waypoints.forEach((wp, index) => {
            this.addWaypointMarker(wp, index);
        });

        this.updatePath();

        // Fit map to mission bounds
        if (this.waypointMarkers.length > 0) {
            const group = L.featureGroup(this.waypointMarkers);
            this.map.fitBounds(group.getBounds().pad(0.1));
        }
    }

    addWaypointMarker(waypoint, index) {
        const marker = L.marker([waypoint.latitude, waypoint.longitude], {
            icon: this.createWaypointIcon(index + 1),
            draggable: this.editMode
        });

        // Popup with waypoint info
        marker.bindPopup(this.createWaypointPopup(waypoint, index));

        // Tooltip
        marker.bindTooltip(`WP${index + 1}: ${waypoint.altitude}m`, {
            permanent: false,
            direction: 'top'
        });

        // Events
        marker.on('click', () => this.selectWaypoint(index));
        marker.on('dragend', (e) => this.onWaypointDragged(index, e));

        marker.addTo(this.map);
        this.waypointMarkers.push(marker);
        this.waypoints.push(waypoint);
    }

    updateWaypointMarker(index, waypoint) {
        if (index < this.waypointMarkers.length) {
            const marker = this.waypointMarkers[index];
            marker.setLatLng([waypoint.latitude, waypoint.longitude]);
            marker.setPopupContent(this.createWaypointPopup(waypoint, index));
            this.waypoints[index] = waypoint;
        }
    }

    removeWaypointMarker(index) {
        if (index < this.waypointMarkers.length) {
            this.map.removeLayer(this.waypointMarkers[index]);
            this.waypointMarkers.splice(index, 1);
            this.waypoints.splice(index, 1);

            // Renumber remaining markers
            this.waypointMarkers.forEach((marker, i) => {
                marker.setIcon(this.createWaypointIcon(i + 1));
            });
        }
    }

    clearWaypointMarkers() {
        this.waypointMarkers.forEach(marker => {
            this.map.removeLayer(marker);
        });
        this.waypointMarkers = [];
        this.waypoints = [];

        if (this.pathLine) {
            this.map.removeLayer(this.pathLine);
            this.pathLine = null;
        }
    }

    createWaypointIcon(number) {
        return L.divIcon({
            className: 'waypoint-marker',
            html: `<div class="waypoint-icon">${number}</div>`,
            iconSize: [30, 30],
            iconAnchor: [15, 15]
        });
    }

    createWaypointPopup(waypoint, index) {
        return `
            <div class="waypoint-popup">
                <h4>${waypoint.name || 'WP' + (index + 1)}</h4>
                <table>
                    <tr><td>Lat:</td><td>${waypoint.latitude.toFixed(6)}</td></tr>
                    <tr><td>Lon:</td><td>${waypoint.longitude.toFixed(6)}</td></tr>
                    <tr><td>Alt:</td><td>${waypoint.altitude}m</td></tr>
                    <tr><td>Speed:</td><td>${waypoint.speed}m/s</td></tr>
                    <tr><td>Hold:</td><td>${waypoint.hold_time}s</td></tr>
                </table>
                <div class="popup-actions">
                    <button onclick="missionPlanner.editWaypoint(${index})">Edit</button>
                    <button onclick="missionPlanner.removeWaypoint(${index})">Delete</button>
                </div>
            </div>
        `;
    }

    updatePath() {
        if (this.pathLine) {
            this.map.removeLayer(this.pathLine);
        }

        if (this.waypoints.length < 2) return;

        const latlngs = this.waypoints.map(wp => [wp.latitude, wp.longitude]);

        this.pathLine = L.polyline(latlngs, {
            color: this.colors.path,
            weight: 3,
            opacity: 0.8,
            dashArray: '10, 5'
        }).addTo(this.map);
    }

    // ==================== Editing ====================

    enableEditMode() {
        this.editMode = true;
        this.waypointMarkers.forEach(marker => {
            marker.dragging.enable();
        });
        this.showNotification('Edit mode enabled - click map to add waypoints', 'info');
    }

    disableEditMode() {
        this.editMode = false;
        this.waypointMarkers.forEach(marker => {
            marker.dragging.disable();
        });
    }

    toggleEditMode() {
        if (this.editMode) {
            this.disableEditMode();
        } else {
            this.enableEditMode();
        }
        return this.editMode;
    }

    selectWaypoint(index) {
        this.selectedWaypoint = index;

        // Highlight selected marker
        this.waypointMarkers.forEach((marker, i) => {
            if (i === index) {
                marker.setIcon(this.createSelectedWaypointIcon(i + 1));
            } else {
                marker.setIcon(this.createWaypointIcon(i + 1));
            }
        });

        // Update UI
        this.updateWaypointEditor(this.waypoints[index], index);
    }

    createSelectedWaypointIcon(number) {
        return L.divIcon({
            className: 'waypoint-marker selected',
            html: `<div class="waypoint-icon selected">${number}</div>`,
            iconSize: [36, 36],
            iconAnchor: [18, 18]
        });
    }

    onWaypointDragged(index, event) {
        const latlng = event.target.getLatLng();
        this.updateWaypoint(index, {
            lat: latlng.lat,
            lon: latlng.lng
        });
    }

    editWaypoint(index) {
        this.selectWaypoint(index);
        this.showWaypointEditor(this.waypoints[index], index);
    }

    // ==================== Survey Patterns ====================

    generateSurvey(pattern, options = {}) {
        const center = options.center || this.map.getCenter();

        this.socket.emit('generate_survey', {
            pattern: pattern,
            center_lat: center.lat,
            center_lon: center.lng,
            altitude: options.altitude || this.defaults.altitude,
            width: options.width || 100,
            height: options.height || 100,
            line_spacing: options.lineSpacing || 10,
            radius: options.radius || 50,
            num_points: options.numPoints || 12,
            angle: options.angle || 0,
            clockwise: options.clockwise !== false
        });
    }

    // ==================== UI Updates ====================

    updateMissionDisplay() {
        const statsEl = document.getElementById('mission-stats');
        if (!statsEl || !this.mission) return;

        const stats = this.mission.statistics || {};

        statsEl.innerHTML = `
            <div class="stat">
                <span class="label">Waypoints:</span>
                <span class="value">${this.mission.waypoints?.length || 0}</span>
            </div>
            <div class="stat">
                <span class="label">Distance:</span>
                <span class="value">${(stats.total_distance / 1000).toFixed(2)} km</span>
            </div>
            <div class="stat">
                <span class="label">Est. Time:</span>
                <span class="value">${Math.ceil(stats.estimated_time / 60)} min</span>
            </div>
            <div class="stat">
                <span class="label">Battery:</span>
                <span class="value">${stats.estimated_battery?.toFixed(1) || 0}%</span>
            </div>
        `;
    }

    updateWaypointList() {
        const listEl = document.getElementById('waypoint-list');
        if (!listEl || !this.mission) return;

        listEl.innerHTML = '';

        (this.mission.waypoints || []).forEach((wp, index) => {
            const item = document.createElement('div');
            item.className = 'waypoint-item';
            item.innerHTML = `
                <span class="wp-number">${index + 1}</span>
                <span class="wp-name">${wp.name || 'WP' + (index + 1)}</span>
                <span class="wp-alt">${wp.altitude}m</span>
                <div class="wp-actions">
                    <button onclick="missionPlanner.selectWaypoint(${index})" title="Select">
                        <i class="icon-target"></i>
                    </button>
                    <button onclick="missionPlanner.removeWaypoint(${index})" title="Delete">
                        <i class="icon-trash"></i>
                    </button>
                </div>
            `;
            item.onclick = () => this.selectWaypoint(index);
            listEl.appendChild(item);
        });
    }

    updateWaypointEditor(waypoint, index) {
        const editorEl = document.getElementById('waypoint-editor');
        if (!editorEl) return;

        editorEl.innerHTML = `
            <h4>Waypoint ${index + 1}</h4>
            <div class="form-group">
                <label>Name</label>
                <input type="text" id="wp-name" value="${waypoint.name || ''}" />
            </div>
            <div class="form-group">
                <label>Altitude (m)</label>
                <input type="number" id="wp-alt" value="${waypoint.altitude}" />
            </div>
            <div class="form-group">
                <label>Speed (m/s)</label>
                <input type="number" id="wp-speed" value="${waypoint.speed}" />
            </div>
            <div class="form-group">
                <label>Hold Time (s)</label>
                <input type="number" id="wp-hold" value="${waypoint.hold_time}" />
            </div>
            <div class="form-group">
                <label>
                    <input type="checkbox" id="wp-passthrough" ${waypoint.pass_through ? 'checked' : ''} />
                    Pass Through
                </label>
            </div>
            <button onclick="missionPlanner.saveWaypointEdit(${index})">Save</button>
        `;
    }

    saveWaypointEdit(index) {
        const updates = {
            name: document.getElementById('wp-name')?.value,
            alt: parseFloat(document.getElementById('wp-alt')?.value),
            speed: parseFloat(document.getElementById('wp-speed')?.value),
            hold_time: parseFloat(document.getElementById('wp-hold')?.value),
            pass_through: document.getElementById('wp-passthrough')?.checked
        };

        this.updateWaypoint(index, updates);
    }

    showWaypointEditor(waypoint, index) {
        // Show modal or panel with waypoint editor
        this.updateWaypointEditor(waypoint, index);
        const panel = document.getElementById('waypoint-panel');
        if (panel) panel.style.display = 'block';
    }

    showValidationResults(data) {
        const container = document.getElementById('validation-results');
        if (!container) {
            this.showNotification(
                data.is_valid ? 'Mission valid' : `Validation failed: ${data.issues.length} issues`,
                data.is_valid ? 'success' : 'warning'
            );
            return;
        }

        container.innerHTML = '';

        if (data.is_valid) {
            container.innerHTML = '<div class="validation-success">Mission is valid</div>';
        } else {
            data.issues.forEach(issue => {
                const el = document.createElement('div');
                el.className = `validation-issue ${issue.severity}`;
                el.innerHTML = `
                    <span class="severity">${issue.severity}</span>
                    <span class="message">${issue.message}</span>
                    ${issue.waypoint_index !== null ? `<span class="wp">WP${issue.waypoint_index + 1}</span>` : ''}
                `;
                container.appendChild(el);
            });
        }
    }

    updateMissionProgress(data) {
        const progressEl = document.getElementById('mission-progress');
        if (!progressEl) return;

        if (!data.active) {
            progressEl.style.display = 'none';
            return;
        }

        progressEl.style.display = 'block';
        progressEl.innerHTML = `
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${data.progress * 100}%"></div>
            </div>
            <div class="progress-info">
                <span>WP ${data.current_waypoint}/${data.total_waypoints}</span>
                <span>${Math.round(data.distance_to_waypoint)}m to next</span>
                <span>${Math.floor(data.mission_time / 60)}:${String(Math.floor(data.mission_time % 60)).padStart(2, '0')}</span>
            </div>
        `;

        // Highlight current waypoint
        this.highlightCurrentWaypoint(data.current_waypoint - 1);
    }

    highlightCurrentWaypoint(index) {
        this.waypointMarkers.forEach((marker, i) => {
            if (i === index) {
                marker.setIcon(this.createActiveWaypointIcon(i + 1));
            } else if (i < index) {
                marker.setIcon(this.createCompletedWaypointIcon(i + 1));
            } else {
                marker.setIcon(this.createWaypointIcon(i + 1));
            }
        });
    }

    createActiveWaypointIcon(number) {
        return L.divIcon({
            className: 'waypoint-marker active',
            html: `<div class="waypoint-icon active">${number}</div>`,
            iconSize: [36, 36],
            iconAnchor: [18, 18]
        });
    }

    createCompletedWaypointIcon(number) {
        return L.divIcon({
            className: 'waypoint-marker completed',
            html: `<div class="waypoint-icon completed">✓</div>`,
            iconSize: [30, 30],
            iconAnchor: [15, 15]
        });
    }

    showNotification(message, type = 'info') {
        // Use existing notification system or create simple one
        if (window.showToast) {
            window.showToast(message, type);
        } else {
            console.log(`[${type}] ${message}`);
        }
    }
}

// Export for use
window.MissionPlanner = MissionPlanner;
