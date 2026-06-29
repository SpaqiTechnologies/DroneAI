/**
 * Geofence Editor Module
 *
 * Provides geofence drawing and editing:
 * - Polygon geofence drawing
 * - Circular geofence
 * - Edit existing fences
 * - Import/export
 */

class GeofenceEditor {
    constructor(map, socket) {
        this.map = map;
        this.socket = socket;

        // Geofence state
        this.geofences = [];
        this.geofenceLayers = [];
        this.activeGeofence = null;

        // Drawing state
        this.drawingMode = false;
        this.drawingType = 'polygon'; // 'polygon' or 'circle'
        this.drawingPoints = [];
        this.drawingLayer = null;

        // Colors
        this.colors = {
            inclusion: { fill: '#00ff88', stroke: '#00cc66' },
            exclusion: { fill: '#ff4444', stroke: '#cc0000' },
            drawing: { fill: '#ffaa00', stroke: '#ff8800' },
            home: { fill: '#0088ff', stroke: '#0066cc' }
        };

        this.init();
    }

    init() {
        this.setupSocketListeners();
        this.setupMapListeners();
    }

    setupSocketListeners() {
        this.socket.on('geofence_created', (data) => {
            this.addGeofenceToMap(data.geofence);
            this.showNotification('Geofence created', 'success');
        });

        this.socket.on('geofence_updated', (data) => {
            this.updateGeofenceOnMap(data.geofence);
        });

        this.socket.on('geofence_deleted', (data) => {
            this.removeGeofenceFromMap(data.id);
        });

        this.socket.on('geofences_list', (data) => {
            this.loadGeofences(data.geofences);
        });

        this.socket.on('geofence_breach', (data) => {
            this.highlightBreach(data);
            this.showNotification('Geofence breach: ' + data.type, 'warning');
        });
    }

    setupMapListeners() {
        this.map.on('click', (e) => {
            if (this.drawingMode) {
                this.addDrawingPoint(e.latlng);
            }
        });

        this.map.on('mousemove', (e) => {
            if (this.drawingMode && this.drawingPoints.length > 0) {
                this.updateDrawingPreview(e.latlng);
            }
        });
    }

    // ==================== Drawing ====================

    startDrawing(type = 'polygon') {
        this.drawingMode = true;
        this.drawingType = type;
        this.drawingPoints = [];
        this.map.getContainer().style.cursor = 'crosshair';
        this.showNotification(`Click to add ${type} points. Double-click or press Enter to finish.`, 'info');
    }

    stopDrawing() {
        this.drawingMode = false;
        this.drawingPoints = [];
        this.map.getContainer().style.cursor = '';

        if (this.drawingLayer) {
            this.map.removeLayer(this.drawingLayer);
            this.drawingLayer = null;
        }
    }

    addDrawingPoint(latlng) {
        if (this.drawingType === 'circle') {
            if (this.drawingPoints.length === 0) {
                this.drawingPoints.push(latlng);
                this.showNotification('Click to set radius', 'info');
            } else {
                // Calculate radius and finish
                const center = this.drawingPoints[0];
                const radius = center.distanceTo(latlng);
                this.finishCircleDrawing(center, radius);
            }
        } else {
            this.drawingPoints.push(latlng);
            this.updateDrawingLayer();
        }
    }

    updateDrawingPreview(latlng) {
        if (this.drawingType === 'circle' && this.drawingPoints.length === 1) {
            const center = this.drawingPoints[0];
            const radius = center.distanceTo(latlng);

            if (this.drawingLayer) {
                this.map.removeLayer(this.drawingLayer);
            }

            this.drawingLayer = L.circle(center, {
                radius: radius,
                color: this.colors.drawing.stroke,
                fillColor: this.colors.drawing.fill,
                fillOpacity: 0.3
            }).addTo(this.map);
        }
    }

    updateDrawingLayer() {
        if (this.drawingLayer) {
            this.map.removeLayer(this.drawingLayer);
        }

        if (this.drawingPoints.length < 2) {
            // Just show markers for now
            this.drawingLayer = L.layerGroup();
            this.drawingPoints.forEach(point => {
                L.circleMarker(point, {
                    radius: 5,
                    color: this.colors.drawing.stroke,
                    fillColor: this.colors.drawing.fill,
                    fillOpacity: 1
                }).addTo(this.drawingLayer);
            });
            this.drawingLayer.addTo(this.map);
        } else {
            // Show polygon
            this.drawingLayer = L.polygon(this.drawingPoints, {
                color: this.colors.drawing.stroke,
                fillColor: this.colors.drawing.fill,
                fillOpacity: 0.3
            }).addTo(this.map);
        }
    }

    finishDrawing(name = 'New Geofence', type = 'inclusion') {
        if (this.drawingPoints.length < 3) {
            this.showNotification('Need at least 3 points for polygon', 'error');
            return;
        }

        const coordinates = this.drawingPoints.map(p => [p.lat, p.lng]);

        this.createPolygonGeofence(name, coordinates, type);
        this.stopDrawing();
    }

    finishCircleDrawing(center, radius) {
        this.createCircularGeofence('Home Geofence', center.lat, center.lng, radius);
        this.stopDrawing();
    }

    // ==================== Geofence Management ====================

    createPolygonGeofence(name, coordinates, type = 'inclusion') {
        this.socket.emit('create_geofence', {
            name: name,
            type: type,
            shape: 'polygon',
            coordinates: coordinates
        });
    }

    createCircularGeofence(name, lat, lng, radius, type = 'inclusion') {
        this.socket.emit('create_geofence', {
            name: name,
            type: type,
            shape: 'circle',
            center: { lat: lat, lng: lng },
            radius: radius
        });
    }

    updateGeofence(id, updates) {
        this.socket.emit('update_geofence', {
            id: id,
            ...updates
        });
    }

    deleteGeofence(id) {
        this.socket.emit('delete_geofence', { id: id });
    }

    loadGeofences(geofences) {
        // Clear existing
        this.clearGeofences();

        // Add new
        geofences.forEach(gf => {
            this.addGeofenceToMap(gf);
        });
    }

    clearGeofences() {
        this.geofenceLayers.forEach(layer => {
            this.map.removeLayer(layer);
        });
        this.geofenceLayers = [];
        this.geofences = [];
    }

    // ==================== Map Display ====================

    addGeofenceToMap(geofence) {
        let layer;
        const colors = geofence.type === 'exclusion' ?
            this.colors.exclusion : this.colors.inclusion;

        if (geofence.shape === 'circle') {
            layer = L.circle([geofence.center.lat, geofence.center.lng], {
                radius: geofence.radius,
                color: colors.stroke,
                fillColor: colors.fill,
                fillOpacity: 0.2,
                weight: 2
            });
        } else {
            layer = L.polygon(geofence.coordinates, {
                color: colors.stroke,
                fillColor: colors.fill,
                fillOpacity: 0.2,
                weight: 2
            });
        }

        // Add popup
        layer.bindPopup(this.createGeofencePopup(geofence));

        // Store reference
        layer.geofenceId = geofence.id;
        layer.addTo(this.map);

        this.geofences.push(geofence);
        this.geofenceLayers.push(layer);
    }

    updateGeofenceOnMap(geofence) {
        const index = this.geofences.findIndex(gf => gf.id === geofence.id);
        if (index === -1) return;

        // Remove old layer
        this.map.removeLayer(this.geofenceLayers[index]);

        // Add updated layer
        const colors = geofence.type === 'exclusion' ?
            this.colors.exclusion : this.colors.inclusion;

        let layer;
        if (geofence.shape === 'circle') {
            layer = L.circle([geofence.center.lat, geofence.center.lng], {
                radius: geofence.radius,
                color: colors.stroke,
                fillColor: colors.fill,
                fillOpacity: 0.2,
                weight: 2
            });
        } else {
            layer = L.polygon(geofence.coordinates, {
                color: colors.stroke,
                fillColor: colors.fill,
                fillOpacity: 0.2,
                weight: 2
            });
        }

        layer.bindPopup(this.createGeofencePopup(geofence));
        layer.geofenceId = geofence.id;
        layer.addTo(this.map);

        this.geofences[index] = geofence;
        this.geofenceLayers[index] = layer;
    }

    removeGeofenceFromMap(id) {
        const index = this.geofences.findIndex(gf => gf.id === id);
        if (index === -1) return;

        this.map.removeLayer(this.geofenceLayers[index]);
        this.geofences.splice(index, 1);
        this.geofenceLayers.splice(index, 1);
    }

    createGeofencePopup(geofence) {
        return `
            <div class="geofence-popup">
                <h4>${geofence.name}</h4>
                <p>Type: ${geofence.type}</p>
                <p>Shape: ${geofence.shape}</p>
                ${geofence.radius ? `<p>Radius: ${geofence.radius.toFixed(0)}m</p>` : ''}
                ${geofence.max_altitude ? `<p>Max Alt: ${geofence.max_altitude}m</p>` : ''}
                <div class="popup-actions">
                    <button onclick="geofenceEditor.editGeofence('${geofence.id}')">Edit</button>
                    <button onclick="geofenceEditor.deleteGeofence('${geofence.id}')">Delete</button>
                </div>
            </div>
        `;
    }

    highlightBreach(data) {
        // Flash the breached geofence
        const layer = this.geofenceLayers.find(l => l.geofenceId === data.geofence_id);
        if (!layer) return;

        const originalStyle = {
            fillOpacity: layer.options.fillOpacity,
            weight: layer.options.weight
        };

        let flashes = 0;
        const flashInterval = setInterval(() => {
            if (flashes % 2 === 0) {
                layer.setStyle({ fillOpacity: 0.6, weight: 4 });
            } else {
                layer.setStyle(originalStyle);
            }
            flashes++;
            if (flashes >= 6) {
                clearInterval(flashInterval);
                layer.setStyle(originalStyle);
            }
        }, 200);
    }

    // ==================== Quick Setup ====================

    createHomeGeofence(lat, lng, radius = 1000, maxAltitude = 120) {
        this.createCircularGeofence('Home Zone', lat, lng, radius, 'inclusion');

        // Also set altitude limit
        this.socket.emit('set_altitude_limit', { max_altitude: maxAltitude });
    }

    createNoFlyZone(name, coordinates) {
        this.createPolygonGeofence(name, coordinates, 'exclusion');
    }

    // ==================== UI ====================

    editGeofence(id) {
        const geofence = this.geofences.find(gf => gf.id === id);
        if (!geofence) return;

        this.activeGeofence = geofence;
        this.showGeofenceEditor(geofence);
    }

    showGeofenceEditor(geofence) {
        const panel = document.getElementById('geofence-editor');
        if (!panel) return;

        panel.innerHTML = `
            <h4>Edit Geofence</h4>
            <div class="form-group">
                <label>Name</label>
                <input type="text" id="gf-name" value="${geofence.name}" />
            </div>
            <div class="form-group">
                <label>Type</label>
                <select id="gf-type">
                    <option value="inclusion" ${geofence.type === 'inclusion' ? 'selected' : ''}>Inclusion</option>
                    <option value="exclusion" ${geofence.type === 'exclusion' ? 'selected' : ''}>Exclusion</option>
                </select>
            </div>
            ${geofence.shape === 'circle' ? `
                <div class="form-group">
                    <label>Radius (m)</label>
                    <input type="number" id="gf-radius" value="${geofence.radius}" />
                </div>
            ` : ''}
            <div class="form-group">
                <label>Max Altitude (m)</label>
                <input type="number" id="gf-max-alt" value="${geofence.max_altitude || 120}" />
            </div>
            <button onclick="geofenceEditor.saveGeofenceEdit('${geofence.id}')">Save</button>
            <button onclick="geofenceEditor.deleteGeofence('${geofence.id}')">Delete</button>
        `;

        panel.style.display = 'block';
    }

    saveGeofenceEdit(id) {
        const updates = {
            name: document.getElementById('gf-name')?.value,
            type: document.getElementById('gf-type')?.value,
            max_altitude: parseFloat(document.getElementById('gf-max-alt')?.value)
        };

        const radiusEl = document.getElementById('gf-radius');
        if (radiusEl) {
            updates.radius = parseFloat(radiusEl.value);
        }

        this.updateGeofence(id, updates);
    }

    updateGeofenceList() {
        const listEl = document.getElementById('geofence-list');
        if (!listEl) return;

        listEl.innerHTML = '';

        this.geofences.forEach(gf => {
            const item = document.createElement('div');
            item.className = `geofence-item ${gf.type}`;
            item.innerHTML = `
                <span class="gf-icon">${gf.type === 'exclusion' ? '🚫' : '✓'}</span>
                <span class="gf-name">${gf.name}</span>
                <span class="gf-type">${gf.shape}</span>
            `;
            item.onclick = () => this.focusGeofence(gf.id);
            listEl.appendChild(item);
        });
    }

    focusGeofence(id) {
        const layer = this.geofenceLayers.find(l => l.geofenceId === id);
        if (layer) {
            this.map.fitBounds(layer.getBounds().pad(0.1));
            layer.openPopup();
        }
    }

    showNotification(message, type = 'info') {
        if (window.showToast) {
            window.showToast(message, type);
        } else {
            console.log(`[${type}] ${message}`);
        }
    }
}

// Export for use
window.GeofenceEditor = GeofenceEditor;
