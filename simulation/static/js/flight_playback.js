/**
 * Flight Log Playback Module
 * Replay recorded flight data with timeline scrubber
 */

class FlightPlayback {
    constructor(map, socket) {
        this.map = map;
        this.socket = socket;
        this.flightLog = [];
        this.playbackIndex = 0;
        this.isPlaying = false;
        this.playbackSpeed = 1;
        this.playbackInterval = null;
        this.pathLine = null;
        this.playbackMarker = null;
        this.startTime = 0;
        this.endTime = 0;

        this.colors = {
            path: '#7c3aed',
            marker: '#00d4ff',
            highlight: '#00ff88'
        };
    }

    init() {
        this.setupSocketListeners();
        console.log('[FlightPlayback] Initialized');
    }

    setupSocketListeners() {
        this.socket.on('flight_logs_list', (data) => {
            this.updateLogsList(data.logs || []);
        });

        this.socket.on('flight_log_data', (data) => {
            this.loadFlightLog(data.records || []);
        });
    }

    updateLogsList(logs) {
        const container = document.getElementById('playback-logs-list');
        if (!container) return;

        if (logs.length === 0) {
            container.innerHTML = `
                <div class="empty-state">No flight logs available</div>
            `;
            return;
        }

        container.innerHTML = logs.map((log, i) => `
            <div class="log-item" onclick="flightPlayback.loadLog('${log.session_id}')">
                <div class="log-info">
                    <span class="log-date">${this.formatDate(log.start_time)}</span>
                    <span class="log-duration">${this.formatDuration(log.duration)}</span>
                </div>
                <span class="log-records">${log.records} pts</span>
            </div>
        `).join('');
    }

    loadLog(sessionId) {
        this.socket.emit('load_flight_log', { session_id: sessionId });
        console.log('[FlightPlayback] Loading log:', sessionId);
    }

    loadFlightLog(records) {
        if (!records || records.length === 0) {
            console.warn('[FlightPlayback] No records to load');
            return;
        }

        this.flightLog = records;
        this.playbackIndex = 0;
        this.startTime = records[0].timestamp;
        this.endTime = records[records.length - 1].timestamp;

        this.updateScrubber();
        this.drawFlightPath();
        this.showPlaybackControls(true);

        console.log('[FlightPlayback] Loaded', records.length, 'records');
    }

    drawFlightPath() {
        // Remove existing path
        if (this.pathLine) {
            this.map.removeLayer(this.pathLine);
        }

        if (this.flightLog.length < 2) return;

        // Create path coordinates
        const latlngs = this.flightLog.map(r => [r.lat, r.lon]);

        // Draw gradient path
        this.pathLine = L.polyline(latlngs, {
            color: this.colors.path,
            weight: 3,
            opacity: 0.7,
            smoothFactor: 1
        }).addTo(this.map);

        // Fit map to path bounds
        this.map.fitBounds(this.pathLine.getBounds().pad(0.1));

        // Add start/end markers
        this.addPathMarkers();
    }

    addPathMarkers() {
        const first = this.flightLog[0];
        const last = this.flightLog[this.flightLog.length - 1];

        // Start marker
        L.marker([first.lat, first.lon], {
            icon: L.divIcon({
                className: 'playback-marker start',
                html: '<div class="marker-dot start"></div>',
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            })
        }).addTo(this.map);

        // End marker
        L.marker([last.lat, last.lon], {
            icon: L.divIcon({
                className: 'playback-marker end',
                html: '<div class="marker-dot end"></div>',
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            })
        }).addTo(this.map);
    }

    togglePlayback() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    play() {
        if (this.flightLog.length === 0) return;

        this.isPlaying = true;
        this.updatePlaybackButton();

        // Resume from end if at end
        if (this.playbackIndex >= this.flightLog.length - 1) {
            this.playbackIndex = 0;
        }

        const intervalMs = 100 / this.playbackSpeed;
        this.playbackInterval = setInterval(() => {
            this.stepForward();
        }, intervalMs);
    }

    pause() {
        this.isPlaying = false;
        this.updatePlaybackButton();

        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
            this.playbackInterval = null;
        }
    }

    stepForward() {
        if (this.playbackIndex >= this.flightLog.length - 1) {
            this.pause();
            return;
        }

        this.playbackIndex++;
        this.updatePlaybackPosition();
    }

    seekTo(percent) {
        const index = Math.floor((percent / 100) * (this.flightLog.length - 1));
        this.playbackIndex = Math.max(0, Math.min(index, this.flightLog.length - 1));
        this.updatePlaybackPosition();
    }

    seekToIndex(index) {
        this.playbackIndex = Math.max(0, Math.min(index, this.flightLog.length - 1));
        this.updatePlaybackPosition();
    }

    updatePlaybackPosition() {
        const record = this.flightLog[this.playbackIndex];
        if (!record) return;

        // Update or create playback marker
        if (this.playbackMarker) {
            this.playbackMarker.setLatLng([record.lat, record.lon]);
        } else {
            this.playbackMarker = L.marker([record.lat, record.lon], {
                icon: L.divIcon({
                    className: 'playback-position-marker',
                    html: `<div class="playback-drone-icon">
                        <svg viewBox="0 0 24 24" width="24" height="24">
                            <path fill="${this.colors.marker}"
                                  d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/>
                        </svg>
                    </div>`,
                    iconSize: [24, 24],
                    iconAnchor: [12, 12]
                }),
                rotationAngle: record.heading || 0
            }).addTo(this.map);
        }

        // Rotate marker based on heading
        const iconEl = this.playbackMarker.getElement();
        if (iconEl) {
            iconEl.style.transform += ` rotate(${record.heading || 0}deg)`;
        }

        // Update scrubber
        this.updateScrubber();

        // Update telemetry display
        this.updateTelemetryDisplay(record);

        // Emit playback state for other components
        this.socket.emit('playback_position', record);
    }

    updateScrubber() {
        const scrubber = document.getElementById('playback-scrubber');
        const current = document.getElementById('playback-current');
        const total = document.getElementById('playback-total');

        if (scrubber) {
            scrubber.max = this.flightLog.length - 1;
            scrubber.value = this.playbackIndex;
        }

        if (current && this.flightLog[this.playbackIndex]) {
            const elapsed = this.flightLog[this.playbackIndex].timestamp - this.startTime;
            current.textContent = this.formatTime(elapsed);
        }

        if (total) {
            const duration = this.endTime - this.startTime;
            total.textContent = this.formatTime(duration);
        }
    }

    updatePlaybackButton() {
        const btn = document.getElementById('playback-toggle-btn');
        if (!btn) return;

        if (this.isPlaying) {
            btn.innerHTML = '&#10074;&#10074;'; // Pause icon
            btn.title = 'Pause';
        } else {
            btn.innerHTML = '&#9658;'; // Play icon
            btn.title = 'Play';
        }
    }

    updateTelemetryDisplay(record) {
        // Update UI elements with playback data
        const elements = {
            'playback-alt': record.alt?.toFixed(1) + ' m',
            'playback-speed': (record.speed || 0).toFixed(1) + ' m/s',
            'playback-heading': (record.heading || 0).toFixed(0) + '°',
            'playback-battery': (record.battery || 0).toFixed(0) + '%'
        };

        for (const [id, value] of Object.entries(elements)) {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        }
    }

    setSpeed(speed) {
        this.playbackSpeed = parseFloat(speed);

        // Restart interval if playing
        if (this.isPlaying) {
            this.pause();
            this.play();
        }
    }

    skipBack() {
        const skipAmount = Math.floor(this.flightLog.length * 0.1); // 10%
        this.seekToIndex(this.playbackIndex - skipAmount);
    }

    skipForward() {
        const skipAmount = Math.floor(this.flightLog.length * 0.1); // 10%
        this.seekToIndex(this.playbackIndex + skipAmount);
    }

    showPlaybackControls(show) {
        const container = document.getElementById('playback-panel');
        if (container) {
            container.style.display = show ? 'block' : 'none';
        }
    }

    // Request available flight logs
    requestLogs() {
        this.socket.emit('get_flight_logs');
    }

    // Clear playback
    clear() {
        this.pause();
        this.flightLog = [];
        this.playbackIndex = 0;

        if (this.pathLine) {
            this.map.removeLayer(this.pathLine);
            this.pathLine = null;
        }

        if (this.playbackMarker) {
            this.map.removeLayer(this.playbackMarker);
            this.playbackMarker = null;
        }

        this.showPlaybackControls(false);
    }

    // Utility functions
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    formatDuration(seconds) {
        if (seconds < 60) return `${Math.floor(seconds)}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
        return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    }

    formatDate(timestamp) {
        const date = new Date(timestamp * 1000);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

// Export for use
window.FlightPlayback = FlightPlayback;
