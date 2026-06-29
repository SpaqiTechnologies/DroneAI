/**
 * Telemetry Charts Module
 * Real-time visualization of drone telemetry data using Chart.js
 */

class TelemetryCharts {
    constructor(socket) {
        this.socket = socket;
        this.charts = {};
        this.bufferSize = 60; // 60 seconds of data
        this.data = {
            timestamps: [],
            altitude: [],
            battery: [],
            speed: [],
            satellites: []
        };

        this.colors = {
            altitude: { border: '#00d4ff', bg: 'rgba(0, 212, 255, 0.1)' },
            battery: { border: '#00ff88', bg: 'rgba(0, 255, 136, 0.1)' },
            speed: { border: '#f0abfc', bg: 'rgba(240, 171, 252, 0.1)' },
            satellites: { border: '#ffc107', bg: 'rgba(255, 193, 7, 0.1)' }
        };

        this.baseConfig = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(10, 10, 30, 0.9)',
                    titleColor: '#00d4ff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(0, 212, 255, 0.3)',
                    borderWidth: 1,
                    padding: 8,
                    displayColors: false
                }
            },
            scales: {
                x: {
                    display: false,
                    grid: { display: false }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.4)',
                        font: { size: 9, family: 'JetBrains Mono' },
                        maxTicksLimit: 4,
                        padding: 4
                    },
                    border: { display: false }
                }
            }
        };
    }

    init() {
        this.createChart('altitude', 'altitude-chart', 0, 150, 'm');
        this.createChart('battery', 'battery-chart', 0, 100, '%');
        this.createChart('speed', 'speed-chart', 0, 30, 'm/s');
        this.setupSocketListener();
        console.log('[TelemetryCharts] Initialized');
    }

    createChart(name, canvasId, min, max, unit) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`[TelemetryCharts] Canvas ${canvasId} not found`);
            return;
        }

        const ctx = canvas.getContext('2d');
        const color = this.colors[name];

        this.charts[name] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: color.border,
                    backgroundColor: color.bg,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBackgroundColor: color.border,
                    borderWidth: 2
                }]
            },
            options: {
                ...this.baseConfig,
                scales: {
                    ...this.baseConfig.scales,
                    y: {
                        ...this.baseConfig.scales.y,
                        min: min,
                        max: max
                    }
                },
                plugins: {
                    ...this.baseConfig.plugins,
                    tooltip: {
                        ...this.baseConfig.plugins.tooltip,
                        callbacks: {
                            label: (ctx) => `${ctx.parsed.y.toFixed(1)} ${unit}`
                        }
                    }
                }
            }
        });
    }

    setupSocketListener() {
        this.socket.on('drone_state', (state) => {
            this.updateData(state);
        });
    }

    updateData(state) {
        const now = new Date().toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });

        // Add new data points
        this.data.timestamps.push(now);
        this.data.altitude.push(state.position?.alt || 0);
        this.data.battery.push(state.battery || 0);
        this.data.speed.push(state.gps?.speed || 0);
        this.data.satellites.push(state.gps?.satellites || 0);

        // Trim to buffer size
        if (this.data.timestamps.length > this.bufferSize) {
            this.data.timestamps.shift();
            this.data.altitude.shift();
            this.data.battery.shift();
            this.data.speed.shift();
            this.data.satellites.shift();
        }

        // Update all charts
        this.updateChart('altitude', this.data.altitude);
        this.updateChart('battery', this.data.battery);
        this.updateChart('speed', this.data.speed);
    }

    updateChart(name, values) {
        const chart = this.charts[name];
        if (!chart) return;

        chart.data.labels = this.data.timestamps;
        chart.data.datasets[0].data = values;
        chart.update('none'); // No animation for performance
    }

    // Get current stats for display
    getStats() {
        const len = this.data.altitude.length;
        if (len === 0) return null;

        return {
            altitude: {
                current: this.data.altitude[len - 1],
                max: Math.max(...this.data.altitude),
                avg: this.data.altitude.reduce((a, b) => a + b, 0) / len
            },
            battery: {
                current: this.data.battery[len - 1],
                min: Math.min(...this.data.battery),
                drain: len > 1 ?
                    this.data.battery[0] - this.data.battery[len - 1] : 0
            },
            speed: {
                current: this.data.speed[len - 1],
                max: Math.max(...this.data.speed),
                avg: this.data.speed.reduce((a, b) => a + b, 0) / len
            }
        };
    }

    // Clear all data
    clear() {
        this.data = {
            timestamps: [],
            altitude: [],
            battery: [],
            speed: [],
            satellites: []
        };

        Object.values(this.charts).forEach(chart => {
            chart.data.labels = [];
            chart.data.datasets[0].data = [];
            chart.update('none');
        });
    }

    // Destroy charts (cleanup)
    destroy() {
        Object.values(this.charts).forEach(chart => chart.destroy());
        this.charts = {};
    }
}

// Export for use
window.TelemetryCharts = TelemetryCharts;
