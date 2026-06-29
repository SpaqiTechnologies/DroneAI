/**
 * 3D Visualization Module
 * Three.js-based 3D drone flight visualization
 */

class Visualization3D {
    constructor(containerId, socket) {
        this.containerId = containerId;
        this.socket = socket;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.drone = null;
        this.pathLine = null;
        this.pathPoints = [];
        this.isActive = false;
        this.homePosition = { lat: 52.52, lon: 13.405 };
        this.animationId = null;

        // Drone state
        this.droneState = {
            position: { lat: 52.52, lon: 13.405, alt: 0 },
            heading: 0,
            armed: false
        };
    }

    init() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.warn('[3D] Container not found:', this.containerId);
            return false;
        }

        this.setupScene(container);
        this.createGround();
        this.createDrone();
        this.setupLighting();
        this.setupControls(container);
        this.setupSocketListeners();

        console.log('[3D] Visualization initialized');
        return true;
    }

    setupScene(container) {
        // Scene with dark background
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a1a);
        this.scene.fog = new THREE.Fog(0x0a0a1a, 80, 200);

        // Camera
        const aspect = container.clientWidth / container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 500);
        this.camera.position.set(30, 25, 30);
        this.camera.lookAt(0, 5, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(this.renderer.domElement);

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    createGround() {
        // Grid helper
        const grid = new THREE.GridHelper(100, 50, 0x00d4ff, 0x1a1a2e);
        grid.material.opacity = 0.3;
        grid.material.transparent = true;
        this.scene.add(grid);

        // Ground plane
        const groundGeo = new THREE.PlaneGeometry(100, 100);
        const groundMat = new THREE.MeshStandardMaterial({
            color: 0x0a0a1a,
            roughness: 0.9,
            metalness: 0.1
        });
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Home marker
        const homeGeo = new THREE.CylinderGeometry(1, 1, 0.2, 32);
        const homeMat = new THREE.MeshStandardMaterial({
            color: 0x00ff88,
            emissive: 0x00ff88,
            emissiveIntensity: 0.3
        });
        const home = new THREE.Mesh(homeGeo, homeMat);
        home.position.y = 0.1;
        this.scene.add(home);

        // Home label ring
        const ringGeo = new THREE.RingGeometry(1.5, 2, 32);
        const ringMat = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.3
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.05;
        this.scene.add(ring);
    }

    createDrone() {
        this.drone = new THREE.Group();

        // Body - sleek design
        const bodyGeo = new THREE.BoxGeometry(1.5, 0.4, 1.5);
        const bodyMat = new THREE.MeshStandardMaterial({
            color: 0x2a2a3a,
            roughness: 0.3,
            metalness: 0.8
        });
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.castShadow = true;
        this.drone.add(body);

        // Arms
        const armGeo = new THREE.BoxGeometry(3, 0.15, 0.2);
        const armMat = new THREE.MeshStandardMaterial({
            color: 0x1a1a2a,
            roughness: 0.5,
            metalness: 0.6
        });

        const arm1 = new THREE.Mesh(armGeo, armMat);
        arm1.rotation.y = Math.PI / 4;
        arm1.castShadow = true;
        this.drone.add(arm1);

        const arm2 = new THREE.Mesh(armGeo, armMat);
        arm2.rotation.y = -Math.PI / 4;
        arm2.castShadow = true;
        this.drone.add(arm2);

        // Propellers with glow
        const propPositions = [
            [1.2, 0.25, 1.2],
            [1.2, 0.25, -1.2],
            [-1.2, 0.25, 1.2],
            [-1.2, 0.25, -1.2]
        ];

        this.propellers = [];
        propPositions.forEach((pos, i) => {
            // Propeller disc
            const propGeo = new THREE.CylinderGeometry(0.6, 0.6, 0.05, 24);
            const propMat = new THREE.MeshStandardMaterial({
                color: 0x00d4ff,
                transparent: true,
                opacity: 0.6,
                emissive: 0x00d4ff,
                emissiveIntensity: 0.2
            });
            const prop = new THREE.Mesh(propGeo, propMat);
            prop.position.set(...pos);
            this.propellers.push(prop);
            this.drone.add(prop);

            // Motor hub
            const hubGeo = new THREE.CylinderGeometry(0.15, 0.15, 0.2, 16);
            const hubMat = new THREE.MeshStandardMaterial({
                color: 0x333344,
                metalness: 0.9,
                roughness: 0.2
            });
            const hub = new THREE.Mesh(hubGeo, hubMat);
            hub.position.set(pos[0], pos[1] + 0.1, pos[2]);
            this.drone.add(hub);
        });

        // LED indicators
        const ledGeo = new THREE.SphereGeometry(0.08, 16, 16);

        // Front LEDs (white)
        const frontLedMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const frontLed1 = new THREE.Mesh(ledGeo, frontLedMat);
        frontLed1.position.set(0.6, 0, 0.75);
        this.drone.add(frontLed1);
        const frontLed2 = new THREE.Mesh(ledGeo, frontLedMat);
        frontLed2.position.set(-0.6, 0, 0.75);
        this.drone.add(frontLed2);

        // Rear LEDs (red)
        this.rearLedMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const rearLed1 = new THREE.Mesh(ledGeo, this.rearLedMat);
        rearLed1.position.set(0.6, 0, -0.75);
        this.drone.add(rearLed1);
        const rearLed2 = new THREE.Mesh(ledGeo, this.rearLedMat);
        rearLed2.position.set(-0.6, 0, -0.75);
        this.drone.add(rearLed2);

        // Camera gimbal
        const gimbalGeo = new THREE.BoxGeometry(0.3, 0.2, 0.3);
        const gimbalMat = new THREE.MeshStandardMaterial({
            color: 0x1a1a2a,
            metalness: 0.7
        });
        const gimbal = new THREE.Mesh(gimbalGeo, gimbalMat);
        gimbal.position.set(0, -0.3, 0.3);
        this.drone.add(gimbal);

        // Point light on drone
        const droneLight = new THREE.PointLight(0x00d4ff, 0.5, 10);
        droneLight.position.set(0, 1, 0);
        this.drone.add(droneLight);

        this.drone.position.y = 5;
        this.scene.add(this.drone);
    }

    setupLighting() {
        // Ambient light
        const ambient = new THREE.AmbientLight(0x404060, 0.4);
        this.scene.add(ambient);

        // Main directional light
        const sun = new THREE.DirectionalLight(0xffffff, 0.6);
        sun.position.set(30, 50, 30);
        sun.castShadow = true;
        sun.shadow.mapSize.width = 1024;
        sun.shadow.mapSize.height = 1024;
        sun.shadow.camera.near = 0.5;
        sun.shadow.camera.far = 100;
        sun.shadow.camera.left = -30;
        sun.shadow.camera.right = 30;
        sun.shadow.camera.top = 30;
        sun.shadow.camera.bottom = -30;
        this.scene.add(sun);

        // Hemisphere light for sky/ground color
        const hemi = new THREE.HemisphereLight(0x4466aa, 0x111122, 0.3);
        this.scene.add(hemi);
    }

    setupControls(container) {
        // Simple orbit controls implementation
        this.controls = {
            target: new THREE.Vector3(0, 5, 0),
            spherical: { radius: 40, phi: Math.PI / 4, theta: Math.PI / 4 },
            isDragging: false,
            lastMouse: { x: 0, y: 0 }
        };

        container.addEventListener('mousedown', (e) => {
            this.controls.isDragging = true;
            this.controls.lastMouse = { x: e.clientX, y: e.clientY };
        });

        container.addEventListener('mousemove', (e) => {
            if (!this.controls.isDragging) return;

            const dx = e.clientX - this.controls.lastMouse.x;
            const dy = e.clientY - this.controls.lastMouse.y;

            this.controls.spherical.theta -= dx * 0.01;
            this.controls.spherical.phi += dy * 0.01;
            this.controls.spherical.phi = Math.max(
                0.1,
                Math.min(Math.PI / 2 - 0.1, this.controls.spherical.phi)
            );

            this.controls.lastMouse = { x: e.clientX, y: e.clientY };
            this.updateCameraPosition();
        });

        container.addEventListener('mouseup', () => {
            this.controls.isDragging = false;
        });

        container.addEventListener('mouseleave', () => {
            this.controls.isDragging = false;
        });

        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.controls.spherical.radius += e.deltaY * 0.05;
            this.controls.spherical.radius = Math.max(
                10,
                Math.min(100, this.controls.spherical.radius)
            );
            this.updateCameraPosition();
        });

        this.updateCameraPosition();
    }

    updateCameraPosition() {
        const s = this.controls.spherical;
        this.camera.position.x =
            s.radius * Math.sin(s.phi) * Math.cos(s.theta) + this.controls.target.x;
        this.camera.position.y =
            s.radius * Math.cos(s.phi) + this.controls.target.y;
        this.camera.position.z =
            s.radius * Math.sin(s.phi) * Math.sin(s.theta) + this.controls.target.z;
        this.camera.lookAt(this.controls.target);
    }

    setupSocketListeners() {
        this.socket.on('drone_state', (state) => {
            if (this.isActive) {
                this.updateDronePosition(state);
            }
        });

        this.socket.on('simulation_started', () => {
            this.pathPoints = [];
            this.updatePath();
        });
    }

    updateDronePosition(state) {
        if (!this.drone) return;

        // Convert GPS to local coordinates (meters from home)
        const x = (state.position.lon - this.homePosition.lon) * 111000 *
            Math.cos(this.homePosition.lat * Math.PI / 180);
        const z = -(state.position.lat - this.homePosition.lat) * 111000;
        const y = state.position.alt / 2; // Scale altitude

        // Update drone position
        this.drone.position.set(x, Math.max(0.5, y), z);

        // Update drone heading
        const heading = (state.gps?.heading || 0) * Math.PI / 180;
        this.drone.rotation.y = -heading;

        // Update camera target to follow drone
        this.controls.target.set(x, y, z);
        this.updateCameraPosition();

        // Animate propellers when armed
        if (state.armed) {
            this.propellers.forEach((prop, i) => {
                prop.rotation.y += 0.5 * (i % 2 === 0 ? 1 : -1);
            });
            this.rearLedMat.color.setHex(0x00ff00);
        } else {
            this.rearLedMat.color.setHex(0xff0000);
        }

        // Add to path trail
        const pathPoint = new THREE.Vector3(x, Math.max(0.5, y), z);
        if (this.pathPoints.length === 0 ||
            pathPoint.distanceTo(this.pathPoints[this.pathPoints.length - 1]) > 0.5) {
            this.pathPoints.push(pathPoint);
            if (this.pathPoints.length > 500) {
                this.pathPoints.shift();
            }
            this.updatePath();
        }
    }

    updatePath() {
        if (this.pathLine) {
            this.scene.remove(this.pathLine);
        }

        if (this.pathPoints.length < 2) return;

        const geometry = new THREE.BufferGeometry().setFromPoints(this.pathPoints);
        const material = new THREE.LineBasicMaterial({
            color: 0x7c3aed,
            transparent: true,
            opacity: 0.6
        });

        this.pathLine = new THREE.Line(geometry, material);
        this.scene.add(this.pathLine);
    }

    animate() {
        if (!this.isActive) return;

        this.animationId = requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }

    activate() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        this.isActive = true;
        container.style.display = 'block';
        this.onResize();
        this.animate();
        console.log('[3D] Activated');
    }

    deactivate() {
        const container = document.getElementById(this.containerId);
        if (container) {
            container.style.display = 'none';
        }

        this.isActive = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        console.log('[3D] Deactivated');
    }

    toggle() {
        if (this.isActive) {
            this.deactivate();
        } else {
            this.activate();
        }
        return this.isActive;
    }

    onResize() {
        const container = document.getElementById(this.containerId);
        if (!container || !this.renderer || !this.camera) return;

        const width = container.clientWidth;
        const height = container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    clearPath() {
        this.pathPoints = [];
        if (this.pathLine) {
            this.scene.remove(this.pathLine);
            this.pathLine = null;
        }
    }

    setHomePosition(lat, lon) {
        this.homePosition = { lat, lon };
    }
}

// Export for use
window.Visualization3D = Visualization3D;
