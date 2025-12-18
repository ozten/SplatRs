class CameraController {
    constructor(canvas) {
        this.canvas = canvas;

        // Orbit parameters (spherical coordinates)
        this.azimuth = 0.0;      // Horizontal rotation (radians)
        this.elevation = 0.3;    // Vertical rotation (radians), clamped to [-π/2, π/2]
        this.distance = 5.0;     // Distance from target
        this.target = [0, 0, 0]; // Look-at point

        // Camera parameters
        this.fovYDeg = 60.0;
        this.width = 640;
        this.height = 480;

        // Movement speeds
        this.orbitSpeed = 0.005;     // Radians per pixel
        this.moveSpeed = 0.5;        // Units per frame (10x faster)
        this.zoomSpeed = 0.1;        // Distance multiplier per wheel tick

        // Input state
        this.keys = new Set();
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Mouse orbit
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });

        window.addEventListener('mouseup', () => {
            this.isDragging = false;
        });

        window.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;

            const dx = e.clientX - this.lastMouseX;
            const dy = e.clientY - this.lastMouseY;

            // Flip azimuth direction so dragging right rotates camera right (world appears to move right)
            this.azimuth += dx * this.orbitSpeed;
            this.elevation = Math.max(-Math.PI / 2 + 0.01,
                                     Math.min(Math.PI / 2 - 0.01,
                                             this.elevation - dy * this.orbitSpeed));

            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });

        // Mouse wheel zoom
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.distance *= (1 + Math.sign(e.deltaY) * this.zoomSpeed);
            this.distance = Math.max(0.1, this.distance); // Prevent negative
        });

        // Keyboard WASD
        window.addEventListener('keydown', (e) => {
            this.keys.add(e.key.toLowerCase());
        });

        window.addEventListener('keyup', (e) => {
            this.keys.delete(e.key.toLowerCase());
        });
    }

    update() {
        // Get right vector for strafing
        const right = this.getRightVector();

        // W/S: Zoom in/out (adjust distance) - always moves toward/away from what you're looking at
        if (this.keys.has('w')) {
            this.distance = Math.max(0.1, this.distance - this.moveSpeed);
        }
        if (this.keys.has('s')) {
            this.distance += this.moveSpeed;
        }

        // A/D: Strafe left/right (move target perpendicular to view)
        if (this.keys.has('a')) {
            this.target[0] -= right[0] * this.moveSpeed;
            this.target[1] -= right[1] * this.moveSpeed;
            this.target[2] -= right[2] * this.moveSpeed;
        }
        if (this.keys.has('d')) {
            this.target[0] += right[0] * this.moveSpeed;
            this.target[1] += right[1] * this.moveSpeed;
            this.target[2] += right[2] * this.moveSpeed;
        }
    }

    getCameraPosition() {
        // Convert spherical to Cartesian
        const x = this.target[0] + this.distance * Math.cos(this.elevation) * Math.sin(this.azimuth);
        const y = this.target[1] + this.distance * Math.sin(this.elevation);
        const z = this.target[2] + this.distance * Math.cos(this.elevation) * Math.cos(this.azimuth);
        return [x, y, z];
    }

    getForwardVector() {
        // Direction from camera to target (normalized)
        const pos = this.getCameraPosition();
        const dx = this.target[0] - pos[0];
        const dy = this.target[1] - pos[1];
        const dz = this.target[2] - pos[2];
        const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
        return [dx/len, dy/len, dz/len];
    }

    getRightVector() {
        // Right = forward × world_up (cross product)
        const fwd = this.getForwardVector();
        const up = [0, 1, 0]; // World up
        return [
            fwd[1] * up[2] - fwd[2] * up[1],
            fwd[2] * up[0] - fwd[0] * up[2],
            fwd[0] * up[1] - fwd[1] * up[0]
        ];
    }

    getRotationMatrix() {
        // Build camera basis vectors
        const forward = this.getForwardVector();
        const right = this.getRightVector();

        // Up = right × forward
        const up = [
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0]
        ];

        // Camera-to-world rotation (column vectors: right, up, -forward)
        // World-to-camera is transpose
        return [
            [right[0], up[0], -forward[0]],
            [right[1], up[1], -forward[1]],
            [right[2], up[2], -forward[2]]
        ];
    }

    toTauriParams() {
        return {
            position: this.getCameraPosition(),
            rotation: this.getRotationMatrix(),
            width: this.width,
            height: this.height,
            fov_y_deg: this.fovYDeg  // Use snake_case to match Rust
        };
    }

    frameModel(boundsMin, boundsMax, suggestedDistance) {
        // Center target on model
        this.target = [
            (boundsMin[0] + boundsMax[0]) / 2,
            (boundsMin[1] + boundsMax[1]) / 2,
            (boundsMin[2] + boundsMax[2]) / 2
        ];

        // Use suggested distance or calculate from bounding box
        if (suggestedDistance) {
            this.distance = suggestedDistance;
        } else {
            const dx = boundsMax[0] - boundsMin[0];
            const dy = boundsMax[1] - boundsMin[1];
            const dz = boundsMax[2] - boundsMin[2];
            const maxDim = Math.max(dx, dy, dz);
            this.distance = maxDim / Math.tan(this.fovYDeg * Math.PI / 360) * 1.5;
        }

        // Reset orientation to look at model from a good angle
        this.azimuth = Math.PI / 4;  // 45 degrees
        this.elevation = Math.PI / 6; // 30 degrees
    }
}
