class CameraController {
    constructor(canvas) {
        this.canvas = canvas;

        // FPS camera state (position + orientation)
        this.position = [0, 0, 5];  // World position
        this.yaw = 0.0;              // Horizontal rotation (radians)
        this.pitch = 0.3;            // Vertical rotation (radians)

        // Camera parameters
        this.fovYDeg = 60.0;
        this.width = 640;
        this.height = 480;

        // Movement settings
        this.moveSpeed = 5.0;              // Units per second
        this.mouseSensitivity = 0.003;     // Radians per pixel

        // Input state
        this.keys = new Set();
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Mouse drag to look around (yaw/pitch)
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

            // Update yaw (left/right) and pitch (up/down)
            this.yaw += dx * this.mouseSensitivity;
            this.pitch -= dy * this.mouseSensitivity;  // Inverted Y

            // Clamp pitch to prevent gimbal lock (±89 degrees)
            const maxPitch = Math.PI / 2 - 0.01;
            this.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.pitch));

            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });

        // Keyboard input
        window.addEventListener('keydown', (e) => {
            this.keys.add(e.key.toLowerCase());
        });

        window.addEventListener('keyup', (e) => {
            this.keys.delete(e.key.toLowerCase());
        });

        // Wheel for movement speed adjustment
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const speedChange = 1 + Math.sign(e.deltaY) * 0.1;
            this.moveSpeed *= speedChange;
            this.moveSpeed = Math.max(0.1, Math.min(50.0, this.moveSpeed));
            console.log(`Movement speed: ${this.moveSpeed.toFixed(2)} units/sec`);
        });
    }

    update(deltaTime) {
        // Calculate basis vectors from yaw/pitch
        const forward = this.getForwardVector();
        const right = this.getRightVector();

        // Accumulate movement direction
        let moveDir = [0, 0, 0];

        // W/S: Forward/Backward
        if (this.keys.has('w')) {
            moveDir[0] += forward[0];
            moveDir[1] += forward[1];
            moveDir[2] += forward[2];
        }
        if (this.keys.has('s')) {
            moveDir[0] -= forward[0];
            moveDir[1] -= forward[1];
            moveDir[2] -= forward[2];
        }

        // A/D: Strafe left/right
        if (this.keys.has('a')) {
            moveDir[0] -= right[0];
            moveDir[1] -= right[1];
            moveDir[2] -= right[2];
        }
        if (this.keys.has('d')) {
            moveDir[0] += right[0];
            moveDir[1] += right[1];
            moveDir[2] += right[2];
        }

        // Q/E: Up/Down (world space)
        if (this.keys.has('q')) {
            moveDir[1] -= 1;
        }
        if (this.keys.has('e')) {
            moveDir[1] += 1;
        }

        // Normalize diagonal movement
        const len = Math.sqrt(moveDir[0]**2 + moveDir[1]**2 + moveDir[2]**2);
        if (len > 0) {
            moveDir[0] /= len;
            moveDir[1] /= len;
            moveDir[2] /= len;

            // Apply speed and delta time (sprint modifier: 2x with Shift)
            const speed = this.moveSpeed * (this.keys.has('shift') ? 2.0 : 1.0);
            this.position[0] += moveDir[0] * speed * deltaTime;
            this.position[1] += moveDir[1] * speed * deltaTime;
            this.position[2] += moveDir[2] * speed * deltaTime;
        }
    }

    getForwardVector() {
        // Forward direction from yaw/pitch (negated for correct movement direction)
        return [
            -Math.sin(this.yaw) * Math.cos(this.pitch),
            -Math.sin(this.pitch),
            -Math.cos(this.yaw) * Math.cos(this.pitch)
        ];
    }

    getRightVector() {
        // Right = forward × world_up (cross product)
        const fwd = this.getForwardVector();
        const up = [0, 1, 0];
        return [
            fwd[1] * up[2] - fwd[2] * up[1],
            fwd[2] * up[0] - fwd[0] * up[2],
            fwd[0] * up[1] - fwd[1] * up[0]
        ];
    }

    getRotationMatrix() {
        // Build rotation from yaw and pitch
        const cy = Math.cos(this.yaw);
        const sy = Math.sin(this.yaw);
        const cp = Math.cos(this.pitch);
        const sp = Math.sin(this.pitch);

        // Camera basis vectors
        const forward = [
            sy * cp,
            sp,
            cy * cp
        ];

        const right = [
            cy,
            0,
            -sy
        ];

        const up = [
            -sy * sp,
            cp,
            -cy * sp
        ];

        // Rotation matrix (world-to-camera)
        // Backend expects row vectors: [right, up, -forward]
        return [
            [right[0], up[0], -forward[0]],
            [right[1], up[1], -forward[1]],
            [right[2], up[2], -forward[2]]
        ];
    }

    toTauriParams() {
        return {
            position: this.position,  // Direct position (simpler!)
            rotation: this.getRotationMatrix(),  // From yaw/pitch
            width: this.width,
            height: this.height,
            fov_y_deg: this.fovYDeg
        };
    }

    setResolution(width, height) {
        // Update render resolution (used when model has training resolution)
        this.width = width;
        this.height = height;
    }

    setCameraPose(position, rotationMatrix) {
        // Set camera to exact position and rotation from COLMAP
        // rotationMatrix is row-major: [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]

        // Set position directly
        this.position = [...position];

        // Extract forward vector (third column, negated to match our convention)
        const forward = [
            -rotationMatrix[0][2],
            -rotationMatrix[1][2],
            -rotationMatrix[2][2]
        ];

        // Extract yaw from forward's XZ projection (negated to flip 180°)
        this.yaw = Math.atan2(-forward[0], -forward[2]);

        // Extract pitch from forward's Y component (negated to flip)
        const fwdXZ = Math.sqrt(forward[0]**2 + forward[2]**2);
        this.pitch = Math.atan2(-forward[1], fwdXZ);

        console.log(`Camera pose set: pos=${this.position}, yaw=${this.yaw}, pitch=${this.pitch}`);
    }

    frameModel(boundsMin, boundsMax, suggestedDistance) {
        // Calculate center of bounding box
        const center = [
            (boundsMin[0] + boundsMax[0]) / 2,
            (boundsMin[1] + boundsMax[1]) / 2,
            (boundsMin[2] + boundsMax[2]) / 2
        ];

        // Calculate distance needed to frame model
        let distance = suggestedDistance;
        if (!distance) {
            const dx = boundsMax[0] - boundsMin[0];
            const dy = boundsMax[1] - boundsMin[1];
            const dz = boundsMax[2] - boundsMin[2];
            const maxDim = Math.max(dx, dy, dz);
            distance = maxDim / Math.tan(this.fovYDeg * Math.PI / 360) * 1.5;
        }

        // Set to nice viewing angle (45° horizontal, 30° vertical)
        this.yaw = Math.PI / 4;
        this.pitch = Math.PI / 6;

        // Position camera behind and above model
        const forward = this.getForwardVector();
        this.position = [
            center[0] - forward[0] * distance,
            center[1] - forward[1] * distance,
            center[2] - forward[2] * distance
        ];

        // Set initial move speed based on model size
        this.moveSpeed = distance * 0.1;

        console.log(`Camera framed at distance ${distance}, position ${this.position}`);
    }
}
