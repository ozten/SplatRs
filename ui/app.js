// Tauri 2.0 API - try multiple paths for compatibility
const invoke = window.__TAURI_INTERNALS__?.invoke ||
               window.__TAURI__?.core?.invoke ||
               window.__TAURI__?.invoke ||
               (async () => { throw new Error('Tauri API not found'); });

// Helper to log to stdout via Rust
async function log(msg) {
    try {
        await invoke('log_to_stdout', { message: msg });
    } catch (e) {
        // Logging failed, ignore
    }
}

let camera = null;
let canvas = null;
let ctx = null;
let isRendering = false;

async function init() {
    await log('init() called');
    canvas = document.getElementById('renderCanvas');
    ctx = canvas.getContext('2d');
    camera = new CameraController(canvas);

    // Draw test pattern to verify canvas works
    drawTestPattern();
    await log('Test pattern drawn');

    // Load available models
    await refreshModelList();

    // Start render loop
    await log('Starting render loop');
    renderLoop();
}

function drawTestPattern() {
    // Draw a colorful test pattern
    ctx.fillStyle = '#ff0000';
    ctx.fillRect(0, 0, 320, 240);
    ctx.fillStyle = '#00ff00';
    ctx.fillRect(320, 0, 320, 240);
    ctx.fillStyle = '#0000ff';
    ctx.fillRect(0, 240, 320, 240);
    ctx.fillStyle = '#ffff00';
    ctx.fillRect(320, 240, 320, 240);

    // Draw text
    ctx.fillStyle = '#ffffff';
    ctx.font = '48px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Canvas Works!', 320, 240);
}

async function refreshModelList() {
    try {
        const models = await invoke('list_models');

        const select = document.getElementById('modelSelect');
        select.innerHTML = '<option value="">-- Select Model --</option>';

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = model.name;
            select.appendChild(option);
        });

        document.getElementById('status').textContent = `Found ${models.length} model(s)`;
    } catch (err) {
        document.getElementById('status').textContent = `Error listing models: ${err}`;
    }
}

async function onModelSelected() {
    const select = document.getElementById('modelSelect');
    const path = select.value;
    if (!path) return;

    await log('onModelSelected called with path: ' + path);

    try {
        document.getElementById('status').textContent = 'Loading model...';
        const metadata = await invoke('load_model', { path });

        await log('Model loaded successfully');

        // Use training resolution if available (non-zero), otherwise keep default 640×480
        if (metadata.training_width > 0 && metadata.training_height > 0) {
            await log(`Using training resolution: ${metadata.training_width}×${metadata.training_height}`);
            camera.setResolution(metadata.training_width, metadata.training_height);
        } else {
            await log('No training resolution in metadata, using default 640×480');
        }

        // Auto-populate dataset path from model metadata
        if (metadata.dataset_path) {
            document.getElementById('datasetRootInput').value = metadata.dataset_path;
            await log(`Auto-populated dataset path: ${metadata.dataset_path}`);
        }

        // Frame camera on model
        camera.frameModel(metadata.bounds_min, metadata.bounds_max, metadata.suggested_camera_distance);

        await log('Camera framed, enabling rendering');

        // Enable rendering
        modelLoaded = true;
        renderAttempts = 0;

        await log('modelLoaded set to true, renderAttempts reset');

        document.getElementById('status').textContent =
            `Loaded ${metadata.num_gaussians} Gaussians - Rendering...`;
    } catch (err) {
        await log('Error loading model: ' + err);
        document.getElementById('status').textContent = `Error: ${err}`;
    }
}

let modelLoaded = false;
let renderAttempts = 0;
let lastFrameTime = performance.now();

async function renderLoop() {
    // Calculate delta time for frame-rate independent movement
    const now = performance.now();
    const deltaTime = (now - lastFrameTime) / 1000;  // Convert to seconds
    lastFrameTime = now;

    camera.update(deltaTime); // Handle WASD input

    if (!modelLoaded) {
        requestAnimationFrame(renderLoop);
        return;
    }

    if (renderAttempts === 0) {
        await log('Starting render loop with model loaded');
    }

    try {
        const params = camera.toTauriParams();

        if (renderAttempts < 3) {
            await log(`Invoking render_frame attempt ${renderAttempts}...`);
        }

        const base64Image = await invoke('render_frame', { cameraParams: params });

        if (renderAttempts < 3) {
            await log(`Got response: ${base64Image.length} bytes total`);
        }

        // Draw image to canvas
        const img = new Image();
        img.onload = async () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            if (renderAttempts < 3) {
                await log('Successfully drew frame to canvas');
                renderAttempts++;
            }
        };
        img.onerror = async (e) => {
            await log('Failed to load image: ' + e);
        };
        img.src = base64Image;

    } catch (err) {
        if (renderAttempts < 3) {
            await log('Render error: ' + err);
            renderAttempts++;
        }
    }

    requestAnimationFrame(renderLoop);
}

async function goToCamera() {
    const cameraIdInput = document.getElementById('cameraIdInput');
    const datasetRootInput = document.getElementById('datasetRootInput');
    const cameraId = parseInt(cameraIdInput.value);
    const datasetRoot = datasetRootInput.value.trim();

    if (isNaN(cameraId) || cameraId < 0) {
        document.getElementById('status').textContent = 'Invalid camera ID';
        return;
    }

    if (!datasetRoot) {
        document.getElementById('status').textContent = 'Please enter a dataset root path';
        return;
    }

    await log(`goToCamera called: id=${cameraId}, dataset=${datasetRoot}`);

    try {
        document.getElementById('status').textContent = `Loading camera ${cameraId}...`;

        const cameraInfo = await invoke('get_camera_by_id', {
            cameraId: cameraId,
            datasetRoot: datasetRoot
        });

        await log(`Camera ${cameraId} loaded: ${JSON.stringify(cameraInfo)}`);

        // Update camera controller with new pose
        camera.setCameraPose(
            cameraInfo.position,
            cameraInfo.rotation
        );

        document.getElementById('status').textContent = `Camera ${cameraId} loaded`;
    } catch (err) {
        await log(`Error loading camera: ${err}`);
        document.getElementById('status').textContent = `Error: ${err}`;
    }
}

window.addEventListener('DOMContentLoaded', () => {
    init();

    // Add Enter key handler for camera ID input
    const cameraIdInput = document.getElementById('cameraIdInput');
    cameraIdInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            goToCamera();
        }
    });
});
