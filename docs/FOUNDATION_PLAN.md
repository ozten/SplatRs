# Foundation Plan: Production-Ready 3DGS Before SuGaR

**Goal:** Build a fast, polished, interactive 3D Gaussian Splatting system before adding SuGaR features.

**Rationale:** A solid foundation will make SuGaR development much smoother and enable rapid iteration.

---

## Four Pillars

### 1. Fast Optimized Training
**Goal:** Match or exceed paper performance (~30min for 30K iterations)

### 2. Fast Optimized Rendering
**Goal:** Real-time rendering (>30 FPS at 1080p)

### 3. Checkpoints & Model I/O
**Goal:** Save/load trained models, resume interrupted training

### 4. Tauri GUI Viewer
**Goal:** Interactive viewer for testing renders, camera movement, parameter tweaking

---

## Detailed Plans

## 1️⃣ Fast Optimized Training

### Current State
- **7.3x speedup** over CPU baseline (4.4s → 602ms/iter)
- Forward on GPU (182x faster), backward on CPU (3.7x faster)
- **Still ~20x slower than paper** (6-7 hours vs 30 minutes for full training)

### Bottlenecks
1. **CPU sorting** (3ms): GPU→CPU→GPU transfers for depth sorting
2. **Naive rasterization** (4ms): No tiling, processes all Gaussians for all pixels
3. **CPU backward** (590ms): Gradient computation on CPU despite GPU forward

### Implementation Plan

#### Option A: GPU Gradients First (Recommended)
**Effort:** 1-2 weeks
**Expected speedup:** 100-200x total (4.4s → 22-44ms/iter)

**Why this first:**
- Biggest bottleneck is backward pass (590ms vs 12ms forward)
- Unlocks 50-100x backward speedup
- Total training time: 6-7 hours → **10-20 minutes** (better than paper!)

**Tasks:**
1. **Design gradient buffer architecture** (1 day)
   - Per-Gaussian gradient storage
   - Atomic accumulation strategy (WGSL limitations)
   - Memory layout for efficient GPU→CPU transfer

2. **Implement rasterization backward shader** (3-4 days)
   - Per-pixel gradient computation (alpha blending backward)
   - Atomic gradient accumulation or per-workgroup buffers
   - Transmittance chain rule (most complex part)
   - Validation against CPU gradients

3. **Implement projection backward shader** (2-3 days)
   - 3D→2D covariance gradients
   - Mean projection gradients
   - Chain through camera transform

4. **Integration & validation** (2-3 days)
   - Replace `render_full_color_grads()` with GPU version
   - Gradient checks for all parameters
   - Compare training convergence CPU vs GPU
   - Numerical stability fixes

**Deliverable:**
- `GpuRenderer::render_with_gradients()` function
- Full GPU training pipeline
- 10-20 minute training for 30K iterations

#### Option B: Tile-Based Rasterization
**Effort:** 2-3 weeks
**Expected speedup:** 30-50x total (4.4s → 88-147ms/iter)

**Why later:**
- More complex than GPU gradients (requires rethinking entire pipeline)
- Smaller immediate impact (forward is already fast)
- Better after GPU gradients prove GPU backend works

**Architecture:**
```
1. Project Gaussians to 2D (GPU) ✅ Already working
2. Assign to tiles (16×16 pixel tiles) (GPU compute)
3. Sort within tiles by depth (GPU radix sort per tile)
4. Rasterize tiles in parallel (GPU compute)
```

**Benefits:**
- Eliminates CPU sort bottleneck
- Better GPU occupancy (each thread block works on a tile)
- Reduces memory (only load relevant Gaussians per tile)
- Enables real-time rendering (>30 FPS)

**Tasks:**
1. Design tile structure (2 days)
2. Implement tile assignment (3 days)
3. GPU radix sort per tile (4-5 days)
4. Tile-based rasterization shader (5-6 days)
5. Integration & validation (3-4 days)

#### Recommended Sequence
1. **GPU gradients first** (1-2 weeks) → 10-20 min training
2. **Tile-based rasterization** (2-3 weeks) → 5-10 min training + real-time rendering

**Total time to paper-level performance:** 3-5 weeks

---

## 2️⃣ Fast Optimized Rendering

### Current State
- GPU forward: ~12ms for 245×136 (182x faster than CPU)
- Not benchmarked at 1080p
- No interactive viewer yet

### Goal
- **Real-time:** >30 FPS (33ms/frame) at 1080p
- Interactive camera movement
- Parameter adjustment (opacity threshold, SH degree, etc.)

### Implementation Plan

**Phase 1: Benchmark Current System**
- Test GPU renderer at 1080p (1920×1080)
- Measure FPS with varying Gaussian counts (10K, 50K, 100K)
- Identify bottlenecks (projection vs rasterization vs sorting)

**Phase 2: Tile-Based Rasterization** (Same as training optimization)
- Essential for real-time at 1080p
- Reduces wasted computation (only process visible Gaussians)

**Phase 3: Camera Caching**
- Pre-compute camera matrices for interactive movement
- Incremental updates for smooth rotation/translation
- LOD system (reduce Gaussian count for distant objects)

**Expected Performance:**
- 10K Gaussians: >100 FPS (smooth)
- 50K Gaussians: >60 FPS (smooth)
- 100K Gaussians: >30 FPS (playable)

---

## 3️⃣ Checkpoints & Model I/O

### Requirements

1. **Save trained models to disk**
   - All Gaussian parameters (position, scale, rotation, opacity, SH)
   - Metadata (scene bounds, camera info, training stats)
   - Compact binary format (not PLY - too verbose)

2. **Load models for rendering**
   - Fast loading (< 1 second for 100K Gaussians)
   - Version compatibility
   - Validation (detect corrupted files)

3. **Resume interrupted training**
   - Save optimizer state (Adam momentum, variance)
   - Save training iteration count
   - Save densification state (Gaussian ages, gradient accumulators)
   - Periodic auto-checkpointing

### File Format Design

#### `.gs` Format (Gaussian Splatting Model)

```
Header (256 bytes):
  - Magic: "GSPLAT\0\0" (8 bytes)
  - Version: u32 (4 bytes)
  - Num Gaussians: u64 (8 bytes)
  - SH degree: u32 (4 bytes)
  - Scene bounds: 6 × f32 (24 bytes)
  - Training iterations: u64 (8 bytes)
  - PSNR: f32 (4 bytes)
  - Reserved: 192 bytes

Gaussian Data (per Gaussian):
  - Position: 3 × f32 (12 bytes)
  - Scale (log): 3 × f32 (12 bytes)
  - Rotation (quaternion): 4 × f32 (16 bytes)
  - Opacity (logit): f32 (4 bytes)
  - SH coefficients: (SH_degree+1)² × 3 × f32
    - Degree 0: 3 × f32 (12 bytes)
    - Degree 1: 9 × f32 (36 bytes)
    - Degree 2: 15 × f32 (60 bytes)
    - Degree 3: 21 × f32 (84 bytes)

Compression: LZ4 or Zstd (optional, for disk storage)
```

**File sizes:**
- 10K Gaussians (SH degree 3): ~2 MB uncompressed, ~500 KB compressed
- 100K Gaussians (SH degree 3): ~20 MB uncompressed, ~5 MB compressed

#### `.gschkpt` Format (Training Checkpoint)

```
Header (256 bytes):
  - Magic: "GSCHKPT\0" (8 bytes)
  - Version: u32
  - Everything from .gs header
  - Optimizer: string (32 bytes, e.g., "Adam")

Gaussian Data: (same as .gs)

Optimizer State (per Gaussian):
  - Adam momentum (m): all parameters × f32
  - Adam variance (v): all parameters × f32
  - Gradient accumulators: all parameters × f32
  - Gaussian age: u32 (for densification)

Training Config:
  - Learning rates (per parameter type)
  - Loss function type
  - Densification thresholds
  - Random seed
```

### Implementation Tasks

**Week 1: Basic Model I/O**
1. Define `.gs` format specification (1 day)
2. Implement `GaussianCloud::save(&Path)` (2 days)
3. Implement `GaussianCloud::load(&Path)` (2 days)
4. Add compression (LZ4) (1 day)
5. CLI: `sugar-render model.gs --camera camera.json --out render.png` (1 day)

**Week 2: Checkpoint System**
1. Define `.gschkpt` format (1 day)
2. Extend trainer to save checkpoints (2 days)
3. Implement resume logic (2 days)
4. Auto-checkpoint every N iterations (1 day)
5. CLI: `sugar-train --resume checkpoint.gschkpt` (1 day)

**Deliverable:**
- Save/load trained models
- Resume training from any checkpoint
- Standalone renderer: `sugar-render model.gs`

---

## 4️⃣ Tauri GUI Viewer

### Goal
Interactive viewer for:
- Loading and rendering trained `.gs` models
- Free camera movement (WASD + mouse look)
- Real-time parameter adjustment
- Training visualization (loss curves, PSNR over time)
- Side-by-side comparison (render vs ground truth)

### Architecture

```
┌─────────────────────────────────────┐
│   Tauri Frontend (TypeScript/Svelte) │
│  - 3D viewport (Three.js or custom)  │
│  - UI controls (sliders, buttons)    │
│  - Training graphs (Chart.js)        │
└─────────────────┬───────────────────┘
                  │ IPC
┌─────────────────▼───────────────────┐
│   Tauri Backend (Rust)               │
│  - Load .gs models                   │
│  - GPU renderer integration          │
│  - Stream frames to frontend         │
│  - Handle training commands          │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   sugar-rs Library                   │
│  - GpuRenderer                       │
│  - GaussianCloud                     │
│  - Trainer (for live training)       │
└─────────────────────────────────────┘
```

### Features (MVP)

**Viewer Tab:**
- Load `.gs` model
- Render at interactive framerate (>30 FPS)
- Camera controls:
  - WASD movement
  - Mouse drag to rotate
  - Scroll to zoom
  - Snap to dataset cameras
- Parameter controls:
  - Opacity threshold slider
  - SH degree selector (0-3)
  - Background color picker
- Export current view as PNG

**Training Tab:**
- Load COLMAP dataset
- Configure training parameters (GUI form)
- Start/pause/resume training
- Live loss curve
- Live PSNR graph
- Preview renders every N iterations
- Save checkpoint button
- Export model when done

**Comparison Tab:**
- Load model + dataset
- Select test view from dataset
- Side-by-side: Render vs Ground Truth
- Overlay mode (slider to blend)
- PSNR/SSIM displayed
- Cycle through test views

### Technology Stack

**Frontend:**
- Framework: **Svelte** (lightweight, reactive)
- 3D: **Three.js** or custom WebGL (for Gaussian rendering shader)
- Charts: **Chart.js** (for loss/PSNR graphs)
- UI: **Tailwind CSS** (for styling)

**Backend:**
- Tauri Rust backend
- IPC commands for model loading, rendering, training control
- Stream RGB frames as Base64 or shared memory

**Rendering Strategy:**

Option A: **Render on Rust backend, stream to frontend**
- Pros: Reuse GPU renderer, guaranteed correctness
- Cons: Bandwidth intensive (streaming 1080p frames)

Option B: **WebGL shader on frontend**
- Pros: Low latency, no streaming
- Cons: Have to reimplement rasterization in GLSL

**Recommendation:** Start with Option A (streaming), migrate to Option B later for lower latency.

### Implementation Plan

**Week 1: Basic Viewer**
1. Tauri project setup (1 day)
2. Load .gs model via IPC (1 day)
3. Render frame on backend, stream to frontend (2 days)
4. Basic camera controls (WASD + mouse) (2 days)

**Week 2: Parameter Controls**
1. Opacity threshold slider (1 day)
2. SH degree selector (1 day)
3. Background color picker (1 day)
4. Export PNG button (1 day)
5. Snap to dataset camera (1 day)

**Week 3: Training Tab**
1. Dataset loading UI (1 day)
2. Training parameter form (2 days)
3. Start/pause training via IPC (2 days)
4. Live loss/PSNR graphs (streaming data) (2 days)

**Week 4: Comparison Tab & Polish**
1. Side-by-side comparison view (2 days)
2. Overlay slider (1 day)
3. PSNR/SSIM calculation (1 day)
4. UI polish and error handling (2 days)

**Deliverable:**
- Standalone Tauri app: `sugar-gui`
- Load models, render interactively, train with live feedback
- Export renders and trained models

---

## Overall Timeline & Dependencies

### Dependency Graph

```
Week 1-2: GPU Gradients
            ↓
Week 3-5: Tile-Based Rasterization
            ↓
Week 2-3: Model I/O & Checkpoints (parallel with gradients)
            ↓
Week 4-7: Tauri GUI (depends on model I/O, benefits from tile-based rendering)
            ↓
Week 8: M10 Validation (depends on fast training)
```

### Parallel Tracks

**Track A: Performance (Critical Path)**
1. GPU Gradients (1-2 weeks)
2. Tile-Based Rasterization (2-3 weeks)
3. **Total: 3-5 weeks** → Paper-level performance

**Track B: Usability (Can parallelize)**
1. Model I/O (1 week) - can start immediately
2. Checkpoints (1 week) - depends on Model I/O
3. **Total: 2 weeks**

**Track C: Tooling (Depends on A & B)**
1. Tauri GUI (3-4 weeks) - depends on Model I/O, benefits from fast rendering
2. **Total: 3-4 weeks**

**Track D: Validation**
1. M10 Benchmarks (1 week) - can run anytime, but faster training helps

### Recommended Order

**Phase 1 (Weeks 1-2): Fast Training Foundation**
- GPU Gradients ← Critical path
- Model I/O (parallel) ← Enables testing

**Phase 2 (Weeks 3-5): Real-Time Rendering**
- Tile-Based Rasterization ← Enables interactive GUI
- Checkpoints (parallel) ← Enables long runs

**Phase 3 (Weeks 6-9): Interactive Tooling**
- Tauri GUI ← Makes everything usable
- M10 Validation (parallel) ← Confirms quality

**Total Time: 9 weeks to production-ready 3DGS system**

---

## Success Criteria

### Fast Training ✅
- [ ] 30K iterations in <30 minutes (match paper)
- [ ] GPU utilization >80% during training
- [ ] Memory usage <8GB for 100K Gaussians

### Fast Rendering ✅
- [ ] >30 FPS at 1080p with 50K Gaussians
- [ ] <100ms to load and prepare model for rendering
- [ ] Smooth camera movement (no stuttering)

### Model I/O ✅
- [ ] Save/load models <1 second
- [ ] Resume training from checkpoint
- [ ] Compressed models <10% of uncompressed size

### GUI ✅
- [ ] Interactive viewer runs at >30 FPS
- [ ] Training visualization updates in real-time
- [ ] Intuitive controls (learnable in <5 minutes)

---

## Next Immediate Steps

Based on this plan, here's what to do **right now**:

### Option 1: Start with GPU Gradients (Fastest path to fast training)
1. Design gradient buffer architecture
2. Implement rasterization backward shader
3. Validate against CPU gradients
4. Integrate with training loop

### Option 2: Start with Model I/O (Fastest path to usable system)
1. Define `.gs` format
2. Implement save/load
3. Create standalone `sugar-render` CLI
4. Test on existing trained models

### Option 3: Prototype Tauri GUI (Fastest path to user feedback)
1. Set up Tauri project
2. Create basic viewer (load model, render, camera controls)
3. Validate UX before investing in training UI

**My recommendation:** Start with **Model I/O (Option 2)** because:
- Quick wins (1 week to working save/load)
- Unblocks GUI development (need models to load)
- Enables testing (save models, render them later)
- Parallelizes with GPU gradients (different code areas)

What would you like to tackle first?
