# M10 Infrastructure Improvements: Coverage, Profiling, and Viewer

## Overview

Three infrastructure improvements needed as blockers for completing M10 (reference-quality 3DGS training):

1. **Code Coverage** - Measure test coverage and identify gaps
2. **Performance Profiling** - Generate flamegraphs to identify bottlenecks
3. **Tauri UI** - Interactive model viewer for validation

**Sequential execution order:** Coverage → Profiling → Tauri UI

**Total timeline:** 8-10 days

---

## Feature 1: Code Coverage with llvm-cov (Day 1)

### Goal
Establish baseline code coverage and identify untested critical paths.

### Implementation Steps

1. **Install tooling** (30 min)
   ```bash
   cargo install cargo-llvm-cov
   rustup component add llvm-tools-preview
   ```

2. **Create configuration files** (1 hour)
   - Create `llvm-cov.toml` with exclusions
   - Add Makefile targets for common workflows
   - Update `.gitignore` for coverage artifacts

3. **Generate baseline report** (1 hour)
   ```bash
   cargo llvm-cov --html --all-features --open
   ```

4. **Document workflow** (2 hours)
   - Create `docs/COVERAGE.md` with usage guide
   - Document current coverage percentages
   - Identify critical gaps (GPU code, error paths)

### Files to Create

**llvm-cov.toml**
```toml
[report]
exclude = [
    "src/bin/*",
    "examples/*",
    "tests/*",
]

[html]
output-dir = "target/coverage/html"
```

**Makefile** (or extend existing)
```makefile
.PHONY: coverage coverage-open coverage-clean

coverage:
	cargo llvm-cov --html --all-features

coverage-open:
	cargo llvm-cov --html --open --all-features

coverage-clean:
	cargo llvm-cov clean --workspace
```

**docs/COVERAGE.md**
- Installation instructions
- Running coverage locally
- Interpreting reports
- Coverage goals (80%+ for core/)
- CI integration notes

### Files to Modify

- `.gitignore` - Add `target/coverage/`, `*.profraw`, `*.profdata`
- `README.md` - Add coverage section (optional)

### Verification

```bash
# Test with unit tests only
cargo llvm-cov --lib --html

# Test with all features including GPU
cargo llvm-cov --html --all-features

# Verify GPU modules are covered
open target/coverage/html/index.html
```

### Integration with M10

- Identify untested gradient paths (critical for correctness)
- Validate training loop has sufficient test coverage
- Find missing error handling tests
- Prevent regressions during M10 optimization

### Time Estimate: 5-6 hours

---

## Feature 2: Performance Profiling with Flamegraph (Day 2-3)

### Goal
Identify performance bottlenecks in 100-iteration micro training using interactive flamegraphs.

### Implementation Steps

1. **Configure debug symbols in release builds** (30 min)
   - Add `[profile.release-with-debug]` to Cargo.toml
   - Enable debug symbols while keeping opt-level=3

2. **Install flamegraph tooling** (15 min)
   ```bash
   cargo install flamegraph
   ```

3. **Create profiling scripts** (3 hours)
   - `scripts/profile-training.sh` - Full 100-iteration training
   - `scripts/profile-forward.sh` - Forward pass only
   - `scripts/profile-backward.sh` - Backward pass only
   - Makefile targets for convenience

4. **Generate baseline flamegraphs** (2 hours)
   - Profile micro training (CPU and GPU)
   - Save to `docs/baselines/`
   - Analyze current hotspots

5. **Document profiling workflow** (2 hours)
   - Create `docs/PROFILING.md`
   - macOS dtrace setup (sudo requirements)
   - Interpreting flamegraphs
   - Common bottlenecks

### Files to Create

**Cargo.toml additions**
```toml
[profile.release-with-debug]
inherits = "release"
debug = true
split-debuginfo = "unpacked"  # macOS-friendly
```

**scripts/profile-training.sh**
```bash
#!/bin/bash
set -e

DATASET="${1:-datasets/tandt_db/tandt/train}"

echo "Profiling micro training (100 iters)..."
sudo cargo flamegraph \
  --profile release-with-debug \
  --features gpu \
  --bin sugar-train \
  --output flamegraph-training.svg \
  -- \
  --preset micro \
  --dataset-root "$DATASET" \
  --iters 100 \
  --out-dir /tmp/profile_output

echo "Flamegraph saved to: flamegraph-training.svg"
open flamegraph-training.svg
```

**Makefile additions**
```makefile
.PHONY: profile-micro profile-clean

profile-micro:
	bash scripts/profile-training.sh

profile-clean:
	rm -f flamegraph-*.svg perf.data*
```

**docs/PROFILING.md**
- Why flamegraphs for Gaussian Splatting
- macOS dtrace setup
- Running profiling scripts
- Interpreting results
- Common bottlenecks (sorting, projection, rasterization)
- Baseline performance metrics

### Files to Modify

- `Cargo.toml` - Add release-with-debug profile
- `.gitignore` - Add `flamegraph*.svg`, `perf.data*`

### Verification

```bash
# Test basic profiling
sudo cargo flamegraph --version

# Profile micro training
bash scripts/profile-training.sh

# Verify flamegraph shows readable function names
open flamegraph-training.svg
```

### Integration with M10

- Identify training bottlenecks for optimization
- Validate GPU speedup claims
- Guide parallelization efforts
- Profile densification logic
- Catch performance regressions

### Time Estimate: 8-10 hours

---

## Feature 3: Tauri UI for Model Exploration (Day 4-8)

### Goal
Interactive viewer with camera control and ground truth comparison for model validation.

### Implementation Steps

1. **Initialize Tauri project** (2 hours)
   ```bash
   cargo install tauri-cli
   cargo tauri init
   ```
   - Choose React + TypeScript
   - Configure `src-tauri/` directory

2. **Implement backend commands** (6-8 hours)
   - `load_model(path)` - Load .gs model into memory
   - `render_with_camera(camera_params)` - Render and return base64 PNG
   - `load_dataset(colmap_path)` - Load COLMAP scene
   - `get_camera(id)` - Get camera parameters
   - `load_ground_truth(id)` - Load GT image

3. **Build frontend UI** (10-15 hours)
   - App.tsx - Main application layout
   - CameraControls - 3D visualization with Three.js
   - RenderView - Display rendered image
   - ComparisonView - Side-by-side rendered vs GT
   - CameraInfo - Display camera parameters

4. **Add 3D camera visualization** (4-6 hours)
   - Three.js scene with camera frustums
   - OrbitControls for interaction
   - Highlight selected camera
   - Sync to render command

5. **Test and debug** (4-6 hours)
   - Test model loading
   - Test rendering command
   - Test dataset loading
   - Test comparison view
   - Test camera controls

6. **Document and package** (3-4 hours)
   - Create `docs/VIEWER.md`
   - Add screenshots
   - Create release build

### Files to Create

**Backend (src-tauri/):**

- `src-tauri/Cargo.toml`
  ```toml
  [dependencies]
  sugar-rs = { path = ".." }
  tauri = { version = "2.0", features = ["shell-open"] }
  serde = { version = "1.0", features = ["derive"] }
  base64 = "0.21"
  image = "0.25"
  nalgebra = "0.33"
  ```

- `src-tauri/src/main.rs` - Tauri app entry point
- `src-tauri/src/commands/model.rs` - Model loading commands
- `src-tauri/src/commands/render.rs` - Rendering command
- `src-tauri/src/commands/dataset.rs` - Dataset exploration commands
- `src-tauri/src/state.rs` - App state (loaded model/dataset)
- `src-tauri/tauri.conf.json` - Tauri configuration

**Frontend (src-tauri/ui/):**

- `src-tauri/ui/package.json` - Frontend dependencies
  ```json
  {
    "dependencies": {
      "react": "^18.2.0",
      "@tauri-apps/api": "^2.0.0",
      "three": "^0.160.0",
      "@react-three/fiber": "^8.15.0",
      "@react-three/drei": "^9.92.0"
    }
  }
  ```

- `src-tauri/ui/src/App.tsx` - Main application
- `src-tauri/ui/src/components/CameraControls.tsx` - 3D controls
- `src-tauri/ui/src/components/RenderView.tsx` - Render display
- `src-tauri/ui/src/components/ComparisonView.tsx` - Side-by-side comparison

**Documentation:**

- `docs/VIEWER.md` - User guide for viewer
- `docs/screenshots/` - UI screenshots

### Files to Modify

- `.gitignore` - Add `src-tauri/target/`, `src-tauri/ui/node_modules/`, `src-tauri/ui/dist/`
- `Cargo.toml` - Add workspace member: `members = [".", "src-tauri"]`
- `README.md` - Add viewer section

### Key Backend Command Implementation

**render_with_camera command:**
```rust
#[tauri::command]
pub async fn render_with_camera(
    fx: f32, fy: f32, cx: f32, cy: f32,
    width: u32, height: u32,
    rotation: [[f32; 3]; 3],
    translation: [f32; 3],
    background: [f32; 3],
    state: State<'_, AppState>,
) -> Result<String, String> {
    // 1. Get model from state
    // 2. Build Camera from params
    // 3. Call render_full_linear
    // 4. Convert to PNG
    // 5. Encode as base64
    // 6. Return to frontend
}
```

### Verification

```bash
# Install and test Tauri
cargo install tauri-cli
cargo tauri init

# Install frontend deps
cd src-tauri/ui && npm install

# Run in dev mode
cargo tauri dev

# Test commands:
# 1. Load .gs model
# 2. Render with camera
# 3. Load dataset
# 4. Compare with ground truth

# Build release
cargo tauri build
```

### Integration with M10

- Visual quality validation (side-by-side with GT)
- Camera pose validation (3D visualization)
- Interactive debugging (test from arbitrary viewpoints)
- Find rendering artifacts
- Dataset quality assessment

### Time Estimate: 29-42 hours (4-5 days)

---

## Critical Files

### For Coverage
- `Cargo.toml` - Add instrumentation configuration
- `src/render/full_diff.rs` - Core rendering to validate
- `tests/gradient_check.rs` - Gradient tests to measure

### For Profiling
- `Cargo.toml` - Add release-with-debug profile
- `src/optim/trainer.rs` - Training loop hotspots
- `src/render/full_diff.rs` - Rendering hotspots

### For Tauri UI
- `src/io/model.rs` - Model loading to expose
- `src/core/camera.rs` - Camera struct for UI
- `src/render/full_diff.rs` - Rendering to call from UI

---

## Sequential Execution Timeline

### Phase 1: Code Coverage (Day 1)
- Install llvm-cov
- Create configuration
- Generate baseline report
- Document workflow
- **Deliverable:** Coverage infrastructure + baseline

### Phase 2: Performance Profiling (Day 2-3)
- Configure release profile
- Create profiling scripts
- Generate flamegraphs
- Analyze bottlenecks
- **Deliverable:** Profiling infrastructure + hotspot analysis

### Phase 3: Tauri UI (Day 4-8)
- Initialize Tauri (Day 4)
- Backend commands (Day 5)
- Frontend UI (Day 6-7)
- Test and debug (Day 7)
- Document and package (Day 8)
- **Deliverable:** Interactive model viewer

---

## Success Criteria

### Coverage
- ✅ HTML coverage reports generated successfully
- ✅ Baseline coverage documented (expect 60-70% initially)
- ✅ Critical gaps identified (GPU code, error handling)
- ✅ Makefile targets work

### Profiling
- ✅ Flamegraphs generated for micro training
- ✅ Hotspots identified (sorting, projection, rasterization)
- ✅ CPU vs GPU comparison documented
- ✅ Scripts work on macOS with dtrace

### Tauri UI
- ✅ Can load .gs models
- ✅ Interactive camera control works
- ✅ Side-by-side comparison renders correctly
- ✅ Ground truth images load from COLMAP
- ✅ Builds and runs on macOS

---

## Risk Mitigation

**Coverage:**
- If GPU features break coverage → Test early with `--features gpu`
- Fallback: tarpaulin with Docker

**Profiling:**
- If dtrace requires sudo → Document security implications
- Fallback: Use Instruments.app GUI manually

**Tauri:**
- If bundle size too large → Use CPU-only rendering
- If Three.js too complex → Simplified 2D camera list
- Fallback: Basic viewer without 3D visualization

---

## How This Helps M10

These three infrastructure improvements provide the tooling needed to:

1. **Validate correctness** - Coverage reveals untested gradient paths
2. **Optimize performance** - Flamegraphs identify training bottlenecks
3. **Assess quality** - Interactive viewer enables visual validation

With these tools in place, M10 can be properly validated against reference implementations with confidence in both correctness and performance.
