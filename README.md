# sugar-rs

A Rust implementation of **SuGaR** (Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering) built on top of 3D Gaussian Splatting.

## Overview

sugar-rs is a high-performance, from-scratch implementation of Gaussian Splatting and SuGaR in Rust. It provides:

- **Fast training** of 3D Gaussian Splatting models from COLMAP scenes
- **High-quality rendering** with GPU acceleration (wgpu)
- **Mesh extraction** capabilities (SuGaR algorithm)
- **Cross-platform support** (Linux, macOS, Windows)
- **Comprehensive verification tests** covering input parsing, rasterization, optimization, and visual quality

This implementation is designed for:
- Research and experimentation with Gaussian Splatting
- Production use cases requiring fast, high-quality novel view synthesis
- Learning and understanding the Gaussian Splatting algorithm in depth

## Features

- ✅ COLMAP dataset loading (cameras, images, point clouds)
- ✅ 3D to 2D Gaussian projection with covariance computation
- ✅ Tile-based rasterization with alpha blending
- ✅ Spherical harmonics (SH) for view-dependent color
- ✅ Differentiable rendering with full backward pass
- ✅ Adam optimizer with per-parameter learning rates
- ✅ Adaptive density control (split, clone, prune)
- ✅ L1 + D-SSIM loss function
- ✅ GPU-accelerated rendering and training (wgpu)
- ✅ Multi-view training with validation
- ✅ Model compression with LZ4
- 🚧 SuGaR mesh extraction (in progress)
- 🚧 Desktop viewer application (in progress)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ozten/SplatRs
cd SplatRs

# 2. Train a model (requires COLMAP dataset in datasets/ directory)
cargo run --bin sugar-train --features gpu --release -- \
  --preset m8 \
  --dataset-root datasets/mipnerf360/garden

# 3. Render a novel view
cargo run --bin sugar-render --features gpu --release -- \
  --model runs/YYYYMMDD_HHMM_m8/final.gs \
  --camera-id 10 \
  --dataset-root datasets/mipnerf360/garden \
  --out render.png
```

**Note:** The `--features gpu` flag is required for GPU acceleration. Without it, the binaries will only support CPU rendering (much slower).

## Building from Source

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- C compiler (for native dependencies)

### Build Commands

**IMPORTANT:** GPU support is **NOT** enabled by default. You must explicitly enable it with `--features gpu` or `--all-features`.

```bash
# Build all binaries with GPU support (debug mode)
cargo build --features gpu

# Build optimized release binaries with GPU support (RECOMMENDED)
cargo build --release --features gpu

# Build specific binary with GPU
cargo build --bin sugar-train --release --features gpu

# Build all features (GPU + viewer + LZ4)
cargo build --release --all-features
```

The compiled binaries will be in:
- Debug: `target/debug/` (when using `cargo build`)
- Release: `target/release/` (when using `cargo build --release`)

**Note:** Always match your build configuration with the binary path. If you build with `cargo build --features gpu`, the binary is in `target/debug/`, not `target/release/`.

### Running Binaries (Recommended Method)

Instead of building and running separately, use `cargo run` to avoid path confusion:

```bash
# Run with cargo (automatically builds if needed)
cargo run --bin sugar-train --features gpu --release -- --preset m8 --dataset-root datasets/garden

# The arguments after '--' are passed to the binary
cargo run --bin sugar-render --features gpu --release -- --model trained.gs --camera-id 0 --dataset-root datasets/garden
```

### Feature Flags

- **`gpu`** (optional, recommended): Enables GPU-accelerated rendering with wgpu. Without this, only CPU rendering is available and training will be significantly slower.
- **`viewer`** (optional): Enables the desktop viewer application (requires `gpu` feature)
- **`lz4`** (default): Enables LZ4 compression for model files

## Binaries

sugar-rs provides four main binaries:

### 1. sugar-train

Train a 3D Gaussian Splatting model from a COLMAP dataset.

**Usage:**
```bash
# Using cargo run (recommended)
cargo run --bin sugar-train --features gpu --release -- --preset <preset> --dataset-root <path> [options]

# Or if you've already built the binary
target/release/sugar-train --preset <preset> --dataset-root <path> [options]
```

**Examples:**

```bash
# Train on Mip-NeRF 360 garden scene with M8 preset (multi-view)
cargo run --bin sugar-train --features gpu --release -- \
  --preset m8 --dataset-root datasets/mipnerf360/garden

# Train with custom parameters
cargo run --bin sugar-train --features gpu --release -- \
  --preset m8 \
  --dataset-root datasets/tandt_db/tandt/train \
  --iters 7000 \
  --downsample 2.0 \
  --max-gaussians 100000

# Single-view overfitting (M7 preset)
cargo run --bin sugar-train --features gpu --release -- \
  --preset m7 \
  --scene datasets/garden/sparse/0 \
  --images datasets/garden/images \
  --image-index 0

# Force CPU rendering (if built without GPU feature, or to test CPU path)
cargo run --bin sugar-train --release -- \
  --preset m8 --dataset-root datasets/garden --cpu
```

**Key Options:**
- `--preset`: Training preset (m7, m8-smoke, m8, m9, m10)
  - `m7`: Single-view overfitting
  - `m8-smoke`: Quick multi-view test (100 iterations)
  - `m8`: Standard multi-view training
  - `m9`: With adaptive density control
  - `m10`: Full production training
- `--dataset-root`: Path to dataset (auto-detects sparse/0 and images/)
- `--scene`: Path to COLMAP sparse reconstruction (sparse/0)
- `--images`: Path to images directory
- `--iters`: Number of training iterations
- `--downsample`: Downsampling factor for images (e.g., 2.0 = half resolution)
- `--max-gaussians`: Maximum number of Gaussians
- `--cpu`: Force CPU rendering (only needed if built with `--features gpu` but want to test CPU path)
- `--seed`: Random seed for reproducibility
- `--out-dir`: Output directory for checkpoints

**Note:** If the binary was built with `--features gpu`, GPU rendering is used by default. Use `--cpu` to force CPU rendering. If built without GPU features, only CPU rendering is available.

**Output:**
- Creates timestamped directory in `runs/YYYYMMDD_HHMM_<preset>/`
- Saves final model as `.gs` file (compressed binary format)
- Saves checkpoints during training (if configured)
- Saves run metadata and training logs

### 2. sugar-render

Render novel views from a trained Gaussian Splatting model.

**Usage:**
```bash
# Using cargo run (recommended)
cargo run --bin sugar-render --features gpu --release -- --model <model.gs> --camera-id <id> --dataset-root <path> --out <output.png>

# Or if you've already built the binary
target/release/sugar-render --model <model.gs> --camera-id <id> --dataset-root <path> --out <output.png>
```

**Examples:**

```bash
# Render using camera 5 from COLMAP dataset
cargo run --bin sugar-render --features gpu --release -- \
  --model runs/20260114_1500_m8/final.gs \
  --camera-id 5 \
  --dataset-root datasets/garden \
  --out render_cam5.png

# Render with custom camera JSON
cargo run --bin sugar-render --features gpu --release -- \
  --model trained.gs \
  --camera-json camera.json \
  --out custom_view.png

# Render with white background
cargo run --bin sugar-render --features gpu --release -- \
  --model trained.gs \
  --camera-id 0 \
  --dataset-root datasets/garden \
  --background 1.0,1.0,1.0 \
  --out white_bg.png

# Render at higher resolution
cargo run --bin sugar-render --features gpu --release -- \
  --model trained.gs \
  --camera-id 0 \
  --dataset-root datasets/garden \
  --width 1920 \
  --height 1080 \
  --out hires.png
```

**Key Options:**
- `--model`: Path to trained model file (.gs)
- `--camera-id`: Camera ID from COLMAP dataset (requires --dataset-root)
- `--camera-json`: Custom camera parameters as JSON file
- `--dataset-root`: Path to COLMAP dataset (for --camera-id)
- `--out`: Output image path (default: render.png)
- `--background`: Background color as R,G,B floats (default: 0,0,0)
- `--width`, `--height`: Override render resolution

**Camera JSON Format:**
```json
{
    "width": 640,
    "height": 480,
    "fx": 525.0,
    "fy": 525.0,
    "cx": 320.0,
    "cy": 240.0,
    "position": [0.0, 0.0, 5.0],
    "rotation": [[1,0,0], [0,1,0], [0,0,1]]
}
```

### 3. sugar-extract

Extract a triangle mesh from a trained SuGaR model.

**Status:** Not yet implemented (see Milestone M14)

**Planned Usage:**
```bash
sugar-extract --model model.ply --output mesh.obj
```

### 4. viewer

Desktop application for interactive viewing of Gaussian Splatting models.

**Status:** In development (requires `viewer` feature)

**Build:**
```bash
cargo build --bin viewer --features viewer
```

**Note:** The viewer feature currently has build issues and is under active development.

## Testing

sugar-rs has a comprehensive test suite with 66 test files covering:
- Unit tests for core math and algorithms
- Integration tests for the full rendering pipeline
- Verification tests validating correctness against reference implementations
- Benchmark tests on standard datasets (Mip-NeRF 360, NeRF Synthetic)
- Visual quality tests (edge sharpness, floater detection, temporal stability)

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output visible
cargo test -- --nocapture

# Run only library unit tests
cargo test --lib

# Run specific test file
cargo test --test verification_phase1_input

# Run specific test by name
cargo test test_colmap_intrinsics

# Run tests in parallel (default)
cargo test

# Run tests sequentially (for debugging)
cargo test -- --test-threads=1
```

### Test Categories

```bash
# Unit tests (core functionality)
cargo test unit_

# Verification tests (Phase 1: Input parsing)
cargo test --test verification_phase1_input

# Verification tests (Phase 2: Rasterization)
cargo test --test verification_phase2_rasterization

# Verification tests (Phase 3: Optimization)
cargo test --test verification_phase3_optimization

# Visual quality tests
cargo test --test verification_visual_edge_sharpness
cargo test --test verification_visual_floater
cargo test --test verification_visual_temporal_stability
cargo test --test verification_visual_view_dependent

# Benchmark tests
cargo test --test verification_benchmark_mipnerf360
cargo test --test verification_benchmark_nerf_synthetic

# GPU tests (requires gpu feature)
cargo test gpu_ --features gpu

# Synthetic scene tests
cargo test --test synthetic_scene_tests
```

### Test Dataset Requirements

Many verification and benchmark tests require datasets to be present in the `datasets/` directory:

**Required datasets:**
- Mip-NeRF 360 scenes (bicycle, garden, stump, etc.) - already present
- Tanks & Temples (in `datasets/tandt_db/`) - already present

**Optional datasets for full verification:**
- NeRF Synthetic (Blender) - for benchmark tests
  - Download from: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
  - Place in: `datasets/nerf_synthetic/`

See `docs/VERIFICATION_SETUP.md` for complete dataset information.

## Development Tools

### Code Coverage

Generate HTML coverage reports using `cargo-llvm-cov`:

```bash
# Install cargo-llvm-cov (first time only)
cargo install cargo-llvm-cov

# Generate HTML coverage report
make coverage

# Generate and open report in browser
make coverage-open

# Generate coverage with test output (for debugging)
make coverage-test

# Clean coverage artifacts
make coverage-clean
```

Coverage reports are saved to `target/llvm-cov/html/index.html`.

### Performance Profiling

Profile training runs to identify performance bottlenecks:

```bash
# Install flamegraph tools (first time only)
cargo install flamegraph

# Profile with custom arguments
make profile PROFILE_ARGS="--preset m8 --dataset-root datasets/garden"

# Quick micro training profile (100 iterations)
make profile-micro

# Clean profiling artifacts
make profile-clean
```

Generates `flamegraph.svg` for visualization. See `docs/PROFILING.md` for details.

## Project Structure

```
sugar-rs/
├── src/
│   ├── lib.rs              # Library root
│   ├── core/               # Core types (Gaussian, Camera, math)
│   ├── io/                 # I/O (COLMAP, PLY, model serialization)
│   ├── render/             # Forward rendering pipeline
│   ├── diff/               # Backward pass (gradients)
│   ├── optim/              # Optimization (Adam, loss, trainer)
│   ├── sugar/              # SuGaR-specific (mesh extraction)
│   ├── gpu/                # GPU backend (wgpu)
│   └── bin/                # Binary executables
│       ├── train.rs        # sugar-train
│       ├── render.rs       # sugar-render
│       ├── extract.rs      # sugar-extract
│       └── viewer.rs       # viewer
├── tests/                  # Integration and verification tests
├── datasets/               # Training datasets (not in git)
├── runs/                   # Training output (not in git)
├── docs/                   # Documentation
├── Cargo.toml              # Rust package manifest
└── Makefile                # Convenience targets
```

## Documentation

- `docs/sugar-rs-architecture.md` - Module architecture and design
- `docs/sugar-rs-milestones.md` - Development milestones
- `docs/VERIFICATION_SETUP.md` - Test setup and dataset requirements
- `docs/PROFILING.md` - Performance profiling guide
- `docs/COVERAGE.md` - Code coverage reports
- `gaussian_splatting_test_plan.md` - Comprehensive test plan

## Troubleshooting

### "GPU rendering requested but not compiled with --features gpu"

This error means you're trying to run a binary that wasn't built with GPU support. Solutions:

1. **Use cargo run with GPU features** (recommended):
   ```bash
   cargo run --bin sugar-train --features gpu --release -- --preset m8 --dataset-root datasets/garden
   ```

2. **Or rebuild with GPU support**:
   ```bash
   cargo build --release --features gpu
   # Now the GPU-enabled binary is at target/release/sugar-train
   ```

3. **Or force CPU rendering** (much slower):
   ```bash
   cargo run --bin sugar-train --release -- --preset m8 --dataset-root datasets/garden --cpu
   ```

### Wrong binary path after building

If you build with `cargo build --features gpu` (no `--release`), the binary is in `target/debug/`, not `target/release/`. Either:
- Build with `--release`: `cargo build --release --features gpu`
- Or use the debug binary: `target/debug/sugar-train`

### Tests requiring datasets

Some verification and benchmark tests require datasets in `datasets/`. See `docs/VERIFICATION_SETUP.md` for details. Tests will skip gracefully if datasets are missing.

## Performance Notes

- **Debug builds** use `opt-level = 1` for faster compilation while maintaining reasonable performance
- **Release builds** use full optimization (`opt-level = 3`, LTO) for production use - always use `--release` for training
- **GPU rendering** requires `--features gpu` and is significantly faster than CPU (10-100x speedup)
- **Image downsampling** (e.g., `--downsample 2.0`) greatly speeds up training at the cost of detail
- **Profiling builds** use `--profile profiling` (release with debug symbols)

## License

Licensed under either:
- MIT License
- Apache License 2.0

at your option.

## References

- **3D Gaussian Splatting for Real-Time Radiance Field Rendering**
  - Kerbl et al., SIGGRAPH 2023
  - Paper: `docs/gaus_splt_2308.04079v1.pdf`
  - https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

- **SuGaR: Surface-Aligned Gaussian Splatting**
  - Guédon & Lepetit, CVPR 2024
  - Paper: `docs/sugar_2311.12775v3.pdf`
  - https://github.com/Anttwo/SuGaR

## Contributing

Contributions are welcome! This project is actively developed and follows standard Rust conventions:

- Run `cargo fmt` before committing
- Run `cargo clippy` to check for common issues
- Add tests for new functionality
- Update documentation as needed

## Author

Austin King (shout@ozten.com)
