# Floater Detection Workflow

## Overview

Floaters are isolated Gaussian artifacts in 3D Gaussian Splatting (3DGS) models that are not attached to visible surfaces. They typically appear as floating blobs, semi-transparent artifacts, or "string-like" structures in empty space. Detecting and minimizing floaters is crucial for high-quality 3DGS reconstructions.

This document describes the complete floater detection workflow for SplatRs, including:
- RGB orbit video rendering for visual inspection
- Depth map orbit rendering for quantitative analysis
- Manual inspection checklist and severity assessment criteria
- Common floater patterns and mitigation strategies

## Quick Start

Run the integration test to verify both scripts are available:

```bash
cargo test --release tc_e2e_010_floater_integration -- --nocapture --ignored
```

This test will print the complete workflow with copy-paste-ready commands.

## Complete Workflow

### Step 1: Train a Model

First, train a 3DGS model on your dataset:

```bash
cargo build --release
./target/release/sugar-train \
  --preset full \
  --dataset-root datasets/bicycle \
  --out-dir runs/floater_test
```

### Step 2: Render RGB Orbit Video

Render a 360° orbital video around the scene for visual inspection:

```bash
python scripts/render_orbit_floater_detection.py \
  --model runs/floater_test/model_final.gs \
  --dataset-root datasets/bicycle \
  --output runs/floater_test/rgb_orbit \
  --frames 360 \
  --elevation 0
```

Options:
- `--frames`: Number of frames (360 = 1 frame per degree, default)
- `--elevation`: Camera elevation angle in degrees
  - `0` = ground level (default)
  - `30` = bird's eye view
  - `90` = top-down view
- `--radius`: Orbit radius (auto-computed from scene if not specified)

### Step 3: Render Depth Map Orbit

Render depth maps along the same orbital path for discontinuity analysis:

```bash
python scripts/render_orbit_depth.py \
  --model runs/floater_test/model_final.gs \
  --dataset-root datasets/bicycle \
  --output runs/floater_test/depth_orbit \
  --frames 360 \
  --elevation 0
```

This generates:
- `rgb_XXXX.png`: RGB frames
- `depth_XXXX.png`: Depth map frames (grayscale visualization)
- `camera_path.json`: Camera path metadata for reference

### Step 4: Create Videos

Create videos from the rendered frame sequences:

```bash
# RGB orbit video
ffmpeg -framerate 30 -i runs/floater_test/rgb_orbit/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 \
  runs/floater_test/floater_rgb.mp4

# Depth orbit video
ffmpeg -framerate 30 -i runs/floater_test/depth_orbit/depth_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 \
  runs/floater_test/floater_depth.mp4

# Side-by-side RGB and depth comparison
ffmpeg -framerate 30 \
  -i runs/floater_test/depth_orbit/rgb_%04d.png \
  -i runs/floater_test/depth_orbit/depth_%04d.png \
  -filter_complex "[0:v][1:v]hstack" \
  runs/floater_test/floater_comparison.mp4
```

The side-by-side comparison video allows you to correlate RGB floaters with depth discontinuities.

## Manual Inspection

### RGB Video Inspection Checklist

Watch the RGB orbit video and check for:

- [ ] **Floating blobs**: Isolated Gaussians visible against sky or uniform backgrounds
- [ ] **Semi-transparent artifacts**: Ghostly structures in empty space
- [ ] **String-like Gaussians**: Linear artifacts connecting unrelated surfaces
- [ ] **Proper parallax**: All visible structures should move naturally as camera orbits
- [ ] **Clean backgrounds**: Sky and empty regions should be clear (no fog/clouds)

### Depth Video Inspection Checklist

Watch the depth orbit video and check for:

- [ ] **Isolated discontinuities**: Sudden depth jumps not connected to surfaces
- [ ] **Floating depth values**: Depth regions disconnected from main geometry
- [ ] **Smooth uniform regions**: Sky, walls, and floors should show smooth depth
- [ ] **Geometric consistency**: Depth structure should match visible geometry
- [ ] **No depth noise**: Smooth areas should have consistent depth values

### Side-by-Side Comparison

Watch the comparison video to verify:

- [ ] **Correlation**: RGB floaters should correspond to depth discontinuities
- [ ] **Geometric validation**: Depth confirms structure of visible objects
- [ ] **Empty space consistency**: Background shows uniform depth values

## Severity Assessment

### Pass Criteria

A model passes floater detection if:
- No significant floaters visible in standard views (0° to 30° elevation)
- Only minor artifacts in extreme views (>60° elevation or edge cases)
- Depth maps show continuous surfaces without isolated spikes

### Severity Levels

If floaters are present, assess severity:

| Severity | Description | Count | Impact |
|----------|-------------|-------|--------|
| **Low** | Minor floaters, barely visible | <3 occurrences | Minimal quality impact |
| **Medium** | Noticeable floaters, affect quality | 3-10 occurrences | Moderate quality degradation |
| **High** | Significant floaters, severely impact quality | >10 occurrences | Unacceptable quality |

### Documentation Template

When documenting floaters, record:

1. **Count**: Approximate number of visible floaters
2. **Viewing angles**: Which elevations/rotations show floaters most clearly
3. **Characteristics**:
   - Size (small/medium/large)
   - Opacity (translucent/semi-opaque/opaque)
   - Location (sky, near surfaces, connecting geometry)
   - Type (blob, string, fog)
4. **Severity rating**: Low/Medium/High for each occurrence
5. **Screenshots/timestamps**: Visual evidence from videos

Example documentation:
```
Floater Report - bicycle scene
- Count: 2 floaters
- Location: Sky region, visible at 45° elevation
- Type: Semi-transparent blobs (medium size)
- Severity: Low (barely visible, only at specific angles)
- Screenshot: floater_rgb.mp4 @ 0:23
```

## Common Floater Patterns

### 1. Sky Floaters

**Appearance**: Semi-transparent blobs floating in sky or background regions

**Detection**:
- RGB: Ghostly structures against uniform sky
- Depth: Isolated depth values in regions that should be far/infinite

**Typical causes**:
- Gaussians optimized to fill view-dependent gaps
- Training views with inconsistent sky appearance

### 2. Surface Floaters

**Appearance**: Disconnected Gaussians near but not on surfaces

**Detection**:
- RGB: Blobs that don't follow surface geometry as camera moves
- Depth: Depth discontinuities near surfaces but separated by gap

**Typical causes**:
- Incomplete surface coverage forcing Gaussians to hover
- Regularization insufficient to prevent detachment

### 3. String Floaters

**Appearance**: Linear artifacts connecting unrelated geometry

**Detection**:
- RGB: Thin visible structures spanning empty space
- Depth: Linear depth gradients connecting surfaces

**Typical causes**:
- View-dependent optimization creating connecting structures
- Sparse training views allowing shortcuts

### 4. Fog Floaters

**Appearance**: Distributed low-opacity artifacts throughout scene (scene appears "foggy")

**Detection**:
- RGB: Reduced contrast and clarity, hazy appearance
- Depth: Noisy depth values throughout

**Typical causes**:
- Many small Gaussians with low opacity
- Opacity regularization too weak

## Advanced Inspection Techniques

### Multiple Elevation Angles

Render orbit paths at different elevations to inspect from various heights:

```bash
# Ground level (0°)
python scripts/render_orbit_floater_detection.py \
  --model runs/floater_test/model_final.gs \
  --dataset-root datasets/bicycle \
  --output runs/floater_test/rgb_orbit_0deg \
  --elevation 0

# Bird's eye view (30°)
python scripts/render_orbit_floater_detection.py \
  --model runs/floater_test/model_final.gs \
  --dataset-root datasets/bicycle \
  --output runs/floater_test/rgb_orbit_30deg \
  --elevation 30

# Top-down view (90°)
python scripts/render_orbit_floater_detection.py \
  --model runs/floater_test/model_final.gs \
  --dataset-root datasets/bicycle \
  --output runs/floater_test/rgb_orbit_90deg \
  --elevation 90
```

### Depth Map Analysis

For quantitative floater analysis:

1. **Identify discontinuities**: Look for sharp edges in depth maps not corresponding to visible geometry
2. **Measure depth gradients**: Floaters show as isolated bright/dark spots
3. **Compare with RGB**: Verify that depth discontinuities correlate with visible artifacts

Depth discontinuities are particularly useful for detecting:
- Sky floaters (depth values inconsistent with far background)
- Surface floaters (depth gaps between floater and surface)
- Fog floaters (depth noise in smooth regions)

### Batch Testing (All Scenes)

For comprehensive testing across multiple scenes:

```bash
for scene in bicycle garden stump room counter kitchen bonsai; do
  echo "Processing scene: $scene"

  # Render RGB orbit
  python scripts/render_orbit_floater_detection.py \
    --model runs/e2e_001_${scene}/model_final.gs \
    --dataset-root datasets/$scene \
    --output runs/e2e_001_${scene}/rgb_orbit \
    --frames 360

  # Render depth orbit
  python scripts/render_orbit_depth.py \
    --model runs/e2e_001_${scene}/model_final.gs \
    --dataset-root datasets/$scene \
    --output runs/e2e_001_${scene}/depth_orbit \
    --frames 360

  # Create videos
  ffmpeg -framerate 30 -i runs/e2e_001_${scene}/rgb_orbit/frame_%04d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    runs/e2e_001_${scene}/floater_rgb.mp4

  ffmpeg -framerate 30 -i runs/e2e_001_${scene}/depth_orbit/depth_%04d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    runs/e2e_001_${scene}/floater_depth.mp4
done
```

## Floater Mitigation Strategies

If floaters are detected, consider these mitigation approaches:

### 1. Training Hyperparameters

- **Increase opacity regularization**: Prevent low-opacity fog floaters
- **Adjust densification threshold**: Control Gaussian proliferation
- **Modify pruning strategy**: More aggressive pruning of low-contribution Gaussians
- **Tune learning rates**: Slower opacity learning may prevent floaters

### 2. Data Quality

- **Increase training view coverage**: More views reduce view-dependent artifacts
- **Improve COLMAP reconstruction**: Better initial point cloud reduces floater initialization
- **Add background masking**: Prevent Gaussians in sky/background regions

### 3. Post-Processing

- **Opacity thresholding**: Remove Gaussians with very low opacity
- **Geometric pruning**: Remove Gaussians far from COLMAP points
- **Manual cleanup**: Remove specific floater Gaussians (for production)

## Implementation Notes

### Script Architecture

Both orbit rendering scripts use identical camera path generation:
- Compute scene center from COLMAP point cloud (median of all points)
- Auto-compute orbit radius as 1.5× 90th percentile distance from center
- Generate spherical camera positions with configurable elevation
- Handle special cases (top-down view requires special up vector)

### Camera Conventions

Camera coordinate system follows OpenCV/COLMAP convention:
- +X: right
- +Y: down (note: negative of world up vector)
- +Z: forward (view direction)

Rotation matrix columns: `[right, -up, view_dir]`

### Dependencies

Required packages:
```bash
pip install pillow numpy
```

Required binaries:
- `sugar-render` (from SplatRs build)
- `ffmpeg` (for video creation)

## Test Cases

### TC-E2E-010: Floater Detection Test Suite

The floater detection test suite includes:

1. **TC-E2E-010 Dataset Verification**: Verify test datasets exist
   ```bash
   cargo test tc_e2e_010_floater_dataset_verification
   ```

2. **TC-E2E-010 RGB Orbit Script**: Generate/verify RGB orbit script
   ```bash
   cargo test tc_e2e_010_floater_generate_orbit_script -- --ignored --nocapture
   ```

3. **TC-E2E-010 Depth Orbit Script**: Verify depth orbit script
   ```bash
   cargo test tc_e2e_010_floater_generate_depth_script -- --ignored --nocapture
   ```

4. **TC-E2E-010-4 Integration Test**: Verify complete workflow
   ```bash
   cargo test tc_e2e_010_floater_integration -- --ignored --nocapture
   ```

All tests are located in `tests/verification_visual_floater.rs`.

## References

- Test implementation: `tests/verification_visual_floater.rs`
- RGB orbit script: `scripts/render_orbit_floater_detection.py`
- Depth orbit script: `scripts/render_orbit_depth.py`
- COLMAP format reference: https://colmap.github.io/format.html

## Troubleshooting

### Script Not Found

If scripts are not found, ensure they exist and are executable:
```bash
ls -l scripts/render_orbit*.py
chmod +x scripts/render_orbit*.py
```

### Rendering Failures

If rendering fails:
1. Verify `sugar-render` is built: `cargo build --release`
2. Check model file exists: `ls runs/*/model_final.gs`
3. Verify dataset has COLMAP structure: `ls datasets/*/sparse/0/`
4. Check console output for error messages

### Video Creation Fails

If ffmpeg fails:
1. Verify frames were rendered: `ls runs/*/orbit/frame_*.png | wc -l`
2. Check ffmpeg is installed: `ffmpeg -version`
3. Try different codec: Replace `libx264` with `libx265`

### No Floaters Visible

If no floaters are detected:
- Try different elevation angles (30°, 60°, 90°)
- Increase orbit radius to get farther from scene
- Check depth maps for subtle discontinuities
- Review training quality (PSNR, SSIM metrics)

This may indicate high-quality training output (which is good).
