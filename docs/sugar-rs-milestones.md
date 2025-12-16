# sugar-rs Milestones

Concrete checkpoints where you can verify progress before moving on.

---

## M1: Parse COLMAP and Visualize Point Cloud

**You're done when:** You can load a COLMAP sparse reconstruction and export the 3D points as a PLY file that opens correctly in MeshLab/Blender.

**Verification:**
- Load `cameras.bin`, `images.bin`, `points3D.bin`
- Export points with colors to PLY
- Visual inspection: points should form recognizable scene structure
- Numeric check: camera count and point count match COLMAP's output

**Why this matters:** Validates your I/O pipeline and basic data structures before any rendering code.

---

## M2: Project Points to Images

**You're done when:** You can project the 3D points through each camera and overlay them on the corresponding input image, and they land in the right places.

**Verification:**
- For each image, project all visible 3D points to 2D
- Draw colored dots on the image at projected locations
- Points should align with actual scene features (corners, edges, etc.)
- Test with 3-4 cameras from different viewpoints

**Why this matters:** Validates camera intrinsics/extrinsics parsing and projection math. If this is wrong, nothing else will work.

---

## M3: Render Spheres (Constant-Size Gaussians)

**You're done when:** You can render an image where each 3D point is drawn as a fixed-size colored circle, with correct depth ordering.

**Verification:**
- Initialize "Gaussians" at point locations with identity covariance
- Render with depth-sorted alpha blending
- Closer objects should occlude farther ones
- Colors should match original point cloud colors

**Why this matters:** Validates your rasterization and blending pipeline without the complexity of varying covariance.

---

## M4: Render Full Gaussians (No SH)

**You're done when:** Rendered image shows elliptical splats with varying sizes and orientations, approximating the scene.

**Verification:**
- Initialize Gaussians with random (small) scales and rotations
- Covariance projection produces visible ellipses
- Splats vary in size based on distance from camera
- Compare output image dimensions/format to reference implementation

**Why this matters:** Validates the 3D→2D covariance projection, which is the trickiest math in the forward pass.

---

## M5: Spherical Harmonics Working

**You're done when:** View-dependent color changes are visible when rendering the same scene from different cameras.

**Verification:**
- Initialize SH coefficients with some variation (not just DC)
- Render same Gaussians from two different viewpoints
- Colors should shift slightly between views
- Compare SH evaluation output to Python reference on same inputs

**Why this matters:** Validates SH basis functions. If these are wrong, view-dependent effects won't train correctly.

---

## M6: Gradient Check Passes for All Operations

**You're done when:** Finite-difference gradient checking passes for every differentiable operation with relative error < 1e-4.

**Verification:**
- For each op: `(f(x+ε) - f(x-ε)) / 2ε ≈ analytical_gradient`
- Test at multiple random inputs, not just one
- Operations to check:
  - [ ] Sigmoid / inverse sigmoid
  - [ ] Quaternion → rotation matrix
  - [ ] Scale + rotation → covariance
  - [ ] 3D → 2D covariance projection
  - [ ] 2D Gaussian evaluation
  - [ ] Alpha blending (single Gaussian)
  - [ ] Alpha blending (multiple Gaussians)
  - [ ] SH evaluation
  - [ ] L1 loss
  - [ ] SSIM loss

**Why this matters:** This is the most critical milestone. Bugs here cause silent training failures that are extremely hard to debug.

---

## M7: Single-Image Overfitting

**You're done when:** Given a single training image, optimization converges to nearly perfect reconstruction (PSNR > 35dB).

**Verification:**
- Use just one camera/image pair
- Initialize ~1000 Gaussians from COLMAP points
- Run optimization for 1000-2000 iterations
- Loss should decrease steadily
- Final rendered image should be visually identical to target
- PSNR > 35dB, SSIM > 0.95

**Why this matters:** Validates the full forward-backward-optimize loop without the complexity of multi-view consistency. If you can't overfit one image, something is broken.

---

## M8: Multi-View Training (Small Scene)

**You're done when:** Training on 10-20 images produces reasonable novel-view synthesis.

**Verification:**
- Hold out 2-3 images for testing
- Train on remaining images for 5000-10000 iterations
- Render held-out viewpoints
- Should be recognizable, though possibly blurry
- PSNR > 20dB on test views

**Why this matters:** Validates that gradients from multiple views combine correctly and that the representation generalizes.

---

## M9: Adaptive Density Control Working

**You're done when:** Gaussian count changes during training (splits, clones, prunes) and quality improves.

**Verification:**
- Log Gaussian count over training
- Should see growth in early iterations (splits/clones)
- Pruning should remove low-opacity Gaussians
- Final quality should exceed M8 (same scene, same iterations)
- Visual: fine details should be sharper

**Why this matters:** Density control is essential for quality. Without it, you're limited by initial point cloud density.

---

## M10: Full 3DGS Training (Reference Quality)

**You're done when:** Training a standard benchmark scene (e.g., Mip-NeRF 360 garden) achieves quality comparable to reference implementation.

**Verification:**
- Train for 30000 iterations
- Compare PSNR/SSIM to published numbers (within 1-2 dB)
- Render video fly-through, should be smooth
- Training time within 2-3x of reference (for CPU) or comparable (for GPU)

**Why this matters:** Validates that your implementation is correct and complete for vanilla 3DGS before adding SuGaR.

---

## M11: GPU Renderer Matches CPU

**You're done when:** GPU renderer produces identical images to CPU renderer (within floating-point tolerance).

**Verification:**
- Render same scene with both backends
- Per-pixel difference < 1e-4
- Test multiple viewpoints
- Test edge cases (Gaussians at image boundary, behind camera)

**Why this matters:** Ensures GPU port is correct before optimizing for speed.

---

## M12: GPU Training End-to-End

**You're done when:** Full training runs on GPU with significant speedup.

**Verification:**
- Train same scene on CPU vs GPU
- Quality should be identical (PSNR within 0.5 dB)
- GPU should be 10-50x faster (depending on scene size)
- Memory usage reasonable (< 8GB for typical scenes)

**Why this matters:** Makes the tool practical for real use. With GPU acceleration, you can iterate quickly on SuGaR development.

---

## M13: SuGaR Regularization Training

**You're done when:** Adding regularization produces visibly flatter, surface-aligned Gaussians.

**Verification:**
- Train same scene with and without regularization
- Visualize Gaussian ellipsoids (export to Blender)
- Regularized: should look like flat discs aligned to surfaces
- Unregularized: should look like fuzzy blobs
- Rendering quality should remain reasonable (may be slightly lower)

**Why this matters:** Validates the regularization losses and their gradients. Flat Gaussians are prerequisite for good mesh extraction.

---

## M14: Mesh Extraction

**You're done when:** Marching cubes produces a watertight mesh that resembles the scene.

**Verification:**
- Extract mesh from regularized Gaussians
- Open in MeshLab/Blender
- Mesh should be recognizable as the scene
- No major holes or floating artifacts
- Vertex count reasonable (100K-1M depending on scene)

**Why this matters:** This is the goal of SuGaR — getting geometry out, not just images.

---

## Summary: Critical Path

```
M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10
 │                        │
 │    (foundation)        │    (working 3DGS - CPU)
 │                        │
 └────────────────────────┴──→ M11 → M12
                                │
                                │    (GPU acceleration)
                                │
                          ──→ M13 → M14
                                │
                                │    (SuGaR complete)
```

**Rationale for GPU-first:** 10-50x speedup enables practical iteration on SuGaR development (M13-M14).

**Minimum viable product:** M12 (GPU training works, can train models efficiently)
**Complete system:** M14 (mesh extraction working)

---

## Time Estimates (Rough)

| Milestone | Estimated Time | Cumulative |
|-----------|---------------|------------|
| M1-M2 | 2-3 days | 3 days |
| M3-M5 | 4-5 days | 1 week |
| M6 | 3-5 days | 2 weeks |
| M7-M8 | 3-4 days | 2.5 weeks |
| M9-M10 | 1 week | 3.5 weeks |
| M11-M12 (GPU) | 1-2 weeks | 5.5 weeks |
| M13-M14 (SuGaR) | 1 week | 6.5 weeks |

These assume focused work and familiarity with Rust. Double if learning Rust concurrently or working part-time.

**Note:** GPU work (M11-M12) moved before SuGaR (M13-M14) to enable fast iteration on subsequent development.
