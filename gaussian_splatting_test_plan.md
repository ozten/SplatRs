# Gaussian Splatting Codebase Verification Test Plan

**Document Version:** 1.0  
**Created:** January 2026  
**Purpose:** Systematic verification of a 3D Gaussian Splatting implementation to identify defects and quality issues

---

## 1. Overview

### 1.1 Scope

This test plan covers end-to-end verification of a 3D Gaussian Splatting (3DGS) pipeline, from input ingestion through rendering and optimization. The goal is to identify which components function correctly and which require remediation before production use.

### 1.2 Testing Approach

Testing proceeds in dependency order: foundational components must pass before dependent components can be meaningfully tested. A failure at a lower level may cause cascading failures at higher levels.

**Test Levels (in order):**
1. Input Pipeline
2. Gaussian Primitive Representation
3. Differentiable Rasterizer
4. Optimization Loop
5. Adaptive Density Control
6. End-to-End Quality Benchmarks

### 1.3 Severity Definitions

| Severity | Definition | Example |
|----------|------------|---------|
| **Critical** | Optimization cannot converge; produces NaN/Inf; fundamentally broken | Incorrect gradients, division by zero |
| **High** | Converges but quality ceiling is significantly limited | Wrong alpha blending order, broken densification |
| **Medium** | Quality degradation in specific scenarios or edge cases | Tile boundary artifacts, numerical precision loss |
| **Low** | Performance issues or code quality concerns not affecting output correctness | Inefficient memory usage, missing error handling |

---

## 2. Reference Materials and Dependencies

### 2.1 Reference Implementations

| Resource | URL | Purpose |
|----------|-----|---------|
| Original 3DGS Paper | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ | Authoritative reference for algorithm behavior |
| Official 3DGS Repository | https://github.com/graphdeco-inria/gaussian-splatting | Reference implementation for comparison |
| diff-gaussian-rasterization | https://github.com/graphdeco-inria/diff-gaussian-rasterization | Reference CUDA rasterizer with gradients |
| gsplat (Nerfstudio) | https://github.com/nerfstudio-project/gsplat | Alternative rasterizer implementation for cross-validation |

### 2.2 Evaluation Libraries

| Library | Installation | Documentation | Purpose |
|---------|--------------|---------------|---------|
| scikit-image (SSIM) | `pip install scikit-image` | https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity | Reference SSIM implementation |
| LPIPS | `pip install lpips` | https://github.com/richzhang/PerceptualSimilarity | Perceptual quality metric |
| PyTorch Image Quality | `pip install piq` | https://github.com/photosynthesis-team/piq | Additional metrics (MS-SSIM, FID, etc.) |
| NumPy | `pip install numpy` | https://numpy.org/doc/stable/ | Numerical reference computations |
| OpenCV | `pip install opencv-python` | https://docs.opencv.org/ | Image I/O and transformations |

### 2.3 Standard Test Datasets

| Dataset | URL | Size | Purpose |
|---------|-----|------|---------|
| NeRF Synthetic (Blender) | https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 | ~400 MB | Synthetic scenes with perfect ground truth cameras |
| Mip-NeRF 360 | https://jonbarron.info/mipnerf360/ | ~4 GB | Real-world unbounded scenes; industry standard benchmark |
| Tanks and Temples | https://www.tanksandtemples.org/download/ | Variable | Large-scale reconstruction benchmark |
| DTU Dataset | https://roboimagedata.compute.dtu.dk/?page_id=36 | ~30 GB | Multi-view stereo benchmark with ground truth |
| COLMAP Sample Data | https://colmap.github.io/datasets.html | Variable | Pre-processed SfM outputs for testing |

### 2.4 Synthetic Test Assets

| Asset | Source | Purpose |
|-------|--------|---------|
| Single colored cube | Create procedurally | Minimal scene for basic verification |
| Overlapping spheres | Create procedurally | Alpha blending and depth sorting verification |
| Checkerboard plane | Create procedurally | High-frequency detail reconstruction |
| Cornell Box | https://www.graphics.cornell.edu/online/box/data.html | Standard graphics test scene |

---

## 3. Test Cases: Input Pipeline

### 3.1 Camera Parameter Ingestion

#### TC-INP-001: COLMAP Camera Intrinsics Parsing

**Objective:** Verify that camera intrinsic parameters are correctly parsed from COLMAP output.

**Prerequisites:**
- COLMAP sparse reconstruction output (cameras.bin or cameras.txt)
- Known ground truth intrinsics for the capture device

**Test Data:**
- Use NeRF Synthetic dataset (known intrinsics: focal length, principal point)
- Alternatively, create synthetic COLMAP output with known values

**Procedure:**
1. Load cameras.bin/txt using the implementation under test
2. Extract focal length (fx, fy), principal point (cx, cy), and distortion parameters
3. Compare against ground truth values

**Pass Criteria:**
- Focal length error < 0.01 pixels
- Principal point error < 0.01 pixels
- Distortion coefficients match to 6 decimal places

**Severity if Failed:** Critical

---

#### TC-INP-002: COLMAP Camera Extrinsics Parsing

**Objective:** Verify that camera pose (rotation + translation) is correctly parsed and represented.

**Prerequisites:**
- COLMAP sparse reconstruction with images.bin or images.txt
- Understanding of coordinate system conventions (COLMAP uses world-to-camera)

**Test Data:**
- NeRF Synthetic dataset with known camera poses
- Create a synthetic 8-camera orbit at radius 4.0 around origin as reference

**Procedure:**
1. Load camera poses from COLMAP output
2. For each camera, extract rotation (as quaternion or matrix) and translation
3. Verify rotation matrices are valid (orthogonal, determinant = +1)
4. Verify camera positions by computing -R^T * t
5. Compare against ground truth poses

**Pass Criteria:**
- Rotation error < 0.1 degrees (compute angle between rotation matrices)
- Translation error < 0.001 units
- All rotation matrices satisfy R^T * R = I within 1e-6

**Severity if Failed:** Critical

---

#### TC-INP-003: Coordinate System Convention

**Objective:** Verify the implementation uses a consistent coordinate system throughout.

**Prerequisites:**
- Documentation of expected coordinate system (OpenGL: +Y up, -Z forward; OpenCV: +Y down, +Z forward)

**Procedure:**
1. Create a synthetic scene with a single point at world position (1, 2, 3)
2. Create a camera at origin looking down -Z axis (OpenGL convention)
3. Project the point using the implementation's projection code
4. Verify the projected coordinates match manual calculation

**Pass Criteria:**
- Projected coordinates match analytical solution within 1e-5
- Document the coordinate system convention used

**Severity if Failed:** Critical (silent failures cause systematic reconstruction errors)

---

#### TC-INP-004: Image Loading and Color Space

**Objective:** Verify images are loaded with correct dimensions, color ordering, and value range.

**Test Data:**
- Create test images with known pixel values:
  - Solid red (255, 0, 0)
  - Solid green (0, 255, 0)
  - Gradient from black to white
  - 8-bit PNG, 16-bit PNG, JPEG, EXR formats

**Procedure:**
1. Load each test image using the implementation
2. Check dimensions match expected
3. Check pixel values at known locations
4. Verify color channel ordering (RGB vs BGR)
5. Verify value range (0-255 vs 0-1 vs 0-65535)

**Pass Criteria:**
- Dimensions exact match
- Pixel values match within format precision (0 for lossless, ±2 for JPEG)
- Color channels in expected order
- Values normalized to expected range

**Severity if Failed:** High

---

#### TC-INP-005: Alpha Channel Handling

**Objective:** Verify transparency/mask information is correctly processed.

**Test Data:**
- RGBA PNG with varying alpha values
- Separate mask image (common in NeRF datasets)

**Procedure:**
1. Load RGBA image, verify alpha channel preserved
2. Load RGB image + separate mask, verify correct association
3. Verify alpha values used correctly in loss computation (masked regions excluded)

**Pass Criteria:**
- Alpha values preserved exactly for PNG
- Masked pixels excluded from loss (verify by checking gradient is zero for masked regions)

**Severity if Failed:** Medium

---

### 3.2 Point Cloud Initialization

#### TC-INP-010: COLMAP Point Cloud Loading

**Objective:** Verify 3D points are correctly loaded from COLMAP sparse reconstruction.

**Test Data:**
- COLMAP points3D.bin or points3D.txt
- NeRF Synthetic dataset (known geometry)

**Procedure:**
1. Load point cloud from COLMAP output
2. Verify point count matches expected
3. Verify point positions are in correct coordinate system
4. Verify RGB colors are loaded (if present)
5. Visualize point cloud overlay on input images to verify alignment

**Pass Criteria:**
- Point count matches source file
- Point positions within 1e-5 of source values
- Visual alignment with input images (manual inspection)

**Severity if Failed:** Critical

---

#### TC-INP-011: Initial Gaussian Parameter Bounds

**Objective:** Verify initial scale, opacity, and SH coefficients are within valid ranges.

**Procedure:**
1. Initialize Gaussians from point cloud
2. For each Gaussian, check:
   - Scale values are positive and non-zero (e.g., 1e-6 < scale < 10.0)
   - Opacity is in valid range (0, 1) or logit space equivalent
   - Quaternion is normalized (||q|| = 1)
   - SH DC component (degree 0) is initialized reasonably (not zero, not extreme)

**Pass Criteria:**
- No scale values ≤ 0 or > reasonable bound
- No opacity values outside valid range
- Quaternion norm = 1.0 ± 1e-6
- No NaN or Inf values in any parameter

**Severity if Failed:** High

---

## 4. Test Cases: Gaussian Primitive Representation

### 4.1 Covariance Matrix Construction

#### TC-COV-001: Scale and Rotation to Covariance Conversion

**Objective:** Verify the 3D covariance matrix is correctly constructed from scale and rotation parameters.

**Background:**
Covariance Σ = R * S * S^T * R^T, where R is rotation matrix from quaternion, S is diagonal scale matrix.

**Procedure:**
1. Create test cases with known scale and rotation:
   - Identity rotation, uniform scale (1, 1, 1) → Σ should be identity
   - Identity rotation, non-uniform scale (2, 1, 0.5) → Σ should be diag(4, 1, 0.25)
   - 90° rotation around Z, uniform scale → Σ should still be identity
   - 45° rotation around Z, scale (2, 1, 1) → compute expected Σ analytically
2. Compute covariance using implementation
3. Compare against analytical result

**Pass Criteria:**
- Covariance matrix elements match within 1e-6
- Covariance matrix is symmetric (Σ = Σ^T)
- Covariance matrix is positive semi-definite (all eigenvalues ≥ 0)

**Severity if Failed:** Critical

---

#### TC-COV-002: Quaternion Normalization Handling

**Objective:** Verify the implementation handles unnormalized quaternions correctly.

**Procedure:**
1. Create Gaussian with unnormalized quaternion (e.g., [2, 0, 0, 0] instead of [1, 0, 0, 0])
2. Compute covariance matrix
3. Verify result matches the normalized quaternion case

**Pass Criteria:**
- Output identical to normalized quaternion input, OR
- Implementation explicitly normalizes before use, OR
- Implementation raises error/warning for unnormalized input

**Severity if Failed:** High (silent incorrect results)

---

#### TC-COV-003: Numerical Stability with Extreme Scales

**Objective:** Verify no numerical issues with very small or very large scales.

**Test Cases:**
- Very small scale: (1e-7, 1e-7, 1e-7)
- Very large scale: (1e3, 1e3, 1e3)
- Mixed extreme: (1e-6, 1.0, 1e6)
- Near-zero single axis: (1e-10, 1.0, 1.0)

**Procedure:**
1. Create Gaussian with each extreme scale
2. Compute covariance matrix
3. Compute 2D projected covariance
4. Compute inverse covariance (used in evaluation)
5. Check for NaN, Inf, or matrix inversion failures

**Pass Criteria:**
- No NaN or Inf values
- Covariance matrices remain symmetric positive semi-definite
- Inverse covariance computable (no singular matrices) or gracefully handled

**Severity if Failed:** High

---

### 4.2 Spherical Harmonics Evaluation

#### TC-SH-001: Degree 0 (DC) Constant Color

**Objective:** Verify SH degree 0 produces view-independent color.

**Procedure:**
1. Create Gaussian with only DC coefficient set (e.g., corresponding to RGB = [0.5, 0.3, 0.8])
2. Evaluate SH from 100 random view directions uniformly distributed on sphere
3. Verify all evaluations produce identical color

**Pass Criteria:**
- Maximum color deviation across all views < 1e-6
- RGB values match expected DC color

**Severity if Failed:** Critical

---

#### TC-SH-002: Higher Degree SH Basis Functions

**Objective:** Verify SH basis functions are correctly implemented.

**Reference:** 
- SH basis function definitions: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
- Reference implementation: https://github.com/google/spherical-harmonics

**Procedure:**
1. For each SH degree (1, 2, 3) and order:
   - Set single SH coefficient to 1.0, all others to 0.0
   - Evaluate at known directions (e.g., +X, +Y, +Z, normalized [1,1,1])
   - Compare against analytical SH basis function values

**Test Directions:**
- (1, 0, 0), (0, 1, 0), (0, 0, 1)
- (-1, 0, 0), (0, -1, 0), (0, 0, -1)
- normalized (1, 1, 0), (1, 0, 1), (0, 1, 1)
- normalized (1, 1, 1)

**Pass Criteria:**
- Evaluated values match reference SH basis within 1e-5

**Severity if Failed:** High

---

#### TC-SH-003: SH Coefficient Count

**Objective:** Verify correct number of SH coefficients for each degree.

**Expected Counts:**
- Degree 0: 1 coefficient (DC only)
- Degree 1: 4 coefficients (1 + 3)
- Degree 2: 9 coefficients (1 + 3 + 5)
- Degree 3: 16 coefficients (1 + 3 + 5 + 7)

**Procedure:**
1. Initialize Gaussians with each SH degree setting
2. Verify parameter tensor shapes match expected coefficient counts per color channel

**Pass Criteria:**
- Coefficient counts exactly match formula (degree + 1)²

**Severity if Failed:** Critical

---

## 5. Test Cases: Differentiable Rasterizer

### 5.1 Depth Sorting and Alpha Blending

#### TC-RAS-001: Front-to-Back Depth Ordering

**Objective:** Verify Gaussians are composited in correct depth order.

**Test Setup:**
- Create two Gaussians at same (x, y) but different z depths
- Front Gaussian: z=1, red, opacity=0.5
- Back Gaussian: z=2, blue, opacity=1.0
- Camera looking down -Z axis

**Expected Result:**
Using alpha-over compositing: C = C_front * α_front + C_back * α_back * (1 - α_front)
= (1,0,0) * 0.5 + (0,0,1) * 1.0 * 0.5 = (0.5, 0, 0.5)

**Procedure:**
1. Render scene with implementation
2. Sample pixel at Gaussian center
3. Compare against analytical result

**Pass Criteria:**
- RGB values within 1/255 of expected

**Severity if Failed:** Critical

---

#### TC-RAS-002: Alpha Blending Accumulation

**Objective:** Verify correct alpha accumulation across multiple overlapping Gaussians.

**Test Setup:**
Create 5 overlapping Gaussians with known colors and opacities. Compute expected output analytically using standard alpha-over formula.

**Reference Formula:**
```
C_out = Σ (c_i * α_i * Π_{j<i}(1 - α_j))
```

**Procedure:**
1. Create stack of 5 semi-transparent Gaussians
2. Render and sample center pixel
3. Compare against manually computed alpha-over result

**Pass Criteria:**
- RGB error < 1/255 per channel
- Alpha error < 0.001

**Severity if Failed:** Critical

---

#### TC-RAS-003: Tile Boundary Handling

**Objective:** Verify Gaussians spanning tile boundaries render correctly.

**Background:**
3DGS uses tile-based rasterization (typically 16x16 pixel tiles). Gaussians near tile boundaries must be handled correctly.

**Test Setup:**
- Create large Gaussian centered exactly on tile boundary (e.g., at pixel 16.0 with 16x16 tiles)
- Create reference by rendering with 1x1 tiles (if possible) or analytical computation

**Procedure:**
1. Render scene at resolution where Gaussian spans 4 tiles
2. Verify no visible seams or discontinuities at tile boundaries
3. Compare pixel values on either side of tile boundary

**Pass Criteria:**
- No visible seams (visual inspection)
- Pixel values continuous across tile boundaries (< 1/255 difference for adjacent pixels)

**Severity if Failed:** Medium

---

#### TC-RAS-004: Early Termination (Transmittance Threshold)

**Objective:** Verify rendering correctly terminates when accumulated opacity reaches threshold.

**Test Setup:**
- Create stack of 100 Gaussians, front ones opaque enough to fully occlude back ones
- Back Gaussians have distinctive color (e.g., bright magenta)

**Procedure:**
1. Render scene
2. Verify back Gaussian colors do not appear in output
3. If possible, instrument code to verify back Gaussians not evaluated

**Pass Criteria:**
- Occluded Gaussian colors not visible
- (Performance check) Rendering time does not scale linearly with occluded Gaussian count

**Severity if Failed:** Low (correctness) / Medium (performance)

---

### 5.2 2D Projection and Covariance

#### TC-RAS-010: 3D to 2D Covariance Projection

**Objective:** Verify 3D Gaussian covariance is correctly projected to 2D screen space.

**Reference:**
EWA splatting: Σ' = J * W * Σ * W^T * J^T
Where J is Jacobian of projection, W is view transform

**Test Setup:**
- Spherical Gaussian (uniform scale) at scene center
- Camera at various distances and angles

**Procedure:**
1. Place spherical Gaussian with scale (1, 1, 1) at origin
2. Render with camera at distance 5 looking at origin
3. Measure rendered splat size in pixels
4. Compare against expected size from projection formula

**Pass Criteria:**
- Rendered splat size within 5% of analytical expectation

**Severity if Failed:** High

---

#### TC-RAS-011: Anisotropic Gaussian Projection

**Objective:** Verify elongated Gaussians project correctly at various orientations.

**Test Setup:**
- Create Gaussian with non-uniform scale (3, 1, 1)
- Rotate to various orientations (0°, 45°, 90° around each axis)
- Render from fixed camera

**Procedure:**
1. For each orientation, render and measure projected ellipse axes
2. Compare against analytically computed 2D projection

**Pass Criteria:**
- Major/minor axis lengths within 5% of expected
- Orientation angle within 2° of expected

**Severity if Failed:** High

---

### 5.3 Gradient Correctness

#### TC-GRAD-001: Position Gradient Finite Difference Check

**Objective:** Verify analytical gradients for Gaussian positions match numerical gradients.

**Procedure:**
1. Create minimal scene: 5-10 Gaussians, single training image
2. Compute loss (L1 or L2)
3. For each Gaussian position (x, y, z):
   - Compute analytical gradient from backward pass
   - Compute numerical gradient: (L(x+ε) - L(x-ε)) / 2ε, with ε = 1e-4
   - Compare analytical vs numerical

**Pass Criteria:**
- Relative error < 1e-3 for most parameters
- Relative error < 1e-2 for all parameters
- Formula: rel_error = |analytical - numerical| / max(|analytical|, |numerical|, 1e-8)

**Severity if Failed:** Critical

---

#### TC-GRAD-002: Scale Gradient Finite Difference Check

**Objective:** Verify analytical gradients for Gaussian scales match numerical gradients.

**Procedure:**
Same as TC-GRAD-001, but perturb scale parameters (sx, sy, sz).

**Note:** Scale may be parameterized as log-scale; verify gradients in the actual parameterization used.

**Pass Criteria:**
- Same as TC-GRAD-001

**Severity if Failed:** Critical

---

#### TC-GRAD-003: Rotation Gradient Finite Difference Check

**Objective:** Verify analytical gradients for Gaussian rotations match numerical gradients.

**Procedure:**
Same as TC-GRAD-001, but perturb quaternion components (qw, qx, qy, qz).

**Note:** Extra care needed due to quaternion normalization. May need to perturb in tangent space or verify gradients project correctly to constraint manifold.

**Pass Criteria:**
- Same as TC-GRAD-001

**Severity if Failed:** Critical

---

#### TC-GRAD-004: Spherical Harmonics Gradient Check

**Objective:** Verify analytical gradients for SH coefficients match numerical gradients.

**Procedure:**
Same as TC-GRAD-001, but perturb SH coefficients for each degree and color channel.

**Pass Criteria:**
- Same as TC-GRAD-001

**Severity if Failed:** Critical

---

#### TC-GRAD-005: Opacity Gradient Finite Difference Check

**Objective:** Verify analytical gradients for opacity match numerical gradients.

**Procedure:**
Same as TC-GRAD-001, but perturb opacity (or logit-opacity) parameter.

**Pass Criteria:**
- Same as TC-GRAD-001

**Severity if Failed:** Critical

---

## 6. Test Cases: Optimization Loop

### 6.1 Loss Function Verification

#### TC-OPT-001: L1 Loss Correctness

**Objective:** Verify L1 loss computation matches reference implementation.

**Reference:** `torch.nn.L1Loss` or `numpy.abs(a - b).mean()`

**Procedure:**
1. Generate two random images A and B (100x100 RGB)
2. Compute L1 loss using implementation
3. Compute L1 loss using reference
4. Compare

**Pass Criteria:**
- Difference < 1e-6

**Severity if Failed:** High

---

#### TC-OPT-002: L2 (MSE) Loss Correctness

**Objective:** Verify L2/MSE loss computation matches reference implementation.

**Reference:** `torch.nn.MSELoss` or `numpy.square(a - b).mean()`

**Procedure:**
Same as TC-OPT-001 but for L2/MSE loss.

**Pass Criteria:**
- Difference < 1e-6

**Severity if Failed:** High

---

#### TC-OPT-003: SSIM Loss Correctness

**Objective:** Verify SSIM computation matches reference implementation.

**Reference:** `skimage.metrics.structural_similarity`
- Documentation: https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

**Procedure:**
1. Generate pairs of test images:
   - Identical images (SSIM should = 1.0)
   - One image with Gaussian noise added
   - Shifted version of image
   - Contrast-adjusted version
2. Compute SSIM using implementation
3. Compute SSIM using scikit-image reference
4. Compare

**Parameters to Match:**
- Window size (typically 11)
- Sigma for Gaussian weighting (typically 1.5)
- K1 and K2 constants (typically 0.01 and 0.03)
- Data range

**Pass Criteria:**
- SSIM values within 0.001 of reference
- Identical images produce SSIM > 0.9999

**Severity if Failed:** High

---

#### TC-OPT-004: D-SSIM Loss Correctness

**Objective:** Verify D-SSIM (1 - SSIM) / 2 formulation if used.

**Procedure:**
1. Verify D-SSIM = (1 - SSIM) / 2
2. Verify gradients flow correctly through D-SSIM

**Pass Criteria:**
- D-SSIM formula correct
- Gradients non-zero and reasonable magnitude

**Severity if Failed:** Medium

---

### 6.2 Optimization Convergence

#### TC-OPT-010: Single Gaussian Fitting

**Objective:** Verify optimizer can fit a single Gaussian to a synthetic target.

**Test Setup:**
1. Create ground truth: single Gaussian with known parameters
2. Render ground truth images from 4 views
3. Initialize with single Gaussian at perturbed position/scale
4. Optimize

**Pass Criteria:**
- Loss decreases monotonically (after initial iterations)
- Final position error < 0.01
- Final scale error < 10%
- Final color error < 0.01 (L1)

**Severity if Failed:** Critical

---

#### TC-OPT-011: Multi-Gaussian Scene Fitting

**Objective:** Verify optimizer can fit multiple Gaussians to a simple synthetic scene.

**Test Setup:**
1. Create ground truth: 10 non-overlapping Gaussians with varied colors
2. Render ground truth images from 8 views
3. Initialize with Gaussians at correct positions but wrong colors/scales
4. Optimize for 1000 iterations

**Pass Criteria:**
- PSNR > 35 dB
- SSIM > 0.95
- Loss converged (< 1% change over last 100 iterations)

**Severity if Failed:** Critical

---

#### TC-OPT-012: NeRF Synthetic Benchmark

**Objective:** Verify implementation achieves expected quality on standard benchmark.

**Test Data:** NeRF Synthetic dataset - "lego" scene
- Download: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1

**Procedure:**
1. Train on "lego" scene for 30,000 iterations (standard protocol)
2. Evaluate on held-out test views
3. Compute PSNR, SSIM, LPIPS

**Reference Metrics (from original 3DGS paper):**
- PSNR: ~35.0 dB
- SSIM: ~0.98
- LPIPS: ~0.02

**Pass Criteria:**
- PSNR within 2 dB of reference
- SSIM within 0.02 of reference
- LPIPS within 0.02 of reference

**Severity if Failed:** High (if significantly worse), Medium (if slightly worse)

---

## 7. Test Cases: Adaptive Density Control

### 7.1 Densification

#### TC-ADC-001: Clone Trigger Condition

**Objective:** Verify Gaussians are cloned when position gradient magnitude exceeds threshold in under-reconstructed regions.

**Test Setup:**
- Create scene with sparse initialization missing a region
- Train until densification iteration
- Monitor which Gaussians get cloned

**Procedure:**
1. Initialize with Gaussians that don't cover part of the scene
2. Run training with densification enabled
3. Verify Gaussians with high position gradients in low-coverage areas are cloned
4. Verify clone is placed at parent position initially

**Pass Criteria:**
- Clone count increases in under-reconstructed regions
- Cloned Gaussians have same position as parent (initially)

**Severity if Failed:** High

---

#### TC-ADC-002: Split Trigger Condition

**Objective:** Verify large Gaussians are split when position gradient is high.

**Test Setup:**
- Initialize with few large Gaussians covering detailed region
- Train until densification iteration

**Procedure:**
1. Create scene with high-frequency detail
2. Initialize with oversized Gaussians (scale > threshold)
3. Run training with densification
4. Verify large Gaussians with high gradients are split into smaller ones

**Pass Criteria:**
- Large Gaussians in detailed regions get split
- Split Gaussians have smaller scale than parent
- Total reconstruction quality improves after split

**Severity if Failed:** High

---

#### TC-ADC-003: Densification Interval

**Objective:** Verify densification occurs at specified iteration intervals.

**Procedure:**
1. Configure densification every N iterations (e.g., N=100)
2. Log Gaussian count at each iteration
3. Verify count only changes at iterations divisible by N

**Pass Criteria:**
- Gaussian count changes only at expected iterations

**Severity if Failed:** Low

---

### 7.2 Pruning

#### TC-ADC-010: Opacity-Based Pruning

**Objective:** Verify low-opacity Gaussians are removed.

**Procedure:**
1. Initialize some Gaussians with very low opacity (e.g., 0.001)
2. Run training with pruning enabled
3. Verify low-opacity Gaussians are removed

**Pass Criteria:**
- Gaussians with opacity below threshold are removed
- Reconstruction quality maintained or improved after pruning

**Severity if Failed:** Medium

---

#### TC-ADC-011: Scale-Based Pruning

**Objective:** Verify excessively large Gaussians are removed.

**Procedure:**
1. Initialize some Gaussians with extreme scale (e.g., 100x scene size)
2. Run training with pruning enabled
3. Verify oversized Gaussians are removed

**Pass Criteria:**
- Gaussians exceeding scale threshold are removed
- No visual artifacts from pruning

**Severity if Failed:** Medium

---

#### TC-ADC-012: Gaussian Count Stability

**Objective:** Verify Gaussian count stabilizes after sufficient training.

**Procedure:**
1. Train for extended iterations (e.g., 30,000)
2. Plot Gaussian count over time
3. Verify count stabilizes (< 1% change per 1000 iterations after convergence)

**Pass Criteria:**
- Count stabilizes after ~20,000 iterations
- Final count reasonable for scene complexity

**Severity if Failed:** Low

---

## 8. Test Cases: End-to-End Quality

### 8.1 Quantitative Benchmarks

#### TC-E2E-001: Mip-NeRF 360 Benchmark

**Objective:** Verify implementation achieves competitive quality on challenging real-world scenes.

**Test Data:** Mip-NeRF 360 outdoor scenes
- Download: https://jonbarron.info/mipnerf360/

**Scenes to Test:**
- bicycle, garden, stump (outdoor)
- room, counter, kitchen, bonsai (indoor)

**Procedure:**
1. Train each scene for 30,000 iterations
2. Evaluate on held-out test views
3. Compute average PSNR, SSIM, LPIPS

**Reference Metrics (outdoor scenes, from 3DGS paper):**
- PSNR: ~25-27 dB
- SSIM: ~0.75-0.85
- LPIPS: ~0.15-0.25

**Pass Criteria:**
- Metrics within 10% of published baselines

**Severity if Failed:** High

---

#### TC-E2E-002: LPIPS Perceptual Quality

**Objective:** Verify perceptual quality using learned metric.

**Reference Library:** 
- LPIPS: https://github.com/richzhang/PerceptualSimilarity
- Installation: `pip install lpips`

**Procedure:**
1. Render test views
2. Compute LPIPS against ground truth using VGG or AlexNet backbone
3. Report per-scene and average metrics

**Pass Criteria:**
- LPIPS values comparable to published baselines

**Severity if Failed:** Medium

---

### 8.2 Visual Quality Assessment

#### TC-E2E-010: Floater Detection

**Objective:** Identify floating Gaussian artifacts not attached to surfaces.

**Procedure:**
1. Render 360° orbit video around reconstruction
2. Manual inspection for floating blobs/artifacts
3. Render depth maps and check for isolated depth discontinuities

**Pass Criteria:**
- No significant floaters visible in standard views
- Floaters documented if present with severity rating

**Severity if Found:** Medium to High depending on severity

---

#### TC-E2E-011: Edge Sharpness

**Objective:** Verify sharp edges are reconstructed without excessive blur.

**Procedure:**
1. Select test views with sharp edges (text, object boundaries)
2. Compute edge sharpness metric or manual comparison with ground truth
3. Document any systematic blurring

**Pass Criteria:**
- Edges visually comparable to ground truth
- No systematic over-smoothing

**Severity if Failed:** Medium

---

#### TC-E2E-012: View-Dependent Effects

**Objective:** Verify specular highlights and view-dependent appearance are captured.

**Procedure:**
1. Select scene with reflective surfaces (e.g., "garden" has shiny leaves)
2. Render novel views and verify specular highlights move correctly
3. Compare against ground truth video if available

**Pass Criteria:**
- Specular highlights change with viewpoint
- No incorrect "baked in" highlights that don't move

**Severity if Failed:** Medium

---

#### TC-E2E-013: Temporal Stability

**Objective:** Verify rendered video sequences are temporally stable without flickering.

**Procedure:**
1. Render smooth camera path (100+ frames)
2. Compute temporal consistency metric (frame-to-frame SSIM)
3. Manual inspection for popping or flickering artifacts

**Pass Criteria:**
- Frame-to-frame SSIM > 0.95 for slow camera motion
- No visible popping or flickering

**Severity if Failed:** Medium

---

## 9. Test Environment Setup

### 9.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1080 (8GB) | NVIDIA RTX 3090 (24GB) |
| CUDA | 11.7 | 11.8 or 12.x |
| RAM | 16 GB | 64 GB |
| Storage | 50 GB SSD | 200 GB NVMe |

### 9.2 Software Dependencies

```
# Core dependencies
Python 3.8+
PyTorch 2.0+ with CUDA support
NumPy
Pillow

# Evaluation
pip install scikit-image  # SSIM reference
pip install lpips         # Perceptual metrics
pip install opencv-python # Image I/O
pip install matplotlib    # Visualization

# Optional
pip install open3d        # Point cloud visualization
pip install piq           # Additional metrics
```

### 9.3 Dataset Preparation

1. **NeRF Synthetic:**
   - Download from Google Drive link above
   - Extract to `./data/nerf_synthetic/`
   - Verify folder structure: `./data/nerf_synthetic/lego/train/`, etc.

2. **Mip-NeRF 360:**
   - Download from author's page
   - Extract to `./data/360_v2/`
   - Verify images and COLMAP sparse folders present

3. **Synthetic Test Scenes:**
   - Generate procedurally using provided test scene generators
   - Store in `./data/synthetic_tests/`

---

## 10. Test Execution and Reporting

### 10.1 Execution Order

Tests should be executed in the following order due to dependencies:

1. **Phase 1: Foundation** (TC-INP-*, TC-COV-*, TC-SH-*)
   - All tests in Sections 3 and 4
   - Stop if any Critical severity test fails

2. **Phase 2: Rasterizer** (TC-RAS-*)
   - All tests in Section 5
   - Gradient checks (TC-GRAD-*) are highest priority

3. **Phase 3: Optimization** (TC-OPT-*, TC-ADC-*)
   - All tests in Sections 6 and 7
   - Single Gaussian fitting must pass before multi-Gaussian tests

4. **Phase 4: End-to-End** (TC-E2E-*)
   - All tests in Section 8
   - Only execute after Phases 1-3 substantially pass

### 10.2 Results Template

For each test case, record:

| Field | Description |
|-------|-------------|
| Test ID | e.g., TC-GRAD-001 |
| Date Executed | YYYY-MM-DD |
| Tester | Name |
| Git Commit | Hash of code under test |
| Result | PASS / FAIL / BLOCKED / SKIPPED |
| Actual Values | Measured metrics |
| Expected Values | From pass criteria |
| Deviation | Difference between actual and expected |
| Notes | Any observations, error messages, etc. |
| Attachments | Screenshots, logs, rendered images |

### 10.3 Defect Report Template

For each failure, create defect report:

| Field | Description |
|-------|-------------|
| Defect ID | Auto-generated |
| Related Test | Test case ID(s) |
| Severity | Critical / High / Medium / Low |
| Component | Input / Covariance / Rasterizer / Optimization / etc. |
| Summary | One-line description |
| Steps to Reproduce | Numbered steps |
| Expected Behavior | What should happen |
| Actual Behavior | What actually happens |
| Root Cause | If determined |
| Suggested Fix | If known |
| Attachments | Logs, images, minimal repro |

---

## 11. Appendices

### Appendix A: Reference SSIM Implementation Comparison

When comparing against scikit-image SSIM, ensure matching parameters:

```python
# Reference call
from skimage.metrics import structural_similarity as ssim
score = ssim(img1, img2, 
             data_range=1.0,        # or 255 if uint8
             channel_axis=2,        # for RGB
             win_size=11,           # must be odd
             gaussian_weights=True,
             sigma=1.5,
             K1=0.01,
             K2=0.03)
```

### Appendix B: Finite Difference Gradient Check Implementation Notes

Recommended approach:
- Use central differences: (f(x+ε) - f(x-ε)) / 2ε
- ε = 1e-4 for float32, 1e-6 for float64
- For each parameter, compute relative error
- Flag parameters with relative error > 1e-2
- Use double precision for gradient checking even if training uses float32

### Appendix C: Quaternion Conventions

Common conventions to verify:
- Order: (w, x, y, z) vs (x, y, z, w)
- Rotation direction: left-multiply vs right-multiply
- Identity quaternion: (1, 0, 0, 0) should produce identity rotation matrix

### Appendix D: Useful External Resources

| Resource | URL |
|----------|-----|
| 3DGS Project Page | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ |
| 3DGS Paper (arXiv) | https://arxiv.org/abs/2308.04079 |
| EWA Splatting Paper | https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf |
| Spherical Harmonics Reference | https://en.wikipedia.org/wiki/Table_of_spherical_harmonics |
| COLMAP Documentation | https://colmap.github.io/format.html |
| PyTorch Gradient Check | https://pytorch.org/docs/stable/autograd.html#torch.autograd.gradcheck |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | January 2026 | [Author] | Initial release |

---

**End of Test Plan**
