# Implementing SuGaR in Rust: Dependency Graph & Roadmap

## Overview

SuGaR (Surface-Aligned Gaussian Splatting) builds on top of 3D Gaussian Splatting (3DGS), so you need most of 3DGS working first. The core challenge is that the rendering pipeline must be **differentiable** — you need gradients flowing back through the rasterizer to the Gaussian parameters.

---

## Conceptual Dependency Graph

```
                                    ┌─────────────────┐
                                    │   SuGaR Mesh    │
                                    │   Extraction    │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │ SuGaR Surface   │
                                    │ Regularization  │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐     ┌─────────▼────────┐    ┌─────────▼─────────┐
           │    Adaptive     │     │  Differentiable  │    │    Loss Function  │
           │ Density Control │     │   Rasterizer     │    │   (L1 + D-SSIM)   │
           └────────┬────────┘     └─────────┬────────┘    └─────────┬─────────┘
                    │                        │                       │
                    └────────────────────────┼───────────────────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              │              │              │
                    ┌─────────▼───┐  ┌───────▼───────┐  ┌───▼───────────┐
                    │   Gaussian  │  │    Camera     │  │   Spherical   │
                    │   Splatting │  │   Projection  │  │   Harmonics   │
                    └─────────┬───┘  └───────┬───────┘  └───┬───────────┘
                              │              │              │
                              └──────────────┼──────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐    ┌──────────▼──────────┐   ┌─────────▼─────────┐
           │  3D Gaussians   │    │  Covariance Math    │   │  Image/COLMAP     │
           │  Representation │    │  (3D → 2D proj)     │   │  Loading          │
           └────────┬────────┘    └──────────┬──────────┘   └───────────────────┘
                    │                        │
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Linear Algebra Core   │
                    │  (nalgebra or custom)   │
                    └─────────────────────────┘
```

---

## Phase 0: Foundation & Tooling

**Goal:** Get your Rust environment ready and pick your dependencies.

### Crate Decisions

| Need | Options | Recommendation |
|------|---------|----------------|
| Linear algebra | `nalgebra`, `glam`, `ultraviolet` | `nalgebra` — most complete, good for learning |
| GPU compute | `wgpu`, `vulkano`, `cuda-sys` | Start CPU-only, add `wgpu` later |
| Image I/O | `image` | Standard choice |
| Autodiff | `dfdx`, `burn`, or manual | Manual first (see below) |
| Mesh output | `ply-rs`, custom | Custom PLY writer is ~50 lines |

### The Autodiff Question

This is your biggest architectural decision:

1. **Manual gradients** — Implement forward pass, then hand-derive and implement backward pass for each operation. More work, but you'll deeply understand the math. The original 3DGS paper has a good appendix on the gradients.

2. **Use a Rust autodiff framework** — `dfdx` or `burn` can do this, but you'll fight the framework when implementing custom CUDA-like kernels.

3. **Hybrid** — Manual gradients for the core rasterizer (it's specialized anyway), autodiff for the loss functions.

**Recommendation:** Manual gradients. The rasterizer is the hard part, and you need to understand it anyway. The 3DGS paper's supplementary material derives all the gradients you need.

### Deliverables
- [ ] Cargo workspace set up
- [ ] `nalgebra` compiling
- [ ] Basic image loading/saving working
- [ ] Decision on autodiff approach documented

---

## Phase 1: Data Structures & I/O

**Goal:** Load a COLMAP reconstruction and represent Gaussians in memory.

### 1.1 COLMAP Parser

COLMAP outputs several formats. The binary format is most common:

```
sparse/
  cameras.bin    # Camera intrinsics (focal length, principal point, distortion)
  images.bin     # Camera extrinsics (R, t) per image + 2D keypoints
  points3D.bin   # 3D points from SfM (initial Gaussian positions)
```

You need to parse:
- Camera intrinsics → focal length, cx, cy
- Camera extrinsics → rotation (quaternion or matrix), translation
- 3D points → initial positions + colors for Gaussians

**Spec:** https://colmap.github.io/format.html

### 1.2 Gaussian Representation

Each Gaussian has:

```rust
struct Gaussian {
    position: Vector3<f32>,        // μ (mean)
    // Covariance stored as scale + rotation (more stable than raw covariance)
    scale: Vector3<f32>,           // s (log-scale for numerical stability)
    rotation: UnitQuaternion<f32>, // q (rotation as quaternion)
    opacity: f32,                  // α (logit-space for optimization)
    sh_coeffs: [Vector3<f32>; 16], // Spherical harmonics (RGB × 16 coeffs for degree 3)
}
```

**Key insight:** The 3D covariance Σ is reconstructed as Σ = R · S · Sᵀ · Rᵀ where S = diag(exp(scale)).

### 1.3 Camera Model

```rust
struct Camera {
    // Intrinsics
    fx: f32, fy: f32,     // Focal lengths
    cx: f32, cy: f32,     // Principal point
    width: u32, height: u32,
    
    // Extrinsics (world → camera)
    rotation: Matrix3<f32>,
    translation: Vector3<f32>,
}

impl Camera {
    fn world_to_camera(&self, p: &Vector3<f32>) -> Vector3<f32>;
    fn project(&self, p_cam: &Vector3<f32>) -> Vector2<f32>;
    fn view_matrix(&self) -> Matrix4<f32>;
}
```

### Deliverables
- [ ] COLMAP binary parser (cameras, images, points3D)
- [ ] Gaussian struct with covariance reconstruction
- [ ] Camera projection working (test: project 3D points, overlay on image)

---

## Phase 2: Forward Rendering (Non-Differentiable)

**Goal:** Render an image from Gaussians. No gradients yet — just get pixels on screen.

### 2.1 Spherical Harmonics

SH encodes view-dependent color. For degree 3, you have 16 coefficients per color channel.

```rust
/// Evaluate SH basis functions for a view direction
fn sh_basis(dir: &Vector3<f32>) -> [f32; 16] {
    // Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, ... Y_3^3
    // Formulas are standard — see paper appendix or Wikipedia
}

/// Get RGB color for a Gaussian from a given view direction
fn evaluate_sh(sh_coeffs: &[[f32; 3]; 16], dir: &Vector3<f32>) -> Vector3<f32> {
    let basis = sh_basis(dir);
    // Dot product of basis with coefficients for each channel
}
```

**Simplification for Phase 2:** Start with just the DC component (sh_coeffs[0]), which gives view-independent color. Add higher-order SH later.

### 2.2 Project 3D Covariance to 2D

This is the trickiest math. Given:
- 3D Gaussian with mean μ and covariance Σ
- Camera with view matrix W and projection J (Jacobian of perspective projection)

The 2D covariance is: Σ' = J · W · Σ · Wᵀ · Jᵀ (taking the upper-left 2×2)

```rust
fn project_gaussian(
    gaussian: &Gaussian,
    camera: &Camera,
) -> Option<Gaussian2D> {
    // 1. Transform mean to camera space
    let mean_cam = camera.world_to_camera(&gaussian.position);
    
    // 2. Check if in front of camera
    if mean_cam.z <= 0.1 { return None; }
    
    // 3. Compute Jacobian of perspective projection at this point
    let J = perspective_jacobian(mean_cam, camera.fx, camera.fy);
    
    // 4. Transform 3D covariance to 2D
    let Σ_3d = gaussian.covariance_matrix();
    let W = camera.rotation; // 3x3 rotation part
    let Σ_2d = J * W * Σ_3d * W.transpose() * J.transpose();
    
    // 5. Project mean to pixel coordinates
    let mean_2d = camera.project(&mean_cam);
    
    Some(Gaussian2D { mean: mean_2d, covariance: Σ_2d, ... })
}
```

### 2.3 Tile-Based Rasterization

The original paper uses a tile-based approach for GPU efficiency. For CPU, you can start simpler:

**Simple approach (slow but correct):**
1. Project all Gaussians to 2D
2. Sort by depth (front-to-back or back-to-front)
3. For each pixel, accumulate contributions from overlapping Gaussians

**Tile-based approach (faster, needed for GPU):**
1. Divide image into 16×16 tiles
2. For each Gaussian, compute which tiles it overlaps (bounding box from 2D covariance)
3. Sort Gaussians within each tile by depth
4. Rasterize per-tile

### 2.4 Alpha Blending

For each pixel, blend Gaussians front-to-back:

```rust
fn blend_gaussians(gaussians: &[Gaussian2D], pixel: Vector2<f32>) -> Vector3<f32> {
    let mut color = Vector3::zeros();
    let mut transmittance = 1.0f32;
    
    for g in gaussians.iter() {  // sorted front-to-back
        // Evaluate 2D Gaussian at pixel
        let diff = pixel - g.mean;
        let power = -0.5 * diff.transpose() * g.covariance_inv * diff;
        let alpha = g.opacity * (-power).exp().min(0.99);
        
        // Accumulate
        color += transmittance * alpha * g.color;
        transmittance *= 1.0 - alpha;
        
        // Early termination
        if transmittance < 0.001 { break; }
    }
    
    color
}
```

### Deliverables
- [ ] SH evaluation (start with DC only)
- [ ] 3D → 2D covariance projection
- [ ] Basic rasterizer (can be naive per-pixel loop)
- [ ] Render a test scene and compare to reference implementation

---

## Phase 3: Differentiable Rasterization

**Goal:** Compute gradients of image loss w.r.t. all Gaussian parameters.

This is the core of the whole system. You need gradients through:
- Alpha blending
- 2D Gaussian evaluation  
- Covariance projection (3D → 2D)
- SH evaluation
- Scale/rotation → covariance reconstruction

### 3.1 Backward Pass Structure

The backward pass mirrors the forward pass in reverse:

```rust
struct GaussianGradients {
    d_position: Vector3<f32>,
    d_scale: Vector3<f32>,
    d_rotation: Vector4<f32>,  // Quaternion gradients
    d_opacity: f32,
    d_sh_coeffs: [[f32; 3]; 16],
}

fn rasterize_backward(
    gaussians: &[Gaussian],
    camera: &Camera,
    d_image: &Image<Vector3<f32>>,  // Gradient of loss w.r.t. output pixels
) -> Vec<GaussianGradients> {
    // ... reverse of forward pass
}
```

### 3.2 Key Gradient Derivations

The 3DGS paper supplementary material has all of these. Key ones:

**Alpha blending backward:**
```
Given: C = Σᵢ cᵢ αᵢ Πⱼ<ᵢ(1-αⱼ)
∂L/∂cᵢ = ∂L/∂C · αᵢ · Tᵢ   where Tᵢ = Πⱼ<ᵢ(1-αⱼ)
∂L/∂αᵢ = ∂L/∂C · (cᵢ·Tᵢ - Σⱼ>ᵢ cⱼ·αⱼ·Tⱼ/(1-αᵢ))
```

**2D Gaussian evaluation backward:**
```
G(x) = exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))
∂G/∂μ = G · Σ⁻¹(x-μ)
∂G/∂Σ⁻¹ = -½ G · (x-μ)(x-μ)ᵀ
```

**Covariance projection backward:**
- Chain rule through J·W·Σ·Wᵀ·Jᵀ
- Gradients w.r.t. Σ₃D, then chain to scale and rotation

### 3.3 Implementation Strategy

1. Implement forward pass saving all intermediates needed for backward
2. Implement backward pass operation by operation
3. **Gradient checking:** Verify each operation with finite differences

```rust
fn gradient_check<F>(f: F, x: &[f32], analytical_grad: &[f32], eps: f32) -> bool 
where F: Fn(&[f32]) -> f32 
{
    for i in 0..x.len() {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[i] += eps;
        x_minus[i] -= eps;
        let numerical = (f(&x_plus) - f(&x_minus)) / (2.0 * eps);
        let diff = (numerical - analytical_grad[i]).abs();
        if diff > 1e-4 { return false; }
    }
    true
}
```

### Deliverables
- [ ] Backward pass for alpha blending
- [ ] Backward pass for 2D Gaussian evaluation
- [ ] Backward pass for covariance projection
- [ ] Backward pass for SH evaluation
- [ ] Backward pass for quaternion → rotation matrix
- [ ] Gradient checking passing for all operations

---

## Phase 4: Optimization Loop

**Goal:** Train Gaussians to reconstruct a scene from images.

### 4.1 Loss Function

```rust
fn compute_loss(rendered: &Image, target: &Image) -> (f32, Image) {
    // L1 loss
    let l1 = (rendered - target).abs().mean();
    
    // D-SSIM loss (structural similarity)
    let ssim = compute_ssim(rendered, target);
    let d_ssim = 1.0 - ssim;
    
    // Combined (paper uses λ=0.2)
    let loss = 0.8 * l1 + 0.2 * d_ssim;
    
    // Return loss and gradient image
    let d_image = 0.8 * d_l1 + 0.2 * d_ssim;
    (loss, d_image)
}
```

**SSIM implementation:** This is a bit involved — 11×11 Gaussian-weighted windows, computing means, variances, covariances. There are reference implementations you can port.

### 4.2 Adam Optimizer

```rust
struct AdamState {
    m: Vec<f32>,  // First moment
    v: Vec<f32>,  // Second moment
    t: u32,       // Timestep
}

impl AdamState {
    fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32, β1: f32, β2: f32, ε: f32) {
        self.t += 1;
        for i in 0..params.len() {
            self.m[i] = β1 * self.m[i] + (1.0 - β1) * grads[i];
            self.v[i] = β2 * self.v[i] + (1.0 - β2) * grads[i].powi(2);
            let m_hat = self.m[i] / (1.0 - β1.powi(self.t));
            let v_hat = self.v[i] / (1.0 - β2.powi(self.t));
            params[i] -= lr * m_hat / (v_hat.sqrt() + ε);
        }
    }
}
```

The paper uses different learning rates for different parameters:
- Position: 0.00016 (decays exponentially)
- Scale: 0.005
- Rotation: 0.001
- Opacity: 0.05
- SH DC: 0.0025
- SH rest: 0.000125

### 4.3 Adaptive Density Control

Every N iterations (paper uses 100), adjust the Gaussian count:

**Densification (splitting/cloning):**
```rust
fn densify(gaussians: &mut Vec<Gaussian>, grad_accum: &[Vector3<f32>], threshold: f32) {
    let mut new_gaussians = Vec::new();
    
    for (i, g) in gaussians.iter().enumerate() {
        let grad_magnitude = grad_accum[i].norm();
        
        if grad_magnitude > threshold {
            if g.scale.max() > some_threshold {
                // Large Gaussian with high gradient → SPLIT into 2 smaller
                let (g1, g2) = split_gaussian(g);
                new_gaussians.push(g1);
                new_gaussians.push(g2);
            } else {
                // Small Gaussian with high gradient → CLONE
                new_gaussians.push(g.clone());
                new_gaussians.push(clone_gaussian(g));
            }
        } else {
            new_gaussians.push(g.clone());
        }
    }
    
    *gaussians = new_gaussians;
}
```

**Pruning:**
```rust
fn prune(gaussians: &mut Vec<Gaussian>, opacity_threshold: f32) {
    gaussians.retain(|g| sigmoid(g.opacity) > opacity_threshold);
}
```

**Opacity reset:** Periodically reset all opacities to near-zero to let optimization "restart" decisions.

### 4.4 Training Loop

```rust
fn train(colmap_path: &Path, output_path: &Path, iterations: u32) {
    // Initialize from COLMAP
    let (cameras, images, points) = load_colmap(colmap_path);
    let mut gaussians = initialize_gaussians(&points);
    let mut optimizer = AdamState::new(/* ... */);
    
    for iter in 0..iterations {
        // Sample random camera
        let cam_idx = rand::random::<usize>() % cameras.len();
        let camera = &cameras[cam_idx];
        let target = &images[cam_idx];
        
        // Forward pass
        let rendered = rasterize(&gaussians, camera);
        
        // Loss
        let (loss, d_image) = compute_loss(&rendered, target);
        
        // Backward pass
        let gradients = rasterize_backward(&gaussians, camera, &d_image);
        
        // Optimizer step
        optimizer.step(&mut gaussians, &gradients);
        
        // Adaptive density control
        if iter % 100 == 0 && iter < 15000 {
            densify_and_prune(&mut gaussians, &grad_accum);
        }
        
        // Logging
        if iter % 1000 == 0 {
            println!("Iter {}: loss={:.4}, num_gaussians={}", iter, loss, gaussians.len());
        }
    }
    
    save_gaussians(&gaussians, output_path);
}
```

### Deliverables
- [ ] SSIM implementation
- [ ] Adam optimizer with per-parameter learning rates
- [ ] Densification (split + clone logic)
- [ ] Pruning
- [ ] Full training loop
- [ ] Successfully reconstruct a simple scene (e.g., single object)

---

## Phase 5: SuGaR Regularization

**Goal:** Add surface-alignment regularization so Gaussians become suitable for mesh extraction.

### 5.1 The SuGaR Idea

Standard 3DGS produces "fuzzy" Gaussians that don't align with surfaces. SuGaR adds:

1. **Flatness regularization** — Encourage Gaussians to be flat (one scale dimension ≪ others)
2. **Alignment regularization** — Encourage flat Gaussians to align with local surface normals
3. **Density regularization** — Encourage consistent density along surfaces

### 5.2 Normal Estimation

For each Gaussian, estimate the local surface normal from neighbors:

```rust
fn estimate_normals(gaussians: &[Gaussian], k: usize) -> Vec<Vector3<f32>> {
    let kdtree = build_kdtree(&gaussians.iter().map(|g| g.position).collect());
    
    gaussians.iter().map(|g| {
        let neighbors = kdtree.nearest_k(&g.position, k);
        let covariance = compute_covariance(&neighbors);
        // Normal is eigenvector of smallest eigenvalue
        let (_, eigenvectors) = symmetric_eigen(&covariance);
        eigenvectors.column(0).into()  // Smallest eigenvalue's eigenvector
    }).collect()
}
```

### 5.3 Regularization Losses

```rust
fn sugar_regularization(gaussians: &[Gaussian], normals: &[Vector3<f32>]) -> f32 {
    let mut loss = 0.0;
    
    for (g, n) in gaussians.iter().zip(normals) {
        // 1. Flatness: penalize if smallest scale isn't much smaller than others
        let scales = g.scale.map(|s| s.exp());
        let min_scale = scales.min();
        let max_scale = scales.max();
        let flatness_loss = min_scale / max_scale;  // Want this small
        
        // 2. Alignment: penalize if Gaussian's flat direction doesn't match normal
        let gaussian_normal = g.smallest_eigenvector();  // Direction of smallest scale
        let alignment_loss = 1.0 - gaussian_normal.dot(n).abs();
        
        loss += flatness_loss + alignment_loss;
    }
    
    loss / gaussians.len() as f32
}
```

### 5.4 Modified Training

```rust
fn train_sugar(/* ... */) {
    // ... same as before, but:
    
    // After some iterations of pure 3DGS, add regularization
    if iter > warmup_iterations {
        let normals = estimate_normals(&gaussians, 16);
        let reg_loss = sugar_regularization(&gaussians, &normals);
        total_loss += lambda_reg * reg_loss;
        // Add regularization gradients to parameter gradients
    }
}
```

### Deliverables
- [ ] KD-tree for nearest neighbor queries
- [ ] Normal estimation via local PCA
- [ ] Flatness regularization loss + gradients
- [ ] Alignment regularization loss + gradients
- [ ] Integration into training loop

---

## Phase 6: Mesh Extraction

**Goal:** Extract a triangle mesh from the regularized Gaussians.

### 6.1 Density Field

SuGaR defines a density field from the Gaussians:

```rust
fn density_at_point(point: &Vector3<f32>, gaussians: &[Gaussian]) -> f32 {
    gaussians.iter().map(|g| {
        let diff = point - g.position;
        let mahalanobis = diff.transpose() * g.covariance_inv() * diff;
        let weight = sigmoid(g.opacity) * (-0.5 * mahalanobis).exp();
        weight
    }).sum()
}
```

### 6.2 Marching Cubes

Extract an isosurface from the density field:

```rust
fn extract_mesh(gaussians: &[Gaussian], resolution: u32, iso_value: f32) -> Mesh {
    // 1. Compute bounding box of Gaussians
    let (min, max) = compute_bounds(gaussians);
    
    // 2. Create 3D grid
    let grid = create_grid(min, max, resolution);
    
    // 3. Evaluate density at grid vertices
    let densities: Vec<f32> = grid.vertices().map(|v| density_at_point(&v, gaussians)).collect();
    
    // 4. Run marching cubes
    marching_cubes(&grid, &densities, iso_value)
}
```

**Marching cubes** is a well-documented algorithm. You can find lookup tables and implementations to port.

### 6.3 Poisson Reconstruction (Alternative)

SuGaR can also use Poisson surface reconstruction:
1. Sample points on Gaussian surfaces
2. Use estimated normals
3. Run Poisson reconstruction

This typically gives smoother results but requires implementing or binding to a Poisson solver.

### 6.4 Mesh Output

```rust
fn save_ply(mesh: &Mesh, path: &Path) {
    // Write PLY header + vertices + faces
    // PLY format is simple text or binary
}

fn save_obj(mesh: &Mesh, path: &Path) {
    // Write OBJ format
    // v x y z
    // f i j k
}
```

### Deliverables
- [ ] Density field evaluation (accelerated with spatial data structure)
- [ ] Marching cubes implementation
- [ ] Mesh export (PLY or OBJ)
- [ ] Visual validation in Blender/MeshLab

---

## Phase 7 (Optional): Mesh Refinement with Bound Gaussians

SuGaR's second stage binds flat Gaussians to the extracted mesh and refines jointly. This is optional but improves quality.

### 7.1 Binding Gaussians to Mesh

```rust
struct BoundGaussian {
    face_idx: u32,           // Which triangle
    barycentric: Vector3<f32>, // Position within triangle
    // ... other learnable params
}
```

### 7.2 Joint Optimization

Optimize both mesh vertex positions and bound Gaussian parameters. This requires gradients through the mesh→Gaussian binding.

---

## Phase 8: Performance Optimization

Once correctness is verified, optimize:

### CPU Optimizations
- [ ] SIMD for Gaussian evaluation (use `packed_simd` or `std::simd`)
- [ ] Parallelize with `rayon`
- [ ] Better memory layout (SoA vs AoS)

### GPU Port
- [ ] Port rasterizer to `wgpu` compute shaders
- [ ] Tile-based culling on GPU
- [ ] Efficient sorting (bitonic sort or radix sort)

---

## Testing Strategy

### Unit Tests
- Covariance projection math (compare to numpy)
- SH evaluation (compare to reference)
- Gradient checking for each differentiable operation

### Integration Tests
- Reconstruct synthetic scenes with known geometry
- Compare rendered images to reference implementation
- Compare extracted meshes to ground truth

### Reference Comparisons
- Run original Python implementation on same data
- Compare intermediate values (projected Gaussians, losses, gradients)

---

## Suggested Timeline

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| 0 | Foundation & tooling | 1-2 days |
| 1 | Data structures & I/O | 2-3 days |
| 2 | Forward rendering | 1 week |
| 3 | Differentiable rasterization | 1-2 weeks |
| 4 | Optimization loop | 1 week |
| 5 | SuGaR regularization | 3-4 days |
| 6 | Mesh extraction | 3-4 days |
| 7 | Mesh refinement (optional) | 1 week |
| 8 | Performance optimization | Ongoing |

**Total:** 5-8 weeks for a working implementation, depending on experience and time investment.

---

## Resources

### Papers (in reading order)
1. **3D Gaussian Splatting** — Main paper + supplementary (has gradient derivations)
2. **SuGaR** — The regularization and mesh extraction additions
3. **EWA Splatting** (Zwicker 2001) — Original splatting math that 3DGS builds on

### Code References
- Original 3DGS: https://github.com/graphdeco-inria/gaussian-splatting
- Original SuGaR: https://github.com/Anttwo/SuGaR
- gsplat (cleaner CUDA): https://github.com/nerfstudio-project/gsplat

### Useful for Rust
- `nalgebra` docs for matrix operations
- Marching cubes lookup tables
- PLY format spec

---

## Appendix: Quick Reference for Key Math

### Quaternion to Rotation Matrix
```
R = | 1-2(y²+z²)   2(xy-wz)    2(xz+wy)  |
    | 2(xy+wz)     1-2(x²+z²)  2(yz-wx)  |
    | 2(xz-wy)     2(yz+wx)    1-2(x²+y²)|
```

### Covariance from Scale + Rotation
```
S = diag(exp(s_x), exp(s_y), exp(s_z))
Σ = R · S · S^T · R^T
```

### 3D to 2D Covariance Projection
```
J = | f_x/z    0     -f_x·x/z² |
    |   0    f_y/z   -f_y·y/z² |

Σ_2D = (J · W · Σ_3D · W^T · J^T)[:2,:2]
```

### Spherical Harmonics (Degree 0-3)
```
Y_0^0 = 0.28209479
Y_1^{-1} = 0.48860251 · y
Y_1^0 = 0.48860251 · z
Y_1^1 = 0.48860251 · x
... (16 total for degree 3)
```
