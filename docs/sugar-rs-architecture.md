# sugar-rs Module Architecture

## Crate Structure

```
sugar-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs                 # Crate root, re-exports
│   │
│   ├── core/                  # Fundamental types
│   │   ├── mod.rs
│   │   ├── gaussian.rs        # Gaussian, Gaussian2D, GaussianCloud
│   │   ├── camera.rs          # Camera intrinsics/extrinsics
│   │   ├── sh.rs              # Spherical harmonics evaluation
│   │   └── math.rs            # Quaternions, sigmoid, covariance math
│   │
│   ├── io/                    # All I/O operations
│   │   ├── mod.rs
│   │   ├── colmap.rs          # Parse cameras.bin, images.bin, points3D.bin
│   │   ├── ply.rs             # Read/write PLY (Gaussians + meshes)
│   │   ├── obj.rs             # Write OBJ meshes
│   │   └── checkpoint.rs      # Save/load training state
│   │
│   ├── render/                # Forward rendering pipeline
│   │   ├── mod.rs
│   │   ├── project.rs         # 3D→2D Gaussian projection
│   │   ├── rasterize.rs       # Tile-based rasterization
│   │   ├── blend.rs           # Alpha compositing
│   │   └── cpu.rs             # CPU renderer tying it together
│   │
│   ├── diff/                  # Differentiable operations (backward passes)
│   │   ├── mod.rs
│   │   ├── project_grad.rs    # Gradients for projection
│   │   ├── rasterize_grad.rs  # Gradients for rasterization
│   │   ├── blend_grad.rs      # Gradients for alpha blending
│   │   └── sh_grad.rs         # Gradients for SH evaluation
│   │
│   ├── optim/                 # Optimization
│   │   ├── mod.rs
│   │   ├── adam.rs            # Adam optimizer with per-param LR
│   │   ├── loss.rs            # L1, SSIM, D-SSIM
│   │   ├── density.rs         # Adaptive density control (split/clone/prune)
│   │   └── trainer.rs         # Main training loop orchestration
│   │
│   ├── sugar/                 # SuGaR-specific functionality
│   │   ├── mod.rs
│   │   ├── regularization.rs  # Flatness + alignment losses
│   │   ├── normals.rs         # Local normal estimation via PCA
│   │   ├── density_field.rs   # Evaluate density at arbitrary points
│   │   ├── marching_cubes.rs  # Isosurface extraction
│   │   └── mesh.rs            # Mesh data structure + operations
│   │
│   ├── gpu/                   # GPU backend (feature-gated)
│   │   ├── mod.rs
│   │   ├── context.rs         # wgpu device/queue setup
│   │   ├── buffers.rs         # GPU buffer management
│   │   ├── pipelines.rs       # Compute pipeline creation
│   │   ├── shaders/           # WGSL shader sources
│   │   │   ├── project.wgsl
│   │   │   ├── sort.wgsl
│   │   │   ├── rasterize.wgsl
│   │   │   └── backward.wgsl
│   │   └── renderer.rs        # GPU renderer implementation
│   │
│   └── bin/                   # CLI entry points
│       ├── train.rs           # sugar-train: run optimization
│       ├── render.rs          # sugar-render: render images from trained model
│       └── extract.rs         # sugar-extract: mesh extraction
│
└── tests/
    ├── gradient_check.rs      # Finite difference gradient verification
    ├── render_compare.rs      # Compare against reference implementation
    └── fixtures/              # Test data (small COLMAP scenes)
```

## Module Responsibilities

### core/
Pure data structures and math. No I/O, no rendering logic. Everything here is `#[derive(Clone, Debug, Serialize, Deserialize)]` friendly.

- **gaussian.rs**: The `Gaussian` struct (position, scale, rotation, opacity, SH coeffs), methods to compute covariance matrix, and `GaussianCloud` as a collection with SoA layout option.
- **camera.rs**: Pinhole camera model. World↔camera transforms, projection, unprojection.
- **sh.rs**: Spherical harmonics basis functions up to degree 3. Pure functions, no state.
- **math.rs**: Quaternion↔matrix conversion, sigmoid/logit, perspective Jacobian.

### io/
All file format handling isolated here. Makes the rest of the crate agnostic to serialization details.

- **colmap.rs**: Binary format parsing. Returns `CameraSet` + initial point cloud.
- **ply.rs**: Gaussian cloud serialization (matches original 3DGS format for interop).
- **checkpoint.rs**: Training state snapshots (Gaussians + optimizer state + iteration).

### render/
Forward pass only. Produces images from Gaussians. No gradient computation.

- **project.rs**: `project_gaussian(g: &Gaussian, cam: &Camera) -> Option<Gaussian2D>`. Handles frustum culling, covariance projection.
- **rasterize.rs**: Tile assignment, depth sorting within tiles.
- **blend.rs**: Front-to-back alpha compositing with early termination.
- **cpu.rs**: `CpuRenderer` struct that owns scratch buffers and orchestrates the pipeline.

### diff/
Backward passes mirroring render/. Each file corresponds to its forward counterpart.

The key insight: backward pass needs intermediate values from forward pass. Either:
1. Recompute them (slower, simpler)
2. Cache them during forward (faster, more memory)

I'd start with recomputation for correctness, optimize later.

### optim/
Training machinery. Doesn't know about rendering details, just takes gradients and updates parameters.

- **adam.rs**: Generic Adam with momentum state. Supports per-parameter learning rates.
- **loss.rs**: Image-space losses. SSIM is the complex one — needs Gaussian-windowed statistics.
- **density.rs**: The split/clone/prune logic. Operates on `GaussianCloud` + accumulated gradients.
- **trainer.rs**: Owns the training loop, calls into render + diff + optimizer.

### sugar/
Everything specific to SuGaR that goes beyond vanilla 3DGS.

- **normals.rs**: KD-tree queries + local PCA for normal estimation.
- **regularization.rs**: Flatness and alignment losses with their gradients.
- **density_field.rs**: Sum-of-Gaussians density evaluation (used by marching cubes).
- **marching_cubes.rs**: Classic algorithm. Lookup tables, edge interpolation.
- **mesh.rs**: Vertices + faces + optional vertex colors/normals.

### gpu/
Feature-gated (`#[cfg(feature = "gpu")]`). Mirrors render/ but on GPU.

- **context.rs**: wgpu instance, adapter, device, queue. Platform-agnostic setup.
- **buffers.rs**: Typed wrappers around wgpu buffers. Upload/download helpers.
- **pipelines.rs**: Compute pipeline compilation, bind group layouts.
- **shaders/**: WGSL source files, included via `include_str!`.
- **renderer.rs**: `GpuRenderer` implementing same trait as `CpuRenderer`.

## Key Traits

```
trait Renderer {
    fn render(&mut self, gaussians: &GaussianCloud, camera: &Camera) -> Image;
}

trait DifferentiableRenderer: Renderer {
    fn render_backward(
        &mut self,
        gaussians: &GaussianCloud,
        camera: &Camera,
        d_image: &Image,
    ) -> GaussianGradients;
}
```

This lets you swap CPU/GPU backends and test them against each other.

## Data Flow

```
                    TRAINING
                    ════════

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   COLMAP    │────▶│  Initialize │────▶│  Gaussian   │
│   Scene     │     │  Gaussians  │     │   Cloud     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                         ┌─────────────────────┘
                         ▼
              ┌─────────────────────┐
              │   Training Loop     │◀─────────────────┐
              └──────────┬──────────┘                  │
                         │                             │
                         ▼                             │
              ┌─────────────────────┐                  │
              │  Sample Camera +    │                  │
              │  Target Image       │                  │
              └──────────┬──────────┘                  │
                         │                             │
                         ▼                             │
              ┌─────────────────────┐                  │
              │  Forward Render     │                  │
              │  (render/)          │                  │
              └──────────┬──────────┘                  │
                         │                             │
                         ▼                             │
              ┌─────────────────────┐                  │
              │  Compute Loss       │                  │
              │  (optim/loss.rs)    │                  │
              └──────────┬──────────┘                  │
                         │                             │
                         ▼                             │
              ┌─────────────────────┐                  │
              │  Backward Pass      │                  │
              │  (diff/)            │                  │
              └──────────┬──────────┘                  │
                         │                             │
                         ▼                             │
              ┌─────────────────────┐                  │
              │  Optimizer Step     │                  │
              │  (optim/adam.rs)    │                  │
              └──────────┬──────────┘                  │
                         │                             │
                         ▼                             │
              ┌─────────────────────┐                  │
              │  Density Control    │──────────────────┘
              │  (optim/density.rs) │
              └─────────────────────┘


                    MESH EXTRACTION
                    ═══════════════

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Trained    │────▶│  Estimate   │────▶│   SuGaR     │
│  Gaussians  │     │  Normals    │     │   Refine    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                         ┌─────────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Build Density      │
              │  Field              │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Marching Cubes     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Export Mesh        │
              │  (PLY/OBJ)          │
              └─────────────────────┘
```

## Suggested Implementation Order

1. **core/** — Get the math right, test quaternion conversions, covariance reconstruction
2. **io/colmap.rs** — Load a real scene
3. **render/** (CPU) — See images, compare to reference
4. **diff/** — Gradient checking, this is where bugs hide
5. **optim/** — Get training running on a simple scene
6. **sugar/regularization.rs** — Add surface alignment
7. **sugar/marching_cubes.rs** — Extract meshes
8. **gpu/** — Port hot paths for speed

## Testing Strategy

```
tests/
├── gradient_check.rs      # For EVERY differentiable op:
│                          #   - Perturb input by ε
│                          #   - Compute (f(x+ε) - f(x-ε)) / 2ε
│                          #   - Compare to analytical gradient
│                          #   - Assert relative error < 1e-4
│
├── render_compare.rs      # Load same scene in Python reference
│                          # Export intermediate values (projected Gaussians, etc.)
│                          # Compare Rust outputs to Python outputs
│
└── integration/
    ├── train_synthetic.rs # Train on synthetic scene with known geometry
    └── mesh_quality.rs    # Compare extracted mesh to ground truth
```

## Dependencies Summary

**Required:**
- nalgebra — matrices, vectors, quaternions
- image — PNG/JPEG I/O
- rayon — parallel iteration
- byteorder — COLMAP binary parsing
- kiddo — KD-tree for neighbor queries

**Optional (gpu feature):**
- wgpu — GPU compute
- pollster — async runtime for wgpu

**Dev:**
- approx — float comparison in tests
