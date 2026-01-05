# SplatRs Agent Context & Guidelines

**Last Updated:** 2025-12-12
**Project:** Implementing SuGaR (Surface-Aligned Gaussian Splatting) in Rust
**Goal:** Working Gaussian Splatting system with mesh extraction from photos

---

## Beads

Use 'bd' for task tracking.

`bd quickstart` for details.

## Python

This is not a python project, but you can activate a venv and install any tools that will help you.

    source .venv/bin/activate

Please don't commit any python related stuff to the repo.

## Project Status

**Current Phase:** Phase 0 (Design & Foundation)
**Current Milestone:** Pre-M1 (project setup & design decisions)

### User Priorities (in order)
1. Working Gaussian Splatting system in Rust
2. Learn deeply about Gaussian Splatting (educational focus)
3. Extract meshes from personal photos
4. Rock-solid, educational unit tests at every level

### Test Data
- **Location:** `/Users/ozten/Projects/GuassianPlay/digital_calipers2_project`
- **Object:** Digital calipers (good test case - mechanical object with clear geometry)
- **COLMAP Data:** Available at `colmap_workspace/sparse/0/` (cameras.bin, images.bin, points3D.bin)

---

## Documentation Structure

### Core Documents
1. **`docs/sugar-rs-architecture.md`** - Module architecture, crate structure, data flow diagrams
2. **`docs/sugar-rs-milestones.md`** - 14 concrete checkpoints with verification criteria (M1-M14)
3. **`docs/sugar-rust-roadmap.md`** - 8 implementation phases with detailed technical guidance

### Milestone to Phase Mapping
- Phase 0 (Foundation) → No milestone (setup only)
- Phase 1 (Data structures & I/O) → M1-M2
- Phase 2 (Forward rendering) → M3-M5
- Phase 3 (Differentiable rasterization) → M6
- Phase 4 (Optimization loop) → M7-M10
- Phase 5 (SuGaR regularization) → M11
- Phase 6 (Mesh extraction) → M12
- Phase 7 (Mesh refinement) → No milestone (optional)
- Phase 8 (Performance/GPU) → M13-M14

**Phases** = what to build (work scope)
**Milestones** = how to verify it works (testable criteria)

---

## Design Decisions (Locked In)

### 1. Data Structure: Array-of-Structs (AoS)
```rust
struct Gaussian {
    position: Vector3<f32>,
    scale: Vector3<f32>,           // Log-space for stability
    rotation: UnitQuaternion<f32>,
    opacity: f32,                  // Logit-space for optimization
    sh_coeffs: [[f32; 3]; 16],     // Degree-3 spherical harmonics
}
```
**Rationale:** Start simple, migrate to SoA later if SIMD/GPU requires it.

### 2. Renderer Ownership: Stateful with Scratch Buffers
```rust
struct CpuRenderer {
    projected_gaussians: Vec<Gaussian2D>,
    tile_bins: Vec<Vec<usize>>,
    depth_sorted_indices: Vec<usize>,
}

impl CpuRenderer {
    fn render(&mut self, gaussians: &[Gaussian], camera: &Camera) -> Image;
}
```
**Rationale:** Avoid per-frame allocations. `&mut self` makes state explicit.

### 3. Gradient Structure: Mirror Gaussian Layout
```rust
struct GaussianGradients {
    d_position: Vec<Vector3<f32>>,
    d_scale: Vec<Vector3<f32>>,
    d_rotation: Vec<Vector4<f32>>,
    d_opacity: Vec<f32>,
    d_sh_coeffs: Vec<[[f32; 3]; 16]>,
}
```
**Rationale:** Type-safe, clear mapping. Optimizer can flatten later if needed.

### 4. Camera Data: Separate from Gaussians
```rust
struct ColmapScene {
    cameras: Vec<Camera>,
    images: Vec<Image>,
}
// Separate:
let mut gaussians: Vec<Gaussian> = initialize(&scene);
```
**Rationale:** Cameras/images are immutable inputs. Gaussians are mutable training state. Clear ownership.

### 5. I/O: Read and Own (No Memory Mapping)
**Rationale:** Don't over-optimize. COLMAP data is small, loaded once.

---

## Design Decisions (Under Discussion)

### Forward-Backward Pass Coupling

**Current question:** How to handle intermediate values needed by backward pass?

**Option 1: Recompute (RECOMMENDED FOR LEARNING)**
```rust
fn render(&mut self, gaussians: &[Gaussian], camera: &Camera) -> Image;
fn render_backward(&mut self, gaussians: &[Gaussian], camera: &Camera, d_image: &Image) -> GaussianGradients;
```
- ✅ Simple, no coupling, easy to understand
- ❌ 2x compute cost
- 📚 Great for learning the math

**Option 2: Cache in Renderer (fast but fragile)**
```rust
fn render(&mut self, ...) -> Image;  // Fills self.cached_data
fn render_backward(&mut self, ...) -> GaussianGradients;  // Uses self.cached_data
```
- ❌ Fragile: calling backward without forward is UB
- ✅ Fast
- ⚠️  Use only after profiling shows need

**Option 3: Explicit Context (type-safe caching)**
```rust
struct RenderContext { /* intermediates */ image: Image }
fn render(&mut self, ...) -> RenderContext;
fn render_backward(&mut self, context: RenderContext, ...) -> GaussianGradients;
```
- ✅ Type-safe: can't call backward without forward
- ✅ Fast
- 📚 Good for production

**Status:** Start with Option 1, migrate to Option 3 if profiling shows need.

---

### Densification Strategy

**Problem:** During training, Vec<Gaussian> changes size (split/clone/prune). This invalidates indices for gradients and optimizer state.

**Option A: Rebuild Vec**
```rust
fn densify(gaussians: &mut Vec<Gaussian>, gradients: &GaussianGradients) {
    let mut new = Vec::new();
    for (i, g) in gaussians.iter().enumerate() {
        if should_split(g, gradients, i) {
            new.push(split_1); new.push(split_2);
        } else if !should_prune(g) {
            new.push(g.clone());
        }
    }
    *gaussians = new;
}
```
- ✅ Simple logic
- ❌ Clones all kept Gaussians

**Option B: Mark-and-Sweep (UNDER CONSIDERATION)**
```rust
struct Gaussian {
    // ... params
    alive: bool,
}
// Mark phase, then retain, then append splits
```
- ✅ Clearer logic
- ✅ Append-only splits (predictable)
- ❌ Extra bool per Gaussian

**Open Question:** How should optimizer momentum behave?
1. Reset to zero (simple)?
2. Inherit from parent (complex but maybe more correct)?

---

### Gradient Accumulation for Densification

**Problem:** Densification uses accumulated gradients over 100 iterations.

**Option A: Separate Buffer**
```rust
struct Trainer {
    grad_accumulator: Vec<Vector3<f32>>,  // Position gradient norms
}
```
- ✅ Clean separation
- ❌ Extra allocation

**Option B: In Gaussian Struct**
```rust
struct Gaussian {
    // ... params
    accumulated_grad_norm: f32,
}
```
- ✅ Fast (no extra vec)
- ❌ Mixes training state with data

**Status:** TBD - need to discuss with user.

---

## Development Principles

### Design Before Code
1. **Always discuss ownership and data flow BEFORE implementation**
2. **Understand the trade-offs** - there's rarely one "right" answer
3. **Start simple** - optimize only when profiling shows need
4. **Make trade-offs explicit** - document WHY decisions were made

### Testing Philosophy (Per User Priority #4)
Every component needs **3 types of tests:**

1. **Unit tests** - Verify math correctness
   ```rust
   #[test]
   fn test_quaternion_to_matrix() {
       let q = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
       let R = quaternion_to_matrix(&q);
       // Assert orthogonality, determinant, etc.
   }
   ```

2. **Visual tests** - Export intermediate results for inspection
   ```rust
   #[test]
   fn test_projection_visual() {
       let gaussians = load_test_scene();
       let camera = test_camera();
       let projected = project_all(&gaussians, &camera);
       export_ply("test_output/projected.ply", &projected);
       // Human inspects in MeshLab/Blender
   }
   ```

3. **Gradient tests** - Educational, verify derivatives
   ```rust
   #[test]
   fn test_covariance_projection_gradients() {
       let epsilon = 1e-5;
       for _ in 0..100 {  // Random inputs
           let numerical_grad = finite_difference(f, x, epsilon);
           let analytical_grad = backward_pass(x);
           assert_relative_eq!(numerical_grad, analytical_grad, epsilon=1e-4);
       }
   }
   ```

### Code Style for Learning
- **Clarity over cleverness** - user wants to learn deeply
- **Explain the math** - comments should teach, not just describe
- **Show intermediate steps** - don't collapse multi-step derivations
- **Make gradients explicit** - even if verbose

### User's Rust Level
- **Intermediate** - knows basics, will hit rough patches
- **Teach as we go** - explain non-obvious ownership patterns
- **Use this as learning opportunity** - especially for advanced patterns

---

## Key Technical References

### Papers (Read These First)
1. **3D Gaussian Splatting** - Main paper + supplementary (has gradient derivations!)
2. **SuGaR** - Surface regularization and mesh extraction
3. **EWA Splatting** (Zwicker 2001) - Original splatting math

### Code References (For Algorithms, Not Style)
- Original 3DGS: https://github.com/graphdeco-inria/gaussian-splatting
- Original SuGaR: https://github.com/Anttwo/SuGaR
- gsplat (cleaner CUDA): https://github.com/nerfstudio-project/gsplat

### Specs
- COLMAP binary format: https://colmap.github.io/format.html
- PLY format: http://paulbourke.net/dataformats/ply/

---

## Critical Path to User's Goal

To extract meshes from photos, we need:
1. **M1-M2:** Load COLMAP data ✓
2. **M3-M5:** Forward rendering (see images)
3. **M6:** Gradient checking (THE critical milestone - bugs here are silent killers)
4. **M7-M10:** Training loop (vanilla 3DGS working)
5. **M11:** SuGaR regularization (flat, surface-aligned Gaussians)
6. **M12:** Mesh extraction (THE GOAL!)

**M13-M14 (GPU)** are optional - skip unless CPU training is too slow.

---

## Dependencies (Cargo.toml)

**Required:**
```toml
nalgebra = "0.32"      # Linear algebra (matrices, quaternions, etc.)
image = "0.24"         # PNG/JPEG I/O
rayon = "1.7"          # Parallelization
byteorder = "1.5"      # COLMAP binary parsing
kiddo = "2.0"          # KD-tree for neighbor queries (SuGaR phase)
```

**Optional (GPU feature):**
```toml
[features]
gpu = ["wgpu", "pollster"]

[dependencies]
wgpu = { version = "0.18", optional = true }
pollster = { version = "0.3", optional = true }
```

**Dev (testing):**
```toml
[dev-dependencies]
approx = "0.5"  # Float comparison in tests
```

---

## Next Steps

### Immediate Actions
1. ✅ Design discussion (DONE)
2. ⏳ **Resolve open design questions:**
   - Forward-backward coupling strategy
   - Densification mutation approach
   - Gradient accumulation location
   - Optimizer state inheritance
3. ⏳ **Project setup:**
   - Initialize Cargo workspace
   - Add dependencies
   - Set up module structure from architecture.md
4. ⏳ **M1: COLMAP parsing** (first concrete milestone)

### Guiding Questions for Each Component
Before implementing anything, ask:
1. **Who owns this data?**
2. **What borrows it and when?**
3. **Does this need to be mutable?**
4. **How will this be tested?** (unit/visual/gradient)
5. **What math is this implementing?** (reference the paper)

---

## Important Reminders for Agents

### User Priorities Drive Decisions
- **Learning > Speed** - choose clarity over optimization
- **Correctness > Features** - gradient checking is NON-NEGOTIABLE
- **Tests as Documentation** - every test should teach something

### Red Flags
- ❌ Optimizing before profiling
- ❌ Implementing features beyond current milestone
- ❌ Skipping gradient checking
- ❌ Writing code without discussing ownership first
- ❌ Sacrificing clarity for "cleverness"

### Green Lights
- ✅ Making trade-offs explicit
- ✅ Starting with simple approaches
- ✅ Exporting intermediate results for visual verification
- ✅ Asking "why" before "how"
- ✅ Teaching through implementation

---

## Vocabulary & Key Concepts

**Gaussian Splatting** - Rendering technique representing scene as 3D Gaussians, projected to 2D and alpha-blended

**Covariance** - Σ = R·S·S^T·R^T where S=diag(exp(scale)), R=rotation matrix from quaternion

**Spherical Harmonics (SH)** - View-dependent color representation (16 coefficients for degree-3)

**Densification** - Adaptive Gaussian count control: split large Gaussians with high gradients, clone small ones, prune low-opacity

**SuGaR Regularization** - Flatness + alignment losses to make Gaussians surface-aligned for mesh extraction

**Marching Cubes** - Isosurface extraction from density field (Gaussians → volumetric density → mesh)

---

## Communication with User

### When to Ask Questions
- Any design ambiguity
- Before implementing complex algorithms
- When trade-offs aren't clear
- When about to make irreversible decisions

### When to Move Forward
- Design is clear and documented
- Implementation is straightforward given design
- Tests are obvious
- Just need to write the code

### Tone
- Technical and precise
- Educational (user wants to learn)
- Honest about trade-offs
- Collaborative (this is a partnership)

---

**End of Agent Context - Keep this updated as project evolves!**

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

