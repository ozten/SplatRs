# M11: GPU Renderer Design

**Goal:** Port forward rendering to GPU for 10-50x speedup, matching CPU output exactly.

---

## Current CPU Algorithm (from `render_full_linear`)

```rust
1. Project all 3D Gaussians → 2D (parallel-friendly)
2. Sort by depth (serial bottleneck)
3. For each pixel (height × width):
     For each Gaussian:
       - Check bounding box
       - Evaluate 2D Gaussian
       - Alpha blend
```

**Performance:** ~9 sec/iter @ 20K Gaussians, 256×256 image
- Projection: Fast (parallelizable)
- Sorting: Moderate (20K log 20K)
- Per-pixel blending: **SLOW** (serial, 256×256 × 20K checks = 1.3B operations)

---

## GPU Strategy: Tile-Based Rasterization

Follow the [3DGS paper approach](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/):

### Phase 1: Preprocessing (GPU Compute)
1. **Project Gaussians** (parallel per-Gaussian)
   - Input: Gaussian positions, scales, rotations, camera
   - Output: 2D means, covariances, colors, depths

2. **Cull & Assign to Tiles** (parallel per-Gaussian)
   - Divide screen into 16×16 tiles
   - Compute which tiles each Gaussian overlaps
   - Output: List of (tile_id, gaussian_id, depth) tuples

3. **Sort per Tile** (GPU radix sort)
   - Sort gaussians within each tile by depth
   - Can use wgpu's compute shaders + bitonic/radix sort

### Phase 2: Rasterization (GPU Fragment/Compute)
4. **Render Tiles** (parallel per-tile)
   - Each work group handles one tile (16×16 pixels)
   - Load sorted Gaussian list for this tile
   - Blend in depth order
   - Write output pixels

---

## Implementation Plan (Incremental)

### Step 1: Basic wgpu Setup (1-2 hours)
- Initialize wgpu device & queue
- Create GPU buffer abstractions
- Basic "hello triangle" style test

### Step 2: Naive GPU Renderer (3-4 hours)
- Port projection to compute shader
- Simple per-pixel loop (no tiling yet)
- Verify output matches CPU

### Step 3: Tile-Based Renderer (8-10 hours)
- Implement tile assignment
- GPU sorting (or CPU fallback initially)
- Tile-parallel rasterization
- Measure speedup

### Step 4: Validation & Testing (2-3 hours)
- Test on multiple scenes
- Edge cases (Gaussians behind camera, at boundaries)
- Per-pixel difference < 1e-4 threshold

---

## Technical Choices

### Language: WGSL (WebGPU Shading Language)
- Native to wgpu
- Simpler than GLSL/HLSL
- Good Rust tooling

### Architecture: Compute-Heavy
- Use compute shaders for projection, sorting, rasterization
- Avoid graphics pipeline complexity initially
- Easier to debug and matches training needs (M12)

### Data Layout
```rust
// CPU → GPU
struct GaussianGPU {
    position: vec3<f32>,    // World space
    scale: vec3<f32>,       // Log-space
    rotation: vec4<f32>,    // Quaternion
    sh_coeffs: array<vec3<f32>, 16>,  // SH coefficients
    opacity: f32,           // Logit space
}

// GPU intermediate
struct Gaussian2DGPU {
    mean: vec2<f32>,        // Pixel space
    cov: vec3<f32>,         // (xx, xy, yy)
    color: vec3<f32>,       // Linear RGB
    opacity: f32,
    depth: f32,
    gaussian_idx: u32,
}
```

### Memory Budget
- 20K Gaussians × 256 bytes = 5MB (manageable)
- Double-buffering for ping-pong = 10MB
- Frame buffer (1024×1024×4×4) = 16MB
- Total: ~30-50MB (well within GPU limits)

---

## Risks & Mitigations

### Risk 1: Sorting on GPU is Hard
**Mitigation:** Start with CPU sorting, port to GPU later
- Still get speedup from parallel projection + rasterization
- Sorting is O(N log N), rasterization is O(N × W × H)

### Risk 2: Floating-Point Precision Differences
**Mitigation:** Careful testing with tolerance
- GPU uses same IEEE 754 as CPU
- Expect differences ~ 1e-6, accept < 1e-4

### Risk 3: wgpu Learning Curve
**Mitigation:** Start simple, iterate
- Begin with compute shaders only (no graphics pipeline)
- Port one function at a time
- Use CPU reference for validation

---

## Success Criteria (M11)

✅ GPU renderer produces images matching CPU:
- Per-pixel RMSE < 1e-4
- Works on calipers dataset
- Works on T&T dataset
- Edge cases handled correctly

🎯 Bonus (nice-to-have for M11):
- 5-10x speedup vs CPU (full tile-based rasterization is M12 optimization)
- Configurable backend (CPU/GPU selectable at runtime)

---

## References

- [3DGS Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [wgpu Tutorial](https://sotrh.github.io/learn-wgpu/)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)
- [GPU Gems: Tile-Based Rendering](https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-37-octree-textures-gpu)

---

## Timeline Estimate

| Task | Time | Cumulative |
|------|------|------------|
| wgpu setup + hello world | 2 hours | 2h |
| Port projection shader | 3 hours | 5h |
| Naive per-pixel rasterizer | 4 hours | 9h |
| Validation & testing | 3 hours | 12h |
| **M11 Complete** | | **~12 hours** |
| (M12) Tile-based optimization | 8 hours | 20h |
| (M12) GPU sorting | 4 hours | 24h |

**Realistic:** 2-3 days focused work for M11.
