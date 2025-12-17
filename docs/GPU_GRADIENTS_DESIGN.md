# GPU Gradients Design Document

**Goal:** Implement GPU-accelerated backward pass for 100-200x training speedup

**Current Performance:**
- Forward: 12ms (GPU) ✅
- Backward: 590ms (CPU with rayon) ❌
- Total: ~602ms/iter

**Target Performance:**
- Forward: 12ms (GPU) ✅
- Backward: ~10-20ms (GPU) 🎯
- Total: ~25-35ms/iter ≈ **20x speedup!**

---

## Architecture Overview

### Current CPU Backward Pass (590ms)

```
For each pixel (parallel with rayon):
  1. Gather contributing Gaussians
  2. Forward pass: alpha blending → pixel color
  3. Backward pass: chain rule → per-Gaussian gradients
  4. Accumulate into thread-local buffers

Final reduction: merge thread-local gradients → per-Gaussian gradients
```

### Proposed GPU Backward Pass (~10-20ms)

```
GPU Shader (per-pixel workgroup):
  1. Gather contributing Gaussians (from sorted list)
  2. Forward pass: alpha blending → pixel color
  3. Backward pass: chain rule → per-Gaussian gradients
  4. Accumulate into workgroup-local buffers

CPU Reduction:
  Download workgroup buffers → reduce → final gradients
```

---

## Key Challenges & Solutions

### Challenge 1: Gradient Accumulation

**Problem:** WGSL only has atomic operations for i32/u32, not f32. Multiple pixels may write to the same Gaussian's gradient simultaneously.

**Solutions:**

**Option A: Atomic CAS with Bitcasting** ⚠️ Complex
```wgsl
// Atomically add to f32 using compare-and-swap
atomicCompareExchangeWeak(&gradient_buffer[i], old_bits, new_bits)
```
- Pros: True atomic f32 accumulation
- Cons: Complex, many CAS retries = slow

**Option B: Per-Workgroup Buffers + CPU Reduction** ✅ Recommended
```wgsl
// Each workgroup writes to its own gradient buffer
workgroup_gradients[workgroup_id][gaussian_id] += gradient
```
- Pros: Simple, no contention, still very fast
- Cons: Need CPU reduction step (but fast - only ~100 workgroups)

**Option C: Fixed-Point Atomic Integers**
```wgsl
// Convert f32 → i32 with fixed-point scaling
atomicAdd(&gradient_i32[i], gradient_as_i32)
```
- Pros: True atomics
- Cons: Precision loss, range limits

**Decision: Option B** - Simplest, proven to work, CPU reduction is negligible

---

## Implementation Plan

### Phase 1: Forward Pass with Intermediate Storage

**Goal:** Modify GPU renderer to save data needed for backward pass

**What to store:**
- Transmittances (T_i for each Gaussian per pixel)
- Alphas (α_i for each Gaussian per pixel)
- Gaussian indices (which Gaussians contribute to each pixel)
- Pixel colors (for validation)

**Memory estimate (245×136 image, 2K Gaussians):**
- Worst case: 10 Gaussians/pixel × 33K pixels = 330K entries
- Per entry: 4 bytes (transmittance) + 4 bytes (alpha) + 4 bytes (index) = 12 bytes
- Total: ~4MB (acceptable!)

**Implementation:**
```rust
struct ForwardIntermediates {
    transmittances: Vec<f32>,  // Per-contribution
    alphas: Vec<f32>,          // Per-contribution
    indices: Vec<u32>,         // Per-contribution
    offsets: Vec<u32>,         // Per-pixel offset into above arrays
    num_contributions: Vec<u32>, // Per-pixel count
}
```

### Phase 2: WGSL Backward Shader

**Shader structure:**
```wgsl
@group(0) @binding(0) var<storage, read> intermediates: ForwardIntermediates;
@group(0) @binding(1) var<storage, read> d_pixels: array<vec4<f32>>;  // Upstream gradients
@group(0) @binding(2) var<storage, read_write> workgroup_gradients: array<Gradients>;

struct Gradients {
    d_color: vec3<f32>,
    d_opacity: f32,
    d_mean_px: vec2<f32>,
    d_cov_2d: vec3<f32>,
}

@compute @workgroup_size(16, 16, 1)
fn backward_pass(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let pixel_idx = global_id.y * width + global_id.x;
    let upstream = d_pixels[pixel_idx];

    // 1. Run alpha blending backward (same as CPU)
    let offset = intermediates.offsets[pixel_idx];
    let count = intermediates.num_contributions[pixel_idx];

    // 2. Accumulate gradients for this pixel's Gaussians
    for (var i = 0u; i < count; i++) {
        let contrib_idx = offset + i;
        let gaussian_idx = intermediates.indices[contrib_idx];
        let alpha = intermediates.alphas[contrib_idx];
        let t = intermediates.transmittances[contrib_idx];

        // Blend backward math (same as CPU blend_backward_with_bg)
        let d_alpha = /* ... */;
        let d_color = upstream * (t * alpha);

        // Write to workgroup-local buffer
        let wg_offset = workgroup_id.y * num_workgroups_x + workgroup_id.x;
        let grad_idx = wg_offset * max_gaussians + gaussian_idx;
        workgroup_gradients[grad_idx].d_color += d_color;
        workgroup_gradients[grad_idx].d_opacity += d_alpha;
        // ... etc
    }
}
```

### Phase 3: CPU Reduction

**After GPU shader completes:**
```rust
// Download workgroup gradients
let workgroup_grads: Vec<Gradients> = read_buffer(&workgroup_buffer);

// Reduce: sum across all workgroups for each Gaussian
let mut final_grads = vec![Gradients::zero(); num_gaussians];
for wg_grad in workgroup_grads.chunks(max_gaussians) {
    for (i, grad) in wg_grad.iter().enumerate() {
        final_grads[i] += *grad;
    }
}
```

**Performance estimate:**
- 100 workgroups × 10K Gaussians = 1M gradient entries
- Reduction: ~1-2ms on CPU (trivial)

### Phase 4: Integration

**Modify GpuRenderer:**
```rust
pub struct GpuRenderer {
    // ... existing fields
    backward_pipeline: ComputePipeline,
    backward_bind_group_layout: BindGroupLayout,
}

impl GpuRenderer {
    pub fn render_with_gradients(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
        d_pixels: &[Vector3<f32>],  // Upstream gradients
    ) -> (
        Vec<Vector3<f32>>,  // Rendered pixels
        GradientBuffers,     // All gradients
    ) {
        // 1. Forward pass (already implemented, but save intermediates)
        let (pixels, intermediates) = self.render_forward_with_intermediates(...);

        // 2. Backward pass (GPU shader)
        let workgroup_gradients = self.run_backward_shader(intermediates, d_pixels);

        // 3. Reduction (CPU)
        let final_gradients = reduce_gradients(workgroup_gradients, num_gaussians);

        (pixels, final_gradients)
    }
}
```

---

## Memory Layout

### Forward Intermediates Buffer

**Sparse storage (variable-length per pixel):**

```
Pixel 0: [T0, α0, idx0] [T1, α1, idx1] ...
Pixel 1: [T0, α0, idx0] [T1, α1, idx1] [T2, α2, idx2] ...
...

offsets[0] = 0
offsets[1] = 2  (pixel 0 had 2 contributions)
offsets[2] = 5  (pixel 1 had 3 contributions)
```

**Buffer structure:**
```rust
struct IntermediatesBuffer {
    data: [f32],        // Packed: [T, α, idx, T, α, idx, ...]
    offsets: [u32],     // Per-pixel offset
    counts: [u32],      // Per-pixel contribution count
}
```

### Workgroup Gradients Buffer

**Layout:**
```
Workgroup 0: [Grad for G0, Grad for G1, ..., Grad for G_max]
Workgroup 1: [Grad for G0, Grad for G1, ..., Grad for G_max]
...
```

**Size:** num_workgroups × max_gaussians × sizeof(Gradients)
- Example: 100 workgroups × 10K Gaussians × 48 bytes/grad = 48MB
- Acceptable for modern GPUs (we have 8-16GB)

---

## Performance Analysis

### Expected Speedup

**Current CPU backward (590ms):**
- Projection: ~50ms
- Per-pixel gradient: ~500ms ← Target for GPU
- Reduction: ~40ms

**GPU backward (~10-20ms):**
- Projection: ~5ms (already on GPU)
- Per-pixel gradient: ~5-10ms (GPU shader)
- Reduction: ~1-2ms (CPU)
- GPU→CPU transfer: ~3-5ms

**Total speedup:** 590ms → 15ms = **39x faster backward!**

**Overall training speedup:**
- Current: 602ms/iter (forward 12ms + backward 590ms)
- Target: 27ms/iter (forward 12ms + backward 15ms)
- **Speedup: 22x total training!**

---

## Validation Strategy

### Correctness Checks

1. **Gradient check against CPU:**
   ```rust
   let cpu_grads = render_full_color_grads_cpu(...);
   let gpu_grads = render_with_gradients_gpu(...);

   for i in 0..num_gaussians {
       assert_relative_eq!(cpu_grads.d_colors[i], gpu_grads.d_colors[i], epsilon=1e-4);
   }
   ```

2. **Finite difference check:**
   ```rust
   // Perturb Gaussian color slightly
   let epsilon = 1e-4;
   let loss_plus = render_and_loss(..., color + epsilon);
   let loss_minus = render_and_loss(..., color - epsilon);
   let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);

   assert_relative_eq!(numerical_grad, gpu_grad, epsilon=1e-3);
   ```

3. **Training convergence:**
   ```rust
   // Train same scene with CPU vs GPU gradients
   // Final PSNR should be within 0.1 dB
   ```

---

## Implementation Timeline

**Day 1: Forward Pass Modifications** (4-6 hours)
- Modify GPU renderer to save intermediates
- Design buffer layout
- Test forward pass still works

**Day 2: WGSL Backward Shader** (6-8 hours)
- Implement alpha blending backward in WGSL
- Implement 2D Gaussian gradient in WGSL
- Workgroup gradient accumulation

**Day 3: CPU Reduction & Integration** (4-6 hours)
- Implement gradient buffer download
- Implement CPU reduction
- Integrate with trainer

**Day 4: Validation & Debugging** (4-6 hours)
- Gradient checks vs CPU
- Fix numerical issues
- Convergence testing

**Day 5: Performance Tuning** (2-4 hours)
- Optimize buffer sizes
- Benchmark on T&T dataset
- Document performance

**Total: 4-5 days** for complete GPU gradients implementation

---

## Risks & Mitigation

### Risk 1: Numerical Precision
- **Problem:** f32 precision may cause gradient errors
- **Mitigation:** Use relative epsilon checks (1e-4), validate on training convergence

### Risk 2: Memory Overflow
- **Problem:** Large scenes may exceed GPU memory
- **Mitigation:** Implement tiling (process image in chunks)

### Risk 3: Complexity
- **Problem:** WGSL shader code is complex and hard to debug
- **Mitigation:** Start simple (color gradients only), add incrementally

### Risk 4: Slower than Expected
- **Problem:** GPU overhead may reduce speedup
- **Mitigation:** Profile with timestamps, optimize hot paths

---

## Success Criteria

✅ **Correctness:**
- GPU gradients match CPU gradients (within 1e-4 relative error)
- Training converges to same PSNR as CPU (within 0.1 dB)

✅ **Performance:**
- Backward pass: < 50ms (10x faster than current 590ms)
- Overall training: < 100ms/iter (6x faster than current 602ms)

✅ **Stretch Goal:**
- Backward pass: < 20ms (30x faster)
- Overall training: < 35ms/iter (17x faster)

---

## Next Steps

1. ✅ Read and understand current CPU backward pass
2. ✅ Design GPU architecture
3. ⏸️ Implement forward pass with intermediates
4. ⏸️ Implement WGSL backward shader
5. ⏸️ Implement CPU reduction
6. ⏸️ Validate and benchmark

Ready to proceed with implementation!
