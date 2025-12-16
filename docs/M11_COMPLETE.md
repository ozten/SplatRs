# M11: GPU Renderer - COMPLETE ✅

**Milestone Goal:** GPU renderer produces identical images to CPU renderer (within floating-point tolerance).

**Status:** ✅ **COMPLETE**

**Date:** December 16, 2025

---

## Achievement Summary

Successfully implemented and validated GPU-accelerated Gaussian splatting renderer using wgpu (WebGPU) on Apple M2 Max (Metal backend).

### Test Results

```
GPU: Apple M2 Max (Metal)
GPU vs CPU Rendering Comparison:
  Pixels:            4096 (64×64 test image)
  Average diff:      0.000001
  Max diff:          0.000204
  Diffs > 1e-4:      15 (0.4%)

✅ GPU rendering matches CPU within tolerance!
```

**Validation:** Per-pixel difference < 1e-3 (M11 requirement met)

---

## Implementation Details

### Architecture

**Two-stage compute pipeline:**

1. **Projection Stage** (3D → 2D)
   - WGSL compute shader
   - Parallel per-Gaussian (256 threads/workgroup)
   - Transforms: world → camera → pixel space
   - Covariance projection via perspective Jacobian
   - SH evaluation (DC term)

2. **Rasterization Stage** (2D → pixels)
   - WGSL compute shader
   - Parallel per-pixel (16×16 workgroups)
   - 2D Gaussian evaluation
   - Alpha blending in depth order
   - Background color support

### Components Added

**Source files:**
- `src/gpu/types.rs` - GPU-friendly data structures (GaussianGPU, Gaussian2DGPU, CameraGPU)
- `src/gpu/shaders.rs` - Projection shader + shader compilation
- `src/gpu/rasterize.wgsl` - Rasterization compute shader
- `src/gpu/renderer.rs` - High-level GpuRenderer interface
- `src/gpu/context.rs` - wgpu device/queue initialization
- `src/gpu/buffers.rs` - Buffer management utilities

**Test files:**
- `tests/m11_gpu_renderer.rs` - GPU vs CPU validation tests

### Key Design Decisions

1. **Compute-only approach** - No graphics pipeline, pure compute shaders
   - Easier to debug
   - Matches training needs (M12)
   - Flexible for optimization

2. **CPU sorting (for now)** - Depth sorting on CPU between stages
   - Simplifies initial implementation
   - Will move to GPU in M12
   - Current bottleneck for performance

3. **Naive rasterization** - Per-pixel loop through all Gaussians
   - Correct but slow (O(N × W × H))
   - Tile-based optimization in M12

---

## Performance Analysis

### Current Performance (Naive Implementation)

**Not yet measured** - Correctness was priority for M11.

**Expected profile:**
- ✅ Projection: Fast (GPU-parallel)
- ⚠️ Sorting: Slow (CPU roundtrip)
- ⚠️ Rasterization: Moderate (GPU but naive algorithm)

### Bottlenecks Identified

1. **CPU sorting** - Full GPU→CPU→GPU roundtrip
   - ~20K Gaussians × 80 bytes/Gaussian = 1.6MB transfer
   - Each direction

2. **Naive rasterization** - No spatial culling
   - Every pixel checks every Gaussian
   - 256×256 × 20K = 1.3B operations

3. **No tile-based culling** - Classic 3DGS optimization missing
   - Tile assignment would reduce work dramatically

---

## M11 Verification Checklist

- [x] GPU renderer produces images matching CPU
- [x] Per-pixel difference < 1e-4 for most pixels
- [x] Max difference < 1e-3 threshold
- [x] Works on real dataset (T&T/calipers)
- [x] Edge cases handled correctly
  - [x] Gaussians behind camera (culled)
  - [x] Gaussians at image boundary
- [x] Test harness in place

---

## Next Steps: M12 Optimizations

### Priority 1: Remove CPU Sorting Bottleneck
- Implement GPU radix sort or bitonic sort
- Keep Gaussians on GPU between stages
- Eliminate CPU↔GPU transfers

### Priority 2: Tile-Based Rasterization
- Divide screen into 16×16 tiles
- Assign Gaussians to overlapping tiles
- Per-tile sorted lists
- Massive culling savings

### Priority 3: Performance Measurement
- Add timing instrumentation
- Measure projection vs sorting vs rasterization time
- Establish baseline for optimization targets

### Target for M12
- **10-50x speedup vs CPU**
- CPU baseline: ~9 sec/iter @ 20K Gaussians
- GPU target: ~0.2-0.5 sec/iter

---

## Lessons Learned

### What Worked Well ✅

1. **Incremental approach** - Built infrastructure first, then rendering
2. **Test-driven** - Validated correctness before optimizing
3. **WGSL readability** - Shader code is clear and maintainable
4. **bytemuck for data** - GPU data structures are clean

### Challenges Overcome 🔧

1. **wgpu API learning curve** - Bind groups, pipeline layouts
2. **WGSL matrix conventions** - Row-major vs column-major
3. **Floating-point precision** - GPU/CPU slight differences (expected)

### Technical Debt 📝

1. **No error handling** - Expects GPU always succeeds
2. **Fixed workgroup sizes** - Should be configurable
3. **SH evaluation** - Only DC term implemented
4. **No memory pooling** - Creates new buffers every frame

---

## Hardware & Software

**Tested on:**
- Device: Apple M2 Max
- Backend: Metal (via wgpu)
- OS: macOS 14.x
- wgpu: 0.19.x

**Dependencies:**
- wgpu: GPU API abstraction
- pollster: Async runtime for blocking GPU ops
- bytemuck: Zero-copy GPU data transfer
- futures: Async utilities

---

## Milestone Complete! 🎉

M11 validates that our GPU implementation is **mathematically correct**.

The foundation is solid - now we can optimize for speed in M12 with confidence that the output will remain correct.

**Time to completion:** ~1 day focused work (as estimated)

**Next milestone:** M12 - GPU Training End-to-End
