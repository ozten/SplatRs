# M12: GPU Training - IN PROGRESS

**Milestone Goal:** GPU training end-to-end with 10-50x speedup over CPU.

**Status:** 🟡 **PARTIAL COMPLETION** - Forward pass GPU-accelerated, backward pass CPU-bound

**Date Started:** December 16, 2025

---

## What's Complete ✅

### 1. GPU Forward Pass Integration

**Single-Image Trainer (`train_single_image_color_only`)**
- ✅ GPU renderer initialization with `--gpu` flag
- ✅ Conditional rendering (GPU if available, CPU fallback)
- ✅ Feature-gated compilation (`#[cfg(feature = "gpu")]`)

**Multi-View Trainer (`train_multiview_color_only`)**
- ✅ GPU renderer initialization
- ✅ All `render_full_linear()` calls use GPU-aware rendering
- ✅ Tested on T&T dataset and calipers dataset

### 2. CLI Integration

**Added:**
- `--gpu` flag to `sugar-train` binary
- `use_gpu` field in `TrainConfig` and `MultiViewTrainConfig`
- Automatic error when GPU requested but not compiled with `--features gpu`

---

## Performance Results 📊

### M7 Single-Image Training (20K Gaussians, 270×480)

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Forward | 4.4s | 0.05s | **88x** |
| Backward | 5.4s | 5.6s | 1.0x |
| **Total** | **9.9s/iter** | **5.65s/iter** | **1.75x** |

**Key Finding:** GPU forward pass is extremely fast, but backward pass dominates total time.

### T&T Multi-View Training (10K Gaussians, 490×273)

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Forward | 2.18s | 0.83s | **2.6x** |
| Backward | 2.2s | 2.3s | 1.0x |
| **Total** | **4.4s/iter** | **3.1s/iter** | **1.4x** |

### M8-Smoke Multi-View (2K Gaussians, 135×240)

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Forward | 140ms | 50ms | **2.8x** |
| Backward | 155ms | 165ms | 0.9x |
| **Total** | **296ms/iter** | **215ms/iter** | **1.4x** |

---

## Analysis 🔍

### Forward Pass Speedup Variance

The GPU forward speedup varies significantly by configuration:
- **M7 (20K Gaussians)**: 88x speedup
- **T&T (10K Gaussians)**: 2.6x speedup
- **M8-smoke (2K Gaussians)**: 2.8x speedup

**Why the variance?**
1. **M7 uses GPU optimally**: Large batch (20K Gaussians), no view switching overhead
2. **Multi-view has overhead**: View loading, camera setup changes between iterations
3. **Smaller scenes see less benefit**: GPU has fixed overhead that dominates with fewer Gaussians

**Absolute GPU forward times:**
- M7: 50ms (20K Gaussians, 270×480)
- T&T: 830ms (10K Gaussians, 490×273)
- M8-smoke: 50ms (2K Gaussians, 135×240)

The T&T forward time (830ms) is surprisingly slow despite having fewer Gaussians than M7. This suggests **additional overhead in multi-view training** (view caching, preloading, etc.).

### Backward Pass Bottleneck

The backward pass (gradient computation) is now the **critical bottleneck**:
- M7: 5.6s backward vs 0.05s forward (112x slower!)
- T&T: 2.3s backward vs 0.83s forward (2.8x slower)

To achieve the M12 goal of 10-50x overall speedup, **GPU gradients are essential**.

---

## What's Left for M12 🎯

### Critical Path: GPU Backward Pass

**Estimated Complexity:** HIGH (3-5 days of focused work)

**Requirements:**
1. **WGSL gradient shaders**
   - Projection gradients (world → camera → pixel)
   - Rasterization gradients (per-pixel alpha blending)
   - SH gradient accumulation

2. **Memory management**
   - Store intermediate values during forward pass
   - Reuse for backward pass (reduce computation)

3. **Gradient buffers**
   - Per-Gaussian gradients (color, opacity, position, scale, rotation, SH)
   - Efficient GPU→CPU transfer for optimizer step

4. **Integration with existing trainers**
   - Replace `render_full_color_grads()` calls
   - Maintain API compatibility

### Secondary Optimizations

**GPU Sorting** (Medium priority)
- Currently sorting 2D Gaussians on CPU (bottleneck in M11 analysis)
- Implement GPU radix sort or bitonic sort
- Eliminate CPU↔GPU transfers
- **Estimated impact:** 2-3x speedup

**Tile-Based Rasterization** (Lower priority)
- Current naive implementation: O(N × W × H)
- Tile-based: O(N × tiles × Gaussians_per_tile)
- **Estimated impact:** 5-10x speedup on rasterization
- **Note:** Less critical if backward pass is the bottleneck

---

## Lessons Learned 💡

### What Worked Well

1. **Incremental approach**: Forward pass first, then backward pass
2. **Conditional rendering**: Clean abstraction with feature gates
3. **Test-driven**: Verified correctness on multiple datasets
4. **Parallel development**: Fixed M11 bug while implementing M12

### Unexpected Findings

1. **Multi-view overhead**: GPU forward pass slower than expected in multi-view scenarios
2. **Backward dominance**: Even 88x forward speedup only gives 1.75x overall (need GPU gradients!)
3. **Dataset-dependent speedup**: Performance varies 2-88x depending on configuration

### Technical Debt

1. **No profiling yet**: Need detailed timing breakdown (projection vs rasterization)
2. **CPU sorting**: Still transferring data GPU→CPU→GPU for depth sorting
3. **Memory management**: Creating new buffers every frame (should pool)
4. **Error handling**: GPU errors not gracefully handled

---

## Usage Examples

### Single-Image Training with GPU
```bash
cargo run --release --features gpu --bin sugar-train -- \
  --preset m7 \
  --dataset-root /path/to/dataset \
  --gpu \
  --iters 1000 \
  --out-dir output/
```

### Multi-View Training with GPU
```bash
cargo run --release --features gpu --bin sugar-train -- \
  --preset m9 \
  --dataset-root datasets/tandt_db/tandt/train \
  --gpu \
  --iters 200 \
  --densify-interval 25 \
  --out-dir output/
```

### CPU-Only (No GPU flag)
```bash
cargo run --release --bin sugar-train -- \
  --preset m7 \
  --dataset-root /path/to/dataset \
  --iters 1000
```

---

## Next Steps

### Immediate (To complete M12 forward-only milestone)

1. **Profile GPU rendering**
   - Add timing instrumentation for projection vs rasterization
   - Identify bottlenecks in multi-view scenario
   - Understand why T&T forward is 830ms vs M7's 50ms

2. **Optimize multi-view GPU path**
   - Investigate view caching overhead
   - Consider GPU-side caching of camera transforms
   - Profile preloading vs on-demand loading

### Future (Full M12: GPU gradients)

1. **Design gradient computation architecture**
   - Decide: single monolithic shader vs separate shaders
   - Memory layout for intermediate values
   - API for gradient accumulation

2. **Implement projection gradients**
   - d(loss)/d(position), d(loss)/d(rotation), d(loss)/d(scale)
   - Covariance gradient computation
   - Camera space transformations

3. **Implement rasterization gradients**
   - d(loss)/d(color), d(loss)/d(opacity)
   - Alpha blending gradient reversal
   - Per-pixel gradient accumulation

4. **Integration testing**
   - Compare GPU gradients vs CPU gradients (correctness)
   - Gradient checks for all parameters
   - End-to-end training validation

---

## Timeline Estimate

**Current Status:** M12 is ~30% complete
- Forward pass GPU: ✅ Done
- Backward pass GPU: ❌ Not started
- Full integration: ⏸️ Pending GPU gradients

**To reach 10-50x speedup:**
- Need GPU gradients (backward pass)
- Estimated: 3-5 days focused implementation
- Complexity: High (shader programming, gradient math, debugging)

**Alternative Approach:**
- Declare M12 "forward-only" complete (current state)
- Create M12b for GPU gradients
- Advantages: Working GPU renderer now, gradients as separate milestone

---

## Hardware & Software

**Tested on:**
- Device: Apple M2 Max
- Backend: Metal (via wgpu)
- OS: macOS 14.x
- Rust: 1.75+
- wgpu: 0.19.x

**Datasets:**
- T&T Train (10K Gaussians, 490×273, 10 views)
- Digital Calipers (20K Gaussians, 270×480, 87 views)

---

## Conclusion

M12 forward pass GPU-acceleration is **working and validated** on multiple datasets. The 1.4-1.75x overall speedup is meaningful but falls short of the 10-50x goal. To achieve that target, **GPU-accelerated gradient computation** is essential.

The foundation is solid, and the forward pass speedup proves the GPU pipeline is working correctly. The next phase (GPU gradients) is technically challenging but builds on this proven infrastructure.

**Recommendation:** Consider this a successful incremental milestone. GPU forward pass provides immediate value, and gradients can be added iteratively.
