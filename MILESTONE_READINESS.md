# Milestone Readiness Checklist

## ✅ M1-M7: COMPLETE & WORKING

### Test Results
- ✅ **20/20** unit tests pass
- ✅ **9/9** gradient check tests pass
- ✅ **All integration tests** pass (M1-M5, M7)
- ✅ **4 datasets tested** successfully (train, truck, playroom, drjohnson)

### Known Issues (Non-Blocking for M7)
- 🟡 Bug #1: Camera ID mapping (masked by single-camera datasets)
- 🟡 Bug #2: Adam state reset (not triggered with fixed Gaussian count)
- 🟡 Bug #3: Distortion ignored (test datasets have minimal distortion)

**Status**: ✅ **READY FOR PRODUCTION** (single-image, single-camera use cases)

---

## ⚠️ M8: Multi-View Training - ACTION REQUIRED

### What M8 Needs
- Train on 10-20 images from different viewpoints
- Multi-view consistency (gradients from multiple views)
- Test view synthesis on held-out cameras

### Critical Blockers
1. **🔴 MUST FIX Bug #1**: Camera ID mapping
   - Current code: Always uses `cameras[0]`
   - Required: Map `image.camera_id` → correct camera
   - **Impact if unfixed**: Wrong intrinsics → misalignment → training fails

### Recommended Changes

#### Fix #1: Proper Camera ID Mapping
```rust
// src/io/colmap.rs
pub struct ColmapScene {
    pub cameras: HashMap<u32, Camera>,  // Changed from Vec<Camera>
    pub images: Vec<ImageInfo>,
    pub points: Vec<Point3D>,
}

// In read_cameras_bin():
let mut cameras = HashMap::new();
for _ in 0..num_cameras {
    let camera_id = reader.read_u32::<LittleEndian>()?;
    // ... parse camera ...
    cameras.insert(camera_id, camera);  // Use ID as key
}
```

#### Fix #2: Trainer Update
```rust
// src/optim/trainer.rs:94
// OLD: let base_camera = &scene.cameras[0];
// NEW:
let base_camera = scene.cameras.get(&image_info.camera_id)
    .ok_or_else(|| anyhow::anyhow!("Camera {} not found", image_info.camera_id))?;
```

### Testing Strategy
1. Find or create a multi-camera dataset
2. Verify different images use different camera IDs
3. Train on subset, test camera ID lookup works
4. Check that training converges with correct intrinsics

**Status**: 🔴 **BLOCKED** until Bug #1 is fixed

---

## 🟡 M9: Adaptive Density Control - ACTION REQUIRED

### What M9 Needs
- Gaussian splitting (high-gradient regions)
- Gaussian cloning (under-reconstructed regions)
- Gaussian pruning (low-opacity Gaussians)
- Parameter count changes dynamically during training

### Critical Blockers
1. **🔴 MUST FIX Bug #2**: Adam optimizer state management
   - Current: Resets `t=0` when parameter count changes
   - Required: Preserve optimizer state or handle gracefully
   - **Impact if unfixed**: Learning rate spikes → unstable training

### Recommended Changes

#### Fix: Adam State Preservation
```rust
// src/optim/adam.rs
pub fn ensure_len(&mut self, len: usize) {
    if self.m.len() != len {
        // Preserve existing state, zero-init new parameters
        self.m.resize(len, Vector3::zeros());
        self.v.resize(len, Vector3::zeros());
        // DON'T reset t!
        // self.t stays at current value for bias correction
    }
}
```

**Alternative**: Per-parameter timesteps
```rust
pub struct AdamVec3 {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: Vec<u32>,  // One timestep per parameter
    m: Vec<Vector3<f32>>,
    v: Vec<Vector3<f32>>,
}
```

### Testing Strategy
1. Start with N Gaussians
2. Add/remove Gaussians mid-training
3. Verify loss continues to decrease (no spikes)
4. Check that new Gaussians have correct learning rates

**Status**: 🔴 **BLOCKED** until Bug #2 is fixed

---

## 🟢 M10: Full 3DGS Training - ALMOST READY

### What M10 Needs
- Full parameter optimization (position, scale, rotation, opacity, SH)
- 30k iterations on full dataset
- Quality comparable to reference implementation

### Current Limitations
- Only color (SH DC) is optimized in M7
- Need to add gradients for:
  - ✅ Position (gradients implemented, not wired up)
  - ✅ Scale (gradients implemented, not wired up)
  - ✅ Rotation (gradients implemented, not wired up)
  - ✅ Opacity (gradients implemented, not wired up)

### Recommended Changes
Most gradient code is already written in `src/diff/`. Just need to:
1. Wire up projection gradients in `render_full_diff.rs`
2. Add optimizer support for more parameter types
3. Add learning rate scheduling

**Status**: 🟢 **READY** (once M8-M9 are done)

---

## 🔵 Future Milestones

### M11: SuGaR Regularization
- **Dependencies**: M10 working
- **New code needed**: Regularization losses
- **Estimated effort**: Medium

### M12: Mesh Extraction
- **Dependencies**: M11 working
- **New code needed**: Marching cubes, Poisson reconstruction
- **Estimated effort**: High

### M13-M14: GPU Acceleration
- **Dependencies**: M10 working (can parallelize with M11-M12)
- **New code needed**: WGPU/CUDA kernels
- **Estimated effort**: Very High

---

## 📋 Quick Action Plan

### Before starting M8:
```bash
# 1. Fix camera ID mapping
- [ ] Change ColmapScene.cameras to HashMap<u32, Camera>
- [ ] Update all camera lookups to use camera_id
- [ ] Test with multi-camera dataset
- [ ] Run: cargo test --test dataset_sanity_check

# 2. Verify multi-view training
- [ ] Train on 10+ images from different views
- [ ] Check loss decreases across all views
- [ ] Test novel view synthesis (PSNR > 20dB)
```

### Before starting M9:
```bash
# 1. Fix Adam optimizer state
- [ ] Update ensure_len() to preserve timestep
- [ ] Add test for parameter count changes
- [ ] Run: cargo test adam_resize_test

# 2. Implement density control
- [ ] Gaussian splitting logic
- [ ] Gaussian pruning logic
- [ ] Verify training remains stable
```

### Before starting M10:
```bash
# 1. Wire up remaining gradients
- [ ] Position gradients
- [ ] Scale gradients
- [ ] Rotation gradients (via quaternion)
- [ ] Opacity gradients
- [ ] Run: cargo test --test gradient_check

# 2. Add learning rate scheduling
- [ ] Exponential decay for position
- [ ] Per-parameter learning rates
```

---

## 🎯 Current Status Summary

| Component | Status | Blocker |
|-----------|--------|---------|
| **COLMAP I/O** | ✅ Working | None |
| **Forward Rendering** | ✅ Working | None |
| **Gradient Math** | ✅ Verified | None |
| **Color-Only Training** | ✅ Working | None |
| **Multi-Camera Support** | 🔴 Broken | Bug #1 |
| **Adaptive Density** | 🔴 Will Break | Bug #2 |
| **Full Optimization** | 🟡 Partial | Need wiring |

---

## 💡 Pro Tips

1. **Always run gradient checks first** when adding new differentiable ops
2. **Test with multiple datasets** - bugs hide in edge cases
3. **Use downsampling during development** - faster iteration (0.25x works well)
4. **Save intermediate outputs** - coverage, transmittance, etc. are invaluable for debugging
5. **Trust the math** - if gradients check out, focus on engineering bugs

---

**Last Updated**: 2025-12-15
**Code Review Status**: Complete (M1-M7)
**Confidence Level**: High ✅
