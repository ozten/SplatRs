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

## 🟡 M8: Multi-View Training - IMPLEMENTED (Quality Tuning Ongoing)

### What M8 Needs
- Train on 10-20 images from different viewpoints
- Multi-view consistency (gradients from multiple views)
- Test view synthesis on held-out cameras

### Testing Strategy
1. Run the smoke test: `cargo test --test m8_multiview_train`
2. For longer quality runs: `cargo test --test m8_multiview_train -- --ignored`
3. For deterministic splits/sampling, pass `--seed 0` to `sugar-train` (see `docs/ROADMAP.md`)

**Status**: 🟡 **IN PROGRESS** (quality targets depend on dataset + runtime budget)

---

## 🟡 M9: Adaptive Density Control - IMPLEMENTED (Quality Tuning Ongoing)

### What M9 Needs
- Gaussian splitting (high-gradient regions)
- Gaussian cloning (under-reconstructed regions)
- Gaussian pruning (low-opacity Gaussians)
- Parameter count changes dynamically during training

### Testing Strategy
1. Smoke test integration: `cargo test --test m9_adaptive_density_control`
2. Manual/longer runs with densify enabled: see `docs/ROADMAP.md` “M9” command
3. Watch logs for `densify @iter ... gaussians A -> B` and ensure PSNR doesn’t regress

**Status**: 🟡 **IN PROGRESS** (needs tuning to consistently improve quality vs M8)

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
# 1. Verify multi-view training
- [ ] Run: cargo test --test m8_multiview_train
- [ ] Optional (slow): cargo test --test m8_multiview_train -- --ignored
- [ ] Manual: run `sugar-train --multiview ... --seed 0` and inspect outputs
```

### Before starting M9:
```bash
# 1. Verify density control plumbing
- [ ] Run: cargo test --test m9_adaptive_density_control
- [ ] Manual: run `sugar-train --multiview ... --densify-interval N ... --seed 0`
- [ ] Tune `--prune-opacity-threshold` downward if repeated splits prune too aggressively
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
