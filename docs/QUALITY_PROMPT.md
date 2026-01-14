# 3D Gaussian Splatting Training Quality Completion Criteria

## Purpose
This document defines the specific, measurable conditions that indicate a high-quality 3DGS training run. Use these criteria to determine when training has successfully completed versus when it has failed and should be aborted.

---

## Overall Success Definition

A **successful high-quality 3DGS training run** produces a model that:
1. Achieves measurable improvement in reconstruction quality (PSNR)
2. Creates appropriate Gaussian coverage of the scene
3. Maintains optimizer stability throughout training
4. Generates visually accurate novel view renderings

---

## Primary Quality Metrics

### 1. PSNR (Peak Signal-to-Noise Ratio) - PRIMARY INDICATOR

**Success Criteria:**
- **Minimum improvement:** +1.0 dB from initial to final PSNR
- **Target improvement by dataset size:**
  - 20-40 images: +1.5 to +3.0 dB
  - 50-75 images: +2.5 to +5.0 dB
  - 100 images: +3.0 to +6.0 dB

**Failure Indicators:**
- ❌ PSNR degrading over time (final < initial)
- ❌ PSNR flat for >3000 consecutive iterations
- ❌ PSNR < 10 dB at any checkpoint after iteration 1000

**Measurement Points:**
- Initial: Iteration 1-100 (before densification)
- Checkpoint: Every val_interval iterations
- Final: Last iteration

**Example Targets:**
```
Initial PSNR: 10.44 dB
Final PSNR:   14.51 dB  ✅ (+4.08 dB improvement)

Initial PSNR: 12.81 dB
Final PSNR:   5.42 dB   ❌ (-7.39 dB degradation - FAILURE)
```

---

### 2. Training Loss - CONVERGENCE INDICATOR

**Success Criteria:**
- Loss decreasing over the first 50% of training
- Loss stabilizing or slowly decreasing in final 25% of training
- Final loss < 0.5 for L2 loss
- Final loss < 0.3 for L1+DSSIM loss

**Failure Indicators:**
- ❌ Loss increasing for >1000 consecutive iterations
- ❌ Loss > 1.0 after iteration 5000
- ❌ Loss fluctuating wildly (variance > 0.2 over 100 iterations)

**Measurement:**
- Track moving average over 100 iterations
- Compare iteration N to iteration N-1000

---

### 3. Visual Quality - PERCEPTUAL VALIDATION

**Success Criteria:**
- Rendered test views show recognizable scene structure
- Major geometric features clearly visible
- Colors reasonably accurate (no solid color failures)
- Minimal black artifacts or floating splats

**Failure Indicators:**
- ❌ Solid color renderings (file size <50 KB)
- ❌ Completely black or white images
- ❌ Unrecognizable scene (no correspondence to input views)
- ❌ Severe artifacts (large black regions, extreme noise)

**File Size Heuristics:**
```
< 50 KB:     Solid color failure ❌
50-150 KB:   Very poor quality ⚠️
150-250 KB:  Marginal quality ⚠️
250-400 KB:  Good quality ✅
> 400 KB:    Excellent detail ✅
```

**Validation:**
- Check rendered test views at iterations 1000, 3000, 5000, final
- Compare rendered vs target side-by-side
- Verify scene is recognizable by iteration 3000

---

## Gaussian Population Metrics

### 4. Gaussian Count - CAPACITY MANAGEMENT

**Success Criteria by Dataset Size:**

**20-40 images:**
- Initial: 200-1,000 Gaussians
- Final: 5,000-15,000 Gaussians
- Growth ratio: 5-15×

**50-75 images:**
- Initial: 400-2,000 Gaussians
- Final: 8,000-20,000 Gaussians
- Growth ratio: 4-10×

**100 images:**
- Initial: 500-3,000 Gaussians
- Final: 15,000-50,000 Gaussians
- Growth ratio: 5-15×

**Failure Indicators:**
- ❌ Zero growth (final == initial): Densification not working
- ❌ Explosive growth (>10× in <3000 iterations): Runaway densification
- ❌ Exceeding GPU capacity (>400K Gaussians on Metal GPU)
- ❌ >100K Gaussians for datasets <100 images

**Healthy Growth Pattern:**
```
Iteration 0:     445 Gaussians (initial point cloud)
Iteration 500:   1,200 Gaussians (first densification)
Iteration 1000:  2,500 Gaussians (gradual growth)
Iteration 5000:  8,000 Gaussians (stabilizing)
Iteration 10000: 8,802 Gaussians (final) ✅

vs.

Iteration 0:     445 Gaussians
Iteration 5000:  73,548 Gaussians
Iteration 8000:  272,112 Gaussians
Iteration 8500:  423,821 Gaussians → GPU CRASH ❌
```

---

### 5. Densification Activity - GROWTH HEALTH

**Success Criteria:**
- Split/clone events concentrated in first 60% of training
- Pruning rate 10-30% of split+clone rate
- Densification tapering off in final 30% of training
- grad_p90 staying between 0.0001 and 0.5

**Failure Indicators:**
- ❌ Zero densification events (densify_split + densify_clone == 0)
- ❌ Excessive pruning (pruned > split + cloned)
- ❌ grad_p90 > 1.0 (gradient explosion)
- ❌ Densification cap hit repeatedly (cap_hit warnings)

**Healthy Densification Pattern:**
```
Iterations 0-3000:    Heavy split/clone activity
Iterations 3000-7000: Moderate activity
Iterations 7000+:     Minimal activity (refinement phase)

Example stats at iteration 5000:
densify_split: 1200
densify_clone: 800
densify_prune: 300  ✅ (15% prune rate)
grad_p90: 0.0521    ✅ (reasonable gradients)
```

---

### 6. Gaussian Health Metrics - QUALITY INDICATORS

**Anisotropy (Aspect Ratio):**
- aniso_median: 1.5-4.0 (reasonable elongation)
- aniso_p90: 2.0-7.0 (some needles OK)
- aniso_max: <10.0 (no extreme needles)

**Opacity Distribution:**
- opacity_median: 0.3-0.8 (useful range)
- opacity_low_pct: 10-40% (some weak Gaussians OK for blending)

**Scale Distribution:**
- scale_median: 0.01-0.5 (scene-dependent, should be stable)
- scale_median should not drift >2× during final 50% of training

**Gradient Magnitudes:**
- pos_grad_median: 0.00001-0.001 (decreasing over time)
- scale_grad_median: 0.00001-0.001
- rot_grad_median: 0.00001-0.001

**Failure Indicators:**
- ❌ aniso_max > 20.0 (severe needle formation)
- ❌ opacity_median < 0.1 (all Gaussians too transparent)
- ❌ opacity_low_pct > 70% (most Gaussians ineffective)
- ❌ Gradient medians >0.01 after iteration 5000 (not converging)

---

## Optimizer Stability Metrics

### 7. Background Color - OPTIMIZATION HEALTH

**Success Criteria:**
- All RGB components in range [0.0, 1.0]
- Stable in final 30% of training (change < 0.1 per component)
- Background not pure black [0,0,0] or pure white [1,1,1]

**Failure Indicators:**
- ❌ Any component < 0.0 or > 1.0 (unphysical, optimizer diverged)
- ❌ Drifting rapidly (>0.3 change per 1000 iterations)
- ❌ Extreme values even if in range (e.g., [0.001, 0.001, 0.001])

**Example:**
```
Iteration 1000: bg=(0.46, 0.55, 0.47) ✅
Iteration 3000: bg=(0.48, 0.54, 0.46) ✅ (stable)

vs.

Iteration 1000: bg=(0.46, 0.55, 0.47) ✅
Iteration 2000: bg=(0.86, 1.00, 0.83) ⚠️ (approaching limit)
Iteration 3000: bg=(0.95, 1.15, 0.84) ❌ (GREEN > 1.0 - DIVERGED)
```

---

### 8. Gradient Statistics - CONVERGENCE HEALTH

**Success Criteria:**
- grad_p50 decreasing over training
- grad_p90 staying <0.5 throughout
- Final grad_p50 <0.0001 (approaching convergence)

**Warning Indicators:**
- ⚠️ grad_p90 >0.05 sustained for >1000 iterations (escalation risk)
- ⚠️ grad_p90 increasing monotonically for 3+ checkpoints
- ⚠️ grad_p90 doubles between consecutive checkpoints

**Failure Indicators:**
- ❌ grad_p50 increasing after iteration 3000
- ❌ grad_p90 >1.0 at any point (explosion)
- ❌ grad_p90 >5.0 (critical: severe gradient escalation, likely failure)
- ❌ grad_p90 >10.0 (catastrophic: abort immediately)
- ❌ grad_p50 >0.001 at final iteration (not converged)

**Healthy Gradient Evolution:**
```
Iteration 1000:  grad_p50=0.0005, grad_p90=0.05
Iteration 5000:  grad_p50=0.0002, grad_p90=0.02  ✅ (decreasing)
Iteration 10000: grad_p50=0.00005, grad_p90=0.005 ✅ (converged)

vs.

Iteration 1000:  grad_p50=0.0000, grad_p90=0.0002
Iteration 5000:  grad_p50=0.0000, grad_p90=0.0052 ⚠️ (escalating)
Iteration 8000:  grad_p50=0.0001, grad_p90=0.0403 ⚠️ (severe escalation)
Iteration 10000: grad_p50=0.0002, grad_p90=0.0817 ❌ (failed, should abort)
```

**Real-World Example (runs/20251222_0041_onehour):**
- Iteration 501: grad_p90=0.0002 ✅
- Iteration 1501: grad_p90=0.0052 ⚠️ (26x increase)
- Iteration 8501: grad_p90=0.0403 ❌ (201x from start)
- Iteration 10000: grad_p90=0.0817 ❌ (408x from start)
- Result: PSNR degraded -2.08 dB, 59K Gaussians, FAILED

---

## Additional Monitoring Metrics

### 9. Densification Rate - GROWTH CONTROL

**Definition:**
```
Densification Rate = (Gaussians_current - Gaussians_prev) / Gaussians_prev
```

**Success Criteria:**
- Rate <50% per 1000 iterations in first 50% of training
- Rate <20% per 1000 iterations in final 50% of training
- Rate decreasing over time (tapering off)

**Warning Indicators:**
- ⚠️ Rate >50% sustained for 2+ consecutive intervals
- ⚠️ Rate increasing in second half of training
- ⚠️ Rate >30% after iteration 5000

**Failure Indicators:**
- ❌ Rate >100% (doubling) in any 1000-iteration interval
- ❌ Rate >50% sustained for 3+ consecutive intervals
- ❌ Exponential growth pattern (rate increasing over time)

**Example Calculations:**
```
Healthy Growth:
Iter 0:    1,000 Gaussians
Iter 1000: 2,500 Gaussians → +150% (acceptable early growth)
Iter 2000: 4,000 Gaussians → +60% (healthy)
Iter 5000: 8,000 Gaussians → +33% (tapering)
Iter 10000: 9,500 Gaussians → +4% (refinement) ✅

vs.

Runaway Growth:
Iter 0:    1,183 Gaussians
Iter 1000: 1,620 Gaussians → +37% (OK)
Iter 2000: 3,500 Gaussians → +116% (WARNING)
Iter 5000: 16,506 Gaussians → +372% (CRITICAL) ❌
Iter 8000: 47,270 Gaussians → +186% (CATASTROPHIC) ❌
```

---

### 10. Initialization Quality - EARLY CHECK

**Purpose:** Verify that initial Gaussian distribution provides adequate scene coverage before investing in full training.

**Check at Iteration 500-1000:**
- Initial Gaussian count appropriate for dataset size:
  - 20 images: 200-1,000 Gaussians minimum
  - 50 images: 500-2,000 Gaussians minimum
  - 75 images: 800-3,000 Gaussians minimum
  - 100+ images: 1,000-5,000 Gaussians minimum

**Warning Signs:**
- ⚠️ Extremely sparse initialization (<500 Gaussians for 50+ images)
- ⚠️ PSNR <10 dB at iteration 500 (very poor initial coverage)
- ⚠️ Massive densification activity in first interval (>100% growth)

**Recommended Action:**
If initialization is inadequate:
1. Re-run COLMAP with denser point cloud
2. Use stratified sampling to ensure coverage
3. Consider pre-densification phase before gradient-based training
4. Adjust `densify_grad_threshold` if starting very sparse

---

## Checkpoint-Based Quality Gates

### Iteration 1000 - Early Validation

**PASS Requirements:**
- ✅ PSNR improved by ≥0.3 dB OR PSNR >11 dB
- ✅ Loss decreased from initial
- ✅ Background RGB all <1.0
- ✅ Gaussians 1.5× to 5× initial count
- ✅ Visual: Scene recognizable in rendered test view

**ABORT if:**
- ❌ PSNR decreased >1.0 dB
- ❌ File size <50 KB (solid color)
- ❌ Background component >1.0

---

### Iteration 3000 - Stability Check

**PASS Requirements:**
- ✅ PSNR improved by ≥0.8 dB from initial
- ✅ PSNR not decreasing (iteration 3000 ≥ iteration 2000)
- ✅ Gaussians 2× to 10× initial count
- ✅ grad_p90 <0.5
- ✅ Visual: Clear scene details visible

**ABORT if:**
- ❌ PSNR degrading for 2+ consecutive checkpoints
- ❌ Gaussians >50K for <100 image datasets
- ❌ Any component of background >1.0

---

### Iteration 5000-8000 - Densification Risk Window

**PASS Requirements:**
- ✅ Gaussian count growth rate slowing (<20% per 1000 iterations)
- ✅ grad_p90 stable or decreasing
- ✅ No GPU crashes or buffer overflows
- ✅ PSNR continuing to improve or stable

**MONITOR:**
- Densification cap not being hit repeatedly
- File sizes staying >150 KB
- Loss not increasing

---

### Final Iteration - Completion Validation

**HIGH QUALITY Requirements:**
- ✅ PSNR improvement ≥1.5 dB (dataset-dependent targets above)
- ✅ Final PSNR ≥12 dB
- ✅ Visual quality: Recognizable, detailed scene
- ✅ File size ≥200 KB
- ✅ Gaussian count in healthy range (see section 4)
- ✅ Background RGB all in [0.0, 1.0]
- ✅ grad_p50 <0.0001 (converged)

**ACCEPTABLE Quality (may require retraining):**
- ⚠️ PSNR improvement +0.5 to +1.5 dB
- ⚠️ Final PSNR 10-12 dB
- ⚠️ File size 100-200 KB
- ⚠️ Some minor artifacts but scene recognizable

**FAILED Training:**
- ❌ PSNR decreased or improved <0.5 dB
- ❌ Final PSNR <10 dB
- ❌ File size <100 KB
- ❌ Solid color or unrecognizable rendering
- ❌ Background RGB outside [0.0, 1.0]
- ❌ Severe visual artifacts

---

## Configuration-Specific Success Criteria

### Micro Preset (20 images, 2000 iterations)
- **Target PSNR improvement:** +1.5 to +3.0 dB
- **Expected final Gaussians:** 8,000-12,000
- **Expected duration:** ~5-10 minutes
- **File size target:** 200-300 KB

### Onehour Preset (75 images, 10K iterations)
- **Target PSNR improvement:** +2.5 to +5.0 dB
- **Expected final Gaussians:** 8,000-15,000
- **Expected duration:** ~60 minutes
- **File size target:** 250-400 KB

### Full Preset (100 images, 30K iterations)
- **Target PSNR improvement:** +3.0 to +6.0 dB
- **Expected final Gaussians:** 25,000-50,000
- **Expected duration:** ~5-7 hours
- **File size target:** 300-500 KB

---

## Early Abort Conditions

Training should be **aborted immediately** if ANY of these occur:

1. **Solid Color Failure:** File size <50 KB at iteration 1000
2. **Background Divergence:** Any RGB component outside [0.0, 1.0] (especially negative values)
3. **PSNR Collapse:** PSNR decreased >2.0 dB from peak
4. **Gradient Explosion:** grad_p90 >10.0 (warning at >5.0, critical at >20.0)
5. **Gaussian Explosion:** Growing >100% per 1000 iterations (warning at >50%)
6. **GPU Crash:** Buffer overflow or memory errors
7. **Persistent Degradation:** PSNR decreasing for 3+ consecutive checkpoints
8. **Densification Rate Runaway:** Gaussian count doubling every 1000 iterations sustained for 2+ intervals

**Rationale:** These conditions indicate fundamental optimization failure that will not recover. Aborting early saves compute time.

**Real-World Example #1 (20251219_2055_onehour - Gradient Explosion):**
- Iteration 2000: grad_p90 spiked from 0.0005 to 34.0 ❌ (gradient explosion)
- Iteration 4000: Background went negative [-0.42, -0.50, -0.35] ❌ (divergence)
- Iteration 10000: PSNR collapsed from 12.81 to 5.42 dB ❌ (-7.39 dB total)
- Result: 114,663 Gaussians (257x growth) producing unusable output
- **Should have aborted at iteration 2000** when gradient explosion occurred

**Real-World Example #2 (20251222_0041_onehour - Gradient Escalation):**
- Iteration 501: 1,333 Gaussians, PSNR=13.69 dB, grad_p90=0.0002 ✅
- Iteration 1501: 2,073 Gaussians (+55%), PSNR=13.91 dB, grad_p90=0.0052 ⚠️ (26x increase)
- Iteration 8501: 37,287 Gaussians (1700% from iter 1501), grad_p90=0.0403 ❌
- Iteration 10000: 59,255 Gaussians (2759% from iter 1501), PSNR=12.15 dB (-2.08 dB degradation) ❌
- **Should have aborted at iteration 3000-5000** when:
  - Gaussian growth rate exceeded 100% per 1000 iters
  - grad_p90 exceeded 0.05 with no quality improvement
  - PSNR started degrading from initial

---

## Post-Training Validation

After training completes, perform final validation:

### 1. Model File Validation
```bash
# Check model was saved
ls -lh runs/YOUR_RUN/model.gs
# Should be 1-10 MB depending on Gaussian count

# Verify metadata
# - num_gaussians matches final count
# - training_psnr matches final PSNR
# - bounds_min/max are reasonable (not NaN/Inf)
```

### 2. Metrics CSV Validation
```bash
# Check CSV has all iterations
wc -l runs/YOUR_RUN/metrics.csv
# Should match: (iters / log_interval) + 1 header

# Verify no NaN values
grep -i nan runs/YOUR_RUN/metrics.csv
# Should return nothing

# Check PSNR trend
tail -20 runs/YOUR_RUN/metrics.csv | awk -F, '{print $3}'
# Should show stable or increasing values
```

### 3. Visual Validation
```bash
# Review final test view
open runs/YOUR_RUN/m8_test_view_rendered.png
open runs/YOUR_RUN/m8_test_view_target.png

# Check for:
# - Scene recognizable ✅
# - Major features present ✅
# - Reasonable colors ✅
# - No massive artifacts ✅
```

### 4. Viewer Validation (if available)
```bash
# Load model in viewer
sugar-viewer runs/YOUR_RUN/model.gs

# Verify:
# - Model loads without errors ✅
# - Novel views render correctly ✅
# - No major holes or artifacts ✅
# - Quality matches static renders ✅
```

---

## Summary Checklist

Use this checklist to quickly determine training success:

**✅ HIGH QUALITY Run:**
- [ ] PSNR improved ≥1.5 dB
- [ ] Final PSNR ≥12 dB
- [ ] Background RGB all in [0.0, 1.0]
- [ ] Gaussian count in healthy range
- [ ] File size ≥200 KB
- [ ] Scene recognizable with good detail
- [ ] No gradient explosions
- [ ] Model file saved successfully

**⚠️ ACCEPTABLE Run (consider retraining):**
- [ ] PSNR improved 0.5-1.5 dB
- [ ] Some minor quality issues
- [ ] File size 100-200 KB
- [ ] Scene recognizable but artifacts present

**❌ FAILED Run (must retrain):**
- [ ] PSNR decreased or <0.5 dB improvement
- [ ] Background RGB outside [0.0, 1.0]
- [ ] Solid color or unrecognizable rendering
- [ ] File size <100 KB
- [ ] Severe artifacts or crashes

---

## Automated Quality Assessment (Future)

Recommended thresholds for automated pass/fail:

```python
def assess_training_quality(metrics_csv, final_png):
    """
    Returns: 'HIGH_QUALITY' | 'ACCEPTABLE' | 'FAILED'
    """
    initial_psnr = metrics_csv.loc[0, 'psnr']
    final_psnr = metrics_csv.loc[-1, 'psnr']
    improvement = final_psnr - initial_psnr

    bg_max = max(abs(metrics_csv.loc[-1, 'bg_r']),
                 abs(metrics_csv.loc[-1, 'bg_g']),
                 abs(metrics_csv.loc[-1, 'bg_b']))

    file_size_kb = os.path.getsize(final_png) / 1024

    # FAILED checks
    if improvement < 0.5:
        return 'FAILED'
    if bg_max > 1.0:
        return 'FAILED'
    if file_size_kb < 100:
        return 'FAILED'
    if final_psnr < 10:
        return 'FAILED'

    # HIGH_QUALITY checks
    if improvement >= 1.5 and final_psnr >= 12 and file_size_kb >= 200:
        return 'HIGH_QUALITY'

    # Otherwise ACCEPTABLE
    return 'ACCEPTABLE'
```

---

## Key Insights: Why Gaussians End Up in Wrong Locations

### Problem: Gradient-Driven vs Quality-Driven Densification

The current densification system (trainer.rs:1031-1257) uses **gradient magnitude** as the primary signal for where to add Gaussians:

```rust
// Line 1153: Only densifies if average gradient is high
if avg_grad > grad_threshold && can_add {
    // Split or clone this Gaussian
}
```

**This creates a fundamental issue:**
- **High gradient** = Gaussian struggling to learn = area needs more representation
- But **high gradient ≠ good location for new Gaussians**
- Gradient explosion (observed in failed runs) triggers massive densification
- New Gaussians added in wrong locations don't improve quality
- Optimizer compensates by distorting background → divergence

### Evidence from Failed Runs

**20251219_2055_onehour timeline:**
1. Iteration 1-2000: Normal gradients (grad_p90 < 1.0), moderate densification
2. **Iteration 2000: grad_p90 spikes to 34.0** (gradient explosion event)
3. Iteration 2000-5000: Massive densification triggered (18,632 splits total)
4. Iteration 4000+: Background diverges (goes negative) trying to compensate
5. Iteration 5000-10000: PSNR collapses as bad Gaussians dominate

**Root causes identified:**
- Fixed `grad_threshold = 0.0002` is too sensitive to gradient explosions
- No quality feedback loop (PSNR/loss improvement) in densification decisions
- Pruning criteria too lenient (opacity < 0.01, aniso > 2.0 log-space)
- When cap is hit, bad Gaussians stay and block good ones from being added

### Recommendations for Quality Improvement

**Short-term fixes:**
1. **Adaptive gradient threshold:** Use percentile-based threshold (e.g., grad_p90 * 0.5) instead of fixed 0.0002
2. **Abort on gradient explosion:** Auto-abort if grad_p90 >20.0
3. **Tighter pruning:** Increase opacity threshold to 0.05, reduce aniso threshold to 1.5 log-space
4. **Background clamping:** Hard clamp background to [0.0, 1.0] during optimization

**Long-term improvements:**
5. **Quality-driven densification:** Only densify if PSNR hasn't improved in last N iterations
6. **Loss-aware splitting:** Track per-Gaussian contribution to loss, split high-loss regions
7. **Probabilistic pruning:** Prune Gaussians with <1% contribution to rendered pixels
8. **Two-phase training:** Dense initial placement (first 30%), then refinement only (last 70%)

---

## References

- **Metrics Definition:** src/optim/trainer.rs:43-132 (CsvLogger, GaussianStats)
- **Densification Logic:** src/optim/trainer.rs:1031-1257 (densify_and_prune)
- **Training Loop:** src/optim/trainer.rs:1266+ (train_multiview_color_only)
- **Known Issues:** docs/QUALITY.md (troubleshooting guide)
- **Failed Run Analysis:** runs/20251219_2055_onehour/metrics.csv (gradient explosion at iter 2000)
- **Progress Documentation:** docs/PROGRESS.md (detailed investigation notes)

---

**Document Version:** 2.1
**Last Updated:** 2025-01-06
**Based on:**
- Analysis of metrics.csv output structure
- Code review of densification/pruning implementation
- Empirical evidence from failed training runs (multiple sessions)
- Root cause analysis of gradient-driven densification failures
- Validation against recent training runs (micro: success, onehour: failure)

**Version 2.1 Enhancements (2025-01-06):**
- Added refined gradient thresholds (warning at >5.0, critical at >10.0 vs previous >20.0)
- Added densification rate monitoring metric (Section 9)
- Added initialization quality checks (Section 10)
- Added second real-world failure example (20251222_0041_onehour with gradient escalation)
- Refined early abort conditions to include densification rate runaway
- Validated all thresholds against empirical data from recent runs
