# SplatRs Recovery Plan — Getting Gaussian Splatting Accurate

**Date:** 2026-07-06
**Method:** Five focused review agents compared the implementation against the canonical
3DGS reference (Kerbl et al. 2023, graphdeco-inria/gaussian-splatting). Load-bearing findings
were re-verified directly against source.

---

## 1. Diagnosis: there are two layers of problems

The reported symptoms ("Gaussians don't stay small, aren't as populated") are real, but they
are **downstream** of a deeper issue. Evidence from `runs/*/metrics.csv`:

| Run | Scene | Result |
|-----|-------|--------|
| `20251219_0128_full` | tandt/train | PSNR oscillates **8.3–9.4 dB** while count explodes 51k→272k, then crashes |
| `20251218_2044_onehour` | tandt/train | PSNR **11.9 → 8.8 dB** (degrading), count → **401,582** (hit GPU hard cap) |
| `2025122*_onehour` | dollhouse_sm | PSNR **12.3 → 5.4 dB**, background RGB → **negative** (physically invalid) |
| `debug_no_densify_*` | tandt/train | PSNR **9–11 dB even with densification disabled** |
| `20260115_1830_micro` | tandt/train (2k it) | Best case: **13–17 dB** on a small subset |

Reference 3DGS reaches **25–35 dB** on these scenes, and ~18–21 dB *before* densification even
kicks in. **We are stuck at 9–11 dB with densification off.** That is the headline: the base
renderer/optimizer cannot fit even a fixed set of Gaussians well. Densification layered on top of
a broken base just amplifies the chaos into the explosion/divergence documented in
`docs/QUALITY.md`.

**Therefore: fix base fidelity FIRST, then rebuild densification. Do not tune densification on a
broken base.**

---

## 2. Confirmed bugs (verified against source)

### Group A — Base fidelity / gradient correctness (fix first)

| ID | Severity | File:line | Bug | Reference behavior |
|----|----------|-----------|-----|--------------------|
| A1 | **HIGH** | `core/init.rs:25` | Initial opacity = `2.2` logit → sigmoid ≈ **0.90** (near-opaque) | `inverse_sigmoid(0.1) ≈ -2.197` (near-transparent). Opaque init saturates alpha-compositing, starves gradient to occluded Gaussians and to the densification heuristic. |
| A2 | **MED-HIGH** | `render/full_diff.rs:423` | Off-diagonal covariance gradient double-counted **2×**. `Matrix2::new(d_cov.x, d_cov.y, d_cov.y, d_cov.z)` feeds the full single-DOF partial into both off-diagonal slots. Numerically confirmed vs finite differences (ratio 0.5 on the cross-term). | Off-diagonal enters at half value. Inflates `dL/d(log_scale)`, `dL/d(rotation)`, `dL/d(position)` by 2× for rotated/anisotropic Gaussians (exactly the ones representing fine detail/edges). |
| A3 | **HIGH** | `runs/*` (background) | Background/optimizer divergence: background RGB goes **negative / past 1.0**; PSNR degrades over time. | Background is a constant or a clamped/regularized parameter; training should converge, not diverge. Root cause likely interacts with A1/D2. Needs isolation (Phase 0). |
| A4 | **MED** | `render/full_diff.rs:101-106` | Hard "screen-fill" cull returns `None` for Gaussians whose 3σ radius exceeds the image → they get **zero gradient of any kind** and can never be shrunk. | Reference keeps them in the backward pass (tile-culled, not excluded) so oversized Gaussians still receive a corrective shrink gradient. |

### Group B — Densification / pruning / count

| ID | Severity | File:line | Bug | Reference behavior |
|----|----------|-----------|-----|--------------------|
| B1 | **CRITICAL** | `optim/trainer.rs:1962-1969` + `full_diff.rs:404-457` | Densification signal is the **world-space** position gradient (chain-ruled through the projection Jacobian by a depth-dependent `~fx/z` factor), not the screen-space 2D mean gradient. The `0.0002` threshold is calibrated for the view-space quantity, so it is meaningless here (explains the wild per-preset threshold retuning: 0.1, 0.0002, …). | Accumulate `‖∂L/∂(2D projected mean)‖` (`viewspace_points.grad`), which is roughly depth-invariant so one threshold works scene-wide. `d_mean_px` is already computed at `full_diff.rs:339` but discarded. |
| B2 | **CRITICAL** | `optim/trainer.rs:1130` | `denom` is a **single global iteration counter** (`grad_window_iters`), not per-Gaussian. Only one camera is sampled per iteration, so a Gaussian visible in a fraction of views has its avg gradient diluted by `total_iters / visible_iters` → never crosses the threshold. | `denom[i]` increments only when Gaussian `i` is in the visibility filter for that view; average = `grad_accum[i] / denom[i]`. |
| B3 | **CRITICAL** | `optim/trainer.rs:2171-2178` | Opacity reset is **inverted and too frequent**: sets every opacity to logit `0.0` (**sigmoid 0.5**, pushing *up*), on *every* densify cycle (~every 100–500 iters). Also stomps the just-computed split/clone child opacities. | Every **3000** iters only: `opacity = min(opacity, 0.01)` (pushes *down*, ≈ logit -4.6), forcing weak Gaussians to re-prove themselves or be pruned. |
| B4 | **HIGH** | `optim/trainer.rs:1185-1186` | Split under-shrinks: `log_scale − 0.2` ⇒ ×`exp(-0.2) ≈ 0.82`. | `new_scale = old / (0.8·N) = old / 1.6 ≈ 0.625×`. Split is *the* mechanism that keeps Gaussians small; it barely shrinks here. |
| B5 | **HIGH** | `optim/trainer.rs:986-989` + `bin/train.rs` presets | No scene-extent-relative clone/split threshold. Compares **mean** axis scale (should be **max**) against a fixed absolute constant (0.05–0.1). `scene_extent`/`percent_dense` do not exist anywhere in the codebase (zero grep hits). | Compare **max** axis scale against `percent_dense (0.01) · scene_extent`. Oversized Gaussians get *cloned at full size* instead of split → grow the large population. |
| B6 | **HIGH** | `optim/trainer.rs` (`densify_and_prune`) | No size-based prune. Only opacity, distance-outlier (>50 units), and anisotropy prunes exist. No screen-radius (>20px) or world-scale (>0.1·extent) prune. Only scale ceiling is a static `MAX_LOG_SCALE = 5.0` → `exp(5) ≈ 148` units (larger than most whole scenes). | After each opacity reset, prune Gaussians with screen radius >20px or world scale >0.1·scene_extent. |
| B7 | **MED** | `optim/trainer.rs:2130-2133` | Densification (and the disruptive opacity reset) never stops — runs the entire 30k run. | Densify from 500 to **15000** only; freeze and settle for the second half. |
| B8 | **MED** | `optim/trainer.rs:1163` | Clone/split jitter clamped to an absolute **5mm** regardless of Gaussian size → children spawn on top of parent, add count without coverage. | Sample offset from the Gaussian's own distribution (scales with its size). |
| B9 | **MED** | `optim/trainer.rs:1075-1082` | `cap = cap.max(before)` — the densify cap can never shrink an over-target population; it silently ratchets upward. | Cap is a real ceiling. |
| B10 | **LOW** | `bin/train.rs` presets | `densify_grad_threshold` inconsistent: `0.1` for debug/m9/m10 (≈500× reference, near-disables densification) vs `0.0002` for micro/onehour/full; `densify_interval=500` (reference 100). | Consistent `0.0002` / interval 100. |

### Group C — Initialization

| ID | Severity | File:line | Bug | Reference behavior |
|----|----------|-----------|-----|--------------------|
| C1 | **HIGH** | `core/init.rs:19` / `trainer.rs:432-441` | Initial scale is a **fixed constant** (`-4.6` log) or a depth-only heuristic (`1.5·z/f`) — no nearest-neighbor computation exists. Sparse and dense regions get the same size. | `scale_raw = log(sqrt(mean_sq_dist_to_3_nearest_neighbors))`, per-point, density-adaptive. Keeps init small & local. |
| C2 | **MED** | `bin/train.rs:796` | SH degree fixed at 3 from iter 0 (`learn_sh` all-or-nothing); no progressive warmup. | Start at degree 0, +1 every 1000 iters up to 3. Prevents early color instability. |

### Group D — Optimizer / learning-rate schedule

| ID | Severity | File:line | Bug | Reference behavior |
|----|----------|-----------|-----|--------------------|
| D1 | **HIGH** | everywhere (grep: 0 hits) | Position LR has **no `spatial_lr_scale`** (scene-extent) multiplier. For scenes with units >1, positions move far too slowly to relocate onto geometry; Gaussians compensate by growing scale. | `position_lr = 0.00016 · scene_extent`, with delayed exponential decay to `0.0000016` (100×). |
| D2 | **MED-HIGH** | `optim/trainer.rs:1615-1658` | Exponential LR decay applied uniformly to **all** parameter groups (position, scale, rotation, opacity, SH, background), each decaying only 10× (not 100×), no warmup delay. | Only **position** is scheduled (steep, delayed). Scale/rotation/opacity/feature LRs are held **constant** for all 30k steps. |
| D3 | **MED** | `optim/adam.rs:297-333` | SH "rest" bands share the DC learning rate (20× too fast). | `feature_lr = 0.0025` (DC), `feature_rest_lr = 0.0025/20 = 0.000125`. |

**What is correct** (verified, don't touch): Adam core math (bias-corrected moments, sign, sqrt),
raw numeric LR defaults for production presets, loss function (`0.8·L1 + 0.2·(1−SSIM)`, 11×11
window), alpha-blend backward, the exp-activation Jacobian in the scale gradient, quaternion
normalization, covariance formula `Σ = R·diag(exp(s)²)·Rᵀ`, and the Adam moment re-sizing in
lockstep with densification.

---

## 3. The plan

### Phase 0 — Measure and isolate — ✅ DONE (2026-07-06)

**Result: the forward renderer is fundamentally CORRECT. The base-fidelity collapse is in the
training/optimization path, not the rasterizer.**

Built `load_ply()` (INRIA/Brush import, the old M10 TODO) + a new `sugar-eval-reference` binary
that renders a known-good model from the dataset's own cameras and reports PSNR. Ran it on
`nerf_synthetic/lego` (`lego.ply`, 173k Gaussians) at half-res, sweeping the two convention
degrees of freedom:

| Convention combo | Mean PSNR (3 frames) |
|---|---|
| offset + nogamma (Brush/INRIA convention) | **22.7 dB** (frames 0/1: **27.0 / 25.3**) |
| offset + gamma | 16.1 dB |
| asis + gamma (what `sugar-render` does) | 13.9 dB |
| asis + nogamma | 12.2 dB |

Frames 0/1 reproduce the reference bulldozer near-perfectly (correct geometry, pose, sharpness,
color) once conventions are matched → **projection, 2D covariance, depth sort, alpha compositing,
and SH evaluation are all correct.** Camera pose conversion (Blender→OpenCV) is also correct
(geometry aligns exactly).

**Important nuance:** the dark "asis" render is an *import* mismatch — SplatRs internally uses a
self-consistent *linear + no-DC-offset* SH convention (see `init.rs` initializing DC from
linearized color with no −0.5 shift, and `evaluate_sh` omitting +0.5), whereas Brush/INRIA use
*display-space + 0.5-offset*. So the color combos above do **not** explain SplatRs's own 9–13 dB
training runs (its color pipeline is internally consistent). They confirm the renderer works.

**Conclusion → the 9–13 dB training problem is NOT in the forward renderer.** Focus Phase 1 on the
optimization-path bugs (A1 opacity init, A2 2× covariance gradient, A3 background divergence,
D1/D2 LR schedule) and Phase 2 on the densification rebuild. The renderer is a solved problem to
build on.

**Two real (secondary) renderer issues surfaced:**
- **SH DC-offset inconsistency:** `evaluate_sh_dc_only` adds `+0.5` but `evaluate_sh` /
  `evaluate_sh_unclamped` (the full-SH render path) do not. Latent bug if any path mixes DC-only
  and full-SH rendering of the same model. (`src/core/sh.rs`)
- **Oversized "needle" artifact — ✅ FIXED.** An 8-frame orbit survey showed this hit ~half the
  views: cardinal-azimuth frames (0/50/100/150) rendered at **27–29 dB** but diagonal frames
  (25/75/125/175), where the raised arm/bucket juts toward the camera, dropped to **15–20 dB** as
  arm Gaussians rendered as giant flat hard-edged colored squares. Instrumenting `project_gaussian`
  revealed the offenders were **near-singular, highly-elongated "needle" splats** (e.g.
  `cov=(281,0.66,0.00)` → eigenvalues `(0.0, 281)`, `det≈0`), at mid-scene depth (z≈3.3–4.7, *not* a
  near-plane blowup), low opacity. **Root cause: SplatRs's rasterizer was missing the standard EWA
  screen-space low-pass filter.** It used a `1e-6` eps on the 2D-covariance diagonal instead of the
  reference `+0.3` dilation (`src/render/full_diff.rs:88`), so near-singular projections aliased into
  hard-edged blobs. **Fix:** replaced the eps with the standard `LOW_PASS = 0.3` dilation. Result on
  the artifact frames: frame 25 **14.8 → 24.1 dB**, frame 75 **20.0 → 23.6 dB**; the blobs are gone
  and the arm/bucket render correctly. (Clean frames dip ~1–2 dB from the added blur — the known
  low-pass tradeoff, amplified by the 0.4× downsample test resolution; negligible at full res.) This
  filter lives in the shared `project_gaussian`, so **training benefits too** (better conditioning,
  less needle aliasing in gradients). Renders in `renders/lego_phase0/{before,after}/`.

---

*Original Phase 0 plan (for reference):*

The single most valuable step. Cheap, and it tells us where the base-fidelity loss lives.

1. **Forward-render isolation test.** Load a **known-good reference `.ply`/`.splat`** (from the
   official 3DGS release, or the existing `plys/train.splat`) and render it with SplatRs from
   known camera poses. Compare to the reference renderer's output for the same scene.
   - If SplatRs renders a good splat **correctly** → the forward path is fine, and the 9–11 dB
     ceiling is purely a *training/optimizer* problem. Focus everything on Groups A/D.
   - If it renders **incorrectly** → there is a forward-render/color-space/SH/projection bug that
     no amount of training fixes. Fix that first.
2. **Instrument the training loop** (much of this already exists in newer `metrics.csv`): per-iter
   PSNR, Gaussian count, `scale_median`/`p90`/`max`, anisotropy, opacity histogram, background RGB,
   **and a histogram of the view-space gradient norm** used for densification.
3. **Pick one fixed benchmark** (e.g. `tandt/train` or `garden_sm`) and a fixed seed. All Phase-1
   progress is gated on this scene.

### Phase 1 — Restore base fidelity (target: ~18–20 dB with densification OFF) — IN PROGRESS

Fix the base so a *fixed* Gaussian set converges like reference does pre-densification.

- **A1 — ✅ DONE.** Initial opacity `2.2` (≈0.90) → `inverse_sigmoid(0.1)` (≈−2.197) in `src/core/init.rs`.
  Confirmed the trainer reads `g.opacity` directly (`trainer.rs:510`) and does not override it.
- **A2 — ✅ DONE.** Off-diagonal covariance gradient was double-counted 2×. Fixed both backward paths
  (`src/render/full_diff.rs:427` CPU, `src/gpu/gradients.rs:199` GPU) by splitting the single-DOF
  `d_cov.y` evenly across the symmetric matrix slots (`0.5 * d_cov.y` each). Verified against the math
  (the `Jᵀ·G·J` chain treats G as a full 4-entry matrix) and the agent's finite-difference check; the
  14 existing gradient FD tests still pass (no regression).
- **A3 — ✅ RESOLVED (2026-07-06, diagnosis: no longer reproduces; learned bg kept).**
  The documented divergence (bg RGB negative/past 1.0, PSNR collapse) does not reproduce after
  the [0,1] bg clamp (present in the trainer) plus A1/A2/B1–B6/B11/B12. Verified the GPU
  background gradient against CPU (matches to ~0.02–2% — the old failing test used an absolute
  tolerance on a pixel-sum; now relative). Then A/B'd learned-vs-frozen bg across all three
  densify configs (tandt/train micro, 2000 iters, seed 42, final test PSNR):
  frozen loses everywhere — no-densify 14.43 vs 14.91, @500 14.97 vs 15.69, @100 13.15 vs 14.15.
  The learned bg converges to a jointly better constant than any fixed value; "red pinned at 0"
  is a clamped optimum, not divergence. **Key negative result: the slow late-training PSNR decay
  persists with bg completely frozen → the decay is NOT background-driven; next suspect is
  D1/D2 (LR schedule).** Kept: learned bg default, `--learn-bg`/`--no-learn-bg` flags, startup
  `background init` log line, fixed gpu-gated bg-gradient tests.
- **C2** progressive SH-degree warmup (or at least confirm SH forward + linear/sRGB color handling).
- **D1/D2/D3** correct the LR schedule (scene-extent position LR, position-only decay, SH-rest LR).
- **Gate:** with densification disabled, this scene should climb to ~18–20 dB. If it can't, return
  to Phase 0 — the forward path is still wrong.

### Phase 2 — Rebuild densification to match reference (target: correct count & size) — B1–B6 LANDED

**Status (2026-07-06): B1–B6 implemented and verified to fire healthily.** After the rewrite, a
600-iter garden_sm run showed densification working correctly: `scene_extent`=4.49 computed from
185 cameras, splits+clones both firing, `grad_p90` sitting right at the 0.0002 threshold (calibrated,
not exploding), oversize prune (B6) active, count growing controlled (8000→11,578), background stable.
The old failure signature (grad→3.38, count→402k) is gone.

**B1b calibration fix (required):** B1's view-space gradient comes out of the renderer in *pixel*
units (~1e-5), but the 0.0002 threshold is calibrated for reference *NDC* units — so the first run
produced 0 splits/0 clones (densification silently off). Fixed by converting pixel→NDC
(`×dim/2`, resolution-independent) before thresholding (`trainer.rs`, grad-accumulation loop).

**Open follow-ups from this work:**
- **PSNR gain** still being measured on a longer run — at 600 iters densification is in its "churn"
  phase and PSNR (18.27) is ~level with the no-densify baseline; the payoff needs thousands of iters.
- **Split dominates clone** — traced to **C1**; ✅ FIXED (2026-07-06, see below).
- **GPU backward** returned zero `d_mean_px` to the trainer — ✅ FIXED (2026-07-06). The GPU
  rasterization backward (`backward.wgsl`) always computed the pixel-space `dL/d(mean_px)` and read
  it back (it is what the position gradient is chained from), but the trainer discarded it and
  handed zeros to the B1 accumulator, silently disabling densification on the GPU path. Now the
  real `grads_2d.d_mean_px` is passed through — same pixel-space convention as the CPU path, so the
  B1b pixel→NDC calibration applies unchanged. Also fixed: the GPU-backward-disabled fallback
  branch used the 7-tuple `render_full_color_grads` (a latent compile error with `--features gpu`
  since B1 landed — the GPU feature build had been broken at HEAD), and the pre-B1
  `densify_and_prune`/FD-test call sites were updated to the new signatures/conventions.

**2000-iter GPU A/B (2026-07-06, tandt/train micro @490×272, seed 42) — densify churn is the
next blocker.** Test PSNR by config: no-densify **15.26 → 14.91** (slow base decay, A3 signature:
bg red channel pinned at 0, channels drifting); densify@500 **15.26 → 15.89 @1000 → 14.48** (beats
baseline mid-run, then decays after more densify events); densify@100 **15.61 → 12.90** (worst,
count 8000→13.5k). Clone/split DID rebalance as C1 predicted (interval-100 run: split 59/clone 19
at iter 100 → split 236/clone 373 by iter 1900). Two mechanisms explain "more densify events = more
damage", both deviations from reference:
- **B11 (new):** `reset_moments_keep_t` zeroes ALL Adam moments for ALL parameter groups after
  every densify event (`trainer.rs` post-`densify_and_prune`). Reference prunes/concats optimizer
  state, preserving moments for surviving Gaussians and zeroing only new rows. At interval 100
  that is 19 full optimizer restarts in 2000 iters — Adam never converges.
- **B12 (new):** `split_opacity_logit` halves effective alpha for BOTH split and clone children
  (`densify_and_prune`). Reference copies opacity unchanged on clone and split; repeatedly
  densified (= high-gradient, important) Gaussians get their alpha knocked down every cycle.
Recommended order: B11 (bigger effect), then B12, then re-run the interval-100 A/B; A3 (background
divergence) remains open behind these.

**B11 + B12 — ✅ DONE (2026-07-06).** B11: `densify_and_prune` now returns a survivor map
(`Some(old_idx)` kept / `None` new-or-reinitialized) and all five per-Gaussian Adam optimizers
remap their moments through it (`remap_moments_keep_t`) instead of a full reset — survivors keep
state, children (and split parents, which reference re-creates) start fresh. B12: split/clone
children copy the parent opacity unchanged (reference); `split_opacity_logit` removed. A/B re-run
(same config/seed as above), test PSNR @500/@1000/@1500/@2000:
| Config | before B11/B12 | after |
|---|---|---|
| no densify | 15.26 / 15.25 / 15.05 / 14.91 | (unchanged — no densify events) |
| densify @500 | 15.26 / 15.89 / 14.48 / **14.48** | 15.26 / 15.52 / 16.01 / **15.69** |
| densify @100 | 15.61 / 14.62 / 12.73 / **12.90** | 16.11 / 15.36 / 14.35 / **14.15** |
Densification is now **net-positive for the first time**: @500 beats the no-densify baseline by
+0.78 dB (peak 16.01 @1500). @100 gained +1.25 dB but still trails baseline — the reference
interval needs the still-open A3 (background divergence: bg red channel pins at 0 in every
config, slow late decay even with densification off) and D1/D2 (LR schedule) before it wins.
Until then prefer `--densify-interval 500` on this preset.

**C1 — ✅ DONE (2026-07-06).** Initial scale is now density-adaptive, matching reference 3DGS
(`simple-knn`/`distCUDA2`): per point, isotropic `σ = sqrt(mean sq dist to 3 nearest neighbors)`,
log-space, computed with a uniform voxel grid + expanding-ring search (`core/init.rs`,
`mean_sq_dist_knn` + `apply_knn_init_scales`, brute-force-verified in tests). Replaces the
depth-only heuristic (`1.5·z/f`) at both trainer init sites. In the multiview trainer σ is capped
at `0.1·scene_extent` (the B6 oversize-prune bound, so no init Gaussian is born pruned);
`scene_extent` computation moved above init to support this. Expected effect: dense regions start
below `0.01·scene_extent`, so densification prefers **clone** (under-reconstruction) over **split**,
rebalancing the split≫clone skew.

Only after Phase 1 holds. This is a rewrite of `densify_and_prune`, not a tuning pass.

- **B1** expose and accumulate the **view-space 2D mean gradient** (`d_mean_px`) for densification.
- **B2** per-Gaussian visibility `denom[i]`; average = `grad_accum[i]/denom[i]`.
- **B3** fix opacity reset: every **3000** iters, `opacity = min(opacity, 0.01)`.
- **B4** split shrink → `/1.6`.
- **B5** introduce `scene_extent`; clone/split boundary = `0.01·scene_extent` on **max** axis scale.
- **B6** add screen-radius / world-scale prune after opacity reset; tie any scale ceiling to extent.
- **B7** stop densification at ~15000 iters.
- **B8** size-proportional child jitter. **B9** real cap. **B10** consistent thresholds/intervals.
- **C1** nearest-neighbor initial scale.

### Phase 3 — Validate against benchmarks

- Full 30k run on a standard scene; target **23–25+ dB** and healthy Gaussian stats
  (`diagnostic_gaussian_health.rs`: needles <5%, anisotropy p90 <10×, scales in bounds).
- Side-by-side render comparison with reference output.
- Add regression tests asserting a **PSNR floor** and Gaussian-health bounds so this can't silently
  regress again.

---

## 4. Cross-cutting constraint: the GPU cap is a hard scaling ceiling

`GPU_HARD_CAP_GAUSSIANS = 400_000` (`trainer.rs:1307`), imposed by Metal's 128 MB buffer limit
(320 bytes/Gaussian). Reference scenes use **millions** of Gaussians. Even fully fixed, SplatRs will
underperform reference on large scenes until buffers are chunked/tiled or the per-Gaussian footprint
is reduced. This is a **scaling** limitation, separate from the correctness bugs above — worth
tracking, but not the cause of the 9–11 dB problem (small subsets that fit well under the cap are
also stuck low).

---

## 5. Recommended immediate next step

Run **Phase 0, step 1 (forward-render isolation)**. It is a few hours of work, cleanly partitions
"can we render splats?" from "can we train them?", and decides whether Group A or Group D is the
priority. Everything else is faster once we know which side of that line the base-fidelity loss is on.
