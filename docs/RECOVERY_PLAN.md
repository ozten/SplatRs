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
- **D1/D2 — ✅ DONE (2026-07-06).** Position LR is now `lr_position · scene_extent` (reference
  `spatial_lr_scale`) with a position-ONLY log-linear decay of 100× over the reference 30k-step
  horizon; all other parameter groups hold constant LR (the old code decayed all six groups 10×
  per run). **Verified — this was the cause of the late-training PSNR decay.** A/B
  (tandt/train micro, 2000 iters GPU, seed 42, test PSNR @500/@1000/@1500/@2000):
  | Config | before D1/D2 | after |
  |---|---|---|
  | no densify | 15.26 / 15.25 / 15.05 / **14.91** | 15.72 / 16.84 / 17.00 / **16.33** |
  | densify @500 | 15.26 / 15.52 / 16.01 / **15.69** | 15.72 / 16.73 / 16.46 / **15.49** |
  | densify @100 | 16.11 / 15.36 / 14.35 / **14.15** | 16.34 / 15.90 / 14.19 / **12.47** |
  Base fitting now CLIMBS (peak 17.0) instead of decaying — +1.4 dB final on no-densify.
  **New bottleneck exposed: densification now hurts relative to the improved baseline.** At
  interval 100 there is a clone runaway (clones/cycle grow 200→542 by iter 1900, count →14.3k,
  bg driven to (0,0,0), PSNR →12.5): faster positions produce larger view-space gradients, more
  Gaussians cross the 0.0002 clone threshold each cycle, and added capacity feeds back. Next
  candidates: **B7** (densify stop horizon — reference stops at 15k/30k; we densify to the end
  of the run), revisiting the B1b pixel→NDC threshold calibration under the corrected LRs, and
  the not-yet-firing B3 opacity reset (interval 3000 > these 2000-iter runs, which is
  reference-consistent but means alpha inflation from cloning goes uncorrected).
- **D3** SH-rest LR (rest bands at DC/20) — still open, needs per-band LR in `AdamSh16`.
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
- **B7 — ✅ DONE (2026-07-06).** Densify only during the first half of training (`≤ iters/2`,
  proportional port of reference 500–15000/30k). Motivated by the 15k validation run
  (interval 500): reset-bounded sawtooth with decaying envelope — opacity resets recovered
  +0.6/+2.85 dB but between resets PSNR fell faster each cycle as additions accelerated
  ~100→550/cycle (count →12.6k, PSNR 16.7→12.8 by 7500; run stopped there). With B7 at 2000
  iters: @500 15.49→16.02, @100 12.47→14.90 final — the post-densify collapse is gone.
- **B8** size-proportional child jitter. **B9** real cap. **B10** consistent thresholds/intervals.

**Full 15k validation run with the complete fix stack (2026-07-06, micro @490×272, interval 500,
seed 42): final 14.47 dB, count 12,593.** Trajectory: 15.72@500 → **16.73 peak @1000** → slides
through the densify phase to 12.78@7500 (last densify event) → settle phase sawtooths with the
3000-iter opacity resets (12.23 → 14.05 → 12.47 → 14.92) → 14.47 final. B7's settle phase
recovered +1.7 dB from the trough (pre-B7 the envelope was still falling at 7500), but the run
ends below the 2000-iter no-densify baseline (16.33) and below its own iter-1000 peak.
**Verdict: the densify phase digs a hole the settle phase only partly climbs out of, and even
pure optimization degrades between opacity resets (12.78→12.23 with zero densify events) — an
over-opacity equilibrium that the resets only mask.** Leading suspects, in order:
1. ~~**Loss: micro trains with L2**~~ — **REFUTED (2026-07-06).** Added a `--loss l2|l1dssim`
   CLI override and re-ran the trio with L1+DSSIM: no-densify 16.24 (vs 16.33), @500 15.71
   (vs 16.02), @100 13.71 (vs 14.90) — slightly worse everywhere at this horizon, and the
   background still gets dragged to pure black. The dark-drift/over-opacity dynamic is NOT
   loss-driven. The **coverage loss weighting** (covered 1.0 / uncovered 0.5, a SplatRs-only
   deviation) was the next suspect — **also refuted (2026-07-06), in an unexpected way: it was
   INERT.** Replacing it with uniform weights (reference behavior) reproduced every run
   bit-for-bit (train losses identical to 6 decimals) — with 8000 C1-sized Gaussians the
   coverage mask saturates, so every pixel already weighed 1.0. The uniform-weights code was
   kept (reference-faithful, removes a periodic coverage computation from the training loop;
   verified zero behavior change). The bg→black drag in densify runs therefore reflects the
   *optimizer's actual preference* as capacity grows — remaining explanation: densified
   Gaussians progressively cover sky pixels, and bg then fits darker residual regions. The
   quality gap itself points back at **densify calibration (B1b under D1 LRs)** as the one
   live lever.
2. **B1b threshold recalibration** — partially confirmed (2026-07-06): sweeping the threshold
   at interval 100 (2000 iters, 15 train views) improves monotonically — 0.0002→14.90,
   0.0004→15.72, 0.0008→15.97 — but the train loss at 0.0008 (0.028) vs no-densify (0.121)
   exposed the real story: densification was fitting the train views far better while testing
   worse ⇒ **overfitting the tiny train set**, not a calibration bug per se.
**Phase-3 15k runs at 100 views (2026-07-06/07) — capacity-to-data ratio is the binding
constraint.** Three 15k runs (100 images, interval 100, seed 42) after the cap fix:
- **Densify-cap bug found & fixed** by the first run — the cap compared the rebuilt array's
  *running length*, so survivors appended after children overshot it every cycle and
  `cap.max(before)` ratcheted it upward compounding ~5–8%/cycle (207k→259k past a 200k cap,
  headed for the 400k Metal buffer limit). Now enforced as an addition budget; unit-tested.
- Capped at 200k, resets throughout: **14.02** final (settle phase climbed 13.58→14.40 but the
  iter-12000 reset ended the run mid-recovery).
- Capped at 200k, resets gated to the densify window (reference behavior, B3 follow-up commit):
  **12.76** final — reset-free settle *overfits* (train loss ↓ while test PSNR ↓, 14.16→12.76).
  The resets had been acting as accidental regularization.
**Conclusion: at 200k Gaussians × 75 half-res views (~10M pixel constraints), supervision per
Gaussian is 2–5× thinner than reference conditions; the settle phase overfits regardless of
schedule. The schedule fixes are reference-correct; the *config* isn't reference-like yet.**
- Capped at **60k**, reference schedule (2026-07-07): **14.25** final — best of the three, and
  the only settle phase that *stabilized and turned upward* (oscillating 14.2–14.5) instead of
  declining. Confirms the capacity story directionally: 60k + reference schedule beats 200k
  with any schedule. All 15k runs still trail the 2000-iter run's 15.33 on the same test set —
  long-horizon payoff needs reference-like data.
**Next levers: all 301 images (`--max-images 0`) and/or fuller resolution (`--downsample`),
with cap ~60–100k. Note the Metal 128 MB buffer limit when raising resolution+count together.**

**All-views run (2026-07-07, 301 images = 225 train/76 test, 100k cap, 15k iters): final
13.45** (initial 11.53; test set is broader/harder than the 100-view runs', so not directly
comparable). Same shape as every capped run: climb to ~13.4 during densify, then the settle
phase OSCILLATES 12.6–13.4 while train loss improves — even at the healthiest supervision
ratio yet (~300 pixel constraints per Gaussian, near reference). **The settle-phase ceiling
has now survived every capacity/data ratio, both losses, and all schedules → the remaining
suspect is representational: the per-step scale ANISOTROPY clamp (`MAX_LOG_ANISOTROPY = 1.6`,
trainer step loop) drags the smaller axes toward the largest EVERY iteration, so Gaussians can
never flatten into the thin surface-aligned splats reference relies on (reference anisotropy
is routinely 10–100×), and the clamp acts as a standing scale-inflation force. Its companion,
the needle prune (anisotropy > 2.0 → prune), reinforces it. Both were added to fight needle
artifacts whose actual root cause (missing EWA low-pass) was fixed in Phase 0 — they are
legacy double-medication. NEXT EXPERIMENT: relax/remove the per-step anisotropy pull and the
needle prune (keep the low-pass), A/B on the 2000-iter trio first.**

**Anisotropy-clamp 2000-iter trio A/B (2026-07-07) — REFUTED at this horizon, with a twist.**
Both clamps are now config knobs (`MultiViewTrainConfig::max_log_anisotropy` /
`needle_prune_log_anisotropy`, 0 = off; CLI `--max-log-aniso`, default 1.6 = legacy, needle
prune at +0.4). Control (clamped) reproduced the documented baselines exactly
(16.33 / 16.03 / 14.90), validating the refactor. Unclamped, test PSNR @2000:
| Config | clamped (1.6/2.0) | unclamped | Δ |
|---|---|---|---|
| no densify | 16.33 | 16.00 | −0.33 |
| densify @500 | 16.03 | 15.44 | −0.59 |
| densify @100 | 14.90 | 14.60 | −0.30 |
The twist: removing the clamp changed almost nothing structurally — `aniso_median` stays 1.0
and p90 ≤1.6 in BOTH arms (the population remains isotropic; only a tail of a few Gaussians
flattened, to 16–132×), `scale_median`/`opacity_median`/bg are identical between arms, and the
trailing train loss is slightly WORSE unclamped (0.125/0.117/0.118 vs 0.125/0.113/0.111) — so
the extreme-tail needles mildly hurt optimization overall rather than overfitting. Two
conclusions: (1) the "standing scale-inflation force" mechanism is refuted — at 2k the clamp
binds almost nowhere; (2) the real anomaly is that our Gaussians do not flatten EVEN WHEN
ALLOWED TO — reference splats develop strong anisotropy early, ours stay isotropic
(possible next suspects: scale-gradient path, rotation coupling, or simply horizon). The 2k
screen cannot see the settle-phase (7.5k+) where the hypothesis lives, so the decisive run is
15k @100 images/60k cap unclamped vs the 14.25 control (`runs/20260707_1127_micro`) — launched
as `runs/ab_aniso15k_free_60k`. Clamp stays default-ON (1.6) pending that result.

**15k settle-phase A/B (2026-07-07, `runs/ab_aniso15k_free_60k`) — CONFIRMED: the clamp was a
real settle-phase ceiling; unclamped is now the default (`--max-log-aniso 0`).** Same config as
the 14.25 control (100 images, interval 100, 60k cap, seed 42), unclamped: **settle mean
(8000–15000) 14.47 vs 14.21 (+0.26 dB)**, ahead at 21/30 validation points, peak 15.75 vs
15.13 (@3000), final 14.36 vs 14.25. Structure did exactly what the hypothesis predicted at
this horizon: unclamped p90 anisotropy reaches 21× (median 2.2, max ~29,000) vs the control's
p90 pinned at the 4.9 ceiling from iter ~2400 — at 15k scale the clamp binds the whole top
decile, which is why the 2k trio (population still isotropic) couldn't see the effect. No
needle artifacts observed — the EWA low-pass is a sufficient defense (a ~29,000:1 tail exists
but does not hurt PSNR; if it ever does, `--max-log-aniso` and the derived needle prune remain
available as knobs). Horizon trade-off: unclamped costs ~−0.3 dB on 2000-iter runs — 2k-trio
baselines under the new default are 16.00 / 15.44 / 14.60 (vs clamped 16.33 / 16.03 / 14.90);
use `--max-log-aniso 1.6` to reproduce legacy short-run numbers. Remaining
(clamp-independent) pathologies, now the top suspects for the residual ceiling: bg drags to
black mid-densify in BOTH arms, median opacity pinned at the 0.01 floor in BOTH arms (half the
population sits near the prune threshold — dead capacity), and late-settle convergence
(deltas fade after ~13k). Next: D3 SH-rest LR, C2 SH warmup, and the opacity-floor pileup.

**Opacity-floor pileup investigation (2026-07-08) — ROOT CAUSE FOUND, and it is the biggest
structural deviation yet: `MAX_CONTRIBUTIONS_PER_PIXEL = 16` in the GPU backward
(`rasterize.wgsl`/`backward.wgsl`).** The forward pass blends ALL contributors per pixel
(image is correct), but only the FIRST 16 non-culled Gaussians per pixel (front-to-back) are
recorded for the backward pass — every Gaussian at depth rank >16 in every pixel it touches
receives EXACTLY ZERO gradient for ALL parameters. Reference 3DGS has no such cap: its
backward re-traverses the sorted list per tile back-to-front, recomputing weights on the fly,
so every contributor down to the T<1e-4 cutoff gets exact gradients with O(1) memory/pixel.
Evidence (`examples/opacity_audit.rs` + `render::full_diff::debug_contrib_stats`, run on the
final `ab_aniso15k_free_60k` model):
- **88.4% of Gaussians (52,978/59,901) sit BIT-EXACTLY at the B3 reset-cap logit
  `inverse_sigmoid(0.01)`** — zero opacity gradient in the 9,000 iters since the iter-6000
  reset. Not a visibility problem: they are in-frustum in 46/75 train views on average (0%
  in zero views), median 3σ footprint ~22 px.
- **Per-pixel contributor counts: p50=209, p90=457, max=1168 — 99.4% of pixels exceed the 16
  slots** (order of magnitude beyond the cap).
- Simulating the GPU recording over 8 train views: only **1.9%** of at-floor Gaussians are
  ever recorded in even one pixel (vs ~21% for healthy ones over the same 8 views) —
  depth order is nearly static, so the starved set is permanent.
This one constant mechanistically explains every open pathology:
1. **Opacity pileup**: resets knock everyone to 0.01; only depth-rank ≤16 re-earn opacity;
   the rest freeze forever (and at 0.01 they sit ABOVE the 0.005 prune threshold, so they
   can never be pruned either — and pruning stops at iters/2 anyway).
2. **Fog / over-opacity equilibrium**: the 53k zombies at α=0.01 stack ~200 deep — a
   permanent gray veil (0.99^200 ≈ 0.13 transmittance through the zombie film alone) that
   gradient descent cannot remove because the veil gets no gradient.
3. **bg→black**: the backward computes the background gradient with `t_final` from only the
   ≤16 recorded contributions (t_final ≈ 0.85 at α=0.01) instead of the true forward
   transmittance (≈0.1 or less) — d_bg is overestimated ~10×, and the residual owned by the
   190+ unrecorded contributors per pixel is misattributed to the background, which rails to
   black trying to cancel the fog.
4. **Train-improves/test-stalls & the settle ceiling**: effective trainable capacity is the
   ~4.6% (2.8k) healthy front Gaussians, not 60k.
5. **Why tests never caught it**: the CPU backward (`render_full_color_grads`) is UNCAPPED —
   CPU/GPU gradient parity only holds on scenes with ≤16 overlaps, which is exactly what unit
   tests use. All real training runs use the GPU path.
Even the 2k-trio observation (median opacity frozen at the 0.1 init) is this: half the
population is never in any pixel's front-16 from iteration 0.
**RECOMMENDED FIX: rewrite the GPU backward to the reference scheme** — drop the
intermediates buffer entirely; store per-pixel true final T (+ contributor count) in the
forward, then walk the same sorted order back-to-front in the backward, recomputing
weight/alpha per Gaussian (deterministic from Gaussian+pixel) and reconstructing T_i
incrementally. This removes both the starvation and the d_bg bias, and shrinks GPU memory
(the 34 MB intermediates buffer goes away). Raising the slot count instead CANNOT work: 64
slots ≈ 136 MB already exceeds the 128 MB Metal buffer limit while still truncating at
p50=209. Secondary (after the backward fix): reconsider reset floor (0.01) vs prune
threshold (0.005) interplay, since zombie mass parked between them is unprunable.

**GPU backward rewrite — ✅ DONE (2026-07-08, commit 15f7d9d), plus a SECOND major bug found
during verification: the GPU projection shader NEVER HAD THE EWA LOW-PASS.** Changes:
- Forward (`rasterize.wgsl`) stores per-pixel `(final transmittance, last blended sorted
  index)` — 8 B/px instead of 256 B/px; backward (`backward.wgsl`) re-walks the sorted list
  back-to-front from that index, re-applies the forward's exact tests, recomputes alpha, and
  reconstructs `T_i = T_{i+1}/(1−a_i)`. Every contributor receives gradients; the gradient
  math itself is unchanged. `d_bg` now uses the true final T (was ~10× overestimated on dense
  pixels). Dead tiled-backward path deleted. Backward cost 31→44 ms/iter @8k Gaussians.
- **Low-pass:** the 0.3 covariance dilation (Phase 0's anti-needle fix) existed only in the
  CPU projection (`full_diff.rs`); the GPU shader added 1e-6. ALL GPU training to date ran
  without it — and without the backward fix's gradients, this went unnoticed because the
  long-broken CPU/GPU parity test (`unit_gpu_gradients_smoke`, compile-stale since the
  `disable_sh` signature change) never ran. Both projections now match (forward parity 6e-5).
  Caveat for interpreting the anisotropy A/Bs above: "EWA low-pass is a sufficient needle
  defense" was concluded from GPU runs that had NO low-pass — needles were rare even
  undefended; with the low-pass actually active the case for unclamped is stronger.
- New regression test `unit_gpu_deep_blend_gradients` (40 stacked Gaussians): on the old
  backward exactly ranks 16..39 get zero gradient; new backward matches the uncapped CPU
  backward at all ranks. Verified the test FAILS on pre-fix HEAD.
- `auto_downsample` now models the real 16 B/px ceiling — tandt (980×545) trains at FULL
  resolution by default. Pass `--downsample 0.5` to reproduce the resolution of all runs
  documented above.
- **First signal (200-iter smoke, no densify, seed 42, 0.5 downsample): test 18.98 dB — +2.3 dB
  above the best value this config ever reached at ANY horizon — and bg settles at sky-blue
  instead of black.** All prior PSNR numbers in this document predate these fixes and are
  superseded as baselines.

**Post-fix 2000-iter trio (2026-07-08, seed 42, --downsample 0.5, unclamped default):**
| Config | pre-fix | post-fix | Δ |
|---|---|---|---|
| no densify | 16.00 | 17.54 | +1.54 |
| densify @500 | 15.44 | **18.91** | +3.47 |
| densify @100 | 14.60 | **18.86** | +4.26 |
The gain grows with densify frequency — densified populations were the most cap-starved.
Densification is now decisively positive (+1.4 over no-densify) and arrests the no-densify
arm's overfit decay (19.1@500→17.5@2000 on 15 train views; densify arms plateau ~18.9).
Population health @2000 (d100): opacity median 0.23 and only 24% below 0.1 (pre-fix: median
frozen at the 0.10 init, 97% below 0.1 at 15k), bg sky-blue (0.00,0.24,0.51), anisotropy
p50/p90 = 3.2/22 (reference-style flattening already at 2k), count 8k→24.8k. 15k validation
run (100 img / interval 100 / 60k cap / 0.5 downsample) launched as
`runs/bwfix_15k_100img_60k` — pre-fix control: 14.25 clamped / 14.36 unclamped.

**15k validation (2026-07-08, `runs/bwfix_15k_100img_60k`): final 16.25 dB — +2.0 dB over the
pre-fix controls (14.25 clamped / 14.36 unclamped), settle mean 15.99 vs 14.47/14.21, peak
16.97 — and the settle phase CLIMBED (15.43 @7500 → 16.25 @15000), the first monotonically
improving settle in the campaign.** The settle-phase ceiling that survived every capacity,
loss, schedule, and clamp experiment is gone: it was gradient starvation all along.
Post-fix opacity audit on the final model: trained fraction (moved off the reset cap) 11.6%
→ 31.9%; healthy+strong (>0.1) 4.6% → 11.0%; still 68.1% bit-exact at the reset cap — the
remaining zero-gradient mechanism is the T<1e-4 early-termination (reference-consistent:
occluded Gaussians legitimately get no gradient; per-pixel contributors are now p50=336).
Remaining follow-ups, in order: (1) the reset floor (0.01) still sits ABOVE the prune
threshold (0.005), so occluded never-recovering mass stays unprunable — reference has the
same mismatch, but with 68% of the population parked there a deviation (reset to <0.005, or
prune-on-visibility) is worth an A/B; (2) anisotropy p90 reached 212 — render-safe under the
low-pass, but watch for view-dependent needle artifacts; (3) full-resolution training is now
memory-feasible (auto-downsample no longer forces 0.5) — the next big data lever; (4) D3
SH-rest LR, C2 SH warmup.

**Critical visual finding on the bwfix models (2026-07-08): PSNR hides needles.** Renders of
`bwfix_15k_100img_60k` show real structure (locomotive "713" legible) but are dominated by
needle streak artifacts (anisotropy p90 = 212×), much worse on novel views. The earlier
"unclamped anisotropy wins" A/B (settle-mean +0.26 dB) was measured on a gradient-starved
population AND on a GPU that lacked the EWA low-pass — it is invalid post-fix. Always eyeball
renders alongside PSNR (`sugar-render --model <run>/model.gs --camera-id N --dataset-root
datasets/tandt_db/tandt/train --out x.png`).

**Aniso-clamp A/B, post-backward-fix (2026-07-08): `--max-log-aniso 3.0` (≈20:1 pull,
needle prune auto at 3.4) vs unclamped default. 2k trio (`runs/ab_aniso3_d{0,500,100}` vs
`ab_bwfix_d{0,500,100}`):**
| Config | unclamped | clamp 3.0 | Δ | aniso max (unc → clamp) |
|---|---|---|---|---|
| no densify | 17.54 | 17.70 | +0.16 | 4291 → 20.1 |
| densify @500 | 18.91 | 18.98 | +0.07 | 3927 → 20.1 |
| densify @100 | 18.86 | 18.86 | ±0.00 | 8240 → 20.1 |
PSNR neutral-to-positive (the legacy 1.6 clamp cost −0.3 dB at 2k; 3.0 costs nothing),
aniso_max pinned at e^3.0 = 20.1, medians/p90 untouched (clamp only bites the extreme tail),
and the d100 test-view render loses most of its radiating needle streaks vs control.
15k validation at the standard config launched as `runs/ab_aniso3_15k_100img_60k`
(control: `bwfix_15k_100img_60k` = 16.25 dB, aniso p90 212).
15k result: 16.17 vs 16.25 final (settle mean 15.96 vs 15.95, settle peak 16.59 vs 16.25) —
a statistical tie on PSNR with needles pinned at 20.1 vs p90=212/max=320k. **But both arms
ran on a broken renderer — see below; the clamp decision moves to the post-fix re-baseline.**

**THREE GPU FORWARD BUGS — found 2026-07-08 while eyeballing the aniso A/B renders, fixed
in commit 836f4d0. ALL prior GPU PSNR numbers (including everything above) ran on a broken
renderer and are superseded.** Discovery chain: sugar-render (CPU forward) drew the
bwfix_15k model as structureless fog while the trainer's own test-view render (GPU forward)
showed structure+needles → `examples/render_compare` (new diagnostic: renders a saved model
through both paths + a CPU replica of the GPU pipeline) measured mean abs pixel diff 0.14 on
the same model+camera → bisected to three GPU-side bugs:
1. **The bitonic depth sort never sorted** (`gpu/sort.wgsl`): the ascending/descending bit
   used bit `stage` of the index instead of bit `stage+1`, violating the bitonic invariant
   from stage 0; additionally, compare-exchanges reaching into the power-of-two pad region
   were skipped instead of treated as +∞ sentinels. Measured on the real 60k model: **45.1%
   adjacent inversions — GPU training composited in near-random depth order for the entire
   campaign.** (Unit test used N=2, which even the broken network sorts.) Fix: correct
   direction bit; projected buffer padded to 2^k with +∞-depth sentinels written by the
   projection shader. Regression test `unit_gpu_sort_order` (300 shuffled translucent
   Gaussians) fails pre-fix, passes post-fix.
2. **No 3σ bounding box on GPU** (`rasterize.wgsl`/`backward.wgsl`): every pixel accumulated
   the 3–6σ tails of ALL splats (alpha at 3σ ≈ 0.011·opacity > the 1e-4 skip), where the CPU
   renderer and reference 3DGS truncate at the 3σ box. Fix: projection stores the 3σ radius
   in the free cov.w slot; rasterize and the backward re-walk apply the CPU's exact bbox test.
3. **The forward projection used Rᵀ for the Gaussian rotation** (`shaders.rs quat_to_matrix`
   passed matrix ROWS to WGSL's column-major mat3x3 constructor) — while
   `project_backward.wgsl` used the correct R. So for every anisotropic Gaussian, the
   world covariance was built with the transposed rotation AND rotation/scale gradients were
   inconsistent with the rendered forward all campaign. Identity-quaternion / isotropic unit
   tests are blind to this. Verified by replica: R with transpose matches old GPU to 1e-6.
After the fixes: **GPU forward == CPU forward to mean abs 3e-6 (max 4e-4) on the real 60k
model at a real camera.** sugar-render eyeballs are now trustworthy. Downstream implications:
- The "settle-phase climb" and all bwfix PSNR numbers were measured under random-order
  compositing with rotation gradients fighting the renderer; densification/clamp/floor
  conclusions must be re-validated. Post-fix re-baseline: `runs/ab_srt_{d0,d500,d100}` and
  `runs/ab_srt_a3_{d0,d500,d100}` (2k trio, unclamped vs clamp 3.0), then a 15k pair.
- Old saved models (.gs) encode rotations under the Rᵀ convention — they render differently
  (worse) on the fixed renderer; do not compare old model files against new renders.

**Post-renderer-fix 2k re-baseline (2026-07-08, seed 42, ds 0.5, micro/20 img):**
| Config | unclamped | clamp 3.0 | Δ | aniso max (unc → clamp) |
|---|---|---|---|---|
| no densify | 18.93 | 18.87 | −0.06 | 2827 → 20.1 |
| densify @500 | 17.92 | 18.53 | **+0.61** | 6771 → 20.1 |
| densify @100 | 18.66 | **19.49** | **+0.83** | 20295 → 20.1 |
19.49 is the best 2k result of the campaign (broken-renderer d100 pairs: 18.86/18.86 —
not comparable, they scored against their own broken compositing). On the fixed renderer
the clamp is a clear win wherever densification runs, neutral without it; aniso p90 is
naturally low (~15) post-fix — the unclamped max-tail (up to 20k:1) is what the clamp
removes. Renders (both d100 arms): locomotive fully legible ("713 WESTERN P"), no needle
streaks, no fog — best visual quality of the campaign at 2k/20 images; the clamped arm is
visibly crisper. d500 dipping below d0 at 2k is the known 15-train-view overfit artifact.
15k pair (`runs/srt15k_unclamped` vs `runs/srt15k_aniso3`, 100 img / 60k cap) launched —
default flip decision on its result.

**Post-fix 15k pair result (2026-07-08) + DEFAULT FLIPPED to `--max-log-aniso 3.0`:**
unclamped 16.68 final / 16.61 settle mean / 17.25 settle peak; clamp 3.0 16.37 / 16.56 /
17.04; clamp leads 9/16 settle evals — a statistical tie on PSNR. Post-fix the unclamped
tail is far milder than before (p90 37.6 vs 212 pre-fix — correct rotation gradients
naturally restrain needles) but still reaches max 161,000:1 vs 20.1 pinned. Renders: both
arms show the best 15k quality of the campaign ("713", ladder rungs, livery stripes all
legible on the test view); unclamped slightly sharper with visible thin streaks, clamped
cleaner and softer. Far-novel views (camera 150, outside the trained third of the orbit)
wash out identically in both arms — a data-coverage limit, not a clamp issue. DECISION:
default flipped to 3.0 (2k densify wins +0.6/+0.8, 15k tie, artifact tail capped 4 orders
of magnitude, cleaner renders); `--max-log-aniso 0` restores reference-faithful unclamped.
Both post-fix 15k arms hit the 60k cap by iter 2500 and their oversize-prune fires all
window (~10-25/cycle); the bg rails to black mid-settle in BOTH arms (returns @~10000) —
clamp-independent, still-open pathology (opacity median parked at the 0.010 reset floor,
88-89% below 0.1 — the reset-floor/settle-prune levers target exactly this).
New baseline for lever A/Bs: `runs/srt15k_aniso3` (16.37 final / 16.56 settle mean).

**Lever A/Bs vs that baseline (2026-07-08, one lever each, clamp 3.0 active):**
| Arm | final | settle mean | count | opac med | % < 0.1 |
|---|---|---|---|---|---|
| baseline (`srt15k_aniso3`) | 16.37 | 16.56 | 59,984 | 0.010 | 89.1 |
| `--opacity-reset-floor 0.004` | 16.62 (+0.25) | 15.08 | 59,931 | **0.142** | **45.2** |
| `--settle-prune-interval 500` | **16.76 (+0.39)** | 16.57 | 58,031 | 0.010 | 84.6 |
Both levers win on final PSNR. floor004 is the structural fix: the parked-at-0.01
unprunable mass is GONE (median opacity 0.142 vs 0.010; sub-floor mass gets pruned at the
next densify pass, and the freed capacity is re-densified — population stays off the cap
until late window). Cost: a deep mid-run dip (settle mean 15.08, train loss overfit
signature @10000) — but the arm finished AT its settle maximum, still climbing, so the
equilibrium is healthier and slower; longer horizons should favor it. sp500 behaves as
designed (settle prunes remove ~100-300/pass — sub-threshold opacity early, then mostly
oversize regrowth; needle prune never fires since the clamp prevents needles) and adds
+0.39 dB at zero health change (the 0.01-parked mass sits above the 0.005 prune threshold,
untouchable without floor004 — the levers are complementary by construction).
Next: combo run (`runs/srt15k_a3_floor004_sp500`) + 30k floor004 horizon test
(`runs/srt30k_a3_floor004`) launched overnight.

**Overnight results (2026-07-09) + `--settle-prune-interval` DEFAULT now 500:**
- Combo (floor004+sp500, 15k): 16.40 final / 16.03 settle mean / peak 16.98@5500 — the
  levers do NOT stack at 15k (floor004's dip dominates; settle prunes remove recovering
  mass). Health is good (opacity median 0.107) but PSNR trails sp500-alone.
- floor004 30k horizon test: peak 17.30@10500, settle(15k+) mean 16.38, final 16.05 —
  the 15k arm's "still climbing" did NOT extrapolate: long settle peaks mid-way then
  decays. AND the population re-parks at the new floor (median 0.004, 86% <0.1) because
  the final reset caps everyone and no pruning runs in settle without sp500 — floor004
  alone just moves the parking lot; only the combo keeps the population healthy.
- DECISION: `--settle-prune-interval 500` becomes the default (+0.39 clean win, A/B'd);
  `--opacity-reset-floor` stays 0.01 opt-in (helps health, mixed PSNR, dip + long-horizon
  decay). Best 15k config now: clamp 3.0 + sp500 = **16.76** (`runs/srt15k_a3_sp500`).
- The dominant remaining pathology across ALL arms: **PSNR peaks mid-run (~17.0-17.3
  during the densify window) then decays into the final** — present with and without
  levers, plus bg→black mid-settle in every arm. Next levers: D3 SH-rest LR / C2 SH warmup
  (SH overfit is a prime decay suspect), full-resolution training, all-views data.

**Full-resolution first attempt (2026-07-09, `runs/srt15k_a3_sp500_fullres`, 980×545,
same 60k cap/config as the 16.76 half-res run): 13.44 final — NEGATIVE at this capacity.**
Mechanism (CSV): cap hit by iter 2500 (opac med 0.222, healthy); the iter-3000 opacity
reset knocked the population to the 0.01 floor and it NEVER recovered — median 0.010 and
99.9% below 0.1 for the remaining 12k iters, PSNR flat at 13.4 (renders through a
permanent 0.01-opacity veil). At 4× pixels/Gaussian, post-reset re-earning is too slow at
this capacity. Full-res needs a proportionally larger cap (200-400k; GPU hard cap 400k)
and possibly gentler/earlier-only resets — NOT just the flag flip. Half-res 0.5 stays the
validated config for A/Bs.

3. **Micro-config confound — CONFIRMED as the root of "densification hurts" (2026-07-06).**
   Re-ran the A/B with `--max-images 100` (75 train / 25 test): densify@100 **beats** its
   no-densify baseline for the first time — 15.33 vs 15.10 final, count 8000→22,618 (~3×,
   reference-scale growth), PSNR climbing monotonically through iter 2000, train loss
   comparable to baseline (no overfit), **and the bg→black pathology disappears** (background
   settles at a sensible gray). With 15 train views, extra capacity memorizes the train set
   and test PSNR pays for it; with 100 views, multi-view consistency constrains the added
   capacity exactly as reference assumes. **Phase-3 validation must use ≥100 images**
   (`--max-images 100` or 0 = all); micro's default 20 stays for its fast-profiling purpose.
   Threshold 0.0002 + interval 100 (reference values) are healthy at real view counts.
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
