# Autonomous work plan — 2026-07-23 → 2026-07-26 (user AFK 3 days)

Standing goal: **maximize validated quality progress on SplatRs** while the user is away,
keeping the GPU saturated with one serial training arm at a time and doing code work in the
idle CPU hours. Every result is analyzed (PSNR + SSIM + LPIPS + visual render check),
recorded in RECOVERY_PLAN.md, committed, and pushed to main. Auto-memory updated at every
milestone so any future session can resume from cold.

## Priorities (in order)

**P0 — Metric instrumentation (no GPU).** Add SSIM to the multi-view eval (reuse the DSSIM
loss's SSIM implementation) as a new metrics.csv column; add a post-run LPIPS batch script
(scratchpad venv has lpips+torch) over the saved per-checkpoint renders. Rationale: the
L1+DSSIM re-test proved the harness is PSNR-blind — PSNR and perception split, and every
future A/B must be judged on all three + visual. Unit test + 2k smoke before relying on it.

**P1 — DSSIM settle re-tune.** The L1+DSSIM arm has the best window peak (17.51) and best
LPIPS (0.496) ever but decays −0.87 in settle under L2-tuned settings. One lever per arm vs
`runs/srt30k_optfloor05_l1dssim` as baseline, ~3h each on the micro/100-img/60k config:
  - reset-window-margin 2500 (resets end early)
  - margin 15000 (= no resets at all; DSSIM population is healthy without them — old
    "reset-free overfits" verdict was L2/pre-backward-fix era)
  - DSSIM-aware floor: --opacity-reset-floor 0.25 (≈2× cut for a 0.5-median population,
    mirroring what 0.05 is for L2)
  - then combine winners / iterate based on SSIM+LPIPS, not PSNR alone.
Success: a config whose settle holds or climbs on SSIM/LPIPS and beats LPIPS 0.496.

**P2 — Step 3: tile-binned rasterizer (code, interleaved with P1 GPU arms).** Per the
roadmap decision (RECOVERY_PLAN): golden parity/regression harness FIRST, naive renderer
kept as a switchable oracle. Order: (a) golden parity harness — render fixed models through
CPU + GPU naive paths, assert pixel parity + PSNR floors, runnable in minutes; (b) forward
tile-binned path behind a flag (tile lists, per-tile depth-sorted splats), parity-checked
against the oracle on real models; (c) backward path if (b) lands clean; (d) bench at 150k+
gaussians — the payoff is step-2's throughput ceiling. Use short GPU parity tests between
arms; never run heavy GPU work concurrently with a training arm.

**P3 — Showcase run (only after P1 yields a winner).** All-views interval-8 half-res 30k
with the best recipe (directly comparable to splatfacto's 22.21). Full-res only if timing
clearly allows and the recipe is stable — the standing "ask before full-res relaunch" note
applies; prefer the half-res showcase.

## Operating rules

- One training process at a time (Metal watchdog); launch detached (nohup); monitor each
  run's status file; always have the next arm decided before the current one finishes.
- Rebuild `--release --features gpu` before any launch (CPU-only-binary gotcha).
- Tests green before every push; main only; no force-push; no destructive ops.
- Analysis discipline: eval-row filtering (500-cadence), fresh same-binary controls when the
  binary changes, ALWAYS a visual render check + LPIPS alongside PSNR/SSIM.
- Delegate parallelizable code/analysis legwork to cheaper-model subagents; review GPU
  kernel changes personally.
- Push notification only for: a major result, a blocker, or the end-of-plan summary.
- If a run/session dies: everything needed to resume is in this doc, RECOVERY_PLAN.md §5–6,
  runs/*.status, and auto-memory.

## Log (append-only, newest last)

- 2026-07-23 ~07:30: plan written; GPU idle; starting P0.
- 2026-07-23 07:35: **P0 DONE** — `compute_ssim` (11×11 Gaussian window, unit-tested) wired
  into the multi-view eval as the `eval_ssim` CSV column (proxy rows log −1); 600-iter smoke
  verified (ssim 0.37 @600). `scripts/compute_lpips_run.py` batch-scores saved checkpoint
  renders. Committed + pushed. **P1 batch launched** (`scripts/dssim_settle_retune.sh`,
  ~9h serial): rg2500 → no-resets → floor25, all l1-dssim on the §6.1 winner config, judged
  SSIM+LPIPS+visual vs baseline `srt30k_optfloor05_l1dssim` (17.51 peak / 0.496 LPIPS /
  15.88 final). Next: P2(a) golden parity harness while the arms run.
- 2026-07-23 ~08:45: P2 spec committed (docs/TILE_RASTER_PLAN.md, source-verified design).
  Part A (golden harness) delegated to a subagent (tests/golden*, compute_psnr pub
  promotion, .gitignore carve-out — report pending). **Tile raster Stage 0 DONE** by hand:
  TileGaussianPair + src/render/tile_math.rs oracle (7 tests; discovered lib.rs gates the
  whole gpu module — stub is dead code — so CPU-visible code cannot live under src/gpu/).
  P1 arm 1 (rg2500) still training. Next: review harness agent's work when it reports;
  then Stage 1 (GPU counting kernel vs the oracle, exact-match gate).
- 2026-07-23 ~09:40: **P2(a) golden harness LANDED** (cf898ea, subagent-built, reviewed):
  CPU AND GPU forward proven bit-exact deterministic (measured drift 0.0 on Metal) —
  parity 0.0/6.3e-5 vs the 2e-3/1e-2 gates; 40 dB PSNR floors vs 16-bit goldens.
  **Tile raster Stage 1 DONE by hand**: count_tile_touches kernel + exact-match gate vs
  the CPU oracle (0 mismatches, 2k+20k fixtures). P1 arm 1 at ~5k/30k (~0.5s/iter, DSSIM
  is pricier) — batch ETA ~18:00-19:00. Next: Stage 2 (pair emission + (tile,depth)
  bitonic sort + tile ranges), then arm-1 analysis when it lands.
- 2026-07-23 ~11:00: **Tile raster Stage 2 DONE** (pair emit + PairSorter + ranges, all
  properties exact on both fixtures). Real GPU bug found & fixed en route: naga/Metal
  compiles vec2 component stores in storage as whole-vector load-modify-write → two
  threads writing .x/.y of one tile's range raced and lost writes; tile_ranges is a flat
  array<u32> now, constraint recorded for all future kernels. P1 arm 1 at ~7k/30k
  (~0.53s/iter → arm1 ~11:45, batch complete ~20:15). Next tick: Stage 3
  (rasterize_tiled.wgsl + render_with_options + oracle parity gate).
- 2026-07-23 ~09:45: **Tile raster Stage 3 DONE — BIT-EXACT parity** (mean/max diff 0.0
  vs oracle on both fixtures, target was 1e-5). Second convention bug found by design
  review before it could bite: binning rect must cover pixel ceil(m+r) (oracle's bbox is
  integer-pixel floor/ceil), high edge now ceil-first everywhere, Stage 1/2 gates still
  exact. RenderOptions + render_with_options landed; naive renderer untouched as oracle.
  P1 arm 1 at ~11.4k/30k. Next: Stage 4 perf bench (tile vs naive at 60k/150k/400k) —
  the payoff measurement — then P1 arm analyses as they land.
- 2026-07-23 ~09:45: Stage-4 bench binary built (examples/bench_tile_raster.rs) — will
  RUN at the P1 batch boundary (~20:15), never beside a training arm. P1 arm 1
  (dssim+rg2500) at 15.3k/30k entering settle; eval_ssim column confirmed live
  (0.541→0.577 climbing through window close; window PSNR 17.26@15000). Arm-1 analysis
  on its DONE event (~11:40).
- 2026-07-23 ~10:15 interim: arm 1 (rg2500) settle still DECAYS under DSSIM — SSIM
  0.577@15k → 0.482@21k, PSNR 17.26→16.57, count melting 60k→24.8k (settle prunes remove
  ~2k/pass under the DSSIM population; margin-2500 alone is not the fix). Arm 2
  (no-resets) is the discriminator: if decay persists with zero resets, the driver is the
  settle-prune × DSSIM interaction. Baseline LPIPS curve computed
  (runs/srt30k_optfloor05_l1dssim/lpips.csv): best 0.460@25.5k, final 0.496 — LPIPS also
  bottoms mid-settle then worsens.
- 2026-07-23 10:39 **arm 1 DONE (dssim+rg2500)**: window peak 17.51 (= baseline), settle
  mean PSNR 16.69 / SSIM 0.520 (peaks 17.51/0.577), final 16.31 (+0.43 vs baseline
  15.88), count 60k→14.3k, best LPIPS 0.418@10.5k (in-window; beats baseline's 0.460).
  Verdict: better endpoint, mechanism NOT fixed — settle decays on all three metrics and
  the population still melts. Render remains crisp. Arm 2 (no-resets) running (~13:45).
- 2026-07-23 ~13:20 **MECHANISM PINNED (arm-2 settle logs)**: the melt is the NEEDLE PRUNE
  — needles=~1k/pass sustained (opacity ~100, oversize ~25) with ZERO resets; the
  L2-tuned 2.8 threshold fights DSSIM's functional edge-following anisotropy in a
  cull-regrow treadmill (renders crisp, no streaks — the anisotropy is signal, not
  pathology, under DSSIM). Queued via scripts/post_batch_20260723.sh after the P1 batch:
  Stage-4 bench on idle GPU, then arm 4 = baseline + --settle-needle-prune-log-aniso 0.
