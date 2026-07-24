# Golden parity harness + tile-binned rasterizer — implementation plan (2026-07-23)

Designed under AUTONOMY_PLAN_20260723.md P2. Self-contained: executable by a session with no
other context. Verified-against-source facts are marked (v).

## Ground truth (verified against source)

- (v) `test_e2e/model.gs` and `plys/train.splat` PREDATE the 836f4d0 renderer fixes (Rᵀ
  rotation convention) — unusable as fixtures. Never compare pre-836f4d0 .gs files against
  current renders (RECOVERY_PLAN.md:566).
- (v) `datasets/` is gitignored and not auto-fetched; a harness depending on it breaks fresh
  clones. Fixtures must be procedurally generated at test time (seeded).
- (v) `.gitignore` blocks `*.png`/`*.gs` — golden images need an explicit carve-out
  (`!/tests/golden/goldens/*.png`), else they silently don't commit.
- (v) GPU forward has NO atomics (deterministic); backward uses fixed-point atomicAdd. CPU
  forward `render_full_linear` (src/render/full_diff.rs:509) is sequential, stable sort —
  bit-exact repeatable expected.
- (v) `GaussianGPU` = 320 B; 400k × 320 B = 128 MB = `max_storage_buffer_binding_size`
  default. `Limits::default()` gives 16 KiB workgroup storage (Metal HW ceiling 32 KiB, but
  the device is created with defaults in src/gpu/context.rs:103).
- (v) Existing tolerances that don't flake on this hardware: max 0.01
  (unit_gpu_render_smoke), mean 2e-3 (unit_gpu_sort_order). Reuse, don't invent.
- (v) Canonical test resolutions: 490×273 half-res, 980×545 full-res.

## Part A — golden parity/regression harness

Fixtures (new `tests/golden/fixtures.rs`, seeded StdRng, patterns from
tests/unit_gpu_sort_order.rs): `smoke_scene()` ~2k gaussians @128×96;
`regression_scene()` ~20k mixed-anisotropy/opacity gaussians scattered in x/y/depth @490×273.

Order of work:
1. **Determinism first** (`tests/golden_determinism.rs`): CPU render twice → assert exact
   `==`; GPU render twice → assert exact (fallback 1e-6 with doc comment if driver breaks
   bit-exactness). These gate the whole tolerance scheme.
2. **Golden IO** (`tests/golden/golden_io.rs`): save/load 16-bit PNG (image crate),
   `compare()` returning (mean_abs, max_abs) — factor out the diff snippet already
   copy-pasted in render_compare.rs / unit_gpu_sort_order.rs.
3. **Parity tests** (`tests/golden_cpu_gpu_parity.rs`): smoke + regression scenes, GPU vs
   CPU, mean < 2e-3 / max < 1e-2.
4. **PSNR floors** (`tests/golden_psnr_floor.rs`): CPU and GPU renders vs
   `tests/golden/goldens/regression_scene_cpu.png`; floor = first-run PSNR minus margin
   (~measured 45 → floor 40). Promote `compute_psnr` in src/optim/trainer.rs to `pub` (only
   non-test source change). Golden regeneration only via env `SUGAR_REGENERATE_GOLDEN=1`.
5. **GPU skip helper** (`tests/golden/gpu_skip.rs`): `GpuRenderer::new()` Err → eprintln +
   return (inside `#[cfg(feature = "gpu")]` per existing convention).
- CPU-vs-golden tolerance: mean < 5e-4, max < 2e-3 (absorbs 16-bit quantization 1.5e-5).
- Layout: helpers in `tests/golden/*.rs` pulled in via `#[path]` mod (std tests/common
  idiom); goldens in `tests/golden/goldens/`.

## Part B — tile-binned forward rasterizer (flag-gated; naive renderer stays as oracle)

Pipeline: projection (UNCHANGED — cov.w already stores the 3σ radius; skip the global
Gaussian2DGPU bitonic sort on the tile path) → tile counting → CPU prefix sum (readback is
4·N B ≤ 1.6 MB; a GPU scan is a later optimization) → pair emission → ONE global bitonic
sort of `TileGaussianPair{key_tile,key_depth,gaussian_idx,pad}` (16 B) keyed (tile, depth)
— depth key = bitcast(z) which order-preserves for positive floats; pad sentinel
key_tile=num_tiles — → boundary-detect kernel → `tile_ranges` → per-tile 16×16 workgroup
raster with shared-memory batches.

- **BATCH_SIZE = 256** (36 B/gaussian shared = 9 KiB < 16 KiB default limit → NO context.rs
  change). 512 needs a required_limits bump to 32 KiB — deferred perf lever.
- Tile grids: 31×18=558 tiles @490×273; 62×35=2170 @980×545.
- Pairs buffer = 16·pow2(K·N): 150k/K=8 → 33.6 MB OK; **400k/K=16 → 134 MB EXCEEDS the
  128 MiB binding limit** — mitigation: after the CPU prefix sum knows total_touches, if it
  exceeds cap → log + fall back to the naive oracle for that frame (same philosophy as the
  render watchdog).
- New sorter: `sort_pairs.wgsl` + PairSorter, copied from the PROVEN sort.wgsl stage/step
  loop (836f4d0 direction-bit bug history — cross-reference in a comment; re-run the
  non-power-of-two padding regression pattern from unit_gpu_sort_order.rs on pairs).
- Raster kernel `rasterize_tiled.wgsl`: collaborative 256-wide batch load, workgroupBarrier,
  per-pixel blend replicating rasterize.wgsl's eval/alpha logic exactly, early-out T<1e-4.
  No workgroup-level early-exit in v1 (Stage-4 perf lever).
- Plumbing: `RenderOptions { disable_sh, tile_rasterizer }` +
  `GpuRenderer::render_with_options(...)`; `render()`/`render_with_sh_mode()` become
  wrappers. CLI `--tile-raster`; env `SUGAR_GPU_TILE_RASTER=1` for ad hoc only.
- Oracle parity target: max ≤ 1e-5 / mean ≤ 1e-6 (same device, same WGSL math); documented
  fallback 1e-4/1e-5 only if measured drift justifies it.
- Watchdog: tile cost is O(pixels × local density), not O(pixels × N) — expect one un-banded
  dispatch to be safe, but MEASURE in Stage 4; keep a tile-row banding fallback keyed off
  total_touches.

### Stages (each with a gate)

0. **DONE (2026-07-23).** `TileGaussianPair` in src/gpu/types.rs + CPU reference math in
   **src/render/tile_math.rs** (NOT src/gpu/ as originally planned: lib.rs:47 feature-gates
   the ENTIRE gpu module — the non-gpu stub in gpu/mod.rs is dead code — so anything that
   must exist in CPU-only builds cannot live under src/gpu/). `tile_touch_rect` /
   `tile_touch_count` / `tile_grid_dims`, TILE_SIZE=16. Gate passed: 7 unit tests
   (inside/spanning/clipped/off-screen/NaN/screen-filling), both feature builds compile.
1. **DONE (2026-07-23).** `tile_bin.wgsl::count_tile_touches` + lazily-built pipeline in
   `GpuRenderer::debug_tile_touch_counts` (returns unsorted projected values + counts so the
   test validates on identical f32 bits; culled convention = `!(cov.w > 0)`, NaN-safe guard).
   Gate passed: exact match on smoke + regression fixtures (tests/unit_gpu_tile_counting.rs),
   with live and multi-tile coverage asserts.
2. **DONE (2026-07-23).** CPU prefix sum → emit_tile_pairs → PairSorter
   (sort_pairs.wgsl, byte-for-byte copy of the proven network incl. the 836f4d0
   direction bit) → identify_tile_ranges. Gate passed (tests/unit_gpu_tile_sort.rs):
   global sortedness, per-tile depth monotonicity, pair conservation vs counts +
   rect membership + depth-key bitcast check, range partition, no sentinel leak —
   all exact, both fixtures. **BUG FOUND & FIXED en route: never let two threads
   write different components of one storage vec2 — naga/Metal compiles a component
   store as load-modify-write of the whole vector, silently racing (observed as
   scattered lost boundary writes). tile_ranges is a flat array<u32> now.** This
   constraint also applies to Stage 3+ kernel design (per-pixel state, gradient
   buffers already use scalar/atomic patterns).
3. **DONE (2026-07-23) — BIT-EXACT.** `rasterize_tiled.wgsl` (16×16 workgroup/tile,
   256-pair shared-memory batches at 40 B/pair = 10 KiB < the 16 KiB default limit,
   workgroup-uniform barriers, per-thread `done` flag) + `RenderOptions` +
   `render_with_options` + `render_tiled` (v1 reuses the binning debug path). Gate
   passed at **mean/max = 0.0 exactly** on both fixtures — beat the 1e-5 target; the
   per-pixel math replicates the oracle expression-for-expression and the ceil-first
   binning convention (fixed this stage in tile_math + both binning kernels: the rect
   must cover pixel `ceil(m+r)`, which can land one tile past `floor((m+r)/16)`).
4. **DONE (2026-07-23, runs/bench_tile_raster_20260723.txt, M2 Max):**

   | N | res | naive ms | tiled ms | speedup | pairs | K |
   |---|---|---|---|---|---|---|
   | 60k | 490×273 | 125.5 | 26.8 | 4.7× | 268k | 5.3 |
   | 60k | 980×545 | 467.2 | 32.5 | 14.4× | 645k | 12.9 |
   | 150k | 490×273 | 495.8 | 43.6 | 11.4× | 668k | 5.3 |
   | 150k | 980×545 | 1568.4 | 61.1 | **25.7×** | 1.61M | 12.9 |
   | 400k | 490×273 | 487.6 | 87.9 | 5.5× | 1.77M | 5.3 |
   | 400k | 980×545 | 899.7 | 237.8 | 3.8× | 4.27M | 12.8 |

   Notes: (a) tiled numbers INCLUDE the v1 overhead (lazy pipelines, CPU prefix sum,
   readbacks) — production integration will be faster still; (b) naive scales
   sub-linearly at 400k because per-pixel early termination (T<1e-4) saturates in dense
   scenes — real effect, benefits both paths; (c) bench parity max ~5e-4..8e-3 on the
   random synthetic scenes is a DEPTH-TIE artifact (random f32 depths collide; tie order
   differs between the depth-only sort and the (tile,depth) sort; both orders are
   legitimate) — the structured golden fixtures remain bit-exact 0.0; (d) 400k full-res
   in ONE un-banded dispatch at 238 ms — the Metal-watchdog ceiling that motivated
   banding is gone on the tile path; (e) BATCH_SIZE=512 / workgroup early-exit levers
   deferred — current numbers already clear the bar for Stage 5.
5. **Full staged plan (2026-07-24, source-verified — see git history of this section for
   the long form; key decisions below are the contract):**
   - **5a** Tiled forward-with-intermediates + lifecycle: pixel_state binding in
     rasterize_tiled.wgsl storing (bitcast(T_final), ABSOLUTE last pair index; sentinel
     0xFFFFFFFF), one whole-vec2 store per owning thread (race-safe). Cache all tile
     pipelines + PairSorter as GpuRenderer fields (built in new()); new GPU-resident
     tile_binning_gpu() reading back ONLY counts (projected/pairs/ranges stay on GPU —
     kills the v1 32-68MB round-trips). Gates: existing parity/count/sort tests + new
     unit_gpu_tile_pixel_state_parity (T within 1e-5; resolved-identity match:
     sorted_buffer[naive.y].gaussian_idx_pad.x == pairs[tiled.y].gaussian_idx) + bench
     non-regression.
   - **5b** backward_tiled.wgsl: one workgroup/tile, batched BACK-TO-FRONT walk of the
     full tile range (batch_end→range_start), per-pixel gate `pair_idx > last_pair_idx →
     continue` (valid because pairs are depth-ascending in-tile), helpers/constants copied
     VERBATIM from backward.wgsl, shared memory = read-only cache (batch_a/b/c + NEW
     batch_idx u32 = 44B×256 = 11KiB < 16KiB), gradients go straight to the UNCHANGED
     global fixed-point atomic buffer. gaussian_idx needs NO indirection (projection never
     reorders on the tile path — pairs carry original indices). 7 storage buffers + 1
     uniform = exact fit under max_storage_buffers_per_shader_stage=8 (VERIFY with a
     pipeline-creation smoke FIRST; fallback = two bind groups). Gate: new
     unit_gpu_tile_backward_gradients — naive-GPU vs tiled-GPU on smoke/regression +
     deep_stack_scene (factored from unit_gpu_deep_blend_gradients), tolerance start
     0.02·max+1e-3 (half the naive-vs-CPU precedent; measure, log, loosen only with
     numbers); plus a hand-computed 2-3-gaussian single-tile tripwire.
   - **5c** bench_tile_raster_backward.rs (same matrix as Stage 4). Banding trigger: any
     unbanded dispatch > ~1s → band by TILE ROWS (tile_row_offset param), else ship
     unbanded. Top empirical risk: atomic contention on tile-spanning gaussians
     (unforecastable from forward's atomic-free numbers).
   - **5d** MultiViewTrainConfig.tile_rasterizer + --tile-raster CLI +
     SUGAR_GPU_TILE_RASTER env default + startup log; only the two render call sites in
     the training loop change (render_with_options / new
     render_with_gradients_and_options; naive body renamed _naive). All config literals
     updated (m8/m9 tests). Single-view TrainConfig explicitly out of scope.
   - **5e** 500-iter training-equivalence smoke (#[ignore], fully procedural — no COLMAP:
     bespoke loop from pub pieces: regression_scene target, perturbed init, loss.rs +
     adam.rs + compute_psnr). CALIBRATE the naive-vs-naive noise floor before locking the
     threshold (start ~0.3 dB final-PSNR delta).
   Ranked risks: (1) 8-buffer exact fit, (2) atomic contention, (3) batch-boundary
   off-by-one in the back-to-front walk (silently drops one boundary pair — the tripwire
   fixture exists for this), (4) uncalibrated 5e threshold, (5) the lifecycle refactor
   touches two passing gates — re-run them in isolation before 5b builds on top.

### Top risks

1. Screen-filling gaussians (cull only rejects radius > max(w,h)) can touch every tile →
   pairs-buffer blowup; mitigation = total_touches cap + oracle fallback, built in Stage 1.
2. K (avg touches/gaussian) unmeasured until Stage 1 runs — buffer table is planning-only.
3. 1e-5 parity tolerance is a prediction; budget Stage-3 time to investigate, don't
   pre-loosen.
4. CPU prefix-sum readback forces a mid-frame sync — measure in Stage 4 before trusting it
   inside the training loop.
5. GPU bit-exact-repeat is asserted from "no atomics" — prove it first (Part A step 1).
