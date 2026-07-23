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
4. Bench 60k/150k/400k × half/full res, naive vs tile; decide BATCH_SIZE=512 and
   early-exit levers from data. Gate: recorded numbers in RECOVERY_PLAN.
5. (outline only) Backward: reuse tile_ranges+pairs to bound the per-pixel re-walk;
   atomicAdd accumulation scheme unchanged; pixel_state's contributor index becomes
   tile-local. Workgroup-shared gradient accumulation is the follow-on optimization.

### Top risks

1. Screen-filling gaussians (cull only rejects radius > max(w,h)) can touch every tile →
   pairs-buffer blowup; mitigation = total_touches cap + oracle fallback, built in Stage 1.
2. K (avg touches/gaussian) unmeasured until Stage 1 runs — buffer table is planning-only.
3. 1e-5 parity tolerance is a prediction; budget Stage-3 time to investigate, don't
   pre-loosen.
4. CPU prefix-sum readback forces a mid-frame sync — measure in Stage 4 before trusting it
   inside the training loop.
5. GPU bit-exact-repeat is asserted from "no atomics" — prove it first (Part A step 1).
