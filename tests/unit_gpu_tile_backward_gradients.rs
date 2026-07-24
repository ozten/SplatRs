//! Tile-binned BACKWARD pass Stage 5b gate (docs/TILE_RASTER_PLAN.md Part B): the tiled
//! backward (`backward_tiled.wgsl`) must reproduce the naive oracle backward's per-Gaussian
//! gradients. Both accumulate into the SAME fixed-point `gradient_atomic` buffer contract
//! and reconstruct T back-to-front identically (`t_i = t_after / (1 - alpha)`); the only
//! difference is WHICH sorted artifact is walked (this tile's `pairs` range vs the globally
//! depth-sorted `Gaussian2D` array) and how the walk is batched (workgroup-cooperative
//! shared-memory batches vs one GPU thread per pixel over the whole array serially). So the
//! parity gate here is much tighter than CPU-vs-GPU parity (see
//! unit_gpu_deep_blend_gradients.rs): both paths share exact WGSL math and the same
//! contributor-selection tests, just different iteration order/grouping.
#![cfg(feature = "gpu")]

#[path = "golden/fixtures.rs"]
mod fixtures;
#[path = "golden/gpu_skip.rs"]
mod gpu_skip;

use fixtures::{deep_stack_scene, regression_scene, sh_constant_color, smoke_scene};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::gpu::{GaussianGradients2D, GpuRenderer, RenderOptions};

/// Run BOTH the naive oracle backward (`render_with_gradients`) and the tiled backward
/// (`render_with_gradients_and_options` with `tile_rasterizer: true`) on the same
/// scene/upstream gradients.
fn run_both(
    gpu: &GpuRenderer,
    gaussians: &[Gaussian],
    camera: &Camera,
    bg: &Vector3<f32>,
    d_pixels: &[Vector3<f32>],
) -> (
    (Vec<Vector3<f32>>, GaussianGradients2D),
    (Vec<Vector3<f32>>, GaussianGradients2D),
) {
    let naive = gpu
        .render_with_gradients(gaussians, camera, bg, d_pixels)
        .expect("naive render_with_gradients failed");
    let tiled = gpu
        .render_with_gradients_and_options(
            gaussians,
            camera,
            bg,
            d_pixels,
            RenderOptions {
                tile_rasterizer: true,
                ..Default::default()
            },
        )
        .expect("tiled render_with_gradients_and_options failed");
    (naive, tiled)
}

fn forward_parity_ok(name: &str, naive_img: &[Vector3<f32>], tiled_img: &[Vector3<f32>]) {
    let mut max_px_diff = 0.0f32;
    for (a, b) in naive_img.iter().zip(tiled_img.iter()) {
        let d = (a - b).abs();
        max_px_diff = max_px_diff.max(d.x.max(d.y).max(d.z));
    }
    eprintln!("[{name}] forward naive-vs-tiled max pixel diff: {max_px_diff:.9}");
    assert!(
        max_px_diff < 1e-4,
        "[{name}] forward images diverge ({max_px_diff}) between naive/tiled rasterizers — \
         a gradient comparison would be meaningless (contributor sets don't match)"
    );
}

/// docs/TILE_RASTER_PLAN.md Stage 5b tolerance: `diff <= 0.02*max(|a|,|b|) + 1e-3`.
fn loose_tol(a: f32, b: f32) -> f32 {
    0.02 * a.abs().max(b.abs()) + 1e-3
}

/// smoke_scene()/regression_scene()/deep_stack_scene() naive-vs-tiled parity: per-component
/// comparison on d_colors/d_opacity_logits/d_mean_px/d_cov_2d/d_background, plus a
/// no-starvation check (same set of gaussians gets nonzero gradients in both paths).
fn check_parity(name: &str, gaussians: &[Gaussian], camera: &Camera, bg: Vector3<f32>) {
    let Some(gpu) = gpu_skip::try_gpu_renderer() else {
        return;
    };
    let num_pixels = (camera.width * camera.height) as usize;
    let d_pixels = vec![Vector3::new(1.0, -0.5, 0.25); num_pixels];

    let ((naive_img, naive_grads), (tiled_img, tiled_grads)) =
        run_both(&gpu, gaussians, camera, &bg, &d_pixels);

    forward_parity_ok(name, &naive_img, &tiled_img);

    let n = gaussians.len();
    assert_eq!(naive_grads.d_colors.len(), n, "[{name}] naive grad length");
    assert_eq!(tiled_grads.d_colors.len(), n, "[{name}] tiled grad length");

    const NONZERO_EPS: f32 = 1e-6;
    let mut naive_nonzero = vec![false; n];
    let mut tiled_nonzero = vec![false; n];

    // Track the worst-case measured relative diff per field (diagnostic only).
    let mut max_rel_color = 0.0f32;
    let mut max_rel_opacity = 0.0f32;
    let mut max_rel_mean = 0.0f32;
    let mut max_rel_cov = 0.0f32;

    let check_component =
        |field: &str, i: usize, c: usize, a: f32, b: f32, max_rel: &mut f32| {
            let diff = (a - b).abs();
            let denom = a.abs().max(b.abs()).max(1e-8);
            *max_rel = max_rel.max(diff / denom);
            assert!(
                diff <= loose_tol(a, b),
                "[{name}] {field} mismatch gaussian {i} component {c}: naive={a} tiled={b} diff={diff} (tol {})",
                loose_tol(a, b)
            );
            if a.abs() > NONZERO_EPS {
                true
            } else {
                b.abs() > NONZERO_EPS
            }
        };

    for i in 0..n {
        let mut any_nonzero_naive = false;
        let mut any_nonzero_tiled = false;
        for c in 0..3 {
            let a = naive_grads.d_colors[i][c];
            let b = tiled_grads.d_colors[i][c];
            check_component("d_color", i, c, a, b, &mut max_rel_color);
            any_nonzero_naive |= a.abs() > NONZERO_EPS;
            any_nonzero_tiled |= b.abs() > NONZERO_EPS;
        }
        {
            let a = naive_grads.d_opacity_logits[i];
            let b = tiled_grads.d_opacity_logits[i];
            check_component("d_opacity_logit", i, 0, a, b, &mut max_rel_opacity);
            any_nonzero_naive |= a.abs() > NONZERO_EPS;
            any_nonzero_tiled |= b.abs() > NONZERO_EPS;
        }
        for c in 0..2 {
            let a = naive_grads.d_mean_px[i][c];
            let b = tiled_grads.d_mean_px[i][c];
            check_component("d_mean_px", i, c, a, b, &mut max_rel_mean);
            any_nonzero_naive |= a.abs() > NONZERO_EPS;
            any_nonzero_tiled |= b.abs() > NONZERO_EPS;
        }
        for c in 0..3 {
            let a = naive_grads.d_cov_2d[i][c];
            let b = tiled_grads.d_cov_2d[i][c];
            check_component("d_cov_2d", i, c, a, b, &mut max_rel_cov);
            any_nonzero_naive |= a.abs() > NONZERO_EPS;
            any_nonzero_tiled |= b.abs() > NONZERO_EPS;
        }
        naive_nonzero[i] = any_nonzero_naive;
        tiled_nonzero[i] = any_nonzero_tiled;
    }

    for c in 0..3 {
        let a = naive_grads.d_background[c];
        let b = tiled_grads.d_background[c];
        let diff = (a - b).abs();
        assert!(
            diff <= loose_tol(a, b),
            "[{name}] d_background mismatch ch {c}: naive={a} tiled={b} diff={diff}"
        );
    }

    eprintln!(
        "[{name}] measured max relative diff: d_color={max_rel_color:.9} d_opacity={max_rel_opacity:.9} d_mean_px={max_rel_mean:.9} d_cov_2d={max_rel_cov:.9}"
    );

    let mut starved = Vec::new();
    let mut naive_nonzero_count = 0usize;
    let mut tiled_nonzero_count = 0usize;
    for i in 0..n {
        if naive_nonzero[i] {
            naive_nonzero_count += 1;
        }
        if tiled_nonzero[i] {
            tiled_nonzero_count += 1;
        }
        if naive_nonzero[i] != tiled_nonzero[i] {
            starved.push((i, naive_nonzero[i], tiled_nonzero[i]));
        }
    }
    eprintln!(
        "[{name}] {naive_nonzero_count}/{n} gaussians have a nonzero naive gradient, {tiled_nonzero_count}/{n} nonzero tiled"
    );
    // Rule out a vacuous pass (both paths trivially all-zero would satisfy the equality
    // check above without ever exercising the backward math).
    assert!(
        naive_nonzero_count > 0 && tiled_nonzero_count > 0,
        "[{name}] no gaussian received a nonzero gradient in either path — comparison is vacuous"
    );
    assert!(
        starved.is_empty(),
        "[{name}] {} gaussian(s) have gradients in one path but not the other (starvation): {:?}",
        starved.len(),
        &starved[..starved.len().min(10)]
    );
}

#[test]
fn tile_backward_gradients_parity_smoke() {
    let (gaussians, camera, bg) = smoke_scene();
    check_parity("smoke", &gaussians, &camera, bg);
}

#[test]
fn tile_backward_gradients_parity_regression() {
    let (gaussians, camera, bg) = regression_scene();
    check_parity("regression", &gaussians, &camera, bg);
}

#[test]
fn tile_backward_gradients_parity_deep_stack() {
    let (gaussians, camera, bg) = deep_stack_scene();
    check_parity("deep_stack", &gaussians, &camera, bg);
}

/// Builds the TRIPWIRE scene: a hand-built single-tile scene of 3 Gaussians at known
/// positions/depths/opacities in a 32x32 (2x2-tile) image. All three project well inside
/// tile (0,0) — their 3-sigma footprints (~4.4px) stay within pixels [0,16) for mean pixel
/// targets 6..11 — so exactly one of the 4 tiles has a non-empty `pairs` range. This is the
/// smallest scene that still exercises the FULL tiled backward machinery (tile_ranges
/// lookup, cooperative shared-memory batch load, single-batch back-to-front walk) while
/// keeping the expected contributor set easy to reason about by hand. Shared by the naive-
/// vs-tiled parity test below and the absgrad-slot test in this file.
///
/// Returns (gaussians, camera, background, upstream d_pixels).
fn tripwire_scene() -> (Vec<Gaussian>, Camera, Vector3<f32>, Vec<Vector3<f32>>) {
    let width = 32u32;
    let height = 32u32;
    let fx = 100.0f32;
    let fy = 100.0f32;
    let cx = 16.0f32;
    let cy = 16.0f32;
    let camera = Camera::new(
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    // sigma_world chosen so the projected 3-sigma footprint (~4.4px) stays inside [0,16)
    // for all three (mean pixels in 6..11), never touching tiles (1,0)/(0,1)/(1,1).
    let log_scale = Vector3::new(-3.6, -3.6, -3.6);

    // (target_px, target_py, depth, opacity_logit, color) — ascending depth (front to back).
    let targets: [(f32, f32, f32, f32, [f32; 3]); 3] = [
        (6.0, 6.0, 2.0, -0.5, [0.9, 0.1, 0.1]),
        (9.0, 8.0, 2.2, 0.5, [0.1, 0.9, 0.1]),
        (11.0, 10.0, 2.4, 1.5, [0.1, 0.1, 0.9]),
    ];
    let gaussians: Vec<Gaussian> = targets
        .iter()
        .map(|&(tx, ty, z, opacity_logit, color)| {
            // Invert the pinhole projection (identity rotation/zero translation, so camera
            // space == world space) to place the mean at exactly the target pixel.
            let x = (tx - cx) * z / fx;
            let y = (ty - cy) * z / fy;
            Gaussian::new(
                Vector3::new(x, y, z),
                log_scale,
                UnitQuaternion::identity(),
                opacity_logit,
                sh_constant_color(Vector3::new(color[0], color[1], color[2])),
            )
        })
        .collect();

    let bg = Vector3::new(0.05, 0.05, 0.05);
    let num_pixels = (width * height) as usize;
    let d_pixels = vec![Vector3::new(1.0, -0.5, 0.25); num_pixels];
    (gaussians, camera, bg, d_pixels)
}

/// Naive and tiled share the exact same fixed-point atomics and (for this scene) the exact
/// same contributor set, so tolerance is much tighter than the random-scene parity above:
/// every gradient component must match within 1e-4 relative (+ a tiny absolute floor to
/// avoid flaking on fixed-point-quantization-level near-zero comparisons).
#[test]
fn tile_backward_gradients_tripwire_single_tile() {
    let Some(gpu) = gpu_skip::try_gpu_renderer() else {
        return;
    };

    let (gaussians, camera, bg, d_pixels) = tripwire_scene();

    let ((naive_img, naive_grads), (tiled_img, tiled_grads)) =
        run_both(&gpu, &gaussians, &camera, &bg, &d_pixels);

    forward_parity_ok("tripwire", &naive_img, &tiled_img);

    let tight_ok = |a: f32, b: f32| -> bool {
        let diff = (a - b).abs();
        diff <= 1e-4 * a.abs().max(b.abs()) + 1e-6
    };

    for i in 0..gaussians.len() {
        let mut any_nonzero_naive = false;
        let mut any_nonzero_tiled = false;
        for c in 0..3 {
            let a = naive_grads.d_colors[i][c];
            let b = tiled_grads.d_colors[i][c];
            assert!(
                tight_ok(a, b),
                "[tripwire] d_color mismatch gaussian {i} ch {c}: naive={a} tiled={b}"
            );
            any_nonzero_naive |= a.abs() > 1e-8;
            any_nonzero_tiled |= b.abs() > 1e-8;
        }
        {
            let a = naive_grads.d_opacity_logits[i];
            let b = tiled_grads.d_opacity_logits[i];
            assert!(
                tight_ok(a, b),
                "[tripwire] d_opacity mismatch gaussian {i}: naive={a} tiled={b}"
            );
            any_nonzero_naive |= a.abs() > 1e-8;
            any_nonzero_tiled |= b.abs() > 1e-8;
        }
        for c in 0..2 {
            let a = naive_grads.d_mean_px[i][c];
            let b = tiled_grads.d_mean_px[i][c];
            assert!(
                tight_ok(a, b),
                "[tripwire] d_mean_px mismatch gaussian {i} axis {c}: naive={a} tiled={b}"
            );
            any_nonzero_naive |= a.abs() > 1e-9;
            any_nonzero_tiled |= b.abs() > 1e-9;
        }
        for c in 0..3 {
            let a = naive_grads.d_cov_2d[i][c];
            let b = tiled_grads.d_cov_2d[i][c];
            assert!(
                tight_ok(a, b),
                "[tripwire] d_cov_2d mismatch gaussian {i} idx {c}: naive={a} tiled={b}"
            );
            any_nonzero_naive |= a.abs() > 1e-9;
            any_nonzero_tiled |= b.abs() > 1e-9;
        }
        assert!(
            any_nonzero_naive,
            "[tripwire] naive gaussian {i} has all-zero gradients"
        );
        assert!(
            any_nonzero_tiled,
            "[tripwire] tiled gaussian {i} has all-zero gradients"
        );
    }

    for c in 0..3 {
        let a = naive_grads.d_background[c];
        let b = tiled_grads.d_background[c];
        assert!(
            tight_ok(a, b),
            "[tripwire] d_background mismatch ch {c}: naive={a} tiled={b}"
        );
    }

    eprintln!("[tripwire] all gradient components within 1e-4 relative; no starved gaussian");
}

/// Absgrad-style accumulator (offsets 16-17 of the fixed-point gradient buffer, see
/// backward.wgsl's `GRADIENT_STRIDE` comment): a new atomic slot pair accumulates
/// `abs(d_mean * d_weight)` per contributor, alongside (not instead of) the pre-existing
/// signed `d_mean_px` slots (offsets 8-9). This test asserts the three properties the
/// implementation is supposed to have, on the TRIPWIRE fixture (small, hand-reasoned-about
/// scene of 3 overlapping isotropic Gaussians under a spatially-uniform upstream gradient —
/// see `tripwire_scene()`):
///
/// (a) `abs_d_mean_px >= |d_mean_px|` element-wise, for both the naive and tiled backward
///     (triangle inequality: sum of |x_i| >= |sum of x_i|; independent fixed-point rounding
///     of the two accumulators gets a small epsilon).
/// (b) naive and tiled abs slots match each other within the same relative tolerance as the
///     existing signed-slot tripwire check above (both paths share the exact same WGSL math
///     and, for this scene, the exact same contributor set).
/// (c) at least one gaussian/component has abs MEANINGFULLY greater than |signed| — i.e. the
///     new slots are not merely a redundant copy of the signed ones. The tripwire scene's
///     Gaussians are isotropic and its upstream gradient (`d_pixels`) is spatially uniform,
///     so each Gaussian's per-pixel `d_mean` contributions are close to antisymmetric around
///     its own mean (positive on one side, negative on the other) while the abs accumulator
///     can't cancel — the textbook case absgrad is designed for.
#[test]
fn tile_backward_gradients_absgrad_slots_tripwire() {
    let Some(gpu) = gpu_skip::try_gpu_renderer() else {
        return;
    };

    let (gaussians, camera, bg, d_pixels) = tripwire_scene();
    // The abs_d_mean_px atomic (offsets 16-17) REUSES FIXED_POINT_SCALE_POSITION (1e9) — same
    // scale as the signed slots (see backward.wgsl's comment above `atomic_add_f32_position`:
    // a dedicated lower scale was tried and reverted because it silently underflowed real
    // training-scale gradients to zero). At that scale, the shared fixture's upstream
    // gradient (d_pixels ≈ (1.0, -0.5, 0.25), used by the signed-only tripwire test above) is
    // unrealistically large — real per-pixel loss gradients are normalized by pixel count and
    // are commonly ~1e-8, vs this fixture's ~1.0 — and DOES overflow the i32 atomic for the
    // (uncancelled) abs sum here: this Gaussian's abs accumulation reaches real magnitude
    // ~2.4-4.5, over the ~2.15 representable ceiling. Scale d_pixels down 100x to a value
    // that still exercises real backward math (nowhere near the atomic_add_* early-out
    // epsilons) while staying safely inside the fixed-point range; this doesn't affect the
    // signed-slot test above, which uses `tripwire_scene()`'s unscaled d_pixels directly.
    let d_pixels: Vec<Vector3<f32>> = d_pixels.iter().map(|v| v * 0.01).collect();

    let ((naive_img, naive_grads), (tiled_img, tiled_grads)) =
        run_both(&gpu, &gaussians, &camera, &bg, &d_pixels);

    forward_parity_ok("absgrad-tripwire", &naive_img, &tiled_img);

    // Fixed-point epsilon: FIXED_POINT_SCALE_POSITION is 1e9, so a single-LSB rounding
    // difference between the independently-accumulated signed/abs atomics is ~1e-9; 1e-3 is
    // the tolerance mandated by the task (generous headroom over the actual quantization
    // floor).
    const FIXED_POINT_EPS: f32 = 1e-3;
    let tight_ok = |a: f32, b: f32| -> bool {
        let diff = (a - b).abs();
        diff <= 1e-4 * a.abs().max(b.abs()) + 1e-6
    };

    let mut max_relative_gap = 0.0f32;
    let mut max_gap_desc = String::new();

    for i in 0..gaussians.len() {
        for (path_name, grads) in [("naive", &naive_grads), ("tiled", &tiled_grads)] {
            for c in 0..2 {
                let signed = grads.d_mean_px[i][c];
                let abs_val = grads.abs_d_mean_px[i][c];
                // (a) abs >= |signed| (within a fixed-point epsilon).
                assert!(
                    abs_val >= signed.abs() - FIXED_POINT_EPS,
                    "[absgrad-tripwire] {path_name} gaussian {i} axis {c}: \
                     abs_d_mean_px={abs_val} < |d_mean_px|={} (eps {FIXED_POINT_EPS})",
                    signed.abs()
                );

                let gap = abs_val - signed.abs();
                let rel_gap = gap / abs_val.max(1e-9);
                if rel_gap > max_relative_gap {
                    max_relative_gap = rel_gap;
                    max_gap_desc = format!(
                        "{path_name} gaussian {i} axis {c}: abs={abs_val:.9} signed={signed:.9} \
                         (|signed|={:.9}) gap={gap:.9} rel_gap={rel_gap:.4}",
                        signed.abs()
                    );
                }
            }
        }

        // (b) naive vs tiled abs slots match within the same tolerance as the signed slots
        // in the sibling test above (both paths share exact WGSL math + contributor set for
        // this scene).
        for c in 0..2 {
            let a = naive_grads.abs_d_mean_px[i][c];
            let b = tiled_grads.abs_d_mean_px[i][c];
            assert!(
                tight_ok(a, b),
                "[absgrad-tripwire] abs_d_mean_px mismatch gaussian {i} axis {c}: naive={a} tiled={b}"
            );
        }
    }

    eprintln!(
        "[absgrad-tripwire] all abs_d_mean_px >= |d_mean_px| (eps {FIXED_POINT_EPS}); \
         naive/tiled abs slots match within 1e-4 relative; \
         largest observed (abs - |signed|)/abs = {max_relative_gap:.4} at [{max_gap_desc}]"
    );

    // (c) at least one gaussian/component shows MEANINGFUL cancellation (abs materially
    // larger than |signed|), i.e. the new slots aren't a redundant copy of the old ones on
    // this fixture. Threshold: the abs accumulator is at least 10% larger than the signed
    // magnitude somewhere. If this ever fails, it means the tripwire fixture stopped
    // exhibiting sign cancellation (e.g. after a fixture edit) rather than that the abs
    // slots are wired up wrong — (a)/(b) above already cover correctness — so note it rather
    // than treat it as an implementation bug.
    const MEANINGFUL_REL_GAP: f32 = 0.10;
    assert!(
        max_relative_gap > MEANINGFUL_REL_GAP,
        "[absgrad-tripwire] NOTE: no gaussian/component showed meaningful abs-vs-signed \
         cancellation on this fixture (largest relative gap {max_relative_gap:.4} <= {MEANINGFUL_REL_GAP}); \
         (a)/(b) still passed (abs slots are wired correctly), but this fixture no longer \
         demonstrates absgrad's benefit — see [{max_gap_desc}]"
    );
}
