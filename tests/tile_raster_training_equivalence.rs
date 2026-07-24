//! Stage-5e training-equivalence smoke (docs/TILE_RASTER_PLAN.md Part B Stage 5e): the
//! tile-binned rasterizer (forward + backward, Stage 5a/5b) must produce a training
//! trajectory statistically indistinguishable from the naive oracle path over many
//! iterations, not just per-frame parity. Fully procedural (NO datasets/): ground truth
//! is `regression_scene()` rendered once by the naive GPU rasterizer; the trainable init
//! is a seeded-perturbed copy of the same scene.
//!
//! Per the plan's "simplest valid design" note this is a comparison of the two BACKENDS,
//! not a full trainer: only opacity logits and the SH DC color term are optimized (both
//! receive gradients directly from `GaussianGradients2D::d_opacity_logits`/`d_colors`
//! with no 2D->3D chain needed — position/scale/rotation stay fixed at their perturbed
//! values for the whole run).
//!
//! `#[ignore]`-gated (500 iters x 3 training runs is slow); run with:
//!   cargo test --release --features gpu --test tile_raster_training_equivalence \
//!     -- --ignored --nocapture
#![cfg(feature = "gpu")]

#[path = "golden/fixtures.rs"]
mod fixtures;
#[path = "golden/gpu_skip.rs"]
mod gpu_skip;

use fixtures::regression_scene;
use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sugar_rs::core::{Camera, Gaussian, SH_C0};
use sugar_rs::gpu::{GpuRenderer, RenderOptions};
use sugar_rs::optim::adam::{AdamF32, AdamVec3};
use sugar_rs::optim::loss::l2_image_loss_and_grad;
use sugar_rs::optim::trainer::compute_psnr;

const N_ITERS: usize = 500;
const CHECKPOINT_INTERVAL: usize = 50;
const OPACITY_LR: f32 = 0.05;
const COLOR_LR: f32 = 0.0025;

/// Standard-normal sample (Box-Muller), scaled by `std`. Hand-rolled because `rand_distr`
/// is not a dependency (Cargo.toml only pulls plain `rand`); deterministic given `rng`.
fn gauss_sample(rng: &mut StdRng, std: f32) -> f32 {
    let u1: f32 = rng.gen_range(1e-7f32..1.0);
    let u2: f32 = rng.gen_range(0.0f32..1.0);
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    z0 * std
}

/// Perturbed copy of `gt`: jitter positions by N(0,0.05), SH DC color by N(0,0.1),
/// opacity logits by N(0,0.2). StdRng seed 7 (docs/TILE_RASTER_PLAN.md Stage 5e).
fn perturbed_init(gt: &[Gaussian]) -> Vec<Gaussian> {
    let mut rng = StdRng::seed_from_u64(7);
    gt.iter()
        .map(|g| {
            let mut g2 = g.clone();
            g2.position += Vector3::new(
                gauss_sample(&mut rng, 0.05),
                gauss_sample(&mut rng, 0.05),
                gauss_sample(&mut rng, 0.05),
            );
            for c in 0..3 {
                g2.sh_coeffs[0][c] += gauss_sample(&mut rng, 0.1);
            }
            g2.opacity += gauss_sample(&mut rng, 0.2);
            g2
        })
        .collect()
}

/// Run the simplified opacity+DC-color-only training loop for `N_ITERS`, returning
/// (checkpoint_iter, psnr) pairs every `CHECKPOINT_INTERVAL` iterations. PSNR at each
/// checkpoint is measured on the PRE-update forward render for that iteration, mirroring
/// the forward-then-separate-backward-call convention in src/optim/trainer.rs (a plain
/// forward render for loss/PSNR, then a second render+gradients call for the backward
/// pass — both real-trainer render call sites this test exercises).
fn run_training(
    gpu: &GpuRenderer,
    init_gaussians: &[Gaussian],
    target: &[Vector3<f32>],
    camera: &Camera,
    bg: &Vector3<f32>,
    tile_rasterizer: bool,
) -> Vec<(usize, f32)> {
    let mut gaussians = init_gaussians.to_vec();
    let n = gaussians.len();

    let mut opacity_logits: Vec<f32> = gaussians.iter().map(|g| g.opacity).collect();
    let mut dc_colors: Vec<Vector3<f32>> = gaussians
        .iter()
        .map(|g| Vector3::new(g.sh_coeffs[0][0], g.sh_coeffs[0][1], g.sh_coeffs[0][2]))
        .collect();

    let mut opacity_opt = AdamF32::new(OPACITY_LR, 0.9, 0.999, 1e-8);
    let mut color_opt = AdamVec3::new(COLOR_LR, 0.9, 0.999, 1e-8);
    opacity_opt.ensure_len(n);
    color_opt.ensure_len(n);

    let opts = RenderOptions {
        tile_rasterizer,
        ..Default::default()
    };

    let mut checkpoints = Vec::new();

    for iter in 1..=N_ITERS {
        // Sync trainable params back into the working gaussians (matches
        // src/optim/trainer.rs's per-iter param->struct write-back convention).
        for (i, g) in gaussians.iter_mut().enumerate() {
            g.opacity = opacity_logits[i];
            g.sh_coeffs[0] = [dc_colors[i].x, dc_colors[i].y, dc_colors[i].z];
        }

        // Forward-only pass for loss/PSNR.
        let rendered = gpu
            .render_with_options(&gaussians, camera, bg, opts)
            .expect("forward render failed");
        let (_loss, d_image) = l2_image_loss_and_grad(&rendered, target);

        if iter % CHECKPOINT_INTERVAL == 0 {
            let psnr = compute_psnr(&rendered, target);
            checkpoints.push((iter, psnr));
        }

        // Separate forward+backward call (matches the real trainer's two-render-call
        // convention; docs/TILE_RASTER_PLAN.md Stage 5d note).
        let (_pixels2, grads2d) = gpu
            .render_with_gradients_and_options(&gaussians, camera, bg, &d_image, opts)
            .expect("backward render failed");

        // dL/d(sh_coeffs[0]) via color = sh_coeffs[0]*SH_C0 + 0.5 (src/core/sh.rs).
        let d_dc: Vec<Vector3<f32>> = grads2d.d_colors.iter().map(|c| *c * SH_C0).collect();

        opacity_opt.step(&mut opacity_logits, &grads2d.d_opacity_logits);
        color_opt.step(&mut dc_colors, &d_dc);
    }

    checkpoints
}

#[test]
#[ignore]
fn tile_raster_training_equivalence() {
    let Some(gpu) = gpu_skip::try_gpu_renderer() else {
        return;
    };

    let (gt_gaussians, camera, bg) = regression_scene();

    // Ground truth: naive GPU render of the UNPERTURBED scene, once.
    let target = gpu
        .render(&gt_gaussians, &camera, &bg)
        .expect("ground truth render failed");

    let init_gaussians = perturbed_init(&gt_gaussians);

    eprintln!("=== calibration: naive vs naive (atomic-order noise floor) ===");
    let naive_run_a = run_training(&gpu, &init_gaussians, &target, &camera, &bg, false);
    let naive_run_b = run_training(&gpu, &init_gaussians, &target, &camera, &bg, false);

    assert_eq!(naive_run_a.len(), naive_run_b.len());
    let mut max_drift = 0.0f32;
    for ((iter_a, psnr_a), (iter_b, psnr_b)) in naive_run_a.iter().zip(naive_run_b.iter()) {
        assert_eq!(iter_a, iter_b);
        let drift = (psnr_a - psnr_b).abs();
        max_drift = max_drift.max(drift);
        eprintln!(
            "  iter {iter_a:>4}: naive_a={psnr_a:.4} dB  naive_b={psnr_b:.4} dB  drift={drift:.4} dB"
        );
    }
    eprintln!("naive-vs-naive max drift across checkpoints: {max_drift:.4} dB");

    eprintln!("=== tiled run ===");
    let tiled_run = run_training(&gpu, &init_gaussians, &target, &camera, &bg, true);
    assert_eq!(tiled_run.len(), naive_run_a.len());
    for ((iter_a, psnr_a), (iter_t, psnr_t)) in naive_run_a.iter().zip(tiled_run.iter()) {
        assert_eq!(iter_a, iter_t);
        eprintln!(
            "  iter {iter_a:>4}: naive_a={psnr_a:.4} dB  tiled={psnr_t:.4} dB  delta={:.4} dB",
            (psnr_a - psnr_t).abs()
        );
    }

    let (final_iter_a, final_naive_psnr) = *naive_run_a.last().unwrap();
    let (final_iter_t, final_tiled_psnr) = *tiled_run.last().unwrap();
    assert_eq!(final_iter_a, N_ITERS);
    assert_eq!(final_iter_t, N_ITERS);

    let gate = (0.3f32).max(3.0 * max_drift);
    let final_delta = (final_tiled_psnr - final_naive_psnr).abs();

    eprintln!(
        "\n=== Stage 5e gate ===\nnaive-vs-naive noise floor (max drift): {max_drift:.4} dB\ngate = max(0.3, 3*{max_drift:.4}) = {gate:.4} dB\nfinal iter {N_ITERS}: naive={final_naive_psnr:.4} dB  tiled={final_tiled_psnr:.4} dB  |delta|={final_delta:.4} dB"
    );

    assert!(
        final_delta < gate,
        "tile-raster training trajectory diverged from naive oracle beyond the calibrated \
         noise floor: |{final_tiled_psnr} - {final_naive_psnr}| = {final_delta} dB >= gate {gate} dB"
    );
}
