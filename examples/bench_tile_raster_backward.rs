//! Stage-5c benchmark: tile-binned BACKWARD pass vs naive oracle backward
//! (docs/TILE_RASTER_PLAN.md Part B Stage 5c).
//!
//! Same synthetic seeded scene/camera matrix as examples/bench_tile_raster.rs (Stage 4
//! forward bench): N in {60k, 150k, 400k} gaussians x res in {490x273, 980x545}. This
//! bench times `render_with_gradients` (naive oracle backward) vs
//! `render_with_gradients_and_options(..., RenderOptions{tile_rasterizer:true,..})` (tiled
//! backward, Stage 5b) with a constant upstream gradient `d_pixels`. It also reports a
//! gradient-parity max-relative-diff column on `d_mean_px` (representative field; see
//! tests/unit_gpu_tile_backward_gradients.rs for the full per-field parity gate).
//!
//! Run ONLY on an idle GPU (never beside a training arm — Metal's cumulative
//! command-buffer watchdog plus contention would skew everything):
//!   cargo run --release --features gpu --example bench_tile_raster_backward
//!
//! Per the plan's banding contract: if any unbanded tiled dispatch exceeds ~1s, that is
//! the signal to band by tile rows (not implemented here — this bench only measures and
//! flags).

use nalgebra::{UnitQuaternion, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sugar_rs::core::{Camera, Gaussian, GaussianCloud};
use sugar_rs::gpu::{GpuRenderer, RenderOptions};
use std::time::Instant;

fn sh_constant_color(r: f32, g: f32, b: f32) -> [[f32; 3]; 16] {
    // DC-only color: SH basis Y00 = 0.28209479; color = 0.5 + DC * Y00.
    let c = 0.282_094_79_f32;
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [(r - 0.5) / c, (g - 0.5) / c, (b - 0.5) / c];
    sh
}

fn bench_scene(n: usize, seed: u64) -> Vec<Gaussian> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let position = Vector3::new(
                rng.gen_range(-3.0..3.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(2.0..12.0),
            );
            let scale = Vector3::new(
                rng.gen_range(-4.5..-2.5),
                rng.gen_range(-4.5..-2.5),
                rng.gen_range(-4.5..-2.5),
            );
            let rotation = UnitQuaternion::from_euler_angles(
                rng.gen_range(0.0..std::f32::consts::TAU),
                rng.gen_range(0.0..std::f32::consts::TAU),
                rng.gen_range(0.0..std::f32::consts::TAU),
            );
            let opacity_logit = rng.gen_range(-2.0..2.0);
            let color = sh_constant_color(rng.gen(), rng.gen(), rng.gen());
            Gaussian::new(position, scale, rotation, opacity_logit, color)
        })
        .collect()
}

fn camera(width: u32, height: u32) -> Camera {
    let f = 0.9 * width as f32;
    Camera {
        width,
        height,
        fx: f,
        fy: f,
        cx: width as f32 / 2.0,
        cy: height as f32 / 2.0,
        rotation: nalgebra::Matrix3::identity(),
        translation: Vector3::new(0.0, 0.0, 0.0),
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// Max relative diff over `d_mean_px` between naive and tiled backward gradients —
/// representative field for the parity column (full per-field gate lives in
/// tests/unit_gpu_tile_backward_gradients.rs).
fn grad_parity_max_rel(
    naive: &sugar_rs::gpu::GaussianGradients2D,
    tiled: &sugar_rs::gpu::GaussianGradients2D,
) -> f32 {
    let mut max_rel = 0.0f32;
    for (a, b) in naive.d_mean_px.iter().zip(tiled.d_mean_px.iter()) {
        for c in 0..2 {
            let av = if c == 0 { a.x } else { a.y };
            let bv = if c == 0 { b.x } else { b.y };
            let diff = (av - bv).abs();
            let denom = av.abs().max(bv.abs()).max(1e-8);
            max_rel = max_rel.max(diff / denom);
        }
    }
    max_rel
}

fn main() {
    let gpu = GpuRenderer::new().expect("GPU required for this benchmark");
    let bg = Vector3::new(0.1, 0.1, 0.12);
    const N_RUNS: usize = 3;
    const BANDING_TRIGGER_MS: f64 = 1000.0;
    let mut banding_triggered = false;

    println!(
        "{:>7} {:>9} | {:>12} {:>12} {:>8} | {:>10} {:>7} | grad_parity(max_rel d_mean_px)",
        "N", "res", "naive_ms", "tiled_ms", "speedup", "pairs", "K"
    );
    for &n in &[60_000usize, 150_000, 400_000] {
        let gaussians = bench_scene(n, 42);
        for &(w, h) in &[(490u32, 273u32), (980, 545)] {
            let cam = camera(w, h);
            let num_pixels = (w * h) as usize;
            let d_pixels = vec![Vector3::new(1.0f32, -0.5, 0.25); num_pixels];

            // Warmup both paths (pipeline compilation, first-touch allocations) — also used
            // for the grad-parity column.
            let (naive_img, naive_grads) = gpu
                .render_with_gradients(&gaussians, &cam, &bg, &d_pixels)
                .expect("naive warmup");
            let (tiled_img, tiled_grads) = gpu
                .render_with_gradients_and_options(
                    &gaussians,
                    &cam,
                    &bg,
                    &d_pixels,
                    RenderOptions { tile_rasterizer: true, ..Default::default() },
                )
                .expect("tiled warmup");

            // Sanity: forward images should not be zeroed-out (watchdog failure mode).
            let naive_sum: f32 = naive_img.iter().map(|p| p.x + p.y + p.z).sum();
            let tiled_sum: f32 = tiled_img.iter().map(|p| p.x + p.y + p.z).sum();
            if naive_sum == 0.0 || tiled_sum == 0.0 {
                eprintln!(
                    "WARNING: N={n} res={w}x{h}: zeroed forward output (naive_sum={naive_sum}, tiled_sum={tiled_sum}) — possible watchdog failure"
                );
            }

            let grad_parity = grad_parity_max_rel(&naive_grads, &tiled_grads);

            let (_, counts) = gpu
                .debug_tile_touch_counts(&gaussians, &cam)
                .expect("counts");
            let pairs: u64 = counts.iter().map(|&c| c as u64).sum();
            let live = counts.iter().filter(|&&c| c > 0).count().max(1);
            let k = pairs as f64 / live as f64;

            let mut naive_times = Vec::new();
            for _ in 0..N_RUNS {
                let t = Instant::now();
                let _ = gpu
                    .render_with_gradients(&gaussians, &cam, &bg, &d_pixels)
                    .expect("naive");
                naive_times.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let mut tiled_times = Vec::new();
            for _ in 0..N_RUNS {
                let t = Instant::now();
                let _ = gpu
                    .render_with_gradients_and_options(
                        &gaussians,
                        &cam,
                        &bg,
                        &d_pixels,
                        RenderOptions { tile_rasterizer: true, ..Default::default() },
                    )
                    .expect("tiled");
                tiled_times.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let nm = median(naive_times);
            let tm = median(tiled_times);
            if tm > BANDING_TRIGGER_MS {
                banding_triggered = true;
                eprintln!(
                    "BANDING TRIGGER: N={n} res={w}x{h}: unbanded tiled backward = {tm:.1}ms > {BANDING_TRIGGER_MS}ms"
                );
            }
            println!(
                "{:>7} {:>4}x{:<4} | {:>12.1} {:>12.1} {:>7.2}x | {:>10} {:>7.2} | {:.3e}",
                n, w, h, nm, tm, nm / tm, pairs, k, grad_parity
            );
        }
    }

    if banding_triggered {
        println!(
            "\nVERDICT: banding TRIGGERED (some unbanded tiled backward dispatch > {BANDING_TRIGGER_MS}ms) — see docs/TILE_RASTER_PLAN.md Stage 5c banding contract (tile_row_offset param); NOT implemented by this bench."
        );
    } else {
        println!(
            "\nVERDICT: banding NOT triggered — all unbanded tiled backward dispatches stayed under {BANDING_TRIGGER_MS}ms. Shipping unbanded."
        );
    }

    // Keep the cloud type in the build graph (mirrors other examples' usage patterns).
    let _ = GaussianCloud::from_gaussians(Vec::new());
}
