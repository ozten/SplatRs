//! Stage-4 benchmark: tile-binned rasterizer vs naive oracle (docs/TILE_RASTER_PLAN.md).
//!
//! Synthetic seeded scenes at 60k/150k/400k gaussians, rendered at half-res (490x273)
//! and full-res (980x545). Reports median wall time over N_RUNS for each path plus
//! binning stats (total pair count, avg tiles/gaussian) and a parity check per config.
//!
//! Run ONLY on an idle GPU (never beside a training arm — Metal's cumulative
//! command-buffer watchdog plus contention would skew everything):
//!   cargo run --release --features gpu --example bench_tile_raster
//!
//! The tile path here is the correctness-first v1 (lazy pipelines, CPU prefix sum,
//! several readbacks) — its numbers INCLUDE that overhead. The interesting comparison
//! is the scaling shape: naive is O(pixels * N), tiled is O(pixels * local density).

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

fn main() {
    let gpu = GpuRenderer::new().expect("GPU required for this benchmark");
    let bg = Vector3::new(0.1, 0.1, 0.12);
    const N_RUNS: usize = 3;

    println!(
        "{:>7} {:>9} | {:>12} {:>12} {:>8} | {:>10} {:>7} | parity(max)",
        "N", "res", "naive_ms", "tiled_ms", "speedup", "pairs", "K"
    );
    for &n in &[60_000usize, 150_000, 400_000] {
        let gaussians = bench_scene(n, 42);
        for &(w, h) in &[(490u32, 273u32), (980, 545)] {
            let cam = camera(w, h);
            // Warmup both paths (pipeline compilation, first-touch allocations).
            let naive_out = gpu.render(&gaussians, &cam, &bg).expect("naive warmup");
            let tiled_out = gpu
                .render_with_options(
                    &gaussians,
                    &cam,
                    &bg,
                    RenderOptions { tile_rasterizer: true, ..Default::default() },
                )
                .expect("tiled warmup");
            let max_diff = naive_out
                .iter()
                .zip(tiled_out.iter())
                .map(|(a, b)| (a - b).abs().max())
                .fold(0.0f32, f32::max);

            let (_, counts) = gpu
                .debug_tile_touch_counts(&gaussians, &cam)
                .expect("counts");
            let pairs: u64 = counts.iter().map(|&c| c as u64).sum();
            let live = counts.iter().filter(|&&c| c > 0).count().max(1);
            let k = pairs as f64 / live as f64;

            let mut naive_times = Vec::new();
            for _ in 0..N_RUNS {
                let t = Instant::now();
                let _ = gpu.render(&gaussians, &cam, &bg).expect("naive");
                naive_times.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let mut tiled_times = Vec::new();
            for _ in 0..N_RUNS {
                let t = Instant::now();
                let _ = gpu
                    .render_with_options(
                        &gaussians,
                        &cam,
                        &bg,
                        RenderOptions { tile_rasterizer: true, ..Default::default() },
                    )
                    .expect("tiled");
                tiled_times.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let nm = median(naive_times);
            let tm = median(tiled_times);
            println!(
                "{:>7} {:>4}x{:<4} | {:>12.1} {:>12.1} {:>7.2}x | {:>10} {:>7.2} | {:.2e}",
                n, w, h, nm, tm, nm / tm, pairs, k, max_diff
            );
        }
    }
    // Keep the cloud type in the build graph (mirrors other examples' usage patterns).
    let _ = GaussianCloud::from_gaussians(Vec::new());
}
