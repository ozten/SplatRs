//! GPU rendering benchmark on T&T train dataset.
//!
//! This example compares CPU vs GPU rendering performance on a real training scenario.
//!
//! Usage:
//!   cargo run --release --features gpu --example train_gpu_benchmark

use nalgebra::{Matrix3, Vector3};
use std::path::PathBuf;
use std::time::Instant;
use sugar_rs::core::{init_from_colmap_points_visible_stratified, Camera};
use sugar_rs::io::load_colmap_scene;
use sugar_rs::render::render_full_linear;

#[cfg(feature = "gpu")]
use sugar_rs::gpu::GpuRenderer;

fn main() {
    let sparse_dir = PathBuf::from("datasets/tandt_db/tandt/train/sparse/0");
    let images_dir = PathBuf::from("datasets/tandt_db/tandt/train/images");

    if !sparse_dir.exists() {
        eprintln!("Error: T&T train dataset not found at {:?}", sparse_dir);
        eprintln!("Expected structure:");
        eprintln!("  datasets/tandt_db/tandt/train/sparse/0/");
        eprintln!("  datasets/tandt_db/tandt/train/images/");
        std::process::exit(1);
    }

    println!("Loading COLMAP scene...");
    let scene = load_colmap_scene(&sparse_dir).expect("Failed to load scene");
    println!("  Cameras: {}", scene.cameras.len());
    println!("  Images:  {}", scene.images.len());
    println!("  Points:  {}", scene.points.len());

    // Use first image
    let image_info = &scene.images[0];
    let base_camera = scene
        .cameras
        .get(&image_info.camera_id)
        .expect("Camera not found");

    let rotation = image_info.rotation.to_rotation_matrix().into_inner();
    let camera_full = Camera::new(
        base_camera.fx,
        base_camera.fy,
        base_camera.cx,
        base_camera.cy,
        base_camera.width,
        base_camera.height,
        rotation,
        image_info.translation,
    );

    // Downsample for faster benchmark
    let downsample = 0.25;
    let camera = Camera::new(
        camera_full.fx * downsample,
        camera_full.fy * downsample,
        camera_full.cx * downsample,
        camera_full.cy * downsample,
        ((camera_full.width as f32) * downsample).round() as u32,
        ((camera_full.height as f32) * downsample).round() as u32,
        camera_full.rotation,
        camera_full.translation,
    );

    println!("\nCamera resolution: {}×{}", camera.width, camera.height);

    // Initialize Gaussians
    let max_gaussians = 10_000;
    println!("Initializing {} Gaussians...", max_gaussians);
    let cloud = init_from_colmap_points_visible_stratified(&scene.points, &camera, max_gaussians, 8);
    let gaussians = cloud.gaussians;
    println!("  Actually got: {} Gaussians", gaussians.len());

    let background = Vector3::new(0.0, 0.0, 0.0);

    // CPU Benchmark
    println!("\n=== CPU Rendering Benchmark ===");
    let cpu_warmup = render_full_linear(&gaussians, &camera, &background);
    println!("  Warmup complete");

    let cpu_iterations = 10;
    let mut cpu_times = Vec::new();

    for i in 0..cpu_iterations {
        let start = Instant::now();
        let _result = render_full_linear(&gaussians, &camera, &background);
        let elapsed = start.elapsed();
        cpu_times.push(elapsed.as_secs_f64());
        println!("  Iteration {}: {:.3}s", i + 1, elapsed.as_secs_f64());
    }

    let cpu_avg = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;
    let cpu_min = cpu_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let cpu_max = cpu_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("  Average: {:.3}s", cpu_avg);
    println!("  Min:     {:.3}s", cpu_min);
    println!("  Max:     {:.3}s", cpu_max);

    // GPU Benchmark
    #[cfg(feature = "gpu")]
    {
        println!("\n=== GPU Rendering Benchmark ===");
        println!("Initializing GPU...");
        let gpu_renderer = GpuRenderer::new().expect("Failed to create GPU renderer");

        println!("  Warmup render...");
        let gpu_warmup = gpu_renderer.render(&gaussians, &camera, &background);
        println!("  Warmup complete");

        // Verify correctness
        println!("  Verifying GPU vs CPU correctness...");
        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0;
        let mut num_large_diffs = 0;
        let mut total_diff = 0.0f32;

        for (i, (cpu, gpu)) in cpu_warmup.iter().zip(gpu_warmup.iter()).enumerate() {
            let diff = (cpu - gpu).norm();
            total_diff += diff;
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
            if diff > 0.1 {
                num_large_diffs += 1;
            }
        }

        let avg_diff = total_diff / cpu_warmup.len() as f32;
        let px_x = max_diff_idx % camera.width as usize;
        let px_y = max_diff_idx / camera.width as usize;

        println!("  Average difference:   {:.6}", avg_diff);
        println!("  Max pixel difference: {:.6}", max_diff);
        println!("  Max diff at pixel:    ({}, {})", px_x, px_y);
        println!("  CPU value:            {:?}", cpu_warmup[max_diff_idx]);
        println!("  GPU value:            {:?}", gpu_warmup[max_diff_idx]);
        println!("  Large diffs (>0.1):   {} / {} ({:.1}%)",
            num_large_diffs,
            cpu_warmup.len(),
            100.0 * num_large_diffs as f32 / cpu_warmup.len() as f32
        );

        if max_diff > 1e-3 {
            eprintln!("  ⚠️  Warning: GPU differs from CPU significantly!");
            eprintln!("      This suggests a bug in GPU renderer with real-world data.");
        } else {
            println!("  ✅ GPU output matches CPU");
        }

        let gpu_iterations = 10;
        let mut gpu_times = Vec::new();

        for i in 0..gpu_iterations {
            let start = Instant::now();
            let _result = gpu_renderer.render(&gaussians, &camera, &background);
            let elapsed = start.elapsed();
            gpu_times.push(elapsed.as_secs_f64());
            println!("  Iteration {}: {:.3}s", i + 1, elapsed.as_secs_f64());
        }

        let gpu_avg = gpu_times.iter().sum::<f64>() / gpu_times.len() as f64;
        let gpu_min = gpu_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let gpu_max = gpu_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("  Average: {:.3}s", gpu_avg);
        println!("  Min:     {:.3}s", gpu_min);
        println!("  Max:     {:.3}s", gpu_max);

        println!("\n=== Performance Comparison ===");
        let speedup = cpu_avg / gpu_avg;
        println!("  CPU average: {:.3}s", cpu_avg);
        println!("  GPU average: {:.3}s", gpu_avg);
        println!("  Speedup:     {:.2}x", speedup);

        if speedup > 1.0 {
            println!("  ✅ GPU is {:.1}% faster", (speedup - 1.0) * 100.0);
        } else {
            println!("  ⚠️  GPU is {:.1}% slower (needs optimization)", (1.0 - speedup) * 100.0);
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("\n⚠️  GPU benchmarking skipped - compile with --features gpu");
    }

    println!("\n✅ Benchmark complete!");
}
