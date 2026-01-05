//! Diagnostic: Analyze Gaussian scale and rotation health
//!
//! Run with: cargo test --test diagnostic_gaussian_health -- --nocapture --ignored
//!
//! This test loads Gaussians and reports statistics on:
//! - Scale distribution (log-space)
//! - Anisotropy (aspect ratios)
//! - Rotation distribution
//! - Identifies problematic "needle" Gaussians

use nalgebra::Vector3;
use std::path::PathBuf;

/// Analyze scale distribution and identify issues
fn analyze_scales(scales_log: &[Vector3<f32>]) -> ScaleAnalysis {
    let n = scales_log.len();
    if n == 0 {
        return ScaleAnalysis::default();
    }

    // Convert log-scales to linear for analysis
    let scales_linear: Vec<Vector3<f32>> = scales_log
        .iter()
        .map(|s| Vector3::new(s.x.exp(), s.y.exp(), s.z.exp()))
        .collect();

    // Compute per-axis statistics
    let mut x_vals: Vec<f32> = scales_linear.iter().map(|s| s.x).collect();
    let mut y_vals: Vec<f32> = scales_linear.iter().map(|s| s.y).collect();
    let mut z_vals: Vec<f32> = scales_linear.iter().map(|s| s.z).collect();

    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    y_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    z_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute anisotropy (max/min scale ratio) for each Gaussian
    let mut anisotropies: Vec<f32> = scales_linear
        .iter()
        .map(|s| {
            let max_s = s.x.max(s.y).max(s.z);
            let min_s = s.x.min(s.y).min(s.z);
            if min_s > 1e-10 {
                max_s / min_s
            } else {
                f32::INFINITY
            }
        })
        .collect();
    anisotropies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Count needles (anisotropy > 10)
    let needles_10x = anisotropies.iter().filter(|&&a| a > 10.0).count();
    let needles_20x = anisotropies.iter().filter(|&&a| a > 20.0).count();
    let needles_50x = anisotropies.iter().filter(|&&a| a > 50.0).count();

    // Compute volume (product of scales)
    let mut volumes: Vec<f32> = scales_linear
        .iter()
        .map(|s| s.x * s.y * s.z)
        .collect();
    volumes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    ScaleAnalysis {
        count: n,
        scale_x_median: x_vals[n / 2],
        scale_y_median: y_vals[n / 2],
        scale_z_median: z_vals[n / 2],
        scale_x_p5: x_vals[n * 5 / 100],
        scale_x_p95: x_vals[n * 95 / 100],
        scale_y_p5: y_vals[n * 5 / 100],
        scale_y_p95: y_vals[n * 95 / 100],
        scale_z_p5: z_vals[n * 5 / 100],
        scale_z_p95: z_vals[n * 95 / 100],
        anisotropy_median: anisotropies[n / 2],
        anisotropy_p90: anisotropies[n * 90 / 100],
        anisotropy_p99: anisotropies[n * 99 / 100],
        anisotropy_max: *anisotropies.last().unwrap_or(&0.0),
        needles_10x,
        needles_20x,
        needles_50x,
        volume_median: volumes[n / 2],
        volume_p5: volumes[n * 5 / 100],
        volume_p95: volumes[n * 95 / 100],
    }
}

#[derive(Default, Debug)]
struct ScaleAnalysis {
    count: usize,
    scale_x_median: f32,
    scale_y_median: f32,
    scale_z_median: f32,
    scale_x_p5: f32,
    scale_x_p95: f32,
    scale_y_p5: f32,
    scale_y_p95: f32,
    scale_z_p5: f32,
    scale_z_p95: f32,
    anisotropy_median: f32,
    anisotropy_p90: f32,
    anisotropy_p99: f32,
    anisotropy_max: f32,
    needles_10x: usize,
    needles_20x: usize,
    needles_50x: usize,
    volume_median: f32,
    volume_p5: f32,
    volume_p95: f32,
}

impl ScaleAnalysis {
    fn print_report(&self) {
        println!("\n=== Scale Analysis ({} Gaussians) ===\n", self.count);

        println!("Per-axis scale (linear, 5th/median/95th percentile):");
        println!("  X: {:.4} / {:.4} / {:.4}", self.scale_x_p5, self.scale_x_median, self.scale_x_p95);
        println!("  Y: {:.4} / {:.4} / {:.4}", self.scale_y_p5, self.scale_y_median, self.scale_y_p95);
        println!("  Z: {:.4} / {:.4} / {:.4}", self.scale_z_p5, self.scale_z_median, self.scale_z_p95);

        println!("\nAnisotropy (max/min scale ratio):");
        println!("  Median: {:.2}x", self.anisotropy_median);
        println!("  P90:    {:.2}x", self.anisotropy_p90);
        println!("  P99:    {:.2}x", self.anisotropy_p99);
        println!("  Max:    {:.2}x", self.anisotropy_max);

        println!("\nNeedle Gaussians (highly stretched):");
        println!("  >10x aspect ratio: {} ({:.1}%)", self.needles_10x, 100.0 * self.needles_10x as f32 / self.count as f32);
        println!("  >20x aspect ratio: {} ({:.1}%)", self.needles_20x, 100.0 * self.needles_20x as f32 / self.count as f32);
        println!("  >50x aspect ratio: {} ({:.1}%)", self.needles_50x, 100.0 * self.needles_50x as f32 / self.count as f32);

        println!("\nVolume (product of scales):");
        println!("  P5/Median/P95: {:.6} / {:.6} / {:.6}", self.volume_p5, self.volume_median, self.volume_p95);

        // Health assessment
        println!("\n=== Health Assessment ===");
        if self.needles_20x as f32 / self.count as f32 > 0.05 {
            println!("  WARNING: >5% of Gaussians are needles (>20x aspect ratio)");
            println!("           This causes blur and may indicate gradient issues");
        }
        if self.anisotropy_p90 > 15.0 {
            println!("  WARNING: P90 anisotropy is {:.1}x (target: <10x)", self.anisotropy_p90);
        }
        if self.scale_x_p95 > 1.0 || self.scale_y_p95 > 1.0 || self.scale_z_p95 > 1.0 {
            println!("  WARNING: Some Gaussians are very large (scale > 1.0)");
        }
        if self.scale_x_p5 < 0.001 || self.scale_y_p5 < 0.001 || self.scale_z_p5 < 0.001 {
            println!("  WARNING: Some Gaussians are very small (scale < 0.001)");
        }
        if self.needles_20x == 0 && self.anisotropy_p90 < 10.0 {
            println!("  GOOD: Gaussian shapes appear healthy");
        }
    }
}

#[test]
#[ignore]
fn analyze_colmap_initial_scales() {
    use sugar_rs::io::load_colmap_scene;

    println!("\n========================================");
    println!("  COLMAP Initial Point Cloud Analysis");
    println!("========================================");

    let sparse_dir = PathBuf::from("datasets/train/sparse/0");
    if !sparse_dir.exists() {
        println!("Dataset not found at {:?}", sparse_dir);
        return;
    }

    let scene = load_colmap_scene(&sparse_dir).expect("Failed to load COLMAP scene");

    println!("\nLoaded {} points from COLMAP", scene.points.len());

    // COLMAP points are initialized with uniform small scale
    // Let's see what the typical initial scale would be
    let initial_log_scale = -4.0f32; // exp(-4) ≈ 0.018
    println!("\nTypical initial log-scale: {} (linear: {:.4})", initial_log_scale, initial_log_scale.exp());

    // Simulate initial Gaussians
    let scales_log: Vec<Vector3<f32>> = (0..scene.points.len())
        .map(|_| Vector3::new(initial_log_scale, initial_log_scale, initial_log_scale))
        .collect();

    let analysis = analyze_scales(&scales_log);
    analysis.print_report();
}

#[test]
#[ignore]
fn analyze_checkpoint_scales() {
    use sugar_rs::io::load_model;

    println!("\n========================================");
    println!("  Checkpoint Gaussian Analysis");
    println!("========================================");

    // Find the most recent .gs file
    let runs_dir = PathBuf::from("runs");
    let mut gs_files: Vec<PathBuf> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&runs_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Ok(subentries) = std::fs::read_dir(&path) {
                    for subentry in subentries.flatten() {
                        let subpath = subentry.path();
                        if subpath.extension().and_then(|s| s.to_str()) == Some("gs") {
                            gs_files.push(subpath);
                        }
                    }
                }
            }
        }
    }

    if gs_files.is_empty() {
        println!("No .gs checkpoint files found in runs/");
        println!("Run training first or specify a checkpoint path");
        return;
    }

    // Sort by modification time (newest first)
    gs_files.sort_by(|a, b| {
        let a_time = std::fs::metadata(a).and_then(|m| m.modified()).ok();
        let b_time = std::fs::metadata(b).and_then(|m| m.modified()).ok();
        b_time.cmp(&a_time)
    });

    let checkpoint_path = &gs_files[0];
    println!("\nAnalyzing: {:?}", checkpoint_path);

    let (cloud, metadata) = load_model(checkpoint_path.to_str().unwrap())
        .expect("Failed to load checkpoint");

    println!("Loaded {} Gaussians", cloud.gaussians.len());
    println!("Training iterations: {}", metadata.training_iterations);

    // Extract scales
    let scales_log: Vec<Vector3<f32>> = cloud.gaussians
        .iter()
        .map(|g| g.scale)
        .collect();

    let analysis = analyze_scales(&scales_log);
    analysis.print_report();

    // Also analyze worst offenders
    println!("\n=== Worst Needle Gaussians ===");
    let mut indexed_anisotropies: Vec<(usize, f32, Vector3<f32>)> = cloud.gaussians
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let s = Vector3::new(g.scale.x.exp(), g.scale.y.exp(), g.scale.z.exp());
            let max_s = s.x.max(s.y).max(s.z);
            let min_s = s.x.min(s.y).min(s.z);
            let aniso = if min_s > 1e-10 { max_s / min_s } else { f32::INFINITY };
            (i, aniso, s)
        })
        .collect();

    indexed_anisotropies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (i, (idx, aniso, scale)) in indexed_anisotropies.iter().take(10).enumerate() {
        let g = &cloud.gaussians[*idx];
        println!("  {}. Gaussian {}: {:.1}x anisotropy", i + 1, idx, aniso);
        println!("     Scale (linear): [{:.4}, {:.4}, {:.4}]", scale.x, scale.y, scale.z);
        println!("     Position: [{:.2}, {:.2}, {:.2}]", g.position.x, g.position.y, g.position.z);
    }
}

#[test]
#[ignore]
fn analyze_scale_gradient_finite_diff() {
    use nalgebra::{Matrix3, UnitQuaternion};
    use sugar_rs::core::{Camera, Gaussian};
    use sugar_rs::render::{render_full_linear, render_full_color_grads};

    println!("\n========================================");
    println!("  Scale Gradient Finite Difference Test");
    println!("========================================");

    // Create a simple scene
    let camera = Camera::new(
        100.0, 100.0, 32.0, 32.0, 64, 64,
        Matrix3::identity(),
        nalgebra::Vector3::zeros(),
    );

    // Test Gaussian with moderate anisotropy
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [0.5, 0.3, 0.2];

    let base_scale = Vector3::new(-2.0, -3.0, -2.5); // Anisotropic
    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 3.0),
        base_scale,
        UnitQuaternion::identity(),
        1.0,
        sh,
    );

    let bg = Vector3::zeros();

    // Render target (slightly different scale)
    let target_gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 3.0),
        Vector3::new(-2.5, -2.5, -2.5), // More spherical target
        UnitQuaternion::identity(),
        1.0,
        sh,
    );
    let target = render_full_linear(&[target_gaussian], &camera, &bg, false);
    let rendered = render_full_linear(&[gaussian.clone()], &camera, &bg, false);

    // Compute L2 loss gradient
    let d_image: Vec<Vector3<f32>> = rendered
        .iter()
        .zip(target.iter())
        .map(|(r, t)| 2.0 * (r - t))
        .collect();

    // Get analytical gradients
    let (_, _, _, _, d_log_scales, _, _) =
        render_full_color_grads(&[gaussian.clone()], &camera, &d_image, &bg, false);

    let analytical = d_log_scales[0];

    // Compute numerical gradients via finite differences
    let eps = 1e-4;
    let mut numerical = Vector3::zeros();

    for axis in 0..3 {
        let loss_fn = |delta: f32| -> f32 {
            let mut scale = base_scale;
            scale[axis] += delta;
            let g = Gaussian::new(
                gaussian.position,
                scale,
                gaussian.rotation,
                gaussian.opacity,
                gaussian.sh_coeffs,
            );
            let rendered = render_full_linear(&[g], &camera, &bg, false);
            rendered
                .iter()
                .zip(target.iter())
                .map(|(r, t)| (r - t).norm_squared())
                .sum::<f32>()
        };

        numerical[axis] = (loss_fn(eps) - loss_fn(-eps)) / (2.0 * eps);
    }

    println!("\nScale gradients (log-space):");
    println!("  Analytical: [{:.6}, {:.6}, {:.6}]", analytical.x, analytical.y, analytical.z);
    println!("  Numerical:  [{:.6}, {:.6}, {:.6}]", numerical.x, numerical.y, numerical.z);

    let diff = (analytical - numerical).norm();
    let rel_err = diff / (numerical.norm() + 1e-8);

    println!("\n  Absolute diff: {:.6}", diff);
    println!("  Relative error: {:.4}%", rel_err * 100.0);

    if rel_err < 0.05 {
        println!("\n  PASS: Scale gradients match within 5%");
    } else {
        println!("\n  FAIL: Scale gradient mismatch > 5%");
    }

    // Interpret the gradient direction
    println!("\nGradient interpretation:");
    println!("  Current scale (linear): [{:.4}, {:.4}, {:.4}]",
             base_scale.x.exp(), base_scale.y.exp(), base_scale.z.exp());
    println!("  Target scale (linear):  [{:.4}, {:.4}, {:.4}]",
             (-2.5f32).exp(), (-2.5f32).exp(), (-2.5f32).exp());

    for axis in 0..3 {
        let axis_name = ["X", "Y", "Z"][axis];
        let grad = analytical[axis];
        let current = base_scale[axis].exp();
        let target_s = (-2.5f32).exp();

        let expected_direction = if current > target_s { "shrink" } else { "grow" };
        let actual_direction = if grad > 0.0 { "shrink" } else { "grow" };

        let correct = expected_direction == actual_direction;
        println!("  {}: gradient={:+.4} -> {} (expected: {}) {}",
                 axis_name, grad, actual_direction, expected_direction,
                 if correct { "OK" } else { "WRONG" });
    }
}
