//! sugar-train: Train Gaussian Splatting model from COLMAP scene
//!
//! Usage:
//!   sugar-train --scene path/to/colmap/sparse/0 --output model.ply

mod train_utils;

use sugar_rs::io::{compute_bounds, save_model, Compression, ModelMetadata};
use sugar_rs::core::GaussianCloud;
use std::path::PathBuf;
use train_utils::auto_downsample;

/// Create timestamped run directory
fn create_run_directory(preset_name: &str) -> std::io::Result<PathBuf> {
    use time::OffsetDateTime;

    // Get current local time
    let now = OffsetDateTime::now_utc();

    // Format timestamp as YYYYMMDD_HHMM in UTC
    // Note: Using UTC to avoid timezone issues. If local time is needed,
    // would need to handle platform-specific timezone access.
    let year = now.year();
    let month = now.month() as u8;
    let day = now.day();
    let hour = now.hour();
    let minute = now.minute();

    // Sanitize preset name
    let sanitized_preset = preset_name
        .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");

    let dir_name = format!(
        "runs/{:04}{:02}{:02}_{:02}{:02}_{}",
        year, month, day, hour, minute, sanitized_preset
    );

    let mut path = PathBuf::from(&dir_name);

    // Handle collisions
    let mut counter = 1;
    while path.exists() {
        path = PathBuf::from(format!("{}.{}", dir_name, counter));
        counter += 1;
    }

    std::fs::create_dir_all(&path)?;
    Ok(path)
}

/// Save run metadata to text file
fn save_run_metadata(
    out_dir: &std::path::Path,
    args: &[String],
    seed_used: Option<u64>,
) -> std::io::Result<()> {
    use std::io::Write;
    use std::time::SystemTime;

    let metadata_path = out_dir.join("run_metadata.txt");
    let mut file = std::fs::File::create(metadata_path)?;

    writeln!(file, "=== Training Run Metadata ===")?;
    writeln!(file)?;
    writeln!(file, "Command:")?;
    let binary_name = std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().to_string()))
        .unwrap_or_else(|| "sugar-train".to_string());
    writeln!(file, "{} {}", binary_name, args[1..].join(" "))?;
    writeln!(file)?;

    writeln!(file, "Started: {:?}", SystemTime::now())?;
    writeln!(file)?;

    // Write seed used for reproducibility
    if let Some(seed) = seed_used {
        writeln!(file, "Seed: {}", seed)?;
        writeln!(file)?;
    }

    writeln!(file, "System:")?;
    writeln!(file, "  Platform: {}", std::env::consts::OS)?;
    writeln!(file, "  Architecture: {}", std::env::consts::ARCH)?;
    writeln!(file, "  Package version: {}", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

fn main() {
    println!("sugar-train v{}", sugar_rs::VERSION);

    // Minimal CLI parsing (no external deps).
    // Example:
    //   sugar-train --scene /path/to/sparse/0 --images /path/to/images --iters 200
    let mut args = std::env::args().skip(1);
    let mut scene: Option<std::path::PathBuf> = None;
    let mut images: Option<std::path::PathBuf> = None;
    let mut iters: usize = 200;
    let mut lr: f32 = 0.05;
    let mut lr_position: f32 = 0.00016;
    let mut lr_rotation: f32 = 0.001;
    let mut lr_scale: f32 = 0.005;
    let mut lr_opacity: f32 = 0.05;
    let mut lr_sh: f32 = 0.0025;
    let mut lr_background: f32 = 0.05;
    let mut downsample: f32 = 0.25;
    let mut downsample_explicit: bool = false; // Track if user explicitly set --downsample
    let mut max_gaussians: usize = 20_000;
    let mut image_index: usize = 0;
    let mut log_interval: usize = 10;
    // A3 (resolved 2026-07-06): the learned background stays ON. The historical divergence
    // (bg negative/past 1.0) no longer reproduces — the [0,1] clamp in the trainer plus the
    // A1/A2/B-series fixes removed it — and a 6-run A/B showed the learned background beats a
    // frozen constant by 0.5–1.0 dB in every densify config (it converges to a jointly better
    // constant; "red pinned at 0" is a clamped optimum, not divergence). Use --no-learn-bg to
    // freeze it for experiments.
    let mut learn_background: bool = true;
    let mut learn_opacity: bool = false;
    let mut learn_position: bool = false;
    let mut learn_scale: bool = false;
    let mut learn_rotation: bool = false;
    let mut learn_sh: bool = false;
    let mut loss: sugar_rs::optim::loss::LossKind = sugar_rs::optim::loss::LossKind::L2;
    let mut dataset_root: Option<std::path::PathBuf> = None;
    let mut multiview: bool = false;
    let mut train_fraction: f32 = 0.8;
    let mut val_interval: usize = 50;
    let mut max_test_views_for_metrics: usize = 0;
    let mut max_images: usize = 0;
    let mut out_dir: Option<std::path::PathBuf> = None;
    let mut preset_name: Option<String> = None;
    let mut densify_interval: usize = 0;
    let mut densify_max_gaussians: usize = 0;
    let mut densify_grad_threshold: f32 = 0.1;
    let mut prune_opacity_threshold: f32 = 0.01;
    let mut split_sigma_threshold: f32 = 0.05;
    let mut seed: Option<u64> = None;
    let mut use_gpu: bool = true;
    let mut disable_sh: bool = false;
    // Per-step anisotropy clamp (log-space max−min axis ratio); 0 disables — reference 3DGS
    // has no such clamp (when enabled, the needle prune runs at +0.4, legacy 1.6 → 2.0).
    // Default 3.0 (≈20:1) since the post-renderer-fix A/B (2026-07-08, commit 836f4d0 fixes):
    // 2k trio +0.61/+0.83 dB in the densify arms (19.49 best-ever 2k), 15k settle-mean tie
    // (16.56 vs 16.61) with the needle tail capped at 20:1 vs 161,000:1 unclamped and visibly
    // cleaner renders. (The old "unclamped wins" verdicts were measured on the broken
    // renderer and are void.) Use --max-log-aniso 0 for reference-faithful unclamped.
    let mut max_log_aniso: f32 = 3.0;
    // Settle-phase prune-only pass every N iters after densification stops (0 = off).
    // Default 500 since the 15k A/B (2026-07-09): +0.39 dB final (16.76 vs 16.37) at equal
    // settle mean, no health cost; removes sub-threshold opacity early, then oversize regrowth
    // (~100/pass) that was previously frozen in for the whole settle phase.
    let mut settle_prune_interval: usize = 500;
    // Opacity resets cap down to this value; reference 0.01. Below the prune threshold
    // (micro: 0.005) makes never-recovering mass prunable.
    let mut opacity_reset_floor: f32 = 0.01;

    fn apply_preset(
        name: &str,
        multiview: &mut bool,
        iters: &mut usize,
        lr: &mut f32,
        lr_position: &mut f32,
        lr_rotation: &mut f32,
        lr_scale: &mut f32,
        lr_opacity: &mut f32,
        lr_sh: &mut f32,
        lr_background: &mut f32,
        downsample: &mut f32,
        max_gaussians: &mut usize,
        image_index: &mut usize,
        log_interval: &mut usize,
        learn_background: &mut bool,
        learn_opacity: &mut bool,
        learn_position: &mut bool,
        learn_scale: &mut bool,
        learn_rotation: &mut bool,
        learn_sh: &mut bool,
        loss: &mut sugar_rs::optim::loss::LossKind,
        train_fraction: &mut f32,
        val_interval: &mut usize,
        max_test_views_for_metrics: &mut usize,
        max_images: &mut usize,
        densify_interval: &mut usize,
        densify_max_gaussians: &mut usize,
        densify_grad_threshold: &mut f32,
        prune_opacity_threshold: &mut f32,
        split_sigma_threshold: &mut f32,
        seed: &mut Option<u64>,
    ) -> Result<(), String> {
        match name {
            "m7" => {
                *multiview = false;
                *iters = 1000;
                *lr = 0.05;
                // Use same LR for all params (only color trained in M7)
                *lr_position = *lr;
                *lr_rotation = *lr;
                *lr_scale = *lr;
                *lr_opacity = *lr;
                *lr_sh = *lr;
                *lr_background = *lr;
                *downsample = 0.25;
                *max_gaussians = 20_000;
                *image_index = 0;
                *log_interval = 10;
                *learn_background = true;
                *learn_opacity = false;
                *learn_position = false;
                *learn_scale = false;
                *learn_rotation = false;
                *learn_sh = false;
                *loss = sugar_rs::optim::loss::LossKind::L2;
                *seed = Some(0);
            }
            "m8-smoke" => {
                *multiview = true;
                *iters = 50;
                *lr = 0.01;
                // Use same LR for all params (only color trained)
                *lr_position = *lr;
                *lr_rotation = *lr;
                *lr_scale = *lr;
                *lr_opacity = *lr;
                *lr_sh = *lr;
                *lr_background = *lr;
                *downsample = 0.125;
                *max_gaussians = 2_000;
                *log_interval = 1;
                *learn_background = true;
                *learn_opacity = false;
                *learn_position = false;
                *learn_scale = false;
                *learn_rotation = false;
                *learn_sh = false;
                *loss = sugar_rs::optim::loss::LossKind::L2;
                *train_fraction = 0.8;
                *val_interval = 10;
                *max_test_views_for_metrics = 1;
                *max_images = 5;
                *densify_interval = 0;
                *densify_max_gaussians = 0;
                *seed = Some(0);
            }
            "m8" => {
                *multiview = true;
                *iters = 500;
                *lr = 0.01;
                // Use same LR for all params (only color trained)
                *lr_position = *lr;
                *lr_rotation = *lr;
                *lr_scale = *lr;
                *lr_opacity = *lr;
                *lr_sh = *lr;
                *lr_background = *lr;
                *downsample = 0.25;
                *max_gaussians = 10_000;
                *log_interval = 10;
                *learn_background = true;
                *learn_opacity = false;
                *learn_position = false;
                *learn_scale = false;
                *learn_rotation = false;
                *learn_sh = false;
                *loss = sugar_rs::optim::loss::LossKind::L2;
                *train_fraction = 0.8;
                *val_interval = 50;
                *max_test_views_for_metrics = 0;
                *max_images = 0;
                *densify_interval = 0;
                *densify_max_gaussians = 0;
                *seed = Some(0);
            }
            "m9" => {
                *multiview = true;
                *iters = 1000;
                *lr = 0.01;
                // Use same LR for all params (only color + opacity trained)
                *lr_position = *lr;
                *lr_rotation = *lr;
                *lr_scale = *lr;
                *lr_opacity = *lr;
                *lr_sh = *lr;
                *lr_background = *lr;
                *downsample = 0.25;
                *max_gaussians = 10_000;
                *log_interval = 10;
                *learn_background = true;
                *learn_opacity = true;
                *learn_position = false;
                *learn_scale = false;
                *learn_rotation = false;
                *learn_sh = false;
                *loss = sugar_rs::optim::loss::LossKind::L2;
                *train_fraction = 0.8;
                *val_interval = 50;
                *max_test_views_for_metrics = 0;
                *max_images = 0;
                *densify_interval = 100;
                *densify_max_gaussians = 80_000;
                *densify_grad_threshold = 0.1;
                *prune_opacity_threshold = 0.01;
                *split_sigma_threshold = 0.05;
                *seed = Some(0);
            }
            "micro" => {
                // Fast preset for GPU profiling and UI dev: ~5 minutes
                *multiview = true;
                *iters = 2000;  // Scaled up for 5-minute target
                *lr = 0.002;
                *lr_position = 0.00016;
                *lr_rotation = 0.001;
                *lr_scale = 0.005;
                *lr_opacity = 0.05;
                *lr_sh = 0.0025;
                *lr_background = 0.001;
                *downsample = 0.40;  // 40% resolution (balanced workload within GPU limits)
                *max_gaussians = 8_000;  // More Gaussians for realistic testing
                *log_interval = 100;  // Log every 100 iterations
                *learn_background = true;
                *learn_opacity = true;
                *learn_position = true;
                *learn_scale = true;
                *learn_rotation = true;
                *learn_sh = true;
                *loss = sugar_rs::optim::loss::LossKind::L2;
                *train_fraction = 0.75;  // 15 train, 5 test with 20 images
                *val_interval = 500;  // Validate at 500, 1000, 1500, 2000
                *max_test_views_for_metrics = 3;
                *max_images = 20;  // More views for better testing
                *densify_interval = 500;  // Densify at 500, 1000, 1500
                *densify_max_gaussians = 15_000;  // Higher cap for realistic growth
                *densify_grad_threshold = 0.0002;
                *prune_opacity_threshold = 0.005;
                *split_sigma_threshold = 0.1;
                *seed = Some(123);  // Fixed seed for reproducible, stable training (seed 0 has bad train/test splits)
            }
            "onehour" => {
                // One-hour preset: Good quality preview run
                *multiview = true;
                *iters = 10_000;  // 5x iterations for better convergence
                *lr = 0.002;
                *lr_position = 0.00016;
                *lr_rotation = 0.001;
                *lr_scale = 0.005;
                *lr_opacity = 0.05;
                *lr_sh = 0.0025;
                *lr_background = 0.001;
                *downsample = 0.40;  // Keep at 40% to avoid GPU memory issues
                *max_gaussians = 25_000;  // More Gaussians for quality
                *log_interval = 100;
                *learn_background = true;
                *learn_opacity = true;
                *learn_position = true;
                *learn_scale = true;
                *learn_rotation = true;
                *learn_sh = true;
                *loss = sugar_rs::optim::loss::LossKind::L1Dssim;  // Better quality loss
                *train_fraction = 0.75;  // 75% train, 25% test
                *val_interval = 500;
                *max_test_views_for_metrics = 5;
                *max_images = 75;  // 4x images for better scene coverage
                *densify_interval = 500;
                *densify_max_gaussians = 50_000;
                *densify_grad_threshold = 0.0002;
                *prune_opacity_threshold = 0.005;
                *split_sigma_threshold = 0.1;
                *seed = Some(123);  // Fixed seed for reproducible, stable training (seed 0 has bad train/test splits)
            }
            "full" => {
                // Full overnight preset: Publication-quality results
                *multiview = true;
                *iters = 30_000;  // Standard 3DGS iteration count
                *lr = 0.002;
                *lr_position = 0.00016;
                *lr_rotation = 0.001;
                *lr_scale = 0.005;
                *lr_opacity = 0.05;
                *lr_sh = 0.0025;
                *lr_background = 0.0001;  // 10× lower than micro/onehour (301 images = 32× more updates)
                *downsample = 0.40;  // Keep at 40% to avoid GPU memory issues
                *max_gaussians = 50_000;
                *log_interval = 500;
                *learn_background = true;
                *learn_opacity = true;
                *learn_position = true;
                *learn_scale = true;
                *learn_rotation = true;
                *learn_sh = true;
                *loss = sugar_rs::optim::loss::LossKind::L1Dssim;
                *train_fraction = 0.8;  // 80% train, 20% test with all images
                *val_interval = 1000;
                *max_test_views_for_metrics = 10;
                *max_images = 0;  // Use all 301 images
                *densify_interval = 500;
                *densify_max_gaussians = 150_000;
                *densify_grad_threshold = 0.0002;
                *prune_opacity_threshold = 0.005;
                *split_sigma_threshold = 0.1;
                *seed = Some(123);  // Fixed seed for reproducible, stable training
            }
            "m10" | "m10-quick" => {
                *multiview = true;
                *iters = 2_000;
                *lr = 0.002; // Fallback (not used when per-param LRs are set)
                // Per-parameter learning rates based on reference Gaussian Splatting
                *lr_position = 0.00016;  // Very small to prevent position explosion
                *lr_rotation = 0.001;    // Moderate for rotation
                *lr_scale = 0.005;       // Higher for scale
                *lr_opacity = 0.05;      // Highest for opacity
                *lr_sh = 0.0025;         // Moderate for spherical harmonics
                *lr_background = 0.001;  // Conservative for background
                *downsample = 0.25;
                *max_gaussians = 20_000;
                *log_interval = 10;
                *learn_background = true;
                *learn_opacity = true;
                *learn_position = true;
                *learn_scale = true;
                *learn_rotation = true;
                *learn_sh = true;
                *loss = sugar_rs::optim::loss::LossKind::L1Dssim;
                *train_fraction = 0.8;
                *val_interval = 50;
                *max_test_views_for_metrics = 0;
                *max_images = 0;
                *densify_interval = 100;
                *densify_max_gaussians = 80_000;
                *densify_grad_threshold = 0.1;
                *prune_opacity_threshold = 0.01;
                *split_sigma_threshold = 0.05;
                *seed = Some(0);
            }
            "debug" => {
                // Debug preset: Ultra-fast iteration for bug hunting
                // Optimized for: single parameter testing, fast loss feedback, tiny models
                *multiview = false;         // Single view for isolation
                *iters = 50;                // Very few iterations
                *lr = 0.01;                 // Larger LR for faster convergence signal
                *lr_position = 0.001;       // 10x higher for visible movement
                *lr_rotation = 0.01;
                *lr_scale = 0.05;
                *lr_opacity = 0.1;
                *lr_sh = 0.025;
                *lr_background = 0.01;
                *downsample = 0.0625;       // 1/16 resolution (~32x32 for 512px)
                *max_gaussians = 10;        // Tiny model
                *log_interval = 1;          // Log every iteration
                *learn_background = false;  // Isolate Gaussian learning
                *learn_opacity = true;
                *learn_position = true;
                *learn_scale = true;
                *learn_rotation = false;    // Disable rotation for simpler gradients
                *learn_sh = true;
                *loss = sugar_rs::optim::loss::LossKind::L2;  // Simpler loss
                *train_fraction = 1.0;      // Use all images for training (no validation)
                *val_interval = 10;
                *max_test_views_for_metrics = 0;
                *max_images = 1;            // Single image
                *densify_interval = 0;      // Disable densification
                *densify_max_gaussians = 10;
                *densify_grad_threshold = 1e9;  // Effectively disable
                *prune_opacity_threshold = 0.0; // Disable pruning
                *split_sigma_threshold = 1e9;
                *seed = Some(42);           // Fixed seed for reproducibility
            }
            other => {
                return Err(format!(
                    "Unknown preset `{other}` (expected one of: m7, m8-smoke, m8, m9, micro, onehour, full, m10, m10-quick, debug)"
                ));
            }
        }
        Ok(())
    }

    while let Some(a) = args.next() {
        match a.as_str() {
            "--preset" => {
                let preset = args.next().unwrap();
                preset_name = Some(preset.clone());
                if let Err(msg) = apply_preset(
                    &preset,
                    &mut multiview,
                    &mut iters,
                    &mut lr,
                    &mut lr_position,
                    &mut lr_rotation,
                    &mut lr_scale,
                    &mut lr_opacity,
                    &mut lr_sh,
                    &mut lr_background,
                    &mut downsample,
                    &mut max_gaussians,
                    &mut image_index,
                    &mut log_interval,
                    &mut learn_background,
                    &mut learn_opacity,
                    &mut learn_position,
                    &mut learn_scale,
                    &mut learn_rotation,
                    &mut learn_sh,
                    &mut loss,
                    &mut train_fraction,
                    &mut val_interval,
                    &mut max_test_views_for_metrics,
                    &mut max_images,
                    &mut densify_interval,
                    &mut densify_max_gaussians,
                    &mut densify_grad_threshold,
                    &mut prune_opacity_threshold,
                    &mut split_sigma_threshold,
                    &mut seed,
                ) {
                    eprintln!("{msg}");
                    return;
                }
            }
            "--dataset-root" => dataset_root = args.next().map(std::path::PathBuf::from),
            "--scene" => scene = args.next().map(std::path::PathBuf::from),
            "--images" => images = args.next().map(std::path::PathBuf::from),
            "--iters" => iters = args.next().unwrap().parse().unwrap(),
            "--lr" => lr = args.next().unwrap().parse().unwrap(),
            "--lr-position" => lr_position = args.next().unwrap().parse().unwrap(),
            "--lr-rotation" => lr_rotation = args.next().unwrap().parse().unwrap(),
            "--lr-scale" => lr_scale = args.next().unwrap().parse().unwrap(),
            "--lr-opacity" => lr_opacity = args.next().unwrap().parse().unwrap(),
            "--lr-sh" => lr_sh = args.next().unwrap().parse().unwrap(),
            "--lr-background" => lr_background = args.next().unwrap().parse().unwrap(),
            "--downsample" => {
                downsample = args.next().unwrap().parse().unwrap();
                downsample_explicit = true;
            }
            "--max-gaussians" => max_gaussians = args.next().unwrap().parse().unwrap(),
            "--image-index" => image_index = args.next().unwrap().parse().unwrap(),
            "--log-interval" => log_interval = args.next().unwrap().parse().unwrap(),
            "--no-learn-bg" => learn_background = false,
            "--learn-bg" => learn_background = true,
            "--loss" => {
                let v = args.next().expect("--loss requires a value: l2 | l1dssim");
                loss = match v.as_str() {
                    "l2" => sugar_rs::optim::loss::LossKind::L2,
                    "l1dssim" | "l1-dssim" => sugar_rs::optim::loss::LossKind::L1Dssim,
                    other => panic!("unknown --loss '{other}': expected l2 | l1dssim"),
                };
            }
            "--learn-opacity" => learn_opacity = true,
            "--learn-position" => learn_position = true,
            "--learn-scale" => learn_scale = true,
            "--learn-rotation" => learn_rotation = true,
            "--learn-sh" => learn_sh = true,
            "--loss" => {
                let v = args.next().unwrap();
                loss = match v.as_str() {
                    "l2" => sugar_rs::optim::loss::LossKind::L2,
                    "l1-dssim" | "l1_dssim" | "l1dssim" => sugar_rs::optim::loss::LossKind::L1Dssim,
                    other => {
                        eprintln!("Unknown --loss {other} (expected: l2 | l1-dssim)");
                        return;
                    }
                };
            }
            "--multiview" => multiview = true,
            "--train-fraction" => train_fraction = args.next().unwrap().parse().unwrap(),
            "--val-interval" => val_interval = args.next().unwrap().parse().unwrap(),
            "--max-test-views" => max_test_views_for_metrics = args.next().unwrap().parse().unwrap(),
            "--max-images" => max_images = args.next().unwrap().parse().unwrap(),
            "--out-dir" => out_dir = Some(args.next().unwrap().into()),
            "--densify-interval" => densify_interval = args.next().unwrap().parse().unwrap(),
            "--densify-max-gaussians" => densify_max_gaussians = args.next().unwrap().parse().unwrap(),
            "--densify-grad-threshold" => densify_grad_threshold = args.next().unwrap().parse().unwrap(),
            "--prune-opacity-threshold" => prune_opacity_threshold = args.next().unwrap().parse().unwrap(),
            "--split-sigma-threshold" => split_sigma_threshold = args.next().unwrap().parse().unwrap(),
            "--seed" => seed = Some(args.next().unwrap().parse().unwrap()),
            "--max-log-aniso" => max_log_aniso = args.next().unwrap().parse().unwrap(),
            "--settle-prune-interval" => settle_prune_interval = args.next().unwrap().parse().unwrap(),
            "--opacity-reset-floor" => opacity_reset_floor = args.next().unwrap().parse().unwrap(),
            "--gpu" => use_gpu = true,
            "--cpu" | "--no-gpu" => use_gpu = false,
            "--disable-sh" => disable_sh = true,
            "--help" | "-h" => {
                eprintln!("Usage:");
                eprintln!("  sugar-train --preset m7|m8-smoke|m8|m9|m10 [--dataset-root <root> | --scene <sparse/0>] [--images <dir>] [overrides...]");
                eprintln!("  Note: presets apply immediately; later flags override preset values.");
                eprintln!("  Note: GPU rendering is enabled by default. Use --cpu to force CPU rendering.");
                eprintln!();
                eprintln!("  # M7 (single-view / overfit)");
                eprintln!("  sugar-train --scene <sparse/0> [--images <dir>] [--iters N] [--lr LR] [--downsample F] [--max-gaussians N] [--image-index I] [--log-interval N] [--loss l2|l1-dssim] [--no-learn-bg] [--learn-opacity] [--learn-position] [--learn-scale] [--learn-rotation] [--learn-sh] [--seed U64] [--out-dir DIR]");
                eprintln!();
                eprintln!("  # M8 (multi-view)");
                eprintln!("  sugar-train --multiview --scene <sparse/0> [--images <dir>] [--max-images N] [--iters N] [--lr LR] [--downsample F] [--max-gaussians N] [--train-fraction F] [--val-interval N] [--max-test-views N] [--log-interval N] [--loss l2|l1-dssim] [--no-learn-bg] [--learn-opacity] [--learn-position] [--learn-scale] [--learn-rotation] [--learn-sh] [--densify-interval N] [--densify-max-gaussians N] [--densify-grad-threshold F] [--prune-opacity-threshold F] [--split-sigma-threshold F] [--max-log-aniso F (default 3.0, 0=off)] [--settle-prune-interval N (default 500, 0=off)] [--opacity-reset-floor F (default 0.01)] [--seed U64] [--out-dir DIR]");
                eprintln!();
                eprintln!("  # Auto-detect paths");
                eprintln!("  sugar-train [--multiview] --dataset-root <root> [--iters N] ...   (auto-detects sparse/0 + images/)");
                return;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                return;
            }
        }
    }

    let (scene, images_dir) = if let Some(root) = dataset_root {
        let sparse = sugar_rs::optim::trainer::guess_sparse0_from_dataset_root(&root)
            .expect("Could not find sparse/0 under --dataset-root");
        let imgs = images
            .or_else(|| sugar_rs::optim::trainer::guess_images_dir_from_sparse(&sparse))
            .expect("Missing --images and couldn't guess images dir");
        (sparse, imgs)
    } else {
        let scene = scene.expect("Missing --scene <colmap sparse/0> (or use --dataset-root)");
        let images_dir = images
            .or_else(|| sugar_rs::optim::trainer::guess_images_dir_from_sparse(&scene))
            .expect("Missing --images and couldn't guess images dir");
        (scene, images_dir)
    };

    // Determine output directory: use --out-dir if specified, otherwise create timestamped directory
    let final_out_dir = if let Some(dir) = out_dir {
        std::fs::create_dir_all(&dir).ok();
        dir
    } else {
        let preset = preset_name.as_deref().unwrap_or("custom");
        create_run_directory(preset)
            .expect("Failed to create run directory")
    };

    // Save run metadata (seed will be updated after training)
    let all_args: Vec<String> = std::env::args().collect();
    save_run_metadata(&final_out_dir, &all_args, None)
        .unwrap_or_else(|e| eprintln!("Warning: Failed to save metadata: {}", e));

    // Auto-calculate downsample factor if not explicitly set
    if !downsample_explicit && use_gpu {
        use sugar_rs::io::load_colmap_scene;
        match load_colmap_scene(&scene) {
            Ok(colmap_scene) => {
                if !colmap_scene.images.is_empty() {
                    let first_image_name = &colmap_scene.images[0].name;
                    let first_image_path = {
                        let direct = images_dir.join(first_image_name);
                        if direct.exists() {
                            direct
                        } else {
                            let mut candidates = Vec::new();
                            if let Some(ext) = direct.extension().and_then(|e| e.to_str()) {
                                let ext_lower = ext.to_ascii_lowercase();
                                if ext_lower == "jpg" || ext_lower == "jpeg" {
                                    let mut png_path = direct.clone();
                                    png_path.set_extension("png");
                                    candidates.push(png_path);
                                } else if ext_lower == "png" {
                                    let mut jpg_path = direct.clone();
                                    jpg_path.set_extension("jpg");
                                    candidates.push(jpg_path);
                                    let mut jpeg_path = direct.clone();
                                    jpeg_path.set_extension("jpeg");
                                    candidates.push(jpeg_path);
                                }
                            } else {
                                let mut png_path = direct.clone();
                                png_path.set_extension("png");
                                candidates.push(png_path);
                                let mut jpg_path = direct.clone();
                                jpg_path.set_extension("jpg");
                                candidates.push(jpg_path);
                                let mut jpeg_path = direct.clone();
                                jpeg_path.set_extension("jpeg");
                                candidates.push(jpeg_path);
                            }
                            candidates.into_iter().find(|p| p.exists()).unwrap_or(direct)
                        }
                    };

                    let max_buffer_size = auto_downsample::get_gpu_max_buffer_size();
                    match auto_downsample::determine_auto_downsample(&first_image_path, max_buffer_size) {
                        Ok((auto_downsample_factor, width, height)) => {
                            // Only warn if we're actually downsampling
                            if auto_downsample_factor < 1.0 {
                                auto_downsample::print_auto_downsample_warning(
                                    width,
                                    height,
                                    max_buffer_size / (1024 * 1024),
                                    auto_downsample_factor,
                                );
                            }
                            downsample = auto_downsample_factor;
                        }
                        Err(e) => {
                            eprintln!("Warning: {}", e);
                            eprintln!("Using default downsample factor: {}", downsample);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to load COLMAP scene for auto-downsample: {}", e);
                eprintln!("Using default downsample factor: {}", downsample);
            }
        }
    }

    // Derive dataset root from scene path before scene is moved (scene is sparse/0, so go up two levels)
    let dataset_path = scene.parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    if multiview {
        let cfg = sugar_rs::optim::trainer::MultiViewTrainConfig {
            sparse_dir: scene,
            images_dir,
            max_gaussians,
            downsample_factor: downsample,
            iters,
            lr,
            lr_position,
            lr_rotation,
            lr_scale,
            lr_opacity,
            lr_sh,
            lr_background,
            learn_background,
            learn_opacity,
            learn_position,
            learn_scale,
            learn_rotation,
            learn_sh,
            loss,
            max_images,
            rng_seed: seed,
            train_fraction,
            val_interval,
                max_test_views_for_metrics,
                log_interval,
            densify_interval,
            densify_max_gaussians,
            densify_grad_threshold,
            prune_opacity_threshold,
            split_sigma_threshold,
            max_log_anisotropy: max_log_aniso,
            // Companion needle prune sits 0.4 above the per-step pull (legacy 1.6 → 2.0).
            needle_prune_log_anisotropy: if max_log_aniso > 0.0 {
                max_log_aniso + 0.4
            } else {
                0.0
            },
            opacity_reset_interval: 3000,
            opacity_reset_floor,
            settle_prune_interval,
            use_gpu,
            csv_output_path: Some(final_out_dir.join("metrics.csv")),
            out_dir: final_out_dir.clone(),
            disable_sh,
        };

        let out = sugar_rs::optim::trainer::train_multiview_color_only(&cfg)
            .expect("Multi-view training failed");

        // Update metadata with actual seed used
        save_run_metadata(&final_out_dir, &all_args, Some(out.seed_used))
            .unwrap_or_else(|e| eprintln!("Warning: Failed to update metadata: {}", e));

        eprintln!(
            "M8 metrics: initial_psnr={:.2}dB final_psnr={:.2}dB train_loss={:.6} gaussians={}->{} densify_events={}",
            out.initial_psnr,
            out.final_psnr,
            out.train_loss,
            out.initial_num_gaussians,
            out.final_num_gaussians,
            out.densify_events
        );

        let rendered_path = final_out_dir.join("m8_test_view_rendered.png");
        let target_path = final_out_dir.join("m8_test_view_target.png");
        out.test_view_sample.save(&rendered_path).ok();
        out.test_view_target.save(&target_path).ok();
        eprintln!("Saved `{}`", rendered_path.display());
        eprintln!("Saved `{}`", target_path.display());

        if std::env::var("SUGAR_DEBUG_TARGET").is_ok() {
            let target = &out.test_view_target;
            let expected_len = (target.width() * target.height() * 3) as usize;
            let raw = target.as_raw();
            if raw.len() != expected_len {
                eprintln!(
                    "[TARGET DEBUG] Raw length mismatch: got {}, expected {}",
                    raw.len(),
                    expected_len
                );
            }

            match image::open(&target_path).map(|img| img.to_rgb8()) {
                Ok(loaded) => {
                    if loaded.width() != target.width() || loaded.height() != target.height() {
                        eprintln!(
                            "[TARGET DEBUG] Reloaded dims mismatch: {}x{} vs {}x{}",
                            loaded.width(),
                            loaded.height(),
                            target.width(),
                            target.height()
                        );
                    } else {
                        let mut max_diff = 0u8;
                        let mut diff_pixels = 0u32;
                        for (a, b) in raw.iter().zip(loaded.as_raw().iter()) {
                            let d = a.abs_diff(*b);
                            if d > max_diff {
                                max_diff = d;
                            }
                            if d > 2 {
                                diff_pixels += 1;
                            }
                        }
                        eprintln!(
                            "[TARGET DEBUG] Reload diff: max_diff={} diff_pixels={}",
                            max_diff,
                            diff_pixels
                        );
                    }
                }
                Err(err) => {
                    eprintln!("[TARGET DEBUG] Failed to reload target: {}", err);
                }
            }
        }

        // Save trained model
        let model_path = final_out_dir.join("model.gs");
        let cloud = GaussianCloud::from_gaussians(out.gaussians);
        let (bounds_min, bounds_max) = compute_bounds(&cloud.gaussians);

        #[cfg(feature = "lz4")]
        let compression = Compression::Lz4;
        #[cfg(not(feature = "lz4"))]
        let compression = Compression::None;

        let metadata = ModelMetadata {
            num_gaussians: cloud.len() as u64,
            sh_degree: 3,
            bounds_min,
            bounds_max,
            training_iterations: iters as u64,
            training_psnr: out.final_psnr,
            compression,
            training_width: out.training_width,
            training_height: out.training_height,
            training_downsample_factor: out.downsample_factor,
            dataset_path,
        };
        save_model(&model_path, &cloud, &metadata).expect("Failed to save model");
        eprintln!("Saved model to `{}`", model_path.display());
    } else {
        let cfg = sugar_rs::optim::trainer::TrainConfig {
            sparse_dir: scene,
            images_dir,
            image_index,
            max_gaussians,
            downsample_factor: downsample,
            iters,
            lr,
            lr_position,
            lr_rotation,
            lr_scale,
            lr_opacity,
            lr_sh,
            lr_background,
            learn_background,
            learn_opacity,
            learn_position,
            learn_scale,
            learn_rotation,
            learn_sh,
            loss,
            log_interval,
            rng_seed: seed,
            use_gpu,
            csv_output_path: Some(final_out_dir.join("metrics.csv")),
            disable_sh,
        };

        let out =
            sugar_rs::optim::trainer::train_single_image_color_only(&cfg).expect("Training failed");
        eprintln!("Training image: {}", out.image_name);

        // Update metadata with actual seed used
        save_run_metadata(&final_out_dir, &all_args, Some(out.seed_used))
            .unwrap_or_else(|e| eprintln!("Warning: Failed to update metadata with seed: {}", e));

        out.target.save(final_out_dir.join("m7_target.png")).ok();
        out.overlay.save(final_out_dir.join("m7_overlay.png")).ok();
        out.coverage.save(final_out_dir.join("m7_coverage.png")).ok();
        out.t_final.save(final_out_dir.join("m7_t_final.png")).ok();
        out.contrib_count
            .save(final_out_dir.join("m7_contrib_count.png"))
            .ok();
        out.initial.save(final_out_dir.join("m7_initial.png")).ok();
        out.final_img.save(final_out_dir.join("m7_final.png")).ok();
        eprintln!("Saved M7 outputs under `{}`", final_out_dir.display());

        // Note: Single-image trainer doesn't currently return gaussians
        // TODO: Add model saving when trainer is updated to return trained gaussians
    }
}
