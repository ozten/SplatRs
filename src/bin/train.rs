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
            other => {
                return Err(format!(
                    "Unknown preset `{other}` (expected one of: m7, m8-smoke, m8, m9, micro, onehour, full, m10, m10-quick)"
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
            "--downsample" => {
                downsample = args.next().unwrap().parse().unwrap();
                downsample_explicit = true;
            }
            "--max-gaussians" => max_gaussians = args.next().unwrap().parse().unwrap(),
            "--image-index" => image_index = args.next().unwrap().parse().unwrap(),
            "--log-interval" => log_interval = args.next().unwrap().parse().unwrap(),
            "--no-learn-bg" => learn_background = false,
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
            "--gpu" => use_gpu = true,
            "--cpu" | "--no-gpu" => use_gpu = false,
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
                eprintln!("  sugar-train --multiview --scene <sparse/0> [--images <dir>] [--max-images N] [--iters N] [--lr LR] [--downsample F] [--max-gaussians N] [--train-fraction F] [--val-interval N] [--max-test-views N] [--log-interval N] [--loss l2|l1-dssim] [--no-learn-bg] [--learn-opacity] [--learn-position] [--learn-scale] [--learn-rotation] [--learn-sh] [--densify-interval N] [--densify-max-gaussians N] [--densify-grad-threshold F] [--prune-opacity-threshold F] [--split-sigma-threshold F] [--seed U64] [--out-dir DIR]");
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
                    let first_image_path = images_dir.join(first_image_name);

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
            use_gpu,
            csv_output_path: Some(final_out_dir.join("metrics.csv")),
            out_dir: final_out_dir.clone(),
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
