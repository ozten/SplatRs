//! sugar-train: Train Gaussian Splatting model from COLMAP scene
//!
//! Usage:
//!   sugar-train --scene path/to/colmap/sparse/0 --output model.ply

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
    let mut downsample: f32 = 0.25;
    let mut max_gaussians: usize = 20_000;
    let mut image_index: usize = 0;
    let mut log_interval: usize = 10;
    let mut learn_background: bool = true;
    let mut learn_opacity: bool = false;
    let mut learn_position: bool = false;
    let mut learn_scale: bool = false;
    let mut learn_rotation: bool = false;
    let mut loss: sugar_rs::optim::loss::LossKind = sugar_rs::optim::loss::LossKind::L2;
    let mut dataset_root: Option<std::path::PathBuf> = None;
    let mut multiview: bool = false;
    let mut train_fraction: f32 = 0.8;
    let mut val_interval: usize = 50;
    let mut max_test_views_for_metrics: usize = 0;
    let mut max_images: usize = 0;
    let mut out_dir: std::path::PathBuf = std::path::PathBuf::from("test_output");
    let mut densify_interval: usize = 0;
    let mut densify_max_gaussians: usize = 0;
    let mut densify_grad_threshold: f32 = 0.1;
    let mut prune_opacity_threshold: f32 = 0.01;
    let mut split_sigma_threshold: f32 = 0.05;
    let mut seed: Option<u64> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--dataset-root" => dataset_root = args.next().map(std::path::PathBuf::from),
            "--scene" => scene = args.next().map(std::path::PathBuf::from),
            "--images" => images = args.next().map(std::path::PathBuf::from),
            "--iters" => iters = args.next().unwrap().parse().unwrap(),
            "--lr" => lr = args.next().unwrap().parse().unwrap(),
            "--downsample" => downsample = args.next().unwrap().parse().unwrap(),
            "--max-gaussians" => max_gaussians = args.next().unwrap().parse().unwrap(),
            "--image-index" => image_index = args.next().unwrap().parse().unwrap(),
            "--log-interval" => log_interval = args.next().unwrap().parse().unwrap(),
            "--no-learn-bg" => learn_background = false,
            "--learn-opacity" => learn_opacity = true,
            "--learn-position" => learn_position = true,
            "--learn-scale" => learn_scale = true,
            "--learn-rotation" => learn_rotation = true,
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
            "--out-dir" => out_dir = args.next().unwrap().into(),
            "--densify-interval" => densify_interval = args.next().unwrap().parse().unwrap(),
            "--densify-max-gaussians" => densify_max_gaussians = args.next().unwrap().parse().unwrap(),
            "--densify-grad-threshold" => densify_grad_threshold = args.next().unwrap().parse().unwrap(),
            "--prune-opacity-threshold" => prune_opacity_threshold = args.next().unwrap().parse().unwrap(),
            "--split-sigma-threshold" => split_sigma_threshold = args.next().unwrap().parse().unwrap(),
            "--seed" => seed = Some(args.next().unwrap().parse().unwrap()),
            "--help" | "-h" => {
                eprintln!("Usage:");
                eprintln!("  # M7 (single-view / overfit)");
                eprintln!("  sugar-train --scene <sparse/0> [--images <dir>] [--iters N] [--lr LR] [--downsample F] [--max-gaussians N] [--image-index I] [--log-interval N] [--loss l2|l1-dssim] [--no-learn-bg] [--learn-opacity] [--learn-position] [--learn-scale] [--learn-rotation] [--seed U64] [--out-dir DIR]");
                eprintln!();
                eprintln!("  # M8 (multi-view)");
                eprintln!("  sugar-train --multiview --scene <sparse/0> [--images <dir>] [--max-images N] [--iters N] [--lr LR] [--downsample F] [--max-gaussians N] [--train-fraction F] [--val-interval N] [--max-test-views N] [--log-interval N] [--loss l2|l1-dssim] [--no-learn-bg] [--learn-opacity] [--learn-position] [--learn-scale] [--learn-rotation] [--densify-interval N] [--densify-max-gaussians N] [--densify-grad-threshold F] [--prune-opacity-threshold F] [--split-sigma-threshold F] [--seed U64] [--out-dir DIR]");
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

    std::fs::create_dir_all(&out_dir).ok();

    if multiview {
        let cfg = sugar_rs::optim::trainer::MultiViewTrainConfig {
            sparse_dir: scene,
            images_dir,
            max_gaussians,
            downsample_factor: downsample,
            iters,
            lr,
            learn_background,
            learn_opacity,
            learn_position,
            learn_scale,
            learn_rotation,
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
        };

        let out = sugar_rs::optim::trainer::train_multiview_color_only(&cfg)
            .expect("Multi-view training failed");

        eprintln!(
            "M8 metrics: initial_psnr={:.2}dB final_psnr={:.2}dB train_loss={:.6} gaussians={}->{} densify_events={}",
            out.initial_psnr,
            out.final_psnr,
            out.train_loss,
            out.initial_num_gaussians,
            out.final_num_gaussians,
            out.densify_events
        );

        let rendered_path = out_dir.join("m8_test_view_rendered.png");
        let target_path = out_dir.join("m8_test_view_target.png");
        out.test_view_sample.save(&rendered_path).ok();
        out.test_view_target.save(&target_path).ok();
        eprintln!("Saved `{}`", rendered_path.display());
        eprintln!("Saved `{}`", target_path.display());
    } else {
        let cfg = sugar_rs::optim::trainer::TrainConfig {
            sparse_dir: scene,
            images_dir,
            image_index,
            max_gaussians,
            downsample_factor: downsample,
            iters,
            lr,
            learn_background,
            learn_opacity,
            learn_position,
            learn_scale,
            learn_rotation,
            loss,
            log_interval,
            rng_seed: seed,
        };

        let out =
            sugar_rs::optim::trainer::train_single_image_color_only(&cfg).expect("Training failed");
        eprintln!("Training image: {}", out.image_name);

        out.target.save(out_dir.join("m7_target.png")).ok();
        out.overlay.save(out_dir.join("m7_overlay.png")).ok();
        out.coverage.save(out_dir.join("m7_coverage.png")).ok();
        out.t_final.save(out_dir.join("m7_t_final.png")).ok();
        out.contrib_count
            .save(out_dir.join("m7_contrib_count.png"))
            .ok();
        out.initial.save(out_dir.join("m7_initial.png")).ok();
        out.final_img.save(out_dir.join("m7_final.png")).ok();
        eprintln!("Saved M7 outputs under `{}`", out_dir.display());
    }
}
