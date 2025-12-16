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
    let mut learn_background: bool = true;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--scene" => scene = args.next().map(std::path::PathBuf::from),
            "--images" => images = args.next().map(std::path::PathBuf::from),
            "--iters" => iters = args.next().unwrap().parse().unwrap(),
            "--lr" => lr = args.next().unwrap().parse().unwrap(),
            "--downsample" => downsample = args.next().unwrap().parse().unwrap(),
            "--max-gaussians" => max_gaussians = args.next().unwrap().parse().unwrap(),
            "--image-index" => image_index = args.next().unwrap().parse().unwrap(),
            "--no-learn-bg" => learn_background = false,
            "--help" | "-h" => {
                eprintln!("Usage:");
                eprintln!("  sugar-train --scene <sparse/0> [--images <dir>] [--iters N] [--lr LR] [--downsample F] [--max-gaussians N] [--image-index I] [--no-learn-bg]");
                return;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                return;
            }
        }
    }

    let scene = scene.expect("Missing --scene <colmap sparse/0>");
    let images_dir = images
        .or_else(|| sugar_rs::optim::trainer::guess_images_dir_from_sparse(&scene))
        .expect("Missing --images and couldn't guess images dir");

    let cfg = sugar_rs::optim::trainer::TrainConfig {
        sparse_dir: scene,
        images_dir,
        image_index,
        max_gaussians,
        downsample_factor: downsample,
        iters,
        lr,
        learn_background,
    };

    let out =
        sugar_rs::optim::trainer::train_single_image_color_only(&cfg).expect("Training failed");
    eprintln!("Training image: {}", out.image_name);

    std::fs::create_dir_all("test_output").ok();
    out.target.save("test_output/m7_target.png").ok();
    out.overlay.save("test_output/m7_overlay.png").ok();
    out.coverage.save("test_output/m7_coverage.png").ok();
    out.t_final.save("test_output/m7_t_final.png").ok();
    out.contrib_count
        .save("test_output/m7_contrib_count.png")
        .ok();
    out.initial.save("test_output/m7_initial.png").ok();
    out.final_img.save("test_output/m7_final.png").ok();
    eprintln!("Saved `test_output/m7_target.png`, `test_output/m7_overlay.png`, `test_output/m7_coverage.png`, `test_output/m7_t_final.png`, `test_output/m7_contrib_count.png`, `test_output/m7_initial.png`, `test_output/m7_final.png`");
}
