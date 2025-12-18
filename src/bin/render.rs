//! sugar-render: Render images from trained Gaussian Splatting model
//!
//! Usage:
//!   sugar-render --model model.gs --camera-id 0 --dataset-root path/to/colmap --out render.png
//!   sugar-render --model model.gs --camera-json camera.json --out render.png

use sugar_rs::core::{quaternion_to_matrix, Camera};
use sugar_rs::io::{load_colmap_scene, load_model};
use sugar_rs::render::full_diff::{linear_vec_to_rgb8_img, render_full_linear};
use nalgebra::Vector3;
use std::path::PathBuf;

fn main() {
    println!("sugar-render v{}", sugar_rs::VERSION);

    // Parse command-line arguments
    let mut args = std::env::args().skip(1);
    let mut model_path: Option<PathBuf> = None;
    let mut camera_id: Option<usize> = None;
    let mut camera_json: Option<PathBuf> = None;
    let mut dataset_root: Option<PathBuf> = None;
    let mut out_path: PathBuf = PathBuf::from("render.png");
    let mut background: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);
    let mut width_override: Option<u32> = None;
    let mut height_override: Option<u32> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model_path = Some(PathBuf::from(args.next().expect("Missing --model argument")));
            }
            "--camera-id" => {
                let id_str = args.next().expect("Missing --camera-id argument");
                camera_id = Some(id_str.parse().expect("Invalid camera ID"));
            }
            "--camera-json" => {
                camera_json = Some(PathBuf::from(
                    args.next().expect("Missing --camera-json argument"),
                ));
            }
            "--dataset-root" => {
                dataset_root = Some(PathBuf::from(
                    args.next().expect("Missing --dataset-root argument"),
                ));
            }
            "--out" => {
                out_path = PathBuf::from(args.next().expect("Missing --out argument"));
            }
            "--background" => {
                let bg_str = args.next().expect("Missing --background argument");
                let parts: Vec<f32> = bg_str
                    .split(',')
                    .map(|s| s.parse().expect("Invalid background color"))
                    .collect();
                if parts.len() != 3 {
                    eprintln!("Error: --background must be three comma-separated floats (e.g., '0.5,0.5,0.5')");
                    std::process::exit(1);
                }
                background = Vector3::new(parts[0], parts[1], parts[2]);
            }
            "--width" => {
                width_override = Some(
                    args.next()
                        .expect("Missing --width argument")
                        .parse()
                        .expect("Invalid width"),
                );
            }
            "--height" => {
                height_override = Some(
                    args.next()
                        .expect("Missing --height argument")
                        .parse()
                        .expect("Invalid height"),
                );
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_help();
                std::process::exit(1);
            }
        }
    }

    // Validate arguments
    let model_path = model_path.expect("Missing --model argument");

    if camera_id.is_none() && camera_json.is_none() {
        eprintln!("Error: Must specify either --camera-id or --camera-json");
        print_help();
        std::process::exit(1);
    }

    if camera_id.is_some() && dataset_root.is_none() {
        eprintln!("Error: --camera-id requires --dataset-root");
        std::process::exit(1);
    }

    // Load model
    println!("Loading model from {:?}...", model_path);
    let (cloud, metadata) = load_model(&model_path).expect("Failed to load model");
    println!(
        "Loaded {} Gaussians (trained for {} iterations, PSNR: {:.2} dB)",
        cloud.len(),
        metadata.training_iterations,
        metadata.training_psnr
    );
    println!(
        "Scene bounds: min={:?}, max={:?}",
        metadata.bounds_min, metadata.bounds_max
    );
    println!(
        "Training resolution: {}×{} (downsample: {})",
        metadata.training_width, metadata.training_height, metadata.training_downsample_factor
    );

    // Check if model has training resolution
    let use_training_resolution = metadata.training_width > 0
        && metadata.training_height > 0
        && width_override.is_none()
        && height_override.is_none();

    if use_training_resolution {
        println!(
            "Using training resolution: {}×{} (downsample: {:.2})",
            metadata.training_width, metadata.training_height, metadata.training_downsample_factor
        );
    } else if width_override.is_some() || height_override.is_some() {
        println!(
            "Using override resolution: {}×{}",
            width_override.unwrap_or(0),
            height_override.unwrap_or(0)
        );
    }

    // Get camera
    let camera = if let Some(cam_id) = camera_id {
        // Load from dataset
        let dataset_root = dataset_root.unwrap();
        let sparse_dir = dataset_root.join("sparse/0");
        println!("Loading camera {} from {:?}...", cam_id, sparse_dir);

        let scene = load_colmap_scene(&sparse_dir).expect("Failed to load COLMAP scene");

        if cam_id >= scene.images.len() {
            eprintln!(
                "Error: camera ID {} out of range (max: {})",
                cam_id,
                scene.images.len() - 1
            );
            std::process::exit(1);
        }

        // Reconstruct Camera from ImageInfo + camera intrinsics
        let image_info = &scene.images[cam_id];
        let camera_intrinsics = scene
            .cameras
            .get(&image_info.camera_id)
            .expect("Camera ID not found in cameras map");

        // Combine intrinsics with extrinsics
        let mut camera = Camera::new(
            camera_intrinsics.fx,
            camera_intrinsics.fy,
            camera_intrinsics.cx,
            camera_intrinsics.cy,
            camera_intrinsics.width,
            camera_intrinsics.height,
            quaternion_to_matrix(&image_info.rotation),
            image_info.translation,
        );

        // Apply training resolution if available and no overrides
        if use_training_resolution {
            let downsample = metadata.training_downsample_factor;
            camera.width = metadata.training_width;
            camera.height = metadata.training_height;
            camera.fx *= downsample;
            camera.fy *= downsample;
            camera.cx *= downsample;
            camera.cy *= downsample;
        } else if let (Some(w), Some(h)) = (width_override, height_override) {
            // Explicit override - scale intrinsics proportionally
            let scale_w = (w as f32) / (camera.width as f32);
            let scale_h = (h as f32) / (camera.height as f32);
            let scale = scale_w.min(scale_h); // Maintain aspect ratio

            camera.width = w;
            camera.height = h;
            camera.fx *= scale;
            camera.fy *= scale;
            camera.cx *= scale;
            camera.cy *= scale;
        }

        camera
    } else if let Some(json_path) = camera_json {
        // Load from JSON
        println!("Loading camera from {:?}...", json_path);
        let json_str = std::fs::read_to_string(&json_path).expect("Failed to read camera JSON");
        let mut camera = serde_json::from_str::<Camera>(&json_str).expect("Failed to parse camera JSON");

        // Apply resolution overrides if specified
        if let (Some(w), Some(h)) = (width_override, height_override) {
            let scale_w = (w as f32) / (camera.width as f32);
            let scale_h = (h as f32) / (camera.height as f32);
            let scale = scale_w.min(scale_h); // Maintain aspect ratio

            camera.width = w;
            camera.height = h;
            camera.fx *= scale;
            camera.fy *= scale;
            camera.cx *= scale;
            camera.cy *= scale;
        }

        camera
    } else {
        unreachable!()
    };

    println!(
        "Rendering {}×{} image...",
        camera.width, camera.height
    );

    // Render
    let pixels = render_full_linear(&cloud.gaussians, &camera, &background);

    // Convert to RGB8 image
    let img = linear_vec_to_rgb8_img(&pixels, camera.width, camera.height);

    // Save
    println!("Saving to {:?}...", out_path);
    img.save(&out_path).expect("Failed to save image");

    println!("Done!");
}

fn print_help() {
    println!(
        r#"sugar-render: Render images from trained Gaussian Splatting model

USAGE:
    sugar-render --model MODEL.gs [OPTIONS]

REQUIRED:
    --model PATH             Path to .gs model file

CAMERA (choose one):
    --camera-id ID           Camera ID from COLMAP dataset (requires --dataset-root)
    --camera-json PATH       Camera parameters as JSON file

OPTIONS:
    --dataset-root PATH      Path to COLMAP dataset root (needed for --camera-id)
    --out PATH               Output image path [default: render.png]
    --background R,G,B       Background color as comma-separated floats [default: 0,0,0]
    --width WIDTH            Override render width (default: use training resolution)
    --height HEIGHT          Override render height (default: use training resolution)
    --help, -h               Print this help message

RESOLUTION HANDLING:
    By default, renders at the training resolution stored in model metadata.
    Use --width and --height together to override for higher or lower resolution renders.
    Old models without metadata will use camera resolution from COLMAP.

EXAMPLES:
    # Render using camera 5 from a COLMAP dataset
    sugar-render --model trained.gs --camera-id 5 --dataset-root datasets/tandt_db/tandt/train --out render.png

    # Render using a custom camera (JSON file)
    sugar-render --model trained.gs --camera-json camera.json --out render.png

    # Render with white background
    sugar-render --model trained.gs --camera-id 0 --dataset-root data/ --background 1.0,1.0,1.0

CAMERA JSON FORMAT:
    {{
        "width": 640,
        "height": 480,
        "fx": 525.0,
        "fy": 525.0,
        "cx": 320.0,
        "cy": 240.0,
        "position": [0.0, 0.0, 5.0],
        "rotation": [[1,0,0], [0,1,0], [0,0,1]]
    }}
"#
    );
}
