//! Training orchestration (M7 + M8).
//!
//! M7: Single-image overfit trainer for validation
//! M8: Multi-view training with train/test split
//!
//! Both optimizers currently train only the SH DC coefficient (color) for each Gaussian.
//!
//! Why color-only for now:
//! - Validates differentiable rendering end-to-end
//! - Keeps the state space small and stable for early debugging
//!
//! Next:
//! - Add opacity + 2D eval + projection gradients for full parameter training
//! - Add Gaussian densification/pruning

use crate::core::{init_from_colmap_points_visible_stratified, Camera, Gaussian};
use crate::io::load_colmap_scene;
use crate::optim::adam::{AdamF32, AdamVec3};
use crate::optim::loss::l2_image_loss_and_grad_weighted;
use crate::render::full_diff::{
    coverage_mask_bool, debug_contrib_count, debug_coverage_mask, debug_final_transmittance,
    debug_overlay_means, downsample_rgb_nearest, linear_vec_to_rgb8_img, render_full_color_grads,
    render_full_linear, rgb8_to_linear_vec,
};
use image::RgbImage;
use nalgebra::{Matrix3, Vector3};
use rand::seq::SliceRandom;
use std::path::{Path, PathBuf};

pub struct TrainConfig {
    pub sparse_dir: PathBuf,
    pub images_dir: PathBuf,
    pub image_index: usize,
    pub max_gaussians: usize,
    pub downsample_factor: f32,
    pub iters: usize,
    pub lr: f32,
    pub learn_background: bool,
    pub learn_opacity: bool,
}

pub struct TrainOutputs {
    pub target: RgbImage,
    pub overlay: RgbImage,
    pub coverage: RgbImage,
    pub t_final: RgbImage,
    pub contrib_count: RgbImage,
    pub initial: RgbImage,
    pub final_img: RgbImage,
    pub background: Vector3<f32>,
    pub image_name: String,
}

fn camera_with_pose(base: &Camera, rotation: Matrix3<f32>, translation: Vector3<f32>) -> Camera {
    Camera::new(
        base.fx,
        base.fy,
        base.cx,
        base.cy,
        base.width,
        base.height,
        rotation,
        translation,
    )
}

fn downsample_camera(camera: &Camera, factor: f32) -> Camera {
    let width = ((camera.width as f32) * factor).round().max(1.0) as u32;
    let height = ((camera.height as f32) * factor).round().max(1.0) as u32;

    Camera::new(
        camera.fx * factor,
        camera.fy * factor,
        camera.cx * factor,
        camera.cy * factor,
        width,
        height,
        camera.rotation,
        camera.translation,
    )
}

fn load_target_image(images_dir: &Path, name: &str) -> anyhow::Result<RgbImage> {
    let path = images_dir.join(name);
    let img = image::open(&path)?.to_rgb8();
    Ok(img)
}

pub fn train_single_image_color_only(cfg: &TrainConfig) -> anyhow::Result<TrainOutputs> {
    let scene = load_colmap_scene(&cfg.sparse_dir)?;
    if scene.cameras.is_empty() || scene.images.is_empty() {
        return Err(anyhow::anyhow!("Scene has no cameras/images"));
    }
    let image_index = cfg.image_index.min(scene.images.len() - 1);
    let image_info = &scene.images[image_index];

    // Use the correct camera for this image (not just cameras[0])
    let base_camera = scene
        .cameras
        .get(&image_info.camera_id)
        .ok_or_else(|| anyhow::anyhow!("Camera {} not found", image_info.camera_id))?;
    let rotation = image_info.rotation.to_rotation_matrix().into_inner();
    let camera_full = camera_with_pose(base_camera, rotation, image_info.translation);
    let camera = downsample_camera(&camera_full, cfg.downsample_factor);

    // Initialize a Gaussian subset that is (roughly) evenly distributed in screen space.
    // This avoids spending most Gaussians on only one part of the image (e.g. the caliper)
    // which makes single-image overfit debugging misleading.
    let cloud =
        init_from_colmap_points_visible_stratified(&scene.points, &camera, cfg.max_gaussians, 8);
    let mut gaussians: Vec<Gaussian> = cloud.gaussians;

    // Heuristic: set per-point isotropic 3D sigma so that the projected footprint is ~constant in pixel-space.
    // This reduces the "salt-and-pepper" look from tiny splats when using sparse COLMAP points.
    let desired_sigma_px = 1.5f32;
    let f_mean = 0.5 * (camera.fx + camera.fy).max(1.0);
    for g in &mut gaussians {
        let z = camera.world_to_camera(&g.position).z;
        if z <= 0.0 {
            continue;
        }
        let sigma_world = (desired_sigma_px * z / f_mean).clamp(1e-4, 1.0);
        let log_sigma = sigma_world.ln();
        g.scale = Vector3::new(log_sigma, log_sigma, log_sigma);
    }

    // Target image.
    let target_full = load_target_image(&cfg.images_dir, &image_info.name)?;
    let target_ds = downsample_rgb_nearest(&target_full, camera.width, camera.height);
    let target_linear = rgb8_to_linear_vec(&target_ds);

    // Debug outputs at training resolution, using the same gaussian subset.
    let overlay = debug_overlay_means(&target_ds, &gaussians, &camera, 1);
    let coverage_bool = coverage_mask_bool(&gaussians, &camera);
    let coverage = debug_coverage_mask(&gaussians, &camera);
    let t_final = debug_final_transmittance(&gaussians, &camera);
    let contrib_count = debug_contrib_count(&gaussians, &camera, 32);

    // Quick sanity: print coverage stats for top vs bottom halves.
    {
        let w = camera.width as usize;
        let h = camera.height as usize;
        let total = w * h;
        let covered = coverage_bool.iter().filter(|&&c| c).count();
        let top = w * (h / 2);
        let covered_top = coverage_bool[..top].iter().filter(|&&c| c).count();
        let covered_bot = coverage_bool[top..].iter().filter(|&&c| c).count();
        eprintln!(
            "gaussians={}  coverage={:.1}%  top={:.1}%  bottom={:.1}%",
            gaussians.len(),
            100.0 * (covered as f32) / (total as f32).max(1.0),
            100.0 * (covered_top as f32) / (top as f32).max(1.0),
            100.0 * (covered_bot as f32) / ((total - top) as f32).max(1.0),
        );
    }

    // Loss weighting: emphasize covered pixels so Gaussian colors get a strong learning signal.
    // Otherwise the loss is dominated by background pixels and updates barely affect Gaussians.
    let weights: Vec<f32> = coverage_bool
        .iter()
        .map(|&c| if c { 1.0 } else { 0.1 })
        .collect();

    // Background color parameter (linear RGB).
    // Initialize to mean target color to reduce initial error for uncovered pixels.
    let mut bg = {
        let mut acc = Vector3::<f32>::zeros();
        for p in &target_linear {
            acc += *p;
        }
        acc / (target_linear.len() as f32).max(1.0)
    };
    let mut bg_opt = AdamVec3::new(cfg.lr, 0.9, 0.999, 1e-8);

    // Optimizer state for SH DC coeffs (RGB).
    let mut adam = AdamVec3::new(cfg.lr, 0.9, 0.999, 1e-8);
    let mut opacity_opt = AdamF32::new(cfg.lr, 0.9, 0.999, 1e-8);

    // Pull initial params.
    let sh0_basis = crate::core::sh_basis(&Vector3::new(0.0, 0.0, 1.0))[0]; // = C0
    let mut params: Vec<Vector3<f32>> = gaussians
        .iter()
        .map(|g| Vector3::new(g.sh_coeffs[0][0], g.sh_coeffs[0][1], g.sh_coeffs[0][2]))
        .collect();
    let mut opacity_logits: Vec<f32> = gaussians.iter().map(|g| g.opacity).collect();

    // Initial render for output.
    let initial_render_u8 = linear_vec_to_rgb8_img(
        &render_full_linear(&gaussians, &camera, &bg),
        camera.width,
        camera.height,
    );

    for iter in 0..cfg.iters {
        // Write params back into gaussians.
        for (i, g) in gaussians.iter_mut().enumerate() {
            g.sh_coeffs[0][0] = params[i].x;
            g.sh_coeffs[0][1] = params[i].y;
            g.sh_coeffs[0][2] = params[i].z;
            if cfg.learn_opacity {
                g.opacity = opacity_logits[i].clamp(-10.0, 10.0);
            }
        }

        // Forward (linear) and loss.
        let rendered_linear = render_full_linear(&gaussians, &camera, &bg);
        let (loss, d_image) =
            l2_image_loss_and_grad_weighted(&rendered_linear, &target_linear, &weights);

        // Backward: get dL/d(color_i) and dL/d(opacity_logit_i) per Gaussian.
        let (_img_u8, d_color, d_opacity_logits, d_bg) =
            render_full_color_grads(&gaussians, &camera, &d_image, &bg);

        // Convert dL/d(color) -> dL/d(SH0 coeff) using basis[0] (assuming clamp inactive).
        let grads: Vec<Vector3<f32>> = d_color.iter().map(|dc| dc * sh0_basis).collect();

        adam.step(&mut params, &grads);
        if cfg.learn_opacity {
            opacity_opt.step(&mut opacity_logits, &d_opacity_logits);
        }
        if cfg.learn_background {
            // AdamVec3 expects slices; update a single bg vector.
            let mut bg_param = vec![bg];
            let bg_grad = vec![d_bg];
            bg_opt.step(&mut bg_param, &bg_grad);
            bg = bg_param[0];
        }

        if iter % 10 == 0 || iter + 1 == cfg.iters {
            eprintln!(
                "iter {iter:4}  loss={loss:.6}  bg=({:.3},{:.3},{:.3})",
                bg.x, bg.y, bg.z
            );
        }
    }

    // Final render.
    let final_render_u8 = {
        for (i, g) in gaussians.iter_mut().enumerate() {
            g.sh_coeffs[0][0] = params[i].x;
            g.sh_coeffs[0][1] = params[i].y;
            g.sh_coeffs[0][2] = params[i].z;
            if cfg.learn_opacity {
                g.opacity = opacity_logits[i].clamp(-10.0, 10.0);
            }
        }
        linear_vec_to_rgb8_img(
            &render_full_linear(&gaussians, &camera, &bg),
            camera.width,
            camera.height,
        )
    };

    Ok(TrainOutputs {
        target: target_ds,
        overlay,
        coverage,
        t_final,
        contrib_count,
        initial: initial_render_u8,
        final_img: final_render_u8,
        background: bg,
        image_name: image_info.name.clone(),
    })
}

/// Try to guess the images directory from a COLMAP sparse path.
///
/// For your calipers dataset, this typically is `.../digital_calipers2_project/input`.
pub fn guess_images_dir_from_sparse(sparse_dir: &Path) -> Option<PathBuf> {
    // Common COLMAP layout:
    //   <root>/sparse/0
    //   <root>/images
    // Or:
    //   <root>/sparse/0
    //   <root>/input
    if let Some(root) = sparse_dir.parent().and_then(|p| p.parent()) {
        let candidate = root.join("input");
        if candidate.is_dir() {
            return Some(candidate);
        }
        let images = root.join("images");
        if images.is_dir() {
            return Some(images);
        }
    }

    // Try sibling "input" at the project root.
    // sparse_dir: <root>/colmap_workspace/sparse/0
    let root = sparse_dir.parent()?.parent()?.parent()?;
    let candidate = root.join("input");
    if candidate.is_dir() {
        return Some(candidate);
    }

    // Common COLMAP layout: <root>/images
    let images = root.join("images");
    if images.is_dir() {
        return Some(images);
    }
    None
}

/// Given a dataset root, try to find a COLMAP `sparse/0` directory.
///
/// This supports layouts like:
/// - `<root>/sparse/0`
/// - `<root>/colmap_workspace/sparse/0`
pub fn guess_sparse0_from_dataset_root(root: &Path) -> Option<PathBuf> {
    let direct = root.join("sparse").join("0");
    if direct.join("cameras.bin").is_file()
        && direct.join("images.bin").is_file()
        && direct.join("points3D.bin").is_file()
    {
        return Some(direct);
    }

    let ws = root.join("colmap_workspace").join("sparse").join("0");
    if ws.join("cameras.bin").is_file()
        && ws.join("images.bin").is_file()
        && ws.join("points3D.bin").is_file()
    {
        return Some(ws);
    }

    None
}

// ============================================================================
// M8: Multi-View Training
// ============================================================================

pub struct MultiViewTrainConfig {
    pub sparse_dir: PathBuf,
    pub images_dir: PathBuf,
    pub max_gaussians: usize,
    pub downsample_factor: f32,
    pub iters: usize,
    pub lr: f32,
    pub learn_background: bool,
    pub learn_opacity: bool,
    pub train_fraction: f32, // Fraction of images for training (rest for testing)
    pub val_interval: usize,  // Validate every N iterations
    /// Limit how many held-out views are used for PSNR reporting.
    /// Use `0` to evaluate all test views (can be slow on large datasets).
    pub max_test_views_for_metrics: usize,
}

pub struct MultiViewTrainOutputs {
    pub initial_psnr: f32,
    pub final_psnr: f32,
    pub train_loss: f32,
    pub num_train_views: usize,
    pub num_test_views: usize,
    pub test_view_sample: RgbImage, // One test view rendering for visual check
    pub test_view_target: RgbImage,
}

/// Compute PSNR between two linear RGB images.
fn compute_psnr(rendered: &[Vector3<f32>], target: &[Vector3<f32>]) -> f32 {
    if rendered.len() != target.len() || rendered.is_empty() {
        return 0.0;
    }

    let mse: f32 = rendered
        .iter()
        .zip(target.iter())
        .map(|(r, t)| {
            let diff = r - t;
            diff.norm_squared()
        })
        .sum::<f32>()
        / (rendered.len() as f32 * 3.0); // Divide by 3 for RGB channels

    if mse < 1e-10 {
        return 100.0; // Cap at very high PSNR for near-perfect match
    }

    // PSNR = 10 * log10(MAX^2 / MSE)
    // For linear RGB in [0, 1], MAX = 1
    10.0 * (1.0 / mse).log10()
}

/// M8: Train on multiple views with train/test split.
///
/// This extends M7 by:
/// - Splitting images into train/test sets
/// - Randomly sampling training views each iteration
/// - Validating on held-out test views
/// - Reporting PSNR metrics
pub fn train_multiview_color_only(
    cfg: &MultiViewTrainConfig,
) -> anyhow::Result<MultiViewTrainOutputs> {
    let scene = load_colmap_scene(&cfg.sparse_dir)?;
    if scene.cameras.is_empty() || scene.images.is_empty() {
        return Err(anyhow::anyhow!("Scene has no cameras/images"));
    }
    if cfg.iters == 0 {
        return Err(anyhow::anyhow!("iters must be > 0"));
    }
    if cfg.val_interval == 0 {
        return Err(anyhow::anyhow!("val_interval must be > 0"));
    }

    // Split images into train/test sets
    let mut image_indices: Vec<usize> = (0..scene.images.len()).collect();
    let mut rng = rand::thread_rng();
    image_indices.shuffle(&mut rng);

    let num_train = ((scene.images.len() as f32) * cfg.train_fraction).max(1.0) as usize;
    let train_indices = &image_indices[..num_train];
    let test_indices = &image_indices[num_train..];

    eprintln!(
        "Multi-view training: {} train views, {} test views",
        train_indices.len(),
        test_indices.len()
    );

    if test_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "No test views available. Need at least 2 images for train/test split."
        ));
    }

    let test_indices_for_metrics: &[usize] = if cfg.max_test_views_for_metrics == 0 {
        test_indices
    } else {
        &test_indices[..cfg.max_test_views_for_metrics.min(test_indices.len())]
    };

    // Use first training view to initialize camera and Gaussians
    let first_train_idx = train_indices[0];
    let first_image_info = &scene.images[first_train_idx];
    let base_camera = scene
        .cameras
        .get(&first_image_info.camera_id)
        .ok_or_else(|| anyhow::anyhow!("Camera {} not found", first_image_info.camera_id))?;

    let rotation = first_image_info.rotation.to_rotation_matrix().into_inner();
    let camera_full = camera_with_pose(base_camera, rotation, first_image_info.translation);
    let camera = downsample_camera(&camera_full, cfg.downsample_factor);

    // Initialize Gaussians from visible points (using first view for now)
    let cloud =
        init_from_colmap_points_visible_stratified(&scene.points, &camera, cfg.max_gaussians, 8);
    let mut gaussians: Vec<Gaussian> = cloud.gaussians;

    // Set per-point isotropic 3D sigma for consistent pixel-space footprint
    let desired_sigma_px = 1.5f32;
    let f_mean = 0.5 * (camera.fx + camera.fy).max(1.0);
    for g in &mut gaussians {
        let z = camera.world_to_camera(&g.position).z;
        if z <= 0.0 {
            continue;
        }
        let sigma_world = (desired_sigma_px * z / f_mean).clamp(1e-4, 1.0);
        let log_sigma = sigma_world.ln();
        g.scale = Vector3::new(log_sigma, log_sigma, log_sigma);
    }

    eprintln!("Initialized {} Gaussians", gaussians.len());

    // Initialize background color (using first view's mean)
    let first_target = load_target_image(&cfg.images_dir, &first_image_info.name)?;
    let first_target_ds = downsample_rgb_nearest(&first_target, camera.width, camera.height);
    let first_target_linear = rgb8_to_linear_vec(&first_target_ds);

    let mut bg = {
        let mut acc = Vector3::<f32>::zeros();
        for p in &first_target_linear {
            acc += *p;
        }
        acc / (first_target_linear.len() as f32).max(1.0)
    };
    let mut bg_opt = AdamVec3::new(cfg.lr, 0.9, 0.999, 1e-8);

    // Optimizer state for SH DC coeffs (RGB)
    let mut adam = AdamVec3::new(cfg.lr, 0.9, 0.999, 1e-8);
    let mut opacity_opt = AdamF32::new(cfg.lr, 0.9, 0.999, 1e-8);

    // Pull initial params
    let sh0_basis = crate::core::sh_basis(&Vector3::new(0.0, 0.0, 1.0))[0]; // = C0
    let mut params: Vec<Vector3<f32>> = gaussians
        .iter()
        .map(|g| Vector3::new(g.sh_coeffs[0][0], g.sh_coeffs[0][1], g.sh_coeffs[0][2]))
        .collect();
    let mut opacity_logits: Vec<f32> = gaussians.iter().map(|g| g.opacity).collect();

    // Compute initial PSNR on test views
    let initial_psnr = {
        let mut psnr_sum = 0.0f32;
        for &test_idx in test_indices_for_metrics {
            let test_image_info = &scene.images[test_idx];
            let test_base_camera = scene
                .cameras
                .get(&test_image_info.camera_id)
                .ok_or_else(|| anyhow::anyhow!("Camera {} not found", test_image_info.camera_id))?;
            let test_rotation = test_image_info.rotation.to_rotation_matrix().into_inner();
            let test_camera_full =
                camera_with_pose(test_base_camera, test_rotation, test_image_info.translation);
            let test_camera = downsample_camera(&test_camera_full, cfg.downsample_factor);

            let test_target = load_target_image(&cfg.images_dir, &test_image_info.name)?;
            let test_target_ds =
                downsample_rgb_nearest(&test_target, test_camera.width, test_camera.height);
            let test_target_linear = rgb8_to_linear_vec(&test_target_ds);

            let rendered = render_full_linear(&gaussians, &test_camera, &bg);
            let psnr = compute_psnr(&rendered, &test_target_linear);
            psnr_sum += psnr;
        }
        psnr_sum / (test_indices_for_metrics.len() as f32)
    };

    eprintln!("Initial test PSNR: {:.2} dB", initial_psnr);

    // Training loop: sample random views
    let mut train_loss = 0.0f32;
    for iter in 0..cfg.iters {
        // Sample a random training view
        let train_idx = *train_indices
            .choose(&mut rng)
            .expect("train_indices is non-empty");
        let train_image_info = &scene.images[train_idx];
        let train_base_camera = scene
            .cameras
            .get(&train_image_info.camera_id)
            .ok_or_else(|| anyhow::anyhow!("Camera {} not found", train_image_info.camera_id))?;
        let train_rotation = train_image_info.rotation.to_rotation_matrix().into_inner();
        let train_camera_full =
            camera_with_pose(train_base_camera, train_rotation, train_image_info.translation);
        let train_camera = downsample_camera(&train_camera_full, cfg.downsample_factor);

        // Load target image
        let train_target = load_target_image(&cfg.images_dir, &train_image_info.name)?;
        let train_target_ds =
            downsample_rgb_nearest(&train_target, train_camera.width, train_camera.height);
        let train_target_linear = rgb8_to_linear_vec(&train_target_ds);

        // Coverage weighting
        let coverage_bool = coverage_mask_bool(&gaussians, &train_camera);
        let weights: Vec<f32> = coverage_bool
            .iter()
            .map(|&c| if c { 1.0 } else { 0.1 })
            .collect();

        // Write params back into gaussians
        for (i, g) in gaussians.iter_mut().enumerate() {
            g.sh_coeffs[0][0] = params[i].x;
            g.sh_coeffs[0][1] = params[i].y;
            g.sh_coeffs[0][2] = params[i].z;
            if cfg.learn_opacity {
                g.opacity = opacity_logits[i].clamp(-10.0, 10.0);
            }
        }

        // Forward and loss
        let rendered_linear = render_full_linear(&gaussians, &train_camera, &bg);
        let (loss, d_image) =
            l2_image_loss_and_grad_weighted(&rendered_linear, &train_target_linear, &weights);
        train_loss = loss; // Track most recent loss

        // Backward
        let (_img_u8, d_color, d_opacity_logits, d_bg) =
            render_full_color_grads(&gaussians, &train_camera, &d_image, &bg);

        // Convert dL/d(color) -> dL/d(SH0 coeff)
        let grads: Vec<Vector3<f32>> = d_color.iter().map(|dc| dc * sh0_basis).collect();

        adam.step(&mut params, &grads);
        if cfg.learn_opacity {
            opacity_opt.step(&mut opacity_logits, &d_opacity_logits);
        }
        if cfg.learn_background {
            let mut bg_param = vec![bg];
            let bg_grad = vec![d_bg];
            bg_opt.step(&mut bg_param, &bg_grad);
            bg = bg_param[0];
        }

        // Validation
        if (iter + 1) % cfg.val_interval == 0 || iter + 1 == cfg.iters {
            let mut test_psnr_sum = 0.0f32;
            for &test_idx in test_indices_for_metrics {
                let test_image_info = &scene.images[test_idx];
                let test_base_camera = scene
                    .cameras
                    .get(&test_image_info.camera_id)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Camera {} not found", test_image_info.camera_id)
                    })?;
                let test_rotation = test_image_info.rotation.to_rotation_matrix().into_inner();
                let test_camera_full = camera_with_pose(
                    test_base_camera,
                    test_rotation,
                    test_image_info.translation,
                );
                let test_camera = downsample_camera(&test_camera_full, cfg.downsample_factor);

                let test_target = load_target_image(&cfg.images_dir, &test_image_info.name)?;
                let test_target_ds =
                    downsample_rgb_nearest(&test_target, test_camera.width, test_camera.height);
                let test_target_linear = rgb8_to_linear_vec(&test_target_ds);

                let rendered = render_full_linear(&gaussians, &test_camera, &bg);
                let psnr = compute_psnr(&rendered, &test_target_linear);
                test_psnr_sum += psnr;
            }
            let avg_test_psnr = test_psnr_sum / (test_indices_for_metrics.len() as f32);

            eprintln!(
                "iter {iter:4}  train_loss={loss:.6}  test_psnr={avg_test_psnr:.2} dB  bg=({:.3},{:.3},{:.3})",
                bg.x, bg.y, bg.z
            );
        }
    }

    // Final validation
    let mut final_psnr_sum = 0.0f32;
    let mut test_view_sample = None;
    let mut test_view_target = None;

    for (i, g) in gaussians.iter_mut().enumerate() {
        g.sh_coeffs[0][0] = params[i].x;
        g.sh_coeffs[0][1] = params[i].y;
        g.sh_coeffs[0][2] = params[i].z;
        if cfg.learn_opacity {
            g.opacity = opacity_logits[i].clamp(-10.0, 10.0);
        }
    }

    for (i, &test_idx) in test_indices_for_metrics.iter().enumerate() {
        let test_image_info = &scene.images[test_idx];
        let test_base_camera = scene
            .cameras
            .get(&test_image_info.camera_id)
            .ok_or_else(|| anyhow::anyhow!("Camera {} not found", test_image_info.camera_id))?;
        let test_rotation = test_image_info.rotation.to_rotation_matrix().into_inner();
        let test_camera_full =
            camera_with_pose(test_base_camera, test_rotation, test_image_info.translation);
        let test_camera = downsample_camera(&test_camera_full, cfg.downsample_factor);

        let test_target = load_target_image(&cfg.images_dir, &test_image_info.name)?;
        let test_target_ds =
            downsample_rgb_nearest(&test_target, test_camera.width, test_camera.height);
        let test_target_linear = rgb8_to_linear_vec(&test_target_ds);

        let rendered = render_full_linear(&gaussians, &test_camera, &bg);
        let psnr = compute_psnr(&rendered, &test_target_linear);
        final_psnr_sum += psnr;

        // Save first test view for visual inspection
        if i == 0 {
            test_view_sample = Some(linear_vec_to_rgb8_img(
                &rendered,
                test_camera.width,
                test_camera.height,
            ));
            test_view_target = Some(test_target_ds);
        }
    }

    let final_psnr = final_psnr_sum / (test_indices_for_metrics.len() as f32);

    eprintln!("\n✅ Multi-view training complete!");
    eprintln!("Initial test PSNR: {:.2} dB", initial_psnr);
    eprintln!("Final test PSNR:   {:.2} dB", final_psnr);
    eprintln!("Improvement:       {:.2} dB", final_psnr - initial_psnr);

    Ok(MultiViewTrainOutputs {
        initial_psnr,
        final_psnr,
        train_loss,
        num_train_views: train_indices.len(),
        num_test_views: test_indices.len(),
        test_view_sample: test_view_sample.unwrap(),
        test_view_target: test_view_target.unwrap(),
    })
}
