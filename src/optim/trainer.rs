//! Training orchestration (M7).
//!
//! This is an intentionally minimal "single-image overfit" trainer that
//! optimizes only the SH DC coefficient (color) for each Gaussian.
//!
//! Why:
//! - Validates differentiable rendering end-to-end
//! - Keeps the state space small and stable for early debugging
//!
//! Next:
//! - Add opacity + 2D eval + projection gradients for full parameter training
//! - Switch to multi-view sampling

use crate::core::{init_from_colmap_points, Camera, Gaussian};
use crate::io::load_colmap_scene;
use crate::optim::adam::AdamVec3;
use crate::optim::loss::l2_image_loss_and_grad;
use crate::render::full_diff::{
    debug_coverage_mask, debug_overlay_means, downsample_rgb_nearest, render_full_color_grads,
    render_full_linear, rgb8_to_linear_vec,
};
use image::RgbImage;
use nalgebra::{Matrix3, Vector3};
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
}

pub struct TrainOutputs {
    pub target: RgbImage,
    pub overlay: RgbImage,
    pub coverage: RgbImage,
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

    let base_camera = &scene.cameras[0];
    let rotation = image_info.rotation.to_rotation_matrix().into_inner();
    let camera_full = camera_with_pose(base_camera, rotation, image_info.translation);
    let camera = downsample_camera(&camera_full, cfg.downsample_factor);

    let cloud = init_from_colmap_points(&scene.points);
    let max_g = cfg.max_gaussians.min(cloud.gaussians.len());
    let mut gaussians: Vec<Gaussian> = cloud.gaussians[..max_g].to_vec();

    // Target image.
    let target_full = load_target_image(&cfg.images_dir, &image_info.name)?;
    let target_ds = downsample_rgb_nearest(&target_full, camera.width, camera.height);
    let target_linear = rgb8_to_linear_vec(&target_ds);

    // Debug outputs at training resolution, using the same gaussian subset.
    let overlay = debug_overlay_means(&target_ds, &gaussians, &camera, 1);
    let coverage = debug_coverage_mask(&gaussians, &camera);

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

    // Pull initial params.
    let sh0_basis = crate::core::sh_basis(&Vector3::new(0.0, 0.0, 1.0))[0]; // = C0
    let mut params: Vec<Vector3<f32>> = gaussians
        .iter()
        .map(|g| Vector3::new(g.sh_coeffs[0][0], g.sh_coeffs[0][1], g.sh_coeffs[0][2]))
        .collect();

    // Initial render for output.
    let initial_render_u8 = {
        let linear = render_full_linear(&gaussians, &camera, &bg);
        // Quantize via image buffer.
        let mut img = RgbImage::new(camera.width, camera.height);
        for (i, p) in linear.iter().enumerate() {
            let x = (i as u32) % camera.width;
            let y = (i as u32) / camera.width;
            img.put_pixel(
                x,
                y,
                image::Rgb([
                    (p.x * 255.0).clamp(0.0, 255.0) as u8,
                    (p.y * 255.0).clamp(0.0, 255.0) as u8,
                    (p.z * 255.0).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
        img
    };

    for iter in 0..cfg.iters {
        // Write params back into gaussians.
        for (i, g) in gaussians.iter_mut().enumerate() {
            g.sh_coeffs[0][0] = params[i].x;
            g.sh_coeffs[0][1] = params[i].y;
            g.sh_coeffs[0][2] = params[i].z;
        }

        // Forward (linear) and loss.
        let rendered_linear = render_full_linear(&gaussians, &camera, &bg);
        let (loss, d_image) = l2_image_loss_and_grad(&rendered_linear, &target_linear);

        // Backward: get dL/d(color_i) per Gaussian.
        let (_img_u8, d_color, d_bg) = render_full_color_grads(&gaussians, &camera, &d_image, &bg);

        // Convert dL/d(color) -> dL/d(SH0 coeff) using basis[0] (assuming clamp inactive).
        let grads: Vec<Vector3<f32>> = d_color.iter().map(|dc| dc * sh0_basis).collect();

        adam.step(&mut params, &grads);
        if cfg.learn_background {
            // AdamVec3 expects slices; update a single bg vector.
            let mut bg_param = vec![bg];
            let bg_grad = vec![d_bg];
            bg_opt.step(&mut bg_param, &bg_grad);
            bg = bg_param[0];
        }

        if iter % 10 == 0 || iter + 1 == cfg.iters {
            eprintln!("iter {iter:4}  loss={loss:.6}  bg=({:.3},{:.3},{:.3})", bg.x, bg.y, bg.z);
        }
    }

    // Final render.
    let final_render_u8 = {
        for (i, g) in gaussians.iter_mut().enumerate() {
            g.sh_coeffs[0][0] = params[i].x;
            g.sh_coeffs[0][1] = params[i].y;
            g.sh_coeffs[0][2] = params[i].z;
        }
        let linear = render_full_linear(&gaussians, &camera, &bg);
        let mut img = RgbImage::new(camera.width, camera.height);
        for (i, p) in linear.iter().enumerate() {
            let x = (i as u32) % camera.width;
            let y = (i as u32) / camera.width;
            img.put_pixel(
                x,
                y,
                image::Rgb([
                    (p.x * 255.0).clamp(0.0, 255.0) as u8,
                    (p.y * 255.0).clamp(0.0, 255.0) as u8,
                    (p.z * 255.0).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
        img
    };

    Ok(TrainOutputs {
        target: target_ds,
        overlay,
        coverage,
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
