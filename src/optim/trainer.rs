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

#[cfg(feature = "gpu")]
use crate::gpu::GpuRenderer;
use crate::optim::adam::{AdamF32, AdamSh16, AdamSo3, AdamVec3};
use crate::optim::loss::{
    l1_dssim_image_loss_and_grad_weighted, l2_image_loss_and_grad_weighted, LossKind,
};
use crate::core::sigmoid;
use crate::render::full_diff::{
    coverage_mask_bool, debug_contrib_count, debug_coverage_mask, debug_final_transmittance,
    debug_overlay_means, downsample_rgb_nearest, linear_vec_to_rgb8_img, render_full_color_grads,
    render_full_linear, rgb8_to_linear_vec,
};
use image::RgbImage;
use nalgebra::{Matrix3, Vector3};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct TrainConfig {
    pub sparse_dir: PathBuf,
    pub images_dir: PathBuf,
    pub image_index: usize,
    pub max_gaussians: usize,
    pub downsample_factor: f32,
    pub iters: usize,
    pub lr: f32, // Default/fallback learning rate
    pub lr_position: f32,
    pub lr_rotation: f32,
    pub lr_scale: f32,
    pub lr_opacity: f32,
    pub lr_sh: f32,
    pub lr_background: f32,
    pub learn_background: bool,
    pub learn_opacity: bool,
    pub loss: LossKind,
    pub learn_position: bool,
    pub learn_scale: bool,
    pub learn_rotation: bool,
    pub learn_sh: bool,
    /// Print per-iteration timing every N iterations (0 disables).
    pub log_interval: usize,
    /// Optional RNG seed for deterministic runs.
    pub rng_seed: Option<u64>,
    /// Use GPU for forward rendering.
    pub use_gpu: bool,
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

    // Initialize GPU renderer if requested
    #[cfg(feature = "gpu")]
    let gpu_renderer = if cfg.use_gpu {
        eprintln!("Initializing GPU renderer...");
        Some(GpuRenderer::new().expect("Failed to initialize GPU"))
    } else {
        None
    };

    #[cfg(not(feature = "gpu"))]
    let gpu_renderer: Option<()> = None;

    #[cfg(not(feature = "gpu"))]
    if cfg.use_gpu {
        return Err(anyhow::anyhow!("GPU rendering requested but not compiled with --features gpu"));
    }

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
    let mut bg_opt = AdamVec3::new(cfg.lr_background, 0.9, 0.999, 1e-8);

    // Optimizer state for SH coeffs (RGB × 16).
    let mut sh_opt = AdamSh16::new(cfg.lr_sh, 0.9, 0.999, 1e-8);
    let mut opacity_opt = AdamF32::new(cfg.lr_opacity, 0.9, 0.999, 1e-8);
    let mut position_opt = AdamVec3::new(cfg.lr_position, 0.9, 0.999, 1e-8);
    let mut scale_opt = AdamVec3::new(cfg.lr_scale, 0.9, 0.999, 1e-8);
    let mut rotation_opt = AdamSo3::new(cfg.lr_rotation, 0.9, 0.999, 1e-8);

    // Pull initial SH params.
    let mut sh_params: Vec<[Vector3<f32>; 16]> = gaussians
        .iter()
        .map(|g| {
            let mut out = [Vector3::<f32>::zeros(); 16];
            for k in 0..16 {
                out[k] = Vector3::new(g.sh_coeffs[k][0], g.sh_coeffs[k][1], g.sh_coeffs[k][2]);
            }
            out
        })
        .collect();
    let mut opacity_logits: Vec<f32> = gaussians.iter().map(|g| g.opacity).collect();
    let mut positions: Vec<Vector3<f32>> = gaussians.iter().map(|g| g.position).collect();
    let mut log_scales: Vec<Vector3<f32>> = gaussians.iter().map(|g| g.scale).collect();
    let mut rotations: Vec<nalgebra::UnitQuaternion<f32>> =
        gaussians.iter().map(|g| g.rotation).collect();

    // Conditional render function: GPU if available, otherwise CPU
    #[cfg(feature = "gpu")]
    let render = |gaussians: &[Gaussian], camera: &Camera, bg: &Vector3<f32>| {
        if let Some(ref renderer) = gpu_renderer {
            renderer.render(gaussians, camera, bg)
        } else {
            render_full_linear(gaussians, camera, bg)
        }
    };

    #[cfg(not(feature = "gpu"))]
    let render = |gaussians: &[Gaussian], camera: &Camera, bg: &Vector3<f32>| {
        render_full_linear(gaussians, camera, bg)
    };

    // Initial render for output.
    let initial_render_u8 = linear_vec_to_rgb8_img(
        &render(&gaussians, &camera, &bg),
        camera.width,
        camera.height,
    );

    for iter in 0..cfg.iters {
        let should_log =
            cfg.log_interval > 0 && (iter == 0 || iter % cfg.log_interval == 0 || iter + 1 == cfg.iters);
        let iter_start = if should_log { Some(Instant::now()) } else { None };

        // Write params back into gaussians.
        for (i, g) in gaussians.iter_mut().enumerate() {
            for k in 0..16 {
                g.sh_coeffs[k][0] = sh_params[i][k].x;
                g.sh_coeffs[k][1] = sh_params[i][k].y;
                g.sh_coeffs[k][2] = sh_params[i][k].z;
            }
            // Even when we don't learn a parameter, keep the struct in sync with the
            // (fixed) parameter vectors so downstream logic has one source of truth.
            g.opacity = opacity_logits[i].clamp(-10.0, 10.0);
            g.position = positions[i];
            g.scale = log_scales[i];
            g.rotation = rotations[i];
        }

        // Forward (linear) and loss.
        if should_log {
            eprintln!(
                "iter {}/{} start  gaussians={}  res={}x{}",
                iter + 1,
                cfg.iters,
                gaussians.len(),
                camera.width,
                camera.height
            );
        }

        let t0 = Instant::now();
        let rendered_linear = render(&gaussians, &camera, &bg);
        let t_forward = t0.elapsed();
        let (loss, d_image) = match cfg.loss {
            crate::optim::loss::LossKind::L2 => {
                l2_image_loss_and_grad_weighted(&rendered_linear, &target_linear, &weights)
            }
            crate::optim::loss::LossKind::L1Dssim => l1_dssim_image_loss_and_grad_weighted(
                &rendered_linear,
                &target_linear,
                &weights,
                camera.width,
                camera.height,
            ),
        };

        // Backward: get dL/d(color_i) and dL/d(opacity_logit_i) per Gaussian.
        let t1 = Instant::now();
        let (_img_u8, d_color, d_opacity_logits, d_positions, d_log_scales, d_rot_vecs, d_bg) = {
            #[cfg(feature = "gpu")]
            if let Some(ref renderer) = gpu_renderer {
                // Use GPU backward pass with sparse atomic gradients (efficient for <10k Gaussians)
                let (_pixels, grads_2d) = renderer.render_with_gradients(&gaussians, &camera, &bg, &d_image);

                // Check if GPU fallback occurred (empty gradients)
                if grads_2d.d_colors.is_empty() {
                    // Fall back to CPU
                    render_full_color_grads(&gaussians, &camera, &d_image, &bg)
                } else {
                    // GPU gradients succeeded
                    let (d_pos, d_scales, d_rots, d_background) = crate::gpu::chain_2d_to_3d_gradients(&grads_2d, &gaussians, &camera);
                    let dummy_img = image::RgbImage::new(camera.width, camera.height);
                    (dummy_img, grads_2d.d_colors, grads_2d.d_opacity_logits, d_pos, d_scales, d_rots, d_background)
                }
            } else {
                // CPU backward pass
                render_full_color_grads(&gaussians, &camera, &d_image, &bg)
            }

            #[cfg(not(feature = "gpu"))]
            render_full_color_grads(&gaussians, &camera, &d_image, &bg)
        };
        let t_backward = t1.elapsed();

        // Convert dL/d(color) -> dL/d(SH coeffs) using per-Gaussian SH basis.
        let mut d_sh: Vec<[Vector3<f32>; 16]> = vec![[Vector3::zeros(); 16]; gaussians.len()];
        if cfg.learn_sh {
            for (i, g) in gaussians.iter().enumerate() {
                let basis = crate::core::sh_basis(&camera.view_direction(&g.position));
                for k in 0..16 {
                    d_sh[i][k] = d_color[i] * basis[k];
                }
            }
        } else {
            // DC-only learning (k=0).
            let sh0 = crate::core::sh_basis(&Vector3::new(0.0, 0.0, 1.0))[0];
            for i in 0..gaussians.len() {
                d_sh[i][0] = d_color[i] * sh0;
            }
        }

        let t2 = Instant::now();
        sh_opt.step(&mut sh_params, &d_sh);
        if cfg.learn_opacity {
            opacity_opt.step(&mut opacity_logits, &d_opacity_logits);
        }
        if cfg.learn_position {
            position_opt.step(&mut positions, &d_positions);

            // Clip positions to scene bounds to prevent Gaussians escaping to infinity
            const MAX_SCENE_RADIUS: f32 = 1000.0;
            for pos in positions.iter_mut() {
                let pos_mag = pos.norm();
                if pos_mag > MAX_SCENE_RADIUS {
                    *pos = *pos * (MAX_SCENE_RADIUS / pos_mag);
                }
            }
        }
        if cfg.learn_scale {
            scale_opt.step(&mut log_scales, &d_log_scales);

            // Clamp log-space scales to prevent exp() overflow
            // exp(20) ≈ 485M (very large), exp(-20) ≈ 2e-9 (very small but valid)
            const MAX_LOG_SCALE: f32 = 20.0;
            const MIN_LOG_SCALE: f32 = -20.0;
            for scale in log_scales.iter_mut() {
                scale.x = scale.x.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);
                scale.y = scale.y.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);
                scale.z = scale.z.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);
            }
        }
        if cfg.learn_rotation {
            rotation_opt.step(&mut rotations, &d_rot_vecs);
        }
        if cfg.learn_background {
            // AdamVec3 expects slices; update a single bg vector.
            let mut bg_param = vec![bg];
            let bg_grad = vec![d_bg];
            bg_opt.step(&mut bg_param, &bg_grad);
            bg = bg_param[0];
        }
        let t_step = t2.elapsed();

        if should_log {
            let total = iter_start.unwrap().elapsed();
            eprintln!(
                "iter {}/{} done   loss={loss:.6}  forward={:.2?} backward={:.2?} step={:.2?} total={:.2?}  bg=({:.3},{:.3},{:.3})",
                iter + 1,
                cfg.iters,
                t_forward,
                t_backward,
                t_step,
                total,
                bg.x, bg.y, bg.z
            );
        }
    }

    // Final render.
    let final_render_u8 = {
        for (i, g) in gaussians.iter_mut().enumerate() {
            for k in 0..16 {
                g.sh_coeffs[k][0] = sh_params[i][k].x;
                g.sh_coeffs[k][1] = sh_params[i][k].y;
                g.sh_coeffs[k][2] = sh_params[i][k].z;
            }
            g.opacity = opacity_logits[i].clamp(-10.0, 10.0);
            g.position = positions[i];
            g.scale = log_scales[i];
            g.rotation = rotations[i];
        }
        linear_vec_to_rgb8_img(
            &render(&gaussians, &camera, &bg),
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
    pub lr: f32, // Default/fallback learning rate
    pub lr_position: f32,
    pub lr_rotation: f32,
    pub lr_scale: f32,
    pub lr_opacity: f32,
    pub lr_sh: f32,
    pub lr_background: f32,
    pub learn_background: bool,
    pub learn_opacity: bool,
    pub loss: LossKind,
    pub learn_position: bool,
    pub learn_scale: bool,
    pub learn_rotation: bool,
    pub learn_sh: bool,
    /// If non-zero, only use the first N images from `images.bin` (for faster iteration).
    pub max_images: usize,
    /// Optional RNG seed for deterministic train/test splits and view sampling.
    pub rng_seed: Option<u64>,
    pub train_fraction: f32, // Fraction of images for training (rest for testing)
    pub val_interval: usize,  // Validate every N iterations
    /// Limit how many held-out views are used for PSNR reporting.
    /// Use `0` to evaluate all test views (can be slow on large datasets).
    pub max_test_views_for_metrics: usize,
    /// Print per-iteration timing every N iterations (0 disables).
    pub log_interval: usize,
    /// Every N iterations, run densification (0 disables).
    pub densify_interval: usize,
    /// Maximum gaussian count after densification (0 disables cap).
    pub densify_max_gaussians: usize,
    /// Split/clone if average position-grad norm exceeds this threshold.
    pub densify_grad_threshold: f32,
    /// If opacity (sigmoid) is below this, prune the gaussian.
    pub prune_opacity_threshold: f32,
    /// If average world sigma (exp(log_scale)) is above this, SPLIT; otherwise CLONE.
    pub split_sigma_threshold: f32,
    /// Use GPU for forward rendering.
    pub use_gpu: bool,
}

pub struct MultiViewTrainOutputs {
    pub initial_psnr: f32,
    pub final_psnr: f32,
    pub train_loss: f32,
    pub num_train_views: usize,
    pub num_test_views: usize,
    pub initial_num_gaussians: usize,
    pub final_num_gaussians: usize,
    pub densify_events: usize,
    pub test_view_sample: RgbImage, // One test view rendering for visual check
    pub test_view_target: RgbImage,
    pub gaussians: Vec<Gaussian>, // Trained Gaussians for model saving
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

fn mean_world_sigma(g: &Gaussian) -> f32 {
    let s = g.actual_scale();
    (s.x + s.y + s.z) / 3.0
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DensifyStats {
    before: usize,
    after: usize,
    kept: usize,
    pruned: usize,
    pruned_outliers: usize,
    split: usize,
    cloned: usize,
    cap_hit: bool,
    grad_p50: f32,
    grad_p90: f32,
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = ((sorted.len() - 1) as f32 * p.clamp(0.0, 1.0)).round() as usize;
    sorted[idx]
}

fn logit_from_alpha(alpha: f32) -> f32 {
    let a = alpha.clamp(1e-6, 1.0 - 1e-6);
    (a / (1.0 - a)).ln()
}

fn split_opacity_logit(parent_logit: f32, children: usize) -> f32 {
    if children <= 1 {
        return parent_logit;
    }
    let a_parent = sigmoid(parent_logit);
    // Preserve total alpha under the approximation:
    // alpha_total = 1 - Π_k (1 - alpha_k), with all children equal.
    // Solve for alpha_child:
    // 1 - (1 - alpha_child)^children = alpha_parent
    // => alpha_child = 1 - (1 - alpha_parent)^(1/children)
    let a_child = 1.0 - (1.0 - a_parent).powf(1.0 / (children as f32));
    logit_from_alpha(a_child)
}

fn densify_and_prune<R: Rng + ?Sized>(
    gaussians: &mut Vec<Gaussian>,
    sh_params: &mut Vec<[Vector3<f32>; 16]>,
    opacity_logits: &mut Vec<f32>,
    positions: &mut Vec<Vector3<f32>>,
    log_scales: &mut Vec<Vector3<f32>>,
    rotations: &mut Vec<nalgebra::UnitQuaternion<f32>>,
    grad_accum: &mut Vec<f32>,
    rng: &mut R,
    iters_in_window: usize,
    max_gaussians: usize,
    grad_threshold: f32,
    prune_opacity_threshold: f32,
    split_sigma_threshold: f32,
) -> DensifyStats {
    let before = gaussians.len();
    if iters_in_window == 0 || gaussians.is_empty() {
        return DensifyStats {
            before,
            after: before,
            kept: before,
            pruned: 0,
            pruned_outliers: 0,
            split: 0,
            cloned: 0,
            cap_hit: false,
            grad_p50: f32::NAN,
            grad_p90: f32::NAN,
        };
    }

    // Compute scene center for outlier detection
    let scene_center = {
        let mut sum = nalgebra::Vector3::zeros();
        for pos in positions.iter() {
            sum += pos;
        }
        sum / (positions.len() as f32)
    };
    const OUTLIER_DISTANCE_THRESHOLD: f32 = 50.0; // 50 meters from center

    let cap = if max_gaussians == 0 {
        usize::MAX
    } else {
        max_gaussians
    };
    // Never drop existing Gaussians just because the post-densify cap is smaller than current size.
    // The cap is intended to limit *additions* during densification.
    let cap = cap.max(before);

    let mut new_gaussians = Vec::with_capacity(gaussians.len());
    let mut new_sh_params = Vec::with_capacity(sh_params.len());
    let mut new_opacity_logits = Vec::with_capacity(opacity_logits.len());
    let mut new_positions = Vec::with_capacity(positions.len());
    let mut new_log_scales = Vec::with_capacity(log_scales.len());
    let mut new_rotations = Vec::with_capacity(rotations.len());
    let mut new_grad_accum = Vec::with_capacity(grad_accum.len());

    let mut kept = 0usize;
    let mut pruned = 0usize;
    let mut pruned_outliers = 0usize;
    let mut split = 0usize;
    let mut cloned = 0usize;
    let mut cap_hit = false;
    let mut kept_avg_grads: Vec<f32> = Vec::new();

    for i in 0..gaussians.len() {
        // Prune outliers: gaussians too far from scene center
        let distance_from_center = (positions[i] - scene_center).norm();
        if distance_from_center > OUTLIER_DISTANCE_THRESHOLD {
            pruned_outliers += 1;
            pruned += 1;
            continue;
        }

        let opacity = sigmoid(opacity_logits[i]);
        if opacity < prune_opacity_threshold {
            pruned += 1;
            continue;
        }

        let avg_grad = grad_accum[i] / (iters_in_window as f32);
        kept_avg_grads.push(avg_grad);

        // Always keep the original (but ensure it matches our parameter vectors).
        let keep_idx = new_gaussians.len();
        let mut keep = gaussians[i].clone();
        for k in 0..16 {
            keep.sh_coeffs[k][0] = sh_params[i][k].x;
            keep.sh_coeffs[k][1] = sh_params[i][k].y;
            keep.sh_coeffs[k][2] = sh_params[i][k].z;
        }
        keep.opacity = opacity_logits[i].clamp(-10.0, 10.0);
        keep.position = positions[i];
        keep.scale = log_scales[i];
        keep.rotation = rotations[i];
        new_gaussians.push(keep);
        new_sh_params.push(sh_params[i]);
        new_opacity_logits.push(opacity_logits[i]);
        new_positions.push(positions[i]);
        new_log_scales.push(log_scales[i]);
        new_rotations.push(rotations[i]);
        new_grad_accum.push(0.0);
        kept += 1;

        let can_add = new_gaussians.len() < cap;
        if !(avg_grad > grad_threshold && can_add) {
            if avg_grad > grad_threshold && !can_add {
                cap_hit = true;
            }
            continue;
        }

        let sigma = mean_world_sigma(&gaussians[i]);
        let jitter = (0.5 * sigma).clamp(1e-4, 5e-3);
        let mut dir = Vector3::new(
            rng.gen_range(-1.0f32..1.0f32),
            rng.gen_range(-1.0f32..1.0f32),
            rng.gen_range(-1.0f32..1.0f32),
        );
        if dir.norm_squared() < 1e-12 {
            dir = Vector3::new(1.0, 0.0, 0.0);
        } else {
            dir = dir.normalize();
        }
        let new_pos = positions[i] + dir * jitter;

        // We are turning 1 Gaussian into 2; split opacity between them so the render doesn't
        // abruptly get more opaque just because we duplicated a primitive.
        let child_opacity_logit = split_opacity_logit(opacity_logits[i], 2).clamp(-10.0, 10.0);
        new_opacity_logits[keep_idx] = child_opacity_logit;
        new_gaussians[keep_idx].opacity = child_opacity_logit;

        if sigma > split_sigma_threshold {
            // SPLIT: add a second gaussian, slightly offset and slightly smaller.
            let mut g2 = gaussians[i].clone();
            for k in 0..16 {
                g2.sh_coeffs[k][0] = sh_params[i][k].x;
                g2.sh_coeffs[k][1] = sh_params[i][k].y;
                g2.sh_coeffs[k][2] = sh_params[i][k].z;
            }
            g2.opacity = child_opacity_logit;
            g2.position = new_pos;
            g2.scale = log_scales[i] - Vector3::new(0.2, 0.2, 0.2);
            g2.rotation = rotations[i];
            new_gaussians.push(g2);
            new_sh_params.push(sh_params[i]);
            new_opacity_logits.push(child_opacity_logit);
            new_positions.push(new_pos);
            new_log_scales.push(log_scales[i] - Vector3::new(0.2, 0.2, 0.2));
            new_rotations.push(rotations[i]);
            new_grad_accum.push(0.0);
            split += 1;
        } else {
            // CLONE: same scale, slight offset.
            let mut g2 = gaussians[i].clone();
            for k in 0..16 {
                g2.sh_coeffs[k][0] = sh_params[i][k].x;
                g2.sh_coeffs[k][1] = sh_params[i][k].y;
                g2.sh_coeffs[k][2] = sh_params[i][k].z;
            }
            g2.opacity = child_opacity_logit;
            g2.position = new_pos;
            g2.scale = log_scales[i];
            g2.rotation = rotations[i];
            new_gaussians.push(g2);
            new_sh_params.push(sh_params[i]);
            new_opacity_logits.push(child_opacity_logit);
            new_positions.push(new_pos);
            new_log_scales.push(log_scales[i]);
            new_rotations.push(rotations[i]);
            new_grad_accum.push(0.0);
            cloned += 1;
        }
    }

    *gaussians = new_gaussians;
    *sh_params = new_sh_params;
    *opacity_logits = new_opacity_logits;
    *positions = new_positions;
    *log_scales = new_log_scales;
    *rotations = new_rotations;
    *grad_accum = new_grad_accum;

    kept_avg_grads.sort_by(|a, b| a.total_cmp(b));
    let grad_p50 = percentile(&kept_avg_grads, 0.50);
    let grad_p90 = percentile(&kept_avg_grads, 0.90);

    DensifyStats {
        before,
        after: gaussians.len(),
        kept,
        pruned,
        pruned_outliers,
        split,
        cloned,
        cap_hit,
        grad_p50,
        grad_p90,
    }
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

    let available_images = if cfg.max_images == 0 {
        scene.images.len()
    } else {
        cfg.max_images.min(scene.images.len())
    };
    if available_images < 2 {
        return Err(anyhow::anyhow!(
            "Need at least 2 images for multi-view training (max_images={})",
            cfg.max_images
        ));
    }

    // Split images into train/test sets
    let mut image_indices: Vec<usize> = (0..available_images).collect();
    let mut rng = if let Some(seed) = cfg.rng_seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };
    image_indices.shuffle(&mut rng);

    let num_train = ((available_images as f32) * cfg.train_fraction).max(1.0) as usize;
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

    #[derive(Clone)]
    struct ViewData {
        camera: Camera,
        target_ds: RgbImage,
        target_linear: Vec<Vector3<f32>>,
    }

    let mut view_cache: HashMap<usize, ViewData> = HashMap::new();
    let should_preload_images = cfg.max_images != 0;
    if should_preload_images {
        for &idx in image_indices.iter() {
            let image_info = &scene.images[idx];
            let base_camera = scene
                .cameras
                .get(&image_info.camera_id)
                .ok_or_else(|| anyhow::anyhow!("Camera {} not found", image_info.camera_id))?;
            let rotation = image_info.rotation.to_rotation_matrix().into_inner();
            let camera_full = camera_with_pose(base_camera, rotation, image_info.translation);
            let camera = downsample_camera(&camera_full, cfg.downsample_factor);

            let target = load_target_image(&cfg.images_dir, &image_info.name)?;
            let target_ds = downsample_rgb_nearest(&target, camera.width, camera.height);
            let target_linear = rgb8_to_linear_vec(&target_ds);

            view_cache.insert(
                idx,
                ViewData {
                    camera,
                    target_ds,
                    target_linear,
                },
            );
        }
        eprintln!(
            "Preloaded {} images (max_images={})",
            view_cache.len(),
            cfg.max_images
        );
    }

    // Use first training view to initialize camera and Gaussians
    let first_train_idx = train_indices[0];
    let first_image_info = &scene.images[first_train_idx];
    let camera = if let Some(v) = view_cache.get(&first_train_idx) {
        v.camera.clone()
    } else {
        let base_camera = scene
            .cameras
            .get(&first_image_info.camera_id)
            .ok_or_else(|| anyhow::anyhow!("Camera {} not found", first_image_info.camera_id))?;
        let rotation = first_image_info.rotation.to_rotation_matrix().into_inner();
        let camera_full = camera_with_pose(base_camera, rotation, first_image_info.translation);
        downsample_camera(&camera_full, cfg.downsample_factor)
    };

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
    let initial_num_gaussians = gaussians.len();

    // Initialize GPU renderer if requested
    #[cfg(feature = "gpu")]
    let gpu_renderer = if cfg.use_gpu {
        eprintln!("Initializing GPU renderer...");
        Some(GpuRenderer::new().expect("Failed to initialize GPU"))
    } else {
        None
    };

    #[cfg(not(feature = "gpu"))]
    let gpu_renderer: Option<()> = None;

    #[cfg(not(feature = "gpu"))]
    if cfg.use_gpu {
        return Err(anyhow::anyhow!("GPU rendering requested but not compiled with --features gpu"));
    }

    // Initialize background color (using first view's mean)
    let first_target_linear = if let Some(v) = view_cache.get(&first_train_idx) {
        v.target_linear.clone()
    } else {
        let first_target = load_target_image(&cfg.images_dir, &first_image_info.name)?;
        let first_target_ds = downsample_rgb_nearest(&first_target, camera.width, camera.height);
        rgb8_to_linear_vec(&first_target_ds)
    };

    let mut bg = {
        let mut acc = Vector3::<f32>::zeros();
        for p in &first_target_linear {
            acc += *p;
        }
        acc / (first_target_linear.len() as f32).max(1.0)
    };
    let mut bg_opt = AdamVec3::new(cfg.lr_background, 0.9, 0.999, 1e-8);

    // Optimizer state for SH coeffs (RGB × 16)
    let mut sh_opt = AdamSh16::new(cfg.lr_sh, 0.9, 0.999, 1e-8);
    let mut opacity_opt = AdamF32::new(cfg.lr_opacity, 0.9, 0.999, 1e-8);
    let mut position_opt = AdamVec3::new(cfg.lr_position, 0.9, 0.999, 1e-8);
    let mut scale_opt = AdamVec3::new(cfg.lr_scale, 0.9, 0.999, 1e-8);
    let mut rotation_opt = AdamSo3::new(cfg.lr_rotation, 0.9, 0.999, 1e-8);

    // Pull initial SH params.
    let mut sh_params: Vec<[Vector3<f32>; 16]> = gaussians
        .iter()
        .map(|g| {
            let mut out = [Vector3::<f32>::zeros(); 16];
            for k in 0..16 {
                out[k] = Vector3::new(g.sh_coeffs[k][0], g.sh_coeffs[k][1], g.sh_coeffs[k][2]);
            }
            out
        })
        .collect();
    let mut opacity_logits: Vec<f32> = gaussians.iter().map(|g| g.opacity).collect();
    let mut positions: Vec<Vector3<f32>> = gaussians.iter().map(|g| g.position).collect();
    let mut log_scales: Vec<Vector3<f32>> = gaussians.iter().map(|g| g.scale).collect();
    let mut rotations: Vec<nalgebra::UnitQuaternion<f32>> =
        gaussians.iter().map(|g| g.rotation).collect();
    let mut grad_accum_pos_norm: Vec<f32> = vec![0.0; gaussians.len()];
    let mut grad_window_iters: usize = 0;

    // Conditional render function: GPU if available, otherwise CPU
    #[cfg(feature = "gpu")]
    let render = |gaussians: &[Gaussian], camera: &Camera, bg: &Vector3<f32>| {
        if let Some(ref renderer) = gpu_renderer {
            renderer.render(gaussians, camera, bg)
        } else {
            render_full_linear(gaussians, camera, bg)
        }
    };

    #[cfg(not(feature = "gpu"))]
    let render = |gaussians: &[Gaussian], camera: &Camera, bg: &Vector3<f32>| {
        render_full_linear(gaussians, camera, bg)
    };

    // Compute initial PSNR on test views
    let initial_psnr = {
        let mut psnr_sum = 0.0f32;
        for &test_idx in test_indices_for_metrics {
            let (test_camera, test_target_linear) = if let Some(v) = view_cache.get(&test_idx) {
                (v.camera.clone(), v.target_linear.clone())
            } else {
                let test_image_info = &scene.images[test_idx];
                let test_base_camera = scene
                    .cameras
                    .get(&test_image_info.camera_id)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Camera {} not found", test_image_info.camera_id)
                    })?;
                let test_rotation = test_image_info.rotation.to_rotation_matrix().into_inner();
                let test_camera_full =
                    camera_with_pose(test_base_camera, test_rotation, test_image_info.translation);
                let test_camera = downsample_camera(&test_camera_full, cfg.downsample_factor);

                let test_target = load_target_image(&cfg.images_dir, &test_image_info.name)?;
                let test_target_ds =
                    downsample_rgb_nearest(&test_target, test_camera.width, test_camera.height);
                let test_target_linear = rgb8_to_linear_vec(&test_target_ds);
                (test_camera, test_target_linear)
            };

            let rendered = render(&gaussians, &test_camera, &bg);
            let psnr = compute_psnr(&rendered, &test_target_linear);
            psnr_sum += psnr;
        }
        psnr_sum / (test_indices_for_metrics.len() as f32)
    };

    eprintln!("Initial test PSNR: {:.2} dB", initial_psnr);

    // Learning rate scheduling: exponential decay
    // Decay all LRs by 10× over the full training (exponential)
    let lr_decay_rate = 0.1f32.powf(1.0 / cfg.iters as f32);
    let initial_lr_position = cfg.lr_position;
    let initial_lr_rotation = cfg.lr_rotation;
    let initial_lr_scale = cfg.lr_scale;
    let initial_lr_opacity = cfg.lr_opacity;
    let initial_lr_sh = cfg.lr_sh;
    let initial_lr_background = cfg.lr_background;

    // Training loop: sample random views
    let mut train_loss = 0.0f32;
    let mut densify_events: usize = 0;
    for iter in 0..cfg.iters {
        // Apply LR schedule: exponential decay
        let lr_multiplier = lr_decay_rate.powi(iter as i32);
        position_opt.lr = initial_lr_position * lr_multiplier;
        rotation_opt.lr = initial_lr_rotation * lr_multiplier;
        scale_opt.lr = initial_lr_scale * lr_multiplier;
        opacity_opt.lr = initial_lr_opacity * lr_multiplier;
        sh_opt.lr = initial_lr_sh * lr_multiplier;
        bg_opt.lr = initial_lr_background * lr_multiplier;
        let should_log =
            cfg.log_interval > 0 && (iter == 0 || iter % cfg.log_interval == 0 || iter + 1 == cfg.iters);
        let iter_start = if should_log { Some(Instant::now()) } else { None };

        // Sample a random training view
        let train_idx = *train_indices
            .choose(&mut rng)
            .expect("train_indices is non-empty");
        let (train_camera, train_target_linear) = if let Some(v) = view_cache.get(&train_idx) {
            (v.camera.clone(), v.target_linear.clone())
        } else {
            let train_image_info = &scene.images[train_idx];
            let train_base_camera = scene
                .cameras
                .get(&train_image_info.camera_id)
                .ok_or_else(|| anyhow::anyhow!("Camera {} not found", train_image_info.camera_id))?;
            let train_rotation = train_image_info.rotation.to_rotation_matrix().into_inner();
            let train_camera_full =
                camera_with_pose(train_base_camera, train_rotation, train_image_info.translation);
            let train_camera = downsample_camera(&train_camera_full, cfg.downsample_factor);

            let train_target = load_target_image(&cfg.images_dir, &train_image_info.name)?;
            let train_target_ds =
                downsample_rgb_nearest(&train_target, train_camera.width, train_camera.height);
            let train_target_linear = rgb8_to_linear_vec(&train_target_ds);
            (train_camera, train_target_linear)
        };

        // Write params back into gaussians
        for (i, g) in gaussians.iter_mut().enumerate() {
            for k in 0..16 {
                g.sh_coeffs[k][0] = sh_params[i][k].x;
                g.sh_coeffs[k][1] = sh_params[i][k].y;
                g.sh_coeffs[k][2] = sh_params[i][k].z;
            }
            g.opacity = opacity_logits[i].clamp(-10.0, 10.0);
            g.position = positions[i];
            g.scale = log_scales[i];
            g.rotation = rotations[i];
        }

        // Coverage weighting (use current params)
        if should_log {
            eprintln!(
                "iter {}/{} start  view={}  gaussians={}  res={}x{}",
                iter + 1,
                cfg.iters,
                scene.images[train_idx].name,
                gaussians.len(),
                train_camera.width,
                train_camera.height
            );
        }

        let t0 = Instant::now();

        // Skip coverage computation when using GPU (it's a full render pass!)
        // With GPU rendering being fast, we can afford to weight all pixels equally
        #[cfg(feature = "gpu")]
        let weights: Vec<f32> = if cfg.use_gpu {
            vec![1.0; (train_camera.width * train_camera.height) as usize]
        } else {
            let coverage_bool = coverage_mask_bool(&gaussians, &train_camera);
            coverage_bool
                .iter()
                .map(|&c| if c { 1.0 } else { 0.1 })
                .collect()
        };

        #[cfg(not(feature = "gpu"))]
        let weights: Vec<f32> = {
            let coverage_bool = coverage_mask_bool(&gaussians, &train_camera);
            coverage_bool
                .iter()
                .map(|&c| if c { 1.0 } else { 0.1 })
                .collect()
        };

        // Forward and loss
        let rendered_linear = render(&gaussians, &train_camera, &bg);
        let (loss, d_image) = match cfg.loss {
            crate::optim::loss::LossKind::L2 => l2_image_loss_and_grad_weighted(
                &rendered_linear,
                &train_target_linear,
                &weights,
            ),
            crate::optim::loss::LossKind::L1Dssim => l1_dssim_image_loss_and_grad_weighted(
                &rendered_linear,
                &train_target_linear,
                &weights,
                train_camera.width,
                train_camera.height,
            ),
        };
        train_loss = loss; // Track most recent loss
        let t_forward = t0.elapsed();

        // Backward
        let t1 = Instant::now();
        let (_img_u8, d_color, d_opacity_logits, d_positions, d_log_scales, d_rot_vecs, d_bg) = {
            #[cfg(feature = "gpu")]
            if let Some(ref renderer) = gpu_renderer {
                // Use GPU backward pass with sparse atomic gradients (efficient for <10k Gaussians)
                let (_pixels, grads_2d) = renderer.render_with_gradients(&gaussians, &train_camera, &bg, &d_image);

                // Check if GPU fallback occurred (empty gradients)
                if grads_2d.d_colors.is_empty() {
                    // Fall back to CPU
                    render_full_color_grads(&gaussians, &train_camera, &d_image, &bg)
                } else {
                    // GPU gradients succeeded
                    let (d_pos, d_scales, d_rots, d_background) = crate::gpu::chain_2d_to_3d_gradients(&grads_2d, &gaussians, &train_camera);
                    let dummy_img = image::RgbImage::new(train_camera.width, train_camera.height);
                    (dummy_img, grads_2d.d_colors, grads_2d.d_opacity_logits, d_pos, d_scales, d_rots, d_background)
                }
            } else {
                // CPU backward pass
                render_full_color_grads(&gaussians, &train_camera, &d_image, &bg)
            }

            #[cfg(not(feature = "gpu"))]
            render_full_color_grads(&gaussians, &train_camera, &d_image, &bg)
        };
        let t_backward = t1.elapsed();

        // Convert dL/d(color) -> dL/d(SH coeffs) using per-Gaussian SH basis.
        let mut d_sh: Vec<[Vector3<f32>; 16]> = vec![[Vector3::zeros(); 16]; gaussians.len()];
        if cfg.learn_sh {
            for (i, g) in gaussians.iter().enumerate() {
                let basis = crate::core::sh_basis(&train_camera.view_direction(&g.position));
                for k in 0..16 {
                    d_sh[i][k] = d_color[i] * basis[k];
                }
            }
        } else {
            // DC-only learning (k=0).
            let sh0 = crate::core::sh_basis(&Vector3::new(0.0, 0.0, 1.0))[0];
            for i in 0..gaussians.len() {
                d_sh[i][0] = d_color[i] * sh0;
            }
        }

        let t2 = Instant::now();
        sh_opt.step(&mut sh_params, &d_sh);
        if cfg.learn_opacity {
            opacity_opt.step(&mut opacity_logits, &d_opacity_logits);
        }
        if cfg.learn_position {
            position_opt.step(&mut positions, &d_positions);

            // Clip positions to scene bounds to prevent Gaussians escaping to infinity
            const MAX_SCENE_RADIUS: f32 = 1000.0;
            for pos in positions.iter_mut() {
                let pos_mag = pos.norm();
                if pos_mag > MAX_SCENE_RADIUS {
                    *pos = *pos * (MAX_SCENE_RADIUS / pos_mag);
                }
            }
        }
        if cfg.learn_scale {
            scale_opt.step(&mut log_scales, &d_log_scales);

            // Clamp log-space scales to prevent exp() overflow
            // exp(20) ≈ 485M (very large), exp(-20) ≈ 2e-9 (very small but valid)
            const MAX_LOG_SCALE: f32 = 20.0;
            const MIN_LOG_SCALE: f32 = -20.0;
            for scale in log_scales.iter_mut() {
                scale.x = scale.x.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);
                scale.y = scale.y.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);
                scale.z = scale.z.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);
            }
        }
        if cfg.learn_rotation {
            rotation_opt.step(&mut rotations, &d_rot_vecs);
        }
        if cfg.learn_background {
            let mut bg_param = vec![bg];
            let bg_grad = vec![d_bg];
            bg_opt.step(&mut bg_param, &bg_grad);
            bg = bg_param[0];
        }
        let t_step = t2.elapsed();

        if cfg.densify_interval > 0 {
            if grad_accum_pos_norm.len() != d_positions.len() {
                grad_accum_pos_norm.resize(d_positions.len(), 0.0);
            }
            for (i, dp) in d_positions.iter().enumerate() {
                grad_accum_pos_norm[i] += dp.norm();
            }
            grad_window_iters += 1;
        }

        // Validation
        if (iter + 1) % cfg.val_interval == 0 || iter + 1 == cfg.iters {
            let mut test_psnr_sum = 0.0f32;
            let mut first_test_rendered: Option<RgbImage> = None;

            for (i, &test_idx) in test_indices_for_metrics.iter().enumerate() {
                let (test_camera, test_target_linear) = if let Some(v) = view_cache.get(&test_idx) {
                    (v.camera.clone(), v.target_linear.clone())
                } else {
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
                    (test_camera, test_target_linear)
                };

                let rendered = render(&gaussians, &test_camera, &bg);
                let psnr = compute_psnr(&rendered, &test_target_linear);
                test_psnr_sum += psnr;

                // Capture first test view for incremental rendering
                if i == 0 {
                    first_test_rendered = Some(linear_vec_to_rgb8_img(
                        &rendered,
                        test_camera.width,
                        test_camera.height,
                    ));
                }
            }
            let avg_test_psnr = test_psnr_sum / (test_indices_for_metrics.len() as f32);

            // Save incremental test view every 100 iterations
            if (iter + 1) % 100 == 0 && first_test_rendered.is_some() {
                let output_path = PathBuf::from(format!("test_output/m8_test_view_rendered_{:04}.png", iter + 1));
                if let Some(parent) = output_path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                first_test_rendered.as_ref().unwrap().save(&output_path)
                    .unwrap_or_else(|e| eprintln!("Warning: Failed to save incremental test view: {}", e));
            }

            // Track scene drift by computing gaussian center and bounds
            let scene_center = {
                let mut sum = nalgebra::Vector3::zeros();
                for g in gaussians.iter() {
                    sum += g.position;
                }
                sum / (gaussians.len() as f32)
            };
            let (scene_min, scene_max) = {
                let mut min = nalgebra::Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
                let mut max = nalgebra::Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
                for g in gaussians.iter() {
                    min.x = min.x.min(g.position.x);
                    min.y = min.y.min(g.position.y);
                    min.z = min.z.min(g.position.z);
                    max.x = max.x.max(g.position.x);
                    max.y = max.y.max(g.position.y);
                    max.z = max.z.max(g.position.z);
                }
                (min, max)
            };

            eprintln!(
                "iter {}/{}  train_loss={loss:.6}  test_psnr={avg_test_psnr:.2} dB  bg=({:.3},{:.3},{:.3})  center=({:.2},{:.2},{:.2})  bounds=[{:.2},{:.2},{:.2}]->[{:.2},{:.2},{:.2}]",
                iter + 1,
                cfg.iters,
                bg.x, bg.y, bg.z,
                scene_center.x, scene_center.y, scene_center.z,
                scene_min.x, scene_min.y, scene_min.z,
                scene_max.x, scene_max.y, scene_max.z
            );
        } else if should_log {
            let total = iter_start.unwrap().elapsed();
            eprintln!(
                "iter {}/{} done   train_loss={loss:.6}  forward={:.2?} backward={:.2?} step={:.2?} total={:.2?}",
                iter + 1,
                cfg.iters,
                t_forward,
                t_backward,
                t_step,
                total
            );
        }

        // Densify/prune after validation (so reported PSNR reflects the trained state),
        // and never on the last iteration (so new Gaussians get at least one update step).
        if cfg.densify_interval > 0
            && grad_window_iters > 0
            && (iter + 1) % cfg.densify_interval == 0
            && (iter + 1) < cfg.iters
        {
            let before = gaussians.len();
            let stats = densify_and_prune(
                &mut gaussians,
                &mut sh_params,
                &mut opacity_logits,
                &mut positions,
                &mut log_scales,
                &mut rotations,
                &mut grad_accum_pos_norm,
                &mut rng,
                grad_window_iters,
                cfg.densify_max_gaussians,
                cfg.densify_grad_threshold,
                cfg.prune_opacity_threshold,
                cfg.split_sigma_threshold,
            );
            // The parameter arrays have been re-built; any per-index optimizer state is invalid.
            // Reset moments but keep timesteps to avoid bias-correction spikes.
            sh_opt.reset_moments_keep_t(sh_params.len());
            opacity_opt.reset_moments_keep_t(opacity_logits.len());
            position_opt.reset_moments_keep_t(positions.len());
            scale_opt.reset_moments_keep_t(log_scales.len());
            rotation_opt.reset_moments_keep_t(rotations.len());
            grad_window_iters = 0;
            densify_events += 1;
            let outlier_msg = if stats.pruned_outliers > 0 {
                format!(" pruned_outliers={}", stats.pruned_outliers)
            } else {
                String::new()
            };
            eprintln!(
                "densify @iter {}/{}: gaussians {} -> {} (kept={} pruned={}{} split={} cloned={} cap_hit={} grad_p50={:.4} grad_p90={:.4})",
                iter + 1,
                cfg.iters,
                before,
                gaussians.len(),
                stats.kept,
                stats.pruned,
                outlier_msg,
                stats.split,
                stats.cloned,
                stats.cap_hit,
                stats.grad_p50,
                stats.grad_p90
            );
        }
    }

    // Final validation
    let mut final_psnr_sum = 0.0f32;
    let mut test_view_sample = None;
    let mut test_view_target = None;

    for (i, g) in gaussians.iter_mut().enumerate() {
        for k in 0..16 {
            g.sh_coeffs[k][0] = sh_params[i][k].x;
            g.sh_coeffs[k][1] = sh_params[i][k].y;
            g.sh_coeffs[k][2] = sh_params[i][k].z;
        }
        g.opacity = opacity_logits[i].clamp(-10.0, 10.0);
        g.position = positions[i];
        g.scale = log_scales[i];
        g.rotation = rotations[i];
    }

    for (i, &test_idx) in test_indices_for_metrics.iter().enumerate() {
        let (test_camera, test_target_ds, test_target_linear) = if let Some(v) = view_cache.get(&test_idx) {
            (v.camera.clone(), v.target_ds.clone(), v.target_linear.clone())
        } else {
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
            (test_camera, test_target_ds, test_target_linear)
        };

        let rendered = render(&gaussians, &test_camera, &bg);
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
        initial_num_gaussians,
        final_num_gaussians: gaussians.len(),
        densify_events,
        test_view_sample: test_view_sample.unwrap(),
        test_view_target: test_view_target.unwrap(),
        gaussians,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{UnitQuaternion, Vector3};
    use rand::{SeedableRng, rngs::StdRng};

    fn empty_sh() -> [[f32; 3]; 16] {
        [[0.0; 3]; 16]
    }

    #[test]
    fn densify_prunes_and_splits() {
        let g1 = Gaussian::new(
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.1f32.ln(), 0.1f32.ln(), 0.1f32.ln()),
            UnitQuaternion::identity(),
            2.0, // sigmoid ~ 0.88
            empty_sh(),
        );
        let g2 = Gaussian::new(
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(0.1f32.ln(), 0.1f32.ln(), 0.1f32.ln()),
            UnitQuaternion::identity(),
            -10.0, // sigmoid ~ 0.000045 => pruned
            empty_sh(),
        );

        let mut gaussians = vec![g1.clone(), g2];
        let mut sh_params = vec![
            [Vector3::new(0.1, 0.2, 0.3); 16],
            [Vector3::new(0.4, 0.5, 0.6); 16],
        ];
        let mut opacity_logits = vec![2.0, -10.0];
        let mut positions = vec![g1.position, Vector3::new(1.0, 0.0, 1.0)];
        let mut log_scales: Vec<Vector3<f32>> = gaussians.iter().map(|g| g.scale).collect();
        let mut rotations: Vec<UnitQuaternion<f32>> = gaussians.iter().map(|g| g.rotation).collect();
        let mut grad_accum = vec![2.0, 0.0]; // avg_grad=0.2 if window=10

        let mut rng = StdRng::seed_from_u64(123);
        let stats = densify_and_prune(
            &mut gaussians,
            &mut sh_params,
            &mut opacity_logits,
            &mut positions,
            &mut log_scales,
            &mut rotations,
            &mut grad_accum,
            &mut rng,
            10,
            10,
            0.1,
            0.01,
            0.05,
        );

        // g2 pruned, and g1 split -> two gaussians remain.
        assert_eq!(gaussians.len(), 2);
        assert_eq!(sh_params.len(), 2);
        assert_eq!(opacity_logits.len(), 2);
        assert_eq!(positions.len(), 2);
        assert_eq!(log_scales.len(), 2);
        assert_eq!(rotations.len(), 2);
        assert_eq!(grad_accum, vec![0.0, 0.0]);

        // First is the original, second is the split copy.
        assert_eq!(gaussians[0].position, g1.position);
        assert_ne!(gaussians[1].position, g1.position);
        assert!(gaussians[1].scale.x < g1.scale.x);
        assert!(gaussians[1].scale.y < g1.scale.y);
        assert!(gaussians[1].scale.z < g1.scale.z);

        // Opacity is split across children to preserve total alpha.
        let expected_child_logit = split_opacity_logit(2.0, 2).clamp(-10.0, 10.0);
        assert_relative_eq!(gaussians[0].opacity, expected_child_logit, epsilon = 1e-6);
        assert_relative_eq!(gaussians[1].opacity, expected_child_logit, epsilon = 1e-6);
        assert_relative_eq!(opacity_logits[0], expected_child_logit, epsilon = 1e-6);
        assert_relative_eq!(opacity_logits[1], expected_child_logit, epsilon = 1e-6);

        let a_parent = sigmoid(2.0);
        let a_child = sigmoid(expected_child_logit);
        let a_total = 1.0 - (1.0 - a_child).powi(2);
        assert_relative_eq!(a_total, a_parent, epsilon = 1e-5);

        assert_eq!(stats.before, 2);
        assert_eq!(stats.after, 2);
        assert_eq!(stats.kept, 1);
        assert_eq!(stats.pruned, 1);
        assert_eq!(stats.split, 1);
        assert_eq!(stats.cloned, 0);
        assert!(!stats.cap_hit);
        assert_relative_eq!(stats.grad_p50, 0.2, epsilon = 1e-6);
        assert_relative_eq!(stats.grad_p90, 0.2, epsilon = 1e-6);
    }

    #[test]
    fn test_compute_psnr_empty_vectors() {
        // Empty vectors should return 0.0
        let empty_rendered: Vec<Vector3<f32>> = vec![];
        let empty_target: Vec<Vector3<f32>> = vec![];

        let psnr = compute_psnr(&empty_rendered, &empty_target);
        assert_eq!(psnr, 0.0, "PSNR for empty vectors should be 0.0");
    }

    #[test]
    fn test_compute_psnr_mismatched_lengths() {
        // Mismatched lengths should return 0.0
        let rendered = vec![Vector3::new(0.5, 0.5, 0.5)];
        let target = vec![Vector3::new(0.5, 0.5, 0.5), Vector3::new(0.6, 0.6, 0.6)];

        let psnr = compute_psnr(&rendered, &target);
        assert_eq!(psnr, 0.0, "PSNR for mismatched lengths should be 0.0");
    }

    #[test]
    fn test_compute_psnr_perfect_match() {
        // Perfect match should be capped at 100.0 dB
        let rendered = vec![
            Vector3::new(0.5, 0.6, 0.7),
            Vector3::new(0.1, 0.2, 0.3),
        ];
        let target = rendered.clone();

        let psnr = compute_psnr(&rendered, &target);
        assert_eq!(psnr, 100.0, "PSNR for perfect match should be capped at 100.0");
    }

    #[test]
    fn test_compute_psnr_normal_case() {
        // Test normal case with known MSE
        let rendered = vec![Vector3::new(0.5, 0.5, 0.5)];
        let target = vec![Vector3::new(0.6, 0.6, 0.6)];

        // MSE = ((0.1^2 + 0.1^2 + 0.1^2)) / 3 = 0.03 / 3 = 0.01
        // PSNR = 10 * log10(1.0 / 0.01) = 10 * log10(100) = 20.0
        let psnr = compute_psnr(&rendered, &target);
        assert_relative_eq!(psnr, 20.0, epsilon = 1e-5);
    }
}
