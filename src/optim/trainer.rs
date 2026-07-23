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
    coverage_mask_bool, debug_contrib_count, debug_coverage_mask,
    debug_final_transmittance, debug_overlay_means, downsample_rgb_bilinear, downsample_rgb_box,
    linear_vec_to_rgb8_img, render_full_color_grads, render_full_color_grads_ext,
    render_full_linear, rgb8_to_linear_vec,
};
use image::RgbImage;
use nalgebra::{Matrix3, Vector2, Vector3};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::io::Write;

/// CSV logger for training metrics
struct CsvLogger {
    file: std::fs::File,
}

impl CsvLogger {
    fn new(path: &Path) -> std::io::Result<Self> {
        let mut file = std::fs::File::create(path)?;
        writeln!(
            file,
            "iteration,loss,psnr,num_gaussians,forward_ms,backward_ms,step_ms,total_ms,densify_split,densify_clone,densify_prune,grad_p50,grad_p90,bg_r,bg_g,bg_b,scale_median,aniso_median,aniso_p90,aniso_max,opacity_median,opacity_low_pct,pos_grad_median,scale_grad_median,rot_grad_median,eval_ssim"
        )?;
        // Make the header visible immediately even if the process runs for a long time
        // before the first row is written (e.g., when `val_interval` is large).
        file.flush()?;
        Ok(CsvLogger { file })
    }

    fn log_iteration(
        &mut self,
        iter: usize,
        loss: f32,
        psnr: f32,
        num_gaussians: usize,
        forward_ms: f32,
        backward_ms: f32,
        step_ms: f32,
        total_ms: f32,
        densify_split: usize,
        densify_clone: usize,
        densify_prune: usize,
        grad_p50: f32,
        grad_p90: f32,
        bg: &Vector3<f32>,
        stats: &GaussianStats,
        // Multi-view eval SSIM for real eval rows; -1.0 on proxy/train-psnr rows (SSIM is
        // only computed when the full test-view eval runs).
        eval_ssim: f32,
    ) -> std::io::Result<()> {
        writeln!(
            self.file,
            "{},{:.6},{:.2},{},{:.2},{:.2},{:.2},{:.2},{},{},{},{:.4},{:.4},{:.6},{:.6},{:.6},{:.4},{:.2},{:.2},{:.2},{:.3},{:.1},{:.6},{:.6},{:.6},{:.4}",
            iter + 1,
            loss,
            psnr,
            num_gaussians,
            forward_ms,
            backward_ms,
            step_ms,
            total_ms,
            densify_split,
            densify_clone,
            densify_prune,
            grad_p50,
            grad_p90,
            bg.x,
            bg.y,
            bg.z,
            stats.scale_median,
            stats.aniso_median,
            stats.aniso_p90,
            stats.aniso_max,
            stats.opacity_median,
            stats.opacity_low_pct,
            stats.pos_grad_median,
            stats.scale_grad_median,
            stats.rot_grad_median,
            eval_ssim,
        )?;
        self.file.flush()
    }
}

/// Statistics about Gaussian health for monitoring training
#[derive(Default, Clone, Debug)]
pub struct GaussianStats {
    /// Median scale (geometric mean of linear x,y,z)
    pub scale_median: f32,
    /// Median anisotropy (max/min scale ratio)
    pub aniso_median: f32,
    /// 90th percentile anisotropy
    pub aniso_p90: f32,
    /// Maximum anisotropy
    pub aniso_max: f32,
    /// Median opacity (after sigmoid)
    pub opacity_median: f32,
    /// Percentage of Gaussians with opacity < 0.1
    pub opacity_low_pct: f32,
    /// Median position gradient magnitude
    pub pos_grad_median: f32,
    /// Median scale gradient magnitude
    pub scale_grad_median: f32,
    /// Median rotation gradient magnitude
    pub rot_grad_median: f32,
}

impl GaussianStats {
    /// Compute statistics from current Gaussian state
    pub fn compute(
        log_scales: &[Vector3<f32>],
        opacity_logits: &[f32],
        d_positions: Option<&[Vector3<f32>]>,
        d_log_scales: Option<&[Vector3<f32>]>,
        d_rot_vecs: Option<&[Vector3<f32>]>,
    ) -> Self {
        let n = log_scales.len();
        if n == 0 {
            return Self::default();
        }

        // Compute scale statistics
        let mut scales_linear: Vec<f32> = log_scales
            .iter()
            .map(|s| ((s.x + s.y + s.z) / 3.0).exp()) // Geometric mean
            .collect();
        scales_linear.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute anisotropy for each Gaussian
        let mut anisotropies: Vec<f32> = log_scales
            .iter()
            .map(|s| {
                let max_s = s.x.max(s.y).max(s.z);
                let min_s = s.x.min(s.y).min(s.z);
                (max_s - min_s).exp() // Ratio in linear space
            })
            .collect();
        anisotropies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute opacity statistics
        let mut opacities: Vec<f32> = opacity_logits.iter().map(|&o| sigmoid(o)).collect();
        opacities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let low_opacity_count = opacities.iter().filter(|&&o| o < 0.1).count();

        // Compute gradient statistics
        let pos_grad_median = if let Some(d_pos) = d_positions {
            let mut norms: Vec<f32> = d_pos.iter().map(|g| g.norm()).collect();
            norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            norms.get(n / 2).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        let scale_grad_median = if let Some(d_scale) = d_log_scales {
            let mut norms: Vec<f32> = d_scale.iter().map(|g| g.norm()).collect();
            norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            norms.get(n / 2).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        let rot_grad_median = if let Some(d_rot) = d_rot_vecs {
            let mut norms: Vec<f32> = d_rot.iter().map(|g| g.norm()).collect();
            norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            norms.get(n / 2).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        Self {
            scale_median: scales_linear.get(n / 2).copied().unwrap_or(0.0),
            aniso_median: anisotropies.get(n / 2).copied().unwrap_or(1.0),
            aniso_p90: anisotropies.get(n * 90 / 100).copied().unwrap_or(1.0),
            aniso_max: anisotropies.last().copied().unwrap_or(1.0),
            opacity_median: opacities.get(n / 2).copied().unwrap_or(0.5),
            opacity_low_pct: 100.0 * low_opacity_count as f32 / n as f32,
            pos_grad_median,
            scale_grad_median,
            rot_grad_median,
        }
    }
}

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
    /// D3: SH rest bands (1..16) train at `lr_sh / lr_sh_rest_div`; DC keeps `lr_sh`.
    /// Reference 3DGS uses 20. `1.0` = uniform across bands (legacy behavior).
    pub lr_sh_rest_div: f32,
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
    /// Optional CSV output path for metrics logging.
    pub csv_output_path: Option<PathBuf>,
    /// Disable SH: treat sh_coeffs[0] as RGB color directly, ignore higher bands.
    pub disable_sh: bool,
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
    pub seed_used: u64,
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

    // Try the exact path first
    if path.exists() {
        let img = crate::io::load_image_to_srgb(&path)
            .map_err(|e| anyhow::anyhow!("Failed to load image with color conversion: {}", e))?;
        return Ok(img);
    }

    // Try alternate extensions (COLMAP might have .jpg but actual files are .png or vice versa)
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(name);
    let alternate_exts = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"];

    for ext in &alternate_exts {
        let alt_path = images_dir.join(format!("{}.{}", stem, ext));
        if alt_path.exists() {
            let img = crate::io::load_image_to_srgb(&alt_path)
                .map_err(|e| anyhow::anyhow!("Failed to load image with color conversion: {}", e))?;
            return Ok(img);
        }
    }

    // No matching file found
    Err(anyhow::anyhow!(
        "Failed to open image: {} (also tried alternate extensions)",
        path.display()
    ))
}

/// Downsample an image intelligently based on the downsample factor.
///
/// For power-of-2 factors (1/2, 1/4, 1/8, etc.), uses box filter for clean anti-aliasing.
/// For fractional factors, falls back to nearest-neighbor.
/// For factor=1.0, returns a clone (no downsampling needed).
fn downsample_image_smart(img: &RgbImage, target_width: u32, target_height: u32, factor: f32) -> RgbImage {
    const EPSILON: f32 = 0.001;

    // No downsampling needed - return clone
    if (factor - 1.0).abs() < EPSILON && img.width() == target_width && img.height() == target_height {
        return img.clone();
    }

    // Check if this is a power-of-2 downsample
    for divisor in [2u32, 4, 8, 16, 32, 64] {
        let expected_factor = 1.0 / (divisor as f32);
        if (factor - expected_factor).abs() < EPSILON {
            // Power-of-2: use box filter
            return downsample_rgb_box(img, divisor);
        }
    }

    // Fractional: use bilinear interpolation (better quality than nearest-neighbor)
    downsample_rgb_bilinear(img, target_width, target_height)
}

pub fn train_single_image_color_only(cfg: &TrainConfig) -> anyhow::Result<TrainOutputs> {
    // When disable_sh is true, force learn_sh to false (DC-only mode)
    let learn_sh = if cfg.disable_sh {
        if cfg.learn_sh {
            eprintln!("Warning: --disable-sh forces learn_sh=false (DC-only mode)");
        }
        false
    } else {
        cfg.learn_sh
    };

    // Generate or use provided seed
    let actual_seed = cfg.rng_seed.unwrap_or_else(|| {
        use std::time::SystemTime;
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });
    eprintln!("Using seed: {}", actual_seed);

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

    // Load target image first to check if dimensions match COLMAP
    let target_full = load_target_image(&cfg.images_dir, &image_info.name)?;

    // If actual image size differs from COLMAP camera (e.g., images were pre-resized),
    // update camera to match actual image, then apply downsample factor
    let camera = if target_full.width() != camera_full.width || target_full.height() != camera_full.height {
        let scale_x = target_full.width() as f32 / camera_full.width as f32;
        let scale_y = target_full.height() as f32 / camera_full.height as f32;
        let adjusted = Camera::new(
            camera_full.fx * scale_x,
            camera_full.fy * scale_y,
            camera_full.cx * scale_x,
            camera_full.cy * scale_y,
            target_full.width(),
            target_full.height(),
            camera_full.rotation,
            camera_full.translation,
        );
        downsample_camera(&adjusted, cfg.downsample_factor)
    } else {
        downsample_camera(&camera_full, cfg.downsample_factor)
    };

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
        match GpuRenderer::new() {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("GPU renderer unavailable, falling back to CPU: {e}");
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "gpu"))]
    let gpu_renderer: Option<()> = None;

    #[cfg(not(feature = "gpu"))]
    if cfg.use_gpu {
        return Err(anyhow::anyhow!("GPU rendering requested but not compiled with --features gpu"));
    }

    // C1: density-adaptive initial scale from 3-nearest-neighbor distances (reference 3DGS),
    // replacing the constant-pixel-footprint depth heuristic. σ clamp range matches the old one.
    crate::core::apply_knn_init_scales(&mut gaussians, 1e-4, 1.0);

    let target_ds = downsample_image_smart(&target_full, camera.width, camera.height, cfg.downsample_factor);
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
    // IMPORTANT: Background pixels need sufficient weight (0.5) to prevent background collapse.
    // 0.1 was too small and caused background gradients to vanish as coverage increased.
    let weights: Vec<f32> = coverage_bool
        .iter()
        .map(|&c| if c { 1.0 } else { 0.5 })
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
    let mut sh_opt = AdamSh16::new(cfg.lr_sh, cfg.lr_sh_rest_div, 0.9, 0.999, 1e-8);
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

    // Capture disable_sh from config for use in closures
    let disable_sh = cfg.disable_sh;

    // Conditional render function: GPU if available, otherwise CPU
    #[cfg(feature = "gpu")]
    let render = |gaussians: &[Gaussian], camera: &Camera, bg: &Vector3<f32>| {
        if let Some(ref renderer) = gpu_renderer {
            match renderer.render(gaussians, camera, bg) {
                Ok(img) => img,
                Err(e) => {
                    eprintln!("GPU render failed, falling back to CPU: {e}");
                    render_full_linear(gaussians, camera, bg, disable_sh)
                }
            }
        } else {
            render_full_linear(gaussians, camera, bg, disable_sh)
        }
    };

    #[cfg(not(feature = "gpu"))]
    let render = |gaussians: &[Gaussian], camera: &Camera, bg: &Vector3<f32>| {
        render_full_linear(gaussians, camera, bg, disable_sh)
    };

    // Initialize CSV logger if requested
    let mut csv_logger = if let Some(ref csv_path) = cfg.csv_output_path {
        match CsvLogger::new(csv_path) {
            Ok(logger) => {
                eprintln!("CSV logging enabled: {:?}", csv_path);
                Some(logger)
            }
            Err(e) => {
                eprintln!("Warning: Failed to create CSV logger: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Initial render for output.
    let initial_render_u8 = linear_vec_to_rgb8_img(
        &render(&gaussians, &camera, &bg),
        camera.width,
        camera.height,
    );

    #[cfg(feature = "gpu")]
    let mut gpu_backward_disabled_reason: Option<String> = None;
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
        let psnr = compute_psnr(&rendered_linear, &target_linear);

        // Backward: get dL/d(color_i) and dL/d(opacity_logit_i) per Gaussian.
        let t1 = Instant::now();
        let (_img_u8, d_color, d_opacity_logits, d_positions, d_log_scales, d_rot_vecs, d_bg) = {
            #[cfg(feature = "gpu")]
            if let Some(ref renderer) = gpu_renderer {
                // Use GPU backward pass with sparse atomic gradients (efficient for <10k Gaussians)
                if let Some(reason) = &gpu_backward_disabled_reason {
                    if iter == 0 {
                        eprintln!("GPU backward disabled, using CPU backward: {reason}");
                    }
                    render_full_color_grads(&gaussians, &camera, &d_image, &bg, disable_sh)
                } else {
                    match renderer.render_with_gradients(&gaussians, &camera, &bg, &d_image) {
                    Ok((_pixels, grads_2d)) => {
                    // TEMPORARY: CPU projection backward is default due to GPU shader bug
                    // Use SUGAR_GPU_GRADIENTS=1 to enable GPU projection backward (experimental)
                    let use_gpu_projection = std::env::var("SUGAR_GPU_GRADIENTS").is_ok();

                    let (d_pos, d_scales, d_rots, d_background) = if use_gpu_projection {
                        // GPU projection backward (experimental - has bugs with full parameter training)
                        let (d_positions, d_log_scales, d_rotations, _d_sh) =
                            renderer.project_gradients_backward(&gaussians, &camera, &grads_2d);
                        (d_positions, d_log_scales, d_rotations, grads_2d.d_background)
                    } else {
                        // CPU projection backward (default, stable)
                        crate::gpu::chain_2d_to_3d_gradients_cpu(&grads_2d, &gaussians, &camera)
                    };

                    let dummy_img = image::RgbImage::new(camera.width, camera.height);
                    (dummy_img, grads_2d.d_colors, grads_2d.d_opacity_logits, d_pos, d_scales, d_rots, d_background)
                    }
                    Err(e) => {
                        eprintln!("GPU backward failed, falling back to CPU: {e}");
                        gpu_backward_disabled_reason = Some(e);
                        render_full_color_grads(&gaussians, &camera, &d_image, &bg, disable_sh)
                    }
                }
                }
            } else {
                // CPU backward pass
                render_full_color_grads(&gaussians, &camera, &d_image, &bg, disable_sh)
            }

            #[cfg(not(feature = "gpu"))]
            render_full_color_grads(&gaussians, &camera, &d_image, &bg, disable_sh)
        };
        let t_backward = t1.elapsed();

        // Convert dL/d(color) -> dL/d(SH coeffs) using per-Gaussian SH basis.
        let mut d_sh: Vec<[Vector3<f32>; 16]> = vec![[Vector3::zeros(); 16]; gaussians.len()];
        if learn_sh {
            for (i, g) in gaussians.iter().enumerate() {
                let basis = crate::core::sh_basis(&camera.view_direction(&g.position));
                for k in 0..16 {
                    d_sh[i][k] = d_color[i] * basis[k];
                }
            }
        } else {
            // DC-only learning (k=0). Uses SH_C0 constant.
            let sh0 = crate::core::SH_C0;
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
            // Clip scale gradients to prevent extreme updates
            const SCALE_GRAD_CLIP: f32 = 1.0;
            let d_log_scales_clipped: Vec<Vector3<f32>> = d_log_scales
                .iter()
                .map(|g| Vector3::new(
                    g.x.clamp(-SCALE_GRAD_CLIP, SCALE_GRAD_CLIP),
                    g.y.clamp(-SCALE_GRAD_CLIP, SCALE_GRAD_CLIP),
                    g.z.clamp(-SCALE_GRAD_CLIP, SCALE_GRAD_CLIP),
                ))
                .collect();
            scale_opt.step(&mut log_scales, &d_log_scales_clipped);

            // Clamp log-space scales to prevent degenerate ellipsoids
            // exp(5) ≈ 148 (large but reasonable), exp(-10) ≈ 4.5e-5 (tiny but valid)
            // Tighter bounds prevent stretched/exploding Gaussians during training
            const MAX_LOG_SCALE: f32 = 5.0;
            const MIN_LOG_SCALE: f32 = -10.0;
            // Max ratio between largest and smallest scale axis (prevents needles)
            // ln(5) ≈ 1.6, so max 5:1 aspect ratio (tightened from 10:1 for sharper results)
            const MAX_LOG_ANISOTROPY: f32 = 1.6;
            for scale in log_scales.iter_mut() {
                // First clamp individual axes
                scale.x = scale.x.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);
                scale.y = scale.y.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);
                scale.z = scale.z.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);

                // Then enforce anisotropy constraint: pull smaller axes toward largest
                let max_s = scale.x.max(scale.y).max(scale.z);
                let min_allowed = max_s - MAX_LOG_ANISOTROPY;
                scale.x = scale.x.max(min_allowed);
                scale.y = scale.y.max(min_allowed);
                scale.z = scale.z.max(min_allowed);
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
            // Clamp background to valid RGB range [0, 1]
            bg.x = bg.x.clamp(0.0, 1.0);
            bg.y = bg.y.clamp(0.0, 1.0);
            bg.z = bg.z.clamp(0.0, 1.0);
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

            // Log to CSV if enabled
            if let Some(ref mut csv) = csv_logger {
                let stats = GaussianStats::compute(
                    &log_scales,
                    &opacity_logits,
                    Some(&d_positions),
                    Some(&d_log_scales),
                    Some(&d_rot_vecs),
                );
                if let Err(e) = csv.log_iteration(
                    iter,
                    loss,
                    psnr,
                    gaussians.len(),
                    t_forward.as_secs_f32() * 1000.0,
                    t_backward.as_secs_f32() * 1000.0,
                    t_step.as_secs_f32() * 1000.0,
                    total.as_secs_f32() * 1000.0,
                    0, // densify_split (M7 doesn't have densification)
                    0, // densify_clone
                    0, // densify_prune
                    0.0, // grad_p50
                    0.0, // grad_p90
                    &bg,
                    &stats,
                    -1.0, // eval_ssim: single-view path, no multi-view eval
                ) {
                    eprintln!("Warning: Failed to write CSV row: {}", e);
                }
            }
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
        seed_used: actual_seed,
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

/// B3 + window-margin gate: should an opacity reset fire at 1-based iteration `iter1`?
/// Resets fire every `interval` iterations inside the densify window, but never in the
/// last `margin` iterations before `window_end` (so the population enters settle having
/// re-earned opacity, with densification still active to restructure).
pub fn opacity_reset_due(iter1: usize, interval: usize, window_end: usize, margin: usize) -> bool {
    interval > 0 && iter1 % interval == 0 && iter1 + margin <= window_end
}

/// C2 SH warmup: how many SH coefficients are active at 0-based iteration `iter`?
/// Reference 3DGS starts DC-only and raises the active degree by one every
/// `oneupSHdegree` interval (1000 iters) up to degree 3; degree d uses (d+1)^2
/// coefficients, so the unlock sequence is 1 → 4 → 9 → 16. `interval == 0` disables
/// the warmup (all 16 active from the start, the pre-C2 behavior).
pub fn active_sh_coeffs(warmup_interval: usize, iter: usize) -> usize {
    if warmup_interval == 0 {
        return 16;
    }
    let degree = (iter / warmup_interval).min(3);
    (degree + 1) * (degree + 1)
}

/// Render-watchdog detector: is this frame essentially dead — pure background, or a
/// constant color (e.g. all zeros from a failed pipeline)? Samples every 7th pixel and
/// reports true when >99% of samples sit within ~1.5/255 of the background constant OR of
/// the first sampled pixel. A working renderer never produces this on a real training view
/// (past init the model always composites varied content); a silently failed GPU pipeline
/// does — see the 2026-07-10 full-res collapse, where frames went background-only with no
/// wgpu error surfacing.
pub fn frame_is_background_only(img: &[Vector3<f32>], bg: &Vector3<f32>) -> bool {
    const EPS: f32 = 1.5 / 255.0;
    if img.is_empty() {
        return true;
    }
    let near = |a: &Vector3<f32>, b: &Vector3<f32>| {
        let d = a - b;
        d.x.abs() < EPS && d.y.abs() < EPS && d.z.abs() < EPS
    };
    let first = img[0];
    let mut bg_px = 0usize;
    let mut const_px = 0usize;
    let mut total = 0usize;
    let mut i = 0;
    while i < img.len() {
        if near(&img[i], bg) {
            bg_px += 1;
        }
        if near(&img[i], &first) {
            const_px += 1;
        }
        total += 1;
        i += 7;
    }
    let limit = 0.99 * (total as f32);
    (bg_px as f32) > limit || (const_px as f32) > limit
}

/// Deterministic interval train/test split matching the nerfstudio/gsplat/MipNeRF360
/// convention (`--eval-mode interval`): indices are ordered by image filename and
/// every Nth position (position % interval == 0, so the lexicographically first image
/// is held out) becomes a test view. Returns (train, test) as indices into `names`,
/// which lets the caller keep its own image ordering (COLMAP `images.bin` is in
/// registration order, not filename order).
pub fn interval_split_by_name(names: &[&str], interval: usize) -> (Vec<usize>, Vec<usize>) {
    let mut by_name: Vec<usize> = (0..names.len()).collect();
    by_name.sort_by(|&a, &b| names[a].cmp(names[b]));
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (pos, &idx) in by_name.iter().enumerate() {
        if pos % interval == 0 {
            test.push(idx);
        } else {
            train.push(idx);
        }
    }
    (train, test)
}

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
    /// D3: SH rest bands (1..16) train at `lr_sh / lr_sh_rest_div`; DC keeps `lr_sh`.
    /// Reference 3DGS uses 20. `1.0` = uniform across bands (legacy behavior).
    pub lr_sh_rest_div: f32,
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
    /// If non-zero, replace the seeded-shuffle split with the deterministic interval
    /// split used by nerfstudio/gsplat/MipNeRF360 (`--eval-mode interval`): images are
    /// sorted by filename and every Nth position (position % N == 0, so the
    /// lexicographically first image is held out) becomes a test view. Ignores
    /// `train_fraction` for the split; `rng_seed` still drives train-view sampling.
    pub eval_interval: usize,
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
    /// Per-step anisotropy clamp: every iteration, pull each Gaussian's smaller log-scale axes
    /// up to within this log-ratio of the largest axis. `0` disables (reference 3DGS has no such
    /// clamp — thin surface-aligned splats routinely reach 10–100× anisotropy). Legacy value 1.6
    /// (≈5:1) predates the EWA low-pass filter, which is the correct anti-needle defense.
    pub max_log_anisotropy: f32,
    /// Prune Gaussians whose log-scale anisotropy (max−min axis) exceeds this at densify time.
    /// `0` disables. Legacy value 2.0 (≈7:1), companion to `max_log_anisotropy`.
    pub needle_prune_log_anisotropy: f32,
    /// Reset (cap-down) opacities every N iterations; 0 disables. Reference default: 3000.
    pub opacity_reset_interval: usize,
    /// Opacity resets cap logits down to this sigmoid-space value. Reference: 0.01 — which sits
    /// ABOVE the usual prune threshold (0.005), so mass that never re-earns opacity parks at the
    /// floor unprunable. Setting this BELOW `prune_opacity_threshold` lets the next prune pass
    /// remove never-recovering Gaussians instead.
    pub opacity_reset_floor: f32,
    /// Skip opacity resets in the last N iterations of the densify window (0 = reference
    /// behavior, resets fire right up to the window end). With the reference schedule the LAST
    /// reset lands AT the window end (e.g. iter 15000 of a 30k run), so the population enters
    /// the settle phase freshly floored with no densification left to restructure — the 30k
    /// control's settle flatlined 0.75 dB below peak. A margin lets the population re-earn
    /// opacity (with densification still active) before the window closes.
    pub opacity_reset_window_margin: usize,
    /// Settle-phase prune: after densification stops (iter > iters/2), run a prune-only pass
    /// every N iterations (0 disables). Applies the same prune rules as densify-time
    /// (sub-threshold opacity, needles, oversize, outliers) but never splits/clones — without
    /// it the population is frozen for the entire settle phase and pathological Gaussians that
    /// survive the densify window can never be removed.
    pub settle_prune_interval: usize,
    /// Settle-decay hunt: freeze ALL spherical-harmonics updates (DC + rest bands) once the
    /// densify window closes (iter > iters/2). Isolates whether continued SH optimization during
    /// the settle phase drives the universal peak-then-decay: every arm peaks mid-densify-window
    /// and settles ~1 dB below. D3 (`--sh-rest-lr-div`) ruled out SH-*rest* LR as the driver; this
    /// is the stronger test (freezes DC too, and freezes rather than merely slowing).
    pub freeze_sh_after_window: bool,
    /// Settle-decay hunt: freeze the learnable background color once the densify window closes
    /// (iter > iters/2). bg→black mid-settle is a universal co-symptom of the decay; freezing bg
    /// at its window-close value tests whether the drifting background drives the PSNR decline.
    pub freeze_bg_in_settle: bool,
    /// C2 SH warmup: raise the active SH degree by one every this many iterations (reference
    /// 3DGS `oneupSHdegree`, 1000), starting DC-only; degree d activates (d+1)^2 coefficients
    /// (1 → 4 → 9 → 16). Locked coefficients are skipped by the SH optimizer entirely, which is
    /// state-identical to reference truncated-degree rendering because rest bands init to zero
    /// (zero coefficient ⇒ zero color contribution ⇒ zero gradient). 0 disables (all bands
    /// train from iter 0, the pre-C2 behavior).
    pub sh_warmup_interval: usize,
    /// Needle-prune log-anisotropy threshold used ONLY by the settle-phase prune pass
    /// (0 disables → falls back to `needle_prune_log_anisotropy`). The normal needle threshold
    /// sits ABOVE the `max_log_anisotropy` clamp (clamp+0.4), so needle-prune never fires — the
    /// population parks AT the clamp (aniso_p90 climbs 17→20 over the 30k settle, clamped but
    /// unpruned). A tighter settle threshold (below the clamp) makes the settle prune actually
    /// remove the needling parked mass instead of just reshaping it (tightening the clamp
    /// inflates scale; pruning frees capacity). Densify-time pruning is unchanged. CLI default
    /// 2.8 (2026-07-14 30k A/B: settle decay eliminated, needling decile pruned, renders clean).
    pub settle_needle_prune_log_aniso: f32,
    /// Use GPU for forward rendering.
    pub use_gpu: bool,
    /// Optional CSV output path for metrics logging.
    pub csv_output_path: Option<PathBuf>,
    /// Output directory for incremental renders.
    pub out_dir: PathBuf,
    /// Save an intermediate `model_<step>.gs` every N iterations (0 = only the final
    /// model). Lets a single run produce the whole iteration grid (e.g. 3k/15k/30k) for
    /// equal-iteration comparison instead of just the final model. `<step>` is the
    /// completed-iteration count; the final iteration is written as `model.gs`, not here.
    pub save_interval: usize,
    /// Disable SH: treat sh_coeffs[0] as RGB color directly, ignore higher bands.
    pub disable_sh: bool,
    /// Render watchdog: abort (saving the model for forensics) when a wgpu fault is
    /// recorded or several consecutive train renders come back as pure background. The
    /// 2026-07-10 full-res run lost the GPU rasterizer mid-training with no surfaced
    /// error and spent 3.5k iterations training against background-only frames.
    pub render_watchdog: bool,
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
    pub training_width: u32,
    pub training_height: u32,
    pub downsample_factor: f32,
    pub seed_used: u64, // Actual seed used (for reproducibility)
}

/// Compute PSNR between two linear RGB images.
pub fn compute_psnr(rendered: &[Vector3<f32>], target: &[Vector3<f32>]) -> f32 {
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

/// Standard SSIM (Wang et al.): 11×11 Gaussian window (σ=1.5), C1=0.01², C2=0.03²,
/// computed per RGB channel on the same linear [0,1] values as `compute_psnr` (so the two
/// columns share a domain), averaged over channels and pixels. Separable convolution with
/// clamp-to-edge padding. `rendered`/`target` are row-major width×height.
pub fn compute_ssim(
    rendered: &[Vector3<f32>],
    target: &[Vector3<f32>],
    width: usize,
    height: usize,
) -> f32 {
    if rendered.len() != target.len() || rendered.len() != width * height || rendered.is_empty() {
        return 0.0;
    }
    const RADIUS: i64 = 5; // 11-tap window
    const C1: f32 = 0.01 * 0.01;
    const C2: f32 = 0.03 * 0.03;
    let kernel: Vec<f32> = {
        let sigma = 1.5f32;
        let mut k: Vec<f32> = (-RADIUS..=RADIUS)
            .map(|i| (-(i as f32).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();
        let sum: f32 = k.iter().sum();
        k.iter_mut().for_each(|v| *v /= sum);
        k
    };
    // Gaussian blur with clamp-to-edge, horizontal then vertical.
    let blur = |src: &[f32]| -> Vec<f32> {
        let mut tmp = vec![0.0f32; src.len()];
        for y in 0..height as i64 {
            for x in 0..width as i64 {
                let mut acc = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sx = (x + ki as i64 - RADIUS).clamp(0, width as i64 - 1);
                    acc += kv * src[(y * width as i64 + sx) as usize];
                }
                tmp[(y * width as i64 + x) as usize] = acc;
            }
        }
        let mut out = vec![0.0f32; src.len()];
        for y in 0..height as i64 {
            for x in 0..width as i64 {
                let mut acc = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sy = (y + ki as i64 - RADIUS).clamp(0, height as i64 - 1);
                    acc += kv * tmp[(sy * width as i64 + x) as usize];
                }
                out[(y * width as i64 + x) as usize] = acc;
            }
        }
        out
    };
    let n = rendered.len();
    let mut ssim_sum = 0.0f64;
    for c in 0..3 {
        let x: Vec<f32> = rendered.iter().map(|v| v[c]).collect();
        let y: Vec<f32> = target.iter().map(|v| v[c]).collect();
        let xx: Vec<f32> = x.iter().map(|v| v * v).collect();
        let yy: Vec<f32> = y.iter().map(|v| v * v).collect();
        let xy: Vec<f32> = x.iter().zip(y.iter()).map(|(a, b)| a * b).collect();
        let mu_x = blur(&x);
        let mu_y = blur(&y);
        let m_xx = blur(&xx);
        let m_yy = blur(&yy);
        let m_xy = blur(&xy);
        for i in 0..n {
            let var_x = (m_xx[i] - mu_x[i] * mu_x[i]).max(0.0);
            let var_y = (m_yy[i] - mu_y[i] * mu_y[i]).max(0.0);
            let cov = m_xy[i] - mu_x[i] * mu_y[i];
            let num = (2.0 * mu_x[i] * mu_y[i] + C1) * (2.0 * cov + C2);
            let den = (mu_x[i] * mu_x[i] + mu_y[i] * mu_y[i] + C1) * (var_x + var_y + C2);
            ssim_sum += (num / den) as f64;
        }
    }
    (ssim_sum / (3.0 * n as f64)) as f32
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
    pruned_needles: usize,
    pruned_oversize: usize,
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

fn densify_and_prune<R: Rng + ?Sized>(
    gaussians: &mut Vec<Gaussian>,
    sh_params: &mut Vec<[Vector3<f32>; 16]>,
    opacity_logits: &mut Vec<f32>,
    positions: &mut Vec<Vector3<f32>>,
    log_scales: &mut Vec<Vector3<f32>>,
    rotations: &mut Vec<nalgebra::UnitQuaternion<f32>>,
    grad_accum: &mut Vec<f32>,
    denom: &mut Vec<f32>,
    rng: &mut R,
    iters_in_window: usize,
    max_gaussians: usize,
    grad_threshold: f32,
    prune_opacity_threshold: f32,
    _split_sigma_threshold: f32,
    // Prune Gaussians whose log-scale anisotropy (max−min axis) exceeds this; 0 disables.
    needle_prune_log_aniso: f32,
    scene_extent: f32,
) -> (DensifyStats, Vec<Option<usize>>) {
    let before = gaussians.len();
    if iters_in_window == 0 || gaussians.is_empty() {
        return (
            DensifyStats {
                before,
                after: before,
                kept: before,
                pruned: 0,
                pruned_outliers: 0,
                pruned_needles: 0,
                pruned_oversize: 0,
                split: 0,
                cloned: 0,
                cap_hit: false,
                grad_p50: f32::NAN,
                grad_p90: f32::NAN,
            },
            (0..before).map(Some).collect(),
        );
    }

    // B5: clone vs split boundary is scene-relative (percent_dense · scene_extent), matching
    // reference 3DGS, so it adapts to the dataset's spatial scale instead of a fixed constant.
    const PERCENT_DENSE: f32 = 0.01;
    let split_size_threshold = PERCENT_DENSE * scene_extent;
    // B6: prune Gaussians whose world-space size exceeds this fraction of the scene extent.
    let oversize_threshold = 0.1 * scene_extent;

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
    // Never drop existing Gaussians just because the post-densify cap is smaller than current
    // size — the cap limits *additions*. Enforce it as an addition BUDGET (cap − current count),
    // decremented per child. The old check compared the rebuilt array's RUNNING length against
    // the cap, but survivors are appended after children stop, so every cycle overshot the cap
    // by the number of parents still unprocessed at the moment the length crossed it — and the
    // next cycle's cap.max(before) ratcheted the cap up to the inflated count (compounding
    // ~5–8%/cycle until the 400k GPU buffer limit).
    let mut add_budget = cap.saturating_sub(before);

    let mut new_gaussians = Vec::with_capacity(gaussians.len());
    let mut new_sh_params = Vec::with_capacity(sh_params.len());
    let mut new_opacity_logits = Vec::with_capacity(opacity_logits.len());
    let mut new_positions = Vec::with_capacity(positions.len());
    let mut new_log_scales = Vec::with_capacity(log_scales.len());
    let mut new_rotations = Vec::with_capacity(rotations.len());
    let mut new_grad_accum = Vec::with_capacity(grad_accum.len());
    let mut new_denom = Vec::with_capacity(denom.len());
    // B11: source-index map for optimizer state — Some(old_i) carries a survivor's Adam
    // moments over, None (children, re-initialized split parents) starts from zero.
    let mut remap: Vec<Option<usize>> = Vec::with_capacity(gaussians.len());

    let mut kept = 0usize;
    let mut pruned = 0usize;
    let mut pruned_outliers = 0usize;
    let mut pruned_needles = 0usize;
    let mut pruned_oversize = 0usize;
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

        // Optionally prune needles: Gaussians with extreme aspect ratios (0 disables)
        let scale = &log_scales[i];
        let max_s = scale.x.max(scale.y).max(scale.z);
        let min_s = scale.x.min(scale.y).min(scale.z);
        if needle_prune_log_aniso > 0.0 && max_s - min_s > needle_prune_log_aniso {
            pruned_needles += 1;
            pruned += 1;
            continue;
        }

        // B6: prune Gaussians grown too large for the scene (world-space size > 0.1·scene_extent).
        if max_s.exp() > oversize_threshold {
            pruned_oversize += 1;
            pruned += 1;
            continue;
        }

        // B2: average the accumulated view-space gradient over the number of iterations this
        // Gaussian was actually visible (its own denom), not over the global window length.
        let avg_grad = grad_accum[i] / denom[i].max(1.0);
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
        new_denom.push(0.0);
        remap.push(Some(i));
        kept += 1;

        let can_add = add_budget > 0;
        if !(avg_grad > grad_threshold && can_add) {
            if avg_grad > grad_threshold && !can_add {
                cap_hit = true;
            }
            continue;
        }
        add_budget -= 1;

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

        // B12: children copy the parent opacity unchanged (reference 3DGS). The old
        // alpha-preserving halving knocked down exactly the high-gradient Gaussians on every
        // densify cycle; the periodic B3 opacity reset is what handles alpha inflation.
        let child_opacity_logit = opacity_logits[i];

        // B5: split when the Gaussian's LARGEST axis exceeds the scene-relative size threshold
        // (over-reconstruction), otherwise clone (under-reconstruction). Uses max axis, not mean,
        // so elongated Gaussians are correctly caught.
        let max_scale_world = log_scales[i].x.max(log_scales[i].y).max(log_scales[i].z).exp();
        if max_scale_world > split_size_threshold {
            // SPLIT: shrink BOTH parent and child so world scale divides by 1.6 (reference /(0.8·N),
            // N=2). In log-space that is subtracting ln(1.6) ≈ 0.470.
            let scale_reduction = Vector3::new(1.6f32.ln(), 1.6f32.ln(), 1.6f32.ln());
            let shrunk_scale = log_scales[i] - scale_reduction;

            // Shrink the parent (kept Gaussian) - CRITICAL FIX
            new_log_scales[keep_idx] = shrunk_scale;
            new_gaussians[keep_idx].scale = shrunk_scale;
            // B11: reference prunes the split source and creates N fresh Gaussians, so the
            // re-initialized (shrunk) parent starts with fresh optimizer state too.
            remap[keep_idx] = None;

            // Create child with same shrunk scale, offset position
            let mut g2 = gaussians[i].clone();
            for k in 0..16 {
                g2.sh_coeffs[k][0] = sh_params[i][k].x;
                g2.sh_coeffs[k][1] = sh_params[i][k].y;
                g2.sh_coeffs[k][2] = sh_params[i][k].z;
            }
            g2.opacity = child_opacity_logit;
            g2.position = new_pos;
            g2.scale = shrunk_scale;
            g2.rotation = rotations[i];
            new_gaussians.push(g2);
            new_sh_params.push(sh_params[i]);
            new_opacity_logits.push(child_opacity_logit);
            new_positions.push(new_pos);
            new_log_scales.push(shrunk_scale);
            new_rotations.push(rotations[i]);
            new_grad_accum.push(0.0);
            new_denom.push(0.0);
            remap.push(None);
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
            new_denom.push(0.0);
            remap.push(None);
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
    *denom = new_denom;

    kept_avg_grads.sort_by(|a, b| a.total_cmp(b));
    let grad_p50 = percentile(&kept_avg_grads, 0.50);
    let grad_p90 = percentile(&kept_avg_grads, 0.90);

    (
        DensifyStats {
            before,
            after: gaussians.len(),
            kept,
            pruned,
            pruned_outliers,
            pruned_needles,
            pruned_oversize,
            split,
            cloned,
            cap_hit,
            grad_p50,
            grad_p90,
        },
        remap,
    )
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
    // When disable_sh is true, force learn_sh to false (DC-only mode)
    let learn_sh = if cfg.disable_sh {
        if cfg.learn_sh {
            eprintln!("Warning: --disable-sh forces learn_sh=false (DC-only mode)");
        }
        false
    } else {
        cfg.learn_sh
    };

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

    // GPU hard cap: Metal on Apple Silicon has 128 MB max buffer size
    // GaussianGPU is ~320 bytes, so theoretical max is ~419K Gaussians
    // We enforce a conservative limit of 400K to stay safely under the buffer limit
    const GPU_HARD_CAP_GAUSSIANS: usize = 400_000;

    #[cfg(feature = "gpu")]
    if cfg.use_gpu && cfg.densify_max_gaussians > GPU_HARD_CAP_GAUSSIANS {
        eprintln!(
            "⚠️  WARNING: densify_max_gaussians ({}) exceeds GPU hard limit ({})",
            cfg.densify_max_gaussians, GPU_HARD_CAP_GAUSSIANS
        );
        eprintln!(
            "⚠️  Metal GPU buffer limit is 128 MB. Setting will be clamped to {} Gaussians.",
            GPU_HARD_CAP_GAUSSIANS
        );
        eprintln!(
            "⚠️  To avoid this warning, set --densify-max-gaussians {} or lower.",
            GPU_HARD_CAP_GAUSSIANS
        );
    }

    // Split images into train/test sets
    let image_indices: Vec<usize> = (0..available_images).collect();

    // Generate seed if not provided (for reproducibility)
    let actual_seed = cfg.rng_seed.unwrap_or_else(|| {
        use std::time::SystemTime;
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });

    let mut rng = StdRng::seed_from_u64(actual_seed);

    let (train_indices, test_indices): (Vec<usize>, Vec<usize>) = if cfg.eval_interval > 0 {
        let names: Vec<&str> = image_indices
            .iter()
            .map(|&i| scene.images[i].name.as_str())
            .collect();
        let (train, test) = interval_split_by_name(&names, cfg.eval_interval);
        let first_test: Vec<&str> = test.iter().take(3).map(|&i| names[i]).collect();
        eprintln!(
            "Split mode: interval:{} over filename order ({} test views, first: {:?})",
            cfg.eval_interval,
            test.len(),
            first_test
        );
        (train, test)
    } else {
        let mut shuffled = image_indices.clone();
        shuffled.shuffle(&mut rng);
        let num_train = ((available_images as f32) * cfg.train_fraction).max(1.0) as usize;
        let test = shuffled.split_off(num_train);
        (shuffled, test)
    };
    let train_indices = &train_indices[..];
    let test_indices = &test_indices[..];

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
    // Always preload images - the cache is needed for training and final output
    {
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

            // Important: Downsample the image independently, then adjust camera to match actual image size
            // This handles cases where COLMAP camera dimensions don't match actual image file dimensions
            let target_ds = if (cfg.downsample_factor - 1.0).abs() < 0.001 {
                target.clone()
            } else {
                // Determine target dimensions from actual image, not camera
                let target_width = ((target.width() as f32) * cfg.downsample_factor).round().max(1.0) as u32;
                let target_height = ((target.height() as f32) * cfg.downsample_factor).round().max(1.0) as u32;
                downsample_image_smart(&target, target_width, target_height, cfg.downsample_factor)
            };

            // Adjust camera to match actual downsampled image dimensions
            let mut camera = camera;
            if camera.width != target_ds.width() || camera.height != target_ds.height() {
                let scale_x = target_ds.width() as f32 / camera.width as f32;
                let scale_y = target_ds.height() as f32 / camera.height as f32;
                camera.width = target_ds.width();
                camera.height = target_ds.height();
                camera.fx *= scale_x;
                camera.fy *= scale_y;
                camera.cx *= scale_x;
                camera.cy *= scale_y;
            }

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

    // B5/B6: scene spatial extent = radius of the camera centers about their centroid.
    // Camera center C = -Rᵀ·t (R is world→camera). Used to make densification thresholds, the
    // oversize prune, and the C1 init-scale cap scene-relative, matching reference 3DGS.
    let scene_extent: f32 = {
        let centers: Vec<Vector3<f32>> = scene
            .images
            .iter()
            .map(|img| {
                let r = crate::core::quaternion_to_matrix(&img.rotation);
                -r.transpose() * img.translation
            })
            .collect();
        if centers.is_empty() {
            1.0
        } else {
            let mean = centers.iter().fold(Vector3::zeros(), |a, c| a + c) / centers.len() as f32;
            centers
                .iter()
                .map(|c| (c - mean).norm())
                .fold(0.0f32, f32::max)
                .max(1e-3)
        }
    };
    eprintln!("scene_extent = {:.4} (from {} cameras)", scene_extent, scene.images.len());

    // Initialize Gaussians from visible points (using first view for now)
    let cloud =
        init_from_colmap_points_visible_stratified(&scene.points, &camera, cfg.max_gaussians, 8);
    let mut gaussians: Vec<Gaussian> = cloud.gaussians;

    // C1: density-adaptive initial scale from 3-nearest-neighbor distances (reference 3DGS),
    // replacing the depth heuristic that made dense regions start too large and over-split.
    // Cap σ at 0.1·scene_extent so no init Gaussian starts beyond the B6 oversize-prune bound.
    crate::core::apply_knn_init_scales(&mut gaussians, 1e-4, 0.1 * scene_extent);

    eprintln!("Initialized {} Gaussians", gaussians.len());
    let initial_num_gaussians = gaussians.len();

    // Initialize GPU renderer if requested
    #[cfg(feature = "gpu")]
    let gpu_renderer = if cfg.use_gpu {
        eprintln!("Initializing GPU renderer...");
        match GpuRenderer::new() {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("GPU renderer unavailable, falling back to CPU: {e}");
                None
            }
        }
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
        let first_target_ds = downsample_image_smart(&first_target, camera.width, camera.height, cfg.downsample_factor);
        rgb8_to_linear_vec(&first_target_ds)
    };

    let mut bg = {
        let mut acc = Vector3::<f32>::zeros();
        for p in &first_target_linear {
            acc += *p;
        }
        acc / (first_target_linear.len() as f32).max(1.0)
    };
    eprintln!(
        "background init = ({:.3},{:.3},{:.3})  learn_background={}",
        bg.x, bg.y, bg.z, cfg.learn_background
    );
    let mut bg_opt = AdamVec3::new(cfg.lr_background, 0.9, 0.999, 1e-8);

    // Optimizer state for SH coeffs (RGB × 16)
    let mut sh_opt = AdamSh16::new(cfg.lr_sh, cfg.lr_sh_rest_div, 0.9, 0.999, 1e-8);
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
    let mut grad_denom: Vec<f32> = vec![0.0; gaussians.len()];
    let mut grad_window_iters: usize = 0;

    // Capture disable_sh from config for use in closures
    let disable_sh = cfg.disable_sh;

    // Conditional render function: GPU if available, otherwise CPU
    #[cfg(feature = "gpu")]
    let render = |gaussians: &[Gaussian], camera: &Camera, bg: &Vector3<f32>| {
        if let Some(ref renderer) = gpu_renderer {
            match renderer.render(gaussians, camera, bg) {
                Ok(img) => img,
                Err(e) => {
                    eprintln!("GPU render failed, falling back to CPU: {e}");
                    render_full_linear(gaussians, camera, bg, disable_sh)
                }
            }
        } else {
            render_full_linear(gaussians, camera, bg, disable_sh)
        }
    };

    #[cfg(not(feature = "gpu"))]
    let render = |gaussians: &[Gaussian], camera: &Camera, bg: &Vector3<f32>| {
        render_full_linear(gaussians, camera, bg, disable_sh)
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

                // Downsample image and adjust camera to match actual image dimensions
                let test_target_ds = if (cfg.downsample_factor - 1.0).abs() < 0.001 {
                    test_target.clone()
                } else {
                    let target_width = ((test_target.width() as f32) * cfg.downsample_factor).round().max(1.0) as u32;
                    let target_height = ((test_target.height() as f32) * cfg.downsample_factor).round().max(1.0) as u32;
                    downsample_image_smart(&test_target, target_width, target_height, cfg.downsample_factor)
                };

                let mut test_camera = test_camera;
                if test_camera.width != test_target_ds.width() || test_camera.height != test_target_ds.height() {
                    let scale_x = test_target_ds.width() as f32 / test_camera.width as f32;
                    let scale_y = test_target_ds.height() as f32 / test_camera.height as f32;
                    test_camera.width = test_target_ds.width();
                    test_camera.height = test_target_ds.height();
                    test_camera.fx *= scale_x;
                    test_camera.fy *= scale_y;
                    test_camera.cx *= scale_x;
                    test_camera.cy *= scale_y;
                }

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

    // Initialize CSV logger if path provided
    let mut csv_logger = if let Some(ref csv_path) = cfg.csv_output_path {
        CsvLogger::new(csv_path).ok()
    } else {
        None
    };

    // D1/D2: reference 3DGS learning-rate schedule.
    // D1: position LR is scaled by the scene extent (reference `spatial_lr_scale`) — a fixed
    //     0.00016 in world units moves positions far too slowly on scenes larger than ~1 unit,
    //     and Gaussians compensate by growing scale instead of relocating onto geometry.
    // D2: ONLY position is scheduled — log-linear (exponential) decay from lr_init·extent down
    //     100× over the reference horizon of 30k steps (reference `get_expon_lr_func`, no
    //     delay). Every other parameter group holds its LR constant for the whole run; the old
    //     uniform 10×-per-run decay starved scale/opacity/SH updates late in training.
    const POSITION_LR_MAX_STEPS: f32 = 30_000.0;
    let position_lr_init = cfg.lr_position * scene_extent;
    let position_lr_final = position_lr_init * 0.01;
    eprintln!(
        "position lr = {:.6} -> {:.8} over {} steps (spatial_lr_scale = {:.3}); other LRs constant",
        position_lr_init, position_lr_final, POSITION_LR_MAX_STEPS as usize, scene_extent
    );
    eprintln!(
        "sh lr = {:.6} (DC), {:.6} (rest bands, div {})",
        cfg.lr_sh,
        cfg.lr_sh / cfg.lr_sh_rest_div,
        cfg.lr_sh_rest_div
    );
    if cfg.freeze_sh_after_window || cfg.freeze_bg_in_settle {
        eprintln!(
            "settle-decay hunt: freeze_sh_after_window = {}, freeze_bg_in_settle = {} (applied for iter > {})",
            cfg.freeze_sh_after_window,
            cfg.freeze_bg_in_settle,
            cfg.iters / 2
        );
    }
    if cfg.settle_needle_prune_log_aniso > 0.0 {
        eprintln!(
            "settle-decay hunt: settle needle-prune log-aniso = {:.2} (settle prunes only; densify-time = {:.2})",
            cfg.settle_needle_prune_log_aniso, cfg.needle_prune_log_anisotropy
        );
    }
    if cfg.sh_warmup_interval > 0 {
        eprintln!(
            "C2 SH warmup: active degree 0->3, +1 every {} iters (coeffs 1->4->9->16, all 16 from iter {})",
            cfg.sh_warmup_interval,
            cfg.sh_warmup_interval * 3
        );
    }
    if cfg.opacity_reset_interval > 0 && cfg.opacity_reset_window_margin > 0 {
        eprintln!(
            "opacity resets: every {} iters, gated to iter <= {} (window end {} − margin {})",
            cfg.opacity_reset_interval,
            (cfg.iters / 2).saturating_sub(cfg.opacity_reset_window_margin),
            cfg.iters / 2,
            cfg.opacity_reset_window_margin
        );
    }

    // Training loop: sample random views
    let mut train_loss = 0.0f32;
    let mut densify_events: usize = 0;
    #[cfg(feature = "gpu")]
    let mut gpu_backward_disabled_reason: Option<String> = None;

    // Track last densification stats for CSV logging
    let mut last_densify_split: usize = 0;
    let mut last_densify_clone: usize = 0;
    let mut last_densify_prune: usize = 0;
    let mut last_grad_p50: f32 = 0.0;
    let mut last_grad_p90: f32 = 0.0;
    let mut last_gaussian_stats: GaussianStats = GaussianStats::default();

    // Render-watchdog state: consecutive background-only frames.
    const WATCHDOG_STRIKES_TO_TRIP: usize = 5;
    let mut watchdog_strikes: usize = 0;

    for iter in 0..cfg.iters {
        // D2: position-only exponential schedule; all other LRs stay at their configured values.
        let t = (iter as f32 / POSITION_LR_MAX_STEPS).min(1.0);
        position_opt.lr = (position_lr_init.ln() * (1.0 - t) + position_lr_final.ln() * t).exp();
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

            // Important: Downsample the image independently, then adjust camera to match actual image size
            // This handles cases where COLMAP camera dimensions don't match actual image file dimensions
            let train_target_ds = if (cfg.downsample_factor - 1.0).abs() < 0.001 {
                train_target.clone()
            } else {
                // Determine target dimensions from actual image, not camera
                let target_width = ((train_target.width() as f32) * cfg.downsample_factor).round().max(1.0) as u32;
                let target_height = ((train_target.height() as f32) * cfg.downsample_factor).round().max(1.0) as u32;
                downsample_image_smart(&train_target, target_width, target_height, cfg.downsample_factor)
            };

            // Adjust camera to match actual downsampled image dimensions
            let mut train_camera = train_camera;
            if train_camera.width != train_target_ds.width() || train_camera.height != train_target_ds.height() {
                let scale_x = train_target_ds.width() as f32 / train_camera.width as f32;
                let scale_y = train_target_ds.height() as f32 / train_camera.height as f32;
                train_camera.width = train_target_ds.width();
                train_camera.height = train_target_ds.height();
                train_camera.fx *= scale_x;
                train_camera.fy *= scale_y;
                train_camera.cx *= scale_x;
                train_camera.cy *= scale_y;
            }

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

        // Uniform loss weights, matching reference 3DGS (no pixel weighting). The old coverage
        // weighting (covered 1.0 / uncovered 0.5) halved the background-restoring gradient from
        // sky pixels while partially-covered pixels pushed at full weight — a mechanism that
        // drags the background dark. It persisted across every other variable A/B'd (learned vs
        // frozen bg, L2 vs L1+DSSIM, densify on/off).
        let weights: Vec<f32> =
            vec![1.0; (train_camera.width * train_camera.height) as usize];

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
        let train_psnr = compute_psnr(&rendered_linear, &train_target_linear);
        let t_forward = t0.elapsed();

        // Render watchdog: a wgpu fault or a streak of background-only frames means the
        // renderer is gone — abort loudly with the model saved rather than keep training
        // against garbage (which destroys the model within a few hundred iterations).
        if cfg.render_watchdog {
            #[cfg(feature = "gpu")]
            let gpu_fault = crate::gpu::gpu_fault_seen();
            #[cfg(not(feature = "gpu"))]
            let gpu_fault = false;

            if frame_is_background_only(&rendered_linear, &bg) {
                watchdog_strikes += 1;
            } else {
                watchdog_strikes = 0;
            }

            if gpu_fault || watchdog_strikes >= WATCHDOG_STRIKES_TO_TRIP {
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
                let cloud = crate::core::GaussianCloud {
                    gaussians: gaussians.clone(),
                };
                let (bounds_min, bounds_max) = crate::io::compute_bounds(&cloud.gaussians);
                let metadata = crate::io::ModelMetadata {
                    num_gaussians: cloud.gaussians.len() as u64,
                    sh_degree: 3,
                    bounds_min,
                    bounds_max,
                    training_iterations: (iter + 1) as u64,
                    training_psnr: train_psnr,
                    compression: crate::io::Compression::None,
                    training_width: train_camera.width,
                    training_height: train_camera.height,
                    training_downsample_factor: cfg.downsample_factor,
                    dataset_path: String::new(),
                };
                let abort_path = cfg.out_dir.join("model_at_watchdog_abort.gs");
                if let Err(e) = crate::io::save_model(&abort_path, &cloud, &metadata) {
                    eprintln!("watchdog: failed to save abort model: {e}");
                }
                let reason = if gpu_fault {
                    "wgpu reported an uncaptured error or device loss".to_string()
                } else {
                    format!(
                        "{WATCHDOG_STRIKES_TO_TRIP} consecutive train renders were pure background \
                         (renderer output is background-only; GPU pipeline presumed dead)"
                    )
                };
                anyhow::bail!(
                    "RENDER WATCHDOG tripped @iter {}/{}: {reason}. Model saved to {:?}. \
                     Disable with --no-render-watchdog if this is a false positive.",
                    iter + 1,
                    cfg.iters,
                    abort_path
                );
            }
        }

        // Backward
        let t1 = Instant::now();
        let (_img_u8, d_color, d_opacity_logits, d_positions, d_log_scales, d_rot_vecs, d_bg, d_mean_px) = {
            #[cfg(feature = "gpu")]
            if let Some(ref renderer) = gpu_renderer {
                // Use GPU backward pass with sparse atomic gradients (efficient for <10k Gaussians)
                if let Some(reason) = &gpu_backward_disabled_reason {
                    if iter == 0 {
                        eprintln!("GPU backward disabled, using CPU backward: {reason}");
                    }
                    render_full_color_grads_ext(&gaussians, &train_camera, &d_image, &bg, disable_sh)
                } else {
                    match renderer.render_with_gradients(&gaussians, &train_camera, &bg, &d_image) {
                    Ok((_pixels, grads_2d)) => {
                    // TEMPORARY: CPU projection backward is default due to GPU shader bug
                    // Use SUGAR_GPU_GRADIENTS=1 to enable GPU projection backward (experimental)
                    let use_gpu_projection = std::env::var("SUGAR_GPU_GRADIENTS").is_ok();

                    let (d_pos, d_scales, d_rots, d_background) = if use_gpu_projection {
                        // GPU projection backward (experimental - has bugs with full parameter training)
                        let (d_positions, d_log_scales, d_rotations, _d_sh) =
                            renderer.project_gradients_backward(&gaussians, &train_camera, &grads_2d);
                        (d_positions, d_log_scales, d_rotations, grads_2d.d_background)
                    } else {
                        // CPU projection backward (default, stable)
                        crate::gpu::chain_2d_to_3d_gradients_cpu(&grads_2d, &gaussians, &train_camera)
                    };

                    let dummy_img = image::RgbImage::new(train_camera.width, train_camera.height);
                    // The GPU rasterization backward accumulates the same pixel-space
                    // dL/d(mean_px) as the CPU path (backward.wgsl, offsets 8-9), so hand it
                    // straight to the B1 densification accumulator.
                    (dummy_img, grads_2d.d_colors, grads_2d.d_opacity_logits, d_pos, d_scales, d_rots, d_background, grads_2d.d_mean_px)
                    }
                    Err(e) => {
                        eprintln!("GPU backward failed, falling back to CPU: {e}");
                        gpu_backward_disabled_reason = Some(e);
                        render_full_color_grads_ext(&gaussians, &train_camera, &d_image, &bg, disable_sh)
                    }
                }
                }
            } else {
                // CPU backward pass
                render_full_color_grads_ext(&gaussians, &train_camera, &d_image, &bg, disable_sh)
            }

            #[cfg(not(feature = "gpu"))]
            render_full_color_grads_ext(&gaussians, &train_camera, &d_image, &bg, disable_sh)
        };
        let t_backward = t1.elapsed();

        // Convert dL/d(color) -> dL/d(SH coeffs) using per-Gaussian SH basis.
        let mut d_sh: Vec<[Vector3<f32>; 16]> = vec![[Vector3::zeros(); 16]; gaussians.len()];
        if learn_sh {
            for (i, g) in gaussians.iter().enumerate() {
                let basis = crate::core::sh_basis(&train_camera.view_direction(&g.position));
                for k in 0..16 {
                    d_sh[i][k] = d_color[i] * basis[k];
                }
            }
        } else {
            // DC-only learning (k=0). Uses SH_C0 constant.
            let sh0 = crate::core::SH_C0;
            for i in 0..gaussians.len() {
                d_sh[i][0] = d_color[i] * sh0;
            }
        }

        let t2 = Instant::now();
        // Settle-decay hunt: optionally freeze SH once the densify window closes (iter > iters/2).
        if !(cfg.freeze_sh_after_window && (iter + 1) > cfg.iters / 2) {
            // C2: only the warmup-unlocked SH coefficients step (16 when warmup is off).
            sh_opt.step_active(
                &mut sh_params,
                &d_sh,
                active_sh_coeffs(cfg.sh_warmup_interval, iter),
            );
        }
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
            // Clip scale gradients to prevent extreme updates
            const SCALE_GRAD_CLIP: f32 = 1.0;
            let d_log_scales_clipped: Vec<Vector3<f32>> = d_log_scales
                .iter()
                .map(|g| Vector3::new(
                    g.x.clamp(-SCALE_GRAD_CLIP, SCALE_GRAD_CLIP),
                    g.y.clamp(-SCALE_GRAD_CLIP, SCALE_GRAD_CLIP),
                    g.z.clamp(-SCALE_GRAD_CLIP, SCALE_GRAD_CLIP),
                ))
                .collect();
            scale_opt.step(&mut log_scales, &d_log_scales_clipped);

            // Clamp log-space scales to prevent degenerate ellipsoids
            // exp(5) ≈ 148 (large but reasonable), exp(-10) ≈ 4.5e-5 (tiny but valid)
            // Tighter bounds prevent stretched/exploding Gaussians during training
            const MAX_LOG_SCALE: f32 = 5.0;
            const MIN_LOG_SCALE: f32 = -10.0;
            for scale in log_scales.iter_mut() {
                // First clamp individual axes
                scale.x = scale.x.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);
                scale.y = scale.y.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);
                scale.z = scale.z.clamp(MIN_LOG_SCALE, MAX_LOG_SCALE);

                // Optional anisotropy constraint: pull smaller axes toward largest.
                // Off by default — reference 3DGS lets splats flatten to 10–100×; the EWA
                // low-pass filter in the projection is the anti-needle defense.
                if cfg.max_log_anisotropy > 0.0 {
                    let max_s = scale.x.max(scale.y).max(scale.z);
                    let min_allowed = max_s - cfg.max_log_anisotropy;
                    scale.x = scale.x.max(min_allowed);
                    scale.y = scale.y.max(min_allowed);
                    scale.z = scale.z.max(min_allowed);
                }
            }
        }
        if cfg.learn_rotation {
            rotation_opt.step(&mut rotations, &d_rot_vecs);
        }
        // Settle-decay hunt: optionally freeze bg once the densify window closes (iter > iters/2).
        if cfg.learn_background && !(cfg.freeze_bg_in_settle && (iter + 1) > cfg.iters / 2) {
            let mut bg_param = vec![bg];
            let bg_grad = vec![d_bg];
            bg_opt.step(&mut bg_param, &bg_grad);
            bg = bg_param[0];
            // Clamp background to valid RGB range [0, 1]
            bg.x = bg.x.clamp(0.0, 1.0);
            bg.y = bg.y.clamp(0.0, 1.0);
            bg.z = bg.z.clamp(0.0, 1.0);
        }
        let t_step = t2.elapsed();

        let is_validation_iter = (iter + 1) % cfg.val_interval == 0 || iter + 1 == cfg.iters;

        // Log a lightweight row at `log_interval` even if validation is infrequent.
        if should_log && !is_validation_iter {
            // Compute Gaussian health stats for monitoring
            last_gaussian_stats = GaussianStats::compute(
                &log_scales,
                &opacity_logits,
                Some(&d_positions),
                Some(&d_log_scales),
                Some(&d_rot_vecs),
            );
            if let Some(ref mut logger) = csv_logger {
                let total_time = iter_start.map(|s| s.elapsed().as_secs_f32() * 1000.0).unwrap_or(0.0);
                let forward_ms = t_forward.as_secs_f32() * 1000.0;
                let backward_ms = t_backward.as_secs_f32() * 1000.0;
                let step_ms = t_step.as_secs_f32() * 1000.0;
                let _ = logger.log_iteration(
                    iter,
                    train_loss,
                    train_psnr,
                    gaussians.len(),
                    forward_ms,
                    backward_ms,
                    step_ms,
                    total_time,
                    last_densify_split,
                    last_densify_clone,
                    last_densify_prune,
                    last_grad_p50,
                    last_grad_p90,
                    &bg,
                    &last_gaussian_stats,
                    -1.0, // eval_ssim: proxy row (train psnr), no multi-view eval
                );
            }
        }

        if cfg.densify_interval > 0 {
            if grad_accum_pos_norm.len() != d_mean_px.len() {
                grad_accum_pos_norm.resize(d_mean_px.len(), 0.0);
                grad_denom.resize(d_mean_px.len(), 0.0);
            }
            // B1: accumulate the VIEW-SPACE mean gradient — the quantity the densify_grad_threshold
            //     (~0.0002, reference 3DGS) is calibrated for, instead of the depth-scaled
            //     world-space position gradient. The renderer returns it in PIXEL units; convert to
            //     NDC ([-1,1]) so the threshold is resolution-independent: dL/d(ndc) = dL/d(px)·(dim/2).
            // B2: a Gaussian is "visible" this iteration iff it received a non-zero view-space
            //     gradient; count those so the per-Gaussian average is over views it appeared in.
            let ndc_sx = train_camera.width as f32 * 0.5;
            let ndc_sy = train_camera.height as f32 * 0.5;
            for (i, d_uv) in d_mean_px.iter().enumerate() {
                if *d_uv != Vector2::zeros() {
                    let gx = d_uv.x * ndc_sx;
                    let gy = d_uv.y * ndc_sy;
                    grad_accum_pos_norm[i] += (gx * gx + gy * gy).sqrt();
                    grad_denom[i] += 1.0;
                }
            }
            grad_window_iters += 1;
        }

        // Validation
        if is_validation_iter {
            let mut test_psnr_sum = 0.0f32;
            let mut test_ssim_sum = 0.0f32;
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

                    // Downsample image and adjust camera to match actual image dimensions
                    let test_target_ds = if (cfg.downsample_factor - 1.0).abs() < 0.001 {
                        test_target.clone()
                    } else {
                        let target_width = ((test_target.width() as f32) * cfg.downsample_factor).round().max(1.0) as u32;
                        let target_height = ((test_target.height() as f32) * cfg.downsample_factor).round().max(1.0) as u32;
                        downsample_image_smart(&test_target, target_width, target_height, cfg.downsample_factor)
                    };

                    let mut test_camera = test_camera;
                    if test_camera.width != test_target_ds.width() || test_camera.height != test_target_ds.height() {
                        let scale_x = test_target_ds.width() as f32 / test_camera.width as f32;
                        let scale_y = test_target_ds.height() as f32 / test_camera.height as f32;
                        test_camera.width = test_target_ds.width();
                        test_camera.height = test_target_ds.height();
                        test_camera.fx *= scale_x;
                        test_camera.fy *= scale_y;
                        test_camera.cx *= scale_x;
                        test_camera.cy *= scale_y;
                    }

                    let test_target_linear = rgb8_to_linear_vec(&test_target_ds);
                    (test_camera, test_target_linear)
                };

                let rendered = render(&gaussians, &test_camera, &bg);
                let psnr = compute_psnr(&rendered, &test_target_linear);
                test_psnr_sum += psnr;
                test_ssim_sum += compute_ssim(
                    &rendered,
                    &test_target_linear,
                    test_camera.width as usize,
                    test_camera.height as usize,
                );

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
            let avg_test_ssim = test_ssim_sum / (test_indices_for_metrics.len() as f32);

            // Log to CSV if logger is enabled
            if let Some(ref mut logger) = csv_logger {
                let total_time = if let Some(start) = iter_start {
                    start.elapsed().as_secs_f32() * 1000.0
                } else {
                    0.0
                };
                let forward_ms = t_forward.as_secs_f32() * 1000.0;
                let backward_ms = t_backward.as_secs_f32() * 1000.0;
                let step_ms = t_step.as_secs_f32() * 1000.0;

                // Update Gaussian health stats for validation logging
                last_gaussian_stats = GaussianStats::compute(
                    &log_scales,
                    &opacity_logits,
                    Some(&d_positions),
                    Some(&d_log_scales),
                    Some(&d_rot_vecs),
                );

                let _ = logger.log_iteration(
                    iter,
                    train_loss,
                    avg_test_psnr,
                    gaussians.len(),
                    forward_ms,
                    backward_ms,
                    step_ms,
                    total_time,
                    last_densify_split,
                    last_densify_clone,
                    last_densify_prune,
                    last_grad_p50,
                    last_grad_p90,
                    &bg,
                    &last_gaussian_stats,
                    avg_test_ssim,
                );
            }

            // Save incremental test view every 100 iterations
            if (iter + 1) % 100 == 0 && first_test_rendered.is_some() {
                let output_path = cfg.out_dir.join(format!("m8_test_view_rendered_{:04}.png", iter + 1));
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
        // B7: densify only during the first HALF of training (reference: iters 500–15000 of
        // 30k), then freeze the population and let it settle. Without this, every cycle keeps
        // injecting full-opacity capacity to the end of the run — the 15k validation run showed
        // a reset-bounded sawtooth with a decaying envelope (16.7 → 12.8 dB by iter 7500) as
        // additions accelerated ~100→550/cycle. Ported proportionally as `iters/2`; the
        // reference 500-iter warmup is effectively covered by the first densify interval.
        if cfg.densify_interval > 0
            && grad_window_iters > 0
            && (iter + 1) % cfg.densify_interval == 0
            && (iter + 1) <= cfg.iters / 2
        {
            let before = gaussians.len();

            // Enforce GPU hard cap to prevent buffer overflow
            #[cfg(feature = "gpu")]
            let effective_max_gaussians = if gpu_renderer.is_some() {
                cfg.densify_max_gaussians.min(GPU_HARD_CAP_GAUSSIANS)
            } else {
                cfg.densify_max_gaussians
            };

            #[cfg(not(feature = "gpu"))]
            let effective_max_gaussians = cfg.densify_max_gaussians;

            let (stats, remap) = densify_and_prune(
                &mut gaussians,
                &mut sh_params,
                &mut opacity_logits,
                &mut positions,
                &mut log_scales,
                &mut rotations,
                &mut grad_accum_pos_norm,
                &mut grad_denom,
                &mut rng,
                grad_window_iters,
                effective_max_gaussians,
                cfg.densify_grad_threshold,
                cfg.prune_opacity_threshold,
                cfg.split_sigma_threshold,
                cfg.needle_prune_log_anisotropy,
                scene_extent,
            );
            // B11: the parameter arrays have been re-built; remap optimizer state through the
            // survivor map so surviving Gaussians keep their Adam moments (reference behavior)
            // and only new/re-initialized ones start from zero. A full reset here restarted the
            // optimizer on every densify event, which degraded PSNR at short intervals.
            sh_opt.remap_moments_keep_t(&remap);
            opacity_opt.remap_moments_keep_t(&remap);
            position_opt.remap_moments_keep_t(&remap);
            scale_opt.remap_moments_keep_t(&remap);
            rotation_opt.remap_moments_keep_t(&remap);

            grad_window_iters = 0;
            densify_events += 1;

            // Track stats for CSV logging
            last_densify_split = stats.split;
            last_densify_clone = stats.cloned;
            last_densify_prune = stats.pruned;
            last_grad_p50 = stats.grad_p50;
            last_grad_p90 = stats.grad_p90;

            let outlier_msg = if stats.pruned_outliers > 0 {
                format!(" pruned_outliers={}", stats.pruned_outliers)
            } else {
                String::new()
            };
            let needle_msg = if stats.pruned_needles > 0 {
                format!(" pruned_needles={}", stats.pruned_needles)
            } else {
                String::new()
            };
            let oversize_msg = if stats.pruned_oversize > 0 {
                format!(" pruned_oversize={}", stats.pruned_oversize)
            } else {
                String::new()
            };
            eprintln!(
                "densify @iter {}/{}: gaussians {} -> {} (kept={} pruned={}{}{}{} split={} cloned={} cap_hit={} grad_p50={:.4} grad_p90={:.4})",
                iter + 1,
                cfg.iters,
                before,
                gaussians.len(),
                stats.kept,
                stats.pruned,
                outlier_msg,
                needle_msg,
                oversize_msg,
                stats.split,
                stats.cloned,
                stats.cap_hit,
                stats.grad_p50,
                stats.grad_p90
            );
        }

        // B3: opacity reset — every opacity_reset_interval iterations, cap opacities DOWNWARD
        // toward ~0.01 (never raise them), forcing weak Gaussians to re-earn their opacity or be
        // pruned. Like reference 3DGS, resets fire only inside the densification window (first
        // half of training, matching B7): the settle phase must run reset-free so the model can
        // converge — the Phase-3 15k run ended mid-recovery from an iter-12000 reset (13.78 at
        // 13500 → only 14.02 at 15000) because resets kept firing after densification stopped.
        // The window-margin gate additionally holds back resets near the window end (see config).
        if opacity_reset_due(
            iter + 1,
            cfg.opacity_reset_interval,
            cfg.iters / 2,
            cfg.opacity_reset_window_margin,
        ) {
            let reset_cap = crate::core::inverse_sigmoid(cfg.opacity_reset_floor);
            for i in 0..opacity_logits.len() {
                opacity_logits[i] = opacity_logits[i].min(reset_cap);
                gaussians[i].opacity = opacity_logits[i];
            }
            opacity_opt.reset_moments_keep_t(opacity_logits.len());
            eprintln!(
                "opacity reset @iter {}/{} (capped to <= {})",
                iter + 1,
                cfg.iters,
                cfg.opacity_reset_floor
            );
        }

        // Settle-phase prune (opt-in): densification (and with it ALL pruning) stops at iters/2,
        // so needles/oversize/dead mass that survive the window — or degrade during settle — are
        // otherwise frozen into the final model. Prune-only: grad threshold ∞ means no
        // split/clone, and no opacity reset fires here, so the settle phase stays convergent.
        // Skip the last iteration so the final eval isn't run on a just-mutated population.
        if cfg.settle_prune_interval > 0
            && (iter + 1) % cfg.settle_prune_interval == 0
            && (iter + 1) > cfg.iters / 2
            && (iter + 1) < cfg.iters
        {
            let before = gaussians.len();
            let (stats, remap) = densify_and_prune(
                &mut gaussians,
                &mut sh_params,
                &mut opacity_logits,
                &mut positions,
                &mut log_scales,
                &mut rotations,
                &mut grad_accum_pos_norm,
                &mut grad_denom,
                &mut rng,
                grad_window_iters.max(1),
                0, // no additions possible at ∞ threshold; cap is irrelevant
                f32::INFINITY,
                cfg.prune_opacity_threshold,
                cfg.split_sigma_threshold,
                // 30k settle-decay hunt: tighter needle threshold for settle prunes only.
                if cfg.settle_needle_prune_log_aniso > 0.0 {
                    cfg.settle_needle_prune_log_aniso
                } else {
                    cfg.needle_prune_log_anisotropy
                },
                scene_extent,
            );
            sh_opt.remap_moments_keep_t(&remap);
            opacity_opt.remap_moments_keep_t(&remap);
            position_opt.remap_moments_keep_t(&remap);
            scale_opt.remap_moments_keep_t(&remap);
            rotation_opt.remap_moments_keep_t(&remap);
            last_densify_prune = stats.pruned;
            if stats.pruned > 0 {
                eprintln!(
                    "settle prune @iter {}/{}: gaussians {} -> {} (opacity={} needles={} oversize={} outliers={})",
                    iter + 1,
                    cfg.iters,
                    before,
                    gaussians.len(),
                    stats.pruned
                        - stats.pruned_needles
                        - stats.pruned_oversize
                        - stats.pruned_outliers,
                    stats.pruned_needles,
                    stats.pruned_oversize,
                    stats.pruned_outliers
                );
            }
        }

        // Periodic checkpoint save (opt-in via --save-interval): write model_<step>.gs so a
        // single run yields the whole iteration grid (e.g. 3k/15k/30k) for equal-iteration
        // comparison, not just the final model. Re-sync gaussians from the live param arrays
        // first (they hold this iter's post-step/densify state); the final iteration is left
        // to the post-loop save as model.gs.
        if cfg.save_interval > 0 && (iter + 1) % cfg.save_interval == 0 && (iter + 1) < cfg.iters {
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
            let cloud = crate::core::GaussianCloud { gaussians: gaussians.clone() };
            let (bounds_min, bounds_max) = crate::io::compute_bounds(&cloud.gaussians);
            let metadata = crate::io::ModelMetadata {
                num_gaussians: cloud.gaussians.len() as u64,
                sh_degree: 3,
                bounds_min,
                bounds_max,
                training_iterations: (iter + 1) as u64,
                training_psnr: train_psnr,
                compression: crate::io::Compression::None,
                training_width: train_camera.width,
                training_height: train_camera.height,
                training_downsample_factor: cfg.downsample_factor,
                dataset_path: String::new(),
            };
            let ckpt_path = cfg.out_dir.join(format!("model_{:06}.gs", iter + 1));
            match crate::io::save_model(&ckpt_path, &cloud, &metadata) {
                Ok(_) => eprintln!(
                    "checkpoint saved: {:?} ({} gaussians)",
                    ckpt_path,
                    cloud.gaussians.len()
                ),
                Err(e) => eprintln!("Warning: failed to save checkpoint {:?}: {}", ckpt_path, e),
            }
        }
    }

    // Final validation
    let mut final_psnr_sum = 0.0f32;
    let mut final_ssim_sum = 0.0f32;
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

            // Downsample image and adjust camera to match actual image dimensions
            let test_target_ds = if (cfg.downsample_factor - 1.0).abs() < 0.001 {
                test_target.clone()
            } else {
                let target_width = ((test_target.width() as f32) * cfg.downsample_factor).round().max(1.0) as u32;
                let target_height = ((test_target.height() as f32) * cfg.downsample_factor).round().max(1.0) as u32;
                downsample_image_smart(&test_target, target_width, target_height, cfg.downsample_factor)
            };

            let mut test_camera = test_camera;
            if test_camera.width != test_target_ds.width() || test_camera.height != test_target_ds.height() {
                let scale_x = test_target_ds.width() as f32 / test_camera.width as f32;
                let scale_y = test_target_ds.height() as f32 / test_camera.height as f32;
                test_camera.width = test_target_ds.width();
                test_camera.height = test_target_ds.height();
                test_camera.fx *= scale_x;
                test_camera.fy *= scale_y;
                test_camera.cx *= scale_x;
                test_camera.cy *= scale_y;
            }

            let test_target_linear = rgb8_to_linear_vec(&test_target_ds);
            (test_camera, test_target_ds, test_target_linear)
        };

        let rendered = render(&gaussians, &test_camera, &bg);
        let psnr = compute_psnr(&rendered, &test_target_linear);
        final_psnr_sum += psnr;
        final_ssim_sum += compute_ssim(
            &rendered,
            &test_target_linear,
            test_camera.width as usize,
            test_camera.height as usize,
        );

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
    let final_ssim = final_ssim_sum / (test_indices_for_metrics.len() as f32);

    eprintln!("\n✅ Multi-view training complete!");
    eprintln!("Initial test PSNR: {:.2} dB", initial_psnr);
    eprintln!("Final test PSNR:   {:.2} dB", final_psnr);
    eprintln!("Final test SSIM:   {:.4}", final_ssim);
    eprintln!("Improvement:       {:.2} dB", final_psnr - initial_psnr);

    // Get training resolution from the first cached view (all have same resolution)
    let first_camera = &view_cache.get(&train_indices[0]).unwrap().camera;

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
        training_width: first_camera.width,
        training_height: first_camera.height,
        downsample_factor: cfg.downsample_factor,
        seed_used: actual_seed,
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
    fn ssim_identical_images_is_one() {
        let (w, h) = (32, 24);
        let img: Vec<Vector3<f32>> = (0..w * h)
            .map(|i| {
                let v = (i % 17) as f32 / 17.0;
                Vector3::new(v, 1.0 - v, 0.5 * v)
            })
            .collect();
        let s = compute_ssim(&img, &img, w, h);
        assert!((s - 1.0).abs() < 1e-4, "SSIM of identical images = {}", s);
    }

    #[test]
    fn ssim_orders_degradations_sensibly() {
        // Structured target; a lightly-noised copy must score far above an inverted copy,
        // and both must be strictly below 1.
        let (w, h) = (32, 24);
        let target: Vec<Vector3<f32>> = (0..w * h)
            .map(|i| {
                let x = (i % w) as f32 / w as f32;
                let y = (i / w) as f32 / h as f32;
                Vector3::new(x, y, ((x * 8.0).sin() * 0.5 + 0.5) * 0.9)
            })
            .collect();
        let noisy: Vec<Vector3<f32>> = target
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let n = if i % 2 == 0 { 0.02 } else { -0.02 };
                Vector3::new(
                    (v.x + n).clamp(0.0, 1.0),
                    (v.y + n).clamp(0.0, 1.0),
                    (v.z + n).clamp(0.0, 1.0),
                )
            })
            .collect();
        let inverted: Vec<Vector3<f32>> = target
            .iter()
            .map(|v| Vector3::new(1.0 - v.x, 1.0 - v.y, 1.0 - v.z))
            .collect();
        let s_noisy = compute_ssim(&noisy, &target, w, h);
        let s_inv = compute_ssim(&inverted, &target, w, h);
        assert!(s_noisy < 1.0 && s_noisy > 0.8, "noisy SSIM = {}", s_noisy);
        assert!(s_inv < 0.3, "inverted SSIM = {}", s_inv);
        assert!(s_noisy > s_inv + 0.4);
    }

    #[test]
    fn interval_split_matches_nerfstudio_convention() {
        // 301 images like tandt/train: interval 8 -> 38 test / 263 train,
        // lexicographically first image held out.
        let names: Vec<String> = (1..=301).map(|i| format!("{:05}.jpg", i)).collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let (train, test) = interval_split_by_name(&refs, 8);
        assert_eq!(test.len(), 38);
        assert_eq!(train.len(), 263);
        assert_eq!(test[0], 0); // "00001.jpg" is a test view
        assert_eq!(test[1], 8);
        // Partition is complete and disjoint.
        let mut all: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..301).collect::<Vec<_>>());
    }

    #[test]
    fn interval_split_follows_filename_order_not_index_order() {
        // Indices are in registration order (like images.bin); the split must be
        // computed over filename order.
        let refs = ["c.jpg", "a.jpg", "d.jpg", "b.jpg"];
        let (train, test) = interval_split_by_name(&refs, 2);
        // Sorted: a(1) b(3) c(0) d(2); positions 0,2 -> test = indices [1, 0]
        assert_eq!(test, vec![1, 0]);
        assert_eq!(train, vec![3, 2]);
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
        let mut grad_accum = vec![2.0, 0.0]; // avg_grad = 2.0/denom = 0.2
        let mut denom = vec![10.0, 0.0]; // B2: g1 visible in all 10 window iterations

        let mut rng = StdRng::seed_from_u64(123);
        // scene_extent = 2.0: split boundary 0.01·2 = 0.02 < σ = 0.1 → split (not clone),
        // and oversize-prune bound 0.1·2 = 0.2 > σ so g1 is not pruned.
        let (stats, remap) = densify_and_prune(
            &mut gaussians,
            &mut sh_params,
            &mut opacity_logits,
            &mut positions,
            &mut log_scales,
            &mut rotations,
            &mut grad_accum,
            &mut denom,
            &mut rng,
            10,
            10,
            0.1,
            0.01,
            0.05,
            0.0, // needle prune disabled (inputs are isotropic; inert either way)
            2.0,
        );

        // g2 pruned, and g1 split -> two gaussians remain.
        assert_eq!(gaussians.len(), 2);
        assert_eq!(sh_params.len(), 2);
        assert_eq!(opacity_logits.len(), 2);
        assert_eq!(positions.len(), 2);
        assert_eq!(log_scales.len(), 2);
        assert_eq!(rotations.len(), 2);
        assert_eq!(grad_accum, vec![0.0, 0.0]);
        assert_eq!(denom, vec![0.0, 0.0]);

        // First is the original, second is the split copy.
        assert_eq!(gaussians[0].position, g1.position);
        assert_ne!(gaussians[1].position, g1.position);
        assert!(gaussians[1].scale.x < g1.scale.x);
        assert!(gaussians[1].scale.y < g1.scale.y);
        assert!(gaussians[1].scale.z < g1.scale.z);

        // B12: parent and child keep the parent's opacity unchanged (reference 3DGS).
        assert_relative_eq!(gaussians[0].opacity, 2.0, epsilon = 1e-6);
        assert_relative_eq!(gaussians[1].opacity, 2.0, epsilon = 1e-6);
        assert_relative_eq!(opacity_logits[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(opacity_logits[1], 2.0, epsilon = 1e-6);

        // B11: a split re-initializes the parent, so BOTH resulting Gaussians start with
        // fresh optimizer state (reference prunes the split source and creates N new ones).
        assert_eq!(remap, vec![None, None]);

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
    fn densify_clone_keeps_parent_optimizer_state() {
        // σ = 0.01 is below the split boundary (0.01·scene_extent = 0.02) → clone path.
        let g = Gaussian::new(
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()),
            UnitQuaternion::identity(),
            1.0,
            empty_sh(),
        );
        let mut gaussians = vec![g.clone()];
        let mut sh_params = vec![[Vector3::zeros(); 16]];
        let mut opacity_logits = vec![1.0];
        let mut positions = vec![g.position];
        let mut log_scales = vec![g.scale];
        let mut rotations = vec![g.rotation];
        let mut grad_accum = vec![2.0];
        let mut denom = vec![10.0];
        let mut rng = StdRng::seed_from_u64(7);

        let (stats, remap) = densify_and_prune(
            &mut gaussians,
            &mut sh_params,
            &mut opacity_logits,
            &mut positions,
            &mut log_scales,
            &mut rotations,
            &mut grad_accum,
            &mut denom,
            &mut rng,
            10,
            10,
            0.1,
            0.01,
            0.05,
            0.0, // needle prune disabled (inputs are isotropic; inert either way)
            2.0,
        );

        assert_eq!(stats.cloned, 1);
        assert_eq!(stats.split, 0);
        // B11: the cloned parent keeps its optimizer state; only the child starts fresh.
        assert_eq!(remap, vec![Some(0), None]);
        // B12: both keep the parent opacity unchanged.
        assert_eq!(opacity_logits, vec![1.0, 1.0]);
    }

    #[test]
    fn densify_respects_max_gaussians_cap() {
        // 4 clone-eligible Gaussians (σ = 0.01 < split boundary 0.02), cap 6 → exactly 2
        // children may be added regardless of how many parents qualify.
        let make = |x: f32| {
            Gaussian::new(
                Vector3::new(x, 0.0, 1.0),
                Vector3::new(0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()),
                UnitQuaternion::identity(),
                1.0,
                empty_sh(),
            )
        };
        let mut gaussians: Vec<Gaussian> = (0..4).map(|i| make(i as f32)).collect();
        let mut sh_params = vec![[Vector3::zeros(); 16]; 4];
        let mut opacity_logits = vec![1.0; 4];
        let mut positions: Vec<Vector3<f32>> = gaussians.iter().map(|g| g.position).collect();
        let mut log_scales: Vec<Vector3<f32>> = gaussians.iter().map(|g| g.scale).collect();
        let mut rotations: Vec<UnitQuaternion<f32>> =
            gaussians.iter().map(|g| g.rotation).collect();
        let mut grad_accum = vec![2.0; 4];
        let mut denom = vec![10.0; 4];
        let mut rng = StdRng::seed_from_u64(7);

        let (stats, remap) = densify_and_prune(
            &mut gaussians,
            &mut sh_params,
            &mut opacity_logits,
            &mut positions,
            &mut log_scales,
            &mut rotations,
            &mut grad_accum,
            &mut denom,
            &mut rng,
            10,
            6, // cap
            0.1,
            0.01,
            0.05,
            0.0, // needle prune disabled (inputs are isotropic; inert either way)
            2.0,
        );

        assert_eq!(gaussians.len(), 6, "cap must bound the final count");
        assert_eq!(stats.cloned + stats.split, 2);
        assert!(stats.cap_hit, "cap_hit must be reported when the budget runs out");
        assert_eq!(remap.len(), 6);
        // All 4 parents survive (kept), regardless of the cap.
        assert_eq!(stats.kept, 4);
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

    #[test]
    fn test_frame_is_background_only() {
        let bg = Vector3::new(0.2, 0.2, 0.2);
        let n = 490 * 273;

        // Pure background frame → trips.
        let empty = vec![bg; n];
        assert!(frame_is_background_only(&empty, &bg));

        // 5% of pixels carry content (well above the 1% tolerance) → does not trip.
        let mut content = vec![bg; n];
        for i in 0..n / 20 {
            content[i * 20] = Vector3::new(0.8, 0.4, 0.1);
        }
        assert!(!frame_is_background_only(&content, &bg));

        // Background-only except one small patch (~2%) — the 2026-07-10 failure shape → trips.
        let mut patch = vec![bg; n];
        for i in 0..n / 50 {
            patch[i] = Vector3::new(0.5, 0.5, 0.9);
        }
        // patch is contiguous at the start; stride-7 sampling still sees ~2% non-bg... which
        // is above 1%, so this must NOT trip — the real failure had >99% background.
        assert!(!frame_is_background_only(&patch, &bg));
        let mut tiny_patch = vec![bg; n];
        for i in 0..n / 300 {
            tiny_patch[i] = Vector3::new(0.5, 0.5, 0.9);
        }
        assert!(frame_is_background_only(&tiny_patch, &bg));

        // Constant frame that does NOT match bg (e.g. all zeros from a dead pipeline while
        // bg is gray) still trips via the constant-frame condition.
        let black = vec![Vector3::zeros(); n];
        assert!(frame_is_background_only(&black, &bg));
    }

    #[test]
    fn test_opacity_reset_due_window_margin() {
        // Reference behavior (margin 0): resets at every interval up to and including window end.
        assert!(opacity_reset_due(3000, 3000, 15000, 0));
        assert!(opacity_reset_due(15000, 3000, 15000, 0));
        assert!(!opacity_reset_due(18000, 3000, 15000, 0)); // settle phase
        assert!(!opacity_reset_due(3001, 3000, 15000, 0)); // off-interval
        assert!(!opacity_reset_due(3000, 0, 15000, 0)); // disabled

        // Margin 2500 on a 30k run (window end 15000): last reset moves to 12000.
        assert!(opacity_reset_due(12000, 3000, 15000, 2500));
        assert!(!opacity_reset_due(15000, 3000, 15000, 2500));

        // Margin 2500 on a 15k run (window end 7500): only the 3000 reset survives.
        assert!(opacity_reset_due(3000, 3000, 7500, 2500));
        assert!(!opacity_reset_due(6000, 3000, 7500, 2500));
    }

    #[test]
    fn test_active_sh_coeffs_warmup_schedule() {
        // Disabled: all 16 coefficients from the start.
        assert_eq!(active_sh_coeffs(0, 0), 16);
        assert_eq!(active_sh_coeffs(0, 30000), 16);

        // Reference schedule (interval 1000): DC-only, then +1 degree per 1000 iters.
        assert_eq!(active_sh_coeffs(1000, 0), 1);
        assert_eq!(active_sh_coeffs(1000, 999), 1);
        assert_eq!(active_sh_coeffs(1000, 1000), 4);
        assert_eq!(active_sh_coeffs(1000, 1999), 4);
        assert_eq!(active_sh_coeffs(1000, 2000), 9);
        assert_eq!(active_sh_coeffs(1000, 3000), 16);
        // Caps at degree 3 regardless of horizon.
        assert_eq!(active_sh_coeffs(1000, 29999), 16);
    }
}
