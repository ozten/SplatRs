///! GPU gradient computation structures and reduction.

use crate::core::Gaussian;
use crate::core::Camera;
use crate::diff::covariance_grad::{
    project_covariance_2d_grad_log_scale, project_covariance_2d_grad_point_cam,
    project_covariance_2d_grad_rotation_vector_at_r0,
};
use crate::diff::project_grad::project_point_grad_point_cam;
use crate::gpu::types::GradientGPU;
use nalgebra::{Matrix2, Vector2, Vector3};

/// Gradient buffers for a set of Gaussians (2D gradients only).
///
/// These are the gradients w.r.t. 2D projected Gaussian parameters.
/// They can be further chained through to 3D parameters if needed.
#[derive(Clone, Debug)]
pub struct GaussianGradients2D {
    /// Gradients w.r.t. Gaussian color (linear RGB)
    pub d_colors: Vec<Vector3<f32>>,

    /// Gradients w.r.t. opacity logit (before sigmoid)
    pub d_opacity_logits: Vec<f32>,

    /// Gradients w.r.t. 2D mean in pixel space
    pub d_mean_px: Vec<Vector2<f32>>,

    /// Gradients w.r.t. 2D covariance (xx, xy, yy)
    pub d_cov_2d: Vec<Vector3<f32>>,

    /// Gradient w.r.t. background color
    pub d_background: Vector3<f32>,
}

impl GaussianGradients2D {
    /// Create zero gradients for a given number of Gaussians.
    pub fn zeros(num_gaussians: usize) -> Self {
        Self {
            d_colors: vec![Vector3::zeros(); num_gaussians],
            d_opacity_logits: vec![0.0; num_gaussians],
            d_mean_px: vec![Vector2::zeros(); num_gaussians],
            d_cov_2d: vec![Vector3::zeros(); num_gaussians],
            d_background: Vector3::zeros(),
        }
    }

    /// Create empty gradients to signal fallback to CPU.
    pub fn empty() -> Self {
        Self {
            d_colors: vec![],
            d_opacity_logits: vec![],
            d_mean_px: vec![],
            d_cov_2d: vec![],
            d_background: Vector3::zeros(),
        }
    }
}

/// Reduce per-pixel gradients to final per-Gaussian gradients.
///
/// Each pixel has written gradients for all Gaussians to its own buffer section.
/// This function sums across all pixels to get the final gradients.
///
/// # Arguments
/// * `pixel_grads` - Flat array of GradientGPU: [px0_g0, px0_g1, ..., px1_g0, px1_g1, ...]
/// * `num_pixels` - Total number of pixels
/// * `num_gaussians` - Number of Gaussians
///
/// # Performance
/// This is a CPU reduction but should be fast (~5-10ms for typical scenes):
/// - 64×64 pixels × 3 Gaussians = ~12K gradient entries (test scene)
/// - 200×200 pixels × 500 Gaussians = ~20M gradient entries (larger scene)
/// - Simple addition, highly cache-friendly
pub fn reduce_pixel_gradients(
    pixel_grads: &[GradientGPU],
    num_pixels: usize,
    num_gaussians: usize,
) -> GaussianGradients2D {
    assert_eq!(
        pixel_grads.len(),
        num_pixels * num_gaussians,
        "Pixel gradients buffer size mismatch"
    );

    let mut final_grads = GaussianGradients2D::zeros(num_gaussians);

    // Sum across all pixels for each Gaussian
    for px_idx in 0..num_pixels {
        let px_base = px_idx * num_gaussians;

        for g_idx in 0..num_gaussians {
            let grad = &pixel_grads[px_base + g_idx];

            final_grads.d_colors[g_idx] += Vector3::new(
                grad.d_color[0],
                grad.d_color[1],
                grad.d_color[2],
            );

            final_grads.d_opacity_logits[g_idx] += grad.d_opacity_logit_pad[0];

            final_grads.d_mean_px[g_idx] += Vector2::new(
                grad.d_mean_px[0],
                grad.d_mean_px[1],
            );

            final_grads.d_cov_2d[g_idx] += Vector3::new(
                grad.d_cov_2d[0],
                grad.d_cov_2d[1],
                grad.d_cov_2d[2],
            );
        }
    }

    // Background gradient would need special handling (not implemented yet)
    // For now, d_background remains zero

    final_grads
}

/// DEPRECATED: Old per-workgroup reduction (had race conditions).
/// Kept for reference. Use reduce_pixel_gradients instead.
#[allow(dead_code)]
pub fn reduce_workgroup_gradients(
    workgroup_grads: &[GradientGPU],
    num_workgroups: usize,
    num_gaussians: usize,
) -> GaussianGradients2D {
    assert_eq!(
        workgroup_grads.len(),
        num_workgroups * num_gaussians,
        "Workgroup gradients buffer size mismatch"
    );

    let mut final_grads = GaussianGradients2D::zeros(num_gaussians);

    // Sum across all workgroups for each Gaussian
    for wg_idx in 0..num_workgroups {
        let wg_base = wg_idx * num_gaussians;

        for g_idx in 0..num_gaussians {
            let grad = &workgroup_grads[wg_base + g_idx];

            final_grads.d_colors[g_idx] += Vector3::new(
                grad.d_color[0],
                grad.d_color[1],
                grad.d_color[2],
            );

            final_grads.d_opacity_logits[g_idx] += grad.d_opacity_logit_pad[0];

            final_grads.d_mean_px[g_idx] += Vector2::new(
                grad.d_mean_px[0],
                grad.d_mean_px[1],
            );

            final_grads.d_cov_2d[g_idx] += Vector3::new(
                grad.d_cov_2d[0],
                grad.d_cov_2d[1],
                grad.d_cov_2d[2],
            );
        }
    }

    final_grads
}

/// Chain 2D gradients from GPU to 3D Gaussian parameters.
///
/// This takes the 2D gradients (d_mean_px, d_cov_2d) and chains them through
/// the projection operations to get gradients w.r.t. 3D parameters (position,
/// scale, rotation).
///
/// Returns: (d_positions, d_log_scales, d_rot_vecs, d_background)
pub fn chain_2d_to_3d_gradients(
    grads_2d: &GaussianGradients2D,
    gaussians: &[Gaussian],
    camera: &Camera,
) -> (Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vector3<f32>) {
    let num_gaussians = gaussians.len();
    let mut d_positions = vec![Vector3::<f32>::zeros(); num_gaussians];
    let mut d_log_scales = vec![Vector3::<f32>::zeros(); num_gaussians];
    let mut d_rot_vecs = vec![Vector3::<f32>::zeros(); num_gaussians];

    for (gi, gaussian) in gaussians.iter().enumerate() {
        let mut d_point_cam_total = Vector3::<f32>::zeros();

        // Chain rule: d_mean_px -> d_point_cam
        let d_uv = grads_2d.d_mean_px[gi];
        if d_uv != Vector2::zeros() {
            // Compute camera-space position
            let point_cam = camera.world_to_camera(&gaussian.position);
            d_point_cam_total += project_point_grad_point_cam(&point_cam, camera.fx, camera.fy, &d_uv);
        }

        // Chain rule: d_cov_2d -> d_point_cam, d_log_scales, d_rot_vecs
        let d_cov = grads_2d.d_cov_2d[gi];
        if d_cov != Vector3::zeros() {
            let point_cam = camera.world_to_camera(&gaussian.position);
            let gaussian_r = crate::core::quaternion_to_matrix(&gaussian.rotation);
            let log_scale = gaussian.scale;
            let d_sigma2d = Matrix2::new(d_cov.x, d_cov.y, d_cov.y, d_cov.z);

            // d_cov_2d -> d_point_cam (via Jacobian dependence)
            d_point_cam_total += project_covariance_2d_grad_point_cam(
                &point_cam,
                camera.fx,
                camera.fy,
                &camera.rotation,
                &gaussian_r,
                &log_scale,
                &d_sigma2d,
            );

            // d_cov_2d -> d_log_scales
            let j = camera.projection_jacobian(&point_cam);
            d_log_scales[gi] = project_covariance_2d_grad_log_scale(
                &camera.rotation,
                &j,
                &gaussian_r,
                &log_scale,
                &d_sigma2d,
            );

            // d_cov_2d -> d_rot_vecs
            d_rot_vecs[gi] = project_covariance_2d_grad_rotation_vector_at_r0(
                &camera.rotation,
                &j,
                &gaussian_r,
                &log_scale,
                &d_sigma2d,
            );
        }

        // d_point_cam -> d_position (via camera rotation transpose)
        // point_cam = R * p_world + t  =>  dL/dp_world = R^T * dL/dpoint_cam
        d_positions[gi] = camera.rotation.transpose() * d_point_cam_total;
    }

    (d_positions, d_log_scales, d_rot_vecs, grads_2d.d_background)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_gradients_simple() {
        // 2 workgroups, 3 Gaussians
        let num_workgroups = 2;
        let num_gaussians = 3;

        let mut wg_grads = vec![GradientGPU::zero(); num_workgroups * num_gaussians];

        // Workgroup 0: gradient for Gaussian 1
        wg_grads[0 * num_gaussians + 1].d_color = [1.0, 2.0, 3.0, 0.0];
        wg_grads[0 * num_gaussians + 1].d_opacity_logit_pad = [0.5, 0.0, 0.0, 0.0];

        // Workgroup 1: gradient for Gaussian 1 (should sum)
        wg_grads[1 * num_gaussians + 1].d_color = [4.0, 5.0, 6.0, 0.0];
        wg_grads[1 * num_gaussians + 1].d_opacity_logit_pad = [1.5, 0.0, 0.0, 0.0];

        let result = reduce_workgroup_gradients(&wg_grads, num_workgroups, num_gaussians);

        // Gaussian 1 should have sum of both workgroup gradients
        assert_eq!(result.d_colors[1], Vector3::new(5.0, 7.0, 9.0));
        assert_eq!(result.d_opacity_logits[1], 2.0);

        // Other Gaussians should be zero
        assert_eq!(result.d_colors[0], Vector3::zeros());
        assert_eq!(result.d_colors[2], Vector3::zeros());
    }
}
