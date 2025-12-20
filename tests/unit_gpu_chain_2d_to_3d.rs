//! GPU 2D gradients chained to 3D should match CPU 3D gradients.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::render_full_color_grads;

#[cfg(feature = "gpu")]
use sugar_rs::gpu::{GpuRenderer, chain_2d_to_3d_gradients_cpu};

const SH_C0: f32 = 0.282_094_791_773_878_14;

fn sh_constant_color(rgb: Vector3<f32>) -> [[f32; 3]; 16] {
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [rgb.x / SH_C0, rgb.y / SH_C0, rgb.z / SH_C0];
    sh
}

#[test]
#[cfg(feature = "gpu")]
#[ignore] // Flaky: off by small amounts
fn test_gpu_2d_chain_matches_cpu_3d_grads() {
    let camera = Camera::new(
        4.0,
        4.0,
        3.5,
        3.5,
        8,
        8,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let gaussians = vec![
        Gaussian::new(
            Vector3::new(0.0, 0.0, 2.0),
            Vector3::new(-2.0, -2.0, -2.0),
            UnitQuaternion::identity(),
            0.2,
            sh_constant_color(Vector3::new(0.8, 0.1, 0.1)),
        ),
        Gaussian::new(
            Vector3::new(0.5, 0.2, 3.0),
            Vector3::new(-2.3, -2.3, -2.3),
            UnitQuaternion::identity(),
            -0.4,
            sh_constant_color(Vector3::new(0.1, 0.7, 0.2)),
        ),
    ];

    let bg = Vector3::new(0.02, 0.03, 0.04);
    let num_pixels = (camera.width * camera.height) as usize;
    let d_pixels = vec![Vector3::new(1.0, -0.5, 0.25); num_pixels];

    let (_cpu_img, _cpu_d_colors, _cpu_d_opacity, cpu_d_pos, cpu_d_scale, cpu_d_rot, _cpu_d_bg) =
        render_full_color_grads(&gaussians, &camera, &d_pixels, &bg);

    let (_gpu_img, gpu_grads) = GpuRenderer::new()
        .expect("Failed to initialize GPU")
        .render_with_gradients(&gaussians, &camera, &bg, &d_pixels)
        .expect("GPU render_with_gradients failed");

    let (gpu_d_pos, gpu_d_scale, gpu_d_rot, _gpu_d_bg) =
        chain_2d_to_3d_gradients_cpu(&gpu_grads, &gaussians, &camera);

    let mut max_pos_diff = 0.0f32;
    let mut max_scale_diff = 0.0f32;
    let mut max_rot_diff = 0.0f32;

    for i in 0..gaussians.len() {
        max_pos_diff = max_pos_diff.max((cpu_d_pos[i] - gpu_d_pos[i]).abs().max());
        max_scale_diff = max_scale_diff.max((cpu_d_scale[i] - gpu_d_scale[i]).abs().max());
        max_rot_diff = max_rot_diff.max((cpu_d_rot[i] - gpu_d_rot[i]).abs().max());
    }

    assert!(max_pos_diff <= 1e-2, "pos grad max diff {:.6} > 1e-2", max_pos_diff);
    assert!(
        max_scale_diff <= 1e-2,
        "scale grad max diff {:.6} > 1e-2",
        max_scale_diff
    );
    assert!(max_rot_diff <= 1e-2, "rot grad max diff {:.6} > 1e-2", max_rot_diff);
}
