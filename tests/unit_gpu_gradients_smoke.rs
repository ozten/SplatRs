//! GPU vs CPU gradient sanity check on a tiny scene.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::render_full_color_grads;

#[cfg(feature = "gpu")]
use sugar_rs::gpu::GpuRenderer;

const SH_C0: f32 = 0.282_094_791_773_878_14;

fn sh_constant_color(rgb: Vector3<f32>) -> [[f32; 3]; 16] {
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [rgb.x / SH_C0, rgb.y / SH_C0, rgb.z / SH_C0];
    sh
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_vs_cpu_gradients_toy_scene() {
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

    let (_cpu_img, cpu_d_colors, cpu_d_opacity, _cpu_d_pos, _cpu_d_scale, _cpu_d_rot, _cpu_d_bg) =
        render_full_color_grads(&gaussians, &camera, &d_pixels, &bg, false);

    let (_gpu_img, gpu_grads) = GpuRenderer::new()
        .expect("Failed to initialize GPU")
        .render_with_gradients(&gaussians, &camera, &bg, &d_pixels)
        .expect("GPU render_with_gradients failed");

    let mut max_color_diff = 0.0f32;
    let mut max_opacity_diff = 0.0f32;
    for i in 0..gaussians.len() {
        let color_diff = (cpu_d_colors[i] - gpu_grads.d_colors[i]).abs().max();
        max_color_diff = max_color_diff.max(color_diff);

        let opacity_diff = (cpu_d_opacity[i] - gpu_grads.d_opacity_logits[i]).abs();
        max_opacity_diff = max_opacity_diff.max(opacity_diff);
    }

    assert!(max_color_diff <= 1e-2, "color grad max diff {:.6} > 1e-2", max_color_diff);
    assert!(max_opacity_diff <= 1e-2, "opacity grad max diff {:.6} > 1e-2", max_opacity_diff);
}
