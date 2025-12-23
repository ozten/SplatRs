//! GPU vs CPU background gradient sanity check.

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
fn test_gpu_background_gradient_matches_cpu() {
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

    let (_cpu_img, _cpu_d_colors, _cpu_d_opacity, _cpu_d_pos, _cpu_d_scale, _cpu_d_rot, cpu_d_bg) =
        render_full_color_grads(&gaussians, &camera, &d_pixels, &bg);

    let (_gpu_img, gpu_grads) = GpuRenderer::new()
        .expect("Failed to initialize GPU")
        .render_with_gradients(&gaussians, &camera, &bg, &d_pixels)
        .expect("GPU render_with_gradients failed");

    let diff = (cpu_d_bg - gpu_grads.d_background).abs().max();
    assert!(diff <= 1e-2, "bg grad max diff {:.6} > 1e-2", diff);
}
