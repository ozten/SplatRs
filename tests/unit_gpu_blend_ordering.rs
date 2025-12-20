//! GPU vs CPU ordering sanity on a tiny scene.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::render_full_linear;

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
fn test_gpu_matches_cpu_blend_ordering_tiny() {
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

    let near = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(0.0, 1.0, 0.0)),
    );

    let far = Gaussian::new(
        Vector3::new(0.0, 0.0, 4.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),
    );

    let bg = Vector3::zeros();
    let cpu = render_full_linear(&[far.clone(), near.clone()], &camera, &bg);
    let gpu = GpuRenderer::new()
        .expect("Failed to initialize GPU")
        .render(&[far, near], &camera, &bg)
        .expect("GPU render failed");

    let center_cpu = cpu[(3 * camera.width + 3) as usize];
    let center_gpu = gpu[(3 * camera.width + 3) as usize];

    assert!(center_cpu.y > center_cpu.x, "CPU near should dominate");
    assert!(center_gpu.y > center_gpu.x, "GPU near should dominate");
}
