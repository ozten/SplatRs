//! Small GPU vs CPU render sanity check.

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
fn test_gpu_vs_cpu_toy_scene() {
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
            1.0,
            sh_constant_color(Vector3::new(1.0, 0.2, 0.1)),
        ),
        Gaussian::new(
            Vector3::new(0.5, 0.2, 3.0),
            Vector3::new(-2.2, -2.2, -2.2),
            UnitQuaternion::identity(),
            0.5,
            sh_constant_color(Vector3::new(0.1, 0.8, 0.2)),
        ),
    ];

    let bg = Vector3::new(0.01, 0.02, 0.03);
    let cpu = render_full_linear(&gaussians, &camera, &bg, false);

    let gpu = GpuRenderer::new()
        .expect("Failed to initialize GPU")
        .render(&gaussians, &camera, &bg)
        .expect("GPU render failed");

    assert_eq!(cpu.len(), gpu.len());

    let mut max_diff = 0.0f32;
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        let diff = (c - g).abs().max();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert!(max_diff <= 0.01, "GPU/CPU max diff {:.6} > 0.01", max_diff);
}
