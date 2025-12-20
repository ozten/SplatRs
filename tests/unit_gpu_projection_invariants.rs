//! GPU vs CPU projection invariants on a tiny scene.

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
fn test_gpu_matches_cpu_projection_tiny() {
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

    let g = Gaussian::new(
        Vector3::new(0.3, -0.2, 2.5),
        Vector3::new(-1.5, -2.0, -1.8),
        UnitQuaternion::identity(),
        0.4,
        sh_constant_color(Vector3::new(0.7, 0.3, 0.1)),
    );

    let bg = Vector3::zeros();
    let cpu = render_full_linear(&[g.clone()], &camera, &bg);
    let gpu = GpuRenderer::new()
        .expect("Failed to initialize GPU")
        .render(&[g], &camera, &bg)
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
