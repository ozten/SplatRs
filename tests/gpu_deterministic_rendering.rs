//! Test that GPU rendering is deterministic (same input → same output)

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};

#[test]
#[cfg(feature = "gpu")]
fn test_deterministic_rendering() {
    use sugar_rs::gpu::GpuRenderer;

    println!("\n=== Deterministic Rendering Test ===\n");

    // Simple scene
    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 5.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        0.0,
        {
            let mut sh = [[0.0f32; 3]; 16];
            sh[0] = [1.0, 0.0, 0.0]; // Red Gaussian
            sh
        },
    );

    let camera = Camera::new(
        100.0, 100.0,
        32.0, 32.0,
        64, 64,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let background = Vector3::new(0.5, 0.5, 0.5);

    let gpu_renderer = match GpuRenderer::new() {
        Ok(r) => r,
        Err(e) => {
            println!("Skipping GPU deterministic test (no adapter/device): {e}");
            return;
        }
    };

    // Render 10 times
    let mut renders = Vec::new();
    for i in 0..10 {
        let pixels = gpu_renderer
            .render(&[gaussian.clone()], &camera, &background)
            .expect("GPU render failed");
        renders.push(pixels);
        println!("Render {}: first pixel = {:?}", i, renders[i][0]);
    }

    // All renders must be identical
    for i in 1..10 {
        assert_eq!(renders[0].len(), renders[i].len());
        for (px_idx, (px0, pxi)) in renders[0].iter().zip(&renders[i]).enumerate() {
            let diff = (px0 - pxi).norm();
            assert!(
                diff < 1e-6,
                "Render {} differs from render 0 at pixel {}: {:?} vs {:?} (diff={})",
                i, px_idx, px0, pxi, diff
            );
        }
    }

    println!("✅ All 10 renders are identical!");
}

#[test]
#[cfg(feature = "gpu")]
fn test_background_rendering() {
    use sugar_rs::gpu::GpuRenderer;

    println!("\n=== Background Rendering Test ===\n");

    // Gaussian far outside view frustum - won't contribute to any pixels
    let far_gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, -1000.0), // Very far behind camera
        Vector3::new(-10.0, -10.0, -10.0), // Small scale
        UnitQuaternion::identity(),
        0.0,
        {
            let mut sh = [[0.0f32; 3]; 16];
            sh[0] = [1.0, 0.0, 0.0];
            sh
        },
    );

    let camera = Camera::new(
        100.0, 100.0,
        32.0, 32.0,
        64, 64,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let background = Vector3::new(0.7, 0.3, 0.1);

    let gpu_renderer = match GpuRenderer::new() {
        Ok(r) => r,
        Err(e) => {
            println!("Skipping GPU background test (no adapter/device): {e}");
            return;
        }
    };
    let pixels = gpu_renderer
        .render(&[far_gaussian], &camera, &background)
        .expect("GPU render failed");

    println!("Testing {} pixels for background color {:?}", pixels.len(), background);

    // All pixels should be background color
    for (i, px) in pixels.iter().enumerate() {
        let diff = (px - background).norm();
        assert!(
            diff < 1e-3,
            "Pixel {} should be background color: expected {:?}, got {:?}",
            i, background, px
        );
    }

    println!("✅ Background rendering correct!");
}
