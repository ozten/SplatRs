//! GPU type conversion tests
//!
//! These tests verify that CPU types (Gaussian, Camera) convert correctly to GPU types
//! (GaussianGPU, CameraGPU). These tests don't require GPU hardware - they validate
//! data structure conversion logic only.

use sugar_rs::{Gaussian, Camera};
use nalgebra::{Vector3, Matrix3, UnitQuaternion};

#[cfg(feature = "gpu")]
use sugar_rs::gpu::{GaussianGPU, CameraGPU, GradientGPU};

#[cfg(feature = "gpu")]
#[test]
fn test_gaussian_gpu_conversion_preserves_data() {
    // Create a test Gaussian with known values
    let position = Vector3::new(1.0, 2.0, 3.0);
    let scale = Vector3::new(0.1, 0.2, 0.3); // Log-space scale
    let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    let opacity = 0.8; // Logit-space opacity

    // Create SH coefficients (16 RGB triplets)
    let mut sh_coeffs = [[0.0f32; 3]; 16];
    sh_coeffs[0] = [0.5, 0.6, 0.7]; // DC component (base color)
    sh_coeffs[1] = [0.1, 0.2, 0.3]; // First SH band

    let gaussian = Gaussian {
        position,
        scale,
        rotation,
        opacity,
        sh_coeffs,
    };

    // Convert to GPU format
    let gpu_gaussian = GaussianGPU::from_gaussian(&gaussian);

    // Verify position (xyz + padding)
    assert_eq!(gpu_gaussian.position[0], 1.0);
    assert_eq!(gpu_gaussian.position[1], 2.0);
    assert_eq!(gpu_gaussian.position[2], 3.0);
    assert_eq!(gpu_gaussian.position[3], 0.0); // padding

    // Verify scale (xyz + padding)
    assert_eq!(gpu_gaussian.scale[0], 0.1);
    assert_eq!(gpu_gaussian.scale[1], 0.2);
    assert_eq!(gpu_gaussian.scale[2], 0.3);
    assert_eq!(gpu_gaussian.scale[3], 0.0); // padding

    // Verify rotation (quaternion uploaded as w,i,j,k for WGSL shader)
    assert_eq!(gpu_gaussian.rotation[0], rotation.w);
    assert_eq!(gpu_gaussian.rotation[1], rotation.i);
    assert_eq!(gpu_gaussian.rotation[2], rotation.j);
    assert_eq!(gpu_gaussian.rotation[3], rotation.k);

    // Verify opacity (scalar + 3 padding)
    assert_eq!(gpu_gaussian.opacity_pad[0], 0.8);
    assert_eq!(gpu_gaussian.opacity_pad[1], 0.0); // padding
    assert_eq!(gpu_gaussian.opacity_pad[2], 0.0); // padding
    assert_eq!(gpu_gaussian.opacity_pad[3], 0.0); // padding

    // Verify SH coefficients
    // DC component (index 0)
    assert_eq!(gpu_gaussian.sh_coeffs[0][0], 0.5); // R
    assert_eq!(gpu_gaussian.sh_coeffs[0][1], 0.6); // G
    assert_eq!(gpu_gaussian.sh_coeffs[0][2], 0.7); // B
    assert_eq!(gpu_gaussian.sh_coeffs[0][3], 0.0); // padding

    // First SH band (index 1)
    assert_eq!(gpu_gaussian.sh_coeffs[1][0], 0.1); // R
    assert_eq!(gpu_gaussian.sh_coeffs[1][1], 0.2); // G
    assert_eq!(gpu_gaussian.sh_coeffs[1][2], 0.3); // B
    assert_eq!(gpu_gaussian.sh_coeffs[1][3], 0.0); // padding

    // Verify other SH coefficients are zero
    for i in 2..16 {
        assert_eq!(gpu_gaussian.sh_coeffs[i][0], 0.0);
        assert_eq!(gpu_gaussian.sh_coeffs[i][1], 0.0);
        assert_eq!(gpu_gaussian.sh_coeffs[i][2], 0.0);
        assert_eq!(gpu_gaussian.sh_coeffs[i][3], 0.0);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_camera_gpu_conversion() {
    // Create a test camera
    let fx = 800.0;
    let fy = 800.0;
    let cx = 400.0;
    let cy = 300.0;
    let width = 800;
    let height = 600;

    // Rotation matrix (identity for simplicity)
    let rotation = Matrix3::identity();
    let translation = Vector3::new(0.0, 0.0, 5.0);

    let camera = Camera::new(
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        rotation,
        translation,
    );

    // Convert to GPU format
    let gpu_camera = CameraGPU::from_camera(&camera);

    // Verify focal parameters (fx, fy, cx, cy)
    assert_eq!(gpu_camera.focal[0], 800.0);
    assert_eq!(gpu_camera.focal[1], 800.0);
    assert_eq!(gpu_camera.focal[2], 400.0);
    assert_eq!(gpu_camera.focal[3], 300.0);

    // Verify dimensions (width, height, padding, padding)
    assert_eq!(gpu_camera.dims[0], 800);
    assert_eq!(gpu_camera.dims[1], 600);
    assert_eq!(gpu_camera.dims[2], 0); // padding
    assert_eq!(gpu_camera.dims[3], 0); // padding

    // Verify rotation matrix (identity)
    // Row 0
    assert_eq!(gpu_camera.rotation[0][0], 1.0);
    assert_eq!(gpu_camera.rotation[0][1], 0.0);
    assert_eq!(gpu_camera.rotation[0][2], 0.0);
    assert_eq!(gpu_camera.rotation[0][3], 0.0); // padding

    // Row 1
    assert_eq!(gpu_camera.rotation[1][0], 0.0);
    assert_eq!(gpu_camera.rotation[1][1], 1.0);
    assert_eq!(gpu_camera.rotation[1][2], 0.0);
    assert_eq!(gpu_camera.rotation[1][3], 0.0); // padding

    // Row 2
    assert_eq!(gpu_camera.rotation[2][0], 0.0);
    assert_eq!(gpu_camera.rotation[2][1], 0.0);
    assert_eq!(gpu_camera.rotation[2][2], 1.0);
    assert_eq!(gpu_camera.rotation[2][3], 0.0); // padding

    // Verify translation
    assert_eq!(gpu_camera.translation[0], 0.0);
    assert_eq!(gpu_camera.translation[1], 0.0);
    assert_eq!(gpu_camera.translation[2], 5.0);
    assert_eq!(gpu_camera.translation[3], 0.0); // padding
}

#[cfg(feature = "gpu")]
#[test]
fn test_gradient_gpu_zero_initialization() {
    let grad = GradientGPU::zero();

    // Verify all color gradient components are zero
    for i in 0..4 {
        assert_eq!(grad.d_color[i], 0.0, "d_color[{}] should be 0.0", i);
    }

    // Verify all opacity gradient components are zero
    for i in 0..4 {
        assert_eq!(grad.d_opacity_logit_pad[i], 0.0, "d_opacity_logit_pad[{}] should be 0.0", i);
    }

    // Verify all 2D mean gradient components are zero
    for i in 0..4 {
        assert_eq!(grad.d_mean_px[i], 0.0, "d_mean_px[{}] should be 0.0", i);
    }

    // Verify all 2D covariance gradient components are zero
    for i in 0..4 {
        assert_eq!(grad.d_cov_2d[i], 0.0, "d_cov_2d[{}] should be 0.0", i);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_gaussian_gpu_conversion_edge_cases() {
    // Test with zero values (except rotation must be valid unit quaternion)
    let zero_gaussian = Gaussian {
        position: Vector3::zeros(),
        scale: Vector3::zeros(),
        rotation: UnitQuaternion::identity(), // Must be valid unit quaternion
        opacity: 0.0,
        sh_coeffs: [[0.0; 3]; 16],
    };

    let gpu_zero = GaussianGPU::from_gaussian(&zero_gaussian);

    // All non-padding position/scale/opacity should be 0.0
    assert_eq!(gpu_zero.position[0], 0.0);
    assert_eq!(gpu_zero.position[1], 0.0);
    assert_eq!(gpu_zero.position[2], 0.0);
    assert_eq!(gpu_zero.scale[0], 0.0);
    assert_eq!(gpu_zero.scale[1], 0.0);
    assert_eq!(gpu_zero.scale[2], 0.0);
    assert_eq!(gpu_zero.opacity_pad[0], 0.0);

    // Test with large values
    let large_gaussian = Gaussian {
        position: Vector3::new(1e6, 1e6, 1e6),
        scale: Vector3::new(10.0, 10.0, 10.0), // Large in log-space
        rotation: UnitQuaternion::from_euler_angles(1.0, 1.0, 1.0),
        opacity: 5.0, // Large in logit-space
        sh_coeffs: [[1.0; 3]; 16],
    };

    let gpu_large = GaussianGPU::from_gaussian(&large_gaussian);

    // Large values should be preserved
    assert_eq!(gpu_large.position[0], 1e6);
    assert_eq!(gpu_large.position[1], 1e6);
    assert_eq!(gpu_large.position[2], 1e6);
    assert_eq!(gpu_large.scale[0], 10.0);
    assert_eq!(gpu_large.opacity_pad[0], 5.0);

    // All SH coefficients should be 1.0
    for i in 0..16 {
        assert_eq!(gpu_large.sh_coeffs[i][0], 1.0);
        assert_eq!(gpu_large.sh_coeffs[i][1], 1.0);
        assert_eq!(gpu_large.sh_coeffs[i][2], 1.0);
    }
}
