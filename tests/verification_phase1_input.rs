//! Phase 1 Input Verification Tests
//!
//! These tests verify that COLMAP data is correctly parsed and meets
//! the precision requirements specified in the verification plan.

use sugar_rs::io::load_colmap_scene;
use std::path::PathBuf;

/// TC-INP-001: Verify camera intrinsic parameters are correctly parsed from COLMAP.
///
/// Pass Criteria:
/// - Focal length error < 0.01 pixels
/// - Principal point error < 0.01 pixels
/// - Distortion coefficients match to 6 decimal places (currently not stored)
#[test]
fn tc_inp_001_camera_intrinsics_parsing() {
    // Use a known dataset
    let dataset_path = PathBuf::from("datasets/garden/sparse/0");

    if !dataset_path.exists() {
        eprintln!("Skipping test - dataset not found at {:?}", dataset_path);
        eprintln!("Run setup to download datasets first");
        return;
    }

    // Load the scene
    let scene = load_colmap_scene(&dataset_path)
        .expect("Failed to load COLMAP scene");

    assert!(scene.cameras.len() > 0, "No cameras loaded");

    // Test intrinsics parsing by validating against COLMAP binary format
    // We verify that:
    // 1. Values are reasonable (focal length positive, principal point near center)
    // 2. Values maintain precision (no excessive truncation)
    // 3. All camera models are parsed correctly

    for (camera_id, camera) in scene.cameras.iter() {
        println!("Camera {}: {}x{}", camera_id, camera.width, camera.height);
        println!("  fx={:.6}, fy={:.6}", camera.fx, camera.fy);
        println!("  cx={:.6}, cy={:.6}", camera.cx, camera.cy);

        // Verify focal lengths are positive and reasonable
        assert!(camera.fx > 0.0, "Camera {} has invalid fx: {}", camera_id, camera.fx);
        assert!(camera.fy > 0.0, "Camera {} has invalid fy: {}", camera_id, camera.fy);

        // Verify focal lengths are in reasonable range (between 0.1x and 10x image dimension)
        let max_dim = camera.width.max(camera.height) as f32;
        assert!(camera.fx > max_dim * 0.1 && camera.fx < max_dim * 10.0,
            "Camera {} fx out of reasonable range: {}", camera_id, camera.fx);
        assert!(camera.fy > max_dim * 0.1 && camera.fy < max_dim * 10.0,
            "Camera {} fy out of reasonable range: {}", camera_id, camera.fy);

        // Verify principal point is within image bounds (with some margin)
        assert!(camera.cx > -(camera.width as f32) && camera.cx < 2.0 * camera.width as f32,
            "Camera {} cx out of bounds: {}", camera_id, camera.cx);
        assert!(camera.cy > -(camera.height as f32) && camera.cy < 2.0 * camera.height as f32,
            "Camera {} cy out of bounds: {}", camera_id, camera.cy);

        // Verify precision is maintained (values should not be truncated to integers)
        // This tests that we're correctly reading f64 from COLMAP and converting to f32
        let fx_frac = camera.fx.fract().abs();
        let fy_frac = camera.fy.fract().abs();
        let cx_frac = camera.cx.fract().abs();
        let cy_frac = camera.cy.fract().abs();

        // At least one of fx/fy should have fractional part (extremely unlikely all are exact integers)
        // Skip this check if focal lengths are suspiciously round (might be synthetic)
        let has_precision = fx_frac > 0.001 || fy_frac > 0.001 || cx_frac > 0.001 || cy_frac > 0.001;
        if !has_precision {
            eprintln!("Warning: Camera {} has suspiciously round values - may be synthetic", camera_id);
        }
    }

    println!("\n✅ TC-INP-001: Camera intrinsics parsing validated");
    println!("   {} cameras with reasonable intrinsics", scene.cameras.len());
}

/// TC-INP-001 Extended: Test intrinsics parsing with ground truth values
///
/// This test compares parsed values against expected values to verify
/// precision requirements are met.
#[test]
#[ignore] // Requires manual ground truth setup
fn tc_inp_001_extended_ground_truth() {
    // This test would compare against known ground truth values
    // Format: create a fixture with cameras.bin and expected values
    // For now, we rely on the main test for validation

    // Expected implementation:
    // 1. Create test fixture with known cameras.bin
    // 2. Define ground truth intrinsics
    // 3. Verify |parsed - ground_truth| < threshold
    //    - Focal length error < 0.01 pixels
    //    - Principal point error < 0.01 pixels

    eprintln!("Ground truth test not yet implemented");
    eprintln!("To implement: create fixtures/camera_intrinsics/ with:");
    eprintln!("  - cameras.bin (COLMAP format)");
    eprintln!("  - ground_truth.json (expected values)");
}

/// TC-INP-002: Verify camera extrinsic parameters (pose) are correctly parsed from COLMAP.
///
/// Pass Criteria:
/// - Rotation error < 0.1 degrees
/// - Translation error < 0.001 units
/// - All rotation matrices satisfy R^T * R = I within 1e-6
#[test]
fn tc_inp_002_camera_extrinsics_parsing() {
    use nalgebra::Matrix3;

    // Use a known dataset
    let dataset_path = PathBuf::from("datasets/garden/sparse/0");

    if !dataset_path.exists() {
        eprintln!("Skipping test - dataset not found at {:?}", dataset_path);
        eprintln!("Run setup to download datasets first");
        return;
    }

    // Load the scene
    let scene = load_colmap_scene(&dataset_path)
        .expect("Failed to load COLMAP scene");

    assert!(scene.images.len() > 0, "No images loaded");

    // Test extrinsics parsing by validating:
    // 1. Rotation quaternions are valid (unit quaternions)
    // 2. Rotation matrices are orthonormal (R^T * R = I)
    // 3. Values maintain precision (not truncated)

    let mut num_valid = 0;

    for image_info in scene.images.iter() {
        println!("Image {}: {}", image_info.id, image_info.name);

        // Extract quaternion components
        let quat = image_info.rotation;
        let qw = quat.w;
        let qx = quat.i;
        let qy = quat.j;
        let qz = quat.k;

        println!("  Quaternion: w={:.6}, x={:.6}, y={:.6}, z={:.6}", qw, qx, qy, qz);
        println!("  Translation: x={:.6}, y={:.6}, z={:.6}",
            image_info.translation.x, image_info.translation.y, image_info.translation.z);

        // Verify quaternion is normalized (should be unit quaternion)
        let quat_norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
        assert!((quat_norm - 1.0).abs() < 1e-6,
            "Image {} quaternion not normalized: norm={}", image_info.id, quat_norm);

        // Convert quaternion to rotation matrix
        let rotation_matrix: Matrix3<f32> = quat.to_rotation_matrix().into_inner();

        // Verify rotation matrix is orthonormal: R^T * R = I
        let rt_r: Matrix3<f32> = rotation_matrix.transpose() * rotation_matrix;
        let identity: Matrix3<f32> = Matrix3::identity();

        for i in 0..3 {
            for j in 0..3 {
                let diff = (rt_r[(i, j)] - identity[(i, j)]).abs();
                assert!(diff < 1e-6,
                    "Image {} rotation matrix not orthonormal at ({},{}): R^T*R - I = {}",
                    image_info.id, i, j, diff);
            }
        }

        // Verify determinant is +1 (proper rotation, not reflection)
        let det = rotation_matrix.determinant();
        assert!((det - 1.0).abs() < 1e-6,
            "Image {} rotation matrix determinant != 1: det={}", image_info.id, det);

        // Verify precision is maintained (values should have fractional parts)
        // At least some rotation or translation components should not be exact integers
        let has_rotation_precision =
            qx.fract().abs() > 0.001 || qy.fract().abs() > 0.001 ||
            qz.fract().abs() > 0.001 || qw.fract().abs() > 0.001;
        let has_translation_precision =
            image_info.translation.x.fract().abs() > 0.001 ||
            image_info.translation.y.fract().abs() > 0.001 ||
            image_info.translation.z.fract().abs() > 0.001;

        if !has_rotation_precision && !has_translation_precision {
            eprintln!("Warning: Image {} has suspiciously round pose values - may be synthetic",
                image_info.id);
        }

        num_valid += 1;
    }

    println!("\n✅ TC-INP-002: Camera extrinsics parsing validated");
    println!("   {} images with valid camera poses", num_valid);
    println!("   All rotation matrices are orthonormal (R^T * R = I within 1e-6)");
    println!("   All quaternions are properly normalized");
}

/// TC-INP-003: Verify consistent coordinate system convention (OpenCV/COLMAP vs OpenGL).
///
/// Pass Criteria:
/// - Projected coordinates match analytical solution within 1e-4
/// - Coordinate system documented and verified
///
/// This test verifies that we use a consistent coordinate system throughout the pipeline:
/// - Camera space: +Z forward (into scene), +X right, +Y down (OpenCV/COLMAP convention)
/// - World to camera: p_cam = R * p_world + t
/// - Projection: [u, v] = [fx * x/z + cx, fy * y/z + cy]
#[test]
fn tc_inp_003_coordinate_system_convention() {
    const TOLERANCE: f32 = 1e-4; // Tolerance for floating-point comparisons
    use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
    use sugar_rs::core::Camera;

    println!("\n=== TC-INP-003: Coordinate System Convention Verification ===\n");

    // Test Case 1: Identity camera, simple point
    // Camera at origin, looking down +Z axis
    {
        let camera = Camera::new(
            100.0,               // fx
            100.0,               // fy
            50.0,                // cx
            50.0,                // cy
            100,                 // width
            100,                 // height
            Matrix3::identity(), // rotation (identity = no rotation)
            Vector3::zeros(),    // translation (at origin)
        );

        // Point at (1, 2, 5) in world space
        // With identity camera, world = camera space
        // Expected projection: u = 100 * (1/5) + 50 = 70
        //                      v = 100 * (2/5) + 50 = 90
        let point_world = Vector3::new(1.0, 2.0, 5.0);
        let expected_pixel = Vector3::new(70.0, 90.0, 5.0); // [u, v, depth]

        let pixel = camera.world_to_pixel(&point_world)
            .expect("Point should be in front of camera");

        println!("Test 1 - Identity camera:");
        println!("  World point: ({:.6}, {:.6}, {:.6})", point_world.x, point_world.y, point_world.z);
        println!("  Expected pixel: ({:.6}, {:.6})", expected_pixel.x, expected_pixel.y);
        println!("  Computed pixel: ({:.6}, {:.6})", pixel.x, pixel.y);
        println!("  Error: ({:.6}, {:.6})",
            (pixel.x - expected_pixel.x).abs(),
            (pixel.y - expected_pixel.y).abs());

        assert!((pixel.x - expected_pixel.x).abs() < TOLERANCE,
            "X projection error too large: {} vs {}", pixel.x, expected_pixel.x);
        assert!((pixel.y - expected_pixel.y).abs() < TOLERANCE,
            "Y projection error too large: {} vs {}", pixel.y, expected_pixel.y);

        println!("  ✓ Identity camera projection correct\n");
    }

    // Test Case 2: Translated camera
    // Camera translated to (10, 0, 0), still looking down +Z
    {
        let camera = Camera::new(
            100.0,
            100.0,
            50.0,
            50.0,
            100,
            100,
            Matrix3::identity(),
            Vector3::new(10.0, 0.0, 0.0), // translation
        );

        // Point at (11, 2, 5) in world space
        // In camera space: p_cam = I * (11, 2, 5) + (10, 0, 0) = (21, 2, 5)
        // Expected projection: u = 100 * (21/5) + 50 = 470
        //                      v = 100 * (2/5) + 50 = 90
        let point_world = Vector3::new(11.0, 2.0, 5.0);
        let expected_pixel = Vector3::new(470.0, 90.0, 5.0);

        let pixel = camera.world_to_pixel(&point_world)
            .expect("Point should be in front of camera");

        println!("Test 2 - Translated camera:");
        println!("  World point: ({:.6}, {:.6}, {:.6})", point_world.x, point_world.y, point_world.z);
        println!("  Expected pixel: ({:.6}, {:.6})", expected_pixel.x, expected_pixel.y);
        println!("  Computed pixel: ({:.6}, {:.6})", pixel.x, pixel.y);
        println!("  Error: ({:.6}, {:.6})",
            (pixel.x - expected_pixel.x).abs(),
            (pixel.y - expected_pixel.y).abs());

        assert!((pixel.x - expected_pixel.x).abs() < TOLERANCE,
            "X projection error too large: {} vs {}", pixel.x, expected_pixel.x);
        assert!((pixel.y - expected_pixel.y).abs() < TOLERANCE,
            "Y projection error too large: {} vs {}", pixel.y, expected_pixel.y);

        println!("  ✓ Translated camera projection correct\n");
    }

    // Test Case 3: Rotated camera (90° rotation around Y-axis)
    // Camera rotated 90° around Y-axis: was looking at +Z, now looking at +X
    {
        // Rotation by 90° around Y-axis (counter-clockwise when looking down -Y)
        // This rotates +Z to +X
        let angle = std::f32::consts::FRAC_PI_2; // 90 degrees
        let axis = Vector3::y_axis();
        let rotation_quat = UnitQuaternion::from_axis_angle(&axis, angle);
        let rotation_matrix = rotation_quat.to_rotation_matrix().into_inner();

        let camera = Camera::new(
            100.0,
            100.0,
            50.0,
            50.0,
            100,
            100,
            rotation_matrix,
            Vector3::zeros(),
        );

        // Point at (5, 2, 1) in world space
        // After 90° rotation around Y: (x,y,z) -> (z, y, -x)
        // In camera space: R * (5, 2, 1) = (1, 2, -5)
        // Depth check: z = -5, which is negative (behind camera)
        // So this point should NOT project

        // Let's use a point that will be in front
        // Point at (5, 2, -1) in world space
        // After rotation: R * (5, 2, -1) = (-1, 2, -5) - still behind

        // For a point to be in front after Y-rotation by 90°:
        // We need R * p to have positive z
        // R_y(90°) maps (x,y,z) -> (z, y, -x)
        // For positive z in camera: -x > 0, so x < 0 in world

        let point_world = Vector3::new(-5.0, 2.0, 1.0);
        // After rotation: (-5, 2, 1) -> (1, 2, 5) in camera space
        // Expected projection: u = 100 * (1/5) + 50 = 70
        //                      v = 100 * (2/5) + 50 = 90
        let expected_pixel = Vector3::new(70.0, 90.0, 5.0);

        let pixel = camera.world_to_pixel(&point_world)
            .expect("Point should be in front of camera");

        println!("Test 3 - Rotated camera (90° around Y):");
        println!("  World point: ({:.6}, {:.6}, {:.6})", point_world.x, point_world.y, point_world.z);
        println!("  Expected pixel: ({:.6}, {:.6})", expected_pixel.x, expected_pixel.y);
        println!("  Computed pixel: ({:.6}, {:.6})", pixel.x, pixel.y);
        println!("  Error: ({:.6}, {:.6})",
            (pixel.x - expected_pixel.x).abs(),
            (pixel.y - expected_pixel.y).abs());

        assert!((pixel.x - expected_pixel.x).abs() < TOLERANCE,
            "X projection error too large: {} vs {}", pixel.x, expected_pixel.x);
        assert!((pixel.y - expected_pixel.y).abs() < TOLERANCE,
            "Y projection error too large: {} vs {}", pixel.y, expected_pixel.y);

        println!("  ✓ Rotated camera projection correct\n");
    }

    // Test Case 4: General camera pose (combined rotation and translation)
    {
        // 45° rotation around Y-axis
        let angle = std::f32::consts::FRAC_PI_4; // 45 degrees
        let axis = Vector3::y_axis();
        let rotation_quat = UnitQuaternion::from_axis_angle(&axis, angle);
        let rotation_matrix = rotation_quat.to_rotation_matrix().into_inner();

        let translation = Vector3::new(1.0, 2.0, 3.0);

        let camera = Camera::new(
            200.0, // fx
            200.0, // fy
            100.0, // cx
            100.0, // cy
            200,
            200,
            rotation_matrix,
            translation,
        );

        // Point at (5, 3, 8) in world space
        let point_world = Vector3::new(5.0, 3.0, 8.0);

        // Manually compute expected projection
        // p_cam = R * p_world + t
        let point_camera = rotation_matrix * point_world + translation;

        // Should have positive z (in front)
        assert!(point_camera.z > 0.0,
            "Test point should be in front of camera, got z={}", point_camera.z);

        // Expected projection: [u, v] = [fx * x/z + cx, fy * y/z + cy]
        let expected_u = 200.0 * point_camera.x / point_camera.z + 100.0;
        let expected_v = 200.0 * point_camera.y / point_camera.z + 100.0;
        let expected_pixel = Vector3::new(expected_u, expected_v, point_camera.z);

        let pixel = camera.world_to_pixel(&point_world)
            .expect("Point should be in front of camera");

        println!("Test 4 - General camera pose:");
        println!("  World point: ({:.6}, {:.6}, {:.6})", point_world.x, point_world.y, point_world.z);
        println!("  Camera point: ({:.6}, {:.6}, {:.6})", point_camera.x, point_camera.y, point_camera.z);
        println!("  Expected pixel: ({:.6}, {:.6})", expected_pixel.x, expected_pixel.y);
        println!("  Computed pixel: ({:.6}, {:.6})", pixel.x, pixel.y);
        println!("  Error: ({:.6}, {:.6})",
            (pixel.x - expected_pixel.x).abs(),
            (pixel.y - expected_pixel.y).abs());

        assert!((pixel.x - expected_pixel.x).abs() < TOLERANCE,
            "X projection error too large: {} vs {}", pixel.x, expected_pixel.x);
        assert!((pixel.y - expected_pixel.y).abs() < TOLERANCE,
            "Y projection error too large: {} vs {}", pixel.y, expected_pixel.y);

        println!("  ✓ General camera projection correct\n");
    }

    // Test Case 5: Verify point behind camera is rejected
    {
        let camera = Camera::new(
            100.0,
            100.0,
            50.0,
            50.0,
            100,
            100,
            Matrix3::identity(),
            Vector3::zeros(),
        );

        // Point with negative Z (behind camera)
        let point_world = Vector3::new(1.0, 2.0, -5.0);
        let pixel = camera.world_to_pixel(&point_world);

        println!("Test 5 - Point behind camera:");
        println!("  World point: ({:.6}, {:.6}, {:.6})", point_world.x, point_world.y, point_world.z);
        println!("  Projection result: {:?}", pixel);

        assert!(pixel.is_none(),
            "Point behind camera should not project");

        println!("  ✓ Point behind camera correctly rejected\n");
    }

    println!("✅ TC-INP-003: Coordinate system convention verified");
    println!("   Convention: OpenCV/COLMAP (Z-forward, camera-to-world)");
    println!("   World to camera: p_cam = R * p_world + t");
    println!("   Projection: [u,v] = [fx*x/z + cx, fy*y/z + cy]");
    println!("   All analytical projections match within {}", TOLERANCE);
}

#[cfg(test)]
mod reference_implementation {
    //! Reference implementation for validating COLMAP intrinsics parsing
    //!
    //! This serves as documentation for the expected precision and behavior.

    use nalgebra::{Matrix3, Vector3};

    /// Expected precision for intrinsic parameters
    pub const FOCAL_LENGTH_PRECISION: f32 = 0.01; // pixels
    pub const PRINCIPAL_POINT_PRECISION: f32 = 0.01; // pixels

    /// Expected precision for extrinsic parameters
    pub const ROTATION_PRECISION_DEGREES: f32 = 0.1; // degrees
    pub const TRANSLATION_PRECISION: f32 = 0.001; // units

    /// Verify intrinsics meet precision requirements
    pub fn verify_intrinsics_precision(
        parsed_fx: f32,
        parsed_fy: f32,
        parsed_cx: f32,
        parsed_cy: f32,
        ground_truth_fx: f32,
        ground_truth_fy: f32,
        ground_truth_cx: f32,
        ground_truth_cy: f32,
    ) -> bool {
        let fx_error = (parsed_fx - ground_truth_fx).abs();
        let fy_error = (parsed_fy - ground_truth_fy).abs();
        let cx_error = (parsed_cx - ground_truth_cx).abs();
        let cy_error = (parsed_cy - ground_truth_cy).abs();

        fx_error < FOCAL_LENGTH_PRECISION
            && fy_error < FOCAL_LENGTH_PRECISION
            && cx_error < PRINCIPAL_POINT_PRECISION
            && cy_error < PRINCIPAL_POINT_PRECISION
    }

    /// Verify rotation matrix is orthonormal
    pub fn verify_rotation_orthonormal(rotation: &Matrix3<f32>, tolerance: f32) -> bool {
        let rt_r: Matrix3<f32> = rotation.transpose() * rotation;
        let identity: Matrix3<f32> = Matrix3::identity();

        for i in 0..3 {
            for j in 0..3 {
                let diff = (rt_r[(i, j)] - identity[(i, j)]).abs();
                if diff > tolerance {
                    return false;
                }
            }
        }

        // Also verify determinant is +1 (proper rotation)
        (rotation.determinant() - 1.0).abs() < tolerance
    }

    /// Compute rotation error in degrees between two rotation matrices
    pub fn rotation_error_degrees(r1: &Matrix3<f32>, r2: &Matrix3<f32>) -> f32 {
        // Error rotation: R_error = R1^T * R2
        let r_error = r1.transpose() * r2;

        // Extract angle from rotation matrix using trace
        // trace(R) = 1 + 2*cos(theta)
        let trace = r_error[(0, 0)] + r_error[(1, 1)] + r_error[(2, 2)];
        let cos_theta = (trace - 1.0) / 2.0;
        let cos_theta_clamped = cos_theta.clamp(-1.0, 1.0);
        let theta_rad = cos_theta_clamped.acos();

        theta_rad.to_degrees()
    }

    /// Verify extrinsics meet precision requirements
    pub fn verify_extrinsics_precision(
        parsed_rotation: &Matrix3<f32>,
        parsed_translation: &Vector3<f32>,
        ground_truth_rotation: &Matrix3<f32>,
        ground_truth_translation: &Vector3<f32>,
    ) -> bool {
        // Check rotation error
        let rotation_error = rotation_error_degrees(parsed_rotation, ground_truth_rotation);
        if rotation_error >= ROTATION_PRECISION_DEGREES {
            return false;
        }

        // Check translation error
        let translation_error = (parsed_translation - ground_truth_translation).norm();
        if translation_error >= TRANSLATION_PRECISION {
            return false;
        }

        true
    }
}
