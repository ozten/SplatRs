//! Synthetic Scene Tests for Bug Hunting
//!
//! These tests use purely synthetic scenes (no real data) to isolate and verify
//! specific rendering and optimization behaviors. Each test has known ground truth.
//!
//! Issues covered:
//! - SplatRs-gv7: Single Gaussian Single Pixel Atomic Test
//! - SplatRs-utk: 2x2 Grid Alpha Blending Test
//! - SplatRs-v19: Depth Ordering Test (enhanced)
//! - SplatRs-dgc: Covariance Projection Test
//! - SplatRs-qsn: SH DC-Only Test
//!
//! Design philosophy:
//! - Remove data complexity entirely (no COLMAP, no real images)
//! - Test one thing at a time
//! - Hand-calculable ground truth where possible

use image::RgbImage;
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{linear_f32_to_srgb_u8, Camera, Gaussian};
use sugar_rs::render::render_full_linear;

/// SH DC coefficient normalization constant: Y_0^0 = 1/(2*sqrt(pi))
const SH_C0: f32 = 0.282_094_791_773_878_14;

/// Create SH coefficients for a constant (view-independent) color.
/// Only sets the DC term (index 0), all higher-order terms are zero.
fn sh_constant_color(rgb: Vector3<f32>) -> [[f32; 3]; 16] {
    let mut sh = [[0.0f32; 3]; 16];
    // DC term: color = SH_C0 * sh[0], so sh[0] = color / SH_C0
    sh[0] = [rgb.x / SH_C0, rgb.y / SH_C0, rgb.z / SH_C0];
    sh
}

/// Create a simple camera looking down the +Z axis.
fn simple_camera(width: u32, height: u32, focal_length: f32) -> Camera {
    Camera::new(
        focal_length,
        focal_length,
        (width as f32) / 2.0,
        (height as f32) / 2.0,
        width,
        height,
        Matrix3::identity(),
        Vector3::zeros(),
    )
}

/// Convert linear RGB Vec to RGB8 image for saving.
fn linear_to_rgb8_image(linear: &[Vector3<f32>], width: u32, height: u32) -> RgbImage {
    let mut img = RgbImage::new(width, height);
    for (i, pixel) in linear.iter().enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        img.put_pixel(
            x,
            y,
            image::Rgb([
                linear_f32_to_srgb_u8(pixel.x),
                linear_f32_to_srgb_u8(pixel.y),
                linear_f32_to_srgb_u8(pixel.z),
            ]),
        );
    }
    img
}

// =============================================================================
// SplatRs-gv7: Single Gaussian Single Pixel Atomic Test
// =============================================================================

/// The atomic test case: verify a single Gaussian renders correctly.
///
/// TEST SETUP:
/// - One Gaussian at (0, 0, 2) - directly in front of camera
/// - Camera at origin looking down +Z
/// - Small 8x8 image
///
/// EXPECTED:
/// - Center pixels should have the Gaussian's color
/// - Edge pixels should be near background color
///
/// This is a VISUAL INSPECTION test - outputs to test_output/
#[test]
fn test_single_gaussian_renders_centered() {
    let width = 16;
    let height = 16;
    let camera = simple_camera(width, height, 8.0);

    // Create a bright red Gaussian at z=2
    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),          // position
        Vector3::new(-2.0, -2.0, -2.0),       // log-scale (exp(-2) ≈ 0.135)
        UnitQuaternion::identity(),
        2.0,                                   // opacity logit (sigmoid(2) ≈ 0.88)
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),  // red
    );

    let bg = Vector3::zeros();
    let rendered = render_full_linear(&[gaussian], &camera, &bg, false);

    // Check center pixel (8, 8) is predominantly red
    let center_idx = (8 * width + 8) as usize;
    let center = rendered[center_idx];

    // Save for visual inspection
    let output_dir = std::path::PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).ok();

    let img = linear_to_rgb8_image(&rendered, width, height);
    img.save(output_dir.join("single_gaussian_centered.png"))
        .expect("Failed to save test image");

    // Automated checks
    assert!(
        center.x > 0.3,
        "Center pixel should be red, got r={:.3} (expected > 0.3)",
        center.x
    );
    assert!(
        center.x > center.y && center.x > center.z,
        "Center should be predominantly red: r={:.3} g={:.3} b={:.3}",
        center.x, center.y, center.z
    );

    // Check corner pixel (0, 0) is near background
    let corner = rendered[0];
    assert!(
        corner.x < 0.1 && corner.y < 0.1 && corner.z < 0.1,
        "Corner should be near black background: r={:.3} g={:.3} b={:.3}",
        corner.x, corner.y, corner.z
    );
}

/// Test that color gradients flow correctly for a single Gaussian.
///
/// SETUP:
/// - Single red Gaussian, target is blue
/// - Verify gradient points from red toward blue
///
/// This validates the color gradient direction without full training.
#[test]
fn test_single_gaussian_color_gradient_direction() {
    use sugar_rs::render::render_full_color_grads;

    let width = 8;
    let height = 8;
    let camera = simple_camera(width, height, 4.0);

    // Gaussian renders as red
    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-1.5, -1.5, -1.5),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),  // red
    );

    // Target is blue
    let target: Vec<Vector3<f32>> = (0..(width * height))
        .map(|_| Vector3::new(0.0, 0.0, 1.0))
        .collect();

    let bg = Vector3::zeros();

    // First render to get the image
    let rendered = render_full_linear(&[gaussian.clone()], &camera, &bg, false);

    // Compute L2 loss gradient: d_image = 2 * (rendered - target)
    let d_image: Vec<Vector3<f32>> = rendered
        .iter()
        .zip(target.iter())
        .map(|(r, t)| 2.0 * (r - t))
        .collect();

    // Get color gradients
    // Returns: (img, d_colors, d_opacity_logits, d_mean_px, d_cov_2d, d_positions, d_bg)
    let (_, d_colors, _, _, _, _, _) = render_full_color_grads(&[gaussian], &camera, &d_image, &bg, false);

    // The color gradient d_colors[0] should point toward blue (negative red, positive blue)
    let d_color = d_colors[0];

    // For L2 loss: gradient = 2*(rendered - target)
    // If rendered is red and target is blue:
    // - red channel: rendered > target -> positive gradient -> should decrease red
    // - blue channel: rendered < target -> negative gradient -> should increase blue
    assert!(
        d_color.x > 0.0,
        "Red channel gradient should be positive (push down): got {:.6}",
        d_color.x
    );
    assert!(
        d_color.z < 0.0,
        "Blue channel gradient should be negative (push up): got {:.6}",
        d_color.z
    );
}

// =============================================================================
// SplatRs-v19: Depth Ordering Test (Enhanced)
// =============================================================================

/// Verify depth sorting: near Gaussian should dominate over far.
///
/// SETUP:
/// - Green Gaussian at z=2 (near)
/// - Red Gaussian at z=4 (far)
///
/// EXPECTED:
/// - Center pixel should be predominantly green (near occludes far)
#[test]
fn test_depth_ordering_near_dominates() {
    let width = 16;
    let height = 16;
    let camera = simple_camera(width, height, 8.0);

    let near = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(0.0, 1.0, 0.0)),  // green
    );

    let far = Gaussian::new(
        Vector3::new(0.0, 0.0, 4.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),  // red
    );

    let bg = Vector3::zeros();

    // Test with both orderings in input (should sort internally)
    let rendered_1 = render_full_linear(&[near.clone(), far.clone()], &camera, &bg, false);
    let rendered_2 = render_full_linear(&[far.clone(), near.clone()], &camera, &bg, false);

    let center_1 = rendered_1[(8 * width + 8) as usize];
    let center_2 = rendered_2[(8 * width + 8) as usize];

    // Both should have green > red (near dominates)
    assert!(
        center_1.y > center_1.x,
        "Order 1: green should dominate. g={:.3} r={:.3}",
        center_1.y, center_1.x
    );
    assert!(
        center_2.y > center_2.x,
        "Order 2: green should dominate. g={:.3} r={:.3}",
        center_2.y, center_2.x
    );

    // Results should be identical regardless of input order
    let diff = (center_1 - center_2).norm();
    assert!(
        diff < 1e-5,
        "Render should be order-independent. diff={:.6}",
        diff
    );
}

/// Verify swapped depth ordering produces different result.
///
/// SETUP:
/// - Config A: Red at z=2 (near), Blue at z=4 (far) -> red dominates
/// - Config B: Blue at z=2 (near), Red at z=4 (far) -> blue dominates
///
/// This confirms depth sorting actually affects the output.
#[test]
fn test_depth_ordering_swap_changes_result() {
    let width = 16;
    let height = 16;
    let camera = simple_camera(width, height, 8.0);
    let bg = Vector3::zeros();

    // Config A: red near, blue far
    let red_near = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),
    );
    let blue_far = Gaussian::new(
        Vector3::new(0.0, 0.0, 4.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(0.0, 0.0, 1.0)),
    );

    let config_a = render_full_linear(&[red_near, blue_far], &camera, &bg, false);
    let center_a = config_a[(8 * width + 8) as usize];

    // Config B: blue near, red far
    let blue_near = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(0.0, 0.0, 1.0)),
    );
    let red_far = Gaussian::new(
        Vector3::new(0.0, 0.0, 4.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),
    );

    let config_b = render_full_linear(&[blue_near, red_far], &camera, &bg, false);
    let center_b = config_b[(8 * width + 8) as usize];

    // Config A: red dominates
    assert!(
        center_a.x > center_a.z,
        "Config A: red should dominate. r={:.3} b={:.3}",
        center_a.x, center_a.z
    );

    // Config B: blue dominates
    assert!(
        center_b.z > center_b.x,
        "Config B: blue should dominate. b={:.3} r={:.3}",
        center_b.z, center_b.x
    );

    // Save for visual inspection
    let output_dir = std::path::PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).ok();

    let img_a = linear_to_rgb8_image(&config_a, width, height);
    let img_b = linear_to_rgb8_image(&config_b, width, height);
    img_a.save(output_dir.join("depth_ordering_red_near.png")).ok();
    img_b.save(output_dir.join("depth_ordering_blue_near.png")).ok();
}

// =============================================================================
// SplatRs-utk: 2x2 Grid Alpha Blending Test
// =============================================================================

/// Test alpha blending with 4 Gaussians in a grid pattern.
///
/// SETUP:
/// - 4 Gaussians at corners: red, green, blue, yellow
/// - Each should dominate one quadrant
///
/// EXPECTED:
/// - Top-left quadrant: red
/// - Top-right quadrant: green
/// - Bottom-left quadrant: blue
/// - Bottom-right quadrant: yellow
#[test]
fn test_2x2_grid_quadrant_dominance() {
    let width = 64;
    let height = 64;
    // Use a larger focal length and image for better separation
    let camera = simple_camera(width, height, 32.0);

    // Position Gaussians in world space so they project to the four quadrant centers
    // With focal=32, cx=cy=32, and z=2:
    //   u = 32 * x/2 + 32 = 16*x + 32
    //   For x=-1.0: u = 32-16 = 16 (left quadrant center)
    //   For x=+1.0: u = 32+16 = 48 (right quadrant center)
    let offset = 1.0;  // world space offset (projects to quadrant centers)
    let z = 2.0;       // base depth
    let log_scale = Vector3::new(-0.5, -0.5, -0.5);  // exp(-0.5) ≈ 0.6, tighter to reduce overlap

    // Slight z offsets to ensure deterministic depth sorting (all visible, minimal occlusion)
    let top_left = Gaussian::new(
        Vector3::new(-offset, -offset, z + 0.01),
        log_scale,
        UnitQuaternion::identity(),
        3.0,  // high opacity
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),  // red
    );

    let top_right = Gaussian::new(
        Vector3::new(offset, -offset, z + 0.02),
        log_scale,
        UnitQuaternion::identity(),
        3.0,
        sh_constant_color(Vector3::new(0.0, 1.0, 0.0)),  // green
    );

    let bottom_left = Gaussian::new(
        Vector3::new(-offset, offset, z + 0.03),
        log_scale,
        UnitQuaternion::identity(),
        3.0,
        sh_constant_color(Vector3::new(0.0, 0.0, 1.0)),  // blue
    );

    let bottom_right = Gaussian::new(
        Vector3::new(offset, offset, z + 0.04),
        log_scale,
        UnitQuaternion::identity(),
        3.0,
        sh_constant_color(Vector3::new(1.0, 1.0, 0.0)),  // yellow
    );

    let bg = Vector3::zeros();
    let rendered = render_full_linear(
        &[top_left, top_right, bottom_left, bottom_right],
        &camera,
        &bg,
        false,
    );

    // Save for visual inspection
    let output_dir = std::path::PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).ok();
    let img = linear_to_rgb8_image(&rendered, width, height);
    img.save(output_dir.join("grid_2x2_quadrants.png")).ok();

    // Sample quadrant centers:
    // Top-left: (16, 16), Top-right: (48, 16)
    // Bottom-left: (16, 48), Bottom-right: (48, 48)
    let tl = rendered[(16 * width + 16) as usize];
    let tr = rendered[(16 * width + 48) as usize];
    let bl = rendered[(48 * width + 16) as usize];
    let br = rendered[(48 * width + 48) as usize];

    // First verify something was rendered
    let total_brightness: f32 = rendered.iter().map(|p| p.x + p.y + p.z).sum();
    assert!(
        total_brightness > 0.1,
        "Should render something. Total brightness: {:.3}",
        total_brightness
    );

    // Check dominance (with tolerance for blending at edges)
    // Top-left should be predominantly red
    assert!(
        tl.x > 0.1 && tl.x > tl.y && tl.x > tl.z,
        "Top-left should be red: r={:.3} g={:.3} b={:.3}",
        tl.x, tl.y, tl.z
    );

    // Top-right should be predominantly green
    assert!(
        tr.y > 0.1 && tr.y > tr.x && tr.y > tr.z,
        "Top-right should be green: r={:.3} g={:.3} b={:.3}",
        tr.x, tr.y, tr.z
    );

    // Bottom-left should be predominantly blue
    assert!(
        bl.z > 0.1 && bl.z > bl.x && bl.z > bl.y,
        "Bottom-left should be blue: r={:.3} g={:.3} b={:.3}",
        bl.x, bl.y, bl.z
    );

    // Bottom-right should be predominantly yellow (r+g)
    // Allow some blue bleed from neighboring Gaussians (< 0.25)
    assert!(
        br.x > 0.1 && br.y > 0.1 && br.z < br.x && br.z < br.y,
        "Bottom-right should be yellow (r,g > b): r={:.3} g={:.3} b={:.3}",
        br.x, br.y, br.z
    );
}

// =============================================================================
// SplatRs-dgc: Covariance Projection Test
// =============================================================================

/// Verify spherical Gaussian projects to circular 2D splat.
///
/// SETUP:
/// - Spherical Gaussian: scale = [s, s, s] (equal all dimensions)
/// - Camera looking straight at it
///
/// EXPECTED:
/// - Projected splat should be rotationally symmetric (circular)
/// - Pixels at equal distance from center should have equal values
#[test]
fn test_spherical_gaussian_projects_circular() {
    let width = 32;
    let height = 32;
    let camera = simple_camera(width, height, 16.0);

    // Spherical Gaussian (equal scales)
    let s = -2.0;  // log-scale
    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(s, s, s),  // spherical!
        UnitQuaternion::identity(),
        3.0,  // high opacity
        sh_constant_color(Vector3::new(1.0, 1.0, 1.0)),  // white
    );

    let bg = Vector3::zeros();
    let rendered = render_full_linear(&[gaussian], &camera, &bg, false);

    // Save for visual inspection
    let output_dir = std::path::PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).ok();
    let img = linear_to_rgb8_image(&rendered, width, height);
    img.save(output_dir.join("spherical_gaussian_circular.png")).ok();

    // Sample at equal distances from center (radius 4 pixels)
    let center_x = (width / 2) as i32;
    let center_y = (height / 2) as i32;
    let radius = 4;

    let sample = |dx: i32, dy: i32| -> f32 {
        let x = (center_x + dx) as usize;
        let y = (center_y + dy) as usize;
        let pixel = rendered[y * (width as usize) + x];
        (pixel.x + pixel.y + pixel.z) / 3.0  // grayscale average
    };

    // Sample at 4 cardinal directions
    let right = sample(radius, 0);
    let left = sample(-radius, 0);
    let up = sample(0, -radius);
    let down = sample(0, radius);

    // All should be approximately equal for a circular splat
    let values = [right, left, up, down];
    let mean = values.iter().sum::<f32>() / 4.0;
    let max_deviation = values.iter().map(|v| (v - mean).abs()).fold(0.0f32, f32::max);

    assert!(
        max_deviation < 0.05,
        "Spherical Gaussian should project circularly. Values: right={:.3} left={:.3} up={:.3} down={:.3}, max_dev={:.3}",
        right, left, up, down, max_deviation
    );
}

/// Verify anisotropic Gaussian projects to elliptical splat.
///
/// SETUP:
/// - Elongated Gaussian: larger scale in X than Y
///
/// EXPECTED:
/// - Projected splat should be wider than tall
#[test]
fn test_anisotropic_gaussian_projects_elliptical() {
    let width = 32;
    let height = 32;
    let camera = simple_camera(width, height, 16.0);

    // Anisotropic Gaussian (elongated in X)
    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-1.0, -3.0, -3.0),  // larger in X (exp(-1) > exp(-3))
        UnitQuaternion::identity(),
        3.0,
        sh_constant_color(Vector3::new(1.0, 1.0, 1.0)),
    );

    let bg = Vector3::zeros();
    let rendered = render_full_linear(&[gaussian], &camera, &bg, false);

    // Save for visual inspection
    let output_dir = std::path::PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).ok();
    let img = linear_to_rgb8_image(&rendered, width, height);
    img.save(output_dir.join("anisotropic_gaussian_elliptical.png")).ok();

    // Sample horizontal and vertical extent
    let center_x = (width / 2) as usize;
    let center_y = (height / 2) as usize;

    // Find extent where value drops below threshold
    let threshold = 0.1;

    let horizontal_extent = (1..16).find(|&dx| {
        let pixel = rendered[center_y * (width as usize) + center_x + dx];
        (pixel.x + pixel.y + pixel.z) / 3.0 < threshold
    }).unwrap_or(15);

    let vertical_extent = (1..16).find(|&dy| {
        let pixel = rendered[(center_y + dy) * (width as usize) + center_x];
        (pixel.x + pixel.y + pixel.z) / 3.0 < threshold
    }).unwrap_or(15);

    assert!(
        horizontal_extent > vertical_extent,
        "Elongated-X Gaussian should be wider than tall. h_extent={} v_extent={}",
        horizontal_extent, vertical_extent
    );
}

// =============================================================================
// SplatRs-qsn: SH DC-Only Test
// =============================================================================

/// Verify DC-only SH produces constant color regardless of view.
///
/// SETUP:
/// - Single Gaussian with only DC term set
/// - Render from two different camera positions
///
/// EXPECTED:
/// - Both renders should produce similar gray colors (view-independent)
#[test]
fn test_sh_dc_only_view_independent() {
    let width = 16;
    let height = 16;
    let focal = 8.0;

    // Camera 1: at origin looking +Z
    let cam1 = Camera::new(
        focal, focal,
        (width as f32) / 2.0, (height as f32) / 2.0,
        width, height,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    // Camera 2: offset in X, looking at same point
    // Use smaller offset to keep Gaussian well in view
    let cam2 = Camera::new(
        focal, focal,
        (width as f32) / 2.0, (height as f32) / 2.0,
        width, height,
        Matrix3::identity(),
        Vector3::new(-0.5, 0.0, 0.0),  // smaller offset
    );

    // DC-only color (gray)
    let target_color = Vector3::new(0.5, 0.5, 0.5);
    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),  // closer to camera
        Vector3::new(-1.5, -1.5, -1.5),  // larger scale for better visibility
        UnitQuaternion::identity(),
        4.0,  // higher opacity
        sh_constant_color(target_color),
    );

    let bg = Vector3::zeros();
    let render1 = render_full_linear(&[gaussian.clone()], &cam1, &bg, false);
    let render2 = render_full_linear(&[gaussian], &cam2, &bg, false);

    // Sample center pixels
    let center1 = render1[(8 * width + 8) as usize];
    let center2 = render2[(8 * width + 8) as usize];

    // Colors should be very similar (small difference due to position change)
    // The main check is that view-dependent SH terms don't cause color shift
    let color_diff = (center1 - center2).norm();

    // Save for visual inspection
    let output_dir = std::path::PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).ok();
    let img1 = linear_to_rgb8_image(&render1, width, height);
    let img2 = linear_to_rgb8_image(&render2, width, height);
    img1.save(output_dir.join("sh_dc_only_view1.png")).ok();
    img2.save(output_dir.join("sh_dc_only_view2.png")).ok();

    // Note: Some difference is expected due to distance/angle change
    // The key test is that both views show balanced gray (r ≈ g ≈ b), not a color shift

    // First check that we rendered something visible
    let center1_brightness = (center1.x + center1.y + center1.z) / 3.0;
    assert!(
        center1_brightness > 0.01,
        "View 1 should render visible Gaussian. brightness={:.3} (r={:.3} g={:.3} b={:.3})",
        center1_brightness, center1.x, center1.y, center1.z
    );

    // Check gray balance (r ≈ g ≈ b) - this is the key DC-only test
    let max_channel_diff_1 = (center1.x - center1.y).abs()
        .max((center1.y - center1.z).abs())
        .max((center1.x - center1.z).abs());

    assert!(
        max_channel_diff_1 < 0.05,
        "View 1 should be gray (balanced channels). r={:.3} g={:.3} b={:.3}",
        center1.x, center1.y, center1.z
    );

    // Check that view 2 also shows balanced gray
    let max_channel_diff_2 = (center2.x - center2.y).abs()
        .max((center2.y - center2.z).abs())
        .max((center2.x - center2.z).abs());

    assert!(
        max_channel_diff_2 < 0.05,
        "View 2 should also be gray (balanced channels). r={:.3} g={:.3} b={:.3}",
        center2.x, center2.y, center2.z
    );
}

// =============================================================================
// End-to-End Gradient Tests (SplatRs-cqq, SplatRs-u1j)
// =============================================================================

/// End-to-end position gradient check via finite differences.
///
/// This tests the full chain: image_loss → pixel_grad → blend → projection → position
#[test]
fn test_end_to_end_position_gradient() {
    use sugar_rs::render::render_full_color_grads;

    // Use larger image and more visible Gaussian for reliable gradient flow
    let width = 32;
    let height = 32;
    let camera = simple_camera(width, height, 16.0);

    // Position offset from center - larger offset for more gradient signal
    let position = Vector3::new(0.2, 0.0, 2.0);
    let gaussian = Gaussian::new(
        position,
        Vector3::new(-1.0, -1.0, -1.0),  // larger scale for more pixel coverage
        UnitQuaternion::identity(),
        2.0,  // sigmoid(2) ≈ 0.88
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),
    );

    // Target: Gaussian at center
    let target_gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-1.0, -1.0, -1.0),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),
    );

    let bg = Vector3::zeros();
    let target = render_full_linear(&[target_gaussian], &camera, &bg, false);
    let rendered = render_full_linear(&[gaussian.clone()], &camera, &bg, false);

    // Verify both Gaussians are visible
    let target_brightness: f32 = target.iter().map(|p| p.x + p.y + p.z).sum();
    let rendered_brightness: f32 = rendered.iter().map(|p| p.x + p.y + p.z).sum();
    assert!(target_brightness > 1.0, "Target should be visible: {}", target_brightness);
    assert!(rendered_brightness > 1.0, "Rendered should be visible: {}", rendered_brightness);

    // Compute L2 loss gradient: d_image = 2 * (rendered - target)
    let d_image: Vec<Vector3<f32>> = rendered
        .iter()
        .zip(target.iter())
        .map(|(r, t)| 2.0 * (r - t))
        .collect();

    // Verify d_image is non-zero (there should be a difference)
    let d_image_magnitude: f32 = d_image.iter().map(|d| d.norm_squared()).sum();
    assert!(d_image_magnitude > 0.01, "d_image should be non-zero: {}", d_image_magnitude);

    // Get analytical gradients
    // Returns: (img, d_colors, d_opacity_logits, d_positions, d_log_scales, d_rot_vecs, d_bg)
    let (_, d_colors, _, d_positions, _, _, _) =
        render_full_color_grads(&[gaussian.clone()], &camera, &d_image, &bg, false);

    // Debug: check intermediate gradients
    let d_color_magnitude = d_colors[0].norm();
    eprintln!("DEBUG: d_color magnitude = {:.6}", d_color_magnitude);
    eprintln!("DEBUG: d_positions[0] = {:?}", d_positions[0]);

    let analytical_d_pos_x = d_positions[0].x;

    // Numerical gradient via finite differences
    let eps = 1e-4;
    let loss_fn = |pos_x: f32| -> f32 {
        let g = Gaussian::new(
            Vector3::new(pos_x, position.y, position.z),
            gaussian.scale,
            gaussian.rotation,
            gaussian.opacity,
            gaussian.sh_coeffs,
        );
        let rendered = render_full_linear(&[g], &camera, &bg, false);
        rendered
            .iter()
            .zip(target.iter())
            .map(|(r, t)| (r - t).norm_squared())
            .sum::<f32>()
    };

    let loss_plus = loss_fn(position.x + eps);
    let loss_minus = loss_fn(position.x - eps);
    let numerical_d_pos_x = (loss_plus - loss_minus) / (2.0 * eps);

    eprintln!("DEBUG: analytical_d_pos_x = {:.6}", analytical_d_pos_x);
    eprintln!("DEBUG: numerical_d_pos_x = {:.6}", numerical_d_pos_x);

    // Check match
    let rel_err = if numerical_d_pos_x.abs() > 1e-6 {
        (analytical_d_pos_x - numerical_d_pos_x).abs() / numerical_d_pos_x.abs()
    } else {
        (analytical_d_pos_x - numerical_d_pos_x).abs()
    };

    assert!(
        rel_err < 0.3 || (analytical_d_pos_x - numerical_d_pos_x).abs() < 0.1,
        "Position gradient mismatch: analytical={:.6} numerical={:.6} rel_err={:.4}",
        analytical_d_pos_x, numerical_d_pos_x, rel_err
    );
}
