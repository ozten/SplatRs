//! Debug test to track down color loss in M3 rendering

use sugar_rs::core::{init_from_colmap_points, evaluate_sh};
use sugar_rs::io::{load_colmap_scene, Point3D};
use nalgebra::Vector3;
use std::path::PathBuf;

#[test]
fn test_color_pipeline() {
    // Test with a known blue color
    let blue_point = Point3D {
        id: 0,
        position: Vector3::new(0.0, 0.0, 1.0),
        color: [0, 0, 255],  // Pure blue
        error: 0.0,
    };

    let cloud = init_from_colmap_points(&[blue_point]);
    let gaussian = &cloud.gaussians[0];

    println!("Input color: [0, 0, 255] (blue)");
    println!("SH DC coefficients: {:?}", gaussian.sh_coeffs[0]);

    // Evaluate color
    let view_dir = Vector3::new(0.0, 0.0, 1.0).normalize();
    let color_vec = evaluate_sh(&gaussian.sh_coeffs, &view_dir);

    println!("Evaluated color (0-1): [{:.3}, {:.3}, {:.3}]",
             color_vec.x, color_vec.y, color_vec.z);

    let color_u8 = [
        (color_vec.x * 255.0) as u8,
        (color_vec.y * 255.0) as u8,
        (color_vec.z * 255.0) as u8,
    ];

    println!("Final color (0-255): {:?}", color_u8);

    // Should be mostly blue
    assert!(color_u8[2] > 200, "Blue channel should be strong");
    assert!(color_u8[0] < 50, "Red channel should be weak");
    assert!(color_u8[1] < 50, "Green channel should be weak");
}

#[test]
fn test_colmap_color_distribution() {
    let colmap_path = PathBuf::from("/Users/ozten/Projects/GuassianPlay/digital_calipers2_project/colmap_workspace/sparse/0");

    if !colmap_path.exists() {
        println!("Skipping test - data not found");
        return;
    }

    let scene = load_colmap_scene(&colmap_path).expect("Failed to load scene");

    // Find min/max RGB values
    let mut min_r = 255u8;
    let mut min_g = 255u8;
    let mut min_b = 255u8;
    let mut max_r = 0u8;
    let mut max_g = 0u8;
    let mut max_b = 0u8;

    let mut sum_r = 0u64;
    let mut sum_g = 0u64;
    let mut sum_b = 0u64;

    for point in &scene.points {
        min_r = min_r.min(point.color[0]);
        min_g = min_g.min(point.color[1]);
        min_b = min_b.min(point.color[2]);

        max_r = max_r.max(point.color[0]);
        max_g = max_g.max(point.color[1]);
        max_b = max_b.max(point.color[2]);

        sum_r += point.color[0] as u64;
        sum_g += point.color[1] as u64;
        sum_b += point.color[2] as u64;
    }

    let count = scene.points.len() as u64;
    println!("Color statistics from {} points:", count);
    println!("  R: min={}, max={}, avg={}", min_r, max_r, sum_r / count);
    println!("  G: min={}, max={}, avg={}", min_g, max_g, sum_g / count);
    println!("  B: min={}, max={}, avg={}", min_b, max_b, sum_b / count);

    // Find some blue points (for the tape)
    let blue_points: Vec<_> = scene.points.iter()
        .filter(|p| p.color[2] > 150 && p.color[2] > p.color[0] && p.color[2] > p.color[1])
        .take(5)
        .collect();

    println!("\nFound {} blue-ish points (B>150, B>R, B>G):", blue_points.len());
    for (i, p) in blue_points.iter().enumerate() {
        println!("  Point {}: RGB = {:?}", i, p.color);
    }

    // Find VERY colorful points (high saturation)
    let colorful_points: Vec<_> = scene.points.iter()
        .filter(|p| {
            let r = p.color[0] as i32;
            let g = p.color[1] as i32;
            let b = p.color[2] as i32;
            let max_diff = (r - g).abs().max((r - b).abs()).max((g - b).abs());
            max_diff > 50  // At least 50 difference between channels
        })
        .take(10)
        .collect();

    println!("\nFound {} highly saturated points (channel diff >50):", colorful_points.len());
    for (i, p) in colorful_points.iter().enumerate() {
        println!("  Point {}: RGB = {:?}", i, p.color);
    }
}
