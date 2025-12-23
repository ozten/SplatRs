// Debug test to investigate downsampling artifacts
// This creates a synthetic smooth gradient and downsamples it
// to see if artifacts are introduced

use image::{RgbImage, Rgb};
use sugar_rs::render::full_diff::{downsample_rgb_box, downsample_rgb_bilinear};

#[test]
#[ignore] // Manual inspection test
fn test_smooth_gradient_downsample() {
    // Create a smooth blue gradient (like sky)
    let width = 1024;
    let height = 768;
    let mut img = RgbImage::new(width, height);

    for y in 0..height {
        let blue_value = ((y as f32 / height as f32) * 255.0) as u8;
        for x in 0..width {
            img.put_pixel(x, y, Rgb([100, 120, blue_value]));
        }
    }

    img.save("test_output/debug_gradient_original.png").unwrap();

    // Downsample using box filter (8x = divisor 8)
    let downsampled_box = downsample_rgb_box(&img, 8);
    downsampled_box.save("test_output/debug_gradient_box8x.png").unwrap();

    // Downsample using bilinear
    let target_width = width / 8;
    let target_height = height / 8;
    let downsampled_bilinear = downsample_rgb_bilinear(&img, target_width, target_height);
    downsampled_bilinear.save("test_output/debug_gradient_bilinear8x.png").unwrap();

    println!("Saved gradient test images to test_output/");
    println!("Original: {}x{}", img.width(), img.height());
    println!("Box filter: {}x{}", downsampled_box.width(), downsampled_box.height());
    println!("Bilinear: {}x{}", downsampled_bilinear.width(), downsampled_bilinear.height());
}
