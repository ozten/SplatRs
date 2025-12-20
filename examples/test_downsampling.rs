// Quick test of downsampling quality
use image::{RgbImage, Rgb};
use sugar_rs::render::full_diff::{downsample_rgb_box, downsample_rgb_bilinear};

fn main() {
    println!("Creating smooth gradient...");

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

    std::fs::create_dir_all("test_output").unwrap();
    img.save("test_output/gradient_original.png").unwrap();
    println!("Saved: test_output/gradient_original.png ({}x{})", img.width(), img.height());

    // Test box filter (8x downsampling)
    let downsampled_box = downsample_rgb_box(&img, 8);
    downsampled_box.save("test_output/gradient_box8x.png").unwrap();
    println!("Saved: test_output/gradient_box8x.png ({}x{})", downsampled_box.width(), downsampled_box.height());

    // Test bilinear
    let target_width = width / 8;
    let target_height = height / 8;
    let downsampled_bilinear = downsample_rgb_bilinear(&img, target_width, target_height);
    downsampled_bilinear.save("test_output/gradient_bilinear8x.png").unwrap();
    println!("Saved: test_output/gradient_bilinear8x.png ({}x{})", downsampled_bilinear.width(), downsampled_bilinear.height());

    println!("\n✅ Done! Check test_output/ for results.");
}
