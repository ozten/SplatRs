//! Golden-image I/O + diff helpers for the golden parity/regression harness
//! (docs/TILE_RASTER_PLAN.md Part A).
//!
//! Golden images are stored as 16-bit PNGs holding *raw linear* [0,1] RGB (no sRGB
//! encoding) so round-tripping only ever costs quantization noise of at most
//! 1/65535 ~= 1.5e-5 absolute per channel — well under the CPU-vs-golden tolerance
//! (mean < 5e-4, max < 2e-3) documented in tests/golden_psnr_floor.rs.
//!
//! `compare()` factors out the mean/max-abs-diff snippet copy-pasted across
//! examples/render_compare.rs and tests/unit_gpu_sort_order.rs.
//!
//! Pulled into `tests/golden_*.rs` test binaries via `#[path]` mod (tests/common idiom).

#![allow(dead_code)]

use image::{ImageBuffer, Rgb};
use nalgebra::Vector3;
use std::path::Path;

/// Convert linear [0,1] f32 RGB pixels to a 16-bit RGB image buffer.
pub fn linear_vec_to_rgb16_img(
    linear: &[Vector3<f32>],
    width: u32,
    height: u32,
) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    assert_eq!(linear.len(), (width * height) as usize);
    let mut img = ImageBuffer::new(width, height);
    let to_u16 = |v: f32| (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
    for (i, p) in linear.iter().enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        img.put_pixel(x, y, Rgb([to_u16(p.x), to_u16(p.y), to_u16(p.z)]));
    }
    img
}

/// Convert a 16-bit RGB image buffer back to linear [0,1] f32 pixels.
pub fn rgb16_img_to_linear_vec(img: &ImageBuffer<Rgb<u16>, Vec<u16>>) -> Vec<Vector3<f32>> {
    img.pixels()
        .map(|p| {
            Vector3::new(
                p[0] as f32 / 65535.0,
                p[1] as f32 / 65535.0,
                p[2] as f32 / 65535.0,
            )
        })
        .collect()
}

/// Save linear [0,1] pixels as a 16-bit PNG at `path`, creating parent directories as
/// needed. Only called from the `SUGAR_REGENERATE_GOLDEN=1` regeneration path.
pub fn save_golden(path: &Path, linear: &[Vector3<f32>], width: u32, height: u32) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create golden directory");
    }
    let img = linear_vec_to_rgb16_img(linear, width, height);
    img.save(path)
        .unwrap_or_else(|e| panic!("failed to save golden {}: {e}", path.display()));
}

/// Load a golden 16-bit PNG from `path` as (linear [0,1] pixels, width, height).
/// Panics with a regeneration hint if the golden is missing or unreadable.
pub fn load_golden(path: &Path) -> (Vec<Vector3<f32>>, u32, u32) {
    let img = image::open(path)
        .unwrap_or_else(|e| {
            panic!(
                "golden image missing or unreadable at {}: {e}\n\
                 Regenerate with: SUGAR_REGENERATE_GOLDEN=1 cargo test --test <this_test> -- --nocapture",
                path.display()
            )
        })
        .into_rgb16();
    let (w, h) = (img.width(), img.height());
    (rgb16_img_to_linear_vec(&img), w, h)
}

/// Mean and max absolute per-channel difference between two equal-length linear pixel
/// buffers: `(mean_abs, max_abs)`.
pub fn compare(a: &[Vector3<f32>], b: &[Vector3<f32>]) -> (f32, f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "compare: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        let m = d.x.max(d.y).max(d.z);
        sum_abs += (d.x + d.y + d.z) as f64;
        if m > max_abs {
            max_abs = m;
        }
    }
    let mean_abs = (sum_abs / (3.0 * a.len().max(1) as f64)) as f32;
    (mean_abs, max_abs)
}
