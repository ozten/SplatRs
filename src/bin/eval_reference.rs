//! eval_reference: Phase-0 forward-render isolation test.
//!
//! Loads a KNOWN-GOOD trained Gaussian model (INRIA/Brush `.ply`) and renders it from
//! the dataset's own cameras, comparing against ground-truth images. This cleanly
//! separates "can we render splats correctly?" from "can we train them?".
//!
//! Default target is the NeRF-synthetic `lego` scene, which ships with a converged
//! 173k-Gaussian model (`lego.ply`), exact synthetic cameras (`transforms_*.json`),
//! and ground-truth images — the canonical 3DGS sanity scene.
//!
//! Because model/renderer conventions are exactly what we're trying to validate, the
//! harness sweeps the two known convention degrees of freedom and reports PSNR for each:
//!   - DC offset:  full-SH render path omits the standard `+0.5` DC offset; `--dc-offset`
//!                 injects `0.5/SH_C0` into the DC coefficient to compensate.
//!   - Output map: `gamma`   = linear->sRGB (what `sugar-render` does today)
//!                 `nogamma` = treat render output directly as sRGB (INRIA/Brush convention)
//!
//! Usage:
//!   sugar-eval-reference \
//!     --ply datasets/nerf_synthetic/lego/lego.ply \
//!     --transforms datasets/nerf_synthetic/lego/transforms_test.json \
//!     --num 3 --downsample 0.5 --out-dir /tmp/lego_eval

use nalgebra::{Matrix3, Matrix4, UnitQuaternion, Vector3};
use std::path::{Path, PathBuf};
use std::time::Instant;

use sugar_rs::core::{linear_f32_to_srgb_u8, Camera, Gaussian};
use sugar_rs::io::load_ply;
use sugar_rs::render::full_diff::render_full_linear_with_depth;

use image::{ImageBuffer, Rgb, RgbImage};

const SH_C0: f32 = 0.282_094_791_773_878_14;

fn main() {
    let mut ply_path =
        PathBuf::from("datasets/nerf_synthetic/lego/lego.ply");
    let mut transforms_path =
        PathBuf::from("datasets/nerf_synthetic/lego/transforms_test.json");
    let mut out_dir = PathBuf::from("/tmp/lego_eval");
    let mut num_frames: usize = 3;
    let mut downsample: f32 = 0.5;
    let mut world_rot = "none".to_string();
    let mut bg = Vector3::new(1.0f32, 1.0, 1.0); // lego trained on white bg
    let mut frames_arg: Option<String> = None; // explicit comma-separated frame indices
    let mut start: usize = 0; // starting frame index (with --stride/--num)
    let mut stride: usize = 1; // step between frames
    let mut all_combos = false; // render all 4 convention combos (slower)

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ply" => ply_path = PathBuf::from(args.next().expect("--ply")),
            "--transforms" => transforms_path = PathBuf::from(args.next().expect("--transforms")),
            "--out-dir" => out_dir = PathBuf::from(args.next().expect("--out-dir")),
            "--num" => num_frames = args.next().expect("--num").parse().expect("int"),
            "--frames" => frames_arg = Some(args.next().expect("--frames")),
            "--start" => start = args.next().expect("--start").parse().expect("int"),
            "--stride" => stride = args.next().expect("--stride").parse().expect("int"),
            "--all-combos" => all_combos = true,
            "--downsample" => downsample = args.next().expect("--downsample").parse().expect("float"),
            "--world-rot" => world_rot = args.next().expect("--world-rot"),
            "--background" => {
                let s = args.next().expect("--background");
                let p: Vec<f32> = s.split(',').map(|x| x.parse().unwrap()).collect();
                bg = Vector3::new(p[0], p[1], p[2]);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    std::fs::create_dir_all(&out_dir).expect("create out dir");

    println!("Loading model from {:?} ...", ply_path);
    let cloud = load_ply(&ply_path).expect("load ply");
    println!("Loaded {} Gaussians", cloud.len());

    let world_rot_mat = world_rotation(&world_rot);
    let base_gaussians: Vec<Gaussian> = if world_rot == "none" {
        cloud.gaussians.clone()
    } else {
        let q = UnitQuaternion::from_matrix(&world_rot_mat);
        cloud
            .gaussians
            .iter()
            .map(|g| {
                let mut g2 = g.clone();
                g2.position = world_rot_mat * g.position;
                g2.rotation = q * g.rotation;
                g2
            })
            .collect()
    };

    // DC-offset-compensated copy (adds the standard +0.5 the full-SH path omits).
    let dc_offset = 0.5 / SH_C0;
    let offset_gaussians: Vec<Gaussian> = base_gaussians
        .iter()
        .map(|g| {
            let mut g2 = g.clone();
            for c in 0..3 {
                g2.sh_coeffs[0][c] += dc_offset;
            }
            g2
        })
        .collect();

    // Parse transforms.json
    let json_str = std::fs::read_to_string(&transforms_path).expect("read transforms");
    let v: serde_json::Value = serde_json::from_str(&json_str).expect("parse transforms json");
    let camera_angle_x = v["camera_angle_x"].as_f64().expect("camera_angle_x") as f32;
    let frames = v["frames"].as_array().expect("frames array");
    let transforms_dir = transforms_path.parent().unwrap_or(Path::new("."));

    // Determine which frame indices to render.
    let indices: Vec<usize> = if let Some(list) = &frames_arg {
        list.split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .filter(|&i| i < frames.len())
            .collect()
    } else {
        (0..num_frames)
            .map(|k| start + k * stride)
            .filter(|&i| i < frames.len())
            .collect()
    };
    let n = indices.len();
    println!(
        "Evaluating {} frame(s) {:?} at downsample {:.2}, world-rot '{}', bg {:?}{}\n",
        n,
        indices,
        downsample,
        world_rot,
        bg,
        if all_combos { " [all combos]" } else { " [offset+nogamma only]" }
    );

    // Accumulate mean PSNR across frames for each convention combo.
    let mut sums = [0.0f64; 4]; // [asis_gamma, asis_nogamma, off_gamma, off_nogamma]
    let labels = ["asis+gamma ", "asis+nogamma", "offset+gamma", "offset+nogam"];

    for &i in &indices {
        let frame = &frames[i];
        let file_path = frame["file_path"].as_str().expect("file_path");
        let rel = file_path.trim_start_matches("./");
        let gt_path = transforms_dir.join(format!("{}.png", rel));

        // Ground truth: composite RGBA over white, then resize to render resolution.
        let gt_full = image::open(&gt_path)
            .unwrap_or_else(|e| panic!("open GT {:?}: {}", gt_path, e))
            .to_rgba8();
        let (w0, h0) = (gt_full.width(), gt_full.height());
        let tw = ((w0 as f32) * downsample).round().max(1.0) as u32;
        let th = ((h0 as f32) * downsample).round().max(1.0) as u32;
        let gt_white_full = composite_over_white(&gt_full);
        let gt = image::imageops::resize(&gt_white_full, tw, th, image::imageops::FilterType::Triangle);

        // Camera intrinsics from FOV; principal point at image center.
        let fx0 = 0.5 * (w0 as f32) / (0.5 * camera_angle_x).tan();
        let camera = Camera::new(
            fx0 * downsample,
            fx0 * downsample, // square pixels
            (w0 as f32 * 0.5) * downsample,
            (h0 as f32 * 0.5) * downsample,
            tw,
            th,
            Matrix3::identity(), // filled below
            Vector3::zeros(),
        );
        let (rot, trans) = blender_c2w_to_opencv_w2c(frame);
        let camera = Camera { rotation: rot, translation: trans, ..camera };

        // Always render the correct convention (offset+nogamma). Render the "asis"
        // variant too only when --all-combos is set (doubles render time).
        let t0 = Instant::now();
        let (lin_off, _) = render_full_linear_with_depth(&offset_gaussians, &camera, &bg, false);
        let off_nogamma = to_rgb8(&lin_off, tw, th, false);
        let p_off_nogamma = psnr(&off_nogamma, &gt);
        sums[3] += p_off_nogamma as f64;

        let stem = format!("frame_{:03}", i);
        gt.save(out_dir.join(format!("{}_gt.png", stem))).unwrap();
        off_nogamma
            .save(out_dir.join(format!("{}_offset_nogamma.png", stem)))
            .unwrap();

        if all_combos {
            let (lin_asis, _) = render_full_linear_with_depth(&base_gaussians, &camera, &bg, false);
            let dt = t0.elapsed().as_secs_f32();
            let asis_gamma = to_rgb8(&lin_asis, tw, th, true);
            let asis_nogamma = to_rgb8(&lin_asis, tw, th, false);
            let off_gamma = to_rgb8(&lin_off, tw, th, true);
            let p = [
                psnr(&asis_gamma, &gt),
                psnr(&asis_nogamma, &gt),
                psnr(&off_gamma, &gt),
                p_off_nogamma,
            ];
            sums[0] += p[0] as f64;
            sums[1] += p[1] as f64;
            sums[2] += p[2] as f64;
            println!(
                "frame {:>3} ({}): {}={:5.2}  {}={:5.2}  {}={:5.2}  {}={:5.2}  dB   [{:.1}s]",
                i, rel, labels[0], p[0], labels[1], p[1], labels[2], p[2], labels[3], p[3], dt
            );
            asis_gamma
                .save(out_dir.join(format!("{}_asis_gamma.png", stem)))
                .unwrap();
            let sbs = side_by_side(&[&gt, &off_nogamma, &asis_gamma]);
            sbs.save(out_dir.join(format!("{}_sbs.png", stem))).unwrap();
        } else {
            let dt = t0.elapsed().as_secs_f32();
            println!(
                "frame {:>3} ({}): offset+nogamma={:5.2} dB   [{:.1}s]",
                i, rel, p_off_nogamma, dt
            );
            let sbs = side_by_side(&[&gt, &off_nogamma]);
            sbs.save(out_dir.join(format!("{}_sbs.png", stem))).unwrap();
        }
    }

    println!("\n=== Mean PSNR over {} frame(s) ===", n);
    if all_combos {
        for k in 0..4 {
            println!("  {} : {:6.2} dB", labels[k], sums[k] / n as f64);
        }
    } else {
        println!("  {} : {:6.2} dB", labels[3], sums[3] / n as f64);
    }
    println!("\nOutputs written to {:?}", out_dir);
    println!("(*_sbs.png = GT | offset+nogamma | asis+gamma)");
}

/// Convert a Blender/OpenGL camera-to-world matrix (from a NeRF transforms.json frame)
/// into an OpenCV/COLMAP world-to-camera (R, t): p_cam = R * p_world + t, +Z forward.
fn blender_c2w_to_opencv_w2c(frame: &serde_json::Value) -> (Matrix3<f32>, Vector3<f32>) {
    let m = frame["transform_matrix"].as_array().expect("transform_matrix");
    let get = |r: usize, c: usize| m[r].as_array().unwrap()[c].as_f64().unwrap() as f32;
    // Row-major 4x4.
    let c2w = Matrix4::new(
        get(0, 0), get(0, 1), get(0, 2), get(0, 3),
        get(1, 0), get(1, 1), get(1, 2), get(1, 3),
        get(2, 0), get(2, 1), get(2, 2), get(2, 3),
        get(3, 0), get(3, 1), get(3, 2), get(3, 3),
    );
    // Flip camera Y and Z axes (OpenGL -> OpenCV): right-multiply by diag(1,-1,-1,1).
    let flip = Matrix4::from_diagonal(&nalgebra::Vector4::new(1.0, -1.0, -1.0, 1.0));
    let c2w_cv = c2w * flip;
    let w2c = c2w_cv.try_inverse().expect("camera matrix not invertible");
    let rot = Matrix3::new(
        w2c[(0, 0)], w2c[(0, 1)], w2c[(0, 2)],
        w2c[(1, 0)], w2c[(1, 1)], w2c[(1, 2)],
        w2c[(2, 0)], w2c[(2, 1)], w2c[(2, 2)],
    );
    let trans = Vector3::new(w2c[(0, 3)], w2c[(1, 3)], w2c[(2, 3)]);
    (rot, trans)
}

/// Optional global world reorientation lever to reconcile point-cloud vs camera frames.
fn world_rotation(name: &str) -> Matrix3<f32> {
    match name {
        "none" => Matrix3::identity(),
        // Rotate -90 deg about X: (x,y,z) -> (x, z, -y)   (Z-up -> Y-up)
        "xm90" => Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0),
        // Rotate +90 deg about X: (x,y,z) -> (x, -z, y)   (Y-up -> Z-up)
        "xp90" => Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0),
        // 180 deg about X: (x,y,z) -> (x,-y,-z)
        "flipyz" => Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0),
        other => panic!("unknown --world-rot '{}'", other),
    }
}

fn composite_over_white(rgba: &image::RgbaImage) -> RgbImage {
    let (w, h) = (rgba.width(), rgba.height());
    let mut out = RgbImage::new(w, h);
    for (x, y, p) in rgba.enumerate_pixels() {
        let a = p[3] as f32 / 255.0;
        let mut c = [0u8; 3];
        for k in 0..3 {
            let v = p[k] as f32 * a + 255.0 * (1.0 - a);
            c[k] = v.round().clamp(0.0, 255.0) as u8;
        }
        out.put_pixel(x, y, Rgb(c));
    }
    out
}

/// Map linear render output to sRGB8. `gamma=true` applies linear->sRGB; `gamma=false`
/// treats the render output as already display-space (INRIA/Brush convention).
fn to_rgb8(linear: &[Vector3<f32>], w: u32, h: u32, gamma: bool) -> RgbImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let px = linear[(y * w + x) as usize];
        let mut c = [0u8; 3];
        for k in 0..3 {
            c[k] = if gamma {
                linear_f32_to_srgb_u8(px[k])
            } else {
                (px[k].clamp(0.0, 1.0) * 255.0).round() as u8
            };
        }
        Rgb(c)
    })
}

fn psnr(a: &RgbImage, b: &RgbImage) -> f32 {
    assert_eq!(a.dimensions(), b.dimensions());
    let (pa, pb) = (a.as_raw(), b.as_raw());
    let mut sse: f64 = 0.0;
    for k in 0..pa.len() {
        let d = pa[k] as f64 - pb[k] as f64;
        sse += d * d;
    }
    let mse = sse / pa.len() as f64;
    if mse <= 1e-9 {
        return 99.0;
    }
    (10.0 * (255.0f64 * 255.0 / mse).log10()) as f32
}

fn side_by_side(imgs: &[&RgbImage]) -> RgbImage {
    let h = imgs.iter().map(|im| im.height()).max().unwrap();
    let total_w: u32 = imgs.iter().map(|im| im.width()).sum::<u32>() + (imgs.len() as u32 - 1) * 4;
    let mut out = RgbImage::from_pixel(total_w, h, Rgb([32, 32, 32]));
    let mut x_off = 0u32;
    for im in imgs {
        for (x, y, p) in im.enumerate_pixels() {
            out.put_pixel(x_off + x, y, *p);
        }
        x_off += im.width() + 4;
    }
    out
}
