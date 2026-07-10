//! Forensic: render a model N times through the GPU renderer in one process and report
//! pairwise divergence + black fraction per run. Distinguishes sync races (in-process
//! nondeterminism) from uninitialized-memory effects (per-process determinism).
//! Usage: gpu_render_repeat <model.gs> <dataset_root> <camera_idx> <runs>

use nalgebra::Vector3;
use sugar_rs::core::Camera;
use sugar_rs::gpu::GpuRenderer;
use sugar_rs::io::{load_colmap_scene, load_model};

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("model.gs");
    let dataset_root = std::path::PathBuf::from(args.next().expect("dataset_root"));
    let cam_idx: usize = args.next().expect("camera_idx").parse().unwrap();
    let runs: usize = args.next().unwrap_or_else(|| "3".into()).parse().unwrap();
    // Comma-separated downsample factors, cycled across runs (e.g. "1.0,0.1" alternates
    // full-res and tiny renders in one process — probes queue poisoning after a failure).
    let ds_list: Vec<f32> = args
        .next()
        .unwrap_or_else(|| "1.0".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();

    let (cloud, _meta) = load_model(&model_path).expect("load model");
    let scene = load_colmap_scene(&dataset_root.join("sparse/0")).expect("load scene");
    let info = &scene.images[cam_idx];
    let base = scene.cameras.get(&info.camera_id).expect("intrinsics");
    let rotation = info.rotation.to_rotation_matrix().into_inner();
    let bg = Vector3::new(0.1, 0.2, 0.3);

    let renderer = GpuRenderer::new().expect("GPU init");
    let mut frames: Vec<(f32, Vec<Vector3<f32>>)> = Vec::new();
    for r in 0..runs {
        let ds = ds_list[r % ds_list.len()];
        let camera = Camera::new(
            base.fx * ds,
            base.fy * ds,
            base.cx * ds,
            base.cy * ds,
            ((base.width as f32 * ds).round() as u32).max(1),
            ((base.height as f32 * ds).round() as u32).max(1),
            rotation,
            info.translation,
        );
        let t = std::time::Instant::now();
        let img = renderer.render(&cloud.gaussians, &camera, &bg).expect("render");
        let bg_frac = img
            .iter()
            .filter(|p| (*p - bg).amax() < 1.5 / 255.0)
            .count() as f32
            / img.len() as f32;
        let mean: f32 = img.iter().map(|p| p.sum()).sum::<f32>() / (3.0 * img.len() as f32);
        println!(
            "run {r} (ds {ds}): {:?}  bg_frac={bg_frac:.3}  mean={mean:.4}",
            t.elapsed()
        );
        frames.push((ds, img));
    }
    for i in 1..frames.len() {
        if frames[i].0 != frames[0].0 {
            continue; // different resolution, not comparable
        }
        let d: f32 = frames[0]
            .1
            .iter()
            .zip(frames[i].1.iter())
            .map(|(a, b)| (a - b).abs().sum())
            .sum::<f32>()
            / (3.0 * frames[0].1.len() as f32);
        println!("run0 vs run{i}: mean abs diff {d:.6}");
    }
}
