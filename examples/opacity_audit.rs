//! Diagnostic: audit a trained model's opacity distribution against geometric visibility.
//!
//! Buckets every Gaussian by final opacity — in particular whether its logit still sits
//! bit-exactly at the B3 reset cap inverse_sigmoid(0.01), which means it received zero
//! opacity gradient since the last reset — and cross-tabs each bucket against how many
//! train-view frusta contain it and its projected screen footprint.
//!
//! Usage:
//!   cargo run --release --example opacity_audit -- \
//!     <model.gs> <sparse_dir> <max_images> <train_fraction> <seed> <downsample>

use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use sugar_rs::core::{inverse_sigmoid, sigmoid, Camera};
use sugar_rs::io::{load_colmap_scene, load_model};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!("usage: opacity_audit <model.gs> <sparse_dir> <max_images> <train_fraction> <seed> <downsample>");
        std::process::exit(1);
    }
    let (cloud, meta) = load_model(&args[1]).expect("load model");
    let scene = load_colmap_scene(Path::new(&args[2])).expect("load scene");
    let max_images: usize = args[3].parse().unwrap();
    let train_fraction: f32 = args[4].parse().unwrap();
    let seed: u64 = args[5].parse().unwrap();
    let downsample: f32 = args[6].parse().unwrap();

    // Replicate the trainer's train/test split exactly (same RNG, same shuffle).
    let available = if max_images == 0 {
        scene.images.len()
    } else {
        max_images.min(scene.images.len())
    };
    let mut idx: Vec<usize> = (0..available).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let num_train = ((available as f32) * train_fraction).max(1.0) as usize;
    let train_idx = &idx[..num_train];

    let cams: Vec<Camera> = train_idx
        .iter()
        .map(|&i| {
            let info = &scene.images[i];
            let base = &scene.cameras[&info.camera_id];
            let f = downsample;
            Camera::new(
                base.fx * f,
                base.fy * f,
                base.cx * f,
                base.cy * f,
                ((base.width as f32) * f).round().max(1.0) as u32,
                ((base.height as f32) * f).round().max(1.0) as u32,
                info.rotation.to_rotation_matrix().into_inner(),
                info.translation,
            )
        })
        .collect();

    println!(
        "model: {} gaussians, {} iters, train views: {}",
        cloud.gaussians.len(),
        meta.training_iterations,
        cams.len()
    );

    let reset_logit = inverse_sigmoid(0.01);
    // bucket key: 0=at-floor(bit-exact reset), 1=below 0.005 (prunable), 2=0.005..0.011 (near floor,
    // moved), 3=0.011..0.1 (regrew some), 4=0.1..0.5, 5=>=0.5
    let bucket_names = [
        "AT-FLOOR (logit == reset cap, zero grad since reset)",
        "< 0.005 (below prune threshold)",
        "0.005..0.011 (near floor, but moved)",
        "0.011..0.10  (partial regrowth)",
        "0.10..0.50   (healthy)",
        ">= 0.50      (strong)",
    ];
    let mut count = [0usize; 6];
    let mut vis_sum = [0usize; 6];   // total in-frustum view count
    let mut vis_zero = [0usize; 6];  // gaussians visible in NO train view
    let mut radius_px: Vec<Vec<f32>> = vec![Vec::new(); 6];

    for g in &cloud.gaussians {
        let op = sigmoid(g.opacity);
        let b = if (g.opacity - reset_logit).abs() < 1e-6 {
            0
        } else if op < 0.005 {
            1
        } else if op < 0.011 {
            2
        } else if op < 0.10 {
            3
        } else if op < 0.50 {
            4
        } else {
            5
        };
        count[b] += 1;

        let max_sigma = g.scale.x.max(g.scale.y).max(g.scale.z).exp();
        let mut nvis = 0usize;
        let mut rsum = 0.0f32;
        for cam in &cams {
            let pc = cam.world_to_camera(&g.position);
            if pc.z <= 0.01 {
                continue;
            }
            if let Some(px) = cam.project(&pc) {
                if px.x >= 0.0 && px.x < cam.width as f32 && px.y >= 0.0 && px.y < cam.height as f32 {
                    nvis += 1;
                    rsum += 3.0 * max_sigma * cam.fx / pc.z;
                }
            }
        }
        vis_sum[b] += nvis;
        if nvis == 0 {
            vis_zero[b] += 1;
        } else {
            radius_px[b].push(rsum / nvis as f32);
        }
    }

    let n = cloud.gaussians.len() as f32;
    let _ = Vector3::<f32>::zeros(); // keep nalgebra import obviously used
    println!("\n{:<55} {:>8} {:>6} {:>9} {:>10} {:>10}", "bucket", "count", "%", "avg#views", "%in-0-view", "med r(px)");
    for b in 0..6 {
        let med = if radius_px[b].is_empty() {
            f32::NAN
        } else {
            let mut v = radius_px[b].clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        println!(
            "{:<55} {:>8} {:>5.1}% {:>9.1} {:>9.1}% {:>10.2}",
            bucket_names[b],
            count[b],
            100.0 * count[b] as f32 / n,
            vis_sum[b] as f32 / count[b].max(1) as f32,
            100.0 * vis_zero[b] as f32 / count[b].max(1) as f32,
            med
        );
    }

    // Phase 2: simulate the GPU backward's 16-slot recording on sample train views and
    // cross-tab "ever recorded" (= receives any gradient) against opacity bucket.
    const SLOTS: usize = 16;
    let sample: Vec<usize> = (0..8).map(|k| k * cams.len() / 8).collect();
    let mut recorded_any = vec![false; cloud.gaussians.len()];
    let mut all_counts: Vec<u32> = Vec::new();
    for &ci in &sample {
        let (counts, recorded) =
            sugar_rs::render::full_diff::debug_contrib_stats(&cloud.gaussians, &cams[ci], SLOTS);
        for (i, r) in recorded.iter().enumerate() {
            if *r {
                recorded_any[i] = true;
            }
        }
        all_counts.extend(counts);
    }
    all_counts.sort_unstable();
    let np = all_counts.len();
    let over = all_counts.iter().filter(|&&c| c > SLOTS as u32).count();
    println!(
        "\nper-pixel contributors over {} sampled train views: p50={} p90={} p99={} max={}  pixels>{} slots: {:.1}%",
        sample.len(),
        all_counts[np / 2],
        all_counts[np * 90 / 100],
        all_counts[np * 99 / 100],
        all_counts[np - 1],
        SLOTS,
        100.0 * over as f32 / np as f32
    );
    println!("\n{:<55} {:>22}", "bucket", "% recorded in >=1 pixel");
    let mut rec = [0usize; 6];
    for (i, g) in cloud.gaussians.iter().enumerate() {
        let op = sigmoid(g.opacity);
        let b = if (g.opacity - reset_logit).abs() < 1e-6 {
            0
        } else if op < 0.005 {
            1
        } else if op < 0.011 {
            2
        } else if op < 0.10 {
            3
        } else if op < 0.50 {
            4
        } else {
            5
        };
        if recorded_any[i] {
            rec[b] += 1;
        }
    }
    for b in 0..6 {
        println!(
            "{:<55} {:>21.1}%",
            bucket_names[b],
            100.0 * rec[b] as f32 / count[b].max(1) as f32
        );
    }
}
