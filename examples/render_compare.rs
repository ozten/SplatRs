//! Render a saved .gs model from a COLMAP camera with BOTH the CPU (full_diff) and GPU
//! forward, to isolate save/load bugs from CPU/GPU forward divergence.
//!
//! Usage:
//!   cargo run --release --features gpu --example render_compare -- \
//!     <model.gs> <dataset_root> <camera_idx> <out_prefix> [bg_r,bg_g,bg_b]

use nalgebra::Vector3;
use sugar_rs::core::Camera;
use sugar_rs::io::{load_colmap_scene, load_model};
use sugar_rs::render::full_diff::{linear_vec_to_rgb8_img, render_full_linear};

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("usage: <model.gs> <dataset_root> <camera_idx> <out_prefix> [bg]");
    let dataset_root = std::path::PathBuf::from(args.next().expect("dataset_root"));
    let cam_idx: usize = args.next().expect("camera_idx").parse().unwrap();
    let out_prefix = args.next().expect("out_prefix");
    let bg = args
        .next()
        .map(|s| {
            let v: Vec<f32> = s.split(',').map(|x| x.parse().unwrap()).collect();
            Vector3::new(v[0], v[1], v[2])
        })
        .unwrap_or_else(Vector3::zeros);

    let (cloud, metadata) = load_model(&model_path).expect("load model");
    eprintln!(
        "Loaded {} Gaussians (training res {}x{}, downsample {})",
        cloud.gaussians.len(),
        metadata.training_width,
        metadata.training_height,
        metadata.training_downsample_factor
    );

    let scene = load_colmap_scene(&dataset_root.join("sparse/0")).expect("load scene");
    let info = &scene.images[cam_idx];
    eprintln!("Camera idx {} -> image {}", cam_idx, info.name);
    let base = scene.cameras.get(&info.camera_id).expect("camera intrinsics");
    let rotation = info.rotation.to_rotation_matrix().into_inner();
    let ds = metadata.training_downsample_factor;
    let camera = Camera::new(
        base.fx * ds,
        base.fy * ds,
        base.cx * ds,
        base.cy * ds,
        metadata.training_width,
        metadata.training_height,
        rotation,
        info.translation,
    );

    // CPU forward (same path as sugar-render)
    let cpu = render_full_linear(&cloud.gaussians, &camera, &bg, false);
    linear_vec_to_rgb8_img(&cpu, camera.width, camera.height)
        .save(format!("{out_prefix}_cpu.png"))
        .expect("save cpu png");
    eprintln!("Saved {out_prefix}_cpu.png");

    // Exact CPU replica of the gpu/sort.wgsl bitonic network, run over the real per-Gaussian
    // depths for this camera (culled entries use the shader's -1.0 sentinel). Reports how
    // sorted the result actually is.
    {
        let mut depths: Vec<f32> = cloud
            .gaussians
            .iter()
            .map(|g| {
                let z = (camera.rotation * g.position + camera.translation).z;
                if z <= 0.2 { -1.0 } else { z }
            })
            .collect();
        let count = depths.len() as u32;
        let padded = count.next_power_of_two();
        let num_stages = (padded as f32).log2() as u32;
        for stage in 0..num_stages {
            for step in 0..=stage {
                let step_within_stage = stage - step;
                let pair_distance = 1u32 << step_within_stage;
                let block_size = 2u32 << step_within_stage;
                for idx in 0..(padded / 2) {
                    if idx >= count {
                        continue; // shader: `if (idx >= params.count) { return; }`
                    }
                    let left = (idx / pair_distance) * block_size + (idx % pair_distance);
                    let right = left + pair_distance;
                    if left >= count || right >= count {
                        continue;
                    }
                    let ascending = ((left >> stage) & 1) == 0;
                    let (l, r) = (depths[left as usize], depths[right as usize]);
                    if (ascending && l > r) || (!ascending && l < r) {
                        depths.swap(left as usize, right as usize);
                    }
                }
            }
        }
        let visible: Vec<f32> = depths.iter().copied().filter(|z| *z >= 0.0).collect();
        let inversions = visible.windows(2).filter(|w| w[0] > w[1]).count();
        eprintln!(
            "Bitonic-replica sortedness: {} adjacent inversions among {} visible depths ({:.1}%)",
            inversions,
            visible.len(),
            100.0 * inversions as f32 / visible.len().max(1) as f32
        );
    }

    // CPU replica of the (fixed) GPU pipeline: same projection math, full ascending depth
    // sort, 3-sigma bbox skip, same alpha/T thresholds. Comparing this against both the CPU
    // renderer and the GPU output pinpoints which side deviates.
    let replica = {
        struct P {
            x: f32,
            y: f32,
            z: f32,
            cxx: f32,
            cxy: f32,
            cyy: f32,
            radius: f32,
            color: Vector3<f32>,
            opacity: f32,
        }
        let mut proj: Vec<P> = Vec::new();
        for g in &cloud.gaussians {
            let pos_cam = camera.rotation * g.position + camera.translation;
            if pos_cam.z <= 0.2 {
                continue;
            }
            let x_px = camera.fx * pos_cam.x / pos_cam.z + camera.cx;
            let y_px = camera.fy * pos_cam.y / pos_cam.z + camera.cy;
            let sigma_world = g.covariance_matrix();
            let sigma_cam = camera.rotation * sigma_world * camera.rotation.transpose();
            let (z, x, y) = (pos_cam.z, pos_cam.x, pos_cam.y);
            let (j00, j02) = (camera.fx / z, -camera.fx * x / (z * z));
            let (j11, j12) = (camera.fy / z, -camera.fy * y / (z * z));
            let s = sigma_cam;
            let cxx = j00 * j00 * s[(0, 0)] + 2.0 * j00 * j02 * s[(0, 2)] + j02 * j02 * s[(2, 2)] + 0.3;
            let cxy = j00 * j11 * s[(0, 1)] + j00 * j12 * s[(0, 2)] + j02 * j11 * s[(1, 2)] + j02 * j12 * s[(2, 2)];
            let cyy = j11 * j11 * s[(1, 1)] + 2.0 * j11 * j12 * s[(1, 2)] + j12 * j12 * s[(2, 2)] + 0.3;
            let max_cov = cxx.max(cyy);
            if max_cov <= 0.0 || max_cov > 1e10 {
                continue;
            }
            let max_dim = camera.width.max(camera.height) as f32;
            if 9.0 * max_cov > max_dim * max_dim {
                continue;
            }
            let color = sugar_rs::core::evaluate_sh_unclamped(
                &g.sh_coeffs,
                &camera.view_direction(&g.position),
            );
            let disc = ((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy).max(0.0).sqrt();
            let lambda_max = (0.5 * (cxx + cyy + disc)).max(0.0);
            proj.push(P {
                x: x_px,
                y: y_px,
                z: pos_cam.z,
                cxx,
                cxy,
                cyy,
                radius: 3.0 * lambda_max.sqrt(),
                color,
                opacity: 1.0 / (1.0 + (-g.opacity).exp()),
            });
        }
        proj.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap());
        let (w, h) = (camera.width as usize, camera.height as usize);
        let mut img = vec![Vector3::<f32>::zeros(); w * h];
        for py in 0..h {
            for px in 0..w {
                let (fx, fy) = (px as f32 + 0.5, py as f32 + 0.5);
                let mut color_acc = Vector3::<f32>::zeros();
                let mut t = 1.0f32;
                for p in &proj {
                    if (px as f32) < (p.x - p.radius).floor()
                        || (px as f32) > (p.x + p.radius).ceil()
                        || (py as f32) < (p.y - p.radius).floor()
                        || (py as f32) > (p.y + p.radius).ceil()
                    {
                        continue;
                    }
                    let det = p.cxx * p.cyy - p.cxy * p.cxy;
                    if det <= 0.0 {
                        continue;
                    }
                    let (dx, dy) = (fx - p.x, fy - p.y);
                    let q = (p.cyy * dx * dx - 2.0 * p.cxy * dx * dy + p.cxx * dy * dy) / det;
                    let alpha = (p.opacity * (-0.5 * q).exp()).min(0.99);
                    if alpha < 1e-4 {
                        continue;
                    }
                    color_acc += t * alpha * p.color;
                    t *= 1.0 - alpha;
                    if t < 1e-4 {
                        break;
                    }
                }
                img[py * w + px] = color_acc + t * bg;
            }
        }
        img
    };
    linear_vec_to_rgb8_img(&replica, camera.width, camera.height)
        .save(format!("{out_prefix}_replica.png"))
        .expect("save replica png");
    let mean_diff = |a: &[Vector3<f32>], b: &[Vector3<f32>]| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs().sum() as f64)
            .sum::<f64>()
            / (3.0 * a.len() as f64)
    };
    eprintln!("replica-vs-CPU: mean abs diff {:.6}", mean_diff(&replica, &cpu));

    // GPU forward (same path as the trainer's validation renders)
    #[cfg(feature = "gpu")]
    {
        let renderer = sugar_rs::gpu::GpuRenderer::new().expect("gpu init");
        let gpu = renderer
            .render(&cloud.gaussians, &camera, &bg)
            .expect("gpu render");
        linear_vec_to_rgb8_img(&gpu, camera.width, camera.height)
            .save(format!("{out_prefix}_gpu.png"))
            .expect("save gpu png");
        eprintln!("Saved {out_prefix}_gpu.png");

        let n = cpu.len().min(gpu.len());
        let mut sum_abs = 0.0f64;
        let mut max_abs = 0.0f32;
        for i in 0..n {
            let d = (cpu[i] - gpu[i]).abs();
            let m = d.x.max(d.y).max(d.z);
            sum_abs += (d.x + d.y + d.z) as f64;
            if m > max_abs {
                max_abs = m;
            }
        }
        eprintln!(
            "CPU-vs-GPU: mean abs diff {:.6}, max abs diff {:.6} (over {} px)",
            sum_abs / (3.0 * n as f64),
            max_abs,
            n
        );
        eprintln!("replica-vs-GPU: mean abs diff {:.6}", mean_diff(&replica, &gpu));

        // DC-only comparison: if this matches while full-SH doesn't, the divergence is in
        // the GPU's SH band 1-3 path (upload layout or eval), which no unit test exercises.
        let cpu_dc = render_full_linear(&cloud.gaussians, &camera, &bg, true);
        let gpu_dc = renderer
            .render_with_sh_mode(&cloud.gaussians, &camera, &bg, true)
            .expect("gpu dc render");
        eprintln!("DC-only CPU-vs-GPU: mean abs diff {:.6}", mean_diff(&cpu_dc, &gpu_dc));

        // Hypothesis probe: WGSL quat_to_matrix passes ROW vectors to the column-major
        // mat3x3 constructor, i.e. the GPU uses R^T. Rebuild the DC-only replica with
        // sigma_world = R^T S^2 R and compare against the GPU DC image.
        let mut transposed_dc = vec![Vector3::<f32>::zeros(); (camera.width * camera.height) as usize];
        {
            struct Q {
                x: f32,
                y: f32,
                z: f32,
                cxx: f32,
                cxy: f32,
                cyy: f32,
                radius: f32,
                color: Vector3<f32>,
                opacity: f32,
            }
            let mut proj: Vec<Q> = Vec::new();
            for g in &cloud.gaussians {
                let pos_cam = camera.rotation * g.position + camera.translation;
                if pos_cam.z <= 0.2 {
                    continue;
                }
                let x_px = camera.fx * pos_cam.x / pos_cam.z + camera.cx;
                let y_px = camera.fy * pos_cam.y / pos_cam.z + camera.cy;
                let r = g.rotation.to_rotation_matrix().into_inner().transpose();
                let sc = g.actual_scale();
                let s2 = nalgebra::Matrix3::from_diagonal(&Vector3::new(sc.x * sc.x, sc.y * sc.y, sc.z * sc.z));
                let sigma_world = r * s2 * r.transpose();
                let sigma_cam = camera.rotation * sigma_world * camera.rotation.transpose();
                let (z, x, y) = (pos_cam.z, pos_cam.x, pos_cam.y);
                let (j00, j02) = (camera.fx / z, -camera.fx * x / (z * z));
                let (j11, j12) = (camera.fy / z, -camera.fy * y / (z * z));
                let s = sigma_cam;
                let cxx = j00 * j00 * s[(0, 0)] + 2.0 * j00 * j02 * s[(0, 2)] + j02 * j02 * s[(2, 2)] + 0.3;
                let cxy = j00 * j11 * s[(0, 1)] + j00 * j12 * s[(0, 2)] + j02 * j11 * s[(1, 2)] + j02 * j12 * s[(2, 2)];
                let cyy = j11 * j11 * s[(1, 1)] + 2.0 * j11 * j12 * s[(1, 2)] + j12 * j12 * s[(2, 2)] + 0.3;
                let max_cov = cxx.max(cyy);
                if max_cov <= 0.0 || max_cov > 1e10 {
                    continue;
                }
                let max_dim = camera.width.max(camera.height) as f32;
                if 9.0 * max_cov > max_dim * max_dim {
                    continue;
                }
                let color = sugar_rs::core::evaluate_sh_dc_only(&g.sh_coeffs);
                let disc = ((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy).max(0.0).sqrt();
                let lambda_max = (0.5 * (cxx + cyy + disc)).max(0.0);
                proj.push(Q {
                    x: x_px,
                    y: y_px,
                    z: pos_cam.z,
                    cxx,
                    cxy,
                    cyy,
                    radius: 3.0 * lambda_max.sqrt(),
                    color,
                    opacity: 1.0 / (1.0 + (-g.opacity).exp()),
                });
            }
            proj.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap());
            let (w, h) = (camera.width as usize, camera.height as usize);
            for py in 0..h {
                for px in 0..w {
                    let (fx, fy) = (px as f32 + 0.5, py as f32 + 0.5);
                    let mut color_acc = Vector3::<f32>::zeros();
                    let mut t = 1.0f32;
                    for p in &proj {
                        if (px as f32) < (p.x - p.radius).floor()
                            || (px as f32) > (p.x + p.radius).ceil()
                            || (py as f32) < (p.y - p.radius).floor()
                            || (py as f32) > (p.y + p.radius).ceil()
                        {
                            continue;
                        }
                        let det = p.cxx * p.cyy - p.cxy * p.cxy;
                        if det <= 0.0 {
                            continue;
                        }
                        let (dx, dy) = (fx - p.x, fy - p.y);
                        let q = (p.cyy * dx * dx - 2.0 * p.cxy * dx * dy + p.cxx * dy * dy) / det;
                        let alpha = (p.opacity * (-0.5 * q).exp()).min(0.99);
                        if alpha < 1e-4 {
                            continue;
                        }
                        color_acc += t * alpha * p.color;
                        t *= 1.0 - alpha;
                        if t < 1e-4 {
                            break;
                        }
                    }
                    transposed_dc[py * w + px] = color_acc + t * bg;
                }
            }
        }
        eprintln!(
            "R^T-replica-DC-vs-GPU-DC: mean abs diff {:.6}  (vs CPU-DC: {:.6})",
            mean_diff(&transposed_dc, &gpu_dc),
            mean_diff(&transposed_dc, &cpu_dc)
        );
    }
}
