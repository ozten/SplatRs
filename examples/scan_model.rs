//! Forensic scan: count non-finite / extreme parameter values in a saved model.
//! Usage: cargo run --release --example scan_model -- <model.gs>

use sugar_rs::io::load_model;

fn main() {
    let path = std::env::args().nth(1).expect("usage: scan_model <model.gs>");
    let (cloud, _meta) = load_model(&path).expect("load model");
    let n = cloud.gaussians.len();

    let mut nan_pos = 0usize;
    let mut nan_scale = 0usize;
    let mut nan_rot = 0usize;
    let mut nan_sh = 0usize;
    let mut nan_opac = 0usize;
    let mut huge_scale = 0usize; // exp(log_scale) > 100 world units
    let mut huge_sh = 0usize; // |coeff| > 100
    let mut max_scale = f32::NEG_INFINITY;
    let mut max_abs_sh = 0.0f32;
    let mut max_abs_pos = 0.0f32;

    for g in &cloud.gaussians {
        if !g.position.iter().all(|v| v.is_finite()) {
            nan_pos += 1;
        } else {
            max_abs_pos = max_abs_pos.max(g.position.iter().fold(0.0f32, |a, v| a.max(v.abs())));
        }
        if !g.scale.iter().all(|v| v.is_finite()) {
            nan_scale += 1;
        } else {
            let m = g.scale.iter().fold(f32::NEG_INFINITY, |a, v| a.max(*v));
            max_scale = max_scale.max(m);
            if m.exp() > 100.0 {
                huge_scale += 1;
            }
        }
        if !g.rotation.quaternion().coords.iter().all(|v| v.is_finite()) {
            nan_rot += 1;
        }
        if !g.opacity.is_finite() {
            nan_opac += 1;
        }
        let mut bad_sh = false;
        for k in 0..16 {
            for c in 0..3 {
                let v = g.sh_coeffs[k][c];
                if !v.is_finite() {
                    bad_sh = true;
                } else {
                    max_abs_sh = max_abs_sh.max(v.abs());
                    if v.abs() > 100.0 {
                        huge_sh += 1;
                    }
                }
            }
        }
        if bad_sh {
            nan_sh += 1;
        }
    }

    // Opacity histogram (sigmoid space) — the visible strong tail is what carries the image.
    let mut op_buckets = [0usize; 5]; // <0.1, 0.1-0.3, 0.3-0.5, 0.5-0.9, >0.9
    for g in &cloud.gaussians {
        let o = 1.0 / (1.0 + (-g.opacity).exp());
        let b = if o < 0.1 {
            0
        } else if o < 0.3 {
            1
        } else if o < 0.5 {
            2
        } else if o < 0.9 {
            3
        } else {
            4
        };
        op_buckets[b] += 1;
    }

    println!("total gaussians: {n}");
    println!(
        "opacity histogram: <0.1: {} | 0.1-0.3: {} | 0.3-0.5: {} | 0.5-0.9: {} | >0.9: {}",
        op_buckets[0], op_buckets[1], op_buckets[2], op_buckets[3], op_buckets[4]
    );
    println!("non-finite: pos={nan_pos} scale={nan_scale} rot={nan_rot} sh={nan_sh} opac={nan_opac}");
    println!("extremes: max log_scale={max_scale:.3} (exp={:.3e}), huge_scale(>100wu)={huge_scale}", max_scale.exp());
    println!("          max |sh|={max_abs_sh:.3e}, sh coeffs >100: {huge_sh}");
    println!("          max |pos|={max_abs_pos:.3e}");
}
