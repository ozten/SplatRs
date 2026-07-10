//! Forensic: save the first N Gaussians of a model, for bisecting the count at which
//! GPU rendering diverges. Usage: truncate_model <in.gs> <N> <out.gs>

use sugar_rs::io::{load_model, save_model};

fn main() {
    let mut args = std::env::args().skip(1);
    let inp = args.next().expect("in.gs");
    let n: usize = args.next().expect("N").parse().unwrap();
    let out = args.next().expect("out.gs");
    let (mut cloud, mut meta) = load_model(&inp).expect("load model");
    cloud.gaussians.truncate(n);
    meta.num_gaussians = cloud.gaussians.len() as u64;
    save_model(&out, &cloud, &meta).expect("save model");
    println!("{}", cloud.gaussians.len());
}
