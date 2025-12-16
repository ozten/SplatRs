//! M7 Visual Test (Step 1): Single-image overfit optimizing color only.
//!
//! This is a deliberately simplified trainer:
//! - Geometry is fixed from COLMAP points
//! - Only SH DC coefficients are optimized
//!
//! It writes `test_output/m7_initial.png` and `test_output/m7_final.png`.

use std::path::PathBuf;

use sugar_rs::optim::trainer::{
    guess_images_dir_from_sparse, train_single_image_color_only, TrainConfig,
};
use sugar_rs::optim::loss::LossKind;

#[test]
#[ignore]
fn test_m7_overfit_color_only_calipers() {
    let sparse_dir = PathBuf::from(
        "/Users/ozten/Projects/GuassianPlay/digital_calipers2_project/colmap_workspace/sparse/0",
    );
    if !sparse_dir.exists() {
        println!("Skipping M7 visual test - COLMAP data not found");
        return;
    }

    let images_dir = guess_images_dir_from_sparse(&sparse_dir).expect("Could not guess images dir");

    let cfg = TrainConfig {
        sparse_dir,
        images_dir,
        image_index: 0,
        max_gaussians: 10_000,
        downsample_factor: 0.25,
        iters: 200,
        lr: 0.05,
        learn_background: true,
        learn_opacity: false,
        loss: LossKind::L2,
    };

    let _out = train_single_image_color_only(&cfg).expect("Training failed");
}
