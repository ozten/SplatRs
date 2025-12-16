//! M7 Visual Test: Single-image overfit (color-only) on the T&T dataset.
//!
//! Dataset path (in-repo):
//!   ./datasets/tandt_db/tandt/train
//!
//! Layout expected:
//!   - sparse/0/{cameras.bin,images.bin,points3D.bin}
//!   - images/*.jpg

use std::path::PathBuf;

use sugar_rs::optim::trainer::{
    guess_images_dir_from_sparse, guess_sparse0_from_dataset_root, train_single_image_color_only,
    TrainConfig,
};
use sugar_rs::optim::loss::LossKind;

#[test]
#[ignore]
fn test_m7_overfit_color_only_tandt_train() {
    let root = PathBuf::from("/Users/ozten/Projects/SplatRs/datasets/tandt_db/tandt/train");
    if !root.exists() {
        println!("Skipping T&T visual test - dataset not found at {:?}", root);
        return;
    }

    let sparse_dir =
        guess_sparse0_from_dataset_root(&root).expect("Could not find sparse/0 under dataset root");
    let images_dir = guess_images_dir_from_sparse(&sparse_dir).expect("Could not guess images dir");

    // Run a short training loop to keep this test reasonably quick when manually invoked.
    let cfg = TrainConfig {
        sparse_dir,
        images_dir,
        image_index: 0,
        max_gaussians: 12_000,
        downsample_factor: 0.125,
        iters: 150,
        lr: 0.05,
        learn_background: true,
        learn_opacity: false,
        loss: LossKind::L2,
    };

    let _out = train_single_image_color_only(&cfg).expect("Training failed");
}
