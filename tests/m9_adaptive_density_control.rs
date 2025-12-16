//! M9 Test: Adaptive density control (split/clone/prune) integrates end-to-end.
//!
//! This is a smoke test that checks:
//! - Training runs with densify enabled
//! - At least one densify event happens
//! - Gaussian count increases (clone/split)
//!
//! The quality-improvement criteria for M9 is intentionally left to an ignored test / manual runs,
//! since it depends heavily on dataset, hyperparameters, and runtime budget.

use std::path::PathBuf;

use sugar_rs::optim::loss::LossKind;
use sugar_rs::optim::trainer::{train_multiview_color_only, MultiViewTrainConfig};

#[test]
fn test_m9_densify_smoke_tandt() {
    let sparse_dir = PathBuf::from("datasets/tandt_db/tandt/train/sparse/0");
    let images_dir = PathBuf::from("datasets/tandt_db/tandt/train/images");

    if !sparse_dir.exists() || !images_dir.exists() {
        println!("Skipping test - T&T train dataset not found");
        return;
    }

    let cfg = MultiViewTrainConfig {
        sparse_dir,
        images_dir,
        max_gaussians: 250,
        downsample_factor: 0.125,
        iters: 6,
        lr: 0.01,
        learn_background: true,
        learn_opacity: false,
        loss: LossKind::L2,
        learn_position: false,
        learn_scale: false,
        learn_rotation: false,
        max_images: 5,
        rng_seed: Some(0),
        train_fraction: 0.8,
        val_interval: 3,
        max_test_views_for_metrics: 1,
        log_interval: 0,
        densify_interval: 2,
        densify_max_gaussians: 2_000,
        densify_grad_threshold: -1.0, // force densify for smoke testing
        prune_opacity_threshold: 0.0, // disable pruning for smoke testing
        split_sigma_threshold: 1e9,   // force CLONE (avoid scale changes)
    };

    let result = train_multiview_color_only(&cfg).expect("Training failed");

    assert!(result.densify_events > 0, "Expected at least one densify event");
    assert!(
        result.final_num_gaussians > result.initial_num_gaussians,
        "Expected gaussian count to increase ({} -> {})",
        result.initial_num_gaussians,
        result.final_num_gaussians
    );
    assert!(result.final_psnr.is_finite());
}
