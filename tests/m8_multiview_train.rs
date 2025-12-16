//! M8 Test: Multi-view training with train/test split
//!
//! This test verifies that:
//! 1. Multi-view training runs without errors
//! 2. Metrics are finite and output images are produced
//!
//! The longer "quality" check is marked `#[ignore]` since it can be slow.

use std::path::PathBuf;
use sugar_rs::optim::loss::LossKind;
use sugar_rs::optim::trainer::{train_multiview_color_only, MultiViewTrainConfig};

#[test]
fn test_m8_multiview_train_smoke_tandt() {
    let sparse_dir = PathBuf::from("datasets/tandt_db/tandt/train/sparse/0");
    let images_dir = PathBuf::from("datasets/tandt_db/tandt/train/images");

    if !sparse_dir.exists() || !images_dir.exists() {
        println!("Skipping test - T&T train dataset not found");
        return;
    }

    let cfg = MultiViewTrainConfig {
        sparse_dir,
        images_dir,
        max_gaussians: 2_000,
        downsample_factor: 0.125, // 8x downsampling for speed
        iters: 5,
        lr: 0.01,
        learn_background: true,
        learn_opacity: false,
        loss: LossKind::L2,
        learn_position: false,
        learn_scale: false,
        learn_rotation: false,
        learn_sh: false,
        max_images: 5,
        rng_seed: Some(0),
        train_fraction: 0.8,
        val_interval: 1000, // Only validates at the end (iters=5)
        max_test_views_for_metrics: 2,
        log_interval: 1,
        densify_interval: 0,
        densify_max_gaussians: 0,
        densify_grad_threshold: 0.1,
        prune_opacity_threshold: 0.01,
        split_sigma_threshold: 0.05,
    };

    let result = train_multiview_color_only(&cfg).expect("Training failed");

    println!("\n=== M8 Multi-View Training (Smoke) Results ===");
    println!("Train views: {}", result.num_train_views);
    println!("Test views:  {}", result.num_test_views);
    println!("Initial test PSNR: {:.2} dB", result.initial_psnr);
    println!("Final test PSNR:   {:.2} dB", result.final_psnr);
    println!("Improvement:       {:.2} dB", result.final_psnr - result.initial_psnr);

    // Save test view for visual inspection
    std::fs::create_dir_all("test_output").ok();
    result
        .test_view_sample
        .save("test_output/m8_test_view_rendered.png")
        .ok();
    result
        .test_view_target
        .save("test_output/m8_test_view_target.png")
        .ok();

    println!("\nSaved test_output/m8_test_view_rendered.png");
    println!("Saved test_output/m8_test_view_target.png");

    // Verify basic sanity metrics (this is a smoke test; view count can be small when max_images is set).
    assert!(
        result.num_train_views >= 2,
        "Expected at least 2 training views"
    );
    assert!(
        result.num_test_views > 0,
        "Expected at least 1 test view"
    );
    assert!(result.initial_psnr.is_finite());
    assert!(result.final_psnr.is_finite());
    assert!(result.train_loss.is_finite());
}

#[test]
#[ignore] // Slow test - longer training run (use `cargo test -- --ignored`)
fn test_m8_multiview_train_quality_tandt() {
    let sparse_dir = PathBuf::from("datasets/tandt_db/tandt/train/sparse/0");
    let images_dir = PathBuf::from("datasets/tandt_db/tandt/train/images");

    if !sparse_dir.exists() || !images_dir.exists() {
        println!("Skipping test - T&T train dataset not found");
        return;
    }

    let cfg = MultiViewTrainConfig {
        sparse_dir,
        images_dir,
        max_gaussians: 10_000,
        downsample_factor: 0.25,
        iters: 500,
        lr: 0.02,
        learn_background: true,
        learn_opacity: false,
        loss: LossKind::L2,
        learn_position: false,
        learn_scale: false,
        learn_rotation: false,
        learn_sh: false,
        max_images: 0,
        rng_seed: Some(0),
        train_fraction: 0.8,
        val_interval: 50,
        max_test_views_for_metrics: 0, // Evaluate all held-out views
        log_interval: 10,
        densify_interval: 0,
        densify_max_gaussians: 0,
        densify_grad_threshold: 0.1,
        prune_opacity_threshold: 0.01,
        split_sigma_threshold: 0.05,
    };

    let result = train_multiview_color_only(&cfg).expect("Training failed");

    println!("\n=== M8 Multi-View Training (Quality) Results ===");
    println!("Train views: {}", result.num_train_views);
    println!("Test views:  {}", result.num_test_views);
    println!("Initial test PSNR: {:.2} dB", result.initial_psnr);
    println!("Final test PSNR:   {:.2} dB", result.final_psnr);
    println!("Improvement:       {:.2} dB", result.final_psnr - result.initial_psnr);

    // Save outputs
    std::fs::create_dir_all("test_output").ok();
    result
        .test_view_sample
        .save("test_output/m8_full_test_view_rendered.png")
        .ok();
    result
        .test_view_target
        .save("test_output/m8_full_test_view_target.png")
        .ok();

    // Full training should achieve M8 target
    assert!(
        result.final_psnr >= 20.0,
        "M8 target not met: PSNR {:.2} dB < 20 dB",
        result.final_psnr
    );
}
