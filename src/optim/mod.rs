//! Optimization components (training loop, losses, density control).
//!
//! This module contains everything needed for training:
//! - Adam optimizer
//! - Loss functions (L1, SSIM)
//! - Adaptive density control (split/clone/prune)
//! - Training orchestration

// TODO: Implement for M7-M10
pub mod adam;
pub mod loss;
// mod density;
pub mod trainer;
