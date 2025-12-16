//! Forward rendering pipeline (CPU implementation).
//!
//! This module implements the forward pass of Gaussian Splatting:
//! - Project 3D Gaussians to 2D
//! - Tile-based rasterization
//! - Alpha blending
//!
//! No gradients computed here - see `diff` module for backward passes.

pub mod simple;
pub mod full;
pub mod full_diff;

// Re-export
pub use simple::SimpleRenderer;
pub use full::FullRenderer;
pub use full_diff::{render_full_color_grads, render_full_linear};

// TODO: Implement full renderer for M4-M5
// mod project;
// mod rasterize;
// mod blend;
// mod cpu;
