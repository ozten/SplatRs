//! Forward rendering pipeline (CPU implementation).
//!
//! This module implements the forward pass of Gaussian Splatting:
//! - Project 3D Gaussians to 2D
//! - Tile-based rasterization
//! - Alpha blending
//!
//! No gradients computed here - see `diff` module for backward passes.

pub mod full;
pub mod full_diff;
pub mod simple;
// CPU reference math for GPU tile-binned rasterization (docs/TILE_RASTER_PLAN.md).
// Lives here rather than src/gpu/ because lib.rs feature-gates the entire gpu module,
// and this must be available to CPU-only builds/tests as the tile-binning oracle.
pub mod tile_math;

// Re-export
pub use full::FullRenderer;
pub use full_diff::{render_full_color_grads, render_full_color_grads_ext, render_full_linear};
pub use simple::SimpleRenderer;

// TODO: Implement full renderer for M4-M5
// mod project;
// mod rasterize;
// mod blend;
// mod cpu;
