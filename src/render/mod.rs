//! Forward rendering pipeline (CPU implementation).
//!
//! This module implements the forward pass of Gaussian Splatting:
//! - Project 3D Gaussians to 2D
//! - Tile-based rasterization
//! - Alpha blending
//!
//! No gradients computed here - see `diff` module for backward passes.

pub mod simple;

// Re-export
pub use simple::SimpleRenderer;

// TODO: Implement full renderer for M4-M5
// mod project;
// mod rasterize;
// mod blend;
// mod cpu;
