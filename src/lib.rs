//! # sugar-rs: Surface-Aligned Gaussian Splatting in Rust
//!
//! This crate implements SuGaR (Surface-Aligned Gaussian Splatting), a technique
//! for reconstructing 3D scenes from images using 3D Gaussian Splatting with
//! surface regularization to enable mesh extraction.
//!
//! ## Architecture
//!
//! The crate is organized into several modules:
//!
//! - `core`: Fundamental data structures (Gaussians, cameras, math utilities)
//! - `io`: File I/O (COLMAP parsing, PLY export, checkpoints)
//! - `render`: Forward rendering pipeline (CPU)
//! - `diff`: Differentiable operations (backward passes)
//! - `optim`: Optimization (Adam, loss functions, density control)
//! - `sugar`: SuGaR-specific functionality (regularization, mesh extraction)
//! - `gpu`: GPU acceleration (feature-gated)
//!
//! ## Learning Path
//!
//! This implementation prioritizes clarity and educational value:
//! 1. Understand the math through explicit implementations
//! 2. Verify correctness through gradient checking
//! 3. Optimize only when profiling shows need
//!
//! See `docs/` for detailed architecture, milestones, and roadmap.

// Core data structures and math
pub mod core;

// I/O operations (COLMAP, PLY, etc.)
pub mod io;

// Forward rendering pipeline
pub mod render;

// Differentiable operations (backward passes)
pub mod diff;

// Optimization (training loop, losses, etc.)
pub mod optim;

// SuGaR-specific functionality
pub mod sugar;

// GPU acceleration (optional)
#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export commonly used types at crate root for convenience
pub use core::{Camera, Gaussian, GaussianCloud};
pub use io::{ColmapScene, LoadError};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
