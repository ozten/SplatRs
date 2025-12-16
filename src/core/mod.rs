//! Core data structures and mathematical operations.
//!
//! This module contains the fundamental types used throughout the system:
//! - `Gaussian`: 3D Gaussian representation
//! - `Camera`: Camera intrinsics and extrinsics
//! - Math utilities: quaternions, covariance, projections
//!
//! All types here are "pure data" - no I/O, no rendering logic.

mod camera;
mod gaussian;
pub mod init;
mod math;
mod sh;

// Re-export public types
pub use camera::Camera;
pub use gaussian::{Gaussian, Gaussian2D, GaussianCloud};
pub use init::{init_from_colmap_points, init_from_colmap_points_visible_stratified};
pub use math::{inverse_sigmoid, perspective_jacobian, quaternion_to_matrix, sigmoid};
pub use sh::{evaluate_sh, evaluate_sh_unclamped, sh_basis};
