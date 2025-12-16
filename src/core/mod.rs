//! Core data structures and mathematical operations.
//!
//! This module contains the fundamental types used throughout the system:
//! - `Gaussian`: 3D Gaussian representation
//! - `Camera`: Camera intrinsics and extrinsics
//! - Math utilities: quaternions, covariance, projections
//!
//! All types here are "pure data" - no I/O, no rendering logic.

mod gaussian;
mod camera;
mod sh;
mod math;
pub mod init;

// Re-export public types
pub use gaussian::{Gaussian, Gaussian2D, GaussianCloud};
pub use camera::Camera;
pub use sh::{evaluate_sh, sh_basis};
pub use math::{inverse_sigmoid, perspective_jacobian, quaternion_to_matrix, sigmoid};
pub use init::init_from_colmap_points;
