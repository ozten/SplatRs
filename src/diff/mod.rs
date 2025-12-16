//! Differentiable operations (backward passes).
//!
//! This module implements gradient computation for all forward operations.
//! Each submodule corresponds to a forward operation in `render`.

// TODO: Implement for M6
pub mod project_grad;
// mod rasterize_grad;
pub mod blend_grad;
pub mod sh_grad;
pub mod math_grad;
pub mod gaussian2d_grad;
pub mod covariance_grad;
pub mod quaternion_grad;
