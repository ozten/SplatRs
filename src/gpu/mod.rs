//! GPU acceleration (feature-gated).
//!
//! This module provides GPU implementations of the rendering pipeline using wgpu.
//! Only available when compiled with --features gpu
//!
//! Architecture (M11-M12):
//! - `context` - wgpu device/queue initialization
//! - `buffers` - GPU buffer management
//! - `shaders` - WGSL shader modules
//! - `renderer` - High-level rendering interface

#[cfg(feature = "gpu")]
mod context;
#[cfg(feature = "gpu")]
mod buffers;
#[cfg(feature = "gpu")]
mod types;
#[cfg(feature = "gpu")]
mod shaders;
#[cfg(feature = "gpu")]
mod renderer;

#[cfg(feature = "gpu")]
pub use context::GpuContext;
#[cfg(feature = "gpu")]
pub use renderer::GpuRenderer;
#[cfg(feature = "gpu")]
pub use types::{
    CameraGPU, ContributionGPU, Gaussian2DGPU, GaussianGPU, GradientGPU,
    MAX_CONTRIBUTIONS_PER_PIXEL,
};

#[cfg(not(feature = "gpu"))]
pub struct GpuRenderer;

#[cfg(not(feature = "gpu"))]
impl GpuRenderer {
    pub fn new() -> Result<Self, String> {
        Err("GPU support not enabled. Compile with --features gpu".to_string())
    }
}
