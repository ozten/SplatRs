//! High-level GPU renderer interface.

use crate::core::{Camera, Gaussian};
use crate::gpu::context::GpuContext;
use nalgebra::Vector3;

pub struct GpuRenderer {
    ctx: GpuContext,
}

impl GpuRenderer {
    /// Create a new GPU renderer.
    pub fn new() -> Result<Self, String> {
        let ctx = GpuContext::new_blocking()?;
        Ok(Self { ctx })
    }

    /// Render Gaussians from a camera viewpoint.
    ///
    /// Returns linear RGB pixel values (matching CPU renderer format).
    pub fn render(
        &self,
        _gaussians: &[Gaussian],
        _camera: &Camera,
        _background: &Vector3<f32>,
    ) -> Vec<Vector3<f32>> {
        // TODO: Implement GPU rendering
        unimplemented!("GPU rendering - see M11")
    }
}
