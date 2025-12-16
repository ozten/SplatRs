//! GPU acceleration (feature-gated).
//!
//! This module provides GPU implementations of the rendering pipeline using wgpu.
//! Only available when compiled with --features gpu

// TODO: Implement for M11-M12 (GPU rendering + training)
// mod context;
// mod buffers;
// mod pipelines;
// mod renderer;

#[cfg(feature = "gpu")]
compile_error!("GPU feature not yet implemented - see M11-M12 in docs/sugar-rs-milestones.md");
