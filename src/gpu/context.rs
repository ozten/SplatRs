//! GPU context management - wgpu device and queue initialization.

use wgpu::{Device, Queue, Instance, Adapter, RequestAdapterOptions, DeviceDescriptor, Features, Limits};

pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    /// Initialize GPU context asynchronously.
    ///
    /// Selects the first available GPU adapter and creates a device with
    /// compute shader support.
    pub async fn new() -> Result<Self, String> {
        // Create wgpu instance (API entry point)
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request an adapter (physical GPU)
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or("Failed to find GPU adapter")?;

        // Log adapter info
        let info = adapter.get_info();
        eprintln!("GPU: {} ({:?})", info.name, info.backend);

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("SuGaR GPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        Ok(Self { device, queue })
    }

    /// Synchronous wrapper using pollster.
    ///
    /// This blocks the current thread until GPU initialization completes.
    /// Use this for CLI tools where async isn't worth the complexity.
    pub fn new_blocking() -> Result<Self, String> {
        pollster::block_on(Self::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run when --features gpu is enabled
    fn test_gpu_context_init() {
        let ctx = GpuContext::new_blocking();
        assert!(ctx.is_ok(), "GPU context initialization failed");
    }
}
