//! GPU context management - wgpu device and queue initialization.

use wgpu::{Device, Features, Instance, Limits, Queue, RequestAdapterOptions};

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
            backends: {
                #[cfg(target_os = "macos")]
                {
                    wgpu::Backends::METAL
                }
                #[cfg(not(target_os = "macos"))]
                {
                    wgpu::Backends::PRIMARY
                }
            },
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

        // Log adapter limits
        let limits = adapter.limits();
        eprintln!("GPU max storage buffer binding size: {} MB",
            limits.max_storage_buffer_binding_size / (1024 * 1024));

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("SuGaR GPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        device.on_uncaptured_error(Box::new(|e| {
            eprintln!("[wgpu] uncaptured error: {e}");
        }));

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
