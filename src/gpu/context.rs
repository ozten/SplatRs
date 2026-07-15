//! GPU context management - wgpu device and queue initialization.

use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::{Device, Features, Instance, Limits, Queue, RequestAdapterOptions};

/// Set when wgpu reports an uncaptured error or device loss. The 2026-07-10 full-res run
/// lost the rasterizer mid-training with zero errors surfacing in the trainer — training
/// then ran 3.5k iterations against background-only frames and destroyed the model.
/// Training loops poll this (see the render watchdog in `optim::trainer`) so a GPU fault
/// aborts the run instead of silently corrupting it.
static GPU_FAULT: AtomicBool = AtomicBool::new(false);

pub fn gpu_fault_seen() -> bool {
    GPU_FAULT.load(Ordering::Relaxed)
}

/// Query the adapter's max storage buffer binding size WITHOUT creating a device.
///
/// The auto-downsample probe used to spin up a full `GpuContext` just to read this limit
/// and then drop it — and as of the 2026-07-13 macOS update, deliberately dropping a device
/// fires the device-lost callback with reason `Unknown`, which set the global fault flag and
/// made the render watchdog abort every auto-downsample training run at iter 1 (explicit
/// `--downsample` runs skipped the probe and were unaffected). Adapters register no fault
/// callbacks and their teardown fires no device-lost, so probing must go through this
/// instead of `GpuContext::new`.
pub fn adapter_max_storage_buffer_binding_size() -> Option<u64> {
    pollster::block_on(async {
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
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;
        Some(adapter.limits().max_storage_buffer_binding_size as u64)
    })
}

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
            GPU_FAULT.store(true, Ordering::Relaxed);
            eprintln!("[wgpu] uncaptured error: {e}");
        }));
        device.set_device_lost_callback(Box::new(|reason, msg| {
            GPU_FAULT.store(true, Ordering::Relaxed);
            eprintln!("[wgpu] DEVICE LOST ({reason:?}): {msg}");
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

    #[test]
    #[ignore] // Only run when --features gpu is enabled
    fn test_adapter_limit_probe_sets_no_fault() {
        // Regression: the auto-downsample probe must not trip the render watchdog.
        // Creating and dropping a probe DEVICE fires the device-lost callback (reason
        // `Unknown` on macOS ≥ the 2026-07-13 update) and poisons the global fault flag,
        // aborting the subsequent training run at iter 1. The adapter-only probe must
        // leave the flag untouched.
        let size = adapter_max_storage_buffer_binding_size();
        assert!(size.is_some(), "no GPU adapter found");
        assert!(size.unwrap() >= 128 * 1024 * 1024);
        assert!(
            !gpu_fault_seen(),
            "adapter limit probe set the GPU fault flag — it must not create a device"
        );
    }
}
