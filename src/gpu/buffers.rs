//! GPU buffer management and data upload.

use wgpu::{Buffer, BufferUsages, Device, Queue};

/// Upload data to a GPU buffer.
///
/// Creates a buffer with the given usage flags and copies data from CPU to GPU.
pub fn create_buffer_init<T: bytemuck::Pod>(
    device: &Device,
    label: &str,
    data: &[T],
    usage: BufferUsages,
) -> Buffer {
    use wgpu::util::DeviceExt;

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

/// Create an empty buffer for output.
pub fn create_buffer(
    device: &Device,
    label: &str,
    size: u64,
    usage: BufferUsages,
) -> Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// Create a zero-initialized output buffer.
///
/// This ensures deterministic behavior by initializing all memory to zero
/// before the GPU shader runs. Critical for output buffers where not all
/// elements may be written by the shader.
///
/// # Why This Matters
///
/// GPU buffers created with `mapped_at_creation: false` contain uninitialized
/// memory (random garbage). This causes non-deterministic rendering when pixels
/// aren't written by shaders, leading to training catastrophes where gradients
/// are computed from garbage values.
///
/// # Usage
///
/// ```ignore
/// // Before (WRONG - uninitialized):
/// let buffer = create_buffer(device, "Output", size, usage);
///
/// // After (CORRECT - zero-initialized):
/// let buffer = create_buffer_zeroed::<[f32; 4]>(device, "Output", count, usage);
/// ```
pub fn create_buffer_zeroed<T: bytemuck::Pod + Clone>(
    device: &Device,
    label: &str,
    count: usize,
    usage: BufferUsages,
) -> Buffer {
    use wgpu::util::DeviceExt;

    let data = vec![T::zeroed(); count];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&data),
        usage,
    })
}

/// Read data back from GPU buffer to CPU.
pub async fn read_buffer<T: bytemuck::Pod>(
    device: &Device,
    queue: &Queue,
    buffer: &Buffer,
    size: usize,
) -> Result<Vec<T>, String> {
    // Create staging buffer for readback
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (size * std::mem::size_of::<T>()) as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy GPU buffer to staging
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(
        buffer,
        0,
        &staging,
        0,
        (size * std::mem::size_of::<T>()) as u64,
    );
    queue.submit(Some(encoder.finish()));

    // Map staging buffer and read data
    let (tx, rx) = futures::channel::oneshot::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);

    rx.await
        .map_err(|_| "Channel closed".to_string())?
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

    let data = staging.slice(..).get_mapped_range();
    let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    Ok(result)
}

/// Blocking wrapper for read_buffer.
pub fn read_buffer_blocking<T: bytemuck::Pod>(
    device: &Device,
    queue: &Queue,
    buffer: &Buffer,
    size: usize,
) -> Result<Vec<T>, String> {
    pollster::block_on(read_buffer(device, queue, buffer, size))
}
