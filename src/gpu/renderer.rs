//! High-level GPU renderer interface.

use crate::core::{Camera, Gaussian};
use crate::gpu::{buffers, context::GpuContext, shaders, types::*};
use nalgebra::{Vector2, Vector3};
use wgpu::{BindGroup, BindGroupLayout, BufferUsages, ComputePipeline};

pub struct GpuRenderer {
    ctx: GpuContext,
    project_pipeline: ComputePipeline,
    rasterize_pipeline: ComputePipeline,
    backward_pipeline: ComputePipeline,
    project_backward_pipeline: ComputePipeline,
    project_bind_group_layout: BindGroupLayout,
    rasterize_bind_group_layout: BindGroupLayout,
    backward_bind_group_layout: BindGroupLayout,
    project_backward_bind_group_layout: BindGroupLayout,
    sorter: crate::gpu::sort::BitonicSorter,
}

/// Rows per banded rasterize/backward submission. The naive per-pixel kernels do
/// O(pixels × gaussians) work; issued as ONE command buffer, that crosses the macOS/Metal
/// ~2s command-buffer watchdog above ~100k Gaussians at full-res pixel counts — Metal then
/// silently aborts the buffer and wgpu 0.19 surfaces no error, so outputs come back zeroed
/// (root-caused 2026-07-10). Splitting the image into row bands, one command buffer each,
/// bounds every buffer's work regardless of Gaussian count. The budget targets well under
/// half the watchdog limit even on a cold/loaded GPU; whole image stays a single band for
/// typical training loads (60k half-res).
fn watchdog_rows_per_band(width: u32, height: u32, num_gaussians: usize) -> u32 {
    const PIXEL_GAUSSIAN_BUDGET: u64 = 2_500_000_000;
    let per_row = (width as u64).max(1) * (num_gaussians as u64).max(1);
    let rows = (PIXEL_GAUSSIAN_BUDGET / per_row).max(16) as u32;
    (rows / 16 * 16).min(height.max(1))
}

impl GpuRenderer {
    /// Create a new GPU renderer.
    pub fn new() -> Result<Self, String> {
        let ctx = GpuContext::new_blocking()?;

        // Create shaders
        let project_shader = shaders::create_project_shader(&ctx.device);
        let rasterize_shader = shaders::create_rasterize_shader(&ctx.device);
        let backward_shader = shaders::create_backward_shader(&ctx.device);
        let project_backward_shader = shaders::create_project_backward_shader(&ctx.device);

        // Create bind group layouts
        let project_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Project Bind Group Layout"),
                    entries: &[
                        // Camera uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Gaussians input
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Gaussians output
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Settings uniform (disable_sh flag)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let rasterize_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Rasterize Bind Group Layout"),
                    entries: &[
                        // Render params uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Gaussians 2D input
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output pixels
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Intermediates buffer (for backward pass)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let backward_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Backward Bind Group Layout"),
                    entries: &[
                        // Backward params uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Intermediates input (from forward pass)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Gaussians 2D input
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Upstream gradients (d_pixels)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Workgroup gradients output
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Per-pixel background gradient output (vec4<f32> per pixel)
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let project_backward_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Project Backward Bind Group Layout"),
                    entries: &[
                        // Camera uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 3D Gaussians input
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 2D Gradients input
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 3D Gradients output
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create pipeline layouts
        let project_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Project Pipeline Layout"),
                    bind_group_layouts: &[&project_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let rasterize_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Rasterize Pipeline Layout"),
                    bind_group_layouts: &[&rasterize_bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create compute pipelines with explicit validation error capture.
        let mut create_pipeline = |label: &str,
                                   layout: &wgpu::PipelineLayout,
                                   module: &wgpu::ShaderModule,
                                   entry: &str|
         -> Result<wgpu::ComputePipeline, String> {
            ctx.device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module,
                    entry_point: entry,
                });
            if let Some(err) = pollster::block_on(ctx.device.pop_error_scope()) {
                return Err(format!("GPU pipeline `{}` failed: {}", label, err));
            }
            Ok(pipeline)
        };

        let project_pipeline = create_pipeline(
            "Project Pipeline",
            &project_pipeline_layout,
            &project_shader,
            "project_gaussians",
        )?;

        let rasterize_pipeline = create_pipeline(
            "Rasterize Pipeline",
            &rasterize_pipeline_layout,
            &rasterize_shader,
            "rasterize",
        )?;

        let backward_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Backward Pipeline Layout"),
                    bind_group_layouts: &[&backward_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let backward_pipeline = create_pipeline(
            "Backward Pipeline",
            &backward_pipeline_layout,
            &backward_shader,
            "backward_pass",
        )?;

        let project_backward_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Project Backward Pipeline Layout"),
                    bind_group_layouts: &[&project_backward_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let project_backward_pipeline = create_pipeline(
            "Project Backward Pipeline",
            &project_backward_pipeline_layout,
            &project_backward_shader,
            "project_backward",
        )?;

        // Create bitonic sorter for GPU-side sorting
        let sorter = crate::gpu::sort::BitonicSorter::new(&ctx.device);

        Ok(Self {
            ctx,
            project_pipeline,
            rasterize_pipeline,
            backward_pipeline,
            project_backward_pipeline,
            project_bind_group_layout,
            rasterize_bind_group_layout,
            backward_bind_group_layout,
            project_backward_bind_group_layout,
            sorter,
        })
    }

    /// Tile-binning Stage-1 validation surface (docs/TILE_RASTER_PLAN.md): project
    /// `gaussians` for `camera`, run the tile-touch counting kernel over the UNSORTED
    /// projection output (index i in the result corresponds to `gaussians[i]`), and
    /// return `(projected, touch_counts)` so tests can validate the GPU kernel against
    /// the CPU oracle (`render::tile_math`) on the exact f32 values the GPU consumed.
    /// Debug/test-only: builds its pipeline lazily per call, never used by rendering.
    pub fn debug_tile_touch_counts(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
    ) -> Result<(Vec<Gaussian2DGPU>, Vec<u32>), String> {
        let num_gaussians = gaussians.len();
        if num_gaussians == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let (tiles_x, tiles_y) =
            crate::render::tile_math::tile_grid_dims(camera.width, camera.height);

        let gaussians_gpu: Vec<GaussianGPU> =
            gaussians.iter().map(GaussianGPU::from_gaussian).collect();
        let camera_gpu = CameraGPU::from_camera(camera);

        let camera_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBin Camera Buffer",
            &[camera_gpu],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let gaussians_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBin Gaussians Buffer",
            &gaussians_gpu,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let projected_bytes = ((num_gaussians as u32).next_power_of_two() as usize)
            * std::mem::size_of::<Gaussian2DGPU>();
        let projected_buffer = buffers::create_buffer(
            &self.ctx.device,
            "TileBin Projected Buffer",
            projected_bytes as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let settings_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBin Settings Buffer",
            &[SettingsGPU::full_sh()],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let project_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TileBin Project Bind Group"),
            layout: &self.project_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gaussians_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: settings_buffer.as_entire_binding() },
            ],
        });

        // Counting kernel: lazily built pipeline (auto layout), debug path only.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TileBinParams {
            tiles_x: u32,
            tiles_y: u32,
            num_gaussians: u32,
            pad: u32,
        }
        let params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBin Params Buffer",
            &[TileBinParams { tiles_x, tiles_y, num_gaussians: num_gaussians as u32, pad: 0 }],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let counts_buffer = buffers::create_buffer(
            &self.ctx.device,
            "TileBin Counts Buffer",
            (num_gaussians * std::mem::size_of::<u32>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        self.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let tile_bin_shader = shaders::create_tile_bin_shader(&self.ctx.device);
        let count_pipeline =
            self.ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Tile Count Pipeline"),
                    layout: None,
                    module: &tile_bin_shader,
                    entry_point: "count_tile_touches",
                });
        if let Some(err) = pollster::block_on(self.ctx.device.pop_error_scope()) {
            return Err(format!("GPU pipeline `Tile Count Pipeline` failed: {}", err));
        }
        let count_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Count Bind Group"),
            layout: &count_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: counts_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TileBin Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TileBin Project Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.project_pipeline);
            pass.set_bind_group(0, &project_bind_group, &[]);
            pass.dispatch_workgroups(
                ((num_gaussians as u32).next_power_of_two() + 255) / 256,
                1,
                1,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tile Count Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&count_pipeline);
            pass.set_bind_group(0, &count_bind_group, &[]);
            pass.dispatch_workgroups((num_gaussians as u32 + 255) / 256, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        let projected: Vec<Gaussian2DGPU> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &projected_buffer,
            num_gaussians,
        )
        .map_err(|e| format!("Failed to read projected Gaussians: {e}"))?;
        let counts: Vec<u32> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &counts_buffer,
            num_gaussians,
        )
        .map_err(|e| format!("Failed to read tile touch counts: {e}"))?;
        Ok((projected, counts))
    }

    /// Render Gaussians from a camera viewpoint.
    ///
    /// Returns linear RGB pixel values (matching CPU renderer format).
    ///
    /// Set SUGAR_GPU_TIMING=1 environment variable for detailed timing.
    pub fn render(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
    ) -> Result<Vec<Vector3<f32>>, String> {
        self.render_with_sh_mode(gaussians, camera, background, false)
    }

    /// Render with explicit SH mode control.
    /// If `disable_sh` is true, only the DC term is used for color (view-independent).
    pub fn render_with_sh_mode(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
        disable_sh: bool,
    ) -> Result<Vec<Vector3<f32>>, String> {
        let enable_timing = std::env::var("SUGAR_GPU_TIMING").is_ok();
        let t_start = if enable_timing { Some(std::time::Instant::now()) } else { None };

        let num_gaussians = gaussians.len();
        let width = camera.width;
        let height = camera.height;
        let num_pixels = (width * height) as usize;

        if num_gaussians == 0 {
            return Ok(vec![*background; num_pixels]);
        }

        let max_storage_binding = self.ctx.device.limits().max_storage_buffer_binding_size as u64;
        let gaussians_bytes = (num_gaussians * std::mem::size_of::<GaussianGPU>()) as u64;
        // Padded to the next power of two: the bitonic sort runs over the full padded array
        // (projection fills the pad region with +inf-depth sentinels).
        let projected_bytes = ((num_gaussians as u32).next_power_of_two() as usize
            * std::mem::size_of::<Gaussian2DGPU>()) as u64;
        let output_bytes = (num_pixels * std::mem::size_of::<[f32; 4]>()) as u64;

        for (label, bytes) in [
            ("Gaussians Buffer", gaussians_bytes),
            ("Projected Buffer", projected_bytes),
            ("Output Buffer", output_bytes),
        ] {
            if bytes > max_storage_binding {
                return Err(format!(
                    "{label} size {} MB exceeds max_storage_buffer_binding_size {} MB (gaussians={}, pixels={} @ {}x{})",
                    bytes / (1024 * 1024),
                    max_storage_binding / (1024 * 1024),
                    num_gaussians,
                    num_pixels,
                    width,
                    height
                ));
            }
        }

        // Convert to GPU format
        let gaussians_gpu: Vec<GaussianGPU> =
            gaussians.iter().map(GaussianGPU::from_gaussian).collect();
        let camera_gpu = CameraGPU::from_camera(camera);

        // Create buffers
        let camera_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Camera Buffer",
            &[camera_gpu],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let gaussians_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Gaussians Buffer",
            &gaussians_gpu,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        let projected_buffer = buffers::create_buffer(
            &self.ctx.device,
            "Projected Buffer",
            projected_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Create settings buffer based on SH mode
        let settings_gpu = if disable_sh {
            SettingsGPU::dc_only()
        } else {
            SettingsGPU::full_sh()
        };
        let settings_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Settings Buffer",
            &[settings_gpu],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        // Create projection bind group
        let project_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Project Bind Group"),
            layout: &self.project_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gaussians_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: projected_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: settings_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute projection
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Project Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Project Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.project_pipeline);
            compute_pass.set_bind_group(0, &project_bind_group, &[]);
            // Dispatch over the PADDED count: threads past num_gaussians write the +inf-depth
            // sort sentinels into the pad region of the projected buffer.
            compute_pass.dispatch_workgroups(
                ((num_gaussians as u32).next_power_of_two() + 255) / 256,
                1,
                1,
            );
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        // Sort projected Gaussians by depth (GPU bitonic sort)
        if enable_timing {
            eprintln!("[GPU] Projection complete: {:?}", t_start.unwrap().elapsed());
        }

        let t_sort = if enable_timing { Some(std::time::Instant::now()) } else { None };

        // GPU-side sort: no download/upload needed!
        // Note: Invalid depths (NaN, etc.) are already marked as z=-1.0 by projection shader,
        // which sorts them to the front (will be skipped during rasterization)
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sort Encoder"),
            });

        self.sorter.sort(
            &self.ctx.device,
            &mut encoder,
            &projected_buffer,
            num_gaussians as u32,
        );

        self.ctx.queue.submit(Some(encoder.finish()));

        if enable_timing {
            eprintln!("[GPU] GPU sort: {:?}", t_sort.unwrap().elapsed());
        }

        // Sorted buffer is the same as projected buffer (in-place sort)
        let sorted_buffer = projected_buffer;

        // Create render params
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RenderParams {
            width: u32,
            height: u32,
            num_gaussians: u32,
            save_intermediates: u32,
            row_offset: u32,
            pad: [u32; 3],
            background: [f32; 4],
        }

        let params = RenderParams {
            width,
            height,
            num_gaussians: num_gaussians as u32,
            save_intermediates: 0, // Don't save intermediates in regular render
            row_offset: 0,
            pad: [0; 3],
            background: [background.x, background.y, background.z, 0.0],
        };

        let output_buffer = buffers::create_buffer_zeroed::<[f32; 4]>(
            &self.ctx.device,
            "Output Buffer",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Create a minimal pixel-state buffer (not written when save_intermediates=0).
        let pixel_state_buffer = buffers::create_buffer_zeroed::<[u32; 2]>(
            &self.ctx.device,
            "Pixel State Buffer (dummy)",
            1,
            BufferUsages::STORAGE,
        );

        // Execute rasterization in row bands — one command buffer per band so no single
        // buffer can hit the Metal watchdog (see watchdog_rows_per_band). Each band gets
        // its own params buffer + bind group (same pattern as the bitonic sorter's passes).
        let rows_per_band = watchdog_rows_per_band(width, height, num_gaussians);
        let mut row0 = 0u32;
        while row0 < height {
            let band_rows = rows_per_band.min(height - row0);
            let band_params = RenderParams {
                row_offset: row0,
                ..params
            };
            let params_buffer = buffers::create_buffer_init(
                &self.ctx.device,
                "Render Params (band)",
                &[band_params],
                BufferUsages::UNIFORM,
            );
            let rasterize_bind_group =
                self.ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Rasterize Bind Group"),
                        layout: &self.rasterize_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: sorted_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: pixel_state_buffer.as_entire_binding(),
                            },
                        ],
                    });

            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Rasterize Encoder"),
                });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Rasterize Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.rasterize_pipeline);
                compute_pass.set_bind_group(0, &rasterize_bind_group, &[]);
                compute_pass.dispatch_workgroups((width + 15) / 16, (band_rows + 15) / 16, 1);
            }
            self.ctx.queue.submit(Some(encoder.finish()));
            // Drain the queue before the next band: Metal's cumulative GPU watchdog kills
            // ALL in-flight command buffers (silently — wgpu 0.19 never checks buffer
            // status) once unfinished work piles up past ~5s. Waiting per band caps
            // in-flight work at one band (~0.4s worst case), far from the cliff.
            self.ctx.device.poll(wgpu::Maintain::Wait);
            row0 += band_rows;
        }

        // Read back results
        let output: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &output_buffer,
            num_pixels,
        )
        .map_err(|e| format!("Failed to read output buffer: {e}"))?;

        // Convert to Vector3
        let result = output
            .iter()
            .map(|rgba| Vector3::new(rgba[0], rgba[1], rgba[2]))
            .collect();

        if enable_timing {
            eprintln!("[GPU] Total render time: {:?}", t_start.unwrap().elapsed());
        }

        Ok(result)
    }

    /// Render Gaussians with gradient computation.
    ///
    /// Runs forward pass (saving intermediates), then backward pass on GPU,
    /// followed by CPU gradient reduction.
    ///
    /// # Arguments
    /// * `gaussians` - Input Gaussians
    /// * `camera` - Camera parameters
    /// * `background` - Background color
    /// * `d_pixels` - Upstream gradients (dL/d(pixel)) for each pixel
    ///
    /// # Returns
    /// * Rendered pixels (linear RGB)
    /// * Gradients w.r.t. Gaussian parameters
    pub fn render_with_gradients(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
        d_pixels: &[Vector3<f32>],
    ) -> Result<(Vec<Vector3<f32>>, crate::gpu::gradients::GaussianGradients2D), String> {

        // Constants for gradient buffer sizing
        const GRADIENT_I32_PER_GAUSSIAN: usize = 16; // 16 i32s = 64 bytes per Gaussian

        let width = camera.width;
        let height = camera.height;
        let num_pixels = (width * height) as usize;
        let num_gaussians = gaussians.len();
        if num_gaussians == 0 {
            return Ok((
                vec![*background; num_pixels],
                crate::gpu::gradients::GaussianGradients2D::zeros(0),
            ));
        }

        let enable_timing = std::env::var("SUGAR_GPU_TIMING").is_ok();
        let t_start = if enable_timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        if d_pixels.len() != num_pixels {
            return Err(format!(
                "d_pixels length must match number of pixels: got {}, expected {} ({}x{})",
                d_pixels.len(),
                num_pixels,
                width,
                height
            ));
        }

        let max_storage_binding = self.ctx.device.limits().max_storage_buffer_binding_size as u64;
        let gaussians_bytes = (num_gaussians * std::mem::size_of::<GaussianGPU>()) as u64;
        // Padded to the next power of two for the bitonic sort (see render path above).
        let projected_bytes = ((num_gaussians as u32).next_power_of_two() as usize
            * std::mem::size_of::<Gaussian2DGPU>()) as u64;
        let output_bytes = (num_pixels * std::mem::size_of::<[f32; 4]>()) as u64;
        // Per-pixel forward state: final transmittance + last contributor index (8 bytes)
        let pixel_state_bytes = (num_pixels * std::mem::size_of::<[u32; 2]>()) as u64;
        let d_pixels_bytes = (num_pixels * std::mem::size_of::<[f32; 4]>()) as u64;
        let gradient_atomic_bytes =
            (num_gaussians * GRADIENT_I32_PER_GAUSSIAN * std::mem::size_of::<i32>()) as u64;
        let d_background_pixels_bytes = (num_pixels * std::mem::size_of::<[f32; 4]>()) as u64;

        for (label, bytes) in [
            ("Gaussians Buffer", gaussians_bytes),
            ("Projected Buffer", projected_bytes),
            ("Sorted Gaussians Buffer", projected_bytes),
            ("Output Buffer", output_bytes),
            ("Pixel State Buffer", pixel_state_bytes),
            ("d_pixels Buffer", d_pixels_bytes),
            ("Per-Gaussian Gradients Buffer", gradient_atomic_bytes),
            ("Per-Pixel Background Gradients Buffer", d_background_pixels_bytes),
        ] {
            if bytes > max_storage_binding {
                return Err(format!(
                    "{label} size {} MB exceeds max_storage_buffer_binding_size {} MB (gaussians={}, pixels={} @ {}x{})",
                    bytes / (1024 * 1024),
                    max_storage_binding / (1024 * 1024),
                    num_gaussians,
                    num_pixels,
                    width,
                    height
                ));
            }
        }

        // Convert to GPU format
        let gaussians_gpu: Vec<GaussianGPU> =
            gaussians.iter().map(GaussianGPU::from_gaussian).collect();
        let camera_gpu = CameraGPU::from_camera(camera);

        // Create buffers
        let camera_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Camera Buffer",
            &[camera_gpu],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let gaussians_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Gaussians Buffer",
            &gaussians_gpu,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        let projected_buffer = buffers::create_buffer(
            &self.ctx.device,
            "Projected Buffer",
            projected_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Create settings buffer (default to full SH for backward compatibility)
        let settings_gpu = SettingsGPU::full_sh();
        let settings_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Settings Buffer",
            &[settings_gpu],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        // Execute projection
        let project_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Project Bind Group"),
            layout: &self.project_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gaussians_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: projected_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: settings_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Project Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Project Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.project_pipeline);
            compute_pass.set_bind_group(0, &project_bind_group, &[]);
            // Dispatch over the PADDED count: threads past num_gaussians write the +inf-depth
            // sort sentinels into the pad region of the projected buffer.
            compute_pass.dispatch_workgroups(
                ((num_gaussians as u32).next_power_of_two() + 255) / 256,
                1,
                1,
            );
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        // GPU-side sort (in-place, no download/upload needed)
        // Note: Invalid depths already marked as z=-1.0 by projection shader
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sort Encoder"),
            });

        self.sorter.sort(
            &self.ctx.device,
            &mut encoder,
            &projected_buffer,
            num_gaussians as u32,
        );

        self.ctx.queue.submit(Some(encoder.finish()));

        // Debug: Check projection results (only download if debugging)
        if std::env::var("SUGAR_GPU_DEBUG").is_ok() {
            let projected: Vec<Gaussian2DGPU> = buffers::read_buffer_blocking(
                &self.ctx.device,
                &self.ctx.queue,
                &projected_buffer,
                num_gaussians,
            )
            .map_err(|e| format!("Failed to read projected Gaussians: {e}"))?;

            let valid_count = projected.iter().filter(|g| g.mean[2] >= 0.0).count();
            let culled_count = projected.len() - valid_count;
            eprintln!("[GPU DEBUG] Projection results (AFTER GPU sorting):");
            eprintln!("  Valid Gaussians: {} / {}", valid_count, num_gaussians);
            eprintln!("  Culled Gaussians: {}", culled_count);

            // Show first few sorted Gaussians
            for (sorted_idx, g) in projected.iter().take(num_gaussians.min(3)).enumerate() {
                eprintln!("  SortedGaussian[{}]: mean_px=({:.2}, {:.2}), depth={:.2}, gaussian_idx_pad.x={} (original index)",
                    sorted_idx, g.mean[0], g.mean[1], g.mean[2], g.gaussian_idx_pad[0]);
            }
        }

        // Sorted buffer is the same as projected buffer (in-place sort)
        let sorted_buffer = projected_buffer;

        // Create render params with save_intermediates=1
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RenderParams {
            width: u32,
            height: u32,
            num_gaussians: u32,
            save_intermediates: u32,
            row_offset: u32,
            pad: [u32; 3],
            background: [f32; 4],
        }

        let params = RenderParams {
            width,
            height,
            num_gaussians: num_gaussians as u32,
            save_intermediates: 1, // SAVE INTERMEDIATES
            row_offset: 0,
            pad: [0; 3],
            background: [background.x, background.y, background.z, 0.0],
        };

        let output_buffer = buffers::create_buffer_zeroed::<[f32; 4]>(
            &self.ctx.device,
            "Output Buffer",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Create per-pixel state buffer (final transmittance + last contributor index)
        let pixel_state_buffer = buffers::create_buffer_zeroed::<[u32; 2]>(
            &self.ctx.device,
            "Pixel State Buffer",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC, // Allow reading back
        );

        // Execute forward pass (rasterization) in watchdog-safe row bands: per-band params
        // buffer + bind group, and drain the queue after each band so in-flight work never
        // accumulates toward Metal's cumulative (~5s) GPU watchdog — which kills ALL
        // in-flight command buffers silently under wgpu 0.19.
        let rows_per_band = watchdog_rows_per_band(width, height, num_gaussians);
        let mut row0 = 0u32;
        while row0 < height {
            let band_rows = rows_per_band.min(height - row0);
            let band_params = RenderParams {
                row_offset: row0,
                ..params
            };
            let params_buffer = buffers::create_buffer_init(
                &self.ctx.device,
                "Render Params (band)",
                &[band_params],
                BufferUsages::UNIFORM,
            );
            let rasterize_bind_group =
                self.ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Rasterize Bind Group"),
                        layout: &self.rasterize_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: sorted_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: pixel_state_buffer.as_entire_binding(),
                            },
                        ],
                    });

            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Forward Encoder"),
                });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Forward Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.rasterize_pipeline);
                compute_pass.set_bind_group(0, &rasterize_bind_group, &[]);
                compute_pass.dispatch_workgroups((width + 15) / 16, (band_rows + 15) / 16, 1);
            }
            self.ctx.queue.submit(Some(encoder.finish()));
            self.ctx.device.poll(wgpu::Maintain::Wait);
            row0 += band_rows;
        }

        // Debug: Inspect per-pixel forward state if requested
        if std::env::var("SUGAR_GPU_DEBUG").is_ok() {
            let state_data = buffers::read_buffer_blocking::<[u32; 2]>(
                &self.ctx.device,
                &self.ctx.queue,
                &pixel_state_buffer,
                num_pixels,
            )
            .map_err(|e| format!("Failed to read pixel state buffer: {e}"))?;

            let mut pixels_with_contribs = 0usize;
            let mut min_t = f32::INFINITY;
            let mut max_t = f32::NEG_INFINITY;
            for state in &state_data {
                if state[1] != 0xFFFFFFFF {
                    pixels_with_contribs += 1;
                    let t = f32::from_bits(state[0]);
                    min_t = min_t.min(t);
                    max_t = max_t.max(t);
                }
            }

            eprintln!("[GPU DEBUG] Forward pass pixel state:");
            eprintln!("  Pixels with contributors: {} / {}", pixels_with_contribs, num_pixels);
            if pixels_with_contribs > 0 {
                eprintln!("  Final transmittance range: {:.6} .. {:.6}", min_t, max_t);
            }
        }

        // Prepare for backward pass
        let _t_backward = if enable_timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Upload upstream gradients (d_pixels)
        let d_pixels_gpu: Vec<[f32; 4]> = d_pixels
            .iter()
            .map(|v| [v.x, v.y, v.z, 0.0])
            .collect();

        let d_pixels_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "d_pixels",
            &d_pixels_gpu,
            BufferUsages::STORAGE,
        );

        // Create per-Gaussian gradient buffer as i32 (initialized to zero)
        // Shader uses fixed-point i32 atomics (Metal-compatible)
        // Each Gaussian has 16 i32s (64 bytes): 4 vec4 fields × 4 components = 16 i32s
        let total_i32s = num_gaussians * GRADIENT_I32_PER_GAUSSIAN;
        let zero_grads: Vec<i32> = vec![0i32; total_i32s];
        let pixel_grads_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Per-Gaussian Gradients (i32)",
            &zero_grads,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Create per-pixel background gradient buffer (vec4<f32> per pixel)
        // Each pixel stores its contribution to d_background, which we'll sum on CPU
        let d_background_pixels_buffer = buffers::create_buffer_zeroed::<[f32; 4]>(
            &self.ctx.device,
            "Per-Pixel Background Gradients",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Create backward params
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct BackwardParams {
            width: u32,
            height: u32,
            num_gaussians: u32,
            tile_start_x: u32,
            tile_start_y: u32,
            tile_width: u32,
            tile_height: u32,
            pad: u32,
            background: [f32; 4],
        }

        if std::env::var("SUGAR_GPU_DEBUG").is_ok() {
            eprintln!("[GPU DEBUG] BackwardParams:");
            eprintln!("  width={}, height={}, num_gaussians={}", width, height, num_gaussians);
        }

        // Banded via the tile params: per-Gaussian gradients accumulate with atomicAdd and
        // the bg-gradient buffer is indexed by global pixel, so row bands are additive.
        // The deep-blend re-walk is the most expensive per-pixel kernel — same Metal
        // cumulative-watchdog constraint as the forward rasterize, so each band gets its
        // own params buffer + bind group and the queue is drained between bands.
        let backward_params = BackwardParams {
            width,
            height,
            num_gaussians: num_gaussians as u32,
            tile_start_x: 0,
            tile_start_y: 0,
            tile_width: width,
            tile_height: height,
            pad: 0,
            background: [background.x, background.y, background.z, 0.0],
        };

        let rows_per_band = watchdog_rows_per_band(width, height, num_gaussians);
        let mut row0 = 0u32;
        while row0 < height {
            let band_rows = rows_per_band.min(height - row0);
            let band_params = BackwardParams {
                tile_start_y: row0,
                tile_height: band_rows,
                ..backward_params
            };
            let backward_params_buffer = buffers::create_buffer_init(
                &self.ctx.device,
                "Backward Params (band)",
                &[band_params],
                BufferUsages::UNIFORM,
            );
            let backward_bind_group =
                self.ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Backward Bind Group"),
                        layout: &self.backward_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: backward_params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: pixel_state_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: sorted_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: d_pixels_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: pixel_grads_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: d_background_pixels_buffer.as_entire_binding(),
                            },
                        ],
                    });

            let wg_x = (width + 15) / 16;
            let wg_y = (band_rows + 15) / 16;

            if std::env::var("SUGAR_GPU_DEBUG").is_ok() {
                eprintln!("[GPU DEBUG] Backward band rows {row0}..{}:", row0 + band_rows);
                eprintln!("  workgroups: ({}, {}, 1)", wg_x, wg_y);
            }

            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Backward Encoder"),
                });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Backward Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.backward_pipeline);
                compute_pass.set_bind_group(0, &backward_bind_group, &[]);
                compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            self.ctx.queue.submit(Some(encoder.finish()));
            self.ctx.device.poll(wgpu::Maintain::Wait);
            row0 += band_rows;
        }

        // Download per-pixel gradients
        let _t_download = if enable_timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Read gradient buffer as i32 (fixed-point)
        // Color/opacity use scale 10^7, position/covariance use scale 10^9
        const FIXED_POINT_SCALE_INV: f32 = 1e-7;
        const FIXED_POINT_SCALE_POSITION_INV: f32 = 1e-9;

        let pixel_grads_i32: Vec<i32> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &pixel_grads_buffer,
            total_i32s,
        )
        .map_err(|e| format!("Failed to read per-Gaussian gradients: {e}"))?;

        // Convert from fixed-point i32 back to f32 by dividing by scale
        let mut final_grads = crate::gpu::gradients::GaussianGradients2D::zeros(num_gaussians);
        for i in 0..num_gaussians {
            let base = i * GRADIENT_I32_PER_GAUSSIAN;
            // d_color: offsets 0-2 (3 is padding)
            final_grads.d_colors[i] = Vector3::new(
                pixel_grads_i32[base + 0] as f32 * FIXED_POINT_SCALE_INV,
                pixel_grads_i32[base + 1] as f32 * FIXED_POINT_SCALE_INV,
                pixel_grads_i32[base + 2] as f32 * FIXED_POINT_SCALE_INV,
            );
            // d_opacity_logit_pad: offset 4 (5-7 are padding)
            final_grads.d_opacity_logits[i] = pixel_grads_i32[base + 4] as f32 * FIXED_POINT_SCALE_INV;
            // d_mean_px: offsets 8-9 (10-11 are padding)
            // Uses higher precision scale (10^9)
            final_grads.d_mean_px[i] = Vector2::new(
                pixel_grads_i32[base + 8] as f32 * FIXED_POINT_SCALE_POSITION_INV,
                pixel_grads_i32[base + 9] as f32 * FIXED_POINT_SCALE_POSITION_INV,
            );
            // d_cov_2d: offsets 12-14 (15 is padding)
            // Uses higher precision scale (10^9)
            final_grads.d_cov_2d[i] = Vector3::new(
                pixel_grads_i32[base + 12] as f32 * FIXED_POINT_SCALE_POSITION_INV,
                pixel_grads_i32[base + 13] as f32 * FIXED_POINT_SCALE_POSITION_INV,
                pixel_grads_i32[base + 14] as f32 * FIXED_POINT_SCALE_POSITION_INV,
            );
        }

        // Download per-pixel background gradients and sum on CPU
        let d_background_pixels: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &d_background_pixels_buffer,
            num_pixels,
        )
        .map_err(|e| format!("Failed to read per-pixel background gradients: {e}"))?;

        // Sum all per-pixel contributions to get total background gradient
        let mut d_background_sum = Vector3::zeros();
        for px in &d_background_pixels {
            d_background_sum += Vector3::new(px[0], px[1], px[2]);
        }
        final_grads.d_background = d_background_sum;

        // Read output pixels
        let output: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &output_buffer,
            num_pixels,
        )
        .map_err(|e| format!("Failed to read output buffer: {e}"))?;

        let pixels = output
            .iter()
            .map(|rgba| Vector3::new(rgba[0], rgba[1], rgba[2]))
            .collect();

        Ok((pixels, final_grads))
    }

    /// Convert 2D gradients to 3D gradients using GPU projection backward pass.
    ///
    /// This takes the 2D gradients from rasterization backward (d_mean_px, d_cov_2d)
    /// and chains them through the projection operations to get gradients w.r.t. 3D
    /// Gaussian parameters (position, scale, rotation).
    ///
    /// # Arguments
    /// * `gaussians` - Original 3D Gaussians (needed for forward data)
    /// * `camera` - Camera parameters
    /// * `gradients_2d` - 2D gradients from rasterization backward pass
    ///
    /// # Returns
    /// Gradients w.r.t. 3D Gaussian parameters (d_position, d_log_scale, d_rotation, d_sh)
    pub fn project_gradients_backward(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        gradients_2d: &crate::gpu::gradients::GaussianGradients2D,
    ) -> (Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<[[f32; 3]; 16]>) {
        let enable_timing = std::env::var("SUGAR_GPU_TIMING").is_ok();
        let t_start = if enable_timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let num_gaussians = gaussians.len();

        // Check if gradients_2d is empty (signals CPU fallback)
        if gradients_2d.d_colors.is_empty() {
            eprintln!("[GPU WARNING] Empty 2D gradients, skipping projection backward");
            return (
                vec![Vector3::zeros(); num_gaussians],
                vec![Vector3::zeros(); num_gaussians],
                vec![Vector3::zeros(); num_gaussians],
                vec![[[0.0; 3]; 16]; num_gaussians],
            );
        }

        // 1. Upload 3D Gaussians
        let gaussians_gpu: Vec<GaussianGPU> = gaussians
            .iter()
            .map(GaussianGPU::from_gaussian)
            .collect();
        let gaussians_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "3D Gaussians",
            &gaussians_gpu,
            BufferUsages::STORAGE,
        );

        // 2. Upload camera params
        let camera_gpu = CameraGPU::from_camera(camera);
        let camera_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Camera",
            &[camera_gpu],
            BufferUsages::UNIFORM,
        );

        // 3. Upload 2D gradients
        let mut gradients_2d_gpu = vec![GradientGPU::zero(); num_gaussians];
        for i in 0..num_gaussians {
            gradients_2d_gpu[i].d_color = [
                gradients_2d.d_colors[i].x,
                gradients_2d.d_colors[i].y,
                gradients_2d.d_colors[i].z,
                0.0,
            ];
            gradients_2d_gpu[i].d_opacity_logit_pad = [
                gradients_2d.d_opacity_logits[i],
                0.0,
                0.0,
                0.0,
            ];
            gradients_2d_gpu[i].d_mean_px = [
                gradients_2d.d_mean_px[i].x,
                gradients_2d.d_mean_px[i].y,
                0.0,
                0.0,
            ];
            gradients_2d_gpu[i].d_cov_2d = [
                gradients_2d.d_cov_2d[i].x,
                gradients_2d.d_cov_2d[i].y,
                gradients_2d.d_cov_2d[i].z,
                0.0,
            ];
        }

        let gradients_2d_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "2D Gradients",
            &gradients_2d_gpu,
            BufferUsages::STORAGE,
        );

        // 4. Create output buffer for 3D gradients
        let gradients_3d_buffer = buffers::create_buffer(
            &self.ctx.device,
            "3D Gradients",
            (num_gaussians * std::mem::size_of::<Gradient3DGPU>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // 5. Create bind group
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Project Backward Bind Group"),
                layout: &self.project_backward_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gaussians_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: gradients_2d_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gradients_3d_buffer.as_entire_binding(),
                    },
                ],
            });

        // 6. Dispatch compute shader
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Project Backward Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Project Backward Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.project_backward_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((num_gaussians as u32 + 255) / 256, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        if enable_timing {
            eprintln!(
                "[GPU] Projection backward dispatch: {:?}",
                t_start.unwrap().elapsed()
            );
        }

        // 7. Download results
        let gradients_3d_gpu: Vec<Gradient3DGPU> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &gradients_3d_buffer,
            num_gaussians,
        )
        .expect("Failed to read 3D gradients");

        if enable_timing {
            eprintln!(
                "[GPU] Total projection backward: {:?}",
                t_start.unwrap().elapsed()
            );
        }

        // 8. Convert to Rust format
        let mut d_positions = vec![Vector3::zeros(); num_gaussians];
        let mut d_log_scales = vec![Vector3::zeros(); num_gaussians];
        let mut d_rotations = vec![Vector3::zeros(); num_gaussians];
        let mut d_sh = vec![[[0.0f32; 3]; 16]; num_gaussians];

        for i in 0..num_gaussians {
            let grad = &gradients_3d_gpu[i];

            d_positions[i] = Vector3::new(grad.d_position[0], grad.d_position[1], grad.d_position[2]);

            d_log_scales[i] = Vector3::new(grad.d_log_scale[0], grad.d_log_scale[1], grad.d_log_scale[2]);

            // NOTE: These are SO(3) vector gradients from the shader
            // They need to be converted to quaternion gradients for the optimizer
            d_rotations[i] = Vector3::new(grad.d_rotation[0], grad.d_rotation[1], grad.d_rotation[2]);

            // SH gradients
            for j in 0..16 {
                d_sh[i][j][0] = grad.d_sh[j][0];
                d_sh[i][j][1] = grad.d_sh[j][1];
                d_sh[i][j][2] = grad.d_sh[j][2];
            }
        }

        (d_positions, d_log_scales, d_rotations, d_sh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watchdog_rows_per_band() {
        // Typical training load (60k gaussians, half-res): a few bands, 16-row aligned.
        let r = watchdog_rows_per_band(490, 273, 60_000);
        assert!(r >= 64 && r <= 273 && r % 16 == 0, "rows={r}");
        // Full-res at 60k: banded, multiple of the 16-row workgroup height.
        let r = watchdog_rows_per_band(980, 545, 60_000);
        assert!(r >= 16 && r < 545 && r % 16 == 0, "rows={r}");
        // The 2026-07-10 failure regime (211k @ full-res): small bands, never zero.
        let r = watchdog_rows_per_band(980, 545, 211_022);
        assert!(r >= 16 && r <= 64, "rows={r}");
        // Worst case (GPU hard cap): still bounded below by one workgroup row.
        assert_eq!(watchdog_rows_per_band(980, 545, 400_000), 16);
        // Degenerate inputs don't panic or return zero.
        assert!(watchdog_rows_per_band(1, 1, 0) >= 1);
        // Per-band work never exceeds the budget by more than one workgroup row.
        let rows = watchdog_rows_per_band(980, 545, 211_022) as u64;
        assert!(rows * 980 * 211_022 < 11_000_000_000);
    }
}
