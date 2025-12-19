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

        // Create compute pipelines
        let project_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Project Pipeline"),
                    layout: Some(&project_pipeline_layout),
                    module: &project_shader,
                    entry_point: "project_gaussians",
                });

        let rasterize_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Rasterize Pipeline"),
                    layout: Some(&rasterize_pipeline_layout),
                    module: &rasterize_shader,
                    entry_point: "rasterize",
                });

        let backward_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Backward Pipeline Layout"),
                    bind_group_layouts: &[&backward_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let backward_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Backward Pipeline"),
                    layout: Some(&backward_pipeline_layout),
                    module: &backward_shader,
                    entry_point: "backward_pass",
                });

        let project_backward_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Project Backward Pipeline Layout"),
                    bind_group_layouts: &[&project_backward_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let project_backward_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Project Backward Pipeline"),
                    layout: Some(&project_backward_pipeline_layout),
                    module: &project_backward_shader,
                    entry_point: "project_backward",
                });

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
        })
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
    ) -> Vec<Vector3<f32>> {
        let enable_timing = std::env::var("SUGAR_GPU_TIMING").is_ok();
        let t_start = if enable_timing { Some(std::time::Instant::now()) } else { None };

        // Convert to GPU format
        let gaussians_gpu: Vec<GaussianGPU> =
            gaussians.iter().map(GaussianGPU::from_gaussian).collect();
        let camera_gpu = CameraGPU::from_camera(camera);

        let num_gaussians = gaussians.len();
        let width = camera.width;
        let height = camera.height;
        let num_pixels = (width * height) as usize;

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
            (num_gaussians * std::mem::size_of::<Gaussian2DGPU>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
            compute_pass.dispatch_workgroups((num_gaussians as u32 + 255) / 256, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        // Sort projected Gaussians by depth (CPU for now)
        if enable_timing {
            eprintln!("[GPU] Projection complete: {:?}", t_start.unwrap().elapsed());
        }

        let t_sort = if enable_timing { Some(std::time::Instant::now()) } else { None };

        let mut projected: Vec<Gaussian2DGPU> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &projected_buffer,
            num_gaussians,
        )
        .expect("Failed to read projected Gaussians");

        if enable_timing {
            eprintln!("[GPU] Download projected: {:?}", t_sort.unwrap().elapsed());
        }

        let t_cpu_sort = if enable_timing { Some(std::time::Instant::now()) } else { None };

        // Filter out invalid Gaussians (NaN or inf depth) before sorting
        let original_count = projected.len();
        projected.retain(|g| g.mean[2].is_finite());
        let filtered_count = original_count - projected.len();
        if filtered_count > 0 {
            eprintln!("[GPU WARNING] Filtered {} Gaussians with invalid depth values", filtered_count);
        }

        projected.sort_by(|a, b| a.mean[2].partial_cmp(&b.mean[2]).unwrap());

        if enable_timing {
            eprintln!("[GPU] CPU sort: {:?}", t_cpu_sort.unwrap().elapsed());
        }

        let t_upload = if enable_timing { Some(std::time::Instant::now()) } else { None };

        // Upload sorted Gaussians
        let sorted_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Sorted Gaussians",
            &projected,
            BufferUsages::STORAGE,
        );

        if enable_timing {
            eprintln!("[GPU] Upload sorted: {:?}", t_upload.unwrap().elapsed());
        }

        // Create render params
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RenderParams {
            width: u32,
            height: u32,
            num_gaussians: u32,
            save_intermediates: u32,
            background: [f32; 4],
        }

        let params = RenderParams {
            width,
            height,
            num_gaussians: num_gaussians as u32,
            save_intermediates: 0, // Don't save intermediates in regular render
            background: [background.x, background.y, background.z, 0.0],
        };

        let params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Render Params",
            &[params],
            BufferUsages::UNIFORM,
        );

        let output_buffer = buffers::create_buffer_zeroed::<[f32; 4]>(
            &self.ctx.device,
            "Output Buffer",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Create dummy intermediates buffer (not used when save_intermediates=0)
        use crate::gpu::types::{ContributionGPU, MAX_CONTRIBUTIONS_PER_PIXEL};
        let intermediates_buffer = buffers::create_buffer(
            &self.ctx.device,
            "Intermediates Buffer (dummy)",
            (num_pixels * MAX_CONTRIBUTIONS_PER_PIXEL as usize
                * std::mem::size_of::<ContributionGPU>()) as u64,
            BufferUsages::STORAGE,
        );

        // Create rasterize bind group
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
                            resource: intermediates_buffer.as_entire_binding(),
                        },
                    ],
                });

        // Execute rasterization
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
            compute_pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        // Read back results
        let output: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &output_buffer,
            num_pixels,
        )
        .expect("Failed to read output");

        // Convert to Vector3
        let result = output
            .iter()
            .map(|rgba| Vector3::new(rgba[0], rgba[1], rgba[2]))
            .collect();

        if enable_timing {
            eprintln!("[GPU] Total render time: {:?}", t_start.unwrap().elapsed());
        }

        result
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
    ) -> (Vec<Vector3<f32>>, crate::gpu::gradients::GaussianGradients2D) {
        use crate::gpu::types::{ContributionGPU, GradientGPU, MAX_CONTRIBUTIONS_PER_PIXEL};

        // Constants for gradient buffer sizing
        const GRADIENT_I32_PER_GAUSSIAN: usize = 16; // 16 i32s = 64 bytes per Gaussian
        const MAX_GPU_BUFFER_SIZE: u64 = 256 * 1024 * 1024; // 256 MB

        // Check if per-Gaussian gradient buffer would exceed GPU limit
        // With sparse atomic gradients, buffer size is just num_gaussians × 64 bytes
        let num_pixels = (camera.width * camera.height) as usize;
        let num_gaussians = gaussians.len();
        let gradient_buffer_size = (num_gaussians * GRADIENT_I32_PER_GAUSSIAN * std::mem::size_of::<i32>()) as u64;

        if gradient_buffer_size > MAX_GPU_BUFFER_SIZE {
            eprintln!("[GPU WARNING] Gradient buffer would be {} GB (> {} MB limit)",
                gradient_buffer_size / (1024 * 1024 * 1024),
                MAX_GPU_BUFFER_SIZE / (1024 * 1024));
            eprintln!("[GPU WARNING] Scene too large for GPU gradients: {} gaussians × {}×{} pixels",
                num_gaussians, camera.width, camera.height);
            eprintln!("[GPU WARNING] Falling back to CPU for backward pass");

            // Return empty gradients to signal fallback
            return (vec![], crate::gpu::gradients::GaussianGradients2D::empty());
        }

        let enable_timing = std::env::var("SUGAR_GPU_TIMING").is_ok();
        let t_start = if enable_timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Convert to GPU format
        let gaussians_gpu: Vec<GaussianGPU> =
            gaussians.iter().map(GaussianGPU::from_gaussian).collect();
        let camera_gpu = CameraGPU::from_camera(camera);

        let num_gaussians = gaussians.len();
        let width = camera.width;
        let height = camera.height;
        let num_pixels = (width * height) as usize;

        // Validate d_pixels
        assert_eq!(
            d_pixels.len(),
            num_pixels,
            "d_pixels length must match number of pixels"
        );

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
            (num_gaussians * std::mem::size_of::<Gaussian2DGPU>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
            compute_pass.dispatch_workgroups((num_gaussians as u32 + 255) / 256, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        // Download, sort, and re-upload projected Gaussians
        let mut projected: Vec<Gaussian2DGPU> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &projected_buffer,
            num_gaussians,
        )
        .expect("Failed to read projected Gaussians");

        // Debug: Check how many Gaussians survived projection
        if std::env::var("SUGAR_GPU_DEBUG").is_ok() {
            let valid_count = projected.iter().filter(|g| g.mean[2] >= 0.0).count();
            let culled_count = projected.len() - valid_count;
            eprintln!("[GPU DEBUG] Projection results (BEFORE sorting):");
            eprintln!("  Valid Gaussians: {} / {}", valid_count, num_gaussians);
            eprintln!("  Culled Gaussians: {}", culled_count);

            // Show projected Gaussians BEFORE sorting
            for (i, g) in projected.iter().take(num_gaussians.min(3)).enumerate() {
                eprintln!("  Gaussian {} (sorted_pos={}): mean_px=({:.2}, {:.2}), depth={:.2}, gaussian_idx_pad.x={}",
                    i, i, g.mean[0], g.mean[1], g.mean[2], g.gaussian_idx_pad[0]);
            }
        }

        projected.sort_by(|a, b| a.mean[2].partial_cmp(&b.mean[2]).unwrap());

        if std::env::var("SUGAR_GPU_DEBUG").is_ok() {
            eprintln!("[GPU DEBUG] Projection results (AFTER sorting by depth):");
            for (sorted_idx, g) in projected.iter().take(num_gaussians.min(3)).enumerate() {
                eprintln!("  SortedGaussian[{}]: depth={:.2}, gaussian_idx_pad.x={} (original index)",
                    sorted_idx, g.mean[2], g.gaussian_idx_pad[0]);
            }
        }

        let sorted_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Sorted Gaussians",
            &projected,
            BufferUsages::STORAGE,
        );

        // Create render params with save_intermediates=1
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RenderParams {
            width: u32,
            height: u32,
            num_gaussians: u32,
            save_intermediates: u32,
            background: [f32; 4],
        }

        let params = RenderParams {
            width,
            height,
            num_gaussians: num_gaussians as u32,
            save_intermediates: 1, // SAVE INTERMEDIATES
            background: [background.x, background.y, background.z, 0.0],
        };

        let params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Render Params",
            &[params],
            BufferUsages::UNIFORM,
        );

        let output_buffer = buffers::create_buffer_zeroed::<[f32; 4]>(
            &self.ctx.device,
            "Output Buffer",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Create intermediates buffer (ACTUALLY USED this time)
        let intermediates_buffer = buffers::create_buffer_zeroed::<ContributionGPU>(
            &self.ctx.device,
            "Intermediates Buffer",
            num_pixels * MAX_CONTRIBUTIONS_PER_PIXEL as usize,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC, // Allow reading back
        );

        // Create rasterize bind group
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
                            resource: intermediates_buffer.as_entire_binding(),
                        },
                    ],
                });

        // Execute forward pass (rasterization)
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
            compute_pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        // Debug: Inspect intermediates buffer if requested
        if std::env::var("SUGAR_GPU_DEBUG").is_ok() {
            let intermediates_data = buffers::read_buffer_blocking::<ContributionGPU>(
                &self.ctx.device,
                &self.ctx.queue,
                &intermediates_buffer,
                num_pixels * MAX_CONTRIBUTIONS_PER_PIXEL as usize,
            ).expect("Failed to read intermediates buffer");

            let mut pixels_with_contribs = 0;
            let mut total_contribs = 0;

            for px_idx in 0..num_pixels {
                let base = px_idx * MAX_CONTRIBUTIONS_PER_PIXEL as usize;
                let mut px_contrib_count = 0;

                for i in 0..MAX_CONTRIBUTIONS_PER_PIXEL as usize {
                    let contrib = &intermediates_data[base + i];
                    if contrib.gaussian_idx == 0xFFFFFFFF {
                        break;
                    }
                    px_contrib_count += 1;
                }

                if px_contrib_count > 0 {
                    pixels_with_contribs += 1;
                    total_contribs += px_contrib_count;
                }
            }

            eprintln!("[GPU DEBUG] Forward pass saved intermediates:");
            eprintln!("  Pixels with contributions: {} / {}", pixels_with_contribs, num_pixels);
            eprintln!("  Total contributions saved: {}", total_contribs);
            eprintln!("  Average contributions per active pixel: {:.1}",
                if pixels_with_contribs > 0 { total_contribs as f32 / pixels_with_contribs as f32 } else { 0.0 });

            // Show first few pixels with contributions
            eprintln!("  First few pixels with contributions:");
            let mut shown = 0;
            for px_idx in 0..num_pixels {
                let base = px_idx * MAX_CONTRIBUTIONS_PER_PIXEL as usize;
                let contrib = &intermediates_data[base];
                if contrib.gaussian_idx != 0xFFFFFFFF {
                    let x = px_idx % (width as usize);
                    let y = px_idx / (width as usize);
                    eprintln!("    Pixel {} (x={}, y={}): gaussian_idx={}, transmittance={:.6}, alpha={:.6}",
                        px_idx, x, y, contrib.gaussian_idx, contrib.transmittance, contrib.alpha);
                    shown += 1;
                    if shown >= 10 { break; }
                }
            }

            // Check specific pixels that will have gradients later
            eprintln!("  Checking pixels that will show gradients (828-830, 874-875):");
            for px_idx in [828, 829, 830, 874, 875] {
                let base = px_idx * MAX_CONTRIBUTIONS_PER_PIXEL as usize;
                let contrib = &intermediates_data[base];
                let has_int = contrib.gaussian_idx != 0xFFFFFFFF;
                eprintln!("    Pixel {}: has_intermediate={}", px_idx, has_int);
                if has_int {
                    eprintln!("      gaussian_idx={}, alpha={:.6}", contrib.gaussian_idx, contrib.alpha);
                }
            }
        }

        // Prepare for backward pass
        let t_backward = if enable_timing {
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

        let backward_params = BackwardParams {
            width,
            height,
            num_gaussians: num_gaussians as u32,
            tile_start_x: 0,      // Non-tiled: full image
            tile_start_y: 0,      // Non-tiled: full image
            tile_width: width,    // Non-tiled: full image
            tile_height: height,  // Non-tiled: full image
            pad: 0,
            background: [background.x, background.y, background.z, 0.0],
        };

        let backward_params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Backward Params",
            &[backward_params],
            BufferUsages::UNIFORM,
        );

        // Create backward bind group
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
                            resource: intermediates_buffer.as_entire_binding(),
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

        // Execute backward pass
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Backward Encoder"),
            });

        {
            let wg_x = (width + 15) / 16;
            let wg_y = (height + 15) / 16;

            if std::env::var("SUGAR_GPU_DEBUG").is_ok() {
                eprintln!("[GPU DEBUG] Backward pass dispatch:");
                eprintln!("  workgroups: ({}, {}, 1)", wg_x, wg_y);
                eprintln!("  total threads: ({}, {})", wg_x * 16, wg_y * 16);
            }

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Backward Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.backward_pipeline);
            compute_pass.set_bind_group(0, &backward_bind_group, &[]);
            compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        // Download per-pixel gradients
        let t_download = if enable_timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Read gradient buffer as i32 (fixed-point representation)
        let pixel_grads_i32: Vec<i32> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &pixel_grads_buffer,
            total_i32s,
        )
        .expect("Failed to read per-Gaussian gradients");

        // Convert from fixed-point i32 to f32 gradients
        // Shader uses FIXED_POINT_SCALE = 10000000.0
        const FIXED_POINT_SCALE: f32 = 10000000.0;
        let mut final_grads = crate::gpu::gradients::GaussianGradients2D::zeros(num_gaussians);
        for i in 0..num_gaussians {
            let base = i * GRADIENT_I32_PER_GAUSSIAN;
            // d_color: offsets 0-2 (3 is padding)
            final_grads.d_colors[i] = Vector3::new(
                pixel_grads_i32[base + 0] as f32 / FIXED_POINT_SCALE,
                pixel_grads_i32[base + 1] as f32 / FIXED_POINT_SCALE,
                pixel_grads_i32[base + 2] as f32 / FIXED_POINT_SCALE,
            );
            // d_opacity_logit_pad: offset 4 (5-7 are padding)
            final_grads.d_opacity_logits[i] = pixel_grads_i32[base + 4] as f32 / FIXED_POINT_SCALE;
            // d_mean_px: offsets 8-9 (10-11 are padding)
            final_grads.d_mean_px[i] = Vector2::new(
                pixel_grads_i32[base + 8] as f32 / FIXED_POINT_SCALE,
                pixel_grads_i32[base + 9] as f32 / FIXED_POINT_SCALE,
            );
            // d_cov_2d: offsets 12-14 (15 is padding)
            final_grads.d_cov_2d[i] = Vector3::new(
                pixel_grads_i32[base + 12] as f32 / FIXED_POINT_SCALE,
                pixel_grads_i32[base + 13] as f32 / FIXED_POINT_SCALE,
                pixel_grads_i32[base + 14] as f32 / FIXED_POINT_SCALE,
            );
        }

        // Download per-pixel background gradients and sum on CPU
        let d_background_pixels: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &d_background_pixels_buffer,
            num_pixels,
        )
        .expect("Failed to read per-pixel background gradients");

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
        .expect("Failed to read output");

        let pixels = output
            .iter()
            .map(|rgba| Vector3::new(rgba[0], rgba[1], rgba[2]))
            .collect();

        (pixels, final_grads)
    }

    /// Calculate optimal tile size based on memory constraints.
    ///
    /// Returns tile size (32 or 16) or 0 if even tiling won't fit in memory.
    fn calculate_tile_size(num_gaussians: usize) -> u32 {
        const MAX_BUFFER: u64 = 128 * 1024 * 1024; // 128 MB (conservative for Apple Silicon)
        let grad_size = std::mem::size_of::<crate::gpu::types::GradientGPU>() as u64;

        // Try 32×32 first (optimal for most cases)
        let tile_32_pixels = 32 * 32;
        let tile_32_buffer = (tile_32_pixels as u64) * (num_gaussians as u64) * grad_size;
        if tile_32_buffer <= MAX_BUFFER {
            return 32;
        }

        // Fall back to 16×16 (for very large scenes)
        let tile_16_pixels = 16 * 16;
        let tile_16_buffer = (tile_16_pixels as u64) * (num_gaussians as u64) * grad_size;
        if tile_16_buffer <= MAX_BUFFER {
            return 16;
        }

        // Still too large - signal CPU fallback
        eprintln!("[GPU WARNING] Even 16×16 tiles would exceed memory limit");
        eprintln!("[GPU WARNING] Required: {} MB per tile", tile_16_buffer / (1024 * 1024));
        0
    }

    /// Render Gaussians with gradient computation using tiled backward pass.
    ///
    /// This method processes the image in tiles to reduce memory usage for large scenes.
    /// Each tile is processed independently on the GPU, then gradients are accumulated on CPU.
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
    pub fn render_with_gradients_tiled(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
        d_pixels: &[Vector3<f32>],
    ) -> (Vec<Vector3<f32>>, crate::gpu::gradients::GaussianGradients2D) {
        use crate::gpu::gradients::accumulate_tile_gradients;
        use crate::gpu::types::{ContributionGPU, GradientGPU, MAX_CONTRIBUTIONS_PER_PIXEL};

        let num_gaussians = gaussians.len();
        let width = camera.width;
        let height = camera.height;
        let num_pixels = (width * height) as usize;

        // Validate inputs
        assert_eq!(
            d_pixels.len(),
            num_pixels,
            "d_pixels length must match number of pixels"
        );

        // Determine optimal tile size
        let tile_size = Self::calculate_tile_size(num_gaussians);
        if tile_size == 0 {
            eprintln!("[GPU WARNING] Scene too large even for tiled gradients");
            eprintln!("[GPU WARNING] Falling back to CPU for backward pass");
            return (vec![], crate::gpu::gradients::GaussianGradients2D::empty());
        }

        eprintln!("[GPU INFO] Using {}×{} tiles for {} gaussians", tile_size, tile_size, num_gaussians);

        let enable_timing = std::env::var("SUGAR_GPU_TIMING").is_ok();
        let t_start = if enable_timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Run forward pass once to get pixels and intermediates
        // (Forward pass is shared - we only tile the backward pass)

        // Convert inputs to GPU format (shared across all tiles)
        let gaussians_gpu: Vec<crate::gpu::types::GaussianGPU> =
            gaussians.iter().map(crate::gpu::types::GaussianGPU::from_gaussian).collect();
        let camera_gpu = crate::gpu::types::CameraGPU::from_camera(camera);

        // Create shared buffers (used for all tiles)
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
            (num_gaussians * std::mem::size_of::<crate::gpu::types::Gaussian2DGPU>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Project Gaussians
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
            ],
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Project Encoder"),
        });

        {
            let workgroups = (num_gaussians as u32 + 255) / 256;
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Project Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.project_pipeline);
            compute_pass.set_bind_group(0, &project_bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        // Download and sort projected Gaussians on CPU
        let projected: Vec<crate::gpu::types::Gaussian2DGPU> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &projected_buffer,
            num_gaussians,
        )
        .expect("Failed to read projected Gaussians");

        let mut projected = projected;
        projected.retain(|g| g.mean[2].is_finite());
        projected.sort_by(|a, b| a.mean[2].partial_cmp(&b.mean[2]).unwrap());

        let sorted_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Sorted Buffer",
            &projected,
            BufferUsages::STORAGE,
        );

        // Rasterize forward pass with intermediates
        let output_buffer = buffers::create_buffer_zeroed::<[f32; 4]>(
            &self.ctx.device,
            "Output Buffer",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let intermediates_buffer = buffers::create_buffer_zeroed::<ContributionGPU>(
            &self.ctx.device,
            "Intermediates Buffer",
            num_pixels * MAX_CONTRIBUTIONS_PER_PIXEL as usize,
            BufferUsages::STORAGE,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RenderParams {
            width: u32,
            height: u32,
            num_gaussians: u32,
            save_intermediates: u32,
            background: [f32; 4],
        }

        let rasterize_params = RenderParams {
            width,
            height,
            num_gaussians: num_gaussians as u32,
            save_intermediates: 1,
            background: [background.x, background.y, background.z, 0.0],
        };

        let rasterize_params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Rasterize Params",
            &[rasterize_params],
            BufferUsages::UNIFORM,
        );

        let rasterize_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rasterize Bind Group"),
            layout: &self.rasterize_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rasterize_params_buffer.as_entire_binding(),
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
                    resource: intermediates_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Rasterize Encoder"),
        });

        {
            let wg_x = (width + 15) / 16;
            let wg_y = (height + 15) / 16;
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Rasterize Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.rasterize_pipeline);
            compute_pass.set_bind_group(0, &rasterize_bind_group, &[]);
            compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        // Download rendered pixels from forward pass
        let output: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &output_buffer,
            num_pixels,
        )
        .expect("Failed to read output pixels");

        let pixels: Vec<Vector3<f32>> = output
            .iter()
            .map(|rgba| Vector3::new(rgba[0], rgba[1], rgba[2]))
            .collect();

        // Initialize final gradients
        let mut final_grads = crate::gpu::gradients::GaussianGradients2D::zeros(num_gaussians);

        // Upload upstream gradients
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

        // Create per-pixel background gradient buffer (shared across all tiles)
        // Each pixel stores its contribution to d_background, which we'll sum on CPU
        let d_background_pixels_buffer = buffers::create_buffer_zeroed::<[f32; 4]>(
            &self.ctx.device,
            "Per-Pixel Background Gradients",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Process each tile
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;

        eprintln!("[GPU INFO] Processing {} tiles ({}×{})", tiles_x * tiles_y, tiles_x, tiles_y);

        let t_tiles_start = if enable_timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                // Compute tile bounds
                let start_x = tile_x * tile_size;
                let start_y = tile_y * tile_size;
                let end_x = (start_x + tile_size).min(width);
                let end_y = (start_y + tile_size).min(height);
                let tile_width = end_x - start_x;
                let tile_height = end_y - start_y;
                let tile_pixels = (tile_width * tile_height) as usize;

                // Create per-tile gradient buffer
                let tile_grad_entries = tile_pixels * num_gaussians;
                let zero_grads = vec![GradientGPU::zero(); tile_grad_entries];
                let tile_grads_buffer = buffers::create_buffer_init(
                    &self.ctx.device,
                    &format!("Tile Gradients ({},{})", tile_x, tile_y),
                    &zero_grads,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                );

                // Create backward params for this tile
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

                let backward_params = BackwardParams {
                    width,
                    height,
                    num_gaussians: num_gaussians as u32,
                    tile_start_x: start_x,
                    tile_start_y: start_y,
                    tile_width,
                    tile_height,
                    pad: 0,
                    background: [background.x, background.y, background.z, 0.0],
                };

                let backward_params_buffer = buffers::create_buffer_init(
                    &self.ctx.device,
                    "Backward Params",
                    &[backward_params],
                    BufferUsages::UNIFORM,
                );

                // Create backward bind group for this tile
                let backward_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Backward Bind Group ({},{})", tile_x, tile_y)),
                    layout: &self.backward_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: backward_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: intermediates_buffer.as_entire_binding(),
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
                            resource: tile_grads_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: d_background_pixels_buffer.as_entire_binding(),
                        },
                    ],
                });

                // Execute backward pass for this tile
                let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("Backward Encoder ({},{})", tile_x, tile_y)),
                });

                {
                    let wg_x = (tile_width + 15) / 16;
                    let wg_y = (tile_height + 15) / 16;
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Backward Pass ({},{})", tile_x, tile_y)),
                        timestamp_writes: None,
                    });
                    compute_pass.set_pipeline(&self.backward_pipeline);
                    compute_pass.set_bind_group(0, &backward_bind_group, &[]);
                    compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
                }

                self.ctx.queue.submit(Some(encoder.finish()));

                // Download tile gradients
                let tile_grads: Vec<GradientGPU> = buffers::read_buffer_blocking(
                    &self.ctx.device,
                    &self.ctx.queue,
                    &tile_grads_buffer,
                    tile_grad_entries,
                )
                .expect("Failed to read tile gradients");

                // Accumulate into final gradients
                accumulate_tile_gradients(&tile_grads, &mut final_grads, tile_pixels, num_gaussians);
            }
        }

        if enable_timing {
            eprintln!(
                "[GPU] All tiles processed: {:?}",
                t_tiles_start.unwrap().elapsed()
            );
        }

        // Download per-pixel background gradients and sum on CPU
        let d_background_pixels: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &d_background_pixels_buffer,
            num_pixels,
        )
        .expect("Failed to read per-pixel background gradients");

        // Sum all per-pixel contributions to get total background gradient
        let mut d_background_sum = Vector3::zeros();
        for px in &d_background_pixels {
            d_background_sum += Vector3::new(px[0], px[1], px[2]);
        }
        final_grads.d_background = d_background_sum;

        (pixels, final_grads)
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
    fn test_calculate_tile_size_small_scene() {
        // 1,000 gaussians with 64 bytes per gradient
        // 32×32 = 1024 pixels × 1000 × 64 = 65.5 MB < 128 MB ✓
        assert_eq!(GpuRenderer::calculate_tile_size(1000), 32);
    }

    #[test]
    fn test_calculate_tile_size_medium_scene() {
        // 3,000 gaussians with 64 bytes per gradient
        // 32×32 = 1024 pixels × 3000 × 64 = 196.6 MB > 128 MB ❌
        // 16×16 = 256 pixels × 3000 × 64 = 49.2 MB < 128 MB ✓
        assert_eq!(GpuRenderer::calculate_tile_size(3000), 16);
    }

    #[test]
    fn test_calculate_tile_size_huge_scene() {
        // 100,000 gaussians with 64 bytes per gradient
        // 16×16 = 256 pixels × 100000 × 64 = 1.6 GB > 128 MB ❌
        // Should return 0 to signal CPU fallback
        assert_eq!(GpuRenderer::calculate_tile_size(100000), 0);
    }
}
