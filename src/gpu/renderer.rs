//! High-level GPU renderer interface.

use crate::core::{Camera, Gaussian};
use crate::gpu::{buffers, context::GpuContext, shaders, types::*};
use nalgebra::Vector3;
use wgpu::{BindGroup, BindGroupLayout, BufferUsages, ComputePipeline};

pub struct GpuRenderer {
    ctx: GpuContext,
    project_pipeline: ComputePipeline,
    rasterize_pipeline: ComputePipeline,
    project_bind_group_layout: BindGroupLayout,
    rasterize_bind_group_layout: BindGroupLayout,
}

impl GpuRenderer {
    /// Create a new GPU renderer.
    pub fn new() -> Result<Self, String> {
        let ctx = GpuContext::new_blocking()?;

        // Create shaders
        let project_shader = shaders::create_project_shader(&ctx.device);
        let rasterize_shader = shaders::create_rasterize_shader(&ctx.device);

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

        Ok(Self {
            ctx,
            project_pipeline,
            rasterize_pipeline,
            project_bind_group_layout,
            rasterize_bind_group_layout,
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

        let output_buffer = buffers::create_buffer(
            &self.ctx.device,
            "Output Buffer",
            (num_pixels * 4 * std::mem::size_of::<f32>()) as u64,
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
}
