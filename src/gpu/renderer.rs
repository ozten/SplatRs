//! High-level GPU renderer interface.

use crate::core::{Camera, Gaussian};
use crate::gpu::{buffers, context::GpuContext, shaders, types::*};
use nalgebra::Vector3;
use wgpu::{BindGroup, BindGroupLayout, BufferUsages, ComputePipeline};

pub struct GpuRenderer {
    ctx: GpuContext,
    project_pipeline: ComputePipeline,
    rasterize_pipeline: ComputePipeline,
    backward_pipeline: ComputePipeline,
    project_bind_group_layout: BindGroupLayout,
    rasterize_bind_group_layout: BindGroupLayout,
    backward_bind_group_layout: BindGroupLayout,
}

impl GpuRenderer {
    /// Create a new GPU renderer.
    pub fn new() -> Result<Self, String> {
        let ctx = GpuContext::new_blocking()?;

        // Create shaders
        let project_shader = shaders::create_project_shader(&ctx.device);
        let rasterize_shader = shaders::create_rasterize_shader(&ctx.device);
        let backward_shader = shaders::create_backward_shader(&ctx.device);

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
                        // Debug info output
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

        Ok(Self {
            ctx,
            project_pipeline,
            rasterize_pipeline,
            backward_pipeline,
            project_bind_group_layout,
            rasterize_bind_group_layout,
            backward_bind_group_layout,
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
        use crate::gpu::gradients::reduce_pixel_gradients;
        use crate::gpu::types::{ContributionGPU, GradientGPU, MAX_CONTRIBUTIONS_PER_PIXEL};

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

        let output_buffer = buffers::create_buffer(
            &self.ctx.device,
            "Output Buffer",
            (num_pixels * 4 * std::mem::size_of::<f32>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Create intermediates buffer (ACTUALLY USED this time)
        let intermediates_buffer = buffers::create_buffer(
            &self.ctx.device,
            "Intermediates Buffer",
            (num_pixels * MAX_CONTRIBUTIONS_PER_PIXEL as usize
                * std::mem::size_of::<ContributionGPU>()) as u64,
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

        if enable_timing {
            eprintln!(
                "[GPU] Forward pass (with intermediates): {:?}",
                t_start.unwrap().elapsed()
            );
        }

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

        // Create per-pixel gradient buffer (initialized to zero)
        // This avoids race conditions - each pixel gets its own gradient section
        let total_grad_entries = num_pixels * num_gaussians;
        let zero_grads = vec![GradientGPU::zero(); total_grad_entries];
        let pixel_grads_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Per-Pixel Gradients",
            &zero_grads,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        // Create backward params
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct BackwardParams {
            width: u32,
            height: u32,
            num_gaussians: u32,
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
            pad: 0,
            background: [background.x, background.y, background.z, 0.0],
        };

        let backward_params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Backward Params",
            &[backward_params],
            BufferUsages::UNIFORM,
        );

        // Create debug buffer (one vec4<u32> per pixel)
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct DebugInfo {
            pixel_idx: u32,
            num_contribs: u32,
            first_grad_idx: u32,
            first_gaussian_idx: u32,
        }
        let zero_debug = vec![DebugInfo { pixel_idx: 0, num_contribs: 0, first_grad_idx: 0, first_gaussian_idx: 0 }; num_pixels];
        let debug_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Debug Info",
            &zero_debug,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
                            resource: debug_buffer.as_entire_binding(),
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

        if enable_timing {
            eprintln!("[GPU] Backward pass: {:?}", t_backward.unwrap().elapsed());
        }

        // Download per-pixel gradients
        let t_download = if enable_timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let pixel_grads: Vec<GradientGPU> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &pixel_grads_buffer,
            num_pixels * num_gaussians,
        )
        .expect("Failed to read pixel gradients");

        if enable_timing {
            eprintln!(
                "[GPU] Download gradients: {:?}",
                t_download.unwrap().elapsed()
            );
        }

        // Debug: Read debug buffer to see what the shader computed
        if std::env::var("SUGAR_GPU_DEBUG").is_ok() {
            let debug_data: Vec<DebugInfo> = buffers::read_buffer_blocking(
                &self.ctx.device,
                &self.ctx.queue,
                &debug_buffer,
                num_pixels,
            ).expect("Failed to read debug buffer");

            eprintln!("[GPU DEBUG] Shader debug info:");
            eprintln!("  Checking pixels 1103-1110 (should have intermediates):");
            for px in 1103..=1110 {
                let info = &debug_data[px];
                let expected_grad_idx = px * num_gaussians + info.first_gaussian_idx as usize;
                eprintln!("    Pixel {}: num_contribs={}, writes_to_grad_idx={} (expected={}), gaussian_idx={}",
                    px, info.num_contribs, info.first_grad_idx, expected_grad_idx, info.first_gaussian_idx);
            }

            eprintln!("  Checking pixels 828-830 (should NOT have intermediates):");
            for px in 828..=830 {
                let info = &debug_data[px];
                eprintln!("    Pixel {}: num_contribs={}, first_grad_idx={}, first_gaussian_idx={}",
                    px, info.num_contribs, info.first_grad_idx, info.first_gaussian_idx);
            }

            // Find pixels where shader found contributions
            let mut pixels_with_shader_contribs = Vec::new();
            for (px_idx, info) in debug_data.iter().enumerate() {
                if info.num_contribs > 0 {
                    pixels_with_shader_contribs.push(px_idx);
                }
            }
            eprintln!("  Total pixels where shader found contributions: {}", pixels_with_shader_contribs.len());
            if pixels_with_shader_contribs.len() <= 10 {
                eprintln!("  First pixels where shader found contribs: {:?}", pixels_with_shader_contribs);
            } else {
                eprintln!("  First 10 pixels where shader found contribs: {:?}", &pixels_with_shader_contribs[0..10]);
            }
        }

        // Debug: Check how many pixels have non-zero gradients
        if std::env::var("SUGAR_GPU_DEBUG").is_ok() {
            let mut pixels_with_grads = 0;
            let mut total_nonzero_grads = 0;

            for px_idx in 0..num_pixels {
                let px_base = px_idx * num_gaussians;
                let mut has_grad = false;

                for g_idx in 0..num_gaussians {
                    let grad = &pixel_grads[px_base + g_idx];
                    let has_nonzero = grad.d_color[0].abs() > 1e-10
                        || grad.d_color[1].abs() > 1e-10
                        || grad.d_color[2].abs() > 1e-10;

                    if has_nonzero {
                        has_grad = true;
                        total_nonzero_grads += 1;
                    }
                }

                if has_grad {
                    pixels_with_grads += 1;
                }
            }

            eprintln!("[GPU DEBUG] Pixels with gradients: {} / {}", pixels_with_grads, num_pixels);
            eprintln!("[GPU DEBUG] Total non-zero gradient entries: {} / {}",
                total_nonzero_grads, num_pixels * num_gaussians);
            eprintln!("[GPU DEBUG] Expected if all pixels contribute: {} entries",
                num_pixels * num_gaussians);

            // Re-read intermediates to compare with gradients
            let intermediates_data = buffers::read_buffer_blocking::<ContributionGPU>(
                &self.ctx.device,
                &self.ctx.queue,
                &intermediates_buffer,
                num_pixels * MAX_CONTRIBUTIONS_PER_PIXEL as usize,
            ).expect("Failed to re-read intermediates");

            // Check pixels 1103-1110 (which have intermediates) to see if they have gradients
            eprintln!("[GPU DEBUG] Checking pixels 1103-1110 (which HAVE intermediates):");
            for px_idx in 1103..=1110 {
                let px_base = px_idx * num_gaussians;
                let int_base = px_idx * MAX_CONTRIBUTIONS_PER_PIXEL as usize;
                let has_intermediate = intermediates_data[int_base].gaussian_idx != 0xFFFFFFFF;
                let mut has_grad = false;
                eprintln!("  Pixel {}:", px_idx);
                for g_idx in 0..num_gaussians {
                    let grad_idx = px_base + g_idx;
                    let grad = &pixel_grads[grad_idx];
                    let norm = (grad.d_color[0].powi(2) + grad.d_color[1].powi(2) + grad.d_color[2].powi(2)).sqrt();
                    if norm > 1e-10 {
                        has_grad = true;
                    }
                    eprintln!("    grad[{}] (g={}): d_color=[{:.12}, {:.12}, {:.12}], norm={:.12}",
                        grad_idx, g_idx, grad.d_color[0], grad.d_color[1], grad.d_color[2], norm);
                }
                eprintln!("    has_intermediate={}, has_gradient={}", has_intermediate, has_grad);
            }

            // Also check pixels 828-830 (which DON'T have intermediates but have gradients)
            eprintln!("[GPU DEBUG] Checking pixels 828-830 (which DON'T have intermediates but have gradients):");
            for px_idx in 828..=830 {
                let px_base = px_idx * num_gaussians;
                let int_base = px_idx * MAX_CONTRIBUTIONS_PER_PIXEL as usize;
                let has_intermediate = intermediates_data[int_base].gaussian_idx != 0xFFFFFFFF;
                eprintln!("  Pixel {}: has_intermediate={}", px_idx, has_intermediate);
                for g_idx in 0..num_gaussians {
                    let grad = &pixel_grads[px_base + g_idx];
                    if grad.d_color[0].abs() > 1e-10 || grad.d_color[1].abs() > 1e-10 || grad.d_color[2].abs() > 1e-10 {
                        eprintln!("    Gaussian {}: d_color=[{:.9}, {:.9}, {:.9}]",
                            g_idx, grad.d_color[0], grad.d_color[1], grad.d_color[2]);
                    }
                }
            }
        }

        // CPU reduction - sum across all pixels
        let t_reduce = if enable_timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let final_grads = reduce_pixel_gradients(&pixel_grads, num_pixels, num_gaussians);

        if enable_timing {
            eprintln!("[GPU] CPU reduction: {:?}", t_reduce.unwrap().elapsed());
        }

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

        if enable_timing {
            eprintln!(
                "[GPU] Total render_with_gradients time: {:?}",
                t_start.unwrap().elapsed()
            );
        }

        (pixels, final_grads)
    }
}
