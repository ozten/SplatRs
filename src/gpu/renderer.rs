//! High-level GPU renderer interface.

use crate::core::{Camera, Gaussian};
use crate::gpu::{buffers, context::GpuContext, shaders, types::*};
use nalgebra::{Vector2, Vector3};
use wgpu::{BindGroup, BindGroupLayout, BufferUsages, ComputePipeline};

/// Options for [`GpuRenderer::render_with_options`] (docs/TILE_RASTER_PLAN.md Part B).
/// Defaults select the naive oracle rasterizer with full SH — identical to `render()`.
#[derive(Copy, Clone, Debug, Default)]
pub struct RenderOptions {
    /// Use only the SH DC term for color (view-independent).
    pub disable_sh: bool,
    /// Use the tile-binned rasterizer instead of the naive per-pixel-loop oracle.
    pub tile_rasterizer: bool,
}

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
    // Tile-binned path (docs/TILE_RASTER_PLAN.md Part B Stage 5a): cached here (built once
    // in `new()`) instead of lazily per-call like the Stage 0-2 debug paths, so the
    // production render_tiled()/debug_render_tiled_pixel_state() paths don't pay pipeline
    // creation cost on every frame. Bind group layouts are fetched per-call via
    // `pipeline.get_bind_group_layout(0)` (auto `layout: None` pipelines).
    pair_sorter: crate::gpu::sort::PairSorter,
    tile_bin_count_pipeline: ComputePipeline,
    tile_bin_emit_pipeline: ComputePipeline,
    tile_bin_ranges_pipeline: ComputePipeline,
    rasterize_tiled_pipeline: ComputePipeline,
    // Tiled backward pass (docs/TILE_RASTER_PLAN.md Part B Stage 5b): cached like the
    // Stage 5a forward pipelines above (built once in `new()`); bind group layout fetched
    // per-call via `backward_tiled_pipeline.get_bind_group_layout(0)`.
    backward_tiled_pipeline: ComputePipeline,
}

/// Fixed-point gradient buffer layout: 16 i32s (64 bytes) per Gaussian. Shared by the
/// naive (backward.wgsl) and tile-binned (backward_tiled.wgsl) backward passes — both
/// atomicAdd into the SAME layout, so both Rust-side readback paths use this constant.
const GRADIENT_I32_PER_GAUSSIAN: usize = 16;

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

/// Convert the fixed-point i32 `gradient_atomic` buffer into f32 [`GaussianGradients2D`].
/// Shared by the naive (`render_with_gradients_naive`, backward.wgsl) and tiled
/// (`render_tiled_with_gradients`, backward_tiled.wgsl) backward passes — both accumulate
/// into the SAME fixed-point layout (`GRADIENT_I32_PER_GAUSSIAN` i32s per Gaussian; see
/// backward.wgsl's `GRADIENT_STRIDE` comment for the exact offsets).
fn convert_fixed_point_gradients(
    pixel_grads_i32: &[i32],
    num_gaussians: usize,
) -> crate::gpu::gradients::GaussianGradients2D {
    // Color/opacity use scale 10^7, position/covariance use scale 10^9 (see backward.wgsl).
    const FIXED_POINT_SCALE_INV: f32 = 1e-7;
    const FIXED_POINT_SCALE_POSITION_INV: f32 = 1e-9;

    let mut final_grads = crate::gpu::gradients::GaussianGradients2D::zeros(num_gaussians);
    for i in 0..num_gaussians {
        let base = i * GRADIENT_I32_PER_GAUSSIAN;
        // d_color: offsets 0-2 (3 is padding)
        final_grads.d_colors[i] = Vector3::new(
            pixel_grads_i32[base] as f32 * FIXED_POINT_SCALE_INV,
            pixel_grads_i32[base + 1] as f32 * FIXED_POINT_SCALE_INV,
            pixel_grads_i32[base + 2] as f32 * FIXED_POINT_SCALE_INV,
        );
        // d_opacity_logit_pad: offset 4 (5-7 are padding)
        final_grads.d_opacity_logits[i] = pixel_grads_i32[base + 4] as f32 * FIXED_POINT_SCALE_INV;
        // d_mean_px: offsets 8-9 (10-11 are padding) — higher precision scale (10^9)
        final_grads.d_mean_px[i] = Vector2::new(
            pixel_grads_i32[base + 8] as f32 * FIXED_POINT_SCALE_POSITION_INV,
            pixel_grads_i32[base + 9] as f32 * FIXED_POINT_SCALE_POSITION_INV,
        );
        // d_cov_2d: offsets 12-14 (15 is padding) — higher precision scale (10^9)
        final_grads.d_cov_2d[i] = Vector3::new(
            pixel_grads_i32[base + 12] as f32 * FIXED_POINT_SCALE_POSITION_INV,
            pixel_grads_i32[base + 13] as f32 * FIXED_POINT_SCALE_POSITION_INV,
            pixel_grads_i32[base + 14] as f32 * FIXED_POINT_SCALE_POSITION_INV,
        );
    }
    final_grads
}

/// Sum per-pixel background gradient contributions on the CPU. Shared by the naive and
/// tiled backward paths.
fn sum_d_background(d_background_pixels: &[[f32; 4]]) -> Vector3<f32> {
    let mut sum = Vector3::zeros();
    for px in d_background_pixels {
        sum += Vector3::new(px[0], px[1], px[2]);
    }
    sum
}

/// GPU-resident tile-binning result (docs/TILE_RASTER_PLAN.md Part B Stage 5a):
/// `projected`/`pairs`/`ranges` stay on the GPU end-to-end — the only readback in
/// [`GpuRenderer::tile_binning_gpu`] is the per-gaussian touch-count array needed for the
/// CPU exclusive prefix sum. This kills the v1 `tile_binning_impl` path's 32-68 MB
/// round-trips (projected re-upload + sorted-pairs/ranges readback) for every render.
struct TileBinningGpu {
    projected_buffer: wgpu::Buffer,
    pairs_buffer: wgpu::Buffer,
    ranges_buffer: wgpu::Buffer,
    total_pairs: u32,
    tiles_x: u32,
    tiles_y: u32,
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

        // Create compute pipelines with explicit validation error capture. `layout: None`
        // (auto layout) is used for the tile-binned pipelines below — their bind group
        // layouts are fetched per-call via `pipeline.get_bind_group_layout(0)`, matching
        // the Stage 0-2 debug paths' convention.
        let mut create_pipeline = |label: &str,
                                   layout: Option<&wgpu::PipelineLayout>,
                                   module: &wgpu::ShaderModule,
                                   entry: &str|
         -> Result<wgpu::ComputePipeline, String> {
            ctx.device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout,
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
            Some(&project_pipeline_layout),
            &project_shader,
            "project_gaussians",
        )?;

        let rasterize_pipeline = create_pipeline(
            "Rasterize Pipeline",
            Some(&rasterize_pipeline_layout),
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
            Some(&backward_pipeline_layout),
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
            Some(&project_backward_pipeline_layout),
            &project_backward_shader,
            "project_backward",
        )?;

        // Create bitonic sorter for GPU-side sorting
        let sorter = crate::gpu::sort::BitonicSorter::new(&ctx.device);

        // Tile-binned path (docs/TILE_RASTER_PLAN.md Part B Stage 5a): cache the pipelines
        // and the pair sorter up front instead of building them lazily per call (the
        // Stage 0-2 debug_tile_* paths still do that — left untouched, they back the
        // Stage 0-2 gates). One shared tile_bin_shader module backs three entry points;
        // wgpu's auto (`layout: None`) reflection derives a distinct bind group layout per
        // entry point from only the bindings that entry point actually references.
        let tile_bin_shader = shaders::create_tile_bin_shader(&ctx.device);
        let rasterize_tiled_shader = shaders::create_rasterize_tiled_shader(&ctx.device);

        let tile_bin_count_pipeline = create_pipeline(
            "Tile Count Pipeline",
            None,
            &tile_bin_shader,
            "count_tile_touches",
        )?;
        let tile_bin_emit_pipeline = create_pipeline(
            "Tile Pair Emit Pipeline",
            None,
            &tile_bin_shader,
            "emit_tile_pairs",
        )?;
        let tile_bin_ranges_pipeline = create_pipeline(
            "Tile Ranges Pipeline",
            None,
            &tile_bin_shader,
            "identify_tile_ranges",
        )?;
        let rasterize_tiled_pipeline = create_pipeline(
            "Rasterize Tiled Pipeline",
            None,
            &rasterize_tiled_shader,
            "rasterize_tiled",
        )?;
        let pair_sorter = crate::gpu::sort::PairSorter::new(&ctx.device);

        // Tiled backward pass (docs/TILE_RASTER_PLAN.md Part B Stage 5b): 1 uniform + 7
        // storage buffers, one bind group — verified against this device's
        // max_storage_buffers_per_shader_stage (8) with a standalone smoke before this
        // kernel was written; passed on Apple M2 Max (Metal), no fallback needed.
        let backward_tiled_shader = shaders::create_backward_tiled_shader(&ctx.device);
        let backward_tiled_pipeline = create_pipeline(
            "Backward Tiled Pipeline",
            None,
            &backward_tiled_shader,
            "backward_pass_tiled",
        )?;

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
            pair_sorter,
            tile_bin_count_pipeline,
            tile_bin_emit_pipeline,
            tile_bin_ranges_pipeline,
            rasterize_tiled_pipeline,
            backward_tiled_pipeline,
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
        self.tile_touch_counts_impl(gaussians, camera, false)
    }

    fn tile_touch_counts_impl(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        disable_sh: bool,
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
            &[if disable_sh {
                SettingsGPU::dc_only()
            } else {
                SettingsGPU::full_sh()
            }],
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

    /// Tile-binning Stage-2 validation surface (docs/TILE_RASTER_PLAN.md): full binning
    /// pipeline — project → count → CPU exclusive prefix sum → pair emission →
    /// (tile, depth) bitonic sort → tile-range boundary detection — returning
    /// `(projected, counts, sorted_pairs, tile_ranges)` for property tests.
    /// Debug/test-only; pipelines built lazily per call.
    pub fn debug_tile_binning(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
    ) -> Result<
        (
            Vec<Gaussian2DGPU>,
            Vec<u32>,
            Vec<TileGaussianPair>,
            Vec<[u32; 2]>,
        ),
        String,
    > {
        self.tile_binning_impl(gaussians, camera, false)
    }

    fn tile_binning_impl(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        disable_sh: bool,
    ) -> Result<
        (
            Vec<Gaussian2DGPU>,
            Vec<u32>,
            Vec<TileGaussianPair>,
            Vec<[u32; 2]>,
        ),
        String,
    > {
        let (projected, counts) = self.tile_touch_counts_impl(gaussians, camera, disable_sh)?;
        let num_gaussians = gaussians.len();
        let (tiles_x, tiles_y) =
            crate::render::tile_math::tile_grid_dims(camera.width, camera.height);
        let num_tiles = (tiles_x * tiles_y) as usize;

        // CPU exclusive prefix sum (v1 simplification per the plan: 4·N bytes readback).
        let mut offsets = vec![0u32; num_gaussians];
        let mut total_pairs: u32 = 0;
        for (i, &c) in counts.iter().enumerate() {
            offsets[i] = total_pairs;
            total_pairs += c;
        }
        if total_pairs == 0 {
            return Ok((projected, counts, Vec::new(), vec![[0u32; 2]; num_tiles]));
        }
        let padded = total_pairs.next_power_of_two();

        // Re-upload projected values so the emit kernel reads the same f32 bits the
        // counting kernel consumed (avoids a second projection dispatch drifting state).
        let projected_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBin Projected (re-upload)",
            &projected,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let offsets_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBin Offsets Buffer",
            &offsets,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        // Pad region pre-filled with key_tile = num_tiles sentinels (sort last).
        let sentinel = TileGaussianPair {
            key_tile: num_tiles as u32,
            key_depth: u32::MAX,
            gaussian_idx: u32::MAX,
            pad: 0,
        };
        let pairs_init = vec![sentinel; padded as usize];
        let pairs_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBin Pairs Buffer",
            &pairs_init,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        let ranges_buffer = buffers::create_buffer_zeroed::<[u32; 2]>(
            &self.ctx.device,
            "TileBin Ranges Buffer",
            num_tiles,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TileBinParams {
            tiles_x: u32,
            tiles_y: u32,
            num_gaussians: u32,
            total_pairs: u32,
        }
        let params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBin Params Buffer (stage 2)",
            &[TileBinParams {
                tiles_x,
                tiles_y,
                num_gaussians: num_gaussians as u32,
                total_pairs,
            }],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        self.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let tile_bin_shader = shaders::create_tile_bin_shader(&self.ctx.device);
        let emit_pipeline =
            self.ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Tile Pair Emit Pipeline"),
                    layout: None,
                    module: &tile_bin_shader,
                    entry_point: "emit_tile_pairs",
                });
        let ranges_pipeline =
            self.ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Tile Ranges Pipeline"),
                    layout: None,
                    module: &tile_bin_shader,
                    entry_point: "identify_tile_ranges",
                });
        if let Some(err) = pollster::block_on(self.ctx.device.pop_error_scope()) {
            return Err(format!("Tile bin stage-2 pipeline creation failed: {}", err));
        }

        let emit_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Pair Emit Bind Group"),
            layout: &emit_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pairs_buffer.as_entire_binding() },
            ],
        });
        let ranges_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Ranges Bind Group"),
            layout: &ranges_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pairs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ranges_buffer.as_entire_binding() },
            ],
        });

        let sorter = crate::gpu::sort::PairSorter::new(&self.ctx.device);

        // Three separate submissions, matching the render() path's phase-per-submission
        // convention (queue.submit boundaries are hard sync points). Note: the lost-write
        // bug originally observed here was NOT an encoder-ordering hazard — it was the
        // vec2<u32> component-store race documented in tile_bin.wgsl (two threads writing
        // .x/.y of one vector); the split submissions are kept for convention/defense.
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TileBin Emit Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tile Pair Emit Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&emit_pipeline);
            pass.set_bind_group(0, &emit_bind_group, &[]);
            pass.dispatch_workgroups((num_gaussians as u32 + 255) / 256, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TileBin Sort Encoder"),
            });
        sorter.sort(&self.ctx.device, &mut encoder, &pairs_buffer, total_pairs);
        self.ctx.queue.submit(Some(encoder.finish()));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TileBin Ranges Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tile Ranges Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ranges_pipeline);
            pass.set_bind_group(0, &ranges_bind_group, &[]);
            pass.dispatch_workgroups((total_pairs + 255) / 256, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        let sorted_pairs: Vec<TileGaussianPair> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &pairs_buffer,
            total_pairs as usize,
        )
        .map_err(|e| format!("Failed to read sorted pairs: {e}"))?;
        let tile_ranges: Vec<[u32; 2]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &ranges_buffer,
            num_tiles,
        )
        .map_err(|e| format!("Failed to read tile ranges: {e}"))?;

        Ok((projected, counts, sorted_pairs, tile_ranges))
    }

    /// Full GPU-resident tile-binning pipeline (docs/TILE_RASTER_PLAN.md Part B Stage 5a):
    /// project → count → read back ONLY counts (4·N bytes) → CPU exclusive prefix sum →
    /// upload offsets → emit (reusing the SAME projected buffer the counting kernel wrote,
    /// no reupload) → `self.pair_sorter.sort` → identify ranges. Never reads
    /// projected/pairs/ranges back — callers that need pairs (e.g. the Stage 5a pixel-state
    /// debug accessor) read them back themselves afterward.
    ///
    /// `total_pairs == 0` (no gaussian touches any tile) returns an empty-but-valid
    /// 1-element pairs buffer and a zeroed ranges buffer: every tile's range stays (0, 0),
    /// so the raster kernel blends nothing for any pixel, matching an all-background frame.
    fn tile_binning_gpu(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        disable_sh: bool,
    ) -> Result<TileBinningGpu, String> {
        let num_gaussians = gaussians.len();
        let (tiles_x, tiles_y) =
            crate::render::tile_math::tile_grid_dims(camera.width, camera.height);
        let num_tiles = (tiles_x * tiles_y) as usize;

        let gaussians_gpu: Vec<GaussianGPU> =
            gaussians.iter().map(GaussianGPU::from_gaussian).collect();
        let camera_gpu = CameraGPU::from_camera(camera);

        let camera_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBinGpu Camera",
            &[camera_gpu],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let gaussians_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBinGpu Gaussians",
            &gaussians_gpu,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let projected_bytes = ((num_gaussians as u32).next_power_of_two() as usize)
            * std::mem::size_of::<Gaussian2DGPU>();
        // No COPY_SRC: unlike the Stage 0-2 debug paths, tile_binning_gpu never reads the
        // projected buffer back — it stays resident on the GPU for the emit kernel and the
        // raster dispatch to consume directly.
        let projected_buffer = buffers::create_buffer(
            &self.ctx.device,
            "TileBinGpu Projected",
            projected_bytes as u64,
            BufferUsages::STORAGE,
        );
        let settings_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBinGpu Settings",
            &[if disable_sh {
                SettingsGPU::dc_only()
            } else {
                SettingsGPU::full_sh()
            }],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let project_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TileBinGpu Project Bind Group"),
            layout: &self.project_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gaussians_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: settings_buffer.as_entire_binding() },
            ],
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TileBinParams {
            tiles_x: u32,
            tiles_y: u32,
            num_gaussians: u32,
            total_pairs: u32,
        }
        let count_params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBinGpu Count Params",
            &[TileBinParams { tiles_x, tiles_y, num_gaussians: num_gaussians as u32, total_pairs: 0 }],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let counts_buffer = buffers::create_buffer(
            &self.ctx.device,
            "TileBinGpu Counts",
            (num_gaussians * std::mem::size_of::<u32>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let count_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TileBinGpu Count Bind Group"),
            layout: &self.tile_bin_count_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: count_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: counts_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TileBinGpu Project+Count Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TileBinGpu Project Pass"),
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
                label: Some("TileBinGpu Count Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_bin_count_pipeline);
            pass.set_bind_group(0, &count_bind_group, &[]);
            pass.dispatch_workgroups((num_gaussians as u32 + 255) / 256, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        // ONLY readback in this function: per-gaussian touch counts (4·N bytes ≤ 1.6 MB at
        // 400k gaussians), needed for the CPU exclusive prefix sum (a GPU scan is a later
        // optimization per the plan).
        let counts: Vec<u32> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &counts_buffer,
            num_gaussians,
        )
        .map_err(|e| format!("Failed to read tile touch counts: {e}"))?;

        let mut offsets = vec![0u32; num_gaussians];
        let mut total_pairs: u32 = 0;
        for (i, &c) in counts.iter().enumerate() {
            offsets[i] = total_pairs;
            total_pairs += c;
        }

        if total_pairs == 0 {
            let dummy_pair = TileGaussianPair {
                key_tile: u32::MAX,
                key_depth: u32::MAX,
                gaussian_idx: u32::MAX,
                pad: 0,
            };
            let pairs_buffer = buffers::create_buffer_init(
                &self.ctx.device,
                "TileBinGpu Pairs (empty)",
                &[dummy_pair],
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            );
            let ranges_buffer = buffers::create_buffer_zeroed::<[u32; 2]>(
                &self.ctx.device,
                "TileBinGpu Ranges (empty)",
                num_tiles,
                BufferUsages::STORAGE,
            );
            return Ok(TileBinningGpu {
                projected_buffer,
                pairs_buffer,
                ranges_buffer,
                total_pairs: 0,
                tiles_x,
                tiles_y,
            });
        }

        let padded = total_pairs.next_power_of_two();
        let offsets_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBinGpu Offsets",
            &offsets,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        // Pad region pre-filled with key_tile = num_tiles sentinels (sort last) — an
        // upload, not a readback, so it doesn't violate the "never read pairs back" rule.
        let sentinel = TileGaussianPair {
            key_tile: num_tiles as u32,
            key_depth: u32::MAX,
            gaussian_idx: u32::MAX,
            pad: 0,
        };
        let pairs_init = vec![sentinel; padded as usize];
        // COPY_SRC: debug/test callers (e.g. debug_render_tiled_pixel_state) read the
        // sorted pairs back for validation; tile_binning_gpu itself never does.
        let pairs_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBinGpu Pairs",
            &pairs_init,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        let ranges_buffer = buffers::create_buffer_zeroed::<[u32; 2]>(
            &self.ctx.device,
            "TileBinGpu Ranges",
            num_tiles,
            BufferUsages::STORAGE,
        );

        let emit_params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileBinGpu Emit Params",
            &[TileBinParams { tiles_x, tiles_y, num_gaussians: num_gaussians as u32, total_pairs }],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let emit_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TileBinGpu Emit Bind Group"),
            layout: &self.tile_bin_emit_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: emit_params_buffer.as_entire_binding() },
                // REUSED from the count pass — no readback/reupload round-trip.
                wgpu::BindGroupEntry { binding: 1, resource: projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pairs_buffer.as_entire_binding() },
            ],
        });
        let ranges_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TileBinGpu Ranges Bind Group"),
            layout: &self.tile_bin_ranges_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: emit_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pairs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ranges_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TileBinGpu Emit Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TileBinGpu Emit Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_bin_emit_pipeline);
            pass.set_bind_group(0, &emit_bind_group, &[]);
            pass.dispatch_workgroups((num_gaussians as u32 + 255) / 256, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TileBinGpu Sort Encoder"),
            });
        self.pair_sorter.sort(&self.ctx.device, &mut encoder, &pairs_buffer, total_pairs);
        self.ctx.queue.submit(Some(encoder.finish()));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TileBinGpu Ranges Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TileBinGpu Ranges Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_bin_ranges_pipeline);
            pass.set_bind_group(0, &ranges_bind_group, &[]);
            pass.dispatch_workgroups((total_pairs + 255) / 256, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        Ok(TileBinningGpu { projected_buffer, pairs_buffer, ranges_buffer, total_pairs, tiles_x, tiles_y })
    }

    /// Render with the tile-binned rasterizer (docs/TILE_RASTER_PLAN.md Part B Stage 5a).
    /// Uses the GPU-resident `tile_binning_gpu` (projected/pairs/ranges never leave the
    /// GPU) and the cached `rasterize_tiled_pipeline`. The naive `render()` remains the
    /// oracle; parity enforced by unit_gpu_tile_raster_parity.
    fn render_tiled(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
        disable_sh: bool,
    ) -> Result<Vec<Vector3<f32>>, String> {
        let binning = self.tile_binning_gpu(gaussians, camera, disable_sh)?;
        let width = camera.width as usize;
        let height = camera.height as usize;
        let pixel_count = width * height;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TileRasterParams {
            width: u32,
            height: u32,
            tiles_x: u32,
            tiles_y: u32,
            save_intermediates: u32,
            pad0: u32,
            pad1: u32,
            pad2: u32,
            background: [f32; 4],
        }
        let params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TileRaster Params",
            &[TileRasterParams {
                width: camera.width,
                height: camera.height,
                tiles_x: binning.tiles_x,
                tiles_y: binning.tiles_y,
                save_intermediates: 0,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                background: [background.x, background.y, background.z, 0.0],
            }],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let output_buffer = buffers::create_buffer(
            &self.ctx.device,
            "TileRaster Output",
            (pixel_count * std::mem::size_of::<[f32; 4]>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        // Dummy 1-element pixel-state buffer: not written when save_intermediates=0 (same
        // pattern as render_with_sh_mode's "Pixel State Buffer (dummy)").
        let pixel_state_buffer = buffers::create_buffer_zeroed::<[u32; 2]>(
            &self.ctx.device,
            "TileRaster Pixel State (dummy)",
            1,
            BufferUsages::STORAGE,
        );

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rasterize Tiled Bind Group"),
            layout: &self.rasterize_tiled_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: binning.projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: binning.pairs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: binning.ranges_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pixel_state_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Rasterize Tiled Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Rasterize Tiled Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rasterize_tiled_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(binning.tiles_x, binning.tiles_y, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        let output: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &output_buffer,
            pixel_count,
        )
        .map_err(|e| format!("Failed to read tiled output: {e}"))?;
        Ok(output
            .into_iter()
            .map(|p| Vector3::new(p[0], p[1], p[2]))
            .collect())
    }

    /// Debug/test-only accessor (docs/TILE_RASTER_PLAN.md Part B Stage 5a gate): runs the
    /// tile-binned forward pass with `save_intermediates=1` and returns
    /// `(pixels, pixel_state, sorted_pairs)`. `pixel_state[i].1` (when not the
    /// `0xFFFFFFFF` sentinel) indexes into the returned `sorted_pairs`, whose
    /// `gaussian_idx` is the ORIGINAL Gaussian index (the tile path never reorders
    /// `projected` — pairs carry original indices, unlike the naive path's globally
    /// depth-sorted buffer).
    pub fn debug_render_tiled_pixel_state(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
    ) -> Result<(Vec<Vector3<f32>>, Vec<[u32; 2]>, Vec<TileGaussianPair>), String> {
        let binning = self.tile_binning_gpu(gaussians, camera, false)?;
        let width = camera.width as usize;
        let height = camera.height as usize;
        let pixel_count = width * height;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TileRasterParams {
            width: u32,
            height: u32,
            tiles_x: u32,
            tiles_y: u32,
            save_intermediates: u32,
            pad0: u32,
            pad1: u32,
            pad2: u32,
            background: [f32; 4],
        }
        let params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Debug TiledPS Params",
            &[TileRasterParams {
                width: camera.width,
                height: camera.height,
                tiles_x: binning.tiles_x,
                tiles_y: binning.tiles_y,
                save_intermediates: 1,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                background: [background.x, background.y, background.z, 0.0],
            }],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let output_buffer = buffers::create_buffer(
            &self.ctx.device,
            "Debug TiledPS Output",
            (pixel_count * std::mem::size_of::<[f32; 4]>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let pixel_state_buffer = buffers::create_buffer_zeroed::<[u32; 2]>(
            &self.ctx.device,
            "Debug TiledPS Pixel State",
            pixel_count,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Debug TiledPS Bind Group"),
            layout: &self.rasterize_tiled_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: binning.projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: binning.pairs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: binning.ranges_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pixel_state_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Debug TiledPS Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Debug TiledPS Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rasterize_tiled_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(binning.tiles_x, binning.tiles_y, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        let output: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &output_buffer,
            pixel_count,
        )
        .map_err(|e| format!("Failed to read tiled output: {e}"))?;
        let pixel_state: Vec<[u32; 2]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &pixel_state_buffer,
            pixel_count,
        )
        .map_err(|e| format!("Failed to read tiled pixel state: {e}"))?;
        // The empty-scene sentinel path leaves a 1-element dummy pairs buffer; read only
        // what's real (or the 1 dummy element so the read isn't zero-sized).
        let pairs_len = binning.total_pairs.max(1) as usize;
        let pairs: Vec<TileGaussianPair> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &binning.pairs_buffer,
            pairs_len,
        )
        .map_err(|e| format!("Failed to read tiled sorted pairs: {e}"))?;

        let pixels = output
            .into_iter()
            .map(|p| Vector3::new(p[0], p[1], p[2]))
            .collect();
        Ok((pixels, pixel_state, pairs))
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

    /// Render with explicit options (docs/TILE_RASTER_PLAN.md Part B): selects between
    /// the naive oracle rasterizer (default) and the flag-gated tile-binned path.
    pub fn render_with_options(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
        opts: RenderOptions,
    ) -> Result<Vec<Vector3<f32>>, String> {
        if opts.tile_rasterizer {
            self.render_tiled(gaussians, camera, background, opts.disable_sh)
        } else {
            self.render_with_sh_mode(gaussians, camera, background, opts.disable_sh)
        }
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

    /// Debug/test-only accessor (docs/TILE_RASTER_PLAN.md Part B Stage 5a gate): runs the
    /// naive oracle's forward pass (full SH, `save_intermediates=1`) and returns
    /// `(pixels, pixel_state, sorted_projected)` — the same forward computation
    /// [`GpuRenderer::render_with_gradients`] runs internally, minus the backward pass,
    /// with the sorted `Gaussian2DGPU` buffer exposed so a test can resolve
    /// `pixel_state[i].1` (a SORTED-buffer index, sentinel `0xFFFFFFFF`) to an ORIGINAL
    /// Gaussian index via `sorted_projected[y].gaussian_idx_pad[0]`.
    pub fn debug_render_naive_pixel_state(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
    ) -> Result<(Vec<Vector3<f32>>, Vec<[u32; 2]>, Vec<Gaussian2DGPU>), String> {
        let num_gaussians = gaussians.len();
        let width = camera.width;
        let height = camera.height;
        let num_pixels = (width * height) as usize;
        if num_gaussians == 0 {
            // No contributors: transmittance stays 1.0 (fully transparent -> all
            // background), sentinel index (matches what the shader would have written).
            return Ok((
                vec![*background; num_pixels],
                vec![[1.0f32.to_bits(), 0xFFFFFFFFu32]; num_pixels],
                Vec::new(),
            ));
        }

        let gaussians_gpu: Vec<GaussianGPU> =
            gaussians.iter().map(GaussianGPU::from_gaussian).collect();
        let camera_gpu = CameraGPU::from_camera(camera);

        let camera_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Debug NaivePS Camera",
            &[camera_gpu],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let gaussians_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Debug NaivePS Gaussians",
            &gaussians_gpu,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let projected_bytes = ((num_gaussians as u32).next_power_of_two() as usize)
            * std::mem::size_of::<Gaussian2DGPU>();
        let projected_buffer = buffers::create_buffer(
            &self.ctx.device,
            "Debug NaivePS Projected",
            projected_bytes as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let settings_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "Debug NaivePS Settings",
            &[SettingsGPU::full_sh()],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let project_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Debug NaivePS Project Bind Group"),
            layout: &self.project_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gaussians_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: settings_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Debug NaivePS Project Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Debug NaivePS Project Pass"),
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
        self.ctx.queue.submit(Some(encoder.finish()));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Debug NaivePS Sort Encoder"),
            });
        self.sorter.sort(&self.ctx.device, &mut encoder, &projected_buffer, num_gaussians as u32);
        self.ctx.queue.submit(Some(encoder.finish()));

        // Sorted buffer is the same as projected buffer (in-place sort).
        let sorted_buffer = projected_buffer;

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
            save_intermediates: 1,
            row_offset: 0,
            pad: [0; 3],
            background: [background.x, background.y, background.z, 0.0],
        };

        let output_buffer = buffers::create_buffer_zeroed::<[f32; 4]>(
            &self.ctx.device,
            "Debug NaivePS Output",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let pixel_state_buffer = buffers::create_buffer_zeroed::<[u32; 2]>(
            &self.ctx.device,
            "Debug NaivePS Pixel State",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let rows_per_band = watchdog_rows_per_band(width, height, num_gaussians);
        let mut row0 = 0u32;
        while row0 < height {
            let band_rows = rows_per_band.min(height - row0);
            let band_params = RenderParams { row_offset: row0, ..params };
            let params_buffer = buffers::create_buffer_init(
                &self.ctx.device,
                "Debug NaivePS Params (band)",
                &[band_params],
                BufferUsages::UNIFORM,
            );
            let rasterize_bind_group =
                self.ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Debug NaivePS Rasterize Bind Group"),
                        layout: &self.rasterize_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: sorted_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: pixel_state_buffer.as_entire_binding() },
                        ],
                    });

            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Debug NaivePS Rasterize Encoder"),
                });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Debug NaivePS Rasterize Pass"),
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

        let output: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &output_buffer,
            num_pixels,
        )
        .map_err(|e| format!("Failed to read output buffer: {e}"))?;
        let pixel_state: Vec<[u32; 2]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &pixel_state_buffer,
            num_pixels,
        )
        .map_err(|e| format!("Failed to read pixel state buffer: {e}"))?;
        let sorted: Vec<Gaussian2DGPU> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &sorted_buffer,
            num_gaussians,
        )
        .map_err(|e| format!("Failed to read sorted projected buffer: {e}"))?;

        let pixels = output
            .into_iter()
            .map(|p| Vector3::new(p[0], p[1], p[2]))
            .collect();
        Ok((pixels, pixel_state, sorted))
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
    ///
    /// Always uses the naive per-pixel-loop oracle backward (full SH — this path has never
    /// plumbed `disable_sh`). See [`GpuRenderer::render_with_gradients_and_options`] to
    /// select the tile-binned backward (docs/TILE_RASTER_PLAN.md Part B Stage 5b).
    pub fn render_with_gradients(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
        d_pixels: &[Vector3<f32>],
    ) -> Result<(Vec<Vector3<f32>>, crate::gpu::gradients::GaussianGradients2D), String> {
        self.render_with_gradients_and_options(
            gaussians,
            camera,
            background,
            d_pixels,
            RenderOptions::default(),
        )
    }

    /// Render with gradients using explicit options (docs/TILE_RASTER_PLAN.md Part B Stage
    /// 5b): selects between the naive oracle backward (default) and the flag-gated
    /// tile-binned backward. `opts.disable_sh` is honored only by the tiled path — the
    /// naive backward has never plumbed it (pre-existing behavior, unchanged here).
    pub fn render_with_gradients_and_options(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
        d_pixels: &[Vector3<f32>],
        opts: RenderOptions,
    ) -> Result<(Vec<Vector3<f32>>, crate::gpu::gradients::GaussianGradients2D), String> {
        if opts.tile_rasterizer {
            self.render_tiled_with_gradients(
                gaussians,
                camera,
                background,
                d_pixels,
                opts.disable_sh,
            )
        } else {
            self.render_with_gradients_naive(gaussians, camera, background, d_pixels)
        }
    }

    /// Naive oracle backward (the original `render_with_gradients` body, renamed per
    /// docs/TILE_RASTER_PLAN.md Part B Stage 5b so `render_with_gradients_and_options` can
    /// dispatch to it). Unchanged behavior.
    fn render_with_gradients_naive(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
        d_pixels: &[Vector3<f32>],
    ) -> Result<(Vec<Vector3<f32>>, crate::gpu::gradients::GaussianGradients2D), String> {
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

        // Read gradient buffer as i32 (fixed-point) and convert to f32 (shared with the
        // tiled backward path — both accumulate into the same layout).
        let pixel_grads_i32: Vec<i32> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &pixel_grads_buffer,
            total_i32s,
        )
        .map_err(|e| format!("Failed to read per-Gaussian gradients: {e}"))?;
        let mut final_grads = convert_fixed_point_gradients(&pixel_grads_i32, num_gaussians);

        // Download per-pixel background gradients and sum on CPU (shared helper).
        let d_background_pixels: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &d_background_pixels_buffer,
            num_pixels,
        )
        .map_err(|e| format!("Failed to read per-pixel background gradients: {e}"))?;
        final_grads.d_background = sum_d_background(&d_background_pixels);

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

    /// Render with gradients using the tile-binned rasterizer/backward pass
    /// (docs/TILE_RASTER_PLAN.md Part B Stage 5b). Calls `tile_binning_gpu` ONCE, runs the
    /// tiled forward with `save_intermediates=1` (same cached `rasterize_tiled_pipeline`
    /// as `render_tiled`), then the tiled backward (`backward_tiled_pipeline`) as its own
    /// queue submission (phase-per-submission convention used throughout this file), then
    /// reads back `gradient_atomic` + `d_background_pixels` and converts fixed-point→f32
    /// with the SAME helpers the naive path uses. `disable_sh` is forwarded to
    /// `tile_binning_gpu`'s projection (the tiled path, unlike the naive backward, does
    /// support it).
    fn render_tiled_with_gradients(
        &self,
        gaussians: &[Gaussian],
        camera: &Camera,
        background: &Vector3<f32>,
        d_pixels: &[Vector3<f32>],
        disable_sh: bool,
    ) -> Result<(Vec<Vector3<f32>>, crate::gpu::gradients::GaussianGradients2D), String> {
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
        let output_bytes = (num_pixels * std::mem::size_of::<[f32; 4]>()) as u64;
        let pixel_state_bytes = (num_pixels * std::mem::size_of::<[u32; 2]>()) as u64;
        let d_pixels_bytes = (num_pixels * std::mem::size_of::<[f32; 4]>()) as u64;
        let gradient_atomic_bytes =
            (num_gaussians * GRADIENT_I32_PER_GAUSSIAN * std::mem::size_of::<i32>()) as u64;
        let d_background_pixels_bytes = (num_pixels * std::mem::size_of::<[f32; 4]>()) as u64;
        for (label, bytes) in [
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

        // ONE tile_binning_gpu call feeds both the forward and backward dispatches below —
        // projected/pairs/tile_ranges stay GPU-resident throughout.
        let binning = self.tile_binning_gpu(gaussians, camera, disable_sh)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TileRasterParams {
            width: u32,
            height: u32,
            tiles_x: u32,
            tiles_y: u32,
            save_intermediates: u32,
            pad0: u32,
            pad1: u32,
            pad2: u32,
            background: [f32; 4],
        }
        let raster_params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TiledGrad Raster Params",
            &[TileRasterParams {
                width,
                height,
                tiles_x: binning.tiles_x,
                tiles_y: binning.tiles_y,
                save_intermediates: 1,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                background: [background.x, background.y, background.z, 0.0],
            }],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let output_buffer = buffers::create_buffer(
            &self.ctx.device,
            "TiledGrad Output",
            output_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        // Consumed only by the backward dispatch below (same submission chain) — no
        // COPY_SRC needed (debug/test inspection of tiled pixel_state goes through
        // debug_render_tiled_pixel_state instead).
        let pixel_state_buffer = buffers::create_buffer_zeroed::<[u32; 2]>(
            &self.ctx.device,
            "TiledGrad Pixel State",
            num_pixels,
            BufferUsages::STORAGE,
        );

        let raster_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TiledGrad Raster Bind Group"),
            layout: &self.rasterize_tiled_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: raster_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: binning.projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: binning.pairs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: binning.ranges_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pixel_state_buffer.as_entire_binding() },
            ],
        });

        // Forward pass: its own submission (phase-per-submission convention — see
        // tile_binning_gpu's Project+Count / Emit / Sort / Ranges phases above).
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TiledGrad Forward Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TiledGrad Forward Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rasterize_tiled_pipeline);
            pass.set_bind_group(0, &raster_bind_group, &[]);
            pass.dispatch_workgroups(binning.tiles_x, binning.tiles_y, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        // Upload upstream gradients.
        let d_pixels_gpu: Vec<[f32; 4]> = d_pixels.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();
        let d_pixels_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TiledGrad d_pixels",
            &d_pixels_gpu,
            BufferUsages::STORAGE,
        );

        let total_i32s = num_gaussians * GRADIENT_I32_PER_GAUSSIAN;
        let zero_grads: Vec<i32> = vec![0i32; total_i32s];
        let gradient_atomic_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TiledGrad Gradient Atomic (i32)",
            &zero_grads,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let d_background_pixels_buffer = buffers::create_buffer_zeroed::<[f32; 4]>(
            &self.ctx.device,
            "TiledGrad d_background_pixels",
            num_pixels,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TileBackwardParams {
            width: u32,
            height: u32,
            tiles_x: u32,
            tiles_y: u32,
            background: [f32; 4],
        }
        let backward_params_buffer = buffers::create_buffer_init(
            &self.ctx.device,
            "TiledGrad Backward Params",
            &[TileBackwardParams {
                width,
                height,
                tiles_x: binning.tiles_x,
                tiles_y: binning.tiles_y,
                background: [background.x, background.y, background.z, 0.0],
            }],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        // Bindings match backward_tiled.wgsl exactly: 0 uniform + 1 projected (read) +
        // 2 pairs (read) + 3 tile_ranges (read) + 4 pixel_state (read) + 5 d_pixels (read)
        // + 6 gradient_atomic (read_write) + 7 d_background_pixels (read_write) — 1
        // uniform + 7 storage buffers in one bind group (verified against this device's
        // max_storage_buffers_per_shader_stage=8 by the Stage 5b pipeline-creation smoke).
        let backward_bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TiledGrad Backward Bind Group"),
            layout: &self.backward_tiled_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: backward_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: binning.projected_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: binning.pairs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: binning.ranges_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pixel_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: d_pixels_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: gradient_atomic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: d_background_pixels_buffer.as_entire_binding() },
            ],
        });

        // Backward pass: its own submission.
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TiledGrad Backward Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TiledGrad Backward Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backward_tiled_pipeline);
            pass.set_bind_group(0, &backward_bind_group, &[]);
            pass.dispatch_workgroups(binning.tiles_x, binning.tiles_y, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        // Readback + fixed-point -> f32 conversion, shared with the naive path.
        let pixel_grads_i32: Vec<i32> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &gradient_atomic_buffer,
            total_i32s,
        )
        .map_err(|e| format!("Failed to read per-Gaussian gradients: {e}"))?;
        let mut final_grads = convert_fixed_point_gradients(&pixel_grads_i32, num_gaussians);

        let d_background_pixels: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &d_background_pixels_buffer,
            num_pixels,
        )
        .map_err(|e| format!("Failed to read per-pixel background gradients: {e}"))?;
        final_grads.d_background = sum_d_background(&d_background_pixels);

        let output: Vec<[f32; 4]> = buffers::read_buffer_blocking(
            &self.ctx.device,
            &self.ctx.queue,
            &output_buffer,
            num_pixels,
        )
        .map_err(|e| format!("Failed to read tiled output buffer: {e}"))?;
        let pixels = output
            .into_iter()
            .map(|p| Vector3::new(p[0], p[1], p[2]))
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
