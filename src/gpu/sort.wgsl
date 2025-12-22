// Bitonic sort shader for sorting Gaussian2D by depth (mean.z)
//
// This implements GPU-side bitonic sort to avoid CPU-GPU-CPU PCI-e bottleneck.
// Bitonic sort is ideal for GPUs: O(log²n) parallel passes, fixed access patterns.

// Gaussian 2D structure (must match types.rs Gaussian2DGPU)
struct Gaussian2D {
    mean: vec4<f32>,          // Pixel space (x,y,depth,pad)
    cov: vec4<f32>,           // 2D covariance (xx,xy,yy,pad)
    color: vec4<f32>,         // Linear RGB
    opacity_pad: vec4<f32>,   // Opacity [0,1]
    gaussian_idx_pad: vec4<u32>, // Source index
}

// Sort parameters for each pass
struct SortParams {
    count: u32,              // Number of elements to sort
    stage: u32,              // Current sorting stage (0..log2(n))
    step_within_stage: u32,  // Current step within stage
    pad: u32,                // Padding for 16-byte alignment
}

@group(0) @binding(0) var<storage, read_write> gaussians: array<Gaussian2D>;
@group(0) @binding(1) var<uniform> params: SortParams;

@compute @workgroup_size(256)
fn bitonic_sort(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.count) { return; }

    // Compute pair indices using bitonic network pattern
    // pair_distance = 2^step_within_stage
    let pair_distance = 1u << params.step_within_stage;
    let block_size = 2u << params.step_within_stage;

    // Compute left and right indices in sorted pair
    let left_idx = (idx / pair_distance) * block_size + (idx % pair_distance);
    let right_idx = left_idx + pair_distance;

    if (right_idx >= params.count) { return; }

    // Determine sort direction (ascending/descending) for bitonic sequence
    // Alternates between ascending and descending for each stage
    let ascending = ((left_idx >> params.stage) & 1u) == 0u;

    // Compare depths (z coordinate in mean.z)
    let left_depth = gaussians[left_idx].mean.z;
    let right_depth = gaussians[right_idx].mean.z;

    // Swap if needed to maintain bitonic property
    let should_swap = (ascending && left_depth > right_depth) ||
                      (!ascending && left_depth < right_depth);

    if (should_swap) {
        // Swap entire Gaussian2D structures
        let temp = gaussians[left_idx];
        gaussians[left_idx] = gaussians[right_idx];
        gaussians[right_idx] = temp;
    }
}
