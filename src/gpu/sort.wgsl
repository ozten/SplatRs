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
    padded_count: u32,       // Buffer length: power of two, pad entries hold +inf depth
    stage: u32,              // Current sorting stage (0..log2(n))
    step_within_stage: u32,  // Current step within stage
    pad: u32,                // Padding for 16-byte alignment
}

@group(0) @binding(0) var<storage, read_write> gaussians: array<Gaussian2D>;
@group(0) @binding(1) var<uniform> params: SortParams;

@compute @workgroup_size(256)
fn bitonic_sort(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    // One thread per compare-exchange pair. The network MUST run over the entire
    // power-of-two padded array — skipping any pair breaks the sortedness guarantee.
    // (The projection pass fills the pad region with +inf-depth sentinels.)
    if (idx >= params.padded_count / 2u) { return; }

    // Compute pair indices using bitonic network pattern
    // pair_distance = 2^step_within_stage
    let pair_distance = 1u << params.step_within_stage;
    let block_size = 2u << params.step_within_stage;

    // Compute left and right indices in sorted pair
    let left_idx = (idx / pair_distance) * block_size + (idx % pair_distance);
    let right_idx = left_idx + pair_distance;

    // Determine sort direction for the bitonic sequence: blocks of size 2^(stage+1)
    // alternate ascending/descending, i.e. bit (stage+1) of the index — NOT bit `stage`
    // (that bug left ~45% adjacent inversions and made GPU compositing depth-random).
    let ascending = (left_idx & (2u << params.stage)) == 0u;

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
