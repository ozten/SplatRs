// Bitonic sort for TileGaussianPair keyed lexicographically (key_tile, key_depth).
//
// Structure is a deliberate byte-for-byte copy of sort.wgsl's proven stage/step loop —
// especially the direction bit `(left_idx & (2u << params.stage))`, which is bit
// (stage+1), NOT bit `stage`: getting that wrong left ~45% adjacent inversions and
// depth-random compositing for the whole early GPU campaign (fixed in 836f4d0). Do not
// "simplify" this expression. Padding entries use key_tile = num_tiles so they sort
// after every real tile (ascending network over the full power-of-two padded array).

struct TileGaussianPair {
    key_tile: u32,
    key_depth: u32,
    gaussian_idx: u32,
    pad: u32,
}

struct SortParams {
    padded_count: u32,       // Buffer length: power of two, pad entries key_tile=num_tiles
    stage: u32,              // Current sorting stage (0..log2(n))
    step_within_stage: u32,  // Current step within stage
    pad: u32,                // Padding for 16-byte alignment
}

@group(0) @binding(0) var<storage, read_write> pairs: array<TileGaussianPair>;
@group(0) @binding(1) var<uniform> params: SortParams;

fn pair_greater(a: TileGaussianPair, b: TileGaussianPair) -> bool {
    if (a.key_tile != b.key_tile) {
        return a.key_tile > b.key_tile;
    }
    return a.key_depth > b.key_depth;
}

@compute @workgroup_size(256)
fn bitonic_sort_pairs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    // One thread per compare-exchange pair. The network MUST run over the entire
    // power-of-two padded array — skipping any pair breaks the sortedness guarantee.
    if (idx >= params.padded_count / 2u) { return; }

    let pair_distance = 1u << params.step_within_stage;
    let block_size = 2u << params.step_within_stage;

    let left_idx = (idx / pair_distance) * block_size + (idx % pair_distance);
    let right_idx = left_idx + pair_distance;

    // Bit (stage+1) of the index decides direction — see file header.
    let ascending = (left_idx & (2u << params.stage)) == 0u;

    let left = pairs[left_idx];
    let right = pairs[right_idx];

    let should_swap = (ascending && pair_greater(left, right)) ||
                      (!ascending && pair_greater(right, left));

    if (should_swap) {
        pairs[left_idx] = right;
        pairs[right_idx] = left;
    }
}
