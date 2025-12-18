# Performance Profiling for SplatRs

This document describes the performance profiling setup for SplatRs, which helps identify performance bottlenecks and optimize hot code paths.

## Overview

Performance profiling helps:
- **Identify bottlenecks**: Which functions consume the most CPU time
- **Optimize hot paths**: Focus optimization efforts where they matter most
- **Validate optimizations**: Measure impact of performance improvements
- **Understand behavior**: Visualize call stacks and execution flow

SplatRs uses `cargo-flamegraph`, which generates interactive flamegraph visualizations showing where your program spends time.

## Quick Start

### Profile a micro training run (100 iterations)

```bash
make profile-micro
```

This runs a 100-iteration training session with the `micro` preset and generates `flamegraph.svg`.

The micro preset includes:
- 100 iterations (~10-15 minutes)
- Multi-view training (12 train views, 3 test views)
- All parameters enabled (position, rotation, scale, opacity, SH, background)
- Adaptive density control (densify every 25 iterations)
- Maximum 3,000 gaussians (grows to ~5,000 with densification)

### Open the flamegraph

```bash
open flamegraph.svg         # macOS
xdg-open flamegraph.svg     # Linux
firefox flamegraph.svg      # Or any browser
```

## Available Make Targets

| Target | Description |
|--------|-------------|
| `make profile-micro` | Profile 100-iteration micro training run |
| `make profile PROFILE_ARGS="..."` | Profile with custom arguments |
| `make profile-clean` | Clean profiling artifacts (*.svg, perf.data, etc.) |

## Manual Profiling

### Using the profiling script

```bash
# Profile micro preset (100 iters)
./scripts/profile-training.sh --preset micro --dataset-root datasets/tandt_db/tandt/train

# Profile M8 preset (500 iters) - longer run
./scripts/profile-training.sh --preset m8 --dataset-root datasets/tandt_db/tandt/train

# Profile custom configuration
./scripts/profile-training.sh --multiview --scene datasets/scene/sparse/0 \
  --images datasets/scene/images --iters 50 --downsample 0.125
```

### Using cargo-flamegraph directly

```bash
# Build with profiling profile
cargo build --profile profiling --bin sugar-train --all-features

# Generate flamegraph
cargo flamegraph --profile profiling --bin sugar-train --all-features -- \
  --preset micro --dataset-root datasets/tandt_db/tandt/train
```

## Understanding Flamegraphs

### What is a flamegraph?

A flamegraph is a visualization of profiled software stack traces:
- **X-axis (width)**: Time spent in a function (proportional to total runtime)
- **Y-axis (height)**: Call stack depth (top is called by bottom)
- **Color**: Random (just for differentiation, not meaningful)

### How to read a flamegraph

1. **Hover over boxes**: See function name, percentage, and sample count
2. **Click to zoom**: Focus on a specific function and its children
3. **Click "Reset zoom"**: Return to full view
4. **Wide boxes = hot functions**: These consume the most CPU time
5. **Tall stacks = deep calls**: May indicate recursion or complex call chains

### Example interpretation

```
┌─────────────────────────────────────────────────────────┐
│ main                                                    │  ← Entry point
├─────────────────────────────────────────────────────────┤
│ train_multiview_color_only                             │  ← Training loop
├──────────────────────┬──────────────────┬──────────────┤
│ forward_pass         │ backward_pass    │ optimizer    │  ← Major phases
│ [50%]                │ [30%]            │ [20%]        │
├──────────┬───────────┼──────────────────┴──────────────┘
│ render   │ loss      │
│ [45%]    │ [5%]      │
└──────────┴───────────┘
```

In this example:
- 50% of time is in forward pass (rendering)
- 30% in backward pass (gradients)
- 20% in optimizer (parameter updates)

## Performance Optimization Workflow

### 1. Generate baseline flamegraph

```bash
make profile-micro
```

### 2. Identify the widest boxes

Look for:
- **Wide boxes at the top**: Hot functions that consume significant time
- **Repeated patterns**: Functions called many times
- **Unexpected wide boxes**: Inefficient code paths

### 3. Common hotspots in Gaussian Splatting

Expected hotspots (these are normal):
- `render_with_gradients` or `render`: Forward pass rendering
- `backward_pass`: Gradient computation
- `apply_gradients` or `step`: Optimizer updates
- `project_gaussians`: Gaussian projection to image space
- `alpha_blending`: Pixel color compositing

Unexpected hotspots (worth investigating):
- Memory allocations (`alloc`, `realloc`)
- Locking/synchronization (`Mutex::lock`, `RwLock`)
- Unnecessary copies (`clone`, `to_vec`)
- Debug printing or logging in hot paths

### 4. Optimize and re-profile

After making changes:
```bash
make profile-micro
# Compare with previous flamegraph
```

### 5. Validate improvements

Look for:
- Reduced width of optimized functions
- Faster total runtime (shown in terminal output)
- Better PSNR if optimization affected numerical quality

## Profiling Configuration

### Cargo profile: `profiling`

Defined in `Cargo.toml`:
```toml
[profile.profiling]
# Inherits from release but includes debug symbols
inherits = "release"
debug = true
# Disable LTO for faster compilation during profiling iterations
lto = false
```

**Why this profile?**
- **Optimized code**: Full release optimizations (opt-level 3)
- **Debug symbols**: Function names appear in flamegraphs
- **Fast compilation**: No LTO means faster incremental builds

### Training presets

The `sugar-train` binary has several presets optimized for different scenarios:

| Preset | Iterations | Use Case |
|--------|-----------|----------|
| `micro` | 100 | Quick profiling runs (~10-15 min) |
| `m8-smoke` | 50 | Smoke testing (very fast) |
| `m8` | 500 | Full M8 validation |
| `m9` | 1000 | With densification |
| `m10` | 2000 | Reference-quality training |

For profiling, **use `micro` or `m8-smoke`** for quick iteration.

## Profiling Tips

### macOS-specific: dtrace permissions

On macOS, cargo-flamegraph uses `dtrace` which requires special permissions:

```bash
# Option 1: Run with sudo (not recommended for regular use)
sudo cargo flamegraph ...

# Option 2: Grant dtrace access to your terminal (recommended)
# System Preferences → Security & Privacy → Privacy → Developer Tools
# Add your terminal app (Terminal.app, iTerm2, etc.)
```

### Linux-specific: perf permissions

On Linux, cargo-flamegraph uses `perf` which may require:

```bash
# Temporarily allow perf for all users
sudo sysctl -w kernel.perf_event_paranoid=-1

# Or run with sudo
sudo cargo flamegraph ...
```

### Profiling GPU code

**Important**: Flamegraphs only show CPU time, not GPU time.

For GPU kernels:
- GPU compute appears as **thin boxes** (CPU just launches kernels)
- Synchronization points show up as **wide boxes** (CPU waits for GPU)
- To profile GPU, use dedicated tools like:
  - **Nsight Compute** (NVIDIA)
  - **AMD GPU Profiler**
  - **Intel VTune** (for Intel GPUs)
  - **Metal System Trace** (macOS Metal)

### Reduce noise in flamegraphs

1. **Disable logging**: Set `log_interval` to a large value or 0
2. **Use deterministic RNG**: Set `--seed 0` for consistent behavior
3. **Profile longer runs**: More samples = more accurate results
4. **Avoid I/O**: Don't save images during profiling (or use ramdisk)

### Compare flamegraphs

To compare before/after optimizations:

```bash
# Before optimization
make profile-micro
mv flamegraph.svg flamegraph-before.svg

# Make changes...

# After optimization
make profile-micro
mv flamegraph.svg flamegraph-after.svg

# Open both in browser tabs and compare widths
```

## Common Performance Patterns

### Pattern: Forward pass dominates (expected)

```
forward_pass [60%]
├─ render [55%]
│  ├─ project_gaussians [25%]
│  └─ alpha_blending [30%]
└─ loss_computation [5%]
```

**Expected**: Forward pass (rendering) is expensive. This is normal.

### Pattern: Backward pass dominates (investigate)

```
backward_pass [70%]
└─ gradient_computation [70%]
   └─ pixel_loop [65%]
```

**Action**: Backward pass should be ~30-40% of total time. If much higher:
- Check for inefficient gradient accumulation
- Consider GPU backward pass (if not already enabled)
- Look for unnecessary recomputation

### Pattern: Memory allocations in hot path

```
render [50%]
└─ alloc::vec::Vec::push [20%]
```

**Action**: Preallocate buffers outside the hot loop:
```rust
// Bad (allocates every iteration)
for iter in 0..iters {
    let mut buffer = Vec::new();  // Allocation in hot path!
    // ...
}

// Good (allocates once)
let mut buffer = Vec::new();
for iter in 0..iters {
    buffer.clear();  // Reuse allocation
    // ...
}
```

### Pattern: Rayon overhead

```
train_loop [100%]
└─ rayon::iter::parallel [80%]
   ├─ actual_work [40%]
   └─ rayon::thread_pool::join [40%]  ← Overhead!
```

**Action**: Rayon overhead is too high. Consider:
- Increase chunk size
- Reduce parallelism for small datasets
- Use sequential iteration for small loops

## Profiling Checklist

Before profiling:
- [ ] Build with `--profile profiling` (or use provided scripts)
- [ ] Use a consistent dataset and seed for reproducibility
- [ ] Disable or minimize logging
- [ ] Ensure no background processes consume CPU

During profiling:
- [ ] Let the run complete (don't Ctrl+C early)
- [ ] Monitor terminal output for errors or warnings
- [ ] Note the total runtime for comparison

After profiling:
- [ ] Identify the top 3 widest boxes
- [ ] Check if they match expectations
- [ ] Look for unexpected allocations or locks
- [ ] Save flamegraph with a descriptive name

## Troubleshooting

### Flamegraph shows `???` or hex addresses

**Cause**: Debug symbols are missing.

**Fix**: Ensure you're using the `profiling` profile:
```bash
cargo flamegraph --profile profiling ...
```

### "Permission denied" on macOS/Linux

**Cause**: `dtrace` (macOS) or `perf` (Linux) requires elevated permissions.

**Fix**: See "Profiling Tips" section above.

### Profiling takes too long

**Cause**: `micro` preset still runs 100 iterations.

**Fix**: Use a faster preset or reduce iterations:
```bash
./scripts/profile-training.sh --preset m8-smoke --dataset-root datasets/tandt_db/tandt/train
# or
./scripts/profile-training.sh --multiview --scene ... --iters 20
```

### Flamegraph is all `rayon` overhead

**Cause**: Profiling very small datasets with high parallelism.

**Fix**: Increase dataset size or reduce parallelism:
```bash
--max-images 25  # More images
--downsample 0.5  # Larger images
```

## Next Steps

After setting up profiling and identifying bottlenecks:

1. **Prioritize optimizations**: Focus on the widest boxes first
2. **Validate correctness**: Ensure optimizations don't break tests
3. **Measure impact**: Re-profile and compare before/after
4. **Document changes**: Note performance improvements in commit messages

## Additional Resources

- [Flamegraph.pl](https://github.com/brendangregg/FlameGraph) - Original flamegraph tool
- [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph) - Rust wrapper
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Profiling Rust Applications](https://fasterthanli.me/articles/profiling-rust-applications)
