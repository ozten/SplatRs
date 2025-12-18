#!/bin/bash
# Profile a SplatRs training run and generate a flamegraph
#
# Usage:
#   ./scripts/profile-training.sh <args-for-sugar-train>
#
# Example:
#   ./scripts/profile-training.sh --help
#   ./scripts/profile-training.sh --sparse datasets/scene/sparse/0 --images datasets/scene/images --iters 100
#
# The flamegraph will be saved as flamegraph.svg in the current directory.

set -e

echo "Building sugar-train with profiling profile..."
cargo build --profile profiling --bin sugar-train --features gpu

echo ""
echo "Running cargo flamegraph..."
echo "Arguments: $@"
echo ""

# Run flamegraph with profiling profile
# --profile profiling uses our custom profiling profile
# --bin sugar-train targets the training binary
# --features gpu enables GPU support
cargo flamegraph --profile profiling --bin sugar-train --features gpu -- "$@"

echo ""
echo "Flamegraph generated: flamegraph.svg"
echo "Open with: open flamegraph.svg (macOS) or xdg-open flamegraph.svg (Linux)"
