#!/bin/bash
# Lever A/Bs vs the new baseline runs/srt15k_aniso3 (16.37 final / 16.56 settle mean).
# One lever at a time, clamp 3.0 default active in both:
#   (a) --opacity-reset-floor 0.004  (below the 0.005 prune threshold: dead mass becomes prunable)
#   (b) --settle-prune-interval 500  (prune-only passes continue after the densify window)
set -uo pipefail
cd /Users/ozten/Projects/SplatRs
BIN=target/release/sugar-train
COMMON="--preset micro --max-images 100 --scene datasets/tandt_db/tandt/train/sparse/0 --dataset-root datasets/tandt_db/tandt/train --images datasets/tandt_db/tandt/train/images --iters 15000 --densify-interval 100 --densify-max-gaussians 60000 --seed 42 --downsample 0.5 --max-log-aniso 3.0"

echo "=== ARM srt15k_a3_floor004 start $(date) ==="
$BIN $COMMON --opacity-reset-floor 0.004 --out-dir runs/srt15k_a3_floor004
echo "=== ARM srt15k_a3_floor004 exit $? $(date) ==="

echo "=== ARM srt15k_a3_sp500 start $(date) ==="
$BIN $COMMON --settle-prune-interval 500 --out-dir runs/srt15k_a3_sp500
echo "=== ARM srt15k_a3_sp500 exit $? $(date) ==="

echo "=== LEVERS PAIR COMPLETE $(date) ==="
