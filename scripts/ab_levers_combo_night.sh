#!/bin/bash
# Overnight follow-ups to the single-lever A/Bs:
#   (1) combo: reset-floor 0.004 + settle-prune 500 at the standard 15k config
#   (2) horizon test: floor004 alone at 30k (the 15k arm ended still climbing)
set -uo pipefail
cd /Users/ozten/Projects/SplatRs
BIN=target/release/sugar-train
COMMON="--preset micro --max-images 100 --scene datasets/tandt_db/tandt/train/sparse/0 --dataset-root datasets/tandt_db/tandt/train --images datasets/tandt_db/tandt/train/images --densify-interval 100 --densify-max-gaussians 60000 --seed 42 --downsample 0.5 --max-log-aniso 3.0"

echo "=== ARM srt15k_a3_floor004_sp500 start $(date) ==="
$BIN $COMMON --iters 15000 --opacity-reset-floor 0.004 --settle-prune-interval 500 --out-dir runs/srt15k_a3_floor004_sp500
echo "=== ARM srt15k_a3_floor004_sp500 exit $? $(date) ==="

echo "=== ARM srt30k_a3_floor004 start $(date) ==="
$BIN $COMMON --iters 30000 --opacity-reset-floor 0.004 --out-dir runs/srt30k_a3_floor004
echo "=== ARM srt30k_a3_floor004 exit $? $(date) ==="

echo "=== NIGHT RUNS COMPLETE $(date) ==="
