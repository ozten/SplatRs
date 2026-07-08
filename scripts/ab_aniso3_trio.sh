#!/bin/bash
# A/B: --max-log-aniso 3.0 (needle prune auto at 3.4) vs unclamped post-bwfix trio.
# Controls: runs/ab_bwfix_d0 (17.54), ab_bwfix_d500 (18.91), ab_bwfix_d100 (18.86).
set -uo pipefail
cd /Users/ozten/Projects/SplatRs
BIN=target/release/sugar-train
COMMON="--preset micro --scene datasets/tandt_db/tandt/train/sparse/0 --dataset-root datasets/tandt_db/tandt/train --images datasets/tandt_db/tandt/train/images --iters 2000 --seed 42 --downsample 0.5 --max-log-aniso 3.0"

echo "=== ARM d0 start $(date) ==="
$BIN $COMMON --densify-interval 0 --out-dir runs/ab_aniso3_d0
echo "=== ARM d0 exit $? $(date) ==="

echo "=== ARM d500 start $(date) ==="
$BIN $COMMON --densify-interval 500 --densify-max-gaussians 200000 --out-dir runs/ab_aniso3_d500
echo "=== ARM d500 exit $? $(date) ==="

echo "=== ARM d100 start $(date) ==="
$BIN $COMMON --densify-interval 100 --densify-max-gaussians 200000 --out-dir runs/ab_aniso3_d100
echo "=== ARM d100 exit $? $(date) ==="

echo "=== TRIO COMPLETE $(date) ==="
