#!/bin/bash
# Post-renderer-fix 15k validation pair (commit 836f4d0): unclamped vs --max-log-aniso 3.0.
# Standard validation config: 100 img / 60k cap / seed 42 / downsample 0.5 / densify 100.
set -uo pipefail
cd /Users/ozten/Projects/SplatRs
BIN=target/release/sugar-train
COMMON="--preset micro --max-images 100 --scene datasets/tandt_db/tandt/train/sparse/0 --dataset-root datasets/tandt_db/tandt/train --images datasets/tandt_db/tandt/train/images --iters 15000 --densify-interval 100 --densify-max-gaussians 60000 --seed 42 --downsample 0.5"

echo "=== ARM srt15k_unclamped start $(date) ==="
$BIN $COMMON --out-dir runs/srt15k_unclamped
echo "=== ARM srt15k_unclamped exit $? $(date) ==="

echo "=== ARM srt15k_aniso3 start $(date) ==="
$BIN $COMMON --max-log-aniso 3.0 --out-dir runs/srt15k_aniso3
echo "=== ARM srt15k_aniso3 exit $? $(date) ==="

echo "=== 15K PAIR COMPLETE $(date) ==="
