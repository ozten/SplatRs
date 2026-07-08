#!/bin/bash
# Post-renderer-fix re-baseline (commit 836f4d0: sort + bbox + rotation fixes).
# 2k trio, unclamped AND --max-log-aniso 3.0, seed 42, downsample 0.5.
# All pre-fix numbers (ab_bwfix_*, ab_aniso3_*) are superseded.
set -uo pipefail
cd /Users/ozten/Projects/SplatRs
BIN=target/release/sugar-train
COMMON="--preset micro --scene datasets/tandt_db/tandt/train/sparse/0 --dataset-root datasets/tandt_db/tandt/train --images datasets/tandt_db/tandt/train/images --iters 2000 --seed 42 --downsample 0.5"

run() {
  local name=$1; shift
  echo "=== ARM $name start $(date) ==="
  $BIN $COMMON "$@" --out-dir runs/$name
  echo "=== ARM $name exit $? $(date) ==="
}

run ab_srt_d0        --densify-interval 0
run ab_srt_d500      --densify-interval 500 --densify-max-gaussians 200000
run ab_srt_d100      --densify-interval 100 --densify-max-gaussians 200000
run ab_srt_a3_d0     --densify-interval 0   --max-log-aniso 3.0
run ab_srt_a3_d500   --densify-interval 500 --densify-max-gaussians 200000 --max-log-aniso 3.0
run ab_srt_a3_d100   --densify-interval 100 --densify-max-gaussians 200000 --max-log-aniso 3.0

echo "=== TRIOS COMPLETE $(date) ==="
