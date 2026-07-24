#!/usr/bin/env bash
# P3 showcase (2026-07-23, AUTONOMY_PLAN_20260723.md): the winning DSSIM recipe at full
# data scale on the one-metric harness — directly comparable to splatfacto's 22.21 on the
# identical 263/38 interval-8 split. Recipe = l1-dssim + settle-needle-prune OFF (the
# round-2 winner: settle mean 17.12, final 17.35, best LPIPS 0.415 in-settle) + 0.05/0.025
# opacity defaults + standard resets. Cap 150k for the 3x view count; checkpoint grid
# every 1500 for the count-vs-quality overlay.
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
STATUS=runs/p3_showcase.status
echo "batch start $(date)" > "$STATUS"

name=srt30k_p3_dssim_int8
out="runs/$name"
echo "[$(date)] START $name" | tee -a "$STATUS"
rm -rf "$out"
"$BIN" --preset micro --max-images 0 --eval-interval 8 --max-test-views 0 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 150000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500 \
  --opacity-reset-floor 0.05 --prune-opacity-threshold 0.025 \
  --loss l1-dssim --settle-needle-prune-log-aniso 0 \
  --save-interval 1500 --out-dir "$out" > "$out.log" 2>&1
rc=$?
final=$(tail -1 "$out/metrics.csv" 2>/dev/null | cut -d, -f1,3)
echo "[$(date)] DONE  $name  rc=$rc  (iter,psnr)=$final" | tee -a "$STATUS"
echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
