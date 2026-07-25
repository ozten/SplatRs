#!/usr/bin/env bash
# Repeat of srt30k_tiled_ctrl (2026-07-24): bounds tiled-path 30k run-to-run variance at
# the 100-img config (first draw: 16.86 vs the naive family's reproducible 17.51-peak).
set -u
cd /Users/ozten/Projects/SplatRs
BIN=./target/release/sugar-train
STATUS=runs/tiled_ctrl_repeat.status
echo "batch start $(date)" > "$STATUS"
name=srt30k_tiled_ctrl_b
out="runs/$name"
echo "[$(date)] START $name" | tee -a "$STATUS"
rm -rf "$out"
"$BIN" --preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500 \
  --opacity-reset-floor 0.05 --prune-opacity-threshold 0.025 \
  --loss l1-dssim --settle-needle-prune-log-aniso 0 --tile-raster \
  --out-dir "$out" > "$out.log" 2>&1
rc=$?
final=$(tail -1 "$out/metrics.csv" 2>/dev/null | cut -d, -f1,3)
echo "[$(date)] DONE  $name  rc=$rc  (iter,psnr)=$final" | tee -a "$STATUS"
echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
