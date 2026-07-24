#!/usr/bin/env bash
# P3-v2 (2026-07-24): the P3 showcase config re-run WITH --tile-raster (Stage 5 complete:
# tiled forward+backward bit-exact, training equivalence 0.0000 dB). Same recipe/config as
# runs/srt30k_p3_dssim_int8 (16.72 final in ~12.5h naive) — this validates the tiled
# trainer at full scale and, if it reproduces ~16.7, becomes the standard fast pipeline
# (~2-3h projected). Checkpoint grid kept for the count-vs-quality overlay.
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
STATUS=runs/p3v2_tiled.status
echo "batch start $(date)" > "$STATUS"

name=srt30k_p3v2_tiled
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
  --save-interval 1500 --tile-raster --out-dir "$out" > "$out.log" 2>&1
rc=$?
final=$(tail -1 "$out/metrics.csv" 2>/dev/null | cut -d, -f1,3)
echo "[$(date)] DONE  $name  rc=$rc  (iter,psnr)=$final" | tee -a "$STATUS"
echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
