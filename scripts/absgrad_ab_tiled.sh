#!/usr/bin/env bash
# Absgrad A/B on the tiled pipeline (2026-07-24): does abs-gradient densification improve
# the DSSIM recipe? Control = the recipe on --tile-raster (fresh same-binary control, also
# revalidates tiled-vs-naive at this config: naive needle-off scored 17.35/17.12). Arm =
# + --densify-absgrad with the 2x threshold convention (0.0004 vs signed default 0.0002).
# Both ~40-60min on the tiled trainer.
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500 \
  --opacity-reset-floor 0.05 --prune-opacity-threshold 0.025 \
  --loss l1-dssim --settle-needle-prune-log-aniso 0 --tile-raster)

STATUS=runs/absgrad_ab_tiled.status
echo "batch start $(date)" > "$STATUS"

run_arm () {
  local name="$1"; shift
  local out="runs/$name"
  echo "[$(date)] START $name  levers: $*" | tee -a "$STATUS"
  rm -rf "$out"
  "$BIN" "${COMMON[@]}" "$@" --out-dir "$out" > "$out.log" 2>&1
  local rc=$?
  local final=$(tail -1 "$out/metrics.csv" 2>/dev/null | cut -d, -f1,3)
  echo "[$(date)] DONE  $name  rc=$rc  (iter,psnr)=$final" | tee -a "$STATUS"
}

run_arm srt30k_tiled_ctrl
run_arm srt30k_tiled_absgrad  --densify-absgrad --densify-grad-threshold 0.0004

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
