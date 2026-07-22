#!/usr/bin/env bash
# Opacity-floor dose-response, round 3 (2026-07-22) — bracket the optimum.
# Curve so far (same binary, fresh control): 1x (0.01/0.005) settle mean 16.18 gap +0.87;
# 5x (0.05/0.025) settle mean 17.25 gap −0.24 final 17.21 = best-ever, DEFAULT as of today;
# 20x (0.2/0.1) settle mean 16.77 gap +0.54 final 16.31 — best window peak (17.31) but the
# aggressive prune removes useful dim mass and settle decays again. This arm: 10x (0.1/0.05).
# Explicit flags (not defaults) so the arm stays same-binary-comparable with the control.
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500)

STATUS=runs/settle_decay_hunt_30k_optfloor3.status
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

run_arm srt30k_sd_optfloor10  --opacity-reset-floor 0.1 --prune-opacity-threshold 0.05

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
