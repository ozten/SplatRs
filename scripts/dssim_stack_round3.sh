#!/usr/bin/env bash
# DSSIM recipe stacking, round 3 (2026-07-24): does rg2500 stack with needle-off?
# Round 1: rg2500 alone = best endpoint (+0.43) but decay persisted. Round 2: needle-off
# alone = decay fixed (17.35). Under L2 the reset-gate did NOT stack with the needle
# lever, but the mechanism differs under DSSIM (window benefits from resets; the gate
# only trims the last one). One arm vs runs/srt30k_dssim_noneedle (17.35/17.12) control.
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
STATUS=runs/dssim_stack_round3.status
echo "batch start $(date)" > "$STATUS"

name=srt30k_dssim_noneedle_rg2500
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
  --loss l1-dssim --settle-needle-prune-log-aniso 0 \
  --opacity-reset-window-margin 2500 --out-dir "$out" > "$out.log" 2>&1
rc=$?
final=$(tail -1 "$out/metrics.csv" 2>/dev/null | cut -d, -f1,3)
echo "[$(date)] DONE  $name  rc=$rc  (iter,psnr)=$final" | tee -a "$STATUS"
echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
