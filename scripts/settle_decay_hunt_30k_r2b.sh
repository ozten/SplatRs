#!/usr/bin/env bash
# Settle-decay hunt, 30k horizon, round 2b (2026-07-14) — relaunch of the combo arm.
# Round 2's second arm (needle25_rg2500) was killed at iter ~2100 by the 2026-07-13 21:30
# OS-update reboot. Meanwhile round 2's first arm proved needle 2.8 DOMINATES 2.5
# (final 16.74 vs 16.20, settle mean 16.68 vs 16.58, slope +0.018 vs -0.049, gap +0.10):
# combining the reset gate with 2.5 would answer a stale question, so the combo arm is
# relaunched on the 2.8 base instead. One lever different from known srt30k_sd_needle28.
#   - needle28_rg2500: --settle-needle-prune-log-aniso 2.8 --opacity-reset-window-margin 2500
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500)

STATUS=runs/settle_decay_hunt_30k_r2b.status
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

run_arm srt30k_sd_needle28_rg2500 --settle-needle-prune-log-aniso 2.8 --opacity-reset-window-margin 2500

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
