#!/usr/bin/env bash
# Opacity-floor dose-response, round 2 (2026-07-22) — the 5x arm won big
# (srt30k_sd_optfloor05: settle mean 17.25 vs control 16.18, gap went NEGATIVE −0.24 —
# first 30k settle phase ever to climb above its window peak; final 17.21 = new best;
# opacity_low_pct 93.2→60.7, count 37.7k→20.5k; visual check confirms crisper chevrons).
# This round tests full splatfacto levels — prune 0.1 / reset-floor 0.2 (20x baseline,
# 4x the winning arm) — to locate the optimum before flipping defaults. Same-binary
# control = srt30k_ctrl_needle28def from the 07:14 batch (hours old, same build).
# Serial (single Metal GPU; watchdog).
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500)

STATUS=runs/settle_decay_hunt_30k_optfloor2.status
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

run_arm srt30k_sd_optfloor20  --opacity-reset-floor 0.2 --prune-opacity-threshold 0.1

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
