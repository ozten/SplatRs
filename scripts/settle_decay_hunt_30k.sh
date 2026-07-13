#!/usr/bin/env bash
# Settle-decay hunt, 30k horizon (2026-07-13) — where the REAL mechanism-#2 decay lives.
# The 15k batch proved the 15k "decay" is a measurement artifact (flat settle). At 30k the
# settle plateau genuinely sits 0.6-0.8 dB below the window peak, and mining srt30k_a3_sp500
# showed three monotonic settle drifts tracking the decline: count 60k→56k, aniso_p90 17→20
# (needling INTO the max_log_aniso=3.0 clamp), scale_median +11%, opacity parked 0.010.
# Two arms, one lever each vs a FRESH same-binary control:
#   - drift arm: --settle-needle-prune-log-aniso 2.5  (settle prunes remove the needling parked
#     mass the normal clamp+0.4=3.4 needle threshold never touches; densify-time unchanged)
#   - reset arm: --opacity-reset-window-margin 5000   (last reset 15000→9000; 6000 iters of
#     densify+re-earn before settle vs the rg2500 run's 12000; targets the window→settle drop)
# Serial (single Metal GPU; concurrent trainers risk the cumulative command-buffer watchdog).
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500)

STATUS=runs/settle_decay_hunt_30k.status
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

run_arm srt30k_ctrl_banded                                      # fresh same-binary control
run_arm srt30k_sd_needle25  --settle-needle-prune-log-aniso 2.5 # geometric-drift arm (highest prior)
run_arm srt30k_sd_rg5000    --opacity-reset-window-margin 5000  # window→settle level-drop arm

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
