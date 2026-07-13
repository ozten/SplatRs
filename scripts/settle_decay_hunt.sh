#!/usr/bin/env bash
# Settle-decay mechanism hunt (2026-07-12) — half-res 60k-cap A/B batch.
# Isolates what degrades the model DURING the settle phase (iter > iters/2): every arm in the
# campaign peaks mid-densify-window (~17.0-17.3) then settles ~0.3-1.1 dB below and never
# re-attains the peak ("mechanism #2"). Each arm differs from the fresh control by ONE lever.
# Serial (single Metal GPU; concurrent trainers risk the cumulative command-buffer watchdog).
#
# Control command = exact reference of runs/srt15k_a3_sp500 (16.76), re-run on the current
# banded-renderer binary (fix d8dd9a7 postdates that run) to remove any drift confound.
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 15000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0)

STATUS=runs/settle_decay_hunt.status
echo "batch start $(date)" > "$STATUS"

# run_arm <name> <full lever args...>  — each arm lists its COMPLETE lever set (no reliance
# on later flags overriding earlier ones).
run_arm () {
  local name="$1"; shift
  local out="runs/$name"
  echo "[$(date)] START $name  levers: $*" | tee -a "$STATUS"
  rm -rf "$out"
  "$BIN" "${COMMON[@]}" "$@" --out-dir "$out" > "$out.log" 2>&1
  local rc=$?
  local final=$(tail -1 "$out/metrics.csv" 2>/dev/null | cut -d, -f1,3)
  echo "[$(date)] DONE  $name  rc=$rc  (iter,col3)=$final" | tee -a "$STATUS"
}

# Order by prior (highest-value first, so a truncated night still captures the best arms):
run_arm srt15k_ctrl_banded  --settle-prune-interval 500                             # fresh control
run_arm srt15k_sd_freezebg  --settle-prune-interval 500 --freeze-bg-in-settle       # bg->black drift
run_arm srt15k_sd_sp0       --settle-prune-interval 0                               # no settle prunes
run_arm srt15k_sd_freezesh  --settle-prune-interval 500 --freeze-sh-after-window    # continued SH opt

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
