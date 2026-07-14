#!/usr/bin/env bash
# Settle-decay hunt, 30k horizon, round 2 (2026-07-13) — refine + combine the confirmed
# drift win. Round 1 proved the needling parked mass drives the 30k decay: needle-prune @2.5
# lifted settle mean +0.24 and cut the gap 32%, but over-pruned (count 60k→34k, −43%) and the
# survivor scale inflated (residual scale-driven slope). Two arms, each one-lever-different
# from a KNOWN point (control srt30k_ctrl_banded and the round-1 needle25 run, both this binary):
#   - needle28: --settle-needle-prune-log-aniso 2.8  (gentler; keep the gain, less count loss —
#     isolates threshold 2.8 vs 2.5)
#   - needle25_rg2500: --settle-needle-prune-log-aniso 2.5 --opacity-reset-window-margin 2500
#     (adds the reset-gate optimum to the proven 2.5 drift win — isolates the reset contribution)
# Control is REUSED (runs/srt30k_ctrl_banded, same binary) — not re-run. Serial (single GPU).
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500)

STATUS=runs/settle_decay_hunt_30k_v2.status
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

run_arm srt30k_sd_needle28        --settle-needle-prune-log-aniso 2.8
run_arm srt30k_sd_needle25_rg2500 --settle-needle-prune-log-aniso 2.5 --opacity-reset-window-margin 2500

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
