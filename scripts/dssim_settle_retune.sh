#!/usr/bin/env bash
# P1: DSSIM settle re-tune (2026-07-23, AUTONOMY_PLAN_20260723.md). Baseline
# runs/srt30k_optfloor05_l1dssim: best-ever window peak (17.51) + LPIPS (0.496) but settle
# decays -0.87 under L2-tuned settings (final 15.88). Its population lives at opacity ~0.5,
# so the 0.05 reset floor is a 10x cut (vs 2x for L2 populations) — these arms re-tune the
# reset/settle levers under l1-dssim, one lever each. Judged on SSIM (new eval_ssim column)
# + LPIPS (scripts/compute_lpips_run.py) + visual, NOT PSNR alone.
# Baseline binary predates the SSIM instrumentation but training math is unchanged by it,
# so the baseline remains a valid control; arms additionally get eval_ssim logged.
# Serial (single Metal GPU; watchdog).
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
  --loss l1-dssim)

STATUS=runs/dssim_settle_retune.status
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

run_arm srt30k_dssim_rg2500     --opacity-reset-window-margin 2500   # resets end early
run_arm srt30k_dssim_noreset    --opacity-reset-window-margin 15000  # margin >= window: no resets at all
run_arm srt30k_dssim_floor25    --opacity-reset-floor 0.25           # ~2x cut for a 0.5-median population

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
