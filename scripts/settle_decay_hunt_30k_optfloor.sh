#!/usr/bin/env bash
# Opacity-floor magnitude arm, 30k horizon (2026-07-22) — next-step-1 hypothesis from the
# count-vs-quality follow-up (RECOVERY_PLAN.md §6.1). The 15k-vs-30k schedule diff against
# splatfacto's resolved config found the one late-phase difference that is neither refuted
# nor already-landed: absolute opacity levels. splatfacto culls at 0.1 / resets to 0.2, so
# its frozen settle population contains nothing dimmer than 0.1; SplatRs prunes at 0.005 /
# floors at 0.01 (20x lower), leaving 90-93% of the settle population below 0.1 with the
# median parked on the floor — even under the needle-2.8 default (srt30k_sd_needle28:
# opacity_low_pct 93.1% at 30000). Mechanism: a large near-transparent cohort adds train-view
# memorization capacity but little test-view signal (the 30k run's overfit signature: train
# loss -33% while held-out PSNR falls). This arm moves both thresholds 5x toward splatfacto,
# keeping the same ~2x reset:prune ratio both systems use. (floor004 tried the OPPOSITE
# direction, lower and unpaired — different experiment.)
# Success: settle-mean-vs-window-peak gap vs the fresh control, and opacity_low_pct@30000
# visibly below 93%. Serial (single Metal GPU; concurrent trainers risk the watchdog).
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500)

STATUS=runs/settle_decay_hunt_30k_optfloor.status
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

run_arm srt30k_ctrl_needle28def                                            # fresh control (binary default needle 2.8)
run_arm srt30k_sd_optfloor05  --opacity-reset-floor 0.05 --prune-opacity-threshold 0.025

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
