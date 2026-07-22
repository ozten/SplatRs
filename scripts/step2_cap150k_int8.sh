#!/usr/bin/env bash
# Step 2 (RECOVERY_PLAN.md §6.2, 2026-07-22): the count-overlap run. SplatRs's curve tops out
# at ~60k gaussians while splatfacto's checkpoints start at ~119k — no shared count range, so
# the capacity term (~2.5-3.5 dB of the ~5.5 dB gap) has never been measured directly. This
# run raises the cap to 150k and, via the one-metric harness, becomes the first SplatRs run
# directly comparable to the hebot ns-eval grid:
#   --eval-interval 8    exact nerfstudio interval split (301 imgs -> 263 train / 38 test,
#                        the same 38 views splatfacto's eval JSONs score)
#   --max-test-views 0   full-test-set PSNR (not micro's 3-view subsample)
#   --save-interval 1500 model_<step>.gs grid mirroring splatfacto's checkpoint cadence
# Opacity thresholds are the 2026-07-22 A/B winners, passed explicitly so this runs on the
# same release binary as the dose-response arms (binary predates the defaults flip).
# Waits for the 10x dose-response arm to release the GPU (serial; Metal watchdog).
# Analysis when done: scripts/plot_count_vs_quality.py overlaying the splatfacto grid.
set -u
cd /Users/ozten/Projects/SplatRs

PREV=runs/settle_decay_hunt_30k_optfloor3.status
for i in $(seq 480); do
  grep -q 'BATCH COMPLETE' "$PREV" 2>/dev/null && break
  [ "$i" -eq 1 ] && echo "waiting for optfloor3 batch to release the GPU..."
  sleep 60
done

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 0 --eval-interval 8 --max-test-views 0 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 150000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500 \
  --opacity-reset-floor 0.05 --prune-opacity-threshold 0.025 \
  --save-interval 1500)

STATUS=runs/step2_cap150k_int8.status
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

run_arm srt30k_cap150k_int8

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
