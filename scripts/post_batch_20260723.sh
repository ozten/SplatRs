#!/usr/bin/env bash
# Batch-boundary work (2026-07-23): waits for the P1 dssim_settle_retune batch, then
# (1) runs the Stage-4 tile-vs-naive bench on the idle GPU (docs/TILE_RASTER_PLAN.md),
# (2) launches P1 arm 4: needle-prune-off under DSSIM. Rationale: arm-2 settle logs show
# needles=~1k/pass sustained with ZERO resets — the L2-tuned needle prune (2.8) fights
# DSSIM's functional anisotropy (edge-following splats; renders are crisp, no streaks),
# a cull-regrow treadmill draining 60k->28k with no densify to refill. Arm 4 = l1dssim
# baseline + --settle-needle-prune-log-aniso 0 (opacity/oversize prunes intact).
set -u
cd /Users/ozten/Projects/SplatRs

PREV=runs/dssim_settle_retune.status
for i in $(seq 480); do
  grep -q 'BATCH COMPLETE' "$PREV" 2>/dev/null && break
  [ "$i" -eq 1 ] && echo "waiting for dssim_settle_retune batch..."
  sleep 60
done

STATUS=runs/post_batch_20260723.status
echo "post-batch start $(date)" > "$STATUS"

echo "[$(date)] BENCH START" | tee -a "$STATUS"
./target/release/examples/bench_tile_raster > runs/bench_tile_raster_20260723.txt 2>&1
echo "[$(date)] BENCH DONE rc=$? (runs/bench_tile_raster_20260723.txt)" | tee -a "$STATUS"

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500 \
  --opacity-reset-floor 0.05 --prune-opacity-threshold 0.025 \
  --loss l1-dssim)

name=srt30k_dssim_noneedle
out="runs/$name"
echo "[$(date)] START $name  levers: --settle-needle-prune-log-aniso 0" | tee -a "$STATUS"
rm -rf "$out"
"$BIN" "${COMMON[@]}" --settle-needle-prune-log-aniso 0 --out-dir "$out" > "$out.log" 2>&1
rc=$?
final=$(tail -1 "$out/metrics.csv" 2>/dev/null | cut -d, -f1,3)
echo "[$(date)] DONE  $name  rc=$rc  (iter,psnr)=$final" | tee -a "$STATUS"
echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
