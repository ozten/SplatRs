#!/usr/bin/env bash
# Overnight 2026-07-14: C2 SH warmup A/B + all-views on the new defaults.
# Defaults now include the 30k decay-hunt winner (--settle-needle-prune-log-aniso 2.8,
# commit e78d5c9); control for both arms' half of the story is runs/srt30k_sd_needle28
# (final 16.74 / settle mean 16.68).
#   - srt30k_c2: --sh-warmup-interval 1000 (reference oneupSHdegree; DC-only start,
#     +1 degree per 1000 iters). One lever different from the needle28 control.
#   - srt30k_allviews_nd28: --max-images 0 (all 301 images) on new defaults. NOT
#     comparable to 100-img numbers (76-view test denominator); read against
#     srt15k_a3_sp500_allviews_r2 (15.44 final @15k) and for bg/novel-view health.
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
COMMON=(--preset micro \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500)

STATUS=runs/overnight_20260714.status
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

run_arm srt30k_c2            --max-images 100 --sh-warmup-interval 1000
run_arm srt30k_allviews_nd28 --max-images 0

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
