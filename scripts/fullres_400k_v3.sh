#!/usr/bin/env bash
# Full-res @400k v3 (2026-07-15): v2 config + current defaults. v2 (15.01 final, settle
# flat 15.0, count 213k, zero watchdog trips) predates the settle needle prune; the ONLY
# behavioral delta here is the new --settle-needle-prune-log-aniso default 2.8 (e78d5c9),
# so v3-vs-v2 is a clean A/B of the 30k decay-hunt winner at full-res. C2 stays off
# (refuted at 30k). Gate rg2500 + sp500 + aniso 3.0 match v2 exactly. ~25h.
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train

STATUS=runs/fullres_400k_v3.status
echo "batch start $(date)" > "$STATUS"

name=srt15k_fullres_400k_v3
out="runs/$name"
echo "[$(date)] START $name" | tee -a "$STATUS"
rm -rf "$out"
"$BIN" --preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 15000 --densify-interval 100 --densify-max-gaussians 400000 \
  --seed 42 --max-log-aniso 3.0 --settle-prune-interval 500 \
  --opacity-reset-window-margin 2500 \
  --out-dir "$out" > "$out.log" 2>&1
rc=$?
final=$(tail -1 "$out/metrics.csv" 2>/dev/null | cut -d, -f1,3)
echo "[$(date)] DONE  $name  rc=$rc  (iter,psnr)=$final" | tee -a "$STATUS"
echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
