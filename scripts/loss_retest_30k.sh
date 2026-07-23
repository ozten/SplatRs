#!/usr/bin/env bash
# Loss re-test at 30k (2026-07-23) — the biggest still-unmatched recipe difference vs
# splatfacto after step 2. Splatfacto trains L1 + 0.2*(1-SSIM); every srt run to date
# trains L2 (micro preset). The old "L1Dssim loses" verdict (2026-07-06, 2k iters) was
# measured on the pre-backward-fix renderer and is stale — SSIM loss is a known 1-2 dB
# lever in reference 3DGS. One arm: the 5x-optfloor winner's exact config + --loss
# l1-dssim, same binary; control = runs/srt30k_sd_optfloor05 (17.21 final / 17.25
# settle mean, L2). Serial (single Metal GPU; watchdog).
set -u
cd /Users/ozten/Projects/SplatRs

BIN=./target/release/sugar-train
COMMON=(--preset micro --max-images 100 \
  --scene datasets/tandt_db/tandt/train/sparse/0 \
  --dataset-root datasets/tandt_db/tandt/train \
  --images datasets/tandt_db/tandt/train/images \
  --iters 30000 --densify-interval 100 --densify-max-gaussians 60000 \
  --seed 42 --downsample 0.5 --max-log-aniso 3.0 --settle-prune-interval 500 \
  --opacity-reset-floor 0.05 --prune-opacity-threshold 0.025)

STATUS=runs/loss_retest_30k.status
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

run_arm srt30k_optfloor05_l1dssim  --loss l1-dssim

echo "[$(date)] BATCH COMPLETE" | tee -a "$STATUS"
