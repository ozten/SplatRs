#!/usr/bin/env bash
# Analyze the settle-decay hunt A/B batch. Reads each run's metrics.csv at VAL iters
# (multiples of 500 = full-test-set evals; the per-100-iter rows are noisy single-view logs).
# For each arm prints: peak psnr@iter, final psnr, peak-final gap (mechanism #2), settle-mean
# (iters > 7500), darkest bg-sum in settle (bg->black), final opacity_median / low%.
# Usage: bash scripts/settle_decay_analyze.sh [run1 run2 ...]  (defaults to the batch's 4 arms)
cd /Users/ozten/Projects/SplatRs
RUNS=("${@:-runs/srt15k_ctrl_banded runs/srt15k_sd_freezebg runs/srt15k_sd_sp0 runs/srt15k_sd_freezesh}")
# also show the pre-banding reference for cross-check
RUNS=(runs/srt15k_a3_sp500 ${RUNS[@]})

printf "%-26s %8s %10s %6s %8s %10s %9s %6s\n" arm final peak@it gap settleμ darkestBG opac_med low%
for r in ${RUNS[@]}; do
  csv="$r/metrics.csv"
  [ -f "$csv" ] || { printf "%-26s  (no metrics.csv)\n" "$(basename $r)"; continue; }
  awk -F, -v name="$(basename $r)" '
    NR>1 && $1%500==0 {
      it=$1; p=$3;
      if (p>peak){peak=p; peakit=it}
      final=p; finalit=it;
      # settle phase: iter > 7500 (densify window is first half of 15000)
      if (it>7500){ ssum+=p; sn++ }
      bgsum=$14+$15+$16;
      if (it>7500 && (dark=="" || bgsum<dark)) dark=bgsum;
      opac=$21; low=$22;
    }
    END{
      gap=peak-final;
      printf "%-26s %8.2f %6.2f@%-4d %+5.2f %8.2f %10.3f %9.3f %5.1f\n",
        name, final, peak, peakit, gap, (sn?ssum/sn:0), dark, opac, low;
    }' "$csv"
done
