#!/usr/bin/env bash
# Discover every run under ns-runs/ and eval each one via eval_all_checkpoints.sh.
# Idempotent (already-evaluated runs are skipped), so run it after every campaign.
# Refuses to start while a training container is live.
# usage: eval_all_runs.sh
set -euo pipefail
shopt -s nullglob

BASE="$HOME/SplatRsBaseline"
BIN="$BASE/bin"

if docker ps --format '{{.Command}}' | grep -q ns-train; then
  echo "a training run is still live (docker ps) — eval after it finishes" >&2
  exit 1
fi

runs=( "$BASE"/ns-runs/*/splatfacto/*/ )
[ "${#runs[@]}" -gt 0 ] || { echo "no runs found under $BASE/ns-runs" >&2; exit 1; }

for d in "${runs[@]}"; do
  rel="${d#"$BASE/ns-runs/"}"
  "$BIN/eval_all_checkpoints.sh" "${rel%/}"
done
