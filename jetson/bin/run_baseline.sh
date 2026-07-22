#!/usr/bin/env bash
# Launch one baseline run by id. Names, log, and output path all derive from the
# id + a fresh timestamp: no collisions, no forgotten --experiment-name (that
# mistake cost a run, 2026-07-17). Runs in the FOREGROUND: wrap in tmux to
# detach; chain with && for sequential runs.
# usage: run_baseline.sh ns-t0 | ns-t1 | ns-t1-all | ns-house
set -euo pipefail

RUN_ID="${1:?usage: run_baseline.sh ns-t0|ns-t1|ns-t1-all|ns-house}"
BASE="$HOME/SplatRsBaseline"
JCB="$BASE/bin/jcb"
STAMP="$(date +%Y%m%d_%H%M%S)"        # unique per launch -> no output-dir collisions
DATA=/workspace/data/tandt_train
CPATH=sparse/0                        # tandt layout; ns-house uses the parser default
ITERS=30000 SAVE=1500
SPLIT="--eval-mode interval --eval-interval 8"

case "$RUN_ID" in
  ns-t0)     ITERS=3000 SAVE=500 ;;
  ns-t1)     ;;
  ns-t1-all) SPLIT="--eval-mode all" ;;
  ns-house)  DATA=/workspace/data/house_flight CPATH=colmap/sparse/0 ;;
  *) echo "unknown run id: $RUN_ID" >&2; exit 1 ;;
esac

# One run at a time: wait (up to 5 min) for any live ns-train container to exit,
# so sequential && chains don't overlap the previous run's shutdown.
# --no-trunc is load-bearing: jcb now prefixes the container command with
# `env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ...`, so `ns-train` sits past docker's
# default 20-char {{.Command}} truncation and the grep would never match (the guard
# would silently never wait, letting sequential runs overlap).
for i in $(seq 60); do
  docker ps --no-trunc --format '{{.Command}}' | grep -q ns-train || break
  [ "$i" -eq 1 ] && echo "waiting for previous run's container to exit..."
  sleep 5
done

LOG="$BASE/ns-runs/${RUN_ID}-${STAMP}.log"
echo "=== $RUN_ID  iters=$ITERS  out=ns-runs/$RUN_ID/splatfacto/$STAMP"
echo "=== log: $LOG"
echo "=== health check within 5 min: tegrastats -> GR3D_FREQ should be 90%+"

"$JCB" ns-train splatfacto \
  --output-dir /workspace/ns-runs \
  --experiment-name "$RUN_ID" \
  --timestamp "$STAMP" \
  --vis tensorboard \
  --machine.seed 42 \
  --max-num-iterations "$ITERS" \
  --steps-per-save "$SAVE" \
  --save-only-latest-checkpoint False \
  --steps-per-eval-all-images 1000 \
  --pipeline.model.camera-optimizer.mode off \
  --pipeline.model.rasterize-mode classic \
  --data "$DATA" \
  colmap \
    --colmap-path "$CPATH" \
    --images-path images \
    --downscale-factor 4 \
    --downscale-rounding-mode ceil \
    $SPLIT \
    --load-3D-points True \
  2>&1 | tee "$LOG"
