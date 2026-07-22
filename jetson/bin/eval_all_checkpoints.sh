#!/usr/bin/env bash
# Walk every checkpoint of ONE completed run: per checkpoint, run ns-eval
# (metrics JSON + rendered eval views) and ns-export gaussian-splat (.ply).
#
# ns-eval always loads the LATEST checkpoint in nerfstudio_models/, so we stash
# all checkpoints and restore them in ascending order — the newly restored one is
# always "latest". Idempotent: skips the run if every checkpoint already has an
# eval JSON. Self-heals an interrupted walk (merges leftovers back into stash).
#
# The stash dir MUST live OUTSIDE nerfstudio_models/ (i.e. beside it, not in it):
# nerfstudio's eval_load_checkpoint calls int() on every entry of load_dir
# (== nerfstudio_models/), and a "stash" subdir slices to "stas" ("stash"[0:-1],
# no '-'/'.'), crashing ns-eval/ns-export with
# `ValueError: invalid literal for int() with base 10: 'stas'` (hit 2026-07-17).
#
# usage: eval_all_checkpoints.sh <run_rel>     e.g. ns-t1/splatfacto/20260717_091500
# Do not run against a run that is still training.
set -euo pipefail
shopt -s nullglob

REL="${1:?usage: eval_all_checkpoints.sh <run-id>/splatfacto/<stamp>}"
BASE="$HOME/SplatRsBaseline"
JCB="$BASE/bin/jcb"
HRUN="$BASE/ns-runs/$REL"        # host view
CRUN="/workspace/ns-runs/$REL"   # container view (same dir through the mount)
CKPTS="$HRUN/nerfstudio_models"
STASH="$HRUN/stash"              # sibling of nerfstudio_models — NEVER inside it (see header)

# ns-export writes splat.ply into its --output-dir; flatten each to a per-step name.
# MUST run only after ownership is reclaimed: the container writes the export dir as
# root, and moving a file OUT of a root-owned dir is EPERM for us — which is exactly
# why this is NOT done inside the eval loop (that raced the container's root writes).
flatten_exports() {
  local d step_dir
  for d in "$HRUN/exports"/iter_*/; do
    [ -d "$d" ] || continue
    step_dir="${d%/}"
    [ -f "$step_dir/splat.ply" ] && mv -f "$step_dir/splat.ply" "${step_dir}.ply" && rmdir "$step_dir"
  done
}

[ -d "$CKPTS" ] || { echo "no nerfstudio_models under $REL — skipping" >&2; exit 0; }

# Container wrote as root: take ownership up front so no mv fails mid-loop.
if [ ! -w "$CKPTS" ] || [ -n "$(find "$HRUN" -maxdepth 3 ! -user "$USER" -print -quit 2>/dev/null)" ]; then
  echo "taking ownership of $REL (container wrote as root)"
  sudo chown -R "$USER:$USER" "$HRUN"
fi

# Self-heal older/broken state: earlier versions stashed INSIDE nerfstudio_models
# (the 'stas' crash above). Move any such leftovers back out before we start.
if [ -d "$CKPTS/stash" ]; then
  old=( "$CKPTS/stash"/step-*.ckpt )
  [ "${#old[@]}" -gt 0 ] && mv "${old[@]}" "$CKPTS/"
  rmdir "$CKPTS/stash" 2>/dev/null || true
fi

# Self-heal a run that exported but couldn't flatten (the in-loop EPERM bug): the
# ownership reclaim above has made any leftover export dir user-owned, so flatten it
# now — this runs even when the idempotency check below decides to skip.
flatten_exports

mkdir -p "$STASH" "$HRUN/eval" "$HRUN/renders" "$HRUN/exports"

main=( "$CKPTS"/step-*.ckpt )
stashed=( "$STASH"/step-*.ckpt )
total=$(( ${#main[@]} + ${#stashed[@]} ))
done_evals=( "$HRUN/eval"/iter_*.json )

if [ "$total" -eq 0 ]; then
  echo "no checkpoints in $REL — skipping"; exit 0
fi
if [ "${#done_evals[@]}" -ge "$total" ]; then
  echo "$REL already evaluated (${#done_evals[@]}/$total) — skipping"; exit 0
fi

# Stash everything (merges with leftovers from an interrupted walk).
[ "${#main[@]}" -gt 0 ] && mv "${main[@]}" "$STASH/"

for ck in $(ls "$STASH"/step-*.ckpt | sort); do
  step=$(basename "$ck" .ckpt | sed 's/^step-0*//')
  mv "$ck" "$CKPTS/"
  echo "=== $REL — evaluating step $step ==="
  "$JCB" ns-eval --load-config "$CRUN/config.yml" \
                 --output-path  "$CRUN/eval/iter_${step}.json" \
                 --render-output-path "$CRUN/renders/iter_${step}"
  "$JCB" ns-export gaussian-splat \
                 --load-config "$CRUN/config.yml" \
                 --output-dir  "$CRUN/exports/iter_${step}"
done

rmdir "$STASH" 2>/dev/null || true
# Eval/render/export outputs were written by the container as root again.
sudo chown -R "$USER:$USER" "$HRUN"
flatten_exports   # exports are user-owned now: splat.ply -> iter_<step>.ply
echo "=== $REL done: $(ls "$HRUN/eval"/iter_*.json | wc -l | tr -d ' ') checkpoints evaluated"
