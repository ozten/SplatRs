# Nerfstudio (splatfacto) baseline on Jetson Orin AGX — apples-to-apples plan

**Goal:** run splatfacto on the Jetson Orin AGX 64GB with the *same inputs, same
train/test split, same resolution, and matched iteration checkpoints* as SplatRs, then
ship the artifacts back to the Mac under `baselines/` so a single comparison script can
answer: **how much of our 15-minute quality gap is quality-per-iteration (training
correctness/levers) vs throughput (rasterizer speed)?**

That decomposition decides where tier-0 auto-research effort goes: quality levers vs
landing tile-binned rasterization.

Conventions used below:

```bash
# On the Mac (this box)
export SPLATRS=/Users/ozten/Projects/SplatRs
export JETSON=hebot        # adjust user/host once, everything else copies
```

**Nerfstudio runs inside a jetson-container on the Orin, NOT a bare python env.**
Everything the container needs must live under one host tree — `~/SplatRsBaseline` —
mounted as `/workspace`. The wrapper is an **executable script, not a `.bashrc`
function** (same pattern as the existing teleop `jc()`, pointed at the baseline tree):
tmux sessions, the eval loop, and any cron/ssh one-liner run in non-interactive shells
that never source `.bashrc`, so a function silently doesn't exist there
(`jcb: command not found` — hit on the first tier-1 launch, 2026-07-15).

**All hebot-side scripts are versioned in this repo under `jetson/bin/`** — that is
the source of truth; never edit copies on the box. Deploy (rerun after any change):

```bash
# Mac → hebot; -a preserves the executable bits
rsync -avh "$SPLATRS/jetson/bin/" "$JETSON:SplatRsBaseline/bin/"
```

| Script | Purpose |
|---|---|
| `jcb` | container wrapper (mounts `~/SplatRsBaseline` as `/workspace`) |
| `run_baseline.sh <run-id>` | launch one training run, all names derived (Phase 4.1) |
| `eval_all_checkpoints.sh <run_rel>` | walk one run's checkpoints: eval + renders + PLY (Phase 4.2) |
| `eval_all_runs.sh` | discover + eval every completed run; idempotent (Phase 4.2) |
| `tb_to_csv.py <run_dir>` | tensorboard scalars → metrics.csv, runs in-container (Phase 4.3) |

One-time on hebot:

```bash
echo 'export BASE=~/SplatRsBaseline' >> ~/.bashrc
echo 'export PATH="$HOME/SplatRsBaseline/bin:$PATH"' >> ~/.bashrc
# interactive usage: jcb ns-train splatfacto ...
# inside tmux/scripts: ALWAYS the full path — $HOME/SplatRsBaseline/bin/jcb
```

(If a non-interactive invocation later fails with `autotag: command not found`, the
same non-interactive-shell issue one level down: edit `jetson/bin/jcb` to replace
`$(autotag nerfstudio)` with the resolved image tag recorded in
`baseline-env/container-tag.txt`, and redeploy.)

**Stale-alias trap (cost ~an hour, 2026-07-17):** a pre-existing `jcb` **alias/function**
(left over from the teleop `jc()` pattern) shadows the PATH `bin/jcb` in interactive
shells — aliases win over PATH lookup. Symptom: you edit + redeploy `bin/jcb`, but bare
`jcb` keeps running the old wrapper, so changes (e.g. a new env var) silently never take
effect, while `bin/jcb` / the full path works. After any `jcb` edit, confirm what bare
`jcb` resolves to: `type jcb` must print `.../SplatRsBaseline/bin/jcb`, not `aliased to`
or `is a function`. If it's stale, remove the old `alias jcb=...` / `jcb()` from
`~/.bashrc` (and `unalias jcb` in the live shell). Scripts are unaffected — they invoke
`$BASE/bin/jcb` by full path — but interactive verification and manual `ns-*` runs are.

Host ↔ container path map (host side left, what `ns-*` commands see right):

| Host (Jetson) | Container |
|---|---|
| `~/SplatRsBaseline/data/tandt_train` | `/workspace/data/tandt_train` |
| `~/SplatRsBaseline/ns-runs` | `/workspace/ns-runs` |
| `~/SplatRsBaseline/bin` (shims, scripts) | `/workspace/bin` |
| `~/SplatRsBaseline/baseline-env` | `/workspace/baseline-env` |

Three container consequences that shape everything below:
- **Anything written outside `/workspace` dies with the container** — output dirs,
  shims, logs all go under the mount.
- **Files the container writes are root-owned on the host** — there's a
  `sudo chown -R $USER:$USER ~/SplatRsBaseline` step after training, before the
  host-side eval loop touches checkpoint files.
- **The launching user must be in the docker group** — done on hebot (2026-07-17,
  `sudo usermod -aG docker ozten`), so launches need no sudo. On a fresh box this is
  the one-time setup, and the gotcha is that only sessions started AFTER the change
  have the group: fully log out and `tmux kill-server` (old tmux servers keep old
  groups forever), then verify `id -nG | grep docker` in the launching shell.
  Without it, jetson-containers falls back to sudo-ing docker and a detached tmux
  launch hangs silently on an invisible password prompt (burned a launch on this,
  2026-07-15). Note the `chown` steps below remain regardless — the container runs
  as root *inside* either way.

All `ns-train splatfacto`, colmap-dataparser, `ns-eval`, and `ns-export` flags below
are taken from the actual `-h` output on the Orin (captured 2026-07-15). The few
remaining **(verify)** marks are runtime behaviors the help text can't confirm
(eval-render filename convention, exporter output name, checkpoint numbering) — each is
checked in minutes on the tier-0 run before anything long depends on it.

## What is and is not comparable

- **Equal-iteration quality (PRIMARY).** PSNR/SSIM/LPIPS on the same test views at the
  same iteration counts. Hardware-independent; this is the honest measure of the
  quality-per-iteration gap.
- **Wall-clock across machines (INVALID).** Orin vs Mac Metal is not a fair perf
  comparison. Wall-clock is only compared *within* a box (each system vs itself across
  tiers), until SplatRs runs on the Jetson too.
- **Method internals are part of the method.** Splatfacto keeps its own default
  schedule — it is the reference algorithm, we don't force ours onto it. We pin only:
  inputs, split, resolution, iteration checkpoints, seed. Its defaults that differ from
  both reference-3DGS and SplatRs get *recorded* (see table below), not overridden.

### Splatfacto defaults worth knowing when reading results (from `-h`, record in manifest)

| Setting | Default | Interpretation caveat |
|---|---|---|
| `camera-optimizer.mode` | **off** | good — matches SplatRs (no pose optimization); still pin explicitly |
| `num-downscales` / `resolution-schedule` | 2 / 3000 | **coarse-to-fine: trains at 1/4 res until iter 3000, 1/2 until 6000, full after.** The tier-0 (3k) checkpoint has never seen full-res training — remember this when reading early-iteration comparisons; eval still renders at full eval res |
| `background-color` | random | vs SplatRs's learned bg constant — a real methodological difference, record it |
| `densify-grad-thresh` / `use-absgrad` | 0.0008 / True | absgrad is a gsplat improvement over reference 3DGS (which uses 0.0002, plain grad) |
| `warmup-length` / `refine-every` | 500 / 100 | densification cadence |
| `reset-alpha-every` | 30 (×refine-every = every 3000 steps) | opacity reset cadence |
| `stop-split-at` | 15000 | densify window end |
| `cull-alpha-thresh` | 0.1 | their prune threshold — note it equals our "opacity_low" bucket boundary |
| `sh-degree` / `sh-degree-interval` | 3 / 1000 | SH ramp |
| `rasterize-mode` | classic | keep classic: antialiased-mode PLY is incompatible with classic-mode viewers/renderers (incl. ours) |
| `max-gs-num` | 1,000,000 | their cap; ours is 400k (Metal 128 MB buffer) |
| `train-cameras-sampling-strategy` / seed | random / 42 | view sampling |
| `machine.seed` | 42 | happens to match our campaign seed; still pass explicitly |
| colmap parser pose normalization | `orientation-method up`, `center-method poses`, `auto-scale-poses True` | nerfstudio reorients/rescales the world to a ±1 box. Irrelevant for image metrics, but it means exported PLYs may be in nerfstudio's normalized frame, not COLMAP world — see Phase 6 |
| `load-3D-points` | True | SfM point init works out of the box (needed for splatfacto) |
| `downscale-rounding-mode` | **floor** | must be overridden to `ceil` (Phases 2.1/4.1) or the expected half-res size is 489×272 vs SplatRs's 490×273 |
| `downscale-factor` | None (auto, target <1600px) | **never rely on auto here** — it probes the 980×545 disk images, picks 1, and trips the camera-size assertion against the 1959×1090 model (Phase 2) |

## Status

Nerfstudio already runs on the Orin (`jetson/ns-viewer-safe` shim exists, a splatfacto
`splat.ply` export is in `jetson/`). This plan is about parity + artifact pipeline, not
installation.

## Phase 1 — Pin the environment (Jetson, ~30 min)

Host steps and container steps are interleaved — the comments say which is which.

```bash
# 1. HOST: lock clocks so wall-clock numbers are stable across runs
sudo nvpmodel -m 0            # MAXN
sudo jetson_clocks
mkdir -p "$BASE"/{data,ns-runs,bin,baseline-env}
sudo nvpmodel -q > "$BASE/baseline-env/power-mode.txt"

# 2. HOST: pin the container image. autotag resolves lazily and can change after a
#    pull — record the exact image + digest every result was produced with.
autotag nerfstudio                     > "$BASE/baseline-env/container-tag.txt"
docker inspect --format '{{index .RepoDigests 0}}' "$(autotag nerfstudio)" \
                                       >> "$BASE/baseline-env/container-tag.txt"
cat /etc/nv_tegra_release              > "$BASE/baseline-env/l4t-version.txt"

# 3. CONTAINER: snapshot the python stack
jcb bash -c "pip freeze | grep -Ei 'nerfstudio|gsplat|torch|numpy|opencv'" \
  > "$BASE/baseline-env/pip-versions.txt"

# 4. CONTAINER: save the help outputs for the record (all four already captured &
#    reflected in this doc, 2026-07-15)
jcb ns-train splatfacto -h        > "$BASE/baseline-env/splatfacto-help.txt" 2>&1
jcb ns-train splatfacto colmap -h > "$BASE/baseline-env/colmap-parser-help.txt" 2>&1
jcb ns-eval -h                    > "$BASE/baseline-env/ns-eval-help.txt" 2>&1
jcb ns-export -h                  > "$BASE/baseline-env/ns-export-help.txt" 2>&1

# 5. CONTAINER: smoke-test headless training end-to-end (200 iters; run after
#    Phase 2 lands the dataset). The downscale flags are NOT optional — see the
#    model-vs-disk resolution mismatch in Phase 2.
jcb ns-train splatfacto \
  --vis tensorboard \
  --output-dir /workspace/ns-runs \
  --max-num-iterations 200 \
  --data /workspace/data/tandt_train \
  colmap --colmap-path sparse/0 \
    --downscale-factor 4 --downscale-rounding-mode ceil
```

The `torch.load(weights_only=...)` incompatibility that required `jetson/ns-viewer-safe`
also hits `ns-eval` and `ns-export` (confirmed 2026-07-17): torch≥2.6 defaults
`torch.load` to `weights_only=True`, and nerfstudio's checkpoints pickle
`numpy._core.multiarray.scalar`, so both die with
`_pickle.UnpicklingError: Weights only load failed ... Unsupported global: ...scalar`.

**Fix (chosen): set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in the container — done once in
`jcb`, so it covers `ns-eval`, `ns-export`, and anything else run through the wrapper.**
torch honors it because nerfstudio's loader calls `torch.load(path,
map_location="cpu")` without passing `weights_only` (the override only applies when the
callsite left it unset). Truthy values: `1`/`y`/`yes`/`true`. Safe here — every
checkpoint is one we trained locally; the flag only disables the untrusted-pickle guard.

**Set it on the container COMMAND, not just as `docker -e`.** `jetson-containers run` is
not a thin `docker run` passthrough — a `-e` before the image can be consumed by its own
CLI and never reach the container (the `-e`-only attempt had no effect, 2026-07-17). `jcb`
now prefixes the command with `env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 …`, which rides in
as part of the command args (the part that demonstrably reaches the container), and keeps
the `-e` too as a second layer. Verify it lands with:
`jcb python3 -c "import os;print(os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'))"`
→ must print `1`.

This replaces the per-entrypoint monkeypatch shim below, which is kept only as a fallback
(e.g. if a future container ships a torch that ignores the env var):

```bash
# FALLBACK ONLY — $BASE/bin/ns-eval-safe  (chmod +x; same pattern for ns-export-safe
# with nerfstudio.scripts.exporter — module paths (verify) against installed version)
#!/usr/bin/env python3
import torch
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, 'weights_only': False})
from nerfstudio.scripts.eval import entrypoint
entrypoint()
```

Invoke the fallback as `jcb /workspace/bin/ns-eval-safe --load-config ...`.

## Phase 2 — Dataset parity (Mac → Jetson)

Primary scene: `datasets/tandt_db/tandt/train` — 301 images @ **980×545**, COLMAP model
at `sparse/0` (bin format). This is the scene with all SplatRs campaign history.
Secondary (after tandt works end-to-end): house-flight capture
(`202060713-house-flight/trimmed.mov` → frames → COLMAP).

**Critical discovered fact (2026-07-15): the sparse model is 2× the images on disk.**
`cameras.bin` declares **1959×1090** cameras while `images/` holds **980×545** files —
the dataset ships pre-downscaled images against a full-res reconstruction. SplatRs
silently rescales intrinsics to the actual image size; nerfstudio asserts
(`The size of image (980, 545) ... does not match the camera parameters ((1959, 1090))`).
Consequently all `images_N` directory names and `--downscale-factor` values are
**relative to the MODEL resolution (1959×1090)**, not the disk images:

| Directory | Resolution | Meaning |
|---|---|---|
| `images/` | 980×545 | model/2 — SplatRs "full-res" |
| `images_2/` | 980×545 | **symlink → `images/`** (so `--downscale-factor 2` works) |
| `images_4/` | 490×273 | ceil(model/4) — the half-res tier, generated in 2.1 |

Set up the links once on **both** machines (`rsync -a` carries the relative symlink):

```bash
# in the dataset root (Mac: $SPLATRS/datasets/tandt_db/tandt/train, Jetson: $BASE/data/tandt_train)
ln -s images images_2
```

And **never rely on auto downscale** (`--downscale-factor` unset): it probes the actual
image size (980 < 1600px → picks factor 1) and then trips the camera-size assertion.
Every run passes an explicit factor: **4 for the half-res tier, 2 for 980×545 runs**,
always with `--downscale-rounding-mode ceil` (1959/4 = 489.75 → 490, 1090/4 = 272.5 →
273 — exact match to our generated set only under ceil).

### 2.1 Pre-generate the half-res tier image set (Mac) — do NOT let nerfstudio auto-downscale

Nerfstudio's colmap parser expects an `images_N/` sibling directory for
`--downscale-factor N`; if it's missing, it offers to generate it **interactively**
(kills headless container runs) — and it would generate it from the mismatched-res base
images anyway. Pre-generate `images_4/` (= 490×273, the half-res tier) on the **Mac,
before pushing** — one canonical image set, covered by the checksum manifest in 2.3:

```bash
cd "$SPLATRS/datasets/tandt_db/tandt/train"
mkdir -p images_4
for f in images/*.jpg; do
  ffmpeg -y -loglevel error -i "$f" -q:v 2 \
    -vf "scale=ceil(iw/2):ceil(ih/2)" "images_4/$(basename "$f")"
done
# Verify: must print 490x273
ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
  -of csv=s=x:p=0 images_4/00001.jpg
ln -sfn images images_2   # the model/2 alias from the table above
```

(The ffmpeg `ceil(iw/2)` halves the 980×545 disk images; the result coincides exactly
with ceil(model/4) = 490×273, which is why `--downscale-rounding-mode ceil` at train
time makes the parser's expected size agree with these files. floor would expect
489×272 and mismatch.)

### 2.2 Push the dataset

```bash
# Mac → Jetson. Trailing slash on src = copy CONTENTS of train/ into tandt_train/.
# Destination is inside the container mount tree. Leave macOS junk behind.
rsync -avh --progress --partial --exclude '.DS_Store' \
  "$SPLATRS/datasets/tandt_db/tandt/train/" \
  "$JETSON:SplatRsBaseline/data/tandt_train/"
```

### 2.3 Verify integrity (both ends must produce identical files)

`LC_ALL=C` on every `sort` is load-bearing: macOS's UTF-8 locale collates
`images_2/` before `images/` while Linux's C locale doesn't, so unpinned sorts
produce identical content in different line orders — a wall of diff output with
matching hashes (learned the hard way, 2026-07-15).

```bash
# Mac (BSD userland: shasum, not sha256sum)
cd "$SPLATRS/datasets/tandt_db/tandt/train" && \
  find images images_4 sparse -type f ! -name '.DS_Store' \
  | LC_ALL=C sort | xargs shasum -a 256 > /tmp/tandt_train.sha256

# Jetson (host side is fine — same bytes the container sees through the mount)
cd "$BASE/data/tandt_train" && \
  find images images_4 sparse -type f ! -name '.DS_Store' \
  | LC_ALL=C sort | xargs sha256sum > /tmp/tandt_train.sha256

# Compare (Mac):
scp "$JETSON:/tmp/tandt_train.sha256" /tmp/tandt_train.jetson.sha256
diff /tmp/tandt_train.sha256 /tmp/tandt_train.jetson.sha256 && echo "DATASET OK"
```

If a diff ever shows mismatched *hashes* for the same path, that's real corruption —
re-rsync. Mismatched *line order* with equal hashes means a locale leaked into a sort.

**GT pixel check (before trusting any cross-system PSNR delta):** SplatRs downsamples
internally from the full-res JPEGs with its own filter. Render nothing — just compare
the two systems' half-res GT for one image (both now exist on the Mac: the `images_4/`
file from 2.1, and SplatRs's downsampled buffer dumped for the same image; diff pixels).
- Same size + small diff (JPEG-decode/filter noise, maxdiff ≲2/255): fine, proceed.
- Size mismatch or big diff: fall back to **staging one shared half-res set as the
  source of truth** — use `images_4/` as *the* images for both systems (SplatRs:
  a run with `--downsample 1` pointed at a dataset whose `images/` is the pre-scaled
  set and whose COLMAP intrinsics are halved — needs a small intrinsics-scaling step,
  verify SplatRs's loader semantics before doing this).

## Phase 3 — Split parity (the non-obvious one)

SplatRs's split is a **seeded `StdRng` shuffle + `train_fraction`**
(`src/optim/trainer.rs:1479-1484`). Nerfstudio's colmap parser eval modes are
`{fraction, filename, interval, all}` (confirmed from `-h`) — `filename` mode "splits
based on filenames containing train/eval", i.e. substrings in the image names, not
external list files, so it cannot reproduce our shuffle without renaming images (which
would break the COLMAP model references).

**Resolution: make SplatRs match nerfstudio, not vice versa — IMPLEMENTED
(2026-07-15).** Nerfstudio's *default* is already `--eval-mode interval
--eval-interval 8` — every 8th frame for eval, which its help text notes is "used by
most academic papers, e.g. MipNerf360, GSplat". SplatRs now has the matching
`--eval-interval N` flag (`MultiViewTrainConfig::eval_interval`, 0 = legacy seeded
shuffle): images sorted by filename, position % N == 0 → test view (so the
lexicographically first image is held out, matching nerfstudio's convention). The sort
matters because COLMAP's `images.bin` is in registration order, not filename order —
`interval_split_by_name` in `trainer.rs` handles it and is unit-tested (301 images →
263/38, first test = lexicographic first). Verified on the real scene:

```
Split mode: interval:8 over filename order (38 test views, first: ["00001.jpg", "00009.jpg", "00017.jpg"])
Multi-view training: 263 train views, 38 test views
```

- nerfstudio side: nothing — the default IS the target split (we still pass the flags
  explicitly in Phase 4.1 so the manifest is self-documenting)
- SplatRs side: `--eval-interval 8`
- Startup-log check when the first nerfstudio run happens: counts must be 263/38 and
  the first eval filenames must match the banner above (pins that nerfstudio's
  index-0-is-eval convention agrees; if its counts differ, its split indexing changed —
  adjust `interval_split_by_name` to match, it's the single source of truth).

Fallback if we must preserve the historical seed-42/100-img split instead: add
`--dump-split` to SplatRs, then patch a 5-line custom eval-mode into the nerfstudio
colmap parser on the Jetson that reads a `test_list.txt`. More fragile; only do this if
interval-split runs turn out not to reproduce the pathologies we care about.

Note the all-views arm: nerfstudio has `--eval-mode all` (verify semantics — train on
all, eval on all); SplatRs's all-views config is the analog. Exact-split parity matters
less there since both see every view.

## Phase 4 — Run matrix (Jetson)

"Res: half" = 490×273 = `--downscale-factor 4` of the 1959×1090 colmap model (see the
Phase 2 resolution table; factor 2 = 980×545 = SplatRs "full-res").

| Run id | Scene | Res | Split | Iters | steps-per-save | Purpose |
|---|---|---|---|---|---|---|
| ns-t0 | tandt/train | half (490×273, df 4) | interval:8 | 3,000 | 500 | tier-0 anchor (≈15-min SplatRs tier); checkpoints 500…3000 |
| ns-t1 | tandt/train | half | interval:8 | 30,000 | 1500 | tier-1 anchor; checkpoints incl. 3000/7500/15000/30000 |
| ns-t1-all | tandt/train | half | all | 30,000 | 1500 | all-views anchor |
| ns-house | house-flight | half | interval:8 | 30,000 | 1500 | own-capture anchor (after tandt works) |

`--steps-per-save` is a single interval, so the 500/1k/2k/3k early grid comes from the
tier-0 run and the 1.5k-multiples grid (1500/3000/…/7500/…/15000/…/30000) from tier-1;
together they cover the checkpoint grid. Checkpoint filenames are `step-%09d.ckpt` and
step numbers are 0-indexed (a 3000-iter run's last checkpoint is `step-000002999.ckpt`)
**(verify the off-by-one on the box)**.

### 4.1 The launcher script (one argument per run — never hand-edit the CLI)

tmux runs on the **host**, wrapping the script; the `tee` is host-side (so the log is
user-owned even though the run outputs are root-owned). All paths the `ns-*` command
sees are container paths, and `jcb` is used by full path — tmux commands run in
non-interactive shells where neither `.bashrc` functions nor its PATH additions exist.

All launches go through **one launcher script** — **`jetson/bin/run_baseline.sh`**
(deployed to `$BASE/bin/`, see Conventions) — that takes a run id and derives the
experiment name, timestamp, output dir, and log name from it. Never launch by
hand-editing the long CLI: a reused `--experiment-name` + `--timestamp` writes INTO
the previous run's directory and a reused `tee` filename truncates the previous log
(both happened 2026-07-17). The script encodes the full pinned `ns-train` command
(the notes below explain every pinned flag), the per-run deltas in its `case` table,
and a one-run-at-a-time guard that waits for any live `ns-train` container before
starting — which is what makes `&&` chains safe.

Launching (host, any shell — the script uses full paths internally):

```bash
# single run, detached:
tmux new-session -d -s baseline "$HOME/SplatRsBaseline/bin/run_baseline.sh ns-t0"

# the tandt campaign, sequentially (ONLY after ns-t0 has passed end-to-end once —
# the first-ever t0 run is validated by a human before anything longer launches):
tmux new-session -d -s baseline "
  B=\$HOME/SplatRsBaseline/bin
  \$B/run_baseline.sh ns-t0 && \$B/run_baseline.sh ns-t1 && \$B/run_baseline.sh ns-t1-all
"
```

`set -euo pipefail` + the `&&` chain means a failed run stops the sequence instead of
burning hours on the next one atop a broken state.

Notes:
- **`--colmap-path sparse/0` is mandatory** — the parser default is `colmap/sparse/0`,
  our layout is `sparse/0` directly. `--downscale-factor 4` must also be explicit
  (confirmed by hitting it, 2026-07-15): the default (None) probes the actual base
  images (980×545, max dim <1600px) and picks factor 1 — which then trips the
  camera-size assertion against the 1959×1090 model.
- `--downscale-rounding-mode ceil` (default floor) keeps the parser's expected size
  (ceil(1959/4)×ceil(1090/4) = 490×273) in agreement with the pre-generated
  `images_4/`, matching SplatRs half-res.
- `--eval-mode interval --eval-interval 8` and `--load-3D-points True` are the
  defaults, passed explicitly so the command line is the complete record.
- `--vis tensorboard` (not the default `viewer`) so the process exits on completion and
  scalars land in TB event files we can parse.
- `--experiment-name` + `--timestamp` pin the output path:
  `/workspace/ns-runs/<run-id>/splatfacto/<stamp>/{config.yml, nerfstudio_models/, ...}`
  — the launcher echoes the exact path at launch; pass that
  `<run-id>/splatfacto/<stamp>` to the eval loop (4.2). `config.yml` embeds
  `/workspace/...` absolute paths, which is exactly why `ns-eval`/`ns-export` must run
  in the container with the same mount — the paths resolve there and nowhere else.
- `--save-only-latest-checkpoint False` is what makes the per-checkpoint eval loop
  possible (default True would leave only the final one).
- Deliberately NOT overridden (recorded instead): background-color, densify thresholds,
  resolution-schedule, absgrad — see the defaults table; they're the method.
- Log GPU state alongside (HOST — tegrastats isn't in the container):
  `tmux new-session -d -s tegra "tegrastats --interval 5000 | ts '%s' > $BASE/ns-runs/<run-id>-<stamp>.tegrastats"`
  (match the launcher's echoed log name).
- **After training completes, reclaim ownership before the eval loop** (the container
  writes as root): `sudo chown -R $USER:$USER $BASE/ns-runs`.
- **Health check within 5 minutes of every launch — do not walk away without it.**
  A healthy splatfacto run shows in tegrastats: `GR3D_FREQ` mostly 90%+ with the GPU
  clock boosted, and progress/ETA lines advancing in the log. `GR3D_FREQ 0%` with one
  CPU core pinned at 100% means the run is broken, not slow — kill and diagnose
  (first tier-0 launch burned 2 days this way, 2026-07-17). Diagnosis order:
  `tail` the log; then
  `jcb python3 -c "import torch; print(torch.cuda.is_available())"` (False = container
  has no GPU access); then `sudo nvpmodel -q` + `jetson_clocks` (clock parked at
  611 MHz = Phase 1 step 1 never ran). Note jetson-containers already passes
  `--shm-size=8g` (observed 2026-07-17), so the classic docker DataLoader
  shared-memory hang is unlikely; if the log stops at image loading anyway, try
  `--pipeline.datamanager.dataloader-num-workers 0`.
- **`permission denied ... docker.sock` at launch:** the shell lacks the docker group
  — a session started before the group grant, or an old tmux server spawning it (hit
  2026-07-17). `id -nG | grep docker` in the launching shell must succeed; fresh ssh
  login + `tmux kill-server` fixes it (see Conventions).
- **Killing a run: tmux alone is not enough.** The container outlives its client —
  after `tmux kill-session -t <s>`, run `docker ps` and `docker stop <id>` or the run
  keeps burning CPU/GPU as a zombie (the first dead tier-0 launch survived 39 h past
  its tmux session, 2026-07-17). **Order matters:** `docker logs --tail 80 <id>`
  BEFORE `docker stop` — jetson-containers runs with `--rm`, so stopping deletes the
  container and its logs, destroying the evidence of why it died/hung.

### 4.1a The four runs, in execution order

All four launch as `run_baseline.sh <run-id>` — the script's `case` table IS the
delta table; the paragraphs below document what each delta means. Run them
**sequentially, never in parallel** — one GPU, and concurrent runs pollute each
other's wall-clock numbers (which are half the point). Runtime estimates are
provisional until run 1 lands; the reliable source is the **ETA nerfstudio prints in
the log** a few minutes into each run.

**Run 1 — `ns-t0`, tier-0 anchor + pipeline shakedown. RUN THIS FIRST, ALONE.**
Deltas:

`--experiment-name ns-t0 --max-num-iterations 3000 --steps-per-save 500`

(and the tee/log filename). Rough estimate: **5–15 min** of training — splatfacto's
coarse-to-fine schedule means all 3,000 iterations run at reduced internal resolution —
plus a few minutes to walk its 7 checkpoints with `eval_all_checkpoints.sh` (4.2).
This run is the designated proving ground for every remaining **(verify)** item:
checkpoint numbering, the stash-loop behavior, ns-eval render naming, the PLY export
name, TB scalar tags, and the split parity check against SplatRs's banner (263/38,
first test `00001.jpg`). Nothing longer launches until this passes end-to-end.

**Run 2 — `ns-t1`, tier-1 anchor.** Deltas:

none — the launcher's defaults ARE this run (`run_baseline.sh ns-t1`). Its defining
values, for comparison against the other runs' deltas:

`iters=30000 save=1500, interval:8 split, tandt`

Rough estimate: **1–2.5 h**. Do NOT extrapolate from run 1's wall-clock (its
iterations were mostly at ¼ internal resolution; this run spends 24k iterations at
full 490×273) — trust the in-log ETA once it passes iter ~6000, and record the final
wall-clock + steady sec/iter into the manifest.

**Run 3 — `ns-t1-all`, all-views anchor.** Deltas:

`--experiment-name ns-t1-all`,
and in the colmap block replace

`--eval-mode interval --eval-interval 8` with `--eval-mode all`

(confirmed semantics: "uses all the images for any split" — its
eval set overlaps train, so held-out metrics for this arm come from our shared script
on true held-out cameras, not ns-eval). Rough estimate: **same as run 2, slightly
longer** (301 train views cached vs 263; per-iteration cost unchanged).

**Run 4 — `ns-house`, own-capture anchor. BLOCKED until the house-flight scene has a
COLMAP reconstruction.**

*Why this run exists:* runs 1–3 stay on tandt (the scene with all campaign history and
published reference numbers — the yardstick for "is the algorithm right"). Run 4 adds
the capture we actually care about, which stresses everything the curated benchmark
doesn't: video frames (motion blur, compression), auto-exposure drift, orbit-style
coverage, self-built COLMAP poses, sky/textureless regions. Splatfacto's result on our
own footage is the **achievability reference** — when SplatRs renders it badly, this
run splits "our trainer is behind" from "this capture is inherently hard." It's also
the candidate sequestered-holdout scene for the auto-research loop.

The tandt scene shipped with `images/` + `sparse/0` (COLMAP's
per-image camera poses + SfM point cloud) — both trainers require that as input: poses
say where each photo was taken, the points seed the initial Gaussians. The
house-flight capture is currently just a video (`202060713-house-flight/trimmed.mov`
on the Mac — probed 2026-07-17: **3840×2160, 30 fps, 5,324 frames, 177 s, 1.1 GB**),
so the pipeline to unblock this run is: extract frames → run COLMAP → the tandt-style
scene layout. The video itself is never transferred (1.1 GB); ~300 extracted 1080p
JPEGs (~150 MB) are.

**Step 1 — extract frames (Mac; ffmpeg is local).** Every 18th frame → ~296 frames
(≈ tandt's 301, so the interval:8 split and run configs transfer; every-40th would
give only 133 — too thin for a 3-min flight). Extract at **1920×1080**: enough
features for COLMAP at a fraction of 4K's cost, and the tiers divide EXACTLY —
`images_2` = 960×540 (≈ tandt full-res tier), `images_4` = 480×270 (≈ tandt half-res
tier), no rounding-mode drama. (`NORM0002.LRV` is the camera's low-res proxy —
ignore.)

```bash
cd $SPLATRS
mkdir -p datasets/house_flight_frames
ffmpeg -i 202060713-house-flight/trimmed.mov \
  -vf "select='not(mod(n,18))',scale=1920:1080" -vsync vfr -q:v 2 \
  datasets/house_flight_frames/%05d.jpg
ls datasets/house_flight_frames | wc -l    # expect ~296
rsync -avh --progress --partial datasets/house_flight_frames/ \
  "$JETSON:SplatRsBaseline/data/house_flight_frames/"
```

**Step 2 — COLMAP (hebot, in the container; AFTER run 3 — COLMAP contends for the
GPU and would pollute a timed run's wall-clock).**

```bash
jcb ns-process-data images \
  --data /workspace/data/house_flight_frames \
  --output-dir /workspace/data/house_flight \
  --matching-method sequential   # frames are video-ordered (verify flag on box)
```

Output: `images/` + auto-generated `images_2..8/`, model at `colmap/sparse/0`
(the parser's DEFAULT path — run 4 needs no `--colmap-path` override, unlike tandt),
`transforms.json`. Model res = image res here, so no model-vs-disk mismatch.
**Check the registration summary it prints:** if far fewer than ~296 images
register, the capture is fighting COLMAP (sky/blur) — re-extract denser (step 12)
before debugging anything else. Rough COLMAP cost: tens of minutes on the Orin GPU.

**Step 3 — ship the scene back to the Mac** (canonical dataset home; SplatRs trains
the same scene for the comparison):

```bash
rsync -avh --progress "$JETSON:SplatRsBaseline/data/house_flight/" \
  "$SPLATRS/datasets/house_flight/"
```

**ATTEMPT 1 FAILED — capture unusable, recapture required (2026-07-17).** The
pipeline above ran cleanly end-to-end (frames → features → matching → bundle adjust),
but COLMAP registered **2 of 295 images (0.68%)**. Frame inspection pinned the cause:
the imagery is sharp and well-exposed, but the flight is a **vertical descent with a
straight-down (nadir) camera** — high over the deck → low over the lawn → ~1 m grass
macro. Consecutive views differ mostly by scale, the flight's ends share no field of
view, and grass/deck textures are self-similar so matches fail geometric
verification. This is a flight-profile problem, not a COLMAP-tuning problem — denser
sampling could at best reconstruct a nadir strip of lawn, which is not the scene this
run exists for. Do NOT train on a 2-pose model ("just to see" shows nothing: two
views can't triangulate; splatfacto would trivially overfit them).

**Recapture recipe (~2 min of flying):** orbit the house with the gimbal at
**30–45° down**, house centered in frame, **two altitude rings** (e.g. ~10 m and
~20 m), slow and smooth — every frame sees the subject, every pair overlaps,
viewpoints diversify. DJI Orbit/POI mode does this automatically. Avoid pure-nadir
segments except an optional single top-down pass for the roof. Then rerun the
pipeline above unchanged — expect ≳90% registration from an orbit.

(The 424 frames in `jetson/dronies_video_frames/` are 160×90 thumbnails — trajectory
documentation only, not trainable data. Their up-and-back "dronie" profile with an
oblique camera IS SfM-friendly; if that clip's full-res source video still exists on
the SD card, it's a usable test article.)

Deltas once unblocked:

`--data /workspace/data/house_flight`,
`--experiment-name ns-house`

Rough estimate: **similar order to run 2**, scaled by that capture's image count and
resolution. Don't start until runs 1–3 have produced a working comparison on tandt.

Total for the tandt campaign (runs 1–3 + eval walks): roughly **an afternoon or
evening**, dominated by runs 2–3.

### 4.2 Per-checkpoint eval + render + PLY export

`ns-eval` takes only `--load-config` and always loads the **latest** checkpoint in the
run's `nerfstudio_models/`. To walk every checkpoint: stash them all, restore in
ascending order, eval after each restore (the newly restored one is always "latest").

**The stash dir must be a SIBLING of `nerfstudio_models/`, never inside it (confirmed
2026-07-17).** nerfstudio's `eval_load_checkpoint` runs `int(x[x.find("-")+1:x.find(".")])`
over *every* entry of `load_dir` (= `nerfstudio_models/`) — files and subdirs alike, no
filtering. A `stash/` subdir slices to `"stash"[0:-1]` = `"stas"` (no `-`/`.`) and crashes
eval **and** export with `ValueError: invalid literal for int() with base 10: 'stas'`.
`eval_all_checkpoints.sh` now stashes to `<run>/stash` (beside `nerfstudio_models/`) and
self-heals any older inside-the-load-dir stash on start.

The script — **`jetson/bin/eval_all_checkpoints.sh`** (deployed to `$BASE/bin/`) —
runs on the **host** (file moves on the host tree) and shells into the container per
checkpoint, juggling both path forms: `$BASE/ns-runs/...` for `mv`/`mkdir`,
`/workspace/ns-runs/...` for the `ns-*` invocations. It handles the root-ownership
problem itself (chowns up front and again at the end — the two sudo prompts; run it
attached), is **idempotent** (skips a run whose every checkpoint already has an eval
JSON), and self-heals an interrupted walk.

Usage — one run (the launcher echoed the `<run-id>/splatfacto/<stamp>` path):

```bash
$HOME/SplatRsBaseline/bin/eval_all_checkpoints.sh ns-t1/splatfacto/20260717_091500
```

Usage — **all runs** (after a campaign): **`jetson/bin/eval_all_runs.sh`** discovers
every run under `ns-runs/`, walks each through `eval_all_checkpoints.sh`, skips
already-evaluated ones, and refuses to start while a training container is live:

```bash
$HOME/SplatRsBaseline/bin/eval_all_runs.sh
```

Note each loop iteration pays container startup (~seconds with the image cached) —
negligible against eval time. The weights_only-load failure is handled globally by
`jcb`'s `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (Phase 1), so the script calls plain
`ns-eval`/`ns-export`; only fall back to the `-safe` shims if that env var stops working.

- `--render-output-path` makes `ns-eval` dump the rendered eval images — those PNG
  pairs are the primary artifact the Mac-side shared metric script consumes. **(verify)**
  the naming convention it uses maps back to source image filenames; if it's bare
  indices, also export the camera list once (`ns-export cameras --load-config ...`) so
  index↔filename mapping is recorded.
- `ns-export gaussian-splat` writes its output as `splat.ply` inside `--output-dir`
  (**confirmed 2026-07-19**). The eval loop renames each to `exports/iter_<step>.ply`,
  but that rename **must happen after** the end-of-loop `chown` — the container writes
  the export dir as root, and moving a file out of a root-owned dir is EPERM. Doing it
  in-loop failed with `mv: ... Permission denied`, and because `set -e` doesn't fire on
  the left of `&&`, the walk marked the run "done" and the idempotency guard then skipped
  it forever. Fixed: `eval_all_checkpoints.sh` flattens exports in a `flatten_exports`
  pass after the chown, and again at startup (self-heals any run left in the broken
  `iter_<step>/splat.ply` state).
- Harmless noise during export: `Cannot load library ... libio_e57.so (libE57Format.so
  ...)` — pymeshlab's E57 plugin failing to load. We export `.ply`, not E57; ignore it.

### 4.3 Extract per-iteration metrics from tensorboard events

Tensorboard's python API lives in the container, so **`jetson/bin/tb_to_csv.py`**
(deployed into the mount at `$BASE/bin/` = `/workspace/bin/`) runs in-container, once
per run:

```bash
jcb python3 /workspace/bin/tb_to_csv.py /workspace/ns-runs/<run-id>/splatfacto/<stamp>
```

It writes `<run_dir>/metrics.csv` and prints every scalar tag it found. Expected tags
**(verify against the printout — nerfstudio's naming varies by version)**:
`Eval Images Metrics Dict (all images)/psnr` / `.../ssim` / `.../lpips`,
`Train Metrics Dict/gaussian_count`, `Train Loss`, timing tags.

## Phase 5 — Ship artifacts to the Mac

Layout next to `runs/` on this box (add `baselines/` to `.gitignore` like `runs/`;
commit only manifests + metrics.csv if we want numbers in-repo):

```
baselines/nerfstudio/<run-id>/           # e.g. ns-t1_tandt_half_int8_20260716
  manifest.json
  config.yml           # splatfacto's resolved config — IS the schedule documentation
  dataparser_transforms.json  # nerfstudio's world reorientation/rescale (see Phase 6)
  metrics.csv          # from tb_to_csv.py
  eval/iter_*.json     # ns-eval outputs
  renders/iter_*/      # rendered eval views (PNG pairs)
  exports/iter_*.ply   # gaussian-splat exports per checkpoint
  ns-t1-<stamp>.log
  ns-t1-<stamp>.tegrastats
```

Pull (Mac) — everything except the multi-GB torch checkpoints:

```bash
STAMP=20260717_091500                      # from the launcher's echo / dir listing
RUNID=ns-t1_tandt_half_int8_${STAMP}
mkdir -p "$SPLATRS/baselines/nerfstudio/$RUNID"
rsync -avh --progress --partial --prune-empty-dirs \
  --exclude 'nerfstudio_models/' --exclude '*.ckpt' --exclude 'stash/' \
  "$JETSON:SplatRsBaseline/ns-runs/ns-t1/splatfacto/$STAMP/" \
  "$SPLATRS/baselines/nerfstudio/$RUNID/"

# The host-side tee'd log + tegrastats live one level up:
rsync -avh "$JETSON:SplatRsBaseline/ns-runs/ns-t1-$STAMP."{log,tegrastats} \
  "$SPLATRS/baselines/nerfstudio/$RUNID/"

# Also grab the environment snapshot once (incl. container-tag.txt):
rsync -avh "$JETSON:SplatRsBaseline/baseline-env/" \
  "$SPLATRS/baselines/nerfstudio/jetson-env/"
```

`manifest.json` (write by hand or script per run):

```json
{
  "run_id": "ns-t1_tandt_half_int8_20260716",
  "system": "nerfstudio-splatfacto",
  "container": "see ../jetson-env/container-tag.txt (autotag-resolved image + digest)",
  "versions_file": "../jetson-env/pip-versions.txt",
  "hardware": "Jetson Orin AGX 64GB, MAXN, jetson_clocks",
  "dataset": "tandt_db/tandt/train",
  "dataset_sha256_manifest": "tandt_train.sha256",
  "resolution": "490x273 (downscale 4 of the 1959x1090 colmap model, ceil, pre-generated images_4)",
  "split": "interval:8 (263 train / 38 test)",
  "iterations": 30000,
  "checkpoints": [1500, 3000, 4500, 7500, 15000, 30000],
  "seed": 42,
  "pinned": {"camera_optimizer": "off", "rasterize_mode": "classic"},
  "recorded_defaults": "see config.yml (coarse-to-fine res schedule, random bg, absgrad 0.0008)",
  "wall_clock_sec": null,
  "sec_per_iter_steady": null
}
```

## Phase 6 — Comparison harness (Mac)

One script (`scripts/compare_baseline.py`), inputs = a SplatRs run dir + a baseline dir:

1. Renders the SplatRs model at the same test cameras / resolution (existing
   `sugar-render` path) — SplatRs must have been trained with the matching
   `--eval-interval 8`.
2. Computes PSNR/SSIM/LPIPS with ONE implementation on both systems' PNG pairs
   (extend `scripts/compute_lpips.py`) — never mix metric implementations across
   systems; `ns-eval`'s JSON numbers are recorded for cross-checking only.
3. Emits the equal-iteration table (both systems at 500/1k/2k/3k/…), side-by-side
   image grids per checkpoint, and Gaussian-count-vs-quality curves (count from the
   `.ply` exports vs our metrics.csv `num_gaussians`).
4. Prints the decomposition verdict:
   - `gap@equal-iters` (dB at 3k, 15k, 30k) → quality-per-iteration gap
   - `iters@equal-wallclock` ratio (each on its own box, reported separately) →
     throughput gap

Bonus same-renderer check: `exports/iter_*.ply` can be rendered in SplatRs's own
renderer (`sugar-render`) to remove renderer differences from the comparison — valid
only while the export's count ≤ 400k (Metal 128 MB buffer cap), so expect tier-0 /
early-tier-1 checkpoints only. Splatfacto's cap is 1M. **Coordinate-frame caveat:**
the colmap parser reorients (`up`), recenters (`poses`), and rescales
(`auto-scale-poses` → ±1 box) the world; whether `ns-export gaussian-splat` undoes
that transform varies by nerfstudio version. Before rendering an exported PLY with
SplatRs's COLMAP-frame cameras, check `dataparser_transforms.json` (saved in the run
dir — the transform + scale) and apply its inverse if the export is in nerfstudio's
normalized frame. A one-image sanity render tells you immediately (wrong frame =
empty/garbage view, not a subtle error).

## Exit criteria / decision

- If `gap@equal-iters` at 3k is large (≳2 dB or gross visual defects at matched
  iteration): tier-0 auto-research targets **quality levers**, baseline renders become
  the per-generation visual reference. (Remember splatfacto's 3k checkpoint trained
  only at 1/4 res — a *fair* early-iteration reading should also glance at 6k+.)
- If `gap@equal-iters` is small and the visible 15-minute difference is mostly
  iteration count: pause lever search, land **tile-binned rasterization** first.
- Either way `baselines/` becomes a standing fixture the auto-research loop scores
  against (and the sequestered-scene holdout lives here too).

## Known risks

- All CLI flags are confirmed from the on-box `-h` output (2026-07-15). The remaining
  **(verify)** items are runtime behaviors: ns-eval render filename convention,
  `ns-export gaussian-splat` flag spelling/output name, checkpoint step numbering
  (0-indexed?), TB scalar tag names, and whether the PLY export is in COLMAP or
  normalized coordinates. Every one is checkable in minutes on the tier-0 run.
- **Container drift:** `autotag nerfstudio` resolves lazily — a `docker pull` or image
  rebuild between runs silently changes the nerfstudio/gsplat versions under identical
  commands. Phase 1 step 2 records the resolved tag + digest; re-check it before any
  run whose numbers will be compared against earlier ones.
- **Root-owned outputs:** everything the container writes lands root-owned on the
  host. The `chown` steps (4.1 note, end of the eval loop) exist because a missed one
  makes the host-side `mv`/stash loop fail with EPERM halfway through — after which
  the checkpoint set is split across two directories; re-run `chown`, merge `stash/`
  back, and restart the loop.
- **Nothing persists outside `/workspace`:** shims, scripts, and outputs must live
  under `~/SplatRsBaseline` or they vanish with the container instance.
- **Silent hang in detached tmux = a hidden password prompt.** Only possible if the
  launching shell lacks the docker group (then jetson-containers falls back to
  sudo-ing docker, and the detached session has nowhere to show the prompt).
  Symptoms: no log output, no GPU load. Rescue: `tmux attach -t <session>`, type the
  password. Prevention is the docker-group membership (Conventions) — obsolete on
  hebot since 2026-07-17, kept here for fresh-box setups.
- Splatfacto will exceed 400k Gaussians well before 30k iters → same-renderer .ply
  comparison limited to early checkpoints; the PNG-pair comparison is unaffected.
- Model-vs-disk resolution mismatch: the tandt `cameras.bin` is 1959×1090 while
  `images/` are 980×545 — all `images_N` names and downscale factors are relative to
  the MODEL (Phase 2 table); an explicit factor is mandatory on every run.
- Resize rounding: nerfstudio's default rounding is **floor** (would expect 489×272 at
  factor 4); SplatRs half-res is 490×273. Guarded twice: ceil-generated `images_4/` +
  `--downscale-rounding-mode ceil` at train time, plus the Phase 2 GT pixel check. If
  sizes mismatch anywhere downstream, crop-to-common is NOT acceptable for PSNR — fix
  the pipeline.
- `ns-eval`/`ns-export` `weights_only` load failures on this torch build (torch≥2.6
  defaults `weights_only=True`; nerfstudio checkpoints pickle a numpy scalar) — **fixed
  2026-07-17 by `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` set in `jcb`** (covers all
  container commands). The `-safe` monkeypatch shim (`jetson/ns-viewer-safe` pattern) is
  the fallback if a future torch ignores the env var.
- The eval-loop stash trick depends on "ns-eval loads the highest-numbered checkpoint
  present" — confirm on the tier-0 run (7 checkpoints, minutes to walk) before relying
  on it for tier-1. **Confirmed 2026-07-17, plus a second constraint the hard way:** the
  stash dir must live OUTSIDE `nerfstudio_models/`. nerfstudio parses `int()` on every
  entry of `load_dir`, so a `stash/` subdir there crashes eval/export with
  `invalid literal for int() with base 10: 'stas'` (`"stash"[0:-1]`). Fixed in
  `eval_all_checkpoints.sh` (stashes to `<run>/stash`, a sibling).
