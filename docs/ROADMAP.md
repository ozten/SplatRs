# Roadmap (GPU-Accelerated 3DGS, Then SuGaR)

This repo's end-goal is **SuGaR** (Surface-Aligned Gaussian Splatting) + mesh extraction.

**Strategy:** Complete vanilla 3D Gaussian Splatting (3DGS) on CPU (M1-M10), then port to GPU (M11-M12) for practical speed, then add SuGaR features (M13-M14).

Use `docs/sugar-rs-milestones.md` as the source-of-truth for "done-ness". This file is a short execution checklist to keep us focused.

## Current Focus: M10 → M12 (Complete 3DGS + GPU)

- [x] **M7** Single-image overfit is reliable (PSNR target).
- [x] **M8** Multi-view training works:
  - [x] Train/test split (hold out 2–3 views)
  - [x] Training view sampling (per-iter view selection)
  - [x] Report PSNR on held-out views
  - [x] Save at least one rendered/target pair for visual inspection
  - [x] Optional: enable opacity learning (`--learn-opacity`) and verify PSNR improves vs color-only
  - [x] Optional: enable position learning (`--learn-position`) and verify held-out views sharpen (use smaller LR)
- [x] **M9** Adaptive density control (split/clone/prune) improves M8 quality.
- [ ] **M10** Reference-quality 3DGS training (CPU validation baseline).
- [ ] **M11** GPU renderer matches CPU (correctness validation).
- [ ] **M12** GPU training end-to-end (10-50x speedup).

## Rules of Engagement

- **Learning + correctness > speed**: prefer explicit math and gradient checks.
- **GPU before SuGaR**: Port to GPU (M11-M12) for practical speed before adding SuGaR (M13-M14).
- **Tests are docs**: for each milestone, keep unit + gradient + (optional) visual tests.

## Where to Look

- Milestones and verification: `docs/sugar-rs-milestones.md`
- Long-form technical guidance: `docs/sugar-rust-roadmap.md`

## How To Run (M7 / M8)

Your calipers dataset root (per `AGENTS.md`) is:

- `/Users/ozten/Projects/GuassianPlay/digital_calipers2_project`

## Presets (Recommended)

Presets set a bundle of reasonable defaults; any flags you pass *after* `--preset` override the preset.

- `cargo run --bin sugar-train -- --preset m7 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`
- `cargo run --bin sugar-train -- --preset m8-smoke --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`
- `cargo run --bin sugar-train -- --preset m9 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`
- `cargo run --bin sugar-train -- --preset m10 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`

M7 (single-image overfit, color-only):

- `cargo run --bin sugar-train -- --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --iters 1000 --lr 0.05 --downsample 0.25 --max-gaussians 20000 --image-index 0 --out-dir test_output`

M8 (multi-view, color-only):

- `cargo run --bin sugar-train -- --multiview --max-images 5 --max-gaussians 2000 --downsample 0.125 --iters 50 --val-interval 10 --max-test-views 1 --log-interval 1 --loss l2 --seed 0 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`

M8 (multi-view, color + opacity):

- `cargo run --bin sugar-train -- --multiview --max-images 5 --max-gaussians 2000 --downsample 0.125 --iters 200 --val-interval 20 --max-test-views 1 --log-interval 1 --loss l1-dssim --learn-opacity --lr 0.01 --seed 0 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`

M8 (multi-view, color + opacity + position):

- `cargo run --bin sugar-train -- --multiview --max-images 5 --max-gaussians 2000 --downsample 0.125 --iters 200 --val-interval 20 --max-test-views 1 --log-interval 1 --loss l1-dssim --learn-opacity --learn-position --lr 0.002 --seed 0 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`

M9 (multi-view + densify/prune enabled):

- `cargo run --bin sugar-train -- --multiview --max-images 10 --max-gaussians 2000 --downsample 0.125 --iters 200 --val-interval 20 --max-test-views 2 --log-interval 1 --loss l2 --densify-interval 25 --densify-max-gaussians 8000 --densify-grad-threshold 0.1 --prune-opacity-threshold 0.01 --split-sigma-threshold 0.05 --seed 0 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`

M10 (multi-view + opacity + position + scale enabled):

- `cargo run --bin sugar-train -- --multiview --max-images 10 --max-gaussians 2000 --downsample 0.125 --iters 200 --val-interval 20 --max-test-views 2 --log-interval 1 --loss l1-dssim --learn-opacity --learn-position --learn-scale --learn-rotation --learn-sh --lr 0.002 --seed 0 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`
