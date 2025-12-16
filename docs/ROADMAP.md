# Roadmap (Vanilla 3D Gaussian Splatting First)

This repo’s end-goal is **SuGaR** (Surface-Aligned Gaussian Splatting) + mesh extraction, but we should **fully finish vanilla 3D Gaussian Splatting (3DGS)** from the first paper before adding any SuGaR-specific regularization.

Use `docs/sugar-rs-milestones.md` as the source-of-truth for “done-ness”. This file is a short execution checklist to keep us focused.

## Current Focus: M7 → M10 (3DGS Training Loop)

- [ ] **M7** Single-image overfit is reliable (PSNR target).
- [ ] **M8** Multi-view training works:
  - [ ] Train/test split (hold out 2–3 views)
  - [ ] Training view sampling (per-iter view selection)
  - [ ] Report PSNR on held-out views
  - [ ] Save at least one rendered/target pair for visual inspection
  - [ ] Optional: enable opacity learning (`--learn-opacity`) and verify PSNR improves vs color-only
  - [ ] Optional: enable position learning (`--learn-position`) and verify held-out views sharpen (use smaller LR)
- [ ] **M9** Adaptive density control (split/clone/prune) improves M8 quality.
- [ ] **M10** Reference-quality 3DGS training (benchmark scene).

## Rules of Engagement

- **Learning + correctness > speed**: prefer explicit math and gradient checks.
- **No SuGaR until M10**: SuGaR regularization + mesh extraction start at M11.
- **Tests are docs**: for each milestone, keep unit + gradient + (optional) visual tests.

## Where to Look

- Milestones and verification: `docs/sugar-rs-milestones.md`
- Long-form technical guidance: `docs/sugar-rust-roadmap.md`

## How To Run (M7 / M8)

Your calipers dataset root (per `AGENTS.md`) is:

- `/Users/ozten/Projects/GuassianPlay/digital_calipers2_project`

M7 (single-image overfit, color-only):

- `cargo run --bin sugar-train -- --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --iters 1000 --lr 0.05 --downsample 0.25 --max-gaussians 20000 --image-index 0 --out-dir test_output`

M8 (multi-view, color-only):

- `cargo run --bin sugar-train -- --multiview --max-images 5 --max-gaussians 2000 --downsample 0.125 --iters 50 --val-interval 10 --max-test-views 1 --log-interval 1 --loss l2 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`

M8 (multi-view, color + opacity):

- `cargo run --bin sugar-train -- --multiview --max-images 5 --max-gaussians 2000 --downsample 0.125 --iters 200 --val-interval 20 --max-test-views 1 --log-interval 1 --loss l1-dssim --learn-opacity --lr 0.01 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`

M8 (multi-view, color + opacity + position):

- `cargo run --bin sugar-train -- --multiview --max-images 5 --max-gaussians 2000 --downsample 0.125 --iters 200 --val-interval 20 --max-test-views 1 --log-interval 1 --loss l1-dssim --learn-opacity --learn-position --lr 0.002 --dataset-root /Users/ozten/Projects/GuassianPlay/digital_calipers2_project --out-dir test_output`
