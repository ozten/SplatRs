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
- [ ] **M9** Adaptive density control (split/clone/prune) improves M8 quality.
- [ ] **M10** Reference-quality 3DGS training (benchmark scene).

## Rules of Engagement

- **Learning + correctness > speed**: prefer explicit math and gradient checks.
- **No SuGaR until M10**: SuGaR regularization + mesh extraction start at M11.
- **Tests are docs**: for each milestone, keep unit + gradient + (optional) visual tests.

## Where to Look

- Milestones and verification: `docs/sugar-rs-milestones.md`
- Long-form technical guidance: `docs/sugar-rust-roadmap.md`
