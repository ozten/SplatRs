#!/usr/bin/env python3
"""Batch LPIPS over a training run's saved test-view renders.

Walks <run_dir>/m8_test_view_rendered_<iter>.png (and the final
m8_test_view_rendered.png, reported as the run's last iteration + 'final'),
scores each against <run_dir>/m8_test_view_target.png with LPIPS (AlexNet),
and writes CSV `iteration,lpips` to stdout or --output.

Needs lpips+torch (the session scratchpad venv has them):
    <venv>/bin/python scripts/compute_lpips_run.py runs/<run> [--every N] [--output f.csv]
"""
import argparse
import re
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--every", type=int, default=1,
                    help="score every Nth checkpoint render (default 1 = all)")
    ap.add_argument("--output", type=Path, default=None,
                    help="write CSV here instead of stdout")
    args = ap.parse_args()

    target = args.run_dir / "m8_test_view_target.png"
    if not target.exists():
        sys.exit(f"no target image at {target}")

    import lpips  # noqa: deferred heavy import
    import torch
    from PIL import Image
    import numpy as np

    loss_fn = lpips.LPIPS(net="alex", verbose=False)

    def to_tensor(p: Path):
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

    t_target = to_tensor(target)

    renders = []
    for p in args.run_dir.glob("m8_test_view_rendered_*.png"):
        m = re.search(r"_(\d+)\.png$", p.name)
        if m:
            renders.append((int(m.group(1)), p))
    renders.sort()
    renders = renders[:: args.every]
    final = args.run_dir / "m8_test_view_rendered.png"
    if final.exists():
        last_iter = renders[-1][0] if renders else 0
        renders.append((max(last_iter, 0) or "final", final))

    lines = ["iteration,lpips"]
    with torch.no_grad():
        for it, p in renders:
            d = loss_fn(to_tensor(p), t_target).item()
            lines.append(f"{it},{d:.6f}")
            print(f"{it},{d:.6f}", file=sys.stderr)

    out = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(out)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
