#!/usr/bin/env python3
"""Extract all tensorboard scalars from a nerfstudio run into <run_dir>/metrics.csv.

Runs INSIDE the container (tensorboard's API lives there):
  jcb python3 /workspace/bin/tb_to_csv.py /workspace/ns-runs/<run-id>/splatfacto/<stamp>
"""
import csv
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
ea = EventAccumulator(run, size_guidance={"scalars": 0})
ea.Reload()
tags = list(ea.Tags()["scalars"])
if not tags:
    sys.exit(f"no scalar tags found under {run} — wrong dir, or --vis was not tensorboard?")

rows = {}
for tag in tags:
    for ev in ea.Scalars(tag):
        rows.setdefault(ev.step, {})[tag] = ev.value

out = f"{run}/metrics.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step"] + tags)
    for step in sorted(rows):
        w.writerow([step] + [rows[step].get(t, "") for t in tags])
print(f"wrote {out}  ({len(rows)} steps, {len(tags)} tags)")
print("tags:", *tags, sep="\n  ")
