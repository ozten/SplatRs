#!/usr/bin/env python3
"""
Plot PSNR vs. Gaussian count for SplatRs runs against a nerfstudio/splatfacto
baseline on the same scene, to diagnose whether SplatRs's quality gap is a
capacity (Gaussian count) gap or a recipe/algorithm gap at equal count.

Data sources:
  - SplatRs: `metrics.csv` files produced by `sugar-train` (columns include
    `iteration`, `psnr`, `num_gaussians`).
  - Splatfacto ns-eval grid: a directory of `iter_<N>.json` files (each with
    `.results.psnr`) produced by `ns-eval`, plus a `iteration,gaussian_count`
    CSV recording the Gaussian count at each checkpoint.
  - Splatfacto nerfstudio training-log CSV (optional): the raw
    `dashboard.csv`/`metrics.csv` nerfstudio writes during training, containing
    a `Train Metrics Dict/gaussian_count` column and an
    `Eval Images Metrics Dict (all images)/psnr` column. NOTE: depending on
    the dataparser's `eval_mode`, this PSNR may be measured on the *training*
    views (eval_mode: all) rather than held out ones -- inflated and not
    directly comparable to the ns-eval grid. The script labels it accordingly.

Usage:
    python plot_count_vs_quality.py \
        --splatrs runs/srt15k_a3_sp500/metrics.csv \
        --splatrs runs/srt30k_a3_sp500/metrics.csv \
        --splatrs-allviews runs/srt30k_allviews_nd28/metrics.csv \
        --splatfacto-eval-dir /path/to/splatfacto_eval \
        --splatfacto-counts /path/to/splatfacto_counts.csv \
        --nerfstudio-metrics baselines/nerfstudio/ns-t1_.../metrics.csv \
        --output /path/to/count_vs_quality.png

This script:
1. Loads each SplatRs run's (iteration, psnr, num_gaussians) trajectory.
2. Loads the splatfacto ns-eval grid by joining iter_<N>.json PSNR values
   against the iteration,gaussian_count CSV.
3. Optionally loads the raw nerfstudio training CSV, aligning its per-step
   `gaussian_count` column against its `(all images)/psnr` eval column.
4. Renders a two-panel PNG: PSNR vs num_gaussians (log-x, "the money plot")
   on the left, and PSNR-vs-iteration / count-vs-iteration stacked on the
   right.
5. Prints an equal-count readout table and end-of-curve PSNR-per-count
   slopes to stdout.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# --- dataviz-skill palette: fixed categorical order, color = system identity ---
COLOR_SPLATRS = "#2a78d6"      # slot 1, blue
COLOR_SPLATFACTO = "#008300"   # slot 2, green
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
AXIS_BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def humanize_count(n):
    """58031 -> '58.0k'; 302489 -> '302k'."""
    n = float(n)
    if n >= 1000:
        return f"{n / 1000:.1f}k" if n < 100_000 else f"{n / 1000:.0f}k"
    return f"{n:.0f}"


def load_splatrs_csv(path, label, color, linestyle, marker):
    """Load a SplatRs metrics.csv and keep only the true held-out multi-view
    eval checkpoints.

    Data quirk: none of the shipped CSVs have blank psnr cells, but the psnr
    column is NOT a single consistent metric. sugar-train logs a row every
    ~100 iterations; most of those rows are a noisy single-view/training-batch
    PSNR proxy (std ~2.5-3 dB across a run, spanning e.g. 7.7-24.8 dB) that is
    NOT comparable across checkpoints. Only the rows logged at the
    `--settle-prune-interval` cadence (every 500 iters in these runs, plus the
    very first row) are the actual multi-view held-out eval PSNR (std
    ~0.5-1 dB, consistent with the runs' documented final numbers, e.g. 16.76
    @ 15k for srt15k_a3_sp500). Those "real eval" rows are identifiable
    without hardcoding the interval: they are immediately followed by a row
    exactly 1 iteration later (an artifact of the eval pass consuming the
    current step before the training loop increments), plus the first and
    last logged rows. We filter to that eval-cadence subset here; the noisy
    interstitial rows are dropped for the money plot / equal-count table
    (kept nowhere -- they would make PSNR-vs-count look like random noise)."""
    df = pd.read_csv(path)
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["psnr"] = pd.to_numeric(df["psnr"], errors="coerce")
    df["num_gaussians"] = pd.to_numeric(df["num_gaussians"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["iteration", "psnr", "num_gaussians"])
    df = df.sort_values("iteration").reset_index(drop=True)

    next_diff = df["iteration"].shift(-1) - df["iteration"]
    eval_mask = (
        (next_diff == 1)
        | (df["iteration"] == df["iteration"].max())
        | (df["iteration"] == df["iteration"].min())
    )
    df = df[eval_mask].reset_index(drop=True)
    n_dropped = n_before - len(df)

    return {
        "label": label,
        "system": "splatrs",
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "df": df[["iteration", "psnr", "num_gaussians"]],
        "n_dropped_rows": n_dropped,
        "source": str(path),
    }


def load_splatfacto_grid(eval_dir, counts_csv, label, color, linestyle, marker):
    """Join iter_<N>.json (.results.psnr) against iteration,gaussian_count CSV."""
    counts = pd.read_csv(counts_csv, header=None, names=["iteration", "num_gaussians"])
    counts["iteration"] = pd.to_numeric(counts["iteration"], errors="coerce")
    counts["num_gaussians"] = pd.to_numeric(counts["num_gaussians"], errors="coerce")

    rows = []
    missing = []
    eval_dir = Path(eval_dir)
    for _, row in counts.iterrows():
        it = int(row["iteration"])
        f = eval_dir / f"iter_{it}.json"
        if not f.exists():
            missing.append(it)
            continue
        with open(f) as fh:
            data = json.load(fh)
        psnr = data.get("results", {}).get("psnr")
        if psnr is None:
            missing.append(it)
            continue
        rows.append({"iteration": it, "psnr": float(psnr), "num_gaussians": row["num_gaussians"]})

    df = pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)
    return {
        "label": label,
        "system": "splatfacto",
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "df": df,
        "n_dropped_rows": len(missing),
        "source": f"{eval_dir} + {counts_csv}",
        "missing_iters": missing,
    }


def load_nerfstudio_training_csv(path, label, color, linestyle, marker,
                                  psnr_col="Eval Images Metrics Dict (all images)/psnr",
                                  count_col="Train Metrics Dict/gaussian_count"):
    """Load nerfstudio's raw per-step training CSV. `psnr_col` is logged only
    at steps_per_eval_all_images intervals; `count_col` is logged much more
    densely (every `steps_per_log`). We align them by exact step match, which
    holds for this dataset because the eval interval (1000) is a multiple of
    the log interval (10)."""
    df = pd.read_csv(path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")

    eval_rows = df[df[psnr_col].notna()][["step", psnr_col]].copy()
    count_rows = df[df[count_col].notna()][["step", count_col]].copy()

    merged = eval_rows.merge(count_rows, on="step", how="left")
    n_unmatched = merged[count_col].isna().sum()
    merged = merged.dropna(subset=[count_col])
    merged = merged.rename(
        columns={"step": "iteration", psnr_col: "psnr", count_col: "num_gaussians"}
    )
    merged = merged.sort_values("iteration").reset_index(drop=True)
    return {
        "label": label,
        "system": "splatfacto",
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "df": merged[["iteration", "psnr", "num_gaussians"]],
        "n_dropped_rows": int(n_unmatched),
        "source": str(path),
    }


def interp_psnr_at_count(df, target_count):
    """Interpolate PSNR at target_count using log10(count) as the x-axis
    (matches the log-x money plot). Returns None if target_count falls
    outside [min(count), max(count)] for this run -- we do NOT extrapolate
    silently; callers should treat None as 'no data at this count'."""
    if df.empty:
        return None
    counts = df["num_gaussians"].to_numpy(dtype=float)
    psnr = df["psnr"].to_numpy(dtype=float)
    order = np.argsort(counts)
    counts_sorted = counts[order]
    psnr_sorted = psnr[order]
    log_c = np.log10(counts_sorted)
    target_log = np.log10(target_count)
    if target_log < log_c.min() or target_log > log_c.max():
        return None
    return float(np.interp(target_log, log_c, psnr_sorted))


def end_slope(df, n_points=15, low_confidence_threshold=0.02):
    """PSNR-per-log10(count) slope using the last n_points eval checkpoints
    (by iteration), fit via least squares.

    The denominator (how far log10(count) actually moved over that window)
    matters as much as the slope itself: once densification has settled to
    equilibrium (splits/clones balanced by settle-prune), count barely moves
    checkpoint to checkpoint and a naive slope estimate blows up on eval
    noise alone. We report the *signed* Δlog10(count) (negative = count is
    net-shrinking over the window, not growing) and a low_confidence flag
    when the window's count range is too narrow (<~2% relative) for the
    slope to mean much, rather than silently hiding the number."""
    if len(df) < 2:
        return None
    tail = df.sort_values("iteration").tail(n_points)
    counts = tail["num_gaussians"].to_numpy(dtype=float)
    psnr = tail["psnr"].to_numpy(dtype=float)
    log_c = np.log10(counts)
    d_log_c = float(log_c[-1] - log_c[0])  # signed: direction of count movement
    count_ratio = float(10 ** d_log_c)     # e.g. 0.97 = count shrank ~3% over the window
    if abs(d_log_c) < 1e-6:
        slope = None
    else:
        slope, _ = np.polyfit(log_c, psnr, 1)
        slope = float(slope)
    return {
        "slope_db_per_decade": slope,
        "d_log10_count": d_log_c,
        "count_ratio": count_ratio,
        "count_first": float(counts[0]),
        "count_last": float(counts[-1]),
        "low_confidence": abs(d_log_c) < low_confidence_threshold,
    }


def build_equal_count_table(runs, targets):
    rows = []
    for target in targets:
        row = {"target_count": target}
        for run in runs:
            val = interp_psnr_at_count(run["df"], target)
            row[run["label"]] = val
        rows.append(row)
    return pd.DataFrame(rows)


def print_equal_count_table(table_df, runs):
    print("\n=== Equal-count PSNR readout (interpolated in log10(count) space) ===")
    print("(blank = target count outside this run's observed [min, max] Gaussian count -- not extrapolated)\n")
    col_width = max(len(r["label"]) for r in runs) + 2
    header = f"{'count':>10} | " + " | ".join(f"{r['label']:<{col_width}}" for r in runs)
    print(header)
    print("-" * len(header))
    for _, row in table_df.iterrows():
        cells = []
        for r in runs:
            v = row[r["label"]]
            cells.append(f"{v:<{col_width}.2f}" if v is not None and not pd.isna(v) else f"{'N/A':<{col_width}}")
        print(f"{humanize_count(row['target_count']):>10} | " + " | ".join(cells))

    # Deltas: for each target count where at least one splatrs and one
    # splatfacto run both have data, report splatfacto - splatrs.
    print("\n--- Deltas (splatfacto - splatrs) at counts where both systems have data ---")
    splatrs_runs = [r for r in runs if r["system"] == "splatrs"]
    splatfacto_runs = [r for r in runs if r["system"] == "splatfacto"]
    any_overlap = False
    for _, row in table_df.iterrows():
        srt_vals = [(r["label"], row[r["label"]]) for r in splatrs_runs if pd.notna(row[r["label"]])]
        sf_vals = [(r["label"], row[r["label"]]) for r in splatfacto_runs if pd.notna(row[r["label"]])]
        if srt_vals and sf_vals:
            any_overlap = True
            for srt_label, srt_v in srt_vals:
                for sf_label, sf_v in sf_vals:
                    print(f"  @ {humanize_count(row['target_count']):>7}: {sf_label} ({sf_v:.2f}) - {srt_label} ({srt_v:.2f}) = {sf_v - srt_v:+.2f} dB")
    if not any_overlap:
        print("  NONE -- no target count has both a SplatRs and a splatfacto data point in range.")
        srt_max = max(r["df"]["num_gaussians"].max() for r in splatrs_runs) if splatrs_runs else float("nan")
        sf_min = min(r["df"]["num_gaussians"].min() for r in splatfacto_runs) if splatfacto_runs else float("nan")
        print(f"  SplatRs max observed count: {humanize_count(srt_max)}  |  splatfacto min observed count: {humanize_count(sf_min)}")
        print("  These ranges do not overlap in the supplied data -- see report for discussion.")


def print_end_slopes(runs, n_points=15):
    print(f"\n=== End-of-curve PSNR-per-count slope (last {n_points} eval checkpoints) ===")
    for r in runs:
        s = end_slope(r["df"], n_points=n_points)
        if s is None:
            print(f"  {r['label']:<58} not enough points")
            continue
        direction = "flat" if abs(s["d_log10_count"]) < 1e-6 else (
            "GROWING" if s["d_log10_count"] > 0 else "SHRINKING"
        )
        conf = "LOW CONFIDENCE" if s["low_confidence"] else "ok"
        slope_str = f"{s['slope_db_per_decade']:+.1f} dB/decade(count)" if s["slope_db_per_decade"] is not None else "undefined (count exactly flat)"
        print(
            f"  {r['label']:<58} {slope_str:<32} count {direction:<9} "
            f"{humanize_count(s['count_first'])} -> {humanize_count(s['count_last'])} "
            f"(x{s['count_ratio']:.3f}, Δlog10count={s['d_log10_count']:+.4f})  [{conf}]"
        )
    print("  Note: a run whose count is SHRINKING (or flat) at the tail is not 'still climbing the count")
    print("  curve' -- settle-prune has reached (near-)equilibrium and further training would need to")
    print("  improve quality at fixed capacity, not from adding more Gaussians.")


def annotate_final_point(ax, run, offset=(6, 4)):
    df = run["df"].sort_values("iteration")
    last = df.iloc[-1]
    iter_str = f"{last['iteration'] / 1000:.0f}k" if last["iteration"] >= 1000 else f"{last['iteration']:.0f}"
    text = f"{last['psnr']:.2f} @ {iter_str}, {humanize_count(last['num_gaussians'])}"
    ax.annotate(
        text,
        xy=(last["num_gaussians"], last["psnr"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=7.5,
        color=INK_SECONDARY,
        fontfamily="sans-serif",
    )


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(True, color=GRID_HAIRLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def make_plot(runs, output_path, title):
    fig = plt.figure(figsize=(14, 7), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1], wspace=0.28, hspace=0.12)

    ax_money = fig.add_subplot(gs[:, 0])
    ax_psnr_iter = fig.add_subplot(gs[0, 1])
    ax_count_iter = fig.add_subplot(gs[1, 1], sharex=ax_psnr_iter)

    # --- Left: the money plot, PSNR vs num_gaussians (log-x) ---
    # Final-point label offsets: stack them apart when two runs' final points
    # land close together (e.g. srt15k/srt30k end within ~2k count, ~0.5 dB).
    label_offsets = [(8, 10), (8, -16), (10, 10), (8, 10), (8, -16), (10, -16)]
    for i, run in enumerate(runs):
        df = run["df"].sort_values("iteration")
        ax_money.plot(
            df["num_gaussians"], df["psnr"],
            color=run["color"], linestyle=run["linestyle"], marker=run["marker"],
            markersize=5, linewidth=1.8, label=run["label"], alpha=0.95,
            markeredgewidth=0, zorder=3,
        )
        annotate_final_point(ax_money, run, offset=label_offsets[i % len(label_offsets)])

    ax_money.set_xscale("log")
    ax_money.set_xlabel("Gaussian count (log scale)", color=INK_SECONDARY, fontsize=9)
    ax_money.set_ylabel("PSNR (dB)", color=INK_SECONDARY, fontsize=9)
    ax_money.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=10)
    ax_money.xaxis.set_major_formatter(FuncFormatter(lambda x, _: humanize_count(x)))
    ax_money.xaxis.set_minor_formatter(NullFormatter())
    style_axes(ax_money)
    # Upper-left is empty in this data (SplatRs never exceeds ~17 dB, splatfacto
    # never has a checkpoint below ~110k Gaussians) -- placing the legend there
    # keeps it clear of every series' markers.
    legend = ax_money.legend(
        loc="upper left", fontsize=7.8, frameon=True, facecolor=SURFACE,
        edgecolor=GRID_HAIRLINE, framealpha=0.95,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    # --- Right, top: PSNR vs iteration ---
    for run in runs:
        df = run["df"].sort_values("iteration")
        ax_psnr_iter.plot(
            df["iteration"], df["psnr"],
            color=run["color"], linestyle=run["linestyle"], marker=None,
            linewidth=1.6, alpha=0.95, zorder=3,
        )
    ax_psnr_iter.set_ylabel("PSNR (dB)", color=INK_SECONDARY, fontsize=8.5)
    ax_psnr_iter.set_title("Training dynamics", color=INK_PRIMARY, fontsize=10.5, loc="left")
    plt.setp(ax_psnr_iter.get_xticklabels(), visible=False)
    style_axes(ax_psnr_iter)

    # --- Right, bottom: num_gaussians vs iteration ---
    for run in runs:
        df = run["df"].sort_values("iteration")
        ax_count_iter.plot(
            df["iteration"], df["num_gaussians"],
            color=run["color"], linestyle=run["linestyle"], marker=None,
            linewidth=1.6, alpha=0.95, zorder=3,
        )
    ax_count_iter.set_xlabel("Iteration", color=INK_SECONDARY, fontsize=8.5)
    ax_count_iter.set_ylabel("Gaussian count", color=INK_SECONDARY, fontsize=8.5)
    ax_count_iter.yaxis.set_major_formatter(FuncFormatter(lambda x, _: humanize_count(x)))
    style_axes(ax_count_iter)

    fig.savefig(output_path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nWrote plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot PSNR vs Gaussian count for SplatRs vs splatfacto (capacity vs recipe diagnostic)"
    )
    parser.add_argument("--splatrs", action="append", default=[], metavar="METRICS_CSV",
                         help="SplatRs metrics.csv path (repeatable); same eval split as splatfacto ns-eval grid")
    parser.add_argument("--splatrs-allviews", type=str, default=None, metavar="METRICS_CSV",
                         help="SplatRs metrics.csv trained/evaluated on the all-views split (NOT comparable split; plotted dashed)")
    parser.add_argument("--splatfacto-eval-dir", type=str, required=True,
                         help="Directory of iter_<N>.json files from ns-eval (each has .results.psnr)")
    parser.add_argument("--splatfacto-counts", type=str, required=True,
                         help="CSV of iteration,gaussian_count for the splatfacto ns-eval grid")
    parser.add_argument("--nerfstudio-metrics", type=str, default=None,
                         help="Optional raw nerfstudio training metrics.csv (has an early low-count curve, "
                              "but PSNR may be measured on training views -- see config.yml eval_mode)")
    parser.add_argument("--nerfstudio-eval-mode", type=str, default=None,
                         help="eval_mode read from the nerfstudio run's config.yml (e.g. 'all', 'interval'); "
                              "used only to label the curve correctly. If 'all', the curve is labeled "
                              "'train-set eval, inflated'.")
    parser.add_argument("--output", type=str,
                         default="/private/tmp/claude-502/-Users-ozten-Projects-SplatRs/"
                                 "084c890a-78f2-4936-8e59-07d98268eb4d/scratchpad/count_vs_quality.png",
                         help="Output PNG path")
    parser.add_argument("--title", type=str,
                         default="SplatRs vs splatfacto: PSNR vs Gaussian count (Tanks & Temples 'train', 490x273)")
    args = parser.parse_args()

    if not args.splatrs:
        print("Error: at least one --splatrs metrics.csv is required", file=sys.stderr)
        sys.exit(1)

    runs = []

    # SplatRs canonical (eval-interval-8 split) runs -- solid / dash-dot, blue
    splatrs_linestyles = ["-", "-.", (0, (3, 1, 1, 1))]
    splatrs_markers = ["o", "s", "D"]
    for i, path in enumerate(args.splatrs):
        path = Path(path)
        run_name = path.parent.name
        label = f"SplatRs {run_name}"
        run = load_splatrs_csv(
            path, label, COLOR_SPLATRS,
            splatrs_linestyles[i % len(splatrs_linestyles)],
            splatrs_markers[i % len(splatrs_markers)],
        )
        runs.append(run)

    # SplatRs all-views split -- dashed, blue, explicitly flagged non-comparable split
    if args.splatrs_allviews:
        path = Path(args.splatrs_allviews)
        run_name = path.parent.name
        label = f"SplatRs {run_name} (all-views split)"
        run = load_splatrs_csv(path, label, COLOR_SPLATRS, "--", "^")
        runs.append(run)

    # Splatfacto ns-eval grid -- solid, green (canonical, held-out interval-8 split)
    sf_grid = load_splatfacto_grid(
        args.splatfacto_eval_dir, args.splatfacto_counts,
        "Splatfacto (ns-eval, held-out interval-8)", COLOR_SPLATFACTO, "-", "o",
    )
    runs.append(sf_grid)

    # Splatfacto raw nerfstudio training log -- dotted, green, flagged if train==eval
    if args.nerfstudio_metrics:
        eval_mode = (args.nerfstudio_eval_mode or "").lower()
        if eval_mode == "all":
            label = "Splatfacto (nerfstudio log, train-set eval -- INFLATED, not comparable)"
        elif eval_mode:
            label = f"Splatfacto (nerfstudio log, eval_mode={eval_mode})"
        else:
            label = "Splatfacto (nerfstudio log, eval split unknown)"
        ns_run = load_nerfstudio_training_csv(
            args.nerfstudio_metrics, label, COLOR_SPLATFACTO, ":", "x",
        )
        runs.append(ns_run)

    # --- Report data quirks found while loading ---
    print("=== Loaded runs ===")
    for r in runs:
        n = len(r["df"])
        cmin, cmax = (r["df"]["num_gaussians"].min(), r["df"]["num_gaussians"].max()) if n else (float("nan"),) * 2
        pmin, pmax = (r["df"]["psnr"].min(), r["df"]["psnr"].max()) if n else (float("nan"),) * 2
        print(f"  {r['label']:<58} n={n:<4} count=[{humanize_count(cmin)}, {humanize_count(cmax)}]  "
              f"psnr=[{pmin:.2f}, {pmax:.2f}]  dropped_rows={r['n_dropped_rows']}  source={r['source']}")
        if r.get("missing_iters"):
            print(f"      missing checkpoints: {r['missing_iters']}")

    # --- Equal-count table ---
    targets = [40_000, 60_000, 80_000, 100_000, 120_000, 150_000, 200_000, 300_000]
    table = build_equal_count_table(runs, targets)
    print_equal_count_table(table, runs)

    # --- End-of-curve slopes ---
    print_end_slopes(runs)

    # --- Plot ---
    make_plot(runs, args.output, args.title)


if __name__ == "__main__":
    main()
