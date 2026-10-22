#!/usr/bin/env python3
"""How aggressive can a mid-run cull be before it destroys designs worth keeping?

`scripts/mosaic_trajectory_rank.py` established that a mosaic trajectory's
in-loop loss at step k carries real rank information about its final re-fold
score, and that a 10% cull never killed a run's winner. This sweeps the two free
parameters of a culling scheme -- the step k at which it fires and the fraction
f it removes -- over the same logged trajectories.

    scripts/mosaic_cull_sweep.py \\
        ../outputs/mosaic/logs/mosaic-42118187.out \\
        ../mosaic_setup/mosaic/logs/hallucinate-39943975_*.out \\
        --out ../docs/figures/mosaic_cull_sweep.png

RISK is the share of the final top decile that the cull destroys. That, not
bottom-decile recall, is the metric a design campaign cares about: a cull that
removes mid-pack trajectories has cost nothing, because mid-pack designs are not
what gets shipped.

BENEFIT is the share of total optimization compute the cull REDIRECTS. In an
evolutionary scheme the population size is held constant and culled members are
replaced by resampled ones, so nothing is saved -- the GPU time is moved off
trajectories that were going nowhere and onto fresh draws. Whether that
redirected compute actually pays off CANNOT be answered from these logs: the
driver records no per-step PSSM state, so a resampled trajectory cannot be
replayed. Only a real run answers that. This figure bounds the risk side alone.

Both parameters are swept against the same 7 runs, so the per-cell numbers share
their run-level effects and should be read as a surface, not as 7 independent
estimates per cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Reuse the log parser rather than copying it: the two scripts must agree about
# what a complete trajectory is, or their conclusions are not comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosaic_trajectory_rank import (
    AXIS,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    RED,
    SURFACE,
    decile_size,
    parse_log,
    style_axes,
)

# The documented sequential blue ramp, steps 100 -> 700. Light means "lost
# nothing", dark means "lost top-decile designs", so danger reads as weight.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
# Ordinal slice colors: four ordered steps, none lighter than step 250 on a
# light surface, per the ordinal contrast rule.
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]

FIRST_STEP = 10
SLICES = [0.10, 0.20, 0.30, 0.50]


def sweep(runs, ks: np.ndarray, fs: np.ndarray):
    """Return (top-decile lost %, winners lost) over the (f, k) grid."""
    lost = np.zeros((len(fs), len(ks)))
    winners = np.zeros((len(fs), len(ks)), dtype=int)

    for i, f in enumerate(fs):
        for j, k in enumerate(ks):
            killed_top = 0
            total_top = 0
            for _, losses, finals in runs:
                n = losses.shape[0]
                size = decile_size(n)
                n_cull = round(f * n)
                if n_cull < 1:
                    continue
                order = np.argsort(finals)  # lower is better
                top = set(order[:size].tolist())
                winner = int(order[0])
                killed = set(np.argsort(losses[:, k])[-n_cull:].tolist())
                killed_top += len(killed & top)
                total_top += size
                winners[i, j] += winner in killed
            lost[i, j] = 100.0 * killed_top / total_top if total_top else np.nan
    return lost, winners


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--npz", type=Path, action="append", default=[],
                    help="a run normalised by scripts/mosaic_pipeline_traces.py")
    ap.add_argument("--boundaries", default="",
                    help="comma-separated step:label marks for panel B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k-step", type=int, default=2)
    args = ap.parse_args()

    runs = []
    for path in sorted(args.logs):
        losses, finals, _ = parse_log(path)
        if losses.shape[0] >= 8:
            runs.append((path.name, losses, finals))
    for path in sorted(args.npz):
        payload = np.load(path)
        runs.append((path.stem, payload["losses"], payload["finals"]))
    if not runs:
        raise SystemExit("no log had enough complete trajectories")

    steps = min(losses.shape[1] for _, losses, _ in runs)
    total = sum(losses.shape[0] for _, losses, _ in runs)
    ks = np.arange(FIRST_STEP, steps, args.k_step)
    fs = np.round(np.arange(0.05, 0.605, 0.05), 2)

    lost, winners = sweep(runs, ks, fs)
    # Benefit is pure geometry: a member culled at step k skips the remaining
    # (steps - k) of its schedule, and that share of the budget moves elsewhere.
    redirected = 100.0 * fs[:, None] * (steps - ks[None, :]) / steps

    print(f"\n{len(runs)} runs, {total} trajectories, {steps} steps")
    print(f"{'f':>6} " + " ".join(f"k={k:<5}" for k in (15, 25, 40, 60, 90)))
    for i, f in enumerate(fs):
        if f not in SLICES:
            continue
        cells = []
        for k in (15, 25, 40, 60, 90):
            j = int(np.argmin(np.abs(ks - k)))
            cells.append(f"{lost[i, j]:4.0f}% {'*' if winners[i, j] else ' '} ")
        print(f"{f:>6.2f} " + " ".join(cells))
    print("  % = share of the final top decile destroyed; * = a run lost its winner")
    print(f"\n{'f':>6} " + " ".join(f"k={k:<5}" for k in (15, 25, 40, 60, 90))
          + "   (lift vs culling at random; 1.0 = no better than blind)")
    for i, f in enumerate(fs):
        if f not in SLICES:
            continue
        cells = []
        for k in (15, 25, 40, 60, 90):
            j = int(np.argmin(np.abs(ks - k)))
            cells.append(f"{(f * 100) / lost[i, j]:5.2f}x " if lost[i, j] else "  inf ")
        print(f"{f:>6.2f} " + " ".join(cells))
    print()

    cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 9.2),
                             gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.3})
    fig.patch.set_facecolor(SURFACE)

    # -- Panel A: the decision surface ---------------------------------------
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    extent = [ks[0], ks[-1], fs[0] * 100, fs[-1] * 100]
    image = ax.imshow(lost, origin="lower", aspect="auto", extent=extent,
                      cmap=cmap, vmin=0, vmax=np.nanmax(lost))

    # Iso-benefit contours: how much of the budget each (k, f) redirects.
    contours = ax.contour(ks, fs * 100, redirected, levels=[5, 10, 20, 30, 40, 50],
                          colors=INK, linewidths=0.9, alpha=0.45)
    ax.clabel(contours, fmt="%d%% redirected", fontsize=7.5, colors=INK_SECONDARY)

    # "Did a run lose its winner" is one Bernoulli trial per run, so this grid
    # has 7 samples per cell. Drawing it as a boundary produced a ribbon that
    # wandered across the whole surface -- sampling noise rendered as structure.
    # It is reported in the printed table instead, where its resolution is
    # visible. The lowest f at which ANY run lost a winner is marked, which is
    # the one statement 7 runs can support.
    rows, cols = np.nonzero(winners > 0)
    if len(rows):
        ax.scatter(ks[cols], fs[rows] * 100, s=9, marker="o", color=RED,
                   linewidths=0, zorder=5,
                   label="a run lost its best design here")

    ax.set_xlabel("cull step k", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("cull fraction f  (% of population)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_title("Risk surface: share of the final top decile destroyed by a cull at (k, f)",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)

    bar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.045)
    bar.set_label("% of final top decile killed", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    bar.outline.set_visible(False)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8.5,
              labelcolor=INK_SECONDARY, loc="lower right", markerscale=1.6)

    # -- Panel B: slices at the cull fractions worth considering --------------
    ax = axes[1]
    style_axes(ax)
    ends = []
    for color, f in zip(ORDINAL, SLICES):
        i = int(np.argmin(np.abs(fs - f)))
        ax.plot(ks, lost[i], color=color, linewidth=2.0, zorder=4)
        ax.plot([], [], color=color, linewidth=2.0, label=f"f = {int(f * 100)}%")
        # Random reference for this f: culling blind destroys exactly f.
        ax.axhline(f * 100, color=color, linewidth=1.0, linestyle=(0, (3, 3)),
                   alpha=0.75, zorder=2)
        ax.annotate(f"random at f={int(f * 100)}%", (ks[0], f * 100), xytext=(4, 3),
                    textcoords="offset points", color=color, fontsize=7.5)
        ends.append([lost[i][-1], f"f = {int(f * 100)}%", color])
    # f = 10% and f = 20% both finish at zero, so their end labels would print
    # on top of each other.
    ends.sort()
    gap = 0.05 * (max(e[0] for e in ends) - min(e[0] for e in ends) or 1.0)
    for index in range(1, len(ends)):
        ends[index][0] = max(ends[index][0], ends[index - 1][0] + gap)
    for y_text, label, color in ends:
        ax.annotate(label, (ks[-1], y_text), xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=8.5, va="center")
    # Stage markers sit along the bottom here: the top of this panel is where
    # the f = 50% line and the legend both live.
    if args.boundaries:
        marks = [piece.split(":", 1) for piece in args.boundaries.split(",")]
    elif steps == 165:
        marks = [("100", "sharpen"), ("150", "final")]
    else:
        marks = []
    for edge, text in ((int(step), label) for step, label in marks):
        if edge < steps:
            ax.axvline(edge, color=AXIS, linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
            ax.annotate(text, (edge, 0.0), xytext=(3, 6), textcoords="offset points",
                        xycoords=("data", "axes fraction"), color=INK_MUTED, fontsize=8)
    ax.set_xlabel("cull step k", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("% of final top decile killed", color=INK_SECONDARY, fontsize=9.5)
    ax.set_title("Slices at the cull fractions worth considering  ·  dashed = culling at random",
                 color=INK, fontsize=11, loc="left", pad=8)
    # Below the axes: the plot area is occupied by four curves and their four
    # random references, and all eight are already labelled in place.
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.set_xlim(ks[0], ks[-1])

    axes[0].set_xlim(ks[0], ks[-1])

    fig.suptitle("How hard can you cull, and when?", color=INK, fontsize=13.5,
                 x=0.012, ha="left", y=1.0, fontweight="bold")
    fig.text(0.012, 0.972,
             f"{total} logged trajectories, {len(runs)} "
             f"run{'s' if len(runs) > 1 else ''}  ·  risk measured on the "
             "final top decile  ·  benefit is compute REDIRECTED, not saved",
             color=INK_MUTED, fontsize=9, ha="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
