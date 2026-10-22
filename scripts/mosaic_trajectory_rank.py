#!/usr/bin/env python3
"""Does a mosaic trajectory's in-loop loss at step k predict its final score?

This is the prerequisite for any early-stopping or population-resampling scheme
on top of `drivers/mosaic/hallucinate_binders.py`. Culling the worst
trajectories at step k only helps if the step-k loss carries rank information
about the score those trajectories would eventually have earned. If it does
not, culling is a random kill with extra steps.

The data already exists and costs nothing to read. `mosaic.optimizers`
unconditionally prints one line per optimization step (`_print_iter`, see
`mosaic_setup/mosaic/src/mosaic/optimizers.py`), so every mosaic run ever logged
carries its full per-step loss trace. The driver then prints one
`[n] loss=<final> (<secs>) <SEQUENCE>` line per completed design with the
expensive re-fold score. That is both halves of the correlation.

    scripts/mosaic_trajectory_rank.py \\
        ../outputs/mosaic/logs/mosaic-42118187.out \\
        ../mosaic_setup/mosaic/logs/hallucinate-39943975_*.out \\
        --out ../docs/figures/mosaic_trajectory_rank.png

Two scores are involved and they are NOT the same quantity. The per-step loss is
the 9-term in-loop objective at 4 samples and 1 recycle -- it is what the
gradient is taken through. The final score is a separate, more expensive
re-fold (iPTM, 3 recycles, 6 samples). The gap between them is precisely what
this script measures, so it is the point rather than a defect.

The bar is not rho > 0. It is the bottom panel: recall of the final worst
decile, against the chance line. A scheme that culls the bottom decile at step k
and recovers the final worst decile no better than chance has bought nothing.

Reports per run and pooled, because ~440 trajectories across ~7 runs is not 440
independent samples -- run-level effects (target, MSA, driver version) would
inflate a pooled-only correlation.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# The dataviz reference palette, light surface. Panel A is a diverging encoding
# (blue <-> red with a neutral gray midpoint) because its three classes are
# ordered -- worst / middle / best -- not three unrelated identities. Line style
# carries identity alongside hue so nothing rests on color alone.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
RED = "#e34948"

# A step-level decision is never taken inside the init transient: the loss falls
# from ~18 to ~3 in the first handful of steps and no culling scheme would fire
# there. Plotting from step 0 would spend the whole y range on that drop.
FIRST_STEP = 10

STEP_RE = re.compile(r"^(\d+) loss: (-?[\d.]+)")
DONE_RE = re.compile(r"^\[(\d+)\] loss=(-?[\d.]+) \((\d+)s\)\s*([A-Z]*)")


def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (per-step losses, final scores, mean seconds/step) for one log.

    Designs are closed by their completion line rather than by counting step
    numbers. The step counter restarts inside each APGM stage (0..99, 0..49,
    0..14), so a reset does not mean a new design, and a design cut off by the
    walltime leaves steps with no completion line at all. Keying on the
    completion line drops exactly those partial trajectories.
    """
    steps: list[float] = []
    traces: list[list[float]] = []
    finals: list[float] = []
    seconds: list[float] = []

    with path.open(errors="replace") as handle:
        for line in handle:
            step = STEP_RE.match(line)
            if step is not None:
                steps.append(float(step.group(2)))
                continue
            done = DONE_RE.match(line)
            if done is not None:
                traces.append(steps)
                finals.append(float(done.group(2)))
                seconds.append(float(done.group(3)))
                steps = []

    if not traces:
        return np.empty((0, 0)), np.empty(0), math.nan

    # Keep only the modal trace length. A log that mixes schedules (the older
    # `campaign-*` runs use 80 or 125 steps per design) would otherwise be
    # silently truncated to its shortest trajectory and compared across
    # incomparable step indices.
    lengths = [len(trace) for trace in traces]
    modal = max(set(lengths), key=lengths.count)
    keep = [index for index, length in enumerate(lengths) if length == modal]
    if len(keep) < len(traces):
        print(
            f"  {path.name}: dropped {len(traces) - len(keep)} trajectories "
            f"whose length != {modal} steps",
            file=sys.stderr,
        )

    losses = np.array([traces[index] for index in keep], dtype=float)
    return losses, np.array([finals[i] for i in keep]), float(np.mean(seconds)) / modal


def decile_size(n: int) -> int:
    """How many designs the bottom 10% is. At least one, or there is no test."""
    return max(1, round(0.1 * n))


def rank_curve(losses: np.ndarray, finals: np.ndarray) -> np.ndarray:
    """Spearman rho between loss at each step and the final score."""
    return np.array(
        [spearmanr(losses[:, k], finals).statistic for k in range(losses.shape[1])]
    )


def recall_curve(losses: np.ndarray, finals: np.ndarray) -> np.ndarray:
    """Fraction of the final worst decile that step k's worst decile catches.

    Lower is better for both scores, so the worst designs are the largest
    values and `argsort` puts them last.
    """
    n = losses.shape[0]
    size = decile_size(n)
    worst_final = set(np.argsort(finals)[-size:].tolist())
    out = np.empty(losses.shape[1])
    for k in range(losses.shape[1]):
        worst_k = set(np.argsort(losses[:, k])[-size:].tolist())
        out[k] = len(worst_k & worst_final) / size
    return out


def class_bands(
    losses: np.ndarray, finals: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Median and IQR of the loss trace, split by what the design finally scored."""
    n = losses.shape[0]
    size = decile_size(n)
    order = np.argsort(finals)
    groups = {
        "best decile (final)": order[:size],
        "middle 80%": order[size : n - size],
        "worst decile (final)": order[n - size :],
    }
    bands = {}
    for label, index in groups.items():
        if len(index) == 0:
            continue
        subset = losses[index]
        bands[label] = (
            np.median(subset, axis=0),
            np.percentile(subset, 25, axis=0),
            np.percentile(subset, 75, axis=0),
        )
    return bands


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def mark_stages(ax, boundaries: list[int]) -> None:
    """The APGM schedule changes algorithm at these steps, not just stepsize."""
    for edge in boundaries:
        ax.axvline(edge, color=AXIS, linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--npz", type=Path, action="append", default=[],
                    help="a run normalised by scripts/mosaic_pipeline_traces.py, "
                         "in place of driver logs")
    ap.add_argument("--boundaries", default="",
                    help="comma-separated step:label marks, e.g. '90:sharp,120:cycle 2'. "
                         "Defaults to the driver's own 100:sharpen,150:final on a "
                         "165-step schedule.")
    ap.add_argument("--out", type=Path, required=True, help="PNG path")
    ap.add_argument("--csv", type=Path, default=None,
                    help="per-step table, so the numbers exist outside the picture")
    ap.add_argument("--loss-label", default="in-loop loss (9-term, 4 samples)",
                    help="y-axis of panel A; the objective differs between runs")
    ap.add_argument("--score-label", default="final re-fold score",
                    help="what the trajectories are finally judged by")
    ap.add_argument("--note", default=f"steps 0-{FIRST_STEP} omitted (init transient)",
                    help="subtitle clause describing this run's init")
    ap.add_argument("--k", type=int, default=25,
                    help="the candidate culling step to call out on the figure")
    args = ap.parse_args()

    runs = []
    for path in sorted(args.logs):
        losses, finals, sec_per_step = parse_log(path)
        if losses.shape[0] < 8:
            print(f"  {path.name}: only {losses.shape[0]} trajectories, skipped",
                  file=sys.stderr)
            continue
        runs.append((path.name, losses, finals, sec_per_step))
    for path in sorted(args.npz):
        payload = np.load(path)
        # No per-design timing survives normalisation, so the reclaimed-GPU
        # figure is reported from the driver's measured 2.43 s/step instead.
        runs.append((path.stem, payload["losses"], payload["finals"], 2.43))

    if not runs:
        raise SystemExit("no log had enough complete trajectories to analyse")

    steps = min(losses.shape[1] for _, losses, _, _ in runs)
    if len({losses.shape[1] for _, losses, _, _ in runs}) > 1:
        print(f"  runs disagree on step count; truncating all to {steps}",
              file=sys.stderr)

    rhos = np.array([rank_curve(losses[:, :steps], finals) for _, losses, finals, _ in runs])
    recalls = np.array([recall_curve(losses[:, :steps], finals) for _, losses, finals, _ in runs])
    pooled_rho = rhos.mean(axis=0)
    pooled_recall = recalls.mean(axis=0)

    total = sum(losses.shape[0] for _, losses, _, _ in runs)
    chance = float(np.mean([decile_size(l.shape[0]) / l.shape[0] for _, l, _, _ in runs]))

    print(f"\n{len(runs)} runs, {total} complete trajectories, {steps} steps each")
    print(f"{'run':<34} {'n':>4} {'rho@k':>7} {'recall@k':>9} {'final sd':>9}")
    for (name, losses, finals, _), rho, recall in zip(runs, rhos, recalls):
        print(f"{name:<34} {losses.shape[0]:>4} {rho[args.k]:>7.3f} "
              f"{recall[args.k]:>9.2f} {np.std(finals):>9.4f}")
    print(f"{'POOLED':<34} {total:>4} {pooled_rho[args.k]:>7.3f} "
          f"{pooled_recall[args.k]:>9.2f}")
    print(f"chance recall for a decile-on-decile test: {chance:.2f}")

    best_k = int(np.argmax(pooled_recall[FIRST_STEP:])) + FIRST_STEP
    print(f"pooled recall peaks at step {best_k} ({pooled_recall[best_k]:.2f})")

    reclaimed = sum(
        decile_size(losses.shape[0]) * (steps - args.k) * sec
        for _, losses, _, sec in runs
    )
    print(f"culling the bottom decile at step {args.k} would free "
          f"{reclaimed / 3600:.1f} GPU-hours across these {len(runs)} runs\n")

    if args.csv is not None:
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "pooled_spearman_rho", "pooled_decile_recall",
                             *[f"rho_{name}" for name, _, _, _ in runs]])
            for k in range(steps):
                writer.writerow([k, f"{pooled_rho[k]:.4f}", f"{pooled_recall[k]:.4f}",
                                 *[f"{rho[k]:.4f}" for rho in rhos]])
        print(f"wrote {args.csv}")

    # Where the optimizer changes algorithm, which is also where a cull would
    # naturally fire. The driver's own schedule is the default; a pipeline run
    # with soft/sharp cycles passes its own.
    if args.boundaries:
        marks = [piece.split(":", 1) for piece in args.boundaries.split(",")]
        boundaries = [int(step) for step, _ in marks]
        labels = [label for _, label in marks]
    elif steps == 165:
        boundaries, labels = [100, 150], ["sharpen", "final"]
    else:
        boundaries, labels = [], []
    window = slice(FIRST_STEP, steps)
    x = np.arange(steps)[window]

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 10.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1, 1], "hspace": 0.22})
    fig.patch.set_facecolor(SURFACE)

    # -- Panel A: do the eventual losers separate early, and when? ------------
    ax = axes[0]
    style_axes(ax)
    mark_stages(ax, boundaries)
    reference = max(runs, key=lambda run: run[1].shape[0])
    name, losses, finals, _ = reference
    spec = {
        "best decile (final)": (BLUE, "-"),
        "middle 80%": (INK_MUTED, (0, (5, 2))),
        "worst decile (final)": (RED, (0, (1.5, 1.5))),
    }
    ends = []
    for label, (median, low, high) in class_bands(losses[:, :steps], finals).items():
        color, dash = spec[label]
        ax.fill_between(x, low[window], high[window], color=color, alpha=0.16,
                        linewidth=0, zorder=2)
        ax.plot(x, median[window], color=color, linewidth=2.0, linestyle=dash,
                zorder=4, label=label)
        ends.append([median[window][-1], label, color])
    # The best decile and the middle 80% converge to the same in-loop loss --
    # which is the finding, not a glitch -- so their end labels would print on
    # top of each other. Push them apart by a minimum gap in data units.
    ends.sort()
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    gap = 0.052 * span
    for index in range(1, len(ends)):
        ends[index][0] = max(ends[index][0], ends[index - 1][0] + gap)
    for y_text, label, color in ends:
        ax.annotate(label, (x[-1], y_text), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=8.5,
                    va="center")
    ax.set_ylabel(args.loss_label, color=INK_SECONDARY, fontsize=9.5)
    ax.set_title(
        f"Trajectories grouped by the score they finally earned  ·  {name}, "
        f"n={losses.shape[0]}",
        color=INK, fontsize=11, loc="left", pad=8)
    # Lower left: the curves are still high there, so the box sits on empty
    # surface instead of over the stage markers along the top.
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="lower left")

    # -- Panel B: does the step-k rank carry final-score information? ---------
    ax = axes[1]
    style_axes(ax)
    mark_stages(ax, boundaries)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    for rho in rhos:
        ax.plot(x, rho[window], color=INK_MUTED, linewidth=1.0, alpha=0.4, zorder=2)
    ax.plot(x, pooled_rho[window], color=BLUE, linewidth=2.0, zorder=4,
            label=f"pooled mean of {len(runs)} runs" if len(runs) > 1 else f"{total} trajectories")
    ax.set_ylabel(f"Spearman rho vs {args.score_label}", color=INK_SECONDARY, fontsize=9.5)
    ax.set_title("Rank agreement between the loss at step k and the "
                 f"{args.score_label}",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="lower right")

    # -- Panel C: the metric a culling scheme actually needs ------------------
    ax = axes[2]
    style_axes(ax)
    mark_stages(ax, boundaries)
    for recall in recalls:
        ax.plot(x, recall[window], color=INK_MUTED, linewidth=1.0, alpha=0.4, zorder=2)
    ax.plot(x, pooled_recall[window], color=BLUE, linewidth=2.0, zorder=4,
            label=f"pooled mean of {len(runs)} runs" if len(runs) > 1 else f"{total} trajectories")
    ax.axhline(chance, color=RED, linewidth=1.5, linestyle=(0, (4, 2.5)), zorder=3,
               label=f"chance ({chance:.2f})")
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall of the final worst decile", color=INK_SECONDARY, fontsize=9.5)
    ax.set_xlabel("optimization step", color=INK_SECONDARY, fontsize=9.5)
    ax.set_title("Would culling the bottom decile at step k kill the right trajectories?",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="upper left")

    for ax in axes:
        ax.set_xlim(FIRST_STEP, steps - 1)
        if boundaries:
            for edge, text in zip(boundaries, labels):
                ax.annotate(text, (edge, 1.0), xytext=(3, -10),
                            textcoords="offset points", xycoords=("data", "axes fraction"),
                            color=INK_MUTED, fontsize=8)
    axes[2].annotate(f"k = {args.k}", (args.k, 0), xytext=(0, 6),
                     textcoords="offset points", color=INK_SECONDARY, fontsize=8.5,
                     ha="center")
    for ax in axes:
        ax.axvline(args.k, color=INK_SECONDARY, linewidth=1.0, alpha=0.5, zorder=1)

    fig.suptitle("Can a mosaic trajectory be written off early?",
                 color=INK, fontsize=13.5, x=0.012, ha="left", y=1.005,
                 fontweight="bold")
    fig.text(0.012, 0.978,
             f"{total} complete trajectories, {len(runs)} "
             f"run{'s' if len(runs) > 1 else ''}  ·  {args.note}  ·  "
             "lower loss is better",
             color=INK_MUTED, fontsize=9, ha="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
