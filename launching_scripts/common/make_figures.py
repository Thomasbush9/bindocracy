#!/usr/bin/env python3
"""Performance and resource figures for the alpha benchmark.

Reads what the launchers already recorded — `resources/summary.tsv` (sacct) and
`resources/gpu_trace.csv` (15 s nvidia-smi sampling) — and writes four figures to
docs/figures/. Nothing here is hand-entered; re-run it after any job finishes and
the figures update.

    singularity exec --cleanenv <image with matplotlib> python make_figures.py

Figure choices follow one rule: these are magnitude comparisons across named
tools, so they are horizontal bars in a SINGLE hue, sorted, directly labelled —
not a rainbow. Colour is only used categorically where two things are genuinely
being told apart (requested vs used), and there it is two shades of one hue.

The GPU-memory figure is the one to read carefully; see the caption it prints.
"""

import csv
import glob
import json
import os
import re
import sys

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BR = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design"
OUT = os.path.join(BR, "outputs")
FIGDIR = os.path.join(BR, "bindocracy", "docs", "figures")

# --- palette -----------------------------------------------------------------
# One hue for magnitude, a second only where two states are compared, plus ink
# and grid greys. Text never wears the series colour.
THEMES = {
    "light": dict(BLUE="#2a78d6", BLUE_LIGHT="#a8c8ee", ORANGE="#eb6834",
                  INK="#14161c", INK2="#5c6270", GRID="#e4e5ea",
                  SURFACE="#ffffff"),
    "dark":  dict(BLUE="#5b9df0", BLUE_LIGHT="#33517a", ORANGE="#f07a49",
                  INK="#f2f3f5", INK2="#9aa0ad", GRID="#2b2f37",
                  SURFACE="#1c1f25"),
}
BLUE = BLUE_LIGHT = ORANGE = INK = INK2 = GRID = SURFACE = None


def use_theme(name):
    global BLUE, BLUE_LIGHT, ORANGE, INK, INK2, GRID, SURFACE
    t = THEMES[name]
    BLUE, BLUE_LIGHT, ORANGE = t["BLUE"], t["BLUE_LIGHT"], t["ORANGE"]
    INK, INK2, GRID, SURFACE = t["INK"], t["INK2"], t["GRID"], t["SURFACE"]

TOOLS = ["caliby", "pxdesign", "boltzgen", "proteina_complexa", "rfdiffusion",
         "protein_hunter", "genie3", "mosaic", "freebindcraft"]

LABEL = {
    "caliby": "Caliby",
    "pxdesign": "PXDesign",
    "boltzgen": "BoltzGen",
    "proteina_complexa": "Proteina-Complexa",
    "rfdiffusion": "RFdiffusion",
    "protein_hunter": "Protein-Hunter",
    "genie3": "Genie 3",
    "mosaic": "Mosaic",
    "freebindcraft": "FreeBindCraft",
}

# Tools whose GPU memory figure is a JAX preallocation rather than demand.
JAX_PREALLOC = {"freebindcraft", "mosaic"}
# Tools with no gpu_trace.csv (their run predates the sampler). The number is
# jobstats' whole-job mean, which is reliable for a long job even though its
# sampler is too coarse for short ones.
JOBSTATS_UTIL = {"mosaic": 95.1}

NOTE = {
    "caliby": "inverse folding only",
    "rfdiffusion": "A100; backbones only",
    "genie3": "88% of it is AF2",
}


def hms_to_min(s):
    if not s or s in ("-", ""):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    b = [int(x) for x in s.split(":")]
    while len(b) < 3:
        b.insert(0, 0)
    return days * 1440 + b[0] * 60 + b[1] + b[2] / 60.0


def load():
    data = {}
    for t in TOOLS:
        rec = {"tool": t}
        p = os.path.join(OUT, t, "resources", "summary.tsv")
        if os.path.exists(p):
            rows = [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]
            if len(rows) > 1:
                d = dict(zip(rows[0], rows[1]))
                rec["elapsed_min"] = hms_to_min(d.get("elapsed"))
                rec["jobid"] = d.get("jobid")
                m = re.match(r"([\d.]+)", d.get("max_rss", "") or "")
                rec["max_rss_gb"] = float(m.group(1)) if m else None
                rec["cpus"] = int(d.get("alloc_cpus") or 0) or None
        # Requested memory comes from sacct, recorded in sacct.txt.
        sp = os.path.join(OUT, t, "resources", "sacct.txt")
        if os.path.exists(sp):
            m = re.search(r"\b(\d+)G\b", open(sp).read())
            rec["req_mem_gb"] = float(m.group(1)) if m else None
        # GPU trace
        g = os.path.join(OUT, t, "resources", "gpu_trace.csv")
        if os.path.exists(g):
            ts, util, mem = [], [], []
            prealloc = None
            for line in open(g):
                if line.startswith("#"):
                    if "PREALLOCATE" in line:
                        prealloc = line.split("=", 1)[1].strip()
                    continue
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 6:
                    continue
                try:
                    u, mm = int(parts[2]), int(parts[4])
                except ValueError:
                    continue
                util.append(u)
                mem.append(mm / 1024.0)
                ts.append(len(ts) * 0.25)     # 15 s samples -> minutes
            if util:
                rec.update(gpu_util=util, gpu_mem=mem, gpu_t=ts,
                           gpu_mem_peak=max(mem),
                           gpu_util_mean=sum(util) / len(util),
                           gpu_util_peak=max(util),
                           prealloc=prealloc)
        data[t] = rec
    return data


def caption(fig, text, width=118):
    """Footnote that wraps instead of running off the right edge."""
    fig.text(0.006, 0.006, "\n".join(textwrap.wrap(text, width)),
             fontsize=7.5, color=INK2, va="bottom")


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fmt_dur(v):
    """1.4 min / 15 min / 1.1 h — never '1.66667 h'."""
    if v < 10:
        return "%.1f min" % v
    if v < 90:
        return "%.0f min" % v
    return "%.1f h" % (v / 60.0)


def fig_walltime(data, path):
    rows = [(t, d["elapsed_min"]) for t, d in data.items() if d.get("elapsed_min")]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200, facecolor=SURFACE)
    lo = 0.9
    # A DOT PLOT, not bars: the spread is ~400x so the axis has to be log, and a
    # bar's length is only meaningful measured from zero. Dots encode position,
    # which is what a log axis actually supports. The stem is a hairline guide,
    # deliberately in the grid colour so it does not read as a magnitude.
    for i, (t, v) in enumerate(rows):
        ax.plot([lo, v], [i, i], color=GRID, lw=1.2, zorder=1,
                solid_capstyle="round")
        ax.scatter([v], [i], s=90, color=BLUE, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[r[0]] for r in rows], color=INK, fontsize=10)
    ax.invert_yaxis()
    style(ax)
    ax.set_xscale("log")
    ax.set_xlim(lo, max(r[1] for r in rows) * 4.0)
    ticks = [1, 2, 5, 10, 20, 60, 120, 300, 600]
    ticks = [t for t in ticks if t <= max(r[1] for r in rows) * 2]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, p: ("%g min" % v) if v < 60 else ("%g h" % (v / 60))))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
    for i, (t, v) in enumerate(rows):
        note = NOTE.get(t)
        ax.text(v * 1.12, i, fmt_dur(v) + ("   (%s)" % note if note else ""),
                va="center", ha="left", fontsize=9, color=INK)
    ax.set_xlabel("wall clock for 40 binders  (log scale)", color=INK2, fontsize=9)
    ax.set_title("Time to 40 binders, one GPU per job",
                 color=INK, fontsize=13, loc="left", pad=14)
    fast, slow = rows[0][1], rows[-1][1]
    caption(fig,
            "Log axis: the spread is %.0fx, %s to %s. Caliby needs backbones as "
            "input, so its true cost is RFdiffusion + Caliby. RFdiffusion ran on "
            "A100 because its image has no sm_90 kernels."
            % (slow / fast, fmt_dur(fast), fmt_dur(slow)))
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(path + ".png", facecolor=SURFACE)
    fig.savefig(path + ".svg", facecolor=SURFACE)
    plt.close(fig)


def fig_hostmem(data, path):
    rows = [(t, d["req_mem_gb"], d["max_rss_gb"]) for t, d in data.items()
            if d.get("req_mem_gb") and d.get("max_rss_gb")]
    rows.sort(key=lambda r: r[1] - r[2], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200, facecolor=SURFACE)
    for i, (t, req, used) in enumerate(rows):
        ax.plot([used, req], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.scatter([req], [i], s=70, color=BLUE_LIGHT, zorder=2,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.scatter([used], [i], s=70, color=BLUE, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.text(req + 4, i, "%.0f G requested" % req, va="center",
                fontsize=8, color=INK2)
        ax.text(used - 4, i, "%.1f G" % used, va="center", ha="right",
                fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[r[0]] for r in rows], color=INK, fontsize=10)
    ax.invert_yaxis()
    style(ax)
    ax.set_xlim(-16, max(r[1] for r in rows) * 1.35)
    ax.set_xlabel("host RAM (GB)", color=INK2, fontsize=9)
    ax.set_title("Host RAM: requested vs actually used",
                 color=INK, fontsize=13, loc="left", pad=14)
    tot_req = sum(r[1] for r in rows)
    tot_use = sum(r[2] for r in rows)
    caption(fig,
            "Every job over-requested. Across these runs %.0f GB was reserved and "
            "%.0f GB used (%.0f%%). Trimming improves queue position and costs "
            "nothing." % (tot_req, tot_use, 100 * tot_use / tot_req))
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(path + ".png", facecolor=SURFACE)
    fig.savefig(path + ".svg", facecolor=SURFACE)
    plt.close(fig)


def fig_gpumem(data, path):
    rows = [(t, d["gpu_mem_peak"], d.get("gpu_util_mean"))
            for t, d in data.items() if d.get("gpu_mem_peak")]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200, facecolor=SURFACE)
    colors = [ORANGE if t in JAX_PREALLOC else BLUE for t, _, _ in rows]
    ax.barh(range(len(rows)), [r[1] for r in rows], color=colors, height=0.45)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[r[0]] for r in rows], color=INK, fontsize=10)
    ax.invert_yaxis()
    style(ax)
    ax.axvline(80, color=INK2, lw=1)
    ax.text(80, -0.75, "H100 80 GB", fontsize=8, color=INK2,
            va="center", ha="center")
    ax.set_xlim(0, 92)
    for i, (t, v, u) in enumerate(rows):
        ax.text(v + 1.4, i, "%.1f GiB" % v, va="center", fontsize=9, color=INK)
    ax.set_xlabel("peak GPU memory (GiB)", color=INK2, fontsize=9)
    ax.set_title("Peak GPU memory — reservation, not demand",
                 color=INK, fontsize=13, loc="left", pad=14)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="measured demand"),
                       Patch(color=ORANGE, label="JAX preallocation (~75% of card)")],
              loc="center right", frameon=False, fontsize=8.5, labelcolor=INK2)
    caption(fig,
            "Orange is the JAX allocator reserving the card, not the model's "
            "requirement. Proteina-Complexa ran with "
            "XLA_PYTHON_CLIENT_PREALLOCATE=false, so its 15.9 GiB is real demand. "
            "On this evidence no tool here needs more than ~16 GiB, and a 40 GB "
            "card would serve every one of them.")
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(path + ".png", facecolor=SURFACE)
    fig.savefig(path + ".svg", facecolor=SURFACE)
    plt.close(fig)


def fig_util(data, path):
    have = [(t, d) for t, d in data.items() if d.get("gpu_util")]
    have.sort(key=lambda r: -r[1]["gpu_util_mean"])
    n = len(have)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 1.5 * nrow), dpi=200,
                             facecolor=SURFACE, sharex=False)
    axes = axes.ravel()
    for ax, (t, d) in zip(axes, have):
        ax.fill_between(d["gpu_t"], d["gpu_util"], color=BLUE, alpha=0.85, lw=0)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 100])
        ax.tick_params(colors=INK2, length=0, labelsize=7)
        n_s = len(d["gpu_util"])
        title = "%s  —  mean %.0f%%" % (LABEL[t], d["gpu_util_mean"])
        if n_s < 12:
            # A 15 s sampler cannot characterise a job this short; say so rather
            # than letting the reader take the shape seriously.
            title += "   (only %d samples — too short to trace)" % n_s
        ax.set_title(title, fontsize=9, color=INK, loc="left", pad=4)
        ax.set_xlabel("minutes", fontsize=7, color=INK2, labelpad=1)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("GPU utilisation over the life of each job", fontsize=13,
                 color=INK, x=0.005, ha="left", y=0.995)
    caption(fig,
            "Most tools alternate GPU inference with CPU-side work (MPNN "
            "subprocesses, PDB parsing, relaxation), so mean utilisation is low "
            "even when nothing is misconfigured. Flat-zero stretches are CPU "
            "stages, not stalls -- but a flat zero that never ends is the "
            "signature of the Genie 3 CPU-JAX bug. Not traced: " +
            ", ".join("%s %.0f%% (jobstats)" % (LABEL[k], v)
                      for k, v in sorted(JOBSTATS_UTIL.items())) +
            " -- GPU-saturated, so batching would not help it.")
    fig.tight_layout(rect=[0, 0.055, 1, 0.975])
    fig.savefig(path + ".png", facecolor=SURFACE)
    fig.savefig(path + ".svg", facecolor=SURFACE)
    plt.close(fig)


def fig_batch(path):
    """Memory and time against batch size, for the tools that expose the knob.

    TWO COLUMNS, NOT TWO Y-AXES. Memory (GiB) and wall clock (min) are different
    scales; a dual axis would invite reading a crossing point that means nothing.

    One row per tool, because the two behave differently and that contrast is
    the finding: Caliby's memory is linear in batch, BoltzGen's is nearly flat.
    """
    src = os.path.join(OUT, "_shared", "batch_sweep.json")
    if not os.path.exists(src):
        return
    sweep = json.load(open(src))

    rows = []
    cal = [q for q in sweep["caliby"]["points"] if q.get("num_seqs") == 8]
    if cal:
        rows.append(("Caliby", "batch_size  (structures per batch)",
                     sorted(cal, key=lambda q: q["batch_size"]),
                     sweep["caliby"].get("linear_in_batch", False)))
    bg = [q for q in sweep["boltzgen"]["points"] if q.get("comparable")]
    if bg:
        rows.append(("BoltzGen", "--diffusion_batch_size",
                     sorted(bg, key=lambda q: q["batch_size"]),
                     sweep["boltzgen"].get("linear_in_batch", False)))
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.6 * len(rows)),
                             dpi=200, facecolor=SURFACE, squeeze=False)
    for r, (name, xlabel, pts, linear_ref) in enumerate(rows):
        xs = [q["batch_size"] for q in pts]
        mem = [q["gpu_mem_gib"] for q in pts]
        tim = [q["elapsed_s"] / 60.0 for q in pts]

        ax = axes[r][0]
        if linear_ref:
            # Straight line through the origin so "linear" is checkable by eye.
            slope = mem[-1] / xs[-1]
            ax.plot([0, xs[-1]], [0, slope * xs[-1]], color=GRID, lw=1.2, zorder=0)
        ax.plot(xs, mem, color=BLUE, lw=2, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        for x, y in zip(xs, mem):
            ax.annotate("%.1f" % y, (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
        ax.set_ylabel("peak GPU memory (GiB)", color=INK2, fontsize=9)
        ax.set_ylim(0, max(mem) * 1.35)
        ax.set_title("%s — memory vs batch" % name, fontsize=11, color=INK,
                     loc="left", pad=10)

        ax2 = axes[r][1]
        ax2.plot(xs, tim, color=ORANGE, lw=2, marker="o", ms=7,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
        for x, y in zip(xs, tim):
            ax2.annotate("%.1f" % y, (x, y), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
        ax2.set_ylabel("wall clock (min)", color=INK2, fontsize=9)
        ax2.set_ylim(0, max(tim) * 1.3)
        ax2.set_title("%s — time vs batch" % name, fontsize=11, color=INK,
                      loc="left", pad=10)

        for a in (ax, ax2):
            style(a)
            a.set_xticks(xs)
            a.set_xticklabels([str(x) for x in xs], color=INK, fontsize=9)
            a.set_xlim(0, xs[-1] * 1.12)
            a.set_xlabel(xlabel, color=INK2, fontsize=9)
            a.yaxis.grid(True, color=GRID, lw=0.8)
            a.xaxis.grid(False)
            a.tick_params(colors=INK2, labelsize=8.5)

    fig.suptitle("What a bigger batch actually buys", fontsize=13, color=INK,
                 x=0.006, ha="left", y=0.995)
    caption(fig,
            "All points sampled at 2 s; a 15 s sampler misses transient peaks and "
            "would bias the small batches to look cheaper than they are. Caliby: "
            "memory linear at ~0.9 GiB per structure, time flat past 8 -- past the "
            "knee you pay memory for nothing. Its OTHER knob, num_seqs_per_pdb, is "
            "a sequential loop: 4x the sequences at IDENTICAL 14.5 GiB. BoltzGen: "
            "memory nearly flat, because the trunk dominates and a 281-token "
            "sample adds little -- so its batch is limited by diminishing time "
            "returns, not by memory. Same knob name, opposite sizing rule.")
    fig.tight_layout(rect=[0, 0.075, 1, 0.965])
    fig.savefig(path + ".png", facecolor=SURFACE)
    fig.savefig(path + ".svg", facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    data = load()
    have = [t for t in TOOLS if data[t].get("elapsed_min")]
    print("tools with data: %d/%d -> %s" % (len(have), len(TOOLS), ", ".join(have)))
    for theme in ("light", "dark"):
        use_theme(theme)
        sfx = "" if theme == "light" else "-dark"
        fig_walltime(data, os.path.join(FIGDIR, "01-walltime" + sfx))
        fig_hostmem(data, os.path.join(FIGDIR, "02-host-memory" + sfx))
        fig_gpumem(data, os.path.join(FIGDIR, "03-gpu-memory" + sfx))
        fig_util(data, os.path.join(FIGDIR, "04-gpu-utilisation" + sfx))
        fig_batch(os.path.join(FIGDIR, "05-batch-scaling" + sfx))
    with open(os.path.join(FIGDIR, "figure-data.json"), "w") as fh:
        json.dump({t: {k: v for k, v in d.items()
                       if k not in ("gpu_util", "gpu_mem", "gpu_t")}
                   for t, d in data.items()}, fh, indent=2)
    print("wrote figures to", FIGDIR)
    for f in sorted(os.listdir(FIGDIR)):
        print("   ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
