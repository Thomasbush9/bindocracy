#!/usr/bin/env python3
"""How well does each scorer separate binders from non-binders?

Reads the six existing `acc_<model>_<target>.csv` files from the mosaic
benchmark and the `metrics.jsonl` shards this harness produces for the newer
models, puts both on the same footing, and plots AUC per model.

AUC here is Mann-Whitney U / (n_pos * n_neg), with averaged ranks for ties --
transcribed from `mosaic/benchmark/common.py::auc` rather than reimplemented,
because several confidence scores saturate near 1.0 on easy designs and a naive
implementation rewards that.

The bar that matters is not 0.5. It is the best **sequence-only control** --
net charge, length, molecular weight -- which cannot know anything about the
target. A model that does not clear that has not earned its GPU time.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

AA_MASS = dict(zip(
    "ARNDCQEGHILKMFPSTWYV",
    [71.08, 156.19, 114.10, 115.09, 103.14, 128.13, 129.12, 57.05, 137.14,
     113.16, 113.16, 128.17, 131.19, 147.18, 97.12, 87.08, 101.10, 186.21,
     163.18, 99.13],
    strict=False,
))

# Direction per metric: True when larger is better. PAE is stored raw.
HIGHER_IS_BETTER = {
    "iptm": True, "bt_iptm": True, "tb_iptm": True, "iptm_min": True,
    "ipsae_min": True, "bt_ipsae": True, "tb_ipsae": True,
    "complex_ptm": True, "binder_ptm": True, "aggregate_score": True,
    "complex_plddt": True, "binder_plddt": True, "iplddt": True,
    "bt_pae": False, "tb_pae": False, "ptm_energy": False,
    "has_clashes": False, "n_clashing_chain_pairs": False,
    "binder_intra_clashes": False,
}


def auc(pos, neg) -> float:
    """Mann-Whitney U / (n_pos * n_neg). Ties get averaged ranks."""
    if not len(pos) or not len(neg):
        return float("nan")
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    vals = [v for v, _ in merged]
    ranks, i = [0.0] * len(vals), 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    r_pos = sum(ranks[k] for k, (_, lab) in enumerate(merged) if lab == 1)
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_auc(values: dict[int, float], labels: dict[int, int], higher: bool) -> float:
    pos = [v for i, v in values.items() if labels.get(i) == 1 and np.isfinite(v)]
    neg = [v for i, v in values.items() if labels.get(i) == 0 and np.isfinite(v)]
    a = auc(pos, neg)
    return a if higher else (1 - a if np.isfinite(a) else a)


def read_existing(
    scores_dir: Path, target: str, tag: str
) -> dict[str, dict[str, dict[int, float]]]:
    """model -> metric -> {design_idx: value}, from the mosaic benchmark CSVs.

    `tag` selects which run. It matters: `acc_` and `macc_` hold different
    protocols, and only `macc_` (matched compute -- same trunk passes and same
    diffusion steps for every backend) reproduces the AUCs the benchmark
    reports. Reading `acc_` instead gives Protenix 0.552 where the report says
    0.680, with nothing in the output to say why.
    """
    out: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(scores_dir.glob(f"{tag}_*_{target}.csv")):
        model = path.stem[len(tag) + 1:-len(f"_{target}")]
        for row in csv.DictReader(path.open()):
            if row["replicate"] != "0":
                continue
            try:
                out[model][row["metric"]][int(row["design_idx"])] = float(row["value"])
            except ValueError:
                continue
    return out


def read_jsonl(root: Path) -> dict[str, dict[str, dict[int, float]]]:
    """model -> metric -> {design_idx: value}, from this harness's shards.

    The design-set FASTA carries the benchmark's own design_idx in its header,
    so the index the driver reports IS the label key -- no join table needed.
    """
    out: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(root.rglob("metrics.jsonl")):
        model = path.relative_to(root).parts[0]
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("failed") or int(row.get("replicate", 0)) != 0:
                continue
            for key, value in (row.get("metrics") or {}).items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    out[model][key][int(row["index"])] = float(value)
    return out


def controls(sequences: dict[int, str]) -> dict[str, dict[int, float]]:
    """Sequence-only properties that cannot know anything about the target."""
    return {
        "net charge": {i: float(sum(s.count(a) for a in "KR")
                                - sum(s.count(a) for a in "DE"))
                       for i, s in sequences.items()},
        "length": {i: float(len(s)) for i, s in sequences.items()},
        "molecular weight": {i: sum(AA_MASS.get(a, 110.0) for a in s) + 18.02
                             for i, s in sequences.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing", type=Path, required=True,
                    help="mosaic benchmark labelled_scores/ directory")
    ap.add_argument("--new", type=Path, required=True,
                    help="root holding <model>/shard*/metrics.jsonl")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--target", default="nipah-glycoprotein-g")
    ap.add_argument("--tag", default="macc",
                    help="which existing benchmark run to compare against. "
                         "macc = matched compute, which is what the reported "
                         "AUCs come from; acc is an earlier, unmatched run.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    labels: dict[int, int] = {}
    for row in csv.DictReader(args.labels.open()):
        labels[int(row["design_idx"])] = int(row["label"])

    sequences: dict[int, str] = {}
    index = None
    chunks: list[str] = []
    for line in args.fasta.read_text().splitlines():
        if line.startswith(">"):
            if index is not None:
                sequences[index] = "".join(chunks)
            index, chunks = int(line[1:].split()[0]), []
        elif line.strip():
            chunks.append(line.strip())
    if index is not None:
        sequences[index] = "".join(chunks)

    per_model = read_existing(args.existing, args.target, args.tag)
    for model, metrics in read_jsonl(args.new).items():
        per_model[model].update(metrics)

    # Each model is scored by its best single metric, which is how the existing
    # benchmark reports it. Reporting a fixed metric instead would penalise
    # models that do not emit it -- Promera and AF3 have no ipSAE, for one.
    results: dict[str, tuple[float, str, int]] = {}
    for model, metrics in sorted(per_model.items()):
        best = (float("nan"), "", 0)
        for metric, values in metrics.items():
            higher = HIGHER_IS_BETTER.get(metric)
            if higher is None:
                continue
            scored = {i: v for i, v in values.items() if i in labels}
            if len(scored) < 20:
                continue
            a = best_auc(scored, labels, higher)
            if np.isfinite(a) and (not np.isfinite(best[0]) or a > best[0]):
                best = (a, metric, len(scored))
        if np.isfinite(best[0]):
            results[model] = best

    control = {}
    for name, values in controls(sequences).items():
        for higher in (True, False):
            a = best_auc(values, labels, higher)
            if np.isfinite(a):
                control[name] = max(control.get(name, 0.0), a)
    bar = max(control.values()) if control else 0.5
    best_ctl = max(control, key=control.get) if control else "none"

    print(f"baseline tag {args.tag!r}; control bar {bar:.3f} ({best_ctl}); "
          f"{sum(labels.values())} binders of {len(labels)}")
    for model, (a, metric, n) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"  {model:16s} {a:.3f}  ({metric}, n={n})")

    plot(results, control, bar, args.out, args.target, labels)
    return 0


def plot(results, control, bar, out: Path, target: str, labels) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#16201f", "#71847f", "#dae4e2"
    ordered = sorted(results.items(), key=lambda kv: kv[1][0])
    names = [m for m, _ in ordered]
    values = [v[0] for _, v in ordered]
    metrics = [v[1] for _, v in ordered]
    # New models are the ones this harness produced; the other six come from
    # the existing benchmark CSVs.
    new = {"promera", "chai1", "af3", "protenix_base"}

    fig, ax = plt.subplots(figsize=(10.5, 0.52 * len(names) + 3.0))
    fig.patch.set_facecolor("white")
    colors = ["#0f6b64" if m in new else "#9fb8b4" for m in names]
    ax.barh(names, values, color=colors, height=.62, zorder=3)
    for i, (v, met) in enumerate(zip(values, metrics, strict=False)):
        ax.text(v + .006, i, f"{v:.3f}", va="center", fontsize=10, color=ink)
        ax.text(.505, i, met, va="center", fontsize=8.6, color="white"
                if v > 0.58 else muted, zorder=4)

    ax.axvline(bar, color="#8a4550", lw=1.4, ls="--", zorder=5)
    ax.text(bar + .004, len(names) - .35,
            f"  best sequence-only control: {bar:.3f}",
            color="#8a4550", fontsize=9.5, va="top")
    ax.axvline(.5, color=muted, lw=1, zorder=2)
    ax.text(.5 + .004, -.85, "chance", color=muted, fontsize=9)

    ax.set_xlim(.5, max(max(values), bar) + .06)
    ax.set_xlabel("AUC — separating binders from non-binders (best single metric)",
                  fontsize=10.5, color=muted)
    ax.grid(axis="x", color=grid, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, labelsize=10)
    for tick, name in zip(ax.get_yticklabels(), names, strict=False):
        tick.set_color(ink)
        if name in new:
            tick.set_fontweight("bold")

    n_pos = sum(labels.values())
    fig.suptitle(f"Does any scorer beat net charge?  ({target})",
                 fontsize=15, color=ink, x=.01, ha="left", y=.98, weight="bold")
    fig.text(.01, .935,
             f"{len(labels)} designs, {n_pos} binders. Teal = added by this "
             f"harness. A model below the dashed line has not earned its GPU time.",
             fontsize=10.5, color=muted, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .90))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
