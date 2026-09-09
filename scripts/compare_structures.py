#!/usr/bin/env python3
"""How similar are the scorers' predictions, and does every one of them run?

This is a **harness check**, not a scoring result. It answers two questions
about the wiring: does each model actually produce a structure for each design,
and how close are those structures to one another. It says nothing about which
designs are good -- the design sets it runs on are small and arbitrary, and no
ranking should be read out of it.

Similarity is measured where it matters for a binder: superpose two predictions
of the same design on the target chain, then take the RMSD over the binder's
alpha carbons. Low means two models put the binder in the same place on the
target; high means they disagree about the pose, which a confidence number
cannot show because each model is separately confident.

Superposition is on the target, deliberately. Superposing on the whole complex
would let a well-predicted binder fold hide a completely different docking
site, which is the disagreement worth seeing.

Usage:
    python scripts/compare_structures.py --root build-logs/verify \\
        --out build-logs/model-agreement.png
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_chains(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """chain index -> (CA coordinates, residue ids), for protein chains."""
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from biotite.structure.io.pdb import PDBFile

    if path.suffix == ".cif":
        atoms = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    else:
        atoms = PDBFile.read(path).get_structure(model=1)
    atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.atom_name == "CA")]

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, chain_id in enumerate(sorted(set(atoms.chain_id))):
        sel = atoms[atoms.chain_id == chain_id]
        out[index] = (sel.coord.astype(np.float64), sel.res_id.copy())
    return out


def split_target_binder(chains, target_length: int):
    """Identify which chain is the target by length, not by name.

    The two drivers disagree on chain order -- the mosaic driver builds
    [binder, target] and the Chai driver writes [target, binder] so the target
    gets chain A, matching `general.target.chain_id`. Keying on the letter
    would silently swap them for one of the two.
    """
    target = binder = None
    for coords, _ in chains.values():
        if abs(len(coords) - target_length) <= 2 and target is None:
            target = coords
        else:
            binder = coords
    return target, binder


def superpose_on_target(mobile_t, mobile_b, ref_t, ref_b):
    """Kabsch on the target, applied to the binder. Returns binder RMSD."""
    n = min(len(mobile_t), len(ref_t))
    if n < 3:
        return None
    a, b = mobile_t[:n], ref_t[:n]
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a_c.T @ b_c)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt

    m = min(len(mobile_b), len(ref_b))
    if m < 3:
        return None
    moved = (mobile_b[:m] - a.mean(0)) @ rot + b.mean(0)
    return float(np.sqrt(((moved - ref_b[:m]) ** 2).sum(-1).mean()))


def collect(root: Path, target_length: int) -> dict[str, dict[str, Path]]:
    """model -> {design key -> structure path}, first replicate only.

    The model name is the first path component *below the root*, not the
    directory before some fixed keyword: the run tree is
    `<root>/<model>/structures/<condition>/...`, and keying on the word
    "structures" picks up the root's own name when the root happens to be
    called that too.
    """
    found: dict[str, dict[str, Path]] = defaultdict(dict)
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".pdb", ".cif") or not path.is_file():
            continue
        rel = path.resolve().relative_to(root)
        if len(rel.parts) < 2:
            continue
        model = rel.parts[0]
        parts = rel.parts

        # mosaic scorer: <model>/structures/<condition>/design-XXXXXX_s0.pdb
        if "structures" in parts:
            if "complex" not in parts:
                continue
            stem = path.stem
            if "_s" in stem and not stem.endswith("_s0"):
                continue
            found[model][stem.split("_s")[0]] = path
        # Chai-1: <model>/folds/design-XXXXXX/pred.model_idx_0.cif
        elif "folds" in parts and path.name.startswith("pred.model_idx_0"):
            found[model][path.parent.name] = path
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-length", type=int, default=201)
    args = ap.parse_args()

    found = collect(args.root, args.target_length)
    models = sorted(found)
    if len(models) < 2:
        print(f"need structures from at least two models, found {models}")
        return 1

    cache: dict[tuple[str, str], tuple] = {}
    for model in models:
        for key, path in found[model].items():
            try:
                t, b = split_target_binder(load_chains(path), args.target_length)
                if t is not None and b is not None:
                    cache[(model, key)] = (t, b)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {model}/{key}: {type(exc).__name__}: {exc}")

    designs = sorted({key for _, key in cache})
    print(f"{len(models)} models x {len(designs)} designs -> {len(cache)} structures")

    pairwise: dict[tuple[str, str], list[float]] = defaultdict(list)
    for a, b in itertools.combinations(models, 2):
        for key in designs:
            if (a, key) in cache and (b, key) in cache:
                at, ab_ = cache[(a, key)]
                bt, bb = cache[(b, key)]
                rmsd = superpose_on_target(at, ab_, bt, bb)
                if rmsd is not None:
                    pairwise[(a, b)].append(rmsd)

    if not pairwise:
        print("no comparable pairs")
        return 1
    plot(models, designs, pairwise, args.out)
    return 0


def plot(models, designs, pairwise, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#16201f", "#71847f", "#dae4e2"
    n = len(models)
    matrix = np.full((n, n), np.nan)
    for (a, b), values in pairwise.items():
        i, j = models.index(a), models.index(b)
        matrix[i, j] = matrix[j, i] = float(np.median(values))
    np.fill_diagonal(matrix, 0.0)

    # Height follows the number of pairs; 21 rows at seven models needs more
    # than one at two.
    height = max(5.0, 1.05 + 0.34 * len(pairwise))
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13.2, height), gridspec_kw={"width_ratios": [1.05, 1]}
    )
    fig.patch.set_facecolor("white")

    # One hue, light to dark: this is a magnitude, not a polarity.
    im = ax.imshow(matrix, cmap="BuPu", vmin=0)
    ax.set_xticks(range(n), models, rotation=35, ha="right", fontsize=10, color=ink)
    ax.set_yticks(range(n), models, fontsize=10, color=ink)
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                shade = "white" if matrix[i, j] > np.nanmax(matrix) * 0.6 else ink
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                        fontsize=10, color=shade)
    ax.set_title("Median binder RMSD between two models, same design (Å)",
                 fontsize=11.5, color=ink, pad=12, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=9)
    # The matrix keeps a square aspect while the pair list grows with the model
    # count, so without this the matrix floats in the middle of a tall column.
    ax.set_anchor("N")
    for spine in ax.spines.values():
        spine.set_visible(False)

    labels = [f"{a} / {b}" for a, b in pairwise]
    order = np.argsort([np.median(v) for v in pairwise.values()])
    labels = [labels[i] for i in order]
    series = [list(pairwise.values())[i] for i in order]
    for row, values in enumerate(series):
        bx.scatter(values, [row] * len(values), s=46, color="#0f6b64",
                   alpha=.75, edgecolor="white", linewidth=.8, zorder=3)
        bx.plot([min(values), max(values)], [row, row], color=grid, lw=2, zorder=1)
    # A yardstick, not decoration: below roughly 2 A two predictions are the
    # same binding mode, and above it they are different hypotheses about where
    # the binder goes, however confident either model was.
    bx.axvline(2.0, color="#8a4550", lw=1.2, ls="--", zorder=2)
    bx.text(2.0, len(labels) - 0.35, "  2 Å — same binding mode",
            color="#8a4550", fontsize=9, va="top")
    bx.set_xlim(left=0)
    bx.set_yticks(range(len(labels)), labels, fontsize=9.5, color=ink)
    bx.set_xlabel("binder RMSD (Å), one point per design", fontsize=10, color=muted)
    bx.set_title("Every model pair, every design", fontsize=11.5, color=ink,
                 pad=12, loc="left")
    bx.grid(axis="x", color=grid, lw=.8)
    bx.set_axisbelow(True)
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.spines["bottom"].set_color(grid)
    bx.tick_params(colors=muted, labelsize=9)

    fig.suptitle("How similar are the scorers\u2019 predictions?",
                 fontsize=15, color=ink, x=.008, ha="left", y=.985, weight="bold")
    fig.text(.008, .945,
             "Harness check: every model folds the same designs, and this is how "
             "far apart the results are. Not a ranking of designs.",
             fontsize=10.5, color=muted, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .925))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
