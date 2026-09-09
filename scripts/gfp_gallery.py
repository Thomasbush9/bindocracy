#!/usr/bin/env python3
"""One GFP fold per model, drawn side by side.

GFP is the check that needs no labels and no controls: it is an 11-strand beta
barrel with a central helix, its structure has been known since 1996, and every
one of these models has seen thousands of them. If a model returns something
that is not a barrel, the wiring is wrong -- which is a different and much more
basic question than whether a model ranks binders well.

Each panel is the alpha-carbon trace, coloured along the chain from N to C so
the barrel's strand order is visible. Beneath each is the model's own confidence
and the RMSD to the panel with the highest confidence, which stands in for a
reference structure without needing to fetch one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_ca(path: Path) -> np.ndarray:
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from biotite.structure.io.pdb import PDBFile

    if path.suffix == ".cif":
        atoms = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    else:
        atoms = PDBFile.read(path).get_structure(model=1)
    atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.atom_name == "CA")]
    return atoms.coord.astype(np.float64)


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n] - a[:n].mean(0), b[:n] - b[:n].mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ rot - b) ** 2).sum(-1).mean()))


def collect(root: Path) -> dict[str, tuple[Path, float | None]]:
    """model -> (structure path, mean pLDDT if the driver reported one)."""
    found: dict[str, tuple[Path, float | None]] = {}
    for status in sorted(root.rglob("metrics.jsonl")):
        model = status.relative_to(root).parts[0]
        plddt = None
        for line in status.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("failed"):
                continue
            values = row.get("metrics") or {}
            plddt = values.get("mono_plddt") or values.get("complex_plddt")
            break
        candidates = [
            p for p in sorted((status.parent).rglob("*"))
            if p.suffix in (".pdb", ".cif") and p.is_file()
        ]
        if candidates:
            found[model] = (candidates[0], plddt)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    found = collect(args.root)
    if not found:
        print(f"no structures under {args.root}")
        return 1

    traces = {}
    for model, (path, plddt) in sorted(found.items()):
        try:
            traces[model] = (load_ca(path), plddt)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {model}: {type(exc).__name__}: {exc}")
    if not traces:
        return 1

    # pLDDT scales differ: mosaic reports 0-1, AF3 and Chai report 0-100.
    normalised = {
        m: (c, (p * 100 if p is not None and p <= 1.0 else p))
        for m, (c, p) in traces.items()
    }
    reference = max(normalised, key=lambda m: normalised[m][1] or -1)
    print(f"{len(normalised)} models; reference = {reference} "
          f"(pLDDT {normalised[reference][1]:.1f})")
    for model, (coords, plddt) in sorted(normalised.items()):
        rmsd = kabsch_rmsd(coords, normalised[reference][0])
        print(f"  {model:16s} {len(coords):>4d} CA  pLDDT "
              f"{plddt if plddt is not None else float('nan'):>6.1f}  "
              f"RMSD to {reference} {rmsd:5.2f} A")
    plot(normalised, reference, args.out)
    return 0


def plot(traces, reference, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#16201f", "#71847f"
    models = sorted(traces)
    cols = min(5, len(models))
    rows = -(-len(models) // cols)
    fig = plt.figure(figsize=(3.5 * cols, 3.9 * rows + 1.0))
    fig.patch.set_facecolor("white")

    for i, model in enumerate(models, start=1):
        coords, plddt = traces[model]
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        # Centre and orient every panel the same way, on the reference, so the
        # barrels are comparable by eye rather than by luck of the frame.
        ref = traces[reference][0]
        n = min(len(coords), len(ref))
        c = coords[:n] - coords[:n].mean(0)
        r = ref[:n] - ref[:n].mean(0)
        u, _, vt = np.linalg.svd(c.T @ r)
        d = np.sign(np.linalg.det(u @ vt))
        c = c @ (u @ np.diag([1.0, 1.0, d]) @ vt)

        t = np.linspace(0, 1, len(c))
        for j in range(len(c) - 1):
            ax.plot(*c[j:j + 2].T, color=plt.cm.viridis(t[j]), lw=1.9)
        ax.set_axis_off()
        span = np.abs(c).max() * 0.95
        ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(-span, span)
        rmsd = kabsch_rmsd(coords, ref)
        confidence = f"pLDDT {plddt:.0f}" if plddt is not None else "pLDDT —"
        label = (f"{model}\n{confidence}   "
                 + ("reference" if model == reference else f"RMSD {rmsd:.1f} Å"))
        ax.set_title(label, fontsize=10.5, color=ink, pad=-2)

    fig.suptitle("The same protein, nine times: GFP folded by every scorer",
                 fontsize=15, color=ink, x=.012, ha="left", y=.985, weight="bold")
    fig.text(.012, .945,
             "Alpha-carbon trace, N-terminus dark to C-terminus yellow, all "
             "superposed on the most confident prediction. GFP is an 11-strand "
             "beta barrel; anything that is not one is a wiring failure, not a "
             "modelling opinion.",
             fontsize=10, color=muted, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .915))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=165, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
