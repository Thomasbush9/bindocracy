#!/usr/bin/env python3
"""Does the binder actually touch the epitope it was conditioned on?

Run 19 asked exactly this and the harness could not answer it: the tool that
enforces an epitope was the one that appeared to miss it, and there was no
independent check. Conditioning on an epitope is not evidence of contacting
one.

Geometry over a pose already on disk -- no GPU, no model, no re-folding. That
is what makes it cheap enough to backfill across every structure ever saved.

Contacts use a heavy-atom distance cutoff rather than CA-CA: a CA-CA cutoff
generous enough to catch a real side-chain contact also catches residues that
are merely nearby, and the question here is whether atoms touch.
"""

from __future__ import annotations

import argparse
import math
import sys

DEFAULT_CUTOFF = 5.0


def parse_hotspots(raw: str) -> set[int]:
    """`110,112,131` -> {109, 111, 130}: 1-based target positions, 0-based out.

    **Positions in the target's FASTA, not author numbering, and no chain
    letter.** Both of those were the obvious design and both are wrong, which a
    real pose showed immediately:

      * Chain letters are not portable. The mosaic driver builds
        `[binder, target]` so its target is chain B, while the Chai-1 and
        AlphaFold 3 drivers write the target as chain A. A hotspot keyed `A110`
        names the binder in one and the target in the other.
      * Residue ids in a predicted pose are positional and 0-based (`0..200`
        for a 201-residue target), not the author numbering a crystal structure
        carries. `A110` from a reference PDB does not address the same residue.

    So the caller resolves author numbering to target FASTA positions -- which
    the harness already does elsewhere, via `general.target.structure_pdb` --
    and this function takes the result. 1-based on the way in because that is
    how sequences are discussed, 0-based internally because that is what the
    coordinates use.
    """
    positions: set[int] = set()
    for token in (piece.strip() for piece in raw.split(",")):
        if not token:
            continue
        if not token.lstrip("-").isdigit():
            raise ValueError(
                f"cannot parse hotspot {token!r}; expected a 1-based target "
                "position such as 110. Chain letters and author numbering are "
                "not accepted -- see this function's docstring for why."
            )
        value = int(token)
        if value < 1:
            raise ValueError(f"hotspot {value} is not a 1-based position")
        positions.add(value - 1)
    return positions


def load_structure(path: str):
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from biotite.structure.io.pdb import PDBFile

    if path.endswith(".cif"):
        atoms = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    else:
        atoms = PDBFile.read(path).get_structure(model=1)
    return atoms[struc.filter_amino_acids(atoms)]


def metrics_for(path: str, binder_length: int, hotspots: set[int],
                cutoff: float) -> dict[str, float]:
    import numpy as np

    atoms = load_structure(path)
    chains = sorted(set(atoms.chain_id))
    if len(chains) < 2:
        raise ValueError(f"expected a complex, found {len(chains)} chain(s)")

    # Identify the binder by length rather than by chain letter: the drivers
    # disagree on chain order -- the mosaic one builds [binder, target] and the
    # co-folding ones write the target as chain A -- so a letter would silently
    # swap them for some models.
    counts = {c: len(set(atoms[atoms.chain_id == c].res_id)) for c in chains}
    binder_chain = min(counts, key=lambda c: abs(counts[c] - binder_length))
    target_chains = [c for c in chains if c != binder_chain]

    binder = atoms[atoms.chain_id == binder_chain]
    target = atoms[np.isin(atoms.chain_id, target_chains)]
    if not len(binder) or not len(target):
        raise ValueError("binder or target chain is empty")

    # Heavy-atom contacts, target residue by target residue.
    distances = np.linalg.norm(
        target.coord[:, None, :] - binder.coord[None, :, :], axis=-1
    )
    nearest_per_atom = distances.min(axis=1)

    # Target residues, in the order they appear, so a position in the target's
    # FASTA maps to the nth distinct residue of the target chain. That holds
    # because every driver here folds the target sequence whole and in order.
    ordered = list(dict.fromkeys(int(r) for r in target.res_id))
    position_of = {res_id: i for i, res_id in enumerate(ordered)}

    contacted: set[int] = set()
    for res_id, nearest in zip(target.res_id, nearest_per_atom, strict=False):
        if nearest <= cutoff:
            contacted.add(position_of[int(res_id)])

    out_of_range = {h for h in hotspots if h >= len(ordered)}
    if out_of_range:
        raise ValueError(
            f"hotspot position(s) {sorted(p + 1 for p in out_of_range)} lie beyond "
            f"the {len(ordered)}-residue target; they are 1-based FASTA positions, "
            "not author numbering"
        )

    hit = hotspots & contacted

    # Mean distance from the binder to the nearest hotspot atom. NaN when the
    # campaign named no hotspots -- an absent question, not a zero answer.
    offset = float("nan")
    if hotspots:
        wanted_ids = {ordered[p] for p in hotspots}
        mask = np.array([int(r) in wanted_ids for r in target.res_id])
        if mask.any():
            offset = float(distances[mask].min(axis=1).mean())

    return {
        "epitope_coverage": (len(hit) / len(hotspots)) if hotspots else float("nan"),
        "n_epitope_contacts": float(len(hit)),
        "n_interface_residues": float(len(contacted)),
        "epitope_offset": offset,
    }


def main() -> int:
    from bindocracy_io import RejectCandidate, run_scoring

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hotspots", default="",
                   help="comma-separated 1-based TARGET FASTA positions, e.g. "
                        "110,112,131. Not author numbering, and no chain letter "
                        "-- see parse_hotspots for why neither is portable.")
    p.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF,
                   help="heavy-atom contact distance in angstroms")
    # Parse function-specific options once, leaving contract paths to the helper.
    options, _ = p.parse_known_args()
    hotspots = parse_hotspots(options.hotspots)
    if not hotspots:
        print("no hotspots given; coverage and offset will be reported as absent",
              file=sys.stderr)

    def score(candidate, args):
        try:
            values = metrics_for(
                candidate["structure"], len(candidate.get("sequence") or ""),
                hotspots, args.cutoff,
            )
        except ValueError as error:
            raise RejectCandidate(str(error)) from error
        # An intentionally absent measurement is not a zero.
        return {key: value for key, value in values.items() if not math.isnan(value)}

    return run_scoring(score, parser=p)


if __name__ == "__main__":
    raise SystemExit(main())
