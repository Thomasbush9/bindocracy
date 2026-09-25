#!/usr/bin/env python3
"""Is the binder closer to one set of target residues than to another?

The shipped `epitope` built-in answers this for the campaign's own hotspots,
and only for those: `functions/run.py:114-128` reads `general.target.hotspots`
and there is no per-run override. So a campaign that needs a SECOND residue set
measured -- a competing surface, a region to avoid, a control patch -- cannot
get it from the built-in, and its metric names are registry-fixed anyway, so a
second run would collide with the first.

This is that second measurement. Same geometry as the built-in, same heavy-atom
convention, same binder-by-length chain identification -- deliberately, so the
two columns are comparable. What differs is that the residue set arrives as
`--positions` and the metrics carry their own names.

Why it exists here. On DIO3 the N-terminal helix (target residues 1-22) takes
the great majority of interfaces, and confidence correlates with binding it at
r = 0.90 across nine rounds of the predecessor campaign. A design 5 A from a
catalytic hotspot and 2 A from that helix is an interface centred on the helix
with the hotspot at its periphery, and four separate rounds mistook one for the
other. d(site) alone cannot distinguish them; d(site) beside d(attractor) can.

Usage as a custom function:

    schema_version: 1
    name: placement
    tool: function
    design_set: /abs/sets/<digest>.json
    structures_from: <the run whose poses to read>
    function:
      name: placement
      script: /abs/examples/functions/residue_set_distance.py
      inputs: [structure, sequence]
      prefix: source_model      # so two models' poses stay separate columns
      args: ["--positions", "68,99,155", "--against", "1-22"]
      metrics:
        set_offset:     {direction: min, unit: angstrom}
        set_contacts:   {direction: max}
        set_coverage:   {direction: max}
        against_offset: {direction: max, unit: angstrom}
        set_is_nearer:  {direction: max}

**`set_is_nearer` is why `--against` exists.** A filter policy compares a metric
to a VALUE; it cannot compare two metrics to each other. So "closer to the site
than to the competing surface" is not expressible as a threshold on two distance
columns -- but it is expressible as `set_is_nearer >= 1`, which makes the gate a
single stored, versioned rule that `designset build --passed-filter` can select
on. Computing the comparison here, once, beside the distances it comes from, is
what keeps the selection reproducible instead of something done by hand in a
notebook.

Directions are declared in the config, not here, because only the caller knows
which set is the target and which is the one to avoid.
"""

from __future__ import annotations

import argparse
import math
import sys

DEFAULT_CUTOFF = 5.0


def parse_positions(raw: str) -> set[int]:
    """`1-22,45,60-62` -> a 0-based set. 1-based in, because sequences are
    discussed that way; 0-based out, because coordinates are indexed that way.

    Positions are 1-based indices into the TARGET's FASTA -- not author
    numbering and not a chain letter. Both of those were the obvious design and
    both are wrong: chain letters are not portable between drivers, and residue
    ids in a predicted pose are positional rather than the numbering a crystal
    structure carries. The built-in's docstring records why at length.
    """
    positions: set[int] = set()
    for token in (piece.strip() for piece in raw.split(",")):
        if not token:
            continue
        start, _, end = token.partition("-")
        bounds = []
        for value in (start, end or start):
            if not value.lstrip("-").isdigit():
                raise ValueError(
                    f"cannot parse position {token!r}; expected 1-based target "
                    "positions such as 68 or a range such as 1-22. Chain "
                    "letters and author numbering are not accepted."
                )
            number = int(value)
            if number < 1:
                raise ValueError(f"position {number} is not 1-based")
            bounds.append(number)
        low, high = min(bounds), max(bounds)
        positions.update(range(low - 1, high))
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


def metrics_for(path: str, binder_length: int, wanted: set[int],
                cutoff: float) -> dict[str, float]:
    """Contacts and mean nearest-atom distance to `wanted`.

    Copied from the built-in rather than imported: a custom function has to run
    inside whatever container the campaign binds, and importing a sibling
    driver would make this depend on the harness layout. The cost of the copy
    is that the two must be kept in step; the comment in each says so.
    """
    import numpy as np

    atoms = load_structure(path)
    chains = sorted(set(atoms.chain_id))
    if len(chains) < 2:
        raise ValueError(f"expected a complex, found {len(chains)} chain(s)")

    # Binder by LENGTH, not by chain letter: the drivers disagree on chain
    # order -- the mosaic one builds [binder, target] while the co-folding ones
    # write the target as chain A -- so a letter would silently swap them.
    counts = {c: len(set(atoms[atoms.chain_id == c].res_id)) for c in chains}
    binder_chain = min(counts, key=lambda c: abs(counts[c] - binder_length))
    target_chains = [c for c in chains if c != binder_chain]

    binder = atoms[atoms.chain_id == binder_chain]
    target = atoms[np.isin(atoms.chain_id, target_chains)]
    if not len(binder) or not len(target):
        raise ValueError("binder or target chain is empty")

    distances = np.linalg.norm(
        target.coord[:, None, :] - binder.coord[None, :, :], axis=-1
    )
    nearest_per_atom = distances.min(axis=1)

    # Target residues in file order, so the nth distinct residue is FASTA
    # position n. That holds because every driver here folds the target
    # sequence whole and in order.
    ordered = list(dict.fromkeys(int(r) for r in target.res_id))
    position_of = {res_id: i for i, res_id in enumerate(ordered)}

    out_of_range = {p for p in wanted if p >= len(ordered)}
    if out_of_range:
        raise ValueError(
            f"position(s) {sorted(p + 1 for p in out_of_range)} lie beyond the "
            f"{len(ordered)}-residue target; they are 1-based FASTA positions, "
            "not author numbering"
        )

    contacted = {
        position_of[int(res_id)]
        for res_id, nearest in zip(target.res_id, nearest_per_atom)
        if nearest <= cutoff
    }
    hit = wanted & contacted

    wanted_ids = {ordered[p] for p in wanted}
    mask = np.array([int(r) in wanted_ids for r in target.res_id])
    offset = float(distances[mask].min(axis=1).mean()) if mask.any() else float("nan")

    return {
        "set_offset": offset,
        "set_contacts": float(len(hit)),
        "set_coverage": len(hit) / len(wanted) if wanted else float("nan"),
    }


def main() -> int:
    from bindocracy_io import RejectCandidate, run_scoring

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positions",
        required=True,
        help="the set of interest: 1-based TARGET FASTA positions/ranges, e.g. 68,99,155",
    )
    parser.add_argument(
        "--against",
        default="",
        help=(
            "an optional competing set, same format. When given, "
            "`against_offset` and `set_is_nearer` are reported too."
        ),
    )
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF,
                        help="heavy-atom contact distance in angstroms")
    options, _ = parser.parse_known_args()
    wanted = parse_positions(options.positions)
    against = parse_positions(options.against) if options.against else set()
    overlap = wanted & against
    if overlap:
        # Not fatal, but it makes `set_is_nearer` mean less than it looks like
        # it means, so it is said out loud rather than absorbed.
        print(f"warning: {len(overlap)} position(s) are in both sets",
              file=sys.stderr)
    if not wanted:
        # Not a silent no-op: an empty set would report NaN for every design and
        # look like a measurement that ran.
        print("--positions selected no residues", file=sys.stderr)
        return 2

    def score(candidate, args):
        structure = candidate["structure"]
        binder_length = len(candidate.get("sequence") or "")
        try:
            values = metrics_for(structure, binder_length, wanted, args.cutoff)
            if against:
                rival = metrics_for(structure, binder_length, against, args.cutoff)
                values["against_offset"] = rival["set_offset"]
                # The comparison, computed here so a filter can threshold it.
                # NaN on either side means the question was not answered, not
                # that the answer is no -- so it is dropped below rather than
                # reported as 0.
                if not (math.isnan(values["set_offset"])
                        or math.isnan(rival["set_offset"])):
                    values["set_is_nearer"] = float(
                        values["set_offset"] < rival["set_offset"]
                    )
        except ValueError as error:
            raise RejectCandidate(str(error)) from error
        # An intentionally absent measurement is not a zero.
        return {k: v for k, v in values.items() if not math.isnan(v)}

    return run_scoring(score, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
