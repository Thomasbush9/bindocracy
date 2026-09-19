#!/usr/bin/env python3
"""Sequence-only properties of a binder. No structure, no model, no GPU.

These are the harness's **negative control arm**, which is why they exist at
all. On the labelled Nipah-G set, binder length alone separates binders from
non-binders at AUC 0.642, and five of the nine folding models score within 0.06
of that. A confidence metric that does not beat these has not earned its GPU
time, and until they are stored beside every real metric that comparison cannot
be made without re-deriving them by hand.

Also a worked example of the function contract: JSONL in, JSONL out, and
nothing imported from bindocracy.
"""

from __future__ import annotations

import argparse
import itertools
import json

# Average residue masses in daltons, water excluded. Matches the table the
# mosaic benchmark's `controls()` uses, so a number computed here and one
# computed there are the same quantity.
AA_MASS = {
    "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
    "Q": 128.13, "E": 129.12, "G": 57.05, "H": 137.14, "I": 113.16,
    "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18, "P": 97.12,
    "S": 87.08, "T": 101.10, "W": 186.21, "Y": 163.18, "V": 99.13,
}
WATER = 18.02
# Kyte-Doolittle positives, the conventional hydrophobic set.
HYDROPHOBIC = set("AILMFWVY")


def longest_run(sequence: str) -> int:
    """Longest stretch of one repeated residue.

    A long homopolymer run is both an expression liability and a common
    artefact of an optimiser exploiting a language-model likelihood.
    """
    best = run = 1 if sequence else 0
    for previous, current in itertools.pairwise(sequence):
        run = run + 1 if current == previous else 1
        best = max(best, run)
    return best


def glycosylation_sequons(sequence: str) -> int:
    """N-X-S/T where X is anything but proline -- the N-linked motif."""
    return sum(
        1
        for i in range(len(sequence) - 2)
        if sequence[i] == "N"
        and sequence[i + 1] != "P"
        and sequence[i + 2] in "ST"
    )


def metrics_for(sequence: str) -> dict[str, float]:
    length = len(sequence)
    return {
        "length": float(length),
        # Net charge at pH 7: K and R positive, D and E negative. Histidine is
        # left out, as the benchmark's control does -- it is only partly
        # protonated at 7 and including it would make this a different number
        # from the one the control bar was computed with.
        "net_charge": float(
            sum(sequence.count(a) for a in "KR") - sum(sequence.count(a) for a in "DE")
        ),
        "molecular_weight": sum(AA_MASS.get(a, 110.0) for a in sequence) + WATER,
        "hydrophobic_fraction": (
            sum(1 for a in sequence if a in HYDROPHOBIC) / length if length else 0.0
        ),
        # Free cysteines: an odd count cannot all pair, and any count at all is
        # an expression liability in a secreted binder.
        "n_cysteines": float(sequence.count("C")),
        "n_glycosylation_motifs": float(glycosylation_sequons(sequence)),
        "max_low_complexity_run": float(longest_run(sequence)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--outputs", required=True)
    args = p.parse_args()

    with open(args.inputs) as source, open(args.outputs, "w") as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sequence = (row.get("sequence") or "").strip().upper()
            if not sequence:
                sink.write(
                    json.dumps({"index": row["index"], "failed": "empty sequence"})
                    + "\n"
                )
                continue
            sink.write(
                json.dumps({"index": row["index"], "metrics": metrics_for(sequence)})
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
