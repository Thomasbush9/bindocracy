#!/usr/bin/env python3
"""Composition sanity check on each tool's binder sequences.

Hallucination and gradient-based methods drift towards low-complexity,
single-residue-rich sequences that score well against the very model being
optimised but are poor real binders. Protein-Hunter already guards against this
with a hardcoded 20% alanine cap; most of the others do not. This prints, per
tool, the sequence count, length range, the most over-represented residue, and
how many sequences look degenerate.

Flags a sequence when any single residue exceeds 25% of it, or when Shannon
entropy over the 20 amino acids falls below 3.0 bits (a natural globular protein
sits around 4.1-4.2).

    seq_qc.py [--outputs <dir>]

Python 3.6 compatible, stdlib only.
"""

import argparse
import csv
import glob
import math
import os
from collections import Counter

DEFAULT_OUTPUTS = ("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
                   "binder_design/outputs")

MAX_SINGLE_FRAC = 0.25
MIN_ENTROPY_BITS = 3.0


def entropy(seq):
    n = len(seq)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log(c / n, 2) for c in Counter(seq).values())


THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def chain_seq_from_pdb(path, chain):
    """One-letter sequence of `chain`, read off the CA atoms in residue order."""
    res = {}
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
                res[int(ln[22:26])] = THREE2ONE.get(ln[17:20].strip(), "X")
    return "".join(res[k] for k in sorted(res))


def csv_col(path, cols, split=None, index=0):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for c in cols:
                if c in row and (row[c] or "").strip():
                    s = row[c].strip()
                    if split:
                        parts = s.split(split)
                        if index >= len(parts):
                            continue
                        s = parts[index]
                    out.append(s.upper())
                    break
    return out


def collect(out):
    j = os.path.join
    got = {}

    seqs = []
    for f in glob.glob(j(out, "mosaic", "designs", "designs_*.txt")):
        lines = open(f).read().split()
        seqs += [l.strip().upper() for l in lines if not l.startswith(">") and l.strip()]
    got["mosaic"] = seqs

    c = sorted(glob.glob(j(out, "boltzgen", "results", "final_ranked_designs",
                           "final_designs_metrics_*.csv")))
    got["boltzgen"] = csv_col(c[-1], ["designed_sequence"]) if c else []

    got["freebindcraft"] = csv_col(j(out, "freebindcraft", "mpnn_design_stats.csv"),
                                   ["Sequence"])

    seqs = []
    for f in glob.glob(j(out, "genie3", "dio3_cut", "sequences", "*.fasta")):
        for line in open(f):
            if not line.startswith(">") and line.strip():
                # "binder_seq:target_seq"
                seqs.append(line.strip().split(":")[0].upper())
    got["genie3"] = seqs

    # Proteina-Complexa's rewards CSV has an `aatype` column, but it holds
    # comma-separated integer residue INDICES, not letters. The readable
    # sequence has to come from chain B of the complex PDB (chain A is the
    # target -- the reverse of RFdiffusion's convention).
    got["proteina_complexa"] = [
        chain_seq_from_pdb(p, "B")
        for p in sorted(glob.glob(j(out, "proteina_complexa", "inference",
                                    "*dio3_cut_v1", "job_*", "*.pdb")))
    ]

    got["protein_hunter"] = csv_col(
        j(out, "protein_hunter", "dio3_cut_boltz", "summary_all_runs.csv"), ["best_seq"])

    c = glob.glob(j(out, "pxdesign", "run01", "out", "design_outputs", "*", "summary.csv"))
    got["pxdesign"] = csv_col(c[0], ["sequence"]) if c else []

    got["rfdiffusion"] = []  # backbones only, poly-glycine

    # Caliby joins chains with ':' alphabetically; RFdiffusion made the binder
    # chain A, so index 0 is the binder.
    got["caliby"] = csv_col(j(out, "caliby", "run1", "seq_des_outputs.csv"),
                            ["seq"], split=":", index=0)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default=DEFAULT_OUTPUTS)
    a = ap.parse_args()

    got = collect(a.outputs)
    print("%-18s %5s %11s %7s %-16s %s" % (
        "tool", "n", "len range", "entropy", "top residue", "flagged"))
    print("-" * 78)
    for tool in ["mosaic", "boltzgen", "freebindcraft", "genie3",
                 "proteina_complexa", "protein_hunter", "pxdesign",
                 "rfdiffusion", "caliby"]:
        seqs = [s for s in got.get(tool, []) if s and s.isalpha()]
        if not seqs:
            print("%-18s %5s %11s %7s %-16s %s" % (tool, "-", "-", "-", "-",
                                                   "(no sequences)"))
            continue
        lens = [len(s) for s in seqs]
        ents = [entropy(s) for s in seqs]
        allc = Counter("".join(seqs))
        total = sum(allc.values())
        top, topn = allc.most_common(1)[0]
        flagged = []
        for s in seqs:
            cc = Counter(s)
            r, rn = cc.most_common(1)[0]
            if rn / len(s) > MAX_SINGLE_FRAC or entropy(s) < MIN_ENTROPY_BITS:
                flagged.append(s)
        print("%-18s %5d %11s %7.2f %-16s %d/%d" % (
            tool, len(seqs), "%d-%d" % (min(lens), max(lens)),
            sum(ents) / len(ents), "%s %.0f%%" % (top, 100.0 * topn / total),
            len(flagged), len(seqs)))

    print()
    print("flagged = any single residue >%.0f%% of the sequence, or entropy <%.1f bits."
          % (MAX_SINGLE_FRAC * 100, MIN_ENTROPY_BITS))
    print("A natural globular protein sits near 4.1-4.2 bits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
