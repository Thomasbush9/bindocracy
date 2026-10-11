"""Compute a weighted residue fraction using bundled data and a sibling module."""
from bindocracy_io import run_scoring
from residues import weighted_fraction


def score(design, args):
    return {"residue_fraction": weighted_fraction(design["sequence"])}


if __name__ == "__main__":
    run_scoring(score)
