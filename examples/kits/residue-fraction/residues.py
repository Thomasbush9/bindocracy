"""Resolve bundled data relative to the module, not the caller's directory."""
import json
from pathlib import Path

WEIGHTS = json.loads(Path(__file__).with_name("weights.json").read_text())


def weighted_fraction(sequence):
    return sum(WEIGHTS.get(residue, 0.0) for residue in sequence) / len(sequence)
