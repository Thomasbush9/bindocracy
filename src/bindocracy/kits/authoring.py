"""Small, runnable CPU examples for authors; no external data or models invented."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml

from bindocracy.kits.bundle import KitRelease, inspect_kit
from bindocracy.kits.models import KitError, safe_segment

KitKind = Literal["optimizer", "scoring_function"]

_SCORER = '''"""Compute a weighted residue fraction using bundled data and a sibling module."""
from bindocracy_io import run_scoring
from residues import weighted_fraction


def score(design, args):
    return {"residue_fraction": weighted_fraction(design["sequence"])}


if __name__ == "__main__":
    run_scoring(score)
'''

_HELPER = '''"""Bundled weights are resolved relative to the bundle, never the caller's cwd."""
import json
from pathlib import Path

WEIGHTS = json.loads(Path(__file__).with_name("weights.json").read_text())


def weighted_fraction(sequence):
    return sum(WEIGHTS.get(residue, 0.0) for residue in sequence) / len(sequence)
'''

_OPTIMIZER = '''"""One deterministic same-length mutation that reduces a sequence-only penalty."""
from bindocracy_io import RejectCandidate, run_optimization


def optimize(parent, context, args):
    sequence = parent["sequence"]
    positions = [index for index, residue in enumerate(sequence) if residue in "KR"]
    if not positions:
        raise RejectCandidate("no positively charged residue to replace")
    position = positions[context["seed"] % len(positions)]
    child = sequence[:position] + "Q" + sequence[position + 1:]
    yield {
        "sequence": child,
        "metrics": {"charge_penalty": float(len(positions) - 1)},
    }


if __name__ == "__main__":
    run_optimization(optimize)
'''


def initialize_kit(destination: Path, *, kind: KitKind, name: str | None = None) -> KitRelease:
    """Create a new complete kit; refuse even an empty existing directory."""
    if kind not in {"optimizer", "scoring_function"}:
        raise KitError("kind must be optimizer or scoring_function")
    destination = Path(destination).absolute()
    kit_name = safe_segment(name or destination.name)
    entrypoint = "optimizer.py" if kind == "optimizer" else "score.py"
    metric = "charge_penalty" if kind == "optimizer" else "residue_fraction"
    manifest = {
        "kit_schema": 1,
        "name": kit_name,
        "version": "1.0.0",
        "kind": kind,
        "summary": "CPU sequence-only charge reduction" if kind == "optimizer" else "CPU weighted residue fraction with bundled data",
        "entrypoint": entrypoint,
        "python": ">=3.12",
        "requires": [],
        "inputs": ["sequence"],
        "metrics": {metric: {"direction": "min" if kind == "optimizer" else "max"}},
        "cost": {"gpu": "none", "compile_seconds": 0, "per_parent_seconds": 0,
                 "note": "Unmeasured example; qualify on your supplied inputs before use."},
    }
    if kind == "optimizer":
        manifest.update(loss_models=[], capabilities={"changes_length": False, "max_children": 1})
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise KitError(f"kit destination already exists: {destination}") from error
    (destination / "kit.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (destination / entrypoint).write_text(_OPTIMIZER if kind == "optimizer" else _SCORER)
    if kind == "scoring_function":
        (destination / "residues.py").write_text(_HELPER)
        (destination / "weights.json").write_text(json.dumps({"K": 1.0, "R": 1.0}, sort_keys=True) + "\n")
    return inspect_kit(destination)
