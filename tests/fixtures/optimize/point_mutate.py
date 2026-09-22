#!/usr/bin/env python3
"""A complete, minimal custom optimizer using the portable JSONL helper.

Not a useful optimizer -- it substitutes the most hydrophobic surface-ish
residues for alanine and calls that an improvement -- but it is a correct one,
and it is the shortest thing that exercises every part of the contract:
n->m children, a declared metric, a failure row, a written trajectory, and a
parent it declines to touch.

Copy this and put your own optimization where `optimize()` is. The helper
supplies argument parsing, row identity, failures, and streaming output.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from bindocracy_io import RejectCandidate, run_optimization

HYDROPHOBIC = "AILMFWVY"


def optimize(sequence: str, rng: random.Random, n_children: int) -> list[tuple[str, float]]:
    """The only part that is about optimization. Returns (sequence, loss) pairs.

    Yours goes here. Fold something, run a gradient, call a model -- the
    contract does not care, it only cares about what comes back.
    """
    children: list[tuple[str, float]] = []
    positions = [
        index for index, residue in enumerate(sequence) if residue in HYDROPHOBIC
    ]
    if not positions:
        return children

    for _ in range(n_children):
        chosen = rng.sample(positions, k=min(2, len(positions)))
        mutated = list(sequence)
        for position in chosen:
            mutated[position] = "A"
        child = "".join(mutated)
        # A stand-in loss: fraction hydrophobic, lower being "better". A real
        # one comes out of whatever model the optimization drove.
        loss = sum(1 for residue in child if residue in HYDROPHOBIC) / len(child)
        children.append((child, loss))
    return children


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--children", type=int, default=1)
    rng = random.Random()
    initialized = False

    def optimize_parent(parent, context, args):
        nonlocal initialized
        structure_dir = Path(context["structure_dir"])
        if not initialized:
            rng.seed(context["seed"])
            (structure_dir / "trajectories").mkdir(parents=True, exist_ok=True)
            initialized = True
        children = optimize(parent["sequence"], rng, min(args.children, context["max_children"]))
        if not children:
            raise RejectCandidate("no hydrophobic position to substitute")

        relative = f"trajectories/{parent['index']}.jsonl"
        (structure_dir / relative).write_text(
            "".join(
                json.dumps({"step": step, "loss": loss}) + "\n"
                for step, (_, loss) in enumerate(children)
            )
        )
        for sequence, loss in children:
            yield {
                "sequence": sequence,
                "metrics": {"loss": loss, "n_mutations": 2},
                "trajectory": relative,
            }

    return run_optimization(optimize_parent, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
