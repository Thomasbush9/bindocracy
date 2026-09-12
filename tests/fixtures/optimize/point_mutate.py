#!/usr/bin/env python3
"""A complete, minimal custom optimizer. Imports nothing from this project.

Not a useful optimizer -- it substitutes the most hydrophobic surface-ish
residues for alanine and calls that an improvement -- but it is a correct one,
and it is the shortest thing that exercises every part of the contract:
n->m children, a declared metric, a failure row, a written trajectory, and a
parent it declines to touch.

Copy this, delete the middle, and put your own optimization where `optimize()`
is. Everything above and below it is the contract.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
    # The three the harness always passes, in this order.
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--context", required=True)
    # Anything after those comes from `args:` in the YAML.
    parser.add_argument("--children", type=int, default=1)
    args = parser.parse_args()

    context = json.loads(Path(args.context).read_text())
    rng = random.Random(context["seed"])
    structure_dir = Path(context["structure_dir"])
    # Never write outside structure_dir, and never report an absolute path:
    # the harness refuses one so that a run directory can be moved.
    trajectories = structure_dir / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)

    n_children = min(args.children, context["max_children"])

    with open(args.outputs, "w") as out:
        for line in Path(args.inputs).read_text().splitlines():
            if not line.strip():
                continue
            parent = json.loads(line)
            index = parent["index"]

            children = optimize(parent["sequence"], rng, n_children)
            if not children:
                # A parent that could not be optimized is REPORTED, not
                # omitted. Absence and refusal look the same in a query and
                # mean opposite things.
                out.write(json.dumps({
                    "parent_index": index,
                    "failed": "no hydrophobic position to substitute",
                }) + "\n")
                continue

            relative = f"trajectories/{index}.jsonl"
            (structure_dir / relative).write_text(
                "".join(
                    json.dumps({"step": step, "loss": loss}) + "\n"
                    for step, (_, loss) in enumerate(children)
                )
            )

            out.writelines(
                json.dumps({
                    "parent_index": index,
                    "child": ordinal,
                    "sequence": sequence,
                    # Only metrics the YAML declares. An undeclared one is
                    # rejected, because a number with no direction sorts
                    # backwards and nothing in the row says so.
                    "metrics": {"loss": loss, "n_mutations": 2},
                    "trajectory": relative,
                }) + "\n"
                for ordinal, (sequence, loss) in enumerate(children)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
