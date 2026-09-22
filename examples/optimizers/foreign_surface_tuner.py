#!/usr/bin/env python3
"""surface_tuner -- a small sequence-only binder polisher.

Originally written outside bindocracy, this standard-library-only optimizer
keeps its own policy and optimization logic. The portable bindocracy_io module
now handles its JSONL transport; the host supplies that module automatically.
For standalone use, put the repository's drivers directory on PYTHONPATH.

What it does, scientifically, is modest and honest about it: it removes
solvent-exposed hydrophobics and walks the binder's net charge toward a target,
by hill-climbing a weighted composition score. No structure is predicted, no
model is consulted, nothing is folded. That is why the run declares
`loss_models: []` -- a claim that the loss saw no structure predictor at all,
not an omission.

The calling convention remains three flags:

    surface_tuner --inputs IN.jsonl --outputs OUT.jsonl --context CTX.json

Usage outside a harness, which is how it was developed:

    surface_tuner --inputs parents.jsonl --outputs children.jsonl \
        --context ctx.json --policy policy.json --variants 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from bindocracy_io import RejectCandidate, run_optimization

__version__ = "0.3.1"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = frozenset("AILMFWVY")
POSITIVE = frozenset("KR")
NEGATIVE = frozenset("DE")
# What an exposed hydrophobic may become. No cysteine (free thiols), no
# proline (backbone), no glycine (flexibility) -- this tool is not allowed to
# decide any of those for you.
SUBSTITUTES = "ADEKNQRST"


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """The objective. Loaded from JSON so a screen can be re-run verbatim."""

    target_charge: float = -2.0
    charge_weight: float = 1.0
    hydrophobic_weight: float = 4.0
    max_substitutions: int = 6
    # A parent already this good is refused rather than churned: an optimizer
    # that always returns something returns noise when there is nothing to do.
    accept_below: float = 0.05
    # Outside this window the tool has no opinion and says nothing at all.
    min_length: int = 20
    max_length: int = 400
    rounds: int = 60

    @classmethod
    def load(cls, path: str | None) -> Policy:
        if not path:
            return cls()
        payload = json.loads(Path(path).read_text())
        known = {key: payload[key] for key in payload if key in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - set(known))
        if unknown:
            raise SystemExit(f"surface_tuner: unknown policy keys {unknown}")
        return cls(**known)


@dataclass
class Attempt:
    sequence: str
    score: float
    start_score: float
    charge: float
    substitutions: int
    history: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------


def net_charge(sequence: str) -> float:
    """Net charge at pH 7, counting K/R as +1 and D/E as -1, plus termini."""
    positive = sum(1 for residue in sequence if residue in POSITIVE)
    negative = sum(1 for residue in sequence if residue in NEGATIVE)
    histidine = 0.1 * sum(1 for residue in sequence if residue == "H")
    return positive - negative + histidine


def exposure_proxy(sequence: str, position: int, window: int = 5) -> float:
    """A cheap stand-in for burial: how hydrophobic the neighbourhood is.

    A hydrophobic residue in a hydrophobic stretch is probably core and is left
    alone; one sitting among polars is probably on the surface. This is a
    proxy, it is wrong sometimes, and it is the reason this optimizer's output
    has to be judged by a folding model that it never saw.
    """
    low = max(0, position - window)
    high = min(len(sequence), position + window + 1)
    neighbourhood = sequence[low:position] + sequence[position + 1 : high]
    if not neighbourhood:
        return 1.0
    buried = sum(1 for residue in neighbourhood if residue in HYDROPHOBIC)
    return 1.0 - buried / len(neighbourhood)


def score(sequence: str, policy: Policy) -> float:
    """Lower is better. Two terms, both in units of 'fraction of the binder'."""
    exposed = sum(
        exposure_proxy(sequence, position)
        for position, residue in enumerate(sequence)
        if residue in HYDROPHOBIC
    )
    hydrophobic_term = exposed / len(sequence)
    charge_gap = abs(net_charge(sequence) - policy.target_charge) / math.sqrt(
        len(sequence)
    )
    return policy.hydrophobic_weight * hydrophobic_term + policy.charge_weight * charge_gap


def tune(sequence: str, policy: Policy, rng: random.Random) -> Attempt:
    """Greedy single-substitution hill climb, capped at max_substitutions."""
    start = score(sequence, policy)
    current, current_score = sequence, start
    history = [start]
    changed = 0

    for _ in range(policy.rounds):
        if changed >= policy.max_substitutions:
            break
        candidates = [
            position
            for position, residue in enumerate(current)
            if residue in HYDROPHOBIC or residue in POSITIVE or residue in NEGATIVE
        ]
        if not candidates:
            break
        rng.shuffle(candidates)
        best = None
        for position in candidates[:12]:
            for replacement in SUBSTITUTES:
                if replacement == current[position]:
                    continue
                trial = current[:position] + replacement + current[position + 1 :]
                trial_score = score(trial, policy)
                if best is None or trial_score < best[1]:
                    best = (trial, trial_score)
        if best is None or best[1] >= current_score - 1e-9:
            break
        current, current_score = best
        history.append(current_score)
        changed += 1

    return Attempt(
        sequence=current,
        score=current_score,
        start_score=start,
        charge=net_charge(current),
        substitutions=sum(1 for was, now in zip(sequence, current) if was != now),
        history=history,
    )


# --------------------------------------------------------------------------
# i/o
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="surface_tuner", description=__doc__)
    parser.add_argument("--policy", type=str, default=None, help="Policy JSON.")
    parser.add_argument("--variants", type=int, default=2, help="Attempts per parent.")
    parser.add_argument("--version", action="version", version=__version__)
    # Load script-owned state once; the helper parses the transport flags.
    options, _ = parser.parse_known_args(argv)
    policy = Policy.load(options.policy)
    rng = random.Random()
    initialized = False
    written = 0

    def optimize(parent, context, args):
        nonlocal initialized, written
        if not initialized:
            rng.seed(context.get("seed", 0))
            workspace = Path(context["structure_dir"]) / "surface_tuner"
            workspace.mkdir(parents=True, exist_ok=True)
            initialized = True
        variants = max(1, min(args.variants, int(context.get("max_children", 1))))
        index = parent["index"]
        sequence = parent["sequence"]
        if not policy.min_length <= len(sequence) <= policy.max_length:
            # No opinion is distinct from an explicit refusal.
            return

        attempts = [tune(sequence, policy, rng) for _ in range(variants)]
        attempts = [attempt for attempt in attempts if attempt.sequence != sequence]
        if not attempts:
            raise RejectCandidate(
                "no substitution lowered the composition score; "
                f"start {score(sequence, policy):.4f}"
            )

        # Best first, so child 0 is the one a later query will reach for.
        attempts.sort(key=lambda attempt: attempt.score)
        seen: set[str] = set()
        ordinal = 0
        for attempt in attempts:
            if attempt.sequence in seen:
                continue
            seen.add(attempt.sequence)
            trail = f"surface_tuner/{index}-{ordinal}.jsonl"
            (Path(context["structure_dir"]) / trail).write_text(
                "".join(
                    json.dumps({"step": step, "loss": value}) + "\n"
                    for step, value in enumerate(attempt.history)
                )
            )
            yield {
                "sequence": attempt.sequence,
                "metrics": {
                    "loss": round(attempt.score, 6),
                    "start_loss": round(attempt.start_score, 6),
                    # Do not shadow the registered net_charge metric.
                    "opt_net_charge": round(attempt.charge, 3),
                    "n_substitutions": attempt.substitutions,
                },
                "trajectory": trail,
            }
            ordinal += 1
            written += 1

    result = run_optimization(optimize, parser=parser, argv=argv)
    print(f"surface_tuner {__version__}: wrote {written} variant(s)", file=sys.stderr)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
