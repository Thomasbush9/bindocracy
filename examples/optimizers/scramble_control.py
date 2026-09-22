#!/usr/bin/env python3
"""scramble_control -- the negative control arm, as an optimizer.

A filter's thresholds are only interpretable against something known to be
bad. `worth_optimizing_smoke23.yaml` says so about itself: its numbers were
read off a visible gap in twenty designs, not against any false-positive rate.
Without a control arm, "four designs cleared 0.80 ipTM" is a fact about four
designs and not yet evidence about the threshold.

This produces that arm. Each child is a **permutation of its parent**: the same
residues, the same length, the same composition, in a different order. Amino
acid composition, molecular weight, net charge, hydrophobic fraction and
cysteine count are all preserved exactly, so every sequence-only control metric
is identical between a design and its scramble. What is destroyed is the
arrangement -- which is the only thing the design process contributed.

That makes the comparison sharp. A predictor scoring scrambles as highly as
designs is responding to composition, not to an interface, and any threshold
placed above the scramble distribution is measuring the wrong thing. This is
the structural counterpart to the sequence-only bar the benchmark already uses,
where binder length alone reaches AUC 0.642.

It is an optimizer rather than a generator because an optimizer is exactly a
parent-to-children contract, and pairing matters here: each scramble is matched
to one real design, so the two distributions differ in arrangement and nothing
else. The run declares `loss_models: []`, which is true and load-bearing --
this script consults no model at all, so every scorer remains available to
judge its output.

Nothing here is a design. Children are marked in their own metrics and the run
name should say so too; they exist to be scored and compared, never to be
ordered or carried into an optimization pass.

    scramble_control --inputs IN.jsonl --outputs OUT.jsonl --context CTX.json
                     [--variants N] [--mode shuffle|reverse]

Standalone development needs the repository's `drivers/` on PYTHONPATH; under
the harness the portable helper is supplied automatically.
"""

from __future__ import annotations

import argparse
import random
import sys

from bindocracy_io import RejectCandidate, run_optimization

__version__ = "1.0.0"

# Below this, a permutation is not a meaningful control: the number of distinct
# arrangements is small and a "scramble" can easily be a near-neighbour of the
# parent rather than an independent draw.
MIN_LENGTH = 20


def identity_to(parent: str, child: str) -> float:
    """Fraction of positions holding the same residue as the parent.

    Reported rather than assumed. A permutation of a low-complexity sequence
    can share most of its positions with the parent by chance, and a control
    that is 70% identical to the thing it controls for is not a control. This
    number is what lets that be checked after the fact instead of trusted.
    """
    same = sum(1 for was, now in zip(parent, child) if was == now)
    return same / len(parent) if parent else 0.0


def shuffle(sequence: str, rng: random.Random) -> str:
    residues = list(sequence)
    rng.shuffle(residues)
    return "".join(residues)


def reverse(sequence: str) -> str:
    """The reversed sequence: one deterministic permutation, not a random one.

    Kept as a second mode because it preserves local composition windows that a
    full shuffle destroys, which makes it the harder control of the two -- a
    predictor that separates designs from reversals is making a stronger claim
    than one that only separates designs from shuffles.
    """
    return sequence[::-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scramble_control", description=__doc__)
    parser.add_argument("--variants", type=int, default=1,
                        help="Scrambles per parent (capped by max_children).")
    parser.add_argument("--mode", choices=("shuffle", "reverse"), default="shuffle",
                        help="Permutation to apply. 'reverse' ignores --variants.")
    parser.add_argument("--max-identity", type=float, default=0.5,
                        help="Refuse a scramble sharing more than this fraction of "
                             "positions with its parent.")
    parser.add_argument("--attempts", type=int, default=32,
                        help="Redraws allowed before giving up on one scramble.")
    parser.add_argument("--version", action="version", version=__version__)

    rng = random.Random()
    seeded = False
    written = 0

    def optimize(parent, context, args):
        nonlocal seeded, written
        if not seeded:
            # Seeded from the run's own seed, so a control arm is reproducible
            # in the same sense the run it controls for is.
            rng.seed(context.get("seed", 0))
            seeded = True

        sequence = parent["sequence"]
        if len(sequence) < MIN_LENGTH:
            raise RejectCandidate(
                f"binder is {len(sequence)} residues; below {MIN_LENGTH} a "
                "permutation is too close to its parent to control for anything"
            )
        if len(set(sequence)) < 2:
            raise RejectCandidate("a homopolymer has no distinct permutations")

        if args.mode == "reverse":
            candidates = [reverse(sequence)]
        else:
            wanted = max(1, min(args.variants, int(context.get("max_children", 1))))
            candidates = []
            seen = {sequence}
            for _ in range(args.attempts):
                if len(candidates) >= wanted:
                    break
                trial = shuffle(sequence, rng)
                if trial in seen or identity_to(sequence, trial) > args.max_identity:
                    continue
                seen.add(trial)
                candidates.append(trial)
            if not candidates:
                raise RejectCandidate(
                    f"no permutation stayed below {args.max_identity:g} identity to "
                    f"the parent in {args.attempts} attempts; this sequence is too "
                    "low-complexity to scramble"
                )

        for child in candidates:
            written += 1
            yield {
                "sequence": child,
                "metrics": {
                    # Named so nothing can mistake a control for a measurement:
                    # neither key is a registered metric.
                    "identity_to_parent": round(identity_to(sequence, child), 4),
                    "is_control": 1.0,
                },
            }

    result = run_optimization(optimize, parser=parser, argv=argv)
    print(f"scramble_control {__version__}: wrote {written} control(s)", file=sys.stderr)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
