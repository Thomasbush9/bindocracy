#!/usr/bin/env python3
"""A worked example of a custom scoring function.

The portable helper owns JSONL and indices; this script owns only the score.
It imports no bindocracy package and runs unchanged inside a container.
"""

import argparse


def main() -> int:
    from bindocracy_io import RejectCandidate, run_scoring

    parser = argparse.ArgumentParser()
    parser.add_argument("--ph", type=float, default=7.0)

    def score(candidate, args):
        sequence = candidate.get("sequence", "")
        if not sequence:
            raise RejectCandidate("no sequence")
        charge = sum(sequence.count(a) for a in "KR") - sum(sequence.count(a) for a in "DE")
        return {
            "net_charge_at_ph": float(charge),
            "fraction_charged": round(
                sum(sequence.count(a) for a in "KRDE") / len(sequence), 4,
            ),
        }

    return run_scoring(score, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
