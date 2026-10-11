"""One deterministic same-length mutation reduces a sequence-only penalty."""
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
