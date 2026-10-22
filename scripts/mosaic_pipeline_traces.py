#!/usr/bin/env python3
"""Normalise a mosaic_test pipeline run into (per-step losses, final scores).

`scripts/mosaic_trajectory_rank.py` and `scripts/mosaic_cull_sweep.py` were
written against the bindocracy driver's own logs. The lab's `mosaic_test`
pipeline produces the same underlying trace from the same `_print_iter` call,
but in a different shape, so this turns one into the other and leaves the two
analysis scripts unchanged.

    scripts/mosaic_pipeline_traces.py \\
        --run /n/holylfs06/.../mosaic_test/pipelines/dio3_cut_r32_catalytic_1k \\
        --out r32.npz

Four differences from the driver logs, each handled here:

1. The runs use `batched_simplex_APGM`, so a step prints one line PER MEMBER,
   tagged `<iter>[<member>]`. Members interleave within a step.
2. The iteration counter restarts in each of the 3 soft/sharp cycles
   (90 + 30 steps), so it cannot delimit anything. Lines are taken in file
   order, which is the trajectory order.
3. There is no completion line carrying a final score. Trajectories are joined
   to `designs_seed<N>.json` by (seed, member) and from there to the two
   critics by SEQUENCE STRING -- never by position or by the `design` field,
   which the run's own handoff documents as a trap (`results` is sorted by
   score, so positional indexing mismatches almost every row).
4. The final score is the critics' ipTM, where HIGHER is better, while both
   analysis scripts assume lower-is-better throughout. It is negated on the way
   out so that `argsort` still puts the best trajectory first.

The final score is min(protenix ipTM, esmfold2 ipTM) -- the run's own agreed
Tier-1 criterion, and genuinely independent of the objective being optimized,
which the driver-log analysis could not claim.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

STEP_RE = re.compile(r"^(\d+)\[(\d+)\] loss: (-?[\d.]+)")
SEED_RE = re.compile(r"--seed\s+(\d+)")


def parse_job(path: Path) -> tuple[int | None, dict[int, list[float]]]:
    """Return (seed, {member: [loss per step]}) for one array task's stdout."""
    seed = None
    traces: dict[int, list[float]] = defaultdict(list)
    with path.open(errors="replace") as handle:
        for line in handle:
            if seed is None:
                found = SEED_RE.search(line)
                if found is not None:
                    seed = int(found.group(1))
            step = STEP_RE.match(line)
            if step is not None:
                traces[int(step.group(2))].append(float(step.group(3)))
    return seed, dict(traces)


def critic_metric(path: Path, metric: str) -> dict[str, float]:
    """sequence -> one critic score, keyed by sequence rather than index.

    `iptm` is the run's Tier-1 criterion; `binder_plddt` is the one the run's
    own handoff says actually discriminates on this target, ipTM having failed
    a scramble control. Both are higher-is-better.
    """
    payload = json.loads(path.read_text())
    rows = payload.get("results", payload)
    out: dict[str, float] = {}
    for row in rows:
        sequence = row.get("sequence")
        value = row.get(metric)
        if isinstance(sequence, str) and isinstance(value, int | float):
            out[sequence.strip().upper()] = float(value)
    return out


def load_run(run: Path, metric: str = "iptm") -> tuple[np.ndarray, np.ndarray, dict]:
    protenix = critic_metric(run / "screen_protenix" / "screen.json", metric)
    esmfold2 = critic_metric(run / "screen_esmfold2" / "screen.json", metric)

    traces: list[list[float]] = []
    finals: list[float] = []
    counts = {"no_seed": 0, "no_designs": 0, "unscreened": 0, "short": 0}

    for job in sorted((run / "optimize").glob("job-*.out")):
        seed, members = parse_job(job)
        if seed is None:
            counts["no_seed"] += 1
            continue
        designs = run / "optimize" / f"designs_seed{seed}.json"
        if not designs.is_file():
            counts["no_designs"] += 1
            continue
        results = json.loads(designs.read_text())["results"]
        for member, trace in sorted(members.items()):
            if member >= len(results):
                continue
            sequence = results[member]["sequence"].strip().upper()
            # A trajectory only counts when BOTH critics scored it; Tier-1 is
            # defined on their agreement, so one-sided rows cannot be ranked.
            if sequence not in protenix or sequence not in esmfold2:
                counts["unscreened"] += 1
                continue
            traces.append(trace)
            finals.append(-min(protenix[sequence], esmfold2[sequence]))

    if not traces:
        raise SystemExit(f"no joinable trajectories in {run}")

    lengths = [len(trace) for trace in traces]
    modal = max(set(lengths), key=lengths.count)
    keep = [i for i, length in enumerate(lengths) if length == modal]
    counts["short"] = len(traces) - len(keep)

    losses = np.array([traces[i] for i in keep], dtype=float)
    scores = np.array([finals[i] for i in keep], dtype=float)
    return losses, scores, {**counts, "steps": modal, "kept": len(keep)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metric", default="iptm",
                    choices=("iptm", "binder_plddt"),
                    help="critic score to rank by; both higher-is-better")
    args = ap.parse_args()

    losses, finals, info = load_run(args.run, args.metric)
    print(f"{args.run.name}: {info['kept']} trajectories x {info['steps']} steps")
    for key in ("no_seed", "no_designs", "unscreened", "short"):
        if info[key]:
            print(f"  dropped {info[key]} ({key})")
    print(f"  best {args.metric} {-finals.min():.3f}   "
          f"median {-np.median(finals):.3f}   worst {-finals.max():.3f}")

    np.savez_compressed(args.out, losses=losses, finals=finals)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
