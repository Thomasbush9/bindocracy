#!/usr/bin/env python3
"""Run a custom optimizer against the real contract, on this machine, no GPU.

The point is to make the first ten iterations of writing an optimizer cost
seconds instead of a queue wait. It drives the **same driver** the cluster runs
and applies the **same validation** the adapter applies, so a script that
passes here fails on the cluster only for reasons that are actually about the
cluster.

    python scripts/validate_optimizer.py \\
        --script my_optimizer.py \\
        --sequences GSHMDIVLTQ... KVFGRCELAA... \\
        --target-fasta target.fasta \\
        --declare loss:min --declare n_mutations:none \\
        --max-children 2

Or against a real frozen set, which is what the run will actually see:

    python scripts/validate_optimizer.py --script my_optimizer.py \\
        --design-set sets/<digest>.json --target-fasta target.fasta \\
        --declare loss:min

It prints what would be stored: the children, their parents, the metrics and
their directions, and every row that was rejected and why.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bindocracy.runs.designset import DesignSet
from bindocracy.store.records import MetricDirection
from bindocracy.tools.optimize.contract import read_outputs

DRIVER = REPO / "drivers" / "optimize" / "run_optimizer.py"


def main() -> int:
    args = parse_args()
    declared = dict(_declaration(item) for item in args.declare)

    with tempfile.TemporaryDirectory(prefix="validate-optimizer-") as tmp:
        root = Path(tmp)
        fasta, manifest, n_parents = _write_design_set(args, root)
        target = _write_target(args, root)

        save_dir = root / "task-0000"
        argv = [
            sys.executable, str(DRIVER),
            "--design-set", str(fasta),
            "--script", str(Path(args.script).resolve()),
            "--target-fasta", str(target),
            "--target-chain", args.target_chain,
            "--work-dir", str(save_dir / "work"),
            "--structure-dir", str(save_dir / "structures"),
            "--save-dir", str(save_dir),
            "--shard", "0", "--num-shards", "1",
            "--seed", str(args.seed),
            "--max-children", str(args.max_children),
            "--inputs-wanted", args.inputs,
            "--declared-metrics", ",".join(sorted(declared)),
            "--timeout-seconds", str(args.timeout_seconds),
        ]
        if manifest is not None:
            argv.extend(("--design-set-manifest", str(manifest)))
        if args.hotspots:
            argv.extend(("--hotspots", args.hotspots))
        if args.script_args:
            argv.extend(("--script-args", *args.script_args))

        print(f"running {args.script} over {n_parents} parent(s)\n", flush=True)
        completed = subprocess.run(argv, check=False)

        children_file = save_dir / "children.jsonl"
        if not children_file.is_file():
            print("\nFAIL: the driver produced no children.jsonl", file=sys.stderr)
            return 1

        rows, counts = read_outputs(
            children_file,
            n_parents=n_parents,
            max_children=args.max_children,
            declared=declared,
        )
        # The driver drops a row whose parent_index is outside the shard before
        # the contract ever sees it, because translating that index is what it
        # is for. That count lives in its status file, and it is the single
        # most likely first mistake in a new script, so it is folded in here
        # rather than left invisible.
        status = save_dir / "status.json"
        if status.is_file():
            details = json.loads(status.read_text()).get("details") or {}
            unmappable = int(details.get("n_unmappable") or 0)
            if unmappable:
                counts["rejected"]["parent_index_out_of_range"] = (
                    counts["rejected"].get("parent_index_out_of_range", 0) + unmappable
                )
                counts["n_lines"] += unmappable
        return _report(rows, counts, declared, completed.returncode, save_dir, args)


def _report(rows, counts, declared, returncode, save_dir, args) -> int:
    print("\n" + "=" * 72)
    print(f"driver exit code     {returncode}")
    print(f"rows written         {counts['n_lines']}")
    print(f"children accepted    {counts['n_children']}")
    print(f"parents refused      {counts['n_failed']}   (a reported `failed` row)")
    if counts["n_torn_lines"]:
        print(f"torn lines           {counts['n_torn_lines']}")

    rejected = counts.get("rejected") or {}
    if rejected:
        print("\nREJECTED ROWS -- these would not be stored:")
        for reason, count in sorted(rejected.items()):
            print(f"  {count:>4}  {reason}")
        print(_explain(rejected))

    children = [row for row in rows if not row.is_failure]
    if children:
        print(f"\nwould store {len(children)} design(s), each with parent_design_id set:")
        for row in children[: args.show]:
            metrics = ", ".join(
                f"{key}={value:g} ({declared[key].upper()})"
                for key, value in sorted(row.metrics.items())
            )
            print(f"  parent {row.parent_index} child {row.child}  "
                  f"len {len(row.sequence)}  {metrics or 'no metrics'}")
            print(f"    {row.sequence[:64]}{'...' if len(row.sequence) > 64 else ''}")
        if len(children) > args.show:
            print(f"  ... and {len(children) - args.show} more")

    for row in rows:
        if row.is_failure:
            print(f"\n  parent {row.parent_index} refused: {row.failed}")

    print(f"\nartifacts under {save_dir}")
    print("  optimizer_output.jsonl   what your script wrote, unaltered")
    print("  children.jsonl           after index translation")
    print("  work/inputs.jsonl        what your script was handed")
    print("  work/context.json        the run-level context")

    ok = returncode == 0 and counts["n_children"] > 0 and not rejected
    print("\n" + ("PASS -- this script satisfies the contract" if ok else "NOT READY"))
    return 0 if ok else 1


def _explain(rejected: dict) -> str:
    hints = {
        "parent_index_out_of_range": (
            "  parent_index is the position in the SHARD you were handed (the "
            "`index` field\n  of each input row), not a design id and not a "
            "position in the whole set."
        ),
        "undeclared_metric": (
            "  Every metric key must appear in the config's `metrics:` block with a\n"
            "  direction. Add it with --declare name:min|max|none here."
        ),
        "sequence_not_canonical": (
            "  Sequences must use only the 20 canonical amino acids. An 'X' is an\n"
            "  unfilled position rather than a residue."
        ),
        "child_beyond_max_children": (
            "  `child` must be 0..max_children-1. Raise max_children in the config if\n"
            "  more children per parent are intended."
        ),
        "path_not_relative_to_structure_dir": (
            "  Report structure/trajectory paths relative to context['structure_dir'],\n"
            "  so the run directory can be moved."
        ),
    }
    lines = [hints[reason] for reason in rejected if reason in hints]
    return "\n" + "\n".join(lines) if lines else ""


def _write_design_set(args, root: Path) -> tuple[Path, Path | None, int]:
    """Either a real frozen set, or a throwaway one from --sequences."""
    if args.design_set:
        manifest = Path(args.design_set).resolve()
        design_set = DesignSet.read(manifest)
        fasta = design_set.fasta_path(manifest)
        if not fasta.is_file():
            raise SystemExit(f"no FASTA beside {manifest}")
        return fasta, manifest, design_set.n_designs

    if not args.sequences:
        raise SystemExit("pass either --design-set or --sequences")
    fasta = root / "parents.fasta"
    fasta.write_text(
        "".join(
            f">{index:06d} tool=validate native=parent-{index}\n{sequence.upper()}\n"
            for index, sequence in enumerate(args.sequences)
        )
    )
    return fasta, None, len(args.sequences)


def _write_target(args, root: Path) -> Path:
    if args.target_fasta:
        return Path(args.target_fasta).resolve()
    # A stand-in, so a sequence-only optimizer can be checked without one. The
    # script is handed it as `context['target_sequence']`; if yours reads that,
    # pass the real thing.
    path = root / "target.fasta"
    path.write_text(">stand-in-target\n" + "G" * 60 + "\n")
    print("note: no --target-fasta given; context['target_sequence'] is a stand-in\n")
    return path


def _declaration(item: str) -> tuple[str, str]:
    name, _, direction = item.partition(":")
    direction = direction or "none"
    valid = {member.value for member in MetricDirection}
    if direction not in valid:
        raise SystemExit(f"--declare {item!r}: direction must be one of {sorted(valid)}")
    if not name:
        raise SystemExit(f"--declare {item!r}: needs a metric name")
    return name, direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--script", required=True, help="Your optimizer.")
    parser.add_argument("--design-set", help="A <digest>.json from `designset build`.")
    parser.add_argument("--sequences", nargs="+", help="Parent sequences, instead of a set.")
    parser.add_argument("--target-fasta")
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--hotspots", default="", help="1-based target FASTA positions, comma separated.")
    parser.add_argument(
        "--declare", action="append", default=[],
        help="A metric your script returns, as name:min|max|none. Repeatable.",
    )
    parser.add_argument("--inputs", default="sequence", help="Comma-separated input kinds.")
    parser.add_argument("--max-children", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--show", type=int, default=5, help="How many children to print.")
    parser.add_argument("--script-args", nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
