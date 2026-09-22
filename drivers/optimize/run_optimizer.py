#!/usr/bin/env python3
"""Run a user's optimization script over one shard of a frozen design set.

Standard library only. This runs inside whatever image the optimizer needs --
mosaic.sif, a torch image, nothing at all -- and those images do not have
`bindocracy` installed and should not need to.

**What this does and does not own.** It owns the marshalling: take this task's
contiguous shard, write `inputs.jsonl` and `context.json`, run the script, time
it, translate shard-local indices back to design-set indices, and write a
status. It does NOT own the rules about what a returned row may say -- the
alphabet, the child accounting, the declared metrics. Those live in
`src/bindocracy/tools/optimize/contract.py` and are applied host-side by the
adapter, in one place, so a rule cannot be enforced here and forgotten there.

The one check that is here is the one that has to be: `parent_index` is
shard-local, and translating it is this script's job, so its range is checked
where the translation happens.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC
from pathlib import Path

CHILDREN_FILE = "children.jsonl"
RAW_FILE = "optimizer_output.jsonl"
INPUTS_FILE = "inputs.jsonl"
CONTEXT_FILE = "context.json"
STATUS_FILE = "status.json"


def main() -> int:
    args = parse_args()
    started = time.time()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = Path(args.structure_dir)
    structure_dir.mkdir(parents=True, exist_ok=True)

    entries = read_design_set(args.design_set_manifest, args.design_set)
    shard = contiguous_shard(entries, args.shard, args.num_shards)
    if not shard:
        # An empty shard is legitimate when jobs exceed designs. Report it
        # rather than exiting non-zero: the run is not failing.
        write_status(
            save_dir, "succeeded", started, {"n_parents": 0}, task_id=args.task_id
        )
        (save_dir / CHILDREN_FILE).write_text("")
        return 0

    resolved = json.loads(Path(args.resolved).read_text()) if args.resolved else {}
    structures = resolved.get("structures") or {}
    parent_metrics = resolved.get("metrics") or {}

    wanted = [kind for kind in args.inputs_wanted.split(",") if kind]
    inputs_path = work_dir / INPUTS_FILE
    write_inputs(inputs_path, shard, wanted, structures, parent_metrics)

    context_path = work_dir / CONTEXT_FILE
    write_context(context_path, args, shard, work_dir, structure_dir)

    raw_path = save_dir / RAW_FILE
    argv = [
        sys.executable,
        str(args.script),
        "--inputs", str(inputs_path),
        "--outputs", str(raw_path),
        "--context", str(context_path),
        *(args.script_args or []),
    ]
    print(f"[optimize] shard {args.shard}/{args.num_shards}: {len(shard)} parents", flush=True)
    print(f"[optimize] {' '.join(argv)}", flush=True)

    completed, error = run_script(argv, args.timeout_seconds, work_dir)

    # The script's rows, with shard-local parent_index translated to the
    # design-set index the adapter joins on. Nothing else is altered: the raw
    # file is kept beside this one so a disagreement between what the script
    # wrote and what was stored can be settled by reading both.
    children, counts = translate(raw_path, shard)
    (save_dir / CHILDREN_FILE).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in children)
    )

    status = "succeeded" if completed and counts["n_lines"] else "failed"
    details = {
        "n_parents": len(shard),
        "shard": args.shard,
        "num_shards": args.num_shards,
        "script": str(args.script),
        "exit_code": error if isinstance(error, int) else None,
        "error": error if isinstance(error, str) else None,
        **counts,
    }
    write_status(
        save_dir, status, started, details,
        task_id=args.task_id,
        n_attempted=len(shard),
        n_produced=counts["n_children"],
    )
    if not completed:
        print(f"[optimize] script did not complete: {error}", file=sys.stderr, flush=True)
        return 1
    if not counts["n_lines"]:
        print(
            "[optimize] script exited 0 but wrote no rows. A run that succeeds and "
            "produces nothing is reported as failed rather than as an optimization "
            "that improved nothing.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"[optimize] {counts['n_children']} children, {counts['n_failed']} refusals, "
        f"{counts['n_unmappable']} unmappable rows",
        flush=True,
    )
    return 0


def run_script(argv: list[str], timeout: int, work_dir: Path) -> tuple[bool, int | str]:
    """Run the user's script, returning whether it completed and why not.

    `cwd` is the task's work directory so a script that writes a relative
    temporary file does not litter wherever the job happened to start, and does
    not collide with another shard doing the same thing.
    """
    environment = dict(os.environ)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    # Archived plans keep the helper beside this driver and the user script.
    # The source-tree driver also supports standalone validation without a
    # bindocracy installation in the script's interpreter.
    helper_dir = Path(__file__).resolve().parent
    if not (helper_dir / "bindocracy_io.py").is_file():
        helper_dir = helper_dir.parent
    if (helper_dir / "bindocracy_io.py").is_file():
        previous = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(helper_dir) + (os.pathsep + previous if previous else "")
    try:
        result = subprocess.run(
            argv, timeout=timeout, cwd=str(work_dir), env=environment, check=False
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except OSError as error:
        return False, f"could not start the script: {error}"
    if result.returncode != 0:
        return False, result.returncode
    return True, 0


def translate(path: Path, shard: list[dict]) -> tuple[list[dict], dict]:
    """Shard-local `parent_index` to design-set `index`.

    A row naming a parent outside the shard is dropped and counted. It cannot
    be stored: the index would silently attach the child to a different design,
    and a design table where some children have the wrong parent is worse than
    one missing a few.
    """
    children: list[dict] = []
    counts = {"n_lines": 0, "n_children": 0, "n_failed": 0, "n_unmappable": 0, "n_torn_lines": 0}
    if not path.is_file():
        return children, counts

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            counts["n_torn_lines"] += 1
            continue
        counts["n_lines"] += 1
        if not isinstance(row, dict):
            counts["n_unmappable"] += 1
            continue
        local = row.get("parent_index")
        if not isinstance(local, int) or isinstance(local, bool) or not 0 <= local < len(shard):
            counts["n_unmappable"] += 1
            continue

        translated = dict(row)
        translated["parent_index"] = shard[local]["index"]
        translated["shard_parent_index"] = local
        children.append(translated)
        if row.get("failed"):
            counts["n_failed"] += 1
        else:
            counts["n_children"] += 1
    return children, counts


def write_inputs(
    path: Path,
    shard: list[dict],
    wanted: list[str],
    structures: dict,
    parent_metrics: dict,
) -> None:
    """One line per parent, carrying only the declared inputs.

    `index` here is SHARD-LOCAL and starts at 0 in every task. That is the same
    convention the scorer uses, and it is what lets a script be tested against
    a handful of designs without knowing anything about sharding.
    """
    with path.open("w") as handle:
        for local, entry in enumerate(shard):
            row: dict = {"index": local, "length": entry["length"]}
            if "sequence" in wanted:
                row["sequence"] = entry["sequence"]
            if "structure" in wanted:
                pose = structures.get(str(entry["index"])) or structures.get(entry["index"])
                if pose:
                    row["structure"] = pose
            if "metrics" in wanted:
                row["metrics"] = (
                    parent_metrics.get(str(entry["index"]))
                    or parent_metrics.get(entry["index"])
                    or {}
                )
            if "provenance" in wanted:
                row["tool"] = entry.get("tool")
                row["run_name"] = entry.get("run_name")
                row["native_id"] = entry.get("native_id")
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_context(
    path: Path, args: argparse.Namespace, shard: list[dict], work_dir: Path, structure_dir: Path
) -> None:
    target = read_single_sequence(Path(args.target_fasta))
    hotspots = [int(spot) for spot in (args.hotspots or "").split(",") if spot.strip()]
    payload = {
        "target_sequence": target,
        "target_chain": args.target_chain,
        "target_msa": args.target_msa,
        "target_structure": args.target_structure,
        # 1-based positions in the target FASTA, resolved and range-checked when
        # the run was planned. Never author numbering.
        "hotspots": hotspots,
        "seed": args.seed,
        "max_children": args.max_children,
        "length_delta": args.length_delta,
        "work_dir": str(work_dir),
        "structure_dir": str(structure_dir),
        "shard": args.shard,
        "num_shards": args.num_shards,
        "n_parents": len(shard),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_design_set(manifest_path: str | None, fasta_path: str) -> list[dict]:
    """The design set's entries, from the manifest if there is one.

    The manifest is preferred because it carries `tool`, `run_name` and
    `native_id`, which a FASTA header only carries by convention. The FASTA is
    the fallback so this script stays runnable by hand against a plain file --
    which is how `scripts/validate_optimizer.py` exercises a user's script
    without a run directory.
    """
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text())
        return [
            {
                "index": entry["index"],
                "sequence": entry["sequence"],
                "length": entry["length"],
                "tool": entry.get("tool"),
                "run_name": entry.get("run_name"),
                "native_id": entry.get("native_id"),
            }
            for entry in manifest["entries"]
        ]

    entries: list[dict] = []
    for header, sequence in read_fasta(Path(fasta_path)):
        # `>000000 tool=... native=...` -- only the leading index is parsed,
        # because the rest of the header is descriptive and a driver that
        # parsed it would be inventing a format. See the Chai-1 driver, which
        # invented one and got it wrong.
        index = int(header.split()[0])
        entries.append({
            "index": index,
            "sequence": sequence,
            "length": len(sequence),
            "tool": None,
            "run_name": None,
            "native_id": None,
        })
    return entries


def contiguous_shard(entries: list[dict], shard: int, num_shards: int) -> list[dict]:
    """This task's block. Contiguous, not strided.

    Entries are ordered by length, so a contiguous block spans few binder
    lengths and therefore few JAX recompilations. A strided shard would hand
    every task the full spread.
    """
    if not 0 <= shard < num_shards:
        raise SystemExit(f"shard {shard} outside 0..{num_shards - 1}")
    per = -(-len(entries) // num_shards)
    return entries[shard * per : (shard + 1) * per]


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, list[str]]] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            records.append((line[1:].strip(), []))
        elif line.strip() and records:
            records[-1][1].append(line.strip())
    return [(header, "".join(parts).upper()) for header, parts in records]


def read_single_sequence(path: Path) -> str:
    records = read_fasta(path)
    if len(records) != 1:
        raise SystemExit(f"{path} holds {len(records)} sequences; expected one")
    return records[0][1]


def write_status(
    save_dir: Path,
    status: str,
    started: float,
    details: dict,
    *,
    task_id: int,
    n_attempted: int = 0,
    n_produced: int = 0,
) -> None:
    """The harness's `TaskStatus` shape, not an invented one.

    `runs/status.py` validates this file with `extra="allow"`, so a tool may
    record more than the harness asks for -- but `task_id` and `status` are
    required, and a driver that omits them makes the whole run uncollectable
    with an error that names pydantic rather than the driver.
    """
    payload = {
        "task_id": task_id,
        "status": status,
        "started_at": _iso(started),
        "finished_at": _iso(time.time()),
        "n_attempted": n_attempted,
        "n_produced": n_produced,
        "exit_code": details.get("exit_code"),
        "error": details.get("error"),
        "output_file": CHILDREN_FILE,
        # Extra, kept: the per-shard accounting a person wants when a run comes
        # back with fewer children than parents.
        "seconds": round(time.time() - started, 2),
        "details": details,
    }
    (save_dir / STATUS_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _iso(stamp: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(stamp, tz=UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-set", required=True)
    parser.add_argument("--design-set-manifest")
    parser.add_argument("--script", required=True)
    parser.add_argument("--target-fasta", required=True)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--target-msa")
    parser.add_argument("--target-structure")
    parser.add_argument("--hotspots", default="")
    parser.add_argument("--resolved")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--structure-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-children", type=int, default=1)
    parser.add_argument("--length-delta", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--inputs-wanted", default="sequence")
    parser.add_argument("--declared-metrics", default="")
    parser.add_argument("--script-args", nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
