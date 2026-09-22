#!/usr/bin/env python3
"""Preflight every run in a workflow index, without submitting anything.

A campaign index is the one artefact nothing validates until submission. The
Snakefile does validate it -- `load_configs` runs each tool's preflight at DAG
time -- but reaching that requires the Slurm profile and a dry run, so in
practice an index is discovered to be stale when somebody tries to launch it.

That is how `scoring_smoke_22.yaml` came to name a scorer that preflight now
refuses: nothing asked the question between the deprecation and the next
attempt to run.

    python scripts/check_index.py /path/to/index.yaml [more.yaml ...]

Prints one line per run -- OK with its tool and task count, or FAIL with the
refusal -- and exits non-zero if any run in any index would not launch. Reads
configs and the filesystem only: no database is opened, no job is submitted,
no GPU is touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bindocracy.runs.index import read_workflow_index
from bindocracy.tools import load_configs, plugin_for


def check(index_path: Path) -> tuple[int, int]:
    """Preflight one index. Returns (checked, failed)."""
    print(f"=== {index_path}")
    try:
        index = read_workflow_index(yaml.safe_load(index_path.read_text()))
    except Exception as error:  # noqa: BLE001 - the message is the product
        print(f"  INDEX UNREADABLE: {type(error).__name__}: {error}")
        return 0, 1

    print(f"  database    {index.database}")
    print(f"  general     {index.general_config}")
    checked = failed = 0
    for request in index.runs:
        checked += 1
        try:
            loaded = load_configs(index.general_config, request.config)
            jobs = plugin_for(loaded.tool).tool_plan(loaded).jobs
        except Exception as error:  # noqa: BLE001
            failed += 1
            lines = str(error).strip().splitlines() or [type(error).__name__]
            print(f"  FAIL  {request.name}")
            print(f"        {type(error).__name__}: {lines[0]}")
            for line in lines[1:]:
                print(f"        {line}")
            continue
        print(f"  OK    {request.name:<36s} tool={loaded.tool:<18s} tasks={jobs}")
    return checked, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indices", nargs="+", type=Path, help="Workflow index YAML files.")
    args = parser.parse_args()

    total = failures = 0
    for path in args.indices:
        checked, failed = check(path)
        total += checked
        failures += failed
        print()

    print(f"{total - failures}/{total} run(s) would launch")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
