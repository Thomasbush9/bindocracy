#!/usr/bin/env python3
"""Check one or more workflow indexes without planning or submitting anything.

    .venv/bin/python scripts/check_index.py index.yaml [more.yaml ...] --site site.yaml

Uses the same report as ``bindocracy campaign check``. Static success alone is
not scheduler readiness: add --probe-site for bounded, read-only Slurm queries.
Blocked or unknown readiness exits nonzero. No campaign database is opened for
writing, run directory reserved, or container probe executed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bindocracy.campaign.check import check_campaign


def check(index_path: Path, site_path: Path | None = None, *, probe_site: bool = False) -> dict:
    """Print a concise rendering of the shared readiness report and return it."""
    report = check_campaign(index_path, site_path, probe_site=probe_site)
    print(f"=== {index_path}: {report['status'].upper()}")
    print(f"  static={report['static_status']} scheduler={report['scheduler_status']}")
    for run in report["runs"]:
        print(
            f"  {run['status'].upper():7s} {run['name']} "
            f"tool={run.get('tool', '?')} tasks={run.get('tasks', '?')}"
        )
        for finding in run["checks"]:
            if finding["status"] != "ready":
                print(f"    {finding['code']}: {finding['message']}")
    for finding in report["checks"] + report["scheduler_checks"]:
        if finding["status"] != "ready":
            print(f"  {finding['status'].upper()} {finding['code']}: {finding['message']}")
    print(f"  {report['scope']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indices", nargs="+", type=Path, help="Workflow index YAML files.")
    parser.add_argument("--site", type=Path, help="Site policy shared by the indexes.")
    parser.add_argument("--probe-site", action="store_true", help="Run read-only Slurm queries.")
    args = parser.parse_args()
    ready = 0
    for path in args.indices:
        report = check(path, args.site, probe_site=args.probe_site)
        ready += report["status"] == "ready"
        print()
    print(f"{ready}/{len(args.indices)} index(es) ready within the reported check scope")
    return 0 if ready == len(args.indices) else 1


if __name__ == "__main__":
    raise SystemExit(main())
