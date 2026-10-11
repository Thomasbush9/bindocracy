#!/usr/bin/env python3
"""A nonadaptive baseline: execute a fixed index and submit a fixed DB selection.

Run through `bindocracy benchmark run` under the same resources as each agent.
This submits real configured workloads; it is not a local scientific simulator.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from bindocracy.benchmark.runner import _broker_request
from bindocracy.benchmark.submission import build_submission
from bindocracy.benchmark.workspace import load_workspace
from bindocracy.campaign.control import status, submit
from bindocracy.campaign.plan import build_plan
from bindocracy.runs.selection import build
from bindocracy.store.query import DesignQuery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=os.environ.get("BINDOCRACY_WORKSPACE"), required=False)
    parser.add_argument("--index", type=Path, help="Fixed workload index; defaults to workspace/workflow.yaml.")
    parser.add_argument("--query", type=Path, help="Predeclared DesignQuery YAML; no adaptive changes.")
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if not args.workspace or args.poll_seconds <= 0:
        parser.error("workspace and positive poll-seconds are required")
    broker = os.environ.get("BINDOCRACY_BROKER")
    if not broker:
        parser.error("run this baseline through the external benchmark runner")
    workspace = load_workspace(args.workspace)
    root = workspace.root / "baseline"
    root.mkdir(exist_ok=False)
    plan = build_plan(args.index or workspace.root / "workflow.yaml", workspace.site_config, root / "plan")
    plan_path = Path(plan["plan_dir"]) / "plan.json"
    print(json.dumps({"event": "plan", "digest": plan["digest"]}), flush=True)
    submit(plan_path, approve=plan["digest"])
    while True:
        observation = status(plan_path)
        state = observation["state"]
        print(json.dumps({"event": "campaign", "state": state}), flush=True)
        if state == "completed":
            break
        if state in {"cancelled", "stopped"}:
            return 1
        time.sleep(args.poll_seconds)
    query = DesignQuery.model_validate(yaml.safe_load(args.query.read_text())) if args.query else DesignQuery(
        distinct_sequences=True, limit=workspace.task.submission.max_candidates,
    )
    _, _, design_set = build(workspace.database, query, root / "selection")
    bundle = root / "submission"
    build_submission(workspace.database, bundle, design_set=design_set, workspace=workspace.root)
    acceptance = _broker_request(broker, "accept", source=str(bundle))
    print(json.dumps({"event": "accepted", "result": acceptance}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
