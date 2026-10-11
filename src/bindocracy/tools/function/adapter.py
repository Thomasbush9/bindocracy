"""Normalize managed function results without writing to the campaign database."""

from __future__ import annotations

import json
from pathlib import Path

from bindocracy.adapters.base import CollectionError, OutputAdapter
from bindocracy.adapters.common import artifact, provenance_artifacts, read_manifest, run_window
from bindocracy.runs.status import read_task_status
from bindocracy.store.records import CollectedRun, MetricRecord, MetricStatus, RunStatus


class FunctionOutputAdapter(OutputAdapter):
    tool = "function"

    def collect(self, run_dir: Path, run):
        manifest = read_manifest(run_dir)
        if manifest.run_id != run.run_id:
            raise CollectionError("function manifest and run identity disagree")
        wanted = {row["design_id"] for row in manifest.workflow["frozen_rows"]}
        metrics = []
        statuses = []
        artifacts = provenance_artifacts(run_dir, run.run_id, manifest, {})
        summaries = {}
        errors = {}
        attempted = 0
        complete = True
        seen = set()
        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            statuses.append(status)
            attempted += status.n_attempted or 0 if status else 0
            path = run_dir / task.designs
            if not path.is_file():
                complete = False
                errors[str(task.task_id)] = "missing worker result"
            else:
                try:
                    payload = json.loads(path.read_text())
                    rows = [MetricRecord.model_validate(row) for row in payload["records"]]
                    for row in rows:
                        if row.run_id != run.run_id or row.design_id not in wanted:
                            raise ValueError("worker metric identity is outside frozen selection")
                        key = (row.design_id, row.name, row.replicate)
                        if key in seen:
                            raise ValueError("duplicate worker measurement")
                        seen.add(key)
                    metrics.extend(rows)
                    summaries[str(task.task_id)] = payload["summary"]
                    if payload.get("error"):
                        errors[str(task.task_id)] = payload["error"]
                except (ValueError, KeyError, TypeError) as error:
                    raise CollectionError(f"invalid function worker result: {path}: {error}") from error
            if status is None or status.status != "succeeded":
                complete = False
                if status is not None and status.error:
                    errors.setdefault(str(task.task_id), status.error)
            for relative, kind in (
                (task.designs, "scored_metrics"), (task.status, "task_status"), (task.log, "log"),
                (f"{task.directory}/function/inputs.jsonl", "function_inputs"),
                (f"{task.directory}/function/metrics.jsonl", "function_raw_metrics"),
                (f"{task.directory}/function/stdout.log", "log"),
                (f"{task.directory}/function/stderr.log", "log"),
            ):
                record = artifact(run_dir, run.run_id, relative, kind)
                if record:
                    artifacts.append(record)
        produced = {metric.design_id for metric in metrics if metric.status == MetricStatus.OK}
        state = RunStatus.FAILED if not produced else RunStatus.SUCCEEDED if complete and not errors and len(produced) == len(wanted) else RunStatus.PARTIAL
        started, finished = run_window(statuses)
        updated = run.model_copy(update={
            "status": state, "n_requested": len(wanted), "n_attempted": attempted,
            "n_produced": len(produced), "started_at": started, "finished_at": finished,
            "output_uri": str(run_dir), "count_details": {
                "tasks": summaries, "errors": errors, "n_failed": len(wanted) - len(produced),
                "design_set_digest": manifest.workflow["design_set_digest"],
                "scope_id": manifest.workflow["scope_id"],
            },
        })
        return CollectedRun(run=updated, metrics=tuple(metrics), artifacts=tuple(artifacts))

    def succeeded(self, run_dir: Path) -> bool:
        try:
            manifest = read_manifest(run_dir)
            return self.collect(run_dir, manifest.to_run_record()).run.status == RunStatus.SUCCEEDED
        except (CollectionError, ValueError, OSError, RuntimeError):
            return False
