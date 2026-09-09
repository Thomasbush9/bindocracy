"""Scoring output -> normalized records.

DRAFT. The one structural difference from every other adapter here: this one
emits **no designs**. A scoring run measures candidates that already exist, so
its metrics reference design IDs created by the generation runs, and inventing
a `DesignRecord` would put a second copy of every design in the table.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from bindocracy.adapters.base import CollectionError, OutputAdapter
from bindocracy.adapters.common import (
    artifact,
    provenance_artifacts,
    read_manifest,
    run_status,
    run_window,
    unique_by_uri,
)
from bindocracy.adapters.scoring import metric_records
from bindocracy.runs.designset import DesignSet
from bindocracy.runs.status import read_task_status
from bindocracy.store.records import (
    ArtifactRecord,
    CollectedRun,
    MetricRecord,
    MetricStatus,
    RunRecord,
)

METRICS_FILE = "metrics.jsonl"


class ScorerOutputAdapter(OutputAdapter):
    tool = "scorer"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = read_manifest(run_dir)
        if manifest.run_id != run.run_id:
            raise CollectionError(
                f"manifest run_id {manifest.run_id} does not match {run.run_id}"
            )

        prefix = manifest.workflow["metric_prefix"]
        index_to_design = _index_map(manifest.workflow["design_set_manifest"])

        metrics: list[MetricRecord] = []
        artifacts: list[ArtifactRecord] = []
        statuses = []
        per_task: dict[str, Any] = {}
        n_attempted = 0
        n_structures = 0
        seen_indices: set[int] = set()
        failures: dict[str, int] = {}

        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            statuses.append(status)
            rows, counts = _read_metrics(
                run_dir / task.designs,
                index_to_design=index_to_design,
                prefix=prefix,
                run_id=run.run_id,
                fallback_time=manifest.created_at,
            )
            metrics.extend(rows)
            n_attempted += counts["n_lines"]
            seen_indices |= counts["indices"]
            for reason, count in counts["failures"].items():
                failures[reason] = failures.get(reason, 0) + count
            # One artifact row per saved pose, joined to the design it is of.
            # This is what makes a later epitope, contact or clash pass a read
            # rather than a re-fold: `SELECT uri FROM artifacts WHERE
            # design_id = ? AND kind = 'predicted_structure'`.
            for relative, design_id, condition, replicate in counts["structures"]:
                record = artifact(
                    run_dir,
                    run.run_id,
                    f"{task.directory}/{relative}",
                    "predicted_structure",
                    design_id=design_id,
                )
                if record is not None:
                    artifacts.append(
                        record.model_copy(update={"metadata": {
                            "model": prefix,
                            "condition": condition,
                            "replicate": replicate,
                        }})
                    )
                    n_structures += 1

            per_task[f"task-{task.task_id:04d}"] = {
                "status": status.status if status else None,
                "n_lines": counts["n_lines"],
                "n_designs": len(counts["indices"]),
                "n_failed_folds": sum(counts["failures"].values()),
            }
            for relative, kind in (
                (task.designs, "scored_metrics"),
                (task.status, "task_status"),
                (task.log, "log"),
            ):
                record = artifact(run_dir, run.run_id, relative, kind)
                if record is not None:
                    artifacts.append(record)

        artifacts.extend(
            provenance_artifacts(
                run_dir,
                run.run_id,
                manifest,
                {"driver": "driver_script", "general": "general_config", "model": "model_config"},
            )
        )

        produced = {
            record.design_id for record in metrics if record.status == MetricStatus.OK
        }
        started, finished = run_window(statuses)
        complete = all(status is not None for status in statuses)

        n_requested = int(manifest.workflow["n_designs"])
        updated = run.model_copy(
            update={
                "status": run_status(len(produced), statuses, complete),
                # The three numbers of harness-design §2, for a run that scores
                # rather than generates: how many designs the set held, how
                # many folds were attempted, how many designs came back with a
                # usable number. A fold that failed is attempted and not
                # produced, which is the distinction that matters.
                "n_requested": n_requested,
                "n_attempted": n_attempted,
                "n_produced": len(produced),
                "count_details": {
                    "tasks": per_task,
                    "n_designs_seen": len(seen_indices),
                    "n_metric_rows": len(metrics),
                    "n_structures": n_structures,
                    "fold_failures": failures,
                    "metric_prefix": prefix,
                    "protocol_sha256": manifest.workflow.get("protocol_sha256"),
                    "design_set_digest": manifest.workflow.get("design_set_digest"),
                    "scope_id": manifest.workflow.get("scope_id"),
                },
                "started_at": started,
                "finished_at": finished,
                "output_uri": str(run_dir),
            }
        )
        return CollectedRun(
            run=updated,
            designs=(),  # a scoring run creates no candidates
            artifacts=tuple(unique_by_uri(artifacts)),
            metrics=tuple(metrics),
        )

    def succeeded(self, run_dir: Path) -> bool:
        """Artifacts exist and parse, per harness-design §6.

        Exit status is not consulted. A shard that scored everything it was
        given and then tripped over a trailing shell command is a success.
        """
        try:
            manifest = read_manifest(run_dir)
        except CollectionError:
            return False
        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            if status is None or status.status != "succeeded":
                return False
            if not (run_dir / task.designs).is_file():
                return False
        return True


def _index_map(manifest_path: str) -> dict[int, str]:
    """Design-set index -> design_id.

    The container never sees a `design_id`; it reports the index it was handed.
    This is the only place the two are joined, and it reads the design set that
    the run was planned with rather than whatever now sits at that path.
    """
    design_set = DesignSet.read(manifest_path)
    return {entry.index: entry.design_id for entry in design_set.entries}


def _read_metrics(
    path: Path,
    *,
    index_to_design: dict[int, str],
    prefix: str,
    run_id: str,
    fallback_time: datetime,
) -> tuple[list[MetricRecord], dict[str, Any]]:
    """One task's JSONL into metric rows, skipping what cannot be trusted."""
    records: list[MetricRecord] = []
    indices: set[int] = set()
    failures: dict[str, int] = {}
    # (task-relative path, design_id, condition, replicate) per saved pose.
    structures: list[tuple[str, str, str | None, int]] = []
    n_lines = 0

    if not path.is_file():
        return records, {
            "n_lines": 0, "indices": indices, "failures": failures,
            "structures": structures,
        }

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A torn final line is what a killed job leaves behind. Count it
            # and keep the rest rather than losing the shard.
            failures["unparseable_line"] = failures.get("unparseable_line", 0) + 1
            continue

        n_lines += 1
        index = row.get("index")
        if index not in index_to_design:
            failures["unknown_index"] = failures.get("unknown_index", 0) + 1
            continue
        indices.add(index)

        if row.get("failed"):
            reason = str(row["failed"]).split(":")[0][:60]
            failures[reason] = failures.get(reason, 0) + 1
            continue

        # Recorded even when the metrics are empty: a pose that exists is
        # worth pointing at whether or not the numbers came out.
        relative = row.get("structure")
        if relative:
            structures.append((
                str(relative),
                index_to_design[index],
                row.get("condition"),
                int(row.get("replicate", 0)),
            ))

        values = row.get("metrics") or {}
        if not values:
            continue

        records.extend(
            metric_records(
                run_id=run_id,
                design_id=index_to_design[index],
                model=prefix,
                values=values,
                replicate=int(row.get("replicate", 0)),
                measured_at=fallback_time,
                details={"condition": row.get("condition"), "seconds": row.get("seconds")},
            )
        )

    return records, {
        "n_lines": n_lines, "indices": indices, "failures": failures,
        "structures": structures,
    }
