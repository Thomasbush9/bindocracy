"""Read a finished Mosaic run directory into normalized records.

The adapter never opens DuckDB. It reads `run.json`, the per-task
`designs.jsonl` and `status.json` written by `hallucinate_binders.py`, and
returns one `CollectedRun` that a later serialized rule ingests.

Two properties matter more than the parsing itself:

* Partial work survives. A task that timed out, crashed, or produced nothing
  still contributes its completed designs, and the run is marked `partial` or
  `failed` rather than discarded.
* Re-collection is deterministic. Every record ID is derived from the run ID
  plus a stable natural key, and every timestamp comes from the files, so
  reparsing an unchanged directory produces byte-identical records.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from bindocracy.adapters.base import CollectionError, OutputAdapter
from bindocracy.adapters.common import (
    artifact,
    provenance_artifacts,
    read_manifest,
    run_window,
    unique_by_uri,
)
from bindocracy.runs.manifest import RunManifest, TaskPlan
from bindocracy.runs.status import read_task_status
from bindocracy.store.records import (
    ArtifactRecord,
    CandidateType,
    CollectedRun,
    DesignRecord,
    MetricDirection,
    MetricRecord,
    RunRecord,
    RunStatus,
    stable_id,
)

METRIC_NAME = "mosaic_ranking_loss"
_SEQUENCE = re.compile(r"[A-Z]+")


class MosaicOutputAdapter(OutputAdapter):
    tool = "mosaic"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = read_manifest(run_dir)
        if manifest.run_id != run.run_id:
            raise CollectionError(
                f"run {run.run_id} does not match manifest run {manifest.run_id}"
            )

        designs: list[DesignRecord] = []
        metrics: list[MetricRecord] = []
        artifacts: list[ArtifactRecord] = []
        seen: set[str] = set()
        per_task: dict[str, Any] = {}
        attempted = 0

        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            records, rejected = _read_designs(run_dir / task.designs, task, seen)
            attempted += status.n_attempted if status and status.n_attempted is not None else len(records)
            per_task[f"{task.task_id:04d}"] = {
                "status": status.status if status else "missing",
                "n_produced": len(records),
                **rejected,
            }

            for record in records:
                design = _design_record(run.run_id, record, manifest.created_at)
                designs.append(design)
                metrics.append(_metric_record(run.run_id, design, record))
            artifacts.extend(_task_artifacts(run_dir, run.run_id, task))

        artifacts.extend(provenance_artifacts(run_dir, run.run_id, manifest, {"driver": "driver_script", "general": "general_config",
     "model": "model_config"}))
        status_value = _run_status(manifest, per_task, len(designs))
        started, finished = run_window(
            [read_task_status(run_dir / task.status) for task in manifest.tasks]
        )

        collected_run = run.model_copy(update={
            "status": status_value,
            "n_requested": manifest.designs_per_task * len(manifest.tasks),
            "n_attempted": max(attempted, len(designs)),
            "n_produced": len(designs),
            "count_details": {"tasks": per_task},
            "started_at": started,
            "finished_at": finished,
        })
        return CollectedRun(
            run=collected_run,
            designs=tuple(designs),
            metrics=tuple(metrics),
            artifacts=tuple(unique_by_uri(artifacts)),
        )

    def succeeded(self, run_dir: Path) -> bool:
        """True when every planned task reported success and wrote designs."""
        manifest = read_manifest(run_dir)
        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            if status is None or status.status != "succeeded":
                return False
            records, _ = _read_designs(run_dir / task.designs, task, set())
            if len(records) < task.n_requested:
                return False
        return True


def _read_designs(
    path: Path, task: TaskPlan, seen: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return the valid records in one designs.jsonl, plus rejection counts.

    Invalid records are skipped rather than fatal: a malformed line must not
    cost the run the designs written before and after it. The driver appends
    and fsyncs whole lines, so only an unterminated final line can be a
    truncation rather than a defect.
    """
    counts = {"n_truncated": 0, "n_invalid": 0}
    if not path.is_file():
        return [], counts

    text = path.read_text()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    elif lines:
        lines.pop()
        counts["n_truncated"] = 1

    prefix = f"task-{task.task_id:04d}-"
    records = []
    for line in lines:
        record = _valid_record(line, prefix, seen)
        if record is None:
            counts["n_invalid"] += 1
            continue
        seen.add(record["native_id"])
        records.append(record)
    return records, counts


def _valid_record(line: str, prefix: str, seen: set[str]) -> dict[str, Any] | None:
    if not line.strip():
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None

    native_id = record.get("native_id")
    sequence = record.get("sequence")
    loss = record.get("ranking_loss")
    if not isinstance(native_id, str) or not native_id.startswith(prefix):
        return None
    if native_id in seen:
        return None
    if not isinstance(sequence, str) or _SEQUENCE.fullmatch(sequence.strip().upper()) is None:
        return None
    if isinstance(loss, bool) or not isinstance(loss, int | float):
        return None
    return record


def _design_record(run_id: str, record: dict[str, Any], fallback: datetime) -> DesignRecord:
    native_id = record["native_id"]
    seed = record.get("seed")
    return DesignRecord(
        design_id=stable_id("design", run_id, native_id),
        run_id=run_id,
        native_id=native_id,
        candidate_type=CandidateType.SEQUENCE,
        sequence=record["sequence"],
        seed=seed if isinstance(seed, int) else None,
        metadata={"seconds": record["seconds"]} if "seconds" in record else None,
        # The driver's own completion time, so re-collection is deterministic.
        created_at=_timestamp(record.get("completed_at")) or fallback,
    )


def _metric_record(run_id: str, design: DesignRecord, record: dict[str, Any]) -> MetricRecord:
    return MetricRecord(
        metric_id=stable_id("metric", run_id, design.design_id, METRIC_NAME, "0"),
        run_id=run_id,
        design_id=design.design_id,
        name=METRIC_NAME,
        value=float(record["ranking_loss"]),
        direction=MetricDirection.MIN,
        measured_at=design.created_at,
    )


def _task_artifacts(run_dir: Path, run_id: str, task: TaskPlan) -> list[ArtifactRecord]:
    wanted = [("native_designs", task.designs), ("task_status", task.status),
              ("log", task.log)]
    return [
        record
        for kind, relative in wanted
        if (record := artifact(run_dir, run_id, relative, kind)) is not None
    ]


def _run_status(manifest: RunManifest, per_task: dict[str, Any], produced: int) -> RunStatus:
    if produced == 0:
        return RunStatus.FAILED
    complete = all(task["status"] == "succeeded" for task in per_task.values())
    if complete and produced == manifest.designs_per_task * len(manifest.tasks):
        return RunStatus.SUCCEEDED
    return RunStatus.PARTIAL


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
