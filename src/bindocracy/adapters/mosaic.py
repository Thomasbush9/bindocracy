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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bindocracy.adapters.base import CollectionError, OutputAdapter
from bindocracy.runs.manifest import RunManifest, TaskPlan
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
_MEDIA_TYPES = {".jsonl": "application/x-ndjson", ".json": "application/json",
                ".log": "text/plain", ".py": "text/x-python"}


class MosaicOutputAdapter(OutputAdapter):
    tool = "mosaic"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = _read_manifest(run_dir)
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
            status = _read_status(run_dir / task.status)
            records, rejected = _read_designs(run_dir / task.designs, task, seen)
            attempted += status.get("n_attempted") or len(records)
            per_task[f"{task.task_id:04d}"] = {
                "status": status.get("status", "missing"),
                "n_produced": len(records),
                **rejected,
            }

            for record in records:
                design = _design_record(run.run_id, record, manifest.created_at)
                designs.append(design)
                metrics.append(_metric_record(run.run_id, design, record))
            artifacts.extend(_task_artifacts(run_dir, run.run_id, task))

        artifacts.extend(_provenance_artifacts(run_dir, run.run_id, manifest))
        status_value = _run_status(manifest, per_task, len(designs))
        started, finished = _run_window(run_dir, manifest)

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
            artifacts=tuple(_unique_by_uri(artifacts)),
        )

    def succeeded(self, run_dir: Path) -> bool:
        """True when every planned task reported success and wrote designs."""
        manifest = _read_manifest(run_dir)
        for task in manifest.tasks:
            status = _read_status(run_dir / task.status)
            if status.get("status") != "succeeded":
                return False
            records, _ = _read_designs(run_dir / task.designs, task, set())
            if len(records) < task.n_requested:
                return False
        return True


def _read_manifest(run_dir: Path) -> RunManifest:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise CollectionError(f"missing run manifest: {manifest_path}")
    return RunManifest.read(manifest_path)


def _read_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise CollectionError(f"invalid status file {path}: {error}") from error


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
        artifact
        for kind, relative in wanted
        if (artifact := _artifact(run_dir, run_id, relative, kind)) is not None
    ]


def _provenance_artifacts(
    run_dir: Path, run_id: str, manifest: RunManifest
) -> list[ArtifactRecord]:
    # `general` and `model` only appear in runs planned before the configs
    # stopped being archived; they are still collected so those runs reparse.
    kinds = {"driver": "driver_script", "general": "general_config",
             "model": "model_config"}
    return [
        artifact
        for key, archived in manifest.provenance.items()
        if (artifact := _artifact(
            run_dir, run_id, archived.path, kinds.get(key, key))) is not None
    ]


def _artifact(run_dir: Path, run_id: str, relative: str, kind: str) -> ArtifactRecord | None:
    path = run_dir / relative
    if not path.is_file():
        return None
    stat = path.stat()
    return ArtifactRecord(
        artifact_id=stable_id("artifact", run_id, relative),
        run_id=run_id,
        kind=kind,
        uri=relative,
        media_type=_MEDIA_TYPES.get(path.suffix),
        size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_mtime, UTC),
    )


def _unique_by_uri(artifacts: list[ArtifactRecord]) -> list[ArtifactRecord]:
    by_uri: dict[str, ArtifactRecord] = {}
    for artifact in artifacts:
        by_uri.setdefault(artifact.uri, artifact)
    return list(by_uri.values())


def _run_status(manifest: RunManifest, per_task: dict[str, Any], produced: int) -> RunStatus:
    if produced == 0:
        return RunStatus.FAILED
    complete = all(task["status"] == "succeeded" for task in per_task.values())
    if complete and produced == manifest.designs_per_task * len(manifest.tasks):
        return RunStatus.SUCCEEDED
    return RunStatus.PARTIAL


def _run_window(run_dir: Path, manifest: RunManifest) -> tuple[datetime | None, datetime | None]:
    starts, finishes = [], []
    for task in manifest.tasks:
        status = _read_status(run_dir / task.status)
        starts.append(_timestamp(status.get("started_at")))
        finishes.append(_timestamp(status.get("finished_at")))
    starts = [value for value in starts if value is not None]
    finishes = [value for value in finishes if value is not None]
    return (min(starts) if starts else None, max(finishes) if finishes else None)


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
