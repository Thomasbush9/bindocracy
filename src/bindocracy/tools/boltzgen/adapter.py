"""Read a finished BoltzGen run directory into normalized records.

Written against the observed output of the 2026-08-27 benchmark run, not the
upstream docs. Three things about that output shape this parser:

* The metrics table has 237 columns. Ingesting all of them would bury the few
  that mean something, so a named subset becomes metrics and the rest stays in
  the CSV, which is recorded as an artifact.
* Requested, produced, and passed are three different numbers. The benchmark
  asked for 40, got 38 rows, and only 7 of those satisfied the tool's own
  filters. All three are recorded; collapsing them would hide the difference.
* Designs are complexes, not bare sequences: each row names a CIF holding the
  binder and the target together.
"""

from __future__ import annotations

import csv
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bindocracy.adapters.base import CollectionError, OutputAdapter
from bindocracy.runs.manifest import RunManifest, TaskPlan
from bindocracy.runs.status import read_task_status
from bindocracy.store.records import (
    ArtifactRecord,
    CandidateType,
    CollectedRun,
    DesignRecord,
    DesignStatus,
    MetricDirection,
    MetricRecord,
    RunRecord,
    RunStatus,
    stable_id,
)

METRICS_FILE = "final_ranked_designs/all_designs_metrics.csv"
# BoltzGen names this directory for the budget -- final_40_designs at 40,
# final_10_designs at 10 -- so it has to be found, not assumed.
# Key the adapter adds to each parsed row; not a BoltzGen column.
NATIVE_ID = "_native_id"
RANKED_DIR = "final_ranked_designs"
STRUCTURE_GLOB = "final_*_designs"

# Native scores worth promoting out of 237 columns, with the direction that
# makes a value good. Anything absent from a row is simply not emitted.
NATIVE_METRICS: dict[str, MetricDirection] = {
    "design_to_target_iptm": MetricDirection.MAX,
    "design_ptm": MetricDirection.MAX,
    "min_design_to_target_pae": MetricDirection.MIN,
    "complex_plddt": MetricDirection.MAX,
    "design_to_target_ipsae": MetricDirection.MAX,
    "quality_score": MetricDirection.MAX,
    "final_rank": MetricDirection.MIN,
}

_SEQUENCE = re.compile(r"[A-Z]+")
_MEDIA_TYPES = {".csv": "text/csv", ".json": "application/json",
                ".log": "text/plain", ".yaml": "application/yaml", ".cif": "chemical/x-cif"}


class BoltzGenOutputAdapter(OutputAdapter):
    tool = "boltzgen"

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
        passed = 0

        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            rows, rejected = _read_metrics(run_dir / task.designs, task, seen)
            attempted += status.n_attempted if status and status.n_attempted is not None else task.n_requested
            task_passed = sum(1 for row in rows if _truthy(row.get("pass_filters")))
            passed += task_passed
            per_task[f"{task.task_id:04d}"] = {
                "status": status.status if status else "missing",
                "n_produced": len(rows),
                "n_passed": task_passed,
                **rejected,
            }

            for row in rows:
                design = _design_record(run.run_id, row, manifest.created_at)
                designs.append(design)
                metrics.extend(_metric_records(run.run_id, design, row))
                artifacts.extend(_structure_artifacts(run_dir, run.run_id, task, design, row))
            artifacts.extend(_task_artifacts(run_dir, run.run_id, task))

        artifacts.extend(_provenance_artifacts(run_dir, run.run_id, manifest))
        started, finished = _run_window(run_dir, manifest)

        collected_run = run.model_copy(update={
            "status": _run_status(manifest, per_task, len(designs)),
            "n_requested": manifest.designs_per_task * len(manifest.tasks),
            "n_attempted": max(attempted, len(designs)),
            "n_produced": len(designs),
            "n_passed": passed,
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
        manifest = _read_manifest(run_dir)
        for task in manifest.tasks:
            rows, _ = _read_metrics(run_dir / task.designs, task, set())
            if not rows:
                return False
        return True


def _read_manifest(run_dir: Path) -> RunManifest:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise CollectionError(f"missing run manifest: {manifest_path}")
    return RunManifest.read(manifest_path)


def _read_metrics(
    path: Path, task: TaskPlan, seen: set[str]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return the valid rows of one metrics CSV, plus rejection counts.

    A row is rejected rather than fatal, so one malformed line does not cost
    the run every design around it.
    """
    counts = {"n_invalid": 0}
    if not path.is_file():
        return [], counts

    with path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows = []
    for row in raw_rows:
        row_id = (row.get("id") or "").strip()
        sequence = (row.get("designed_sequence") or "").strip().upper()
        if not row_id:
            counts["n_invalid"] += 1
            continue
        # BoltzGen numbers designs from 0 within each task, so two tasks of one
        # run produce the same ids. Qualify them the way Mosaic's driver
        # already does, or the second task is rejected as duplicates.
        native_id = f"task-{task.task_id:04d}-{row_id}"
        if native_id in seen:
            counts["n_invalid"] += 1
            continue
        # X is a legal letter but an unknown residue, so a sequence carrying
        # one is not orderable. BoltzGen tracks this itself in `has_x`; check
        # the sequence directly so a missing column cannot let one through.
        if not sequence or _SEQUENCE.fullmatch(sequence) is None or "X" in sequence:
            counts["n_invalid"] += 1
            continue
        seen.add(native_id)
        rows.append({**row, NATIVE_ID: native_id})
    return rows, counts


def _design_record(run_id: str, row: dict[str, str], fallback: datetime) -> DesignRecord:
    native_id = row[NATIVE_ID]
    sequence = row["designed_sequence"].strip().upper()
    return DesignRecord(
        design_id=stable_id("design", run_id, native_id),
        run_id=run_id,
        native_id=native_id,
        # A BoltzGen design is a folded complex, not a bare sequence.
        candidate_type=CandidateType.COMPLEX,
        sequence=sequence,
        status=DesignStatus.PRODUCED if _truthy(row.get("pass_filters")) else DesignStatus.PARTIAL,
        metadata={
            "pass_filters": _truthy(row.get("pass_filters")),
            "final_rank": _number(row.get("final_rank")),
            "file_name": (row.get("file_name") or "").strip() or None,
        },
        created_at=fallback,
    )


def _metric_records(run_id: str, design: DesignRecord, row: dict[str, str]) -> list[MetricRecord]:
    records = []
    for name, direction in NATIVE_METRICS.items():
        value = _number(row.get(name))
        if value is None:
            continue
        records.append(
            MetricRecord(
                metric_id=stable_id("metric", run_id, design.design_id, name, "0"),
                run_id=run_id,
                design_id=design.design_id,
                name=f"boltzgen_{name}",
                value=value,
                direction=direction,
                measured_at=design.created_at,
            )
        )
    return records


def _structure_artifacts(
    run_dir: Path, run_id: str, task: TaskPlan, design: DesignRecord, row: dict[str, str]
) -> list[ArtifactRecord]:
    """The ranked CIF for one design, whose filename carries its rank."""
    file_name = (row.get("file_name") or "").strip()
    rank = _number(row.get("final_rank"))
    structures = _structure_dir(run_dir, task)
    if not file_name or rank is None or structures is None:
        return []
    relative = f"{task.directory}/{RANKED_DIR}/{structures.name}/rank{int(rank):02d}_{file_name}"
    artifact = _artifact(run_dir, run_id, relative, "design_complex")
    if artifact is None:
        return []
    return [artifact.model_copy(update={"design_id": design.design_id})]


def _structure_dir(run_dir: Path, task: TaskPlan) -> Path | None:
    """Find the ranked-design directory, whatever budget named it."""
    ranked = run_dir / task.directory / RANKED_DIR
    matches = sorted(path for path in ranked.glob(STRUCTURE_GLOB) if path.is_dir())
    return matches[0] if matches else None


def _task_artifacts(run_dir: Path, run_id: str, task: TaskPlan) -> list[ArtifactRecord]:
    wanted = [("native_design_table", task.designs), ("task_status", task.status),
              ("log", task.log)]
    return [
        artifact
        for kind, relative in wanted
        if (artifact := _artifact(run_dir, run_id, relative, kind)) is not None
    ]


def _provenance_artifacts(
    run_dir: Path, run_id: str, manifest: RunManifest
) -> list[ArtifactRecord]:
    kinds = {"spec": "design_spec", "driver": "driver_script"}
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
    # BoltzGen filters its own pool, so producing fewer than requested is
    # normal rather than partial; an incomplete task is what makes it partial.
    return RunStatus.SUCCEEDED if complete else RunStatus.PARTIAL


def _run_window(run_dir: Path, manifest: RunManifest) -> tuple[datetime | None, datetime | None]:
    starts, finishes = [], []
    for task in manifest.tasks:
        status = read_task_status(run_dir / task.status)
        if status is None:
            continue
        starts.append(status.started_at)
        finishes.append(status.finished_at)
    starts = [value for value in starts if value is not None]
    finishes = [value for value in finishes if value is not None]
    return (min(starts) if starts else None, max(finishes) if finishes else None)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}
