"""Read a finished Proteina-Complexa run directory into normalized records.

The tool writes its results across two trees, and they answer different
questions:

* `inference/<stem>/all_rewards_*.csv` -- every candidate the run generated,
  scored by the AF2 reward. Its sequences are integer `aatype` vectors, not
  letters, so it counts candidates but cannot name them.
* `evaluation_results/<stem>/binder_results_*.csv` -- the candidates that
  survived the reward filter, refolded and scored. This is the only file with a
  readable sequence in it, so a design is a row here.
* `evaluation_results/<stem>/all_successes_protein_binder_self.csv` -- the
  subset that cleared the evaluation thresholds. A verdict, not a design list.

So generating, surviving, and passing are three different counts, and this
records all three. In the archived benchmark they were 40, 40 and 2.

Two shapes of this tool's output need saying out loud:

* `filter.dedup_sequence` drops identical sequences before the top-N cut, so
  **fewer rows than `keep_per_job` is legitimate**, not truncation.
* The evaluation stage *copies* every sample directory from `inference/` into
  `evaluation_results/` before refolding, so each design's structure exists
  twice on disk. Only the evaluated copy is recorded, or every design would
  carry two identical artifact rows.
"""

from __future__ import annotations

import csv
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
    run_status,
    run_window,
    unique_by_uri,
)
from bindocracy.runs.manifest import RunManifest, TaskPlan
from bindocracy.runs.status import read_task_status
from bindocracy.store.records import (
    ArtifactRecord,
    CandidateType,
    CollectedRun,
    DecisionKind,
    DecisionRecord,
    DesignRecord,
    MetricDirection,
    MetricRecord,
    RunRecord,
    stable_id,
)
from bindocracy.tools.proteina_complexa.launch import (
    criteria_file,
    output_stem,
    rewards_file,
    successes_file,
)

# The evaluated columns worth carrying, and which direction is better. Every
# one is a `self_*` column: the shipped `sequence_types: [self]` means the
# design's sequence is the model's own, not an inverse-folding redesign of it.
METRICS: tuple[tuple[str, str, MetricDirection], ...] = (
    ("self_complex_i_pTM", "proteina_complex_iptm", MetricDirection.MAX),
    ("self_complex_pLDDT", "proteina_complex_plddt", MetricDirection.MAX),
    ("self_complex_i_pAE", "proteina_complex_ipae", MetricDirection.MIN),
    ("self_complex_avg_ipSAE", "proteina_complex_avg_ipsae", MetricDirection.MAX),
    ("self_binder_scRMSD_ca", "proteina_binder_scrmsd_ca", MetricDirection.MIN),
    ("self_esm_pseudo_perplexity", "proteina_esm_pseudo_perplexity", MetricDirection.MIN),
)

# The evaluation gate, named for the file that defines it. It is a conjunction
# of three thresholds, and the run records all three rather than whichever one
# happened to be configurable.
SUCCESS_FILTER = "proteina_complexa_protein_binder"

_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")


class ProteinaComplexaOutputAdapter(OutputAdapter):
    tool = "proteina_complexa"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = read_manifest(run_dir)
        if manifest.run_id != run.run_id:
            raise CollectionError(
                f"run {run.run_id} does not match manifest run {manifest.run_id}"
            )
        task_name = _task_name(manifest)

        designs: list[DesignRecord] = []
        metrics: list[MetricRecord] = []
        decisions: list[DecisionRecord] = []
        artifacts: list[ArtifactRecord] = []
        seen: set[str] = set()
        per_task: dict[str, Any] = {}
        passed = 0
        generated = 0
        judged = True

        for task in manifest.tasks:
            task_dir = run_dir / task.directory
            status = read_task_status(run_dir / task.status)

            rows, rejected = _read_designs(run_dir / task.designs, task, seen)
            n_generated = _count_rows(task_dir / rewards_file(task_name))
            generated += n_generated

            criteria = _read_criteria(task_dir / criteria_file(task_name))
            successes, orphans = _read_successes(
                task_dir / successes_file(task_name), rows
            )
            # A verdict table that never appeared is not a genuine zero: it is
            # an evaluation stage that did not finish. Counting it as zero
            # would report a real result the run never measured.
            complete = criteria is not None and successes is not None
            judged = judged and complete
            task_passed = len(successes) if successes is not None else 0
            passed += task_passed

            per_task[f"{task.task_id:04d}"] = {
                "status": status.status if status else "missing",
                # What the tool drew, before its own reward filter.
                "n_generated": n_generated,
                # What survived the filter and reached evaluation. Lower than
                # keep_per_job is legitimate: dedup_sequence drops identical
                # sequences before the top-N cut.
                "n_produced": len(rows),
                "n_passed": task_passed if complete else None,
                # Empty on a healthy run. A sequence marked successful that no
                # design row names is a stale or corrupted verdict table.
                "successes_naming_no_design": sorted(orphans),
                # §1.7 of known-issues reports every ipSAE as 0.0; it is not,
                # but many are, and the count is what makes that checkable
                # rather than a claim.
                "n_zero_avg_ipsae": sum(
                    1 for row in rows if _number(row.get("self_complex_avg_ipSAE")) == 0.0
                ),
                **rejected,
            }

            for row in rows:
                design = _design_record(run.run_id, row, task, manifest.created_at)
                designs.append(design)
                metrics.extend(_metric_records(run.run_id, design, row))
                if complete:
                    decisions.append(
                        _success_decision(
                            run.run_id, design, task, criteria, successes
                        )
                    )
                record = _design_artifact(run_dir, run.run_id, task, design, row)
                if record is not None:
                    artifacts.append(record)
            artifacts.extend(_task_artifacts(run_dir, run.run_id, task, task_name))

        artifacts.extend(
            provenance_artifacts(run_dir, run.run_id, manifest, {"registry": "target_registry"})
        )
        statuses = [read_task_status(run_dir / task.status) for task in manifest.tasks]
        started, finished = run_window(statuses)

        collected_run = run.model_copy(update={
            "status": run_status(len(designs), statuses, complete=judged),
            "n_requested": manifest.designs_per_task * len(manifest.tasks),
            # What the tool drew on the way there, which is larger than what it
            # kept whenever the search runs more than one replica.
            "n_attempted": max(generated, len(designs)),
            "n_produced": len(designs),
            # Counted only over tasks whose evaluation actually ran.
            "n_passed": passed if judged else None,
            "count_details": {"tasks": per_task},
            "started_at": started,
            "finished_at": finished,
        })
        return CollectedRun(
            run=collected_run,
            designs=tuple(designs),
            metrics=tuple(metrics),
            decisions=tuple(decisions),
            artifacts=tuple(unique_by_uri(artifacts)),
        )

    def succeeded(self, run_dir: Path) -> bool:
        manifest = read_manifest(run_dir)
        task_name = _task_name(manifest)
        for task in manifest.tasks:
            rows, _ = _read_designs(run_dir / task.designs, task, set())
            if not rows:
                return False
            if not (run_dir / task.directory / successes_file(task_name)).is_file():
                return False
        return True


def _task_name(manifest: RunManifest) -> str:
    """The registry key this run designed against, from the stored config."""
    task_name = manifest.config.model_config_json["registry"]["task_name"]
    return str(task_name)


def _read_designs(
    path: Path, task: TaskPlan, seen: set[str]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Valid rows of one binder_results table, plus rejection counts.

    A malformed row is skipped rather than fatal: it must not cost the run the
    designs written before and after it.
    """
    counts = {"n_invalid": 0, "n_duplicate": 0}
    if not path.is_file():
        return [], counts

    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            native_id = _native_id(row, task)
            sequence = (row.get("self_sequence") or "").strip().upper()
            if native_id is None or _SEQUENCE.fullmatch(sequence) is None:
                counts["n_invalid"] += 1
                continue
            if native_id in seen:
                # Proteina numbers samples per task, so two tasks can produce
                # the same directory name. Qualified IDs make that impossible;
                # a collision here is a genuinely repeated row.
                counts["n_duplicate"] += 1
                continue
            seen.add(native_id)
            rows.append(row)
    return rows, counts


def _native_id(row: dict[str, str], task: TaskPlan) -> str | None:
    """The sample directory stem, qualified by task.

    `pdb_path` is `.../job_0_n_271_id_0_bon_orig2_r0/job_0_n_271_id_0_bon_orig2_r0.pdb`,
    and the stem carries the whole identity: which job, what length, which
    sample, and which best-of-n replica of it.
    """
    raw = (row.get("pdb_path") or "").strip()
    if not raw:
        return None
    stem = Path(raw).stem
    return f"task-{task.task_id:04d}-{stem}" if stem else None


def _count_rows(path: Path) -> int:
    """Rows in a generation-stage table, which counts candidates only.

    Its sequences are integer `aatype` vectors rather than letters, so it can
    say how many were drawn but not which they were.
    """
    if not path.is_file():
        return 0
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _read_criteria(path: Path) -> dict[str, Any] | None:
    """The thresholds the evaluation gate applied, as it recorded them.

    Read from the file rather than assumed, because the whole protocol is the
    verdict: three thresholds joined by AND, only one of which is prominent in
    any configuration.
    """
    if not path.is_file():
        return None
    try:
        criteria = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return criteria if isinstance(criteria, dict) else None


def _read_successes(
    path: Path, rows: list[dict[str, str]]
) -> tuple[set[str] | None, set[str]]:
    """Sequences the gate passed, and any it names that no design row does.

    The verdict table is evidence, not truth. A sequence marked successful that
    the primary table never produced is a stale or corrupted file marking the
    wrong design as passed, so it is counted rather than silently trusted.
    """
    if not path.is_file():
        return None, set()

    produced = {(row.get("self_sequence") or "").strip().upper() for row in rows}
    passed: set[str] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = (row.get("binder_sequence") or "").strip().upper()
            if sequence:
                passed.add(sequence)
    return passed, passed - produced


def _design_record(
    run_id: str, row: dict[str, str], task: TaskPlan, fallback: datetime
) -> DesignRecord:
    native_id = _native_id(row, task)
    assert native_id is not None  # _read_designs rejected the alternative
    sequence = row["self_sequence"].strip().upper()
    length = _number(row.get("L"))
    return DesignRecord(
        design_id=stable_id("design", run_id, native_id),
        run_id=run_id,
        native_id=native_id,
        # Proteina-Complexa co-generates sequence and structure, and every
        # sample is written as the target plus the binder, so a design is a
        # complex rather than a bare sequence or backbone.
        candidate_type=CandidateType.COMPLEX,
        sequence=sequence,
        length=int(length) if length else len(sequence),
        # Per task, not per design: the seed is a run-level Hydra override and
        # the tool records no per-sample seed anywhere.
        seed=None,
        created_at=fallback,
    )


def _metric_records(
    run_id: str, design: DesignRecord, row: dict[str, str]
) -> list[MetricRecord]:
    records = []
    for column, name, direction in METRICS:
        value = _number(row.get(column))
        if value is None:
            continue
        records.append(
            MetricRecord(
                metric_id=stable_id("metric", run_id, design.design_id, name, "0"),
                run_id=run_id,
                design_id=design.design_id,
                name=name,
                value=value,
                direction=direction,
                measured_at=design.created_at,
            )
        )
    return records


def _success_decision(
    run_id: str,
    design: DesignRecord,
    task: TaskPlan,
    criteria: dict[str, Any],
    successes: set[str],
) -> DecisionRecord:
    """One design's verdict, carrying the whole protocol that produced it."""
    return DecisionRecord(
        decision_id=stable_id("decision", run_id, design.design_id, SUCCESS_FILTER),
        run_id=run_id,
        design_id=design.design_id,
        kind=DecisionKind.FILTER,
        name=SUCCESS_FILTER,
        passed=design.sequence in successes,
        scope_id=filter_scope(run_id, task.task_id),
        # The thresholds as the run recorded them. A verdict without them says
        # a design passed without saying what it passed.
        reason={"thresholds": criteria.get("thresholds", {}),
                "sequence_types": criteria.get("sequence_types", [])},
        created_at=design.created_at,
    )


def filter_scope(run_id: str, task_id: int) -> str:
    """The pool a verdict was reached within -- one task, not the whole run."""
    return stable_id("scope", run_id, f"task-{task_id:04d}", SUCCESS_FILTER)


def _design_artifact(
    run_dir: Path, run_id: str, task: TaskPlan, design: DesignRecord, row: dict[str, str]
) -> ArtifactRecord | None:
    """The evaluated complex, and only that copy.

    Evaluation copies each sample directory out of `inference/`, so the same
    structure exists under both roots. `pdb_path` names the evaluated one.
    """
    raw = (row.get("pdb_path") or "").strip()
    if not raw or Path(raw).is_absolute():
        # The table records the path the tool wrote from its own working
        # directory. An absolute one belongs to another machine's run.
        return None
    relative = f"{task.directory}/{raw.removeprefix('./')}"
    return artifact(run_dir, run_id, relative, "design_complex", design_id=design.design_id)


def _task_artifacts(
    run_dir: Path, run_id: str, task: TaskPlan, task_name: str
) -> list[ArtifactRecord]:
    wanted = [
        ("native_designs", task.designs),
        ("generation_rewards", f"{task.directory}/{rewards_file(task_name)}"),
        ("success_table", f"{task.directory}/{successes_file(task_name)}"),
        ("success_criteria", f"{task.directory}/{criteria_file(task_name)}"),
        ("task_status", task.status),
        ("log", task.log),
    ]
    return [
        record
        for kind, relative in wanted
        if (record := artifact(run_dir, run_id, relative, kind)) is not None
    ]


def _number(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


__all__ = [
    "METRICS",
    "SUCCESS_FILTER",
    "ProteinaComplexaOutputAdapter",
    "filter_scope",
    "output_stem",
]
