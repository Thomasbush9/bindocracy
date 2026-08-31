"""Read a finished PXDesign run directory into normalized records.

Written against the observed output of the 2026-08-26 benchmark run. Three
things about that output shape this parser:

* **The table is always full.** The CLI appends `--min_total_return N
  --max_success_return N`, so `summary.csv` has exactly `--N_sample` rows
  whether or not that many designs passed anything. Twenty-four of the
  benchmark's forty rows failed every filter. They are still designs the run
  produced; what they are not is hits.
* **There are four verdicts, not one.** AF2-IG-easy, AF2-IG, Protenix and
  Protenix-basic each get a column, and they disagree: 16, 0, 5 and 6 of the
  forty. Each becomes its own decision, and `n_passed` counts the designs that
  satisfied every family the run's preset actually ran.
* **The structure a row points at is already relative** -- to
  `design_outputs/<task_name>/` -- and names the filter it survived
  (`passing-Protenix-basic/rank_1.cif`) or the raw design
  (`orig_designed/rank_40.cif`).

Which verdicts to expect is read from the manifest, never inferred from the
table. An `extended` run whose Protenix stage died writes the same table an
AF2-only run does, and guessing from the columns present would turn a broken
run into a successful one with a smaller filter set.
"""

from __future__ import annotations

import csv
import math
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
    DesignStatus,
    MetricDirection,
    MetricRecord,
    RunRecord,
    stable_id,
)

SUMMARY_FILE = "summary.csv"
DESIGN_OUTPUTS = "design_outputs"
# PXDesign's own record of which ranking mode actually ran, which is not the
# same question as which --preset was asked for.
TASK_INFO = "task_info.json"
# The resolved configuration PXDesign wrote for itself, after the CLI, the
# preset and the Hydra overrides had all had their say.
RESOLVED_CONFIG = "config.yaml"

# The scores worth promoting out of 31 columns, with the direction that makes a
# value good. Anything absent from a row is simply not emitted.
NATIVE_METRICS: dict[str, MetricDirection] = {
    "af2_plddt": MetricDirection.MAX,
    "af2_ptm": MetricDirection.MAX,
    "af2_iptm": MetricDirection.MAX,
    "af2_pAE": MetricDirection.MIN,
    "af2_ipAE": MetricDirection.MIN,
    "af2_monomer_plddt": MetricDirection.MAX,
    "af2_bound_unbound_RMSD": MetricDirection.MIN,
    "af2_binder_pred_design_rmsd": MetricDirection.MIN,
    "af2_complex_pred_design_rmsd": MetricDirection.MIN,
    "ptx_plddt": MetricDirection.MAX,
    "ptx_ptm": MetricDirection.MAX,
    "ptx_iptm": MetricDirection.MAX,
    "ptx_pred_design_rmsd": MetricDirection.MIN,
    "Rg": MetricDirection.NONE,
}

# PXDesign's four verdicts, by the column that carries each.
FILTERS: dict[str, str] = {
    "AF2-IG-easy-success": "pxdesign_af2ig_easy",
    "AF2-IG-success": "pxdesign_af2ig",
    "Protenix-success": "pxdesign_protenix",
    "Protenix-basic-success": "pxdesign_protenix_basic",
}
AF2_FILTERS = ("AF2-IG-easy-success", "AF2-IG-success")
PROTENIX_FILTERS = ("Protenix-success", "Protenix-basic-success")

# Which verdict columns each preset is *expected* to write. `preview` disables
# the Protenix filters through Hydra overrides; `extended` runs both families.
EXPECTED_FILTERS: dict[str, tuple[str, ...]] = {
    "preview": AF2_FILTERS,
    "extended": AF2_FILTERS + PROTENIX_FILTERS,
}
# What counts as a hit under each preset: one design that satisfied every model
# family the preset ran.
HIT_FILTERS: dict[str, tuple[str, ...]] = {
    "preview": ("AF2-IG-easy-success",),
    "extended": ("AF2-IG-easy-success", "Protenix-success"),
}
RANK_NAME = "pxdesign_rank"

# Keys the adapter adds to each parsed row; neither is a PXDesign column.
NATIVE_ID = "_native_id"
RANK = "_rank"

_SEQUENCE = re.compile(r"[A-Z]+")


class PXDesignOutputAdapter(OutputAdapter):
    tool = "pxdesign"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = read_manifest(run_dir)
        if manifest.run_id != run.run_id:
            raise CollectionError(
                f"run {run.run_id} does not match manifest run {manifest.run_id}"
            )
        preset = _preset_of(manifest)
        expected = _expectations(manifest)

        designs: list[DesignRecord] = []
        metrics: list[MetricRecord] = []
        artifacts: list[ArtifactRecord] = []
        decisions: list[DecisionRecord] = []
        seen: set[str] = set()
        per_task: dict[str, Any] = {}
        passed = 0
        evaluated = True

        for task in manifest.tasks:
            task_name = task_name_of(task)
            status = read_task_status(run_dir / task.status)
            rows, rejected, columns = _read_summary(
                run_dir / task.designs, task, seen, expected
            )
            # Which verdicts this task was supposed to write, and which of them
            # it actually did. A missing one means the filter stage did not
            # finish, not that the preset never asked for it.
            missing = [name for name in EXPECTED_FILTERS[preset] if name not in columns]
            malformed = _shape_problems(rows, task)
            complete = not missing and not malformed and bool(rows)
            evaluated = evaluated and complete
            task_passed = (
                sum(1 for row in rows if _is_hit(row, preset)) if complete else 0
            )
            passed += task_passed
            per_task[f"{task.task_id:04d}"] = {
                "status": status.status if status else "missing",
                "n_produced": len(rows),
                # Only meaningful when every expected verdict is present; a
                # count taken from half the filters would read as a real zero.
                "n_passed": task_passed if complete else None,
                # Per-filter counts, because the four disagree and a single
                # n_passed hides by how much.
                "n_by_filter": {
                    FILTERS[column]: sum(1 for row in rows if _truthy(row.get(column)))
                    for column in EXPECTED_FILTERS[preset]
                    if column in columns
                },
                # Empty on a healthy run. Non-empty means this task's table is
                # missing verdicts the preset should have produced.
                "missing_filters": [FILTERS[column] for column in missing],
                # Empty on a healthy run. Non-empty means the table is not the
                # shape this task was planned to produce.
                "shape_problems": malformed,
                **rejected,
            }

            for row in rows:
                design = _design_record(run.run_id, row, manifest.created_at)
                designs.append(design)
                metrics.extend(_metric_records(run.run_id, design, row))
                decisions.extend(
                    _decision_records(run.run_id, design, row, task, preset, columns)
                )
                artifacts.extend(
                    _design_artifacts(run_dir, run.run_id, task, task_name, design, row)
                )
            artifacts.extend(_task_artifacts(run_dir, run.run_id, task, task_name))

        artifacts.extend(provenance_artifacts(
            run_dir, run.run_id, manifest, {"spec": "design_spec"},
        ))
        statuses = [read_task_status(run_dir / task.status) for task in manifest.tasks]
        started, finished = run_window(statuses)
        requested = manifest.designs_per_task * len(manifest.tasks)

        collected_run = run.model_copy(update={
            # PXDesign pads its table to exactly what was asked for, so fewer
            # rows than requested is work that did not finish, never filtering.
            # A task missing an expected verdict, or holding a table that is
            # not the planned shape, is unfinished for the same reason -- and
            # an aggregate row count would let one task's shortfall be covered
            # by another's surplus.
            "status": run_status(len(designs), statuses, complete=evaluated),
            "n_requested": requested,
            "n_attempted": max(_attempted(statuses, manifest.tasks), len(designs)),
            "n_produced": len(designs),
            # Counted only over tasks whose filters all ran, so an interrupted
            # evaluation cannot report hits it did not measure.
            "n_passed": passed if evaluated else None,
            "count_details": {"preset": preset, "tasks": per_task},
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
        preset = _preset_of(manifest)
        expected = _expectations(manifest)
        for task in manifest.tasks:
            rows, _, columns = _read_summary(
                run_dir / task.designs, task, set(), expected
            )
            if not rows or any(name not in columns for name in EXPECTED_FILTERS[preset]):
                return False
        return True


class Expectations:
    """What the manifest says this run's rows must look like.

    Checked rather than assumed, because every one of these is a way for the
    table to be well-formed and still not describe the run that was planned.
    """

    def __init__(self, task_name: str, binder_length: int | None) -> None:
        self.task_name = task_name
        self.binder_length = binder_length

    def rejects(self, row: dict[str, str], sequence: str) -> bool:
        if self.binder_length is not None and len(sequence) != self.binder_length:
            return True
        name = (row.get("task_name") or "").strip()
        return bool(name) and name != self.task_name


def _preset_of(manifest: RunManifest) -> str:
    """The preset this run was planned with, which decides what to expect."""
    preset = manifest.workflow.get("preset")
    if preset not in EXPECTED_FILTERS:
        raise CollectionError(
            f"run {manifest.run_id} records preset {preset!r}, which is not one "
            f"of {', '.join(sorted(EXPECTED_FILTERS))}. Without it there is no "
            "way to tell a preset that skipped Protenix from a run whose "
            "Protenix stage failed."
        )
    return str(preset)


def _expectations(manifest: RunManifest) -> Expectations:
    length = manifest.workflow.get("binder_length")
    return Expectations(
        task_name=task_name_of(manifest.tasks[0]),
        binder_length=int(length) if isinstance(length, int) else None,
    )


def _shape_problems(rows: list[dict[str, Any]], task: TaskPlan) -> list[str]:
    """How this task's table differs from the one the plan asked for.

    PXDesign pads to exactly `--N_sample` rows ranked 1..N, so anything else is
    a table that did not come out of the run as planned. Checked per task
    because a run-level count lets one task's shortfall hide behind another's
    surplus.
    """
    problems = []
    if len(rows) != task.n_requested:
        problems.append(f"{len(rows)} rows, expected {task.n_requested}")
    ranks = [row[RANK] for row in rows]
    if len(set(ranks)) != len(ranks):
        problems.append("duplicate ranks")
    expected = set(range(1, task.n_requested + 1))
    if set(ranks) != expected and len(rows) == task.n_requested:
        problems.append(f"ranks are not 1..{task.n_requested}")
    return problems


def task_name_of(task: TaskPlan) -> str:
    """The results directory PXDesign wrote, from the task's own designs path.

    Taken from the manifest rather than from the stored spec so the directory
    the adapter reads and the file the plan named can never be two places.
    """
    relative = Path(task.designs).relative_to(task.directory)
    return relative.parts[1]


def _read_summary(
    path: Path, task: TaskPlan, seen: set[str], expected: Expectations
) -> tuple[list[dict[str, Any]], dict[str, int], set[str]]:
    """One summary table's valid rows, its rejection counts, and its columns.

    A row is rejected rather than fatal, so one malformed line does not cost
    the run the designs around it.
    """
    counts = {"n_invalid": 0}
    if not path.is_file():
        return [], counts, set()

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        raw_rows = list(reader)

    rows = []
    for row in raw_rows:
        rank = _rank(row)
        sequence = (row.get("sequence") or "").strip().upper()
        if rank is None:
            counts["n_invalid"] += 1
            continue
        # PXDesign ranks from 1 within each output directory, and each task has
        # its own, so two tasks of one run produce the same ranks. Qualify
        # them, or the second task collects as duplicates of the first.
        native_id = f"task-{task.task_id:04d}-rank-{rank:04d}"
        # X is a legal letter but an unknown residue, so a sequence carrying
        # one cannot be ordered.
        if (
            native_id in seen
            or not sequence
            or _SEQUENCE.fullmatch(sequence) is None
            or "X" in sequence
            or expected.rejects(row, sequence)
        ):
            counts["n_invalid"] += 1
            continue
        seen.add(native_id)
        rows.append({**row, NATIVE_ID: native_id, RANK: rank})
    return rows, counts, columns


def _design_record(run_id: str, row: dict[str, Any], fallback: datetime) -> DesignRecord:
    native_id = row[NATIVE_ID]
    return DesignRecord(
        design_id=stable_id("design", run_id, native_id),
        run_id=run_id,
        native_id=native_id,
        # The structure a row points at holds the binder and the target
        # together, whichever filter stage chose it.
        candidate_type=CandidateType.COMPLEX,
        sequence=(row.get("sequence") or "").strip().upper(),
        # A complete row is a produced design, including one the table was
        # padded with. Which filters it cleared is a decision, not a lesser
        # kind of existence.
        status=DesignStatus.PRODUCED,
        metadata={
            # Which stage's structure was kept: 'ptx', 'af2' or 'orig'.
            "chosen_struct_type": (row.get("chosen_struct_type") or "").strip() or None,
            "task_name": (row.get("task_name") or "").strip() or None,
        },
        created_at=fallback,
    )


def _metric_records(
    run_id: str, design: DesignRecord, row: dict[str, Any]
) -> list[MetricRecord]:
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
                name=f"pxdesign_{name}",
                value=value,
                direction=direction,
                measured_at=design.created_at,
            )
        )
    return records


def _decision_records(
    run_id: str,
    design: DesignRecord,
    row: dict[str, Any],
    task: TaskPlan,
    preset: str,
    columns: set[str],
) -> list[DecisionRecord]:
    """PXDesign's own verdicts, and where the design ranked among its siblings.

    Only the verdicts this preset ran *and* actually wrote become decisions.
    An absent column is a filter whose answer nobody knows, and recording it as
    a failure would be inventing one.
    """
    records = [
        DecisionRecord(
            decision_id=stable_id("decision", run_id, design.design_id, FILTERS[column]),
            run_id=run_id,
            design_id=design.design_id,
            kind=DecisionKind.FILTER,
            name=FILTERS[column],
            passed=_truthy(row[column]),
            created_at=design.created_at,
        )
        for column in EXPECTED_FILTERS[preset]
        if column in columns and column in row
    ]
    records.append(
        DecisionRecord(
            decision_id=stable_id("decision", run_id, design.design_id, RANK_NAME),
            run_id=run_id,
            design_id=design.design_id,
            kind=DecisionKind.RANK,
            name=RANK_NAME,
            rank=row[RANK],
            # PXDesign ranks each task's pool on its own, so a two-task run has
            # two designs ranked first and a run-scoped rank would record that
            # as a contradiction.
            scope_id=rank_scope(run_id, task.task_id),
            created_at=design.created_at,
        )
    )
    return records


def rank_scope(run_id: str, task_id: int) -> str:
    """The pool a PXDesign rank is meaningful within: one task's designs."""
    return stable_id("rank-scope", run_id, f"task-{task_id:04d}")


def _is_hit(row: dict[str, Any], preset: str) -> bool:
    """Did this design satisfy every filter family this preset ran."""
    return all(_truthy(row.get(column)) for column in HIT_FILTERS[preset])


def _design_artifacts(
    run_dir: Path,
    run_id: str,
    task: TaskPlan,
    task_name: str,
    design: DesignRecord,
    row: dict[str, Any],
) -> list[ArtifactRecord]:
    """The structure this row kept, at the relative path the table names.

    The path comes from the tool, so it is treated as untrusted: anything
    absolute or climbing out of the task directory is dropped rather than
    recorded as an artifact of this run.
    """
    chosen = (row.get("chosen_struct_path") or "").strip()
    if not chosen or not _is_contained(chosen):
        return []
    relative = f"{task.directory}/{DESIGN_OUTPUTS}/{task_name}/{chosen}"
    record = artifact(run_dir, run_id, relative, "design_complex",
                      design_id=design.design_id)
    return [record] if record is not None else []


def _is_contained(relative: str) -> bool:
    """True when a tool-supplied path stays inside the directory it names."""
    candidate = Path(relative)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _task_artifacts(
    run_dir: Path, run_id: str, task: TaskPlan, task_name: str
) -> list[ArtifactRecord]:
    wanted = [
        ("native_design_table", task.designs),
        ("filter_mode", f"{task.directory}/{DESIGN_OUTPUTS}/{task_name}/{TASK_INFO}"),
        ("resolved_config", f"{task.directory}/{RESOLVED_CONFIG}"),
        ("task_status", task.status),
        ("log", task.log),
    ]
    return [
        record
        for kind, relative in wanted
        if (record := artifact(run_dir, run_id, relative, kind)) is not None
    ]


def _attempted(statuses: list, tasks: tuple[TaskPlan, ...]) -> int:
    """What the run set out to produce, as the tasks themselves reported it."""
    total = 0
    for status, task in zip(statuses, tasks, strict=True):
        if status is not None and status.n_attempted is not None:
            total += status.n_attempted
        else:
            total += task.n_generated if task.n_generated is not None else task.n_requested
    return total


def _rank(row: dict[str, str]) -> int | None:
    """A rank is a position, so a fractional one is a parse error, not a 1."""
    value = _number(row.get("rank"))
    if value is None or value < 1 or value != int(value):
        return None
    return int(value)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}
