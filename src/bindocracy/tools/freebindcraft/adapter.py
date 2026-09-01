"""Read a finished FreeBindCraft run directory into normalized records.

Written against the observed output of the 2026-08-26 benchmark run (49
trajectories, 396 MPNN sequences, 41 accepted). Four things about that output
shape this parser:

* **A design is an MPNN sequence, and there are two kinds of them.** Every
  redesigned backbone is predicted with AF2 and checked against a handful of
  base confidence filters *before* anything else is computed. The 233 that
  failed that check appear only in `rejected_mpnn_full_stats.csv`, carrying a
  sequence and no measurements at all; the 163 that passed it are scored in
  full and appear in `mpnn_design_stats.csv`. Both were produced. Only the
  second kind has metrics, and the first is recorded as a sequence rather than
  a complex because BindCraft keeps no structure for it.

* **`mpnn_design_stats.csv` and `rejected_mpnn_full_stats.csv` partition it.**
  A scored design is accepted exactly when it is named in the first and not in
  the second -- 163 - 122 = 41, matching both `Accepted/` and
  `final_design_stats.csv`. All three are checked against each other, because
  the run's own verdict is what `n_passed` means.

* **The ranked table is written only on the way out the door.** BindCraft
  writes `final_design_stats.csv` inside the check that ends the loop, so a
  task that stops on `max_trajectories` instead leaves it with a header and no
  rows while `Accepted/` is full. That is a documented outcome, not a failure,
  so `n_passed` is taken from the tables and the rank decisions are simply
  absent.

* **Eight of the metrics are constants.** With PyRosetta absent,
  `pr_alternative_utils` fills `dG`, `Binder_Energy_Score`, `PackStat`, and the
  hydrogen-bond counts with fixed values "chosen to pass active filters". They
  are not measurements, so they are not emitted as metrics -- storing -10.0 as
  a design's dG would put a constant into the one table meant for comparing
  tools. Which filters were thereby inert is recorded on the run instead.

Trajectories are not designs and are not recorded as such. The census of them
-- attempted, successful, clashing, low-confidence -- goes in `count_details`,
because it is the only thing that distinguishes a task that ran out of budget
from one that ran out of luck.
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
from bindocracy.tools.freebindcraft.launch import (
    ACCEPTED_DIR,
    DESIGNS_FILE,
    FAILURE_FILE,
    FINAL_FILE,
    REJECTED_DIR,
    REJECTED_FILE,
    TRAJECTORY_DIRS,
    TRAJECTORY_FILE,
)

# Per-model scores, written as `Average_X` plus `1_X` .. `5_X`. Only the models
# that ran carry a value; a multimer design run predicts with two of the five,
# so three of the columns are empty on every row and are simply not emitted.
REPLICATED_METRICS: dict[str, MetricDirection] = {
    "pLDDT": MetricDirection.MAX,
    "pTM": MetricDirection.MAX,
    "i_pTM": MetricDirection.MAX,
    "pAE": MetricDirection.MIN,
    "i_pAE": MetricDirection.MIN,
    "ipSAE": MetricDirection.MAX,
    "i_pLDDT": MetricDirection.MAX,
    "ss_pLDDT": MetricDirection.MAX,
}

# Written the same way, but only the average is worth a row: these vary little
# between the two prediction models and the per-model columns are in the
# archived table for anyone who wants them.
AVERAGED_METRICS: dict[str, MetricDirection] = {
    "Unrelaxed_Clashes": MetricDirection.MIN,
    "Relaxed_Clashes": MetricDirection.MIN,
    "Surface_Hydrophobicity": MetricDirection.NONE,
    "ShapeComplementarity": MetricDirection.MAX,
    "dSASA": MetricDirection.MAX,
    "Interface_SASA_%": MetricDirection.NONE,
    "Interface_Hydrophobicity": MetricDirection.NONE,
    "n_InterfaceResidues": MetricDirection.MAX,
    "Hotspot_RMSD": MetricDirection.MIN,
    "Target_RMSD": MetricDirection.MIN,
    "Binder_pLDDT": MetricDirection.MAX,
    "Binder_pTM": MetricDirection.MAX,
    "Binder_pAE": MetricDirection.MIN,
    "Binder_RMSD": MetricDirection.MIN,
}

# Plain columns, with no per-model form.
SINGLE_METRICS: dict[str, MetricDirection] = {
    "MPNN_score": MetricDirection.MIN,
    "MPNN_seq_recovery": MetricDirection.MAX,
}

# BindCraft predicts each design with at most five AF2 models.
MODELS = (1, 2, 3, 4, 5)

FILTER_NAME = "freebindcraft_filters"
RANK_NAME = "freebindcraft_rank"

# Columns of `rejected_mpnn_full_stats.csv` that are not filter flags.
REJECTED_KEYS = ("Design", "Sequence")

# Keys the adapter adds to each parsed row; neither is a BindCraft column.
NATIVE_ID = "_native_id"
DESIGN = "_design"

_SEQUENCE = re.compile(r"[A-Z]+")
# `<binder_name>_l<length>_s<seed>_mpnn<n>` -- the trajectory is everything
# before `_mpnn`, which is what ties a design to its backbone.
_TRAJECTORY = re.compile(r"^(?P<trajectory>.+)_mpnn(?P<index>\d+)$")


class FreeBindCraftOutputAdapter(OutputAdapter):
    tool = "freebindcraft"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = read_manifest(run_dir)
        if manifest.run_id != run.run_id:
            raise CollectionError(
                f"run {run.run_id} does not match manifest run {manifest.run_id}"
            )
        expected = Expectations.of(manifest)

        designs: list[DesignRecord] = []
        metrics: list[MetricRecord] = []
        artifacts: list[ArtifactRecord] = []
        decisions: list[DecisionRecord] = []
        seen: set[str] = set()
        per_task: dict[str, Any] = {}
        passed = 0
        evaluated = True

        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            table = _read_task(run_dir, task, seen, expected)
            problems = table.shape_problems(task)
            complete = not problems and bool(table.scored)
            evaluated = evaluated and complete
            passed += len(table.accepted)

            per_task[f"{task.task_id:04d}"] = {
                "status": status.status if status else "missing",
                # The census that says why a task stopped. `budget` is what
                # `max_trajectories` counts: successful hallucinations only.
                "trajectories": {
                    "attempted": table.trajectories_attempted,
                    "successful": table.trajectories["successful"],
                    "clashing": table.trajectories["clashing"],
                    "low_confidence": table.trajectories["low_confidence"],
                    "budget": task.n_generated,
                },
                "n_attempted": len(table.scored) + len(table.unscored),
                # Fully scored, with metrics and a kept structure.
                "n_scored": len(table.scored),
                # Predicted, then dropped by the base AF2 filters before any
                # interface metric was computed. Produced, but not measured.
                "n_rejected_before_scoring": len(table.unscored),
                "n_passed": len(table.accepted),
                # False when the task stopped on its trajectory budget: the
                # ranked table is written only by the check that ends the loop
                # on having enough designs.
                "ranked": bool(table.ranks),
                # BindCraft's own running tally, which counts trajectory
                # terminations and design rejections in one table and is
                # cumulative across a resumed design path.
                "failure_counters": table.failure_counts(),
                # Empty on a healthy run.
                "shape_problems": problems,
                **table.rejections,
            }

            for row in (*table.scored, *table.unscored):
                design = _design_record(run.run_id, row, manifest.created_at)
                designs.append(design)
                if row.get(DESIGN) in table.scored_names:
                    metrics.extend(_metric_records(run.run_id, design, row))
                decisions.extend(_decision_records(run.run_id, design, row, task, table))
                artifacts.extend(
                    _design_artifact(run_dir, run.run_id, task, design, row, table)
                )
            artifacts.extend(_task_artifacts(run_dir, run.run_id, task))

        artifacts.extend(provenance_artifacts(run_dir, run.run_id, manifest, {
            "driver": "driver",
            "target": "target_settings",
            "filters": "filter_set",
            "advanced": "advanced_settings",
        }))
        statuses = [read_task_status(run_dir / task.status) for task in manifest.tasks]
        started, finished = run_window(statuses)

        collected_run = run.model_copy(update={
            "status": run_status(len(designs), statuses, complete=evaluated),
            "n_requested": manifest.designs_per_task * len(manifest.tasks),
            # Every MPNN sequence the run predicted, whether or not it was
            # scored. The trajectories behind them are in count_details: they
            # are candidate backbones, not candidate designs, and a single
            # number mixing the two would answer neither question.
            "n_attempted": len(designs),
            "n_produced": len(designs),
            # Taken from the tables rather than from the ranked file, which a
            # task that exhausted its trajectory budget never writes.
            "n_passed": passed if evaluated else None,
            "count_details": {
                # PyRosetta is not in this image, so eight interface metrics
                # are constants and any threshold on them was inert. Recorded
                # here because it is the difference between a filter that
                # rejected nothing and a filter that measured nothing.
                "pyrosetta": manifest.workflow.get("pyrosetta"),
                "inert_filters": list(manifest.workflow.get("inert_filters") or ()),
                "tasks": per_task,
            },
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
        expected = Expectations.of(manifest)
        for task in manifest.tasks:
            table = _read_task(run_dir, task, set(), expected)
            if not table.scored:
                return False
        return True


class Expectations:
    """What the manifest says this run's rows must look like.

    Every one of these is a way for the table to be well-formed and still not
    describe the run that was planned. The epitope check is the one that could
    not be made anywhere else: BindCraft stamps `target_hotspot_residues` onto
    every row it writes, so a task launched with the wrong string -- or with an
    empty one, which means no epitope at all -- says so in its own output and
    nowhere else.
    """

    def __init__(
        self,
        *,
        hotspot_string: str,
        binder_length: tuple[int, int] | None,
        documents: dict[str, str],
    ) -> None:
        self.hotspot_string = hotspot_string
        self.binder_length = binder_length
        self.documents = documents

    @classmethod
    def of(cls, manifest: RunManifest) -> Expectations:
        workflow = manifest.workflow
        low = workflow.get("binder_min_length")
        high = workflow.get("binder_max_length")
        return cls(
            hotspot_string=str(workflow.get("hotspot_string") or ""),
            binder_length=(int(low), int(high))
            if isinstance(low, int) and isinstance(high, int)
            else None,
            # BindCraft records the stem of each settings file it was given.
            # The driver keeps the archived names, so these are checkable.
            documents={
                "TargetSettings": str(workflow.get("target_settings_name") or ""),
                "Filters": str(workflow.get("filters_name") or ""),
                "AdvancedSettings": str(workflow.get("advanced_settings_name") or ""),
            },
        )

    def rejects(self, row: dict[str, str], sequence: str) -> bool:
        """True when a scored row did not come out of the run that was planned."""
        if (row.get("Target_Hotspot") or "").strip() != self.hotspot_string:
            return True
        for column, expected in self.documents.items():
            if expected and (row.get(column) or "").strip() != expected:
                return True
        length = _number(row.get("Length"))
        if length is not None and int(length) != len(sequence):
            return True
        return self.binder_length is not None and not (
            self.binder_length[0] <= len(sequence) <= self.binder_length[1]
        )


class TaskTable:
    """One task's five tables and four directories, read together.

    Kept as one object because none of the numbers means anything alone: an
    accepted design is one the scored table names and the rejected table does
    not, and whether that agrees with `Accepted/` is the check that catches a
    stale or half-written file.
    """

    def __init__(
        self,
        scored: list[dict[str, Any]],
        unscored: list[dict[str, Any]],
        rejected_failures: dict[str, list[str]],
        ranks: dict[str, int],
        trajectories: dict[str, int],
        trajectory_rows: int,
        failures: dict[str, int],
        structures: dict[str, dict[str, str]],
        rejections: dict[str, int],
    ) -> None:
        self.scored = scored
        self.unscored = unscored
        self.rejected_failures = rejected_failures
        self.ranks = ranks
        self.trajectories = trajectories
        self.trajectory_rows = trajectory_rows
        self.failures = failures
        self.structures = structures
        self.rejections = rejections

        self.scored_names = {row[DESIGN] for row in scored}
        # A scored design the rejected table does not name is one BindCraft
        # accepted. This is the tool's own verdict, and it is what n_passed
        # counts; `Accepted/` and the ranked table are checked against it.
        self.accepted = self.scored_names - set(rejected_failures)

    @property
    def trajectories_attempted(self) -> int:
        """Every hallucination the task started, however it ended.

        The three outcome directories partition them: a trajectory is relaxed,
        or moved to Clashing, or moved to LowConfidence. Only the first counts
        towards `max_trajectories`.
        """
        return sum(self.trajectories.values())

    def failure_counts(self) -> dict[str, int]:
        """BindCraft's own tally of which filter rejected how many designs."""
        return {name: count for name, count in self.failures.items() if count}

    def shape_problems(self, task: TaskPlan) -> list[str]:
        """How this task's output differs from the run it was planned as.

        Checked per task, because a run-level count lets one task's shortfall
        hide behind another's surplus.
        """
        problems = []
        accepted_files = set(self.structures["accepted"])
        if accepted_files != self.accepted:
            problems.append(
                f"{len(self.accepted)} designs the tables call accepted, "
                f"{len(accepted_files)} structures in Accepted/"
            )
        rejected_files = set(self.structures["rejected"])
        scored_rejects = self.scored_names & set(self.rejected_failures)
        if rejected_files != scored_rejects:
            problems.append(
                f"{len(scored_rejects)} scored designs rejected, "
                f"{len(rejected_files)} structures in Rejected/"
            )
        if self.ranks:
            if set(self.ranks) != self.accepted:
                problems.append("the ranked table does not name the accepted designs")
            if sorted(self.ranks.values()) != list(range(1, len(self.ranks) + 1)):
                problems.append(f"ranks are not 1..{len(self.ranks)}")
        elif len(self.accepted) >= task.n_requested:
            # The loop writes the ranked table in the same check that ends it,
            # so enough designs and no ranking means the file is missing.
            problems.append(
                f"{len(self.accepted)} designs accepted but nothing was ranked"
            )
        if self.trajectory_rows != self.trajectories["successful"]:
            problems.append(
                f"{self.trajectory_rows} trajectory rows, "
                f"{self.trajectories['successful']} relaxed trajectories"
            )
        if task.n_generated is not None and self.trajectories["successful"] > task.n_generated:
            problems.append(
                f"{self.trajectories['successful']} successful trajectories, "
                f"budget was {task.n_generated}"
            )
        return problems


def _read_task(
    run_dir: Path, task: TaskPlan, seen: set[str], expected: Expectations
) -> TaskTable:
    """Everything one task wrote, parsed and cross-checked."""
    task_dir = run_dir / task.directory
    # Scored first: a design that failed the base AF2 filters is in the
    # rejected table alone, one that failed later is in both, and only the
    # first kind is a design this run has nowhere else.
    scored, scored_counts = _read_scored(task_dir / DESIGNS_FILE, task, seen, expected)
    rejected_failures, unscored, rejection_counts = _read_rejected(
        task_dir / REJECTED_FILE, task, seen
    )
    ranks, rank_counts = _read_ranks(task_dir / FINAL_FILE)
    return TaskTable(
        scored=scored,
        unscored=unscored,
        rejected_failures=rejected_failures,
        ranks=ranks,
        trajectories={
            name: _count_pdbs(task_dir / relative)
            for name, relative in TRAJECTORY_DIRS.items()
        },
        trajectory_rows=_count_rows(task_dir / TRAJECTORY_FILE),
        failures=_read_failures(task_dir / FAILURE_FILE),
        structures={
            "accepted": _structures(task_dir / ACCEPTED_DIR),
            "rejected": _structures(task_dir / REJECTED_DIR),
        },
        rejections={**scored_counts, **rejection_counts, **rank_counts},
    )


def _read_scored(
    path: Path, task: TaskPlan, seen: set[str], expected: Expectations
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """`mpnn_design_stats.csv`: one row per fully scored design.

    A row is rejected rather than fatal, so one malformed line does not cost
    the run the designs around it.
    """
    counts = {"n_invalid": 0, "n_foreign": 0}
    rows: list[dict[str, Any]] = []
    for row in _rows(path):
        name = (row.get("Design") or "").strip()
        sequence = (row.get("Sequence") or "").strip().upper()
        # Designs are numbered per trajectory and trajectory seeds are drawn at
        # random, so two tasks can produce the same name. Qualify them, or the
        # second task collects as duplicates of the first.
        native_id = f"task-{task.task_id:04d}-{name}"
        if (
            not name
            or native_id in seen
            # X is a legal letter but an unknown residue, so a sequence
            # carrying one cannot be ordered.
            or not sequence
            or _SEQUENCE.fullmatch(sequence) is None
            or "X" in sequence
        ):
            counts["n_invalid"] += 1
            continue
        if expected.rejects(row, sequence):
            counts["n_foreign"] += 1
            continue
        seen.add(native_id)
        rows.append({**row, NATIVE_ID: native_id, DESIGN: name})
    return rows, counts


def _read_rejected(
    path: Path, task: TaskPlan, seen: set[str]
) -> tuple[dict[str, list[str]], list[dict[str, Any]], dict[str, int]]:
    """`rejected_mpnn_full_stats.csv`: every design BindCraft turned down.

    Two kinds of row, distinguished only by whether the design also appears in
    the scored table. Both carry a sequence and a 0/1 flag per filter.

    The flags for the early-rejection kind cover only the base AF2 metrics.
    BindCraft derives their names from the failing column with
    `''.join(parts[1:])`, so a multi-word metric like `Binder_Energy_Score`
    becomes `BinderEnergyScore` and matches no column -- harmless here, because
    the base filters are all single-word, but it means an absent flag is not
    evidence a filter passed.
    """
    counts = {"n_rejected_invalid": 0}
    failures: dict[str, list[str]] = {}
    unscored: list[dict[str, Any]] = []
    for row in _rows(path):
        name = (row.get("Design") or "").strip()
        sequence = (row.get("Sequence") or "").strip().upper()
        if not name or name in failures:
            counts["n_rejected_invalid"] += 1
            continue
        failures[name] = [
            column
            for column, value in row.items()
            if column not in REJECTED_KEYS and str(value).strip() == "1"
        ]
        native_id = f"task-{task.task_id:04d}-{name}"
        if (
            native_id not in seen
            and sequence
            and _SEQUENCE.fullmatch(sequence) is not None
            and "X" not in sequence
        ):
            unscored.append({**row, NATIVE_ID: native_id, DESIGN: name})
            seen.add(native_id)
    return failures, unscored, counts


def _read_ranks(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """`final_design_stats.csv`: the accepted designs, ranked.

    Header-only is the normal shape of a task that stopped on its trajectory
    budget, so an empty result here is a fact about the run rather than a
    parse failure.
    """
    counts = {"n_unranked": 0}
    ranks: dict[str, int] = {}
    for row in _rows(path):
        name = (row.get("Design") or "").strip()
        rank = _number(row.get("Rank"))
        if not name or rank is None or rank < 1 or rank != int(rank) or name in ranks:
            counts["n_unranked"] += 1
            continue
        ranks[name] = int(rank)
    return ranks, counts


def _read_failures(path: Path) -> dict[str, int]:
    """`failure_csv.csv`: one row of counters, one column per filter.

    `Trajectory_WrongHotspot` is always zero. The column is created but
    `update_failures` is never called with it anywhere in this fork, so it is a
    counter of nothing rather than evidence that every binder found the
    epitope.
    """
    for row in _rows(path):
        return {
            column: int(value)
            for column, value in row.items()
            if str(value).strip().lstrip("-").isdigit()
        }
    return {}


def _design_record(run_id: str, row: dict[str, Any], created_at: datetime) -> DesignRecord:
    native_id = row[NATIVE_ID]
    # The scored table carries the whole row; the rejected table carries the
    # design, its sequence and the filter flags, and nothing else.
    scored = "Protocol" in row
    match = _TRAJECTORY.fullmatch(row[DESIGN])
    return DesignRecord(
        design_id=stable_id("design", run_id, native_id),
        run_id=run_id,
        native_id=native_id,
        # A scored design has a relaxed complex in `Accepted/` or `Rejected/`.
        # One dropped by the base AF2 filters has neither: the prediction was
        # made and discarded, so what survives it is the sequence.
        candidate_type=CandidateType.COMPLEX if scored else CandidateType.SEQUENCE,
        sequence=(row.get("Sequence") or "").strip().upper(),
        seed=_integer(row.get("Seed")),
        status=DesignStatus.PRODUCED,
        metadata={
            # The backbone this sequence redesigns. Several designs share one,
            # which is the unit `max_trajectories` counts.
            "trajectory": match.group("trajectory") if match else None,
            "protocol": (row.get("Protocol") or "").strip() or None,
            "helicity": _number(row.get("Helicity")),
            "interface_residues": (row.get("InterfaceResidues") or "").strip() or None,
            # BindCraft's own sequence notes: a cysteine, or no UV-absorbing
            # residue, both of which matter for ordering rather than for
            # structure.
            "notes": (row.get("Notes") or "").strip() or None,
            "scored": scored,
        },
        created_at=created_at,
    )


def _metric_records(
    run_id: str, design: DesignRecord, row: dict[str, Any]
) -> list[MetricRecord]:
    """Every measured score on one design; the eight constants are left out."""
    records = []
    for name, direction in SINGLE_METRICS.items():
        records.extend(_metric(run_id, design, name, row.get(name), direction, 0))
    for name, direction in AVERAGED_METRICS.items():
        records.extend(
            _metric(run_id, design, name, row.get(f"Average_{name}"), direction, 0)
        )
    for name, direction in REPLICATED_METRICS.items():
        records.extend(
            _metric(run_id, design, name, row.get(f"Average_{name}"), direction, 0)
        )
        for model in MODELS:
            # A multimer design run predicts with two of the five models, so
            # three of these are empty on every row.
            records.extend(
                _metric(run_id, design, name, row.get(f"{model}_{name}"), direction, model)
            )
    return records


def _metric(
    run_id: str,
    design: DesignRecord,
    name: str,
    raw: Any,
    direction: MetricDirection,
    replicate: int,
) -> list[MetricRecord]:
    value = _number(raw)
    if value is None:
        return []
    return [
        MetricRecord(
            metric_id=stable_id(
                "metric", run_id, design.design_id, name, str(replicate)
            ),
            run_id=run_id,
            design_id=design.design_id,
            name=f"freebindcraft_{name}",
            value=value,
            direction=direction,
            # 0 is the average across the models that ran; 1..5 are the models
            # themselves, and the filters act on both.
            replicate=replicate,
            measured_at=design.created_at,
        )
    ]


def _decision_records(
    run_id: str,
    design: DesignRecord,
    row: dict[str, Any],
    task: TaskPlan,
    table: TaskTable,
) -> list[DecisionRecord]:
    """BindCraft's verdict on one design, and where it ranked among its siblings."""
    name = row[DESIGN]
    scored = name in table.scored_names
    failed = table.rejected_failures.get(name)
    records = [
        DecisionRecord(
            decision_id=stable_id("decision", run_id, design.design_id, FILTER_NAME),
            run_id=run_id,
            design_id=design.design_id,
            kind=DecisionKind.FILTER,
            name=FILTER_NAME,
            passed=failed is None,
            reason=None if failed is None else {
                # Which gate turned it down. The base filters run on the AF2
                # prediction alone and stop the design before any interface
                # metric is computed; the rest run on the full scored row.
                "stage": "interface_filters" if scored else "af2_base",
                "failed": failed,
            },
            created_at=design.created_at,
        )
    ]
    rank = table.ranks.get(name)
    if rank is not None:
        records.append(
            DecisionRecord(
                decision_id=stable_id("decision", run_id, design.design_id, RANK_NAME),
                run_id=run_id,
                design_id=design.design_id,
                kind=DecisionKind.RANK,
                name=RANK_NAME,
                rank=rank,
                # BindCraft ranks each design path's accepted pool on its own,
                # and each task has one, so a two-task run has two designs
                # ranked first.
                scope_id=rank_scope(run_id, task.task_id),
                created_at=design.created_at,
            )
        )
    return records


def rank_scope(run_id: str, task_id: int) -> str:
    """The pool a FreeBindCraft rank is meaningful within: one task's designs."""
    return stable_id("rank-scope", run_id, f"task-{task_id:04d}")


def _design_artifact(
    run_dir: Path,
    run_id: str,
    task: TaskPlan,
    design: DesignRecord,
    row: dict[str, Any],
    table: TaskTable,
) -> list[ArtifactRecord]:
    """The relaxed complex BindCraft kept for this design, if it kept one.

    One file per design: the best of the predicted models by pLDDT, copied into
    `Accepted/` or `Rejected/`. A design the base filters dropped has none.
    """
    name = row[DESIGN]
    for kind, directory in (("accepted", ACCEPTED_DIR), ("rejected", REJECTED_DIR)):
        filename = table.structures[kind].get(name)
        if filename is None:
            continue
        record = artifact(
            run_dir, run_id, f"{task.directory}/{directory}/{filename}",
            "design_complex", design_id=design.design_id,
        )
        return [record] if record is not None else []
    return []


def _task_artifacts(run_dir: Path, run_id: str, task: TaskPlan) -> list[ArtifactRecord]:
    wanted = [
        ("native_design_table", task.designs),
        ("rejected_design_table", f"{task.directory}/{REJECTED_FILE}"),
        ("ranked_design_table", f"{task.directory}/{FINAL_FILE}"),
        ("trajectory_table", f"{task.directory}/{TRAJECTORY_FILE}"),
        ("filter_failure_counts", f"{task.directory}/{FAILURE_FILE}"),
        ("task_status", task.status),
        ("log", task.log),
    ]
    return [
        record
        for kind, relative in wanted
        if (record := artifact(run_dir, run_id, relative, kind)) is not None
    ]


def _structures(directory: Path) -> dict[str, str]:
    """Design name to the structure file kept for it, from one listing.

    BindCraft names them `<design>_model<n>.pdb`, and `<n>` is whichever model
    scored highest, so the design cannot be turned into a filename without
    looking. Sorted so a directory holding two files for one design resolves
    the same way every time.
    """
    if not directory.is_dir():
        return {}
    found: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix != ".pdb" or path.name.startswith("."):
            continue
        found.setdefault(path.name.rsplit("_model", 1)[0], path.name)
    return found


def _count_pdbs(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.iterdir()
        if path.suffix == ".pdb" and not path.name.startswith(".")
    )


def _count_rows(path: Path) -> int:
    return sum(1 for _ in _rows(path))


def _rows(path: Path) -> list[dict[str, str]]:
    """One CSV's rows, or none when the tool never wrote the file."""
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number == int(number) else None
