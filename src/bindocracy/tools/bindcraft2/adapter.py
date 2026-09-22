"""Read a finished BindCraft 2 run directory into normalized records.

Written against the observed output of the 2026-09-22 DIO3 run: 184
trajectories, 248 refolded candidates, 11 accepted designs on four H100s in
1h23m. Five things about that output shape this parser.

* **A design is a refolded candidate, not a trajectory and not an accepted
  design.** A trajectory is a gradient-designed backbone; only 29 of the 184 ran
  the whole way through, and the other 155 were terminated at a stage gate and
  carry no sequence at all. Each completed trajectory is redesigned into several
  sequences, and each of those is refolded and scored. Those 248 candidates are
  the designs: they have a sequence, a complex structure and a full metric row.
  Trajectories are recorded as a census in `count_details`, because it is the
  only thing that distinguishes a task that ran out of budget from one that ran
  out of luck.

* **Passing and being accepted are different facts, and the gap is large.** 28
  candidates passed every filter, but only 11 became accepted designs, because a
  trajectory contributes at most one: the 28 passing candidates span exactly 11
  trajectories. So `outcome == "passed"` is the filter verdict and the ranked
  table is a second, per-trajectory selection on top of it. Both are recorded,
  as a filter decision and a selection decision, because a query asking "how
  many designs cleared the filters" and one asking "how many designs did this
  run deliver" have different right answers here.

* **The ranked table renames its designs, so it cannot be joined by name.** A
  candidate is `<design>_candidate3`; the accepted design built from it is
  `<design>_seq0`. Nothing in either name maps to the other. They are matched on
  `(hash, Binder_Sequence)` instead -- the trajectory hash plus the exact
  sequence -- which resolved all 11 of 11 on the observed run, every one of them
  onto a candidate whose outcome was `passed`. A ranked row matching no
  candidate is counted rather than dropped, because it would mean a stale or
  truncated table marking the wrong design as delivered.

* **Which passing candidate a trajectory promotes is not derivable from these
  tables.** It is not the first and it is not the best `i_pDAE`: on the observed
  run the lowest-`i_pDAE` passing candidate was the accepted one in only 2 of 11
  trajectories. So the promotion is recorded as an observed fact via the
  sequence match, and no selection rule is asserted. Inventing one would put a
  guess in the database.

* **A task overshoots its design budget rather than hitting it.** 20 workers
  accept independently, so a budget of 10 returned 11. That is recorded as an
  expected surplus, not a shape problem; a *shortfall* is what means the
  trajectory budget ran out first.

Metric names are BindCraft 2's own column names, unchanged. Where a name
coincides with FreeBindCraft's -- `pLDDT`, `pTM`, `i_pTM`, `i_pAE`,
`Surface_Hydrophobicity`, `Target_RMSD` -- it is the same quantity and the two
tools compare directly. Where BC2 measures something its predecessor did not,
the new name is the honest one: `Unbound_Binder_pLDDT` is the binder folded
alone, which is not FreeBindCraft's `Binder_pLDDT`, and renaming either to match
the other would manufacture a comparison that does not exist.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
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
from bindocracy.tools.bindcraft2.launch import (
    CAMPAIGN_DIR,
    METADATA_FILE,
    RANKED_FILE,
    REFOLDED_FILE,
    STATE_FILE,
    SUMMARY_FILE,
    TRAJECTORIES_FILE,
    WORKERS_DIR,
)

# BindCraft 2's own column names, with the direction declared once. A metric
# stored with the wrong direction sorts backwards and nothing about the row says
# so. Counts and compositions that are neither good nor bad are NONE rather than
# omitted: they are what a later filter reads.
METRICS: dict[str, MetricDirection] = {
    # Confidence. i_pDAE is the campaign's own ranking metric and is normalized
    # predicted alignment error, so lower is better.
    "i_pDAE": MetricDirection.MIN,
    "i_pTM": MetricDirection.MAX,
    "i_pAE": MetricDirection.MIN,
    "pLDDT": MetricDirection.MAX,
    "pTM": MetricDirection.MAX,
    "SS_pLDDT": MetricDirection.MAX,
    "Target_pLDDT": MetricDirection.MAX,
    # The binder predicted on its own, which is not the binder chain of the
    # complex. A design confident only in complex is a design that does not fold.
    "Unbound_Binder_pLDDT": MetricDirection.MAX,
    # Epitope. These are what an epitope-targeted campaign is judged on, and
    # they are the reason BC2 can verify an epitope its predecessor could not.
    "Hotspot_Contact_Fraction": MetricDirection.MAX,
    "Off_Epitope_Contact_Fraction": MetricDirection.MIN,
    "Epitope_Residues_Contacted": MetricDirection.MAX,
    # Interface geometry.
    "Interface_Residues": MetricDirection.NONE,
    "Interface_BuriedArea": MetricDirection.MAX,
    "Interface_BuriedArea_Fraction": MetricDirection.MAX,
    "Interface_Hydrophobicity": MetricDirection.NONE,
    "Backbone_Clashes": MetricDirection.MIN,
    "All_Atom_Clashes": MetricDirection.MIN,
    "Target_RMSD": MetricDirection.MIN,
    # Composition. None of these is better high or low on its own; they are read
    # by the filters and by whoever decides what can be ordered.
    "Surface_Hydrophobicity": MetricDirection.NONE,
    "Binder_Helix_Fraction": MetricDirection.NONE,
    "Binder_BetaSheet_Fraction": MetricDirection.NONE,
    "Binder_Loop_Fraction": MetricDirection.NONE,
    "Interface_Helix_Fraction": MetricDirection.NONE,
    "Interface_BetaSheet_Fraction": MetricDirection.NONE,
    "Interface_Loop_Fraction": MetricDirection.NONE,
    "Binder_Disulfides": MetricDirection.NONE,
    "Binder_Cysteines": MetricDirection.NONE,
    # A free cysteine is a liability for anything that has to be made.
    "Binder_Free_Cysteines": MetricDirection.MIN,
    "Binder_Mass_kDa": MetricDirection.NONE,
    "Binder_pI": MetricDirection.NONE,
    "Binder_Net_Charge": MetricDirection.NONE,
    "Binder_Extinction": MetricDirection.NONE,
}

FILTER_NAME = "bindcraft2_filters"
SELECTION_NAME = "bindcraft2_accepted"
RANK_NAME = "bindcraft2_rank"

# The flat layout BC2 keeps for folders written before it had stages. A task
# directory is always fresh, so the staged names are what appear; both are
# resolved anyway, because a table read from the wrong place reads as absent and
# an absent table reads as a task that produced nothing.
LEGACY_TABLES = {
    TRAJECTORIES_FILE: f"{CAMPAIGN_DIR}/trajectories.csv",
    REFOLDED_FILE: f"{CAMPAIGN_DIR}/candidates.csv",
    RANKED_FILE: f"{CAMPAIGN_DIR}/accepted.csv",
}


@dataclass(frozen=True)
class Candidate:
    """One refolded, scored candidate: a design with a sequence and a structure."""

    native_id: str
    design: str
    trajectory_hash: str
    sequence: str
    length: int | None
    outcome: str
    failed_filters: tuple[str, ...]
    row: dict[str, str]

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"


@dataclass
class TaskTables:
    """The three stage tables of one task, cross-checked against each other."""

    task_id: int
    trajectories: list[dict[str, str]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    ranked: list[dict[str, str]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    # Ranked design name -> the candidate it was built from, matched on the
    # trajectory hash and the exact sequence.
    promoted: dict[str, Candidate] = field(default_factory=dict)
    unmatched_ranked: list[str] = field(default_factory=list)

    @property
    def terminated(self) -> dict[str, int]:
        """Why each trajectory stopped; an empty value means it completed."""
        census: dict[str, int] = {}
        for row in self.trajectories:
            stage = (row.get("terminated") or "").strip() or "completed"
            census[stage] = census.get(stage, 0) + 1
        return census

    @property
    def passing(self) -> list[Candidate]:
        return [candidate for candidate in self.candidates if candidate.passed]

    def shape_problems(self, task: TaskPlan) -> list[str]:
        """What the tables say that they should not, per task rather than in total.

        A surplus of accepted designs is not a problem: workers accept
        independently and a task overshoots. A shortfall is not a problem
        either -- it means the trajectory budget ended the task -- but it is
        recorded as such rather than silently.
        """
        problems: list[str] = []
        if self.unmatched_ranked:
            problems.append(
                f"{len(self.unmatched_ranked)} accepted design(s) match no scored "
                "candidate on (trajectory hash, sequence): "
                + ", ".join(sorted(self.unmatched_ranked)[:5])
            )
        ranks = [_integer(row.get("rank")) for row in self.ranked]
        present = sorted(rank for rank in ranks if rank is not None)
        if len(present) != len(self.ranked):
            problems.append(f"{len(self.ranked) - len(present)} accepted design(s) carry no rank")
        elif present and present != list(range(1, len(present) + 1)):
            problems.append(f"ranks are not 1..{len(present)}: {present}")
        if len({candidate.native_id for candidate in self.candidates}) != len(self.candidates):
            problems.append("the candidate table repeats a design name")
        promoted = {candidate.native_id for candidate in self.promoted.values()}
        not_passing = sorted(
            candidate.native_id
            for candidate in self.promoted.values()
            if not candidate.passed
        )
        if not_passing:
            problems.append(
                "accepted design(s) built from a candidate the filters rejected: "
                + ", ".join(not_passing[:5])
            )
        if len(promoted) != len(self.promoted):
            problems.append("two accepted designs were built from one candidate")
        # BC2's own tally, which is written independently of the tables.
        scored = _integer((self.state.get("rejections") or {}).get("candidates_scored"))
        if scored is not None and scored != len(self.candidates):
            problems.append(
                f"the campaign state counts {scored} scored candidates and the "
                f"table holds {len(self.candidates)}"
            )
        accepted = _integer(self.state.get("accepted"))
        if accepted is not None and accepted != len(self.ranked):
            problems.append(
                f"the campaign state counts {accepted} accepted designs and the "
                f"ranked table holds {len(self.ranked)}"
            )
        for label, count in self.skipped.items():
            problems.append(f"{count} unreadable row(s) in the {label} table")
        return problems


class BindCraft2OutputAdapter(OutputAdapter):
    tool = "bindcraft2"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = read_manifest(run_dir)
        if manifest.run_id != run.run_id:
            raise CollectionError(
                f"run {run.run_id} does not match manifest run {manifest.run_id}"
            )

        designs: list[DesignRecord] = []
        metrics: list[MetricRecord] = []
        decisions: list[DecisionRecord] = []
        artifacts: list[ArtifactRecord] = []
        per_task: dict[str, Any] = {}
        passed = 0
        accepted_total = 0
        evaluated = True

        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            tables = _read_task(run_dir, task)
            problems = tables.shape_problems(task)
            # A task that produced no candidate at all is not incomplete if its
            # every trajectory was terminated by a stage gate: that is a real
            # zero. It is incomplete when the tables disagree with each other.
            complete = not problems
            evaluated = evaluated and complete
            passed += len(tables.passing)
            accepted_total += len(tables.ranked)

            scope = stable_id("scope", run.run_id, f"task-{task.task_id:04d}")
            by_native: dict[str, DesignRecord] = {}
            for candidate in tables.candidates:
                design = _design_record(run.run_id, candidate, manifest)
                designs.append(design)
                by_native[candidate.native_id] = design
                metrics.extend(_metric_records(run.run_id, design, candidate))
                decisions.extend(_filter_decisions(run.run_id, design, candidate))
                artifacts.extend(
                    _candidate_artifacts(run_dir, run.run_id, task, design, candidate)
                )

            for row in tables.ranked:
                candidate = tables.promoted.get(row["design"])
                design = by_native.get(candidate.native_id) if candidate else None
                if design is None:
                    continue
                decisions.extend(
                    _acceptance_decisions(run.run_id, design, row, scope)
                )
                artifacts.extend(
                    _accepted_artifacts(run_dir, run.run_id, task, design, row)
                )

            artifacts.extend(_task_artifacts(run_dir, run.run_id, task))
            per_task[f"{task.task_id:04d}"] = _task_details(task, tables, status, problems)

        artifacts.extend(
            provenance_artifacts(
                run_dir, run.run_id, manifest, {"settings": "campaign_settings"}
            )
        )
        statuses = [read_task_status(run_dir / task.status) for task in manifest.tasks]
        started, finished = run_window(statuses)

        collected = run.model_copy(
            update={
                "status": run_status(len(designs), statuses, complete=evaluated),
                "n_requested": manifest.designs_per_task * len(manifest.tasks),
                # Every candidate that was refolded and scored. The trajectories
                # behind them are in count_details: they are candidate backbones,
                # not candidate designs, and one number mixing the two would
                # answer neither question.
                "n_attempted": len(designs),
                "n_produced": len(designs),
                # The filter verdict, not the delivered count. `accepted` in
                # count_details is what the run handed over, and it is smaller
                # because a trajectory promotes at most one of its passing
                # candidates.
                "n_passed": passed if evaluated else None,
                "count_details": {
                    "accepted": accepted_total,
                    "reproducible": manifest.workflow.get("reproducible"),
                    "tasks": per_task,
                },
                "started_at": started,
                "finished_at": finished,
            }
        )
        return CollectedRun(
            run=collected,
            designs=tuple(designs),
            metrics=tuple(metrics),
            decisions=tuple(decisions),
            artifacts=tuple(unique_by_uri(artifacts)),
        )

    def succeeded(self, run_dir: Path) -> bool:
        manifest = read_manifest(run_dir)
        return all(
            _resolve(run_dir / task.directory, TRAJECTORIES_FILE) is not None
            for task in manifest.tasks
        )


def _task_details(
    task: TaskPlan, tables: TaskTables, status: Any, problems: list[str]
) -> dict[str, Any]:
    rejections = tables.state.get("rejections") or {}
    accepted = len(tables.ranked)
    return {
        "status": status.status if status else "missing",
        # The census that says why a task stopped. Every attempt counts against
        # the budget, including one terminated at the first gate.
        "trajectories": {
            "attempted": len(tables.trajectories),
            "budget": task.n_generated,
            "terminated": tables.terminated,
        },
        "n_attempted": len(tables.candidates),
        "n_scored": len(tables.candidates),
        "n_passed": len(tables.passing),
        # What the run delivered, which is at most one per trajectory and so is
        # smaller than the number that passed.
        "accepted": accepted,
        "requested": task.n_requested,
        # Workers accept independently, so a surplus is expected and a shortfall
        # means the trajectory budget ended the task first.
        "surplus": accepted - task.n_requested,
        "budget_exhausted": accepted < task.n_requested,
        # BC2's own tally of which filter rejected what, written independently
        # of the tables and therefore worth keeping beside them.
        "failed_filters": rejections.get("failed_filters") or {},
        "candidates_scored": rejections.get("candidates_scored"),
        "candidates_rejected": rejections.get("candidates_rejected"),
        "shape_problems": problems,
    }


def _read_task(run_dir: Path, task: TaskPlan) -> TaskTables:
    task_dir = run_dir / task.directory
    tables = TaskTables(task_id=task.task_id)
    tables.state = _read_json(task_dir / STATE_FILE)

    trajectories, skipped = _read_rows(_resolve(task_dir, TRAJECTORIES_FILE), ("design",))
    tables.trajectories = trajectories
    if skipped:
        tables.skipped["trajectory"] = skipped

    candidate_rows, skipped = _read_rows(
        _resolve(task_dir, REFOLDED_FILE), ("design", "Binder_Sequence")
    )
    if skipped:
        tables.skipped["candidate"] = skipped
    tables.candidates = [
        _candidate(row, task.task_id)
        for row in candidate_rows
        if (row.get("Binder_Sequence") or "").strip()
    ]
    tables.skipped["candidate"] = tables.skipped.get("candidate", 0) + (
        len(candidate_rows) - len(tables.candidates)
    )
    if not tables.skipped.get("candidate"):
        tables.skipped.pop("candidate", None)

    ranked, skipped = _read_rows(_resolve(task_dir, RANKED_FILE), ("design",))
    tables.ranked = ranked
    if skipped:
        tables.skipped["ranked"] = skipped

    # The ranked table renames its designs, so the join is on the trajectory
    # hash plus the exact sequence. Anything unmatched is counted, never
    # silently dropped: it would mean a stale table naming a design the scored
    # table does not hold.
    by_key = {
        (candidate.trajectory_hash, candidate.sequence): candidate
        for candidate in tables.candidates
    }
    for row in tables.ranked:
        key = ((row.get("hash") or "").strip(), (row.get("Binder_Sequence") or "").strip().upper())
        match = by_key.get(key)
        if match is None:
            tables.unmatched_ranked.append(row["design"])
        else:
            tables.promoted[row["design"]] = match
    return tables


def _candidate(row: dict[str, str], task_id: int) -> Candidate:
    design = (row.get("design") or "").strip()
    failed = tuple(
        name.strip()
        for name in (row.get("failed_filters") or "").split(",")
        if name.strip()
    )
    return Candidate(
        # Qualified by task. A design name carries the campaign name, the binder
        # length and the trajectory hash, and two tasks of one run share the
        # first two, so an unqualified name could collide and a collision would
        # silently halve a two-task run's designs.
        native_id=f"task-{task_id:04d}-{design}",
        design=design,
        trajectory_hash=(row.get("hash") or "").strip(),
        sequence=(row.get("Binder_Sequence") or "").strip().upper(),
        length=_integer(row.get("length")),
        outcome=(row.get("outcome") or "").strip(),
        failed_filters=failed,
        row=row,
    )


def _resolve(task_dir: Path, staged: str) -> Path | None:
    """The staged table, or the flat one an older campaign folder would hold."""
    for relative in (staged, LEGACY_TABLES.get(staged, staged)):
        path = task_dir / relative
        if path.is_file():
            return path
    return None


def _read_rows(
    path: Path | None, required: tuple[str, ...]
) -> tuple[list[dict[str, str]], int]:
    """Every readable row, and a count of the ones that were not.

    One malformed line must not cost the designs written before and after it.
    """
    if path is None:
        return [], 0
    rows: list[dict[str, str]] = []
    skipped = 0
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if all((row.get(key) or "").strip() for key in required):
                    rows.append(row)
                else:
                    skipped += 1
    except OSError as error:
        raise CollectionError(f"cannot read BindCraft 2 table {path}: {error}") from error
    return rows, skipped


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def _design_record(
    run_id: str, candidate: Candidate, manifest: RunManifest
) -> DesignRecord:
    return DesignRecord(
        design_id=stable_id("design", run_id, candidate.native_id),
        run_id=run_id,
        native_id=candidate.native_id,
        # Every candidate here was refolded as a complex and its structure kept.
        candidate_type=CandidateType.COMPLEX,
        sequence=candidate.sequence,
        status=DesignStatus.PRODUCED,
        metadata={
            # The backbone this sequence redesigns. Several candidates share
            # one, and it is the unit `max_trajectories` counts and the unit a
            # promotion to an accepted design happens within.
            "trajectory_hash": candidate.trajectory_hash,
            "trajectory_design": candidate.design.rsplit("_candidate", 1)[0] or None,
            "candidate": _integer(candidate.design.rsplit("_candidate", 1)[-1]),
            "outcome": candidate.outcome or None,
            "core_profile": (candidate.row.get("settings_core") or "").strip() or None,
            "bindcraft_version": (candidate.row.get("bindcraft_version") or "").strip() or None,
            "interface_binder_residues": (
                candidate.row.get("Interface_Binder_Residues") or ""
            ).strip() or None,
            "interface_target_residues": (
                candidate.row.get("Interface_Target_Residues") or ""
            ).strip() or None,
        },
        # From the manifest, never the clock: re-collecting an unchanged
        # directory must produce an identical bundle.
        created_at=manifest.created_at,
    )


def _metric_records(
    run_id: str, design: DesignRecord, candidate: Candidate
) -> list[MetricRecord]:
    records = []
    for name, direction in METRICS.items():
        value = _number(candidate.row.get(name))
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


def _filter_decisions(
    run_id: str, design: DesignRecord, candidate: Candidate
) -> list[DecisionRecord]:
    """The campaign's verdict on one candidate, and each threshold that rejected it.

    The aggregate is what `n_passed` counts. The per-filter rows are what makes
    "which threshold is costing this campaign its designs" a query rather than a
    grep, and on the observed run the answer was `i_pTM` by a wide margin.
    """
    decisions = [
        DecisionRecord(
            decision_id=stable_id("decision", run_id, design.design_id, "filter", FILTER_NAME),
            run_id=run_id,
            design_id=design.design_id,
            kind=DecisionKind.FILTER,
            name=FILTER_NAME,
            value=candidate.outcome or None,
            passed=candidate.passed,
            reason={"failed_filters": list(candidate.failed_filters)}
            if candidate.failed_filters
            else None,
            created_at=design.created_at,
        )
    ]
    for name in candidate.failed_filters:
        decisions.append(
            DecisionRecord(
                decision_id=stable_id(
                    "decision", run_id, design.design_id, "filter", name
                ),
                run_id=run_id,
                design_id=design.design_id,
                kind=DecisionKind.FILTER,
                name=name,
                passed=False,
                created_at=design.created_at,
            )
        )
    return decisions


def _acceptance_decisions(
    run_id: str, design: DesignRecord, row: dict[str, str], scope: str
) -> list[DecisionRecord]:
    """Being promoted to an accepted design, and the rank it was given.

    Two decisions rather than one. Passing the filters is already recorded; this
    is the separate fact that this candidate is the one its trajectory delivered,
    which 17 of the 28 passing candidates on the observed run were not.
    """
    rank = _integer(row.get("rank"))
    decisions = [
        DecisionRecord(
            decision_id=stable_id(
                "decision", run_id, design.design_id, "selection", SELECTION_NAME
            ),
            run_id=run_id,
            design_id=design.design_id,
            kind=DecisionKind.SELECTION,
            name=SELECTION_NAME,
            value=row["design"],
            passed=True,
            reason={
                # The name BC2 gave the delivered design, which is not the
                # candidate's name and is what the ranked structure is called.
                "accepted_as": row["design"],
                "promoted_within": "trajectory",
            },
            created_at=design.created_at,
        )
    ]
    if rank is not None:
        decisions.append(
            DecisionRecord(
                decision_id=stable_id("decision", run_id, design.design_id, "rank", RANK_NAME),
                run_id=run_id,
                design_id=design.design_id,
                kind=DecisionKind.RANK,
                name=RANK_NAME,
                rank=rank,
                # Ranked within the task: each task is its own campaign with its
                # own accepted pool, so a rank means nothing across tasks.
                scope_id=scope,
                created_at=design.created_at,
            )
        )
    return decisions


def _candidate_artifacts(
    run_dir: Path, run_id: str, task: TaskPlan, design: DesignRecord, candidate: Candidate
) -> list[ArtifactRecord]:
    records = []
    for relative, kind in (
        (f"{CAMPAIGN_DIR}/2_Refolded/Complexes/{candidate.design}.cif", "complex_structure"),
        (
            f"{CAMPAIGN_DIR}/2_Refolded/BinderMonomer/{candidate.design}_monomer.cif",
            "binder_structure",
        ),
    ):
        record = artifact(
            run_dir, run_id, f"{task.directory}/{relative}", kind, design_id=design.design_id
        )
        if record is not None:
            records.append(record)
    return records


def _accepted_artifacts(
    run_dir: Path, run_id: str, task: TaskPlan, design: DesignRecord, row: dict[str, str]
) -> list[ArtifactRecord]:
    record = artifact(
        run_dir,
        run_id,
        f"{task.directory}/{CAMPAIGN_DIR}/3_Ranked/{row['design']}.cif",
        "accepted_structure",
        design_id=design.design_id,
    )
    return [record] if record is not None else []


def _task_artifacts(run_dir: Path, run_id: str, task: TaskPlan) -> list[ArtifactRecord]:
    records = []
    for relative, kind in (
        (TRAJECTORIES_FILE, "trajectory_table"),
        (REFOLDED_FILE, "native_design_table"),
        (RANKED_FILE, "accepted_table"),
        (STATE_FILE, "campaign_state"),
        (METADATA_FILE, "campaign_metadata"),
        (SUMMARY_FILE, "campaign_summary"),
        (f"{WORKERS_DIR}/campaign_settings.json", "resolved_settings"),
    ):
        record = artifact(run_dir, run_id, f"{task.directory}/{relative}", kind)
        if record is not None:
            records.append(record)
    workers = run_dir / task.directory / WORKERS_DIR
    if workers.is_dir():
        for log in sorted(workers.glob("worker_*.log")):
            record = artifact(
                run_dir, run_id, f"{task.directory}/{WORKERS_DIR}/{log.name}", "worker_log"
            )
            if record is not None:
                records.append(record)
    record = artifact(run_dir, run_id, task.log, "task_log")
    if record is not None:
        records.append(record)
    return records


def _number(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and float(number).is_integer() else None
