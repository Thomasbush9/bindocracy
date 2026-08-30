"""Read a finished Genie 3 run directory into normalized records.

Written against the observed output of the 2026-08-26 benchmark run. Three
things about that output shape this parser:

* **One design is several rows.** `results/info.csv` has a row per design *per
  AF2 model*: 40 designs folded by 5 models is 200 rows. The rows of one design
  agree on the sequence and differ only in the fold, so they collapse to one
  design whose metrics carry a `replicate`. Treating a row as a design would
  have multiplied the campaign's design count by five.
* **A backbone is not a design.** Genie 3 diffuses UNK backbones, ProteinMPNN
  writes `num_seq` sequences for each, and only then is there anything with a
  sequence. `pdbs/` therefore holds fewer files than there are designs, and it
  is counted separately: backbones with no rows is exactly what a generation
  that succeeded and an evaluation that failed looks like.
* **Producing and passing are different facts**, as they are for BoltzGen. The
  v0 reducer applies its own success filters (complex scRMSD, binder pTM,
  interface PAE, hotspot coverage) and writes the winners to
  `results/v0_success/`. Every complete row is `produced`; the verdict is a
  decision. The benchmark produced 40 and passed 0.
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
    RunStatus,
    stable_id,
)

RESULTS_FILE = "results/info.csv"
SUCCESS_FILE = "results/v0_success/success_info.csv"
BACKBONE_DIR = "pdbs"
STRUCTURE_DIR = "structures"
SEQUENCE_DIR = "sequences"
# Written by the driver from the archived template: the config this task ran.
RENDERED_CONFIG = "experiment.yaml"

# The scores worth promoting out of 51 columns, with the direction that makes a
# value good. Anything absent from a row is simply not emitted.
NATIVE_METRICS: dict[str, MetricDirection] = {
    "iptm": MetricDirection.MAX,
    "ptm": MetricDirection.MAX,
    "actifptm": MetricDirection.MAX,
    "binder_ptm": MetricDirection.MAX,
    "min_interface_pae": MetricDirection.MIN,
    "avg_interface_pae": MetricDirection.MIN,
    "avg_plddt": MetricDirection.MAX,
    "avg_binder_plddt": MetricDirection.MAX,
    "complex_scrmsd": MetricDirection.MIN,
    "binder_scrmsd": MetricDirection.MIN,
    "target_hotspot_coverage": MetricDirection.MAX,
}

FILTER_NAME = "genie3_v0_success"

_SEQUENCE = re.compile(r"[A-Z]+")


class Genie3OutputAdapter(OutputAdapter):
    tool = "genie3"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = read_manifest(run_dir)
        if manifest.run_id != run.run_id:
            raise CollectionError(
                f"run {run.run_id} does not match manifest run {manifest.run_id}"
            )

        designs: list[DesignRecord] = []
        metrics: list[MetricRecord] = []
        artifacts: list[ArtifactRecord] = []
        decisions: list[DecisionRecord] = []
        seen: set[str] = set()
        per_task: dict[str, Any] = {}
        passed = 0
        reduced_all = True

        for task in manifest.tasks:
            selection = selection_of(task)
            status = read_task_status(run_dir / task.status)
            groups, rejected = _read_results(
                run_dir / task.designs, task, seen, _binder_lengths(manifest)
            )
            successful = _successful_names(run_dir / task.directory / selection / SUCCESS_FILE)
            reduced = successful is not None
            reduced_all = reduced_all and reduced
            task_passed = (
                sum(1 for group in groups if group.name in successful) if reduced else 0
            )
            passed += task_passed
            per_task[f"{task.task_id:04d}"] = {
                "status": status.status if status else "missing",
                "n_produced": len(groups),
                # None where the reducer never wrote its table: the designs
                # exist and nothing has judged them.
                "n_passed": task_passed if reduced else None,
                "reducer": "complete" if reduced else "missing",
                # Backbones with no designs is generation that worked and
                # evaluation that did not; the two counts separate them.
                "n_backbones": _count_backbones(run_dir / task.directory / selection),
                **rejected,
            }

            for group in groups:
                design = _design_record(run.run_id, group, manifest.created_at)
                designs.append(design)
                metrics.extend(_metric_records(run.run_id, design, group))
                if reduced:
                    decisions.append(
                        _filter_decision(run.run_id, design, group.name in successful)
                    )
                artifacts.extend(
                    _design_artifacts(run_dir, run.run_id, task, selection, design, group)
                )
            artifacts.extend(
                _task_artifacts(run_dir, run.run_id, task, selection, groups)
            )

        artifacts.extend(provenance_artifacts(
            run_dir, run.run_id, manifest,
            {"experiment": "experiment_config", "driver": "driver_script"},
        ))
        statuses = [read_task_status(run_dir / task.status) for task in manifest.tasks]
        started, finished = run_window(statuses)
        requested = manifest.designs_per_task * len(manifest.tasks)

        collected_run = run.model_copy(update={
            "status": _run_status(len(designs), statuses, requested, reduced_all),
            "n_requested": requested,
            "n_attempted": max(_attempted(statuses, manifest), len(designs)),
            "n_produced": len(designs),
            # None where any task's reducer did not run: nothing measured it.
            "n_passed": passed if reduced_all else None,
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
        for task in manifest.tasks:
            groups, _ = _read_results(
                run_dir / task.designs, task, set(), _binder_lengths(manifest)
            )
            if not groups:
                return False
        return True


class DesignGroup:
    """The rows of one design: same sequence, one fold each."""

    def __init__(self, name: str, native_id: str, rows: list[dict[str, str]]) -> None:
        self.name = name
        self.native_id = native_id
        self.rows = rows

    @property
    def canonical(self) -> dict[str, str]:
        """The row the design's own fields come from: its best-ranked fold."""
        ranked = [(rank, row) for row in self.rows if (rank := _rank(row)) is not None]
        return min(ranked, key=lambda pair: pair[0])[1] if ranked else self.rows[0]

    @property
    def sequence(self) -> str:
        return _sequence_of(self.canonical)

    @property
    def folds_agree(self) -> bool:
        """Every fold of one design refolded the same sequence.

        They are five predictions of one ProteinMPNN sequence, so a group whose
        rows disagree is not one design -- most likely two designs whose names
        collided, and taking the best-ranked row would silently pick one.
        """
        return len({_sequence_of(row) for row in self.rows}) == 1

    @property
    def replicates(self) -> list[int]:
        """Which fold each row is.

        `rank` is Genie 3's own ordering of a design's folds and is what makes
        replicate 0 the best one. It is only usable when every row has a
        distinct rank; otherwise position in the file is the only thing left
        that is deterministic, and a metric ID must be.
        """
        ranks = [_rank(row) for row in self.rows]
        if all(rank is not None for rank in ranks) and len(set(ranks)) == len(ranks):
            return [rank - 1 for rank in ranks]  # type: ignore[misc]
        return list(range(len(self.rows)))


def _sequence_of(row: dict[str, str]) -> str:
    return (row.get("binder_seq") or "").strip().upper()


def selection_of(task: TaskPlan) -> str:
    """The problem this task designed against, which names its output tree.

    Taken from the task's own results path rather than from the run's workflow
    metadata, so the directory the adapter reads and the file the manifest
    planned can never be two different places.
    """
    relative = Path(task.designs).relative_to(task.directory)
    return relative.parts[0]


def _read_results(
    path: Path, task: TaskPlan, seen: set[str], lengths: tuple[int, int] | None = None
) -> tuple[list[DesignGroup], dict[str, int]]:
    """Group one results table into designs, plus rejection counts.

    A design is rejected rather than fatal, so one malformed row does not cost
    the run the designs around it.
    """
    counts = {"n_invalid": 0}
    if not path.is_file():
        return [], counts

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    ordered: list[str] = []
    by_name: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            counts["n_invalid"] += 1
            continue
        if name not in by_name:
            by_name[name] = []
            ordered.append(name)
        by_name[name].append(row)

    groups = []
    for name in ordered:
        # Genie 3 numbers designs from 0 within each output root, and each task
        # has its own, so two tasks of one run produce the same names. Qualify
        # them, or the second task collects as duplicates of the first.
        native_id = f"task-{task.task_id:04d}-{name}"
        group = DesignGroup(name, native_id, by_name[name])
        sequence = group.sequence
        # X is a legal letter but an unknown residue, so a sequence carrying
        # one cannot be ordered.
        if (
            native_id in seen
            or not sequence
            or _SEQUENCE.fullmatch(sequence) is None
            or "X" in sequence
            # The problem set, not any harness field, decides how long a binder
            # may be; a sequence outside that range did not come from it.
            or (lengths is not None and not lengths[0] <= len(sequence) <= lengths[1])
            or not group.folds_agree
        ):
            counts["n_invalid"] += 1
            continue
        seen.add(native_id)
        groups.append(group)
    return groups, counts


def _successful_names(path: Path) -> set[str] | None:
    """The designs Genie 3's own v0 reducer called successes, or None.

    The reducer writes `success_info.csv` unconditionally -- the benchmark's
    zero-hit run left a header and no rows. So an **empty** table means every
    design was judged and none passed, while a **missing** one means the
    reducer never ran, and the two must not collapse into the same answer.
    None is the second case: no verdicts exist, so none are recorded.
    """
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        return {
            name
            for row in csv.DictReader(handle)
            if (name := (row.get("name") or "").strip())
        }


def _design_record(run_id: str, group: DesignGroup, fallback: datetime) -> DesignRecord:
    row = group.canonical
    return DesignRecord(
        design_id=stable_id("design", run_id, group.native_id),
        run_id=run_id,
        native_id=group.native_id,
        # A Genie 3 design is a refolded complex of binder and target, not the
        # bare sequence ProteinMPNN wrote.
        candidate_type=CandidateType.COMPLEX,
        sequence=group.sequence,
        # A complete row is a produced design. Whether it satisfied the v0
        # filters is a decision, not a lesser kind of existence.
        status=DesignStatus.PRODUCED,
        # No seed: `info.csv`'s seed column is ColabFold's, not the diffusion
        # seed that made this backbone. That one is in the task's rendered
        # experiment.yaml, which is collected as an artifact.
        metadata={
            "backbone": (row.get("domain") or "").strip() or None,
            "resample_id": (row.get("resample_id") or "").strip() or None,
            "n_folds": len(group.rows),
        },
        created_at=fallback,
    )


def _metric_records(
    run_id: str, design: DesignRecord, group: DesignGroup
) -> list[MetricRecord]:
    """One metric per score per fold. Five AF2 models is five replicates."""
    records = []
    for replicate, row in zip(group.replicates, group.rows, strict=True):
        for name, direction in NATIVE_METRICS.items():
            value = _number(row.get(name))
            if value is None:
                continue
            records.append(
                MetricRecord(
                    metric_id=stable_id(
                        "metric", run_id, design.design_id, name, str(replicate)
                    ),
                    run_id=run_id,
                    design_id=design.design_id,
                    name=f"genie3_{name}",
                    value=value,
                    direction=direction,
                    replicate=replicate,
                    measured_at=design.created_at,
                )
            )
    return records


def _filter_decision(run_id: str, design: DesignRecord, passed: bool) -> DecisionRecord:
    """Genie 3's own verdict: did this design clear the v0 success filters.

    There is no rank decision to go with it. `info.csv`'s `rank` orders a single
    design's five folds, not the designs against each other, and recording it as
    a rank would claim an ordering the tool never produced.
    """
    return DecisionRecord(
        decision_id=stable_id("decision", run_id, design.design_id, FILTER_NAME),
        run_id=run_id,
        design_id=design.design_id,
        kind=DecisionKind.FILTER,
        name=FILTER_NAME,
        passed=passed,
        created_at=design.created_at,
    )


def _design_artifacts(
    run_dir: Path,
    run_id: str,
    task: TaskPlan,
    selection: str,
    design: DesignRecord,
    group: DesignGroup,
) -> list[ArtifactRecord]:
    """The refolded complex this design *is*, and only that.

    One design owns exactly one complex, because `structures/<name>/` is keyed
    by the design. The backbone and the ProteinMPNN FASTA are keyed by the
    *domain*, which `num_seq` designs share, so they are collected once per
    task instead -- see `_backbone_artifacts`.

    The path is rebuilt from the layout rather than read out of the table's
    `design_filepath`, which holds the absolute path of the machine that wrote
    it and stops resolving the moment a run directory is moved or copied.
    """
    complex_file = Path(row_value(group.canonical, "design_filepath")).name
    if not complex_file:
        return []
    relative = f"{task.directory}/{selection}/{STRUCTURE_DIR}/{group.name}/{complex_file}"
    record = artifact(run_dir, run_id, relative, "design_complex",
                      design_id=design.design_id)
    return [record] if record is not None else []


def row_value(row: dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def _backbone_artifacts(
    run_dir: Path, run_id: str, task: TaskPlan, selection: str,
    groups: list[DesignGroup],
) -> list[ArtifactRecord]:
    """The diffused backbones and their sequences, once each.

    With `num_seq > 1` several designs come off one backbone and share its PDB
    and its FASTA. Attaching those to a design would put a `design_id` on a
    file the design does not own, and `artifacts` is unique per (run, uri), so
    which design won would come down to collection order.
    """
    base = f"{task.directory}/{selection}"
    records = []
    for backbone in dict.fromkeys(
        domain for group in groups if (domain := row_value(group.canonical, "domain"))
    ):
        for relative, kind in (
            (f"{base}/{BACKBONE_DIR}/{backbone}.pdb", "design_backbone"),
            (f"{base}/{SEQUENCE_DIR}/{backbone}.fasta", "designed_sequences"),
        ):
            if (record := artifact(run_dir, run_id, relative, kind)) is not None:
                records.append(record)
    return records


def _task_artifacts(
    run_dir: Path, run_id: str, task: TaskPlan, selection: str,
    groups: list[DesignGroup],
) -> list[ArtifactRecord]:
    wanted = [
        ("native_design_table", task.designs),
        # What this task actually ran: the template with its seed, output root
        # and sample count filled in.
        ("rendered_config", f"{task.directory}/{RENDERED_CONFIG}"),
        ("success_table", f"{task.directory}/{selection}/{SUCCESS_FILE}"),
        ("task_status", task.status),
        ("log", task.log),
    ]
    records = [
        record
        for kind, relative in wanted
        if (record := artifact(run_dir, run_id, relative, kind)) is not None
    ]
    return records + _backbone_artifacts(run_dir, run_id, task, selection, groups)


def _binder_lengths(manifest: RunManifest) -> tuple[int, int] | None:
    """The binder length range the problem set asked for, if it was recorded."""
    minimum = manifest.workflow.get("binder_min_length")
    maximum = manifest.workflow.get("binder_max_length")
    if isinstance(minimum, int) and isinstance(maximum, int):
        return (minimum, maximum)
    return None


def _count_backbones(output_root: Path) -> int:
    directory = output_root / BACKBONE_DIR
    return len(list(directory.glob("*.pdb"))) if directory.is_dir() else 0


def _attempted(statuses: list, manifest: RunManifest) -> int:
    """What the run set out to produce, as the tasks themselves reported it."""
    total = 0
    for status, task in zip(statuses, manifest.tasks, strict=True):
        if status is not None and status.n_attempted is not None:
            total += status.n_attempted
        else:
            total += task.n_generated if task.n_generated is not None else task.n_requested
    return total


def _run_status(
    produced: int, statuses: list, requested: int, reduced: bool
) -> RunStatus:
    """Genie 3 keeps everything it folds, so a short run really is partial.

    Unlike BoltzGen it has no budget that trims its own pool: every backbone
    that gets a sequence gets a row, so fewer rows than asked for means work
    that did not finish -- and so does a reducer that never wrote its verdicts,
    even when every design is present.
    """
    return run_status(produced, statuses, complete=produced >= requested and reduced)


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
