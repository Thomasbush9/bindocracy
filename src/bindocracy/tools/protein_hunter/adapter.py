"""Read a finished Protein-Hunter run directory into normalized records.

Written against the observed output of the 2026-08-26 benchmark run. Three
things about that output shape this parser:

* **The table is wide, not long.** `summary_all_runs.csv` has one row per
  *trajectory*, with a block of `cycle_N_*` columns across it. A design is a
  cell block, not a row: 40 trajectories at 5 cycles is 200 designs, each its
  own sequence, because MPNN emits exactly one sequence per cycle.
* **Cycle 0 is not a design.** It is the fold of the starting mostly-X binder,
  before any redesign, and its sequence column is empty in all forty rows.
* **`best_*` can be blank while the trajectory worked.** A 20% alanine cap
  excludes a cycle from the best-of selection, and a trajectory whose every
  cycle was excluded gets no `best_iptm` at all -- six of the benchmark's
  forty. Those trajectories still produced five sequences each, and reading a
  blank `best_iptm` as failure would throw away thirty designs.

The tool's own verdict lives in a second file, `summary_high_iptm.csv`, which
lists the (trajectory, cycle) pairs that cleared both thresholds -- and which
the pipeline does not write at all when nothing clears. Neither are the
`high_iptm_*` directories. So absence has two meanings and the file cannot
separate them; what separates them is whether the task produced every design it
was asked for, because the summary table is written last.
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

SUMMARY_FILE = "summary_all_runs.csv"
HIGH_IPTM_FILE = "summary_high_iptm.csv"
STRUCTURE_DIR = "high_iptm_pdb"
SPEC_DIR = "high_iptm_yaml"

# The per-cycle columns, by the suffix each carries. `alanine` is a count, and
# a high one is what the pipeline's own cap excludes a cycle for.
NATIVE_METRICS: dict[str, MetricDirection] = {
    "iptm": MetricDirection.MAX,
    "plddt": MetricDirection.MAX,
    "iplddt": MetricDirection.MAX,
    "alanine": MetricDirection.MIN,
}
SEQUENCE_SUFFIX = "seq"

FILTER_NAME = "protein_hunter_high_iptm"
BEST_NAME = "protein_hunter_best_cycle"

_CYCLE_COLUMN = re.compile(r"^cycle_(?P<cycle>\d+)_(?P<field>[a-z]+)$")
_SEQUENCE = re.compile(r"[A-Z]+")


class Design:
    """One (trajectory, cycle) pair: one sequence, one co-fold, one verdict."""

    def __init__(self, run_id: int, cycle: int, native_id: str,
                 sequence: str, values: dict[str, float]) -> None:
        self.run_id = run_id
        self.cycle = cycle
        self.native_id = native_id
        self.sequence = sequence
        self.values = values

    @property
    def key(self) -> tuple[int, int]:
        return (self.run_id, self.cycle)


class ProteinHunterOutputAdapter(OutputAdapter):
    tool = "protein_hunter"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = read_manifest(run_dir)
        if manifest.run_id != run.run_id:
            raise CollectionError(
                f"run {run.run_id} does not match manifest run {manifest.run_id}"
            )
        lengths = _binder_lengths(manifest)
        cycles = _cycles_of(manifest)
        protocol = hit_protocol_summary(manifest)

        designs: list[DesignRecord] = []
        metrics: list[MetricRecord] = []
        artifacts: list[ArtifactRecord] = []
        decisions: list[DecisionRecord] = []
        seen: set[str] = set()
        per_task: dict[str, Any] = {}
        passed = 0
        judged_all = True

        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            parsed, rejected, trajectories, without_best = _read_summary(
                run_dir / task.designs, task, seen, lengths
            )
            malformed = _shape_problems(parsed, task, trajectories, cycles)
            verdicts, verdict_counts = _high_iptm(
                run_dir / task.directory / HIGH_IPTM_FILE, parsed
            )
            if verdicts is None and not malformed:
                # The pipeline writes neither the threshold table nor the
                # high_iptm_* directories when nothing clears them. On a task
                # whose table is exactly the shape it was planned to be, the
                # pipeline reached its end, so that is a real zero rather than
                # an evaluation that did not happen.
                verdicts = {}
            judged = verdicts is not None
            judged_all = judged_all and judged and not malformed
            task_passed = (
                sum(1 for design in parsed if design.key in verdicts) if judged else 0
            )
            passed += task_passed
            per_task[f"{task.task_id:04d}"] = {
                "status": status.status if status else "missing",
                "n_produced": len(parsed),
                # None where the threshold table is absent: the designs exist
                # and nothing has judged them.
                "n_passed": task_passed if judged else None,
                "n_trajectories": trajectories,
                # Trajectories whose every cycle tripped the alanine cap, so
                # the tool recorded no best. Their designs are still here; this
                # is the count that says the run was not as clean as it looks.
                "n_trajectories_without_best": without_best,
                # Empty on a healthy run. Non-empty means the table is not the
                # shape this task was planned to produce.
                "shape_problems": malformed,
                **rejected,
                **verdict_counts,
            }

            for design in parsed:
                record = _design_record(run.run_id, design, manifest.created_at)
                designs.append(record)
                metrics.extend(_metric_records(run.run_id, record, design))
                if judged:
                    decisions.append(_filter_decision(
                        run.run_id, record, design.key in verdicts, protocol
                    ))
                artifacts.extend(_design_artifacts(
                    run_dir, run.run_id, task, record,
                    verdicts.get(design.key) if verdicts else None,
                ))
            decisions.extend(_best_decisions(
                run_dir / task.designs, run.run_id, parsed, manifest.created_at
            ))
            artifacts.extend(_task_artifacts(run_dir, run.run_id, task))

        artifacts.extend(provenance_artifacts(
            run_dir, run.run_id, manifest, {"driver": "driver_script"},
        ))
        statuses = [read_task_status(run_dir / task.status) for task in manifest.tasks]
        started, finished = run_window(statuses)
        requested = manifest.designs_per_task * len(manifest.tasks)

        collected_run = run.model_copy(update={
            # Every cycle emits a sequence, so fewer than asked for is work
            # that did not finish -- and so is a missing threshold table.
            "status": run_status(
                len(designs), statuses,
                complete=len(designs) >= requested and judged_all,
            ),
            "n_requested": requested,
            "n_attempted": max(_attempted(statuses, manifest.tasks), len(designs)),
            "n_produced": len(designs),
            "n_passed": passed if judged_all else None,
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
        cycles = _cycles_of(manifest)
        for task in manifest.tasks:
            parsed, _, trajectories, _ = _read_summary(
                run_dir / task.designs, task, set(), _binder_lengths(manifest)
            )
            if not parsed or _shape_problems(parsed, task, trajectories, cycles):
                return False
        return True


def _cycles_of(manifest: RunManifest) -> int | None:
    """How many cycles each trajectory was configured to run."""
    cycles = manifest.workflow.get("cycles")
    return cycles if isinstance(cycles, int) and cycles > 0 else None


def _shape_problems(
    designs: list[Design], task: TaskPlan, trajectories: int, cycles: int | None
) -> list[str]:
    """How this task's table differs from the one the plan asked for.

    Checked per task and per trajectory, because totals hide the interesting
    cases: one trajectory short of its cycles while another carries an extra
    still adds up to the requested number of designs.

    This is also what decides whether an absent threshold table is a real zero,
    so it is the one place that answers "did this task finish".
    """
    problems = []
    if len(designs) != task.n_requested:
        problems.append(f"{len(designs)} designs, expected {task.n_requested}")
    if cycles is None:
        return problems

    expected_trajectories = task.n_requested // cycles
    if trajectories != expected_trajectories:
        problems.append(f"{trajectories} trajectories, expected {expected_trajectories}")

    expected = set(range(1, cycles + 1))
    by_trajectory: dict[int, set[int]] = {}
    for design in designs:
        by_trajectory.setdefault(design.run_id, set()).add(design.cycle)
    for run_id, found in sorted(by_trajectory.items()):
        if found != expected:
            problems.append(
                f"trajectory {run_id} has cycles {sorted(found)}, expected "
                f"1..{cycles}"
            )
    return problems


def _read_summary(
    path: Path, task: TaskPlan, seen: set[str], lengths: tuple[int, int] | None
) -> tuple[list[Design], dict[str, int], int, int]:
    """Unpack the wide table into designs, plus the counts that explain it.

    Returns the designs, the rejection counts, how many trajectories the table
    held, and how many of those recorded no best cycle.
    """
    counts = {"n_invalid": 0}
    if not path.is_file():
        return [], counts, 0, 0

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or ())
        rows = list(reader)

    cycles = _cycle_numbers(columns)
    designs: list[Design] = []
    trajectories = 0
    without_best = 0

    for row in rows:
        run_id = _integer(row.get("run_id"))
        if run_id is None:
            counts["n_invalid"] += 1
            continue
        trajectories += 1
        if not (row.get("best_iptm") or "").strip():
            without_best += 1

        for cycle in cycles:
            sequence = (row.get(f"cycle_{cycle}_{SEQUENCE_SUFFIX}") or "").strip().upper()
            # Cycle 0 is the fold of the starting mostly-X binder, before any
            # redesign; it has no sequence and is not a design.
            if not sequence:
                continue
            native_id = f"task-{task.task_id:04d}-run-{run_id:04d}-cycle-{cycle}"
            if (
                native_id in seen
                or _SEQUENCE.fullmatch(sequence) is None
                or "X" in sequence
                # The pipeline was told a length range; a sequence outside it
                # did not come from this configuration.
                or (lengths is not None and not lengths[0] <= len(sequence) <= lengths[1])
            ):
                counts["n_invalid"] += 1
                continue
            seen.add(native_id)
            designs.append(Design(
                run_id=run_id,
                cycle=cycle,
                native_id=native_id,
                sequence=sequence,
                values=_cycle_values(row, cycle),
            ))
    return designs, counts, trajectories, without_best


def _cycle_numbers(columns: list[str]) -> list[int]:
    """Which cycles this table actually carries, in order."""
    found = set()
    for column in columns:
        match = _CYCLE_COLUMN.match(column)
        if match:
            found.add(int(match.group("cycle")))
    return sorted(found)


def _cycle_values(row: dict[str, str], cycle: int) -> dict[str, float]:
    values = {}
    for field in NATIVE_METRICS:
        number = _number(row.get(f"cycle_{cycle}_{field}"))
        if number is not None:
            values[field] = number
    return values


def _high_iptm(
    path: Path, designs: list[Design]
) -> tuple[dict[tuple[int, int], dict[str, str]] | None, dict[str, int]]:
    """The (trajectory, cycle) pairs that cleared the thresholds, cross-checked.

    None means the file is absent, which is either a run in which nothing
    passed or an evaluation that did not finish; the caller decides which from
    the shape of the primary table.

    This is a second file naming designs that the first file also names, so it
    is checked against it rather than believed: a verdict keyed to a design
    that is not there, or carrying a different sequence or ipTM from the one
    the primary table recorded, is a stale or corrupted file marking the wrong
    design as passed.
    """
    counts = {"n_verdicts_unmatched": 0, "n_verdicts_duplicated": 0,
              "n_verdicts_inconsistent": 0}
    if not path.is_file():
        return None, counts

    by_key = {design.key: design for design in designs}
    verdicts: dict[tuple[int, int], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = _integer(row.get("run_id"))
            cycle = _integer(row.get("cycle"))
            if run_id is None or cycle is None:
                counts["n_verdicts_unmatched"] += 1
                continue
            key = (run_id, cycle)
            if key in verdicts:
                counts["n_verdicts_duplicated"] += 1
                continue
            design = by_key.get(key)
            if design is None:
                counts["n_verdicts_unmatched"] += 1
                continue
            if not _agrees(row, design):
                counts["n_verdicts_inconsistent"] += 1
                continue
            verdicts[key] = row
    return verdicts, counts


def _agrees(row: dict[str, str], design: Design) -> bool:
    """Whether a verdict row describes the design the primary table recorded."""
    sequence = (row.get("sequence") or "").strip().upper()
    if sequence and sequence != design.sequence:
        return False
    for column, field in (("iptm", "iptm"), ("plddt", "plddt")):
        value = _number(row.get(column))
        recorded = design.values.get(field)
        if value is not None and recorded is not None and not math.isclose(
            value, recorded, rel_tol=1e-6, abs_tol=1e-9
        ):
            return False
    return True


def _design_record(run_id: str, design: Design, fallback: datetime) -> DesignRecord:
    return DesignRecord(
        design_id=stable_id("design", run_id, design.native_id),
        run_id=run_id,
        native_id=design.native_id,
        # Every cycle is scored by co-folding the binder with the target, so
        # what was measured is a complex even where no structure was kept.
        candidate_type=CandidateType.COMPLEX,
        sequence=design.sequence,
        status=DesignStatus.PRODUCED,
        # No seed: the pipeline has no seed flag at all, so nothing here can
        # say which draw produced this sequence.
        metadata={"trajectory": design.run_id, "cycle": design.cycle},
        created_at=fallback,
    )


def _metric_records(
    run_id: str, record: DesignRecord, design: Design
) -> list[MetricRecord]:
    return [
        MetricRecord(
            metric_id=stable_id("metric", run_id, record.design_id, name, "0"),
            run_id=run_id,
            design_id=record.design_id,
            name=f"protein_hunter_{name}",
            value=value,
            direction=NATIVE_METRICS[name],
            measured_at=record.created_at,
        )
        for name, value in design.values.items()
    ]


def _filter_decision(
    run_id: str, record: DesignRecord, passed: bool, protocol: str
) -> DecisionRecord:
    """Did this cycle survive everything that gates `summary_high_iptm.csv`.

    The name follows the tool's own file, so a row can be traced back to it.
    The name is also narrower than the test: membership needs the ipTM and
    pLDDT thresholds *and* an alanine cap *and*, when the campaign names an
    epitope, a contact check -- two of which are hard-coded upstream. The
    protocol is carried on every row so a query does not have to go and find
    it, and in full on the run's `hit_protocol`.
    """
    return DecisionRecord(
        decision_id=stable_id("decision", run_id, record.design_id, FILTER_NAME),
        run_id=run_id,
        design_id=record.design_id,
        kind=DecisionKind.FILTER,
        name=FILTER_NAME,
        value=protocol,
        passed=passed,
        created_at=record.created_at,
    )


def hit_protocol_summary(manifest: RunManifest) -> str:
    """The gate `protein_hunter_high_iptm` actually applies, in one line."""
    protocol = manifest.workflow.get("hit_protocol")
    if not isinstance(protocol, dict):
        return ""
    parts = [
        f"iptm>{protocol.get('iptm_above')}",
        f"plddt>{protocol.get('plddt_above')}",
        f"alanine<={protocol.get('alanine_fraction_at_most')}",
    ]
    contacts = protocol.get("contacts")
    if isinstance(contacts, dict):
        parts.append(
            f"contacts>={contacts.get('min_residues_contacted')}"
            f"@{contacts.get('cutoff_angstroms')}A"
        )
    return " & ".join(parts)


def _best_decisions(
    path: Path, run_id: str, designs: list[Design], created_at: datetime
) -> list[DecisionRecord]:
    """Which cycle each trajectory kept as its best.

    A selection rather than a rank: the table names one winner per trajectory
    and says nothing about the order of the rest. A trajectory whose cycles all
    tripped the alanine cap names none, and then nothing is recorded for it.
    """
    best = _best_cycles(path)
    records = []
    for design in designs:
        if best.get(design.run_id) != design.cycle:
            continue
        design_id = stable_id("design", run_id, design.native_id)
        records.append(DecisionRecord(
            decision_id=stable_id("decision", run_id, design_id, BEST_NAME),
            run_id=run_id,
            design_id=design_id,
            kind=DecisionKind.SELECTION,
            name=BEST_NAME,
            passed=True,
            # The pool this selection was made within: one trajectory's cycles.
            scope_id=trajectory_scope(run_id, design.run_id),
            created_at=created_at,
        ))
    return records


def _best_cycles(path: Path) -> dict[int, int]:
    """The winning cycle per trajectory, where the tool recorded one."""
    if not path.is_file():
        return {}
    best = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = _integer(row.get("run_id"))
            cycle = _integer(row.get("best_cycle"))
            if run_id is not None and cycle is not None:
                best[run_id] = cycle
    return best


def trajectory_scope(run_id: str, trajectory: int) -> str:
    """The pool a best-cycle selection is meaningful within."""
    return stable_id("trajectory-scope", run_id, f"run-{trajectory:04d}")


def _design_artifacts(
    run_dir: Path, run_id: str, task: TaskPlan, record: DesignRecord,
    verdict: dict[str, str] | None,
) -> list[ArtifactRecord]:
    """The co-folded complex and its spec, kept only for what passed.

    The pipeline writes structures for the designs that cleared its thresholds
    and nothing for the rest, so most produced designs have no file. The names
    come from the threshold table, so they are treated as untrusted.
    """
    if verdict is None:
        return []
    wanted = [
        (STRUCTURE_DIR, (verdict.get("pdb_filename") or "").strip(), "design_complex"),
        (SPEC_DIR, (verdict.get("yaml_filename") or "").strip(), "design_spec"),
    ]
    records = []
    for directory, filename, kind in wanted:
        if not filename or not _is_contained(filename):
            continue
        relative = f"{task.directory}/{directory}/{filename}"
        found = artifact(run_dir, run_id, relative, kind, design_id=record.design_id)
        if found is not None:
            records.append(found)
    return records


def _is_contained(relative: str) -> bool:
    """True when a tool-supplied path stays inside the directory it names."""
    candidate = Path(relative)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _task_artifacts(run_dir: Path, run_id: str, task: TaskPlan) -> list[ArtifactRecord]:
    wanted = [
        ("native_design_table", task.designs),
        ("threshold_table", f"{task.directory}/{HIGH_IPTM_FILE}"),
        ("task_status", task.status),
        ("log", task.log),
    ]
    return [
        record
        for kind, relative in wanted
        if (record := artifact(run_dir, run_id, relative, kind)) is not None
    ]


def _binder_lengths(manifest: RunManifest) -> tuple[int, int] | None:
    minimum = manifest.workflow.get("min_binder_length")
    maximum = manifest.workflow.get("max_binder_length")
    if isinstance(minimum, int) and isinstance(maximum, int):
        return (minimum, maximum)
    return None


def _attempted(statuses: list, tasks: tuple[TaskPlan, ...]) -> int:
    """What the run set out to produce, as the tasks themselves reported it."""
    total = 0
    for status, task in zip(statuses, tasks, strict=True):
        if status is not None and status.n_attempted is not None:
            total += status.n_attempted
        else:
            total += task.n_generated if task.n_generated is not None else task.n_requested
    return total


def _integer(value: Any) -> int | None:
    """An index is a whole number, so a fractional one is a parse error."""
    number = _number(value)
    if number is None or number < 0 or number != int(number):
        return None
    return int(number)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number
