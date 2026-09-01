"""Reading a finished Proteina-Complexa run.

The fixtures are unmodified rows from the 2026-08 benchmark against DIO3-cut:
four designs from `binder_results_*.csv`, the two of them the evaluation gate
passed, the criteria file that judged them, and four rows of the generation
rewards table.

What the tests are really about is that generating, keeping and passing are
three different numbers, and that a missing verdict table is an unfinished run
rather than a genuine zero.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
from conftest import write_proteina_complexa_configs

from bindocracy.runs.manifest import RunManifest
from bindocracy.tools import collect_run, load_configs, plan
from bindocracy.tools.proteina_complexa.adapter import SUCCESS_FILTER
from bindocracy.tools.proteina_complexa.launch import (
    criteria_file,
    designs_file,
    rewards_file,
    successes_file,
)

FIXTURES = Path(__file__).parent / "fixtures" / "proteina_complexa"
TASK_NAME = "test_target"


def build_run(tmp_path: Path, *, jobs: int = 1, omit_successes: bool = False,
              truncate_designs: int | None = None) -> Path:
    """Plan a run, then fill its task directories with real output."""
    general, model = write_proteina_complexa_configs(
        tmp_path, sampling={"jobs": jobs, "samples_per_job": 4, "keep_per_job": 4}
    )
    loaded = load_configs(general, model)
    manifest = plan(loaded, tmp_path / "run", name="pcx")

    for task in manifest.tasks:
        task_dir = manifest.directory / task.directory
        for relative in (designs_file(TASK_NAME), rewards_file(TASK_NAME)):
            (task_dir / relative).parent.mkdir(parents=True, exist_ok=True)

        _write_designs(task_dir / designs_file(TASK_NAME), truncate_designs)
        shutil.copy(FIXTURES / "all_rewards.csv", task_dir / rewards_file(TASK_NAME))
        shutil.copy(FIXTURES / "success_criteria.json", task_dir / criteria_file(TASK_NAME))
        if not omit_successes:
            shutil.copy(FIXTURES / "all_successes.csv", task_dir / successes_file(TASK_NAME))

        (task_dir / "status.json").write_text(json.dumps({
            "task_id": task.task_id,
            "status": "succeeded",
            "started_at": "2026-08-26T19:00:00+00:00",
            "finished_at": "2026-08-27T01:21:00+00:00",
            "exit_code": 0,
            "written_by": "harness",
        }))
        (manifest.directory / task.log).parent.mkdir(parents=True, exist_ok=True)
        (manifest.directory / task.log).write_text("proteina-complexa log\n")
    return manifest.directory / "run.json"


def _write_designs(path: Path, limit: int | None) -> None:
    with (FIXTURES / "binder_results.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)[: limit if limit is not None else None]
        fields = reader.fieldnames
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --- the three counts --------------------------------------------------------


def test_generating_keeping_and_passing_are_three_numbers(tmp_path: Path) -> None:
    collected = collect_run(build_run(tmp_path))

    assert collected.run.n_attempted == 4   # rows in the rewards table
    assert collected.run.n_produced == 4    # rows in binder_results
    assert collected.run.n_passed == 2      # rows in all_successes
    assert len(collected.designs) == 4


def test_every_design_carries_the_sequence_the_model_generated(tmp_path: Path) -> None:
    """`sequence_types: [self]` -- not an inverse-folding redesign."""
    collected = collect_run(build_run(tmp_path))

    for design in collected.designs:
        assert design.sequence and design.sequence.isupper()
        assert design.length == len(design.sequence)
        assert design.candidate_type == "complex"


def test_the_verdict_carries_the_whole_protocol(tmp_path: Path) -> None:
    """Not just the configurable threshold: all three, joined by AND."""
    collected = collect_run(build_run(tmp_path))
    decisions = [d for d in collected.decisions if d.name == SUCCESS_FILTER]

    assert len(decisions) == 4
    assert sum(1 for d in decisions if d.passed) == 2
    thresholds = decisions[0].reason["thresholds"]
    assert set(thresholds) == {"i_pAE", "pLDDT", "scRMSD_ca"}
    # A rank or verdict without the pool it was reached in is not reproducible.
    assert all(d.scope_id for d in decisions)


def test_metrics_come_across_with_a_direction(tmp_path: Path) -> None:
    collected = collect_run(build_run(tmp_path))
    by_name = {m.name: m for m in collected.metrics}

    assert by_name["proteina_complex_iptm"].direction == "max"
    assert by_name["proteina_binder_scrmsd_ca"].direction == "min"
    assert by_name["proteina_complex_ipae"].direction == "min"


# --- partial and broken runs -------------------------------------------------


def test_a_missing_verdict_table_is_not_a_genuine_zero(tmp_path: Path) -> None:
    """The evaluation stage did not finish; it did not find nothing."""
    collected = collect_run(build_run(tmp_path, omit_successes=True))

    assert collected.run.n_produced == 4
    assert collected.run.n_passed is None
    assert collected.run.status == "partial"
    assert not collected.decisions


def test_a_short_table_is_reported_per_task(tmp_path: Path) -> None:
    collected = collect_run(build_run(tmp_path, truncate_designs=2))
    task = collected.run.count_details["tasks"]["0000"]

    assert task["n_produced"] == 2
    assert task["n_generated"] == 4


def test_the_zero_ipsae_count_is_recorded(tmp_path: Path) -> None:
    """known-issues §1.7 claims every ipSAE is 0.0; the count makes that checkable."""
    collected = collect_run(build_run(tmp_path))
    task = collected.run.count_details["tasks"]["0000"]

    assert "n_zero_avg_ipsae" in task
    assert task["n_zero_avg_ipsae"] < 4


def test_a_success_naming_no_design_is_counted_not_trusted(tmp_path: Path) -> None:
    """A verdict table is evidence; a stale one marks the wrong design passed."""
    manifest_path = build_run(tmp_path)
    manifest = RunManifest.read(manifest_path)
    successes = manifest.directory / manifest.tasks[0].directory / successes_file(TASK_NAME)
    with successes.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    rows.append({**rows[0], "binder_sequence": "MKTAYIAKQRQISFVKSHFSRQ"})
    with successes.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    collected = collect_run(manifest_path)
    task = collected.run.count_details["tasks"]["0000"]

    assert task["successes_naming_no_design"] == ["MKTAYIAKQRQISFVKSHFSRQ"]


def test_no_designs_at_all_is_a_failed_run(tmp_path: Path) -> None:
    collected = collect_run(build_run(tmp_path, truncate_designs=0))

    assert collected.run.n_produced == 0
    assert collected.run.status == "failed"


# --- the properties every adapter owes ---------------------------------------


def test_two_tasks_do_not_collide(tmp_path: Path) -> None:
    """Proteina numbers samples per task, so IDs are qualified by task."""
    collected = collect_run(build_run(tmp_path, jobs=2))
    native_ids = [design.native_id for design in collected.designs]

    assert len(native_ids) == 8
    assert len(set(native_ids)) == 8
    assert sum(1 for name in native_ids if name.startswith("task-0001-")) == 4


def test_recollection_is_deterministic(tmp_path: Path) -> None:
    manifest_path = build_run(tmp_path)

    first = collect_run(manifest_path)
    second = collect_run(manifest_path)

    assert first.content_hash() == second.content_hash()


def test_each_design_is_recorded_once_despite_the_duplicate_on_disk(
    tmp_path: Path,
) -> None:
    """Evaluation copies every sample directory out of inference/."""
    manifest_path = build_run(tmp_path)
    manifest = RunManifest.read(manifest_path)
    task_dir = manifest.directory / manifest.tasks[0].directory

    # Follow the paths the rows actually record rather than rebuilding them:
    # the fixture carries the benchmark's own directory stem, and the whole
    # point is that the adapter reads the table instead of guessing.
    with (FIXTURES / "binder_results.csv").open(newline="") as handle:
        recorded = [row["pdb_path"] for row in csv.DictReader(handle)]
    for path in recorded:
        evaluated = task_dir / path.removeprefix("./")
        evaluated.parent.mkdir(parents=True, exist_ok=True)
        evaluated.write_text("ATOM\nEND\n")
        # The same structure, left behind under the generation root.
        duplicate = task_dir / path.removeprefix("./").replace(
            "evaluation_results/", "inference/", 1
        )
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        duplicate.write_text("ATOM\nEND\n")

    recollected = collect_run(manifest_path)
    complexes = [a for a in recollected.artifacts if a.kind == "design_complex"]

    assert len(complexes) == 4
    assert all("evaluation_results" in a.uri for a in complexes)


def test_the_registry_is_archived_as_an_artifact(tmp_path: Path) -> None:
    collected = collect_run(build_run(tmp_path))

    assert any(a.kind == "target_registry" for a in collected.artifacts)


@pytest.mark.parametrize("kind", ["generation_rewards", "success_table", "native_designs"])
def test_the_evidence_files_are_recorded(tmp_path: Path, kind: str) -> None:
    collected = collect_run(build_run(tmp_path))

    assert any(a.kind == kind for a in collected.artifacts)


def test_an_orphan_success_does_not_inflate_n_passed(tmp_path: Path) -> None:
    """A verdict table is evidence, and an orphan row is bad evidence.

    n_passed and the per-design verdicts have to agree: counting a sequence no
    design produced reports a hit the run never made.
    """
    manifest_path = build_run(tmp_path)
    manifest = RunManifest.read(manifest_path)
    successes = manifest.directory / manifest.tasks[0].directory / successes_file(TASK_NAME)
    with successes.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    rows.append({**rows[0], "binder_sequence": "MKTAYIAKQRQISFVKSHFSRQ"})
    with successes.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    collected = collect_run(manifest_path)
    passing_decisions = sum(1 for d in collected.decisions if d.passed)

    assert collected.run.n_passed == passing_decisions == 2
    assert collected.run.count_details["tasks"]["0000"]["n_passed"] == 2
