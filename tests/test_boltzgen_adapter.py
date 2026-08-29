"""BoltzGen collection, against real benchmark rows.

The fixture is four unmodified rows from the 2026-08-27 run: two that pass the
tool's own filters and two that do not, including the alanine-heavy design that
motivated `--filter_biased`. The point of most of these tests is that BoltzGen
counts differently from Mosaic, and the schema has to keep the difference.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml
from conftest import write_boltzgen_task

from bindocracy.tools import collect_run, load_configs, plan
from bindocracy.tools.boltzgen.adapter import BoltzGenOutputAdapter


def planned(boltzgen_configs, run_dir: Path):
    return plan(load_configs(*boltzgen_configs), run_dir)


def collect(manifest):
    return BoltzGenOutputAdapter().collect(manifest.directory, manifest.to_run_record())


def test_designs_and_native_scores_are_read(boltzgen_configs, tmp_path: Path) -> None:
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status={})

    collected = collect(manifest)

    # Qualified by task: BoltzGen's own ids restart at 0 in every task.
    assert [d.native_id for d in collected.designs] == [
        "task-0000-dio3_cut_binder_14", "task-0000-dio3_cut_binder_18",
        "task-0000-dio3_cut_binder_33", "task-0000-dio3_cut_binder_39",
    ]
    first = collected.designs[0]
    assert first.length == 90
    assert first.sequence.startswith("PPLTDADAREL")
    scores = {m.name: m.value for m in collected.metrics if m.design_id == first.design_id}
    assert scores["boltzgen_design_to_target_iptm"] == pytest.approx(0.63545)
    assert scores["boltzgen_min_design_to_target_pae"] == pytest.approx(3.31294)
    directions = {m.name: m.direction for m in collected.metrics}
    assert directions["boltzgen_design_to_target_iptm"] == "max"
    assert directions["boltzgen_min_design_to_target_pae"] == "min"


def test_produced_and_passed_are_different_numbers(boltzgen_configs, tmp_path: Path) -> None:
    """The counting trap: BoltzGen filters its own pool, Mosaic does not."""
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status={"n_attempted": 8})

    run = collect(manifest).run

    assert run.n_requested == 4      # budget
    assert run.n_attempted == 8      # backbones generated
    assert run.n_produced == 4       # rows that survived into the table
    assert run.n_passed == 2         # rows the tool's own filters accepted
    assert run.count_details["tasks"]["0000"]["n_passed"] == 2


def test_the_filter_verdict_is_a_decision_not_a_lesser_design(
    boltzgen_configs, tmp_path: Path
) -> None:
    """Producing a design and passing a filter are different facts.

    Every complete row is `produced`; whether BoltzGen's own filters accepted
    it is a decision, so a later filtering pass adds a verdict rather than
    contradicting this one.
    """
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status={})

    collected = collect(manifest)
    by_id = {d.native_id: d for d in collected.designs}
    verdicts = {d.design_id: d for d in collected.decisions if d.kind == "filter"}

    assert {d.status for d in collected.designs} == {"produced"}
    assert verdicts[by_id["task-0000-dio3_cut_binder_14"].design_id].passed is True
    assert verdicts[by_id["task-0000-dio3_cut_binder_39"].design_id].passed is False


def test_a_rank_is_scoped_to_the_task_that_produced_it(
    boltzgen_configs, tmp_path: Path
) -> None:
    """BoltzGen ranks each task's pool from 1, so the run is the wrong scope.

    A two-task run has two designs ranked first. Recording both against the
    run would say something false; recording them against their task does not.
    """
    general_path, model_path = boltzgen_configs
    raw = yaml.safe_load(model_path.read_text())
    raw["sampling"]["jobs"] = 2
    model_path.write_text(yaml.safe_dump(raw))
    manifest = plan(load_configs(general_path, model_path), tmp_path / "run")
    for task_id in (0, 1):
        write_boltzgen_task(manifest.directory, task_id, status={})

    ranks = [d for d in collect(manifest).decisions if d.kind == "rank"]

    assert len(ranks) == 8
    assert len({d.scope_id for d in ranks}) == 2
    firsts = [d for d in ranks if d.rank == 1]
    assert len(firsts) == 2
    assert firsts[0].scope_id != firsts[1].scope_id


def test_designs_are_complexes_not_bare_sequences(boltzgen_configs, tmp_path: Path) -> None:
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status={})

    collected = collect(manifest)

    assert {d.candidate_type for d in collected.designs} == {"complex"}
    structures = [a for a in collected.artifacts if a.kind == "design_complex"]
    assert len(structures) == 4
    assert all(a.design_id is not None for a in structures)
    assert any("rank01_dio3_cut_binder_14.cif" in a.uri for a in structures)


def test_the_archived_spec_is_an_artifact(boltzgen_configs, tmp_path: Path) -> None:
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status={})

    kinds = {a.kind for a in collect(manifest).artifacts}

    assert "design_spec" in kinds
    assert {"native_design_table", "task_status", "log"} <= kinds


def test_a_missing_table_is_a_failed_run(boltzgen_configs, tmp_path: Path) -> None:
    manifest = planned(boltzgen_configs, tmp_path / "run")
    (manifest.directory / "logs").mkdir(exist_ok=True)

    collected = collect(manifest)

    assert collected.run.status == "failed"
    assert collected.designs == ()
    assert collected.run.n_produced == 0


def test_rows_without_a_usable_sequence_are_skipped(boltzgen_configs, tmp_path: Path) -> None:
    """`has_x` designs carry unknown residues and cannot be ordered."""
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status={})
    table = manifest.path("tasks/0000/final_ranked_designs/all_designs_metrics.csv")
    rows = list(csv.DictReader(table.open(newline="")))
    rows[1]["designed_sequence"] = "ACDXFG"
    rows[2]["id"] = ""
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    collected = collect(manifest)

    assert len(collected.designs) == 2
    assert collected.run.count_details["tasks"]["0000"]["n_invalid"] == 2


def test_recollection_is_deterministic(boltzgen_configs, tmp_path: Path) -> None:
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status={})
    path = manifest.directory / "run.json"

    assert collect_run(path).content_hash() == collect_run(path).content_hash()


def test_collection_dispatches_to_the_boltzgen_adapter(
    boltzgen_configs, tmp_path: Path
) -> None:
    """The manifest says boltzgen, so the mosaic adapter never sees this run."""
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status={})

    collected = collect_run(manifest.directory / "run.json")

    assert collected.run.tool == "boltzgen"
    assert all(m.name.startswith("boltzgen_") for m in collected.metrics)


def test_the_structure_directory_is_found_not_assumed(
    boltzgen_configs, tmp_path: Path
) -> None:
    """BoltzGen names it for the budget: final_10_designs at budget 10.

    Hardcoding the benchmark's `final_40_designs` silently dropped every
    structure artifact for any other budget.
    """
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status={})
    ranked = manifest.path("tasks/0000/final_ranked_designs")
    (ranked / "final_40_designs").rename(ranked / "final_10_designs")

    structures = [a for a in collect(manifest).artifacts if a.kind == "design_complex"]

    assert len(structures) == 4
    assert all("final_10_designs" in a.uri for a in structures)


def test_a_task_with_no_status_file_still_collects(boltzgen_configs, tmp_path: Path) -> None:
    """BoltzGen writes no status of its own; the harness supplies one."""
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status=None)

    collected = collect(manifest)

    assert len(collected.designs) == 4
    assert collected.run.count_details["tasks"]["0000"]["status"] == "missing"
    assert collected.run.status == "partial"


def test_two_tasks_do_not_collide_on_native_ids(boltzgen_configs, tmp_path: Path) -> None:
    """BoltzGen numbers designs per task, so task 1 repeats task 0's ids.

    Mosaic embeds the task in its native_id; BoltzGen's comes from the CSV and
    restarts at 0 every task. Without qualification the second task's designs
    are all rejected as duplicates, and a two-task run silently loses half its
    output.
    """
    general_path, model_path = boltzgen_configs
    raw = yaml.safe_load(model_path.read_text())
    raw["sampling"]["jobs"] = 2
    model_path.write_text(yaml.safe_dump(raw))
    manifest = plan(load_configs(general_path, model_path), tmp_path / "run")
    for task_id in (0, 1):
        write_boltzgen_task(manifest.directory, task_id, status={})

    collected = collect(manifest)

    assert len(collected.designs) == 8
    assert len({d.native_id for d in collected.designs}) == 8
    assert collected.run.count_details["tasks"]["0001"]["n_invalid"] == 0


def test_attempted_counts_what_was_generated_not_the_budget(
    boltzgen_configs, tmp_path: Path
) -> None:
    """n_requested is the budget; n_attempted is num_designs.

    Falling back to n_requested made a run that generated 8 and kept 4 look
    like it had only ever tried 4.
    """
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status=None)  # no native count

    run = collect(manifest).run

    assert manifest.tasks[0].n_requested == 4    # budget
    assert manifest.tasks[0].n_generated == 8    # num_designs
    assert run.n_requested == 4
    assert run.n_attempted == 8


def test_a_native_attempt_count_wins_over_the_configured_one(
    boltzgen_configs, tmp_path: Path
) -> None:
    manifest = planned(boltzgen_configs, tmp_path / "run")
    write_boltzgen_task(manifest.directory, 0, status={"n_attempted": 6})

    assert collect(manifest).run.n_attempted == 6
