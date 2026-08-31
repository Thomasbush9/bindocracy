"""Reading a finished Protein-Hunter run.

The fixture is three unmodified trajectories of the 2026-08-26 benchmark run:
one that cleared the thresholds five times, one that cleared them once, and one
whose every cycle tripped the 20% alanine cap so the tool recorded no best for
it at all. That last one is the point — its five sequences are still designs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    PROTEIN_HUNTER_FIXTURE,
    write_protein_hunter_configs,
    write_protein_hunter_task,
)

from bindocracy.adapters.base import CollectionError
from bindocracy.store.records import (
    CandidateType,
    DecisionKind,
    DesignStatus,
    RunStatus,
)
from bindocracy.tools import collect_run, load_configs, plan
from bindocracy.tools.protein_hunter.adapter import (
    BEST_NAME,
    FILTER_NAME,
    ProteinHunterOutputAdapter,
)

# The alanine-capped trajectory: five sequences, no best, nothing passing.
CAPPED = 4


@pytest.fixture
def run(tmp_path: Path):
    """A planned three-trajectory, five-cycle, one-task run."""
    general, model = write_protein_hunter_configs(
        tmp_path, sampling={"jobs": 1, "trajectories_per_job": 3, "cycles": 5}
    )
    return plan(load_configs(general, model), tmp_path / "run")


@pytest.fixture
def two_task_run(tmp_path: Path):
    general, model = write_protein_hunter_configs(
        tmp_path, sampling={"jobs": 2, "trajectories_per_job": 3, "cycles": 5}
    )
    return plan(load_configs(general, model), tmp_path / "run")


def collected(manifest):
    return collect_run(manifest.directory / "run.json")


# --- unpacking the wide table ----------------------------------------------


def test_a_design_is_a_cycle_not_a_trajectory(run) -> None:
    """Three trajectories at five cycles is fifteen sequences."""
    write_protein_hunter_task(run.directory, 0, status={})
    result = collected(run)

    assert result.run.n_produced == 15
    assert result.run.count_details["tasks"]["0000"]["n_trajectories"] == 3
    assert {d.metadata["cycle"] for d in result.designs} == {1, 2, 3, 4, 5}


def test_cycle_zero_is_not_a_design(run) -> None:
    """It is the fold of the starting mostly-X binder, with no sequence."""
    write_protein_hunter_task(run.directory, 0, status={})
    result = collected(run)

    assert 0 not in {design.metadata["cycle"] for design in result.designs}
    assert all(design.sequence for design in result.designs)
    # Skipped as "not a design", not rejected as malformed: three empty
    # cycle_0 cells must not show up as three invalid rows.
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 0


def test_each_design_carries_its_own_scores(run) -> None:
    write_protein_hunter_task(run.directory, 0, status={})
    result = collected(run)

    first = next(d for d in result.designs
                 if d.native_id == "task-0000-run-0000-cycle-5")
    scores = {m.name: m.value for m in result.metrics if m.design_id == first.design_id}
    assert scores["protein_hunter_iptm"] == pytest.approx(0.8472905158996582)
    assert scores["protein_hunter_alanine"] == 12
    assert first.candidate_type == CandidateType.COMPLEX
    assert first.status == DesignStatus.PRODUCED


# --- the alanine cap -------------------------------------------------------


def test_a_trajectory_with_no_best_still_produced_its_designs(run) -> None:
    """Six of the benchmark's forty had every cycle excluded by the cap."""
    write_protein_hunter_task(run.directory, 0, status={})
    result = collected(run)

    capped = [d for d in result.designs if d.metadata["trajectory"] == CAPPED]
    assert len(capped) == 5
    assert result.run.count_details["tasks"]["0000"]["n_trajectories_without_best"] == 1


def test_no_best_is_selected_for_a_capped_trajectory(run) -> None:
    """The tool named no winner, so the harness does not invent one."""
    write_protein_hunter_task(run.directory, 0, status={})
    result = collected(run)

    by_id = {d.design_id: d.metadata["trajectory"] for d in result.designs}
    selected = [by_id[d.design_id] for d in result.decisions if d.name == BEST_NAME]
    assert CAPPED not in selected
    assert sorted(selected) == [0, 1]


def test_each_best_is_scoped_to_its_own_trajectory(run) -> None:
    """Two trajectories each name a winner; they are not competing."""
    write_protein_hunter_task(run.directory, 0, status={})
    result = collected(run)

    best = [d for d in result.decisions if d.name == BEST_NAME]
    assert all(decision.kind == DecisionKind.SELECTION for decision in best)
    assert len({decision.scope_id for decision in best}) == len(best)


# --- the tool's own verdict ------------------------------------------------


def test_passing_is_membership_in_the_threshold_table(run) -> None:
    write_protein_hunter_task(run.directory, 0, status={})
    result = collected(run)

    assert result.run.n_produced == 15
    assert result.run.n_passed == 6
    verdicts = [d for d in result.decisions if d.name == FILTER_NAME]
    assert len(verdicts) == 15
    assert sum(1 for decision in verdicts if decision.passed) == 6


def test_no_threshold_table_on_a_finished_task_is_a_real_zero(run) -> None:
    """The pipeline writes neither the table nor high_iptm_* when nothing clears.

    Observed on the cluster: a completed one-trajectory run whose only
    above-threshold cycle was excluded by the alanine cap left no threshold
    table and no directories at all.
    """
    write_protein_hunter_task(run.directory, 0, thresholds=False, status={})
    result = collected(run)

    assert result.run.n_produced == 15
    assert result.run.n_passed == 0
    assert result.run.status == RunStatus.SUCCEEDED
    verdicts = [d for d in result.decisions if d.name == FILTER_NAME]
    assert len(verdicts) == 15 and not any(decision.passed for decision in verdicts)


def test_no_threshold_table_on_an_unfinished_task_judges_nothing(run) -> None:
    """Short of what it was asked for, absence could mean either thing."""
    write_protein_hunter_task(run.directory, 0, trajectories=2, thresholds=False,
                              status={})
    result = collected(run)

    assert result.run.n_produced == 10
    assert result.run.n_passed is None
    assert result.run.status == RunStatus.PARTIAL
    assert result.run.count_details["tasks"]["0000"]["n_passed"] is None
    assert not [d for d in result.decisions if d.name == FILTER_NAME]


def test_an_empty_threshold_table_is_a_real_zero(run) -> None:
    """Nothing cleared, and every design was judged and told so."""
    header = (PROTEIN_HUNTER_FIXTURE / "summary_high_iptm.csv").read_text().splitlines()[0]
    write_protein_hunter_task(run.directory, 0, status={})
    (run.directory / "tasks" / "0000" / "summary_high_iptm.csv").write_text(header + "\n")
    result = collected(run)

    assert result.run.n_passed == 0
    assert result.run.status == RunStatus.SUCCEEDED
    verdicts = [d for d in result.decisions if d.name == FILTER_NAME]
    assert len(verdicts) == 15 and not any(d.passed for d in verdicts)


# --- counts and malformed output -------------------------------------------


def test_fewer_trajectories_than_asked_for_is_partial(run) -> None:
    write_protein_hunter_task(run.directory, 0, trajectories=2, status={})
    result = collected(run)

    assert result.run.n_produced == 10
    assert result.run.n_requested == 15
    assert result.run.status == RunStatus.PARTIAL


def test_a_truncated_table_keeps_the_trajectories_it_completed(run) -> None:
    text = (PROTEIN_HUNTER_FIXTURE / "summary_all_runs.csv").read_text()
    write_protein_hunter_task(run.directory, 0, summary=text[: int(len(text) * 0.5)],
                              status={})
    result = collected(run)

    assert result.run.n_produced >= 5
    assert all(design.sequence for design in result.designs)


def test_a_row_with_no_trajectory_id_is_skipped_and_counted(run) -> None:
    lines = (PROTEIN_HUNTER_FIXTURE / "summary_all_runs.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    orphan = "," + body[0].split(",", 1)[1]
    write_protein_hunter_task(run.directory, 0,
                              summary="\n".join([header, orphan, *body]) + "\n",
                              status={})
    result = collected(run)

    assert result.run.n_produced == 15
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_a_sequence_outside_the_configured_range_is_not_a_design(run) -> None:
    """The pipeline was told 65..120; anything else is not from this config."""
    lines = (PROTEIN_HUNTER_FIXTURE / "summary_all_runs.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    columns = header.split(",")
    column = columns.index("cycle_1_seq")
    fields = body[0].split(",")
    fields[column] = "ACDEF"
    write_protein_hunter_task(run.directory, 0,
                              summary="\n".join([header, ",".join(fields), *body[1:]]) + "\n",
                              status={})
    result = collected(run)

    assert result.run.n_produced == 14
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_a_missing_summary_is_not_a_crash(run) -> None:
    write_protein_hunter_task(run.directory, 0, status={})
    (run.directory / run.tasks[0].designs).unlink()
    result = collected(run)

    assert result.run.n_produced == 0
    assert result.run.status == RunStatus.FAILED


def test_a_run_directory_for_another_run_is_refused(run) -> None:
    write_protein_hunter_task(run.directory, 0, status={})
    other = run.to_run_record().model_copy(update={"run_id": "not-this-run"})

    with pytest.raises(CollectionError, match="does not match manifest"):
        ProteinHunterOutputAdapter().collect(run.directory, other)


# --- two tasks -------------------------------------------------------------


def test_two_tasks_do_not_collect_as_duplicates_of_each_other(two_task_run) -> None:
    """Trajectories are numbered from 0 in each task's own save directory."""
    for task_id in (0, 1):
        write_protein_hunter_task(two_task_run.directory, task_id, status={})
    result = collected(two_task_run)

    assert result.run.n_produced == 30
    assert len({design.native_id for design in result.designs}) == 30
    assert len({design.design_id for design in result.designs}) == 30


def test_a_task_that_never_ran_does_not_cost_the_one_that_did(two_task_run) -> None:
    write_protein_hunter_task(two_task_run.directory, 0, status={})
    result = collected(two_task_run)

    assert result.run.n_produced == 15
    assert result.run.status == RunStatus.PARTIAL
    assert result.run.count_details["tasks"]["0001"]["status"] == "missing"


# --- artifacts and determinism ---------------------------------------------


def test_only_the_designs_that_passed_have_a_structure(run) -> None:
    """The pipeline writes structures for what cleared its thresholds only."""
    write_protein_hunter_task(run.directory, 0, status={})
    result = collected(run)

    complexes = [a for a in result.artifacts if a.kind == "design_complex"]
    assert len(complexes) == 6
    assert all(artifact.design_id is not None for artifact in complexes)
    for record in complexes:
        assert (run.directory / record.uri).is_file()


def test_a_structure_path_climbing_out_of_the_run_is_dropped(run) -> None:
    """The filenames come from the tool's table, so they are not trusted."""
    write_protein_hunter_task(run.directory, 0, status={})
    table = run.directory / "tasks" / "0000" / "summary_high_iptm.csv"
    lines = table.read_text().splitlines()
    header, body = lines[0], lines[1:]
    columns = header.split(",")
    fields = body[0].split(",")
    fields[columns.index("pdb_filename")] = "../../../run.json"
    table.write_text("\n".join([header, ",".join(fields), *body[1:]]) + "\n")
    result = collected(run)

    assert all(".." not in Path(a.uri).parts for a in result.artifacts)
    assert len([a for a in result.artifacts if a.kind == "design_complex"]) == 5


def test_what_the_run_configured_is_collected(run) -> None:
    write_protein_hunter_task(run.directory, 0, status={})
    kinds = {artifact.kind for artifact in collected(run).artifacts}

    assert {"native_design_table", "threshold_table", "design_complex",
            "design_spec", "driver_script", "task_status", "log"} <= kinds


def test_recollecting_an_unchanged_directory_is_identical(run) -> None:
    write_protein_hunter_task(run.directory, 0, status={})

    assert collected(run).content_hash() == collected(run).content_hash()


def test_succeeded_needs_every_task_to_have_finished(two_task_run) -> None:
    """A finished task with nothing above threshold is still a success."""
    adapter = ProteinHunterOutputAdapter()
    write_protein_hunter_task(two_task_run.directory, 0, status={})

    assert not adapter.succeeded(two_task_run.directory)
    write_protein_hunter_task(two_task_run.directory, 1, trajectories=2, status={})
    assert not adapter.succeeded(two_task_run.directory)
    # Complete, and with no threshold table at all -- which is what zero hits
    # looks like on disk.
    write_protein_hunter_task(two_task_run.directory, 1, thresholds=False, status={})
    assert adapter.succeeded(two_task_run.directory)


def test_the_manifest_is_what_says_how_long_a_binder_may_be(run) -> None:
    """The range is a config field, not something the output declares."""
    manifest_path = run.directory / "run.json"
    document = json.loads(manifest_path.read_text())
    assert document["workflow"]["min_binder_length"] == 65
    assert document["workflow"]["max_binder_length"] == 120
