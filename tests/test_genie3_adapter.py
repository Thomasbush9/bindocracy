"""Reading a finished Genie 3 run.

The fixture is ten unmodified rows of the 2026-08-26 benchmark run: two designs
folded five times each. That shape is the whole point — a row is a fold, not a
design, and a parser that misses the difference multiplies the campaign's
design count by five.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from conftest import GENIE3_FIXTURE, write_genie3_configs, write_genie3_task

from bindocracy.adapters.base import CollectionError
from bindocracy.store.records import (
    CandidateType,
    DecisionKind,
    DesignStatus,
    MetricDirection,
    RunStatus,
)
from bindocracy.tools import collect_run, load_configs, plan
from bindocracy.tools.genie3.adapter import FILTER_NAME, Genie3OutputAdapter

FIRST = "task-0000-dio3_cut_0-resample_0"
SECOND = "task-0000-dio3_cut_1-resample_0"


@pytest.fixture
def run(tmp_path: Path):
    """A planned two-backbone, one-task run, ready for task directories."""
    general, model = write_genie3_configs(
        tmp_path, sampling={"jobs": 1, "backbones_per_job": 2, "seed_base": 100}
    )
    return plan(load_configs(general, model), tmp_path / "run")


@pytest.fixture
def two_task_run(tmp_path: Path):
    general, model = write_genie3_configs(
        tmp_path, sampling={"jobs": 2, "backbones_per_job": 2, "seed_base": 100}
    )
    return plan(load_configs(general, model), tmp_path / "run")


def collected(manifest):
    return collect_run(manifest.directory / "run.json")


# --- one design is several rows --------------------------------------------


def test_five_folds_of_one_design_collapse_to_one_design(run) -> None:
    write_genie3_task(run.directory, 0, status={})
    result = collected(run)

    assert [design.native_id for design in result.designs] == [FIRST, SECOND]
    assert result.run.n_produced == 2
    assert all(design.metadata["n_folds"] == 5 for design in result.designs)


def test_every_fold_becomes_a_replicate_of_the_same_metric(run) -> None:
    write_genie3_task(run.directory, 0, status={})
    result = collected(run)

    design = next(d for d in result.designs if d.native_id == FIRST)
    iptm = [m for m in result.metrics if m.design_id == design.design_id
            and m.name == "genie3_iptm"]
    # Genie 3's rank orders a design's folds, so replicate 0 is its best one.
    assert sorted(metric.replicate for metric in iptm) == [0, 1, 2, 3, 4]
    assert next(m.value for m in iptm if m.replicate == 0) == 0.16
    assert {metric.direction for metric in iptm} == {MetricDirection.MAX}


def test_the_design_takes_its_sequence_from_its_best_ranked_fold(run) -> None:
    write_genie3_task(run.directory, 0, status={})
    result = collected(run)

    first = next(d for d in result.designs if d.native_id == FIRST)
    assert first.length == 90
    assert first.candidate_type == CandidateType.COMPLEX
    assert first.status == DesignStatus.PRODUCED
    assert first.metadata["backbone"] == "dio3_cut_0"


def test_folds_with_no_usable_rank_still_get_distinct_replicates(
    run, tmp_path: Path
) -> None:
    """A metric ID has to be unique whether or not `rank` survived the write."""
    rows = (GENIE3_FIXTURE / "info.csv").read_text().splitlines()
    header, body = rows[0], rows[1:]
    blanked = [",".join(["" if index == 6 else field
                         for index, field in enumerate(line.split(","))])
               for line in body]
    write_genie3_task(run.directory, 0, results="\n".join([header, *blanked]) + "\n",
                      status={})
    result = collected(run)

    design = next(d for d in result.designs if d.native_id == FIRST)
    replicates = sorted(m.replicate for m in result.metrics
                        if m.design_id == design.design_id and m.name == "genie3_iptm")
    assert replicates == [0, 1, 2, 3, 4]
    assert len({metric.metric_id for metric in result.metrics}) == len(result.metrics)


# --- produced, passed, and attempted ---------------------------------------


def test_producing_and_passing_are_different_facts(run) -> None:
    write_genie3_task(run.directory, 0, successes=("dio3_cut_1-resample_0",), status={})
    result = collected(run)

    assert result.run.n_produced == 2
    assert result.run.n_passed == 1
    verdicts = {
        decision.design_id: decision.passed
        for decision in result.decisions
        if decision.name == FILTER_NAME
    }
    by_id = {design.design_id: design.native_id for design in result.designs}
    assert {by_id[design_id]: passed for design_id, passed in verdicts.items()} == {
        FIRST: False, SECOND: True,
    }
    assert all(d.kind == DecisionKind.FILTER for d in result.decisions)


def test_a_run_where_nothing_passed_is_still_a_successful_run(run) -> None:
    """The benchmark produced 40 designs and passed none of them.

    The reducer ran and wrote a header-only table, so every design has a
    verdict and every verdict is False.
    """
    write_genie3_task(run.directory, 0, status={})
    result = collected(run)

    assert result.run.status == RunStatus.SUCCEEDED
    assert result.run.n_passed == 0
    assert result.run.count_details["tasks"]["0000"]["reducer"] == "complete"
    verdicts = [d.passed for d in result.decisions if d.name == FILTER_NAME]
    assert verdicts == [False, False]


def test_a_reducer_that_never_ran_is_not_everything_failing(run) -> None:
    """A missing success table is an unfinished evaluation, not zero hits.

    Genie 3 writes `success_info.csv` whether or not anything passed, so its
    absence means the reduce step did not happen. Recording that as `passed:
    False` for every design would put a measurement in the database that
    nothing measured.
    """
    write_genie3_task(run.directory, 0, reducer=False, status={})
    result = collected(run)

    assert result.run.n_produced == 2
    assert result.run.n_passed is None
    assert result.run.status == RunStatus.PARTIAL
    assert result.run.count_details["tasks"]["0000"]["reducer"] == "missing"
    assert result.run.count_details["tasks"]["0000"]["n_passed"] is None
    assert not [d for d in result.decisions if d.name == FILTER_NAME]


def test_no_rank_decision_is_invented(run) -> None:
    """`rank` orders one design's folds; Genie 3 never ranks designs."""
    write_genie3_task(run.directory, 0, status={})

    assert not [d for d in collected(run).decisions if d.kind == DecisionKind.RANK]


def test_backbones_without_designs_are_counted_separately(run) -> None:
    """Generation that worked and evaluation that did not, in one number."""
    header = (GENIE3_FIXTURE / "info.csv").read_text().splitlines()[0]
    write_genie3_task(run.directory, 0, results=header + "\n",
                      status={"status": "failed", "exit_code": 1})
    result = collected(run)

    task = result.run.count_details["tasks"]["0000"]
    assert result.run.n_produced == 0
    assert result.run.status == RunStatus.FAILED
    assert task["n_backbones"] == 2
    assert task["n_produced"] == 0


def test_fewer_designs_than_asked_for_is_partial(run) -> None:
    write_genie3_task(run.directory, 0, designs=1, status={})
    result = collected(run)

    assert result.run.n_produced == 1
    assert result.run.n_requested == 2
    assert result.run.status == RunStatus.PARTIAL


def test_a_task_that_never_ran_does_not_cost_the_one_that_did(two_task_run) -> None:
    write_genie3_task(two_task_run.directory, 0, status={})
    result = collected(two_task_run)

    assert result.run.n_produced == 2
    assert result.run.status == RunStatus.PARTIAL
    assert result.run.count_details["tasks"]["0001"]["status"] == "missing"


# --- malformed output ------------------------------------------------------


def test_a_truncated_table_keeps_the_designs_it_completed(run) -> None:
    text = (GENIE3_FIXTURE / "info.csv").read_text()
    write_genie3_task(run.directory, 0, results=text[: int(len(text) * 0.7)], status={})
    result = collected(run)

    assert result.run.n_produced >= 1
    assert all(design.sequence for design in result.designs)


def test_a_row_with_no_name_is_skipped_and_counted(run) -> None:
    rows = (GENIE3_FIXTURE / "info.csv").read_text().splitlines()
    header, body = rows[0], rows[1:]
    orphan = "," + body[0].split(",", 1)[1]
    write_genie3_task(run.directory, 0, results="\n".join([header, orphan, *body]) + "\n",
                      status={})
    result = collected(run)

    assert result.run.n_produced == 2
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_a_sequence_with_an_unknown_residue_is_not_a_design(run) -> None:
    """X is a legal letter and an unorderable residue."""
    rows = (GENIE3_FIXTURE / "info.csv").read_text().splitlines()
    header, body = rows[0], rows[1:]
    spoiled = []
    for line in body:
        fields = line.split(",")
        if fields[0].endswith("_0-resample_0"):
            fields[17] = "X" + fields[17][1:]
        spoiled.append(",".join(fields))
    write_genie3_task(run.directory, 0, results="\n".join([header, *spoiled]) + "\n",
                      status={})
    result = collected(run)

    assert [design.native_id for design in result.designs] == [SECOND]
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_a_missing_results_table_is_not_a_crash(run) -> None:
    write_genie3_task(run.directory, 0, status={})
    (run.directory / run.tasks[0].designs).unlink()
    result = collected(run)

    assert result.run.n_produced == 0
    assert result.run.status == RunStatus.FAILED


def test_a_run_directory_for_another_run_is_refused(run, tmp_path: Path) -> None:
    write_genie3_task(run.directory, 0, status={})
    other = run.to_run_record().model_copy(update={"run_id": "not-this-run"})

    with pytest.raises(CollectionError, match="does not match manifest"):
        Genie3OutputAdapter().collect(run.directory, other)


# --- two tasks -------------------------------------------------------------


def test_two_tasks_do_not_collect_as_duplicates_of_each_other(two_task_run) -> None:
    """Genie 3 numbers designs per output root, and each task has its own."""
    for task_id in (0, 1):
        write_genie3_task(two_task_run.directory, task_id, status={})
    result = collected(two_task_run)

    assert result.run.n_produced == 4
    assert len({design.native_id for design in result.designs}) == 4
    assert len({design.design_id for design in result.designs}) == 4
    assert sorted(design.native_id for design in result.designs) == [
        "task-0000-dio3_cut_0-resample_0", "task-0000-dio3_cut_1-resample_0",
        "task-0001-dio3_cut_0-resample_0", "task-0001-dio3_cut_1-resample_0",
    ]


def test_each_task_keeps_its_own_structures(two_task_run) -> None:
    for task_id in (0, 1):
        write_genie3_task(two_task_run.directory, task_id, status={})
    result = collected(two_task_run)

    complexes = {a.design_id: a.uri for a in result.artifacts if a.kind == "design_complex"}
    assert len(complexes) == 4
    by_id = {design.design_id: design.native_id for design in result.designs}
    for design_id, uri in complexes.items():
        assert uri.startswith(f"tasks/{by_id[design_id][5:9]}/")


# --- artifacts and determinism ---------------------------------------------


def test_a_shared_backbone_is_not_owned_by_one_of_its_designs(run) -> None:
    """`num_seq` designs come off one backbone and share its PDB and FASTA.

    `artifacts` is unique per (run, uri), so attaching those to a design would
    make the winning `design_id` depend on collection order and would claim an
    ownership that does not exist.
    """
    write_genie3_task(run.directory, 0, status={})
    result = collected(run)

    shared = [a for a in result.artifacts
              if a.kind in {"design_backbone", "designed_sequences"}]
    assert shared, "the backbones and their sequences are still collected"
    assert all(artifact.design_id is None for artifact in shared)
    # The refolded complex is per design, so that one keeps its owner.
    complexes = [a for a in result.artifacts if a.kind == "design_complex"]
    assert len(complexes) == 2
    assert all(artifact.design_id is not None for artifact in complexes)


def test_a_binder_longer_than_the_problem_set_allows_is_not_a_design(run) -> None:
    """The problem set decides the length range; no harness field does."""
    rows = (GENIE3_FIXTURE / "info.csv").read_text().splitlines()
    header, body = rows[0], rows[1:]
    stretched = []
    for line in body:
        fields = line.split(",")
        if fields[0].endswith("_0-resample_0"):
            fields[17] = "A" * 500
        stretched.append(",".join(fields))
    write_genie3_task(run.directory, 0, results="\n".join([header, *stretched]) + "\n",
                      status={})
    result = collected(run)

    assert [design.native_id for design in result.designs] == [SECOND]
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_folds_that_disagree_on_the_sequence_are_not_one_design(run) -> None:
    """Five folds of one design are five predictions of one sequence."""
    rows = (GENIE3_FIXTURE / "info.csv").read_text().splitlines()
    header, body = rows[0], rows[1:]
    spoiled = list(body)
    fields = spoiled[1].split(",")
    fields[17] = fields[17][:-1] + ("A" if fields[17][-1] != "A" else "C")
    spoiled[1] = ",".join(fields)
    write_genie3_task(run.directory, 0, results="\n".join([header, *spoiled]) + "\n",
                      status={})
    result = collected(run)

    assert [design.native_id for design in result.designs] == [SECOND]
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_a_fractional_rank_is_not_rounded_into_a_replicate(run) -> None:
    """`1.5` is a parse error; silently reading it as rank 1 collides."""
    rows = (GENIE3_FIXTURE / "info.csv").read_text().splitlines()
    header, body = rows[0], rows[1:]
    fields = body[0].split(",")
    fields[6] = "1.5"
    fixed = [",".join(fields), *body[1:]]
    write_genie3_task(run.directory, 0, results="\n".join([header, *fixed]) + "\n",
                      status={})
    result = collected(run)

    design = next(d for d in result.designs if d.native_id == FIRST)
    replicates = sorted(m.replicate for m in result.metrics
                        if m.design_id == design.design_id and m.name == "genie3_iptm")
    # One unusable rank makes the whole group fall back to file position,
    # which is the only remaining deterministic answer.
    assert replicates == [0, 1, 2, 3, 4]
    assert len({metric.metric_id for metric in result.metrics}) == len(result.metrics)


def test_what_the_task_actually_ran_is_collected(run) -> None:
    """The rendered config carries the seed, the root, and the sample count."""
    write_genie3_task(run.directory, 0, status={})
    kinds = {artifact.kind for artifact in collected(run).artifacts}

    assert {"rendered_config", "experiment_config", "driver_script",
            "native_design_table", "design_complex", "design_backbone",
            "designed_sequences", "task_status", "log"} <= kinds


def test_structure_paths_are_rebuilt_rather_than_read_from_the_table(
    run, tmp_path: Path
) -> None:
    """`design_filepath` is absolute and belongs to the machine that wrote it."""
    write_genie3_task(run.directory, 0, status={})
    result = collected(run)

    with (run.directory / run.tasks[0].designs).open(newline="") as handle:
        recorded = next(csv.DictReader(handle))["design_filepath"]
    assert not Path(recorded).is_relative_to(run.directory)

    complexes = [a.uri for a in result.artifacts if a.kind == "design_complex"]
    for uri in complexes:
        assert not Path(uri).is_absolute()
        assert (run.directory / uri).is_file()


def test_recollecting_an_unchanged_directory_is_identical(run) -> None:
    write_genie3_task(run.directory, 0, successes=("dio3_cut_1-resample_0",), status={})

    assert collected(run).content_hash() == collected(run).content_hash()


def test_succeeded_needs_every_task_to_have_produced_something(two_task_run) -> None:
    write_genie3_task(two_task_run.directory, 0, status={})
    adapter = Genie3OutputAdapter()

    assert not adapter.succeeded(two_task_run.directory)
    write_genie3_task(two_task_run.directory, 1, status={})
    assert adapter.succeeded(two_task_run.directory)
