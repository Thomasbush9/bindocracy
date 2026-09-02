"""Reading a FreeBindCraft run directory, against real benchmark output.

The fixture is two trajectories of the 2026-08-26 run: seven fully scored
designs, two more that the base AF2 filters dropped before scoring, two
accepted. That shape is the whole point -- the accepted set is a *derived*
fact, the difference between two tables, and three independent files have to
agree about it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import (
    write_freebindcraft_configs,
    write_freebindcraft_task,
)

from bindocracy.adapters.base import CollectionError
from bindocracy.store.records import CandidateType, DecisionKind, RunStatus
from bindocracy.tools import collect_run, load_configs, plan
from bindocracy.tools.freebindcraft.adapter import FreeBindCraftOutputAdapter

ACCEPTED = "dio3_cut_l85_s877586_mpnn4"
REJECTED = "dio3_cut_l85_s877586_mpnn3"
# In `rejected_mpnn_full_stats.csv` and nowhere else: predicted, then dropped
# by the base AF2 filters before a single interface metric was computed.
UNSCORED = "dio3_cut_l85_s877586_mpnn1"


@pytest.fixture
def run(freebindcraft_configs, tmp_path: Path):
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, status={})
    return manifest


def collected(manifest):
    return collect_run(manifest.directory / "run.json")


def design(bundle, native: str):
    return next(d for d in bundle.designs if d.native_id.endswith(native))


def decision(bundle, design_id: str, kind: DecisionKind):
    return next(
        d for d in bundle.decisions if d.design_id == design_id and d.kind == kind
    )


# --- the counts -------------------------------------------------------------


def test_every_mpnn_sequence_is_a_produced_design(run) -> None:
    """Nine were made; two of them were never measured. Both are produced."""
    bundle = collected(run)

    assert bundle.run.n_produced == 9
    assert bundle.run.n_attempted == 9
    assert bundle.run.n_passed == 2
    assert bundle.run.status == RunStatus.SUCCEEDED


def test_an_unscored_design_is_a_sequence_and_not_a_complex(run) -> None:
    """BindCraft keeps no structure for one the base filters dropped."""
    bundle = collected(run)

    assert design(bundle, UNSCORED).candidate_type == CandidateType.SEQUENCE
    assert design(bundle, ACCEPTED).candidate_type == CandidateType.COMPLEX
    assert not [m for m in bundle.metrics if m.design_id == design(bundle, UNSCORED).design_id]


def test_the_two_kinds_of_rejection_are_distinguished(run) -> None:
    bundle = collected(run)

    early = decision(bundle, design(bundle, UNSCORED).design_id, DecisionKind.FILTER)
    late = decision(bundle, design(bundle, REJECTED).design_id, DecisionKind.FILTER)

    assert early.passed is False and early.reason["stage"] == "af2_base"
    assert late.passed is False and late.reason["stage"] == "interface_filters"
    assert late.reason["failed"]


def test_the_trajectory_census_says_why_a_task_stopped(run) -> None:
    """Only the relaxed ones count towards the budget; the others still cost."""
    task = collected(run).run.count_details["tasks"]["0000"]

    assert task["trajectories"] == {
        "attempted": 4,
        "successful": 2,
        "clashing": 1,
        "low_confidence": 1,
        "budget": 4,
    }
    assert task["n_scored"] == 7
    assert task["n_rejected_before_scoring"] == 2


# --- what the run's own verdict means ---------------------------------------


def test_the_eight_constants_are_not_recorded_as_measurements(run) -> None:
    """Without PyRosetta, dG is -10.0 on every design by construction."""
    bundle = collected(run)
    names = {metric.name for metric in bundle.metrics}

    assert "freebindcraft_dG" not in names
    assert "freebindcraft_PackStat" not in names
    assert "freebindcraft_n_InterfaceHbonds" not in names
    assert "freebindcraft_i_pTM" in names
    assert "freebindcraft_ShapeComplementarity" in names


def test_the_run_names_the_filters_that_could_not_bite(run) -> None:
    details = collected(run).run.count_details

    assert details["pyrosetta"] is False
    assert details["inert_filters"] == ["dG"]


def test_per_model_scores_are_replicates_of_one_metric(run) -> None:
    """A multimer design run predicts with two of the five models."""
    bundle = collected(run)
    accepted = design(bundle, ACCEPTED)
    iptm = sorted(
        metric.replicate
        for metric in bundle.metrics
        if metric.design_id == accepted.design_id and metric.name == "freebindcraft_i_pTM"
    )

    # 0 is the average; 1 and 2 are the models that ran. 3-5 are empty columns.
    assert iptm == [0, 1, 2]


def test_a_design_carries_the_backbone_it_redesigns(run) -> None:
    bundle = collected(run)

    assert design(bundle, ACCEPTED).metadata["trajectory"] == "dio3_cut_l85_s877586"
    assert design(bundle, ACCEPTED).seed == 877586


# --- the ranked table -------------------------------------------------------


def test_ranks_are_scoped_to_the_task_that_produced_them(run) -> None:
    bundle = collected(run)
    rank = decision(bundle, design(bundle, ACCEPTED).design_id, DecisionKind.RANK)

    assert rank.rank == 1
    assert rank.scope_id is not None


def test_a_rejected_design_is_never_ranked(run) -> None:
    bundle = collected(run)
    ranked = {d.design_id for d in bundle.decisions if d.kind == DecisionKind.RANK}

    assert design(bundle, REJECTED).design_id not in ranked
    assert len(ranked) == 2


def test_a_task_that_ran_out_of_trajectories_still_reports_what_it_accepted(
    tmp_path: Path,
) -> None:
    """BindCraft fills the ranks in only in the check that ends the loop.

    It appends a row for each design as it accepts it, with `Rank` empty, and
    rewrites the file with ranks only once it has enough designs. So a task
    that stops on its trajectory budget leaves a table that names every
    accepted design and ranks none of them -- measured on run 19, where both
    tasks did exactly this. The names are still evidence, and n_passed still
    comes from the tables.
    """
    general, model = write_freebindcraft_configs(
        tmp_path, sampling={"designs_per_job": 4, "max_trajectories": 2}
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, ranked=False, status={})
    bundle = collected(manifest)

    task = bundle.run.count_details["tasks"]["0000"]

    assert bundle.run.n_passed == 2
    assert task["ranked"] is False
    # The rows are there; only the ranks are not, and they still have to name
    # the designs the other two tables say were accepted.
    assert task["n_final_rows"] == 2
    assert task["n_final_unranked"] == 2
    assert task["shape_problems"] == []
    assert not [d for d in bundle.decisions if d.kind == DecisionKind.RANK]


# --- the tables have to agree ------------------------------------------------


def test_structures_that_disagree_with_the_tables_are_a_shape_problem(
    freebindcraft_configs, tmp_path: Path
) -> None:
    """`Accepted/` is the file BindCraft's own stopping check counts."""
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, status={})
    accepted = manifest.directory / "tasks/0000/bindcraft/Accepted"
    next(accepted.glob(f"{ACCEPTED}_model*.pdb")).unlink()
    bundle = collected(manifest)

    problems = bundle.run.count_details["tasks"]["0000"]["shape_problems"]
    assert any("Accepted/" in problem for problem in problems)
    # Counted, but not measured: an incomplete task cannot report hits.
    assert bundle.run.n_passed is None
    assert bundle.run.status == RunStatus.PARTIAL


def test_a_run_that_accepted_enough_but_ranked_nothing_is_incomplete(
    freebindcraft_configs, tmp_path: Path
) -> None:
    """Two accepted designs is the stopping condition, so the table is due."""
    general, model = write_freebindcraft_configs(
        tmp_path, sampling={"designs_per_job": 2, "max_trajectories": 40}
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, ranked=False, status={})

    problems = collected(manifest).run.count_details["tasks"]["0000"]["shape_problems"]
    assert any("nothing was ranked" in problem for problem in problems)


def test_a_task_over_its_trajectory_budget_is_a_shape_problem(
    tmp_path: Path,
) -> None:
    general, model = write_freebindcraft_configs(
        tmp_path, sampling={"designs_per_job": 2, "max_trajectories": 1}
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, status={})

    problems = collected(manifest).run.count_details["tasks"]["0000"]["shape_problems"]
    assert any("budget was 1" in problem for problem in problems)


# --- rows that did not come from this run ------------------------------------


def test_a_row_stamped_with_another_runs_settings_is_dropped(
    freebindcraft_configs, tmp_path: Path
) -> None:
    """BindCraft stamps the stem of each settings file onto every row.

    A design path reused across configurations would otherwise collect a
    previous run's designs as this run's, and every count would look normal.
    """
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, status={})
    table = manifest.directory / "tasks/0000/bindcraft/mpnn_design_stats.csv"
    table.write_text(table.read_text().replace("relaxed_filters", "no_filters"))
    bundle = collected(manifest)
    task = bundle.run.count_details["tasks"]["0000"]

    assert task["n_foreign"] == 7
    assert task["n_scored"] == 0
    # Nothing is reported as having passed, and the structures BindCraft kept
    # say plainly that the table did not describe this run.
    assert bundle.run.n_passed is None
    assert task["shape_problems"]


def test_a_row_reporting_the_wrong_epitope_is_dropped(
    tmp_path: Path,
) -> None:
    """The only place a task launched against the wrong epitope would show.

    The fixture rows carry an empty `Target_Hotspot`, so a run planned with an
    epitope must not collect them as its own.
    """
    general, model = write_freebindcraft_configs(tmp_path, hotspots=["A56", "A57"])
    manifest = plan(load_configs(general, model), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, status={})

    assert collected(manifest).run.count_details["tasks"]["0000"]["n_foreign"] == 7


def test_a_truncated_table_costs_only_its_broken_row(
    freebindcraft_configs, tmp_path: Path
) -> None:
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, status={})
    table = manifest.directory / "tasks/0000/bindcraft/mpnn_design_stats.csv"
    lines = table.read_text().splitlines()
    table.write_text("\n".join([*lines, lines[-1].split(",", 3)[0] + ",,,"]) + "\n")

    assert collected(manifest).run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_two_tasks_do_not_collect_as_duplicates_of_each_other(
    tmp_path: Path,
) -> None:
    """Design names are per trajectory, and trajectory seeds are random."""
    general, model = write_freebindcraft_configs(tmp_path, sampling={"jobs": 2})
    manifest = plan(load_configs(general, model), tmp_path / "run")
    for task_id in (0, 1):
        write_freebindcraft_task(manifest.directory, task_id, status={})
    bundle = collected(manifest)

    assert bundle.run.n_produced == 18
    assert len({d.native_id for d in bundle.designs}) == 18
    assert len({d.design_id for d in bundle.designs}) == 18


# --- collection is a pure function of the directory --------------------------


def test_re_collecting_an_unchanged_run_is_identical(run) -> None:
    assert collected(run).content_hash() == collected(run).content_hash()


def test_a_manifest_for_another_run_is_refused(run, tmp_path: Path) -> None:
    root = tmp_path / "other"
    root.mkdir()
    other = plan(load_configs(*write_freebindcraft_configs(root)), tmp_path / "b")
    adapter = FreeBindCraftOutputAdapter()

    with pytest.raises(CollectionError, match="does not match manifest"):
        adapter.collect(run.directory, other.to_run_record())


def test_a_task_whose_every_trajectory_died_is_a_failed_run(
    freebindcraft_configs, tmp_path: Path
) -> None:
    """Clashing and low-confidence trajectories reach no MPNN stage at all."""
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, structures=False, status={})
    design_path = manifest.directory / "tasks/0000/bindcraft"
    for name in ("mpnn_design_stats.csv", "rejected_mpnn_full_stats.csv",
                 "final_design_stats.csv", "trajectory_stats.csv"):
        table = design_path / name
        table.write_text(table.read_text().splitlines()[0] + "\n")
    bundle = collected(manifest)

    assert bundle.run.n_produced == 0
    assert bundle.run.n_passed is None
    assert bundle.run.status == RunStatus.FAILED


def test_only_the_scored_table_says_which_designs_were_measured(
    freebindcraft_configs, tmp_path: Path
) -> None:
    """A design in both tables is one scored design, not two candidates."""
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, status={})
    bundle = collected(manifest)
    task = bundle.run.count_details["tasks"]["0000"]

    # Seven scored and seven rejected rows, five of which name the same design.
    assert task["n_scored"] + task["n_rejected_before_scoring"] == len(bundle.designs)
    assert design(bundle, REJECTED).candidate_type == CandidateType.COMPLEX


def test_succeeded_asks_only_whether_every_task_scored_something(
    tmp_path: Path,
) -> None:
    """A task whose every trajectory died leaves nothing to parse."""
    general, model = write_freebindcraft_configs(tmp_path, sampling={"jobs": 2})
    manifest = plan(load_configs(general, model), tmp_path / "run")
    adapter = FreeBindCraftOutputAdapter()

    write_freebindcraft_task(manifest.directory, 0, status={})
    assert adapter.succeeded(manifest.directory) is False

    write_freebindcraft_task(manifest.directory, 1, status={})
    assert adapter.succeeded(manifest.directory) is True


def test_a_final_table_that_names_the_wrong_designs_is_a_shape_problem(
    freebindcraft_configs, tmp_path: Path
) -> None:
    """Three files have to agree about what was accepted, ranked or not.

    The scored and rejected tables partition the designs and `Accepted/` holds
    the structures; the final table is the third witness, and it names each
    design as it is accepted rather than only at the end. A table naming
    something else is a stale file marking the wrong design as passed.
    """
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    write_freebindcraft_task(manifest.directory, 0, status={})
    table = manifest.directory / "tasks/0000/bindcraft/final_design_stats.csv"
    lines = table.read_text().splitlines()
    table.write_text("\n".join(lines[:-1]) + "\n")
    bundle = collected(manifest)

    problems = bundle.run.count_details["tasks"]["0000"]["shape_problems"]
    assert any("the final table names" in problem for problem in problems)
    assert bundle.run.n_passed is None
