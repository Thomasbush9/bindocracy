"""Reading a BindCraft 2 run directory, against real output.

The fixture is unmodified rows from the 2026-09-22 DIO3 run: one trajectory that
ran the whole way through and produced six scored candidates, three of which
passed every filter and one of which became the accepted design, plus two
trajectories terminated at different stages.

That shape is the point. Three counts differ here and a reader has to be able to
tell them apart: 3 trajectories attempted, 6 candidates produced, 3 passing the
filters, 1 delivered. A parser that conflates any two of them reports a number
nobody asked for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import write_bindcraft2_configs, write_bindcraft2_task

from bindocracy.adapters.base import CollectionError
from bindocracy.store.records import DecisionKind
from bindocracy.tools import collect_run, load_configs, plan
from bindocracy.tools.bindcraft2.adapter import BindCraft2OutputAdapter

SUCCEEDED = {"task_id": 0, "status": "succeeded", "n_produced": 6}


def collected(
    root: Path, *, tasks: int = 1, sampling: dict | None = None,
    status: dict | None = SUCCEEDED, **task_kwargs,
):
    general, model = write_bindcraft2_configs(
        root, sampling={"jobs": tasks, **(sampling or {})},
    )
    manifest = plan(load_configs(general, model), root / "run")
    for task_id in range(tasks):
        write_bindcraft2_task(
            manifest.directory,
            task_id,
            status={**status, "task_id": task_id} if status is not None else None,
            **task_kwargs,
        )
    return manifest, collect_run(manifest.directory / "run.json")


# --- the three counts ------------------------------------------------------


def test_a_design_is_a_candidate_not_a_trajectory(tmp_path: Path) -> None:
    _, run = collected(tmp_path)

    assert run.run.n_produced == 6
    assert run.run.n_attempted == 6
    assert len(run.designs) == 6


def test_passing_and_being_delivered_are_different_numbers(tmp_path: Path) -> None:
    """Three candidates cleared the filters; one became the accepted design."""
    _, run = collected(tmp_path)

    assert run.run.n_passed == 3
    assert run.run.count_details["accepted"] == 1


def test_the_trajectory_census_says_why_the_task_stopped(tmp_path: Path) -> None:
    _, run = collected(tmp_path)
    task = run.run.count_details["tasks"]["0000"]

    assert task["trajectories"]["attempted"] == 3
    assert task["trajectories"]["terminated"] == {"completed": 1, "screen": 1, "anneal": 1}
    assert task["trajectories"]["budget"] == 6


def test_missing_trajectory_budget_does_not_establish_exhaustion(tmp_path: Path) -> None:
    manifest, _ = collected(tmp_path)
    manifest_path = manifest.directory / "run.json"
    document = json.loads(manifest_path.read_text())
    document["tasks"][0].pop("n_generated")
    manifest_path.write_text(json.dumps(document))

    run = collect_run(manifest_path)
    task = run.run.count_details["tasks"]["0000"]

    assert task["trajectories"]["budget"] is None
    assert task["budget_exhausted"] is False
    assert len(run.designs) == 6


def test_a_successful_exit_with_shortfall_does_not_prove_exhaustion(tmp_path: Path) -> None:
    """The fixture records three attempts, not the full six-attempt budget."""
    _, run = collected(tmp_path, status={**SUCCEEDED, "exit_code": 0})
    task = run.run.count_details["tasks"]["0000"]

    assert task["requested"] == 2
    assert task["surplus"] == -1
    assert task["budget_exhausted"] is False
    assert task["shape_problems"] == []


@pytest.mark.parametrize("budget", [2, 3])
def test_recorded_trajectories_exhaust_the_budget_with_shortfall(
    tmp_path: Path, budget: int,
) -> None:
    _, run = collected(tmp_path, sampling={"max_trajectories": budget})
    task = run.run.count_details["tasks"]["0000"]

    assert task["trajectories"]["attempted"] == 3
    assert task["trajectories"]["budget"] == budget
    assert task["surplus"] == -1
    assert task["budget_exhausted"] is True
    assert task["shape_problems"] == []
    assert len(run.designs) == 6


@pytest.mark.parametrize("budget", [3, 6])
def test_sufficient_accepted_outputs_are_not_budget_exhaustion(
    tmp_path: Path, budget: int,
) -> None:
    _, run = collected(
        tmp_path, sampling={"designs_per_job": 1, "max_trajectories": budget},
    )
    task = run.run.count_details["tasks"]["0000"]

    assert task["accepted"] == task["requested"] == 1
    assert task["budget_exhausted"] is False
    assert run.run.status == "succeeded"


@pytest.mark.parametrize("empty_tables", [False, True])
def test_configuration_crash_before_any_attempt_does_not_exhaust_budget(
    tmp_path: Path, empty_tables: bool,
) -> None:
    general, model = write_bindcraft2_configs(
        tmp_path, sampling={"max_trajectories": 60},
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")
    task_plan = manifest.tasks[0]
    task_dir = manifest.directory / task_plan.directory
    task_dir.mkdir(parents=True, exist_ok=True)
    (manifest.directory / task_plan.status).write_text(json.dumps({
        "task_id": 0, "status": "failed", "exit_code": 1,
        "error": "ValueError: invalid campaign configuration",
    }))
    if empty_tables:
        campaign = task_dir / "campaign"
        trajectories = campaign / "1_Trajectories"
        trajectories.mkdir(parents=True)
        (trajectories / "!_Trajectories.csv").write_text("trajectory,design,terminated\n")
        (campaign / ".campaign_state.json").write_text(json.dumps({
            "trajectories": 0, "accepted": 0, "attempted": [],
        }))
        (campaign / "summary.csv").write_text(
            "campaign,scope,metric,samples,mean,std,min,max\n"
            "fixture,campaign,trajectories,1,0,0,0,0\n"
        )

    run = collect_run(manifest.directory / "run.json")
    task = run.run.count_details["tasks"]["0000"]

    assert task["status"] == "failed"
    assert task["trajectories"]["attempted"] == 0
    assert task["trajectories"]["budget"] == 60
    assert task["budget_exhausted"] is False
    assert run.run.status == "failed"
    assert run.run.n_produced == 0
    assert not run.designs


@pytest.mark.parametrize("status", [
    {"task_id": 0, "status": "failed", "exit_code": 1},
    {"task_id": 0, "status": "partial", "exit_code": 143},
    None,
])
@pytest.mark.parametrize("budget", [3, 6])
def test_interrupted_collection_keeps_outputs_and_budget_evidence(
    tmp_path: Path, status: dict | None, budget: int,
) -> None:
    _, run = collected(
        tmp_path, status=status, sampling={"max_trajectories": budget},
    )
    task = run.run.count_details["tasks"]["0000"]

    assert task["status"] == (status["status"] if status else "missing")
    assert task["budget_exhausted"] is (budget == 3)
    assert task["accepted"] == 1
    assert len(run.designs) == run.run.n_produced == 6
    assert run.run.status == "partial"


def test_claimed_but_unrecorded_trajectories_do_not_prove_exhaustion(
    tmp_path: Path,
) -> None:
    manifest, _ = collected(
        tmp_path, status={"task_id": 0, "status": "failed", "exit_code": 1},
    )
    state_path = manifest.directory / "tasks/0000/campaign/.campaign_state.json"
    state = json.loads(state_path.read_text())
    # Upstream reserves the budget slot before the trajectory runs.
    state["trajectories"] = 6
    state_path.write_text(json.dumps(state))

    run = collect_run(manifest.directory / "run.json")
    task = run.run.count_details["tasks"]["0000"]

    assert task["trajectories"]["attempted"] == 3
    assert task["budget_exhausted"] is False
    assert len(run.designs) == 6
    assert run.run.status == "partial"


# --- the join the tables do not make -------------------------------------


def test_the_accepted_design_is_matched_to_its_candidate_by_sequence(
    tmp_path: Path,
) -> None:
    """`_seq0` and `_candidate4` are the same design under two names."""
    _, run = collected(tmp_path)

    selections = [
        decision for decision in run.decisions if decision.kind == DecisionKind.SELECTION
    ]
    assert len(selections) == 1
    selected = next(
        design for design in run.designs if design.design_id == selections[0].design_id
    )
    assert selected.native_id.endswith("_candidate4")
    assert selections[0].reason["accepted_as"].endswith("_seq0")


def test_the_delivered_design_is_one_the_filters_passed(tmp_path: Path) -> None:
    _, run = collected(tmp_path)

    selected = {
        decision.design_id
        for decision in run.decisions
        if decision.kind == DecisionKind.SELECTION
    }
    verdicts = {
        decision.design_id: decision.passed
        for decision in run.decisions
        if decision.kind == DecisionKind.FILTER and decision.name == "bindcraft2_filters"
    }
    assert all(verdicts[design_id] for design_id in selected)


def test_an_accepted_design_matching_no_candidate_is_reported(tmp_path: Path) -> None:
    """A stale ranked table naming a design the scored table does not hold."""
    fixture = Path("tests/fixtures/bindcraft2/ranked.csv").read_text().splitlines()
    tampered = fixture[0] + "\n" + fixture[1].replace("c7f74f2f18ee1581", "deadbeefdeadbeef")
    _, run = collected(tmp_path, ranked_rows=tampered + "\n")

    problems = run.run.count_details["tasks"]["0000"]["shape_problems"]
    assert any("match no scored candidate" in problem for problem in problems)
    assert run.run.n_passed is None


def test_a_rank_is_scoped_to_its_own_task(tmp_path: Path) -> None:
    """Each task is its own campaign, so a rank means nothing across tasks."""
    _, run = collected(tmp_path, tasks=2)

    ranks = [decision for decision in run.decisions if decision.kind == DecisionKind.RANK]
    assert len(ranks) == 2
    assert len({decision.scope_id for decision in ranks}) == 2
    assert {decision.rank for decision in ranks} == {1}


# --- metrics and the filter verdict ---------------------------------------


def test_every_metric_carries_a_direction_and_a_real_value(tmp_path: Path) -> None:
    _, run = collected(tmp_path)
    by_name = {metric.name: metric for metric in run.metrics}

    assert by_name["i_pTM"].direction == "max"
    # The campaign's ranking metric is normalized error, so lower is better.
    assert by_name["i_pDAE"].direction == "min"
    assert by_name["Off_Epitope_Contact_Fraction"].direction == "min"
    assert all(metric.value is not None for metric in run.metrics)


def test_a_rejection_names_the_threshold_that_rejected_it(tmp_path: Path) -> None:
    """"Which filter is costing this campaign designs" has to be a query."""
    _, run = collected(tmp_path)

    named = [
        decision.name
        for decision in run.decisions
        if decision.kind == DecisionKind.FILTER and decision.passed is False
        and decision.name != "bindcraft2_filters"
    ]
    assert named == ["Unbound_Binder_pLDDT"] * 3


def test_the_campaign_state_is_kept_beside_the_tables(tmp_path: Path) -> None:
    _, run = collected(tmp_path)
    task = run.run.count_details["tasks"]["0000"]

    assert task["candidates_scored"] == 6
    assert task["failed_filters"] == {"Unbound_Binder_pLDDT": 3}


def test_tables_disagreeing_with_the_state_file_is_a_shape_problem(
    tmp_path: Path,
) -> None:
    """The state file is written independently, so it is worth checking against."""
    rows = Path("tests/fixtures/bindcraft2/refolded.csv").read_text().splitlines()
    _, run = collected(tmp_path, candidate_rows="\n".join(rows[:4]) + "\n")

    problems = run.run.count_details["tasks"]["0000"]["shape_problems"]
    assert any("scored candidates" in problem for problem in problems)


# --- determinism, artifacts, and the empty case ---------------------------


def test_re_collecting_an_unchanged_directory_is_identical(tmp_path: Path) -> None:
    """Re-ingestion must not look like a conflicting rewrite."""
    manifest, first = collected(tmp_path)
    second = collect_run(manifest.directory / "run.json")

    assert first.content_hash() == second.content_hash()


def test_native_ids_are_unique_across_tasks(tmp_path: Path) -> None:
    """Two tasks share a campaign name and a binder length, so names collide."""
    _, run = collected(tmp_path, tasks=2)

    assert len(run.designs) == 12
    assert len({design.native_id for design in run.designs}) == 12


def test_each_candidate_keeps_its_structures(tmp_path: Path) -> None:
    _, run = collected(tmp_path)

    kinds = [artifact.kind for artifact in run.artifacts]
    assert kinds.count("complex_structure") == 6
    assert kinds.count("binder_structure") == 6
    assert kinds.count("accepted_structure") == 1
    assert "campaign_state" in kinds


def test_a_task_that_accepted_nothing_still_collects(tmp_path: Path) -> None:
    """No ranked table must not prevent collecting scored candidates."""
    _, run = collected(tmp_path, ranked=False)

    assert run.run.n_produced == 6
    assert run.run.count_details["accepted"] == 0
    assert not [d for d in run.decisions if d.kind == DecisionKind.RANK]


def test_a_malformed_row_costs_only_itself(tmp_path: Path) -> None:
    rows = Path("tests/fixtures/bindcraft2/refolded.csv").read_text().splitlines()
    broken = [rows[0], ",,,", *rows[1:]]
    _, run = collected(tmp_path, candidate_rows="\n".join(broken) + "\n")

    assert len(run.designs) == 6
    problems = run.run.count_details["tasks"]["0000"]["shape_problems"]
    assert any("unreadable" in problem for problem in problems)


def test_a_run_record_from_another_run_is_refused(tmp_path: Path) -> None:
    """Collecting one run's directory as another run would merge two campaigns.

    Unreachable through `collect_run`, which derives the record from the
    manifest it just read. The guard is for a caller that supplies its own, so
    the adapter is exercised directly here -- routing it through `collect_run`
    would assert nothing, which is what the first draft of this test did.
    """
    general, model = write_bindcraft2_configs(tmp_path)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    write_bindcraft2_task(manifest.directory, 0, status=SUCCEEDED)
    foreign = manifest.to_run_record().model_copy(
        update={"run_id": "00000000-0000-0000-0000-000000000000"}
    )

    with pytest.raises(CollectionError, match="does not match manifest"):
        BindCraft2OutputAdapter().collect(manifest.directory, foreign)
