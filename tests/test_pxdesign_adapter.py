"""Reading a finished PXDesign run.

The fixture is four unmodified rows of the 2026-08-26 benchmark run, one per
distinct verdict-and-structure combination it produced: a dual-filter hit, two
designs only AF2 liked, and one the table was padded with. That last one is the
point — PXDesign always returns exactly as many rows as it was asked for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import PXDESIGN_FIXTURE, write_pxdesign_configs, write_pxdesign_task

from bindocracy.adapters.base import CollectionError
from bindocracy.store.records import (
    CandidateType,
    DecisionKind,
    DesignStatus,
    RunStatus,
)
from bindocracy.tools import collect_run, load_configs, plan
from bindocracy.tools.pxdesign.adapter import FILTERS, RANK_NAME, PXDesignOutputAdapter

HIT = "task-0000-rank-0001"
PADDED = "task-0000-rank-0017"


@pytest.fixture
def run(tmp_path: Path):
    """A planned four-design, one-task run, ready for a task directory."""
    general, model = write_pxdesign_configs(
        tmp_path,
        sampling={"jobs": 1, "designs_per_job": 17, "seed_base": 100, "preset": "extended"},
    )
    return plan(load_configs(general, model), tmp_path / "run")


@pytest.fixture
def two_task_run(tmp_path: Path):
    general, model = write_pxdesign_configs(
        tmp_path,
        sampling={"jobs": 2, "designs_per_job": 17, "seed_base": 100, "preset": "extended"},
    )
    return plan(load_configs(general, model), tmp_path / "run")


def collected(manifest):
    return collect_run(manifest.directory / "run.json")


# --- produced, passed, and the padding -------------------------------------


def test_a_padded_failure_is_still_a_produced_design(run) -> None:
    """Twenty-four of the benchmark's forty rows failed every filter."""
    write_pxdesign_task(run.directory, 0, status={})
    result = collected(run)

    assert result.run.n_produced == 17
    assert {design.status for design in result.designs} == {DesignStatus.PRODUCED}
    padded = next(d for d in result.designs if d.native_id == PADDED)
    assert padded.metadata["chosen_struct_type"] == "orig"


def test_passing_is_the_dual_filter_and_not_the_row_count(run) -> None:
    write_pxdesign_task(run.directory, 0, status={})
    result = collected(run)

    assert result.run.n_produced == 17
    # Sixteen of the seventeen cleared AF2-IG-easy; five also cleared Protenix.
    # `n_passed` is the second number, not the first and not the row count.
    assert result.run.n_passed == 5
    passing = {
        name: {d.design_id for d in result.decisions if d.name == name and d.passed}
        for name in ("pxdesign_af2ig_easy", "pxdesign_protenix")
    }
    assert len(passing["pxdesign_af2ig_easy"]) == 16
    assert len(passing["pxdesign_protenix"]) == 5
    assert passing["pxdesign_protenix"] <= passing["pxdesign_af2ig_easy"]
    hit = next(d for d in result.designs if d.native_id == HIT)
    assert hit.design_id in passing["pxdesign_protenix"]


def test_all_four_verdicts_are_kept_because_they_disagree(run) -> None:
    """AF2-IG-easy, AF2-IG, Protenix and Protenix-basic are different filters."""
    write_pxdesign_task(run.directory, 0, status={})
    result = collected(run)

    by_name: dict[str, list[bool]] = {}
    for decision in result.decisions:
        if decision.kind == DecisionKind.FILTER:
            by_name.setdefault(decision.name, []).append(bool(decision.passed))

    assert set(by_name) == set(FILTERS.values())
    counts = {name: sum(passed) for name, passed in by_name.items()}
    assert counts == {
        "pxdesign_af2ig_easy": 16,
        "pxdesign_af2ig": 0,
        "pxdesign_protenix": 5,
        "pxdesign_protenix_basic": 6,
    }
    assert result.run.count_details["tasks"]["0000"]["n_by_filter"] == counts


def test_a_run_that_passed_nothing_is_still_a_successful_run(run) -> None:
    """The whole point of a padded table: 40 rows, 0 hits, exit code 0."""
    text = (PXDESIGN_FIXTURE / "summary.csv").read_text().replace("True", "False")
    write_pxdesign_task(run.directory, 0, summary=text, status={})
    result = collected(run)

    assert result.run.status == RunStatus.SUCCEEDED
    assert result.run.n_produced == 17
    assert result.run.n_passed == 0


def drop_protenix_columns() -> str:
    """The fixture table as a run with no Protenix stage would have written it."""
    lines = (PXDESIGN_FIXTURE / "summary.csv").read_text().splitlines()
    header = lines[0].split(",")
    drop = [header.index(column) for column in ("Protenix-success", "Protenix-basic-success")]
    return "\n".join(
        ",".join(field for index, field in enumerate(line.split(",")) if index not in drop)
        for line in lines
    ) + "\n"


def test_a_preview_run_is_judged_on_the_filters_it_ran(tmp_path: Path) -> None:
    """`preview` disables Protenix, so the AF2 verdict is the only one there is."""
    general, model = write_pxdesign_configs(
        tmp_path,
        sampling={"jobs": 1, "designs_per_job": 17, "seed_base": 0, "preset": "preview"},
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")
    write_pxdesign_task(manifest.directory, 0, summary=drop_protenix_columns(), status={})
    result = collected(manifest)

    assert result.run.status == RunStatus.SUCCEEDED
    assert result.run.n_passed == 16
    names = {d.name for d in result.decisions if d.kind == DecisionKind.FILTER}
    assert names == {"pxdesign_af2ig_easy", "pxdesign_af2ig"}
    assert result.run.count_details["tasks"]["0000"]["missing_filters"] == []


def test_an_extended_run_missing_protenix_is_not_a_preview_run(run) -> None:
    """The same table from an `extended` run means its filter stage died.

    Inferring the preset from the columns present would turn an interrupted
    evaluation into a successful AF2-only run, with `n_passed` counted from
    half the filters it was supposed to apply.
    """
    write_pxdesign_task(run.directory, 0, summary=drop_protenix_columns(), status={})
    result = collected(run)

    assert result.run.status == RunStatus.PARTIAL
    assert result.run.n_passed is None
    assert result.run.count_details["tasks"]["0000"]["n_passed"] is None
    assert result.run.count_details["tasks"]["0000"]["missing_filters"] == [
        "pxdesign_protenix", "pxdesign_protenix_basic",
    ]
    # The designs are still produced, and no Protenix verdict is invented.
    assert result.run.n_produced == 17
    names = {d.name for d in result.decisions if d.kind == DecisionKind.FILTER}
    assert names == {"pxdesign_af2ig_easy", "pxdesign_af2ig"}


def test_a_manifest_with_no_preset_cannot_be_collected(run) -> None:
    """It is the only thing that separates the two cases above."""
    manifest_path = run.directory / "run.json"
    document = json.loads(manifest_path.read_text())
    del document["workflow"]["preset"]
    manifest_path.write_text(json.dumps(document))
    write_pxdesign_task(run.directory, 0, status={})

    with pytest.raises(CollectionError, match="preset"):
        collect_run(manifest_path)


def test_fewer_rows_than_asked_for_is_partial(run) -> None:
    """PXDesign pads rather than truncates, so a short table is unfinished work."""
    write_pxdesign_task(run.directory, 0, designs=2, status={})
    result = collected(run)

    assert result.run.n_produced == 2
    assert result.run.n_requested == 17
    assert result.run.status == RunStatus.PARTIAL
    assert result.run.count_details["tasks"]["0000"]["shape_problems"] == [
        "2 rows, expected 17",
    ]


# --- designs, metrics and ranks --------------------------------------------


def test_a_design_is_a_complex_with_the_binder_sequence(run) -> None:
    write_pxdesign_task(run.directory, 0, status={})
    result = collected(run)

    hit = next(d for d in result.designs if d.native_id == HIT)
    assert hit.candidate_type == CandidateType.COMPLEX
    assert hit.length == 80
    assert hit.sequence.startswith("SEKLEKFEEML")


def test_scores_from_both_model_families_are_promoted(run) -> None:
    write_pxdesign_task(run.directory, 0, status={})
    result = collected(run)

    hit = next(d for d in result.designs if d.native_id == HIT)
    scores = {m.name: m.value for m in result.metrics if m.design_id == hit.design_id}
    assert scores["pxdesign_af2_iptm"] == 0.87
    assert scores["pxdesign_ptx_iptm"] == 0.9319
    assert scores["pxdesign_af2_ipAE"] == 7.66


def test_the_rank_is_scoped_to_the_task_that_produced_it(two_task_run) -> None:
    """Two tasks each rank their own pool, so each has a design ranked first."""
    for task_id in (0, 1):
        write_pxdesign_task(two_task_run.directory, task_id, status={})
    result = collected(two_task_run)

    firsts = [d for d in result.decisions if d.name == RANK_NAME and d.rank == 1]
    assert len(firsts) == 2
    assert len({decision.scope_id for decision in firsts}) == 2


def test_two_tasks_do_not_collect_as_duplicates_of_each_other(two_task_run) -> None:
    """PXDesign ranks from 1 in each output directory, and each task has one."""
    for task_id in (0, 1):
        write_pxdesign_task(two_task_run.directory, task_id, status={})
    result = collected(two_task_run)

    assert result.run.n_produced == 34
    assert len({design.native_id for design in result.designs}) == 34
    assert len({design.design_id for design in result.designs}) == 34


# --- malformed output ------------------------------------------------------


def test_a_truncated_table_keeps_the_designs_it_completed(run) -> None:
    text = (PXDESIGN_FIXTURE / "summary.csv").read_text()
    write_pxdesign_task(run.directory, 0, summary=text[: int(len(text) * 0.6)], status={})
    result = collected(run)

    assert result.run.n_produced >= 1
    assert all(design.sequence for design in result.designs)


def test_a_row_with_no_rank_is_skipped_and_counted(run) -> None:
    lines = (PXDESIGN_FIXTURE / "summary.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    orphan = "," + body[0].split(",", 1)[1]
    write_pxdesign_task(run.directory, 0, summary="\n".join([header, orphan, *body]) + "\n",
                        status={})
    result = collected(run)

    assert result.run.n_produced == 17
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_a_repeated_rank_is_not_a_second_design(run) -> None:
    lines = (PXDESIGN_FIXTURE / "summary.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    write_pxdesign_task(run.directory, 0,
                        summary="\n".join([header, *body, body[0]]) + "\n", status={})
    result = collected(run)

    assert result.run.n_produced == 17
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_a_sequence_with_an_unknown_residue_is_not_a_design(run) -> None:
    """X is a legal letter and an unorderable residue."""
    lines = (PXDESIGN_FIXTURE / "summary.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    spoiled = list(body)
    fields = spoiled[0].split(",")
    fields[2] = "X" + fields[2][1:]
    spoiled[0] = ",".join(fields)
    write_pxdesign_task(run.directory, 0, summary="\n".join([header, *spoiled]) + "\n",
                        status={})
    result = collected(run)

    assert result.run.n_produced == 16
    assert HIT not in {design.native_id for design in result.designs}


def test_a_sequence_of_the_wrong_length_is_not_a_design(run) -> None:
    """The spec fixes `binder_length`, so a row that disagrees is not from it."""
    lines = (PXDESIGN_FIXTURE / "summary.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    fields = body[0].split(",")
    fields[2] = fields[2][:-5]
    write_pxdesign_task(run.directory, 0,
                        summary="\n".join([header, ",".join(fields), *body[1:]]) + "\n",
                        status={})
    result = collected(run)

    assert result.run.n_produced == 16
    assert HIT not in {design.native_id for design in result.designs}
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_a_row_from_another_task_name_is_not_a_design(run) -> None:
    """One results directory holds one task's designs, and rows say which."""
    lines = (PXDESIGN_FIXTURE / "summary.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    fields = body[0].split(",")
    fields[1] = "some_other_target"
    write_pxdesign_task(run.directory, 0,
                        summary="\n".join([header, ",".join(fields), *body[1:]]) + "\n",
                        status={})
    result = collected(run)

    assert result.run.n_produced == 16
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_a_fractional_rank_is_a_parse_error_not_a_rank(run) -> None:
    """Truncating `9.5` to 9 would invent a position the tool never assigned.

    The rank is deliberately one no other row holds, so the duplicate check
    cannot be what rejects it.
    """
    lines = (PXDESIGN_FIXTURE / "summary.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    fields = body[1].split(",")
    fields[0] = "19.5"
    write_pxdesign_task(run.directory, 0,
                        summary="\n".join([header, body[0], ",".join(fields), *body[2:]]) + "\n",
                        status={})
    result = collected(run)

    assert result.run.n_produced == 16
    assert result.run.count_details["tasks"]["0000"]["n_invalid"] == 1
    assert "task-0000-rank-0019" not in {d.native_id for d in result.designs}


def test_a_structure_path_climbing_out_of_the_run_is_dropped(run) -> None:
    """`chosen_struct_path` comes from the tool, so it is not trusted."""
    lines = (PXDESIGN_FIXTURE / "summary.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    columns = header.split(",")
    chosen = columns.index("chosen_struct_path")
    fields = body[0].split(",")
    # Four levels up from design_outputs/<task_name>/ is the run root, where
    # run.json really exists -- so only the traversal check can refuse this.
    fields[chosen] = "../../../../run.json"
    write_pxdesign_task(run.directory, 0,
                        summary="\n".join([header, ",".join(fields), *body[1:]]) + "\n",
                        status={})
    result = collected(run)

    # The design is still produced; only the bogus artifact is refused.
    assert result.run.n_produced == 17
    complexes = {a.design_id for a in result.artifacts if a.kind == "design_complex"}
    hit = next(d for d in result.designs if d.native_id == HIT)
    assert hit.design_id not in complexes
    assert all(not Path(a.uri).is_absolute() and ".." not in Path(a.uri).parts
               for a in result.artifacts)


def test_a_missing_table_is_not_a_crash(run) -> None:
    write_pxdesign_task(run.directory, 0, status={})
    (run.directory / run.tasks[0].designs).unlink()
    result = collected(run)

    assert result.run.n_produced == 0
    assert result.run.status == RunStatus.FAILED


def test_a_run_directory_for_another_run_is_refused(run) -> None:
    write_pxdesign_task(run.directory, 0, status={})
    other = run.to_run_record().model_copy(update={"run_id": "not-this-run"})

    with pytest.raises(CollectionError, match="does not match manifest"):
        PXDesignOutputAdapter().collect(run.directory, other)


# --- artifacts and determinism ---------------------------------------------


def test_each_design_keeps_the_structure_its_row_chose(run) -> None:
    """The path names the filter it survived, or the raw design if none did."""
    write_pxdesign_task(run.directory, 0, status={})
    result = collected(run)

    complexes = {a.design_id: a.uri for a in result.artifacts if a.kind == "design_complex"}
    by_id = {d.design_id: d.native_id for d in result.designs}
    chosen = {by_id[design_id]: uri for design_id, uri in complexes.items()}

    assert chosen[HIT].endswith("passing-Protenix-basic/rank_1.cif")
    assert chosen[PADDED].endswith("orig_designed/rank_17.cif")
    for uri in chosen.values():
        assert not Path(uri).is_absolute()
        assert (run.directory / uri).is_file()


def test_what_the_run_actually_configured_is_collected(run) -> None:
    write_pxdesign_task(run.directory, 0, status={})
    kinds = {artifact.kind for artifact in collected(run).artifacts}

    assert {"native_design_table", "design_complex", "design_spec",
            # PXDesign's own record of the mode it ran and the config it
            # resolved, which is not the same as the preset that was asked for.
            "filter_mode", "resolved_config", "task_status", "log"} <= kinds


def test_recollecting_an_unchanged_directory_is_identical(run) -> None:
    write_pxdesign_task(run.directory, 0, status={})

    assert collected(run).content_hash() == collected(run).content_hash()


def test_succeeded_needs_every_task_to_have_produced_something(two_task_run) -> None:
    write_pxdesign_task(two_task_run.directory, 0, status={})
    adapter = PXDesignOutputAdapter()

    assert not adapter.succeeded(two_task_run.directory)
    write_pxdesign_task(two_task_run.directory, 1, status={})
    assert adapter.succeeded(two_task_run.directory)
