"""Output has to be the shape the run was planned to produce.

Both collectors once decided a run was complete from an aggregate count, so one
task producing too few rows could hide behind another producing extra. That
matters most for Protein-Hunter, where a complete table plus an absent
threshold file is read as a genuine zero-hit result.

The alignment checks are here for the same reason: an a3m for another protein
is a well-formed file that folds the wrong target and looks entirely normal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import (
    PROTEIN_HUNTER_FIXTURE,
    PXDESIGN_FIXTURE,
    pxdesign_spec,
    write_protein_hunter_configs,
    write_protein_hunter_task,
    write_pxdesign_configs,
    write_pxdesign_msa,
    write_pxdesign_task,
)

from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.store.records import RunStatus
from bindocracy.tools import collect_run, load_configs, plan

OTHER_PROTEIN = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQ"


def collected(manifest):
    return collect_run(manifest.directory / "run.json")


# --- an alignment must be of this target -----------------------------------


def test_protein_hunter_refuses_an_alignment_of_another_protein(tmp_path: Path) -> None:
    general, model = write_protein_hunter_configs(tmp_path)
    (tmp_path / "target.a3m").write_text(f">someone_else\n{OTHER_PROTEIN}\n")

    with pytest.raises(ConfigPreflightError, match="not an alignment of the campaign"):
        load_configs(general, model)


def test_pxdesign_refuses_an_alignment_of_another_protein(tmp_path: Path) -> None:
    cif = tmp_path / "target.cif"
    cif.write_text("data_target\n#\n")
    msa = write_pxdesign_msa(tmp_path)
    (msa / "non_pairing.a3m").write_text(f">someone_else\n{OTHER_PROTEIN}\n")
    general, model = write_pxdesign_configs(tmp_path, spec=pxdesign_spec(cif, msa))

    with pytest.raises(ConfigPreflightError, match="not an alignment of the campaign"):
        load_configs(general, model)


def test_a3m_insertions_and_gaps_are_not_part_of_the_query(tmp_path: Path) -> None:
    """Lower case is an insertion and `-` is a gap; neither is a residue."""
    general, model = write_protein_hunter_configs(tmp_path)
    (tmp_path / "target.a3m").write_text(">target\nACDdEF-G\n>hit\nACDEFG\n")

    # The campaign target is ACDEFG, and this query normalizes to it.
    assert load_configs(general, model).preflight.msa is not None


# --- PXDesign: exactly the planned rows and ranks ---------------------------


@pytest.fixture
def pxdesign_run(tmp_path: Path):
    general, model = write_pxdesign_configs(
        tmp_path,
        sampling={"jobs": 2, "designs_per_job": 17, "seed_base": 0, "preset": "extended"},
    )
    return plan(load_configs(general, model), tmp_path / "run")


def test_one_task_short_cannot_hide_behind_another(pxdesign_run) -> None:
    """The aggregate count would have been 17 + 17 either way."""
    write_pxdesign_task(pxdesign_run.directory, 0, designs=12, status={})
    write_pxdesign_task(pxdesign_run.directory, 1, status={})
    result = collected(pxdesign_run)

    tasks = result.run.count_details["tasks"]
    assert tasks["0000"]["shape_problems"] == ["12 rows, expected 17"]
    assert tasks["0001"]["shape_problems"] == []
    assert result.run.status == RunStatus.PARTIAL


def test_a_repeated_rank_is_a_shape_problem(pxdesign_run) -> None:
    lines = (PXDESIGN_FIXTURE / "summary.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    write_pxdesign_task(pxdesign_run.directory, 0,
                        summary="\n".join([header, *body[:-1], body[0]]) + "\n",
                        status={})
    write_pxdesign_task(pxdesign_run.directory, 1, status={})
    result = collected(pxdesign_run)

    # The duplicate is rejected as a design, which leaves the table short.
    assert result.run.count_details["tasks"]["0000"]["shape_problems"] == [
        "16 rows, expected 17",
    ]
    assert result.run.status == RunStatus.PARTIAL


def test_a_complete_pair_of_tasks_is_a_complete_run(pxdesign_run) -> None:
    for task_id in (0, 1):
        write_pxdesign_task(pxdesign_run.directory, task_id, status={})
    result = collected(pxdesign_run)

    assert result.run.n_produced == 34
    assert result.run.status == RunStatus.SUCCEEDED
    assert all(task["shape_problems"] == []
               for task in result.run.count_details["tasks"].values())


# --- Protein-Hunter: every trajectory, every cycle --------------------------


@pytest.fixture
def hunter_run(tmp_path: Path):
    general, model = write_protein_hunter_configs(
        tmp_path, sampling={"jobs": 1, "trajectories_per_job": 3, "cycles": 5}
    )
    return plan(load_configs(general, model), tmp_path / "run")


def test_a_trajectory_missing_a_cycle_is_a_shape_problem(hunter_run) -> None:
    """Totals hide it: one trajectory short and another long still sums right."""
    lines = (PROTEIN_HUNTER_FIXTURE / "summary_all_runs.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    columns = header.split(",")
    fields = body[0].split(",")
    fields[columns.index("cycle_3_seq")] = ""
    write_protein_hunter_task(hunter_run.directory, 0,
                              summary="\n".join([header, ",".join(fields), *body[1:]]) + "\n",
                              status={})
    result = collected(hunter_run)

    problems = result.run.count_details["tasks"]["0000"]["shape_problems"]
    assert "14 designs, expected 15" in problems
    assert any("cycles [1, 2, 4, 5], expected 1..5" in problem for problem in problems)
    assert result.run.status == RunStatus.PARTIAL


def test_an_unfinished_task_is_not_read_as_zero_hits(hunter_run) -> None:
    """The whole reason exact shape matters for this tool."""
    write_protein_hunter_task(hunter_run.directory, 0, trajectories=2,
                              thresholds=False, status={})
    result = collected(hunter_run)

    assert result.run.n_passed is None
    assert result.run.status == RunStatus.PARTIAL


def test_a_complete_task_with_no_hits_is_still_a_zero(hunter_run) -> None:
    write_protein_hunter_task(hunter_run.directory, 0, thresholds=False, status={})
    result = collected(hunter_run)

    assert result.run.n_passed == 0
    assert result.run.status == RunStatus.SUCCEEDED
    assert result.run.count_details["tasks"]["0000"]["shape_problems"] == []


# --- Protein-Hunter: the second table is checked against the first ----------


def rewrite_verdicts(run_dir: Path, transform) -> None:
    table = run_dir / "tasks" / "0000" / "summary_high_iptm.csv"
    lines = table.read_text().splitlines()
    table.write_text("\n".join(transform(lines[0], lines[1:])) + "\n")


def test_a_verdict_for_a_design_that_is_not_there_is_counted(hunter_run) -> None:
    write_protein_hunter_task(hunter_run.directory, 0, status={})
    rewrite_verdicts(hunter_run.directory, lambda header, body: [
        header, *body, ",".join(["99", "3", *body[0].split(",")[2:]]),
    ])
    result = collected(hunter_run)

    assert result.run.count_details["tasks"]["0000"]["n_verdicts_unmatched"] == 1
    assert result.run.n_passed == 6


def test_a_repeated_verdict_key_does_not_overwrite(hunter_run) -> None:
    write_protein_hunter_task(hunter_run.directory, 0, status={})
    rewrite_verdicts(hunter_run.directory, lambda header, body: [header, *body, body[0]])
    result = collected(hunter_run)

    assert result.run.count_details["tasks"]["0000"]["n_verdicts_duplicated"] == 1
    assert result.run.n_passed == 6


def test_a_verdict_disagreeing_with_the_primary_table_is_refused(hunter_run) -> None:
    """A stale second file would otherwise mark the wrong design as passed."""
    write_protein_hunter_task(hunter_run.directory, 0, status={})

    def spoil(header: str, body: list[str]) -> list[str]:
        columns = header.split(",")
        fields = body[0].split(",")
        fields[columns.index("sequence")] = "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM"
        return [header, ",".join(fields), *body[1:]]

    rewrite_verdicts(hunter_run.directory, spoil)
    result = collected(hunter_run)

    assert result.run.count_details["tasks"]["0000"]["n_verdicts_inconsistent"] == 1
    assert result.run.n_passed == 5


def test_a_verdict_with_the_wrong_score_is_refused(hunter_run) -> None:
    write_protein_hunter_task(hunter_run.directory, 0, status={})

    def spoil(header: str, body: list[str]) -> list[str]:
        columns = header.split(",")
        fields = body[0].split(",")
        fields[columns.index("iptm")] = "0.999"
        return [header, ",".join(fields), *body[1:]]

    rewrite_verdicts(hunter_run.directory, spoil)
    result = collected(hunter_run)

    assert result.run.count_details["tasks"]["0000"]["n_verdicts_inconsistent"] == 1
    assert result.run.n_passed == 5


def test_every_verdict_row_carries_the_gate_it_applied(hunter_run) -> None:
    """`high_iptm` names one of four conditions; the value names them all."""
    write_protein_hunter_task(hunter_run.directory, 0, status={})
    result = collected(hunter_run)

    values = {d.value for d in result.decisions if d.name == "protein_hunter_high_iptm"}
    assert values == {"iptm>0.7 & plddt>0.7 & alanine<=0.2"}
