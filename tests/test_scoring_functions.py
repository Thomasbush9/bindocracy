"""Scoring functions: the conditions/functions split, and the custom contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.functions.contract import (
    FunctionInput,
    declared_but_absent,
    read_outputs,
    write_inputs,
)
from bindocracy.functions.models import CustomFunction, FunctionsConfig
from bindocracy.functions.runner import (
    FunctionError,
    command_for,
    preflight_custom,
    run_custom,
)
from bindocracy.tools.scorer.config import MOVED_TO_FUNCTIONS
from bindocracy.tools.scorer.preflight import preflight_scorer

WHEN = datetime(2026, 9, 11, tzinfo=UTC)
EXAMPLE = Path(__file__).resolve().parent / "fixtures" / "functions" / "net_charge.py"


def charge_function(**over) -> CustomFunction:
    spec = {
        "name": "charge",
        "script": str(EXAMPLE),
        "inputs": ["sequence"],
        "metrics": {
            "net_charge_at_ph": {"direction": "none"},
            "fraction_charged": {"direction": "none"},
        },
    }
    spec.update(over)
    return CustomFunction.model_validate(spec)


def designs(n: int = 3, structure: Path | None = None) -> list[FunctionInput]:
    return [
        FunctionInput(index=i, design_id=f"d{i}",
                      sequence="ACDEFGHIKLKRKRDE"[: 6 + i],
                      target_sequence="MKTAYIAK", structure=structure)
        for i in range(n)
    ]


# --- the split ---------------------------------------------------------------


def test_a_reader_that_folds_nothing_is_refused(scorer_configs) -> None:
    _, model = scorer_configs
    with pytest.raises(ValueError, match="fold nothing"):
        model.readers.__class__(complex=False, monomer=False)


@pytest.mark.parametrize("flag", sorted(MOVED_TO_FUNCTIONS))
def test_the_flags_that_never_worked_are_now_refused(scorer_configs, flag) -> None:
    """They validated, launched, succeeded and measured nothing -- the launcher
    only ever emitted complex and monomer. Failing is strictly better."""
    general, model = scorer_configs
    asking = model.model_copy(
        update={"readers": model.readers.model_copy(update={flag: True})}
    )
    with pytest.raises(ConfigPreflightError) as error:
        preflight_scorer(general, asking)
    message = str(error.value)
    assert f"readers.{flag}" in message
    assert f"functions.{flag}" in message, "the refusal must say where it moved"


def test_archived_configs_still_validate() -> None:
    """All twelve stored scorer configs name all four reader keys, and
    `configs_of()` validates with extra='forbid'. Dropping the fields would
    strand every historical run."""
    from bindocracy.tools.scorer.config import Readers

    archived = Readers.model_validate(
        {"complex": True, "monomer": True, "epitope": False, "inverse_folding": False}
    )
    assert archived.complex and not archived.epitope


# --- the config contract -----------------------------------------------------


def test_a_custom_metric_must_declare_its_direction() -> None:
    """The registry exists because a number stored the wrong way round sorts
    backwards and nothing in the row says so."""
    with pytest.raises(ValueError):
        CustomFunction.model_validate({
            "name": "x", "script": str(EXAMPLE),
            "metrics": {"score": {"unit": "kcal/mol"}},   # no direction
        })


def test_a_custom_function_cannot_redefine_a_registered_metric() -> None:
    """Two definitions under one name is the failure the model prefixes exist
    to prevent."""
    with pytest.raises(ValueError, match="already registered"):
        charge_function(metrics={"iptm": {"direction": "max"}})


def test_an_empty_functions_block_is_refused() -> None:
    with pytest.raises(ValueError, match="compute nothing"):
        FunctionsConfig.model_validate({})


def test_duplicate_function_names_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        FunctionsConfig.model_validate({
            "custom": [charge_function().model_dump(mode="json"),
                       charge_function().model_dump(mode="json")]
        })


# --- the file contract -------------------------------------------------------


def test_a_function_is_handed_only_what_it_asked_for(tmp_path) -> None:
    """A function declaring `inputs: [sequence]` never sees a structure path,
    so it cannot come to depend on one without saying so."""
    path = tmp_path / "in.jsonl"
    write_inputs(path, designs(structure=Path("/tmp/x.pdb")), ["sequence"])
    row = json.loads(path.read_text().splitlines()[0])
    assert "sequence" in row and "index" in row
    assert "structure" not in row and "target_sequence" not in row


def test_a_torn_final_line_does_not_lose_the_file(tmp_path) -> None:
    """What a killed process leaves behind."""
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"index": 0, "metrics": {"a": 1.0}}\n'
        '{"index": 1, "metrics": {"a": 2.0}}\n'
        '{"index": 2, "metr'
    )
    assert [o.index for o in read_outputs(path)] == [0, 1]


def test_a_promised_metric_that_did_not_arrive_is_reported() -> None:
    assert declared_but_absent(["a", "b"], {"a": 1.0}) == ("b",)


# --- running one -------------------------------------------------------------


def test_a_custom_function_runs_and_its_metrics_are_stored(tmp_path) -> None:
    result = run_custom(
        charge_function(), designs(3),
        run_id="run-1", work_dir=tmp_path / "charge", measured_at=WHEN,
    )
    assert result.n_scored == 3
    names = {r.name for r in result.records}
    # Stored under the function's name, so two functions may both compute a
    # `score` and the column still says which produced it.
    assert names == {"charge_net_charge_at_ph", "charge_fraction_charged"}
    assert not result.failures and not result.incomplete


def test_the_script_bytes_are_hashed_into_every_row(tmp_path) -> None:
    """A run has to record which bytes scored it, not which path they came
    from, or a script edited between two runs makes them look comparable."""
    result = run_custom(
        charge_function(), designs(2),
        run_id="run-1", work_dir=tmp_path / "charge", measured_at=WHEN,
    )
    assert result.script_sha256
    assert all(r.details["script_sha256"] == result.script_sha256 for r in result.records)
    # And the script itself is archived beside its output.
    assert (tmp_path / "charge" / EXAMPLE.name).is_file()


def test_an_undeclared_metric_is_refused(tmp_path) -> None:
    """Returning something the config never declared means storing a metric
    whose direction nothing records."""
    narrow = charge_function(metrics={"net_charge_at_ph": {"direction": "none"}})
    with pytest.raises(FunctionError, match="undeclared metric"):
        run_custom(narrow, designs(2), run_id="r", work_dir=tmp_path / "n",
                   measured_at=WHEN)


def test_a_missing_script_is_refused_before_it_runs() -> None:
    with pytest.raises(FunctionError, match="script not found"):
        preflight_custom(charge_function(script="/nonexistent/score.py"))


def test_designs_without_a_required_input_are_counted_not_dropped(tmp_path) -> None:
    """A function needing a structure cannot score a design that has none. That
    is a fact worth recording, not a silent omission."""
    needs_structure = charge_function(inputs=["sequence", "structure"])
    result = run_custom(
        needs_structure, designs(3),          # none have a structure
        run_id="r", work_dir=tmp_path / "s", measured_at=WHEN,
    )
    assert result.n_scored == 0
    assert result.failures["missing_required_input"] == 3


def test_a_container_changes_the_command_not_the_contract(tmp_path) -> None:
    plain = command_for(charge_function(), tmp_path / "i", tmp_path / "o")
    boxed = command_for(
        charge_function(container=tmp_path / "img.sif"), tmp_path / "i", tmp_path / "o"
    )
    import sys

    # The harness's own interpreter on the host -- there is no bare `python` on
    # this cluster's PATH, and a host function should see the dependencies the
    # harness installed.
    assert plain[0] == sys.executable
    assert boxed[0] == "singularity" and "--cleanenv" in boxed
    # Same two flags either way.
    assert plain[-4:] == boxed[-4:]
