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


# --- the built-ins, on the same runner --------------------------------------


def test_the_builtins_are_the_same_type_the_runner_takes() -> None:
    """They go through `run_custom` unchanged, so the extension point is
    exercised by the harness rather than merely offered to others."""
    from bindocracy.functions.models import ScoringFunction

    resolved = FunctionsConfig.model_validate(
        {"sequence": True, "epitope": True}
    ).resolve()
    assert all(isinstance(f, ScoringFunction) for f in resolved)
    assert {f.name for f in resolved} == {"sequence", "epitope"}


def test_a_builtin_declares_nothing_because_its_metrics_are_registered() -> None:
    from bindocracy.functions.models import builtin

    sequence = builtin("sequence")
    assert set(sequence.specs) == {
        "length", "net_charge", "molecular_weight", "hydrophobic_fraction",
        "n_cysteines", "n_glycosylation_motifs", "max_low_complexity_run",
    }


def test_an_enabled_but_unimplemented_function_is_refused() -> None:
    """The exact failure the split exists to stop: a flag that validates,
    launches and measures nothing."""
    with pytest.raises(ValueError, match="not implemented"):
        FunctionsConfig.model_validate({"inverse_folding": True}).resolve()


def test_sequence_metrics_are_prefixed_by_the_function(tmp_path) -> None:
    """A binder's net charge does not depend on which model folded it."""
    from bindocracy.functions.models import builtin

    result = run_custom(
        builtin("sequence"), designs(3),
        run_id="r", work_dir=tmp_path / "seq", measured_at=WHEN,
    )
    assert result.n_scored == 3
    assert {r.name for r in result.records} == {
        "sequence_length", "sequence_net_charge", "sequence_molecular_weight",
        "sequence_hydrophobic_fraction", "sequence_n_cysteines",
        "sequence_n_glycosylation_motifs", "sequence_max_low_complexity_run",
    }


def test_epitope_metrics_are_prefixed_by_the_model_that_folded(tmp_path) -> None:
    """Geometry read off one model's pose belongs to that model. Two models
    disagree about where the binder sits by a median 22.7 A, and one
    unprefixed `epitope_coverage` column would hide that entirely."""
    from bindocracy.functions.models import builtin
    from bindocracy.functions.runner import _prefix_for

    epitope = builtin("epitope")
    row = FunctionInput(index=0, design_id="d", sequence="AAA",
                        structure=Path("/tmp/p.pdb"), source_model="boltz2")
    assert _prefix_for(epitope, row) == "boltz2"
    # A pose whose model was not recorded falls back rather than inventing one.
    anonymous = FunctionInput(index=0, design_id="d", sequence="AAA",
                              structure=Path("/tmp/p.pdb"))
    assert _prefix_for(epitope, anonymous) == "epitope"


# --- what the sequence function actually computes ---------------------------


def sequence_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "drivers" / "functions" / "sequence_metrics.py"
    spec = importlib.util.spec_from_file_location("sequence_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_net_charge_matches_the_benchmark_control() -> None:
    """This is the control bar: on Nipah-G, these sequence-only properties
    reach AUC 0.642 and five of nine folding models sit within 0.06 of that.
    It has to be the same quantity the benchmark computed, or the comparison
    is meaningless."""
    m = sequence_module()
    assert m.metrics_for("KKKRRR")["net_charge"] == 6.0
    assert m.metrics_for("DDDEEE")["net_charge"] == -6.0
    # Histidine is excluded, as the benchmark's control does.
    assert m.metrics_for("HHHH")["net_charge"] == 0.0


def test_a_glycosylation_sequon_needs_a_non_proline() -> None:
    m = sequence_module()
    assert m.glycosylation_sequons("NGS") == 1
    assert m.glycosylation_sequons("NPS") == 0, "N-P-S is not a sequon"
    assert m.glycosylation_sequons("NGT") == 1


def test_low_complexity_finds_the_longest_run() -> None:
    m = sequence_module()
    assert m.longest_run("AAABBBB") == 4
    assert m.longest_run("ABCDE") == 1
    assert m.longest_run("") == 0


# --- what the epitope function actually computes ----------------------------


def epitope_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "drivers" / "functions" / "epitope_metrics.py"
    spec = importlib.util.spec_from_file_location("epitope_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hotspots_are_target_positions_not_author_numbering() -> None:
    """Both obvious designs were wrong, and a real pose showed it: chain
    letters are not portable between drivers, and residue ids in a predicted
    structure are positional and 0-based."""
    m = epitope_module()
    assert m.parse_hotspots("110,112,131") == {109, 111, 130}
    with pytest.raises(ValueError, match="1-based target position"):
        m.parse_hotspots("A110")
    with pytest.raises(ValueError, match="not a 1-based position"):
        m.parse_hotspots("0")


def two_chain_pdb(path: Path, binder_n: int, target_n: int, gap: float) -> Path:
    """A toy complex: two CA-only chains `gap` angstroms apart.

    Chain A is the LONGER one, deliberately -- the mosaic driver writes
    [binder, target] while the co-folding drivers write the target first, so a
    function keying on the chain letter would get this backwards for half the
    panel. The binder must be found by length.
    """
    lines, serial = [], 1
    for chain, count, x in (("A", target_n, 0.0), ("B", binder_n, gap)):
        for i in range(count):
            lines.append(
                f"ATOM  {serial:>5d}  CA  ALA {chain}{i + 1:>4d}    "
                f"{x:>8.3f}{i * 3.8:>8.3f}{0.0:>8.3f}  1.00  0.00           C"
            )
            serial += 1
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_the_binder_is_found_by_length_not_by_chain_letter(tmp_path) -> None:
    """Chain A here is the 8-residue target and chain B the 3-residue binder,
    which is the layout the mosaic driver produces inverted. A real pose showed
    this: its chains were A=binder(60) and B=target(201)."""
    m = epitope_module()
    pose = two_chain_pdb(tmp_path / "c.pdb", binder_n=3, target_n=8, gap=4.0)
    values = m.metrics_for(str(pose), binder_length=3, hotspots=set(), cutoff=5.0)
    # Every target residue within reach is counted, and none of the binder's.
    assert 0 < values["n_interface_residues"] <= 8


def test_an_out_of_range_hotspot_is_refused_not_scored_as_a_miss(tmp_path) -> None:
    """A hotspot past the end of the target means the numbering is wrong.
    Reporting coverage 0.0 would read as a real measurement of a real miss."""
    m = epitope_module()
    pose = two_chain_pdb(tmp_path / "c.pdb", binder_n=3, target_n=8, gap=4.0)
    with pytest.raises(ValueError, match="beyond"):
        m.metrics_for(str(pose), 3, {99}, 5.0)


def test_a_distant_binder_contacts_nothing(tmp_path) -> None:
    """Far apart, there is no interface at all -- and coverage of a named
    hotspot is a real 0.0 rather than an absent measurement."""
    m = epitope_module()
    pose = two_chain_pdb(tmp_path / "far.pdb", binder_n=3, target_n=8, gap=40.0)
    values = m.metrics_for(str(pose), 3, {2}, 5.0)
    assert values["n_interface_residues"] == 0
    assert values["epitope_coverage"] == 0.0
    assert values["epitope_offset"] > 30


def test_no_hotspots_means_absent_not_zero(tmp_path) -> None:
    """`epitope_coverage` with nothing to cover is NaN, which the driver drops
    so no row is stored. A stored 0.0 would claim the binder missed an epitope
    nobody named."""
    import math

    m = epitope_module()
    pose = two_chain_pdb(tmp_path / "c.pdb", binder_n=3, target_n=8, gap=4.0)
    values = m.metrics_for(str(pose), 3, set(), 5.0)
    assert math.isnan(values["epitope_coverage"])
    assert math.isnan(values["epitope_offset"])
    # The interface itself is still a real measurement.
    assert values["n_interface_residues"] > 0


# --- the built-ins have to be reachable from a command ----------------------


def test_a_builtin_function_can_be_named_and_a_custom_one_supplied(tmp_path):
    """Until 2026-09-22 neither built-in could be run by any command.

    `FunctionRunConfig.function` accepted only a `CustomFunction`, so `epitope`
    was refused twice over: as a built-in for declaring no `metrics`, and as a
    custom function for declaring metric names the registry already owns. It
    was implemented, tested, documented as a worked example, and unreachable --
    the same shape as the `readers.epitope` bug one level up.
    """
    from bindocracy.functions.run import FunctionRunConfig

    named = FunctionRunConfig.model_validate({
        "name": "epitope-pilot", "design_set": str(tmp_path / "set.json"),
        "builtin": "epitope", "structures_from": "score-boltz2-pilot",
    })
    assert named.builtin == "epitope"
    assert named.function is None

    # Exactly one, and one is required.
    with pytest.raises(ValueError, match="exactly one"):
        FunctionRunConfig.model_validate({
            "name": "neither", "design_set": str(tmp_path / "set.json"),
        })
    with pytest.raises(ValueError, match="exactly one"):
        FunctionRunConfig.model_validate({
            "name": "both", "design_set": str(tmp_path / "set.json"),
            "builtin": "sequence",
            "function": {"name": "charge", "script": str(tmp_path / "s.py"),
                         "metrics": {"my_charge": {"direction": "none"}}},
        })


def test_the_epitope_function_is_given_the_campaign_hotspots(tmp_path):
    """Resolved to FASTA positions, or the run is refused.

    A built-in with no `--hotspots` reports coverage and offset as absent --
    correctly, since no epitope was named -- and stores only the two geometry
    metrics. That is a run that succeeds and does not answer the question it
    was started for, so a campaign naming no hotspots is refused instead.
    """
    from bindocracy.config.models import GeneralConfig
    from bindocracy.config.preflight import ConfigPreflightError
    from bindocracy.functions.run import FunctionRunConfig, resolve_function

    fasta = tmp_path / "target.fasta"
    fasta.write_text(">t\n" + "A" * 201 + "\n")
    payload = {
        "schema_version": 1,
        "campaign": {"name": "c"},
        "target": {"name": "t", "sequence_fasta": str(fasta), "chain_id": "A",
                   "hotspots": ["A110", "A112", "A131"]},
        "cluster": {"executor": "slurm", "account": "acct",
                    "default_partition": "gpu"},
    }
    general = GeneralConfig.model_validate(payload)
    config = FunctionRunConfig.model_validate({
        "name": "epi", "design_set": str(tmp_path / "set.json"),
        "builtin": "epitope", "structures_from": "score-boltz2",
    })

    function = resolve_function(config, general, "A" * 201)
    assert function.args == ("--hotspots", "110,112,131")
    assert function.prefix == "source_model"
    # A built-in declares no metrics of its own; they come from the registry.
    assert set(function.specs) == {
        "epitope_coverage", "n_epitope_contacts", "n_interface_residues",
        "epitope_offset",
    }

    # Author numbering that is not FASTA position is refused, not clamped.
    payload["target"]["hotspots"] = ["A410"]
    with pytest.raises(ConfigPreflightError, match="outside the target"):
        resolve_function(config, GeneralConfig.model_validate(payload), "A" * 201)

    # No epitope named at all.
    payload["target"]["hotspots"] = []
    with pytest.raises(ConfigPreflightError, match="names none"):
        resolve_function(config, GeneralConfig.model_validate(payload), "A" * 201)

    # A sequence-only function needs nothing from the campaign.
    sequence_config = config.model_copy(
        update={"builtin": "sequence", "structures_from": None}
    )
    assert resolve_function(sequence_config, general, "A" * 201).args == ()
