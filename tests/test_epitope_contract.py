"""No tool may silently ignore the campaign's epitope.

Three of the five once did. A campaign that names an epitope and a tool that
cannot honour it is a comparison quietly becoming meaningless: one tool designs
against a patch while another designs against the whole surface, and both
report designs. Every tool now either uses the epitope or refuses the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import (
    pxdesign_spec,
    write_boltzgen_configs,
    write_configs,
    write_genie3_configs,
    write_protein_hunter_configs,
    write_proteina_complexa_configs,
    write_pxdesign_configs,
    write_pxdesign_msa,
)

from bindocracy.config.preflight import (
    ConfigPreflightError,
    hotspot_numbers,
    parse_hotspots,
)
from bindocracy.tools import launch_spec, load_configs, plan

EPITOPE = ["A2", "A4"]


def with_hotspots(general: Path, hotspots: list[str]) -> Path:
    document = yaml.safe_load(general.read_text())
    document["target"]["hotspots"] = hotspots
    general.write_text(yaml.safe_dump(document))
    return general


def flag(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


# --- the shared reading ----------------------------------------------------


def test_a_hotspot_carries_a_chain_and_a_number() -> None:
    parsed = parse_hotspots(("A110", "112"))

    assert [(spot.chain, spot.number) for spot in parsed] == [("A", 110), ("", 112)]
    assert str(parsed[0]) == "A110"


def test_numbers_are_what_survives_a_renumbering() -> None:
    """Tools rename chains and renumber; the number is what stays comparable."""
    assert hotspot_numbers(("A110", "B110", "112")) == {110, 112}


def test_an_unreadable_hotspot_is_refused() -> None:
    with pytest.raises(ConfigPreflightError, match="cannot read a residue"):
        parse_hotspots(("the catalytic pocket",))


# --- each tool either uses it or refuses -----------------------------------


def test_mosaic_refuses_an_epitope_it_does_not_pass_on(tmp_path: Path) -> None:
    """Its loss has the term; the driver just builds it without an epitope.

    So the refusal is a placeholder for threading `epitope_idx` through, not a
    statement that Mosaic cannot be conditioned.
    """
    general, model = write_configs(tmp_path)
    with_hotspots(general, EPITOPE)

    with pytest.raises(ConfigPreflightError, match="BinderTargetContact"):
        load_configs(general, model)


def test_boltzgen_refuses_a_spec_that_binds_nothing(tmp_path: Path) -> None:
    """The spec is archived and never rewritten, so it can drift silently."""
    general, model = write_boltzgen_configs(tmp_path)
    with_hotspots(general, EPITOPE)

    with pytest.raises(ConfigPreflightError, match="names no binding site"):
        load_configs(general, model)


def test_boltzgen_accepts_a_spec_that_covers_the_epitope(tmp_path: Path) -> None:
    general, model = write_boltzgen_configs(tmp_path)
    with_hotspots(general, EPITOPE)
    spec_path = tmp_path / "binder_spec.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    spec["entities"][1]["file"]["binding_types"] = [
        {"chain": {"id": "A", "binding": "1..6"}}
    ]
    spec_path.write_text(yaml.safe_dump(spec))

    loaded = load_configs(general, model)
    assert loaded.model.spec.contents["entities"][1]["file"]["binding_types"]


def test_boltzgen_refuses_a_spec_that_binds_somewhere_else(tmp_path: Path) -> None:
    general, model = write_boltzgen_configs(tmp_path)
    with_hotspots(general, EPITOPE)
    spec_path = tmp_path / "binder_spec.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    spec["entities"][1]["file"]["binding_types"] = [
        {"chain": {"id": "A", "binding": "5,6"}}
    ]
    spec_path.write_text(yaml.safe_dump(spec))

    with pytest.raises(ConfigPreflightError, match="does not cover"):
        load_configs(general, model)


def test_genie3_still_compares_its_problem_sets_epitope(tmp_path: Path) -> None:
    general, model = write_genie3_configs(tmp_path)
    with_hotspots(general, ["A10", "A12", "A99"])

    with pytest.raises(ConfigPreflightError, match="conditions on residues"):
        load_configs(general, model)


def test_pxdesign_still_compares_its_specs_epitope(tmp_path: Path) -> None:
    cif = tmp_path / "target.cif"
    cif.write_text("data_target\n#\n")
    spec = pxdesign_spec(cif, write_pxdesign_msa(tmp_path), hotspots=[40, 99])
    general, model = write_pxdesign_configs(tmp_path, spec=spec)
    with_hotspots(general, EPITOPE)

    with pytest.raises(ConfigPreflightError, match="conditions on residues"):
        load_configs(general, model)



def test_proteina_complexa_still_compares_its_registrys_epitope(tmp_path: Path) -> None:
    general, model = write_proteina_complexa_configs(tmp_path)
    with_hotspots(general, EPITOPE)

    with pytest.raises(ConfigPreflightError, match="conditions on no residues"):
        load_configs(general, model)


def test_proteina_complexa_resolves_the_epitope_against_the_structure(
    tmp_path: Path,
) -> None:
    """Agreeing with the campaign is not enough; the residues must exist.

    The mask is built by string match against each CA atom, and a hotspot that
    matches nothing is silently dropped -- so a registry can agree with the
    campaign and still condition on nothing at all.
    """
    general, model = write_proteina_complexa_configs(tmp_path, hotspots=["A2", "A4444"])

    with pytest.raises(ConfigPreflightError, match="A4444"):
        load_configs(general, model)


def test_proteina_complexa_records_the_epitope_it_ran_with(tmp_path: Path) -> None:
    general, model = write_proteina_complexa_configs(tmp_path, hotspots=EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")

    assert manifest.workflow["hotspots"] == EPITOPE
    assert manifest.workflow["target_input"] == "A1-201"


# --- Protein-Hunter maps it onto the flags the pipeline reads --------------


def test_protein_hunter_conditions_on_the_campaign_epitope(tmp_path: Path) -> None:
    """Upstream reads --contact_residues for generation, resampling and hits."""
    general, model = write_protein_hunter_configs(tmp_path)
    with_hotspots(general, EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    assert flag(argv, "--contact-residues") == "2,4"
    assert flag(argv, "--contact-cutoff") == "15.0"
    assert flag(argv, "--max-contact-filter-retries") == "6"
    assert "--contact-filter" in argv


def test_a_hotspot_free_campaign_passes_no_contact_flags(
    protein_hunter_configs, tmp_path
) -> None:
    """Conditioning on nothing is different from conditioning on everything."""
    manifest = plan(load_configs(*protein_hunter_configs), tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    assert "--contact-residues" not in argv
    assert manifest.workflow["contacts"] == {"conditioned": False}


def test_the_manifest_records_how_the_epitope_was_enforced(tmp_path: Path) -> None:
    general, model = write_protein_hunter_configs(tmp_path)
    with_hotspots(general, EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")

    assert manifest.workflow["contacts"] == {
        "conditioned": True,
        "residues": [2, 4],
        "cutoff_angstroms": 15.0,
        "resample_on_miss": True,
        "max_retries": 6,
    }


def test_the_recorded_hit_protocol_is_the_whole_gate(tmp_path: Path) -> None:
    """Two of its four conditions are hard-coded upstream and in no config."""
    general, model = write_protein_hunter_configs(tmp_path)
    with_hotspots(general, EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    protocol = manifest.workflow["hit_protocol"]

    assert protocol["iptm_above"] == 0.7
    assert protocol["plddt_above"] == 0.7
    # Hard-coded as `alanine_percentage <= 0.20` in pipeline.py.
    assert protocol["alanine_fraction_at_most"] == 0.20
    # Hard-coded as "at least 2 contacted" in model_utils.py.
    assert protocol["contacts"]["min_residues_contacted"] == 2


def test_turning_the_resampling_filter_off_says_so(tmp_path: Path) -> None:
    general, model = write_protein_hunter_configs(
        tmp_path, contacts={"filter": False, "cutoff": 10.0}
    )
    with_hotspots(general, EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    assert "--no-contact-filter" in argv
    assert flag(argv, "--contact-cutoff") == "10.0"
    assert manifest.workflow["contacts"]["resample_on_miss"] is False


def test_an_epitope_on_another_chain_is_refused(tmp_path: Path) -> None:
    """Protein-Hunter folds one target chain and cannot condition on another."""
    general, model = write_protein_hunter_configs(tmp_path)
    with_hotspots(general, ["B2", "B4"])

    with pytest.raises(ConfigPreflightError, match="cannot condition on another"):
        load_configs(general, model)


def test_an_epitope_past_the_end_of_the_target_is_refused(tmp_path: Path) -> None:
    """It numbers the target 1..N over the FASTA it is handed."""
    general, model = write_protein_hunter_configs(tmp_path)
    with_hotspots(general, ["A2", "A900"])

    with pytest.raises(ConfigPreflightError, match="outside the target sequence"):
        load_configs(general, model)


def test_the_driver_accepts_the_contact_flags(
    protein_hunter_driver, tmp_path: Path
) -> None:
    """The connector builds them on the login node; the driver parses them."""
    general, model = write_protein_hunter_configs(tmp_path)
    with_hotspots(general, EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    parsed = protein_hunter_driver.parse_args(list(argv[argv.index("python") + 2 :]))
    assert parsed.contact_residues == "2,4"
    assert parsed.contact_cutoff == 15.0
    assert parsed.contact_filter is True

    command = protein_hunter_driver.pipeline_command(parsed)
    assert command[command.index("--contact_residues") + 1] == "2,4"
    assert "--no_contact_filter" not in command
