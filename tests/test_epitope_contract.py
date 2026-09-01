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
    freebindcraft_target,
    pxdesign_spec,
    write_boltzgen_configs,
    write_configs,
    write_freebindcraft_configs,
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


def test_mosaic_conditions_its_contact_loss_on_the_campaign_epitope(
    tmp_path: Path,
) -> None:
    """`BinderTargetContact` slices its contact matrix to the epitope columns.

    The indices are 0-based positions in the target sequence, so a campaign
    A2/A4 becomes [1, 3]. Indexing by enumeration position instead is the bug
    in docs/known-issues.md section 1.4.
    """
    general, model = write_configs(tmp_path)
    with_hotspots(general, EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    assert flag(argv, "--epitope") == "1,3"
    assert manifest.workflow["epitope_idx"] == [1, 3]


def test_a_hotspot_free_mosaic_run_passes_an_empty_epitope(configs, tmp_path) -> None:
    """Empty is a value: it is the loss with the whole target as the partner."""
    manifest = plan(load_configs(*configs), tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    assert flag(argv, "--epitope") == ""
    assert manifest.workflow["epitope_enforcement"] == {"conditioned": False}


def test_mosaic_records_how_little_its_epitope_enforces(tmp_path: Path) -> None:
    """One of nine loss terms, at a 20 A cutoff, never checked afterwards."""
    general, model = write_configs(tmp_path)
    with_hotspots(general, EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    enforcement = manifest.workflow["epitope_enforcement"]

    assert enforcement["conditioned"] is True
    assert enforcement["contact_distance_angstroms"] == 20.0
    assert enforcement["verified_after_generation"] is False


def test_mosaic_refuses_an_epitope_it_cannot_place(tmp_path: Path) -> None:
    """The FASTA is six residues; a hotspot past it would slice silently."""
    general, model = write_configs(tmp_path)
    with_hotspots(general, ["A2", "A900"])

    with pytest.raises(ConfigPreflightError, match="outside the target sequence"):
        load_configs(general, model)


def test_mosaic_refuses_an_epitope_on_another_chain(tmp_path: Path) -> None:
    general, model = write_configs(tmp_path)
    with_hotspots(general, ["B2", "B4"])

    with pytest.raises(ConfigPreflightError, match="cannot condition on another"):
        load_configs(general, model)


def test_the_mosaic_driver_accepts_and_bounds_the_epitope(
    driver, tmp_path: Path
) -> None:
    """The connector builds the flag on a login node; the driver parses it.

    The bounds check is repeated in the driver because this is the last place
    the numbers exist before they become an array slice, and JAX would clip an
    out-of-range index rather than raise.
    """
    general, model = write_configs(tmp_path)
    with_hotspots(general, EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    argv = list(launch_spec(manifest, 0).argv)

    assert driver.parse_epitope(flag(tuple(argv), "--epitope"), 6) == [1, 3]
    assert driver.parse_epitope("", 6) is None
    with pytest.raises(SystemExit, match="outside a target"):
        driver.parse_epitope("1,900", 6)


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


# --- FreeBindCraft writes the epitope rather than reading it ---------------


def test_freebindcraft_renders_the_campaign_epitope_onto_the_command(
    tmp_path: Path,
) -> None:
    """The harness owns `target_hotspot_residues`, so it cannot drift.

    BindCraft's hotspots are chain-then-number against the author numbering of
    the starting PDB, which is exactly how a campaign writes them, so this is a
    rendering rather than a mapping.
    """
    general, model = write_freebindcraft_configs(tmp_path, hotspots=EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    assert flag(argv, "--hotspots") == "A2,A4"
    assert manifest.workflow["hotspots"] == ["A2", "A4"]


def test_a_hotspot_free_freebindcraft_campaign_says_so(
    freebindcraft_configs, tmp_path: Path
) -> None:
    """Empty is a value: BindCraft reads it as `hotspot=None`, the whole surface."""
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")

    assert flag(launch_spec(manifest, 0).argv, "--hotspots") == ""
    assert manifest.workflow["hotspots"] == []


def test_freebindcraft_resolves_the_epitope_against_the_structure(
    tmp_path: Path,
) -> None:
    """ColabDesign asserts on a hotspot that matches no CA atom.

    It is loud, but it is loud inside the container, once Slurm has already
    allocated a GPU. Checked here instead.
    """
    general, model = write_freebindcraft_configs(tmp_path, hotspots=["A2", "A4444"])

    with pytest.raises(ConfigPreflightError, match="4444"):
        load_configs(general, model)


def test_freebindcraft_refuses_an_epitope_on_a_chain_it_does_not_design_against(
    tmp_path: Path,
) -> None:
    general, model = write_freebindcraft_configs(tmp_path, hotspots=["B2", "B4"])

    with pytest.raises(ConfigPreflightError, match="chain A alone"):
        load_configs(general, model)


def test_freebindcraft_refuses_an_authored_epitope(tmp_path: Path) -> None:
    """One place decides what the epitope is, and it is not the target JSON."""
    general, model = write_freebindcraft_configs(
        tmp_path,
        hotspots=EPITOPE,
        target=freebindcraft_target(tmp_path / "target.pdb",
                                    target_hotspot_residues="A2,A4"),
    )

    with pytest.raises(ConfigPreflightError, match="target_hotspot_residues"):
        load_configs(general, model)


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


def test_freebindcraft_records_how_little_an_epitope_enforces(tmp_path: Path) -> None:
    """A hotspot biases the backbone search and is never checked again.

    It restricts the interface contact loss during hallucination, where a
    "contact" is two binder residues within 20 A. Nothing after that stage
    looks at the epitope: no filter mentions it, and this fork's
    `Trajectory_WrongHotspot` counter is created but never incremented. A run
    that says only "conditioned on A41,A42,A44" would claim more than it did.
    """
    general, model = write_freebindcraft_configs(tmp_path, hotspots=EPITOPE)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    enforcement = manifest.workflow["epitope_enforcement"]

    assert enforcement["conditioned"] is True
    assert enforcement["contact_distance_angstroms"] == 20.0
    assert enforcement["verified_after_generation"] is False


def test_a_hotspot_free_freebindcraft_run_claims_no_conditioning(
    freebindcraft_configs, tmp_path: Path
) -> None:
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")

    assert manifest.workflow["epitope_enforcement"] == {"conditioned": False}
