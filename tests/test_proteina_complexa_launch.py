"""What a Proteina-Complexa task must be launched with, and what it must not.

Two of these guard silent wrong answers rather than crashes: a `++root_path`
that would leave generation unseeded, and a registry that can disagree with the
campaign while every other check passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import (
    proteina_registry,
    write_proteina_complexa_configs,
    write_proteina_target_pdb,
)

from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.tools import launch_spec, load_configs, plan
from bindocracy.tools.proteina_complexa.launch import CONTAINER_TARGET, RUN_NAME

EPITOPE = ["A45", "A67"]


def value_of(argv: tuple[str, ...], key: str) -> str:
    """The value of one `++key=value` Hydra override."""
    for item in argv:
        if item.startswith(f"++{key}="):
            return item.split("=", 1)[1]
    raise AssertionError(f"{key} not in argv: {argv}")


def spec_for(tmp_path: Path, **kwargs):
    general, model = write_proteina_complexa_configs(tmp_path, **kwargs)
    loaded = load_configs(general, model)
    manifest = plan(loaded, tmp_path / "run", name="pcx")
    return manifest, launch_spec(manifest, 0)


# --- the command -------------------------------------------------------------


def test_the_task_directory_is_the_output_directory(tmp_path: Path) -> None:
    """`--pwd` is the only thing placing output; there is no flag for it."""
    manifest, spec = spec_for(tmp_path)
    task_dir = manifest.directory / manifest.tasks[0].directory

    assert "--pwd" in spec.argv
    assert spec.argv[spec.argv.index("--pwd") + 1] == str(task_dir)
    assert task_dir in spec.mkdirs


def test_root_path_is_never_passed(tmp_path: Path) -> None:
    """It would place output *and* silently disable seeding.

    generate.py calls setup() only when root_path is None, and setup() is the
    sole caller of L.seed_everything and the sole place seed + job_id is
    applied. A run with ++root_path records a seed that did nothing.
    """
    _, spec = spec_for(tmp_path)

    assert not any(item.startswith("++root_path") for item in spec.argv)


def test_each_task_samples_from_its_own_seed(tmp_path: Path) -> None:
    general, model = write_proteina_complexa_configs(tmp_path, sampling={"jobs": 2})
    loaded = load_configs(general, model)
    manifest = plan(loaded, tmp_path / "run", name="pcx")

    first = value_of(launch_spec(manifest, 0).argv, "seed")
    second = value_of(launch_spec(manifest, 1).argv, "seed")

    assert (first, second) == ("5", "6")


def test_the_campaign_target_is_bound_over_the_registry_path(tmp_path: Path) -> None:
    general, _ = write_proteina_complexa_configs(tmp_path)
    _, spec = spec_for(tmp_path)
    target = yaml.safe_load(general.read_text())["target"]["structure_pdb"]

    assert f"{target}:{CONTAINER_TARGET}:ro" in spec.argv


def test_the_archived_registry_is_what_gets_bound(tmp_path: Path) -> None:
    """Not the authored one: the working tree may have moved on."""
    manifest, spec = spec_for(tmp_path)
    archived = manifest.directory / manifest.provenance["registry"].path

    assert any(item.startswith(f"{archived}:") for item in spec.argv)
    assert archived.is_file()


def test_the_search_is_configured_rather_than_defaulted(tmp_path: Path) -> None:
    """Every in-image default that is wrong for a campaign is overridden."""
    _, spec = spec_for(tmp_path, sampling={"samples_per_job": 40, "keep_per_job": 40,
                                           "replicas": 2})

    assert value_of(spec.argv, "generation.dataloader.dataset.nres.nsamples") == "40"
    assert value_of(spec.argv, "generation.search.best_of_n.replicas") == "2"
    assert value_of(spec.argv, "generation.filter.filter_samples_limit") == "40"
    assert value_of(spec.argv, "run_name") == RUN_NAME


def test_jax_does_not_preallocate_the_gpu(tmp_path: Path) -> None:
    """AF2 is JAX and the model is torch, on one device."""
    _, spec = spec_for(tmp_path)

    assert spec.env["SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert spec.env["TMPDIR"] == spec.env["SINGULARITYENV_TMPDIR"]
    assert not spec.env["TMPDIR"].startswith("/n/")


# --- the run's shape ---------------------------------------------------------


def test_generating_and_keeping_are_different_numbers(tmp_path: Path) -> None:
    general, model = write_proteina_complexa_configs(
        tmp_path, sampling={"samples_per_job": 40, "replicas": 3, "keep_per_job": 40}
    )
    loaded = load_configs(general, model)
    manifest = plan(loaded, tmp_path / "run", name="pcx")

    assert manifest.tasks[0].n_generated == 120
    assert manifest.tasks[0].n_requested == 40


def test_a_task_cannot_keep_more_than_it_drew(tmp_path: Path) -> None:
    general, model = write_proteina_complexa_configs(
        tmp_path, sampling={"samples_per_job": 4, "keep_per_job": 1000}
    )
    loaded = load_configs(general, model)
    manifest = plan(loaded, tmp_path / "run", name="pcx")

    assert manifest.tasks[0].n_requested == 4


def test_the_registry_is_stored_whole_rather_than_by_path(tmp_path: Path) -> None:
    """Editing the registry must change model_config_id, not just the file."""
    general, model = write_proteina_complexa_configs(tmp_path, hotspots=[])
    loaded = load_configs(general, model)

    stored = loaded.model.registry.contents
    assert stored is not None
    assert stored["target_dict_cfg"]["test_target"]["binder_length"] == [70, 110]


# --- refusals ----------------------------------------------------------------


def test_a_registry_naming_another_target_is_refused(tmp_path: Path) -> None:
    registry = proteina_registry("some_other_target")
    general, model = write_proteina_complexa_configs(tmp_path, registry=registry)

    with pytest.raises(ConfigPreflightError, match="has no target 'test_target'"):
        load_configs(general, model)


def test_a_registry_pointing_outside_the_bind_is_refused(tmp_path: Path) -> None:
    """A target_path the launcher does not bind is a different structure."""
    registry = proteina_registry(target_path="./assets/target_data/PD1.pdb")
    general, model = write_proteina_complexa_configs(tmp_path, registry=registry)

    with pytest.raises(ConfigPreflightError, match="binds the campaign structure"):
        load_configs(general, model)


def test_a_campaign_without_structure_pdb_is_refused(tmp_path: Path) -> None:
    general, model = write_proteina_complexa_configs(tmp_path)
    document = yaml.safe_load(general.read_text())
    del document["target"]["structure_pdb"]
    general.write_text(yaml.safe_dump(document))

    with pytest.raises(ConfigPreflightError, match="no target.structure_pdb"):
        load_configs(general, model)


def test_a_nonsense_binder_length_is_refused(tmp_path: Path) -> None:
    registry = proteina_registry(binder_length=[110, 70])
    general, model = write_proteina_complexa_configs(tmp_path, registry=registry)

    with pytest.raises(ConfigPreflightError, match="ascending range"):
        load_configs(general, model)


# --- the epitope -------------------------------------------------------------


def test_the_campaign_epitope_reaches_the_registry(tmp_path: Path) -> None:
    general, model = write_proteina_complexa_configs(tmp_path, hotspots=EPITOPE)
    loaded = load_configs(general, model)

    assert sorted(loaded.preflight.hotspots) == sorted(EPITOPE)


def test_an_epitope_the_registry_ignores_is_refused(tmp_path: Path) -> None:
    """The campaign names one and the registry does not: §1.7b."""
    general, model = write_proteina_complexa_configs(tmp_path)
    document = yaml.safe_load(general.read_text())
    document["target"]["hotspots"] = EPITOPE
    general.write_text(yaml.safe_dump(document))

    with pytest.raises(ConfigPreflightError, match="conditions on no residues"):
        load_configs(general, model)


def test_a_hotspot_that_resolves_to_nothing_is_refused(tmp_path: Path) -> None:
    """The check the tool itself does not do.

    load_target_from_pdb marks the mask True for each CA whose chain+number is
    listed and does nothing with the rest, so an unresolvable hotspot yields an
    all-False mask and a run that looks entirely normal.
    """
    general, model = write_proteina_complexa_configs(tmp_path, hotspots=["A45", "A9999"])

    with pytest.raises(ConfigPreflightError, match="A9999"):
        load_configs(general, model)


def test_a_hotspot_outside_the_crop_says_so(tmp_path: Path) -> None:
    """It exists in the structure, which is a different mistake from a typo."""
    registry = proteina_registry("test_target", hotspots=["A20"], target_input="A1-10")
    general, model = write_proteina_complexa_configs(
        tmp_path, registry=registry, hotspots=["A20"]
    )

    with pytest.raises(ConfigPreflightError, match="outside the crop"):
        load_configs(general, model)


def test_a_hotspot_on_a_chain_the_structure_lacks_is_refused(tmp_path: Path) -> None:
    write_proteina_target_pdb(tmp_path / "target.pdb")
    registry = proteina_registry("test_target", hotspots=["B45"], target_input="A1-201")
    general, model = write_proteina_complexa_configs(
        tmp_path, registry=registry, hotspots=["B45"]
    )

    with pytest.raises(ConfigPreflightError, match="B45"):
        load_configs(general, model)
