"""What a BindCraft 2 task is told to do, and what preflight refuses first.

The seam this guards is that BC2 needs no driver: every value the harness owns
travels on the command line as a `--set` override, including the whole target
block. If that stops being true, a run designs against something other than the
campaign target and the output looks entirely normal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import bindcraft2_settings, write_bindcraft2_configs

from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.tools import launch_spec, load_configs, plan


def setting(argv: tuple[str, ...], key: str) -> str:
    """The value of one `--set KEY=VALUE`, which is two argv entries."""
    for index, item in enumerate(argv):
        if item == "--set" and argv[index + 1].startswith(f"{key}="):
            return argv[index + 1].split("=", 1)[1]
    raise AssertionError(f"no --set {key} in {argv}")


def planned(root: Path, **kwargs):
    general, model = write_bindcraft2_configs(root, **kwargs)
    loaded = load_configs(general, model)
    return loaded, plan(loaded, root / "run")


# --- the command -----------------------------------------------------------


def test_the_campaign_document_is_run_from_the_archive(tmp_path: Path) -> None:
    """The working tree may have moved on since the run was planned."""
    _, manifest = planned(tmp_path)
    spec = launch_spec(manifest, 0)

    archived = manifest.directory / manifest.provenance["settings"].path
    assert str(archived) in spec.argv
    assert str(tmp_path / "dio3_cut.json") not in spec.argv


def test_no_driver_is_used(tmp_path: Path) -> None:
    """The whole point of this plugin: BC2 overrides everything on the CLI."""
    _, manifest = planned(tmp_path)
    spec = launch_spec(manifest, 0)

    assert manifest.provenance.keys() == {"settings"}
    assert spec.argv[:2] == ("singularity", "exec")
    assert "bindcraft" in spec.argv and "design" in spec.argv


def test_the_target_block_carries_the_campaign_structure_and_chain(tmp_path: Path) -> None:
    _, manifest = planned(tmp_path)
    spec = launch_spec(manifest, 0)

    target = json.loads(setting(spec.argv, "targets"))
    assert target["target_path"] == str((tmp_path / "target.pdb").resolve())
    assert target["chains"] == "A"


def test_the_epitope_reaches_the_command(tmp_path: Path) -> None:
    _, manifest = planned(tmp_path, hotspots=["A2", "A4"])
    spec = launch_spec(manifest, 0)

    assert json.loads(setting(spec.argv, "targets"))["hotspots"] == "A2,A4"


def test_a_campaign_with_no_epitope_names_none(tmp_path: Path) -> None:
    """Absent rather than empty: BC2 reads no key as the whole surface."""
    _, manifest = planned(tmp_path)
    spec = launch_spec(manifest, 0)

    assert "hotspots" not in json.loads(setting(spec.argv, "targets"))


def test_the_four_counters_come_from_the_config(tmp_path: Path) -> None:
    _, manifest = planned(tmp_path)
    spec = launch_spec(manifest, 0)

    assert setting(spec.argv, "number_of_final_designs") == "2"
    assert setting(spec.argv, "max_trajectories") == "6"
    assert setting(spec.argv, "campaign_seed") == "19"


def test_each_task_gets_its_own_seed_and_its_own_directory(tmp_path: Path) -> None:
    """A campaign resumes by default, so a shared directory is two tasks
    continuing each other's work and both reporting the same designs."""
    _, manifest = planned(tmp_path, sampling={"jobs": 2})

    first, second = launch_spec(manifest, 0), launch_spec(manifest, 1)
    assert setting(first.argv, "campaign_seed") == "19"
    assert setting(second.argv, "campaign_seed") == "20"
    assert setting(first.argv, "project_folder") != setting(second.argv, "project_folder")


def test_the_allocated_gpus_survive_cleanenv(tmp_path: Path, monkeypatch) -> None:
    """`--cleanenv` drops the variable BC2 reads to find its cards."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    _, manifest = planned(tmp_path)

    assert launch_spec(manifest, 0).env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_scratch_is_node_local_and_unique_per_task(tmp_path: Path) -> None:
    _, manifest = planned(tmp_path, sampling={"jobs": 2})

    first = launch_spec(manifest, 0).env["TMPDIR"]
    second = launch_spec(manifest, 1).env["TMPDIR"]
    assert first != second
    assert first.startswith(str(tmp_path / "nodetmp"))


def test_a_worker_cap_reaches_the_container_only_when_set(tmp_path: Path) -> None:
    _, manifest = planned(tmp_path)
    assert "SINGULARITYENV_BINDCRAFT_WORKERS_PER_GPU" not in launch_spec(manifest, 0).env

    _, capped = planned(tmp_path / "capped", sampling={"workers_per_gpu": 5})
    assert launch_spec(capped, 0).env["SINGULARITYENV_BINDCRAFT_WORKERS_PER_GPU"] == "5"


def test_resources_come_from_the_config(tmp_path: Path) -> None:
    _, manifest = planned(tmp_path)
    resources = launch_spec(manifest, 0).resources

    assert resources["gres"] == "gpu:4"
    assert resources["cpus_per_task"] == 32
    assert resources["mem_mb"] == 192 * 1024


# --- the plan --------------------------------------------------------------


def test_the_plan_counts_attempts_not_successes(tmp_path: Path) -> None:
    _, manifest = planned(tmp_path)

    assert manifest.designs_per_task == 2
    assert manifest.tasks[0].n_generated == 6
    assert manifest.workflow["reproducible"] is True


def test_the_document_is_folded_into_the_stored_config(tmp_path: Path) -> None:
    """The thresholds are the definition of n_passed and live nowhere else."""
    loaded, _ = planned(tmp_path)

    assert loaded.model.settings.contents == bindcraft2_settings()


def test_an_epitope_without_a_ceiling_is_recorded_as_unverified(tmp_path: Path) -> None:
    """Conditioning is a bias; only a threshold makes it a requirement."""
    _, without = planned(
        tmp_path / "bias", hotspots=["A2"], settings={"binder_lengths": [65]}
    )
    assert without.workflow["epitope_enforcement"]["verified_after_generation"] is False

    _, enforced = planned(tmp_path / "filtered", hotspots=["A2"])
    enforcement = enforced.workflow["epitope_enforcement"]
    assert enforcement["verified_after_generation"] is True
    assert enforcement["epitope_filters"] == ["min_hotspot_contact_final"]


# --- what preflight refuses ------------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("project_folder", "/tmp/elsewhere"),
        ("number_of_final_designs", 99),
        ("max_trajectories", 99),
        ("campaign_seed", 7),
        ("resume", False),
        ("targets", [{"name": "other"}]),
        ("target", "hPDL1"),
        ("workers_per_gpu", 3),
    ],
)
def test_a_harness_owned_setting_is_refused_not_overwritten(
    tmp_path: Path, key: str, value: object
) -> None:
    """`--set` wins, so an authored value would be recorded and not used."""
    general, model = write_bindcraft2_configs(
        tmp_path, settings=bindcraft2_settings(**{key: value})
    )

    with pytest.raises(ConfigPreflightError, match=key):
        load_configs(general, model)


def test_a_campaign_without_binder_lengths_is_refused(tmp_path: Path) -> None:
    """It decides the padded complex size, and so the cost of the run."""
    general, model = write_bindcraft2_configs(tmp_path, settings={})

    with pytest.raises(ConfigPreflightError, match="binder_lengths"):
        load_configs(general, model)


def test_an_unresolvable_epitope_is_refused_before_a_gpu_is_taken(tmp_path: Path) -> None:
    general, model = write_bindcraft2_configs(tmp_path, hotspots=["A2", "A4444"])

    with pytest.raises(ConfigPreflightError, match="4444"):
        load_configs(general, model)


def test_an_epitope_on_another_chain_is_refused(tmp_path: Path) -> None:
    general, model = write_bindcraft2_configs(tmp_path, hotspots=["B2"])

    with pytest.raises(ConfigPreflightError, match="chain"):
        load_configs(general, model)


def test_a_campaign_with_no_structure_is_refused(tmp_path: Path) -> None:
    """BC2 reads mmCIF too, but the epitope is verified against the PDB."""
    general, model = write_bindcraft2_configs(tmp_path)
    document = yaml.safe_load(general.read_text())
    del document["target"]["structure_pdb"]
    general.write_text(yaml.safe_dump(document))

    with pytest.raises(ConfigPreflightError, match="structure_pdb"):
        load_configs(general, model)
