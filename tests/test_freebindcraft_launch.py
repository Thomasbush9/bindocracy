"""What FreeBindCraft must refuse, and what one task's command has to say.

BindCraft is configured almost entirely by three JSON documents, so every check
worth having is about one of them disagreeing with the campaign while the run
still looks entirely normal. The three keys the driver writes are the sharpest
case: an authored `number_of_final_designs` is not the one the run uses, an
authored `design_path` makes two tasks resume each other, and an authored
epitope is one more thing that can drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import (
    FREEBINDCRAFT_TARGET,
    freebindcraft_advanced,
    freebindcraft_filters,
    freebindcraft_target,
    write_freebindcraft_configs,
)

from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.tools import launch_spec, load_configs, plan
from bindocracy.tools.freebindcraft.launch import DESIGNS_FILE


def flag(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


def rewrite(path: Path, **overrides) -> None:
    document = json.loads(path.read_text())
    for key, value in overrides.items():
        if value is None:
            document.pop(key, None)
        else:
            document[key] = value
    path.write_text(json.dumps(document, indent=2))


# --- the three documents ---------------------------------------------------


def test_a_valid_pair_loads(freebindcraft_configs) -> None:
    loaded = load_configs(*freebindcraft_configs)

    assert loaded.tool == "freebindcraft"
    assert loaded.preflight.binder_length == (65, 150)
    assert loaded.preflight.hotspot_string == ""


def test_all_three_documents_are_folded_into_the_stored_config(
    freebindcraft_configs,
) -> None:
    """A path is not a record of what a run asked for, or of what passed."""
    model = load_configs(*freebindcraft_configs).model

    assert model.target.contents["binder_name"] == "dio3_cut"
    assert model.advanced.contents["design_algorithm"] == "4stage"
    assert model.filters.contents["Average_pLDDT"]["threshold"] == 0.8


@pytest.mark.parametrize(
    "key, value",
    [
        ("design_path", "/somewhere/else"),
        ("number_of_final_designs", 40),
        ("target_hotspot_residues", "A56"),
    ],
)
def test_the_keys_the_driver_writes_may_not_be_authored(
    tmp_path: Path, key: str, value: object
) -> None:
    general, model = write_freebindcraft_configs(tmp_path)
    rewrite(tmp_path / "dio3_cut_target.json", **{key: value})

    with pytest.raises(ConfigPreflightError, match=key):
        load_configs(general, model)


def test_a_target_document_aimed_elsewhere_is_refused(tmp_path: Path) -> None:
    """It carries its own structure path, so it can drift and still validate."""
    other = tmp_path / "someone_else.pdb"
    other.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
    general, model = write_freebindcraft_configs(tmp_path)
    rewrite(tmp_path / "dio3_cut_target.json", starting_pdb=str(other))

    with pytest.raises(ConfigPreflightError, match="not the campaign target"):
        load_configs(general, model)


def test_a_relative_starting_pdb_is_refused(tmp_path: Path) -> None:
    general, model = write_freebindcraft_configs(tmp_path)
    rewrite(tmp_path / "dio3_cut_target.json", starting_pdb="./target.pdb")

    with pytest.raises(ConfigPreflightError, match="relative path"):
        load_configs(general, model)


def test_a_target_document_naming_another_chain_is_refused(tmp_path: Path) -> None:
    general, model = write_freebindcraft_configs(tmp_path)
    rewrite(tmp_path / "dio3_cut_target.json", chains="B")

    with pytest.raises(ConfigPreflightError, match="chain 'A'"):
        load_configs(general, model)


def test_an_advanced_profile_with_its_own_budget_is_refused(tmp_path: Path) -> None:
    """Two answers to how long a task runs is worse than none."""
    general, model = write_freebindcraft_configs(
        tmp_path, advanced=freebindcraft_advanced(max_trajectories=40)
    )

    with pytest.raises(ConfigPreflightError, match="max_trajectories"):
        load_configs(general, model)


def test_an_advanced_profile_without_mpnn_is_refused(tmp_path: Path) -> None:
    """With MPNN off there is no design table, and so nothing to collect."""
    general, model = write_freebindcraft_configs(
        tmp_path, advanced=freebindcraft_advanced(enable_mpnn=False)
    )

    with pytest.raises(ConfigPreflightError, match="disable MPNN"):
        load_configs(general, model)


def test_a_campaign_without_a_pdb_is_refused(tmp_path: Path) -> None:
    general, model = write_freebindcraft_configs(tmp_path)
    document = yaml.safe_load(general.read_text())
    document["target"].pop("structure_pdb")
    general.write_text(yaml.safe_dump(document))

    with pytest.raises(ConfigPreflightError, match="target.structure_pdb"):
        load_configs(general, model)


# --- filters that cannot bite ----------------------------------------------


def test_a_threshold_on_a_placeholder_metric_is_recorded_as_inert(
    freebindcraft_configs, tmp_path: Path
) -> None:
    """`Average_dG` is a constant without PyRosetta: it measures nothing.

    The filter set is not refused -- every file the image ships thresholds at
    least one of these, so refusing would leave none usable -- but a run whose
    n_passed was decided partly by constants has to say so.
    """
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")

    assert manifest.workflow["inert_filters"] == ["dG"]
    assert "i_pTM" in manifest.workflow["active_filters"]
    assert manifest.workflow["pyrosetta"] is False


def test_a_filter_set_with_no_placeholders_is_wholly_enforced(tmp_path: Path) -> None:
    general, model = write_freebindcraft_configs(
        tmp_path,
        filters={"Average_i_pTM": {"threshold": 0.5, "higher": True}},
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")

    assert manifest.workflow["active_filters"] == ["i_pTM"]
    assert manifest.workflow["inert_filters"] == []


def test_a_null_threshold_is_not_a_filter(tmp_path: Path) -> None:
    general, model = write_freebindcraft_configs(
        tmp_path,
        filters=freebindcraft_filters(Average_i_pTM={"threshold": None, "higher": True}),
    )

    assert "i_pTM" not in load_configs(general, model).preflight.active_filters


# --- the plan ---------------------------------------------------------------


def test_the_plan_separates_designs_from_the_trajectory_budget(
    freebindcraft_configs, tmp_path: Path
) -> None:
    """Designs are a stopping condition; trajectories are what a task spends."""
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    task = manifest.tasks[0]

    assert task.n_requested == 2
    assert task.n_generated == 4
    assert task.designs == f"tasks/0000/{DESIGNS_FILE}"


def test_the_run_records_that_it_cannot_be_reproduced(
    freebindcraft_configs, tmp_path: Path
) -> None:
    """BindCraft draws every seed from numpy's unseeded global RNG."""
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")

    assert manifest.workflow["reproducible"] is False


def test_every_consumed_document_is_archived(
    freebindcraft_configs, tmp_path: Path
) -> None:
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")

    assert sorted(manifest.provenance) == ["advanced", "driver", "filters", "target"]
    assert list(manifest.inputs) == ["freebindcraft_target_pdb"]


# --- one task's command -----------------------------------------------------


def test_the_command_runs_the_archived_driver(
    freebindcraft_configs, tmp_path: Path
) -> None:
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    spec = launch_spec(manifest, 0)

    assert spec.argv[:4] == ("singularity", "exec", "--cleanenv", "--nv")
    assert flag(spec.argv, "--target-template").startswith(str(manifest.directory))
    assert flag(spec.argv, "--final-designs") == "2"
    assert flag(spec.argv, "--max-trajectories") == "4"
    assert flag(spec.argv, "--rank-by") == "i_pTM"
    assert "--no-plots" in spec.argv and "--no-animations" in spec.argv


def test_each_task_gets_its_own_design_path(
    tmp_path: Path,
) -> None:
    """Two tasks sharing one would resume each other, not race.

    The design loop skips any trajectory whose PDB already exists, so a second
    task pointed at the first one's directory finishes its leftovers and both
    report the same designs.
    """
    general, model = write_freebindcraft_configs(tmp_path, sampling={"jobs": 2})
    manifest = plan(load_configs(general, model), tmp_path / "run")

    paths = {
        flag(launch_spec(manifest, task_id).argv, "--design-path")
        for task_id in (0, 1)
    }
    assert len(paths) == 2


def test_scratch_is_node_local_and_the_gpu_is_forwarded(
    freebindcraft_configs, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    env = launch_spec(manifest, 0).env

    assert env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] == "3"
    assert env["TMPDIR"].startswith(str(tmp_path / "nodetmp"))


def test_resources_reach_slurm(freebindcraft_configs, tmp_path: Path) -> None:
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    resources = launch_spec(manifest, 0).resources

    assert resources["gres"] == "gpu:1"
    assert resources["cpus_per_task"] == 8
    assert resources["mem_mb"] == 64 * 1024
    assert resources["runtime"] == 240


# --- the driver's own contract ----------------------------------------------


def test_the_driver_accepts_what_the_launcher_builds(
    freebindcraft_driver, freebindcraft_configs, tmp_path: Path
) -> None:
    """The two never meet at runtime: one runs on a login node, one in the image."""
    manifest = plan(load_configs(*freebindcraft_configs), tmp_path / "run")
    argv = list(launch_spec(manifest, 0).argv)
    parsed = freebindcraft_driver.parse_args(argv[argv.index("bindcraft-python") + 2:])

    assert parsed.final_designs == 2
    assert parsed.max_trajectories == 4
    assert parsed.hotspots == ""
    assert parsed.plots is False


def test_the_driver_writes_the_three_keys_and_nothing_else(
    freebindcraft_driver, tmp_path: Path
) -> None:
    template = freebindcraft_target(tmp_path / "target.pdb")
    rendered = freebindcraft_driver.render_target(
        template, design_path="/task/bindcraft", final_designs=3, hotspots="A56,A57"
    )

    assert rendered["design_path"] == "/task/bindcraft"
    assert rendered["number_of_final_designs"] == 3
    assert rendered["target_hotspot_residues"] == "A56,A57"
    assert rendered["lengths"] == template["lengths"]
    # The template is not mutated: the archive is what the run executes.
    assert "design_path" not in template


def test_the_driver_writes_an_empty_epitope_rather_than_omitting_it(
    freebindcraft_driver, tmp_path: Path
) -> None:
    """An absent key is a KeyError; an empty one is `hotspot=None`, on purpose."""
    rendered = freebindcraft_driver.render_target(
        freebindcraft_target(tmp_path / "target.pdb"),
        design_path="/task", final_designs=1, hotspots="",
    )

    assert rendered["target_hotspot_residues"] == ""


def test_the_driver_gives_the_task_its_trajectory_budget(freebindcraft_driver) -> None:
    rendered = freebindcraft_driver.render_advanced(
        freebindcraft_advanced(), max_trajectories=7
    )

    assert rendered["max_trajectories"] == 7


def test_the_driver_drops_the_opencl_jits_noise(freebindcraft_driver) -> None:
    """One line per compiled kernel, thousands of them, burying the real log."""
    noise = "Failed to read file: /tmp/dep-1a2b3c.d\n"

    assert freebindcraft_driver._NOISE.search(noise)
    assert not freebindcraft_driver._NOISE.search("Starting trajectory: dio3_cut_l85\n")


def test_the_driver_refuses_a_descriptor_limit_it_cannot_raise(
    freebindcraft_driver, monkeypatch
) -> None:
    """Below 65,536 the run fails hours in, looking like something else."""
    monkeypatch.setattr(
        freebindcraft_driver.resource, "getrlimit", lambda _: (1024, 4096)
    )

    with pytest.raises(SystemExit, match="4096"):
        freebindcraft_driver.raise_open_files()


def test_the_target_is_the_one_the_campaign_names(freebindcraft_configs) -> None:
    loaded = load_configs(*freebindcraft_configs)

    assert loaded.preflight.target_sequence == FREEBINDCRAFT_TARGET
    assert loaded.preflight.target_length == 201
