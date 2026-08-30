"""Genie 3's config, its preflight, and the command one task runs.

The checks worth having here are the ones whose absence is silent: a problem
set that is not the campaign's target, a JAX overlay that is missing so the AF2
stage runs on the CPU forever, a folding mode that needs the network, and a
template that answers a question the harness has already answered.
"""

from __future__ import annotations

import shutil
from itertools import pairwise
from pathlib import Path

import pytest
import yaml
from conftest import (
    GENIE3_SELECTION,
    genie3_experiment,
    write_genie3_configs,
    write_genie3_problemset,
)
from pydantic import ValidationError

from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.tools import launch_spec, load_configs, plan, resources
from bindocracy.tools.genie3.config import Genie3Config
from bindocracy.tools.genie3.launch import CONTAINER_LIGHTNING_LOGS, CONTAINER_PYTHON


def flag(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


def flags(argv: tuple[str, ...], name: str) -> list[str]:
    return [value for option, value in pairwise(argv) if option == name]


# --- configuration ---------------------------------------------------------


def test_a_genie3_config_loads_and_keeps_its_experiment(genie3_configs) -> None:
    loaded = load_configs(*genie3_configs)

    assert isinstance(loaded.model, Genie3Config)
    assert loaded.tool == "genie3"
    # The template is folded in, so the stored config says what the run asked
    # for rather than where the question was written down.
    assert loaded.model.experiment.contents == loaded.preflight.experiment
    assert loaded.model.experiment.contents["evaluation"]["version"] == "binder"


def test_a_recovered_config_carrying_its_experiment_still_validates(genie3_configs) -> None:
    loaded = load_configs(*genie3_configs)
    stored = loaded.model.model_dump(mode="json")

    assert Genie3Config.model_validate(stored).experiment.contents is not None


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    general, model = write_genie3_configs(tmp_path)
    document = yaml.safe_load(model.read_text())
    document["sampling"]["designs_per_job"] = 4
    model.write_text(yaml.safe_dump(document))

    with pytest.raises(ValidationError):
        load_configs(general, model)


# --- preflight -------------------------------------------------------------


def test_a_problem_set_for_another_protein_is_refused(tmp_path: Path) -> None:
    """The silent-wrong-answer case: everything else about the run is fine."""
    dataset = write_genie3_problemset(tmp_path, sequence="MMMMMMMMMMMMMMMM")
    general, model = write_genie3_configs(
        tmp_path, experiment=genie3_experiment(dataset)
    )

    with pytest.raises(ConfigPreflightError, match="not the campaign target"):
        load_configs(general, model)


def test_a_missing_jax_overlay_is_refused(tmp_path: Path) -> None:
    """Without it the AF2 stage runs on the CPU and never finishes, silently."""
    general, model = write_genie3_configs(tmp_path)
    shutil.rmtree(tmp_path / "overlays" / "jax" / "jax_plugins")

    with pytest.raises(ConfigPreflightError, match="run on the CPU"):
        load_configs(general, model)


@pytest.mark.parametrize("marker", ["cudnn/libcudnn.so.9", "nvcc/bin/ptxas"])
def test_each_overlay_is_checked_by_a_file_that_proves_it_built(
    tmp_path: Path, marker: str
) -> None:
    general, model = write_genie3_configs(tmp_path)
    (tmp_path / "overlays" / marker).unlink()

    with pytest.raises(ConfigPreflightError, match="run on the CPU"):
        load_configs(general, model)


def test_a_folding_mode_that_needs_the_network_is_refused(tmp_path: Path) -> None:
    dataset = write_genie3_problemset(tmp_path)
    experiment = genie3_experiment(dataset)
    experiment["evaluation"]["folding"]["mode"] = "msa"
    general, model = write_genie3_configs(tmp_path, experiment=experiment)

    with pytest.raises(ConfigPreflightError, match="api.colabfold.com"):
        load_configs(general, model)


def test_a_template_with_no_evaluation_stage_is_refused(tmp_path: Path) -> None:
    """Generation alone produces UNK backbones and no sequences at all."""
    dataset = write_genie3_problemset(tmp_path)
    experiment = genie3_experiment(dataset)
    del experiment["evaluation"]
    general, model = write_genie3_configs(tmp_path, experiment=experiment)

    with pytest.raises(ConfigPreflightError, match="no sequences"):
        load_configs(general, model)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("paths", "rootdir", "/tmp/elsewhere"),
        ("paths", "outdir", "/tmp/elsewhere"),
        ("experiment", "seed", 7),
    ],
)
def test_a_template_cannot_answer_what_the_harness_answers(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    dataset = write_genie3_problemset(tmp_path)
    experiment = genie3_experiment(dataset)
    experiment[section][key] = value
    general, model = write_genie3_configs(tmp_path, experiment=experiment)

    with pytest.raises(ConfigPreflightError, match=f"{section}.{key}"):
        load_configs(general, model)


def test_a_template_cannot_set_its_own_sample_count(tmp_path: Path) -> None:
    dataset = write_genie3_problemset(tmp_path)
    experiment = genie3_experiment(dataset)
    experiment["generation"]["dataset"]["n_sample"] = 40
    general, model = write_genie3_configs(tmp_path, experiment=experiment)

    with pytest.raises(ConfigPreflightError, match="generation.dataset.n_sample"):
        load_configs(general, model)


def test_a_problem_set_with_no_hotspots_is_refused(tmp_path: Path) -> None:
    """Genie 3's reducer reads them unconditionally; there is no free mode."""
    dataset = write_genie3_problemset(tmp_path, hotspots=[])
    general, model = write_genie3_configs(tmp_path, experiment=genie3_experiment(dataset))

    with pytest.raises(ConfigPreflightError, match="declares no hotspots"):
        load_configs(general, model)


def test_a_problem_set_conditioning_on_another_epitope_is_refused(tmp_path: Path) -> None:
    """The problem set renumbers residues, so only the numbers can be compared."""
    dataset = write_genie3_problemset(tmp_path, hotspots=["B10", "B12", "B13"])
    general, model = write_genie3_configs(tmp_path, experiment=genie3_experiment(dataset))
    document = yaml.safe_load(general.read_text())
    document["target"]["hotspots"] = ["A10", "A12", "A99"]
    general.write_text(yaml.safe_dump(document))

    with pytest.raises(ConfigPreflightError, match="conditions on residues"):
        load_configs(general, model)


def test_a_renamed_chain_is_not_a_different_epitope(tmp_path: Path) -> None:
    """A45 in the campaign and B45 in the problem set are the same residue."""
    dataset = write_genie3_problemset(tmp_path, hotspots=["B10", "B12", "B13"])
    general, model = write_genie3_configs(tmp_path, experiment=genie3_experiment(dataset))
    document = yaml.safe_load(general.read_text())
    document["target"]["hotspots"] = ["A13", "A10", "A12"]
    general.write_text(yaml.safe_dump(document))

    assert load_configs(general, model).preflight.hotspots == ("B10", "B12", "B13")


def test_a_relative_dataset_path_is_refused(tmp_path: Path) -> None:
    """Preflight and the container resolve it against different directories."""
    dataset = write_genie3_problemset(tmp_path)
    experiment = genie3_experiment(dataset)
    experiment["paths"]["dataset"] = "genie3_dataset"
    general, model = write_genie3_configs(tmp_path, experiment=experiment)

    with pytest.raises(ConfigPreflightError, match="absolute"):
        load_configs(general, model)


def test_the_overlays_are_recorded_by_version_not_only_by_path(tmp_path: Path) -> None:
    """A path says where a build was, not which build it was."""
    general, model = write_genie3_configs(tmp_path)
    (tmp_path / "overlays" / "jax" / "jax_cuda12_plugin-0.6.2.dist-info").mkdir()
    manifest = plan(load_configs(general, model), tmp_path / "run")

    overlays = manifest.workflow["jax_overlays"]
    assert overlays["plugin"]["packages"] == {"jax_cuda12_plugin": "0.6.2"}
    assert set(overlays) == {"plugin", "cudnn", "cuda_nvcc"}
    assert overlays["cudnn"]["path"].endswith("overlays/cudnn")


def test_more_than_one_selection_is_refused(tmp_path: Path) -> None:
    dataset = write_genie3_problemset(tmp_path)
    experiment = genie3_experiment(dataset)
    experiment["generation"]["dataset"]["selections"] = f"{GENIE3_SELECTION},other"
    general, model = write_genie3_configs(tmp_path, experiment=experiment)

    with pytest.raises(ConfigPreflightError, match="names 2 problems"):
        load_configs(general, model)


# --- planning --------------------------------------------------------------


def test_designs_are_backbones_times_sequences_per_backbone(tmp_path: Path) -> None:
    """The counting trap: 4 backbones at num_seq 3 is 12 designs, not 4."""
    dataset = write_genie3_problemset(tmp_path)
    experiment = genie3_experiment(dataset)
    experiment["evaluation"]["inverse_folding"]["num_seq"] = 3
    general, model = write_genie3_configs(
        tmp_path, experiment=experiment,
        sampling={"jobs": 2, "backbones_per_job": 4, "seed_base": 0},
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")

    assert manifest.designs_per_task == 12
    assert manifest.to_run_record().n_requested == 24
    assert manifest.workflow["backbones_per_task"] == 4
    assert manifest.workflow["sequences_per_backbone"] == 3


def test_the_plan_names_the_results_table_the_adapter_reads(genie3_configs, tmp_path) -> None:
    manifest = plan(load_configs(*genie3_configs), tmp_path / "run")

    assert manifest.tasks[0].designs == (
        f"tasks/0000/{GENIE3_SELECTION}/results/info.csv"
    )


def test_the_plan_records_what_only_the_problem_set_knows(genie3_configs, tmp_path) -> None:
    """Binder length and epitope are in the problem JSON, not in any config."""
    manifest = plan(load_configs(*genie3_configs), tmp_path / "run")

    assert manifest.workflow["binder_min_length"] == 60
    assert manifest.workflow["binder_max_length"] == 120
    assert manifest.workflow["hotspots"] == ["B10", "B12", "B13"]
    assert manifest.workflow["cond_strategy"] == "extended"


def test_the_run_digests_the_problem_set_it_read(genie3_configs, tmp_path) -> None:
    manifest = plan(load_configs(*genie3_configs), tmp_path / "run")

    assert set(manifest.inputs) == {
        "genie3_problem", "genie3_target_pdb", "genie3_target_fasta",
        "genie3_target_pdb_chain_0",
    }
    manifest.verify_inputs()


def test_editing_the_problem_set_stops_a_planned_run(genie3_configs, tmp_path) -> None:
    loaded = load_configs(*genie3_configs)
    manifest = plan(loaded, tmp_path / "run")
    problem = Path(manifest.inputs["genie3_problem"].uri)
    problem.write_text(problem.read_text().replace('"binder_min_length": 60',
                                                   '"binder_min_length": 90'))

    with pytest.raises(Exception, match="no longer match"):
        manifest.verify_inputs()


def test_both_the_template_and_the_driver_are_archived(genie3_configs, tmp_path) -> None:
    manifest = plan(load_configs(*genie3_configs), tmp_path / "run")

    assert set(manifest.provenance) == {"experiment", "driver"}
    for archived in manifest.provenance.values():
        assert manifest.path(archived.path).is_file()


# --- launch ----------------------------------------------------------------


def test_one_task_runs_the_archived_driver_on_the_archived_template(
    genie3_configs, tmp_path
) -> None:
    loaded = load_configs(*genie3_configs)
    manifest = plan(loaded, tmp_path / "run")
    spec = launch_spec(manifest, 0)

    assert spec.argv[:4] == ("singularity", "exec", "--cleanenv", "--nv")
    assert CONTAINER_PYTHON in spec.argv
    assert flag(spec.argv, "--template") == str(
        manifest.path(manifest.provenance["experiment"].path)
    )
    assert spec.argv[spec.argv.index(CONTAINER_PYTHON) + 1] == str(
        manifest.path(manifest.provenance["driver"].path)
    )
    assert flag(spec.argv, "--rootdir") == str(manifest.path("tasks/0000"))
    assert flag(spec.argv, "--n-sample") == "4"


def test_each_task_diffuses_from_its_own_seed(tmp_path: Path) -> None:
    """Two tasks on one seed would generate the same backbones twice."""
    general, model = write_genie3_configs(
        tmp_path, sampling={"jobs": 3, "backbones_per_job": 2, "seed_base": 100}
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")

    seeds = [flag(launch_spec(manifest, task).argv, "--seed") for task in range(3)]
    assert seeds == ["100", "101", "102"]


def test_the_container_gets_the_three_overlays(genie3_configs, tmp_path) -> None:
    loaded = load_configs(*genie3_configs)
    spec = launch_spec(plan(loaded, tmp_path / "run"), 0)
    runtime = loaded.model.runtime

    assert spec.env["SINGULARITYENV_PYTHONPATH"] == str(runtime.jax_plugin_overlay)
    # cuDNN goes on LD_LIBRARY_PATH, never PYTHONPATH: an overlay copy there
    # would shadow the whole `nvidia` package, not only libcudnn.
    assert spec.env["SINGULARITYENV_LD_LIBRARY_PATH"] == str(runtime.cudnn_overlay)
    assert spec.env["SINGULARITYENV_XLA_FLAGS"] == (
        f"--xla_gpu_cuda_data_dir={runtime.cuda_nvcc_overlay}"
    )
    assert spec.env["SINGULARITYENV_PATH"].startswith(f"{runtime.cuda_nvcc_overlay}/bin:")


def test_writable_space_is_bound_over_the_read_only_lightning_logs(
    genie3_configs, tmp_path
) -> None:
    """Lightning makedirs into /opt/genie3 and the run dies before generating."""
    manifest = plan(load_configs(*genie3_configs), tmp_path / "run")
    spec = launch_spec(manifest, 0)
    lightning = manifest.path("tasks/0000/lightning_logs")

    assert f"{lightning}:{CONTAINER_LIGHTNING_LOGS}" in flags(spec.argv, "--bind")
    # The bind source has to exist before singularity starts.
    assert lightning in spec.mkdirs


def test_nothing_is_bound_that_the_site_already_binds(genie3_configs, tmp_path) -> None:
    """`/n` is a system bind path here, so the only bind is the one that must
    put writable space over a path inside the image."""
    manifest = plan(load_configs(*genie3_configs), tmp_path / "run")
    bound = flags(launch_spec(manifest, 0).argv, "--bind")

    lightning = manifest.path("tasks/0000/lightning_logs")
    assert bound == [f"{lightning}:{CONTAINER_LIGHTNING_LOGS}"]


def test_tmpdir_is_node_local_and_unique_per_task(genie3_configs, tmp_path) -> None:
    """TMPDIR on Lustre kills any Triton-JIT tool with Errno 39."""
    manifest = plan(load_configs(*genie3_configs), tmp_path / "run")
    specs = [launch_spec(manifest, 0)]

    assert specs[0].env["TMPDIR"] == specs[0].env["SINGULARITYENV_TMPDIR"]
    assert manifest.run_id[:8] in specs[0].env["TMPDIR"]
    assert Path(specs[0].env["TMPDIR"]) in specs[0].mkdirs


def test_the_allocated_device_survives_cleanenv(genie3_configs, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    manifest = plan(load_configs(*genie3_configs), tmp_path / "run")

    assert launch_spec(manifest, 0).env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] == "3"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert "SINGULARITYENV_CUDA_VISIBLE_DEVICES" not in launch_spec(manifest, 0).env


def test_resources_come_from_the_config(genie3_configs) -> None:
    loaded = load_configs(*genie3_configs)

    assert resources(loaded) == {
        "slurm_account": "test-account",
        "slurm_partition": "test-gpu",
        "gres": "gpu:1",
        "cpus_per_task": 16,
        "mem_mb": 96 * 1024,
        "runtime": 8 * 60,
    }


def test_the_launch_reads_the_manifest_not_the_edited_yaml(
    genie3_configs, tmp_path
) -> None:
    loaded = load_configs(*genie3_configs)
    manifest = plan(loaded, tmp_path / "run")
    document = yaml.safe_load(Path(genie3_configs[1]).read_text())
    document["sampling"]["backbones_per_job"] = 999
    Path(genie3_configs[1]).write_text(yaml.safe_dump(document))

    assert flag(launch_spec(manifest, 0).argv, "--n-sample") == "4"
