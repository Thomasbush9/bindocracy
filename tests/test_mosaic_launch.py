from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bindocracy.tools import launch_spec, load_configs, plan


@pytest.fixture
def planned(configs, tmp_path: Path):
    loaded = load_configs(*configs)
    return loaded, plan(loaded, tmp_path / "run")


def test_argv_passes_every_run_dependent_value(planned) -> None:
    loaded, manifest = planned

    spec = launch_spec(manifest, 1)

    assert spec.argv[:2] == (str(loaded.model.runtime.exec_wrapper), "python")
    flags = dict(zip(spec.argv[3::2], spec.argv[4::2], strict=True))
    assert flags == {
        "--target-fasta": str(loaded.general.target.sequence_fasta),
        "--target-msa": str(loaded.general.target.msa),
        "--binder-length": "70",
        "--task-id": "1",
        "--seed-base": "0",
        "--n-designs": "4",
        "--max-runtime": "1.0",
        "--save-dir": str(manifest.directory / "tasks" / "0001"),
        "--soft-steps": "100",
        "--sharpen-steps": "50",
        "--final-steps": "15",
        # Always passed, and empty when the campaign names no epitope: the
        # driver reads that as the loss with the whole target as its contact
        # partner, which is a different objective rather than a missing one.
        "--epitope": "",
    }


def test_a_shortened_schedule_reaches_the_driver(configs, tmp_path: Path) -> None:
    general_path, model_path = configs
    raw = yaml.safe_load(model_path.read_text())
    raw["sampling"]["optimizer"] = {"soft_steps": 10, "sharpen_steps": 5, "final_steps": 2}
    model_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    loaded = load_configs(general_path, model_path)

    argv = launch_spec(plan(loaded, tmp_path / "run"), 0).argv

    flags = dict(zip(argv[3::2], argv[4::2], strict=True))
    assert flags["--soft-steps"] == "10"
    assert flags["--sharpen-steps"] == "5"
    assert flags["--final-steps"] == "2"


def test_it_runs_the_archived_driver_not_the_authored_one(planned) -> None:
    loaded, manifest = planned

    driver = Path(launch_spec(manifest, 0).argv[2])

    assert driver == manifest.directory / "provenance" / "hallucinate_binders.py"
    assert driver != loaded.model.driver.script


def test_a_legacy_manifest_does_not_receive_the_new_epitope_flag(planned) -> None:
    """Its archived driver predates --epitope and would reject the argument."""
    _, manifest = planned
    workflow = {key: value for key, value in manifest.workflow.items() if key != "epitope_idx"}
    legacy = manifest.model_copy(update={"workflow": workflow})

    assert "--epitope" not in launch_spec(legacy, 0).argv


def test_environment_carries_what_the_wrapper_reads(planned) -> None:
    loaded, manifest = planned

    env = launch_spec(manifest, 0).env

    assert env["MOSAIC_SIF"] == str(loaded.model.runtime.container)
    assert env["MOSAIC_WEIGHTS"] == str(loaded.model.runtime.weights)
    assert env["MOSAIC_SCRATCH"] == str(loaded.model.runtime.scratch)
    assert env["SINGULARITYENV_SSL_CERT_FILE"].endswith("ca-certificates.crt")


def test_resources_come_from_the_validated_configs(planned) -> None:
    _, manifest = planned

    resources = launch_spec(manifest, 0).resources

    assert resources == {
        "slurm_account": "test-account",
        "slurm_partition": "test-gpu",
        "gres": "gpu:1",
        "cpus_per_task": 8,
        "mem_mb": 32 * 1024,
        "runtime": 120,
    }


def test_expected_outputs_and_log_are_inside_the_run(planned) -> None:
    _, manifest = planned

    spec = launch_spec(manifest, 0)

    assert spec.outputs == (
        manifest.directory / "tasks" / "0000" / "designs.jsonl",
        manifest.directory / "tasks" / "0000" / "status.json",
    )
    assert spec.log == manifest.directory / "logs" / "task-0000.log"


def test_command_is_shell_safe(planned) -> None:
    _, manifest = planned

    command = launch_spec(manifest, 0).command

    assert command.startswith("env MOSAIC_SCRATCH=")
    assert ";" not in command and "&&" not in command


def test_unknown_task_is_an_error(planned) -> None:
    _, manifest = planned

    with pytest.raises(KeyError, match="no task 9"):
        launch_spec(manifest, 9)
