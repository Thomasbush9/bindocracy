from __future__ import annotations

from pathlib import Path

import pytest

from bindocracy.config import load_mosaic_configs
from bindocracy.runs import mosaic_launch_spec, plan_mosaic_run


@pytest.fixture
def planned(configs, tmp_path: Path):
    loaded = load_mosaic_configs(*configs)
    return loaded, plan_mosaic_run(loaded, tmp_path / "run")


def test_argv_passes_every_run_dependent_value(planned) -> None:
    loaded, manifest = planned

    spec = mosaic_launch_spec(loaded, manifest, 1)

    assert spec.argv[:2] == (str(loaded.mosaic.runtime.exec_wrapper), "python")
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
    }


def test_it_runs_the_archived_driver_not_the_authored_one(planned) -> None:
    loaded, manifest = planned

    driver = Path(mosaic_launch_spec(loaded, manifest, 0).argv[2])

    assert driver == manifest.directory / "provenance" / "hallucinate_binders.py"
    assert driver != loaded.mosaic.driver.script


def test_environment_carries_what_the_wrapper_reads(planned) -> None:
    loaded, manifest = planned

    env = mosaic_launch_spec(loaded, manifest, 0).env

    assert env["MOSAIC_SIF"] == str(loaded.mosaic.runtime.container)
    assert env["MOSAIC_WEIGHTS"] == str(loaded.mosaic.runtime.weights)
    assert env["MOSAIC_SCRATCH"] == str(loaded.mosaic.runtime.scratch)
    assert env["SINGULARITYENV_SSL_CERT_FILE"].endswith("ca-certificates.crt")


def test_resources_come_from_the_validated_configs(planned) -> None:
    loaded, manifest = planned

    resources = mosaic_launch_spec(loaded, manifest, 0).resources

    assert resources == {
        "slurm_account": "test-account",
        "slurm_partition": "test-gpu",
        "slurm_extra": "--gres=gpu:1",
        "cpus_per_task": 8,
        "mem_mb": 32 * 1024,
        "runtime": 120,
    }


def test_expected_outputs_and_log_are_inside_the_run(planned) -> None:
    loaded, manifest = planned

    spec = mosaic_launch_spec(loaded, manifest, 0)

    assert spec.outputs == (
        manifest.directory / "tasks" / "0000" / "designs.jsonl",
        manifest.directory / "tasks" / "0000" / "status.json",
    )
    assert spec.log == manifest.directory / "logs" / "task-0000.log"


def test_command_is_shell_safe(planned) -> None:
    loaded, manifest = planned

    command = mosaic_launch_spec(loaded, manifest, 0).command

    assert command.startswith("env MOSAIC_SCRATCH=")
    assert ";" not in command and "&&" not in command


def test_unknown_task_is_an_error(planned) -> None:
    loaded, manifest = planned

    with pytest.raises(KeyError, match="no task 9"):
        mosaic_launch_spec(loaded, manifest, 9)
