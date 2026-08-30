"""The connector/driver seam for Genie 3.

Nothing else covers it: the connector builds an argument vector on the login
node and the driver parses it inside a container, so the two can drift apart
while every other test still passes. The bug is only visible once a GPU job has
started. These tests parse the real connector's argv with the real driver's
parser, and check that what the driver writes is where the adapter looks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import genie3_experiment, write_genie3_problemset

from bindocracy.tools import launch_spec, load_configs, plan
from bindocracy.tools.genie3.adapter import RENDERED_CONFIG, selection_of
from bindocracy.tools.genie3.launch import CONTAINER_PYTHON


def parse(driver, argv: tuple[str, ...]):
    """Parse the connector's argv with the driver's own parser."""
    return driver.parse_args(list(argv[argv.index(CONTAINER_PYTHON) + 2 :]))


def test_the_driver_accepts_exactly_what_the_connector_launches(
    genie3_driver, genie3_configs, tmp_path: Path
) -> None:
    loaded = load_configs(*genie3_configs)
    manifest = plan(loaded, tmp_path / "run")
    spec = launch_spec(manifest, 0)

    # argparse exits non-zero on an unknown or missing flag, so this failing
    # means the connector would launch a job that dies immediately.
    parsed = parse(genie3_driver, spec.argv)

    assert Path(parsed.template) == manifest.path(manifest.provenance["experiment"].path)
    assert parsed.n_sample == loaded.model.sampling.backbones_per_job
    assert parsed.seed == loaded.model.sampling.seed_base
    assert parsed.num_devices == loaded.model.resources.gpus


def test_the_driver_writes_into_the_directory_the_adapter_reads(
    genie3_driver, genie3_configs, tmp_path: Path
) -> None:
    loaded = load_configs(*genie3_configs)
    manifest = plan(loaded, tmp_path / "run")
    task = manifest.tasks[0]
    parsed = parse(genie3_driver, launch_spec(manifest, 0).argv)

    rootdir = Path(parsed.rootdir)
    assert rootdir == manifest.path(task.directory)
    # Genie 3 writes <rootdir>/<selection>/results/info.csv, which is what the
    # manifest planned as this task's designs file.
    assert rootdir / selection_of(task) / "results" / "info.csv" == manifest.path(task.designs)
    assert Path(parsed.config_out) == manifest.path(f"{task.directory}/{RENDERED_CONFIG}")


def test_rendering_sets_the_three_keys_the_harness_owns(genie3_driver, tmp_path: Path) -> None:
    template = genie3_experiment(write_genie3_problemset(tmp_path))
    rendered = genie3_driver.render(template, rootdir="/runs/0000", seed=7, n_sample=3)

    assert rendered["paths"]["rootdir"] == "/runs/0000"
    assert rendered["experiment"]["seed"] == 7
    assert rendered["generation"]["dataset"]["n_sample"] == 3


def test_rendering_changes_nothing_else(genie3_driver, tmp_path: Path) -> None:
    template = genie3_experiment(write_genie3_problemset(tmp_path))
    original = yaml.safe_dump(template, sort_keys=True)
    rendered = genie3_driver.render(template, rootdir="/runs/0000", seed=7, n_sample=3)

    # The authored document is not mutated, and every authored key survives.
    assert yaml.safe_dump(template, sort_keys=True) == original
    assert rendered["evaluation"] == template["evaluation"]
    assert rendered["generation"]["dataset"]["cond_strategy"] == "extended"
    assert rendered["paths"]["dataset"] == template["paths"]["dataset"]


def test_an_empty_parameter_cache_is_refused(genie3_driver, tmp_path: Path, monkeypatch) -> None:
    """Colabfold would otherwise download 4 GB on a node with no route out."""
    monkeypatch.setattr(genie3_driver, "CONTAINER_PARAMS", tmp_path / "absent")

    with pytest.raises(SystemExit, match="AF2 parameters"):
        genie3_driver.prepare_cache(tmp_path / "cache")


def test_the_cache_points_at_the_parameters_in_the_image(
    genie3_driver, tmp_path: Path, monkeypatch
) -> None:
    params = tmp_path / "image-params"
    params.mkdir()
    (params / genie3_driver.PARAMS_MARKER).write_bytes(b"fixture")
    monkeypatch.setattr(genie3_driver, "CONTAINER_PARAMS", params)

    linked = genie3_driver.prepare_cache(tmp_path / "cache")

    assert linked == tmp_path / "cache" / "colabfold" / "params"
    assert linked.is_symlink() and linked.resolve() == params
    # Re-running a task must not trip over the link it made last time.
    assert genie3_driver.prepare_cache(tmp_path / "cache") == linked


def test_the_driver_runs_genie3_from_the_repository_root(genie3_driver) -> None:
    """ProteinMPNN, IPSAE, TM-align and DSSP all have repo-relative paths."""
    assert str(genie3_driver.REPO_ROOT) == "/opt/genie3"


def test_the_command_runs_the_all_in_one_path(genie3_driver, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(genie3_driver.shutil, "which", lambda name: f"/opt/bin/{name}")
    command = genie3_driver.genie3_command(tmp_path / "experiment.yaml", tmp_path / "logs", 2)

    assert command[:2] == ["/opt/bin/genie3", "run"]
    # --log-dir is effectively mandatory: the default is under the read-only
    # /opt/genie3.
    assert "--log-dir" in command
    assert command[command.index("--num-devices") + 1] == "2"


def test_a_container_without_genie3_says_so(genie3_driver, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(genie3_driver.shutil, "which", lambda name: None)

    with pytest.raises(SystemExit, match="genie3 is not on PATH"):
        genie3_driver.genie3_command(tmp_path / "experiment.yaml", tmp_path / "logs", 1)
