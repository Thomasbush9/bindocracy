from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bindocracy.config import GeneralConfig, MosaicConfig, load_mosaic_configs
from bindocracy.runs import ManifestError, RunManifest
from bindocracy.tools import plan


def test_plan_creates_the_run_layout(configs, tmp_path: Path) -> None:
    loaded = load_mosaic_configs(*configs)

    manifest = plan(loaded, tmp_path / "run", name="config_01")

    run_dir = manifest.directory
    assert (run_dir / "run.json").is_file()
    assert [task.task_id for task in manifest.tasks] == [0, 1]
    assert manifest.designs_per_task == 4
    assert manifest.name == "config_01"
    assert manifest.to_run_record().n_requested == 8
    for relative in ("provenance/hallucinate_binders.py", "tasks/0000", "tasks/0001",
                     "logs"):
        assert (run_dir / relative).exists()


def test_only_the_executed_driver_is_archived(configs, tmp_path: Path) -> None:
    """Configs live in the database, so copying the YAML would be a third copy.

    The driver is different in kind: it is executed, and its content is in no
    other store.
    """
    loaded = load_mosaic_configs(*configs)

    manifest = plan(loaded, tmp_path / "run")

    assert list(manifest.provenance) == ["driver"]
    assert list((manifest.directory / "provenance").iterdir()) == [
        manifest.path("provenance/hallucinate_binders.py")
    ]


def test_the_manifest_carries_the_whole_config(configs, tmp_path: Path) -> None:
    """A run is replayable from run.json alone, with no file on the share."""
    loaded = load_mosaic_configs(*configs)

    manifest = plan(loaded, tmp_path / "run")

    rebuilt = MosaicConfig.model_validate(manifest.config.model_config_json)
    assert rebuilt == loaded.model
    assert GeneralConfig.model_validate(manifest.config.general_config_json) == loaded.general


def test_archived_driver_is_a_copy_not_a_reference(configs, tmp_path: Path) -> None:
    loaded = load_mosaic_configs(*configs)
    manifest = plan(loaded, tmp_path / "run")
    archived = manifest.path(manifest.provenance["driver"].path)
    original = manifest.provenance["driver"].source_uri

    before = archived.read_text()
    Path(original).write_text("# the working tree moved on\n")

    assert archived.read_text() == before


def test_replanning_reuses_the_run_id_and_inputs(configs, tmp_path: Path) -> None:
    loaded = load_mosaic_configs(*configs)
    first = plan(loaded, tmp_path / "run")

    second = plan(load_mosaic_configs(*configs), tmp_path / "run")

    assert second.run_id == first.run_id
    assert second.provenance == first.provenance
    assert second.created_at == first.created_at


def test_replanning_a_different_config_in_the_same_directory_fails(
    configs, tmp_path: Path
) -> None:
    plan(load_mosaic_configs(*configs), tmp_path / "run")

    general_path, model_path = configs
    raw = yaml.safe_load(model_path.read_text())
    raw["sampling"]["binder_length"] = 90
    model_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ManifestError, match="was planned for model_config_id"):
        plan(load_mosaic_configs(general_path, model_path), tmp_path / "run")


def test_the_same_config_planned_twice_is_two_runs(configs, tmp_path: Path) -> None:
    loaded = load_mosaic_configs(*configs)

    first = plan(loaded, tmp_path / "run-a")
    second = plan(loaded, tmp_path / "run-b")

    assert first.model_config_id == second.model_config_id
    assert first.run_id != second.run_id


def test_manifest_round_trips_through_json(configs, tmp_path: Path) -> None:
    manifest = plan(load_mosaic_configs(*configs), tmp_path / "run")

    assert RunManifest.read(manifest.directory / "run.json") == manifest
