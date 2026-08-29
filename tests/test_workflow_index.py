"""The workflow index: named executions, not filenames.

A run directory used to be the config's filename stem, which meant the
filename decided run identity. Re-running a config required copying it, two
tools could not share a filename, and `run_10.yaml`/`run_11.yaml` accumulated
purely to obtain fresh directories.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bindocracy.config import ConfigLoadError
from bindocracy.runs.index import WorkflowIndex, load_workflow_index, read_workflow_index


def index_mapping(tmp_path: Path, runs: list[dict]) -> dict:
    return {
        "database": str(tmp_path / "campaign.duckdb"),
        "run_root": str(tmp_path / "runs"),
        "general_config": str(tmp_path / "general.yaml"),
        "runs": runs,
    }


def test_an_execution_names_itself(tmp_path: Path) -> None:
    index = read_workflow_index(index_mapping(tmp_path, [
        {"name": "mosaic-run-12", "config": str(tmp_path / "mosaic.yaml")},
    ]))

    assert index.names == ("mosaic-run-12",)
    assert index.run_dir("mosaic-run-12") == tmp_path / "runs" / "mosaic-run-12"
    assert index.config_for("mosaic-run-12") == tmp_path / "mosaic.yaml"


def test_one_config_can_be_executed_twice_under_two_names(tmp_path: Path) -> None:
    """The property the filename scheme could not express."""
    same = str(tmp_path / "mosaic.yaml")
    index = read_workflow_index(index_mapping(tmp_path, [
        {"name": "mosaic-run-12", "config": same},
        {"name": "mosaic-run-13", "config": same},
    ]))

    assert index.names == ("mosaic-run-12", "mosaic-run-13")
    assert index.run_dir("mosaic-run-12") != index.run_dir("mosaic-run-13")
    assert index.config_for("mosaic-run-12") == index.config_for("mosaic-run-13")


def test_two_tools_may_share_a_config_filename(tmp_path: Path) -> None:
    index = read_workflow_index(index_mapping(tmp_path, [
        {"name": "mosaic-smoke", "config": str(tmp_path / "mosaic" / "smoke.yaml")},
        {"name": "boltzgen-smoke", "config": str(tmp_path / "boltzgen" / "smoke.yaml")},
    ]))

    assert index.names == ("mosaic-smoke", "boltzgen-smoke")


def test_duplicate_run_names_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError, match="both named 'twice'"):
        read_workflow_index(index_mapping(tmp_path, [
            {"name": "twice", "config": str(tmp_path / "a.yaml")},
            {"name": "twice", "config": str(tmp_path / "b.yaml")},
        ]))


def test_an_unknown_key_is_a_typo_worth_failing_on(tmp_path: Path) -> None:
    """`models:` groupings used to be ignored, so a config could sit under the
    wrong tool heading and nothing would say so."""
    mapping = index_mapping(tmp_path, [{"name": "r", "config": str(tmp_path / "a.yaml")}])
    mapping["models"] = {"mosaic": ["stale.yaml"]}

    with pytest.raises(ConfigLoadError, match="invalid workflow index"):
        read_workflow_index(mapping)


def test_an_empty_index_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError):
        read_workflow_index(index_mapping(tmp_path, []))


def test_a_run_name_cannot_contain_a_path_separator(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError):
        read_workflow_index(index_mapping(tmp_path, [
            {"name": "../escape", "config": str(tmp_path / "a.yaml")},
        ]))


def test_an_index_round_trips_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    mapping = index_mapping(tmp_path, [{"name": "r1", "config": str(tmp_path / "a.yaml")}])
    path.write_text(yaml.safe_dump(mapping, sort_keys=False))

    assert load_workflow_index(path) == WorkflowIndex.model_validate(mapping)
