"""Initialization preserves existing work and freezes portable task identities."""
from pathlib import Path

import pytest
import yaml
from conftest import write_configs

from bindocracy.benchmark.workspace import (
    capture_environment,
    initialize_workspace,
    load_workspace,
)
from bindocracy.runs.selection import read_only

ROOT = Path(__file__).resolve().parents[1]


def provision(directory):
    directory.mkdir()
    general, model = write_configs(directory)
    source = yaml.safe_load(general.read_text())
    spec = directory / "environment.yaml"
    spec.write_text(yaml.safe_dump({
        "name": "fixture-installation",
        "harness_revision": "test-revision",
        "harness_path": str(ROOT),
        "lock_path": str(ROOT / "uv.lock"),
        "worker_partition": "workers",
        "site": {
            "controller": {"account": "test", "partition": "controllers", "cpus": 1,
                           "memory_gb": 1, "walltime": "00:02:00", "gpus": 0},
            "max_workers": 1, "max_total_gpus": 1,
        },
        "tool_configs": {"mosaic": model.name},
    }))
    environment = directory / "environment.json"
    capture_environment(spec, environment)
    task = directory / "task.yaml"
    task.write_text(yaml.safe_dump({
        "task_id": "example", "version": "1", "prompt": "Produce an auditable submission.",
        "target": source["target"], "allowed_tools": ["mosaic"],
        "budget": {"mode": "allocation", "walltime_seconds": 120,
                   "max_gpus": 1, "max_cpus": 8, "max_memory_gb": 32},
    }))
    return environment, task


def test_workspace_is_usable_and_existing_work_is_never_replaced(tmp_path):
    environment, task = provision(tmp_path / "installed")
    root = tmp_path / "workspace"
    workspace = initialize_workspace(environment, task, root, attempt_id="attempt-1")
    loaded = load_workspace(root)
    assert loaded == workspace
    assert yaml.safe_load(loaded.general_config.read_text())["cluster"]["default_partition"] == "workers"
    with read_only(workspace.database) as connection:
        assert connection.execute("SELECT count(*) FROM designs").fetchone()[0] == 0
    marker = root / "my-work.txt"
    marker.write_text("preserve me")
    with pytest.raises(FileExistsError):
        initialize_workspace(environment, task, root, attempt_id="attempt-2")
    assert marker.read_text() == "preserve me"
    assert load_workspace(root).attempt_id == "attempt-1"


def test_same_task_at_different_input_paths_has_same_identity(tmp_path):
    first = provision(tmp_path / "site-a")
    second = provision(tmp_path / "site-b")
    a = initialize_workspace(*first, tmp_path / "agent-a")
    b = initialize_workspace(*second, tmp_path / "agent-b")
    assert a.task_digest == b.task_digest
    assert a.attempt_id != b.attempt_id


def test_changed_task_input_and_prompt_are_rejected(tmp_path):
    environment, task = provision(tmp_path / "installed")
    workspace = initialize_workspace(environment, task, tmp_path / "workspace")
    prompt = workspace.root / "prompt.txt"
    prompt.write_text("Different instructions")
    with pytest.raises(ValueError, match="prompt differs"):
        load_workspace(workspace.root)
    prompt.write_text(workspace.task.prompt)
    workspace.task.target.sequence_fasta.write_text(">target\nACDEFA\n")
    with pytest.raises(ValueError, match="content changed"):
        load_workspace(workspace.root)


def test_invalid_input_does_not_publish_a_partial_workspace(tmp_path):
    environment, task = provision(tmp_path / "installed")
    data = yaml.safe_load(task.read_text())
    data["allowed_tools"] = ["unavailable"]
    task.write_text(yaml.safe_dump(data))
    root = tmp_path / "workspace"
    with pytest.raises(ValueError, match="no captured installed template"):
        initialize_workspace(environment, task, root)
    assert not root.exists()


def test_source_symlinks_are_not_silently_followed(tmp_path):
    environment, task = provision(tmp_path / "installed")
    source = tmp_path / "installed" / "target.fasta"
    original = tmp_path / "original.fasta"
    source.rename(original)
    source.symlink_to(original)
    with pytest.raises(ValueError, match="Symlink"):
        initialize_workspace(environment, task, tmp_path / "workspace")


def test_capture_output_cannot_mutate_its_own_asset_inventory(tmp_path):
    provision(tmp_path / "installed")
    spec = tmp_path / "installed" / "environment.yaml"
    tree = tmp_path / "weights"
    tree.mkdir()
    (tree / "existing").write_bytes(b"unchanged")
    raw = yaml.safe_load(spec.read_text())
    raw["assets"] = {"weights": str(tree)}
    spec.write_text(yaml.safe_dump(raw))
    output = tree / "environment.json"
    with pytest.raises(ValueError, match="conflicts with an inventoried"):
        capture_environment(spec, output)
    assert not output.exists()
    assert (tree / "existing").read_bytes() == b"unchanged"


def test_unreadable_directory_cannot_receive_a_content_identity(tmp_path):
    import os

    from bindocracy.benchmark.workspace import identify_asset

    if os.geteuid() == 0:
        pytest.skip("root bypasses directory mode permissions")
    subtree = tmp_path / "unreadable"
    subtree.mkdir()
    (subtree / "hidden-data").write_bytes(b"must not be omitted")
    subtree.chmod(0)
    try:
        with pytest.raises(PermissionError):
            identify_asset(tmp_path)
    finally:
        subtree.chmod(0o700)


def test_external_tool_override_cannot_escape_environment_identity(tmp_path, monkeypatch):
    from bindocracy.tools import registry

    provision(tmp_path / "installed")

    class ExternalMosaic(type(registry.plugin_for("mosaic"))):
        pass

    monkeypatch.setitem(registry._PLUGINS, "mosaic", ExternalMosaic())
    with pytest.raises(ValueError, match="captured harness implementation"):
        capture_environment(tmp_path / "installed" / "environment.yaml", tmp_path / "external.json")
