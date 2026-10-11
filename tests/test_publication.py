"""Publication never adopts existing paths or removes intervening user files."""
from pathlib import Path

import pytest

from bindocracy.publication import publish_directory


def test_existing_empty_directory_is_not_adopted(tmp_path):
    stage, destination = tmp_path / "stage", tmp_path / "destination"
    stage.mkdir()
    (stage / "result").write_bytes(b"complete")
    destination.mkdir()
    identity = destination.stat().st_ino
    with pytest.raises(FileExistsError):
        publish_directory(stage, destination)
    assert destination.stat().st_ino == identity
    assert not (destination / "result").exists()
    assert (stage / "result").read_bytes() == b"complete"


def test_completed_tree_is_published_without_remaining_stage(tmp_path):
    stage, destination = tmp_path / "stage", tmp_path / "destination"
    stage.mkdir()
    (stage / "result").write_bytes(b"complete")
    publish_directory(stage, destination)
    assert (destination / "result").read_bytes() == b"complete"
    assert not stage.exists()


def test_failed_rename_preserves_intervening_files(tmp_path, monkeypatch):
    stage, destination = tmp_path / "stage", tmp_path / "destination"
    stage.mkdir()
    (stage / "result").write_bytes(b"complete")
    original = Path.rename

    def interrupted_rename(source, target):
        (target / "user-file").write_bytes(b"preserve")
        return original(source, target)

    monkeypatch.setattr(Path, "rename", interrupted_rename)
    with pytest.raises(OSError):
        publish_directory(stage, destination)
    assert (destination / "user-file").read_bytes() == b"preserve"
    assert (stage / "result").read_bytes() == b"complete"
