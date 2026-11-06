"""Recover workflow markers without repeating completed native computation.

Receipts live outside task outputs so Snakemake's failed-output cleanup cannot
remove the only record of completion. A receipt is bound to the frozen manifest;
it never promotes a failed/partial native result to success.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from bindocracy.runs.manifest import RunManifest
from bindocracy.runs.status import HarnessError, TaskStatus, read_task_status, write_task_status


def _identity(manifest: RunManifest) -> str:
    return hashlib.sha256(manifest.model_dump_json().encode()).hexdigest()


def _receipt(manifest: RunManifest, task_id: int) -> Path:
    return manifest.directory / ".task-completions" / f"{task_id:04d}.json"


def record_completion(manifest: RunManifest, task_id: int) -> None:
    task = next(task for task in manifest.tasks if task.task_id == task_id)
    status = read_task_status(manifest.path(task.status))
    if status is None or status.task_id != task_id:
        raise HarnessError(f"missing or mismatched completion status for task {task_id}")
    payload = {"schema_version": 1, "manifest_sha256": _identity(manifest),
               "task_id": task_id, "status": status.model_dump(mode="json")}
    path = _receipt(manifest, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def restore_completion(manifest: RunManifest, task_id: int) -> TaskStatus | None:
    """Return preserved completion, restoring only an absent workflow marker."""
    task = next(task for task in manifest.tasks if task.task_id == task_id)
    path = _receipt(manifest, task_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if (payload["schema_version"] != 1 or payload["task_id"] != task_id
                or payload["manifest_sha256"] != _identity(manifest)):
            raise ValueError("receipt does not match frozen manifest/task")
        status = TaskStatus.model_validate(payload["status"])
        if status.task_id != task_id:
            raise ValueError("receipt status belongs to another task")
        manifest.verify_inputs()
        if status.status == "succeeded" and not manifest.path(task.designs).is_file():
            raise ValueError("successful receipt has missing native output")
        current = read_task_status(manifest.path(task.status))
        if current is not None and current != status:
            raise ValueError("completion receipt conflicts with task status")
        if current is None:
            write_task_status(manifest.path(task.status), status)
        return status
    except (ValueError, KeyError, OSError) as error:
        raise HarnessError(f"cannot recover completion from {path}: {error}") from error
