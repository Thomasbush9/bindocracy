"""A lost workflow marker must not duplicate native designs on resumption."""
import json

import pytest

from bindocracy.runs.completion import record_completion, restore_completion
from bindocracy.runs.status import HarnessError, TaskStatus, read_task_status, write_task_status
from bindocracy.runs.steps import generate_step
from bindocracy.tools import load_configs
from bindocracy.tools.registry import plan


@pytest.fixture
def completed_task(configs, tmp_path):
    manifest = plan(load_configs(*configs), tmp_path / "run", name="receipt-test")
    task = manifest.tasks[0]
    generate_step(manifest.directory / "run.json", task.task_id,
                  manifest.path(task.log), manifest.path(task.status))
    return manifest, task


def test_resume_lost_marker_does_not_append_duplicate_designs(completed_task):
    manifest, task = completed_task
    designs = manifest.path(task.designs).read_text()
    original = read_task_status(manifest.path(task.status))
    manifest.path(task.status).unlink()
    generate_step(manifest.directory / "run.json", task.task_id,
                  manifest.path(task.log), manifest.path(task.status))
    assert manifest.path(task.designs).read_text() == designs
    assert read_task_status(manifest.path(task.status)) == original
    assert len([json.loads(line) for line in designs.splitlines()]) == task.n_requested


def test_receipt_cannot_recover_under_changed_run_identity(completed_task):
    manifest, task = completed_task
    manifest.path(task.status).unlink()
    altered = manifest.model_copy(update={"name": "different-run"})
    with pytest.raises(HarnessError, match="does not match frozen manifest"):
        restore_completion(altered, task.task_id)
    assert not manifest.path(task.status).exists()


def test_missing_native_output_is_not_promoted_to_success(completed_task):
    manifest, task = completed_task
    manifest.path(task.status).unlink()
    manifest.path(task.designs).unlink()
    with pytest.raises(HarnessError, match="missing native output"):
        restore_completion(manifest, task.task_id)
    assert not manifest.path(task.status).exists()


def test_conflicting_marker_is_not_overwritten(completed_task):
    manifest, task = completed_task
    failure = TaskStatus(task_id=task.task_id, status="failed", exit_code=7)
    write_task_status(manifest.path(task.status), failure, overwrite=True)
    with pytest.raises(HarnessError, match="conflicts with task status"):
        restore_completion(manifest, task.task_id)
    assert read_task_status(manifest.path(task.status)) == failure


def test_failed_native_result_remains_failed_when_marker_recovered(configs, tmp_path):
    manifest = plan(load_configs(*configs), tmp_path / "run", name="failed-receipt")
    task = manifest.tasks[0]
    failure = TaskStatus(task_id=task.task_id, status="failed", exit_code=3,
                         error="native failure", written_by="harness")
    write_task_status(manifest.path(task.status), failure)
    record_completion(manifest, task.task_id)
    manifest.path(task.status).unlink()
    assert restore_completion(manifest, task.task_id) == failure
    assert read_task_status(manifest.path(task.status)).status == "failed"
