"""Who writes status.json, and what an ordinary tool failure does.

The gap these cover was invisible for two different reasons. The BoltzGen
fixture used to write a status file the tool never produces, so nothing noticed
that a tool which merely exits leaves the workflow with no declared output. And
`run_task` raised on a non-zero exit, so Snakemake failed the rule and never
ran collection — which contradicted the documented promise that failed and
partial runs are kept as campaign history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bindocracy.runs.status import (
    HarnessError,
    TaskStatus,
    harness_status,
    read_task_status,
    run_task,
    write_task_status,
)


def test_a_tool_that_only_exits_still_gets_a_status_file(tmp_path: Path) -> None:
    status = tmp_path / "status.json"

    code = run_task((sys.executable, "-c", "print('done')"), {}, tmp_path / "t.log", status, 0)

    assert code == 0
    recorded = read_task_status(status)
    assert recorded.status == "succeeded"
    assert recorded.exit_code == 0
    assert recorded.task_id == 0
    assert recorded.written_by == "harness"
    assert (tmp_path / "t.log").read_text().strip() == "done"


def test_a_status_the_tool_wrote_itself_is_not_overwritten(tmp_path: Path) -> None:
    """Mosaic's driver knows the difference between partial and failed."""
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"task_id": 0, "status": "partial", "n_produced": 9}))

    run_task((sys.executable, "-c", "pass"), {}, tmp_path / "t.log", status, 0)

    recorded = read_task_status(status)
    assert recorded.status == "partial"
    assert recorded.n_produced == 9
    assert recorded.written_by == "tool"


def test_a_tool_failure_is_recorded_and_does_not_raise(tmp_path: Path) -> None:
    """The whole point: collection must still run, so the run reaches the database."""
    status = tmp_path / "status.json"

    code = run_task((sys.executable, "-c", "raise SystemExit(3)"), {},
                    tmp_path / "t.log", status, 1)

    assert code == 3
    recorded = read_task_status(status)
    assert recorded.status == "failed"
    assert recorded.exit_code == 3
    assert recorded.task_id == 1
    assert recorded.error == "exit code 3"


def test_a_harness_failure_does_raise(tmp_path: Path) -> None:
    """Being unable to start the tool is not the tool failing."""
    status = tmp_path / "status.json"

    with pytest.raises(HarnessError, match="could not run task"):
        run_task(("/nonexistent/binary",), {}, tmp_path / "t.log", status, 0)

    # even then, what happened is recorded
    assert read_task_status(status).status == "failed"


def test_node_local_directories_are_created_before_the_container_starts(
    tmp_path: Path
) -> None:
    """The BoltzGen runscript mkdir -p's under TMPDIR but not TMPDIR itself."""
    node_tmp = tmp_path / "nodetmp" / "run-0000"
    env = {"TMPDIR": str(node_tmp),
           "SINGULARITYENV_BOLTZGEN_RUNTIME_CACHE": str(node_tmp / "cache")}

    run_task((sys.executable, "-c", "pass"), env, tmp_path / "t.log",
             tmp_path / "status.json", 0)

    assert node_tmp.is_dir()
    assert (node_tmp / "cache").is_dir()


def test_write_task_status_is_atomic(tmp_path: Path) -> None:
    written = write_task_status(tmp_path / "status.json",
                                harness_status(0, "succeeded"))

    assert written is True
    assert list(tmp_path.glob("*.tmp")) == []
    assert read_task_status(tmp_path / "status.json").status == "succeeded"


def test_a_status_file_is_validated_not_trusted(tmp_path: Path) -> None:
    """Collection used to pass raw dictionaries around."""
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"task_id": 0, "status": "exploded"}))

    with pytest.raises(HarnessError, match="invalid status file"):
        read_task_status(status)


def test_a_tool_may_record_more_than_the_harness_asks_for(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    status.write_text(json.dumps({
        "task_id": 0, "status": "succeeded", "n_produced": 5,
        "gpu_hours": 1.25, "output_file": "designs.jsonl",
    }))

    recorded = read_task_status(status)

    assert recorded.n_produced == 5
    assert recorded.output_file == "designs.jsonl"
    assert recorded.gpu_hours == 1.25


def test_a_missing_status_file_is_not_an_error(tmp_path: Path) -> None:
    assert read_task_status(tmp_path / "absent.json") is None


def test_task_status_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError):
        TaskStatus(task_id=0, status="succeeded", n_produced=-1)
