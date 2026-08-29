"""The harness writes status.json for tools that do not write their own.

This gap was invisible in the unit tests because the BoltzGen fixture wrote a
status file the tool never produces. The workflow declares status.json as the
output of every generate job, so without this a successful ten-minute BoltzGen
run would have failed Snakemake with "missing output files".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bindocracy.runs import run_task, write_task_status


def test_a_tool_that_only_exits_still_gets_a_status_file(tmp_path: Path) -> None:
    status = tmp_path / "status.json"

    run_task((sys.executable, "-c", "print('done')"), {}, tmp_path / "t.log", status, 0)

    payload = json.loads(status.read_text())
    assert payload["status"] == "succeeded"
    assert payload["exit_code"] == 0
    assert payload["task_id"] == 0
    assert payload["written_by"] == "harness"
    assert (tmp_path / "t.log").read_text().strip() == "done"


def test_a_status_the_tool_wrote_itself_is_not_overwritten(tmp_path: Path) -> None:
    """Mosaic's driver knows the difference between partial and failed."""
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"task_id": 0, "status": "partial", "n_produced": 9}))

    run_task((sys.executable, "-c", "pass"), {}, tmp_path / "t.log", status, 0)

    payload = json.loads(status.read_text())
    assert payload["status"] == "partial"
    assert payload["n_produced"] == 9
    assert "written_by" not in payload


def test_a_failing_task_is_recorded_and_then_raised(tmp_path: Path) -> None:
    status = tmp_path / "status.json"

    with pytest.raises(subprocess.CalledProcessError):
        run_task((sys.executable, "-c", "raise SystemExit(3)"), {},
                 tmp_path / "t.log", status, 1)

    payload = json.loads(status.read_text())
    assert payload["status"] == "failed"
    assert payload["exit_code"] == 3
    assert payload["task_id"] == 1


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
    written = write_task_status(tmp_path / "status.json", 0,
                                started_at="2026-08-29T00:00:00+00:00", status="succeeded")

    assert written is True
    assert list(tmp_path.glob("*.tmp")) == []
