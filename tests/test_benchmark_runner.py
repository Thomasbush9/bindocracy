"""Real local subprocesses and authenticated local broker; never real Slurm."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bindocracy.benchmark import runner, runner_cli
from bindocracy.benchmark import workspace as workspace_module
from bindocracy.benchmark.models import EnvironmentManifest, TaskSpec, WorkspaceManifest


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    fasta = root / "target.fasta"
    fasta.write_text(">target\nACDEFGHIK\n")
    task = TaskSpec(task_id="cpu-task", version="1", prompt="Run a CPU command",
                    target={"name": "target", "sequence_fasta": fasta, "chain_id": "A"},
                    budget={"walltime_seconds": 10, "max_gpus": 0, "max_cpus": 2, "max_memory_gb": 2},
                    allowed_tools=("function",))
    environment = EnvironmentManifest(name="local", harness_revision="test", harness_path=tmp_path,
                                      lock_path=tmp_path / "lock", harness_sha256="a" * 64, lock_sha256="b" * 64,
                                      site={"controller": {"account": "test", "partition": "cpu", "cpus": 1,
                                                            "memory_gb": 1, "walltime": "00:00:02", "gpus": 0},
                                            "max_workers": 1, "max_total_gpus": 1})
    value = WorkspaceManifest(attempt_id="attempt-a", root=root, database=root / "database.duckdb",
                              general_config=root / "general.yaml", site_config=root / "site.yaml",
                              bindings=root / "bindings.yaml", task=task, environment=environment,
                              task_digest="c" * 64, environment_digest="d" * 64,
                              input_assets={}, created_at=datetime.now(UTC))
    monkeypatch.setattr(workspace_module, "load_workspace", lambda _: value)
    return value


def test_real_command_records_output_metadata_and_unknown_usage(workspace, tmp_path):
    directory = tmp_path / "attempt"
    result = runner.run_attempt(workspace.root, directory,
                                [sys.executable, "-c", "import os,sys; print(os.environ['BINDOCRACY_ATTEMPT_ID']); print('problem', file=sys.stderr)"],
                                development=True, agent={"name": "test-agent", "repeat": 2})
    assert result["attempt"]["state"] == "completed"
    assert result["attempt"]["agent"]["isolation"] == "unisolated/development"
    assert (directory / "stdout.log").read_text().strip() == "attempt-a"
    assert (directory / "stderr.log").read_text().strip() == "problem"
    assert result["attempt"]["usage"]["reported_totals"]["cost_usd"] is None
    assert result["budget"]["closed"] is True
    assert result["budget"]["reservations"]["agent"]["charged"]["cpu_seconds"] > 0
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert {event["event"] for event in events} >= {"started", "output", "finished"}
    assert all(event["timestamp"] for event in events)
    assert (directory / "target.fasta").read_bytes() == workspace.task.target.sequence_fasta.read_bytes()


def test_deadline_kills_process_group(workspace, tmp_path, monkeypatch):
    short = workspace.model_copy(update={"task": workspace.task.model_copy(update={
        "budget": workspace.task.budget.model_copy(update={"walltime_seconds": 1})})})
    monkeypatch.setattr(workspace_module, "load_workspace", lambda _: short)
    directory = tmp_path / "deadline"
    command = "import subprocess,time; p=subprocess.Popen(['sleep','60']); print(p.pid,flush=True); time.sleep(60)"
    started = time.monotonic()
    result = runner.run_attempt(short.root, directory, [sys.executable, "-c", command], development=True)
    assert result["attempt"]["state"] == "timed_out"
    assert time.monotonic() - started < 8
    child = int((directory / "stdout.log").read_text().strip())
    stat = Path(f"/proc/{child}/stat")
    assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_strict_prerequisites_and_control_placement_fail_before_launch(workspace, tmp_path):
    with pytest.raises(runner.RunnerError, match="distinct agent UID"):
        runner.run_attempt(workspace.root, tmp_path / "production", ["not-executed"])
    with pytest.raises(runner.RunnerError, match="outside agent workspace"):
        runner.run_attempt(workspace.root, workspace.root / "control", ["not-executed"], development=True)
    assert not (tmp_path / "production").exists()


def test_failed_launch_is_durable_and_recovery_never_relaunches(workspace, tmp_path):
    directory = tmp_path / "failed"
    result = runner.run_attempt(workspace.root, directory, ["/missing/agent/executable"], development=True)
    assert result["attempt"]["state"] == "infrastructure_failed"
    assert result["budget"]["closed"]
    recovered = runner.recover(directory)
    assert recovered["attempt"]["state"] == "infrastructure_failed"
    assert recovered["attempt"]["pid"] is None


def test_usage_interface_preserves_unknowns_and_is_idempotent(workspace, tmp_path):
    directory = tmp_path / "usage"
    runner.run_attempt(workspace.root, directory, [sys.executable, "-c", "pass"], development=True)
    record = {"event_id": "call-1", "provider": "provider", "model": "model", "input_tokens": 12}
    first = runner.record_usage(directory, record)
    assert runner.record_usage(directory, record) == first
    assert first["reported_totals"] == {"input_tokens": 12, "output_tokens": None, "cost_usd": None}
    with pytest.raises(runner.RunnerError, match="different data"):
        runner.record_usage(directory, {**record, "input_tokens": 13})


def test_broker_accepts_structured_usage_from_real_agent(workspace, tmp_path):
    script = """import socket,os,json
s=socket.socket(socket.AF_UNIX); s.connect(os.environ['BINDOCRACY_BROKER'])
request={'operation':'usage','arguments':{'record':{'event_id':'call','provider':'local','model':'test'}}}
s.sendall(json.dumps(request).encode()+b'\\n')
reply=json.loads(s.makefile('rb').readline()); assert 'result' in reply, reply
print('broker accepted')
"""
    result = runner.run_attempt(workspace.root, tmp_path / "broker", [sys.executable, "-c", script], development=True)
    assert result["attempt"]["state"] == "completed"
    assert result["attempt"]["usage"]["records"][0]["event_id"] == "call"


def test_cancel_closes_active_attempt_and_recover_refuses_live_runner(workspace, tmp_path):
    directory = tmp_path / "cancel"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runner.run_attempt, workspace.root, directory,
                             [sys.executable, "-c", "import time; time.sleep(60)"], development=True)
        until = time.monotonic() + 5
        while not (directory / "process.json").exists() and time.monotonic() < until:
            time.sleep(0.02)
        assert (directory / "process.json").exists()
        with pytest.raises(runner.RunnerError, match="still active"):
            runner.recover(directory)
        cancelled = runner.cancel(directory)
        assert cancelled["budget"]["closed"]
        assert future.result(timeout=10)["attempt"]["state"] == "cancelled"


def test_genuine_cli_cpu_smoke(workspace, tmp_path):
    directory = tmp_path / "smoke"
    result = CliRunner().invoke(runner_cli.app, ["smoke", str(workspace.root), str(directory)])
    assert result.exit_code == 0, result.output
    assert "benchmark local CPU smoke" in (directory / "stdout.log").read_text()
    assert runner.status(directory)["attempt"]["state"] == "completed"


def test_broker_rejects_another_uid(workspace, tmp_path):
    import os
    import tempfile
    directory = tmp_path / "peer"
    runner.run_attempt(workspace.root, directory, [sys.executable, "-c", "pass"], development=True)
    with tempfile.TemporaryDirectory(prefix="bd-peer-") as sockets:
        socket_path = Path(sockets) / "broker.sock"
        with (
            runner._serve_broker(directory, socket_path, os.getuid() + 1),
            pytest.raises(runner.RunnerError, match="not the configured agent"),
        ):
            runner._broker_request(socket_path, "usage", record={})


def test_cleanup_cancels_exact_tags_and_keeps_unknown_reserved(workspace, tmp_path):
    from datetime import timedelta

    from bindocracy.benchmark.budget import BudgetLedger
    from bindocracy.benchmark.models import AttemptRecord

    directory = tmp_path / "scheduler"
    directory.mkdir()
    now = datetime.now(UTC)
    record = AttemptRecord(attempt_id=workspace.attempt_id, workspace=workspace.root,
                           task_id=workspace.task.task_id, task_digest=workspace.task_digest,
                           environment_digest=workspace.environment_digest,
                           created_at=now, started_at=now, state="running",
                           deadline=now + timedelta(seconds=100))
    (directory / "attempt.json").write_text(record.model_dump_json())
    ledger = BudgetLedger(directory)
    ledger.initialize(workspace.task.budget, attempt_id=workspace.attempt_id,
                      deadline=now + timedelta(seconds=100), scheduler_enabled=True)
    ledger.reserve("known", gpus=0, cpus=1, walltime_seconds=10, tag="owned:exact")
    ledger.reserve("lost", gpus=0, cpus=1, walltime_seconds=10, tag="owned:unknown")

    class Scheduler:
        def __init__(self):
            self.cancelled = []

        def snapshot(self, since):
            return [{"id": "123", "tag": "owned:exact", "state": "RUNNING"},
                    {"id": "999", "tag": "not-ours", "state": "RUNNING"}]

        def cancel(self, job_id):
            self.cancelled.append(job_id)

        def accounting(self, job_id):
            return {"terminal": True, "cpus": 1, "gpus": 0, "elapsed_seconds": 4}

    scheduler = Scheduler()
    result = runner.cancel(directory, scheduler=scheduler)
    assert scheduler.cancelled == ["123"]
    assert result["budget"]["closed"]
    assert result["budget"]["reservations"]["known"]["terminal"]
    assert not result["budget"]["reservations"]["lost"]["terminal"]
    assert result["attempt"]["cleanup"]["unresolved"] == ["lost"]
