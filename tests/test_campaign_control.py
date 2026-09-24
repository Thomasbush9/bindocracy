"""Lifecycle contracts against a deterministic Slurm transport, never real jobs."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from bindocracy.campaign import control
from bindocracy.tools import load_configs
from bindocracy.tools.registry import plan as plan_run


class Scheduler:
    def __init__(self):
        self.jobs = {}
        self.cancelled = []
        self.environments = []
        self.args = []
        self.accept_hook = None
        self.hide = False
        self.transport = control.Slurm(run=self.run, sbatch="/fake/sbatch")

    def run(self, argv, **kwargs):
        self.environments.append(kwargs["env"])
        name = Path(argv[0]).name
        if name == "sbatch":
            job_id = str(100 + len(self.jobs))
            tag = control._option(argv[1:], "--comment")
            self.jobs[job_id] = {"tag": tag, "state": "PENDING"}
            self.args.append(argv)
            if self.accept_hook:
                self.accept_hook(argv, job_id)
            return subprocess.CompletedProcess(argv, 0, job_id + "\n", "")
        if name == "scancel":
            self.cancelled.append(argv[1])
            self.jobs[argv[1]]["state"] = "CANCELLED"
            return subprocess.CompletedProcess(argv, 0, "", "")
        if name in {"squeue", "sacct"}:
            rows = (
                []
                if self.hide
                else [
                    f"{job_id}|{job['state']}|{job['tag']}"
                    for job_id, job in self.jobs.items()
                    if name == "sacct" or job["state"] not in control._TERMINAL
                ]
            )
            return subprocess.CompletedProcess(argv, 0, "\n".join(rows), "")
        raise AssertionError(f"unexpected scheduler call: {argv}")


@pytest.fixture
def campaign(configs, tmp_path, monkeypatch):
    manifest = plan_run(load_configs(*configs), tmp_path / "run", name="run_with_underscores")
    directory = tmp_path / "plan with spaces"
    directory.mkdir()
    frozen = {
        "digest": "sha256:" + "a" * 64,
        "plan_dir": str(directory),
        "python": sys.executable,
        "snakemake": str(tmp_path / "snakemake"),
        "snakefile": str(directory / "Snakefile"),
        "workflow_index": str(directory / "index.yaml"),
        "database": str(tmp_path / "campaign.duckdb"),
        "run_root": str(tmp_path),
        "site": {
            "controller": {
                "account": "science",
                "partition": "cpu",
                "cpus": 2,
                "memory_gb": 8,
                "walltime": "01:00:00",
                "gpus": 0,
            },
            "max_workers": 3,
            "max_total_gpus": 3,
            "nofile": 4096,
            "latency_wait": 60,
        },
        "runs": [
            {
                "name": manifest.name,
                "manifest": str(manifest.directory / "run.json"),
                "tasks": len(manifest.tasks),
            }
        ],
    }
    path = directory / "plan.json"
    path.write_text(json.dumps(frozen))

    def load(path, verify=True):
        result = json.loads(Path(path).read_text())
        if verify:
            manifest.verify_inputs()
        return result

    monkeypatch.setattr(control, "load_plan", load)
    return path, frozen, manifest


def worker_args(manifest, task=0):
    return [
        "--parsable",
        "--job-name=executor-uuid",
        "--export=ALL",
        "--comment",
        f"rule_generate_wildcards_{manifest.name}_{task:04d}",
        "--gres=gpu:1",
        "--wrap",
        "printf 'worker allocation\\n'",
    ]


def test_approval_and_concurrent_submit_launch_exactly_once(campaign):
    path, frozen, _ = campaign
    scheduler = Scheduler()
    with pytest.raises(control.CampaignError, match="exactly"):
        control.submit(path, "a" * 64, scheduler=scheduler.transport)
    assert scheduler.jobs == {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(control.submit, path, frozen["digest"], scheduler=scheduler.transport)
            for _ in range(2)
        ]
        results = [future.result() for future in futures]
    assert len(scheduler.jobs) == 1
    assert results[0]["attempts"][0]["id"] == results[1]["attempts"][0]["id"]


def test_lost_submission_reply_is_reconciled_not_duplicated(campaign):
    path, frozen, _ = campaign
    scheduler = Scheduler()

    def timeout_after_acceptance(argv, job_id):
        persisted = json.loads((Path(frozen["plan_dir"]) / "control/journal.json").read_text())
        assert persisted["attempts"][0]["jobs"][0]["id"] is None
        raise subprocess.TimeoutExpired(argv, 120)

    scheduler.accept_hook = timeout_after_acceptance
    with pytest.raises(control.CampaignError, match="unknown"):
        control.submit(path, frozen["digest"], scheduler=scheduler.transport)
    scheduler.hide = True
    with pytest.raises(control.CampaignError, match="unknown"):
        control.resume(path, frozen["digest"], scheduler=scheduler.transport)
    scheduler.hide = False
    result = control.submit(path, frozen["digest"], scheduler=scheduler.transport)
    assert result["attempts"][0]["jobs"][0]["id"] == "100"
    assert len(scheduler.jobs) == 1


def test_cancel_closes_gate_and_only_cancels_exact_owned_jobs(campaign):
    path, frozen, manifest = campaign
    scheduler = Scheduler()
    started = control.submit(path, frozen["digest"], scheduler=scheduler.transport)
    attempt = started["attempts"][0]["id"]
    worker = control.worker_submit(
        path, attempt, worker_args(manifest), scheduler=scheduler.transport
    )
    scheduler.jobs["999"] = {"state": "RUNNING", "tag": "another-campaign"}
    result = control.cancel(path, scheduler=scheduler.transport)
    assert scheduler.cancelled == ["100", worker]
    assert result["state"] == "cancelled"
    with pytest.raises(control.CampaignError, match="cancelled"):
        control.worker_submit(
            path, attempt, worker_args(manifest, 1), scheduler=scheduler.transport
        )
    control.cancel(path, scheduler=scheduler.transport)
    assert scheduler.cancelled == ["100", worker]
    assert scheduler.jobs["999"]["state"] == "RUNNING"


def test_late_worker_acceptance_is_cancelled_after_submission_lock(campaign):
    path, frozen, manifest = campaign
    scheduler = Scheduler()
    started = control.submit(path, frozen["digest"], scheduler=scheduler.transport)
    attempt = started["attempts"][0]["id"]
    accepted = Event()
    release = Event()

    def block_after_acceptance(argv, job_id):
        accepted.set()
        assert release.wait(10)
        raise subprocess.TimeoutExpired(argv, 120)

    scheduler.accept_hook = block_after_acceptance
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker = pool.submit(
            control.worker_submit,
            path,
            attempt,
            worker_args(manifest),
            scheduler=scheduler.transport,
        )
        assert accepted.wait(10)
        cancellation = pool.submit(control.cancel, path, scheduler=scheduler.transport)
        release.set()
        with pytest.raises(control.CampaignError, match="unknown"):
            worker.result()
        assert cancellation.result()["state"] == "cancelled"
    assert scheduler.cancelled == ["100", "101"]


def test_reused_scheduler_id_with_different_tag_is_never_cancelled(campaign):
    path, frozen, _ = campaign
    scheduler = Scheduler()
    control.submit(path, frozen["digest"], scheduler=scheduler.transport)
    scheduler.jobs["100"]["tag"] = "unrelated-recycled-id"
    result = control.cancel(path, scheduler=scheduler.transport)
    assert result["state"] == "unknown"
    assert scheduler.cancelled == []
    with pytest.raises(control.CampaignError, match="unknown"):
        control.resume(path, frozen["digest"], scheduler=scheduler.transport)


def test_resume_retains_scientific_outputs_and_requires_terminal_workers(campaign):
    path, frozen, manifest = campaign
    scheduler = Scheduler()
    started = control.submit(path, frozen["digest"], scheduler=scheduler.transport)
    attempt = started["attempts"][0]["id"]
    control.worker_submit(path, attempt, worker_args(manifest), scheduler=scheduler.transport)
    scheduler.jobs["100"]["state"] = "FAILED"
    with pytest.raises(control.CampaignError, match="live"):
        control.resume(path, frozen["digest"], scheduler=scheduler.transport)
    task = manifest.tasks[0]
    status_path = manifest.path(task.status)
    status_path.write_text(
        json.dumps({"task_id": task.task_id, "status": "succeeded", "metric": 42})
    )
    manifest.path(task.designs).write_text("scientific result\n")
    before = {file: file.read_bytes() for file in (status_path, manifest.path(task.designs), path)}
    scheduler.jobs["101"]["state"] = "COMPLETED"
    result = control.resume(path, frozen["digest"], scheduler=scheduler.transport)
    assert len(result["attempts"]) == 2
    assert result["attempts"][1]["id"] != attempt
    assert {file: file.read_bytes() for file in before} == before
    assert result["artifacts"][0]["tasks"][0]["status"] == "succeeded"


def test_worker_verification_preserves_allocation_but_scheduler_calls_are_clean(
    campaign, monkeypatch
):
    path, frozen, manifest = campaign
    scheduler = Scheduler()
    for key in ("SLURM_JOB_ID", "SBATCH_PARTITION", "SRUN_CPUS_PER_TASK"):
        monkeypatch.setenv(key, "inherited")
    started = control.submit(path, frozen["digest"], scheduler=scheduler.transport)
    attempt = started["attempts"][0]["id"]
    worker = control.worker_submit(
        path, attempt, worker_args(manifest), scheduler=scheduler.transport
    )
    job = control.status(path, scheduler=scheduler.transport)["attempts"][0]["jobs"][1]
    monkeypatch.setenv("SLURM_JOB_ID", worker)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    control._verify_worker(path, attempt, job["token"])
    import os

    assert os.environ["SLURM_JOB_ID"] == worker
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert all(
        not any(key.startswith(("SLURM_", "SBATCH_", "SRUN_")) for key in environment)
        for environment in scheduler.environments
    )
    manifest.path(next(iter(manifest.provenance.values())).path).write_text("changed input")
    with pytest.raises(RuntimeError, match="no longer match"):
        control._verify_worker(path, attempt, job["token"])
    # Recovery commands remain available even after an input changes.
    assert control.cancel(path, scheduler=scheduler.transport)["state"] == "cancelled"


def test_resume_archives_interrupted_append_only_shard_before_restarting(campaign):
    path, frozen, manifest = campaign
    scheduler = Scheduler()
    control.submit(path, frozen["digest"], scheduler=scheduler.transport)
    scheduler.jobs["100"]["state"] = "FAILED"
    task = manifest.tasks[0]
    output = manifest.path(task.designs)
    old_record = json.dumps({"native_id": "task-0000-design-000000", "sequence": "OLD"})
    output.write_text(old_record + "\n")
    manifest.path(task.log).write_text("interrupted after first design\n")
    result = control.resume(path, frozen["digest"], scheduler=scheduler.transport)
    recovery = result["attempts"][-1]["recovery"]
    archived_task = next(
        Path(item["archive"])
        for item in recovery
        if item["source"] == str(manifest.path(task.directory))
    )
    archived_output = archived_task / output.relative_to(manifest.path(task.directory))
    assert archived_output.read_text() == old_record + "\n"
    assert manifest.path(task.directory).is_dir()
    assert not output.exists()
    # The driver's ordinary append mode now starts with an empty shard; its
    # restarted native IDs cannot collide with preserved partial evidence.
    with output.open("a") as stream:
        stream.write(json.dumps({"native_id": "task-0000-design-000000", "sequence": "NEW"}) + "\n")
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records == [{"native_id": "task-0000-design-000000", "sequence": "NEW"}]
    assert archived_output.read_text() == old_record + "\n"


def test_scheduler_process_does_not_inherit_parent_gpu_visibility(tmp_path, monkeypatch):
    """A controller's GPU mask must not constrain a different worker allocation."""
    executable = tmp_path / "sbatch"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if 'CUDA_VISIBLE_DEVICES' in os.environ:\n"
        "    sys.stderr.write('inherited controller GPU allocation')\n"
        "    sys.exit(1)\n"
        "print('800')\n"
    )
    executable.chmod(0o700)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert control.Slurm(sbatch=str(executable)).submit(["--parsable"]) == "800"


def test_generated_resume_controller_accepts_completed_outputs_without_metadata(
    configs, tmp_path, monkeypatch
):
    """An interrupted controller may have completed outputs but no local metadata."""
    from bindocracy.campaign.plan import build_plan

    index = tmp_path / "index.yaml"
    site = tmp_path / "site.yaml"
    index.write_text(
        json.dumps(
            {
                "database": str(tmp_path / "campaign.duckdb"),
                "run_root": str(tmp_path / "runs"),
                "general_config": str(configs[0]),
                "runs": [{"name": "completed", "config": str(configs[1])}],
            }
        )
    )
    site.write_text(
        json.dumps(
            {
                "controller": {
                    "account": "test",
                    "partition": "cpu",
                    "cpus": 2,
                    "memory_gb": 4,
                    "walltime": "01:00:00",
                    "gpus": 0,
                },
                "max_workers": 2,
                "max_total_gpus": 2,
            }
        )
    )
    plan_path = tmp_path / "plan" / "plan.json"
    plan = build_plan(index, site, plan_path.parent)
    work = tmp_path / "initial-work"
    work.mkdir()
    initial = subprocess.run(
        [
            plan["snakemake"],
            "--snakefile",
            plan["snakefile"],
            "--configfile",
            plan["workflow_index"],
            "--directory",
            str(work),
            "--executor",
            "local",
            "--cores",
            "2",
            "--resources",
            "db_writer=1",
            "gpu=2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert initial.returncode == 0, initial.stderr
    outputs = [
        tmp_path / "runs" / "completed" / "tasks" / f"{task:04d}" / name
        for task in range(2)
        for name in ("status.json", "designs.jsonl")
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs}
    scheduler = Scheduler()
    # Even a regression that tries submitting a worker cannot reach real Slurm.
    forbidden = tmp_path / "never-submit" / "sbatch"
    forbidden.parent.mkdir()
    forbidden.write_text("#!/bin/sh\nexit 99\n")
    forbidden.chmod(0o700)
    scheduler.transport.sbatch = str(forbidden)
    control.submit(plan_path, plan["digest"], scheduler=scheduler.transport)
    scheduler.jobs["100"]["state"] = "FAILED"
    control.resume(plan_path, plan["digest"], scheduler=scheduler.transport)
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    completed = subprocess.run(
        ["bash", scheduler.args[-1][-1]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs}


def test_worker_guard_covers_entire_native_shell_payload(campaign, tmp_path):
    import shlex

    path, frozen, manifest = campaign
    scheduler = Scheduler()
    started = control.submit(path, frozen["digest"], scheduler=scheduler.transport)
    marker = tmp_path / "must-not-run"
    args = worker_args(manifest)
    args[-1] = f"true; touch {shlex.quote(str(marker))}"
    control.worker_submit(
        path,
        started["attempts"][0]["id"],
        args,
        scheduler=scheduler.transport,
    )
    payload = next(
        arg.removeprefix("--wrap=") for arg in scheduler.args[-1] if arg.startswith("--wrap=")
    )
    # The real child verifier rejects this fixture's synthetic, unapproved plan.
    # Every native command must remain behind that guard, not just the first.
    result = subprocess.run(
        ["sh", "-c", payload],
        capture_output=True,
        text=True,
        check=False,
    )
    assert not marker.exists()
    assert result.returncode != 0
