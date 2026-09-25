"""Qualification safety through real campaign control and an injected Slurm runner.

No cluster jobs or scientific models are executed by this module.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import write_configs

from bindocracy.campaign import control
from bindocracy.campaign.qualify import QualificationError, qualify
from bindocracy.tools import load_configs
from bindocracy.tools.registry import plan as plan_run


class Canary:
    def __init__(self, path, frozen, manifest):
        self.path, self.frozen, self.manifest = path, frozen, manifest
        self.now = 0
        self.jobs = {"999": {"state": "RUNNING", "tag": "unrelated"}}
        self.cancelled = []
        self.calls = []
        self.complete = True
        self.accounting = True
        self.interrupt = False
        self.lose_reply = False
        self.failure = False
        self.task_status = "succeeded"

    def load(self, path, verify=True):
        return self.frozen

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def run(self, argv, **kwargs):
        assert 0 < kwargs["timeout"] <= 120
        self.calls.append(argv)
        name = Path(argv[0]).name
        if name == "sbatch":
            job_id = str(100 + len(self.jobs))
            self.jobs[job_id] = {
                "tag": control._option(argv, "--comment"),
                "state": "RUNNING",
            }
            if self.lose_reply:
                self.lose_reply = False
                raise subprocess.TimeoutExpired(argv, 1)
            return subprocess.CompletedProcess(argv, 0, job_id, "")
        if name == "scancel":
            self.cancelled.append(argv[1])
            self.jobs[argv[1]]["state"] = "CANCELLED"
            return subprocess.CompletedProcess(argv, 0, "", "")
        if name in {"sacct", "squeue"}:
            if self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt
            rows = []
            for job_id, job in self.jobs.items():
                if name == "sacct" and not self.accounting:
                    continue
                # Retain terminal states in this artificial queue to prove that
                # even a completed lifecycle view cannot substitute for sacct.
                rows.append(f"{job_id}|{job['state']}|{job['tag']}")
            return subprocess.CompletedProcess(argv, 0, "\n".join(rows), "")
        raise AssertionError(argv)

    def status(self, path, *, scheduler):
        journal = json.loads((Path(self.frozen["plan_dir"]) / "control/journal.json").read_text())
        attempt = journal["attempts"][-1]
        if not attempt["cancel_requested"]:
            if not any(job["role"] == "worker" for job in attempt["jobs"]):
                control.worker_submit(
                    path,
                    attempt["id"],
                    [
                        "--parsable",
                        "--comment=rule_generate_wildcards_canary_0000",
                        "--wrap=true",
                    ],
                    scheduler=control.Slurm(run=self.run),
                )
            if self.complete or len(journal["attempts"]) > 1:
                for job in self.jobs.values():
                    if job["tag"].startswith("bd:") and job["state"] != "CANCELLED":
                        job["state"] = "FAILED" if self.failure else "COMPLETED"
                if not self.failure:
                    (self.manifest.directory / "generation.done").touch()
                    (self.manifest.directory / "ingested.json").write_text("{}")
                    task = self.manifest.tasks[0]
                    self.manifest.path(task.designs).parent.mkdir(parents=True, exist_ok=True)
                    self.manifest.path(task.designs).write_text("{}\n")
                    self.manifest.path(task.status).write_text(
                        json.dumps(
                            {
                                "task_id": task.task_id,
                                "status": self.task_status,
                            }
                        )
                    )
        return control.status(path, scheduler=scheduler)

    def invoke(self, **kwargs):
        options = {
            "approve": self.frozen["digest"],
            "loader": self.load,
            "runner": self.run,
            "clock": self.clock,
            "sleep": self.sleep,
            "timeout": 3,
            "poll_interval": 1,
            "lifecycle": SimpleNamespace(
                submit=control.submit,
                status=self.status,
                cancel=control.cancel,
                resume=control.resume,
            ),
        }
        options.update(kwargs)
        return qualify(self.path, **options)


@pytest.fixture
def canary(tmp_path, monkeypatch):
    configs = write_configs(tmp_path, sampling={"jobs": 1, "designs_per_job": 1})
    manifest = plan_run(load_configs(*configs), tmp_path / "run", name="canary")
    directory = tmp_path / "plan"
    directory.mkdir()
    frozen = {
        "digest": "sha256:" + "a" * 64,
        "plan_dir": str(directory),
        "python": sys.executable,
        "site": {
            "controller": {
                "gpus": 0,
                "cpus": 1,
                "memory_gb": 1,
                "walltime": "00:05:00",
                "account": "test",
                "partition": "cpu",
            },
            "max_workers": 1,
            "max_total_gpus": 1,
            "nofile": 4096,
        },
        "runs": [
            {
                "name": "canary",
                "manifest": str(manifest.directory / "run.json"),
                "tasks": 1,
                "resources": {
                    "gpu": 0,
                    "gres": "gpu:0",
                    "cpus_per_task": 1,
                    "mem_mb": 1024,
                    "runtime": 1,
                },
            }
        ],
    }
    path = directory / "plan.json"
    path.write_text(json.dumps(frozen))
    result = Canary(path, frozen, manifest)
    monkeypatch.setattr(control, "load_plan", result.load)
    return result


def test_missing_or_wrong_approval_never_contacts_scheduler(canary):
    report = canary.invoke(approve=None)
    assert report["outcome"] == "inconclusive"
    assert report["reason"] == "approval_required"
    assert report["live_qualified"] is False
    with pytest.raises(QualificationError, match="exactly"):
        canary.invoke(approve="a" * 64)
    assert canary.calls == []


@pytest.mark.parametrize("mutation", ["tasks", "gpu", "memory", "walltime", "workers"])
def test_unsafe_scope_is_refused_before_submission(canary, mutation):
    if mutation == "tasks":
        canary.frozen["runs"][0]["tasks"] = 50
    elif mutation == "gpu":
        canary.frozen["runs"][0]["resources"].update(gpu=1, gres="gpu:1")
    elif mutation == "memory":
        canary.frozen["runs"][0]["resources"]["mem_mb"] = 1000000
    elif mutation == "walltime":
        canary.frozen["site"]["controller"]["walltime"] = "24:00:00"
    else:
        canary.frozen["site"]["max_workers"] = 50
    with pytest.raises(QualificationError):
        canary.invoke()
    assert canary.calls == []


def test_gpu_requires_additional_opt_in_and_still_has_a_ceiling(canary):
    canary.frozen["runs"][0]["resources"].update(gpu=2, gres="gpu:2")
    with pytest.raises(QualificationError, match="one GPU"):
        canary.invoke(allow_gpu=True)
    canary.frozen["runs"][0]["resources"].update(gpu=1, gres="gpu:1")
    assert canary.invoke(allow_gpu=True)["outcome"] == "passed"


def test_previously_active_plan_is_not_reused_or_cancelled(canary):
    control.submit(canary.path, canary.frozen["digest"], scheduler=control.Slurm(run=canary.run))
    before = list(canary.calls)
    with pytest.raises(QualificationError, match="previously-active"):
        canary.invoke()
    assert canary.calls == before
    assert canary.cancelled == []


def test_atomic_freshness_race_never_cancels_the_other_submitter(canary):
    def submit(path, approve, **kwargs):
        control.submit(path, approve, scheduler=control.Slurm(run=canary.run))
        return control.submit(path, approve, **kwargs)

    lifecycle = SimpleNamespace(
        submit=submit, status=canary.status, cancel=control.cancel, resume=control.resume
    )
    report = canary.invoke(lifecycle=lifecycle)
    assert report["outcome"] == "inconclusive"
    assert canary.cancelled == []
    assert report["cleanup"] is None


def test_existing_task_evidence_is_not_overwritten(canary):
    output = canary.manifest.path(canary.manifest.tasks[0].designs)
    output.write_text("preserve this evidence\n")
    with pytest.raises(QualificationError, match="fresh"):
        canary.invoke()
    assert output.read_text() == "preserve this evidence\n"
    assert canary.calls == []


@pytest.mark.parametrize("cause", ["timeout", "interrupt", "lost_reply"])
def test_incomplete_execution_only_cleans_up_owned_jobs(canary, cause):
    canary.complete = False
    canary.interrupt = cause == "interrupt"
    canary.lose_reply = cause == "lost_reply"
    report = canary.invoke()
    assert report["outcome"] == "inconclusive"
    assert report["live_qualified"] is False
    assert canary.jobs["999"]["state"] == "RUNNING"
    assert set(canary.cancelled) == {job_id for job_id in canary.jobs if job_id != "999"}
    assert report["cleanup"]["state"] == "cancelled"
    assert json.loads(Path(report["report_path"]).read_text())["reason"] == report["reason"]


def test_missing_accounting_is_not_success_even_with_completed_artifacts(canary):
    canary.accounting = False
    report = canary.invoke()
    assert any(view["state"] == "completed" for view in report["observations"])
    assert report["outcome"] == "inconclusive"
    assert report["live_qualified"] is False
    assert (canary.manifest.directory / "ingested.json").exists()


def test_confirmed_completion_is_honestly_labelled_injected(canary):
    report = canary.invoke()
    assert report["outcome"] == "passed"
    assert report["evidence_kind"] == "injected"
    assert report["live_qualified"] is False
    assert report["cleanup"] is None
    assert {event["source"] for event in report["scheduler_evidence"]} == {"squeue", "sacct"}
    assert all(
        row["id"] != "999" for event in report["scheduler_evidence"] for row in event["rows"]
    )


def test_completed_scheduler_does_not_qualify_failed_tool_outcomes(canary):
    canary.task_status = "failed"
    report = canary.invoke()
    assert report["outcome"] == "failed"
    assert report["reason"] == "canary_tasks_did_not_succeed"
    assert report["live_qualified"] is False


def test_terminal_scheduler_failure_is_failed_not_inconclusive(canary):
    canary.failure = True
    report = canary.invoke()
    assert report["outcome"] == "failed"
    assert report["reason"] == "campaign_did_not_complete"
    assert canary.jobs["999"]["state"] == "RUNNING"


def test_cancel_resume_requires_accounted_worker_cancellation(canary):
    canary.complete = False
    report = canary.invoke(mode="cancel-resume")
    assert report["outcome"] == "passed"
    assert report["cancellation_verified"] is True
    final = report["observations"][-1]
    assert len(final["attempts"]) == 2
    assert final["state"] == "completed"
    assert canary.jobs["999"]["state"] == "RUNNING"
    assert report["live_qualified"] is False


def test_fast_canary_does_not_claim_to_have_exercised_cancellation(canary):
    report = canary.invoke(mode="cancel-resume")
    assert report["outcome"] == "inconclusive"
    assert report["reason"] == "worker_finished_before_cancellation"
    assert "cancellation_verified" not in report


def test_lifecycle_error_preserves_partial_outputs_during_scoped_cleanup(canary):
    canary.complete = False
    output = canary.manifest.path(canary.manifest.tasks[0].designs)

    def status(path, *, scheduler):
        canary.status(path, scheduler=scheduler)
        output.write_text("partial scientific evidence\n")
        raise RuntimeError("controller inspection failed")

    report = canary.invoke(
        lifecycle=SimpleNamespace(
            submit=control.submit,
            status=status,
            cancel=control.cancel,
            resume=control.resume,
        )
    )
    assert report["reason"] == "lifecycle_error"
    assert report["outcome"] == "inconclusive"
    assert output.read_text() == "partial scientific evidence\n"
    assert report["cleanup"]["state"] == "cancelled"
    assert canary.jobs["999"]["state"] == "RUNNING"


def test_cancel_resume_does_not_resume_with_missing_accounting(canary):
    canary.complete = False
    canary.accounting = False
    report = canary.invoke(mode="cancel-resume")
    assert report["outcome"] == "inconclusive"
    assert "cancellation_verified" not in report
    journal = json.loads((Path(canary.frozen["plan_dir"]) / "control/journal.json").read_text())
    assert len(journal["attempts"]) == 1
