"""Opt-in, bounded qualification of a fresh frozen CPU canary through campaign control.

This module observes the existing transport; it does not schedule workers. Tests may
inject a lifecycle and subprocess runner, but such evidence is never live qualification.
"""

from __future__ import annotations

import contextlib
import json
import math
import signal
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from bindocracy.campaign import control
from bindocracy.campaign.plan import load_plan
from bindocracy.config.models import slurm_walltime_seconds
from bindocracy.runs.manifest import RunManifest, write_json_atomic


class QualificationError(ValueError):
    """The plan or qualification request is unsafe to execute."""


def _bounded(value, maximum, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{label} must be numeric")
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise QualificationError(f"{label} must be positive and at most {maximum}")


def _check_plan(plan, allow_gpu):
    runs = plan["runs"]
    if len(runs) != 1 or runs[0]["tasks"] != 1 or plan["site"]["max_workers"] != 1:
        raise QualificationError("qualification requires exactly one run, one task, one worker")
    controller = plan["site"]["controller"]
    worker = runs[0]["resources"]
    if allow_gpu:
        if controller["gpus"] not in {0, 1} or worker["gpu"] not in {0, 1}:
            raise QualificationError("GPU opt-in allows at most one GPU per allocation")
        if not 0 < plan["site"]["max_total_gpus"] <= 2:
            raise QualificationError("GPU opt-in requires a total GPU ceiling at most two")
        if worker.get("gres") != f"gpu:{worker['gpu']}":
            raise QualificationError("worker GPU directives disagree")
    elif controller["gpus"] != 0 or worker["gpu"] != 0 or worker.get("gres") != "gpu:0":
        raise QualificationError("CPU-only qualification refuses GPUs without --allow-gpu")
    allowed = {
        "slurm_account",
        "slurm_partition",
        "gres",
        "gpu",
        "cpus_per_task",
        "mem_mb",
        "runtime",
    }
    if set(worker) - allowed:
        raise QualificationError("qualification refuses extra scheduler resource directives")
    for value, maximum, label in (
        (controller["cpus"], 2, "controller CPUs"),
        (controller["memory_gb"], 8 if allow_gpu else 4, "controller memory GiB"),
        (
            slurm_walltime_seconds(controller["walltime"]),
            900 if allow_gpu else 600,
            "controller seconds",
        ),
        (worker["cpus_per_task"], 8 if allow_gpu else 2, "worker CPUs"),
        (worker["mem_mb"], 65536 if allow_gpu else 4096, "worker memory MiB"),
        (worker["runtime"], 10 if allow_gpu else 5, "worker minutes"),
    ):
        _bounded(value, maximum, label)
    manifest = RunManifest.read(runs[0]["manifest"])
    if len(manifest.tasks) != 1:
        raise QualificationError("qualification requires exactly one manifest task")
    task = manifest.tasks[0]
    _bounded(task.n_requested, 2, "requested records")
    if task.n_generated is not None:
        _bounded(task.n_generated, 2, "generated records")
    for label in ("n_designs", "n_predictions"):
        if runs[0].get(label) is not None:
            _bounded(runs[0][label], 2, label)
    # Planning is allowed to create empty task directories, not execution evidence.
    task_dir = manifest.path(task.directory)
    if task_dir.exists() and any(task_dir.iterdir()):
        raise QualificationError("qualification requires fresh task directories")
    for relative in (
        task.designs,
        task.status,
        task.log,
        "collected.json",
        "ingested.json",
        "generation.done",
        ".campaign-recovery",
    ):
        if manifest.path(relative).exists():
            raise QualificationError(f"qualification refuses prior execution evidence: {relative}")


class _ObservedSlurm(control.Slurm):
    """Retain source-specific, exact-attempt tag evidence from the real transport."""

    def __init__(self, runner, clock, deadline):
        self.runner = runner
        self.clock = clock
        self.deadline = deadline
        self.attempt_prefixes = set()
        self.started = False
        self.evidence = []
        super().__init__(run=self._run)

    def submit(self, args, *, input=None):
        tag = control._option(args, "--comment")
        self.attempt_prefixes.add(tag.rsplit(":", 1)[0] + ":")
        # Set before sbatch: a lost response can still have created an allocation.
        self.started = True
        return super().submit(args, input=input)

    def cancel(self, job_id):
        # control.cancel verifies current tags; this additional fence excludes any
        # attempt another operator may have resumed concurrently.
        owned = any(
            row["id"] == job_id
            for event in self.evidence
            if not event["returncode"]
            for row in event["rows"]
        )
        if owned:
            super().cancel(job_id)

    def _run(self, argv, **kwargs):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(argv, 0)
        kwargs["timeout"] = min(kwargs.get("timeout", 15), 15, remaining)
        result = self.runner(argv, **kwargs)
        source = Path(argv[0]).name
        if source in {"squeue", "sacct"} and self.attempt_prefixes:
            rows = []
            for line in result.stdout.splitlines():
                fields = line.strip().split("|")
                if (
                    len(fields) >= 3
                    and fields[1].strip()
                    and any(
                        fields[2].strip().startswith(prefix) for prefix in self.attempt_prefixes
                    )
                ):
                    rows.append(
                        {
                            "id": fields[0],
                            "state": fields[1].split()[0].rstrip("+"),
                            "tag": fields[2].strip(),
                        }
                    )
            self.evidence.append({"source": source, "returncode": result.returncode, "rows": rows})
        return result


@contextlib.contextmanager
def _interrupt_on_term():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _jobs(view):
    return [job for attempt in view["attempts"] for job in attempt["jobs"]]


def _accounted(jobs, evidence, states):
    rows = [
        row
        for event in evidence
        if event["source"] == "sacct" and not event["returncode"]
        for row in event["rows"]
    ]
    return bool(jobs) and all(
        any(
            row["id"] == job["id"] and row["tag"] == job["tag"] and row["state"] in states
            for row in rows
        )
        for job in jobs
    )


def qualify(
    plan_path,
    approve=None,
    *,
    mode="completion",
    allow_gpu=False,
    timeout=900,
    poll_interval=5,
    lifecycle=None,
    runner=None,
    loader=None,
    clock=None,
    sleep=None,
):
    """Return and persist an honest report; no exact approval means no scheduler access.

    A plan must be exclusively operated during qualification. ``cancel-resume``
    additionally requires an observed active worker and accounted cancellation
    before using the ordinary resume path. A fast canary may be inconclusive.
    """
    if mode not in {"completion", "cancel-resume"}:
        raise QualificationError("mode must be completion or cancel-resume")
    _bounded(timeout, 1800, "timeout seconds")
    _bounded(poll_interval, 30, "poll interval seconds")
    if poll_interval < 1:
        raise QualificationError("poll interval must be at least one second")
    injected = any(value is not None for value in (lifecycle, runner, loader, clock, sleep))
    lifecycle = lifecycle or control
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    plan_path = Path(plan_path).resolve()
    plan = (loader or load_plan)(plan_path)
    _check_plan(plan, allow_gpu)
    if approve is not None and approve != plan["digest"]:
        raise QualificationError("--approve must exactly match the frozen plan digest")
    directory = Path(plan["plan_dir"]) / "control"
    journal_path = directory / "journal.json"
    # This is only the early refusal; submit(require_fresh=True) repeats it atomically.
    if journal_path.exists() and json.loads(journal_path.read_text()).get("attempts"):
        raise QualificationError("qualification refuses a previously-active plan")
    report = {
        "schema_version": 1,
        "digest": plan["digest"],
        "mode": mode,
        "allow_gpu": allow_gpu,
        "started_at": datetime.now(UTC).isoformat(),
        "outcome": "inconclusive",
        "live_qualified": False,
        "evidence_kind": "injected" if injected else "live",
        "reason": "approval_required",
        "observations": [],
        "scheduler_evidence": [],
        "cleanup": None,
    }
    if approve is None:
        report["evidence_kind"] = "not_executed"
        return report
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / f"qualification-{uuid4().hex}.json"
    report["report_path"] = str(report_path)
    scheduler = _ObservedSlurm(runner or subprocess.run, clock, clock() + timeout)
    cancel_requested = False
    resumed = mode == "completion"
    cleanup_needed = False

    def observe(view):
        report["observations"].append(view)
        report["scheduler_evidence"] = scheduler.evidence
        write_json_atomic(report_path, report)

    try:
        with _interrupt_on_term():
            observe(lifecycle.submit(plan_path, approve, scheduler=scheduler, require_fresh=True))
            cleanup_needed = True
            while clock() < scheduler.deadline:
                view = lifecycle.status(plan_path, scheduler=scheduler)
                observe(view)
                jobs = _jobs(view)
                if mode == "cancel-resume" and not resumed:
                    if not cancel_requested:
                        active_workers = [
                            job
                            for job in jobs
                            if job["role"] == "worker"
                            and job.get("scheduler_confirmed")
                            and job["state"] in {"PENDING", "RUNNING"}
                        ]
                        if active_workers:
                            observe(lifecycle.cancel(plan_path, scheduler=scheduler))
                            cancel_requested = True
                        elif view["state"] in {"completed", "stopped"}:
                            report["reason"] = "worker_finished_before_cancellation"
                            break
                    elif view["state"] == "cancelled":
                        workers = [job for job in jobs if job["role"] == "worker"]
                        if _accounted(workers, scheduler.evidence, {"CANCELLED"}) and _accounted(
                            jobs, scheduler.evidence, control._TERMINAL - {"NOT_SUBMITTED"}
                        ):
                            report["cancellation_verified"] = True
                            observe(lifecycle.resume(plan_path, approve, scheduler=scheduler))
                            resumed = True
                else:
                    latest = view["attempts"][-1]["jobs"] if view["attempts"] else []
                    roles = {job["role"] for job in latest}
                    if view["state"] == "completed":
                        tasks = [task for run in view["artifacts"] for task in run["tasks"]]
                        if not tasks or any(
                            task["status"] != "succeeded" or not task["output_exists"]
                            for task in tasks
                        ):
                            report["outcome"] = "failed"
                            report["reason"] = "canary_tasks_did_not_succeed"
                            break
                        if roles == {"controller", "worker"} and _accounted(
                            latest, scheduler.evidence, {"COMPLETED"}
                        ):
                            report["outcome"] = "passed"
                            report["reason"] = "completion_and_accounting_verified"
                            report["live_qualified"] = not injected
                            cleanup_needed = False
                            break
                    elif view["state"] in {"stopped", "cancelled"}:
                        report["outcome"] = "failed"
                        report["reason"] = "campaign_did_not_complete"
                        break
                sleep(min(poll_interval, max(0, scheduler.deadline - clock())))
            else:
                report["reason"] = "timeout_or_incomplete_accounting"
    except KeyboardInterrupt:
        report["reason"] = "interrupted"
    except Exception as error:  # noqa: BLE001 - persist evidence and clean up even unexpected failures
        report["reason"] = "lifecycle_error"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        # Only a transport submission performed by this invocation grants cleanup ownership.
        # A require_fresh refusal (including races) must never cancel another invocation.
        if scheduler.started and (cleanup_needed or report["outcome"] != "passed"):
            scheduler.deadline = clock() + 60
            try:
                with _interrupt_on_term():
                    cleanup = lifecycle.cancel(plan_path, scheduler=scheduler)
                    while cleanup["state"] in {"active", "cancelling", "unknown"}:
                        if clock() >= scheduler.deadline:
                            break
                        sleep(min(poll_interval, max(0, scheduler.deadline - clock())))
                        cleanup = lifecycle.cancel(plan_path, scheduler=scheduler)
                    report["cleanup"] = cleanup
            except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - cleanup failure must remain explicit evidence
                report["cleanup"] = {"state": "unknown", "error": str(error)}
        report["scheduler_evidence"] = scheduler.evidence
        report["finished_at"] = datetime.now(UTC).isoformat()
        write_json_atomic(report_path, report)
    return report
