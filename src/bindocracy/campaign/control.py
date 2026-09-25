"""Durable Slurm lifecycle around the ordinary Snakemake workflow.

Every scheduler call has a fsynced intent and a unique scheduler comment before
submission. An ambiguous reply is *not* permission to submit again. The same
lock guards cancellation's submission barrier and the sbatch transport shim.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from bindocracy.campaign.plan import load_plan
from bindocracy.runs.manifest import RunManifest
from bindocracy.runs.status import read_task_status


class CampaignError(RuntimeError):
    """The campaign cannot safely perform the requested transition."""


_TERMINAL = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
    "NOT_SUBMITTED",
}
_PREFIXES = ("SLURM_", "SBATCH_", "SRUN_")


def _clean_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_PREFIXES) and key != "CUDA_VISIBLE_DEVICES"
    }


class Slurm:
    """Scheduler transport seam; tests inject a subprocess-compatible runner."""

    def __init__(self, run=subprocess.run, sbatch=None):
        self.run = run
        self.sbatch = sbatch or shutil.which("sbatch") or "sbatch"

    def _call(self, argv, *, input=None):
        try:
            result = self.run(
                argv,
                input=input,
                text=True,
                capture_output=True,
                check=False,
                env=_clean_environment(),
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise CampaignError(f"scheduler outcome unknown: {error}") from error
        if result.returncode:
            raise CampaignError(f"scheduler command failed: {result.stderr.strip()}")
        return result.stdout

    def submit(self, args, *, input=None):
        output = self._call([self.sbatch, *args], input=input).strip()
        # Federated submissions require cluster-aware accounting/cancellation.
        # Do not silently discard the cluster suffix and cancel a local ID.
        if not re.fullmatch(r"[1-9][0-9]*", output):
            raise CampaignError(f"unrecognized sbatch reply; outcome unknown: {output!r}")
        return output

    def snapshot(self, since):
        args = ["--user", str(os.getuid())]
        local_start = datetime.fromisoformat(since).astimezone().strftime("%Y-%m-%dT%H:%M:%S")
        accounting = self._call(
            [
                "sacct",
                *args,
                "--allocations",
                "--noheader",
                "--parsable2",
                "--starttime",
                local_start,
                "--format=JobIDRaw,State,Comment%256",
            ]
        )
        queue = self._call(["squeue", *args, "--noheader", "--format=%i|%T|%k"])
        rows = {}
        for line in (accounting + "\n" + queue).splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 3 and re.fullmatch(r"[1-9][0-9]*", fields[0]):
                job_id, state, tag = fields[:3]
                rows[job_id] = {
                    "id": job_id,
                    "state": state.split()[0].rstrip("+"),
                    "tag": tag.strip(),
                }
        return list(rows.values())

    def cancel(self, job_id):
        self._call(["scancel", job_id])


def _directory(plan):
    return Path(plan["plan_dir"]) / "control"


@contextlib.contextmanager
def _locked(plan):
    directory = _directory(plan)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "journal.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield directory


def _atomic(path, value):
    """Persist both contents and rename before a scheduler side effect."""
    fd, temporary = tempfile.mkstemp(prefix=".journal-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(plan):
    path = _directory(plan) / "journal.json"
    if not path.exists():
        return {"schema_version": 1, "digest": plan["digest"], "attempts": []}
    journal = json.loads(path.read_text())
    if journal["digest"] != plan["digest"]:
        raise CampaignError("journal belongs to a different frozen plan")
    return journal


def _save(plan, journal):
    _atomic(_directory(plan) / "journal.json", journal)


def _attempt(journal, attempt_id):
    for attempt in journal["attempts"]:
        if attempt["id"] == attempt_id:
            return attempt
    raise CampaignError("unknown controller attempt")


def _intent(plan, attempt, role, **association):
    token = uuid4().hex
    entry = {
        "token": token,
        "tag": f"bd:{plan['digest'].split(':')[-1][:16]}:{attempt['id']}:{token}",
        "role": role,
        "id": None,
        "state": "UNKNOWN",
        **association,
    }
    attempt["jobs"].append(entry)
    return entry


def _reconcile(plan, journal, scheduler):
    if not journal["attempts"]:
        return
    rows = scheduler.snapshot(journal["attempts"][0]["created_at"])
    by_tag = {}
    for row in rows:
        by_tag.setdefault(row["tag"], []).append(row)
    for attempt in journal["attempts"]:
        for job in attempt["jobs"]:
            job["scheduler_confirmed"] = False
            matches = by_tag.get(job["tag"], [])
            if len(matches) > 1:
                raise CampaignError(f"multiple scheduler jobs share intent {job['tag']}")
            if matches:
                match = matches[0]
                if job["id"] is not None and job["id"] != match["id"]:
                    raise CampaignError("scheduler identity changed for a recorded intent")
                job.update(id=match["id"], state=match["state"])
                job["scheduler_confirmed"] = True
                job.pop("error", None)
            elif job["state"] not in _TERMINAL:
                job["state"] = "UNKNOWN"
    _save(plan, journal)


def _artifacts(plan):
    result = []
    for run in plan["runs"]:
        try:
            manifest = RunManifest.read(run["manifest"])
        except (OSError, ValueError) as error:
            result.append(
                {
                    "name": run["name"],
                    "tasks": [],
                    "ingested": False,
                    "generation_complete": False,
                    "error": str(error),
                }
            )
            continue
        tasks = []
        for task in manifest.tasks:
            try:
                status = read_task_status(manifest.path(task.status))
                state = status.status if status is not None else "missing"
                if status is not None and status.task_id != task.task_id:
                    state = "invalid"
            except (RuntimeError, OSError):
                state = "invalid"
            tasks.append(
                {
                    "task": task.task_id,
                    "status": state,
                    "output_exists": manifest.path(task.designs).is_file(),
                }
            )
        result.append(
            {
                "name": run["name"],
                "tasks": tasks,
                "ingested": (manifest.directory / "ingested.json").is_file(),
                "generation_complete": (manifest.directory / "generation.done").is_file(),
            }
        )
    return result


def _view(plan, journal):
    attempts = journal["attempts"]
    artifacts = _artifacts(plan)
    state = "planned"
    if attempts:
        latest = attempts[-1]
        jobs = latest["jobs"]
        if any(job["state"] == "UNKNOWN" for job in jobs):
            state = "unknown"
        elif any(job["state"] not in _TERMINAL for job in jobs):
            state = "cancelling" if latest["cancel_requested"] else "active"
        elif latest["cancel_requested"]:
            state = "cancelled"
        elif all(job["state"] == "COMPLETED" for job in jobs) and all(
            run["generation_complete"] and run["ingested"] for run in artifacts
        ):
            state = "completed"
        else:
            state = "stopped"
    return {"digest": plan["digest"], "state": state, "attempts": attempts, "artifacts": artifacts}


def status(plan_path, *, scheduler=None):
    """Reconcile exact scheduler tags and read artifacts; never write metrics."""
    plan = load_plan(plan_path, verify=False)
    with _locked(plan):
        journal = _read(plan)
        _reconcile(plan, journal, scheduler or Slurm())
        return _view(plan, journal)


def _approved(plan_path, approve):
    plan = load_plan(plan_path)
    if approve != plan["digest"]:
        raise CampaignError("--approve must exactly match the frozen plan digest")
    return plan


def _controller_args(plan, entry, script):
    controller = plan["site"]["controller"]
    args = [
        "--parsable",
        f"--job-name=bd-controller-{entry['token']}",
        f"--comment={entry['tag']}",
        "--export=ALL",
        "--nodes=1",
        "--ntasks=1",
        f"--account={controller['account']}",
        f"--partition={controller['partition']}",
        f"--cpus-per-task={controller['cpus']}",
        f"--mem={controller['memory_gb']}G",
        f"--time={controller['walltime']}",
        f"--output={script.parent / 'controller-%j.log'}",
        f"--chdir={script.parent}",
    ]
    if controller["gpus"]:
        args.append(f"--gpus={controller['gpus']}")
    if controller.get("constraint"):
        args.append(f"--constraint={controller['constraint']}")
    return [*args, str(script)]


def _launcher(plan, plan_path, attempt, scheduler):
    directory = _directory(plan) / attempt["id"]
    directory.mkdir()
    bin_dir = directory / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "sbatch"
    shim.write_text(
        "#!/bin/sh\nexec "
        + shlex.join(
            [
                plan["python"],
                "-m",
                "bindocracy.campaign.control",
                "worker-submit",
                str(plan_path),
                attempt["id"],
            ]
        )
        + ' "$@"\n'
    )
    shim.chmod(0o700)
    script = directory / "controller.sh"
    script.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        + f"ulimit -n {int(plan['site']['nofile'])}\nexec "
        + shlex.join(
            [
                plan["python"],
                "-m",
                "bindocracy.campaign.control",
                "run",
                str(plan_path),
                attempt["id"],
            ]
        )
        + "\n"
    )
    script.chmod(0o700)
    attempt["sbatch"] = (
        str(Path(scheduler.sbatch).resolve()) if "/" in scheduler.sbatch else scheduler.sbatch
    )
    return script


def _recover_interrupted(plan, journal, attempt):
    """Move interrupted shards aside before drivers can append duplicate IDs.

    Archives live beside each run, guaranteeing a same-filesystem rename even
    when the control directory is on a different mount. Record the source and
    destination before every move; a crash leaves preserved, discoverable data.
    """
    for run, observed in zip(plan["runs"], _artifacts(plan), strict=True):
        manifest = RunManifest.read(run["manifest"])
        interrupted = [
            task
            for task, state in zip(manifest.tasks, observed["tasks"], strict=True)
            if state["status"] in {"missing", "invalid"}
            or (state["status"] == "succeeded" and not state["output_exists"])
        ]
        if not interrupted:
            continue
        if observed["ingested"]:
            raise CampaignError(
                f"cannot restart damaged tasks of already-ingested run {run['name']}"
            )
        recovery = manifest.directory / ".campaign-recovery" / attempt["id"]
        sources = [
            pair
            for task in interrupted
            for pair in (
                (manifest.path(task.directory), recovery / f"{task.task_id:04d}" / "task"),
                (manifest.path(task.log), recovery / f"{task.task_id:04d}" / "task.log"),
            )
        ]
        sources.extend(
            (manifest.directory / name, recovery / name)
            for name in ("collected.json", "generation.done")
        )
        for source, destination in sources:
            if not source.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            operation = {"source": str(source), "archive": str(destination), "moved": False}
            attempt.setdefault("recovery", []).append(operation)
            _save(plan, journal)
            os.replace(source, destination)
            for parent in (source.parent, destination.parent):
                fd = os.open(parent, os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            operation["moved"] = True
            _save(plan, journal)
        for task in interrupted:
            manifest.path(task.directory).mkdir(parents=True, exist_ok=True)


def _submit(plan, plan_path, journal, scheduler, *, recover=False):
    attempt = {
        "id": uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "cancel_requested": False,
        "jobs": [],
    }
    journal["attempts"].append(attempt)
    script = _launcher(plan, plan_path, attempt, scheduler)
    entry = _intent(plan, attempt, "controller")
    entry["state"] = "NOT_SUBMITTED"
    _save(plan, journal)
    if recover:
        _recover_interrupted(plan, journal, attempt)
    entry["state"] = "UNKNOWN"
    _save(plan, journal)
    try:
        entry["id"] = scheduler.submit(_controller_args(plan, entry, script))
        entry["state"] = "SUBMITTED"
    except CampaignError as error:
        entry["error"] = str(error)
        _save(plan, journal)
        raise
    _save(plan, journal)
    return _view(plan, journal)


def submit(plan_path, approve, *, scheduler=None, require_fresh=False):
    """Launch once. Repeated submit reports the existing attempt, never duplicates."""
    plan_path = Path(plan_path).resolve()
    plan = _approved(plan_path, approve)
    scheduler = scheduler or Slurm()
    with _locked(plan):
        journal = _read(plan)
        if journal["attempts"]:
            if require_fresh:
                raise CampaignError("qualification requires a fresh plan with no previous attempts")
            _reconcile(plan, journal, scheduler)
            return _view(plan, journal)
        return _submit(plan, plan_path, journal, scheduler)


def cancel(plan_path, *, scheduler=None):
    """Close submission gate before stopping controllers and tagged workers.

    Repeat while state is cancelling/unknown: accounting may lag acceptance.
    Unknown outcomes remain unresolved rather than authorizing a second launch.
    """
    plan = load_plan(plan_path, verify=False)
    scheduler = scheduler or Slurm()
    with _locked(plan):
        journal = _read(plan)
        for attempt in journal["attempts"]:
            attempt["cancel_requested"] = True
        _save(plan, journal)
        _reconcile(plan, journal, scheduler)
        for role in ("controller", "worker"):
            for attempt in journal["attempts"]:
                for job in attempt["jobs"]:
                    if (
                        job["role"] == role
                        and job.get("scheduler_confirmed")
                        and job["state"] not in _TERMINAL
                    ):
                        scheduler.cancel(job["id"])
            _reconcile(plan, journal, scheduler)
        return _view(plan, journal)


def resume(plan_path, approve, *, scheduler=None):
    """Resume the same DAG only after every previous scheduler job is terminal."""
    plan_path = Path(plan_path).resolve()
    plan = _approved(plan_path, approve)
    scheduler = scheduler or Slurm()
    with _locked(plan):
        journal = _read(plan)
        _reconcile(plan, journal, scheduler)
        if not journal["attempts"]:
            raise CampaignError("plan has not been submitted; use submit")
        if any(
            job["state"] not in _TERMINAL
            for attempt in journal["attempts"]
            for job in attempt["jobs"]
        ):
            raise CampaignError(
                "cannot resume while an earlier job is live or its outcome is unknown"
            )
        return _submit(plan, plan_path, journal, scheduler, recover=True)


def _allowed(plan, journal, attempt_id):
    attempt = _attempt(journal, attempt_id)
    if journal["attempts"][-1]["id"] != attempt_id or attempt["cancel_requested"]:
        raise CampaignError("attempt is cancelled or superseded")
    return attempt


def _option(args, name):
    values = []
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            values.append(args[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    if len(values) != 1:
        raise CampaignError(f"expected exactly one {name} in executor submission")
    return values[0]


def _replace_option(args, name, value):
    result = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == name:
            skip = True
        elif not arg.startswith(name + "="):
            result.append(arg)
    return [*result, f"{name}={value}"]


def worker_submit(plan_path, attempt_id, args, *, scheduler=None):
    """sbatch transport shim, invoked only by the unmodified Slurm executor."""
    plan = load_plan(plan_path)
    if any(arg.split("=", 1)[0] in {"--array", "-a", "--clusters", "-M"} for arg in args):
        raise CampaignError("campaign tracking requires individual, local-cluster jobs")
    comment = _option(args, "--comment")
    association = [
        (run["name"], task)
        for run in plan["runs"]
        for task in range(run["tasks"])
        if comment == f"rule_generate_wildcards_{run['name']}_{task:04d}"
    ]
    if len(association) != 1:
        raise CampaignError(f"unrecognized worker identity: {comment!r}")
    wrapped = _option(args, "--wrap")
    with _locked(plan):
        journal = _read(plan)
        attempt = _allowed(plan, journal, attempt_id)
        run, task = association[0]
        if any(job.get("run") == run and job.get("task") == task for job in attempt["jobs"]):
            raise CampaignError("worker already has a submission intent in this attempt")
        entry = _intent(plan, attempt, "worker", run=run, task=task)
        _save(plan, journal)
        verify = shlex.join(
            [
                plan["python"],
                "-m",
                "bindocracy.campaign.control",
                "verify-worker",
                str(Path(plan_path).resolve()),
                attempt_id,
                entry["token"],
            ]
        )
        wrapped = f"ulimit -n {int(plan['site']['nofile'])} && {verify} && ( {wrapped}\n)"
        args = _replace_option(_replace_option(args, "--comment", entry["tag"]), "--wrap", wrapped)
        try:
            entry["id"] = (scheduler or Slurm(sbatch=attempt["sbatch"])).submit(args)
            entry["state"] = "SUBMITTED"
        except CampaignError as error:
            entry["error"] = str(error)
            _save(plan, journal)
            raise
        _save(plan, journal)
        return entry["id"]


def _verify_worker(plan_path, attempt_id, token):
    plan = load_plan(plan_path)
    with _locked(plan):
        journal = _read(plan)
        attempt = _allowed(plan, journal, attempt_id)
        entries = [
            job for job in attempt["jobs"] if job["token"] == token and job["role"] == "worker"
        ]
        if len(entries) != 1:
            raise CampaignError("worker has no durable submission intent")
        job_id = os.environ.get("SLURM_JOB_ID", "")
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            raise CampaignError("worker is not inside its scheduler allocation")
        if entries[0]["id"] not in (None, job_id):
            raise CampaignError("worker allocation does not match submission intent")
        entries[0].update(id=job_id, state="RUNNING")
        _save(plan, journal)
    # Deliberately leave SLURM_* and CUDA visibility untouched in workers.


def _run_controller(plan_path, attempt_id):
    plan = load_plan(plan_path)
    directory = _directory(plan)
    with (directory / "controller.lock").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignError("another controller is still executing this plan") from error
        with _locked(plan):
            journal = _read(plan)
            attempt = _allowed(plan, journal, attempt_id)
            controller = next(job for job in attempt["jobs"] if job["role"] == "controller")
            job_id = os.environ.get("SLURM_JOB_ID")
            if not job_id or controller["id"] not in (None, job_id):
                raise CampaignError("controller is not inside its recorded allocation")
            controller.update(id=job_id, state="RUNNING")
            _save(plan, journal)
        env = _clean_environment()
        env["PATH"] = str(directory / attempt_id / "bin") + os.pathsep + env.get("PATH", "")
        env["SNAKEMAKE_PROFILE"] = "none"
        workdir = directory / "work"
        workdir.mkdir(exist_ok=True)
        base = [
            plan["snakemake"],
            "--snakefile",
            plan["snakefile"],
            "--configfile",
            plan["workflow_index"],
            "--directory",
            str(workdir),
            "--profile",
            "none",
            "--workflow-profile",
            "none",
            "--persistence-backend",
            "file",
        ]
        if len(journal["attempts"]) > 1:
            from snakemake.io import IOFile
            from snakemake.persistence.file import FilePersistence

            subprocess.run([*base, "--unlock"], env=env, check=True)
            persistence = FilePersistence(path=workdir / ".snakemake", nolock=True)
            for run in plan["runs"]:
                manifest = RunManifest.read(run["manifest"])
                for task in manifest.tasks:
                    value = read_task_status(manifest.path(task.status))
                    if value is not None:
                        # Absence is normal for adopted outputs or a repeated
                        # recovery. Use Snakemake's idempotent operation rather
                        # than its CLI, which errors on absent metadata.
                        persistence.cleanup_metadata(IOFile(str(manifest.path(task.status))))
        site = plan["site"]
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (int(site["nofile"]), resource.getrlimit(resource.RLIMIT_NOFILE)[1]),
        )
        command = [
            *base,
            "--executor",
            "slurm",
            "--jobs",
            str(site["max_workers"]),
            "--local-cores",
            str(site["controller"]["cpus"]),
            "--resources",
            "db_writer=1",
            f"gpu={site['max_total_gpus'] - site['controller']['gpus']}",
            "--latency-wait",
            str(site["latency_wait"]),
            "--rerun-incomplete",
            "--rerun-triggers",
            "mtime",
            "--printshellcmds",
        ]
        result = subprocess.run(command, env=env, check=False)
        with _locked(plan):
            journal = _read(plan)
            _attempt(journal, attempt_id)["controller_exit_code"] = result.returncode
            _save(plan, journal)
        return result.returncode


def _main():
    action, plan_path, attempt_id, *args = sys.argv[1:]
    try:
        if action == "worker-submit":
            print(worker_submit(plan_path, attempt_id, args), flush=True)
            return 0
        if action == "verify-worker":
            _verify_worker(plan_path, attempt_id, args[0])
            return 0
        if action == "run":
            return _run_controller(plan_path, attempt_id)
        raise CampaignError(f"unknown internal action: {action}")
    except (CampaignError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
