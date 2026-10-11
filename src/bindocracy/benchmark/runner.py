"""Organizer-side attempt lifecycle, explicit local development and provisioned isolation.

Filesystem layout alone is NOT isolation. Production requires a separate agent
identity, protected harness/control files and externally confined scheduler jobs.
"""
from __future__ import annotations

import contextlib
import json
import os
import selectors
import signal
import socket
import socketserver
import struct
import subprocess
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bindocracy.benchmark.budget import BudgetLedger, atomic_json, file_lock
from bindocracy.benchmark.models import AttemptRecord


class RunnerError(RuntimeError):
    pass


def _now():
    return datetime.now(UTC)


def _record(directory):
    return AttemptRecord.model_validate_json((Path(directory) / "attempt.json").read_text())


def _update(directory, **changes):
    directory = Path(directory)
    with file_lock(directory / "attempt.lock"):
        record = _record(directory)
        record = AttemptRecord.model_validate({**record.model_dump(), **changes})
        atomic_json(directory / "attempt.json", record.model_dump(mode="json"))
        return record


def _event(directory, kind, **fields):
    with (
        file_lock(Path(directory) / "events.lock"),
        (Path(directory) / "events.jsonl").open("a") as stream,
    ):
        stream.write(json.dumps(dict(timestamp=_now().isoformat(), event=kind, **fields)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def record_usage(attempt_dir, record):
    """Append a provider-reported usage record, never infer missing tokens or cost.

Required: unique event_id, provider and model. Optional nonnegative input_tokens,
output_tokens and cost_usd; omitted values remain unknown. Duplicate identical
records are idempotent; conflicting duplicates are rejected.
"""
    directory = Path(attempt_dir)
    required = {"event_id", "provider", "model"}
    allowed = required | {"input_tokens", "output_tokens", "cost_usd", "source"}
    if not required <= record.keys() or record.keys() - allowed:
        raise RunnerError("usage needs event_id/provider/model and only documented fields")
    if any(not isinstance(record[k], str) or not record[k] for k in required):
        raise RunnerError("usage identity fields must be nonempty strings")
    for key in ("input_tokens", "output_tokens", "cost_usd"):
        value = record.get(key)
        if value is not None:
            import math
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RunnerError(f"invalid usage {key}")
            if key.endswith("tokens") and not isinstance(value, int):
                raise RunnerError("token usage must be integral")
    with file_lock(directory / "attempt.lock"):
        attempt = _record(directory)
        usage = dict(attempt.usage)
        records = usage.get("records", [])
        previous = next((r for r in records if r["event_id"] == record["event_id"]), None)
        if previous is not None:
            if previous != record:
                raise RunnerError("usage event identity reused with different data")
            return usage
        records = [*records, record]
        totals = {key: sum(r[key] for r in records) if all(r.get(key) is not None for r in records) else None
                  for key in ("input_tokens", "output_tokens", "cost_usd")}
        usage.update(records=records, reported_totals=totals,
                     completeness="reported_records_only; unreported usage unknown")
        atomic_json(directory / "attempt.json", attempt.model_copy(update={"usage": usage}).model_dump(mode="json"))
        return usage


def _process_identity(pid):
    try:
        # comm may contain spaces and parentheses; fields after its last ')' are stable.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def _stop_process(directory, *, grace=2.0):
    identity_path = Path(directory) / "process.json"
    if not identity_path.exists():
        return {"process": "not_started"}
    identity = json.loads(identity_path.read_text())
    pid = identity["pid"]
    if _process_identity(pid) != identity:
        return {"process": "not_live_or_identity_changed", "descendants": "unknown_after_leader_loss"}
    try:
        os.killpg(pid, signal.SIGTERM)
        until = time.monotonic() + grace
        while time.monotonic() < until:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return {"process": "terminated"}
            time.sleep(0.02)
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return {"process": "group_killed"}


def _scheduler_cleanup(directory, scheduler=None):
    from bindocracy.campaign.control import _TERMINAL, Slurm
    ledger = BudgetLedger(directory)
    value = ledger.snapshot()
    reservations = value["reservations"]
    if not any(r["tag"] for r in reservations.values()):
        return {"scheduler": "no_jobs", "unresolved": []}
    scheduler = scheduler or Slurm(timeout=10)
    unresolved, cancelled = [], []
    cleanup_deadline = time.monotonic() + 30
    try:
        rows = scheduler.snapshot(_record(directory).created_at.isoformat())
    except (RuntimeError, OSError) as error:
        return {"scheduler": "unknown", "error": str(error), "unresolved": list(reservations)}
    by_tag = {}
    for row in rows:
        by_tag.setdefault(row["tag"], []).append(row)
    for key, entry in reservations.items():
        if not entry["tag"] or entry["terminal"]:
            continue
        if time.monotonic() >= cleanup_deadline:
            unresolved.append(key)
            continue
        matches = by_tag.get(entry["tag"], [])
        if len(matches) != 1 or entry["scheduler_id"] not in (None, matches[0]["id"]):
            unresolved.append(key)
            continue
        row = matches[0]
        ledger.record_job(key, row["id"], tag=entry["tag"], state=row["state"])
        try:
            if row["state"] not in _TERMINAL:
                scheduler.cancel(row["id"])
                cancelled.append(row["id"])
            # Cancellation is not terminal allocation accounting. Keep reserved.
            accounting = scheduler.accounting(row["id"]) if hasattr(scheduler, "accounting") else None
            if accounting and accounting["terminal"]:
                ledger.reconcile(key, scheduler_id=row["id"], tag=entry["tag"], **accounting)
            else:
                unresolved.append(key)
        except (RuntimeError, OSError) as error:
            unresolved.append(key)
            _event(directory, "scheduler_cleanup_error", reservation=key, error=str(error))
    return {"scheduler": "pending_accounting" if unresolved else "accounted", "cancelled": cancelled, "unresolved": unresolved}


def status(attempt_dir, *, reconcile=False, scheduler=None):
    directory = Path(attempt_dir).resolve()
    if reconcile:
        record = _record(directory)
        if record.state not in {"prepared", "running"} and (directory / "budget.json").exists():
            cleanup = {**record.cleanup, **_scheduler_cleanup(directory, scheduler)}
            _update(directory, cleanup=cleanup)
    record = _record(directory)
    budget = BudgetLedger(directory).snapshot() if (directory / "budget.json").exists() else None
    return {"attempt": record.model_dump(mode="json"), "budget": budget}


def cancel(attempt_dir, *, scheduler=None, reason="cancelled"):
    directory = Path(attempt_dir).resolve()
    ledger = BudgetLedger(directory)
    if not ledger.path.exists():
        record = _update(directory, state="infrastructure_failed", finished_at=_now(),
                         error="startup interrupted before budget initialization",
                         cleanup={"process": "not_started", "scheduler": "no_jobs"})
        return {"attempt": record.model_dump(mode="json"), "budget": None}
    ledger.close(reason)
    record = _record(directory)
    if record.state in {"prepared", "running"}:
        _update(directory, state=reason, finished_at=_now())
    cleanup = {**_stop_process(directory), **_scheduler_cleanup(directory, scheduler)}
    socket_path = record.agent.get("broker_socket")
    if socket_path and Path(socket_path).is_socket():
        Path(socket_path).unlink()
        cleanup["broker"] = "socket_removed"
    _update(directory, cleanup=cleanup, jobs=tuple(ledger.snapshot()["reservations"].values()))
    _event(directory, "cleanup", **cleanup)
    return status(directory)


def recover(attempt_dir, *, scheduler=None):
    """Fail closed after a runner crash; never restart an uncertain command."""
    directory = Path(attempt_dir).resolve()
    with (directory / "runner.lock").open("a") as guard:
        try:
            import fcntl
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunnerError("attempt runner is still active; use cancel instead") from error
        return cancel(directory, scheduler=scheduler, reason="infrastructure_failed")


def _broker_request(socket_path, operation, **arguments):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(300)
        client.connect(str(socket_path))
        client.sendall(json.dumps({"operation": operation, "arguments": arguments}).encode() + b"\n")
        stream = client.makefile("rb")
        reply = json.loads(stream.readline(8 * 1024 * 1024))
    if "error" in reply:
        raise RunnerError(reply["error"])
    return reply["result"]


class _Broker(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    block_on_close = False


@contextlib.contextmanager
def _serve_broker(directory, socket_path, agent_uid):
    from bindocracy.campaign import control

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                self.request.settimeout(15)
                _, uid, _ = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if uid != agent_uid:
                    raise RunnerError("broker peer is not the configured agent identity")
                request = json.loads(self.rfile.readline(1024 * 1024))
                operation, arguments = request["operation"], request["arguments"]
                if operation == "accept":
                    from bindocracy.benchmark.submission import accept_submission
                    BudgetLedger(directory).check()
                    source = Path(arguments["source"]).resolve()
                    root = _record(directory).workspace.resolve()
                    if not source.is_relative_to(root):
                        raise RunnerError("submission source is outside this agent workspace")
                    result = accept_submission(source, directory)
                elif operation == "usage":
                    result = record_usage(directory, arguments["record"])
                elif operation in {"submit", "resume", "status", "cancel"}:
                    permitted = {"plan_path", "approve", "require_fresh"}
                    if arguments.keys() - permitted:
                        raise RunnerError("unsupported campaign broker arguments")
                    record = _record(directory)
                    plan_path = Path(arguments["plan_path"]).resolve()
                    if not plan_path.is_relative_to(record.workspace.resolve()):
                        raise RunnerError("campaign plan is outside the assigned workspace")
                    plan = control.load_plan(arguments["plan_path"], verify=operation in {"submit", "resume"})
                    from bindocracy.benchmark.budget import ledger_for_plan
                    ledger = ledger_for_plan(plan)
                    if ledger is None or ledger.directory != directory:
                        raise RunnerError("plan does not belong to this broker attempt")
                    record = _record(directory)
                    if any(plan.get(field) != getattr(record, field)
                           for field in ("attempt_id", "task_digest", "environment_digest")):
                        raise RunnerError("plan identity differs from organizer attempt")
                    if operation in {"submit", "resume"}:
                        from bindocracy.campaign.plan import _executables
                        python, snakemake = _executables()
                        if plan["python"] != python or plan["snakemake"] != snakemake:
                            raise RunnerError("plan requires untrusted execution binaries")
                        environment = json.loads((directory / "environment.json").read_text())
                        trusted_workflow = Path(environment["harness_path"]) / "workflow" / "Snakefile"
                        if Path(plan["snakefile"]).read_bytes() != trusted_workflow.read_bytes():
                            raise RunnerError("plan modifies the organizer workflow")
                        from bindocracy.runs.manifest import RunManifest
                        task = json.loads((directory / "task.json").read_text())
                        for run in plan["runs"]:
                            if RunManifest.read(run["manifest"]).tool not in task["allowed_tools"]:
                                raise RunnerError("plan uses a tool outside the task policy")
                        ledger.check()
                    result = getattr(control, operation)(**arguments, scheduler=control.Slurm(timeout=10),
                                                         _expected_digest=plan["digest"])
                else:
                    raise RunnerError("unsupported broker operation")
                response = {"result": result}
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                response = {"error": str(error)}
            self.wfile.write(json.dumps(response).encode() + b"\n")

    if socket_path.exists():
        raise RunnerError("broker socket path already exists")
    server = _Broker(str(socket_path), Handler)
    socket_path.chmod(0o600)
    if os.geteuid() == 0:
        os.chown(socket_path, agent_uid, -1)
    elif agent_uid != os.getuid():
        # A preprovisioned traversable directory and ACL are required for separate UID.
        socket_path.chmod(0o666)
        # Authentication uses SO_PEERCRED, not socket mode alone.
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        socket_path.unlink(missing_ok=True)


def run_attempt(workspace, attempt_dir, command, *, development=False, agent_uid=None,
                isolation=None, agent=None, launcher=(), broker_socket=None, scheduler=None):
    from bindocracy.benchmark.workspace import load_workspace

    workspace = load_workspace(workspace)
    directory = Path(attempt_dir).resolve()
    if directory == workspace.root or directory.is_relative_to(workspace.root):
        raise RunnerError("organizer attempt directory must be outside agent workspace")
    if not command or any(not isinstance(a, str) or not a for a in command):
        raise RunnerError("agent command must be a nonempty argv")
    uid = os.getuid() if agent_uid is None else agent_uid
    required = ("protected_control", "protected_harness", "isolated_agent", "isolated_scheduler_jobs", "controlled_scheduler", "network_policy")
    if not development:
        if uid == os.getuid() or not isolation or any(isolation.get(k) is not True for k in required):
            raise RunnerError("production requires distinct agent UID and explicit external isolation prerequisites")
        if os.geteuid() != 0 and not launcher:
            raise RunnerError("distinct UID requires root privilege drop or an organizer-provisioned launcher")
        if broker_socket is None:
            raise RunnerError("production requires a broker socket in a preprovisioned traversable directory")
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    with contextlib.ExitStack() as stack:
        stack.enter_context(file_lock(directory / "runner.lock"))
        if broker_socket is None:
            import tempfile
            socket_directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="bd-broker-"))
            broker_socket = Path(socket_directory) / "control.sock"
        return _execute_attempt(workspace, directory, command, development=development,
                                uid=uid, isolation=isolation, agent=agent, launcher=launcher,
                                socket_path=Path(broker_socket).absolute(), scheduler=scheduler)


def _execute_attempt(workspace, directory, command, *, development, uid, isolation,
                     agent, launcher, socket_path, scheduler):
    started = _now()
    deadline = started + timedelta(seconds=workspace.task.budget.walltime_seconds)
    record = AttemptRecord(
        attempt_id=workspace.attempt_id, workspace=workspace.root,
        task_id=workspace.task.task_id, task_digest=workspace.task_digest,
        environment_digest=workspace.environment_digest,
        command=tuple(command), created_at=started,
        agent={**(agent or {}), "agent_uid": uid, "organizer_uid": os.getuid(),
               "isolation": "unisolated/development" if development else "externally_provisioned",
               "prerequisites": isolation or {}, "launcher": list(launcher),
               "broker_socket": str(socket_path)},
        usage={"records": [], "reported_totals": {"input_tokens": None, "output_tokens": None, "cost_usd": None},
               "completeness": "unknown"},
    )
    atomic_json(directory / "attempt.json", record.model_dump(mode="json"))
    for filename, model in (("task.json", workspace.task), ("environment.json", workspace.environment), ("workspace.json", workspace)):
        atomic_json(directory / filename, model.model_dump(mode="json"))
        (directory / filename).chmod(0o400)
    import shutil
    shutil.copyfile(workspace.task.target.sequence_fasta, directory / "target.fasta")
    (directory / "target.fasta").chmod(0o400)
    target_asset = workspace.input_assets.get("target:sequence_fasta")
    if target_asset is not None:
        import hashlib
        if hashlib.sha256((directory / "target.fasta").read_bytes()).hexdigest() != target_asset.sha256:
            raise RunnerError("target input changed during organizer snapshot")
    ledger = BudgetLedger(directory)
    ledger.initialize(workspace.task.budget, attempt_id=workspace.attempt_id, deadline=deadline,
                      scheduler_enabled=not development)
    env = dict(os.environ)
    env.update(BINDOCRACY_ATTEMPT_ID=workspace.attempt_id, BINDOCRACY_WORKSPACE=str(workspace.root),
               BINDOCRACY_ATTEMPT_DIR=str(directory), BINDOCRACY_BROKER=str(socket_path),
               BINDOCRACY_LOCAL_CPU_ALLOCATION="agent",
               BINDOCRACY_DEVELOPMENT="1" if development else "0", PYTHONDONTWRITEBYTECODE="1")
    # Agent cannot redirect Python imports into a writable replacement harness.
    if not development:
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
    launch_options = {}
    if uid != os.getuid() and os.geteuid() == 0 and not launcher:
        import pwd
        account = pwd.getpwuid(uid)
        launch_options.update(user=uid, group=account.pw_gid, extra_groups=[])
        env.update(HOME=account.pw_dir, USER=account.pw_name, LOGNAME=account.pw_name)
    stopped = threading.Event()
    old_signals = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_signals[sig] = signal.signal(sig, lambda *_: stopped.set())
    process = None
    local_start = time.monotonic()
    cutoff = local_start + max(0, (deadline - _now()).total_seconds())
    try:
        with _serve_broker(directory, socket_path, uid):
            remaining = min((deadline - _now()).total_seconds(),
                            ledger.snapshot()["monotonic_deadline"] - time.monotonic()) - 0.01
            ledger.reserve("agent", gpus=0, cpus=1, walltime_seconds=remaining,
                           role="agent", memory_gb=1)
            _update(directory, state="running", started_at=started, deadline=deadline)
            _event(directory, "started", command=list(command), deadline=deadline.isoformat())
            with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
                import shutil
                import sys
                argv = [*launcher, *command]
                if shutil.which(argv[0], path=env.get("PATH")) is None:
                    raise FileNotFoundError(f"agent launch executable does not exist: {argv[0]}")
                # The trusted child waits for durable PID identity before exec.
                # If the runner dies before release, pipe EOF exits without running.
                bootstrap = "import os,sys; token=sys.stdin.buffer.read(1); sys.exit(125) if token!=b'1' else os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
                process = subprocess.Popen([sys.executable, "-I", "-S", "-c", bootstrap, *argv],
                                           cwd=workspace.root, env=env, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           start_new_session=True, **launch_options)
                atomic_json(directory / "process.json", _process_identity(process.pid))
                _update(directory, pid=process.pid)
                process.stdin.write(b"1")
                process.stdin.close()
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ, ("stdout", stdout))
                selector.register(process.stderr, selectors.EVENT_READ, ("stderr", stderr))
                state = "completed"
                while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                    if stopped.is_set() or _record(directory).state != "running":
                        state = "cancelled"
                        break
                    if time.monotonic() >= cutoff:
                        state = "timed_out"
                        break
                    for key, _ in selector.select(timeout=min(0.1, max(0, cutoff - time.monotonic()))):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        channel, stream = key.data
                        stream.write(data)
                        stream.flush()
                        _event(directory, "output", channel=channel, text=data.decode("utf-8", errors="replace"))
                ledger.close(state)
                # Keep the session leader unreaped until its group has been killed;
                # background children must not outlive even a successful command.
                if os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                    process_cleanup = _stop_process(directory)
                else:
                    process_cleanup = {"process": "exited_and_group_cleanup"}
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=5)
                ledger.reconcile("agent", gpus=0, cpus=1,
                                 elapsed_seconds=time.monotonic() - local_start, terminal=True)
                for key in list(selector.get_map().values()):
                    os.set_blocking(key.fileobj.fileno(), False)
                    while True:
                        try:
                            data = os.read(key.fileobj.fileno(), 65536)
                        except BlockingIOError:
                            break
                        if not data:
                            break
                        channel, stream = key.data
                        stream.write(data)
                        _event(directory, "output", channel=channel, text=data.decode("utf-8", errors="replace"))
                selector.close()
                process.stdout.close()
                process.stderr.close()
            if state == "completed" and process.returncode:
                state = "agent_failed"
            current = _record(directory)
            if current.state == "running":
                _update(directory, state=state, finished_at=_now(), returncode=process.returncode)
            cleanup = {**process_cleanup, **_scheduler_cleanup(directory, scheduler)}
            _update(directory, cleanup=cleanup, jobs=tuple(ledger.snapshot()["reservations"].values()))
            _event(directory, "finished", state=_record(directory).state, returncode=process.returncode)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        ledger.close("infrastructure_failed")
        _update(directory, state="infrastructure_failed", finished_at=_now(), error=str(error))
        cleanup = {**_stop_process(directory), **_scheduler_cleanup(directory, scheduler)}
        terminal = process is None
        if process is not None:
            try:
                process.wait(timeout=5)
                terminal = True
            except subprocess.TimeoutExpired:
                cleanup["process"] = "kill_unconfirmed"
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        local_reservation = ledger.snapshot()["reservations"].get("agent")
        if terminal and local_reservation is not None and not local_reservation["terminal"]:
            ledger.reconcile("agent", gpus=0, cpus=1,
                             elapsed_seconds=time.monotonic() - local_start, terminal=True)
        _update(directory, cleanup=cleanup)
        _event(directory, "infrastructure_failed", error=str(error))
    finally:
        for sig, previous in old_signals.items():
            signal.signal(sig, previous)
    return status(directory)
