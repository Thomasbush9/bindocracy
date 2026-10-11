"""Organizer-owned, fail-closed allocation ledger for controlled submission routes.

A file lock serializes reservations and scheduler submission against gate closure.
This is not a scheduler credential firewall: provision that externally.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from bindocracy.benchmark.models import BudgetPolicy


class BudgetError(RuntimeError):
    """A request cannot safely be admitted or reconciled."""


def atomic_json(path: Path, value) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def file_lock(path: Path):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


class BudgetLedger:
    def __init__(self, attempt_dir):
        self.directory = Path(attempt_dir).resolve()
        self.path = self.directory / "budget.json"

    @contextlib.contextmanager
    def _locked(self):
        with file_lock(self.directory / "budget.lock"):
            value = json.loads(self.path.read_text())
            yield value

    def initialize(self, policy: BudgetPolicy, *, attempt_id: str, deadline: datetime,
                   scheduler_enabled=False):
        with file_lock(self.directory / "budget.lock"):
            if self.path.exists():
                raise BudgetError("budget ledger already exists")
            atomic_json(self.path, {
                "schema_version": 1, "attempt_id": attempt_id,
                "policy": policy.model_dump(mode="json"), "deadline": deadline.isoformat(),
                "boot_id": _boot_id(),
                "monotonic_deadline": time.monotonic() + max(0, (deadline - datetime.now(UTC)).total_seconds()),
                "scheduler_enabled": scheduler_enabled, "closed": False,
                "reason": None, "reservations": {},
            })

    def snapshot(self):
        with self._locked() as value:
            return value

    def _check(self, value):
        expired = datetime.now(UTC) >= datetime.fromisoformat(value["deadline"])
        expired |= value["boot_id"] != _boot_id()
        expired |= time.monotonic() >= value["monotonic_deadline"]
        if value["closed"] or expired:
            raise BudgetError("benchmark submission gate is closed or expired")

    def check(self):
        with self._locked() as value:
            self._check(value)

    def close(self, reason):
        with self._locked() as value:
            value.update(closed=True, reason=value["reason"] or reason)
            atomic_json(self.path, value)

    def _reserve(self, value, reservation_id, *, gpus, cpus, walltime_seconds,
                 memory_gb=0, tag=None, role="worker", plan=None):
        self._check(value)
        numbers = (gpus, cpus, walltime_seconds, memory_gb)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in numbers):
            raise BudgetError("allocation resources must be finite numbers")
        if not isinstance(gpus, int) or not isinstance(cpus, int):
            raise BudgetError("GPU and CPU allocations must be integral")
        if gpus < 0 or cpus <= 0 or walltime_seconds <= 0 or memory_gb < 0:
            raise BudgetError("invalid allocation resources")
        request = {"gpus": gpus, "cpus": cpus, "walltime_seconds": walltime_seconds,
                   "memory_gb": memory_gb, "tag": tag, "role": role, "plan": plan}
        previous = value["reservations"].get(reservation_id)
        if previous is not None:
            if any(previous[k] != v for k, v in request.items()):
                raise BudgetError("reservation identity reused with different resources")
            return previous
        policy = value["policy"]
        remaining = min((datetime.fromisoformat(value["deadline"]) - datetime.now(UTC)).total_seconds(),
                        value["monotonic_deadline"] - time.monotonic())
        if walltime_seconds > remaining:
            raise BudgetError("allocation walltime exceeds remaining attempt time")
        active = [r for r in value["reservations"].values() if not r["terminal"]]
        for resource, maximum in (("gpus", "max_gpus"), ("cpus", "max_cpus"), ("memory_gb", "max_memory_gb")):
            if sum(r[resource] for r in active) + request[resource] > policy[maximum]:
                raise BudgetError(f"concurrent {resource} ceiling exceeded")
        for resource, limit in (("gpus", "gpu_seconds"), ("cpus", "cpu_seconds")):
            total = sum(r["charged"][limit] if r["terminal"] else r[resource] * r["walltime_seconds"]
                        for r in value["reservations"].values())
            if policy[limit] is not None and total + request[resource] * walltime_seconds > policy[limit]:
                raise BudgetError(f"cumulative {limit} budget exhausted")
        result = dict(request, id=reservation_id, scheduler_id=None, state="RESERVED", terminal=False, charged=None)
        value["reservations"][reservation_id] = result
        atomic_json(self.path, value)
        return result

    def reserve(self, reservation_id, *, gpus, cpus, walltime_seconds, **kwargs):
        with self._locked() as value:
            return self._reserve(value, reservation_id, gpus=gpus, cpus=cpus,
                                 walltime_seconds=walltime_seconds, **kwargs)

    @contextlib.contextmanager
    def submission(self, reservation_id, **resources):
        """Persist before transport; uncertain exceptions NEVER release a reservation."""
        with self._locked() as value:
            if not value["scheduler_enabled"]:
                raise BudgetError("scheduler access is disabled for unisolated development attempts")
            entry = self._reserve(value, reservation_id, **resources)
            if entry["state"] != "RESERVED":
                raise BudgetError("submission intent has already been dispatched")
            entry["state"] = "UNKNOWN"
            atomic_json(self.path, value)
            yield entry
            atomic_json(self.path, value)

    def record_job(self, reservation_id, job_id, *, tag, state="SUBMITTED"):
        with self._locked() as value:
            entry = value["reservations"][reservation_id]
            if entry["tag"] != tag or entry["scheduler_id"] not in (None, str(job_id)):
                raise BudgetError("scheduler identity differs from durable reservation")
            entry.update(scheduler_id=str(job_id), state=state)
            atomic_json(self.path, value)

    def reconcile(self, reservation_id, *, gpus, cpus, elapsed_seconds,
                  terminal, scheduler_id=None, tag=None):
        """Only authoritative terminal allocation accounting releases unused time."""
        if not terminal:
            raise BudgetError("terminal scheduler accounting is required before refund")
        if any(not math.isfinite(x) or x < 0 for x in (gpus, cpus, elapsed_seconds)):
            raise BudgetError("invalid terminal allocation accounting")
        with self._locked() as value:
            entry = value["reservations"][reservation_id]
            if entry["tag"] != tag or entry["scheduler_id"] != scheduler_id:
                raise BudgetError("accounting identity does not match reservation")
            charged = {"gpu_seconds": gpus * elapsed_seconds, "cpu_seconds": cpus * elapsed_seconds}
            if entry["terminal"] and entry["charged"] != charged:
                raise BudgetError("terminal accounting changed for settled reservation")
            entry.update(terminal=True, state="ACCOUNTED", charged=charged)
            policy = value["policy"]
            if gpus > entry["gpus"] or cpus > entry["cpus"] or elapsed_seconds > entry["walltime_seconds"]:
                value.update(closed=True, reason="allocation exceeded reservation")
            for limit in ("gpu_seconds", "cpu_seconds"):
                spent = sum(r["charged"][limit] for r in value["reservations"].values() if r["terminal"])
                if policy[limit] is not None and spent > policy[limit]:
                    value.update(closed=True, reason=f"{limit} exceeded")
            atomic_json(self.path, value)


def ledger_for_plan(plan):
    """Frozen identity is preferred; missing organizer policy is never optional."""
    benchmark = plan.get("benchmark") or {}
    directory = benchmark.get("attempt_dir") or plan.get("benchmark_control") or os.environ.get("BINDOCRACY_ATTEMPT_DIR")
    attempt_id = benchmark.get("attempt_id") or plan.get("attempt_id") or os.environ.get("BINDOCRACY_ATTEMPT_ID")
    if not directory:
        if attempt_id:
            raise BudgetError("benchmark attempt requires organizer budget ledger")
        return None
    ledger = BudgetLedger(directory)
    value = ledger.snapshot()
    if value["attempt_id"] != attempt_id:
        raise BudgetError("frozen plan and organizer attempt identity differ")
    record = json.loads((ledger.directory / "attempt.json").read_text())
    for field in ("task_digest", "environment_digest"):
        if plan.get(field) is not None and plan[field] != record[field]:
            raise BudgetError(f"frozen plan {field} differs from organizer record")
    return ledger
