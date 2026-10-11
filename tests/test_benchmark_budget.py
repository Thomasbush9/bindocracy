"""Budget admission contracts; no test contacts a scheduler."""
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from bindocracy.benchmark.budget import BudgetError, BudgetLedger, ledger_for_plan
from bindocracy.benchmark.models import BudgetPolicy
from bindocracy.campaign import control


def _reserve_one(arguments):
    directory, identity = arguments
    try:
        BudgetLedger(directory).reserve(identity, gpus=1, cpus=1, walltime_seconds=40)
        return True
    except BudgetError:
        return False


@pytest.fixture
def ledger(tmp_path):
    ledger = BudgetLedger(tmp_path)
    ledger.initialize(BudgetPolicy(mode="compute", walltime_seconds=300,
                                  max_gpus=2, max_cpus=4, max_memory_gb=16,
                                  gpu_seconds=80, cpu_seconds=160),
                      attempt_id="attempt-a", deadline=datetime.now(UTC) + timedelta(seconds=300),
                      scheduler_enabled=True)
    return ledger


def test_process_reservation_race_never_overspends(ledger):
    with ProcessPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(_reserve_one, [(ledger.directory, str(i)) for i in range(8)]))
    assert sum(accepted) == 2
    assert len(ledger.snapshot()["reservations"]) == 2


def test_reservation_retry_is_idempotent_and_conflict_fails(ledger):
    first = ledger.reserve("one", gpus=1, cpus=1, walltime_seconds=40)
    assert ledger.reserve("one", gpus=1, cpus=1, walltime_seconds=40) == first
    with pytest.raises(BudgetError, match="different resources"):
        ledger.reserve("one", gpus=1, cpus=1, walltime_seconds=30)
    assert len(ledger.snapshot()["reservations"]) == 1


def test_lost_submission_reply_remains_reserved_until_terminal_accounting(ledger):
    with (
        pytest.raises(TimeoutError),
        ledger.submission("one", gpus=2, cpus=2, walltime_seconds=40, tag="exact"),
    ):
        raise TimeoutError("accepted but reply was lost")
    reservation = ledger.snapshot()["reservations"]["one"]
    assert reservation["state"] == "UNKNOWN"
    with pytest.raises(BudgetError):
        ledger.reserve("two", gpus=1, cpus=1, walltime_seconds=1)
    with (
        pytest.raises(BudgetError, match="already been dispatched"),
        ledger.submission("one", gpus=2, cpus=2, walltime_seconds=40, tag="exact"),
    ):
        pytest.fail("must not retry uncertain transport")
    ledger.record_job("one", "123", tag="exact")
    with pytest.raises(BudgetError, match="terminal"):
        ledger.reconcile("one", gpus=2, cpus=2, elapsed_seconds=10,
                         terminal=False, scheduler_id="123", tag="exact")
    ledger.reconcile("one", gpus=2, cpus=2, elapsed_seconds=10,
                     terminal=True, scheduler_id="123", tag="exact")
    ledger.reconcile("one", gpus=2, cpus=2, elapsed_seconds=10,
                     terminal=True, scheduler_id="123", tag="exact")
    ledger.reserve("two", gpus=1, cpus=1, walltime_seconds=40)
    assert ledger.snapshot()["reservations"]["one"]["charged"]["gpu_seconds"] == 20


def test_closed_and_expired_attempts_fail_closed(ledger, tmp_path):
    ledger.close("cancelled")
    with pytest.raises(BudgetError, match="closed or expired"):
        ledger.reserve("one", gpus=0, cpus=1, walltime_seconds=1)
    other = tmp_path / "expired"
    other.mkdir()
    expired = BudgetLedger(other)
    expired.initialize(BudgetPolicy(walltime_seconds=1, max_gpus=0, max_cpus=1, max_memory_gb=1),
                       attempt_id="expired", deadline=datetime.now(UTC) - timedelta(seconds=1))
    with pytest.raises(BudgetError, match="closed or expired"):
        expired.check()


def test_development_cannot_submit_scheduler_jobs(tmp_path):
    ledger = BudgetLedger(tmp_path)
    ledger.initialize(BudgetPolicy(walltime_seconds=30, max_gpus=0, max_cpus=1, max_memory_gb=1),
                      attempt_id="dev", deadline=datetime.now(UTC) + timedelta(seconds=30))
    with (
        pytest.raises(BudgetError, match="disabled"),
        ledger.submission("one", gpus=0, cpus=1, walltime_seconds=1),
    ):
        pytest.fail("scheduler must never be invoked")


def test_controller_resources_and_actual_job_identity_are_reserved(ledger):
    (ledger.directory / "attempt.json").write_text(json.dumps({
        "attempt_id": "attempt-a", "task_digest": "task", "environment_digest": "env",
    }))
    plan = {"benchmark": {"attempt_id": "attempt-a", "attempt_dir": str(ledger.directory)},
            "plan_dir": str(ledger.directory)}
    entry = {"tag": "exact-controller-tag", "role": "controller", "id": None, "state": "UNKNOWN"}
    args = ["--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=2", "--mem=2G", "--time=00:00:20", "controller.sh"]
    with control._benchmark_submission(plan, entry, args):
        # This is the injected scheduler transport side effect.
        entry.update(id="900", state="SUBMITTED")
    record = ledger.snapshot()["reservations"][entry["tag"]]
    assert record["cpus"] == 2
    assert record["gpus"] == 0
    assert record["scheduler_id"] == "900"
    assert "--no-requeue" in args
    assert ledger_for_plan(plan).directory == ledger.directory


def test_cumulative_compute_limit_remains_after_concurrent_refund(ledger):
    ledger.reserve("one", gpus=2, cpus=2, walltime_seconds=40)
    ledger.reconcile("one", gpus=2, cpus=2, elapsed_seconds=40, terminal=True)
    with pytest.raises(BudgetError, match="cumulative gpu_seconds"):
        ledger.reserve("two", gpus=1, cpus=1, walltime_seconds=1)
