"""Bounded Slurm accounting parsing; no scheduler calls or scientific jobs."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "campaign_watch_gpu_collector",
    Path(__file__).resolve().parents[1] / "scripts/campaign_watch/collector.py",
)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def observe(monkeypatch, output, *, keys=None, query_error=None):
    keys = keys if keys is not None else {("123", -1): "123"}
    records = {key: {"state": "RUNNING", "current": True, "raw_job_id": allocation}
               for key, allocation in keys.items()}
    campaign = SimpleNamespace(root=Path("/campaign"), gpu_keys=lambda: set(keys))
    calls = []

    def command(args, *, allow_partial=False):
        calls.append((args, allow_partial))
        if isinstance(output, Exception):
            raise output
        return output, query_error

    monkeypatch.setattr(collector, "bounded_command", command)
    result = collector.gpu_snapshot([campaign], records, None)
    return result[campaign.root], calls


def test_native_gpu_step_supersedes_zero_batch_and_ignores_extern(monkeypatch):
    result, calls = observe(monkeypatch,
        "123.batch|gres/gpuutil=0,gres/gpumem=0\n"
        "123.extern|energy=90\n"
        "123.0|gres/gpuutil=97,gres/gpumem=61730M\n")
    assert calls == [(["sstat", "--allsteps", "-j", "123", "--noheader", "--parsable2",
                      "--format=JobID%64,TRESUsageInAve%1024"], True)]
    assert result["sampled_jobs"] == result["memory_sampled_jobs"] == 1
    assert result["mean_percent"] == 97
    assert result["mean_memory_mib"] == 61730
    assert result["low_util_jobs"] == 0
    assert result["error"] is None


@pytest.mark.parametrize("native", ["", "energy=90", "gres/gpuutil=N/A,gres/gpumem=N/A"])
def test_missing_native_evidence_never_falls_back_to_wrapper_zero(monkeypatch, native):
    result, _ = observe(monkeypatch,
        f"123.batch|gres/gpuutil=0,gres/gpumem=0\n123.0|{native}\n")
    assert result["sampled_jobs"] == result["memory_sampled_jobs"] == 0
    assert result["mean_percent"] is result["mean_memory_mib"] is None
    assert result["observed_at"] is None
    assert "Missing/invalid readings" in result["error"]


def test_batch_only_legacy_workload_still_reports_accounting(monkeypatch):
    result, _ = observe(monkeypatch,
        "123.batch|gres/gpuutil=50,gres/gpumem=1024M\n123.extern|energy=90\n")
    assert result["mean_percent"] == 50
    assert result["mean_memory_mib"] == 1024
    assert result["error"] is None


def test_native_over_100_is_unknown_not_clipped_or_replaced_by_wrapper(monkeypatch):
    result, _ = observe(monkeypatch,
        "123.batch|gres/gpuutil=0,gres/gpumem=0\n"
        "123.0|gres/gpuutil=102,gres/gpumem=45470M\n")
    assert result["sampled_jobs"] == result["low_util_jobs"] == 0
    assert result["mean_percent"] is None
    assert result["memory_sampled_jobs"] == 1
    assert result["mean_memory_mib"] == 45470
    assert "utilization 1, memory 0" in result["error"]


def test_multiple_native_steps_have_one_explicit_allocation_mean(monkeypatch):
    result, _ = observe(monkeypatch,
        "123.batch|gres/gpuutil=0,gres/gpumem=0\n"
        "123.0|gres/gpuutil=20,gres/gpumem=1000M\n"
        "123.1|gres/gpuutil=80,gres/gpumem=3000M\n")
    assert result["eligible_jobs"] == result["queried_jobs"] == result["sampled_jobs"] == 1
    assert result["mean_percent"] == 50
    assert result["min_percent"] == result["max_percent"] == 50
    assert result["mean_memory_mib"] == 2000
    assert "unweighted step means" in result["error"]


def test_incomplete_multistep_metric_does_not_look_like_complete_coverage(monkeypatch):
    result, _ = observe(monkeypatch,
        "123.0|gres/gpuutil=20,gres/gpumem=1000M\n"
        "123.1|gres/gpuutil=102,gres/gpumem=3000M\n")
    assert result["sampled_jobs"] == 0
    assert result["mean_percent"] is None
    assert result["mean_memory_mib"] == 2000
    assert "Missing/invalid readings" in result["error"]


def test_duplicate_native_rows_are_unknown_without_wrapper_fallback(monkeypatch):
    result, _ = observe(monkeypatch,
        "123.batch|gres/gpuutil=0,gres/gpumem=0\n"
        "123.0|gres/gpuutil=20,gres/gpumem=1000M\n"
        "123.0|gres/gpuutil=80,gres/gpumem=3000M\n")
    assert result["sampled_jobs"] == result["memory_sampled_jobs"] == 0
    assert "duplicate sstat rows" in result["error"]


def test_allocation_aliases_are_queried_and_counted_once(monkeypatch):
    result, calls = observe(monkeypatch,
        "123.batch|gres/gpuutil=0,gres/gpumem=0\n"
        "123.0|gres/gpuutil=80,gres/gpumem=3000M\n",
        keys={("100", 2): "123", ("123", -1): "123"})
    assert calls[0][0][3] == "123"
    assert result["eligible_jobs"] == result["queried_jobs"] == result["sampled_jobs"] == 1
    assert result["mean_percent"] == 80


def test_partial_command_failure_keeps_native_evidence_with_diagnostic(monkeypatch):
    result, _ = observe(monkeypatch,
        "123.0|gres/gpuutil=43,gres/gpumem=3062M\n",
        query_error="sstat timed out; only complete rows retained")
    assert result["mean_percent"] == 43
    assert "timed out" in result["error"]


def test_command_failure_reports_unknown_not_zero(monkeypatch):
    result, _ = observe(monkeypatch, OSError("sstat unavailable"))
    assert result["sampled_jobs"] == result["memory_sampled_jobs"] == 0
    assert result["mean_percent"] is result["observed_at"] is None
    assert "sstat unavailable" in result["error"]


def test_allsteps_retains_eight_allocation_query_bound(monkeypatch):
    keys = {(str(index), -1): str(index) for index in range(100, 120)}
    result, calls = observe(monkeypatch, "", keys=keys)
    assert result["eligible_jobs"] == 20
    assert result["queried_jobs"] == 8
    assert len(calls) == 1
    assert len(calls[0][0][3].split(",")) == 8
