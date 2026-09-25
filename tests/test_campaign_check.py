"""Readiness observes existing execution contracts without creating executions."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from bindocracy.campaign import check as readiness
from bindocracy.cli import app


def _request(tmp_path, configs, *, controller_gpus=0, max_total_gpus=2, extra_runs=()):
    general, model = configs
    index = tmp_path / "index.yaml"
    index.write_text(
        yaml.safe_dump(
            {
                "database": str(tmp_path / "campaign.duckdb"),
                "run_root": str(tmp_path / "runs"),
                "general_config": str(general),
                "runs": [{"name": "valid", "config": str(model)}, *extra_runs],
            }
        )
    )
    site = tmp_path / "site.yaml"
    site.write_text(
        yaml.safe_dump(
            {
                "controller": {
                    "account": "test-account",
                    "partition": "test-controller",
                    "cpus": 2,
                    "memory_gb": 4,
                    "walltime": "01:00:00",
                    "gpus": controller_gpus,
                },
                "max_workers": 8,
                "max_total_gpus": max_total_gpus,
            }
        )
    )
    return index, site


def _snapshot(root):
    return {
        str(path.relative_to(root)): (
            path.stat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for path in root.rglob("*")
    }


def _scheduler(monkeypatch, *, state="up", association="test-account|\n"):
    monkeypatch.setattr(readiness.shutil, "which", lambda name: f"/slurm/{name}")

    def query(command, **kwargs):
        name = Path(command[0]).name
        output = {
            "sinfo": f"test-controller|up\ntest-gpu*|{state}\n",
            "scontrol": "ClusterName = current-cluster\n",
            # A foreign cluster's association must not establish access here.
            "sacctmgr": association if "cluster=current-cluster" in command else "test-account|\n",
        }[name]
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(readiness.subprocess, "run", query)


def test_valid_static_check_does_not_write_files_or_open_database(configs, tmp_path, monkeypatch):
    index, site = _request(tmp_path, configs)
    database = tmp_path / "campaign.duckdb"
    database.write_bytes(b"an existing database must not be opened by generation preflight")
    before = _snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("static readiness must not invoke a subprocess or open the campaign DB")

    import duckdb

    monkeypatch.setattr(duckdb, "connect", forbidden)
    monkeypatch.setattr(readiness.subprocess, "run", forbidden)
    report = readiness.check_campaign(index, site)
    assert report["static_status"] == "ready"
    assert report["status"] == report["scheduler_status"] == "unknown"
    assert report["runs"][0]["tasks"] == 2
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / "runs").exists()


def test_missing_database_is_not_created(configs, tmp_path):
    index, site = _request(tmp_path, configs)
    report = readiness.check_campaign(index, site)
    assert report["static_status"] == "ready"
    assert not (tmp_path / "campaign.duckdb").exists()
    assert not (tmp_path / "runs").exists()


def test_frozen_index_cannot_claim_readiness_after_database_change(configs, tmp_path):
    from bindocracy.campaign.plan import build_plan

    index, site = _request(tmp_path, configs)
    plan = build_plan(index, site, tmp_path / "plan")
    frozen = Path(plan["workflow_index"])
    assert readiness.check_campaign(frozen, site)["static_status"] == "ready"
    changed = yaml.safe_load(frozen.read_text())
    changed["database"] = str(tmp_path / "other.duckdb")
    copied_index = tmp_path / "modified-index.yaml"
    copied_index.write_text(yaml.safe_dump(changed))
    before = _snapshot(tmp_path)
    report = readiness.check_campaign(copied_index, site)
    assert report["static_status"] == "blocked"
    assert any(
        check["code"] == "frozen_plan" and check["status"] == "blocked"
        for check in report["checks"]
    )
    assert _snapshot(tmp_path) == before


def test_all_independent_run_errors_are_collected(configs, tmp_path):
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("tool: not-registered\n")
    missing = tmp_path / "missing.yaml"
    index, site = _request(
        tmp_path,
        configs,
        extra_runs=(
            {"name": "unknown-tool", "config": str(unknown)},
            {"name": "missing-config", "config": str(missing)},
        ),
    )
    report = readiness.check_campaign(index, site)
    assert report["status"] == "blocked"
    assert [run["status"] for run in report["runs"]] == ["ready", "blocked", "blocked"]
    assert "not-registered" in report["runs"][1]["checks"][0]["message"]
    assert str(missing) in report["runs"][2]["checks"][0]["message"]


@pytest.mark.parametrize("controller_gpus,worker_gpus", [(2, 1), (1, 2)])
def test_controller_and_worker_budget_violations(configs, tmp_path, controller_gpus, worker_gpus):
    model = yaml.safe_load(configs[1].read_text())
    model["resources"]["gpus"] = worker_gpus
    configs[1].write_text(yaml.safe_dump(model))
    index, site = _request(tmp_path, configs, controller_gpus=controller_gpus)
    report = readiness.check_campaign(index, site)
    assert report["static_status"] == report["status"] == "blocked"
    findings = report["checks"] + report["runs"][0]["checks"]
    assert any(
        item["status"] == "blocked" and item["code"] in {"site", "resources"} for item in findings
    )
    assert not (tmp_path / "runs").exists()


def test_missing_scheduler_is_unknown_not_ready(configs, tmp_path, monkeypatch):
    index, site = _request(tmp_path, configs)
    monkeypatch.setattr(readiness.shutil, "which", lambda name: None)
    report = readiness.check_campaign(index, site, probe_site=True)
    assert report["static_status"] == "ready"
    assert report["status"] == report["scheduler_status"] == "unknown"
    assert any("not on PATH" in check["message"] for check in report["scheduler_checks"])


@pytest.mark.parametrize("failure", ["permissions", "timeout"])
def test_unavailable_scheduler_queries_are_unknown(configs, tmp_path, monkeypatch, failure):
    index, site = _request(tmp_path, configs)
    monkeypatch.setattr(readiness.shutil, "which", lambda name: f"/slurm/{name}")

    def unavailable(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=1, stdout="", stderr="Permission denied")

    monkeypatch.setattr(readiness.subprocess, "run", unavailable)
    report = readiness.check_campaign(index, site, probe_site=True)
    assert report["status"] == report["scheduler_status"] == "unknown"


def test_visible_site_is_ready_but_partition_down_is_blocked(configs, tmp_path, monkeypatch):
    index, site = _request(tmp_path, configs)
    _scheduler(monkeypatch)
    report = readiness.check_campaign(index, site, probe_site=True)
    assert report["status"] == "ready"
    _scheduler(monkeypatch, state="down")
    report = readiness.check_campaign(index, site, probe_site=True)
    assert report["static_status"] == "ready"
    assert report["status"] == report["scheduler_status"] == "blocked"


def test_foreign_cluster_association_does_not_establish_readiness(configs, tmp_path, monkeypatch):
    index, site = _request(tmp_path, configs)
    _scheduler(monkeypatch, association="")
    report = readiness.check_campaign(index, site, probe_site=True)
    assert report["status"] == report["scheduler_status"] == "unknown"


def test_optimizer_dependencies_remain_unknown_without_declared_bindings(
    optimize_config_files, tmp_path
):
    index, site = _request(tmp_path, optimize_config_files)
    report = readiness.check_campaign(index, site)
    assert report["static_status"] == "unknown"
    assert any(
        check["code"] == "kit_dependencies" and check["status"] == "unknown"
        for check in report["runs"][0]["checks"]
    )


@pytest.mark.parametrize("blocked", [False, True])
def test_cli_retains_report_on_nonzero_readiness(configs, tmp_path, blocked):
    index, site = _request(tmp_path, configs)
    if blocked:
        configs[1].unlink()
    result = CliRunner().invoke(
        app, ["--json", "campaign", "check", str(index), "--site", str(site)]
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["result"]["status"] == ("blocked" if blocked else "unknown")
    assert payload["result"]["runs"][0]["name"] == "valid"
