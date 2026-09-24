"""Frozen plans constrain the actual plugin workflow, not an alternate scheduler."""

import json
import subprocess
from pathlib import Path

import duckdb
import pytest
import yaml
from typer.testing import CliRunner

from bindocracy.campaign.plan import PlanError, build_plan, load_plan
from bindocracy.campaign.plan_cli import app
from bindocracy.runs.manifest import RunManifest


def _request(tmp_path, configs, *, controller_gpus=0, max_total_gpus=2):
    general, model = configs
    index = tmp_path / "index.yaml"
    index.write_text(
        yaml.safe_dump(
            {
                "database": str(tmp_path / "campaign.duckdb"),
                "run_root": str(tmp_path / "runs"),
                "general_config": str(general),
                "runs": [{"name": "execution", "config": str(model)}],
            }
        )
    )
    site = tmp_path / "site.yaml"
    site.write_text(
        yaml.safe_dump(
            {
                "controller": {
                    "account": "test",
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
    return index, site, tmp_path / "plan"


def test_generation_slots_are_not_guaranteed_candidates(configs, tmp_path):
    request = _request(tmp_path, configs)
    plan = build_plan(*request)
    run = plan["runs"][0]
    assert run["tasks"] == 2
    assert run["n_designs"] is None
    assert run["n_predictions"] is None
    assert run["scope"]["generation_request_slots"] == 8
    assert run["scope"]["guaranteed_outputs"] is False
    assert not Path(plan["database"]).exists()
    assert build_plan(*request)["digest"] == plan["digest"]
    result = CliRunner().invoke(app, ["show", str(request[2] / "plan.json")])
    assert result.exit_code == 0, result.output
    assert plan["digest"] in result.output
    assert "8 generation request slots" in result.output


def test_evaluation_uses_exact_membership_not_ceiling_times_shards(chai1_config_files, tmp_path):
    general, model = chai1_config_files
    config = yaml.safe_load(model.read_text())
    config["sharding"]["jobs"] = 3
    model.write_text(yaml.safe_dump(config))
    plan = build_plan(*_request(tmp_path, (general, model), controller_gpus=1))
    run = plan["runs"][0]
    assert run["tasks"] == 3
    assert run["n_designs"] == 4  # ceiling(4 / 3) * 3 would falsely claim six
    assert run["n_predictions"] == 20
    assert run["scope"]["shard_candidates"] == [2, 2, 0]
    assert run["scope"]["by_generator"] == {"mosaic": 4}
    assert run["scope"]["by_source_run"] == {"fixture": 4}
    assert plan["budget"]["worker_gpu_budget"] == 1
    assert plan["site"]["controller"]["gpus"] == 1


def test_frozen_workflow_ignores_edited_authored_yaml_and_ingests(configs, tmp_path):
    request = _request(tmp_path, configs)
    plan = build_plan(*request)
    # The old workload must still execute, even when live YAML cannot load.
    configs[0].write_text("not: a valid general config\n")
    configs[1].write_text("tool: nonexistent\n")
    request[0].write_text("runs: []\n")
    assert load_plan(request[2] / "plan.json")["digest"] == plan["digest"]
    command = [
        plan["python"],
        "-m",
        "snakemake",
        "--snakefile",
        plan["snakefile"],
        "--configfile",
        plan["workflow_index"],
        "--executor",
        "local",
        "--cores",
        "2",
        "--resources",
        "db_writer=1",
        "gpu=2",
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with duckdb.connect(plan["database"], read_only=True) as connection:
        assert connection.execute(
            "SELECT status, n_requested, n_produced FROM runs"
        ).fetchone() == (
            "succeeded",
            8,
            8,
        )
    assert load_plan(request[2] / "plan.json")["digest"] == plan["digest"]
    statuses = sorted((Path(plan["run_root"]) / "execution" / "tasks").glob("*/status.json"))
    before = {path: path.stat().st_mtime_ns for path in statuses}
    resumed = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert {path: path.stat().st_mtime_ns for path in statuses} == before


@pytest.mark.parametrize(
    "changed", ["manifest", "archive", "input", "workflow", "snapshot", "container"]
)
def test_changed_execution_closure_refuses_loading(configs, tmp_path, changed):
    request = _request(tmp_path, configs)
    plan = build_plan(*request)
    manifest = RunManifest.read(plan["runs"][0]["manifest"])
    paths = {
        "manifest": Path(plan["runs"][0]["manifest"]),
        "archive": manifest.path(next(iter(manifest.provenance.values())).path),
        "input": Path(next(iter(manifest.inputs.values())).uri),
        "workflow": Path(plan["snakefile"]),
        "snapshot": Path(
            yaml.safe_load(Path(plan["workflow_index"]).read_text())["general_config"]
        ),
        "container": Path(manifest.container),
    }
    paths[changed].write_bytes(paths[changed].read_bytes() + b"\nchanged\n")
    with pytest.raises(PlanError, match="changed"):
        load_plan(request[2] / "plan.json")
    # Cancellation/history must remain possible when execution inputs go away.
    assert load_plan(request[2] / "plan.json", verify=False)["digest"] == plan["digest"]


def test_plan_budget_tampering_is_not_a_new_approval(configs, tmp_path):
    request = _request(tmp_path, configs)
    plan = build_plan(*request)
    plan["site"]["max_total_gpus"] += 1
    path = request[2] / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(PlanError, match="digest mismatch"):
        load_plan(path, verify=False)


def test_controller_and_worker_must_fit_together(configs, tmp_path):
    request = _request(tmp_path, configs, controller_gpus=1, max_total_gpus=2)
    model = yaml.safe_load(configs[1].read_text())
    model["resources"]["gpus"] = 2
    configs[1].write_text(yaml.safe_dump(model))
    with pytest.raises(PlanError, match="worker plus controller"):
        build_plan(*request)
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "campaign.duckdb").exists()


def test_replanning_refuses_to_rebind_existing_general_configuration(configs, tmp_path):
    request = _request(tmp_path, configs)
    plan = build_plan(*request)
    manifest_path = Path(plan["runs"][0]["manifest"])
    before = manifest_path.read_bytes()
    general = yaml.safe_load(configs[0].read_text())
    general["cluster"]["account"] = "different-account"
    configs[0].write_text(yaml.safe_dump(general))
    with pytest.raises(PlanError, match="authored configuration changed"):
        build_plan(*request)
    with pytest.raises(PlanError, match="different execution"):
        build_plan(request[0], request[1], tmp_path / "another-plan")
    assert manifest_path.read_bytes() == before


def test_two_plans_cannot_schedule_the_same_run_directory(configs, tmp_path):
    request = _request(tmp_path, configs)
    first = build_plan(*request)
    with pytest.raises(PlanError, match="different campaign plan"):
        build_plan(request[0], request[1], tmp_path / "second-plan")
    assert load_plan(request[2] / "plan.json")["digest"] == first["digest"]


def test_a_claimed_design_set_digest_does_not_override_membership(chai1_config_files, tmp_path):
    request = _request(tmp_path, chai1_config_files)
    model = yaml.safe_load(chai1_config_files[1].read_text())
    path = Path(model["design_set"])
    design_set = json.loads(path.read_text())
    design_set["entries"][0]["design_id"] = "substituted-candidate"
    path.write_text(json.dumps(design_set))
    with pytest.raises(PlanError, match="design-set membership"):
        build_plan(*request)
    assert not (tmp_path / "runs").exists()


def test_optimizer_metric_enums_freeze_as_consumable_configuration(optimize_config_files, tmp_path):
    from bindocracy.runs.index import load_workflow_index
    from bindocracy.tools import load_configs

    plan = build_plan(*_request(tmp_path, optimize_config_files))
    index = load_workflow_index(plan["workflow_index"])
    frozen = load_configs(index.general_config, index.runs[0].config)
    assert frozen.model.metrics["loss"].direction == "min"
    assert frozen.model.metrics["n_mutations"].direction == "none"
    assert plan["runs"][0]["scope"]["input_candidates"] == 3
    assert plan["runs"][0]["scope"]["shard_candidates"] == [3]
