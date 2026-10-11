"""Runnable authored kits use the same frozen bundles and contracts as production."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from bindocracy.cli import app
from bindocracy.functions.contract import FunctionInput
from bindocracy.functions.runner import run_custom
from bindocracy.kits.authoring import initialize_kit
from bindocracy.kits.bundle import cache_kit, inspect_kit
from bindocracy.kits.models import KitError
from bindocracy.kits.optimizer import optimizer_config
from bindocracy.kits.resolve import load_lock, resolve_kit, verify_lock, write_lock
from bindocracy.kits.scoring import function_config, validate_function_lock
from bindocracy.kits.smoke import smoke_kit


def _resolved(tmp_path, kind, request, script=None):
    source = tmp_path / "source"
    release = initialize_kit(source, kind=kind)
    if script is not None:
        (source / release.manifest.entrypoint).write_text(script)
    release = cache_kit(source, tmp_path / "cache")
    lock = resolve_kit(release.root, approved_digest=release.digest)
    lock_path = write_lock(lock, tmp_path / "kit-lock.json")
    adapter = optimizer_config if kind == "optimizer" else function_config
    model = adapter(lock, request, lock_path)
    config = tmp_path / "resolved.yaml"
    config.write_text(yaml.safe_dump(model.model_dump(mode="json")))
    return source, lock, model, config


def _request(design_set, *, optimizer=False):
    return {
        "schema_version": 1, "name": "example", "design_set": str(design_set),
        "resources": {"gpus": 0, "cpus": 1, "memory_gb": 1, "walltime": "00:01:00"},
        **({"sharding": {"jobs": 1}} if optimizer else {}),
    }


@pytest.fixture
def general(tmp_path):
    target = tmp_path / "target.fasta"
    target.write_text(">real-test-target\nACDEFGHIK\n")
    path = tmp_path / "general.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": 1, "campaign": {"name": "kit-tests"},
        "target": {"name": "test-target", "sequence_fasta": str(target), "chain_id": "A"},
        "cluster": {"executor": "slurm", "account": "test", "default_partition": "test"},
    }))
    return path


@pytest.mark.parametrize("kind", ["optimizer", "scoring_function"])
def test_init_inspect_check_resolve_both_kinds(tmp_path, kind):
    cli = CliRunner()
    source = tmp_path / "authored"
    initialized = cli.invoke(app, ["--json", "kit", "init", str(source), "--kind", kind])
    assert initialized.exit_code == 0, initialized.output
    first = inspect_kit(source)
    repeated = cli.invoke(app, ["--json", "kit", "init", str(source), "--kind", kind])
    assert repeated.exit_code != 0
    assert inspect_kit(source).digest == first.digest
    inspected = cli.invoke(app, ["--json", "kit", "inspect", str(source)])
    assert inspected.exit_code == 0, inspected.output
    checked = cli.invoke(app, ["--json", "kit", "check", str(source)])
    assert checked.exit_code == 1  # Explicitly unapproved, not a runtime test.
    assert json.loads(checked.stdout)["result"]["runtime_qualified"] is False
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump(_request(tmp_path / "set.json", optimizer=kind == "optimizer")))
    resolved = cli.invoke(app, [
        "--json", "kit", "resolve", str(source), "--cache", str(tmp_path / "cache"),
        "--config", str(request), "--output-dir", str(tmp_path / "resolved"),
        "--approve", first.digest,
    ])
    assert resolved.exit_code == 0, resolved.output
    result = json.loads(resolved.stdout)["result"]
    assert Path(result["config"]).name == ("optimize.yaml" if kind == "optimizer" else "function.yaml")
    assert load_lock(Path(result["lock"])).manifest.kind == kind


def test_multifile_scorer_runs_complete_immutable_revision(tmp_path):
    source, lock, model, _ = _resolved(tmp_path, "scoring_function", _request(tmp_path / "set.json"))
    (source / "weights.json").write_text('{"K": 9.0}\n')
    newer = cache_kit(source, tmp_path / "cache")
    assert newer.digest != lock.bundle_digest
    shutil.rmtree(source)
    result = run_custom(
        model.function,
        [FunctionInput(index=0, design_id="a", sequence="KRAA"),
         FunctionInput(index=1, design_id="b", sequence="AAAA")],
        run_id="scorer-closure", work_dir=tmp_path / "work", measured_at=datetime.now(UTC),
    )
    assert result.n_scored == 2
    assert sorted(record.value for record in result.records) == [0.0, 0.5]
    assert (tmp_path / "work" / "kit" / "weights.json").is_file()
    assert not tuple(lock.kit_root.rglob("__pycache__"))
    verify_lock(lock)


@pytest.mark.parametrize("mutation", ["add", "remove", "change"])
def test_scoring_inventory_drift_is_refused(tmp_path, mutation):
    _, lock, model, _ = _resolved(tmp_path, "scoring_function", _request(tmp_path / "set.json"))
    if mutation == "add":
        (lock.kit_root / "unlisted.py").write_text("value = 1\n")
    elif mutation == "remove":
        (lock.kit_root / "residues.py").unlink()
    else:
        (lock.kit_root / "weights.json").write_text('{"K": 2.0}\n')
    with pytest.raises(KitError, match="inventory|digest"):
        validate_function_lock(model.function)


@pytest.mark.parametrize("field", ["script", "args", "metrics", "inputs", "kit_lock", "container", "container_python", "dev_source"])
def test_scoring_kit_owned_overrides_refused(tmp_path, field):
    request = _request(tmp_path / "set.json")
    _, lock, model, _ = _resolved(tmp_path, "scoring_function", request)
    value = model.function.model_dump(mode="json")[field]
    with pytest.raises(KitError, match="kit-owned"):
        function_config(lock, {**request, "function": {field: value}}, model.function.kit_lock)


@pytest.mark.parametrize("kind", ["optimizer", "scoring_function"])
def test_wrong_kind_adapter_is_refused(tmp_path, kind):
    request = _request(tmp_path / "set.json", optimizer=kind == "optimizer")
    _, lock, _, _ = _resolved(tmp_path, kind, request)
    wrong = function_config if kind == "optimizer" else optimizer_config
    with pytest.raises(KitError, match="requires an? (optimizer|scoring_function) kit"):
        wrong(lock, {}, tmp_path / "other-lock.json")


def test_scoring_manifest_rejects_optimizer_only_fields(tmp_path):
    release = initialize_kit(tmp_path / "kit", kind="scoring_function")
    path = release.root / "kit.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["capabilities"] = {"changes_length": False, "max_children": 1}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(KitError, match="capabilities"):
        inspect_kit(release.root)


@pytest.mark.parametrize("sequences,produced,refused", [(["KRAA", "AAAA"], 1, 1), (["AAAA"], 0, 1)])
def test_optimizer_smoke_runs_template_and_accepts_all_refusals(
    tmp_path, write_design_set, general, sequences, produced, refused,
):
    _, _, _, config = _resolved(tmp_path, "optimizer", _request(write_design_set(sequences), optimizer=True))
    result = smoke_kit(config, general, tmp_path / "smoke", local_cpu=True)
    assert result["status"] == "passed" and result["contract_valid"] is True
    assert result["n_produced"] == produced and result["n_refused"] == refused
    assert Path(result["log"]).is_file()


@pytest.mark.parametrize("output,valid", [("", True), ('{"parent_index":0,"sequence":"X"}\n', False)])
def test_empty_optimizer_output_is_valid_but_malformed_contract_is_not(
    tmp_path, write_design_set, general, output, valid,
):
    script = (
        "import argparse\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "for flag in ('--inputs','--outputs','--context'): p.add_argument(flag)\n"
        "args=p.parse_args()\n"
        f"Path(args.outputs).write_text({output!r})\n"
    )
    _, _, _, config = _resolved(tmp_path, "optimizer", _request(write_design_set(["AAAA"]), optimizer=True), script)
    if valid:
        result = smoke_kit(config, general, tmp_path / "smoke", local_cpu=True)
        assert result["contract_valid"] and result["scientific_output"] == "zero_output"
    else:
        with pytest.raises(KitError, match="production contract"):
            smoke_kit(config, general, tmp_path / "smoke", local_cpu=True)
        report = json.loads((tmp_path / "smoke" / "smoke-report.json").read_text())
        assert report["status"] == "failed"


def test_smoke_requires_explicit_execution_and_refuses_gpu_local(tmp_path, write_design_set, general):
    request = _request(write_design_set(["KRAA"]), optimizer=True)
    request["resources"]["gpus"] = 1
    _, _, _, config = _resolved(tmp_path, "optimizer", request)
    with pytest.raises(KitError, match="choose exactly one"):
        smoke_kit(config, general, tmp_path / "implicit")
    with pytest.raises(KitError, match="refuses GPU"):
        smoke_kit(config, general, tmp_path / "gpu", local_cpu=True)
    assert not (tmp_path / "gpu").exists()


def test_smoke_enforces_input_bound_before_execution(tmp_path, write_design_set, general):
    _, _, _, config = _resolved(tmp_path, "optimizer", _request(write_design_set(["KAAA", "RAAA", "AAAA"]), optimizer=True))
    with pytest.raises(KitError, match="real inputs"):
        smoke_kit(config, general, tmp_path / "smoke", local_cpu=True, max_inputs=2)
    assert not (tmp_path / "smoke").exists()


def test_smoke_timeout_preserves_logs_and_failure_report(tmp_path, write_design_set, general):
    _, _, _, config = _resolved(
        tmp_path, "optimizer", _request(write_design_set(["KAAA"]), optimizer=True),
        "import time\nprint('entered smoke', flush=True)\ntime.sleep(30)\n",
    )
    with pytest.raises(KitError, match="exceeded|runtime exited"):
        smoke_kit(config, general, tmp_path / "smoke", local_cpu=True, timeout_seconds=1)
    assert (tmp_path / "smoke" / "smoke-report.json").is_file()
    logs = list((tmp_path / "smoke" / "run" / "logs").glob("*.log"))
    assert logs and "entered smoke" in logs[0].read_text()


@pytest.mark.parametrize("mode", ["template", "refusal", "empty", "malformed", "crash"])
def test_scoring_smoke_uses_production_database_and_bundle(tmp_path, general, mode):
    from bindocracy.runs.selection import build
    from bindocracy.store import CampaignStore, create_database
    from bindocracy.store.query import DesignQuery
    from bindocracy.store.records import (
        CandidateType,
        CollectedRun,
        ConfigRecord,
        DesignRecord,
        RunKind,
        RunRecord,
        RunStatus,
    )

    database = create_database(tmp_path / "campaign.duckdb")
    config_record = ConfigRecord(
        general_name="kit-tests", general_schema_version=1,
        general_config_json=yaml.safe_load(general.read_text()),
        model_name="source", tool="fixture", model_schema_version=1,
        model_config_json={"name": "source"},
    )
    run = RunRecord(
        run_id="source", name="source", tool="fixture", kind=RunKind.GENERATE,
        model_config_id=config_record.model_config_id, status=RunStatus.SUCCEEDED,
        n_requested=1, n_attempted=1, n_produced=1,
    )
    with CampaignStore(database) as store:
        store.ingest(CollectedRun(run=run, designs=(
            DesignRecord(design_id="binder", run_id="source", native_id="0",
                         candidate_type=CandidateType.SEQUENCE, sequence="KRAA"),
        )), configs=[config_record])
    _, _, design_set = build(database, DesignQuery(), tmp_path / "sets")
    request = {**_request(design_set), "database": str(database)}
    script = None
    if mode != "template":
        output = {
            "refusal": '{"index":0,"failed":"timeout"}\n',
            "empty": "",
            "malformed": '{"index":0,"metrics":{"undeclared":1}}\n',
            "crash": "",
        }[mode]
        script = (
            "import argparse\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser()\n"
            "p.add_argument('--inputs'); p.add_argument('--outputs')\n"
            "args=p.parse_args()\n"
            f"Path(args.outputs).write_text({output!r})\n"
        )
        if mode == "crash":
            script += "raise SystemExit(3)\n"
    _, _, _, config = _resolved(tmp_path, "scoring_function", request, script)
    if mode in {"malformed", "crash"}:
        with pytest.raises(KitError, match="contract|runtime"):
            smoke_kit(config, general, tmp_path / "smoke", local_cpu=True)
    else:
        report = smoke_kit(config, general, tmp_path / "smoke", local_cpu=True)
        assert report["contract_valid"] is True
        assert report["n_produced"] == (1 if mode == "template" else 0)
        assert report["scientific_output"] == ("produced" if mode == "template" else "zero_output")


def test_plan_only_prepares_approved_campaign_route_without_execution(
    tmp_path, write_design_set, general,
):
    request = _request(write_design_set(["KRAA"]), optimizer=True)
    request["resources"]["gpus"] = 1
    _, _, _, config = _resolved(tmp_path, "optimizer", request)
    site = tmp_path / "site.yaml"
    site.write_text(yaml.safe_dump({
        "controller": {"account": "test", "partition": "test", "cpus": 1,
                       "memory_gb": 1, "walltime": "00:02:00", "gpus": 0},
        "max_workers": 1, "max_total_gpus": 1,
    }))
    report = smoke_kit(
        config, general, tmp_path / "smoke", plan_only=True, site=site,
        database=tmp_path / "campaign.duckdb",
    )
    assert report["status"] == "planned"
    assert report["executed"] is False and report["runtime_qualified"] is False
    assert Path(report["plan"]).is_file()
    assert not list((tmp_path / "smoke" / "runs").rglob("optimizer_output.jsonl"))


def test_archived_bundle_dependency_check_does_not_require_source_bundle(tmp_path):
    from bindocracy.kits.resolve import verify_dependencies

    source = initialize_kit(tmp_path / "kit", kind="scoring_function").root
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "values.txt").write_text("original\n")
    manifest = yaml.safe_load((source / "kit.yaml").read_text())
    manifest["requires"] = [{"name": "external", "kind": "source_overlay"}]
    (source / "kit.yaml").write_text(yaml.safe_dump(manifest))
    bindings = tmp_path / "bindings.yaml"
    bindings.write_text(yaml.safe_dump({
        "bindings_schema": 1, "external": {"kind": "source_overlay", "path": str(asset)},
    }))
    lock = resolve_kit(source, bindings=bindings, approved_digest=inspect_kit(source).digest)
    shutil.rmtree(source)
    verify_dependencies(lock)
    (asset / "values.txt").write_text("changed\n")
    with pytest.raises(KitError, match="dependency identity"):
        verify_dependencies(lock)


def test_gpu_required_optimizer_refuses_cpu_request(tmp_path):
    source = tmp_path / "requires-gpu"
    initialize_kit(source, kind="optimizer")
    path = source / "kit.yaml"
    declaration = yaml.safe_load(path.read_text())
    declaration["cost"]["gpu"] = "required"
    path.write_text(yaml.safe_dump(declaration))
    release = inspect_kit(source)
    lock = resolve_kit(source, approved_digest=release.digest)
    with pytest.raises(KitError, match="zero-GPU"):
        optimizer_config(lock, _request(tmp_path / "set.json", optimizer=True), tmp_path / "lock.json")


def test_unlocked_scoring_source_overlay_is_refused(tmp_path):
    from bindocracy.functions.models import CustomFunction

    with pytest.raises(ValueError, match="verified scoring kit lock"):
        CustomFunction(name="custom", script=tmp_path / "score.py", dev_source=tmp_path,
                       metrics={"custom_metric": {"direction": "max"}})
