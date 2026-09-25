"""Consumer contracts for discovery, error handling and exact scope previews."""

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from bindocracy.cli import app
from bindocracy.store import CampaignStore, create_database
from bindocracy.store.records import CollectedRun, DesignRecord, MetricRecord, RunRecord
from bindocracy.tools import load_configs


@pytest.fixture
def database(configs, tmp_path):
    record = load_configs(*configs).to_record()
    path = create_database(tmp_path / "protocol.duckdb")
    generator = RunRecord(
        run_id="gen",
        name="generation",
        tool="mosaic",
        kind="generate",
        model_config_id=record.model_config_id,
        status="succeeded",
        n_requested=3,
        n_produced=3,
    )
    designs = tuple(
        DesignRecord(
            design_id=identifier,
            run_id="gen",
            native_id=identifier,
            candidate_type="sequence",
            sequence=sequence,
        )
        for identifier, sequence in [("a", "ACDEFG"), ("b", "ACDEFG"), ("c", "GGGGGG")]
    )
    with CampaignStore(path) as store:
        store.ingest(CollectedRun(run=generator, designs=designs), configs=[record])
        for run_id in ("first", "second"):
            run = RunRecord(
                run_id=run_id,
                name="evaluation",
                tool="scorer",
                kind="evaluate",
                model_config_id=record.model_config_id,
                status="partial",
            )
            metrics = (
                MetricRecord(
                    metric_id=f"{run_id}-0",
                    run_id=run_id,
                    design_id="a",
                    name="scorer_iptm",
                    value=0.8,
                    direction="max",
                    replicate=0,
                ),
                MetricRecord(
                    metric_id=f"{run_id}-1",
                    run_id=run_id,
                    design_id="a",
                    name="scorer_iptm",
                    value=None,
                    status="failed",
                    direction="max",
                    replicate=1,
                ),
            )
            store.ingest(CollectedRun(run=run, metrics=metrics))
    return path


def invoke(*args):
    return CliRunner().invoke(app, ["--json", *map(str, args)])


def result_of(result):
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    envelope = json.loads(result.stdout)
    assert envelope["schema_version"] == 1 and envelope["ok"] is True
    return envelope["result"]


def test_discovery_schema_can_validate_and_reject_real_configuration(configs):
    names = result_of(invoke("tools", "list"))["tools"]
    assert "mosaic" in names
    schema = result_of(invoke("config", "schema", "--tool", "mosaic"))
    validator = Draft202012Validator(schema)
    authored = yaml.safe_load(configs[1].read_text())
    validator.validate(authored)
    assert list(validator.iter_errors({**authored, "misspelled_option": 1}))
    described = result_of(invoke("tools", "describe", "mosaic"))
    assert described["availability"] == "registered"
    assert described["launchability"] == "requires_preflight"


def test_parser_failure_is_json_on_stderr_not_a_success_document():
    result = invoke("config", "check", "--general", "missing.yaml")
    assert result.exit_code == 2
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "invalid_arguments"


def test_json_mode_never_succeeds_with_unstructured_help_output():
    result = invoke("campaign", "--help")
    assert result.exit_code == 2 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "invalid_arguments"


def test_validation_failure_has_field_locations_and_does_not_leak_json_mode(configs):
    general, model = configs
    raw = yaml.safe_load(model.read_text())
    raw["unrecognized_field"] = 1
    model.write_text(yaml.safe_dump(raw))
    result = invoke("config", "check", "--general", general, "--model", model, "--no-preflight")
    assert result.exit_code == 2 and result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "validation_error"
    assert any(entry["loc"] == ["unrecognized_field"] for entry in error["details"])
    human = CliRunner().invoke(app, ["version"])
    assert human.exit_code == 0 and not human.stdout.startswith("{")


def test_config_check_is_read_only_and_distinguishes_schema_from_preflight(configs, tmp_path):
    general, model = configs
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    checked = result_of(invoke("config", "check", "--general", general, "--model", model))
    assert checked["valid"] and checked["preflight"]
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    general_data = yaml.safe_load(general.read_text())
    Path(general_data["target"]["sequence_fasta"]).unlink()
    schema = result_of(
        invoke("config", "check", "--general", general, "--model", model, "--no-preflight")
    )
    assert schema["valid"] and not schema["preflight"]
    assert "model_config_id" not in schema
    failed = invoke("config", "check", "--general", general, "--model", model)
    assert failed.exit_code != 0 and json.loads(failed.stderr)["ok"] is False


def test_preview_matches_frozen_membership_with_limit_and_deduplication(database, tmp_path):
    before = database.read_bytes()
    preview = result_of(
        invoke("designset", "preview", database, "--limit", "2", "--distinct-sequences")
    )
    assert database.read_bytes() == before
    assert preview["n_designs"] == 1
    frozen = result_of(
        invoke(
            "designset",
            "build",
            database,
            "--out-dir",
            tmp_path / "sets",
            "--limit",
            "2",
            "--distinct-sequences",
        )
    )
    manifest = json.loads(Path(frozen["manifest"]).read_text())
    assert preview["n_designs"] == manifest["n_designs"]
    assert preview["query"] == manifest["query"]
    assert [entry["design_id"] for entry in manifest["entries"]] == ["a"]
    empty = result_of(invoke("designset", "preview", database, "--min-length", "100"))
    assert empty["n_designs"] == 0 and empty["by_tool"] == {}
    refused = invoke(
        "designset", "build", database, "--out-dir", tmp_path / "empty", "--min-length", "100"
    )
    assert refused.exit_code != 0 and not (tmp_path / "empty").exists()


def test_metric_discovery_preserves_run_and_replica_scope(database):
    all_metrics = result_of(invoke("filter", "metrics", database))["metrics"]
    assert {row["run_id"] for row in all_metrics} == {"first", "second"}
    scoped = result_of(invoke("filter", "metrics", database, "--run", "first"))["metrics"]
    assert len(scoped) == 1
    assert scoped[0]["replicates"] == [0, 1]
    assert scoped[0]["n_rows"] == 2 and scoped[0]["n_valid"] == 1
    ambiguous = invoke("filter", "metrics", database, "--run", "evaluation")
    assert ambiguous.exit_code == 2
    assert json.loads(ambiguous.stderr)["error"]["code"] == "selection_error"
