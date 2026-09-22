"""Public I/O seam: temporary database -> real script process -> database.

No models, containers, GPUs, or mocked subprocesses. The tiny callbacks only
compute a sequence property or append a residue to exercise transport.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from bindocracy.cli import app
from bindocracy.config.load import load_yaml
from bindocracy.config.models import GeneralConfig
from bindocracy.runs import ingest_bundle, write_collected
from bindocracy.runs.selection import read_only
from bindocracy.store import CampaignStore, create_database
from bindocracy.store.query import DesignQuery, select_designs, select_metrics
from bindocracy.store.records import (
    CollectedRun,
    ConfigRecord,
    DesignRecord,
    MetricRecord,
    RunRecord,
)
from bindocracy.tools import collect_run, launch_spec, load_configs, plan

CLI = CliRunner()
ROOT = Path(__file__).resolve().parents[1]
SCORE_SCRIPT = """from bindocracy_io import run_scoring

def score(candidate, args):
    return {"lysines": candidate["sequence"].count("K")}

run_scoring(score)
"""
OPTIMIZE_SCRIPT = """from bindocracy_io import run_optimization

def optimize(parent, context, args):
    yield {"sequence": parent["sequence"] + "A", "metrics": {"edit_cost": 1}}

run_optimization(optimize)
"""


@pytest.fixture
def io_campaign(tmp_path):
    target = tmp_path / "target.fasta"
    target.write_text(">target\nMKTAYIAK\n")
    general_path = tmp_path / "general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1, "campaign": {"name": "io-test"},
        "target": {"name": "target", "sequence_fasta": str(target), "chain_id": "A"},
        "cluster": {"executor": "slurm", "account": "unused", "default_partition": "unused"},
    }))
    general = load_yaml(general_path, GeneralConfig)
    config = ConfigRecord(
        general_name="io-test", general_schema_version=1,
        general_config_json=general.model_dump(mode="json"),
        model_name="parents", tool="fixture", model_schema_version=1,
        model_config_json={"name": "parents"},
    )
    database = create_database(tmp_path / "campaign.duckdb")
    parents = RunRecord(run_id="parents", name="parents", tool="fixture", kind="generate",
                        model_config_id=config.model_config_id, status="succeeded")
    sequences = ("KKAA", "KAAAA", "AAAAAA", "KKKAAAA")
    designs = tuple(DesignRecord(
        design_id=f"d{i}", run_id="parents", native_id=f"parent-{i}",
        candidate_type="sequence", sequence=sequence,
        created_at=datetime(2026, 9, 10 + i, tzinfo=UTC),
    ) for i, sequence in enumerate(sequences))
    evaluator = RunRecord(run_id="measured", name="measured-loss", tool="fixture", kind="evaluate",
                          model_config_id=config.model_config_id, status="succeeded")
    metrics = tuple(MetricRecord(run_id="measured", design_id=f"d{i}", name="trial_loss",
                                 value=value, direction="min")
                    for i, value in enumerate((0.2, 0.8, 0.4)))  # d3 is unmeasured.
    with CampaignStore(database) as store:
        store.ingest(CollectedRun(run=parents, designs=designs), configs=[config])
        store.ingest(CollectedRun(run=evaluator, metrics=metrics))
    return database, general_path


def freeze(database, directory, *flags):
    result = CLI.invoke(app, ["designset", "build", str(database), "--out-dir", str(directory), *flags])
    assert result.exit_code == 0, result.output
    return next(directory.glob("*.json"))


def score_config(tmp_path, manifest, script_text=SCORE_SCRIPT):
    script = tmp_path / "score.py"
    script.write_text(script_text)
    config = tmp_path / "function.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "sequence-count", "tool": "function",
        "design_set": str(manifest),
        "function": {"name": "counts", "script": str(script), "inputs": ["sequence"],
                     "metrics": {"lysines": {"direction": "none"}}},
    }))
    return config


def score(database, general, config, output):
    return CLI.invoke(app, ["function", "run", str(database), "--general", str(general),
                           "--config", str(config), "--output-dir", str(output)])


def stored_counts(database):
    with read_only(database) as connection:
        return {row.design_id: row.value for row in select_metrics(connection, names=["counts_lysines"])}


def test_sequences_roundtrip_through_scorer_and_database(io_campaign, tmp_path):
    database, general = io_campaign
    manifest = freeze(database, tmp_path / "sets")
    config = score_config(tmp_path, manifest)
    result = score(database, general, config, tmp_path / "scored")
    assert result.exit_code == 0, result.output
    assert stored_counts(database) == {"d0": 2, "d1": 1, "d2": 0, "d3": 3}
    with read_only(database) as connection:
        assert {d.design_id for d in select_designs(connection)} == {"d0", "d1", "d2", "d3"}


@pytest.mark.parametrize(("flags", "expected"), [
    (("--tool", "fixture", "--min-length", "5", "--max-length", "6"), {"d1": 1, "d2": 0}),
    (("--created-after", "2026-09-11T02:00:00+02:00", "--created-before", "2026-09-13T00:00:00Z"),
     {"d1": 1, "d2": 0}),
])
def test_metadata_and_date_selection_reaches_only_matching_sequences(io_campaign, tmp_path, flags, expected):
    database, general = io_campaign
    manifest = freeze(database, tmp_path / "sets", *flags)
    result = score(database, general, score_config(tmp_path, manifest), tmp_path / "scored")
    assert result.exit_code == 0, result.output
    assert stored_counts(database) == expected


def test_loss_filter_is_scoped_and_missing_loss_never_passes(io_campaign, tmp_path):
    database, general = io_campaign
    all_designs = freeze(database, tmp_path / "all")
    policy = tmp_path / "filter.yaml"
    policy.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "loss-policy", "tool": "filter",
        "design_set": str(all_designs), "evaluator_runs": ["measured-loss"],
        "filter_set": {"name": "low_loss", "rules": [{"name": "loss_gate", "thresholds": [
            {"metric": "trial_loss", "op": "<=", "value": 0.4, "aggregate": "mean"},
        ]}]},
    }))
    applied = CLI.invoke(app, ["filter", "apply", str(database), "--general", str(general),
                              "--filter", str(policy), "--output-dir", str(tmp_path / "filtered")])
    assert applied.exit_code == 0, applied.output
    chosen = freeze(database, tmp_path / "chosen", "--passed-filter", "low_loss", "--filter-run", "loss-policy")
    result = score(database, general, score_config(tmp_path, chosen), tmp_path / "scored")
    assert result.exit_code == 0, result.output
    assert stored_counts(database) == {"d0": 2, "d2": 0}


def optimize_roundtrip(database, general, manifest, tmp_path, script_text=OPTIMIZE_SCRIPT):
    script = tmp_path / "optimizer.py"
    script.write_text(script_text)
    model_path = tmp_path / "optimize.yaml"
    model_path.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "transport", "tool": "optimize", "design_set": str(manifest),
        "script": str(script), "driver_script": str(ROOT / "drivers/optimize/run_optimizer.py"),
        "inputs": ["sequence"], "loss_models": [], "metrics": {"edit_cost": {"direction": "min"}},
        "max_children": 1, "length_delta": 1, "seed": 7, "runtime": {}, "sharding": {"jobs": 1},
        "resources": {"gpus": 1, "cpus": 1, "memory_gb": 1, "walltime": "00:01:00"},
    }))
    loaded = load_configs(general, model_path)
    planned = plan(loaded, tmp_path / "optimized")
    for task in planned.tasks:
        spec = launch_spec(planned, task.task_id)
        for directory in spec.mkdirs:
            directory.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(spec.argv, env={**os.environ, **spec.env},
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
    collected = collect_run(planned.directory / "run.json")
    bundle = write_collected(collected, planned.directory / "collected.json")
    assert ingest_bundle(database, bundle)
    return collected


def test_selected_sequences_roundtrip_as_optimizer_children(io_campaign, tmp_path):
    database, general = io_campaign
    manifest = freeze(database, tmp_path / "sets", "--max-length", "5")
    result = optimize_roundtrip(database, general, manifest, tmp_path)
    assert {(d.parent_design_id, d.sequence) for d in result.designs} == {("d0", "KKAAA"), ("d1", "KAAAAA")}
    with read_only(database) as connection:
        children = select_designs(connection, DesignQuery(tools=("optimize",)))
        assert {d.sequence for d in children} == {"KKAAA", "KAAAAA"}
        metrics = select_metrics(connection, design_ids=[d.design_id for d in children])
        assert {(m.name, m.value, m.direction) for m in metrics} == {("transport_edit_cost", 1, "min")}
    assert not ingest_bundle(database, Path(result.run.output_uri) / "collected.json")


@pytest.mark.parametrize("bad_result", [
    '{"lysines": float("nan")}',
    '{"undeclared": 1}',
])
def test_invalid_scores_do_not_enter_database(io_campaign, tmp_path, bad_result):
    database, general = io_campaign
    manifest = freeze(database, tmp_path / "sets")
    script = SCORE_SCRIPT.replace('{"lysines": candidate["sequence"].count("K")}', bad_result)
    result = score(database, general, score_config(tmp_path, manifest, script), tmp_path / "scored")
    assert result.exit_code != 0
    assert stored_counts(database) == {}


def test_invalid_optimizer_child_is_rejected_without_creating_design(io_campaign, tmp_path):
    database, general = io_campaign
    manifest = freeze(database, tmp_path / "sets", "--limit", "1")
    script = OPTIMIZE_SCRIPT.replace('parent["sequence"] + "A"', 'parent["sequence"] + "X"')
    result = optimize_roundtrip(database, general, manifest, tmp_path, script)
    assert not result.designs and result.run.status == "failed"
    with read_only(database) as connection:
        assert select_designs(connection, DesignQuery(tools=("optimize",))) == ()


def test_invalid_input_identity_is_refused_by_portable_parser(tmp_path):
    script = tmp_path / "score.py"
    script.write_text(SCORE_SCRIPT)
    inputs, outputs = tmp_path / "inputs.jsonl", tmp_path / "outputs.jsonl"
    inputs.write_text('{"index": 0, "sequence": "KK"}\n{"index": 0, "sequence": "AA"}\n')
    result = subprocess.run([sys.executable, str(script), "--inputs", str(inputs), "--outputs", str(outputs)],
                            env={**os.environ, "PYTHONPATH": str(ROOT / "drivers")},
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()


def test_invalid_selection_never_produces_a_design_set(io_campaign, tmp_path):
    database, _ = io_campaign
    for i, flags in enumerate([
        ["--created-after", "2026-09-13T00:00:00Z", "--created-before", "2026-09-11T00:00:00Z"],
        ["--created-after", "2026-09-11T00:00:00"],
        ["--passed-filter", "low_loss"],
    ]):
        destination = tmp_path / f"invalid-{i}"
        result = CLI.invoke(app, ["designset", "build", str(database), "--out-dir", str(destination), *flags])
        assert result.exit_code != 0
        assert not list(destination.glob("*.json"))
