"""Ranking's observable cohorts, coverage, provenance, and restart semantics."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest
import yaml
from typer.testing import CliRunner

from bindocracy.config.models import GeneralConfig
from bindocracy.ranking.models import RankingPolicy
from bindocracy.ranking.run import RankingError, apply_ranking, run_ranking
from bindocracy.runs.designset import DesignSet
from bindocracy.runs.selection import SelectionError, build, select
from bindocracy.store import CampaignStore, create_database
from bindocracy.store.query import DesignQuery
from bindocracy.store.records import (
    CollectedRun,
    ConfigRecord,
    DecisionRecord,
    DesignRecord,
    MetricRecord,
    RunRecord,
)


@pytest.fixture
def campaign(tmp_path):
    """Create a genuine database from typed records, with no external tools."""
    general = GeneralConfig.model_validate(
        {
            "schema_version": 1,
            "campaign": {"name": "rank-test"},
            "target": {
                "name": "target",
                "sequence_fasta": str(tmp_path / "target.fasta"),
                "chain_id": "A",
            },
            "cluster": {"executor": "slurm", "account": "test", "default_partition": "test"},
        }
    )

    def make(designs, metrics, *, query=None, native_passes=None):
        database = create_database(tmp_path / "campaign.duckdb")
        config = ConfigRecord(
            general_name="rank-test",
            general_schema_version=1,
            general_config_json=general.model_dump(mode="json"),
            model_name="fixture",
            tool="fixture",
            model_schema_version=1,
            model_config_json={"name": "fixture"},
        )
        with CampaignStore(database) as store:
            for tool in sorted({tool for _, tool, _ in designs}):
                run = RunRecord(
                    run_id=tool,
                    name=tool,
                    tool=tool,
                    kind="generate",
                    model_config_id=config.model_config_id,
                    status="succeeded",
                )
                store.ingest(
                    CollectedRun(
                        run=run,
                        designs=tuple(
                            DesignRecord(
                                design_id=design_id,
                                run_id=tool,
                                native_id=design_id,
                                candidate_type="sequence",
                                sequence=sequence,
                            )
                            for design_id, generator, sequence in designs
                            if generator == tool
                        ),
                        decisions=tuple(
                            DecisionRecord(
                                run_id=tool,
                                design_id=design_id,
                                kind="filter",
                                name="native_pass",
                                passed=design_id in native_passes,
                            )
                            for design_id, generator, _ in designs
                            if generator == tool and native_passes is not None
                        ),
                    ),
                    configs=[config],
                )
            for evaluator in sorted({row[0] for row in metrics} | {"eval"}):
                run = RunRecord(
                    run_id=evaluator,
                    name=f"name-{evaluator}",
                    tool="scorer",
                    kind="evaluate",
                    model_config_id=config.model_config_id,
                    status="succeeded",
                )
                store.ingest(
                    CollectedRun(
                        run=run,
                        metrics=tuple(
                            MetricRecord(
                                metric_id=f"{evaluator}-{index}",
                                run_id=evaluator,
                                design_id=design_id,
                                name=name,
                                replicate=replica,
                                value=value,
                                status=status,
                            )
                            for index, (
                                run_id,
                                design_id,
                                name,
                                replica,
                                value,
                                status,
                            ) in enumerate(metrics)
                            if run_id == evaluator
                        ),
                    )
                )
        _, _, manifest = build(database, query or DesignQuery(), tmp_path / "source")
        return database, general, manifest

    return make


def policy(manifest, **updates):
    return RankingPolicy.model_validate(
        {
            "name": "ranking",
            "design_set": str(manifest),
            "evaluator_runs": ["eval"],
            "group_by": "global",
            "inputs": {
                "q": {
                    "metric": "quality",
                    "replicas": [0],
                    "aggregate": "mean",
                }
            },
            "priorities": [{"inputs": ["q"], "direction": "max"}],
            "deduplicate_sequences": True,
            "shortage": "truncate",
            "cohorts": [
                {"name": "best", "end": "head", "count": 2},
                {"name": "worst", "end": "tail", "count": 1},
            ],
            **updates,
        }
    )


def metric(design_id, value, *, replica=0, name="quality", status="ok", run="eval"):
    return run, design_id, name, replica, value, status


def members(collected, name):
    return {
        decision.design_id
        for decision in collected.decisions
        if decision.kind == "filter" and decision.name == name and decision.passed
    }


def compute(campaign_data, tmp_path, **updates):
    database, general, manifest = campaign_data
    return run_ranking(
        database=database,
        general=general,
        policy=policy(manifest, **updates),
        output_dir=tmp_path / "rank",
    )[0]


def test_missing_failed_nonfinite_and_incomplete_replicas_are_not_tail(campaign, tmp_path):
    designs = [(f"d{i}", "alpha", "A" * (i + 2)) for i in range(5)]
    metrics = [
        metric("d0", 0.9),
        metric("d0", 0.7, replica=1),
        metric("d1", 0.99),
        metric("d2", 0.8),
        metric("d2", None, replica=1, status="failed"),
        metric("d3", float("inf")),
        metric("d3", 0.4, replica=1),
    ]
    result = compute(
        campaign(designs, metrics),
        tmp_path,
        inputs={
            "q": {
                "metric": "quality",
                "replicas": [0, 1],
                "aggregate": "mean",
                "min_coverage": 2,
            }
        },
    )
    assert members(result, "best") == {"d0"}
    assert members(result, "worst") == set()
    assert result.run.n_produced == 1
    rejected = {
        decision.design_id: decision.reason
        for decision in result.decisions
        if decision.kind == "filter" and decision.name == "selected" and not decision.passed
    }
    assert set(rejected) == {"d1", "d2", "d3", "d4"}
    assert rejected["d2"]["inputs"]["q"]["replicas"]["1"] == "failed"
    assert rejected["d3"]["inputs"]["q"]["replicas"]["0"] == "nonfinite_or_missing"


def test_ties_and_duplicate_representatives_follow_rank_not_source_order(campaign, tmp_path):
    data = campaign(
        [
            ("a", "alpha", "AAAA"),
            ("b", "alpha", "AAAA"),
            ("c", "alpha", "CCCC"),
            ("d", "alpha", "DDDD"),
        ],
        [metric("a", 0.1), metric("b", 0.9), metric("c", 0.9), metric("d", 0.2)],
    )
    result = compute(data, tmp_path)
    assert members(result, "best") == {"b", "c"}
    assert members(result, "worst") == {"d"}
    assert {
        decision.design_id: decision.rank
        for decision in result.decisions
        if decision.kind == "rank"
    } == {"b": 1, "c": 2, "d": 3}
    duplicate = next(
        decision
        for decision in result.decisions
        if decision.design_id == "a" and decision.name == "selected"
    )
    assert duplicate.reason["representative"] == "b"


def test_per_generator_ranks_reset_and_head_tail_are_disjoint(campaign, tmp_path):
    data = campaign(
        [
            ("a", "alpha", "AA"),
            ("b", "alpha", "BB"),
            ("c", "beta", "CC"),
            ("d", "beta", "DD"),
            ("e", "beta", "EE"),
        ],
        [metric(name, value) for name, value in zip("abcde", [1, 2, 3, 4, 5], strict=True)],
    )
    result = compute(data, tmp_path, group_by="generator")
    assert members(result, "best") == {"a", "b", "d", "e"}
    assert members(result, "worst") == {"c"}
    assert members(result, "best").isdisjoint(members(result, "worst"))
    firsts = [
        decision for decision in result.decisions if decision.kind == "rank" and decision.rank == 1
    ]
    assert {decision.design_id for decision in firsts} == {"b", "e"}
    assert len({decision.scope_id for decision in firsts}) == 2
    assert result.run.count_details["groups"]["alpha"]["shortage"] == 1
    with pytest.raises(RankingError, match="requested"):
        compute(data, tmp_path, group_by="generator", shortage="error")


def test_named_conservative_min_and_lexicographic_direction(campaign, tmp_path):
    data = campaign(
        [("a", "pxdesign", "AA"), ("b", "pxdesign", "BB"), ("c", "pxdesign", "CC")],
        [
            metric("a", 0.99),
            metric("a", 0.3, name="other"),
            metric("a", 1, name="error"),
            metric("b", 0.6),
            metric("b", 0.6, name="other"),
            metric("b", 3, name="error"),
            metric("c", 0.6),
            metric("c", 0.9, name="other"),
            metric("c", 1, name="error"),
        ],
    )
    result = compute(
        data,
        tmp_path,
        group_by="generator",
        priorities=[],
        inputs={
            name: {"metric": metric_name, "replicas": [0], "aggregate": "mean"}
            for name, metric_name in [("q", "quality"), ("o", "other"), ("e", "error")]
        },
        by_generator={
            "pxdesign": [
                {"inputs": ["q", "o"], "direction": "max"},
                {"inputs": ["e"], "direction": "min"},
            ]
        },
    )
    assert {
        decision.design_id: decision.rank
        for decision in result.decisions
        if decision.kind == "rank"
    } == {"c": 1, "b": 2, "a": 3}


def test_replica_zero_aggregate_is_not_pooled_with_individual_folds(campaign, tmp_path):
    data = campaign(
        [("a", "fbc", "AA"), ("b", "genie", "BB")],
        [
            metric("a", 0.8),
            metric("a", 0.1, replica=1),
            *[metric("b", value, replica=i) for i, value in enumerate([0.8, 0.7, 0.6, 0.5, 0.4])],
        ],
    )
    result = compute(
        data,
        tmp_path,
        group_by="generator",
        priorities=[],
        inputs={
            "fbc": {"metric": "quality", "replicas": [0], "aggregate": "mean"},
            "genie": {
                "metric": "quality",
                "replicas": [0, 1, 2, 3, 4],
                "aggregate": "mean",
                "min_coverage": 5,
            },
        },
        by_generator={
            "fbc": [{"inputs": ["fbc"], "direction": "max"}],
            "genie": [{"inputs": ["genie"], "direction": "max"}],
        },
    )
    scores = {
        decision.design_id: decision.reason["scores"][0]
        for decision in result.decisions
        if decision.kind == "rank"
    }
    assert scores == pytest.approx({"a": 0.8, "b": 0.6})


def test_explicit_source_and_evaluator_scoping(campaign, tmp_path):
    data = campaign(
        [("a", "alpha", "AA"), ("b", "alpha", "BB"), ("outside", "beta", "CC")],
        [
            metric("a", 0.8),
            metric("b", 0.2),
            metric("outside", 1),
            metric("a", 0.1, run="other"),
            metric("b", 0.99, run="other"),
        ],
        query=DesignQuery(tools=("alpha",)),
    )
    result = compute(data, tmp_path, cohorts=[{"name": "best", "end": "head", "count": 1}])
    assert members(result, "best") == {"a"}
    assert {decision.design_id for decision in result.decisions} == {"a", "b"}
    changed = compute(
        data,
        tmp_path,
        evaluator_runs=["other"],
        cohorts=[{"name": "best", "end": "head", "count": 1}],
    )
    assert members(changed, "best") == {"b"}
    assert changed.run.run_id != result.run.run_id
    ambiguous = compute(data, tmp_path, evaluator_runs=["eval", "other"])
    assert ambiguous.run.n_passed == 0
    scoped = compute(
        data,
        tmp_path,
        evaluator_runs=["eval", "other"],
        inputs={
            "q": {
                "metric": "quality",
                "replicas": [0],
                "aggregate": "mean",
                "evaluator_runs": ["eval"],
            }
        },
    )
    assert members(scoped, "best") == {"a", "b"}


def test_restart_safety_metric_revision_and_downstream_retrieval(campaign, tmp_path):
    database, general, manifest = campaign(
        [("a", "alpha", "AA"), ("b", "alpha", "BB"), ("c", "alpha", "CC")],
        [metric("a", 0.9), metric("b", 0.8), metric("c", 0.1)],
    )
    configured = policy(manifest)
    args = {
        "database": database,
        "general": general,
        "policy": configured,
        "output_dir": tmp_path / "rank",
    }
    first, inserted, outputs = apply_ranking(**args)
    bundle = Path(first.run.output_uri) / "collected.json"
    before = bundle.read_bytes()
    second, repeated, _ = apply_ranking(**args)
    assert inserted and not repeated
    assert second.run.run_id == first.run.run_id
    assert bundle.read_bytes() == before
    for name, expected in [("best", {"a", "b"}), ("worst", {"c"}), ("selected", {"a", "b", "c"})]:
        rows, query = select(
            database, DesignQuery(passed_filter=(name,), filter_runs=(first.run.run_id,))
        )
        assert {row.design_id for row in rows} == expected
        exported = DesignSet.read(outputs[name])
        assert {entry.design_id for entry in exported.entries} == expected
        assert exported.query == query.model_dump(mode="json")
    with duckdb.connect(str(database)) as connection:
        connection.execute("UPDATE metrics SET value = 0.99 WHERE design_id = 'c'")
    revised, changed, _ = apply_ranking(**args)
    assert changed and revised.run.run_id != first.run.run_id
    assert members(revised, "best") == {"a", "c"}
    old, _ = select(database, DesignQuery(passed_filter=("best",), filter_runs=(first.run.run_id,)))
    assert {row.design_id for row in old} == {"a", "b"}


def test_empty_eligibility_exports_empty_sets_and_no_fake_rank(campaign, tmp_path):
    database, general, manifest = campaign([("a", "alpha", "AA")], [])
    result, _, outputs = apply_ranking(
        database=database, general=general, policy=policy(manifest), output_dir=tmp_path / "rank"
    )
    assert result.run.n_passed == 0
    assert not any(decision.kind == "rank" for decision in result.decisions)
    for path in outputs.values():
        assert DesignSet.read(path).entries == ()
        assert ">" not in path.with_suffix(".fasta").read_text()


def test_tampered_source_cannot_reuse_claimed_identity(campaign, tmp_path):
    database, general, manifest = campaign([("a", "alpha", "AA")], [metric("a", 0.9)])
    original = json.loads(manifest.read_text())
    original["entries"][0]["sequence"] = "BB"
    manifest.write_text(json.dumps(original))
    with pytest.raises(RankingError, match="identity"):
        run_ranking(
            database=database,
            general=general,
            policy=policy(manifest),
            output_dir=tmp_path / "rank",
        )


def test_rank_cli_and_existing_designset_cli(campaign, tmp_path):
    from bindocracy.cli import app

    database, general, manifest = campaign(
        [("a", "alpha", "AA"), ("b", "alpha", "BB")], [metric("a", 0.9), metric("b", 0.1)]
    )
    general_path = tmp_path / "general.yaml"
    general_path.write_text(yaml.safe_dump(general.model_dump(mode="json")))
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy(manifest).model_dump(mode="json")))
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "rank",
            "apply",
            str(database),
            "--general",
            str(general_path),
            "--policy",
            str(policy_path),
            "--output-dir",
            str(tmp_path / "rank"),
        ],
    )
    assert result.exit_code == 0, result.output
    run_id = next(
        line.removeprefix("run: ")
        for line in result.output.splitlines()
        if line.startswith("run: ")
    )
    selection = runner.invoke(
        app,
        [
            "designset",
            "build",
            str(database),
            "--passed-filter",
            "selected",
            "--filter-run",
            run_id,
            "--out-dir",
            str(tmp_path / "selected"),
        ],
    )
    assert selection.exit_code == 0, selection.output
    exported = DesignSet.read(next((tmp_path / "selected").glob("*.json")))
    assert {entry.design_id for entry in exported.entries} == {"a", "b"}


def test_native_filter_membership_is_explicit_eligibility(campaign, tmp_path):
    database, general, _ = campaign(
        [("a", "alpha", "AA"), ("b", "alpha", "BB")],
        [metric("a", 0.1), metric("b", 0.99)],
        native_passes={"a"},
    )
    source, _, manifest = build(
        database,
        DesignQuery(passed_filter=("native_pass",), filter_runs=("alpha",)),
        tmp_path / "native-source",
    )
    assert {entry.design_id for entry in source.entries} == {"a"}
    ranked = compute((database, general, manifest), tmp_path)
    assert members(ranked, "selected") == {"a"}
    with pytest.raises(SelectionError, match="no filter decisions"):
        select(database, DesignQuery(passed_filter=("native_pass",), filter_runs=("eval",)))
    with pytest.raises(SelectionError, match="no.*run named"):
        select(database, DesignQuery(passed_filter=("native_pass",), filter_runs=("absent",)))


def test_explicit_minimum_coverage_allows_only_requested_replica_subset(campaign, tmp_path):
    data = campaign(
        [("a", "alpha", "AA"), ("b", "alpha", "BB")],
        [
            metric("a", 0.8),
            metric("a", 0.0, replica=5),
            metric("b", 0.6),
            metric("b", 0.6, replica=1),
        ],
    )
    result = compute(
        data,
        tmp_path,
        inputs={
            "q": {
                "metric": "quality",
                "replicas": [0, 1],
                "aggregate": "mean",
                "min_coverage": 1,
            }
        },
        cohorts=[{"name": "best", "end": "head", "count": 1}],
    )
    assert members(result, "best") == {"a"}
    winning = next(
        decision
        for decision in result.decisions
        if decision.kind == "rank" and decision.design_id == "a"
    )
    assert winning.reason["inputs"]["q"]["coverage"] == 1
    assert winning.reason["scores"] == [0.8]


def test_changed_source_or_policy_has_a_distinct_decision_identity(campaign, tmp_path):
    database, general, manifest = campaign(
        [("a", "alpha", "AA"), ("b", "beta", "BB")],
        [metric("a", 0.1), metric("b", 0.9)],
    )
    first = compute((database, general, manifest), tmp_path)
    _, _, subset = build(database, DesignQuery(tools=("alpha",)), tmp_path / "subset")
    narrower = compute((database, general, subset), tmp_path)
    other_policy = compute(
        (database, general, manifest),
        tmp_path,
        cohorts=[{"name": "best", "end": "head", "count": 1}],
    )
    assert len({run.run.run_id for run in (first, narrower, other_policy)}) == 3
    assert members(narrower, "selected") == {"a"}
    assert members(other_policy, "selected") == {"b"}
