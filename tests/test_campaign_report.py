"""Scientific scope and offline integrity contracts, without schedulers or GPUs."""

import csv
import json
import math

import duckdb
import pytest
from pyarrow import parquet

from bindocracy.campaign.report import (
    audit_campaign,
    campaign_report,
    export_campaign,
    write_report,
)
from bindocracy.runs.inputs import digest_of
from bindocracy.runs.manifest import RunManifest, TaskPlan
from bindocracy.runs.selection import SelectionError
from bindocracy.runs.staging import staged_digest, write_collected
from bindocracy.store import CampaignStore
from bindocracy.store.records import (
    ArtifactRecord,
    CollectedRun,
    ConfigRecord,
    DecisionRecord,
    DesignRecord,
    MetricRecord,
    RunRecord,
    canonical_json,
    sha256_text,
)


@pytest.fixture
def config():
    return ConfigRecord(
        general_name="campaign",
        general_schema_version=1,
        general_config_json={"target": "fixture"},
        model_name="fixture",
        tool="fixture",
        model_schema_version=1,
        model_config_json={"replicas": 3},
    )


@pytest.fixture
def database(tmp_path, config):
    database = tmp_path / "campaign.duckdb"
    generate = RunRecord(
        run_id="generate",
        name="generated",
        tool="fixture",
        kind="generate",
        model_config_id=config.model_config_id,
        status="partial",
        n_requested=4,
        n_attempted=3,
        n_produced=2,
    )
    designs = tuple(
        DesignRecord(
            design_id=f"d{i}",
            run_id=generate.run_id,
            native_id=f"native{i}",
            candidate_type="sequence",
            sequence="ACDE",
            status="partial" if i else "produced",
        )
        for i in range(2)
    )
    with CampaignStore.create(database) as store:
        store.ingest(CollectedRun(run=generate, designs=designs), configs=[config])
        for run_id, values in (
            ("score-a", (0.2, 0.8, float("inf"))),
            ("score-b", (10.0, 20.0, float("nan"))),
        ):
            run = RunRecord(
                run_id=run_id,
                name="scoring",
                tool="fixture",
                kind="evaluate",
                model_config_id=config.model_config_id,
                status="partial",
                n_requested=6,
                n_attempted=4,
                n_produced=3,
                workflow_metadata={"scope_id": run_id, "protocol_sha256": run_id},
            )
            metrics = tuple(
                MetricRecord(
                    run_id=run_id,
                    design_id="d0",
                    name="model_score",
                    replicate=i,
                    value=value,
                    direction="max",
                )
                for i, value in enumerate(values)
            )
            metrics += (
                MetricRecord(
                    run_id=run_id,
                    design_id="d1",
                    name="model_score",
                    replicate=0,
                    value=None,
                    status="failed",
                    direction="max",
                ),
            )
            decisions = tuple(
                DecisionRecord(
                    run_id=run_id, design_id="d0", kind="filter", name=f"rule{i}", passed=True
                )
                for i in range(2)
            )
            store.ingest(CollectedRun(run=run, metrics=metrics, decisions=decisions))
        failed = RunRecord(
            run_id="failed",
            name="failed",
            tool="fixture",
            kind="evaluate",
            model_config_id=config.model_config_id,
            status="failed",
            n_requested=2,
            n_attempted=2,
            n_produced=0,
            error={"message": "tool failed"},
        )
        store.ingest(CollectedRun(run=failed))
        for identifier, passed in (("rank-a", True), ("rank-b", False)):
            rank = RunRecord(
                run_id=identifier,
                name="ranking",
                tool="fixture",
                kind="rank",
                model_config_id=config.model_config_id,
                status="succeeded",
            )
            store.ingest(
                CollectedRun(
                    run=rank,
                    decisions=(
                        DecisionRecord(
                            run_id=identifier,
                            design_id="d0",
                            kind="rank",
                            name="ordering",
                            rank=1,
                            scope_id=identifier,
                        ),
                        DecisionRecord(
                            run_id=identifier,
                            design_id="d0",
                            kind="filter",
                            name="top",
                            passed=passed,
                            scope_id=identifier,
                        ),
                    ),
                )
            )
    return database


def test_partial_failed_and_evaluation_counts_are_not_workflow_success(database):
    before = database.read_bytes()
    result = campaign_report(database, runs=("generate", "failed", "score-a"))
    outcomes = {item["run"]["run_id"]: item["scientific_outcome"] for item in result["runs"]}
    assert outcomes["generate"]["status"] == "partial"
    assert outcomes["generate"]["attempted_not_produced"] == 1
    assert outcomes["generate"]["design_status_counts"] == {"produced": 1, "partial": 1}
    assert outcomes["score-a"]["new_design_rows"] == 0
    assert outcomes["failed"]["status"] == "failed"
    assert outcomes["failed"]["produced"] == 0
    assert outcomes["failed"]["attempted_not_produced"] == 2
    assert all(item["workflow_completion"].startswith("not inferred") for item in result["runs"])
    assert database.read_bytes() == before


def test_run_ambiguity_and_scoped_finite_replica_summaries(database):
    with pytest.raises(SelectionError, match="named 'scoring'"):
        campaign_report(database, runs=("scoring",))
    with pytest.raises(SelectionError, match="no run"):
        campaign_report(database, runs=("absent",))
    result = campaign_report(database, runs=("score-a", "score-b", "score-a"))
    assert len(result["runs"]) == 2
    metrics = {item["run"]["run_id"]: item["metrics"][0] for item in result["runs"]}
    assert metrics["score-a"]["mean"] == pytest.approx(0.5)
    assert metrics["score-b"]["mean"] == pytest.approx(15)
    assert metrics["score-a"]["nonfinite_rows"] == 1
    assert metrics["score-b"]["nonfinite_rows"] == 1
    assert metrics["score-a"]["status_counts"] == {"ok": 3, "failed": 1}
    assert metrics["score-a"]["replica_coverage"] == [
        {"design_id": "d0", "observed": [0, 1, 2], "finite_ok": [0, 1]},
        {"design_id": "d1", "observed": [0], "finite_ok": []},
    ]
    json.dumps(result, allow_nan=False)


def test_cohort_membership_is_bound_to_exact_decision_run(database):
    result = campaign_report(database, runs=("rank-a", "rank-b"))
    cohorts = {item["run"]["run_id"]: item["cohorts"] for item in result["runs"]}
    for run_id, expected in (("rank-a", True), ("rank-b", False)):
        top = next(cohort for cohort in cohorts[run_id] if cohort["name"] == "top")
        assert top["scope_id"] == run_id
        assert top["members"][0]["passed"] is expected
        assert top["members"][0]["design_id"] == "d0"


def test_csv_parquet_preserve_replicas_failed_rows_and_provenance(database, tmp_path):
    csv_path, parquet_path = tmp_path / "metrics.csv", tmp_path / "metrics.parquet"
    first = export_campaign(database, csv_path, runs=("score-a",))
    second = export_campaign(database, parquet_path, runs=("score-a",))
    assert first["rows"] == second["rows"] == 4  # Two decisions do not multiply rows.
    with csv_path.open() as handle:
        csv_rows = list(csv.DictReader(handle))
    parquet_rows = parquet.read_table(parquet_path).to_pylist()
    assert {(row["design_id"], row["replicate"]) for row in parquet_rows} == {
        ("d0", 0),
        ("d0", 1),
        ("d0", 2),
        ("d1", 0),
    }
    assert all(
        row["run_id"] == "score-a" and row["producing_run_id"] == "generate" for row in parquet_rows
    )
    assert all(
        json.loads(row["owner_workflow_metadata"])["scope_id"] == "score-a" for row in csv_rows
    )
    assert all(json.loads(row["model_config_json"])["replicas"] == 3 for row in parquet_rows)
    assert next(row for row in csv_rows if row["status"] == "failed")["value"] == "\\N"
    assert next(row for row in parquet_rows if row["status"] == "failed")["value"] is None
    assert math.isinf(
        next(row for row in parquet_rows if row["design_id"] == "d0" and row["replicate"] == 2)[
            "value"
        ]
    )


def test_empty_exports_keep_schema_and_refuse_source_or_existing_outputs(database, tmp_path):
    result = export_campaign(database, tmp_path / "failed.parquet", runs=("failed",))
    assert result["rows"] == 0
    table = parquet.read_table(tmp_path / "failed.parquet")
    assert "replicate" in table.column_names and "provenance_model_config_id" in table.column_names
    before = database.read_bytes()
    with pytest.raises(ValueError, match="source database"):
        export_campaign(database, database)
    alias = tmp_path / "alias.csv"
    alias.symlink_to(database)
    with pytest.raises(ValueError, match="source database"):
        export_campaign(database, alias)
    artifact = tmp_path / "output.json"
    artifact.write_text("scientific input")
    with pytest.raises(FileExistsError):
        write_report(campaign_report(database), artifact)
    assert artifact.read_text() == "scientific input"
    assert database.read_bytes() == before
    empty = tmp_path / "empty.duckdb"
    with CampaignStore.create(empty):
        pass
    assert campaign_report(empty)["runs"] == []


@pytest.fixture
def frozen(tmp_path, config):
    directory = tmp_path / "run"
    directory.mkdir()
    task_dir = directory / "tasks" / "0000"
    task_dir.mkdir(parents=True)
    manifest = RunManifest(
        run_id="frozen",
        name="frozen-run",
        tool="fixture",
        kind="generate",
        general_config_id=config.general_config_id,
        model_config_id=config.model_config_id,
        created_at=config.created_at,
        run_dir=str(directory),
        tasks=(
            TaskPlan(
                task_id=0,
                directory="tasks/0000",
                designs="tasks/0000/designs.jsonl",
                status="tasks/0000/status.json",
                log="tasks/0000/task.log",
                n_requested=1,
            ),
        ),
        designs_per_task=1,
        resources={},
        provenance={},
        container="fixture.sif",
        container_digest=None,
        code_revision=None,
        workflow={},
        config=config,
    )
    manifest_path = directory / "run.json"
    manifest_path.write_text(manifest.model_dump_json())
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    database = tmp_path / "campaign.duckdb"
    plan = {
        "schema_version": 1,
        "plan_dir": str(plan_dir),
        "database": str(database),
        "runs": [
            {
                "name": manifest.name,
                "tool": manifest.tool,
                "kind": "generate",
                "tasks": 1,
                "manifest": str(manifest_path),
            }
        ],
        "artifacts": {str(manifest_path): digest_of(manifest_path).sha256},
        "referenced_inputs": {},
    }
    plan["digest"] = sha256_text(canonical_json(plan))
    plan_path = plan_dir / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return plan_path, manifest, database


def complete(frozen, *, status="succeeded"):
    _plan_path, manifest, database = frozen
    task = manifest.tasks[0]
    output = manifest.path(task.designs)
    output.write_text('{"native_id":"n0","sequence":"ACDE"}\n' if status != "failed" else "")
    manifest.path(task.status).write_text(
        json.dumps({"task_id": 0, "status": status, "n_produced": int(status != "failed")})
    )
    artifacts = []
    for path in (output, manifest.path(task.status)):
        digest = digest_of(path)
        artifacts.append(
            ArtifactRecord(
                run_id=manifest.run_id,
                kind="fixture",
                uri=str(path.relative_to(manifest.directory)),
                sha256=digest.sha256,
                size_bytes=digest.size_bytes,
            )
        )
    run = manifest.to_run_record().model_copy(
        update={"status": status, "n_attempted": 1, "n_produced": int(status != "failed")}
    )
    designs = (
        ()
        if status == "failed"
        else (
            DesignRecord(
                design_id="d0",
                run_id=run.run_id,
                native_id="n0",
                candidate_type="sequence",
                sequence="ACDE",
            ),
        )
    )
    bundle = CollectedRun(run=run, designs=designs, artifacts=tuple(artifacts))
    bundle_path = write_collected(bundle, manifest.directory / "collected.json")
    with CampaignStore.create(database) as store:
        store.ingest(bundle, configs=[manifest.config], digest=staged_digest(bundle_path))
    (manifest.directory / "generation.done").touch()
    (manifest.directory / "ingested.json").write_text(
        json.dumps({"database": str(database), "inserted": True})
    )
    return bundle


def test_unstarted_work_is_incomplete_not_corrupt_and_creates_nothing(frozen):
    path, _manifest, database = frozen
    before = {file: file.read_bytes() for file in path.parent.parent.rglob("*") if file.is_file()}
    result = audit_campaign(path)
    assert result["integrity"] == "incomplete"
    assert result["runs"][0]["tasks"][0]["status"] == "unstarted"
    assert not database.exists()
    assert before == {
        file: file.read_bytes() for file in path.parent.parent.rglob("*") if file.is_file()
    }


@pytest.mark.parametrize("status", ["failed", "partial", "succeeded"])
def test_ingested_outcome_can_be_verified_without_scientific_success(frozen, status):
    complete(frozen, status=status)
    path, _manifest, database = frozen
    before = database.read_bytes()
    result = audit_campaign(path)
    assert result["integrity"] == "verified"
    assert result["runs"][0]["ingestion"] == "ingested"
    assert result["runs"][0]["tasks"][0]["status"] == status
    assert database.read_bytes() == before


def test_audit_does_not_normalize_corrupted_database_evidence(frozen):
    complete(frozen)
    path, _, database = frozen
    with duckdb.connect(str(database)) as connection:
        connection.execute("UPDATE designs SET sequence = 'acde' WHERE design_id = 'd0'")
    result = audit_campaign(path)
    assert result["integrity"] == "corrupt"
    differences = next(
        item["differences"]
        for item in result["findings"]
        if item["code"] == "database_bundle_mismatch"
    )
    assert any(item["table"] == "designs" and item["changed"] == ["d0"] for item in differences)


def test_marker_alone_cannot_prove_ingestion(frozen):
    path, manifest, database = frozen
    (manifest.directory / "ingested.json").write_text(
        json.dumps({"database": str(database), "inserted": True})
    )
    result = audit_campaign(path)
    assert result["integrity"] == "corrupt"
    assert "bundle_missing" in {item["code"] for item in result["findings"]}
    assert not database.exists()


def test_bundle_hash_and_actual_rows_are_checked_independently(frozen):
    complete(frozen)
    path, _manifest, database = frozen
    with duckdb.connect(str(database)) as connection:
        connection.execute("UPDATE designs SET sequence = 'AAAA' WHERE design_id = 'd0'")
    before = database.read_bytes()
    result = audit_campaign(path)
    assert result["integrity"] == "corrupt"
    difference = next(
        item for item in result["findings"] if item["code"] == "database_bundle_mismatch"
    )
    assert {"table": "designs", "missing": [], "extra": [], "changed": ["d0"]} in difference[
        "differences"
    ]
    assert database.read_bytes() == before


def test_changed_bundle_and_corrupt_artifact_are_detected(frozen):
    complete(frozen)
    path, manifest, _database = frozen
    bundle_path = manifest.directory / "collected.json"
    raw = json.loads(bundle_path.read_text())
    raw["collected"]["run"]["n_produced"] = 0
    bundle_path.write_text(json.dumps(raw))
    manifest.path(manifest.tasks[0].designs).write_text('{"native_id":"n0","sequence":"AAAA"}\n')
    result = audit_campaign(path)
    codes = {item["code"] for item in result["findings"]}
    assert "database_bundle_mismatch" in codes
    assert "artifact_hash_mismatch" in codes


def test_stale_marker_and_missing_database_rows_are_not_valid(frozen):
    complete(frozen)
    path, _manifest, database = frozen
    with duckdb.connect(str(database)) as connection:
        connection.execute("DELETE FROM designs WHERE design_id = 'd0'")
    result = audit_campaign(path)
    assert result["integrity"] == "corrupt"
    difference = next(
        item for item in result["findings"] if item["code"] == "database_bundle_mismatch"
    )
    assert {"table": "designs", "missing": ["d0"], "extra": [], "changed": []} in difference[
        "differences"
    ]


def test_optimizer_parent_counts_are_not_subtracted_from_child_counts(database, config):
    with CampaignStore(database) as store:
        store.ingest(
            CollectedRun(
                run=RunRecord(
                    run_id="optimization",
                    name="optimization",
                    tool="optimize",
                    kind="optimize",
                    model_config_id=config.model_config_id,
                    status="partial",
                    n_requested=4,
                    n_attempted=3,
                    n_produced=1,
                )
            )
        )
    result = campaign_report(database, runs=("optimization",))
    outcome = result["runs"][0]["scientific_outcome"]
    assert outcome["attempted_not_produced"] is None
    assert outcome["requested"] == 4 and outcome["produced"] == 1
    assert outcome["count_unit"] == "requested/attempted parents; produced children"


def test_native_csv_tasks_are_audited_as_csv_not_jsonl(frozen):
    path, manifest, _ = frozen
    task = manifest.tasks[0].model_copy(update={"designs": "tasks/0000/designs.csv"})
    manifest = manifest.model_copy(update={"tasks": (task,)})
    manifest_path = manifest.directory / "run.json"
    manifest_path.write_text(manifest.model_dump_json())
    plan = json.loads(path.read_text())
    plan["artifacts"][str(manifest_path)] = digest_of(manifest_path).sha256
    del plan["digest"]
    plan["digest"] = sha256_text(canonical_json(plan))
    path.write_text(json.dumps(plan))
    manifest.path(task.status).write_text(
        json.dumps({"task_id": 0, "status": "succeeded", "n_produced": 1})
    )
    manifest.path(task.designs).write_text("native_id,sequence\nn0,ACDE\n")
    assert audit_campaign(path)["integrity"] == "incomplete"
    manifest.path(task.designs).write_text("native_id,sequence\nn0,ACDE,unexpected\n")
    result = audit_campaign(path)
    assert result["integrity"] == "corrupt"
    assert "task_artifact_invalid" in {item["code"] for item in result["findings"]}
