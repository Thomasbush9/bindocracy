"""Portable real-DB bundles, lineage and organizer cutoff behavior."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bindocracy.benchmark.models import (
    AssetIdentity,
    AttemptRecord,
    EnvironmentManifest,
    SubmissionPolicy,
    TaskSpec,
    WorkspaceManifest,
)
from bindocracy.benchmark.submission import (
    SubmissionError,
    _digest,
    accept_submission,
    archive_evidence,
    build_submission,
    check_submission,
)
from bindocracy.benchmark.workspace import _task_digest
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.designset import build_design_set
from bindocracy.runs.inputs import TargetDigest
from bindocracy.runs.selection import read_only
from bindocracy.store import CampaignStore, create_database
from bindocracy.store.query import DesignQuery, select_designs
from bindocracy.store.records import (
    ArtifactRecord,
    CollectedRun,
    ConfigRecord,
    DesignRecord,
    RunRecord,
)


def campaign_bundle(root: Path, *, model_config=None):
    """Real typed database + real files; no scientific model or scheduler stub."""
    source = root / "source"
    source.mkdir(parents=True)
    fasta = source / "target.fasta"
    fasta.write_text(">target\nACDEFGHIK\n")
    general = GeneralConfig.model_validate({
        "schema_version": 1, "campaign": {"name": "benchmark-test"},
        "target": {"name": "target", "sequence_fasta": str(fasta), "chain_id": "A"},
        "cluster": {"executor": "slurm", "account": "test", "default_partition": "test"},
    })
    config = ConfigRecord(general_name="benchmark-test", general_schema_version=1,
                          general_config_json=general.model_dump(mode="json"), model_name="fixture",
                          tool="fixture", model_schema_version=1, model_config_json=model_config or {"name": "fixture"})
    database = create_database(source / "campaign.duckdb")
    with CampaignStore(database) as store:
        store.assert_target("target", TargetDigest.of("target", "ACDEFGHIK").sequence_sha256)
        parent = RunRecord(run_id="parent-run", name="parent", tool="fixture", kind="generate",
                           status="succeeded", model_config_id=config.model_config_id)
        store.ingest(CollectedRun(run=parent, designs=(DesignRecord(design_id="ancestor", run_id=parent.run_id,
                     native_id="ancestor", candidate_type="sequence", sequence="ACDE"),)), configs=[config])
        run = RunRecord(run_id="child-run", name="child", tool="fixture", kind="optimize", status="succeeded",
                        parent_run_id=parent.run_id, model_config_id=config.model_config_id)
        store.ingest(CollectedRun(run=run, designs=(
            DesignRecord(design_id="child-a", parent_design_id="ancestor", run_id=run.run_id,
                         native_id="a", candidate_type="sequence", sequence="ACDEF"),
            DesignRecord(design_id="child-b", parent_design_id="ancestor", run_id=run.run_id,
                         native_id="b", candidate_type="sequence", sequence="ACDEFG"),
        )))
        evaluator = RunRecord(run_id="eval", name="eval", tool="fixture", kind="evaluate", status="succeeded",
                              output_uri=str(source), model_config_id=config.model_config_id)
        artifacts = []
        for identity in ("child-a", "child-b"):
            structure = source / f"{identity}.pdb"
            structure.write_text("HEADER    SOFTWARE TEST ONLY\nEND\n")
            artifacts.append(ArtifactRecord(run_id="eval", design_id=identity, kind="predicted_structure",
                             uri=structure.name, sha256="sha256:" + hashlib.sha256(structure.read_bytes()).hexdigest(),
                             size_bytes=structure.stat().st_size, metadata={"condition": "complex", "replicate": 0}))
        store.ingest(CollectedRun(run=evaluator, artifacts=tuple(artifacts)))
    with read_only(database) as connection:
        designs = {row.design_id: row for row in select_designs(connection)}
    # Nondefault order is deliberate: no accidental re-sort during submission.
    design_set = build_design_set([designs["child-b"], designs["child-a"]], database=database, query=DesignQuery())
    output = root / "bundle"
    build_submission(database, output, design_set=design_set, structures_from="eval",
                     policy=SubmissionPolicy(require_structures=True))
    return source, output, general


def resign(bundle: Path, manifest: dict | None = None):
    manifest = manifest or json.loads((bundle / "manifest.json").read_text())
    manifest["files"] = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in manifest["files"]}
    manifest["digest"] = _digest({key: value for key, value in manifest.items() if key != "digest"})
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def authorized_attempt(root: Path, bundle: Path, general: GeneralConfig):
    now = datetime.now(UTC)
    task = TaskSpec(task_id="target-task", version="1", prompt="Design test candidates.", target=general.target,
                    budget={"walltime_seconds": 60, "max_gpus": 0, "max_cpus": 1, "max_memory_gb": 1},
                    submission={"require_structures": True}, allowed_tools=("fixture",))
    assets = {"target": AssetIdentity(path=general.target.sequence_fasta,
              sha256=hashlib.sha256(general.target.sequence_fasta.read_bytes()).hexdigest())}
    task_digest = _task_digest(task, assets)
    environment = EnvironmentManifest(name="test", harness_revision="test", harness_path=root,
                    lock_path=root / "lock", harness_sha256="a" * 64, lock_sha256="b" * 64,
                    site={"controller": {"account": "test", "partition": "test", "cpus": 1,
                          "memory_gb": 1, "walltime": "00:01:00", "gpus": 0}, "max_workers": 1, "max_total_gpus": 0})
    workspace = WorkspaceManifest(attempt_id="attempt-1", root=root / "source", database=root / "source/campaign.duckdb",
                    general_config=root / "source/general.yaml", site_config=root / "source/site.yaml",
                    bindings=root / "source/bindings.yaml", task=task, environment=environment, input_assets=assets,
                    task_digest=task_digest, environment_digest="env-1", created_at=now)
    record = AttemptRecord(attempt_id="attempt-1", workspace=workspace.root, task_id=task.task_id,
                    task_digest=task_digest, environment_digest="env-1", agent={"name": "test-agent", "repeat": 0},
                    state="running", created_at=now, started_at=now, deadline=now + timedelta(seconds=60))
    control = root / "control"
    control.mkdir()
    for name, model in (("attempt", record), ("task", task), ("workspace", workspace), ("environment", environment)):
        (control / f"{name}.json").write_text(model.model_dump_json())
    shutil.copyfile(general.target.sequence_fasta, control / "target.fasta")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest.update(attempt_id=record.attempt_id, task_id=task.task_id, task_version=task.version,
                    task_digest=task_digest, environment_digest=record.environment_digest)
    resign(bundle, manifest)
    return control, record


def test_bundle_survives_source_removal_and_preserves_lineage(tmp_path):
    source, bundle, _ = campaign_bundle(tmp_path)
    shutil.rmtree(source)
    manifest = check_submission(bundle)
    assert [candidate["design_id"] for candidate in manifest["candidates"]] == ["child-b", "child-a"]
    lineage = json.loads((bundle / "provenance/lineage.json").read_text())
    assert set(lineage["designs"]) == {"ancestor", "child-a", "child-b"}
    assert lineage["runs"]["child-run"]["parent_run_id"] == "parent-run"
    assert all((bundle / candidate["structures"][0]["path"]).is_file() for candidate in manifest["candidates"])


@pytest.mark.parametrize("mutation", ["bytes", "extra", "symlink", "escape", "duplicate", "count", "lineage"])
def test_bundle_rejects_tampering(tmp_path, mutation):
    _, bundle, _ = campaign_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if mutation == "bytes":
        (bundle / "candidates.fasta").write_text("changed")
    elif mutation == "extra":
        (bundle / "unlisted.txt").write_text("unlisted")
    elif mutation == "symlink":
        path = bundle / "candidates.fasta"
        path.unlink()
        path.symlink_to(tmp_path / "source/target.fasta")
    elif mutation == "escape":
        manifest["files"]["../outside"] = "0" * 64
        manifest["digest"] = _digest({k: v for k, v in manifest.items() if k != "digest"})
        (bundle / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "duplicate":
        manifest["candidates"][1]["design_id"] = manifest["candidates"][0]["design_id"]
        resign(bundle, manifest)
    elif mutation == "count":
        manifest["candidate_count"] = 1
        resign(bundle, manifest)
    else:
        path = bundle / "provenance/lineage.json"
        lineage = json.loads(path.read_text())
        del lineage["designs"]["ancestor"]
        path.write_text(json.dumps(lineage))
        resign(bundle)
    with pytest.raises((SubmissionError, ValueError)):
        check_submission(bundle)


def test_acceptance_copies_immutable_bytes_and_updates_latest(tmp_path):
    source, bundle, general = campaign_bundle(tmp_path)
    control, record = authorized_attempt(tmp_path, bundle, general)
    accepted = accept_submission(bundle, control, now=record.started_at)
    archived = Path(accepted["path"])
    assert archived.parent == control / "accepted"
    assert archived.stat().st_mode & 0o222 == 0
    (bundle / "candidates.fasta").write_text("changed afterward")
    shutil.rmtree(source)
    assert check_submission(archived)["digest"] == accepted["digest"]
    assert json.loads((control / "attempt.json").read_text())["latest_submission"] == accepted


def test_cutoff_is_checked_after_copy_and_archive_publish(tmp_path):
    _, bundle, general = campaign_bundle(tmp_path)
    control, record = authorized_attempt(tmp_path, bundle, general)
    times = iter([record.started_at, record.started_at, record.deadline])
    with pytest.raises(SubmissionError, match="deadline"):
        accept_submission(bundle, control, now=lambda: next(times))
    assert json.loads((control / "attempt.json").read_text())["latest_submission"] is None


def test_copy_race_cannot_be_accepted(tmp_path, monkeypatch):
    import bindocracy.benchmark.submission as module
    _, bundle, general = campaign_bundle(tmp_path)
    control, record = authorized_attempt(tmp_path, bundle, general)
    original = module._copy_file
    changed = False
    def raced(source, destination, expected=None):
        nonlocal changed
        if not changed:
            changed = True
            (bundle / "candidates.fasta").write_text("race")
        return original(source, destination, expected)
    monkeypatch.setattr(module, "_copy_file", raced)
    with pytest.raises(SubmissionError, match="hash"):
        accept_submission(bundle, control, now=record.started_at)
    assert json.loads((control / "attempt.json").read_text())["latest_submission"] is None


def test_acceptance_checks_trusted_policy_identity_and_target(tmp_path):
    _, bundle, general = campaign_bundle(tmp_path)
    control, record = authorized_attempt(tmp_path, bundle, general)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["attempt_id"] = "other-attempt"
    resign(bundle, manifest)
    with pytest.raises(SubmissionError, match="unauthorized"):
        accept_submission(bundle, control, now=record.started_at)
    manifest["attempt_id"] = record.attempt_id
    resign(bundle, manifest)
    (control / "target.fasta").write_text(">target\nAAAA\n")
    with pytest.raises(SubmissionError, match="organizer target"):
        accept_submission(bundle, control, now=record.started_at)


def test_evidence_is_separate_and_requires_stopped_attempt(tmp_path):
    _, bundle, general = campaign_bundle(tmp_path)
    control, record = authorized_attempt(tmp_path, bundle, general)
    with pytest.raises(SubmissionError, match="stop"):
        archive_evidence(control, tmp_path / "evidence")
    stopped = record.model_copy(update={"state": "completed"})
    (control / "attempt.json").write_text(stopped.model_dump_json())
    evaluations = tmp_path / "evaluation"
    evaluations.mkdir()
    (evaluations / "stderr.log").write_text("preserved organizer diagnostic")
    evidence = archive_evidence(control, tmp_path / "evidence", evaluations=(evaluations,))
    assert evidence["kind"] == "experiment_evidence"
    assert "workspace/campaign.duckdb" in evidence["files"]
    assert "control/attempt.json" in evidence["files"]
    assert "evaluations/0/stderr.log" in evidence["files"]
    with pytest.raises(SubmissionError, match="unsupported"):
        check_submission(tmp_path / "evidence")


def test_multifile_kit_provenance_is_portable_and_complete(tmp_path):
    from bindocracy.kits.resolve import resolve_kit, write_lock
    example = Path(__file__).resolve().parents[1] / "examples/kits/residue-fraction"
    kit_root = tmp_path / "custom-kit"
    shutil.copytree(example, kit_root)
    lock = resolve_kit(kit_root)
    lock_path = tmp_path / "kit.lock.json"
    write_lock(lock, lock_path)
    source, bundle, _ = campaign_bundle(tmp_path, model_config={"custom": {"kit_lock": str(lock_path)}})
    shutil.rmtree(source)
    shutil.rmtree(kit_root)
    lock_path.unlink()
    manifest = check_submission(bundle)
    kit = manifest["kits"][0]
    helper = next(name for name in kit["code_files"] if name.endswith("/residues.py"))
    assert (bundle / helper).is_file()
    assert any("/dependencies/bindocracy-io/" in name for name in kit["code_files"])
    # Even a recomputed outer manifest cannot remove part of the locked closure.
    (bundle / helper).unlink()
    del manifest["files"][helper]
    del kit["code_files"][helper]
    resign(bundle, manifest)
    with pytest.raises(SubmissionError, match="closure"):
        check_submission(bundle)


def test_latest_acceptance_retains_earlier_archive_and_enforces_stopped_state(tmp_path):
    _, bundle, general = campaign_bundle(tmp_path)
    control, record = authorized_attempt(tmp_path, bundle, general)
    first = accept_submission(bundle, control, now=record.started_at)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["created_at"] = (record.started_at + timedelta(seconds=1)).isoformat()
    resign(bundle, manifest)
    second = accept_submission(bundle, control, now=record.started_at + timedelta(seconds=2))
    assert first["digest"] != second["digest"]
    assert check_submission(first["path"])["digest"] == first["digest"]
    current = json.loads((control / "attempt.json").read_text())
    assert current["latest_submission"] == second
    current["state"] = "completed"
    (control / "attempt.json").write_text(json.dumps(current))
    with pytest.raises(SubmissionError, match="not running"):
        accept_submission(bundle, control, now=record.started_at + timedelta(seconds=3))


def test_organizer_count_policy_overrides_bundled_policy(tmp_path):
    _, bundle, general = campaign_bundle(tmp_path)
    control, record = authorized_attempt(tmp_path, bundle, general)
    task = json.loads((control / "task.json").read_text())
    task["submission"]["max_candidates"] = 1
    (control / "task.json").write_text(json.dumps(task))
    # The relaxed policy in an agent-built bundle cannot expand organizer limits.
    with pytest.raises(SubmissionError, match="count outside"):
        accept_submission(bundle, control, now=record.started_at)
