"""Execute a deterministic CPU command solely to exercise the grading contract."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_benchmark_submission import authorized_attempt, campaign_bundle

from bindocracy.benchmark.evaluation import (
    EvaluationError,
    EvaluationProtocol,
    aggregate_reports,
    evaluate_submission,
)
from bindocracy.benchmark.models import AttemptRecord
from bindocracy.benchmark.submission import _digest, accept_submission

SMOKE_SCORER = '''import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--bundle", type=Path, required=True)
p.add_argument("--result", type=Path, required=True)
a = p.parse_args()
assert a.bundle.parent.name == "accepted"
m = json.loads((a.bundle / "manifest.json").read_text())
protocol = json.loads(Path(os.environ["BINDOCRACY_EVALUATION_PROTOCOL"]).read_text())
rows = [{"index": c["index"], "status": "ok", "metrics": {"length": len(c["sequence"])}} for c in m["candidates"]]
r = {"schema_version": 1, "bundle_digest": m["digest"], "protocol_id": protocol["protocol_id"],
     "protocol_version": protocol["version"], "protocol_digest": os.environ["BINDOCRACY_EVALUATION_PROTOCOL_DIGEST"],
     "status": "ok", "score": sum(x["metrics"]["length"] for x in rows) / len(rows),
     "candidate_results": rows, "details": {"scientific_validation": False}}
a.result.write_text(json.dumps(r, allow_nan=False))
print("SOFTWARE CONTRACT DEMONSTRATION ONLY")
'''


def protocol_for(root, task_digest="task-one", *, source=SMOKE_SCORER, timeout=5):
    script = root / "organizer_scorer.py"
    script.write_text(source)
    executable = Path(sys.executable).resolve()
    return EvaluationProtocol(protocol_id="length-smoke", version="1", scientific=False,
                argv=(str(executable), str(script)),
                files={str(executable): hashlib.sha256(executable.read_bytes()).hexdigest(),
                       str(script): hashlib.sha256(script.read_bytes()).hexdigest()},
                task_versions={"target-task": "1"}, task_digests={"target-task": task_digest},
                timeout_seconds=timeout, score_min=0, score_max=100, failure_score=0)


def accepted_attempt(root):
    source, bundle, general = campaign_bundle(root)
    control, record = authorized_attempt(root, bundle, general)
    accepted = accept_submission(bundle, control, now=record.started_at)
    stopped = record.model_copy(update={"state": "completed", "latest_submission": accepted,
                                       "usage": {"cpu_seconds": 10.5, "gpu_seconds": 0}})
    (control / "attempt.json").write_text(stopped.model_dump_json())
    return source, bundle, control, stopped


def test_real_command_reads_accepted_bundle_without_original_workspace(tmp_path):
    source, bundle, control, record = accepted_attempt(tmp_path)
    protocol = protocol_for(tmp_path, record.task_digest)
    shutil.rmtree(source)
    shutil.rmtree(bundle)
    output = tmp_path / "evaluation"
    report = evaluate_submission(control, protocol, output)
    assert report["status"] == "completed"
    assert report["score"] == 5.5
    assert report["scientific"] is False
    assert report["usage"]["cpu_seconds"] == 10.5
    assert report == json.loads((output / "report.json").read_text())
    assert "DEMONSTRATION" in (output / "stdout.log").read_text()
    assert len(report["result"]["candidate_results"]) == 2


@pytest.mark.parametrize("mutation", ["nonfinite", "missing_member", "extra_key", "identity"])
def test_invalid_structured_evaluator_output_is_not_scientific_failure(tmp_path, mutation):
    _, _, control, record = accepted_attempt(tmp_path)
    additions = {
        "nonfinite": 'r["score"] = float("nan")',
        "missing_member": 'r["candidate_results"].pop()',
        "extra_key": 'r["unexpected"] = 1',
        "identity": 'r["bundle_digest"] = "other"',
    }
    source = SMOKE_SCORER.replace('a.result.write_text(json.dumps(r, allow_nan=False))',
                                 additions[mutation] + '\na.result.write_text(json.dumps(r))')
    protocol = protocol_for(tmp_path, record.task_digest, source=source)
    report = evaluate_submission(control, protocol, tmp_path / "evaluation")
    assert report["status"] == "invalid_evaluator_output"
    assert report["score"] is None


def test_grader_tamper_and_timeout_are_infrastructure_failures(tmp_path):
    _, _, control, record = accepted_attempt(tmp_path)
    protocol = protocol_for(tmp_path, record.task_digest)
    Path(protocol.argv[1]).write_text("raise SystemExit(0)")
    report = evaluate_submission(control, protocol, tmp_path / "changed-grader")
    assert report["status"] == "infrastructure_failure"
    assert "asset changed" in report["error"]
    protocol = protocol_for(tmp_path, record.task_digest, source="import time; time.sleep(60)", timeout=0.1)
    report = evaluate_submission(control, protocol, tmp_path / "timeout")
    assert report["status"] == "infrastructure_failure"
    assert "walltime" in report["error"]
    assert (tmp_path / "timeout/stderr.log").is_file()


def make_record(identity, agent, task, digest, repeat):
    return AttemptRecord(attempt_id=identity, workspace=Path("/unavailable-workspace"), task_id=task,
                         task_digest=digest, environment_digest="same-env", agent={"name": agent, "repeat": repeat},
                         state="completed", created_at=datetime.now(UTC), usage={"cpu_seconds": repeat + 1})


def make_report(record, protocol, score, status="completed"):
    report = {"schema_version": 1, "kind": "benchmark_evaluation", "attempt_id": record.attempt_id,
              "agent": record.agent["name"], "task_id": record.task_id, "task_digest": record.task_digest,
              "task_version": protocol.task_versions[record.task_id], "environment_digest": record.environment_digest,
              "protocol_digest": protocol.digest, "protocol_id": protocol.protocol_id,
              "protocol_version": protocol.version, "score": score, "status": status}
    report["digest"] = _digest(report)
    return report


def test_task_balanced_aggregation_keeps_missing_and_failed_attempts(tmp_path):
    protocol = protocol_for(tmp_path).model_copy(update={"task_versions": {"a": "1", "b": "1"}, "task_digests": {"a": "aa", "b": "bb"}})
    records = [make_record(f"a-{i}", "agent", "a", "aa", i) for i in range(4)]
    records.append(make_record("b-0", "agent", "b", "bb", 0))
    reports = [make_report(records[0], protocol, 8), make_report(records[1], protocol, 4),
               make_report(records[2], protocol, None, "infrastructure_failure"), make_report(records[4], protocol, 10)]
    first = aggregate_reports(reports, attempts=records, protocol=protocol, seed=17, bootstrap_samples=100)
    second = aggregate_reports(list(reversed(reports)), attempts=records, protocol=protocol, seed=17, bootstrap_samples=100)
    summary = first["agents"]["agent"]
    assert summary["task_balanced_mean"] == 6.5  # ((8+4+0+0)/4 + 10)/2, not 22/5.
    assert summary["attempts"] == 5
    assert summary["completed"] == 3
    assert summary["outcomes"]["missing_report"] == 1
    assert summary["outcomes"]["infrastructure_failure"] == 1
    assert len(summary["reported_usage"]) == 5
    assert first == second


def test_incomparable_protocols_and_task_versions_are_refused(tmp_path):
    protocol = protocol_for(tmp_path)
    record = make_record("one", "agent", "target-task", "task-one", 0)
    report = make_report(record, protocol.model_copy(update={"version": "different"}), 5)
    with pytest.raises(EvaluationError, match="incomparable"):
        aggregate_reports([report], attempts=[record], protocol=protocol)
    changed = record.model_copy(update={"task_digest": "changed-task"})
    with pytest.raises(EvaluationError, match="incomparable task"):
        aggregate_reports([], attempts=[changed], protocol=protocol)


def test_baseline_pairing_includes_failures_and_refuses_cherry_picked_roster(tmp_path):
    protocol = protocol_for(tmp_path)
    records = [make_record(f"{agent}-{repeat}", agent, "target-task", "task-one", repeat)
               for agent in ("baseline", "agent") for repeat in range(2)]
    reports = [make_report(records[0], protocol, 2), make_report(records[1], protocol, 4),
               make_report(records[2], protocol, 8)]
    result = aggregate_reports(reports, attempts=records, protocol=protocol, baseline="baseline", bootstrap_samples=100)
    assert result["paired_comparisons"]["agent"]["task_balanced_mean_difference"] == 1
    with pytest.raises(EvaluationError, match="matching"):
        aggregate_reports(reports, attempts=records[:-1], protocol=protocol, baseline="baseline", bootstrap_samples=10)
    with pytest.raises(EvaluationError, match="duplicate"):
        aggregate_reports([reports[0], reports[0]], attempts=records, protocol=protocol)


def test_missing_submission_has_explicit_agent_outcome(tmp_path):
    _, bundle, general = campaign_bundle(tmp_path)
    control, record = authorized_attempt(tmp_path, bundle, general)
    record = record.model_copy(update={"state": "agent_failed"})
    (control / "attempt.json").write_text(record.model_dump_json())
    protocol = protocol_for(tmp_path, record.task_digest)
    report = evaluate_submission(control, protocol, tmp_path / "evaluation")
    assert report["status"] == "agent_failure"
    assert report["score"] is None
    assert not (tmp_path / "evaluation/stdout.log").exists()
