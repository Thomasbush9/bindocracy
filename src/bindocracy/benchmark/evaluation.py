"""Organizer-owned held-out evaluation and predeclared, task-balanced summaries.

A passing software contract or a synthetic smoke score is not binding evidence.
The evaluator's scientific validity and operating-system isolation are external
organizer responsibilities, not consequences of these content hashes.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import signal
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from bindocracy.benchmark.models import AttemptRecord
from bindocracy.benchmark.submission import (
    SubmissionError,
    _atomic_json,
    _digest,
    _json,
    _read,
    check_submission,
)


class EvaluationError(ValueError):
    """An organizer protocol, result, or comparison was refused."""


class EvaluationProtocol(BaseModel):
    """Trusted input, supplied out-of-band by the organizer, never by a bundle."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    protocol_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    # Every executable/script/config/model asset used by the grader should be
    # pinned here. At minimum the actual argv[0] executable is mandatory.
    files: dict[str, str] = Field(min_length=1)
    task_versions: dict[str, str] = Field(min_length=1)
    task_digests: dict[str, str] = Field(min_length=1)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_output_bytes: int = Field(default=16 * 1024 * 1024, gt=0)
    score_min: float = Field(allow_inf_nan=False)
    score_max: float = Field(allow_inf_nan=False)
    failure_score: float = Field(allow_inf_nan=False)
    direction: Literal["min", "max"] = "max"
    scientific: bool = True
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_policy(self):
        if not self.score_min <= self.failure_score <= self.score_max:
            raise ValueError("failure_score must be in the declared score range")
        if set(self.task_versions) != set(self.task_digests):
            raise ValueError("task_versions and task_digests must name the same tasks")
        if not Path(self.argv[0]).is_absolute() or self.argv[0] not in self.files:
            raise ValueError("argv[0] must be an absolute, content-pinned executable")
        if any(not item or "\x00" in item for item in self.argv):
            raise ValueError("argv contains empty or NUL arguments")
        for argument in self.argv[1:]:
            if Path(argument).is_file() and str(Path(argument).absolute()) not in self.files:
                raise ValueError("every file argument must be explicitly content-pinned")
        for path, digest in self.files.items():
            if (not Path(path).is_absolute() or len(digest.removeprefix("sha256:")) != 64
                    or any(char not in "0123456789abcdef" for char in digest.removeprefix("sha256:"))):
                raise ValueError("protocol files require absolute paths and SHA-256 digests")
        return self

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class CandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    index: StrictInt = Field(ge=0)
    status: Literal["ok", "scientific_failure"]
    metrics: dict[str, StrictFloat | StrictInt] = Field(default_factory=dict)


class EvaluatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    bundle_digest: str
    protocol_id: str
    protocol_version: str
    protocol_digest: str
    status: Literal["ok", "scientific_failure"]
    score: StrictFloat | StrictInt
    candidate_results: list[CandidateResult]
    details: dict[str, Any] = Field(default_factory=dict)


def load_protocol(path: str | Path) -> EvaluationProtocol:
    return EvaluationProtocol.model_validate(_json(_read(Path(path))))


def _verify_protocol(protocol: EvaluationProtocol) -> None:
    for path, digest in protocol.files.items():
        if hashlib.sha256(_read(Path(path))).hexdigest() != digest.removeprefix("sha256:"):
            raise EvaluationError(f"organizer evaluator asset changed: {path}")


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return True


def _validate_result(payload: Any, manifest: dict, protocol: EvaluationProtocol) -> dict:
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise EvaluationError("result requires an integer schema_version")
    result = EvaluatorResult.model_validate(payload)
    if (result.bundle_digest != manifest["digest"] or result.protocol_id != protocol.protocol_id
            or result.protocol_version != protocol.version or result.protocol_digest != protocol.digest):
        raise EvaluationError("evaluator returned a different bundle or protocol identity")
    if not _finite_tree(payload) or not protocol.score_min <= result.score <= protocol.score_max:
        raise EvaluationError("evaluator returned nonfinite or out-of-range values")
    if [row.index for row in result.candidate_results] != list(range(manifest["candidate_count"])):
        raise EvaluationError("evaluator candidate membership/order is incomplete or duplicated")
    return result.model_dump(mode="json")


def _stop(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run(protocol: EvaluationProtocol, bundle: Path, output: Path) -> tuple[int, str | None]:
    result_path = output / "result.json"
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(output),
                   **protocol.environment, "BINDOCRACY_EVALUATION_PROTOCOL": str(output / "protocol.json"),
                   "BINDOCRACY_EVALUATION_PROTOCOL_DIGEST": protocol.digest,
                   "PYTHONDONTWRITEBYTECODE": "1"}
    argv = [*protocol.argv, "--bundle", str(bundle), "--result", str(result_path)]
    started = time.monotonic()
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, cwd=output, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        reason = None
        try:
            while True:
                returncode = process.poll()
                size = sum(path.stat().st_size for path in (output / "stdout.log", output / "stderr.log", result_path) if path.exists())
                if size > protocol.max_output_bytes:
                    reason = "evaluator output exceeded byte limit"
                elif time.monotonic() - started >= protocol.timeout_seconds:
                    reason = "evaluator exceeded walltime limit"
                if reason or returncode is not None:
                    break
                time.sleep(min(0.05, protocol.timeout_seconds / 10))
        finally:
            # Also kill descendants after the leader exits (even successfully).
            _stop(process)
    return process.returncode, reason


def _agent(record: AttemptRecord) -> str:
    name = record.agent.get("name") or record.agent.get("label") or record.agent.get("id")
    if not isinstance(name, str) or not name:
        raise EvaluationError(f"attempt {record.attempt_id} lacks an explicit agent label")
    return name


def _record(value: AttemptRecord | str | Path | dict) -> AttemptRecord:
    if isinstance(value, AttemptRecord):
        return value
    if isinstance(value, dict):
        return AttemptRecord.model_validate(value)
    path = Path(value)
    return AttemptRecord.model_validate_json(_read(path / "attempt.json" if path.is_dir() else path))


def evaluate_submission(attempt_dir: str | Path, protocol: EvaluationProtocol | str | Path,
                        output_dir: str | Path) -> dict:
    """Evaluate only the latest accepted archive, never an arbitrary candidate path.

    Emits a durable report for missing/invalid submissions and grader failures as
    well as valid science. It never substitutes fabricated scientific metrics.
    """
    if not isinstance(protocol, EvaluationProtocol):
        protocol = load_protocol(protocol)
    directory, output = Path(attempt_dir).absolute(), Path(output_dir).absolute()
    record = _record(directory)
    if any(Path(path).resolve().is_relative_to(Path(record.workspace).resolve()) for path in protocol.files):
        raise EvaluationError("grader assets must not reside in the agent workspace")
    if record.state in {"prepared", "running"}:
        raise EvaluationError("evaluation requires a stopped attempt")
    task = _json(_read(directory / "task.json"))
    if (protocol.task_versions.get(record.task_id) != task["version"]
            or protocol.task_digests.get(record.task_id) != record.task_digest):
        raise EvaluationError("task version/digest is outside organizer protocol")
    if output.exists() or output.resolve().is_relative_to(Path(record.workspace).resolve()):
        raise EvaluationError("evaluation output must be new and outside agent workspace")
    output.mkdir(parents=True)
    _atomic_json(output / "protocol.json", protocol.model_dump(mode="json"))
    report = {"schema_version": 1, "kind": "benchmark_evaluation", "attempt_id": record.attempt_id,
              "agent": _agent(record), "repeat": record.agent.get("repeat", record.attempt_id),
              "task_id": record.task_id, "task_version": task["version"], "task_digest": record.task_digest,
              "environment_digest": record.environment_digest, "protocol_id": protocol.protocol_id,
              "protocol_version": protocol.version, "protocol_digest": protocol.digest,
              "scientific": protocol.scientific, "attempt_state": record.state, "usage": record.usage,
              "accepted_submission": record.latest_submission, "score": None, "result": None,
              "status": "missing_submission", "error": None, "created_at": datetime.now(UTC).isoformat()}
    acceptance = record.latest_submission
    if not acceptance:
        if record.state == "infrastructure_failed":
            report["status"] = "infrastructure_failure"
        elif record.state in {"agent_failed", "timed_out", "budget_exhausted", "cancelled"}:
            report["status"] = "agent_failure"
    else:
        try:
            digest = acceptance["digest"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise SubmissionError("invalid accepted digest")
            bundle = directory / "accepted" / digest
            if Path(acceptance["path"]).absolute() != bundle or acceptance.get("attempt_id") != record.attempt_id or acceptance.get("task_digest") != record.task_digest:
                raise SubmissionError("accepted archive identity/path differs from attempt")
            moment = datetime.fromisoformat(acceptance["accepted_at"])
            if record.deadline is None or moment >= record.deadline:
                raise SubmissionError("acceptance timestamp is not before cutoff")
            manifest = check_submission(bundle)
            if manifest["digest"] != digest or manifest["attempt_id"] != record.attempt_id or manifest["task_digest"] != record.task_digest:
                raise SubmissionError("accepted bundle identity mismatch")
        except (OSError, ValueError, KeyError, TypeError) as error:
            report.update(status="invalid_submission", error=str(error))
        else:
            try:
                _verify_protocol(protocol)
                returncode, reason = _run(protocol, bundle, output)
                report["returncode"] = returncode
                _verify_protocol(protocol)
                if check_submission(bundle)["digest"] != digest:
                    raise EvaluationError("accepted bundle changed during evaluation")
                if reason or returncode:
                    raise EvaluationError(reason or f"evaluator exited {returncode}")
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                report.update(status="infrastructure_failure", error=str(error))
            else:
                try:
                    payload = _json(_read(output / "result.json"))
                    result = _validate_result(payload, manifest, protocol)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    report.update(status="invalid_evaluator_output", error=str(error))
                else:
                    report.update(status="completed" if result["status"] == "ok" else "scientific_failure",
                                  score=result["score"], result=result)
    report["finished_at"] = datetime.now(UTC).isoformat()
    report["digest"] = _digest(report)
    _atomic_json(output / "report.json", report)
    return report


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _interval(samples: list[float]) -> list[float]:
    samples.sort()
    return [samples[int((len(samples) - 1) * 0.025)], samples[int((len(samples) - 1) * 0.975)]]


def _bootstrap(groups: dict[str, list[float]], iterations: int, rng: random.Random) -> list[float]:
    tasks = sorted(groups)
    samples = []
    for _ in range(iterations):
        selected = rng.choices(tasks, k=len(tasks))
        samples.append(_mean([_mean(rng.choices(groups[task], k=len(groups[task]))) for task in selected]))
    return _interval(samples)


def aggregate_reports(reports: list[dict | str | Path], *,
                      attempts: list[AttemptRecord | dict | str | Path],
                      protocol: EvaluationProtocol | str | Path,
                      bootstrap_samples: int = 2000, seed: int = 0,
                      baseline: str | None = None) -> dict:
    """Use an organizer-declared COMPLETE attempt roster, including absent reports.

    Every failed/missing attempt contributes protocol.failure_score. Means weight
    tasks equally, not whichever task received more attempts. Hierarchical
    bootstrap resamples tasks and attempts within task with a deterministic seed.
    Paired comparisons require identical explicit (task, repeat) slots.
    """
    if not isinstance(protocol, EvaluationProtocol):
        protocol = load_protocol(protocol)
    if bootstrap_samples < 1:
        raise EvaluationError("bootstrap_samples must be positive")
    records = sorted((_record(item) for item in attempts), key=lambda record: record.attempt_id)
    if not records or len({r.attempt_id for r in records}) != len(records):
        raise EvaluationError("attempt roster must be nonempty and unique")
    roster = {record.attempt_id: record for record in records}
    indexed = {}
    for item in reports:
        report = item if isinstance(item, dict) else _json(_read(Path(item)))
        identity = report.get("attempt_id")
        if identity not in roster or identity in indexed:
            raise EvaluationError("report is duplicate or outside declared attempt roster")
        if (report.get("schema_version") != 1 or report.get("kind") != "benchmark_evaluation"
                or report.get("protocol_digest") != protocol.digest
                or report.get("protocol_id") != protocol.protocol_id or report.get("protocol_version") != protocol.version):
            raise EvaluationError("incomparable evaluator protocols")
        if report.get("digest") != _digest({key: value for key, value in report.items() if key != "digest"}):
            raise EvaluationError("evaluation report digest mismatch")
        record = roster[identity]
        if (report.get("task_id") != record.task_id or report.get("task_digest") != record.task_digest
                or report.get("task_version") != protocol.task_versions.get(record.task_id)
                or report.get("agent") != _agent(record) or report.get("environment_digest") != record.environment_digest):
            raise EvaluationError("report task/agent/environment differs from attempt roster")
        indexed[identity] = report
    agents = defaultdict(lambda: defaultdict(list))
    outcomes = defaultdict(Counter)
    entries = []
    pairs = defaultdict(dict)
    usage = defaultdict(list)
    for record in records:
        if protocol.task_digests.get(record.task_id) != record.task_digest:
            raise EvaluationError("incomparable task versions/digests in roster")
        label = _agent(record)
        report = indexed.get(record.attempt_id)
        status = report["status"] if report else "missing_report"
        allowed = {"completed", "scientific_failure", "missing_submission", "agent_failure", "infrastructure_failure", "invalid_submission", "invalid_evaluator_output", "missing_report"}
        if status not in allowed:
            raise EvaluationError("unknown report outcome")
        score = protocol.failure_score
        if report and status in {"completed", "scientific_failure"}:
            score = report.get("score")
            if type(score) not in (int, float) or not math.isfinite(score) or not protocol.score_min <= score <= protocol.score_max:
                raise EvaluationError("report score is nonfinite or outside protocol range")
        agents[label][record.task_id].append(float(score))
        outcomes[label][status] += 1
        repeat = record.agent.get("repeat")
        if repeat is not None:
            if type(repeat) not in (str, int):
                raise EvaluationError("repeat identifiers must be strings or integers")
            key = (record.task_id, repeat)
            if key in pairs[label]:
                raise EvaluationError("duplicate agent/task/repeat roster slot")
            pairs[label][key] = float(score)
        usage[label].append({"attempt_id": record.attempt_id, "reported_usage": record.usage})
        entries.append({"attempt_id": record.attempt_id, "agent": label, "task_id": record.task_id,
                        "repeat": repeat, "status": status, "attempt_state": record.state, "included_score": score,
                        "score_source": "evaluator" if status in {"completed", "scientific_failure"} else "predeclared_failure_score"})
    summaries = {}
    for label in sorted(agents):
        groups = agents[label]
        if set(groups) != set(protocol.task_versions):
            raise EvaluationError("each agent must have roster slots for every protocol task; declare missing attempts explicitly")
        totals = defaultdict(float)
        usage_counts = Counter()
        for item in usage[label]:
            for key, value in item["reported_usage"].items():
                if type(value) in (int, float) and math.isfinite(value):
                    totals[key] += value
                    usage_counts[key] += 1
        summaries[label] = {"task_balanced_mean": _mean([_mean(groups[t]) for t in sorted(groups)]),
                            "confidence_interval_95": _bootstrap(groups, bootstrap_samples, random.Random(f"{seed}:{label}")),
                            "tasks": {task: {"mean": _mean(values), "attempts": len(values)} for task, values in sorted(groups.items())},
                            "attempts": sum(outcomes[label].values()), "completed": outcomes[label]["completed"],
                            "scientific_failures": outcomes[label]["scientific_failure"], "outcomes": dict(outcomes[label]),
                            "reported_usage": usage[label], "reported_usage_totals": dict(totals),
                            "usage_reporting_counts": dict(usage_counts),
                            "attempt_states": dict(Counter(item["attempt_state"] for item in entries if item["agent"] == label))}
    comparisons = {}
    if baseline is not None:
        if baseline not in summaries:
            raise EvaluationError("baseline label is absent")
        for label in sorted(summaries):
            if label == baseline:
                continue
            if (len(pairs[label]) != sum(outcomes[label].values()) or len(pairs[baseline]) != sum(outcomes[baseline].values())
                    or pairs[label].keys() != pairs[baseline].keys()):
                raise EvaluationError("paired comparison requires complete matching task/repeat slots; no cherry-picking")
            differences = defaultdict(list)
            for task, repeat in sorted(pairs[label], key=lambda item: (item[0], str(item[1]))):
                differences[task].append(pairs[label][(task, repeat)] - pairs[baseline][(task, repeat)])
            comparisons[label] = {"baseline": baseline, "difference": "agent_minus_baseline",
                                  "task_balanced_mean_difference": _mean([_mean(v) for v in differences.values()]),
                                  "confidence_interval_95": _bootstrap(differences, bootstrap_samples, random.Random(f"{seed}:paired:{label}"))}
    return {"schema_version": 1, "protocol_digest": protocol.digest, "protocol_id": protocol.protocol_id,
            "protocol_version": protocol.version, "scientific": protocol.scientific, "direction": protocol.direction,
            "failure_score": protocol.failure_score, "bootstrap": {"method": "hierarchical_task_then_attempt_percentile", "samples": bootstrap_samples, "seed": seed},
            "agents": summaries, "paired_comparisons": comparisons, "attempts": entries}
