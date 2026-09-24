"""Rank a frozen population and ingest ordinary scoped decision records."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bindocracy.config.models import GeneralConfig
from bindocracy.ranking.models import MetricInput, RankingPolicy
from bindocracy.runs.designset import DesignSet, build_design_set, write_design_set
from bindocracy.runs.ingest import ingest_collected
from bindocracy.runs.selection import SelectionError, read_only, resolve_run_ids
from bindocracy.runs.staging import read_collected, write_collected
from bindocracy.store.query import DesignQuery, DesignRow, MetricRow, select_metrics
from bindocracy.store.records import (
    CollectedRun,
    ConfigRecord,
    DecisionRecord,
    RunRecord,
    canonical_json,
    sha256_text,
    stable_id,
    utc_now,
)


class RankingError(ValueError):
    """The source, recipe, or requested cohort cannot be ranked faithfully."""


def _source(database: str | Path, path: Path) -> DesignSet:
    source = DesignSet.read(path)
    expected = sha256_text(
        canonical_json([[entry.design_id, entry.sequence] for entry in source.entries])
    )
    ids = [entry.design_id for entry in source.entries]
    if (
        source.digest != expected
        or source.scope_id != stable_id("design-set", expected)
        or source.n_designs != len(ids)
        or len(set(ids)) != len(ids)
    ):
        raise RankingError("source DesignSet has inconsistent identity or membership")
    if any(entry.index != index for index, entry in enumerate(source.entries)):
        raise RankingError("source DesignSet indices do not match its frozen order")
    if not ids:
        return source
    with read_only(database) as connection:
        rows = connection.execute(
            "SELECT d.design_id, d.sequence, r.tool, r.name, d.native_id "
            "FROM designs d JOIN runs r ON d.run_id = r.run_id "
            f"WHERE d.design_id IN ({', '.join('?' for _ in ids)})",
            ids,
        ).fetchall()
    actual = {row[0]: row[1:] for row in rows}
    for entry in source.entries:
        if actual.get(entry.design_id) != (
            entry.sequence,
            entry.tool,
            entry.run_name,
            entry.native_id,
        ) or entry.length != len(entry.sequence):
            raise RankingError(f"source design {entry.design_id!r} differs from the database")
    return source


def _input_score(spec: MetricInput, rows: list[MetricRow]) -> tuple[float | None, dict[str, Any]]:
    replicas: dict[int, list[MetricRow]] = defaultdict(list)
    for row in rows:
        if row.replicate in spec.replicas:
            replicas[row.replicate].append(row)
    values: list[float] = []
    problems: dict[str, str] = {}
    for replica in sorted(spec.replicas):
        found = replicas[replica]
        if len(found) > 1:
            return None, {
                "excluded": "ambiguous_evaluator",
                "replica": replica,
                "run_ids": sorted(row.run_id for row in found),
            }
        if not found:
            problems[str(replica)] = "missing"
        elif found[0].status != "ok":
            problems[str(replica)] = found[0].status
        elif found[0].value is None or not math.isfinite(found[0].value):
            problems[str(replica)] = "nonfinite_or_missing"
        else:
            values.append(found[0].value)
    required = spec.min_coverage if spec.min_coverage is not None else len(spec.replicas)
    audit: dict[str, Any] = {"coverage": len(values), "required": required, "replicas": problems}
    if len(values) < required:
        return None, {**audit, "excluded": "incomplete_metrics"}
    aggregate = {"mean": statistics.fmean, "median": statistics.median, "min": min, "max": max}[
        spec.aggregate
    ]
    try:
        score = float(aggregate(values))
    except (OverflowError, ValueError):
        score = math.nan
    if not math.isfinite(score):
        return None, {**audit, "excluded": "nonfinite_aggregate"}
    return score, {**audit, "value": score}


def run_ranking(
    *,
    database: str | Path,
    general: GeneralConfig,
    policy: RankingPolicy,
    output_dir: str | Path,
    general_source: Path | None = None,
) -> tuple[CollectedRun, ConfigRecord]:
    """Compute a rank run without writing; metrics, source and policy define identity.

    Duplicates are collapsed within each ranking scope, keeping its best-ranked
    representative. Head takes priority over tail on a truncated shortage.
    Failed measurements never become tail candidates. No implicit pass gating.
    """
    source = _source(database, policy.design_set)
    try:
        with read_only(database) as connection:
            runs = tuple(sorted(set(resolve_run_ids(connection, policy.evaluator_runs))))
            inputs: dict[str, MetricInput] = {}
            for name, spec in policy.inputs.items():
                selected = (
                    tuple(sorted(set(resolve_run_ids(connection, spec.evaluator_runs))))
                    if spec.evaluator_runs
                    else runs
                )
                if not set(selected) <= set(runs):
                    raise RankingError(f"input {name!r} names runs outside evaluator_runs")
                inputs[name] = spec.model_copy(update={"evaluator_runs": selected})
            metrics = select_metrics(
                connection,
                design_ids=[entry.design_id for entry in source.entries],
                names=sorted({spec.metric for spec in inputs.values()}),
                run_ids=runs,
            )
    except SelectionError as error:
        raise RankingError(str(error)) from error
    resolved = policy.model_copy(update={"evaluator_runs": runs, "inputs": inputs})
    indexed: dict[tuple[str, str], list[MetricRow]] = defaultdict(list)
    used_metrics = []
    for row in metrics:
        if any(
            row.name == spec.metric
            and row.run_id in spec.evaluator_runs
            and row.replicate in spec.replicas
            for spec in inputs.values()
        ):
            indexed[row.design_id, row.name].append(row)
            item = asdict(row)
            if row.value is not None and not math.isfinite(row.value):
                item["value"] = str(row.value)
            used_metrics.append(item)
    data_digest = sha256_text(canonical_json(sorted(used_metrics, key=canonical_json)))
    source_payload = source.model_dump(mode="json", exclude={"created_at", "database"})
    model_payload = resolved.model_dump(mode="json", exclude={"design_set"})
    model_payload["source"] = source_payload
    record = ConfigRecord(
        general_name=general.campaign.name,
        general_schema_version=general.schema_version,
        general_config_json=general.model_dump(mode="json"),
        general_source_uri=str(general_source) if general_source else None,
        model_name=policy.name,
        tool="rank",
        model_schema_version=1,
        model_config_json=model_payload,
    )
    digest = sha256_text(
        canonical_json(
            {
                "config": record.model_config_id,
                "metrics": data_digest,
            }
        )
    )
    run_id = stable_id("rank-run", digest)
    started = utc_now()
    groups: dict[str, list[Any]] = defaultdict(list)
    for entry in source.entries:
        groups[entry.tool if policy.group_by == "generator" else "global"].append(entry)
    if not source.entries and policy.shortage == "error":
        raise RankingError("source is empty; no requested cohort can be filled")
    provenance = {
        "digest": digest,
        "source_digest": source.digest,
        "source_scope_id": source.scope_id,
        "metrics_digest": data_digest,
        "policy_config_id": record.model_config_id,
        "evaluator_runs": list(runs),
    }
    decisions = []
    counts = {cohort.name: 0 for cohort in policy.cohorts}
    counts[policy.union_name] = 0
    group_summary = {}
    excluded_counts: dict[str, int] = defaultdict(int)
    for group, entries in sorted(groups.items()):
        recipe = policy.by_generator.get(group, policy.priorities)
        if not recipe:
            raise RankingError(f"no metric recipe for generator {group!r}")
        scope_id = stable_id("rank-scope", digest, group)
        audits = {}
        ranked = []
        for entry in entries:
            scores = {}
            audit: dict[str, Any] = {"inputs": {}}
            for name in sorted({name for priority in recipe for name in priority.inputs}):
                spec = inputs[name]
                score, detail = _input_score(
                    spec,
                    [
                        row
                        for row in indexed[entry.design_id, spec.metric]
                        if row.run_id in spec.evaluator_runs
                    ],
                )
                audit["inputs"][name] = detail
                if score is None:
                    audit["excluded"] = "incomplete_metrics"
                else:
                    scores[name] = score
            if "excluded" not in audit:
                values = [min(scores[name] for name in priority.inputs) for priority in recipe]
                audit["scores"] = values
                key = tuple(
                    value if priority.direction == "min" else -value
                    for value, priority in zip(values, recipe, strict=True)
                )
                ranked.append((key, entry.design_id, entry.sequence))
            audits[entry.design_id] = audit
        ranked.sort()
        representatives = {}
        unique = []
        for key, design_id, sequence in ranked:
            if policy.deduplicate_sequences and sequence in representatives:
                audits[design_id].update(
                    excluded="duplicate_sequence", representative=representatives[sequence]
                )
            else:
                representatives[sequence] = design_id
                unique.append(design_id)
        requested = sum(cohort.count for cohort in policy.cohorts)
        if len(unique) < requested and policy.shortage == "error":
            raise RankingError(
                f"scope {group!r} has {len(unique)} eligible unique designs; {requested} requested"
            )
        remaining = list(unique)
        members = {}
        for cohort in sorted(policy.cohorts, key=lambda cohort: cohort.end != "head"):
            chosen = (
                remaining[: cohort.count] if cohort.end == "head" else remaining[-cohort.count :]
            )
            members[cohort.name] = set(chosen)
            chosen_ids = set(chosen)
            remaining = [design_id for design_id in remaining if design_id not in chosen_ids]
        members[policy.union_name] = set().union(*members.values())
        ranks = {design_id: index for index, design_id in enumerate(unique, 1)}
        for entry in sorted(entries, key=lambda entry: entry.design_id):
            design_id = entry.design_id
            reason = {**provenance, "group": group, **audits[design_id]}
            if design_id in ranks:
                decisions.append(
                    DecisionRecord(
                        decision_id=stable_id("rank-decision", run_id, design_id),
                        run_id=run_id,
                        design_id=design_id,
                        kind="rank",
                        name=policy.name,
                        rank=ranks[design_id],
                        scope_id=scope_id,
                        reason=reason,
                        created_at=started,
                    )
                )
            else:
                excluded_counts[reason["excluded"]] += 1
            for name, selected in members.items():
                passed = design_id in selected
                decisions.append(
                    DecisionRecord(
                        decision_id=stable_id("rank-membership", run_id, design_id, name),
                        run_id=run_id,
                        design_id=design_id,
                        kind="filter",
                        name=name,
                        passed=passed,
                        scope_id=scope_id,
                        reason={**reason, "cohort": name, "rank": ranks.get(design_id)},
                        created_at=started,
                    )
                )
                counts[name] += int(passed)
        group_summary[group] = {
            "source": len(entries),
            "eligible": len(unique),
            "shortage": max(0, requested - len(unique)),
            "cohorts": {name: len(ids) for name, ids in members.items()},
        }
    collected = CollectedRun(
        run=RunRecord(
            run_id=run_id,
            name=policy.name,
            tool="rank",
            kind="rank",
            model_config_id=record.model_config_id,
            status="succeeded",
            n_requested=source.n_designs,
            n_attempted=source.n_designs,
            n_produced=sum(group["eligible"] for group in group_summary.values()),
            n_passed=counts[policy.union_name],
            count_details={
                **provenance,
                "cohorts": counts,
                "groups": group_summary,
                "excluded": dict(excluded_counts),
                "empty_source": not source.entries,
            },
            output_uri=str(Path(output_dir).resolve() / digest.removeprefix("sha256:")),
            created_at=started,
            started_at=started,
            finished_at=utc_now(),
        ),
        decisions=tuple(decisions),
    )
    return collected, record


def summarise(collected: CollectedRun) -> str:
    run = collected.run
    details = run.count_details or {}
    lines = [
        f"rank {run.name}",
        f"run: {run.run_id}",
        f"source: {run.n_requested}; eligible: {run.n_produced}; selected: {run.n_passed}",
    ]
    for group, counts in details.get("groups", {}).items():
        lines.append(
            f"  {group}: eligible={counts['eligible']}, shortage={counts['shortage']}, cohorts={counts['cohorts']}"
        )
    lines.append(f"excluded: {details.get('excluded', {})}")
    if not run.n_passed:
        lines.append("No designs selected; exported cohorts are empty (no fallback population).")
    return "\n".join(lines)


def apply_ranking(
    *,
    database: str | Path,
    general: GeneralConfig,
    policy: RankingPolicy,
    output_dir: str | Path,
    general_source: Path | None = None,
) -> tuple[CollectedRun, bool, dict[str, Path]]:
    """Stage, ingest, and export each cohort plus its union, including empty sets.

    Each run has its own content-addressed directory. Reapplication leaves
    existing staging and cohort manifests untouched and ingestion is a no-op.
    """
    collected, config = run_ranking(
        database=database,
        general=general,
        policy=policy,
        output_dir=output_dir,
        general_source=general_source,
    )
    directory = Path(collected.run.output_uri)
    bundle = directory / "collected.json"
    if bundle.exists():
        if read_collected(bundle).verdict_hash() != collected.verdict_hash():
            raise RankingError(f"staged ranking differs under the same identity: {bundle}")
    else:
        write_collected(collected, bundle)
    config_path = directory / "config.json"
    if not config_path.exists():
        config_path.write_text(config.model_dump_json(indent=2) + "\n")
    inserted = ingest_collected(database, collected, config=config)
    source = DesignSet.read(policy.design_set)
    manifests = {}
    for name in (collected.run.count_details or {})["cohorts"]:
        selected = {
            decision.design_id
            for decision in collected.decisions
            if decision.kind == "filter" and decision.name == name and decision.passed
        }
        entries = sorted(
            (entry for entry in source.entries if entry.design_id in selected),
            key=lambda entry: (entry.length, entry.design_id),
        )
        query = DesignQuery(passed_filter=(name,), filter_runs=(collected.run.run_id,))
        if entries:
            rows = tuple(
                DesignRow(
                    entry.design_id,
                    "",
                    entry.run_name,
                    entry.tool,
                    entry.native_id,
                    entry.sequence,
                    entry.length,
                    "sequence",
                    {},
                )
                for entry in entries
            )
            design_set = build_design_set(rows, database=database, query=query)
        else:
            empty_digest = sha256_text(canonical_json([]))
            design_set = DesignSet(
                digest=empty_digest,
                scope_id=stable_id("design-set", empty_digest),
                database=str(database),
                query=query.model_dump(mode="json"),
                created_at=utc_now(),
                n_designs=0,
                by_tool={},
                length_range=None,
                distinct_lengths=0,
                entries=(),
            )
        _, manifests[name] = write_design_set(design_set, directory / "cohorts" / name)
    summary = directory / "summary.txt"
    if not summary.exists():
        summary.write_text(summarise(collected) + "\n")
    return collected, inserted, manifests
