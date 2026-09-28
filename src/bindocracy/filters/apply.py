"""Applying a filter set to scored designs.

Reached by `bindocracy filter apply` and `bindocracy designset build`.
See docs/custom-optimization.md for the stage that consumes it.

Pure functions over records: nothing here opens a database, launches anything,
or knows what a run is. That keeps the interesting part -- how a threshold
becomes a verdict -- testable with a list of floats, and it is the same
separation the output adapters already keep.

A filter pass produces `DecisionRecord`s and nothing else. No design is
deleted, no metric is rewritten, and a design that fails is a row saying so
with the numbers that failed it. Re-filtering under different thresholds is a
second filter run beside the first, not a correction of it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bindocracy.filters.models import OPERATORS, Aggregation, FilterRule, FilterSet, Threshold
from bindocracy.store.query import MetricRow
from bindocracy.store.records import DecisionKind, DecisionRecord, stable_id

_AGGREGATORS = {
    "mean": statistics.fmean,
    "median": statistics.median,
    "min": min,
    "max": max,
    "first": lambda values: values[0],
}


class FilterEvidenceError(ValueError):
    """Measurements cannot be assigned unambiguously to filter replicates."""


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """One comparison, with everything needed to explain it later."""

    metric: str
    aggregate: Aggregation
    op: str
    threshold: float
    observed: float | None
    n_replicates: int
    passed: bool
    missing: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "aggregate": self.aggregate,
            "op": self.op,
            "threshold": self.threshold,
            "observed": self.observed,
            "n_replicates": self.n_replicates,
            "passed": self.passed,
            "missing": self.missing,
        }


@dataclass(frozen=True, slots=True)
class RuleResult:
    rule: str
    passed: bool
    thresholds: tuple[ThresholdResult, ...]

    @property
    def failures(self) -> tuple[ThresholdResult, ...]:
        return tuple(result for result in self.thresholds if not result.passed)


def aggregate_metric(
    values: Sequence[float], aggregate: Aggregation
) -> float:
    """Collapse finite replicates to a finite number, refusing unusable evidence."""
    if not values:
        raise ValueError("cannot aggregate an empty set of replicates")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("cannot aggregate nonfinite replicates")
    result = float(_AGGREGATORS[aggregate](values))
    if not math.isfinite(result):
        raise ValueError("nonfinite aggregate")
    return result


def metric_values(rows: Iterable[MetricRow]) -> dict[str, list[float]]:
    """Group usable measurements by metric name, in replicate order.

    Failed or nonfinite measurements do not count as evidence. Ownership is
    checked before dropping them: two evaluators claiming the same replicate
    cannot be silently pooled, even when only one measurement is usable.
    """
    grouped: dict[str, list[tuple[int, float]]] = {}
    owners: dict[tuple[str, str, int], str] = {}
    for row in rows:
        key = (row.design_id, row.name, row.replicate)
        if key in owners:
            raise FilterEvidenceError(
                f"ambiguous evaluator ownership for design {row.design_id!r}, "
                f"metric {row.name!r}, replicate {row.replicate}: "
                f"runs {sorted((owners[key], row.run_id))}"
            )
        owners[key] = row.run_id
        if row.status != "ok" or row.value is None or not math.isfinite(row.value):
            continue
        grouped.setdefault(row.name, []).append((row.replicate, float(row.value)))
    return {
        name: [value for _, value in sorted(pairs)] for name, pairs in grouped.items()
    }


def evaluate_threshold(
    threshold: Threshold, values: dict[str, list[float]]
) -> ThresholdResult:
    """Test one threshold, treating absent or unusable evidence as failure."""
    replicates = [
        value for value in values.get(threshold.metric, []) if math.isfinite(value)
    ]
    observed = None
    if len(replicates) >= threshold.min_replicates:
        try:
            observed = aggregate_metric(replicates, threshold.aggregate)
        except (OverflowError, ValueError):
            pass
    return ThresholdResult(
        metric=threshold.metric,
        aggregate=threshold.aggregate,
        op=threshold.op,
        threshold=threshold.value,
        observed=observed,
        n_replicates=len(replicates),
        passed=observed is not None and OPERATORS[threshold.op](observed, threshold.value),
        missing=observed is None,
    )


def evaluate_rule(rule: FilterRule, values: dict[str, list[float]]) -> RuleResult:
    results = tuple(evaluate_threshold(threshold, values) for threshold in rule.thresholds)
    passed = (
        any(result.passed for result in results)
        if rule.any_of
        else all(result.passed for result in results)
    )
    return RuleResult(rule=rule.name, passed=passed, thresholds=results)


def evaluate_design(
    filter_set: FilterSet, rows: Iterable[MetricRow]
) -> tuple[RuleResult, ...]:
    """Every rule's verdict for one design."""
    values = metric_values(rows)
    return tuple(evaluate_rule(rule, values) for rule in filter_set.rules)


def decisions_for_design(
    *,
    run_id: str,
    design_id: str,
    filter_set: FilterSet,
    rows: Iterable[MetricRow],
    created_at: datetime,
) -> tuple[DecisionRecord, ...]:
    """One decision per rule, plus one for the set as a whole.

    The per-rule rows are what make a filter debuggable -- "which requirement
    did this design miss" is the question actually asked -- and the set-level
    row is what downstream selection reads. Storing only the second would make
    every rejection look identical.
    """
    results = evaluate_design(filter_set, rows)
    gating = set(filter_set.gating)
    records = [
        DecisionRecord(
            decision_id=stable_id("decision", run_id, design_id, "filter", result.rule),
            run_id=run_id,
            design_id=design_id,
            kind=DecisionKind.FILTER,
            name=result.rule,
            passed=result.passed,
            reason={
                "gating": result.rule in gating,
                "thresholds": [item.as_json() for item in result.thresholds],
            },
            created_at=created_at,
        )
        for result in results
    ]

    overall = all(result.passed for result in results if result.rule in gating)
    records.append(
        DecisionRecord(
            decision_id=stable_id("decision", run_id, design_id, "filter", filter_set.name),
            run_id=run_id,
            design_id=design_id,
            kind=DecisionKind.FILTER,
            name=filter_set.name,
            passed=overall,
            reason={
                "gating_rules": sorted(gating),
                "failed_rules": sorted(
                    result.rule
                    for result in results
                    if result.rule in gating and not result.passed
                ),
            },
            created_at=created_at,
        )
    )
    return tuple(records)


def apply_filter_set(
    *,
    run_id: str,
    filter_set: FilterSet,
    design_ids: Sequence[str],
    metrics: Iterable[MetricRow],
    created_at: datetime,
) -> tuple[tuple[DecisionRecord, ...], dict[str, Any]]:
    """Filter a whole design set, returning decisions and a countable summary.

    `design_ids` drives the loop rather than the metric rows, so a design that
    was never scored still gets an explicit failing verdict instead of being
    absent. Absence and rejection look the same in a query and mean opposite
    things; `known-issues.md` records the run where that cost a day.
    """
    by_design: dict[str, list[MetricRow]] = {design_id: [] for design_id in design_ids}
    for row in metrics:
        if row.design_id in by_design:
            by_design[row.design_id].append(row)

    records: list[DecisionRecord] = []
    passed_by_rule: dict[str, int] = {rule.name: 0 for rule in filter_set.rules}
    unscored = 0
    n_passed = 0

    for design_id in design_ids:
        rows = by_design[design_id]
        if not rows:
            unscored += 1
        design_records = decisions_for_design(
            run_id=run_id,
            design_id=design_id,
            filter_set=filter_set,
            rows=rows,
            created_at=created_at,
        )
        records.extend(design_records)
        # The set-level record is the last one and carries the set's name,
        # which `FilterSet` refuses to let collide with a rule's. Counted
        # positionally rather than by name so the two tallies cannot merge even
        # if that guarantee were ever relaxed.
        *rule_records, overall_record = design_records
        for record in rule_records:
            if record.passed:
                passed_by_rule[record.name] += 1
        if overall_record.passed:
            n_passed += 1

    summary = {
        "filter_set": filter_set.name,
        "n_designs": len(design_ids),
        "n_passed": n_passed,
        "n_unscored": unscored,
        "passed_by_rule": passed_by_rule,
        "metrics_used": list(filter_set.metrics_used()),
    }
    return tuple(records), summary
