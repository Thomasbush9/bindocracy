"""Filtering scored designs into a verdict per design, stored and explainable.

DRAFT -- not wired into anything yet. See docs/scoring-stage.md.
"""

from bindocracy.filters.apply import (
    RuleResult,
    ThresholdResult,
    aggregate_metric,
    apply_filter_set,
    decisions_for_design,
    evaluate_design,
    metric_values,
)
from bindocracy.filters.models import (
    Aggregation,
    Comparison,
    FilterRule,
    FilterSet,
    Threshold,
)

__all__ = [
    "Aggregation",
    "Comparison",
    "FilterRule",
    "FilterSet",
    "RuleResult",
    "Threshold",
    "ThresholdResult",
    "aggregate_metric",
    "apply_filter_set",
    "decisions_for_design",
    "evaluate_design",
    "metric_values",
]
