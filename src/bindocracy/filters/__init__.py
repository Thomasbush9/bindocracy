"""Filtering scored designs into a verdict per design, stored and explainable.

Reached by `bindocracy filter apply` and `bindocracy designset build`.
See docs/custom-optimization.md for the stage that consumes it.
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
from bindocracy.filters.config import FilterConfig, build_filter_run
from bindocracy.filters.models import (
    Aggregation,
    Comparison,
    FilterRule,
    FilterSet,
    Threshold,
)
from bindocracy.filters.run import FilterRunError, run_filter

__all__ = [
    "Aggregation",
    "Comparison",
    "FilterConfig",
    "FilterRule",
    "FilterRunError",
    "FilterSet",
    "RuleResult",
    "Threshold",
    "ThresholdResult",
    "aggregate_metric",
    "apply_filter_set",
    "build_filter_run",
    "decisions_for_design",
    "evaluate_design",
    "metric_values",
    "run_filter",
]
