"""Chai-1 output -> normalized records.

Subclasses the scorer's adapter rather than copying it. The two tools share the
part that matters -- a `metrics.jsonl` of `{index, metrics, replicate, ...}`
rows joined to design IDs through the design-set manifest -- and differ only in
the container that produced it. Everything the adapter reads comes from the
manifest (`metric_prefix`, `design_set_manifest`), so the tool name is the only
override needed.

Like the scorer, this emits no designs: a scoring run measures candidates that
already exist.
"""

from __future__ import annotations

from bindocracy.tools.scorer.adapter import METRICS_FILE, ScorerOutputAdapter

__all__ = ["METRICS_FILE", "Chai1OutputAdapter"]


class Chai1OutputAdapter(ScorerOutputAdapter):
    tool = "chai1"
