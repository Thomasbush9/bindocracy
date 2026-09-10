"""Upstream OpenFold3 output -> normalized records.

The scorer's adapter under a fourth tool name. All four scoring plugins share
the `metrics.jsonl` contract, so the seam is the file, not the container.
"""

from __future__ import annotations

from bindocracy.tools.scorer.adapter import METRICS_FILE, ScorerOutputAdapter

__all__ = ["METRICS_FILE", "OF3UpstreamOutputAdapter"]


class OF3UpstreamOutputAdapter(ScorerOutputAdapter):
    tool = "of3_upstream"
