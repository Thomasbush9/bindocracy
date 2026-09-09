"""AlphaFold 3 output -> normalized records.

The scorer's adapter with a different tool name. All three scoring plugins
share the `metrics.jsonl` contract, so the seam is the file, not the container.
"""

from __future__ import annotations

from bindocracy.tools.scorer.adapter import METRICS_FILE, ScorerOutputAdapter

__all__ = ["METRICS_FILE", "AF3OutputAdapter"]


class AF3OutputAdapter(ScorerOutputAdapter):
    tool = "af3"
