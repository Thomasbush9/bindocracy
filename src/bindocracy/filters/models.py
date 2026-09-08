"""What a filter is allowed to say.

DRAFT -- not wired into anything yet. See docs/scoring-stage.md.

A filter set is a *stored, versioned document*, not numbers in a script. That is
the whole point of this module. Every threshold in the campaign so far lives
inside the tool that applied it, which is why the seven filter verdicts already
in the database cannot be compared with one another and why nobody can say what
`pxdesign_af2ig` required without reading PXDesign's source.

Three rules follow from that, and they are enforced here rather than left to
whoever writes the YAML:

1. **A threshold names its aggregation.** Metrics are stored per replicate, and
   "ipTM above 0.6" is ambiguous across six samples. Mean, median, min and max
   are different filters and a config that does not choose is rejected.
2. **A missing metric is a failure, not a pass.** The alternative silently
   admits every design a scorer failed on, which is the exact shape of the
   silent-success bugs `known-issues.md` is organised around.
3. **A rule records why.** Each verdict carries the observed value beside the
   threshold it was tested against, so a decision is explainable from the
   database alone months later.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# How several replicates of one metric collapse to the single number a
# threshold tests. `first` means replicate 0 and exists for metrics that are
# not sampled at all -- a sequence liability, a length -- where a mean over one
# value would only obscure that there is nothing to average.
Aggregation = Literal["mean", "median", "min", "max", "first"]

Comparison = Literal[">=", ">", "<=", "<", "==", "!="]

OPERATORS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


class FilterModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Threshold(FilterModel):
    """One comparison against one aggregated metric.

    `metric` is the stored metric name, including its scorer prefix
    (`esmfold2_iptm`, not `iptm`). Requiring the prefix is deliberate: an
    unprefixed threshold would silently apply to whichever model happened to
    write that column, which is how a held-out judge stops being held out.
    """

    metric: str = Field(min_length=1)
    op: Comparison
    value: float
    aggregate: Aggregation

    # Replicates whose status is not 'ok' are dropped before aggregating. If
    # that leaves nothing, the threshold fails. Set this to allow a design
    # through on partial evidence, and say why in the config comment.
    min_replicates: int = Field(default=1, ge=1)

    def describe(self) -> str:
        return f"{self.aggregate}({self.metric}) {self.op} {self.value:g}"


class FilterRule(FilterModel):
    """A named verdict: every threshold must hold, unless `any_of` is set.

    One rule becomes one `DecisionRecord` per design, so the name is what shows
    up in the database and should read as a claim about the design --
    `confident_interface`, `contacts_epitope`, `expressible` -- rather than as
    a step number.
    """

    name: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    description: str = ""
    thresholds: tuple[Threshold, ...] = Field(min_length=1)
    any_of: bool = False

    @model_validator(mode="after")
    def check_metrics_distinct(self) -> FilterRule:
        """Two thresholds on the same metric and aggregation contradict or duplicate.

        Either is a config bug worth refusing: the second is dead weight and the
        first is a filter whose meaning depends on evaluation order.
        """
        seen = [(t.metric, t.aggregate) for t in self.thresholds]
        duplicates = {key for key in seen if seen.count(key) > 1}
        if duplicates:
            metric, aggregate = next(iter(duplicates))
            raise ValueError(f"rule {self.name!r} tests {aggregate}({metric}) more than once")
        return self


class FilterSet(FilterModel):
    """A campaign's filtering policy, stored whole with the run that applied it.

    Serialised into the filter run's config, so `decisions` can always be traced
    back to the exact thresholds that produced them. Changing any number here
    changes the config hash and therefore produces a new run rather than
    silently reinterpreting an old one.
    """

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str = ""
    rules: tuple[FilterRule, ...] = Field(min_length=1)

    # A design passes the set when it passes every rule named here. Empty means
    # every rule. Kept explicit so a rule can be recorded for information
    # without gating anything -- an epitope-contact verdict is worth storing on
    # every design whether or not it is allowed to exclude one.
    gating_rules: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check_rule_names(self) -> FilterSet:
        names = [rule.name for rule in self.rules]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate rule name: {min(duplicates)}")
        unknown = set(self.gating_rules) - set(names)
        if unknown:
            raise ValueError(f"gating_rules names no such rule: {min(unknown)}")
        return self

    @property
    def gating(self) -> tuple[str, ...]:
        return self.gating_rules or tuple(rule.name for rule in self.rules)

    def metrics_used(self) -> tuple[str, ...]:
        """Every metric this set reads, for validating against the database.

        A filter set naming a metric no numeric row exists for would fail every
        design and look like a strict filter rather than a typo, so the CLI
        checks this before applying anything.
        """
        return tuple(sorted({t.metric for rule in self.rules for t in rule.thresholds}))
