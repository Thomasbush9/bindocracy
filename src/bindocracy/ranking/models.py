"""Explicit, reusable ranking recipes; no native-pass or weighted-score defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from bindocracy.config.models import ConfigModel


class MetricInput(ConfigModel):
    """Aggregate only the named replicas; coverage counts finite successful rows.

    A metric/replica appearing in multiple evaluator runs is ambiguous, not an
    extra replicate. Narrow evaluator_runs on this input to disambiguate it.
    """

    metric: str = Field(min_length=1)
    evaluator_runs: tuple[str, ...] = ()
    replicas: tuple[int, ...] = Field(min_length=1)
    aggregate: Literal["mean", "min", "max", "median"]
    min_coverage: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def coverage_is_possible(self) -> Self:
        if len(set(self.replicas)) != len(self.replicas) or min(self.replicas) < 0:
            raise ValueError("replicas must be unique nonnegative integers")
        if self.min_coverage is not None and self.min_coverage > len(self.replicas):
            raise ValueError("min_coverage exceeds the replica subset")
        return self


class Priority(ConfigModel):
    """One lexicographic priority: a named input, or conservative minimum.

    Multiple inputs are combined only by minimum, after each has independently
    met its coverage requirement. There are no weights or missing-input fallbacks.
    """

    inputs: tuple[str, ...] = Field(min_length=1)
    combine: Literal["min"] = "min"
    direction: Literal["min", "max"]


class Cohort(ConfigModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    end: Literal["head", "tail"]
    count: int = Field(gt=0)


class RankingPolicy(ConfigModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    tool: Literal["rank"] = "rank"
    design_set: Path
    evaluator_runs: tuple[str, ...] = Field(min_length=1)
    group_by: Literal["global", "generator"]
    inputs: dict[str, MetricInput] = Field(min_length=1)
    priorities: tuple[Priority, ...] = ()
    by_generator: dict[str, tuple[Priority, ...]] = Field(default_factory=dict)
    deduplicate_sequences: bool
    cohorts: tuple[Cohort, ...] = Field(min_length=1, max_length=2)
    shortage: Literal["error", "truncate"]
    union_name: str = Field(default="selected", pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")

    @model_validator(mode="after")
    def coherent_recipe(self) -> Self:
        if not self.priorities and not self.by_generator:
            raise ValueError("supply priorities or by_generator recipes")
        if any(not recipe for recipe in self.by_generator.values()):
            raise ValueError("generator recipes must not be empty")
        if self.group_by == "global" and self.by_generator:
            raise ValueError("generator-specific recipes require group_by: generator")
        for recipe in (self.priorities, *self.by_generator.values()):
            for priority in recipe:
                if unknown := set(priority.inputs) - self.inputs.keys():
                    raise ValueError(f"unknown metric inputs: {sorted(unknown)}")
        names = [cohort.name for cohort in self.cohorts]
        if len(set(names)) != len(names) or self.union_name in names:
            raise ValueError("cohort names and union_name must be distinct")
        if len({cohort.end for cohort in self.cohorts}) != len(self.cohorts):
            raise ValueError("at most one head and one tail cohort are allowed")
        return self
