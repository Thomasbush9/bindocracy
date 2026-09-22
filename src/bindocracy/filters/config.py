"""The filter run: a policy document, applied to a frozen design set.

Reached by `bindocracy filter apply` and `bindocracy designset build`.
See docs/custom-optimization.md for the stage that consumes it.

**Filtering is not a tool, and this module is where that is argued rather than
worked around.** `ToolPlugin` describes something with a container, a task
count, a launch command and a run directory to parse. A filter has none of
those: it reads the database, evaluates arithmetic, and writes verdicts. Forcing
it into the plugin seam would mean inventing a container it does not need and a
task fan-out over work that takes under a second.

`docs/adding-a-tool.md` says to name a missing contract instead of working
around it, so: the missing contract is a **database-to-database run**. Scoring
is a real plugin because it genuinely runs a model on a GPU. Filtering,
clustering and ranking are all this shape instead, and they share one path --
build a `CollectedRun` in process, write a staging bundle, ingest it. The
existing `collect` and `ingest` commands then work unchanged, which is the
whole reason for expressing it this way rather than writing to DuckDB directly.

One thing genuinely does need to change in the library; see `FilterResources`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from bindocracy.config.models import ConfigModel, GeneralConfig
from bindocracy.filters.apply import apply_filter_set
from bindocracy.filters.models import FilterSet
from bindocracy.runs.designset import DesignSet
from bindocracy.store.query import MetricRow
from bindocracy.store.records import (
    CollectedRun,
    ConfigRecord,
    DecisionRecord,
    RunKind,
    RunRecord,
    RunStatus,
    stable_id,
    utc_now,
)


class FilterResources(ConfigModel):
    """What a filter run costs, which is not a SLURM allocation.

    Deliberately not `ResourceConfig`. That type describes a scheduled job --
    GPUs, a partition, a walltime -- because until now every run was one. A
    filter runs in the calling process against an open database and takes under
    a second on this campaign's 3,302 designs, so it has no walltime to declare
    and no GPU to request. Reusing `ResourceConfig` would mean inventing both.

    If filtering ever does become a scheduled job, `ResourceConfig.gpus` is
    `gt=0` and would need widening to `ge=0` first. It does not need widening
    now, and doing it speculatively would put a GPU field on a thing that has
    no GPU.
    """

    cpus: int = Field(default=1, gt=0)
    memory_gb: int = Field(default=4, gt=0)


class FilterConfig(ConfigModel):
    """What a filter run is, stored whole so its verdicts stay explainable.

    The `FilterSet` is embedded rather than referenced by path. A config that
    only names a thresholds file leaves the database recording where the policy
    was, not what it was -- the same argument `ToolPlugin.resolve` already makes
    for every tool that reads an external spec, and the reason editing that file
    must change `model_config_id`.
    """

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    tool: Literal["filter"] = "filter"
    resources: FilterResources = FilterResources()

    # The frozen candidate set this policy was applied to. Required: a filter
    # over "whatever was in the database at the time" cannot be reproduced, and
    # its verdicts cannot be scoped.
    design_set: Path

    # Which evaluator runs supply the metrics. Named explicitly so a filter
    # cannot silently start reading a newer scoring run's numbers, which would
    # change every verdict without changing the config.
    evaluator_runs: tuple[str, ...] = Field(min_length=1)

    filter_set: FilterSet

    # Permit thresholds on a model that already had a say in these designs.
    # Off by default: `run.py::check_independence` explains what it prevents
    # and why "held out" cannot be read off the run graph after the fact.
    allow_self_selection: bool = False


def build_filter_run(
    *,
    config: FilterConfig,
    general: GeneralConfig,
    design_set: DesignSet,
    metrics: tuple[MetricRow, ...],
    model_config_id: str,
    run_name: str,
    output_uri: str,
    started_at: datetime | None = None,
    independence: dict[str, Any] | None = None,
) -> CollectedRun:
    """Apply a filter set and package the result for the ordinary ingest path.

    Emits `DecisionRecord`s only. No designs, because a filter creates none; no
    metrics, because it measures nothing. `n_requested` and `n_produced` are
    both the size of the design set -- every design gets a verdict, including
    the ones no scorer reached -- and `n_passed` is the count that cleared the
    gating rules. That is harness-design §2's three numbers meaning what they
    say for a run that is not generation.
    """
    started = started_at or utc_now()
    design_ids = [entry.design_id for entry in design_set.entries]

    # Content-derived, not `new_id()`. Every other run in this harness is
    # restart-safe because its run directory carries a manifest with a fixed
    # run_id, so re-collecting an already-ingested run is a no-op. A filter has
    # no run directory, so a random id would make a second `filter apply` of
    # the same policy write a second set of identical verdicts under a new run
    # -- and the design set built from "passed worth_optimizing" would then
    # depend on which of the two duplicates a query happened to see.
    #
    # The three inputs that can change a verdict are the policy, the candidate
    # set, and the evaluator runs supplying the numbers. Re-running with any of
    # them changed is a new run beside the old one, which is the semantics
    # `apply.py` already documents; re-running with none of them changed is the
    # same run, and `CampaignStore.ingest` recognises it.
    run_id = stable_id(
        "filter-run",
        model_config_id,
        design_set.digest,
        *sorted(config.evaluator_runs),
    )

    decisions, summary = apply_filter_set(
        run_id=run_id,
        filter_set=config.filter_set,
        design_ids=design_ids,
        metrics=metrics,
        created_at=started,
    )

    run = RunRecord(
        run_id=run_id,
        name=run_name,
        tool="filter",
        kind=RunKind.FILTER,
        model_config_id=model_config_id,
        status=RunStatus.SUCCEEDED,
        n_requested=len(design_ids),
        n_attempted=len(design_ids),
        n_produced=len(design_ids),
        n_passed=summary["n_passed"],
        count_details={
            **summary,
            "design_set_digest": design_set.digest,
            "scope_id": design_set.scope_id,
            "evaluator_runs": list(config.evaluator_runs),
            "campaign": general.campaign.name,
            # Which models had already shaped these designs when the policy was
            # applied, and whether any of them is one it selected on. Stored
            # with the verdicts so the independence of a selection is a fact in
            # the database rather than a claim in a comment.
            "independence": independence or {},
        },
        output_uri=output_uri,
        created_at=started,
        started_at=started,
        finished_at=utc_now(),
    )
    return CollectedRun(run=run, decisions=decisions)


def filter_config_record(
    *, general: GeneralConfig, config: FilterConfig, general_source: Path | None = None
) -> ConfigRecord:
    """The `configs` row a filter run points at.

    `runs.model_config_id` is NOT NULL and foreign-keyed, so a filter run needs
    a config row like any other. Embedding the filter set in `model_config_json`
    is what makes the thresholds recoverable with `bindocracy config export`
    long after the YAML has moved on.
    """
    return ConfigRecord(
        general_name=general.campaign.name,
        general_schema_version=general.schema_version,
        general_config_json=general.model_dump(mode="json"),
        general_source_uri=str(general_source) if general_source else None,
        model_name=config.name,
        tool="filter",
        model_schema_version=config.schema_version,
        model_config_json=config.model_dump(mode="json"),
    )


def gating_decisions(decisions: tuple[DecisionRecord, ...], filter_set: FilterSet) -> tuple[
    DecisionRecord, ...
]:
    """Just the set-level verdicts, for callers that only want pass or fail."""
    return tuple(record for record in decisions if record.name == filter_set.name)


def summarise(run: RunRecord) -> str:
    """A few lines a person can read after a filter pass."""
    details: dict[str, Any] = run.count_details or {}
    lines = [
        f"filter {details.get('filter_set', run.name)}",
        f"  designs   {run.n_requested}",
        f"  passed    {run.n_passed}",
        f"  unscored  {details.get('n_unscored', 0)}",
    ]
    lines.extend(
        f"    {rule:<28s} {count}"
        for rule, count in sorted((details.get("passed_by_rule") or {}).items())
    )
    return "\n".join(lines)
