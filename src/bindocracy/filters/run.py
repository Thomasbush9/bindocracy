"""Applying a stored filter policy to a frozen design set, end to end.

`apply.py` decides verdicts and `config.py` packages them; this is the part
that reads the database, checks the policy against what is actually in it, and
writes the result back through the ordinary ingest path. It was the missing
link that kept `filters/` marked DRAFT.

One check here earns its place: **a filter set is validated against the metric
names the named evaluator runs actually produced, before anything is applied.**
A threshold on a metric no row exists for fails every design, and a filter that
rejects everything looks like a strict policy rather than a typo. The campaign
has 35 registered metric names and stores them prefixed, so `iptm` instead of
`boltz2_iptm` is a plausible mistake with an implausible-looking result.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.filters.config import FilterConfig, build_filter_run, filter_config_record
from bindocracy.runs.designset import DesignSet
from bindocracy.runs.selection import SelectionError, read_only, resolve_run_ids
from bindocracy.store.query import MetricRow, select_metrics
from bindocracy.store.records import CollectedRun, ConfigRecord, utc_now


class FilterRunError(RuntimeError):
    """A filter run cannot be applied as configured."""


def metrics_for_filter(
    database: str | Path, config: FilterConfig, design_set: DesignSet
) -> tuple[MetricRow, ...]:
    """Every measurement the policy needs, from the runs it named.

    Scoped three ways at once -- to the design set, to the evaluator runs, and
    to the metric names the policy reads -- because each scope is a different
    mistake. Unscoped designs would let a verdict depend on rows outside the
    frozen set; unscoped runs would let a newer scoring run silently change
    every verdict without changing the config; unscoped names would pull the
    whole metrics table into memory to use a handful of columns.
    """
    design_ids = [entry.design_id for entry in design_set.entries]
    names = list(config.filter_set.metrics_used())

    with read_only(database) as connection:
        run_ids = resolve_run_ids(connection, config.evaluator_runs, kind="evaluate")
        rows = select_metrics(
            connection, design_ids=design_ids, names=names, run_ids=list(run_ids)
        )
        available = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT name FROM metrics WHERE run_id IN "
                f"({', '.join('?' for _ in run_ids)})",
                list(run_ids),
            ).fetchall()
        }

    unknown = sorted(set(names) - available)
    if unknown:
        close = sorted(name for name in available if unknown[0].split("_")[-1] in name)
        hint = f"; did you mean one of {close[:4]}" if close else ""
        raise FilterRunError(
            f"filter set {config.filter_set.name!r} tests metric {unknown[0]!r}, which "
            f"the evaluator run(s) {list(config.evaluator_runs)} never produced{hint}.\n"
            "Refusing rather than applying it: a threshold on an absent metric fails "
            "every design, which reads as a strict filter rather than a typo."
        )
    return rows


def run_filter(
    *,
    database: str | Path,
    general: GeneralConfig,
    config: FilterConfig,
    output_dir: str | Path,
    general_source: Path | None = None,
    run_name: str | None = None,
    started_at: datetime | None = None,
) -> tuple[CollectedRun, ConfigRecord]:
    """Apply one filter policy and return the bundle, unwritten.

    Returning rather than ingesting keeps this testable without a database
    writer and lets the CLI decide whether to ingest, which is the same split
    `collect` and `ingest` already have.
    """
    manifest_path = Path(config.design_set)
    if not manifest_path.is_file():
        raise FilterRunError(f"no design-set manifest at {manifest_path}")
    design_set = DesignSet.read(manifest_path)

    try:
        metrics = metrics_for_filter(database, config, design_set)
    except SelectionError as error:
        raise FilterRunError(str(error)) from error

    record = filter_config_record(
        general=general, config=config, general_source=general_source
    )
    collected = build_filter_run(
        config=config,
        general=general,
        design_set=design_set,
        metrics=metrics,
        model_config_id=record.model_config_id,
        run_name=run_name or config.name,
        output_uri=str(Path(output_dir).resolve()),
        started_at=started_at or utc_now(),
    )
    return collected, record
