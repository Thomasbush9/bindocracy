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

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from bindocracy.config.models import GeneralConfig
from bindocracy.filters.config import FilterConfig, build_filter_run, filter_config_record
from bindocracy.runs.designset import DesignSet
from bindocracy.runs.selection import SelectionError, read_only, resolve_run_ids
from bindocracy.store.query import MetricRow, select_metrics
from bindocracy.store.records import CollectedRun, ConfigRecord, utc_now
from bindocracy.tools.optimize.preflight import KNOWN_MODELS


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


def models_that_designed(
    database: str | Path, design_set: DesignSet
) -> dict[str, int]:
    """Which models already had a say in these designs, and over how many.

    `designs.metadata.loss_models` is written by the optimizer adapter onto
    every child: the models whose scores were the objective the child was
    produced to improve. A model that shaped a sequence is not an independent
    opinion about it, and asking it afterwards measures its agreement with
    itself.

    Only what is recorded is reported. Generators do not write `loss_models`
    today, so a design straight out of hallucination contributes nothing here
    even though the generator had a loss of its own. That is stated rather than
    guessed at: a table of "which model each tool probably optimizes against"
    would make this check look complete while resting on assumptions nothing in
    the database supports.
    """
    design_ids = [entry.design_id for entry in design_set.entries]
    if not design_ids:
        return {}
    placeholders = ", ".join("?" for _ in design_ids)
    counts: dict[str, int] = {}
    with read_only(database) as connection:
        rows = connection.execute(
            "SELECT json_extract_string(metadata, '$.loss_models') FROM designs "
            f"WHERE design_id IN ({placeholders}) AND metadata IS NOT NULL",
            design_ids,
        ).fetchall()
    for (raw,) in rows:
        if not raw:
            continue
        try:
            models = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(models, list):
            continue
        for model in models:
            if isinstance(model, str):
                counts[model] = counts.get(model, 0) + 1
    return counts


def metric_model(name: str) -> str | None:
    """The model a stored metric name is prefixed with, if it names one.

    Longest match wins, so `protenix_mini_iptm` is protenix_mini rather than
    protenix. A name matching no known model -- a custom function's output, a
    generator's own reported score -- returns None and is not the subject of
    this check.
    """
    candidates = [model for model in KNOWN_MODELS if name.startswith(f"{model}_")]
    return max(candidates, key=len) if candidates else None


def check_independence(
    config: FilterConfig, designed_by: dict[str, int], n_designs: int
) -> dict[str, Any]:
    """Refuse a filter that selects on a model which helped make the designs.

    The failure this prevents is quiet and it has already happened here: a
    smoke filter gated on `esmfold2_iptm` while ESMFold2 was named as the
    campaign's held-out judge, and a filter feeding an AF2-driven optimizer
    could equally have gated on AF2. Neither run would fail, and neither
    number would look wrong; the selection would simply be measuring a model's
    agreement with itself and reporting it as independent evidence.

    Held out is a property of the whole path from design to verdict, not of
    which stage a model runs in. That cannot be inferred from the run graph
    after the fact, so it is checked here, where the selection is made, from
    what the designs themselves record.

    `allow_self_selection` exists because there is one honest reason to do it
    -- re-selecting within an optimizer's own output to rank its children --
    and the config should have to say so out loud.
    """
    overlap = {
        metric: designed_by[model]
        for metric in config.filter_set.metrics_used()
        if (model := metric_model(metric)) is not None and model in designed_by
    }
    audit = {
        "designed_by": dict(sorted(designed_by.items())),
        "n_designs": n_designs,
        "self_selecting_metrics": dict(sorted(overlap.items())),
        "allowed": config.allow_self_selection,
    }
    if overlap and not config.allow_self_selection:
        lines = "\n".join(
            f"    {metric:<32s} shaped {count} of {n_designs} designs"
            for metric, count in sorted(overlap.items())
        )
        raise FilterRunError(
            f"filter set {config.filter_set.name!r} selects on model(s) that already "
            "had a say in these designs, recorded in their own loss_models:\n"
            f"{lines}\n"
            "  A model that shaped a sequence is not an independent opinion about "
            "it, and a filter gated on one measures its agreement with itself while "
            "reading as a second opinion.\n"
            "  Drop those thresholds, or set allow_self_selection: true and say in "
            "the config why ranking within one model's own output is what you meant."
        )
    return audit


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

    try:
        audit = check_independence(
            config, models_that_designed(database, design_set), design_set.n_designs
        )
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
        independence=audit,
    )
    return collected, record
