"""Reading designs and metrics back out of a campaign database.

Reached by `bindocracy filter apply` and `bindocracy designset build`.
See docs/custom-optimization.md for the stage that consumes it.

`CampaignStore` is write-only by design: generation never needs to read, and
keeping it that way made the single-writer rule easy to hold. A scoring stage
breaks that assumption, because its *input* is the database. This module is the
read half, and it is deliberately separate:

- It takes a `duckdb.DuckDBPyConnection`, not a `CampaignStore`, so it can be
  handed a read-only connection and cannot write by accident.
- It returns frozen dataclasses, not `DesignRecord`s. A row read back is not a
  record being written, and giving them the same type invites round-tripping a
  design into a second copy of itself.
- It does no aggregation. Metrics come back per replicate; collapsing them is a
  decision the caller states explicitly (see `bindocracy.filters`).

The queries here are the only place SQL is written against `designs` and
`metrics`. Everything downstream works on the dataclasses.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import duckdb
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from bindocracy.store.records import CandidateType


@dataclass(frozen=True, slots=True)
class DesignRow:
    """One design as a scorer sees it: identity, sequence, and provenance.

    `tool` and `run_name` are the *producing* run's, carried so a scoring run
    can report per-tool breakdowns without a second query, and so the frozen
    design set stays readable by a human opening the FASTA.
    """

    design_id: str
    run_id: str
    run_name: str
    tool: str
    native_id: str
    sequence: str
    length: int
    candidate_type: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One measurement, at one replicate, by one evaluator run."""

    design_id: str
    run_id: str
    name: str
    value: float | None
    replicate: int
    status: str
    direction: str


class DesignQuery(BaseModel):
    """Which designs a scoring or filtering pass should consider.

    Every field narrows; an empty query selects every design in the campaign
    that has a sequence. The defaults are deliberate: `require_sequence` is on
    because a scorer folds sequences, and a backbone-only design has nothing to
    fold. Nothing here filters on quality -- selecting the good ones is what
    the scoring stage exists to do, and pre-filtering on each tool's own
    verdict would reintroduce exactly the incomparability being fixed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tools: tuple[str, ...] = ()
    run_names: tuple[str, ...] = ()
    candidate_types: tuple[CandidateType, ...] = ()
    min_length: int | None = Field(default=None, gt=0)
    max_length: int | None = Field(default=None, gt=0)
    require_sequence: bool = True

    # Creation time of the DESIGN, not its producing/evaluating run. Require an
    # explicit timezone so selection does not depend on the caller's location.
    # Half-open windows compose without counting the boundary twice.
    created_after: AwareDatetime | None = None
    created_before: AwareDatetime | None = None

    # Skip designs an evaluator has already scored. Named by run, not by tool,
    # because rescoring under a changed protocol is a new run and must not be
    # silently skipped as "already done".
    exclude_scored_by_run: tuple[str, ...] = ()

    # Keep only designs a filter passed. `passed_filter` names the decisions
    # -- a rule name, or the filter set's own name for the gating verdict --
    # and every one of them must have passed.
    #
    # `filter_runs` is REQUIRED alongside it, and that is the interesting rule
    # here. Re-filtering under changed thresholds produces a second filter run
    # *beside* the first rather than correcting it (see `filters/apply.py`), so
    # "passed a rule called confident_interface" is ambiguous the moment a
    # policy has been revised: the union of two contradictory policies reads as
    # a selection somebody made, and nobody made it. Naming the run is what
    # makes a design set reproducible.
    # Run IDs, or names that resolve to exactly one run. A name is NOT
    # unique -- re-filtering under changed thresholds writes a second run with
    # the same config `name` -- so `runs.selection.select` resolves names to
    # IDs and refuses an ambiguous one. Going through that path is what makes
    # the guarantee hold; this predicate accepts either form so a resolved
    # query and a hand-written one both work.
    passed_filter: tuple[str, ...] = ()
    filter_runs: tuple[str, ...] = ()

    # Deduplicate identical sequences across tools. Off by default: the campaign
    # currently has zero cross-tool duplicates, so switching it on would hide
    # nothing and cost a subquery. Worth having the day two tools converge.
    distinct_sequences: bool = False

    limit: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def check_filter_scope(self) -> DesignQuery:
        """A filter selection must say which filter run it trusts.

        Symmetric with `FilterConfig.evaluator_runs`, which is required for the
        same reason: a policy applied to "whatever was in the database" cannot
        be reproduced, and neither can a selection over "whatever policy was
        applied".
        """
        if self.passed_filter and not self.filter_runs:
            raise ValueError(
                "passed_filter requires filter_runs: a rule name alone is ambiguous "
                "once a policy has been revised, because re-filtering writes a "
                "second filter run beside the first rather than replacing it"
            )
        if self.filter_runs and not self.passed_filter:
            raise ValueError(
                "filter_runs without passed_filter narrows nothing; name the "
                "decision(s) a design must have passed"
            )
        return self

    @model_validator(mode="after")
    def check_length_window(self) -> DesignQuery:
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.max_length < self.min_length
        ):
            raise ValueError("max_length cannot be below min_length")
        return self

    @model_validator(mode="after")
    def check_creation_window(self) -> DesignQuery:
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after >= self.created_before
        ):
            raise ValueError("created_before must be later than created_after")
        return self


# Designs come back sorted by length, then by design_id. The ordering is not
# cosmetic: every JAX structure model recompiles per binder length, and the
# campaign holds 78 distinct lengths, so a shard cut from a length-sorted list
# spans a handful of compilations instead of all of them. `design_id` breaks
# ties so the order is total and the digest of a design set is reproducible.
_ORDER_BY = "ORDER BY d.length, d.design_id"

_SELECT = """
SELECT d.design_id, d.run_id, r.name, r.tool, d.native_id,
       d.sequence, d.length, d.candidate_type, d.metadata
FROM designs d
JOIN runs r ON r.run_id = d.run_id
"""


def select_designs(
    connection: duckdb.DuckDBPyConnection, query: DesignQuery | None = None
) -> tuple[DesignRow, ...]:
    """Every design matching `query`, ordered by length then ID.

    The order is part of the contract: `bindocracy.runs.designset` hashes this
    sequence of rows, so an unchanged database and an unchanged query must
    produce a byte-identical design set.
    """
    query = query or DesignQuery()
    where, params = _predicates(query)
    sql = _SELECT + (f"WHERE {' AND '.join(where)}\n" if where else "") + _ORDER_BY
    if query.limit is not None:
        sql += f"\nLIMIT {int(query.limit)}"

    rows = connection.execute(sql, params).fetchall()
    designs = tuple(
        DesignRow(
            design_id=row[0],
            run_id=row[1],
            run_name=row[2],
            tool=row[3],
            native_id=row[4],
            sequence=row[5],
            length=row[6],
            candidate_type=row[7],
            metadata=_json_or_empty(row[8]),
        )
        for row in rows
    )
    return _first_per_sequence(designs) if query.distinct_sequences else designs


def select_metrics(
    connection: duckdb.DuckDBPyConnection,
    *,
    design_ids: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
    run_ids: Sequence[str] | None = None,
) -> tuple[MetricRow, ...]:
    """Measurements for the given designs, one row per replicate.

    Nothing is collapsed. A six-sample scoring run returns six rows per metric
    per design, and which of them a threshold applies to is stated by the
    filter set rather than assumed here.
    """
    where: list[str] = []
    params: list[Any] = []
    for column, values in (
        ("design_id", design_ids),
        ("name", names),
        ("run_id", run_ids),
    ):
        if values is not None:
            if not values:
                return ()
            where.append(f"{column} IN ({_placeholders(values)})")
            params.extend(values)

    sql = (
        "SELECT design_id, run_id, name, value, replicate, status, direction FROM metrics"
        + (f" WHERE {' AND '.join(where)}" if where else "")
        + " ORDER BY design_id, name, replicate"
    )
    return tuple(
        MetricRow(
            design_id=row[0],
            run_id=row[1],
            name=row[2],
            value=row[3],
            replicate=row[4],
            status=row[5],
            direction=row[6],
        )
        for row in connection.execute(sql, params).fetchall()
    )


def metric_names(connection: duckdb.DuckDBPyConnection) -> tuple[str, ...]:
    """Every metric name present, for building and validating a filter set."""
    return tuple(
        row[0]
        for row in connection.execute("SELECT DISTINCT name FROM metrics ORDER BY 1").fetchall()
    )


def count_designs(
    connection: duckdb.DuckDBPyConnection, query: DesignQuery | None = None
) -> int:
    """How many designs a query would return, without materialising them."""
    query = query or DesignQuery()
    where, params = _predicates(query)
    sql = "SELECT count(*) FROM designs d JOIN runs r ON r.run_id = d.run_id" + (
        f" WHERE {' AND '.join(where)}" if where else ""
    )
    return int(connection.execute(sql, params).fetchone()[0])


def _predicates(query: DesignQuery) -> tuple[list[str], list[Any]]:
    where: list[str] = []
    params: list[Any] = []

    if query.require_sequence:
        where.append("d.sequence IS NOT NULL")
    if query.tools:
        where.append(f"r.tool IN ({_placeholders(query.tools)})")
        params.extend(query.tools)
    if query.run_names:
        where.append(f"r.name IN ({_placeholders(query.run_names)})")
        params.extend(query.run_names)
    if query.candidate_types:
        types = [str(value) for value in query.candidate_types]
        where.append(f"d.candidate_type IN ({_placeholders(types)})")
        params.extend(types)
    if query.min_length is not None:
        where.append("d.length >= ?")
        params.append(query.min_length)
    if query.max_length is not None:
        where.append("d.length <= ?")
        params.append(query.max_length)
    if query.created_after is not None:
        where.append("d.created_at >= ?")
        params.append(query.created_after)
    if query.created_before is not None:
        where.append("d.created_at < ?")
        params.append(query.created_before)
    if query.passed_filter:
        # One EXISTS per named decision, so a design must have passed all of
        # them rather than any. A design with no decision row at all fails,
        # which is `filters/models.py` rule 2 -- a missing verdict is not a
        # pass -- applied at selection time.
        for name in query.passed_filter:
            where.append(
                "EXISTS (SELECT 1 FROM decisions x JOIN runs xr ON xr.run_id = x.run_id "
                "WHERE x.design_id = d.design_id AND x.kind = 'filter' "
                "AND x.passed AND x.name = ? AND ("
                f"xr.run_id IN ({_placeholders(query.filter_runs)}) OR "
                f"xr.name IN ({_placeholders(query.filter_runs)})))"
            )
            params.append(name)
            params.extend(query.filter_runs)
            params.extend(query.filter_runs)

    if query.exclude_scored_by_run:
        where.append(
            "d.design_id NOT IN "
            f"(SELECT design_id FROM metrics WHERE run_id IN "
            f"({_placeholders(query.exclude_scored_by_run)}))"
        )
        params.extend(query.exclude_scored_by_run)

    return where, params


def _placeholders(values: Sequence[Any]) -> str:
    return ", ".join("?" for _ in values)


def _first_per_sequence(designs: tuple[DesignRow, ...]) -> tuple[DesignRow, ...]:
    """Keep the first design of each distinct sequence, in the given order.

    "First" is well defined only because `select_designs` imposes a total
    order, which is why deduplication happens here rather than in SQL.
    """
    seen: set[str] = set()
    kept: list[DesignRow] = []
    for design in designs:
        if design.sequence in seen:
            continue
        seen.add(design.sequence)
        kept.append(design)
    return tuple(kept)


def _json_or_empty(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}
