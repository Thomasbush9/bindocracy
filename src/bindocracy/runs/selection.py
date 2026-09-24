"""Freezing a database query into a design set, and the read side that needs.

The missing half of the scoring stage: `designset.py` can freeze rows into a
content-addressed set and `store/query.py` can select rows, but nothing joined
them, so every design set so far was built by hand. An optimization run makes
that gap load-bearing -- it reads designs a filter chose -- so this is the join.

Read-only on purpose. `CampaignStore` is the single writer and this never opens
it: a selection must not be able to modify the campaign it is selecting from,
and a read-only connection is a cheaper guarantee than a code review.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from bindocracy.runs.designset import DesignSet, build_design_set, write_design_set
from bindocracy.store.query import DesignQuery, DesignRow, select_designs


class SelectionError(RuntimeError):
    """A query selected nothing, or named something the database does not have."""


@contextmanager
def read_only(database: str | Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """A read-only connection to an existing campaign database.

    DuckDB creates a database file when asked to open a missing one, so the
    existence check is not decoration: without it, a typo in a path produces an
    empty database and a query that legitimately returns nothing, which reads
    as "no designs passed the filter".
    """
    path = Path(database)
    if not path.is_file():
        raise SelectionError(f"no campaign database at {path}")
    connection = duckdb.connect(str(path), read_only=True)
    try:
        yield connection
    finally:
        connection.close()


def known_run_names(connection: duckdb.DuckDBPyConnection, kind: str | None = None) -> tuple[
    str, ...
]:
    """Run names present, for checking a config before it is applied."""
    sql = "SELECT DISTINCT name FROM runs"
    params: list[str] = []
    if kind is not None:
        sql += " WHERE kind = ?"
        params.append(kind)
    return tuple(row[0] for row in connection.execute(sql + " ORDER BY 1", params).fetchall())


def resolve_run_ids(
    connection: duckdb.DuckDBPyConnection, names: tuple[str, ...], *, kind: str | None = None
) -> tuple[str, ...]:
    """Run names or IDs to run IDs, refusing anything ambiguous or absent.

    Two refusals, and the second is the one that matters.

    **Absent.** A name that resolves to no run silently contributes no rows.
    For an evaluator list that means a filter reading no metrics and failing
    every design, which looks like a strict policy rather than a typo -- the
    failure mode `filters/models.py` rule 2 is written against.

    **Ambiguous.** A run *name* is not unique. Re-filtering under changed
    thresholds writes a second filter run beside the first, and both carry the
    config's `name`, so `--filter-run my-policy` after a revision would select
    the union of two contradictory policies -- and the union of a strict filter
    and a loose one is the loose one. That is precisely the hazard
    `DesignQuery.check_filter_scope` refuses to leave implicit, so leaving it
    reachable through an ambiguous name would give the guarantee away. The
    run_id is offered in the error because it is the thing that is unique.
    """
    if not names:
        return ()
    placeholders = ", ".join("?" for _ in names)
    sql = (
        f"SELECT run_id, name FROM runs WHERE (name IN ({placeholders}) "
        f"OR run_id IN ({placeholders}))"
    )
    params: list[str] = [*names, *names]
    if kind is not None:
        sql += " AND kind = ?"
        params.append(kind)
    rows = connection.execute(sql + " ORDER BY created_at", params).fetchall()

    qualifier = f" {kind}" if kind else ""
    resolved: list[str] = []
    for wanted in names:
        matches = [row for row in rows if wanted in (row[0], row[1])]
        if not matches:
            available = ", ".join(known_run_names(connection, kind)) or "none"
            raise SelectionError(
                f"no{qualifier} run named {wanted!r} in this database; "
                f"available{qualifier} runs: {available}"
            )
        if len(matches) > 1:
            ids = "\n  ".join(row[0] for row in matches)
            raise SelectionError(
                f"{len(matches)}{qualifier} runs are named {wanted!r}, which is what "
                "re-running a policy with changed thresholds produces. Selecting on "
                "the name would union them, and the union of a strict filter and a "
                "loose one is the loose one. Name the run_id instead (oldest "
                "first):\n  " + ids
            )
        resolved.append(matches[0][0])
    return tuple(resolved)


def select(database: str | Path, query: DesignQuery) -> tuple[tuple[DesignRow, ...], DesignQuery]:
    """Run one query, refusing an empty result. Returns rows and the resolved query.

    The resolved query is returned rather than discarded because it is what
    belongs in the design set: a manifest naming a filter run by a name two
    runs share does not describe a reproducible selection, and `DesignSet`
    stores the query precisely so that the selection can be re-derived.

    Empty is refused here rather than at design-set construction so the message
    can name the query. A set of zero designs is never what somebody meant, and
    the ways to get one -- a filter nothing passed, a run name that does not
    exist, a length window with nothing in it -- need different fixes.
    """
    with read_only(database) as connection:
        if query.passed_filter:
            run_ids = resolve_run_ids(connection, query.filter_runs)
            decision_runs = connection.execute(
                "SELECT DISTINCT run_id FROM decisions WHERE kind = 'filter' AND run_id IN "
                f"({', '.join('?' for _ in run_ids)})",
                list(run_ids),
            ).fetchall()
            unsupported = set(run_ids) - {row[0] for row in decision_runs}
            if unsupported:
                raise SelectionError(
                    f"run(s) have no filter decisions: {', '.join(sorted(unsupported))}"
                )
            query = query.model_copy(update={"filter_runs": run_ids})
        designs = select_designs(connection, query)

    if not designs:
        raise SelectionError(
            f"query selected no designs from {database}\n  {_describe(query)}"
        )
    return designs, query


def build(
    database: str | Path,
    query: DesignQuery,
    directory: str | Path,
) -> tuple[DesignSet, Path, Path]:
    """Select, freeze, and write `<digest>.fasta` + `<digest>.json`.

    Returns the set and both paths. Writing is idempotent: the same query
    against an unchanged database produces the same digest and therefore the
    same two filenames, and `write_design_set` returns the existing pair rather
    than rewriting it.
    """
    designs, resolved = select(database, query)
    design_set = build_design_set(designs, database=database, query=resolved)
    fasta, manifest = write_design_set(design_set, directory)
    return design_set, fasta, manifest


def _describe(query: DesignQuery) -> str:
    """The narrowing clauses a person would want to see in an empty-result error."""
    parts = [
        f"{field}={value!r}"
        for field, value in sorted(query.model_dump(mode="json").items())
        if value not in ((), None, False)
    ]
    return ", ".join(parts) if parts else "no narrowing clauses"
