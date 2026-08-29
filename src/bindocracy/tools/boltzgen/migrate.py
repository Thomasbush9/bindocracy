"""Convert BoltzGen rows written under the old interpretation.

Runs collected before the semantics changed recorded three things differently:
a design that failed BoltzGen's own filters was stored as `partial`, as though
it were incomplete; the filter verdict lived in `designs.metadata`; and
`final_rank` was a numeric metric. Newly collected runs are correct, so without
this the same table would hold two meanings at once and any query across both
would be wrong.

This is data semantics rather than schema shape, so it is a command rather than
an automatic migration on open: it rewrites rows, and rewriting rows should be
something a person asks for.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb

from bindocracy.store.records import DecisionKind, stable_id
from bindocracy.tools.boltzgen.adapter import FILTER_NAME, RANK_NAME, rank_scope

_TASK = re.compile(r"^task-(\d{4})-")


@dataclass(frozen=True)
class Backfill:
    """What the migration did, or would do."""

    designs: int
    filters: int
    ranks: int
    metrics_removed: int
    statuses_corrected: int

    @property
    def empty(self) -> bool:
        return self.designs == 0


def task_of_native_id(native_id: str) -> int:
    """Which task produced this design.

    Ranks are scoped per task because BoltzGen ranks each task's pool from 1.
    Designs collected before native IDs were task-qualified came from
    single-task runs, so they belong to task 0.
    """
    match = _TASK.match(native_id)
    return int(match.group(1)) if match else 0


def backfill_boltzgen_decisions(database: str | Path, *, dry_run: bool = False) -> Backfill:
    """Rewrite historical BoltzGen rows into filter and rank decisions."""
    con = duckdb.connect(str(database), read_only=dry_run)
    try:
        rows = con.execute("""
            SELECT d.design_id, d.run_id, d.native_id, d.status,
                   CAST(d.metadata AS VARCHAR)
            FROM designs d JOIN runs r USING (run_id)
            WHERE r.tool = 'boltzgen'
              AND json_extract(d.metadata, '$.pass_filters') IS NOT NULL
        """).fetchall()
        if not rows:
            return Backfill(0, 0, 0, 0, 0)

        filters = ranks = statuses = 0
        if not dry_run:
            con.execute("BEGIN TRANSACTION")
        try:
            for design_id, run_id, native_id, status, metadata in rows:
                meta = json.loads(metadata)
                task_id = task_of_native_id(native_id)

                if not dry_run:
                    con.execute(
                        # created_at is taken from the design in SQL rather
                        # than carried through Python, which would need a
                        # timezone library just to hand it back unchanged.
                        "INSERT INTO decisions (decision_id, run_id, design_id, kind, "
                        "name, passed, created_at) "
                        "SELECT ?, ?, ?, ?, ?, ?, created_at FROM designs "
                        "WHERE design_id = ? ON CONFLICT DO NOTHING",
                        [stable_id("decision", run_id, design_id, FILTER_NAME), run_id,
                         design_id, DecisionKind.FILTER.value, FILTER_NAME,
                         bool(meta.get("pass_filters")), design_id],
                    )
                filters += 1

                rank = meta.get("final_rank")
                if rank is not None:
                    if not dry_run:
                        con.execute(
                            "INSERT INTO decisions (decision_id, run_id, design_id, kind, "
                            "name, rank, scope_id, created_at) "
                            "SELECT ?, ?, ?, ?, ?, ?, ?, created_at FROM designs "
                            "WHERE design_id = ? ON CONFLICT DO NOTHING",
                            [stable_id("decision", run_id, design_id, RANK_NAME), run_id,
                             design_id, DecisionKind.RANK.value, RANK_NAME, int(rank),
                             rank_scope(run_id, task_id), design_id],
                        )
                    ranks += 1

                if status != "produced":
                    statuses += 1
                if not dry_run:
                    # A complete design is produced; the verdict is a decision.
                    con.execute(
                        "UPDATE designs SET status = 'produced', metadata = ? "
                        "WHERE design_id = ?",
                        [json.dumps({"file_name": meta.get("file_name")},
                                    sort_keys=True, separators=(",", ":")), design_id],
                    )

            removed = con.execute(
                "SELECT count(*) FROM metrics WHERE name = ?", [RANK_NAME]
            ).fetchone()[0]
            if not dry_run:
                con.execute("DELETE FROM metrics WHERE name = ?", [RANK_NAME])
                con.execute("COMMIT")
        except Exception:
            if not dry_run:
                con.execute("ROLLBACK")
            raise
    finally:
        con.close()

    return Backfill(
        designs=len(rows), filters=filters, ranks=ranks,
        metrics_removed=removed, statuses_corrected=statuses,
    )
