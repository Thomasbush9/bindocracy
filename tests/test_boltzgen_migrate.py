"""Historical BoltzGen rows must come to mean what new ones mean.

Runs collected before the semantics changed stored a filtered-out design as
`partial`, kept the verdict in `designs.metadata`, and recorded `final_rank` as
a metric. New runs do none of those. Left alone the table would hold two
interpretations at once, and any query spanning both would be wrong in a way
nothing announces.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from bindocracy.store import CampaignStore, create_database
from bindocracy.store.records import (
    CollectedRun,
    ConfigRecord,
    DesignRecord,
    MetricRecord,
    RunRecord,
)
from bindocracy.tools.boltzgen.migrate import backfill_boltzgen_decisions, task_of_native_id


def old_style_bundle(config: ConfigRecord, run_id: str, rows: list[tuple[str, bool, int]]):
    """A bundle in the pre-change shape: verdict in metadata, rank as a metric."""
    run = RunRecord(run_id=run_id, name="bg", tool="boltzgen", kind="generate",
                    model_config_id=config.model_config_id, status="succeeded",
                    n_requested=len(rows), n_produced=len(rows))
    designs, metrics = [], []
    for index, (native_id, passed, rank) in enumerate(rows):
        design = DesignRecord(
            design_id=f"{run_id}-d{index}", run_id=run_id, native_id=native_id,
            candidate_type="complex", sequence="ACDEFG",
            status="produced" if passed else "partial",
            metadata={"file_name": f"{native_id}.cif", "pass_filters": passed,
                      "final_rank": float(rank)},
        )
        designs.append(design)
        metrics.append(MetricRecord(
            metric_id=f"{run_id}-m{index}", run_id=run_id, design_id=design.design_id,
            name="boltzgen_final_rank", value=float(rank), direction="min"))
    return CollectedRun(run=run, designs=tuple(designs), metrics=tuple(metrics))


@pytest.fixture
def legacy(tmp_path: Path) -> Path:
    database = create_database(tmp_path / "campaign.duckdb")
    config = ConfigRecord(
        general_name="c", general_schema_version=1, general_config_json={"a": 1},
        model_name="bg", tool="boltzgen", model_schema_version=1,
        model_config_json={"tool": "boltzgen"})
    with CampaignStore(database) as store:
        store.ingest(old_style_bundle(config, "run-1", [
            ("task-0000-bg_0", True, 1), ("task-0000-bg_1", False, 2),
            ("task-0001-bg_0", False, 1),
        ]), configs=[config])
    return database


def test_a_task_is_read_from_a_qualified_native_id() -> None:
    assert task_of_native_id("task-0003-bg_7") == 3


def test_an_unqualified_native_id_came_from_a_single_task_run() -> None:
    """boltzgen_run10 predates task-qualified IDs, and had one task."""
    assert task_of_native_id("binder_spec_run10_7") == 0


def test_a_dry_run_reports_without_writing(legacy: Path) -> None:
    result = backfill_boltzgen_decisions(legacy, dry_run=True)

    assert (result.designs, result.filters, result.ranks) == (3, 3, 3)
    assert result.statuses_corrected == 2
    con = duckdb.connect(str(legacy), read_only=True)
    assert con.execute("SELECT count(*) FROM decisions").fetchone() == (0,)
    con.close()


def test_the_backfill_converts_verdicts_and_ranks(legacy: Path) -> None:
    backfill_boltzgen_decisions(legacy)

    con = duckdb.connect(str(legacy), read_only=True)
    assert con.execute(
        "SELECT kind, count(*) FROM decisions GROUP BY 1 ORDER BY 1"
    ).fetchall() == [("filter", 3), ("rank", 3)]
    assert con.execute(
        "SELECT count(*) FROM decisions WHERE kind='filter' AND passed"
    ).fetchone() == (1,)
    # every design is produced; the verdict is no longer its status
    assert con.execute("SELECT DISTINCT status FROM designs").fetchall() == [("produced",)]
    # the metric is gone, having become a decision
    assert con.execute(
        "SELECT count(*) FROM metrics WHERE name='boltzgen_final_rank'"
    ).fetchone() == (0,)
    con.close()


def test_ranks_keep_their_per_task_scope(legacy: Path) -> None:
    """Two tasks each rank from 1; a run-wide scope would say something false."""
    backfill_boltzgen_decisions(legacy)

    con = duckdb.connect(str(legacy), read_only=True)
    scopes = con.execute(
        "SELECT count(DISTINCT scope_id) FROM decisions WHERE kind='rank'"
    ).fetchone()[0]
    firsts = con.execute(
        "SELECT count(*) FROM decisions WHERE kind='rank' AND rank=1"
    ).fetchone()[0]
    con.close()

    assert scopes == 2
    assert firsts == 2


def test_metadata_keeps_only_what_is_still_metadata(legacy: Path) -> None:
    backfill_boltzgen_decisions(legacy)

    con = duckdb.connect(str(legacy), read_only=True)
    meta = json.loads(con.execute(
        "SELECT CAST(metadata AS VARCHAR) FROM designs LIMIT 1").fetchone()[0])
    con.close()

    assert set(meta) == {"file_name"}


def test_running_it_twice_is_a_no_op(legacy: Path) -> None:
    backfill_boltzgen_decisions(legacy)

    second = backfill_boltzgen_decisions(legacy)

    assert second.empty
    con = duckdb.connect(str(legacy), read_only=True)
    assert con.execute("SELECT count(*) FROM decisions").fetchone() == (6,)
    con.close()


def test_a_database_with_nothing_to_migrate_is_left_alone(tmp_path: Path) -> None:
    database = create_database(tmp_path / "empty.duckdb")

    assert backfill_boltzgen_decisions(database).empty
