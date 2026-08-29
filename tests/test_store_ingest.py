from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from conftest import design_line, write_task

from bindocracy.adapters import collect_run
from bindocracy.adapters.mosaic import METRIC_NAME
from bindocracy.config import load_mosaic_configs
from bindocracy.runs import ingest_bundle, read_collected, write_collected
from bindocracy.store import CampaignStore, IngestConflictError, create_database
from bindocracy.tools import plan


@pytest.fixture
def staged(configs, tmp_path: Path) -> tuple[Path, Path]:
    """A collected two-task run and an initialized empty database."""
    manifest = plan(load_mosaic_configs(*configs), tmp_path / "run")
    for task_id in (0, 1):
        write_task(manifest.directory, task_id,
                   [design_line(task_id, index) for index in range(4)], status={})

    bundle = write_collected(
        collect_run(manifest.directory / "run.json"),
        manifest.directory / "collected.json",
    )
    return create_database(tmp_path / "campaign.duckdb"), bundle


def test_the_bundle_survives_serialization(staged) -> None:
    _, bundle = staged

    collected = read_collected(bundle)

    assert collected.run.n_produced == 8
    assert len(collected.designs) == 8
    assert len(collected.metrics) == 8


def test_ingest_writes_run_designs_metrics_and_artifacts(staged) -> None:
    database, bundle = staged
    collected = read_collected(bundle)

    assert ingest_bundle(database, bundle) is True

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM configs").fetchone() == (1,)
    assert con.execute(
        "SELECT status, n_requested, n_attempted, n_produced FROM runs "
        "WHERE tool = 'mosaic' AND kind = 'generate'"
    ).fetchone() == ("succeeded", 8, 8, 8)
    assert con.execute("SELECT count(*) FROM designs").fetchone() == (8,)
    assert con.execute(
        "SELECT count(*) FROM metrics WHERE name = ? AND direction = 'min'",
        [METRIC_NAME],
    ).fetchone() == (8,)
    kinds = {row[0] for row in con.execute("SELECT DISTINCT kind FROM artifacts").fetchall()}
    assert {"native_designs", "task_status", "log", "driver_script"} <= kinds
    assert con.execute(
        "SELECT value FROM metrics JOIN designs USING (design_id) WHERE native_id = ?",
        ["task-0000-design-000000"],
    ).fetchone() == (-0.5,)
    assert con.execute("SELECT run_id FROM runs").fetchone() == (collected.run.run_id,)
    con.close()


def test_design_history_carries_both_config_ids(staged) -> None:
    database, bundle = staged
    ingest_bundle(database, bundle)
    collected = read_collected(bundle)

    con = duckdb.connect(str(database), read_only=True)
    rows = con.execute(
        "SELECT DISTINCT general_config_id, model_config_id, producing_run_name "
        "FROM design_history"
    ).fetchall()
    expected_model = con.execute("SELECT model_config_id FROM configs").fetchone()[0]
    expected_general = con.execute("SELECT general_config_id FROM configs").fetchone()[0]
    con.close()

    assert rows == [(expected_general, expected_model, collected.run.name)]
    assert collected.run.model_config_id == expected_model


def test_reingesting_the_same_bundle_is_a_no_op(staged) -> None:
    database, bundle = staged
    assert ingest_bundle(database, bundle) is True

    assert ingest_bundle(database, bundle) is False

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM runs").fetchone() == (1,)
    assert con.execute("SELECT count(*) FROM designs").fetchone() == (8,)
    con.close()


def test_a_changed_bundle_for_a_known_run_is_refused(staged, tmp_path: Path) -> None:
    database, bundle = staged
    ingest_bundle(database, bundle)
    collected = read_collected(bundle)

    # Same run, fewer designs: an integrity error, never a second copy.
    trimmed = collected.model_copy(update={
        "designs": collected.designs[:1],
        "metrics": collected.metrics[:1],
    })
    with CampaignStore(database) as store, pytest.raises(IngestConflictError):
        store.ingest(trimmed)

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM designs").fetchone() == (8,)
    con.close()
