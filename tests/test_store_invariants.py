"""What the database guarantees regardless of which tool produced the rows.

The store is the one module every tool shares, so its invariants are the ones
that break quietly when a tool is added. These tests use hand-built records
rather than a parsed run directory, so they fail for schema reasons only.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from bindocracy.store import (
    ArtifactRecord,
    CampaignStore,
    CollectedRun,
    ConfigRecord,
    DesignRecord,
    MetricRecord,
    RunRecord,
    create_database,
)


def config_record(tool: str, model_name: str) -> ConfigRecord:
    """A minimal config pair for one tool. IDs derive from the JSON content."""
    return ConfigRecord(
        general_name="campaign",
        general_schema_version=1,
        general_config_json={"campaign": {"name": "campaign"}},
        model_name=model_name,
        tool=tool,
        model_schema_version=1,
        model_config_json={"tool": tool, "name": model_name},
    )


def bundle(config: ConfigRecord, *, run_id: str, sequences: tuple[str, ...]) -> CollectedRun:
    run = RunRecord(
        run_id=run_id,
        name=f"{config.tool}-run",
        tool=config.tool,
        kind="generate",
        model_config_id=config.model_config_id,
        status="succeeded",
        n_requested=len(sequences),
        n_produced=len(sequences),
    )
    designs = tuple(
        DesignRecord(
            run_id=run_id,
            design_id=f"{run_id}-design-{index}",
            native_id=f"native-{index}",
            candidate_type="sequence",
            sequence=sequence,
        )
        for index, sequence in enumerate(sequences)
    )
    metrics = tuple(
        MetricRecord(
            metric_id=f"{run_id}-metric-{index}",
            run_id=run_id,
            design_id=design.design_id,
            name=f"{config.tool}_score",
            value=float(index),
            direction="min",
        )
        for index, design in enumerate(designs)
    )
    return CollectedRun(run=run, designs=designs, metrics=metrics)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return create_database(tmp_path / "campaign.duckdb")


def test_ingest_is_atomic_when_the_config_is_missing(database: Path) -> None:
    """A run referencing an absent config must leave the database untouched."""
    config = config_record("mosaic", "m1")
    orphan = bundle(config, run_id="run-1", sequences=("ACDEFG",))

    with CampaignStore(database) as store, pytest.raises(duckdb.Error):
        store.ingest(orphan)  # configs deliberately not passed

    con = duckdb.connect(str(database), read_only=True)
    for table in ("runs", "designs", "metrics"):
        assert con.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)
    con.close()


def test_a_failed_ingest_does_not_block_a_later_correct_one(database: Path) -> None:
    config = config_record("mosaic", "m1")
    good = bundle(config, run_id="run-1", sequences=("ACDEFG",))

    with CampaignStore(database) as store:
        with pytest.raises(duckdb.Error):
            store.ingest(good)
        assert store.ingest(good, configs=[config]) is True

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM designs").fetchone() == (1,)
    con.close()


def test_two_tools_coexist_and_stay_distinguishable(database: Path) -> None:
    """Adding a tool must not merge or shadow another tool's rows."""
    mosaic = config_record("mosaic", "mosaic-01")
    toy = config_record("toy", "toy-01")

    with CampaignStore(database) as store:
        store.ingest(bundle(mosaic, run_id="run-m", sequences=("ACDEFG", "HIKLMN")),
                     configs=[mosaic])
        store.ingest(bundle(toy, run_id="run-t", sequences=("PQRSTV",)), configs=[toy])

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute(
        "SELECT producing_tool, count(*) FROM design_history "
        "GROUP BY producing_tool ORDER BY producing_tool"
    ).fetchall() == [("mosaic", 2), ("toy", 1)]
    # Each tool's designs carry its own model config, and both share the general one.
    assert con.execute(
        "SELECT count(DISTINCT model_config_id) FROM design_history"
    ).fetchone() == (2,)
    assert con.execute(
        "SELECT count(DISTINCT general_config_id) FROM design_history"
    ).fetchone() == (1,)
    assert con.execute(
        "SELECT count(*) FROM metrics WHERE name = 'toy_score'"
    ).fetchone() == (1,)
    con.close()


def test_the_same_general_config_groups_tools(database: Path) -> None:
    mosaic = config_record("mosaic", "mosaic-01")
    toy = config_record("toy", "toy-01")

    assert mosaic.general_config_id == toy.general_config_id
    assert mosaic.model_config_id != toy.model_config_id


def test_a_design_cannot_repeat_a_native_id_within_a_run(database: Path) -> None:
    config = config_record("mosaic", "m1")
    collected = bundle(config, run_id="run-1", sequences=("ACDEFG", "HIKLMN"))
    clashing = collected.model_copy(update={
        "designs": (
            collected.designs[0],
            collected.designs[1].model_copy(update={"native_id": "native-0"}),
        ),
        "metrics": (),
    })

    with CampaignStore(database) as store, pytest.raises(duckdb.Error):
        store.ingest(clashing, configs=[config])

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM designs").fetchone() == (0,)
    con.close()


def test_one_metric_per_design_name_and_replicate(database: Path) -> None:
    config = config_record("mosaic", "m1")
    collected = bundle(config, run_id="run-1", sequences=("ACDEFG",))
    duplicated = collected.model_copy(update={
        "metrics": (
            collected.metrics[0],
            collected.metrics[0].model_copy(update={"metric_id": "other", "value": 9.0}),
        )
    })

    with CampaignStore(database) as store, pytest.raises(duckdb.Error):
        store.ingest(duplicated, configs=[config])

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM metrics").fetchone() == (0,)
    con.close()


def test_replicates_of_one_metric_are_allowed(database: Path) -> None:
    """Same metric, same design, different replicate: a legitimate re-measure."""
    config = config_record("mosaic", "m1")
    collected = bundle(config, run_id="run-1", sequences=("ACDEFG",))
    replicated = collected.model_copy(update={
        "metrics": (
            collected.metrics[0],
            collected.metrics[0].model_copy(update={"metric_id": "r1", "replicate": 1}),
        )
    })

    with CampaignStore(database) as store:
        assert store.ingest(replicated, configs=[config]) is True

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM metrics").fetchone() == (2,)
    con.close()


def test_an_artifact_uri_is_unique_within_a_run(database: Path) -> None:
    config = config_record("mosaic", "m1")
    collected = bundle(config, run_id="run-1", sequences=("ACDEFG",))
    duplicated = collected.model_copy(update={
        "artifacts": (
            ArtifactRecord(artifact_id="a1", run_id="run-1", kind="log", uri="logs/t.log"),
            ArtifactRecord(artifact_id="a2", run_id="run-1", kind="log", uri="logs/t.log"),
        )
    })

    with CampaignStore(database) as store, pytest.raises(duckdb.Error):
        store.ingest(duplicated, configs=[config])

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)
    con.close()


def test_configs_are_inserted_once_across_repeated_ingests(database: Path) -> None:
    config = config_record("mosaic", "m1")

    with CampaignStore(database) as store:
        store.ingest(bundle(config, run_id="run-1", sequences=("ACDEFG",)), configs=[config])
        store.ingest(bundle(config, run_id="run-2", sequences=("HIKLMN",)), configs=[config])

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM configs").fetchone() == (1,)
    assert con.execute("SELECT count(*) FROM runs").fetchone() == (2,)
    con.close()
