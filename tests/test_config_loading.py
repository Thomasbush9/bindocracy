from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
import yaml
from pydantic import ValidationError

from bindocracy.config import (
    MosaicConfig,
    config_yaml_from_db,
    load_mosaic_configs,
    load_yaml,
    recover_config_yaml,
)
from bindocracy.store import CampaignStore


def _write_config_fixture(root: Path) -> tuple[Path, Path]:
    fasta = root / "target.fasta"
    msa = root / "target.a3m"
    script = root / "hallucinate.py"
    container = root / "mosaic.sif"
    weights = root / "weights"
    (weights / "boltz").mkdir(parents=True)
    fasta.write_text(">target\nACDEFG\n")
    msa.write_text(">target\nACDEFG\n")
    script.write_text("print('fixture')\n")
    container.write_bytes(b"fixture")

    general_path = root / "general.yaml"
    general_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign": {"name": "test-campaign"},
                "target": {
                    "name": "test-target",
                    "sequence_fasta": str(fasta),
                    "msa": str(msa),
                    "chain_id": "A",
                    "hotspots": [],
                },
                "cluster": {
                    "executor": "slurm",
                    "account": "test-account",
                    "default_partition": "test-gpu",
                    "max_concurrent_jobs": 2,
                },
            },
            sort_keys=False,
        )
    )

    model_path = root / "mosaic.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "mosaic-test",
                "tool": "mosaic",
                "driver": {"script": str(script), "archive": True},
                "sampling": {
                    "binder_length": 70,
                    "jobs": 2,
                    "designs_per_job": 4,
                    "max_runtime_hours": 1,
                    "seed_base": 0,
                },
                "runtime": {
                    "container": str(container),
                    "weights": str(weights),
                },
                "resources": {
                    "gpus": 1,
                    "cpus": 8,
                    "memory_gb": 32,
                    "walltime": "02:00:00",
                },
            },
            sort_keys=False,
        )
    )
    return general_path, model_path


def test_load_mosaic_configs_returns_typed_models(tmp_path: Path) -> None:
    general_path, model_path = _write_config_fixture(tmp_path)

    loaded = load_mosaic_configs(general_path, model_path)

    assert loaded.general.target.sequence_fasta == tmp_path / "target.fasta"
    assert loaded.mosaic.sampling.jobs == 2
    assert loaded.preflight.target_length == 6
    record = loaded.to_record()
    assert record.general_name == "test-campaign"
    assert record.model_name == "mosaic-test"
    assert record.general_config_id is not None
    assert record.model_config_id is not None
    assert record.general_config_hash is not None
    assert record.model_config_hash is not None
    assert record.general_source_uri == str(general_path.resolve())
    assert record.model_source_uri == str(model_path.resolve())


def test_unknown_mosaic_key_is_rejected(tmp_path: Path) -> None:
    _, model_path = _write_config_fixture(tmp_path)
    raw = yaml.safe_load(model_path.read_text())
    raw["scrpit"] = "typo.py"
    model_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_yaml(model_path, MosaicConfig)


def test_config_load_populates_only_configs_and_is_idempotent(tmp_path: Path) -> None:
    general_path, model_path = _write_config_fixture(tmp_path)
    loaded = load_mosaic_configs(general_path, model_path)
    record = loaded.to_record()
    database = tmp_path / "campaign.duckdb"

    with CampaignStore.create(database) as store:
        first_insert = store.add_configs([record])
        second_insert = store.add_configs([record])

    assert first_insert == (record.model_config_id,)
    assert second_insert == ()

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM configs").fetchone() == (1,)
    stored = con.execute(
        "SELECT general_config_id, model_config_id FROM configs"
    ).fetchone()
    assert stored == (record.general_config_id, record.model_config_id)
    for table in ("runs", "designs", "artifacts", "metrics", "decisions"):
        assert con.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)
    con.close()


def test_recover_general_and_model_yaml(tmp_path: Path) -> None:
    general_path, model_path = _write_config_fixture(tmp_path)
    loaded = load_mosaic_configs(general_path, model_path)
    record = loaded.to_record()
    assert record.general_config_id is not None
    assert record.model_config_id is not None
    database = tmp_path / "campaign.duckdb"

    with CampaignStore.create(database) as store:
        store.add_configs([record])

    general_yaml = config_yaml_from_db(database, record.general_config_id)
    assert yaml.safe_load(general_yaml) == loaded.general.model_dump(mode="json")

    recovered_model = tmp_path / "recovered" / "mosaic.yaml"
    recover_config_yaml(database, record.model_config_id, recovered_model)
    assert yaml.safe_load(recovered_model.read_text()) == loaded.mosaic.model_dump(mode="json")
