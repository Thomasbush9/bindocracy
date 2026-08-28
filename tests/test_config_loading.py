from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
import yaml
from conftest import write_configs
from pydantic import ValidationError

from bindocracy.config import (
    MosaicConfig,
    config_yaml_from_db,
    load_mosaic_configs,
    load_yaml,
    recover_config_yaml,
)
from bindocracy.store import CampaignStore


def test_load_mosaic_configs_returns_typed_models(tmp_path: Path) -> None:
    general_path, model_path = write_configs(tmp_path)

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
    _, model_path = write_configs(tmp_path)
    raw = yaml.safe_load(model_path.read_text())
    raw["scrpit"] = "typo.py"
    model_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_yaml(model_path, MosaicConfig)


def test_config_load_populates_only_configs_and_is_idempotent(tmp_path: Path) -> None:
    general_path, model_path = write_configs(tmp_path)
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
    general_path, model_path = write_configs(tmp_path)
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
