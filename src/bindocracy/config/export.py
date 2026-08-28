"""Recover authored-style YAML from durable JSON configuration rows."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import yaml


class ConfigNotFoundError(KeyError):
    """No general or model configuration has the requested ID."""


def config_yaml_from_db(database: str | Path, config_id: str) -> str:
    """Return YAML for a general_config_id or model_config_id."""
    con = duckdb.connect(str(database), read_only=True)
    try:
        row = con.execute(
            "SELECT CAST(model_config_json AS VARCHAR) FROM configs "
            "WHERE model_config_id = ?",
            [config_id],
        ).fetchone()
        if row is None:
            row = con.execute(
                "SELECT CAST(general_config_json AS VARCHAR) FROM configs "
                "WHERE general_config_id = ? LIMIT 1",
                [config_id],
            ).fetchone()
    finally:
        con.close()

    if row is None:
        raise ConfigNotFoundError(config_id)
    payload = json.loads(row[0])
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def recover_config_yaml(
    database: str | Path,
    config_id: str,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write recovered configuration YAML without overwriting by default."""
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    text = config_yaml_from_db(database, config_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    return output_path
