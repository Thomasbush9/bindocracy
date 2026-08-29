"""YAML loading and conversion to config-table records.

Nothing here knows which tools exist. A tool's module supplies its config type
and its preflight; this file turns that pair into a validated `LoadedConfigs`
and a `ConfigRecord`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from bindocracy.config.models import GeneralConfig, ToolConfig
from bindocracy.store.records import ConfigRecord


class ConfigLoadError(ValueError):
    """A configuration file could not be read as a YAML mapping."""


@dataclass(frozen=True)
class LoadedConfigs:
    """A validated general + model pair, plus that tool's preflight result."""

    general_path: Path
    model_path: Path
    general: GeneralConfig
    model: ToolConfig
    preflight: Any

    @property
    def tool(self) -> str:
        return self.model.tool

    def to_record(self) -> ConfigRecord:
        """Build one general+model pair for the config table."""
        return ConfigRecord(
            general_name=self.general.campaign.name,
            general_schema_version=self.general.schema_version,
            general_config_json=self.general.model_dump(mode="json"),
            general_source_uri=str(self.general_path.resolve()),
            general_source_sha256=sha256_file(self.general_path),
            model_name=self.model.name,
            tool=self.model.tool,
            model_schema_version=self.model.schema_version,
            model_config_json=self.model.model_dump(mode="json"),
            model_source_uri=str(self.model_path.resolve()),
            model_source_sha256=sha256_file(self.model_path),
        )


def sha256_file(path: str | Path) -> str:
    """Hash the exact bytes of a source config file."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def load_yaml[ModelT: BaseModel](path: str | Path, model_type: type[ModelT]) -> ModelT:
    """Load one YAML mapping and validate it with the requested Pydantic model."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text())
    except OSError as error:
        raise ConfigLoadError(f"cannot read config {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigLoadError(f"invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw, dict):
        raise ConfigLoadError(f"{config_path} must contain a YAML mapping")
    return model_type.model_validate(raw)


def read_tool_name(path: str | Path) -> str:
    """Peek at a model config's `tool` key, to choose how to load the rest."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ConfigLoadError(f"cannot read config {config_path}: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("tool"), str):
        raise ConfigLoadError(f"{config_path} has no 'tool' key naming its tool")
    return raw["tool"]


def load_pair(
    general_path: str | Path,
    model_path: str | Path,
    model_type: type[ToolConfig],
    preflight: Callable[[GeneralConfig, Any], Any],
) -> LoadedConfigs:
    """Load, validate, and preflight one general + model config pair."""
    general_config_path = Path(general_path)
    model_config_path = Path(model_path)
    general = load_yaml(general_config_path, GeneralConfig)
    model = load_yaml(model_config_path, model_type)
    return LoadedConfigs(
        general_path=general_config_path,
        model_path=model_config_path,
        general=general,
        model=model,
        preflight=preflight(general, model),
    )
