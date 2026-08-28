"""YAML loading and conversion to config-table records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from bindocracy.config.models import GeneralConfig, MosaicConfig
from bindocracy.config.preflight import MosaicPreflight, preflight_mosaic
from bindocracy.store.records import ConfigRecord


class ConfigLoadError(ValueError):
    """A configuration file could not be read as a YAML mapping."""


@dataclass(frozen=True)
class LoadedMosaicConfigs:
    """Validated general and Mosaic configs plus their preflight result."""

    general_path: Path
    model_path: Path
    general: GeneralConfig
    mosaic: MosaicConfig
    preflight: MosaicPreflight

    def to_record(self) -> ConfigRecord:
        """Build one general+model pair for the config table."""
        return ConfigRecord(
            general_name=self.general.campaign.name,
            general_schema_version=self.general.schema_version,
            general_config_json=self.general.model_dump(mode="json"),
            general_source_uri=str(self.general_path.resolve()),
            general_source_sha256=sha256_file(self.general_path),
            model_name=self.mosaic.name,
            tool=self.mosaic.tool,
            model_schema_version=self.mosaic.schema_version,
            model_config_json=self.mosaic.model_dump(mode="json"),
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


def load_mosaic_configs(
    general_path: str | Path,
    model_path: str | Path,
) -> LoadedMosaicConfigs:
    """Load, validate, and preflight one general + Mosaic config pair."""
    general_config_path = Path(general_path)
    model_config_path = Path(model_path)
    general = load_yaml(general_config_path, GeneralConfig)
    mosaic = load_yaml(model_config_path, MosaicConfig)
    preflight = preflight_mosaic(general, mosaic)
    return LoadedMosaicConfigs(
        general_path=general_config_path,
        model_path=model_config_path,
        general=general,
        mosaic=mosaic,
        preflight=preflight,
    )
