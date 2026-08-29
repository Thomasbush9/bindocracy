"""Filesystem checks BoltzGen needs, including the one that matters most.

A BoltzGen spec carries its own structure path, so it can drift away from the
campaign target while every other check still passes -- and the run then
designs against the wrong protein and looks entirely normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import ConfigPreflightError, read_single_fasta
from bindocracy.tools.boltzgen.config import BoltzGenConfig


@dataclass(frozen=True)
class BoltzGenPreflight:
    target_sequence: str
    spec_files: tuple[Path, ...]
    spec: dict

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


def spec_file_paths(spec: dict) -> tuple[Path, ...]:
    """Every structure a BoltzGen spec pulls geometry from."""
    entities = spec.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ConfigPreflightError("BoltzGen spec has no 'entities' list")
    return tuple(
        Path(entity["file"]["path"])
        for entity in entities
        if isinstance(entity, dict)
        and isinstance(entity.get("file"), dict)
        and "path" in entity["file"]
    )


def preflight_boltzgen(general: GeneralConfig, boltzgen: BoltzGenConfig) -> BoltzGenPreflight:
    """Check paths and, crucially, that the spec aims at the right structure.

    Designing against the wrong geometry is the silent-wrong-answer class from
    docs/harness-design.md section 5: the run completes and the output looks
    entirely normal. The spec carries its own target path, so it can drift away
    from the campaign's target without anything noticing. Assert it here.
    """
    required_files = {
        "BoltzGen spec": boltzgen.spec.template,
        "BoltzGen container": boltzgen.runtime.container,
        "target FASTA": general.target.sequence_fasta,
    }
    errors = [
        f"{description} does not exist: {path}"
        for description, path in required_files.items()
        if not path.is_file()
    ]

    if general.target.structure_cif is None:
        errors.append("BoltzGen needs target.structure_cif, which is not set")
    elif not general.target.structure_cif.is_file():
        errors.append(f"target structure does not exist: {general.target.structure_cif}")

    if errors:
        raise ConfigPreflightError("\n".join(errors))

    try:
        spec = yaml.safe_load(boltzgen.spec.template.read_text())
    except yaml.YAMLError as error:
        raise ConfigPreflightError(f"invalid YAML in BoltzGen spec: {error}") from error
    if not isinstance(spec, dict):
        raise ConfigPreflightError(f"BoltzGen spec must be a mapping: {boltzgen.spec.template}")

    spec_files = spec_file_paths(spec)
    missing = [path for path in spec_files if not path.is_file()]
    if missing:
        raise ConfigPreflightError(
            "BoltzGen spec references files that do not exist:\n"
            + "\n".join(str(path) for path in missing)
        )

    target = general.target.structure_cif.resolve()
    if target not in {path.resolve() for path in spec_files}:
        raise ConfigPreflightError(
            f"BoltzGen spec does not reference the campaign target {target}.\n"
            f"It references: {', '.join(str(p) for p in spec_files) or 'no structure'}.\n"
            "A spec aimed at another structure designs against the wrong target silently."
        )

    return BoltzGenPreflight(
        target_sequence=read_single_fasta(general.target.sequence_fasta),
        spec_files=spec_files,
        spec=spec,
    )
