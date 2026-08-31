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
from bindocracy.config.preflight import (
    ConfigPreflightError,
    hotspot_numbers,
    read_single_fasta,
)
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

    _require_matching_binding_sites(general, spec)

    return BoltzGenPreflight(
        target_sequence=read_single_fasta(general.target.sequence_fasta),
        spec_files=spec_files,
        spec=spec,
    )


def binding_residues(spec: dict) -> set[int]:
    """Every residue a BoltzGen spec names as a binding site.

    `binding: 95..110,143` is a comma-separated list whose items are either a
    single residue or an inclusive `a..b` range, in 1-based label_seq_id.
    """
    residues: set[int] = set()
    for entity in spec.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        for binding_type in (entity.get("file") or {}).get("binding_types") or []:
            chain = (binding_type or {}).get("chain") or {}
            for item in str(chain.get("binding", "")).split(","):
                item = item.strip()
                if not item:
                    continue
                start, _, end = item.partition("..")
                try:
                    low = int(start)
                    high = int(end) if end else low
                except ValueError as error:
                    raise ConfigPreflightError(
                        f"cannot read a binding site from {item!r}; expected a "
                        "residue or an inclusive range such as '95..110'"
                    ) from error
                residues.update(range(min(low, high), max(low, high) + 1))
    return residues


def _require_matching_binding_sites(general: GeneralConfig, spec: dict) -> None:
    """A campaign epitope and a spec's binding sites must not disagree.

    BoltzGen expresses an epitope in the spec, which the harness archives and
    never rewrites, so the two can drift apart with nothing noticing -- and a
    spec with no binding site designs against the whole surface while the rest
    of the campaign designs against one patch.
    """
    if not general.target.hotspots:
        return
    campaign = hotspot_numbers(general.target.hotspots)
    theirs = binding_residues(spec)
    if not theirs:
        raise ConfigPreflightError(
            f"the campaign names an epitope ({sorted(campaign)}), but the "
            "BoltzGen spec names no binding site, so it would design against "
            "the whole surface.\nAdd a `binding_types` entry to the spec's "
            "file entity, or clear target.hotspots."
        )
    if not campaign <= theirs:
        raise ConfigPreflightError(
            f"the BoltzGen spec binds residues {sorted(theirs)}, which does not "
            f"cover the campaign epitope {sorted(campaign)}.\n"
            "Spec binding sites are 1-based label_seq_id, not author numbering."
        )
