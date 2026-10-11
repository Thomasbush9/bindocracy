"""Resolve mutable campaign data once, before workers are scheduled."""

from __future__ import annotations

import json
from dataclasses import dataclass

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import read_single_fasta
from bindocracy.functions.models import ScoringFunction
from bindocracy.functions.run import (
    FunctionRunConfig,
    FunctionRunError,
    _structures,
    resolve_function,
)
from bindocracy.functions.runner import preflight_custom
from bindocracy.runs.designset import DesignSet
from bindocracy.runs.inputs import TargetDigest
from bindocracy.runs.selection import read_only
from bindocracy.store.query import select_designs
from bindocracy.store.records import canonical_json, sha256_text, stable_id


@dataclass(frozen=True)
class FrozenFunction:
    target_sequence: str
    design_set: DesignSet
    function: ScoringFunction
    rows: list[dict]
    structures: dict
    metric_sources: dict
    skipped: dict


def resolve_metrics(connection, config, entries):
    """Resolve exact IDs, preserving failed/missing measurements as evidence."""
    values = {entry.design_id: {} for entry in entries}
    sources = {entry.design_id: {} for entry in entries}
    skipped = {}
    for alias, request in config.metric_inputs.items():
        if not alias.strip():
            raise FunctionRunError("metric input aliases must be nonempty")
        source_run = connection.execute(
            "SELECT r.run_id, r.model_config_id, c.general_config_id, "
            "c.model_source_sha256, c.general_source_sha256, c.model_config_hash, c.general_config_hash FROM runs r "
            "JOIN configs c ON c.model_config_id = r.model_config_id WHERE r.run_id = ?",
            [request.run_id],
        ).fetchall()
        if len(source_run) != 1:
            raise FunctionRunError(f"metric input {alias!r} requires an exact, existing run_id: {request.run_id}")
        run_id, config_id, general_id, model_hash, general_hash, model_config_hash, general_config_hash = source_run[0]
        for entry in entries:
            rows = connection.execute(
                "SELECT metric_id, value, status, details, measured_at, unit, direction FROM metrics "
                "WHERE run_id = ? AND design_id = ? AND name = ? AND replicate = ?",
                [run_id, entry.design_id, request.metric, request.replicate],
            ).fetchall()
            if len(rows) > 1:
                raise FunctionRunError(f"ambiguous metric input {alias!r} for design {entry.design_id}")
            evidence = {
                "run_id": run_id, "model_config_id": config_id, "general_config_id": general_id,
                "model_source_sha256": model_hash, "general_source_sha256": general_hash,
                "model_config_hash": model_config_hash, "general_config_hash": general_config_hash,
                "metric": request.metric, "replicate": request.replicate,
                "metric_id": None, "value": None, "status": "missing",
            }
            if rows:
                metric_id, value, status, details, measured_at, unit, direction = rows[0]
                evidence.update(metric_id=metric_id, value=value, status=status, unit=unit, direction=direction,
                                details=json.loads(details) if isinstance(details, str) else details,
                                measured_at=measured_at.isoformat() if measured_at else None)
            sources[entry.design_id][alias] = evidence
            if evidence["status"] != "ok" or evidence["value"] is None:
                if config.missing_metrics == "error":
                    raise FunctionRunError(
                        f"metric input {alias!r} for design {entry.design_id} is {evidence['status']}: "
                        f"run={run_id} metric={request.metric} replicate={request.replicate}; "
                        f"details={evidence.get('details')}"
                    )
                skipped.setdefault(entry.design_id, []).append(alias)
            values[entry.design_id][alias] = evidence["value"]
    return values, sources, skipped


def freeze_function(general: GeneralConfig, config: FunctionRunConfig) -> FrozenFunction:
    design_set = DesignSet.read(config.design_set)
    entries = design_set.entries
    if not entries or design_set.n_designs != len(entries):
        raise FunctionRunError("design set must contain its declared, nonempty membership")
    if [entry.index for entry in entries] != list(range(len(entries))):
        raise FunctionRunError("design-set indices must be contiguous and ordered from zero")
    if len({entry.design_id for entry in entries}) != len(entries):
        raise FunctionRunError("design set contains duplicate design IDs")
    digest = sha256_text(canonical_json([[entry.design_id, entry.sequence] for entry in entries]))
    if design_set.digest != digest or design_set.scope_id != stable_id("design-set", digest):
        raise FunctionRunError("design-set identity does not match its frozen members")
    target_sequence = read_single_fasta(general.target.sequence_fasta)
    target = TargetDigest.of(general.target.name, target_sequence)
    function = resolve_function(config, general, target_sequence)
    preflight_custom(function)
    if config.database is None:
        raise FunctionRunError("function planning requires database for frozen candidate verification")
    with read_only(config.database) as connection:
        stored_target = connection.execute("SELECT value FROM _meta WHERE key = 'target_sha256'").fetchone()
        if stored_target is not None and stored_target[0] != target.sequence_sha256:
            raise FunctionRunError("general config target does not match the named campaign database")
        designs = {row.design_id: row for row in select_designs(connection)}
        for entry in entries:
            row = designs.get(entry.design_id)
            if row is None:
                raise FunctionRunError(f"frozen design is absent from named database: {entry.design_id}")
            if row.sequence != entry.sequence or row.length != entry.length:
                raise FunctionRunError(f"frozen sequence identity differs for design {entry.design_id}")
        structures, _ = _structures(connection, config, function, design_set)
        values, sources, skipped = resolve_metrics(connection, config, entries)
    rows = [
        {"index": entry.index, "design_id": entry.design_id, "sequence": entry.sequence,
         "target_sequence": target_sequence,
         "source_model": structures[entry.design_id][1] if structures else None,
         "metrics": values[entry.design_id]}
        for entry in entries
    ]
    return FrozenFunction(target_sequence, design_set, function, rows, structures, sources, skipped)
