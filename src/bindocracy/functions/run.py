"""Run a custom scoring function against a frozen database selection."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from bindocracy.config.models import ConfigModel, GeneralConfig
from bindocracy.config.preflight import ConfigPreflightError, read_single_fasta
from bindocracy.functions.contract import FunctionInput
from bindocracy.functions.models import CustomFunction
from bindocracy.functions.runner import FunctionError, preflight_custom, run_custom, write_summary
from bindocracy.runs.designset import DesignSet, DesignSetError
from bindocracy.runs.inputs import TargetDigest
from bindocracy.runs.selection import SelectionError, read_only, resolve_run_ids
from bindocracy.store.query import select_designs
from bindocracy.store.records import (
    CollectedRun,
    ConfigRecord,
    RunKind,
    RunRecord,
    RunStatus,
    canonical_json,
    new_id,
    sha256_text,
    stable_id,
    utc_now,
)


class FunctionRunError(RuntimeError):
    """A frozen selection cannot be scored as configured."""


class FunctionRunConfig(ConfigModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    tool: Literal["function"] = "function"
    design_set: Path
    function: CustomFunction
    # An evaluator run name or ID, never a model-wide search across runs.
    structures_from: str | None = None


def _structures(connection, config: FunctionRunConfig, design_set: DesignSet):
    """Resolve exactly one complex pose at replicate zero per frozen member."""
    if "structure" not in config.function.inputs:
        return {}, None
    if config.structures_from is None:
        raise FunctionRunError("structure input requires structures_from: an evaluator run name or ID")
    source_run = resolve_run_ids(connection, (config.structures_from,), kind="evaluate")[0]
    design_ids = [entry.design_id for entry in design_set.entries]
    placeholders = ", ".join("?" for _ in design_ids)
    rows = connection.execute(
        "SELECT a.design_id, r.output_uri, a.uri, "
        "json_extract_string(a.metadata, '$.model') "
        "FROM artifacts a JOIN runs r ON r.run_id = a.run_id "
        f"WHERE a.run_id = ? AND a.design_id IN ({placeholders}) "
        "AND a.kind = 'predicted_structure' "
        "AND json_extract_string(a.metadata, '$.condition') = 'complex' "
        "AND json_extract_string(a.metadata, '$.replicate') = '0'",
        [source_run, *design_ids],
    ).fetchall()
    resolved: dict[str, tuple[Path, str | None]] = {}
    for design_id, output_uri, uri, model in rows:
        if design_id in resolved:
            raise FunctionRunError(f"ambiguous complex replicate 0 structure for design {design_id}")
        path = Path(uri)
        if not path.is_absolute():
            if not output_uri:
                raise FunctionRunError(f"relative structure path has no run output directory: {uri}")
            path = Path(output_uri) / path
        if not path.is_file():
            raise FunctionRunError(f"required structure not found: {path}")
        if config.function.prefix == "source_model" and not model:
            raise FunctionRunError(f"structure for design {design_id} has no source model")
        resolved[design_id] = (path.resolve(), model)
    missing = [entry.design_id for entry in design_set.entries if entry.design_id not in resolved]
    if missing:
        raise FunctionRunError(
            f"{len(missing)} designs have no complex replicate 0 structure in evaluator "
            f"run {config.structures_from!r}: {', '.join(missing[:5])}"
        )
    return resolved, source_run


def run_function(
    *,
    database: str | Path,
    general: GeneralConfig,
    config: FunctionRunConfig,
    output_dir: str | Path,
    general_source: Path | None = None,
) -> tuple[CollectedRun, ConfigRecord]:
    """Score frozen members and return the ordinary atomic-ingest bundle.

    A new output directory is mandatory. A failed attempt remains on disk for
    diagnosis; neither it nor a completed run may be silently overwritten.
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FunctionRunError(f"output directory already exists; refusing to overwrite: {output}")
    try:
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
        preflight_custom(config.function)
        with read_only(database) as connection:
            stored_target = connection.execute(
                "SELECT value FROM _meta WHERE key = 'target_sha256'"
            ).fetchone()
            if stored_target is not None and stored_target[0] != target.sequence_sha256:
                raise FunctionRunError("general config target does not match the named campaign database")
            wanted_ids = {entry.design_id for entry in entries}
            designs = {
                row.design_id: row for row in select_designs(connection)
                if row.design_id in wanted_ids
            }
            for entry in entries:
                row = designs.get(entry.design_id)
                if row is None:
                    raise FunctionRunError(f"frozen design is absent from named database: {entry.design_id}")
                if row.sequence != entry.sequence or row.length != entry.length:
                    raise FunctionRunError(f"frozen sequence identity differs for design {entry.design_id}")
            structures, source_run = _structures(connection, config, design_set)
    except (DesignSetError, SelectionError, ConfigPreflightError, OSError, FunctionError) as error:
        raise FunctionRunError(str(error)) from error

    record = ConfigRecord(
        general_name=general.campaign.name,
        general_schema_version=general.schema_version,
        general_config_json=general.model_dump(mode="json"),
        general_source_uri=str(general_source) if general_source else None,
        model_name=config.name,
        tool=config.tool,
        model_schema_version=config.schema_version,
        model_config_json=config.model_dump(mode="json"),
    )
    inputs = [
        FunctionInput(
            index=entry.index, design_id=entry.design_id, sequence=entry.sequence,
            target_sequence=target_sequence,
            structure=structures[entry.design_id][0] if structures else None,
            source_model=structures[entry.design_id][1] if structures else None,
        )
        for entry in entries
    ]
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FunctionRunError(f"output directory already exists: {output}") from error
    (output / "config.json").write_text(record.model_dump_json(indent=2) + "\n")
    (output / "design_set.json").write_text(design_set.model_dump_json(indent=2) + "\n")
    (output / "target.fasta").write_text(f">{target.name}\n{target_sequence}\n")
    started = utc_now()
    run_id = new_id()
    try:
        result = run_custom(
            config.function, inputs, run_id=run_id,
            work_dir=output / "function", measured_at=started,
        )
    except FunctionError as error:
        raise FunctionRunError(str(error)) from error
    write_summary(output / "summary.json", [result])
    status = RunStatus.SUCCEEDED
    if result.failures or result.incomplete or result.n_scored != len(entries):
        status = RunStatus.PARTIAL if result.n_scored else RunStatus.FAILED
    run = RunRecord(
        run_id=run_id, name=config.name, tool=config.tool, kind=RunKind.EVALUATE,
        model_config_id=record.model_config_id, status=status,
        n_requested=len(entries), n_attempted=result.n_inputs, n_produced=result.n_scored,
        count_details={
            **result.summary,
            "n_failed": len(entries) - result.n_scored,
            "design_set_digest": design_set.digest,
            "scope_id": design_set.scope_id,
        },
        workflow_metadata={
            "function": config.function.name,
            "script_sha256": result.script_sha256,
            "helper_sha256": result.helper_sha256,
            "target": target.model_dump(mode="json"),
            "structures_from": source_run,
            "database": str(Path(database).resolve()),
        },
        output_uri=str(output), created_at=started, started_at=started, finished_at=utc_now(),
    )
    return CollectedRun(run=run, metrics=result.records), record
