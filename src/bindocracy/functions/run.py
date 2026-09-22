"""Run a custom scoring function against a frozen database selection."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from bindocracy.config.models import ConfigModel, GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    read_single_fasta,
    target_hotspot_positions,
)
from bindocracy.functions.contract import FunctionInput
from bindocracy.functions.models import (
    BUILTIN_FUNCTIONS,
    CustomFunction,
    ScoringFunction,
    builtin,
)
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


# The functions this repository ships, named rather than restated. A built-in
# declares no metrics because its metrics are already in the registry with a
# fixed meaning; naming it is the whole config.
BuiltinName = Literal["epitope", "sequence"]


class FunctionRunConfig(ConfigModel):
    """One function run: a script, or the name of one this repository ships.

    Exactly one of `builtin` and `function`. They are separate fields rather
    than one union because they carry opposite obligations: a custom function
    must declare what its numbers mean, and a built-in must not, since
    restating a registered metric's direction in YAML is a second place for it
    to be wrong.

    Until 2026-09-22 only `function` existed, so `epitope` and `sequence` were
    implemented, tested, documented as the two worked examples -- and reachable
    from no command. A config naming `epitope` was refused twice over: as a
    built-in for declaring no `metrics`, and as a custom function for declaring
    metric names the registry already owns. That is the same shape as the
    `readers.epitope` bug one level up, so it is fixed the same way: the thing
    that cannot run is refused at the config, and the thing that should run has
    a way to be named.
    """

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    tool: Literal["function"] = "function"
    design_set: Path
    # A function this repository ships: `epitope` or `sequence`.
    builtin: BuiltinName | None = None
    # A user-supplied script. See docs/scoring-functions.md tier 3.
    function: CustomFunction | None = None
    # An evaluator run name or ID, never a model-wide search across runs.
    structures_from: str | None = None

    @model_validator(mode="after")
    def exactly_one_function(self) -> Self:
        if (self.builtin is None) == (self.function is None):
            raise ValueError(
                "name exactly one of `builtin` (a function this repository ships: "
                f"{sorted(BUILTIN_FUNCTIONS)}) or `function` (a script of your own)"
            )
        return self


def resolve_function(
    config: FunctionRunConfig, general: GeneralConfig, target_sequence: str
) -> ScoringFunction:
    """The function this run executes, with what only the campaign can supply.

    `epitope` is the case that needs anything: it measures contacts against the
    campaign's hotspots, and it takes them as 1-based positions in the target's
    FASTA rather than author numbering, because residue ids in a predicted pose
    are positional. The mapping is the shared, checked one every other stage
    uses, so the epitope a design was optimized against and the epitope it is
    measured against are the same residues by construction.

    A campaign with no `target.hotspots` is refused rather than run. The driver
    would report coverage and offset as absent -- correctly, since no epitope
    was named -- and store only the two geometry metrics, which is a run that
    succeeds and does not answer the question it was started for.
    """
    if config.function is not None:
        return config.function
    if config.builtin != "epitope":
        return builtin(config.builtin)

    hotspots = target_hotspot_positions(
        general.target.hotspots,
        chain_id=general.target.chain_id,
        target_length=len(target_sequence),
        target_name=general.target.name,
    )
    if not hotspots:
        raise ConfigPreflightError(
            "the epitope function measures contacts against target.hotspots, and "
            f"campaign {general.campaign.name!r} names none. Coverage and offset "
            "would be stored as absent and the run would answer nothing. Name the "
            "epitope in the general config, or run a function that does not need "
            "one."
        )
    return builtin("epitope", args=("--hotspots", ",".join(str(spot) for spot in hotspots)))


def _structures(
    connection, config: FunctionRunConfig, function: ScoringFunction, design_set: DesignSet
):
    """Resolve exactly one complex pose at replicate zero per frozen member."""
    if "structure" not in function.inputs:
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
        if function.prefix == "source_model" and not model:
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
        function = resolve_function(config, general, target_sequence)
        preflight_custom(function)
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
            structures, source_run = _structures(connection, config, function, design_set)
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
            function, inputs, run_id=run_id,
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
            "function": function.name,
            # What the script was actually invoked with, beyond --inputs and
            # --outputs. For the epitope function this is the resolved hotspot
            # list, so the epitope a measurement was made against is readable
            # from the run rather than re-derived from the general config.
            "function_args": list(function.args),
            "script_sha256": result.script_sha256,
            "helper_sha256": result.helper_sha256,
            "target": target.model_dump(mode="json"),
            "structures_from": source_run,
            "database": str(Path(database).resolve()),
        },
        output_uri=str(output), created_at=started, started_at=started, finished_at=utc_now(),
    )
    return CollectedRun(run=run, metrics=result.records), record
