"""Refuse an optimization run that cannot work, before any GPU is allocated.

The two checks worth reading are `_resolve_structures` and the `loss_models`
one. Both catch a mistake that would otherwise surface per-design inside a
container, after the allocation, looking like the script's fault:

* an optimizer asking for poses over designs nothing has folded
* a `loss_models` entry naming no model this campaign scores with, which would
  exclude nothing later while looking like it had
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bindocracy.config.load import sha256_file
from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import ConfigPreflightError, target_hotspot_positions
from bindocracy.runs.designset import DesignSet, DesignSetError
from bindocracy.runs.selection import SelectionError, read_only
from bindocracy.store.records import sha256_text
from bindocracy.tools.optimize.config import OptimizeConfig
from bindocracy.tools.scorer.config import ScoringModelName

# Prefixes a `loss_models` entry may name: the mosaic scorer's models plus the
# tools that are their own plugin. Checked so a typo cannot pass as a claim
# about which model had its say -- the whole value of the field is that a later
# selection can trust it.
KNOWN_MODELS: frozenset[str] = frozenset(
    {
        *ScoringModelName.__args__,
        "chai1",
        "af3",
        "of3_upstream",
        "protenix_mini",
        "protenix_base",
    }
)

# The artifact kind the scorer writes one row of per saved pose.
STRUCTURE_KIND = "predicted_structure"


class OptimizePreflightError(RuntimeError):
    """An optimization run is misconfigured or its inputs are missing."""


@dataclass(frozen=True, slots=True)
class OptimizePreflight:
    design_set: DesignSet
    fasta_path: Path
    n_designs: int
    distinct_lengths: int
    target_length: int
    target_sequence: str
    script_sha256: str
    dev_source_sha256: str | None
    # design-set index -> absolute pose path, empty unless `structure` was asked
    # for. Resolved at PLAN time and carried in the manifest, so a task launches
    # against the poses the run was planned with rather than whatever is on
    # disk when it starts.
    structures: dict[int, str]
    # design-set index -> {stored metric name: value}, empty unless asked for.
    metrics: dict[int, dict[str, float]]
    # 1-based positions in the target's FASTA. See `_resolve_hotspots`.
    hotspots: tuple[int, ...] = ()
    n_unfolded: int = 0


def preflight_optimize(general: GeneralConfig, model: OptimizeConfig) -> OptimizePreflight:
    manifest = Path(model.design_set)
    if not manifest.is_file():
        raise OptimizePreflightError(
            f"no design-set manifest at {manifest}. Build one with:\n"
            "  bindocracy designset build DB --out-dir sets/ "
            "--passed-filter <rule> --filter-run <run_id>"
        )
    try:
        design_set = DesignSet.read(manifest)
    except DesignSetError as error:
        raise OptimizePreflightError(str(error)) from error

    fasta = design_set.fasta_path(manifest)
    if not fasta.is_file():
        raise OptimizePreflightError(
            f"design-set manifest {manifest} has no FASTA beside it at {fasta}"
        )
    if design_set.n_designs == 0:
        raise OptimizePreflightError(f"design set {design_set.digest} is empty")

    script = Path(model.script)
    if not script.is_file():
        raise OptimizePreflightError(f"optimizer script not found at {script}")

    driver = Path(model.driver_script)
    if not driver.is_file():
        raise OptimizePreflightError(f"driver script not found at {driver}")

    if model.runtime.container is not None and not Path(model.runtime.container).exists():
        raise OptimizePreflightError(f"container not found at {model.runtime.container}")

    dev_source_sha256 = None
    if model.runtime.dev_source is not None:
        source = Path(model.runtime.dev_source)
        if not source.is_dir():
            raise OptimizePreflightError(f"dev_source is not a directory: {source}")
        dev_source_sha256 = _tree_digest(source)

    unknown = sorted(set(model.loss_models) - KNOWN_MODELS)
    if unknown:
        raise OptimizePreflightError(
            f"loss_models names {unknown}, which is not a model this campaign scores "
            f"with. Known: {sorted(KNOWN_MODELS)}.\n"
            "Refused on a typo rather than stored, because its only purpose is to let "
            "a later selection exclude the models that already had a say in the "
            "design -- and a misspelled name would exclude nothing while looking "
            "like it had."
        )

    target = Path(general.target.sequence_fasta)
    if not target.is_file():
        raise OptimizePreflightError(f"target FASTA not found at {target}")
    target_sequence = _read_single_sequence(target)

    structures, n_unfolded = _resolve_structures(model, design_set)
    metrics = _resolve_metrics(model, design_set)
    hotspots = _resolve_hotspots(general, len(target_sequence))

    return OptimizePreflight(
        design_set=design_set,
        fasta_path=fasta,
        n_designs=design_set.n_designs,
        distinct_lengths=design_set.distinct_lengths,
        target_length=len(target_sequence),
        target_sequence=target_sequence,
        script_sha256=sha256_file(script),
        dev_source_sha256=dev_source_sha256,
        structures=structures,
        metrics=metrics,
        hotspots=hotspots,
        n_unfolded=n_unfolded,
    )


def _resolve_hotspots(general: GeneralConfig, target_length: int) -> tuple[int, ...]:
    """The campaign epitope as 1-based positions in the target's FASTA.

    The mapping, its checks and the reasoning live in
    `config/preflight.py::target_hotspot_positions`, shared with the epitope
    function so that the epitope a run is planned against and the epitope it is
    later measured against cannot drift apart.
    """
    try:
        return target_hotspot_positions(
            general.target.hotspots,
            chain_id=general.target.chain_id,
            target_length=target_length,
            target_name=general.target.name,
        )
    except ConfigPreflightError as error:
        raise OptimizePreflightError(str(error)) from error


def _resolve_structures(
    model: OptimizeConfig, design_set: DesignSet
) -> tuple[dict[int, str], int]:
    """Parent poses, from the database the design set was built against.

    Resolved here rather than in the driver for the same reason the design set
    itself is frozen: a run must launch against the inputs it was planned with.
    A pose resolved at launch could be a different replicate of a rescoring run
    that landed in between, and nothing in the output would say so.

    The pose is chosen by `structures_from` (which model predicted it) and then
    the lowest replicate, which is the deterministic choice rather than the best
    one. Picking the *best* pose would mean ranking by a confidence metric here,
    which is a selection decision and belongs in a filter where it gets
    recorded.
    """
    if "structure" not in model.inputs:
        return {}, 0

    if model.structures_from is None:
        raise OptimizePreflightError(
            "inputs includes 'structure', so structures_from must name which model's "
            f"poses to use (one of {sorted(KNOWN_MODELS)}). Two models disagree about "
            "where a binder sits by a median 22.7 A, so an unqualified 'the structure' "
            "is not a well defined input."
        )

    database = Path(design_set.database)
    design_ids = [entry.design_id for entry in design_set.entries]
    by_design: dict[str, str] = {}
    try:
        with read_only(database) as connection:
            placeholders = ", ".join("?" for _ in design_ids)
            # `artifacts.uri` is RUN-DIRECTORY-RELATIVE, by the convention
            # `adapters/common.py::artifact` sets for every tool here -- which
            # is what lets a run directory be moved. So it has to be joined to
            # the producing run's `output_uri` to become a path. Reading the
            # column as a path is the mistake this join exists to prevent.
            # `json_extract_string` rather than `->>`, and the ordering column
            # is aliased `rep` rather than `replicate`: the latter is a DuckDB
            # built-in function name, and `ORDER BY replicate` resolves to the
            # function instead of the alias, failing with a cast error that
            # names the metadata column.
            rows = connection.execute(
                "SELECT a.design_id, r.output_uri, a.uri, "
                "COALESCE(json_extract_string(a.metadata, '$.replicate'), '0') AS rep "
                "FROM artifacts a JOIN runs r ON r.run_id = a.run_id "
                f"WHERE a.kind = ? AND a.design_id IN ({placeholders}) "
                "AND json_extract_string(a.metadata, '$.model') = ? "
                "ORDER BY a.design_id, rep",
                [STRUCTURE_KIND, *design_ids, model.structures_from],
            ).fetchall()
    except SelectionError as error:
        raise OptimizePreflightError(
            f"cannot resolve parent structures: {error}. The design set records the "
            f"database it was built from ({database}); it must still be readable."
        ) from error

    for design_id, output_uri, uri, _ in rows:
        # `Path(base) / absolute` yields the absolute path, so this is correct
        # whether the stored uri is relative (the convention) or absolute.
        by_design.setdefault(design_id, str(Path(output_uri or "") / uri))

    structures: dict[int, str] = {}
    missing: list[str] = []
    for entry in design_set.entries:
        uri = by_design.get(entry.design_id)
        if uri is None:
            missing.append(entry.native_id)
            continue
        if not Path(uri).is_file():
            missing.append(f"{entry.native_id} (row points at a missing file)")
            continue
        structures[entry.index] = str(Path(uri).resolve())

    if missing and not model.allow_unfolded:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise OptimizePreflightError(
            f"{len(missing)} of {design_set.n_designs} parents have no "
            f"{model.structures_from!r} pose on disk: {shown}\n"
            "Score the set with that model first, or set allow_unfolded: true to "
            "optimize only the parents that have one. Refusing by default because a "
            "run that silently optimizes a subset reports a smaller number with no "
            "row saying why."
        )
    return structures, len(missing)


def _resolve_metrics(
    model: OptimizeConfig, design_set: DesignSet
) -> dict[int, dict[str, float]]:
    """The parent metrics a script asked for, averaged over replicates.

    Mean, and stated here rather than configurable, because these are context
    for a decision the script makes rather than a threshold: `filters/` is
    where an aggregation is a choice worth recording. A script that needs the
    spread should read the database itself.
    """
    if "metrics" not in model.inputs:
        return {}

    database = Path(design_set.database)
    design_ids = [entry.design_id for entry in design_set.entries]
    names = list(model.metric_inputs)
    index_of = {entry.design_id: entry.index for entry in design_set.entries}

    with read_only(database) as connection:
        placeholders = ", ".join("?" for _ in design_ids)
        name_placeholders = ", ".join("?" for _ in names)
        rows = connection.execute(
            "SELECT design_id, name, avg(value) FROM metrics "
            f"WHERE design_id IN ({placeholders}) AND name IN ({name_placeholders}) "
            "AND status = 'ok' AND value IS NOT NULL "
            "GROUP BY 1, 2",
            [*design_ids, *names],
        ).fetchall()
        available = {
            row[0]
            for row in connection.execute(
                f"SELECT DISTINCT name FROM metrics WHERE name IN ({name_placeholders})",
                names,
            ).fetchall()
        }

    unknown = sorted(set(names) - available)
    if unknown:
        raise OptimizePreflightError(
            f"metric_inputs names {unknown}, which this database has no rows for. "
            "Stored names carry their scorer prefix; `bindocracy filter metrics DB` "
            "lists what exists."
        )

    resolved: dict[int, dict[str, float]] = {}
    for design_id, name, value in rows:
        resolved.setdefault(index_of[design_id], {})[name] = float(value)
    return resolved


def _read_single_sequence(path: Path) -> str:
    """The one sequence in a target FASTA, refusing a multi-record file."""
    records: list[list[str]] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            records.append([])
        elif line.strip() and records:
            records[-1].append(line.strip())
    if len(records) != 1:
        raise OptimizePreflightError(
            f"{path} holds {len(records)} sequences; a target FASTA must hold one"
        )
    return "".join(records[0]).upper()


def _tree_digest(source: Path) -> str:
    """One digest over every .py under a bound source tree.

    Same purpose as the scorer's: when an image is bound over, the container
    digest no longer determines the result, so the bytes that replaced it have
    to be recorded instead.
    """
    parts = [
        f"{path.relative_to(source)}:{sha256_file(path)}"
        for path in sorted(source.rglob("*.py"))
        if path.is_file()
    ]
    return sha256_text("\n".join(parts))
