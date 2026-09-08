"""Freezing a set of designs into a file a run can be planned against.

DRAFT -- not wired into anything yet. See docs/scoring-stage.md.

This is the one genuinely new contract the scoring stage needs. Every
generation tool's inputs are files named in its config, digested by `plan_run`
and verified before launch. A scorer's input is a *query result*, which has
none of those properties: it changes whenever a generation run lands, it cannot
be digested, and it cannot be re-read months later to find out what was scored.

So a design set is materialised to disk before any run is planned, and from
that point it behaves like every other input:

    design_sets/<digest>.fasta    what the scorer reads
    design_sets/<digest>.json     what it means -- query, counts, provenance

The file is content-addressed, which buys three things at once. Two scoring
runs over the same candidates share one file and one identity. A rank decision
gets a `scope_id` that genuinely names the candidate set it ranked, which the
schema requires and which is otherwise easy to fake. And re-running the same
query after new designs land produces a *different* digest, so the change is
visible instead of silent.

Nothing here touches Snakemake. Building a design set is a CLI step the user
runs once; the scoring run that follows is an ordinary run whose config names
the resulting path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from bindocracy.store.query import DesignQuery, DesignRow
from bindocracy.store.records import canonical_json, sha256_text, stable_id, utc_now

# Wrap FASTA sequence lines at the usual width. Binders here are 60-149
# residues, so this is one or two lines each and the file stays greppable.
FASTA_WIDTH = 60


class DesignSetError(RuntimeError):
    """A design set is empty, malformed, or inconsistent with its manifest."""


class DesignSetEntry(BaseModel):
    """One design in a frozen set.

    `index` is the design's position in the set and is what a sharded scorer
    reports back, so a task's output can be joined to a design without the
    driver ever knowing a `design_id`. Drivers run inside a container and know
    nothing about the database; this is the whole of what crosses that line.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    design_id: str
    sequence: str
    length: int = Field(gt=0)
    tool: str
    run_name: str
    native_id: str


class DesignSet(BaseModel):
    """The manifest beside a design-set FASTA.

    Carries the query that produced it, not just the result. A set that records
    only its members cannot answer "was this every design, or only the ones
    somebody thought were promising", and that distinction is the difference
    between a benchmark and a selection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    digest: str
    scope_id: str
    database: str
    query: dict[str, Any]
    created_at: datetime
    n_designs: int = Field(ge=0)
    by_tool: dict[str, int]
    length_range: tuple[int, int] | None
    distinct_lengths: int
    entries: tuple[DesignSetEntry, ...]

    @classmethod
    def read(cls, path: str | Path) -> Self:
        manifest = Path(path)
        if not manifest.is_file():
            raise DesignSetError(f"missing design-set manifest: {manifest}")
        return cls.model_validate_json(manifest.read_text())

    def fasta_path(self, manifest_path: str | Path) -> Path:
        """The FASTA beside a manifest, by convention rather than by record.

        Storing the sibling path inside the manifest would make the pair
        un-relocatable, which is exactly the mistake `known-issues.md` records
        for the Mosaic scripts that pointed at a moved directory.
        """
        return Path(manifest_path).with_suffix(".fasta")

    def shard(self, shard: int, num_shards: int) -> tuple[DesignSetEntry, ...]:
        """This shard's entries, as a contiguous block.

        Contiguous, not strided. Entries are ordered by length, so a contiguous
        block spans few binder lengths and therefore few JAX recompilations; a
        strided shard would hand every task the full spread. On this campaign
        that is 78 distinct lengths split across tasks instead of 78 in each.
        """
        if not 0 <= shard < num_shards:
            raise ValueError(f"shard {shard} outside 0..{num_shards - 1}")
        total = len(self.entries)
        per = -(-total // num_shards)  # ceiling division
        return self.entries[shard * per : (shard + 1) * per]


def build_design_set(
    designs: Sequence[DesignRow],
    *,
    database: str | Path,
    query: DesignQuery,
    created_at: datetime | None = None,
) -> DesignSet:
    """Turn selected rows into a frozen, content-addressed set.

    The digest covers the members and their order, and nothing else -- not the
    timestamp, not the database path. Two campaigns that select the same
    designs get the same digest, and re-running the same query one minute later
    is recognisably the same set.
    """
    if not designs:
        raise DesignSetError("a design set must contain at least one design")

    entries = tuple(
        DesignSetEntry(
            index=index,
            design_id=design.design_id,
            sequence=design.sequence,
            length=design.length,
            tool=design.tool,
            run_name=design.run_name,
            native_id=design.native_id,
        )
        for index, design in enumerate(designs)
    )

    digest = sha256_text(
        canonical_json([[entry.design_id, entry.sequence] for entry in entries])
    )
    lengths = [entry.length for entry in entries]
    by_tool: dict[str, int] = {}
    for entry in entries:
        by_tool[entry.tool] = by_tool.get(entry.tool, 0) + 1

    return DesignSet(
        digest=digest,
        scope_id=stable_id("design-set", digest),
        database=str(database),
        query=query.model_dump(mode="json"),
        created_at=created_at or utc_now(),
        n_designs=len(entries),
        by_tool=dict(sorted(by_tool.items())),
        length_range=(min(lengths), max(lengths)),
        distinct_lengths=len(set(lengths)),
        entries=entries,
    )


def write_design_set(design_set: DesignSet, directory: str | Path) -> tuple[Path, Path]:
    """Write `<digest>.fasta` and `<digest>.json`, returning both paths.

    Refuses to overwrite a manifest whose digest matches but whose content
    differs, which can only mean a hash collision or a corrupted file -- either
    way not something to paper over.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    stem = design_set.digest.removeprefix("sha256:")[:16]
    fasta_path = out / f"{stem}.fasta"
    manifest_path = out / f"{stem}.json"

    if manifest_path.is_file():
        existing = DesignSet.read(manifest_path)
        if existing.digest != design_set.digest:
            raise DesignSetError(
                f"{manifest_path} holds digest {existing.digest}, not {design_set.digest}"
            )
        return fasta_path, manifest_path

    fasta_path.write_text(_render_fasta(design_set))
    manifest_path.write_text(design_set.model_dump_json(indent=2) + "\n")
    return fasta_path, manifest_path


def read_fasta_entries(path: str | Path) -> tuple[tuple[str, str], ...]:
    """Parse a design-set FASTA into (header, sequence) pairs.

    Used by the in-container driver, which has no access to pydantic or to this
    package -- so keep this function importable in isolation and free of any
    dependency a container might not have.
    """
    header: str | None = None
    chunks: list[str] = []
    entries: list[tuple[str, str]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(chunks)))
            header, chunks = line[1:], []
        else:
            chunks.append(line)
    if header is not None:
        entries.append((header, "".join(chunks)))
    return tuple(entries)


def _render_fasta(design_set: DesignSet) -> str:
    """One record per design, headed by its index and its provenance.

    The index comes first because it is the join key. Everything after it is
    for the human who opens the file, and a driver must not parse it: a tool
    name is not stable enough to key on and a `design_id` has no business
    inside a container.
    """
    lines: list[str] = []
    for entry in design_set.entries:
        lines.append(
            f">{entry.index:06d} tool={entry.tool} run={entry.run_name} "
            f"native={entry.native_id} len={entry.length}"
        )
        lines.extend(
            entry.sequence[start : start + FASTA_WIDTH]
            for start in range(0, len(entry.sequence), FASTA_WIDTH)
        )
    return "\n".join(lines) + "\n"


def summarise(design_set: DesignSet) -> str:
    """A few lines a person can read before committing a GPU to this set."""
    low, high = design_set.length_range or (0, 0)
    tools = ", ".join(f"{tool} {count}" for tool, count in design_set.by_tool.items())
    return "\n".join(
        (
            f"design set {design_set.digest.removeprefix('sha256:')[:16]}",
            f"  designs          {design_set.n_designs}",
            f"  tools            {tools}",
            f"  length           {low}-{high} ({design_set.distinct_lengths} distinct)",
            f"  scope_id         {design_set.scope_id}",
        )
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a manifest as plain JSON, for callers that cannot import pydantic."""
    return json.loads(Path(path).read_text())
