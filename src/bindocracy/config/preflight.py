"""Filesystem and content checks performed after structural validation.

What lives here is what more than one tool needs and no tool owns: reading the
campaign target, reading the epitope it names, and checking that an alignment
is an alignment *of that target*. Anything a single tool needs stays in that
tool's own preflight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class ConfigPreflightError(ValueError):
    """A structurally valid configuration cannot run in this environment."""


@dataclass(frozen=True)
class Hotspot:
    """One residue of the campaign's epitope, as the campaign writes it."""

    chain: str
    number: int

    def __str__(self) -> str:
        return f"{self.chain}{self.number}"


_HOTSPOT = re.compile(r"(?P<chain>[A-Za-z]*)(?P<number>\d+)")


def parse_hotspots(residues: tuple[str, ...]) -> tuple[Hotspot, ...]:
    """Read `target.hotspots` into chains and numbers, or refuse.

    Every tool that conditions on an epitope needs these, and every one of them
    expresses the epitope differently -- integers on one chain, `B110` after a
    renumbering, a range string. Parsing is shared; the mapping is not.
    """
    parsed = []
    for residue in residues:
        match = _HOTSPOT.fullmatch(str(residue).strip())
        if match is None:
            raise ConfigPreflightError(
                f"cannot read a residue from hotspot {residue!r}; expected a "
                "number, optionally prefixed by a chain, such as 'A110'"
            )
        parsed.append(
            Hotspot(chain=match.group("chain").upper(), number=int(match.group("number")))
        )
    return tuple(parsed)


def hotspot_numbers(residues: tuple[str, ...]) -> set[int]:
    """The residue numbers of an epitope, with the chain dropped.

    What is comparable between a campaign and a tool that renumbers or renames
    chains -- which most of them do.
    """
    return {hotspot.number for hotspot in parse_hotspots(residues)}


def require_alignment_of(msa: Path, target_sequence: str, *, described_as: str) -> None:
    """Refuse an alignment whose query is not the campaign target.

    An a3m for a different protein is a well-formed file that produces a
    normal-looking run against the wrong target. The query is the first record;
    in a3m the insertions relative to it are lower case and gaps are dashes, so
    stripping both is what recovers the sequence it aligns.
    """
    query = _first_record(msa)
    if query is None:
        raise ConfigPreflightError(f"{described_as} has no sequences: {msa}")
    if query != target_sequence:
        raise ConfigPreflightError(
            f"{described_as} is not an alignment of the campaign target.\n"
            f"  campaign: {len(target_sequence)} aa\n"
            f"  {msa}: {len(query)} aa as its query\n"
            "An alignment of another protein is a well-formed file that folds "
            "the wrong target and looks entirely normal."
        )


def _first_record(msa: Path) -> str | None:
    """The query sequence of an a3m, normalized to plain residues."""
    body: list[str] = []
    seen_header = False
    with msa.open() as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                if seen_header:
                    break
                seen_header = True
            elif seen_header and line:
                body.append(line)
    if not seen_header:
        return None
    # Lower case is an insertion relative to the query and `-`/`.` are gaps;
    # neither belongs to the sequence the alignment is of.
    query = re.sub(r"[a-z.\-]", "", "".join(body)).upper()
    return query or None


def read_single_fasta(path: Path) -> str:
    """Read one FASTA sequence and return an uppercase, whitespace-free string."""
    headers = 0
    sequence_lines: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            headers += 1
            continue
        sequence_lines.append(line)

    if headers > 1:
        raise ConfigPreflightError(f"target FASTA must contain one sequence: {path}")

    sequence = re.sub(r"\s+", "", "".join(sequence_lines)).upper()
    if not sequence or re.fullmatch(r"[A-Z]+", sequence) is None:
        raise ConfigPreflightError(f"target FASTA has no valid amino-acid sequence: {path}")
    return sequence
