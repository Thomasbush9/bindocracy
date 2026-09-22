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


def target_hotspot_positions(
    residues: tuple[str, ...],
    *,
    chain_id: str,
    target_length: int,
    target_name: str,
) -> tuple[int, ...]:
    """The campaign epitope as 1-based positions in the target's own FASTA.

    `target.hotspots` is author numbering with an optional chain (`A110`),
    which is how a crystal structure numbers it. What anything reading a
    *predicted* pose can use is a position in the sequence it was handed,
    because residue ids in a prediction are positional -- 0..200 for a
    201-residue target -- and carry none of the author numbering.

    The mapping is assumed to be the identity and then CHECKED rather than
    assumed silently. A hotspot on another chain, or past the end of the
    sequence, proves it is not the identity here, and that raises instead of
    scoring a confident zero later: an epitope measured on the wrong numbering
    is not a weaker measurement, it is a wrong one that looks like a result.

    Shared because three stages need the same mapping from the same field --
    optimization plans with it, the epitope function measures against it, and a
    filter gates on what that function produced. Three copies of this would be
    three chances for them to disagree about what the epitope is.
    """
    if not residues:
        return ()

    parsed = parse_hotspots(residues)
    chain = chain_id.upper()
    wrong_chain = sorted({spot.chain for spot in parsed if spot.chain and spot.chain != chain})
    if wrong_chain:
        raise ConfigPreflightError(
            f"target.hotspots names chain(s) {wrong_chain} but the target chain is "
            f"{chain!r}; an epitope on another chain is not this target's epitope"
        )

    numbers = sorted({spot.number for spot in parsed})
    out_of_range = [number for number in numbers if not 1 <= number <= target_length]
    if out_of_range:
        raise ConfigPreflightError(
            f"hotspot(s) {out_of_range} fall outside the target's 1..{target_length} "
            "residues, so the campaign's author numbering is not the same as FASTA "
            f"position for this target ({target_name}).\n"
            "Map them against target.structure_pdb and write FASTA positions in the "
            "config. Refusing rather than clamping: an epitope measured on the wrong "
            "numbering reads as a real miss."
        )
    return tuple(numbers)


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
