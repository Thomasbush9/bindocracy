"""The exact rows a custom optimizer reads and writes.

One module, so a person writing a script has one file to read and the harness
and the driver cannot drift apart about what a row means.

**Why this is not the scoring-function contract.** A function is 1:1 on
`index`: one design in, one set of numbers out. An optimizer is n->m -- ten
parents can yield forty children, or three, or none -- so a row is keyed by
`(parent_index, child)` and the harness cannot assume a child exists for every
parent. That difference is the whole reason for a separate shape rather than a
reused one.

**Why the run-level fields live in their own file.** A function is handed
`target_sequence` on every line, which was tolerable for a metric read off one
pose. An optimizer needs the target, its alignment, the epitope, a seed, and
somewhere to write structures -- none of which vary per design. Repeating them
on every line would be a hundred copies of an a3m path and an invitation for a
script to read one parent's copy and apply it to another.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# What a script may ask to be handed per parent. Anything not asked for is
# absent from the row, so a script cannot come to depend on an input it did not
# declare -- which is what lets the harness know an optimizer needs no
# structures and can therefore run over designs nothing has folded.
OptimizerInputKind = Literal["sequence", "structure", "metrics", "provenance"]

# The 20 canonical amino acids. Deliberately not a broader alphabet: `X` in an
# optimized binder is a gap the optimizer failed to fill, `U`/`O` are not
# synthesizable by the ordinary route, and `B`/`Z`/`J` are ambiguity codes that
# mean the script did not decide. A design table is downstream of this, and a
# sequence that cannot be ordered is not a candidate.
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
_SEQUENCE = re.compile(rf"^[{CANONICAL_AA}]+$")

INPUT_FILE = "inputs.jsonl"
OUTPUT_FILE = "children.jsonl"
CONTEXT_FILE = "context.json"


class ContractError(ValueError):
    """An optimizer returned something the harness will not store."""


@dataclass(frozen=True, slots=True)
class OptimizerInput:
    """One parent design as the script sees it."""

    index: int
    sequence: str
    length: int
    structure: Path | None = None
    source_model: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    tool: str | None = None
    run_name: str | None = None
    native_id: str | None = None

    def as_row(self, wants: Iterable[str]) -> dict[str, Any]:
        """Only what the script asked for.

        `index` and `length` are always present: the first is the join key and
        the second is the one property of a parent an optimizer cannot get
        wrong by recomputing, since a length that disagrees with the sequence is
        how a shard silently scores the wrong design.
        """
        wanted = set(wants)
        row: dict[str, Any] = {"index": self.index, "length": self.length}
        if "sequence" in wanted:
            row["sequence"] = self.sequence
        if "structure" in wanted and self.structure is not None:
            row["structure"] = str(self.structure)
            row["source_model"] = self.source_model
        if "metrics" in wanted:
            row["metrics"] = dict(self.metrics)
        if "provenance" in wanted:
            row["tool"] = self.tool
            row["run_name"] = self.run_name
            row["native_id"] = self.native_id
        return row


@dataclass(frozen=True, slots=True)
class OptimizerContext:
    """Everything about the run that does not vary per design.

    Written once as `context.json` and passed with `--context`.
    """

    target_sequence: str
    target_chain: str
    seed: int
    max_children: int
    work_dir: Path
    structure_dir: Path
    shard: int
    num_shards: int
    n_parents: int
    target_msa: Path | None = None
    target_structure: Path | None = None
    # 1-based positions in the TARGET's FASTA, never author numbering and never
    # a chain letter. Same convention the epitope scoring function settled on,
    # and for the same two reasons: residue ids in a predicted pose are
    # positional, and the drivers here do not agree on which chain is the
    # target. See docs/scoring-functions.md.
    hotspots: tuple[int, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "target_sequence": self.target_sequence,
            "target_chain": self.target_chain,
            "target_msa": str(self.target_msa) if self.target_msa else None,
            "target_structure": (
                str(self.target_structure) if self.target_structure else None
            ),
            "hotspots": list(self.hotspots),
            "seed": self.seed,
            "max_children": self.max_children,
            "work_dir": str(self.work_dir),
            "structure_dir": str(self.structure_dir),
            "shard": self.shard,
            "num_shards": self.num_shards,
            "n_parents": self.n_parents,
        }


@dataclass(frozen=True, slots=True)
class OptimizerOutput:
    """One child the script produced, or one parent it could not optimize."""

    parent_index: int
    child: int = 0
    sequence: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    structure: str | None = None
    trajectory: str | None = None
    seconds: float | None = None
    failed: str | None = None

    @property
    def is_failure(self) -> bool:
        return self.failed is not None


def write_inputs(path: Path, inputs: Sequence[OptimizerInput], wants: Iterable[str]) -> int:
    wants = tuple(wants)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for item in inputs:
            handle.write(json.dumps(item.as_row(wants), sort_keys=True) + "\n")
    return len(inputs)


def write_context(path: Path, context: OptimizerContext) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context.as_json(), indent=2, sort_keys=True) + "\n")
    return path


def normalize_sequence(raw: Any, *, where: str) -> str:
    """A child's sequence, or a refusal naming what was wrong with it.

    Strict, because this writes to the `designs` table. A bad custom scoring
    function adds a useless column that a query can ignore; a bad custom
    optimizer puts sequences that look like real candidates into the campaign
    permanently, and nothing downstream can tell them from designs a model
    produced.
    """
    if not isinstance(raw, str):
        raise ContractError(f"{where}: sequence must be a string, got {type(raw).__name__}")
    sequence = "".join(raw.split()).upper()
    if not sequence:
        raise ContractError(f"{where}: sequence is empty")
    if not _SEQUENCE.match(sequence):
        bad = sorted({character for character in sequence if character not in CANONICAL_AA})
        raise ContractError(
            f"{where}: sequence contains {bad} which are not among the 20 canonical "
            f"amino acids ({CANONICAL_AA}). An 'X' is an unfilled position rather "
            "than a residue, and B/Z/J are ambiguity codes; either way it is not "
            "something the campaign can order."
        )
    return sequence


def read_outputs(
    path: Path, *, n_parents: int, max_children: int, declared: Iterable[str]
) -> tuple[tuple[OptimizerOutput, ...], dict[str, Any]]:
    """Parse and check an optimizer's output file.

    **A bad row is rejected and counted; it does not abort the file.** That
    split matters. One non-canonical sequence out of five hundred children is a
    bug in one branch of a script, and throwing away the other 499 would cost a
    GPU-day to punish a typo. What is never done is the third option -- storing
    it anyway -- because a design table downstream of this cannot tell a
    silently-accepted row from a real candidate.

    So: `rejected` counts why, the run's `count_details` carries it, and the
    good children are stored. Only a file-level problem raises, because there is
    then nothing to salvage and nothing true to report.

    The five row-level rejections, each because the tidy version is
    indistinguishable from a real result:

    * `parent_index` outside this shard -- the child would attach to the wrong
      parent
    * a duplicate `(parent_index, child)` -- one of the two is lost, and which
      one depends on file order
    * `child` at or beyond `max_children` -- a runaway loop looks productive
    * an undeclared metric -- stored with no direction, it sorts backwards
    * a non-canonical sequence -- see `normalize_sequence`

    A torn final line is counted separately and the rest kept: that is what a
    killed job leaves behind, and it is the one case where salvaging is not a
    judgement about the script at all.
    """
    declared_metrics = set(declared)
    rows: list[OptimizerOutput] = []
    counts: dict[str, Any] = {
        "n_lines": 0,
        "n_children": 0,
        "n_failed": 0,
        "n_torn_lines": 0,
        "rejected": {},
    }
    seen: set[tuple[int, int]] = set()

    if not path.is_file():
        raise ContractError(f"optimizer wrote no output file at {path}")

    def reject(reason: str) -> None:
        counts["rejected"][reason] = counts["rejected"].get(reason, 0) + 1

    for number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            counts["n_torn_lines"] += 1
            continue
        counts["n_lines"] += 1
        where = f"line {number}"

        if not isinstance(row, dict):
            reject("not_an_object")
            continue

        parent = row.get("parent_index")
        if not isinstance(parent, int) or isinstance(parent, bool):
            reject("parent_index_not_an_integer")
            continue
        if not 0 <= parent < n_parents:
            # Indices are positions in the shard this task was handed, not in
            # the whole design set. Getting that wrong is the most likely first
            # mistake in a new script, so it is named rather than counted alone.
            reject("parent_index_out_of_range")
            continue

        if row.get("failed"):
            rows.append(OptimizerOutput(parent_index=parent, failed=str(row["failed"])[:200]))
            counts["n_failed"] += 1
            continue

        child = row.get("child", 0)
        if not isinstance(child, int) or isinstance(child, bool) or child < 0:
            reject("child_not_a_non_negative_integer")
            continue
        if child >= max_children:
            # Keyed on the ordinal rather than on arrival order, so which rows
            # are "excess" does not depend on how the file was written.
            reject("child_beyond_max_children")
            continue
        if (parent, child) in seen:
            reject("duplicate_parent_child")
            continue

        try:
            sequence = normalize_sequence(row.get("sequence"), where=where)
        except ContractError:
            reject("sequence_not_canonical")
            continue

        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict):
            reject("metrics_not_an_object")
            continue
        if set(metrics) - declared_metrics:
            reject("undeclared_metric")
            continue
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in metrics.values()
        ):
            reject("metric_not_a_number")
            continue

        try:
            structure = _relative(row.get("structure"), where=where)
            trajectory = _relative(row.get("trajectory"), where=where)
        except ContractError:
            reject("path_not_relative_to_structure_dir")
            continue

        seen.add((parent, child))
        rows.append(
            OptimizerOutput(
                parent_index=parent,
                child=child,
                sequence=sequence,
                metrics={key: float(value) for key, value in metrics.items()},
                structure=structure,
                trajectory=trajectory,
                seconds=(
                    float(row["seconds"])
                    if isinstance(row.get("seconds"), (int, float))
                    and not isinstance(row.get("seconds"), bool)
                    else None
                ),
            )
        )
        counts["n_children"] += 1

    return tuple(rows), counts


def _relative(value: Any, *, where: str) -> str | None:
    """A structure or trajectory path, which must be relative to structure_dir.

    Absolute paths are refused because they do not survive the run directory
    being moved, which `known-issues.md` records as having already cost this
    campaign a day when the Mosaic scripts pointed at a relocated tree.
    """
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    if Path(text).is_absolute():
        raise ContractError(
            f"{where}: {text!r} is absolute; write it under the context's "
            "structure_dir and report the relative path, so the run directory "
            "can be moved"
        )
    if ".." in Path(text).parts:
        raise ContractError(f"{where}: {text!r} escapes structure_dir")
    return text
