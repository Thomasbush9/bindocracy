"""Portable JSONL callbacks for scoring and optimization (standard library only).

The harness supplies this module beside archived scripts. Scripts own model
loading and add their options to an ArgumentParser; this module owns transport
and row identity. Raise RejectCandidate for an expected, per-candidate refusal.
Every other exception propagates so programming or model failures fail the job.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


class RejectCandidate(Exception):
    """An expected refusal, recorded as a failed row instead of aborting the job."""


def _arguments(parser, argv, *, optimization):
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    if optimization:
        parser.add_argument("--context", required=True, type=Path)
    args = parser.parse_args(argv)
    sources = [args.inputs]
    if optimization:
        sources.append(args.context)
    for source in sources:
        if source.resolve() == args.outputs.resolve() or (
            args.outputs.exists() and source.samefile(args.outputs)
        ):
            raise ValueError("output path must not overwrite an input or context file")
    return args


def _inputs(path):
    seen = set()
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"input line {number}: expected a JSON object")
            index = row.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError(f"input line {number}: index must be a nonnegative integer")
            if index in seen:
                raise ValueError(f"input line {number}: duplicate index {index}")
            seen.add(index)
            yield row


def _metrics(metrics):
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary of finite numbers")
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metric {name!r} must be a finite number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be a finite number")
    return metrics


def _write(sink, row):
    sink.write(json.dumps(row, allow_nan=False) + "\n")
    sink.flush()


def _failure(sink, identity, error):
    reason = str(error).strip()
    if not reason:
        raise ValueError("RejectCandidate requires a nonempty reason") from error
    _write(sink, {**identity, "failed": reason})


def run_scoring(score, *, parser=None, argv=None):
    """Call ``score(candidate, args) -> metric dict`` for each input row.

    Add script-specific options to ``parser``; --inputs and --outputs belong
    to this helper. Model initialization remains the caller's responsibility.
    Returns zero on completion, including explicitly rejected candidates.
    """
    args = _arguments(parser, argv, optimization=False)
    with args.outputs.open("w", encoding="utf-8") as sink:
        for candidate in _inputs(args.inputs):
            identity = {"index": candidate["index"]}
            try:
                metrics = score(candidate, args)
            except RejectCandidate as error:
                _failure(sink, identity, error)
            else:
                _write(sink, {**identity, "metrics": _metrics(metrics)})
    return 0


def run_optimization(optimize, *, parser=None, argv=None):
    """Call ``optimize(parent, context, args) -> iterable of child dicts``.

    Children contain sequence and optional metrics/structure/trajectory/seconds.
    Identity is owned here: parent_index and zero-based child ordinals are attached
    in emission order. Empty iterables deliberately emit nothing. An explicit
    rejection writes a parent failure row; already flushed children remain if a
    generator rejects or crashes later. Sequence and artifact policies remain
    the host optimizer contract's responsibility, not this transport's.
    """
    args = _arguments(parser, argv, optimization=True)
    with args.context.open(encoding="utf-8") as source:
        context = json.load(source)
    if not isinstance(context, dict):
        raise TypeError("context must be a JSON object")
    with args.outputs.open("w", encoding="utf-8") as sink:
        for parent in _inputs(args.inputs):
            identity = {"parent_index": parent["index"]}
            try:
                for ordinal, child in enumerate(optimize(parent, context, args)):
                    if not isinstance(child, dict):
                        raise TypeError("an optimizer child must be a dictionary")
                    unexpected = child.keys() - {
                        "sequence", "metrics", "structure", "trajectory", "seconds"
                    }
                    if unexpected:
                        raise ValueError(f"unsupported optimizer child fields: {unexpected}")
                    if "sequence" not in child:
                        raise ValueError("an optimizer child requires a sequence")
                    _metrics(child.get("metrics", {}))
                    _write(sink, {**child, **identity, "child": ordinal})
            except RejectCandidate as error:
                _failure(sink, identity, error)
    return 0
