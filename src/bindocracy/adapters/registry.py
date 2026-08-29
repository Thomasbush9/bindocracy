"""Map a tool name to the adapter that reads that tool's output.

This is the seam that keeps tools independent. An adapter knows how to parse
its own run directory and nothing else; it does not import this module, and it
does not know which other tools exist. The registry is the one place that knows
the set, so adding a tool is a new module plus one line here, and removing one
is deleting both.

Collection dispatches on the `tool` recorded in the run manifest rather than on
anything the caller passes, so a run can only ever be parsed by the adapter for
the tool that produced it.
"""

from __future__ import annotations

from pathlib import Path

from bindocracy.adapters.base import OutputAdapter
from bindocracy.adapters.mosaic import MosaicOutputAdapter
from bindocracy.runs.manifest import RunManifest
from bindocracy.store.records import CollectedRun


class UnknownToolError(KeyError):
    """No adapter is registered for the tool that produced this run."""


_ADAPTERS: dict[str, type[OutputAdapter]] = {}


def register(adapter_type: type[OutputAdapter]) -> type[OutputAdapter]:
    """Register one adapter class under its own `tool` name."""
    if not adapter_type.tool:
        raise ValueError(f"{adapter_type.__name__} must define a tool name")
    _ADAPTERS[adapter_type.tool] = adapter_type
    return adapter_type


def unregister(tool: str) -> None:
    """Remove a tool. Adapters for other tools are unaffected."""
    _ADAPTERS.pop(tool, None)


def registered_tools() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def adapter_for(tool: str) -> OutputAdapter:
    """Return a fresh adapter instance for one tool."""
    try:
        adapter_type = _ADAPTERS[tool]
    except KeyError as error:
        known = ", ".join(registered_tools()) or "none"
        raise UnknownToolError(f"no adapter registered for {tool!r}; known: {known}") from error
    return adapter_type()


def collect_run(manifest_path: str | Path) -> CollectedRun:
    """Collect the run described by one `run.json`, whatever tool made it."""
    manifest = RunManifest.read(manifest_path)
    return adapter_for(manifest.tool).collect(manifest.directory, manifest.to_run_record())


register(MosaicOutputAdapter)
