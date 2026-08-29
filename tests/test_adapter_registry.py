"""The tool seam: adding or removing a tool must not disturb the others.

The test that matters here is `test_a_new_tool_needs_no_changes_to_mosaic`. It
defines a whole second tool inside the test — a different output format, a
different adapter — and drives it through the same collection path. If someone
later reintroduces a `if tool == "mosaic"` branch anywhere between the manifest
and the staging bundle, that test is what fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import design_line, write_task

from bindocracy.adapters import (
    OutputAdapter,
    UnknownToolError,
    adapter_for,
    collect_run,
    register,
    registered_tools,
    unregister,
)
from bindocracy.adapters.mosaic import MosaicOutputAdapter
from bindocracy.config import load_mosaic_configs
from bindocracy.runs import RunManifest, plan_mosaic_run
from bindocracy.store.records import CandidateType, CollectedRun, DesignRecord, RunStatus


class ToyAdapter(OutputAdapter):
    """A second tool with an unrelated output format: one sequence per line."""

    tool = "toy"

    def collect(self, run_dir: Path, run) -> CollectedRun:
        lines = (run_dir / "toy_output.txt").read_text().split()
        designs = tuple(
            DesignRecord(
                run_id=run.run_id,
                native_id=f"toy-{index}",
                candidate_type=CandidateType.SEQUENCE,
                sequence=sequence,
            )
            for index, sequence in enumerate(lines)
        )
        return CollectedRun(
            run=run.model_copy(update={
                "status": RunStatus.SUCCEEDED, "n_produced": len(designs)
            }),
            designs=designs,
        )

    def succeeded(self, run_dir: Path) -> bool:
        return (run_dir / "toy_output.txt").is_file()


@pytest.fixture
def isolated_registry():
    """Restore the process-wide registry, so tests cannot leak tools."""
    before = registered_tools()
    yield
    for tool in registered_tools():
        if tool not in before:
            unregister(tool)


@pytest.fixture
def toy_run(configs, tmp_path: Path) -> Path:
    """A run directory whose manifest says the tool is `toy`, not `mosaic`."""
    manifest = plan_mosaic_run(load_mosaic_configs(*configs), tmp_path / "run")
    toy = manifest.model_copy(update={"tool": "toy"})
    (manifest.directory / "run.json").write_text(toy.model_dump_json(indent=2))
    (manifest.directory / "toy_output.txt").write_text("ACDEFG\nHIKLMN\n")
    return manifest.directory / "run.json"


def test_mosaic_is_registered_by_default() -> None:
    assert "mosaic" in registered_tools()
    assert isinstance(adapter_for("mosaic"), MosaicOutputAdapter)


def test_an_unknown_tool_says_what_is_known() -> None:
    with pytest.raises(UnknownToolError, match="no adapter registered for 'nope'"):
        adapter_for("nope")
    with pytest.raises(UnknownToolError, match="mosaic"):
        adapter_for("nope")


def test_a_new_tool_needs_no_changes_to_mosaic(isolated_registry, toy_run: Path) -> None:
    """One register() call is the entire cost of adding a tool."""
    register(ToyAdapter)

    collected = collect_run(toy_run)

    assert collected.run.tool == "toy"
    assert [design.native_id for design in collected.designs] == ["toy-0", "toy-1"]
    assert collected.run.n_produced == 2
    # Mosaic is untouched by the arrival of another tool.
    assert isinstance(adapter_for("mosaic"), MosaicOutputAdapter)


def test_removing_a_tool_leaves_the_others_working(
    isolated_registry, configs, tmp_path: Path
) -> None:
    register(ToyAdapter)
    unregister("toy")

    assert "toy" not in registered_tools()
    assert "mosaic" in registered_tools()

    manifest = plan_mosaic_run(load_mosaic_configs(*configs), tmp_path / "run")
    write_task(manifest.directory, 0, [design_line(0, 0)], status={})
    write_task(manifest.directory, 1, [design_line(1, 0)], status={})
    assert len(collect_run(manifest.directory / "run.json").designs) == 2


def test_an_unregistered_tool_cannot_be_collected(isolated_registry, toy_run: Path) -> None:
    with pytest.raises(UnknownToolError, match="'toy'"):
        collect_run(toy_run)


def test_collection_dispatches_on_the_manifest_not_the_caller(
    isolated_registry, configs, tmp_path: Path
) -> None:
    """A mosaic run is parsed by the mosaic adapter, whatever else is loaded."""
    register(ToyAdapter)
    manifest = plan_mosaic_run(load_mosaic_configs(*configs), tmp_path / "run")
    for task_id in (0, 1):
        write_task(manifest.directory, task_id, [design_line(task_id, 0)], status={})
    # A toy output file sitting in a mosaic run must be ignored entirely.
    (manifest.directory / "toy_output.txt").write_text("WWWWWW\n")

    collected = collect_run(manifest.directory / "run.json")

    assert collected.run.tool == "mosaic"
    assert all(design.native_id.startswith("task-") for design in collected.designs)
    assert "WWWWWW" not in {design.sequence for design in collected.designs}


def test_every_registered_adapter_satisfies_the_contract(isolated_registry) -> None:
    register(ToyAdapter)

    for tool in registered_tools():
        adapter = adapter_for(tool)
        assert isinstance(adapter, OutputAdapter)
        assert adapter.tool == tool


def test_an_adapter_without_a_tool_name_is_refused() -> None:
    class Nameless(OutputAdapter):
        tool = ""

        def collect(self, run_dir, run):  # pragma: no cover - never called
            raise NotImplementedError

        def succeeded(self, run_dir):  # pragma: no cover - never called
            raise NotImplementedError

    with pytest.raises(ValueError, match="must define a tool name"):
        register(Nameless)


def test_the_manifest_is_the_only_source_of_the_tool_name(
    configs, tmp_path: Path
) -> None:
    """Nothing infers the tool from a path or a filename."""
    manifest = plan_mosaic_run(load_mosaic_configs(*configs), tmp_path / "nothing-here")

    assert RunManifest.read(manifest.directory / "run.json").tool == "mosaic"
    assert json.loads((manifest.directory / "run.json").read_text())["tool"] == "mosaic"
