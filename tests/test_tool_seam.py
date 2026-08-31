"""The tool seam: adding or removing a tool must not disturb the others.

The test that matters here is `test_a_new_tool_needs_no_changes_to_mosaic`. It
defines a whole second tool inside the test — its own config, its own output
format, its own adapter — and drives it through the same collection path. If
someone reintroduces a `if tool == "mosaic"` branch anywhere between the
manifest and the staging bundle, that test is what fails.

There is one registry. A tool registered through it is reachable from the
workflow and from the CLI alike, which was not true when collection had a
registry of its own.

These are the fast in-process checks. `tests/test_third_tool.py` takes the
harder one: a complete foreign tool, planned and launched and ingested through
the real Snakefile without editing anything under `src/bindocracy/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import design_line, write_task

from bindocracy.adapters import OutputAdapter
from bindocracy.config.load import LoadedConfigs
from bindocracy.runs import LaunchSpec, RunManifest, ToolPlan
from bindocracy.store.records import CandidateType, CollectedRun, DesignRecord, RunStatus
from bindocracy.tools import (
    ToolPlugin,
    UnknownToolError,
    collect_run,
    load_configs,
    plan,
    plugin_for,
    register,
    registered_tools,
    unregister,
)
from bindocracy.tools.mosaic.adapter import MosaicOutputAdapter


class ToyAdapter(OutputAdapter):
    """An unrelated output format: one sequence per line."""

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


class ToyPlugin(ToolPlugin):
    tool = "toy"
    config_type = None  # never loaded from YAML in these tests
    adapter_type = ToyAdapter

    def preflight(self, general, model) -> None:  # pragma: no cover - not exercised
        return None

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:  # pragma: no cover
        raise NotImplementedError

    def launch_spec(self, loaded, manifest, task_id) -> LaunchSpec:  # pragma: no cover
        raise NotImplementedError

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


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
    manifest = plan(load_configs(*configs), tmp_path / "run")
    toy = manifest.model_copy(update={"tool": "toy"})
    (manifest.directory / "run.json").write_text(toy.model_dump_json(indent=2))
    (manifest.directory / "toy_output.txt").write_text("ACDEFG\nHIKLMN\n")
    return manifest.directory / "run.json"


def test_the_built_in_tools_are_registered() -> None:
    assert registered_tools() == (
        "boltzgen", "genie3", "mosaic", "protein_hunter", "pxdesign",
    )
    assert isinstance(plugin_for("mosaic").adapter(), MosaicOutputAdapter)


def test_an_unknown_tool_says_what_is_known() -> None:
    with pytest.raises(UnknownToolError, match="no plugin registered for 'nope'"):
        plugin_for("nope")
    with pytest.raises(UnknownToolError, match="mosaic"):
        plugin_for("nope")


def test_a_new_tool_needs_no_changes_to_mosaic(isolated_registry, toy_run: Path) -> None:
    """One register() call is the entire cost of adding a tool."""
    register(ToyPlugin)

    collected = collect_run(toy_run)

    assert collected.run.tool == "toy"
    assert [design.native_id for design in collected.designs] == ["toy-0", "toy-1"]
    assert collected.run.n_produced == 2
    assert isinstance(plugin_for("mosaic").adapter(), MosaicOutputAdapter)


def test_one_registration_reaches_collection_too(isolated_registry, toy_run: Path) -> None:
    """Registering the plugin is enough; collection has no registry of its own."""
    assert "toy" not in registered_tools()
    with pytest.raises(UnknownToolError, match="'toy'"):
        collect_run(toy_run)

    register(ToyPlugin)

    assert "toy" in registered_tools()
    assert collect_run(toy_run).run.tool == "toy"


def test_removing_a_tool_leaves_the_others_working(
    isolated_registry, configs, tmp_path: Path
) -> None:
    register(ToyPlugin)
    unregister("toy")

    assert "toy" not in registered_tools()
    assert "mosaic" in registered_tools()

    manifest = plan(load_configs(*configs), tmp_path / "run")
    write_task(manifest.directory, 0, [design_line(0, 0)], status={})
    write_task(manifest.directory, 1, [design_line(1, 0)], status={})
    assert len(collect_run(manifest.directory / "run.json").designs) == 2


def test_collection_dispatches_on_the_manifest_not_the_caller(
    isolated_registry, configs, tmp_path: Path
) -> None:
    """A mosaic run is parsed by the mosaic adapter, whatever else is loaded."""
    register(ToyPlugin)
    manifest = plan(load_configs(*configs), tmp_path / "run")
    for task_id in (0, 1):
        write_task(manifest.directory, task_id, [design_line(task_id, 0)], status={})
    # A toy output file sitting in a mosaic run must be ignored entirely.
    (manifest.directory / "toy_output.txt").write_text("WWWWWW\n")

    collected = collect_run(manifest.directory / "run.json")

    assert collected.run.tool == "mosaic"
    assert all(design.native_id.startswith("task-") for design in collected.designs)
    assert "WWWWWW" not in {design.sequence for design in collected.designs}


def test_every_registered_plugin_satisfies_the_contract(isolated_registry) -> None:
    register(ToyPlugin)

    for tool in registered_tools():
        plugin = plugin_for(tool)
        assert isinstance(plugin, ToolPlugin)
        assert plugin.tool == tool
        assert isinstance(plugin.adapter(), OutputAdapter)
        assert plugin.adapter().tool == tool


def test_a_plugin_without_a_tool_name_is_refused() -> None:
    class Nameless(ToyPlugin):
        tool = ""

    with pytest.raises(ValueError, match="must define a tool name"):
        register(Nameless)


def test_the_manifest_is_the_only_source_of_the_tool_name(
    configs, tmp_path: Path
) -> None:
    """Nothing infers the tool from a path or a filename."""
    manifest = plan(load_configs(*configs), tmp_path / "nothing-here")

    assert RunManifest.read(manifest.directory / "run.json").tool == "mosaic"
    assert json.loads((manifest.directory / "run.json").read_text())["tool"] == "mosaic"
