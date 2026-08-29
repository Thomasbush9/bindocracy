"""The one place that knows which tools exist.

Dispatch is always on the `tool` name written in the model config or the run
manifest, never on a path, a filename, or an argument the caller passes. That
way a config can only be planned by its own tool, and a run can only be parsed
by the tool that produced it.
"""

from __future__ import annotations

from pathlib import Path

from bindocracy.config.load import LoadedConfigs, read_tool_name
from bindocracy.runs.launch import LaunchSpec
from bindocracy.runs.manifest import RunManifest, plan_run
from bindocracy.store.records import CollectedRun
from bindocracy.tools.base import ToolPlugin


class UnknownToolError(KeyError):
    """No plugin is registered for this tool."""


_PLUGINS: dict[str, ToolPlugin] = {}


def register(plugin_type: type[ToolPlugin]) -> type[ToolPlugin]:
    """Register a tool, and its output adapter along with it."""
    plugin = plugin_type()
    if not plugin.tool:
        raise ValueError(f"{plugin_type.__name__} must define a tool name")
    _PLUGINS[plugin.tool] = plugin
    return plugin_type


def unregister(tool: str) -> None:
    _PLUGINS.pop(tool, None)


def registered_tools() -> tuple[str, ...]:
    return tuple(sorted(_PLUGINS))


def plugin_for(tool: str) -> ToolPlugin:
    try:
        return _PLUGINS[tool]
    except KeyError as error:
        known = ", ".join(registered_tools()) or "none"
        raise UnknownToolError(f"no plugin registered for {tool!r}; known: {known}") from error


def load_configs(general_path: str | Path, model_path: str | Path) -> LoadedConfigs:
    """Load any general + model pair, choosing the tool from the config itself."""
    return plugin_for(read_tool_name(model_path)).load(general_path, model_path)


def plan(loaded: LoadedConfigs, run_dir: str | Path, *, name: str | None = None) -> RunManifest:
    """Plan a run for whichever tool the loaded config names."""
    plugin = plugin_for(loaded.tool)
    return plan_run(loaded, plugin.tool_plan(loaded), run_dir, name=name)


def collect_run(manifest_path: str | Path) -> CollectedRun:
    """Collect a run with the adapter belonging to the tool that produced it."""
    manifest = RunManifest.read(manifest_path)
    adapter = plugin_for(manifest.tool).adapter()
    return adapter.collect(manifest.directory, manifest.to_run_record())


def launch_spec(manifest: RunManifest, task_id: int) -> LaunchSpec:
    """Build one task's command from the planned run, not from live YAML."""
    return plugin_for(manifest.tool).launch_spec(manifest, task_id)


def resources(loaded: LoadedConfigs) -> dict:
    return plugin_for(loaded.tool).resources(loaded)
