"""Everything the harness needs to know about one tool, in one object.

A tool is four decisions the generic machinery cannot make for it: how its
config validates, how many tasks a run splits into, what command runs one task,
and how to read the result. `ToolPlugin` is those four and nothing else.

Adding a tool is one module implementing this, plus one line in the registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from bindocracy.adapters.base import OutputAdapter
from bindocracy.config.load import LoadedConfigs, load_pair
from bindocracy.config.models import GeneralConfig, ToolConfig
from bindocracy.runs.launch import LaunchSpec
from bindocracy.runs.manifest import RunManifest, ToolPlan


class ToolPlugin(ABC):
    tool: str
    config_type: type[ToolConfig]
    adapter_type: type[OutputAdapter]

    def load(self, general_path: str | Path, model_path: str | Path) -> LoadedConfigs:
        """Validate a general + model pair and run this tool's preflight."""
        return load_pair(general_path, model_path, self.config_type, self.preflight)

    def adapter(self) -> OutputAdapter:
        return self.adapter_type()

    @abstractmethod
    def preflight(self, general: GeneralConfig, model: Any) -> Any:
        """Check what this tool needs from the environment before any GPU work."""

    @abstractmethod
    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        """Task count, per-task output name, and the inputs worth archiving."""

    @abstractmethod
    def launch_spec(
        self, loaded: LoadedConfigs, manifest: RunManifest, task_id: int
    ) -> LaunchSpec:
        """The argv, environment, resources, and expected outputs of one task."""

    @abstractmethod
    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        """Slurm resources, resolvable before a run manifest exists."""
