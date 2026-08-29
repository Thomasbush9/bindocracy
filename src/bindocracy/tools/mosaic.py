"""Mosaic: hallucinate a binder by optimizing a soft sequence through Boltz-2.

Everything Mosaic-specific lives here or in the two modules this points at.
"""

from __future__ import annotations

from typing import Any

from bindocracy.adapters.mosaic import MosaicOutputAdapter
from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig, MosaicConfig
from bindocracy.config.preflight import MosaicPreflight, preflight_mosaic
from bindocracy.runs.launch import LaunchSpec, mosaic_launch_spec, mosaic_resources
from bindocracy.runs.manifest import DESIGNS_FILE, RunManifest, ToolPlan
from bindocracy.tools.base import ToolPlugin


class MosaicPlugin(ToolPlugin):
    tool = "mosaic"
    config_type = MosaicConfig
    adapter_type = MosaicOutputAdapter

    def preflight(self, general: GeneralConfig, model: MosaicConfig) -> MosaicPreflight:
        return preflight_mosaic(general, model)

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        sampling = loaded.model.sampling
        return ToolPlan(
            jobs=sampling.jobs,
            designs_per_task=sampling.designs_per_job,
            designs_file=DESIGNS_FILE,
            # The driver is executed, and its content is in no other store.
            archives={"driver": loaded.model.driver.script},
            container=loaded.model.runtime.container,
            workflow={
                "target_length": loaded.preflight.target_length,
                "binder_length": sampling.binder_length,
                "seed_base": sampling.seed_base,
            },
        )

    def launch_spec(
        self, loaded: LoadedConfigs, manifest: RunManifest, task_id: int
    ) -> LaunchSpec:
        return mosaic_launch_spec(loaded, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return mosaic_resources(loaded)
