"""BoltzGen: diffuse a binder backbone against the target's geometry.

The contrast with Mosaic is the point of this module. BoltzGen consumes the
target as *structure* rather than sequence, is driven by an authored design
spec rather than a script, writes a 237-column CSV rather than JSON lines, and
filters its own pool so that produced and passed are different numbers.

None of that reaches the generic machinery.
"""

from __future__ import annotations

from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources
from bindocracy.runs.manifest import RunManifest, ToolPlan
from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.boltzgen.adapter import METRICS_FILE, BoltzGenOutputAdapter
from bindocracy.tools.boltzgen.config import BoltzGenConfig
from bindocracy.tools.boltzgen.launch import boltzgen_launch_spec
from bindocracy.tools.boltzgen.preflight import BoltzGenPreflight, preflight_boltzgen


class BoltzGenPlugin(ToolPlugin):
    tool = "boltzgen"
    config_type = BoltzGenConfig
    adapter_type = BoltzGenOutputAdapter

    def preflight(self, general: GeneralConfig, model: BoltzGenConfig) -> BoltzGenPreflight:
        return preflight_boltzgen(general, model)

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        sampling = loaded.model.sampling
        return ToolPlan(
            jobs=sampling.jobs,
            # What the run asks for is the budget: the designs meant to survive
            # into final_ranked_designs, not the backbones generated on the way.
            designs_per_task=sampling.budget,
            generated_per_task=sampling.num_designs,
            designs_file=METRICS_FILE,
            # The spec is bound into the container and consumed by the run, so
            # it is archived for the same reason Mosaic's driver is.
            archives={"spec": loaded.model.spec.template},
            # The spec is a referenced input whose contents matter; the config
            # only names its path. Digest every structure it pulls geometry
            # from, so replacing one in place is visible in the run record.
            inputs={
                f"spec_structure_{index}": path
                for index, path in enumerate(loaded.preflight.spec_files)
            },
            container=loaded.model.runtime.container,
            workflow={
                "target_length": loaded.preflight.target_length,
                "protocol": sampling.protocol,
                "num_designs": sampling.num_designs,
                "budget": sampling.budget,
                "filter_biased": sampling.filter_biased,
                # Parsed, not just referenced: model_config_json records the
                # spec's path, and a path does not say what was designed.
                "spec": loaded.preflight.spec,
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, model = self.configs_of(manifest)
        return boltzgen_launch_spec(general, model, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)
