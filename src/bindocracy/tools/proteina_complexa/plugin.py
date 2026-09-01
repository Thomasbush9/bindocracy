"""Proteina-Complexa: draw many binders, score every one with AF2, keep a few.

The contrast with the other tools is what this module is for. Proteina-Complexa
co-generates sequence and structure, so a design arrives as a complex rather
than a backbone needing inverse folding. It runs AF2 inside the sampling loop
as a reward, which is why generating and keeping are different numbers and why
the walltime is what it is. And it reads its target through a registry file
rather than through flags, so the epitope, the crop and the binder length all
arrive together or not at all.

None of that reaches the generic machinery.
"""

from __future__ import annotations

from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources
from bindocracy.runs.manifest import RunManifest, ToolPlan
from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.proteina_complexa.adapter import ProteinaComplexaOutputAdapter
from bindocracy.tools.proteina_complexa.config import ProteinaComplexaConfig
from bindocracy.tools.proteina_complexa.launch import (
    CONFIG_STEM,
    RUN_NAME,
    designs_file,
    proteina_complexa_launch_spec,
)
from bindocracy.tools.proteina_complexa.preflight import (
    ProteinaComplexaPreflight,
    preflight_proteina_complexa,
)

# What the image's evaluation stage ships, and therefore what a design's
# sequence *is*: the model's own, not an inverse-folding redesign of it. The
# alternatives (`mpnn`, `mpnn_fixed`) are commented out in binder_evaluate.yaml
# and would change what the campaign is comparing, so the run records it.
SEQUENCE_TYPES = ("self",)


class ProteinaComplexaPlugin(ToolPlugin):
    tool = "proteina_complexa"
    config_type = ProteinaComplexaConfig
    adapter_type = ProteinaComplexaOutputAdapter

    def preflight(
        self, general: GeneralConfig, model: ProteinaComplexaConfig
    ) -> ProteinaComplexaPreflight:
        return preflight_proteina_complexa(general, model)

    def resolve(self, loaded: LoadedConfigs) -> ProteinaComplexaConfig:
        """Fold the target registry into the config that gets stored.

        The crop, the epitope and the binder length all live in that file, and
        the config only names its path. Left alone, the configs table cannot
        answer what a run asked for, and editing the registry in place would
        change the science without changing model_config_id.
        """
        model = loaded.model
        if model.registry.contents == loaded.preflight.registry:
            return model
        return model.model_copy(update={
            "registry": model.registry.model_copy(
                update={"contents": loaded.preflight.registry}
            )
        })

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        model = loaded.model
        sampling = model.sampling
        preflight: ProteinaComplexaPreflight = loaded.preflight

        # What the search actually draws before its reward filter keeps any.
        # `replicas` is the best-of-n width, so it multiplies rather than adds.
        generated = (
            sampling.samples_per_job
            * sampling.nrepeat_per_sample
            * sampling.replicas
        )
        minimum, maximum = preflight.binder_length

        return ToolPlan(
            jobs=sampling.jobs,
            # The filter keeps top-N by reward, so a task cannot return more
            # than it drew however large the budget is.
            designs_per_task=min(sampling.keep_per_job, generated),
            generated_per_task=generated,
            designs_file=designs_file(model.registry.task_name),
            # The registry is bound into the container and read by the run, so
            # it is archived where BoltzGen archives its spec. The pipeline
            # config, the Hydra tree and the weights are all inside the image.
            archives={"registry": model.registry.template},
            # The structure is the only file the tool opens that the harness
            # owns; everything else it reads ships in the image.
            inputs=dict(preflight.input_files),
            container=model.runtime.container,
            workflow={
                "target_length": preflight.target_length,
                "task_name": model.registry.task_name,
                # The crop the run designed against. A campaign target and a
                # contig that trims it are different proteins to this tool.
                "target_input": preflight.target_input,
                "hotspots": list(preflight.hotspots),
                # Binder length is in the registry, not in any harness field.
                "binder_min_length": minimum,
                "binder_max_length": maximum,
                # Generating, keeping and drawing are three numbers here, and a
                # run row that records one of them cannot reconstruct the rest.
                "samples_per_task": sampling.samples_per_job,
                "nrepeat_per_sample": sampling.nrepeat_per_sample,
                "search_algorithm": "best-of-n",
                "replicas": sampling.replicas,
                "keep_per_task": sampling.keep_per_job,
                "batch_size": sampling.batch_size,
                "seed_base": sampling.seed_base,
                # Not affected by `++seed`: the AF2 reward carries its own,
                # hard-coded in binder_generate.yaml.
                "reward_seed": sampling.reward_seed,
                # Decides whether a design's sequence is the model's own or an
                # inverse-folding redesign, which is what makes this tool
                # comparable with the others at all.
                "sequence_types": list(SEQUENCE_TYPES),
                # Part of the output contract: Hydra names both output roots
                # after these, so collection depends on them.
                "config_stem": CONFIG_STEM,
                "run_name": RUN_NAME,
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, model = self.configs_of(manifest)
        return proteina_complexa_launch_spec(general, model, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)
