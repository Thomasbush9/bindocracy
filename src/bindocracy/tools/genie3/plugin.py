"""Genie 3: diffuse a backbone, design a sequence onto it, then refold it.

The contrast with the other two tools is what this module is for. Genie 3
consumes the target as a *problem set* rather than a FASTA or a CIF, is driven
by its own experiment YAML that the harness must write one of per task, does
three stages in one job so a design is a backbone and a sequence and a complex,
and scores every design five times over.

None of that reaches the generic machinery.
"""

from __future__ import annotations

from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources
from bindocracy.runs.manifest import RunManifest, ToolPlan
from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.genie3.adapter import RESULTS_FILE, Genie3OutputAdapter
from bindocracy.tools.genie3.config import Genie3Config
from bindocracy.tools.genie3.launch import genie3_launch_spec
from bindocracy.tools.genie3.preflight import (
    Genie3Preflight,
    overlay_packages,
    preflight_genie3,
)


class Genie3Plugin(ToolPlugin):
    tool = "genie3"
    config_type = Genie3Config
    adapter_type = Genie3OutputAdapter

    def preflight(self, general: GeneralConfig, model: Genie3Config) -> Genie3Preflight:
        return preflight_genie3(general, model)

    def resolve(self, loaded: LoadedConfigs) -> Genie3Config:
        """Fold the experiment template into the config that gets stored.

        Genie 3's hyperparameters live in that file -- how the interface is
        conditioned, how many sequences per backbone, how many AF2 models and
        recycles -- and the config only names its path. Left alone, the configs
        table cannot answer what a run asked for, and editing the template in
        place would change the science without changing model_config_id.
        """
        model = loaded.model
        if model.experiment.contents == loaded.preflight.experiment:
            return model
        return model.model_copy(update={
            "experiment": model.experiment.model_copy(
                update={"contents": loaded.preflight.experiment}
            )
        })

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        sampling = loaded.model.sampling
        preflight: Genie3Preflight = loaded.preflight
        minimum, maximum = preflight.binder_length_range
        return ToolPlan(
            jobs=sampling.jobs,
            # A task's designs are its backbones times the sequences
            # ProteinMPNN writes for each, which is the count that ends up in
            # results/info.csv. The backbone count alone would under-report a
            # run by a factor of num_seq.
            designs_per_task=sampling.backbones_per_job * preflight.sequences_per_backbone,
            designs_file=f"{preflight.selection}/{RESULTS_FILE}",
            # Both are consumed by the run: the driver is executed, and the
            # template is what it renders this task's config from.
            archives={
                "experiment": loaded.model.experiment.template,
                "driver": loaded.model.driver.script,
            },
            # Genie 3 reads the target through its problem set and never opens
            # the campaign FASTA, the MSA, or the CIF the other tools use.
            inputs=dict(preflight.input_files),
            container=loaded.model.runtime.container,
            workflow={
                "target_length": preflight.target_length,
                "selection": preflight.selection,
                # Binder length is sampled per design from the problem set, not
                # set in any config the harness holds -- so record it, or the
                # run row cannot say what lengths it asked for.
                "binder_min_length": minimum,
                "binder_max_length": maximum,
                # Genie 3 has no hotspot-free binder mode, so every run here is
                # epitope-conditioned whether the campaign names one or not.
                "hotspots": list(preflight.hotspots),
                "cond_strategy": preflight.cond_strategy,
                "backbones_per_task": sampling.backbones_per_job,
                "sequences_per_backbone": preflight.sequences_per_backbone,
                "seed_base": sampling.seed_base,
                # The image alone no longer determines the result: without
                # these the AF2 stage silently runs on the CPU. Recorded by
                # package version as well as by path, because a path says
                # where a build was, not which build it was.
                "jax_overlays": _overlays(loaded.model.runtime),
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, model = self.configs_of(manifest)
        return genie3_launch_spec(general, model, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)


def _overlays(runtime: Any) -> dict[str, dict[str, Any]]:
    """Each JAX overlay's path and the wheel versions found on it."""
    return {
        label: {"path": str(path), "packages": overlay_packages(path)}
        for label, path in (
            ("plugin", runtime.jax_plugin_overlay),
            ("cudnn", runtime.cudnn_overlay),
            ("cuda_nvcc", runtime.cuda_nvcc_overlay),
        )
    }
