"""PXDesign: diffuse a binder, then let two model families vote on it.

The contrast with the other tools is what this module is for. PXDesign returns
exactly as many rows as it was asked for -- padding the table with designs that
failed everything -- and scores each one with two independent confidence
models that disagree. So `n_produced` is never the interesting number here, and
there are four verdicts rather than one.

Its CLI is also the one that lies: `--preset` defaults to a mode with no
filters at all, and the shared options overwrite the container's own sampling
schedule. Both are answered in the config rather than left to the default.
"""

from __future__ import annotations

from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources
from bindocracy.runs.manifest import RunManifest, ToolPlan
from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.pxdesign.adapter import (
    DESIGN_OUTPUTS,
    SUMMARY_FILE,
    PXDesignOutputAdapter,
)
from bindocracy.tools.pxdesign.config import PXDesignConfig
from bindocracy.tools.pxdesign.launch import pxdesign_launch_spec
from bindocracy.tools.pxdesign.preflight import PXDesignPreflight, preflight_pxdesign


class PXDesignPlugin(ToolPlugin):
    tool = "pxdesign"
    config_type = PXDesignConfig
    adapter_type = PXDesignOutputAdapter

    def preflight(self, general: GeneralConfig, model: PXDesignConfig) -> PXDesignPreflight:
        return preflight_pxdesign(general, model)

    def resolve(self, loaded: LoadedConfigs) -> PXDesignConfig:
        """Fold the input spec into the config that gets stored.

        The target, its MSA, the crop, any epitope and the binder length all
        live in that file, and the config only names its path. Left alone, the
        configs table cannot answer what a run asked for, and editing the spec
        in place would change the science without changing model_config_id.
        """
        model = loaded.model
        if model.spec.contents == loaded.preflight.spec:
            return model
        return model.model_copy(
            update={"spec": model.spec.model_copy(update={"contents": loaded.preflight.spec})}
        )

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        sampling = loaded.model.sampling
        preflight: PXDesignPreflight = loaded.preflight
        return ToolPlan(
            jobs=sampling.jobs,
            # Exactly what the table will hold: the CLI pads it to --N_sample
            # with failed designs rather than returning fewer rows.
            designs_per_task=sampling.designs_per_job,
            designs_file=f"{DESIGN_OUTPUTS}/{preflight.task_name}/{SUMMARY_FILE}",
            # The spec is passed to the container and consumed by the run, so
            # it is archived where BoltzGen archives its own.
            archives={"spec": loaded.model.spec.template},
            # PXDesign reads geometry and a precomputed MSA directory, and
            # never the campaign FASTA or its single a3m.
            inputs=dict(preflight.input_files),
            container=loaded.model.runtime.container,
            workflow={
                "target_length": preflight.target_length,
                "task_name": preflight.task_name,
                # Binder length is in the spec, not in any harness field.
                "binder_length": preflight.binder_length,
                "hotspots": list(preflight.hotspots),
                "preset": sampling.preset,
                "diffusion_steps": sampling.diffusion_steps,
                "seed_base": sampling.seed_base,
                # The schedule that actually ran, which is not the container's
                # default and not the CLI's either.
                "eta": {
                    "type": sampling.eta_type,
                    "min": sampling.eta_min,
                    "max": sampling.eta_max,
                },
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, model = self.configs_of(manifest)
        return pxdesign_launch_spec(general, model, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)
