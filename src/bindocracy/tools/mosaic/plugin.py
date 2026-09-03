"""Mosaic: hallucinate a binder by optimizing a soft sequence through Boltz-2.

Everything Mosaic-specific lives here or in the two modules this points at.
"""

from __future__ import annotations

from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources
from bindocracy.runs.manifest import DESIGNS_FILE, RunManifest, ToolPlan
from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.mosaic.adapter import MosaicOutputAdapter
from bindocracy.tools.mosaic.config import MosaicConfig
from bindocracy.tools.mosaic.launch import mosaic_launch_spec
from bindocracy.tools.mosaic.preflight import MosaicPreflight, preflight_mosaic


class MosaicPlugin(ToolPlugin):
    tool = "mosaic"
    config_type = MosaicConfig
    adapter_type = MosaicOutputAdapter

    def preflight(self, general: GeneralConfig, model: MosaicConfig) -> MosaicPreflight:
        return preflight_mosaic(general, model)

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        sampling = loaded.model.sampling
        inputs = {
            "target_fasta": loaded.general.target.sequence_fasta,
            "target_msa": loaded.general.target.msa,
            "exec_wrapper": loaded.model.runtime.exec_wrapper,
        }
        # Planning reads the structure only to map author-numbered hotspots
        # onto FASTA positions. Record it whenever that mapping was required.
        if loaded.general.target.hotspots:
            assert loaded.general.target.structure_pdb is not None
            inputs["target_structure"] = loaded.general.target.structure_pdb
        return ToolPlan(
            jobs=sampling.jobs,
            designs_per_task=sampling.designs_per_job,
            designs_file=DESIGNS_FILE,
            # The driver is executed, and its content is in no other store.
            archives={"driver": loaded.model.driver.script},
            inputs=inputs,
            container=loaded.model.runtime.container,
            workflow={
                "target_length": loaded.preflight.target_length,
                "binder_length": sampling.binder_length,
                "seed_base": sampling.seed_base,
                # The campaign epitope as this tool expresses it, and how much
                # it enforces. Empty when the campaign names none.
                "hotspots": list(loaded.general.target.hotspots),
                "epitope_idx": list(loaded.preflight.epitope_idx),
                "epitope_enforcement": _epitope_enforcement(loaded.preflight),
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, mosaic = self.configs_of(manifest)
        return mosaic_launch_spec(general, mosaic, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)


def _epitope_enforcement(preflight: MosaicPreflight) -> dict[str, Any]:
    """What an epitope does to a Mosaic run, stated rather than assumed.

    One of nine loss terms. `BinderTargetContact` slices its binder-by-target
    contact matrix down to the epitope columns, then averages each binder
    residue's three best contact log-probabilities at a 20 A cutoff -- so the
    optimiser is pushed towards the patch during hallucination, and nothing
    afterwards re-checks it. The ranking re-fold scores iPTM and ipSAE over the
    whole complex and never mentions the epitope.

    The same shape as FreeBindCraft, and weaker than Protein-Hunter, where the
    epitope is a resampling filter and part of the hit gate.
    """
    if not preflight.epitope_idx:
        return {"conditioned": False}
    return {
        "conditioned": True,
        # 0-based into the target sequence, which is what the loss slices by.
        "epitope_idx": list(preflight.epitope_idx),
        "mechanism": "BinderTargetContact restricted to the epitope columns, "
        "during hallucination only",
        # BinderTargetContact's own default, and not a harness field.
        "contact_distance_angstroms": 20.0,
        "verified_after_generation": False,
    }
