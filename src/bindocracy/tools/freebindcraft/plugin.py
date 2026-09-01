"""FreeBindCraft: hallucinate a backbone, redesign it, and judge the result.

The contrast with the other tools is what this module is for. FreeBindCraft is
the only one here that runs an *open-ended loop*: it hallucinates a backbone,
redesigns it with ProteinMPNN, re-predicts every sequence, filters, and repeats
until it has enough accepted designs or has spent its trajectory budget. So
`designs_per_task` is a stopping condition rather than an output size, the
number of candidates is not knowable at planning time, and the run can end two
ways that leave visibly different output.

It is also the only one whose own filters are partly inert. PyRosetta is not in
this image, and the eight interface metrics that need it are filled with
constants chosen to pass -- so a run that records only its thresholds would
describe a filter stricter than the one that ran. The inert ones are named on
the run.

None of that reaches the generic machinery.
"""

from __future__ import annotations

from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources
from bindocracy.runs.manifest import RunManifest, ToolPlan
from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.freebindcraft.adapter import FreeBindCraftOutputAdapter
from bindocracy.tools.freebindcraft.config import FreeBindCraftConfig
from bindocracy.tools.freebindcraft.launch import (
    DESIGNS_FILE,
    freebindcraft_launch_spec,
)
from bindocracy.tools.freebindcraft.preflight import (
    PLACEHOLDER_METRICS,
    FreeBindCraftPreflight,
    preflight_freebindcraft,
)


class FreeBindCraftPlugin(ToolPlugin):
    tool = "freebindcraft"
    config_type = FreeBindCraftConfig
    adapter_type = FreeBindCraftOutputAdapter

    def preflight(
        self, general: GeneralConfig, model: FreeBindCraftConfig
    ) -> FreeBindCraftPreflight:
        return preflight_freebindcraft(general, model)

    def resolve(self, loaded: LoadedConfigs) -> FreeBindCraftConfig:
        """Fold all three authored documents into the config that gets stored.

        The target definition, the design protocol and the filter set are the
        science, and the config only names their paths. Left alone, the configs
        table cannot answer what a run asked for or what `n_passed` meant, and
        editing any of the three in place would change the experiment without
        changing model_config_id.
        """
        model = loaded.model
        preflight: FreeBindCraftPreflight = loaded.preflight
        contents = {
            "target": preflight.target,
            "filters": preflight.filters,
            "advanced": preflight.advanced,
        }
        if all(
            getattr(model, field).contents == document
            for field, document in contents.items()
        ):
            return model
        return model.model_copy(update={
            field: getattr(model, field).model_copy(update={"contents": document})
            for field, document in contents.items()
        })

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        model = loaded.model
        sampling = model.sampling
        advanced = loaded.preflight.advanced
        preflight: FreeBindCraftPreflight = loaded.preflight
        minimum, maximum = preflight.binder_length

        return ToolPlan(
            jobs=sampling.jobs,
            # The condition that ends the loop, not a table size: BindCraft
            # stops at the first check where at least this many designs sit in
            # Accepted/, so a task can finish with one or two more.
            designs_per_task=sampling.designs_per_job,
            # Successful hallucinations, which is what `max_trajectories`
            # counts -- not designs. A trajectory yields between zero and
            # `max_mpnn_sequences` accepted designs, and the number of MPNN
            # sequences attempted along the way is not knowable here at all.
            generated_per_task=sampling.max_trajectories,
            designs_file=DESIGNS_FILE,
            # All three documents are executed by the run: the driver renders
            # two of them per task and passes the third through. The driver
            # itself is code, and is in no other store.
            archives={
                "driver": model.driver.script,
                "target": model.target.template,
                "filters": model.filters.template,
                "advanced": model.advanced.template,
            },
            # The structure is the only file this tool opens that the harness
            # owns. AF2 parameters, ProteinMPNN weights, DSSP, FASPR and sc-rs
            # all ship in the image and are covered by the container digest.
            inputs=dict(preflight.input_files),
            container=model.runtime.container,
            workflow={
                "target_length": preflight.target_length,
                # What every trajectory, design and structure file is named
                # after, and so what the adapter reads names against.
                "binder_name": preflight.binder_name,
                "binder_min_length": minimum,
                "binder_max_length": maximum,
                # The campaign epitope as this tool expresses it, and the
                # string the driver writes. Empty means BindCraft was given
                # `hotspot=None` and designed against the whole surface, which
                # is a decision the run has to be able to state.
                "hotspots": list(preflight.hotspots),
                "hotspot_string": preflight.hotspot_string,
                # How much an epitope actually buys here, which is less than
                # the word suggests and is not recorded anywhere else.
                "epitope_enforcement": _epitope_enforcement(preflight, advanced),
                "max_trajectories": sampling.max_trajectories,
                "designs_per_task": sampling.designs_per_job,
                # BindCraft stamps the stem of each settings file onto every
                # row it writes, and the driver keeps the archived names, so
                # these are what a collected row is checked against.
                "target_settings_name": model.target.template.stem,
                "filters_name": model.filters.template.stem,
                "advanced_settings_name": model.advanced.template.stem,
                "design_algorithm": advanced.get("design_algorithm"),
                # Sequences drawn per trajectory, and how many of them may be
                # accepted before the trajectory is abandoned. Together they
                # bound the designs one backbone can contribute.
                "mpnn_sequences_per_trajectory": advanced.get("num_seqs"),
                "max_accepted_per_trajectory": advanced.get("max_mpnn_sequences"),
                # Decides how many AF2 models score each design, and therefore
                # how many of the five per-model columns carry a value.
                "use_multimer_design": advanced.get("use_multimer_design"),
                "rank_by": model.runtime.rank_by,
                # The complete definition of n_passed, and the part of it that
                # could not bite. Every filter file the image ships thresholds
                # at least one metric that is a constant without PyRosetta.
                "active_filters": list(preflight.active_filters),
                "inert_filters": list(preflight.inert_filters),
                "placeholder_metrics": list(PLACEHOLDER_METRICS),
                "pyrosetta": False,
                # Recorded because it is false. BindCraft draws every
                # trajectory's seed and length from numpy's unseeded global RNG
                # and has no flag that changes it, so this run cannot be
                # reproduced and two tasks differ by chance rather than design.
                "reproducible": False,
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, model = self.configs_of(manifest)
        return freebindcraft_launch_spec(general, model, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)


def _epitope_enforcement(
    preflight: FreeBindCraftPreflight, advanced: dict
) -> dict[str, Any]:
    """What an epitope does to a FreeBindCraft run, stated rather than assumed.

    Less than the word suggests, and worth recording because nothing in the
    output says it. A hotspot enters ColabDesign as `opt["hotspot"]`, which
    restricts the interface contact loss to those residues during
    *hallucination* -- and "contact" there means `inter_contact_number` binder
    residues within `inter_contact_distance`, which the default profile sets to
    20 A. That is a bias on the backbone search, not a constraint.

    Nothing checks it afterwards. The MPNN redesign and the AF2 re-prediction
    can drift off the patch, none of the filters mentions the epitope, and this
    fork's `Trajectory_WrongHotspot` counter is created but never incremented
    -- `update_failures` is called with it nowhere. So a run can accept designs
    that do not touch the residues it was conditioned on, which the first
    epitope run here did: both accepted designs landed on a neighbouring patch.

    Contrast Protein-Hunter, where the epitope is a resampling filter and part
    of the hit gate, and Proteina-Complexa, where it is a mask on the target.
    """
    if not preflight.hotspots:
        return {"conditioned": False}
    return {
        "conditioned": True,
        "residues": list(preflight.hotspots),
        "mechanism": "i_con loss restricted to the hotspot residues, during "
                     "hallucination only",
        "contact_distance_angstroms": advanced.get("inter_contact_distance"),
        "contacts_per_residue": advanced.get("inter_contact_number"),
        # No stage after hallucination looks at the epitope again, and the
        # counter that would have said so is dead in this fork.
        "verified_after_generation": False,
    }
