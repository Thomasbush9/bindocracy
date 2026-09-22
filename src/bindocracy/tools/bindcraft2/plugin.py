"""BindCraft 2: gradient-design a backbone, redesign it, refold it, rank it.

The contrast with the other tools is what this module is for, and here it is
mostly a contrast with its own predecessor.

FreeBindCraft needed a driver, because BindCraft 1 had no override for its
output directory, its design count, its epitope or its trajectory budget. BC2
exposes every one of them as `--set KEY=VALUE`, including the whole target block
as JSON, so this is the shortest generation plugin in the tree after PXDesign:
the archived campaign document goes to the container as it stands, and the
harness's values ride on the command line.

Three other differences are worth recording on the run rather than discovering
later.

**A run here is reproducible.** BC2 draws every trajectory from `campaign_seed`.
FreeBindCraft drew from numpy's unseeded global RNG with no flag to change it, so
its plans record `reproducible: false`; this one records true, and two tasks of
one run differ because their seeds differ rather than by chance.

**The epitope is verified after generation, not only conditioned on.** In
FreeBindCraft a hotspot biases the hallucination loss and nothing afterwards
checks it, so a run could accept designs that never touch the patch -- and the
first epitope run here did exactly that. BC2 measures
`Hotspot_Contact_Fraction` on every refolded candidate and can threshold it, so
an epitope is a filter and not only a bias. Whether the campaign actually
thresholds it is recorded, because conditioning without a ceiling is the weaker
thing wearing the same word.

**Nothing is inert.** FreeBindCraft's image has no PyRosetta, so eight of its
interface metrics are constants chosen to pass and any threshold on them
measured nothing. BC2 computes everything it filters on, so there is no
placeholder list here and no `pyrosetta: false` to qualify `n_passed` with.
"""

from __future__ import annotations

from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources
from bindocracy.runs.manifest import RunManifest, ToolPlan
from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.bindcraft2.adapter import BindCraft2OutputAdapter
from bindocracy.tools.bindcraft2.config import BindCraft2Config
from bindocracy.tools.bindcraft2.launch import DESIGNS_FILE, bindcraft2_launch_spec
from bindocracy.tools.bindcraft2.preflight import (
    BindCraft2Preflight,
    preflight_bindcraft2,
)

# The filters that would make the campaign epitope a requirement rather than a
# bias. Either one gives a design that misses the patch nowhere to hide.
EPITOPE_CEILINGS = (
    "min_hotspot_contact_final",
    "min_epitope_residues_contacted_final",
    "max_off_epitope_contact_final",
)


class BindCraft2Plugin(ToolPlugin):
    tool = "bindcraft2"
    config_type = BindCraft2Config
    adapter_type = BindCraft2OutputAdapter

    def preflight(
        self, general: GeneralConfig, model: BindCraft2Config
    ) -> BindCraft2Preflight:
        return preflight_bindcraft2(general, model)

    def resolve(self, loaded: LoadedConfigs) -> BindCraft2Config:
        """Fold the campaign document into the config that gets stored.

        The binder format, the loss weights, the stage schedule and the
        acceptance thresholds all live in that file, and the config only names
        its path. Left alone, the configs table cannot answer what a run asked
        for or what `n_passed` meant, and editing the document in place would
        change the science without changing model_config_id.
        """
        model = loaded.model
        preflight: BindCraft2Preflight = loaded.preflight
        if model.settings.contents == preflight.settings:
            return model
        return model.model_copy(
            update={
                "settings": model.settings.model_copy(
                    update={"contents": preflight.settings}
                )
            }
        )

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        model = loaded.model
        sampling = model.sampling
        preflight: BindCraft2Preflight = loaded.preflight
        minimum, maximum = preflight.binder_length_range

        return ToolPlan(
            jobs=sampling.jobs,
            # The condition that ends a task, not a table size. Workers accept
            # independently, so a task overshoots: a budget of 10 returned 11 on
            # the 2026-09-22 run.
            designs_per_task=sampling.designs_per_job,
            # Trajectory ATTEMPTS, which is what `max_trajectories` counts --
            # not designs and not successful trajectories. Only 29 of 184
            # attempts ran the whole way through on that run.
            generated_per_task=sampling.max_trajectories,
            designs_file=DESIGNS_FILE,
            # One document, executed from the archive. No driver: every value
            # the harness owns is a `--set` override.
            archives={"settings": model.settings.template},
            # The structure is the only file this tool opens that the harness
            # owns. AlphaFold parameters and all three ProteinMPNN variants are
            # baked into the image and covered by the container digest, and BC2
            # reads no alignment at all.
            inputs=dict(preflight.input_files),
            container=model.runtime.container,
            workflow={
                "target_length": preflight.target_length,
                # The exact block the launch passes with `--set`, stored so a
                # launch cannot derive an epitope different from the one the run
                # recorded.
                "bindcraft2_target": _target_block(loaded, preflight),
                "binder_lengths": list(preflight.binder_lengths),
                "binder_min_length": minimum,
                "binder_max_length": maximum,
                # The campaign epitope as BC2 expresses it. Empty means the
                # campaign named none and the whole surface was in play, which
                # is a decision the run has to be able to state.
                "hotspots": list(preflight.hotspots),
                "hotspot_string": preflight.hotspot_string,
                # What an epitope actually buys here, which unlike its
                # predecessor is more than a bias.
                "epitope_enforcement": _epitope_enforcement(preflight),
                "max_trajectories": sampling.max_trajectories,
                "designs_per_task": sampling.designs_per_job,
                "campaign_seed": sampling.seed_base,
                "workers_per_gpu": sampling.workers_per_gpu,
                # The thresholds the campaign authored. The rest of the filter
                # set is inside the image and covered by the container digest,
                # so this is the part of `n_passed` the campaign chose.
                "authored_filters": list(preflight.authored_filters),
                # Recorded because it is true, which its predecessor's was not.
                # BC2 draws every trajectory from campaign_seed.
                "reproducible": True,
                "target_chain": preflight.chain_id,
                "campaign_settings_name": model.settings.template.stem,
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, model = self.configs_of(manifest)
        return bindcraft2_launch_spec(general, model, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)


def _target_block(loaded: LoadedConfigs, preflight: BindCraft2Preflight) -> dict[str, Any]:
    """The campaign target as BC2's `targets` list, resolved once at planning.

    One entry: this harness designs against one target, and a second would be a
    multi-target campaign whose filters and rotation are a different experiment.
    The path is absolute, so nothing depends on the working directory the job
    happens to start in.
    """
    target: dict[str, Any] = {
        "name": loaded.general.target.name,
        "target_path": str(preflight.target_pdb.resolve()),
        "chains": preflight.chain_id,
    }
    if preflight.hotspot_string:
        target["hotspots"] = preflight.hotspot_string
    return target


def _epitope_enforcement(preflight: BindCraft2Preflight) -> dict[str, Any]:
    """What an epitope does to a BindCraft 2 run, stated rather than assumed.

    More than in FreeBindCraft, and the difference is the whole reason to record
    it. A hotspot steers the interface contact loss during gradient design, as
    it did before; but BC2 also measures `Hotspot_Contact_Fraction` and
    `Epitope_Residues_Contacted` on every refolded candidate, so a campaign that
    thresholds either one rejects a design that drifted off the patch instead of
    accepting it.

    `verified_after_generation` is therefore a property of the campaign, not of
    the tool. Conditioned without a ceiling, this is the same bias FreeBindCraft
    applied and the same silence afterwards.
    """
    if not preflight.hotspots:
        return {"conditioned": False, "verified_after_generation": False}
    ceilings = [
        name for name in EPITOPE_CEILINGS if name in set(preflight.authored_filters)
    ]
    return {
        "conditioned": True,
        "residues": list(preflight.hotspots),
        "mechanism": "interface contact loss restricted to the hotspot residues "
                     "during gradient design, and measured again on every "
                     "refolded candidate",
        # False means the epitope steered generation and then nothing checked
        # it, which is a campaign that can accept a design missing the patch.
        "verified_after_generation": bool(ceilings),
        "epitope_filters": ceilings,
    }
