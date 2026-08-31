"""Protein-Hunter: search sequence space by co-folding and redesigning.

The contrast with the other four is that nothing here is diffused and nothing
is conditioned on geometry. Each trajectory starts from a mostly-X binder and
loops {Boltz-2 co-fold, LigandMPNN redesign}, so a design is a cycle's sequence
and a trajectory is five of them. The target reaches the tool as a string in an
argument vector rather than as a file, and there is no seed anywhere in the
pipeline -- both of which the plan records rather than papers over.
"""

from __future__ import annotations

import re
from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources
from bindocracy.runs.manifest import RunManifest, ToolPlan
from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.protein_hunter.adapter import (
    SUMMARY_FILE,
    ProteinHunterOutputAdapter,
)
from bindocracy.tools.protein_hunter.config import ProteinHunterConfig
from bindocracy.tools.protein_hunter.launch import protein_hunter_launch_spec
from bindocracy.tools.protein_hunter.preflight import (
    ProteinHunterPreflight,
    preflight_protein_hunter,
)


class ProteinHunterPlugin(ToolPlugin):
    tool = "protein_hunter"
    config_type = ProteinHunterConfig
    adapter_type = ProteinHunterOutputAdapter

    def preflight(
        self, general: GeneralConfig, model: ProteinHunterConfig
    ) -> ProteinHunterPreflight:
        return preflight_protein_hunter(general, model)

    # No `resolve`: this tool references no external scientific file. Its
    # hyperparameters are all fields of the config, and the target is the
    # campaign's own FASTA, which the run digests. There is nothing to fold in.

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        sampling = loaded.model.sampling
        preflight: ProteinHunterPreflight = loaded.preflight
        return ToolPlan(
            jobs=sampling.jobs,
            # Every cycle emits exactly one sequence, so the designs a task
            # produces is trajectories times cycles. Counting trajectories
            # alone would under-report the run by a factor of `cycles`.
            designs_per_task=sampling.trajectories_per_job * sampling.cycles,
            designs_file=SUMMARY_FILE,
            # The driver is executed, and its content is in no other store.
            archives={"driver": loaded.model.driver.script},
            # The FASTA is read to build the command line; the alignment, when
            # there is one, is what the driver seeds the MSA cache from.
            inputs=dict(preflight.input_files),
            container=loaded.model.runtime.container,
            workflow={
                "target_length": preflight.target_length,
                # What the tool names its structure files after.
                "design_name": design_name(loaded.general.target.name),
                "trajectories_per_task": sampling.trajectories_per_job,
                "cycles": sampling.cycles,
                "min_binder_length": sampling.min_binder_length,
                "max_binder_length": sampling.max_binder_length,
                "percent_x": sampling.percent_x,
                "omit_aa": sampling.omit_aa,
                "msa_mode": loaded.model.msa.mode,
                # The campaign epitope as this tool expresses it, and how it
                # is enforced. Empty when the campaign names none.
                "contact_residues": preflight.contact_residues,
                "contacts": _contacts(loaded.model, preflight),
                # The complete definition of n_passed for this run. Two of
                # these four conditions are hard-coded upstream and appear in
                # no config at all, so recording only the thresholds would
                # describe a filter looser than the one that ran.
                "hit_protocol": _hit_protocol(loaded.model, preflight),
                "high_iptm_threshold": loaded.model.filters.high_iptm_threshold,
                "high_plddt_threshold": loaded.model.filters.high_plddt_threshold,
                # Recorded because it is false, and because every other tool
                # here records a seed. The pipeline has no seed flag, so this
                # run cannot be reproduced or split deterministically.
                "reproducible": False,
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, model = self.configs_of(manifest)
        return protein_hunter_launch_spec(general, model, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)


def design_name(target_name: str) -> str:
    """A filename-safe name for the target, which the tool prefixes outputs with.

    Derived rather than authored so it cannot drift from the campaign target,
    and reduced to word characters because the chai pipeline strips everything
    else and the two should not disagree about what a run is called.
    """
    return re.sub(r"\W+", "_", target_name).strip("_") or "target"


def _contacts(model: ProteinHunterConfig, preflight: ProteinHunterPreflight) -> dict[str, Any]:
    """How the epitope is enforced, or that there was none to enforce."""
    if not preflight.contact_residues:
        return {"conditioned": False}
    return {
        "conditioned": True,
        "residues": [int(value) for value in preflight.contact_residues.split(",")],
        "cutoff_angstroms": model.contacts.cutoff,
        "resample_on_miss": model.contacts.filter,
        "max_retries": model.contacts.max_retries,
    }


def _hit_protocol(
    model: ProteinHunterConfig, preflight: ProteinHunterPreflight
) -> dict[str, Any]:
    """Every condition a design must meet to reach `summary_high_iptm.csv`.

    `protein_hunter_high_iptm` is named for the first of these and gated by all
    of them. Two are hard-coded upstream and appear in no configuration, so a
    run recording only its two thresholds would describe a filter that lets
    more through than the one that actually ran.
    """
    protocol = {
        "iptm_above": model.filters.high_iptm_threshold,
        "plddt_above": model.filters.high_plddt_threshold,
        # Hard-coded in pipeline.py: alanine_percentage <= 0.20.
        "alanine_fraction_at_most": 0.20,
    }
    if preflight.contact_residues:
        # Hard-coded in model_utils.binder_binds_contacts: at least two of the
        # named residues must have a binder CA within the cutoff.
        protocol["contacts"] = {
            "residues": [int(v) for v in preflight.contact_residues.split(",")],
            "cutoff_angstroms": model.contacts.cutoff,
            "min_residues_contacted": 2,
        }
    return protocol
