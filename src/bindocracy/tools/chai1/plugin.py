"""Chai-1: co-fold every design in a set against the target, and record it.

A second scorer beside the mosaic one, over the same design set and the same
metric registry. See `config.py` for why it is a separate plugin rather than a
seventh `ScoringModelName`.
"""

from __future__ import annotations

import math
from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources
from bindocracy.runs.manifest import RunManifest, ToolPlan
from bindocracy.store.records import RunKind
from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.chai1.adapter import METRICS_FILE, Chai1OutputAdapter
from bindocracy.tools.chai1.config import Chai1Config
from bindocracy.tools.chai1.launch import chai1_launch_spec
from bindocracy.tools.chai1.preflight import Chai1Preflight, preflight_chai1


class Chai1Plugin(ToolPlugin):
    tool = "chai1"
    config_type = Chai1Config
    adapter_type = Chai1OutputAdapter

    def preflight(self, general: GeneralConfig, model: Chai1Config) -> Chai1Preflight:
        return preflight_chai1(general, model)

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        chai: Chai1Config = loaded.model
        pre: Chai1Preflight = loaded.preflight
        jobs = chai.sharding.jobs
        per_task = -(-pre.n_designs // jobs)  # ceiling, matches DesignSet.shard

        inputs = {
            "target_fasta": loaded.general.target.sequence_fasta,
            "design_set": pre.fasta_path,
            "design_set_manifest": chai.design_set,
        }
        # Digested, so the run records the alignment bytes it folded against
        # rather than the directory it looked in.
        if pre.msa_path is not None:
            inputs["target_msa_pqt"] = pre.msa_path

        return ToolPlan(
            kind=RunKind.EVALUATE,
            jobs=jobs,
            designs_per_task=per_task,
            designs_file=METRICS_FILE,
            archives={"driver": chai.driver_script},
            inputs=inputs,
            container=chai.runtime.container,
            workflow={
                "model": "chai1",
                "metric_prefix": chai.metric_prefix,
                "protocol_sha256": chai.protocol_hash,
                "protocol": chai.protocol_fields,
                "readers": ["complex"],
                # Chai's assets are embedded and mounted read-only, so unlike
                # the mosaic scorer there is no source overlay to record: the
                # container digest alone determines the result.
                "design_set_digest": pre.design_set.digest,
                "design_set_manifest": str(chai.design_set),
                "design_set_fasta": str(pre.fasta_path),
                "scope_id": pre.design_set.scope_id,
                "n_designs": pre.n_designs,
                "distinct_lengths": pre.distinct_lengths,
                "target_length": pre.target_length,
                "target_msa_pqt": str(pre.msa_path) if pre.msa_path else None,
                # Every fold produces this many structures, and each becomes a
                # metric replicate. Recorded because it multiplies both the
                # walltime and the row count, and neither is obvious from the
                # two knobs that set it.
                "samples_per_design": chai.protocol.total_samples,
                "lengths_per_shard": math.ceil(pre.distinct_lengths / jobs),
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, chai = self.configs_of(manifest)
        return chai1_launch_spec(general, chai, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)
