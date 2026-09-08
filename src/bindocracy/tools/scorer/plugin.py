"""Scorer: fold every design in a set with one model, and record what it says.

DRAFT. One plugin serves every structural model; the model is a config field,
not a tool name. See `config.py` for why.
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
from bindocracy.tools.scorer.adapter import METRICS_FILE, ScorerOutputAdapter
from bindocracy.tools.scorer.config import ScorerConfig
from bindocracy.tools.scorer.launch import scorer_launch_spec
from bindocracy.tools.scorer.preflight import ScorerPreflight, preflight_scorer


class ScorerPlugin(ToolPlugin):
    tool = "scorer"
    config_type = ScorerConfig
    adapter_type = ScorerOutputAdapter

    def preflight(self, general: GeneralConfig, model: ScorerConfig) -> ScorerPreflight:
        return preflight_scorer(general, model)

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        scorer: ScorerConfig = loaded.model
        pre: ScorerPreflight = loaded.preflight
        jobs = scorer.sharding.jobs
        per_task = -(-pre.n_designs // jobs)  # ceiling, matches DesignSet.shard

        inputs = {
            "target_fasta": loaded.general.target.sequence_fasta,
            "design_set": pre.fasta_path,
            "design_set_manifest": scorer.design_set,
            "exec_wrapper": scorer.runtime.exec_wrapper,
        }
        if scorer.model.use_target_msa and loaded.general.target.msa is not None:
            inputs["target_msa"] = loaded.general.target.msa

        return ToolPlan(
            kind=RunKind.EVALUATE,
            jobs=jobs,
            designs_per_task=per_task,
            designs_file=METRICS_FILE,
            archives={"driver": scorer.driver_script},
            inputs=inputs,
            container=scorer.runtime.container,
            workflow={
                "model": scorer.model.name,
                "metric_prefix": scorer.metric_prefix,
                # The comparability key. Two runs sharing this measured the
                # same quantity the same way, whatever they scored.
                "protocol_sha256": scorer.protocol_hash,
                "protocol": scorer.protocol,
                "readers": [
                    name
                    for name, on in (
                        ("complex", scorer.readers.complex),
                        ("monomer", scorer.readers.monomer),
                    )
                    if on
                ],
                # Where the code came from. None means the image alone
                # determined the result, which is the goal state.
                "dev_source": (
                    str(scorer.runtime.dev_source) if scorer.runtime.dev_source else None
                ),
                "dev_source_sha256": pre.dev_source_sha256,
                "design_set_digest": pre.design_set.digest,
                "design_set_manifest": str(scorer.design_set),
                "design_set_fasta": str(pre.fasta_path),
                # What a rank decision over these metrics must be scoped to.
                "scope_id": pre.design_set.scope_id,
                "n_designs": pre.n_designs,
                "distinct_lengths": pre.distinct_lengths,
                "target_length": pre.target_length,
                # Recorded because a JIT recompile per length is the dominant
                # fixed cost, and this is what predicts it.
                "lengths_per_shard": math.ceil(pre.distinct_lengths / jobs),
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, scorer = self.configs_of(manifest)
        return scorer_launch_spec(general, scorer, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)
