"""Optimize: improve a frozen design set with a user's script, and keep the children.

The fourth shape in this harness. A generator creates candidates from nothing;
a scorer measures candidates that exist; a filter decides about them. An
optimizer reads a frozen set like a scorer and emits designs like a generator,
which is why it is a plugin (it has a container, a fan-out and a run directory)
rather than the database-to-database shape `filters/config.py` describes.
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
from bindocracy.tools.optimize.adapter import CHILDREN_FILE, OptimizeOutputAdapter
from bindocracy.tools.optimize.config import OptimizeConfig
from bindocracy.tools.optimize.launch import optimize_launch_spec
from bindocracy.tools.optimize.preflight import OptimizePreflight, preflight_optimize


class OptimizePlugin(ToolPlugin):
    tool = "optimize"
    config_type = OptimizeConfig
    adapter_type = OptimizeOutputAdapter

    def preflight(self, general: GeneralConfig, model: OptimizeConfig) -> OptimizePreflight:
        return preflight_optimize(general, model)

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        model: OptimizeConfig = loaded.model
        pre: OptimizePreflight = loaded.preflight
        jobs = model.sharding.jobs
        per_task = -(-pre.n_designs // jobs)  # ceiling, matches DesignSet.shard

        inputs = {
            "target_fasta": loaded.general.target.sequence_fasta,
            "design_set": pre.fasta_path,
            "design_set_manifest": model.design_set,
            "optimizer_script": model.script,
        }
        if loaded.general.target.msa is not None:
            inputs["target_msa"] = loaded.general.target.msa

        return ToolPlan(
            kind=RunKind.OPTIMIZE,
            jobs=jobs,
            designs_per_task=per_task,
            designs_file=CHILDREN_FILE,
            archives={"driver": model.driver_script},
            inputs=inputs,
            # A host-run optimizer has no container. `ToolPlan.container` is
            # typed as one, so the script stands in for it -- it is the thing
            # whose bytes determined the result, which is what the field is
            # recorded for.
            container=model.runtime.container or model.script,
            workflow={
                "optimizer": model.name,
                "metric_prefix": model.metric_prefix,
                # The comparability key. Excludes the design set, so optimizing
                # more parents later extends a measurement.
                "protocol_sha256": model.protocol_hash,
                "protocol": model.protocol,
                # WHICH MODELS THE LOSS SAW. The field this whole tool records
                # that nothing before it did; see `config.py`.
                "loss_models": list(model.loss_models),
                # The bytes that optimized, not the path they were read from.
                # A script edited between two runs cannot make them look
                # comparable.
                "script_sha256": pre.script_sha256,
                "dev_source": (
                    str(model.runtime.dev_source) if model.runtime.dev_source else None
                ),
                "dev_source_sha256": pre.dev_source_sha256,
                "declared_metrics": {
                    key: value.model_dump(mode="json")
                    for key, value in model.metrics.items()
                },
                "max_children": model.max_children,
                "length_delta": model.length_delta,
                "seed": model.seed,
                "inputs_wanted": list(model.inputs),
                # Resolved from the database at PLAN time and carried here, so
                # a task launches against the poses and numbers the run was
                # planned with. The container never opens the campaign.
                "parent_structures": {
                    str(index): path for index, path in sorted(pre.structures.items())
                },
                "parent_metrics": {
                    str(index): values for index, values in sorted(pre.metrics.items())
                },
                "structures_from": model.structures_from,
                "n_unfolded_parents": pre.n_unfolded,
                # 1-based target FASTA positions, range-checked at plan time.
                "hotspots": list(pre.hotspots),
                "design_set_digest": pre.design_set.digest,
                "design_set_manifest": str(model.design_set),
                "design_set_fasta": str(pre.fasta_path),
                # What a rank decision over these children must be scoped to.
                "scope_id": pre.design_set.scope_id,
                "n_designs": pre.n_designs,
                "distinct_lengths": pre.distinct_lengths,
                "target_length": pre.target_length,
                "lengths_per_shard": math.ceil(pre.distinct_lengths / jobs),
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, model = self.configs_of(manifest)
        return optimize_launch_spec(general, model, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)
