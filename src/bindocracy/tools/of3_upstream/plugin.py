"""Upstream OpenFold3: the official image, not mosaic's port.

See `config.py` for why this is a separate model rather than a swap.
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
from bindocracy.tools.of3_upstream.adapter import (
    METRICS_FILE,
    OF3UpstreamOutputAdapter,
)
from bindocracy.tools.of3_upstream.config import OF3UpstreamConfig
from bindocracy.tools.of3_upstream.launch import of3_upstream_launch_spec
from bindocracy.tools.of3_upstream.preflight import (
    OF3UpstreamPreflight,
    preflight_of3_upstream,
)


class OF3UpstreamPlugin(ToolPlugin):
    tool = "of3_upstream"
    config_type = OF3UpstreamConfig
    adapter_type = OF3UpstreamOutputAdapter

    def preflight(
        self, general: GeneralConfig, model: OF3UpstreamConfig
    ) -> OF3UpstreamPreflight:
        return preflight_of3_upstream(general, model)

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        of3: OF3UpstreamConfig = loaded.model
        pre: OF3UpstreamPreflight = loaded.preflight
        jobs = of3.sharding.jobs
        per_task = -(-pre.n_designs // jobs)

        inputs = {
            "target_fasta": loaded.general.target.sequence_fasta,
            "design_set": pre.fasta_path,
            "design_set_manifest": of3.design_set,
        }
        if pre.msa_path is not None:
            inputs["target_msa"] = pre.msa_path

        return ToolPlan(
            kind=RunKind.EVALUATE,
            jobs=jobs,
            designs_per_task=per_task,
            designs_file=METRICS_FILE,
            archives={"driver": of3.driver_script},
            inputs=inputs,
            container=of3.runtime.container,
            workflow={
                "model": "of3_upstream",
                "metric_prefix": of3.metric_prefix,
                "protocol_sha256": of3.protocol_hash,
                "protocol": of3.protocol_fields,
                "readers": ["complex"],
                # The weights are not in the image, so the container digest
                # alone does not determine the result -- as for AF3, and unlike
                # Chai-1.
                "checkpoint": str(of3.runtime.checkpoint),
                "msa_server": False,
                "design_set_digest": pre.design_set.digest,
                "design_set_manifest": str(of3.design_set),
                "design_set_fasta": str(pre.fasta_path),
                "scope_id": pre.design_set.scope_id,
                "n_designs": pre.n_designs,
                "distinct_lengths": pre.distinct_lengths,
                "target_length": pre.target_length,
                "samples_per_design": of3.protocol.num_diffusion_samples,
                "lengths_per_shard": math.ceil(pre.distinct_lengths / jobs),
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, of3 = self.configs_of(manifest)
        return of3_upstream_launch_spec(general, of3, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)
