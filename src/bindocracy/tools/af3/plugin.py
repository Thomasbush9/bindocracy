"""AlphaFold 3: co-fold every design in a set against the target.

The third scorer, over the same design set and metric registry as the mosaic
scorer and Chai-1. See `config.py` for why it is its own plugin.
"""

from __future__ import annotations

import math
from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources
from bindocracy.runs.manifest import RunManifest, ToolPlan
from bindocracy.store.records import RunKind
from bindocracy.tools.af3.adapter import METRICS_FILE, AF3OutputAdapter
from bindocracy.tools.af3.config import AF3Config
from bindocracy.tools.af3.launch import af3_launch_spec
from bindocracy.tools.af3.preflight import AF3Preflight, preflight_af3
from bindocracy.tools.base import ToolPlugin


class AF3Plugin(ToolPlugin):
    tool = "af3"
    config_type = AF3Config
    adapter_type = AF3OutputAdapter

    def preflight(self, general: GeneralConfig, model: AF3Config) -> AF3Preflight:
        return preflight_af3(general, model)

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        af3: AF3Config = loaded.model
        pre: AF3Preflight = loaded.preflight
        jobs = af3.sharding.jobs
        per_task = -(-pre.n_designs // jobs)

        inputs = {
            "target_fasta": loaded.general.target.sequence_fasta,
            "design_set": pre.fasta_path,
            "design_set_manifest": af3.design_set,
        }
        if pre.msa_path is not None:
            inputs["target_msa"] = pre.msa_path

        return ToolPlan(
            kind=RunKind.EVALUATE,
            jobs=jobs,
            designs_per_task=per_task,
            designs_file=METRICS_FILE,
            archives={"driver": af3.driver_script},
            inputs=inputs,
            container=af3.runtime.container,
            workflow={
                "model": "af3",
                "metric_prefix": af3.metric_prefix,
                "protocol_sha256": af3.protocol_hash,
                "protocol": af3.protocol_fields,
                "readers": ["complex"],
                # Recorded because the weights are NOT in the container, so the
                # container digest alone does not determine the result here --
                # unlike Chai-1, and for a licensing reason rather than an
                # oversight.
                "model_dir": str(af3.runtime.model_dir),
                "data_pipeline": "disabled",
                "design_set_digest": pre.design_set.digest,
                "design_set_manifest": str(af3.design_set),
                "design_set_fasta": str(pre.fasta_path),
                "scope_id": pre.design_set.scope_id,
                "n_designs": pre.n_designs,
                "distinct_lengths": pre.distinct_lengths,
                "target_length": pre.target_length,
                "samples_per_design": af3.protocol.num_diffn_samples,
                "lengths_per_shard": math.ceil(pre.distinct_lengths / jobs),
            },
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        general, af3 = self.configs_of(manifest)
        return af3_launch_spec(general, af3, manifest, task_id)

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)
