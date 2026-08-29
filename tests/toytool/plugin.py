"""The whole of a tool, in one file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from bindocracy.adapters import OutputAdapter
from bindocracy.adapters.common import artifact, read_manifest, run_status, run_window
from bindocracy.config.load import LoadedConfigs
from bindocracy.config.models import ConfigModel, GeneralConfig, ToolConfig
from bindocracy.config.preflight import ConfigPreflightError, read_single_fasta
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest, ToolPlan
from bindocracy.runs.status import read_task_status
from bindocracy.store.records import (
    CandidateType,
    CollectedRun,
    DesignRecord,
    MetricDirection,
    MetricRecord,
    RunRecord,
    stable_id,
)
from bindocracy.tools.base import ToolPlugin

OUTPUT_FILE = "toy_designs.csv"


class ToySamplingConfig(ConfigModel):
    jobs: int = Field(default=1, gt=0)
    designs_per_job: int = Field(gt=0)


class ToyRuntimeConfig(ConfigModel):
    script: Path


class ToyConfig(ToolConfig):
    schema_version: Literal[1]
    tool: Literal["toy"]
    sampling: ToySamplingConfig
    runtime: ToyRuntimeConfig


class ToyPreflight(ConfigModel):
    target_sequence: str

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


class ToyAdapter(OutputAdapter):
    tool = "toy"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = read_manifest(run_dir)
        designs, metrics, artifacts = [], [], []
        statuses = []
        for task in manifest.tasks:
            statuses.append(read_task_status(run_dir / task.status))
            path = run_dir / task.designs
            if path.is_file():
                for row in csv.DictReader(path.open(newline="")):
                    design = DesignRecord(
                        design_id=stable_id("design", run.run_id, row["name"]),
                        run_id=run.run_id, native_id=row["name"],
                        candidate_type=CandidateType.SEQUENCE, sequence=row["seq"],
                        created_at=manifest.created_at,
                    )
                    designs.append(design)
                    metrics.append(MetricRecord(
                        metric_id=stable_id("metric", run.run_id, design.design_id, "toy_score", "0"),
                        run_id=run.run_id, design_id=design.design_id,
                        name="toy_score", value=float(row["score"]),
                        direction=MetricDirection.MAX, measured_at=design.created_at,
                    ))
            record = artifact(run_dir, run.run_id, task.designs, "native_design_table")
            if record is not None:
                artifacts.append(record)

        started, finished = run_window(statuses)
        requested = manifest.designs_per_task * len(manifest.tasks)
        return CollectedRun(
            run=run.model_copy(update={
                "status": run_status(len(designs), statuses, len(designs) >= requested),
                "n_requested": requested, "n_attempted": requested,
                "n_produced": len(designs),
                "started_at": started, "finished_at": finished,
            }),
            designs=tuple(designs), metrics=tuple(metrics), artifacts=tuple(artifacts),
        )

    def succeeded(self, run_dir: Path) -> bool:
        manifest = read_manifest(run_dir)
        return all((run_dir / t.designs).is_file() for t in manifest.tasks)


class ToyPlugin(ToolPlugin):
    tool = "toy"
    config_type = ToyConfig
    adapter_type = ToyAdapter

    def preflight(self, general: GeneralConfig, model: ToyConfig) -> ToyPreflight:
        if not model.runtime.script.is_file():
            raise ConfigPreflightError(f"toy script does not exist: {model.runtime.script}")
        return ToyPreflight(target_sequence=read_single_fasta(general.target.sequence_fasta))

    def tool_plan(self, loaded: LoadedConfigs) -> ToolPlan:
        return ToolPlan(
            jobs=loaded.model.sampling.jobs,
            designs_per_task=loaded.model.sampling.designs_per_job,
            designs_file=OUTPUT_FILE,
            archives={"script": loaded.model.runtime.script},
            container=loaded.model.runtime.script,
            workflow={"target_length": loaded.preflight.target_length},
            inputs={"target_fasta": loaded.general.target.sequence_fasta},
        )

    def launch_spec(self, manifest: RunManifest, task_id: int) -> LaunchSpec:
        import sys

        _, model = self.configs_of(manifest)
        task = task_of(manifest, task_id)
        run_dir = manifest.directory
        return LaunchSpec(
            argv=(sys.executable, str(run_dir / manifest.provenance["script"].path),
                  str(run_dir / task.directory), str(task.n_requested), str(task_id)),
            env={},
            resources=slurm_resources(
                GeneralConfig.model_validate(manifest.config.general_config_json).cluster,
                model.resources),
            log=run_dir / task.log,
            outputs=(run_dir / task.designs,),
        )

    def resources(self, loaded: LoadedConfigs) -> dict[str, Any]:
        return slurm_resources(loaded.general.cluster, loaded.model.resources)
