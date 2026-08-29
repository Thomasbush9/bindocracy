"""Generic run machinery: manifests, launch specs, execution, staging."""

from bindocracy.runs.ingest import ingest_bundle
from bindocracy.runs.launch import LaunchSpec, task_of
from bindocracy.runs.manifest import (
    MANIFEST_NAME,
    ManifestError,
    RunManifest,
    ToolPlan,
    plan_run,
)
from bindocracy.runs.staging import read_collected, write_collected
from bindocracy.runs.status import run_task, write_task_status

__all__ = [
    "MANIFEST_NAME",
    "LaunchSpec",
    "ManifestError",
    "RunManifest",
    "ToolPlan",
    "ingest_bundle",
    "plan_run",
    "read_collected",
    "run_task",
    "task_of",
    "write_collected",
    "write_task_status",
]
