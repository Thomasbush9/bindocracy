"""Run identity: plan a run, launch its tasks, stage its collected output."""

from bindocracy.runs.ingest import ingest_bundle
from bindocracy.runs.launch import LaunchSpec, mosaic_launch_spec, mosaic_resources
from bindocracy.runs.manifest import (
    MANIFEST_NAME,
    ManifestError,
    RunManifest,
    plan_mosaic_run,
)
from bindocracy.runs.staging import read_collected, write_collected

__all__ = [
    "MANIFEST_NAME",
    "LaunchSpec",
    "ManifestError",
    "RunManifest",
    "ingest_bundle",
    "mosaic_launch_spec",
    "mosaic_resources",
    "plan_mosaic_run",
    "read_collected",
    "write_collected",
]
