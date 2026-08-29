"""The single DuckDB writer of the generation workflow.

The staging bundle carries no configuration, so the config pair comes from the
run manifest, which captured it when the run was planned. That keeps ingestion
independent of authored YAML that may have been edited since.
"""

from __future__ import annotations

from pathlib import Path

from bindocracy.runs.manifest import MANIFEST_NAME, RunManifest
from bindocracy.runs.staging import read_collected
from bindocracy.store import CampaignStore


def ingest_bundle(database: str | Path, collected_path: str | Path) -> bool:
    """Ingest one `collected.json`. False means it was already present."""
    collected = read_collected(collected_path)
    manifest = RunManifest.read(Path(collected.run.output_uri) / MANIFEST_NAME)
    if manifest.run_id != collected.run.run_id:
        raise ValueError(
            f"bundle {collected_path} claims run {collected.run.run_id}, but its "
            f"manifest describes run {manifest.run_id}"
        )
    with CampaignStore(database) as store:
        # One database, one target. Checked before anything is written.
        if manifest.target is not None:
            store.assert_target(manifest.target.name, manifest.target.sequence_sha256)
        return store.ingest(collected, configs=[manifest.config])
