"""The single DuckDB writer of the generation workflow.

The staging bundle carries no configuration, so the config pair comes from the
run manifest, which captured it when the run was planned. That keeps ingestion
independent of authored YAML that may have been edited since.
"""

from __future__ import annotations

from pathlib import Path

from bindocracy.runs.manifest import MANIFEST_NAME, RunManifest
from bindocracy.runs.staging import read_collected, staged_digest
from bindocracy.store import CampaignStore, CollectedRun, ConfigRecord


def ingest_bundle(database: str | Path, collected_path: str | Path) -> bool:
    """Ingest one `collected.json`. False means it was already present."""
    collected = read_collected(collected_path)
    manifest = RunManifest.read(Path(collected.run.output_uri) / MANIFEST_NAME)
    if manifest.run_id != collected.run.run_id:
        raise ValueError(
            f"bundle {collected_path} claims run {collected.run.run_id}, but its "
            f"manifest describes run {manifest.run_id}"
        )
    target = (
        (manifest.target.name, manifest.target.sequence_sha256)
        if manifest.target is not None
        else None
    )
    with CampaignStore(database) as store:
        return store.ingest(
            collected,
            configs=[manifest.config],
            digest=staged_digest(collected_path),
            target=target,
        )


def ingest_collected(
    database: str | Path,
    collected: CollectedRun,
    *,
    config: ConfigRecord,
    target: tuple[str, str] | None = None,
) -> bool:
    """Ingest a run that has no run directory. False means already present.

    The database-to-database shape `filters/config.py` names: filtering,
    clustering and ranking read the campaign, evaluate arithmetic, and write
    verdicts. They have no container, no task fan-out and no run directory, so
    there is no `run.json` for `ingest_bundle` to read the config pair out of --
    the caller passes it instead, because it built it.

    Everything downstream is unchanged. `CampaignStore.ingest` is already
    generic over what a `CollectedRun` holds, and the restart-safety it
    provides (an identical re-ingest is a no-op, a changed one raises) applies
    here exactly as it does to a GPU run.
    """
    with CampaignStore(database) as store:
        return store.ingest(
            collected,
            configs=[config],
            # The verdicts, not the wall clock. See `CollectedRun.verdict_hash`.
            digest=collected.verdict_hash(),
            target=target,
        )
