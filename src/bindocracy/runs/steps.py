"""One function per workflow rule.

The Snakefile should say what depends on what, not how each step works. Keeping
the bodies here means they are importable, testable without Snakemake, and
readable by someone who does not know Snakemake's dialect — and it is what
`snakemake --lint` asks for when it objects to long `run:` directives.

Every step writes a one-line log, because a rule with no log leaves a failure
visible only in whichever terminal happened to run it.
"""

from __future__ import annotations

from pathlib import Path

from bindocracy.config.load import LoadedConfigs
from bindocracy.runs.ingest import ingest_bundle
from bindocracy.runs.launch import LaunchSpec
from bindocracy.runs.manifest import RunManifest, write_json_atomic
from bindocracy.runs.staging import write_collected
from bindocracy.runs.status import run_task


def prepare_step(
    loaded: LoadedConfigs, run_dir: str | Path, name: str, log_path: str | Path
) -> RunManifest:
    """Create or reuse this run's identity, directory, and archived inputs."""
    from bindocracy.tools.registry import plan

    manifest = plan(loaded, run_dir, name=name)
    _log(log_path, f"planned {manifest.run_id} with {len(manifest.tasks)} task(s)")
    return manifest


def generate_step(
    manifest_path: str | Path, task_id: int, log_path: str | Path, status_path: str | Path
) -> int:
    """Run one task of a planned run.

    Reads the manifest rather than the authored YAML, and refuses to start if a
    recorded input has changed since planning. The tool's own output goes to
    `log_path`; an ordinary tool failure is recorded in `status_path` rather
    than raised, so collection still runs.
    """
    from bindocracy.tools.registry import launch_spec

    manifest = RunManifest.read(manifest_path)
    manifest.verify_inputs()
    spec: LaunchSpec = launch_spec(manifest, task_id)
    return run_task(spec.argv, spec.env, log_path, status_path, task_id)


def collect_step(
    manifest_path: str | Path, output_path: str | Path, log_path: str | Path
) -> Path:
    """Parse every planned task into a staging bundle, keeping partial work."""
    from bindocracy.tools.registry import collect_run

    collected = collect_run(manifest_path)
    written = write_collected(collected, output_path)
    _log(
        log_path,
        f"{collected.run.status}: {collected.run.n_produced}"
        f"/{collected.run.n_requested} produced",
    )
    return written


def ingest_step(
    database: str | Path,
    bundle_path: str | Path,
    output_path: str | Path,
    log_path: str | Path,
) -> bool:
    """Write one bundle to the campaign database, once."""
    inserted = ingest_bundle(database, bundle_path)
    write_json_atomic(output_path, {"database": str(database), "inserted": inserted})
    _log(log_path, "ingested" if inserted else "already ingested")
    return inserted


def _log(path: str | Path, message: str) -> None:
    log = Path(path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(message + "\n")
