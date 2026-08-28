"""The run manifest: persistent identity and layout for one execution.

`run.json` is the stable handoff between planning, generation, collection, and
ingestion. It is created before any GPU work starts and is never rewritten, so
a restarted Snakemake execution reuses the same `run_id` and the same archived
inputs instead of quietly starting a second run.

Layout, all paths inside the manifest relative to the run directory:

    runs/<name>/
    |-- run.json
    |-- provenance/{general.yaml,mosaic.yaml,<driver>.py}
    |-- tasks/0000/{designs.jsonl,status.json}
    |-- logs/
    `-- collected.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict

from bindocracy.config.load import LoadedMosaicConfigs, sha256_file
from bindocracy.store.records import ConfigRecord, RunKind, RunRecord, new_id, utc_now

MANIFEST_NAME = "run.json"
DESIGNS_FILE = "designs.jsonl"
STATUS_FILE = "status.json"


class ManifestError(RuntimeError):
    """An existing run directory cannot be reused for the requested configs."""


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskPlan(ManifestModel):
    """One Mosaic task: one process, one GPU, one output directory."""

    task_id: int
    directory: str
    designs: str
    status: str
    log: str
    n_requested: int


class ArchivedFile(ManifestModel):
    """A file copied into provenance/ so the run stays inspectable."""

    path: str
    source_uri: str
    sha256: str


class RunManifest(ManifestModel):
    schema_version: Literal[1] = 1
    run_id: str
    name: str
    tool: str
    kind: RunKind
    general_config_id: str
    model_config_id: str
    created_at: datetime
    run_dir: str
    tasks: tuple[TaskPlan, ...]
    designs_per_task: int
    resources: dict[str, Any]
    provenance: dict[str, ArchivedFile]
    container: str
    container_digest: str | None
    code_revision: str | None
    workflow: dict[str, Any]
    # Carried whole so ingestion can insert the config pair without re-reading
    # and re-validating YAML that may have changed since the run was planned.
    config: ConfigRecord

    @property
    def directory(self) -> Path:
        return Path(self.run_dir)

    def path(self, relative: str) -> Path:
        return self.directory / relative

    @classmethod
    def read(cls, manifest_path: str | Path) -> Self:
        return cls.model_validate_json(Path(manifest_path).read_text())

    def to_run_record(self) -> RunRecord:
        """The run row this manifest describes, before any output is parsed."""
        return RunRecord(
            run_id=self.run_id,
            name=self.name,
            tool=self.tool,
            kind=self.kind,
            model_config_id=self.model_config_id,
            n_requested=self.designs_per_task * len(self.tasks),
            container_digest=self.container_digest,
            code_revision=self.code_revision,
            workflow_metadata=self.workflow,
            resources=self.resources,
            output_uri=self.run_dir,
            created_at=self.created_at,
        )


def plan_mosaic_run(
    loaded: LoadedMosaicConfigs,
    run_dir: str | Path,
    *,
    name: str | None = None,
) -> RunManifest:
    """Create the run directory and manifest, or reuse a matching existing one.

    Restart-safe: an existing manifest for the same config pair is returned
    unchanged, keeping its run ID and its archived copies of the inputs.
    """
    directory = Path(run_dir).resolve()
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.is_file():
        return _reuse(manifest_path, loaded)

    sampling = loaded.mosaic.sampling
    config = loaded.to_record()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "provenance").mkdir(exist_ok=True)
    (directory / "logs").mkdir(exist_ok=True)

    tasks = []
    for task_id in range(sampling.jobs):
        task_dir = f"tasks/{task_id:04d}"
        (directory / task_dir).mkdir(parents=True, exist_ok=True)
        tasks.append(
            TaskPlan(
                task_id=task_id,
                directory=task_dir,
                designs=f"{task_dir}/{DESIGNS_FILE}",
                status=f"{task_dir}/{STATUS_FILE}",
                log=f"logs/task-{task_id:04d}.log",
                n_requested=sampling.designs_per_job,
            )
        )

    provenance = {
        "general": _archive(loaded.general_path, directory, "general.yaml"),
        "model": _archive(loaded.model_path, directory, "mosaic.yaml"),
        "driver": _archive(
            loaded.mosaic.driver.script, directory, loaded.mosaic.driver.script.name
        ),
    }

    manifest = RunManifest(
        run_id=new_id(),
        name=name or loaded.mosaic.name,
        tool=loaded.mosaic.tool,
        kind=RunKind.GENERATE,
        general_config_id=config.general_config_id,
        model_config_id=config.model_config_id,
        created_at=utc_now(),
        run_dir=str(directory),
        tasks=tuple(tasks),
        designs_per_task=sampling.designs_per_job,
        resources=loaded.mosaic.resources.model_dump(mode="json"),
        provenance=provenance,
        container=str(loaded.mosaic.runtime.container),
        container_digest=None,
        code_revision=_code_revision(),
        workflow={
            "engine": "snakemake",
            "target_length": loaded.preflight.target_length,
            "binder_length": sampling.binder_length,
            "seed_base": sampling.seed_base,
        },
        config=config,
    )
    _write_atomic(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    return manifest


def _reuse(manifest_path: Path, loaded: LoadedMosaicConfigs) -> RunManifest:
    manifest = RunManifest.read(manifest_path)
    config = loaded.to_record()
    if manifest.model_config_id != config.model_config_id:
        raise ManifestError(
            f"{manifest_path} was planned for model_config_id="
            f"{manifest.model_config_id}, not {config.model_config_id}; "
            "use a different run directory"
        )
    return manifest


def _archive(source: Path, directory: Path, filename: str) -> ArchivedFile:
    """Copy one input into provenance/ and record where it came from."""
    destination = directory / "provenance" / filename
    shutil.copyfile(source, destination)
    return ArchivedFile(
        path=f"provenance/{filename}",
        source_uri=str(source.resolve()),
        sha256=sha256_file(destination),
    )


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _code_revision() -> str | None:
    """The bindocracy commit that planned this run, when there is one."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Serialize JSON-compatible data so a reader never sees a half file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(destination, json.dumps(payload, indent=2) + "\n")
    return destination
