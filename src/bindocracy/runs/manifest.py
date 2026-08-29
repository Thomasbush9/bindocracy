"""The run manifest: persistent identity and layout for one execution.

`run.json` is the stable handoff between planning, generation, collection, and
ingestion. It is created before any GPU work starts and is never rewritten, so
a restarted Snakemake execution reuses the same `run_id` and the same archived
inputs instead of quietly starting a second run.

Layout, all paths inside the manifest relative to the run directory:

    runs/<name>/
    |-- run.json
    |-- provenance/<driver>.py       the code that ran, archived and executed
    |-- tasks/0000/{designs.jsonl,status.json}
    |-- logs/
    `-- collected.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict

from bindocracy.config.load import LoadedConfigs, sha256_file
from bindocracy.store.records import ConfigRecord, RunKind, RunRecord, new_id, utc_now

MANIFEST_NAME = "run.json"
DESIGNS_FILE = "designs.jsonl"
STATUS_FILE = "status.json"


@dataclass(frozen=True)
class ToolPlan:
    """What a tool's plugin must supply to have a run planned for it.

    Everything below this line is generic. A tool contributes its task count,
    its per-task output name, and the files worth archiving; it never reaches
    into the manifest itself.
    """

    jobs: int
    designs_per_task: int
    designs_file: str
    archives: dict[str, Path]
    container: Path
    workflow: dict[str, Any]


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


def plan_run(
    loaded: LoadedConfigs,
    tool_plan: ToolPlan,
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

    config = loaded.to_record()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "provenance").mkdir(exist_ok=True)
    (directory / "logs").mkdir(exist_ok=True)

    tasks = []
    for task_id in range(tool_plan.jobs):
        task_dir = f"tasks/{task_id:04d}"
        (directory / task_dir).mkdir(parents=True, exist_ok=True)
        tasks.append(
            TaskPlan(
                task_id=task_id,
                directory=task_dir,
                designs=f"{task_dir}/{tool_plan.designs_file}",
                status=f"{task_dir}/{STATUS_FILE}",
                log=f"logs/task-{task_id:04d}.log",
                n_requested=tool_plan.designs_per_task,
            )
        )

    # Only inputs that are *consumed by the run* are archived -- Mosaic's driver
    # script, BoltzGen's design spec. The configs are carried whole in `config`
    # below and stored in the database, so copying their YAML would be a third
    # copy that nothing reads, and would make a file on a shared filesystem
    # load-bearing again.
    provenance = {
        label: _archive(source, directory, source.name)
        for label, source in tool_plan.archives.items()
    }

    manifest = RunManifest(
        run_id=new_id(),
        name=name or loaded.model.name,
        tool=loaded.model.tool,
        kind=RunKind.GENERATE,
        general_config_id=config.general_config_id,
        model_config_id=config.model_config_id,
        created_at=utc_now(),
        run_dir=str(directory),
        tasks=tuple(tasks),
        designs_per_task=tool_plan.designs_per_task,
        resources=loaded.model.resources.model_dump(mode="json"),
        provenance=provenance,
        container=str(tool_plan.container),
        container_digest=None,
        code_revision=_code_revision(),
        workflow={"engine": "snakemake", **tool_plan.workflow},
        config=config,
    )
    _write_atomic(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    return manifest


def _reuse(manifest_path: Path, loaded: LoadedConfigs) -> RunManifest:
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
