"""The workflow index: which executions to perform, by name.

A run directory used to be named after its config's filename stem, which made
the filename do two jobs at once. Re-running the same configuration meant
copying it under a new name, two tools could not share a filename, and a
directory of `run_10.yaml`, `run_11.yaml` accumulated for no scientific reason.

An execution now names itself:

    runs:
      - name: mosaic-run-12
        config: /path/to/mosaic.yaml

The name is the run directory; the config is the science. Running the same
config twice means two entries with two names, which is exactly what "a run ID
identifies an execution, not a configuration" was always supposed to mean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from bindocracy.config.load import ConfigLoadError
from bindocracy.config.models import ConfigModel


class RunRequest(ConfigModel):
    """One execution: a directory name and the model config to run in it."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    config: Path


class WorkflowIndex(ConfigModel):
    """An execution index. It holds no science, only what to run and where."""

    database: Path
    run_root: Path
    general_config: Path
    runs: tuple[RunRequest, ...] = Field(min_length=1)
    # Present only in a frozen campaign index. Native workflow indexes retain
    # their normal authored-config behavior.
    campaign_plan: Path | None = None

    @model_validator(mode="after")
    def names_must_be_unique(self) -> Self:
        seen: set[str] = set()
        for request in self.runs:
            if request.name in seen:
                raise ValueError(
                    f"two runs are both named {request.name!r}; the name is the "
                    "run directory, so they would collide"
                )
            seen.add(request.name)
        return self

    def run_dir(self, name: str) -> Path:
        return self.run_root / name

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(request.name for request in self.runs)

    def config_for(self, name: str) -> Path:
        for request in self.runs:
            if request.name == name:
                return request.config
        raise KeyError(f"no run named {name!r} in this workflow index")


def read_workflow_index(config: dict) -> WorkflowIndex:
    """Validate a Snakemake config mapping as a workflow index.

    Strict: an unknown key is a typo worth failing on, and a config listed
    under the wrong tool heading used to be silently accepted.
    """
    try:
        return WorkflowIndex.model_validate(config)
    except Exception as error:
        raise ConfigLoadError(f"invalid workflow index: {error}") from error


def load_workflow_index(path: str | Path) -> WorkflowIndex:
    return read_workflow_index(yaml.safe_load(Path(path).read_text()))
