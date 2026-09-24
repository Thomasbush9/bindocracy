"""Strict site policy for a CPU-planned, explicitly approved campaign."""

from pydantic import Field, field_validator

from bindocracy.config.models import ConfigModel, slurm_walltime_seconds


class Controller(ConfigModel):
    account: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    partition: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    cpus: int = Field(gt=0, strict=True)
    memory_gb: int = Field(gt=0, strict=True)
    walltime: str
    gpus: int = Field(ge=0, strict=True)
    constraint: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.&|*\[\]-]+$")

    @field_validator("walltime")
    @classmethod
    def positive_walltime(cls, value: str) -> str:
        if slurm_walltime_seconds(value) <= 0:
            raise ValueError("controller walltime must be positive")
        return value


class Site(ConfigModel):
    controller: Controller
    max_workers: int = Field(gt=0, strict=True)
    max_total_gpus: int = Field(gt=0, strict=True)
    nofile: int = Field(default=65536, ge=1024, strict=True)
    latency_wait: int = Field(default=60, ge=0, strict=True)
