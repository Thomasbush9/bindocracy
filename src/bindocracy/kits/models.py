"""Declarative kit metadata; parsing never imports or executes kit code."""

from __future__ import annotations

import math
import re
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bindocracy.config.models import ConfigModel
from bindocracy.functions.models import FunctionInputKind, MetricDeclaration


class KitError(ValueError):
    """A kit, release, or integrity check was refused."""


_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\Z")
_ARGUMENT = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
Scalar = StrictStr | StrictBool | StrictInt | StrictFloat
Number = StrictInt | StrictFloat
Nonempty = Annotated[StrictStr, Field(min_length=1)]


def safe_segment(value: str) -> str:
    if not isinstance(value, str) or not _SEGMENT.fullmatch(value):
        raise KitError(f"Not a path-safe name/version: {value!r}")
    return value


def safe_relative(value: str) -> str:
    """Accept only unambiguous portable POSIX relative file paths."""
    if (
        not isinstance(value, str) or not value or "\\" in value or ":" in value
        or len(value) > 4096 or value.count("/") >= 128
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise KitError(f"Unsafe relative kit path: {value!r}")
    return value


class Verification(ConfigModel):
    files: list[StrictStr] = Field(default_factory=list)
    import_: StrictStr | None = Field(default=None, alias="import")
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


    @field_validator("files")
    @classmethod
    def safe_files(cls, value: list[str]) -> list[str]:
        for item in value:
            safe_relative(item)
        if len(value) != len(set(value)):
            raise ValueError("Duplicate verification file")
        return value

    @field_validator("import_")
    @classmethod
    def module_name(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value, re.ASCII):
            raise ValueError("verify.import must be a dotted Python module name")
        return value


class Requirement(ConfigModel):
    name: StrictStr
    kind: Literal["container", "weights", "source_overlay", "harness_module"]
    optional: StrictBool = False
    version: Nonempty | None = None
    provided_by: Literal["harness"] | None = None
    provides: Literal["interpreter"] | None = None
    target: StrictStr | None = None
    mount: Literal["read_only"] | None = None
    why: Nonempty | None = None
    verify: Verification = Field(default_factory=Verification)

    _name = field_validator("name")(safe_segment)

    @field_validator("target")
    @classmethod
    def absolute_target(cls, value: str | None) -> str | None:
        if value is not None:
            if not value.startswith("/"):
                raise ValueError("Dependency target must be absolute")
            safe_relative(value[1:])
        return value

    @model_validator(mode="after")
    def coherent_requirement(self) -> Self:
        if self.provides and self.kind != "container":
            raise ValueError("Only a container can provide the interpreter")
        if self.provided_by and self.kind != "harness_module":
            raise ValueError("Only a harness_module can be provided by the harness")
        if self.target and self.kind != "source_overlay":
            raise ValueError("Only a source_overlay can declare a target")
        return self


class Capabilities(ConfigModel):
    changes_length: StrictBool
    max_children: Annotated[StrictInt, Field(ge=1, le=64)]


class Cost(ConfigModel):
    gpu: Literal["none", "required"]
    compile_seconds: Number = Field(ge=0, allow_inf_nan=False)
    per_parent_seconds: Number = Field(ge=0, allow_inf_nan=False)
    measured_on: dict[str, JsonValue] = Field(default_factory=dict)
    scales_with: list[Nonempty] = Field(default_factory=list)
    note: Nonempty | None = None

    @field_validator("measured_on")
    @classmethod
    def finite_json(cls, value: dict) -> dict:
        pending = list(value.values())
        while pending:
            item = pending.pop()
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Cost evidence must contain finite JSON numbers")
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        return value


class Parameter(ConfigModel):
    type: Literal["string", "integer", "number", "boolean"]
    required: StrictBool = False
    enum: list[Scalar] | None = Field(default=None, min_length=1)
    minimum: Number | None = Field(default=None, allow_inf_nan=False)
    maximum: Number | None = Field(default=None, allow_inf_nan=False)

    def validate_value(self, value: object) -> None:
        allowed = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,)}
        if type(value) not in allowed[self.type]:
            raise ValueError(f"Expected parameter type {self.type}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Parameter numbers must be finite")
        if self.enum is not None and not any(type(value) is type(item) and value == item for item in self.enum):
            raise ValueError("Parameter is not in its declared enum")
        if self.minimum is not None and value < self.minimum:
            raise ValueError("Parameter is below its minimum")
        if self.maximum is not None and value > self.maximum:
            raise ValueError("Parameter is above its maximum")

    @model_validator(mode="after")
    def coherent_bounds(self) -> Self:
        if self.type not in {"integer", "number"} and (self.minimum is not None or self.maximum is not None):
            raise ValueError("Only numeric parameters can have bounds")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Parameter minimum exceeds maximum")
        if self.enum is not None:
            for value in self.enum:
                self.validate_value(value)
        return self


class KitManifest(ConfigModel):
    kit_schema: Literal[1]
    name: StrictStr
    version: StrictStr
    kind: Literal["optimizer"]
    summary: Nonempty
    entrypoint: StrictStr
    python: Nonempty
    requires: list[Requirement]
    inputs: list[FunctionInputKind] = Field(min_length=1)
    loss_models: list[Nonempty]
    capabilities: Capabilities
    metrics: dict[Nonempty, MetricDeclaration]
    cost: Cost
    args: dict[StrictStr, Scalar] = Field(default_factory=dict)
    dependency_args: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    parameters: dict[StrictStr, Parameter] = Field(default_factory=dict)
    publisher: StrictStr | None = None
    license: Nonempty | None = None
    source: Nonempty | None = None

    _name = field_validator("name", "version")(safe_segment)
    _entrypoint = field_validator("entrypoint")(safe_relative)

    @field_validator("publisher")
    @classmethod
    def publisher_claim(cls, value: str | None) -> str | None:
        return safe_segment(value) if value is not None else None

    @field_validator("version")
    @classmethod
    def exact_version(cls, value: str) -> str:
        if value.lower() == "latest":
            raise ValueError("A kit must declare an exact version, not latest")
        return value

    @field_validator("kit_schema", mode="before")
    @classmethod
    def integer_schema(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("kit_schema must be integer 1")
        return value

    @field_validator("python")
    @classmethod
    def python_constraint(cls, value: str) -> str:
        try:
            SpecifierSet(value)
        except InvalidSpecifier as exc:
            raise ValueError("Invalid Python version constraint") from exc
        return value

    @model_validator(mode="after")
    def coherent_manifest(self) -> Self:
        names = [requirement.name for requirement in self.requires]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate dependency name")
        for field in (self.inputs, self.loss_models):
            if len(field) != len(set(field)):
                raise ValueError("Duplicate input or loss model")
        for key in self.args.keys() | self.parameters.keys():
            if not _ARGUMENT.fullmatch(key):
                raise ValueError(f"Invalid argument name: {key!r}")
        for key, value in self.args.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Argument numbers must be finite")
            if key in self.parameters:
                self.parameters[key].validate_value(value)
        flags = list(self.dependency_args.values())
        if len(flags) != len(set(flags)):
            raise ValueError("Duplicate dependency argument flag")
        for name, flag in self.dependency_args.items():
            if name not in names:
                raise ValueError(f"Dependency argument names undeclared dependency {name!r}")
            if not flag.startswith("--") or not _ARGUMENT.fullmatch(flag[2:]):
                raise ValueError("Dependency argument must be one --name token")
            if flag[2:] in self.args or flag[2:] in self.parameters:
                raise ValueError("Dependency arguments cannot collide with parameters")
        return self
