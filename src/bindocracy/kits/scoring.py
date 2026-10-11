"""Resolve scoring kits to the ordinary function runtime, with one locked contract."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from bindocracy.kits.models import KitError
from bindocracy.kits.resolve import KitLock, load_lock, verify_lock

if TYPE_CHECKING:
    from bindocracy.functions.models import CustomFunction
    from bindocracy.functions.run import FunctionRunConfig


_OWNED = frozenset({"script", "args", "inputs", "metrics", "kit_lock", "container", "container_python", "dev_source"})


def _owned_fields(lock: KitLock) -> dict[str, Any]:
    if lock.manifest.kind != "scoring_function":
        raise KitError("scoring configuration requires a scoring_function kit")
    return {
        "script": lock.kit_root / lock.manifest.entrypoint,
        "args": tuple(lock.args),
        "inputs": tuple(lock.manifest.inputs),
        "metrics": lock.manifest.metrics,
        "container": Path(lock.runtime["container"]) if lock.runtime.get("container") else None,
        "container_python": lock.runtime.get("container_python", "python"),
        "dev_source": Path(lock.runtime["dev_source"]) if lock.runtime.get("dev_source") else None,
    }


def function_config(lock: KitLock, request: dict, lock_path: Path) -> FunctionRunConfig:
    """Inject reviewed code/metrics/runtime; retain campaign-owned resource choices.

    The optional function mapping may choose name, prefix and timeout only.
    Digest approval is an execution decision, never a sandbox guarantee.
    """
    from bindocracy.functions.run import FunctionRunConfig

    verify_lock(lock)
    owned = _owned_fields(lock)
    if not isinstance(request, dict):
        raise KitError("scoring request must be a mapping")
    overrides = _OWNED.intersection(request)
    if overrides or "builtin" in request or "runtime" in request:
        raise KitError("scoring request cannot override kit-owned fields or select a builtin/runtime")
    choices = request.get("function", {})
    if not isinstance(choices, dict) or set(choices) - {"name", "prefix", "timeout_seconds"}:
        raise KitError("scoring request function may only set name, prefix and timeout_seconds; kit-owned fields cannot be overridden")
    model = FunctionRunConfig.model_validate({
        **request,
        "function": {
            "name": request.get("name"),
            **choices,
            **owned,
            "kit_lock": Path(lock_path).resolve(),
        },
    })
    return model


def validate_function_lock(function: CustomFunction) -> KitLock | None:
    """Verify the complete bundle and agreement with the effective function."""
    if function.kit_lock is None:
        return None
    lock = load_lock(function.kit_lock)
    for field, expected in _owned_fields(lock).items():
        if getattr(function, field) != expected:
            raise KitError(f"kit-owned function field {field!r} differs from its lock")
    return lock
