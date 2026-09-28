"""Turn a reviewed kit lock and campaign choices into an optimizer configuration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from bindocracy.kits.models import KitError
from bindocracy.kits.resolve import KitLock, load_lock, verify_lock

if TYPE_CHECKING:
    from bindocracy.tools.optimize.config import OptimizeConfig


_DRIVER = Path(__file__).resolve().parents[3] / "drivers" / "optimize" / "run_optimizer.py"
_OWNED = frozenset({"script", "args", "inputs", "loss_models", "metrics", "driver_script", "kit_lock"})


def _owned_fields(lock: KitLock) -> dict[str, Any]:
    return {
        "script": lock.kit_root / lock.manifest.entrypoint,
        "args": tuple(lock.args),
        "inputs": tuple(lock.manifest.inputs),
        "loss_models": tuple(lock.manifest.loss_models),
        "metrics": lock.manifest.metrics,
        "driver_script": _DRIVER,
    }


def _validate_effective(lock: KitLock, model: OptimizeConfig) -> None:
    from bindocracy.tools.optimize.config import OptimizerRuntime

    if model.tool != "optimize":
        raise KitError("optimizer kit configuration must use tool='optimize'")
    for field, expected in _owned_fields(lock).items():
        if getattr(model, field) != expected:
            raise KitError(f"kit-owned configuration field {field!r} differs from its lock")
    expected_runtime = OptimizerRuntime.model_validate(lock.runtime)
    for field in ("container", "container_python", "dev_source"):
        if getattr(model.runtime, field) != getattr(expected_runtime, field):
            raise KitError(f"kit-owned runtime field {field!r} differs from its lock")
    capabilities = lock.manifest.capabilities
    if model.max_children > capabilities.max_children:
        raise KitError(
            f"max_children={model.max_children} exceeds kit capability "
            f"max_children={capabilities.max_children}"
        )
    if not capabilities.changes_length and model.length_delta != 0:
        raise KitError("length_delta must be zero for a kit declaring changes_length=false")


def optimizer_config(lock: KitLock, request: dict, lock_path: Path) -> OptimizeConfig:
    """Keep campaign choices separate from locked code and its declarations.

    Approval permits execution of reviewed/trusted code; it is not a sandbox or
    a statement about a publisher's identity.
    """
    from bindocracy.tools.optimize.config import OptimizeConfig

    verify_lock(lock)
    if not isinstance(request, dict):
        raise KitError("optimizer request must be a mapping")
    overrides = sorted(_OWNED.intersection(request))
    if overrides:
        raise KitError(f"optimizer request cannot override kit-owned fields: {overrides}")
    runtime = request.get("runtime", {})
    if not isinstance(runtime, dict) or set(runtime) - {"scratch"}:
        raise KitError("optimizer request runtime may only set scratch")
    model = OptimizeConfig.model_validate(
        {
            **request,
            **_owned_fields(lock),
            "kit_lock": Path(lock_path).resolve(),
            "runtime": {**lock.runtime, **runtime},
        }
    )
    _validate_effective(lock, model)
    return model


def validate_optimizer_lock(model: OptimizeConfig) -> KitLock | None:
    """Recheck the entire closure and its agreement with the effective config."""
    if model.kit_lock is None:
        return None
    lock = load_lock(model.kit_lock)
    _validate_effective(lock, model)
    return lock
