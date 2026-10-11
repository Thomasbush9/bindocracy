"""Resolve public attempt identity without treating it as authorization."""
from __future__ import annotations

import os
import re
from pathlib import Path


def execution_identity(
    *, workspace: str | Path | None = None, attempt_id: str | None = None,
) -> dict[str, str]:
    workspace_path = workspace or os.environ.get("BINDOCRACY_WORKSPACE")
    requested = attempt_id or os.environ.get("BINDOCRACY_ATTEMPT_ID")
    identity: dict[str, str] = {}
    if workspace_path:
        from bindocracy.benchmark.workspace import load_workspace

        manifest = load_workspace(workspace_path)
        if requested is not None and requested != manifest.attempt_id:
            raise ValueError("attempt identity differs from initialized workspace")
        requested = manifest.attempt_id
        identity.update(
            task_digest=manifest.task_digest,
            environment_digest=manifest.environment_digest,
        )
    if requested is not None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", requested):
            raise ValueError("attempt_id must contain only letters, digits, '.', '_' or '-'")
        identity["attempt_id"] = requested
    return identity
