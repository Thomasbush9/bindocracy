"""Tool-specific rendering and output normalization."""

from bindocracy.adapters.base import CollectionError, OutputAdapter
from bindocracy.adapters.registry import (
    UnknownToolError,
    adapter_for,
    collect_run,
    register,
    registered_tools,
    unregister,
)

__all__ = [
    "CollectionError",
    "OutputAdapter",
    "UnknownToolError",
    "adapter_for",
    "collect_run",
    "register",
    "registered_tools",
    "unregister",
]
