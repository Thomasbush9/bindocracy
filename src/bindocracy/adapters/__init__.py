"""The contract every tool's output adapter implements.

Adapters themselves live with their tool, in tools/<tool>/adapter.py.
"""

from bindocracy.adapters.base import CollectionError, OutputAdapter

__all__ = ["CollectionError", "OutputAdapter"]
