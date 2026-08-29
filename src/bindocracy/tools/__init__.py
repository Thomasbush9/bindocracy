"""One module per tool, and the registry that knows the set.

Adding a tool: write its module, then add one `register(...)` line below.
Removing one: delete both. Nothing else in the harness names a tool.
"""

from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.boltzgen import BoltzGenPlugin
from bindocracy.tools.mosaic import MosaicPlugin
from bindocracy.tools.registry import (
    DuplicateRunNameError,
    UnknownToolError,
    collect_run,
    index_configs,
    launch_spec,
    load_configs,
    plan,
    plugin_for,
    register,
    registered_tools,
    resources,
    unregister,
)

register(MosaicPlugin)
register(BoltzGenPlugin)

__all__ = [
    "BoltzGenPlugin",
    "DuplicateRunNameError",
    "MosaicPlugin",
    "ToolPlugin",
    "UnknownToolError",
    "collect_run",
    "index_configs",
    "launch_spec",
    "load_configs",
    "plan",
    "plugin_for",
    "register",
    "registered_tools",
    "resources",
    "unregister",
]
