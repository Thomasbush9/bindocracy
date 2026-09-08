"""One module per tool, and the registry that knows the set.

Adding a tool: write its module, then add one `register(...)` line below.
Removing one: delete both. Nothing else in the harness names a tool.
"""

import os
from importlib import import_module

from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.boltzgen import BoltzGenPlugin
from bindocracy.tools.freebindcraft import FreeBindCraftPlugin
from bindocracy.tools.genie3 import Genie3Plugin
from bindocracy.tools.mosaic import MosaicPlugin
from bindocracy.tools.protein_hunter import ProteinHunterPlugin
from bindocracy.tools.proteina_complexa import ProteinaComplexaPlugin
from bindocracy.tools.pxdesign import PXDesignPlugin
from bindocracy.tools.registry import (
    UnknownToolError,
    collect_run,
    launch_spec,
    load_configs,
    plan,
    plugin_for,
    register,
    registered_tools,
    resources,
    unregister,
)
from bindocracy.tools.scorer import ScorerPlugin

register(MosaicPlugin)
register(BoltzGenPlugin)
register(Genie3Plugin)
register(PXDesignPlugin)
register(ProteinHunterPlugin)
register(ProteinaComplexaPlugin)
register(FreeBindCraftPlugin)
register(ScorerPlugin)

# A tool does not have to live in this repository. Anything named in
# BINDOCRACY_PLUGINS (comma-separated importable modules) is imported here and
# registers itself the same way, which is also how the seam test drives a whole
# foreign tool through the real workflow.
for _module in os.environ.get("BINDOCRACY_PLUGINS", "").split(","):
    if _module.strip():
        import_module(_module.strip())

__all__ = [
    "BoltzGenPlugin",
    "FreeBindCraftPlugin",
    "Genie3Plugin",
    "MosaicPlugin",
    "PXDesignPlugin",
    "ProteinHunterPlugin",
    "ProteinaComplexaPlugin",
    "ScorerPlugin",
    "ToolPlugin",
    "UnknownToolError",
    "collect_run",
    "launch_spec",
    "load_configs",
    "plan",
    "plugin_for",
    "register",
    "registered_tools",
    "resources",
    "unregister",
]
