"""A complete third tool, defined outside the library.

Its whole purpose is to be foreign: a different config shape, a different
output format, a different notion of what a design is. If driving it through
the real Snakefile requires editing anything in `src/bindocracy/`, the plugin
contract is not actually a contract.
"""

from bindocracy.tools import register
from tests.toytool.plugin import ToyPlugin

register(ToyPlugin)

__all__ = ["ToyPlugin"]
