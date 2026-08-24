"""haunt — local-first verbatim memory for AI agents."""

__version__ = "0.2.0"

from haunt.paths import haunt_home, namespace_db_path
from haunt.store import Store, observe, list_namespaces

__all__ = [
    "__version__",
    "haunt_home",
    "namespace_db_path",
    "Store",
    "observe",
    "list_namespaces",
]
