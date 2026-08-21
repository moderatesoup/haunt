"""engram (package: lore) — local-first verbatim memory for AI agents."""

__version__ = "0.1.0"

from lore.paths import lore_home, namespace_db_path
from lore.store import Store, observe, list_namespaces

__all__ = [
    "__version__",
    "lore_home",
    "namespace_db_path",
    "Store",
    "observe",
    "list_namespaces",
]
