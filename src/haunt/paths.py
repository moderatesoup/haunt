"""Filesystem layout: ~/.haunt (or $HAUNT_HOME). Absolute paths only."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

SAFE_NS = re.compile(r"[^a-zA-Z0-9._-]+")


def haunt_home() -> Path:
    raw = (
        os.environ.get("HAUNT_HOME")
        or os.environ.get("LORE_HOME")
        or os.environ.get("ENGRAM_HOME")
    )
    if raw:
        return Path(raw).expanduser().resolve()
    default = Path.home() / ".haunt"
    legacy = Path.home() / ".lore"
    if not default.exists() and legacy.exists():
        return legacy.resolve()
    return default.resolve()


# Keep lore_home as an alias for backwards compatibility in internal code
lore_home = haunt_home


def models_dir() -> Path:
    raw = os.environ.get("HAUNT_MODEL_CACHE") or os.environ.get("LORE_MODEL_CACHE")
    if raw:
        return Path(raw).expanduser().resolve()
    return haunt_home() / "models"


def registry_path() -> Path:
    return haunt_home() / "registry.db"


def namespaces_dir() -> Path:
    return haunt_home() / "namespaces"


def bin_dir() -> Path:
    return haunt_home() / "bin"


def namespace_db_path(name: str) -> Path:
    return namespaces_dir() / f"{safe_name(name)}.db"


def safe_name(name: str) -> str:
    cleaned = SAFE_NS.sub("-", name.strip()).strip("-.")
    if not cleaned:
        return "default"
    return cleaned[:80]


def infer_namespace(cwd: Path | None = None) -> str:
    """Infer a namespace from git remote, repo folder, or cwd name."""
    env = (
        os.environ.get("HAUNT_NAMESPACE")
        or os.environ.get("LORE_NAMESPACE")
        or os.environ.get("ENGRAM_NAMESPACE")
    )
    if env:
        return safe_name(env)
    if cwd is None:
        proj = os.environ.get("CURSOR_PROJECT_DIR") or os.environ.get("CLAUDE_PROJECT_DIR")
        root = Path(proj).expanduser().resolve() if proj else Path.cwd().resolve()
    else:
        root = cwd.resolve()
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if remote.returncode == 0:
            url = remote.stdout.strip().rstrip("/")
            leaf = url.split("/")[-1].removesuffix(".git")
            if leaf:
                return safe_name(leaf)
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if top.returncode == 0:
            return safe_name(Path(top.stdout.strip()).name)
    except (OSError, subprocess.SubprocessError):
        pass
    if root.name and root.name not in {".", "/", ""}:
        return safe_name(root.name)
    return "default"


def resolve_namespace(name: str | None = None, cwd: Path | None = None) -> str:
    if name:
        return safe_name(name)
    return infer_namespace(cwd)


def ensure_layout() -> Path:
    home = haunt_home()
    for p in (home, namespaces_dir(), bin_dir(), models_dir()):
        p.mkdir(parents=True, exist_ok=True)
    return home
