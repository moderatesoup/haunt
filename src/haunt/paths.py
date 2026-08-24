"""Filesystem layout: ~/.haunt (or $HAUNT_HOME). Absolute paths only."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

SAFE_NS = re.compile(r"[^a-zA-Z0-9._-]+")

DIR_MODE = 0o700
FILE_MODE = 0o600


def haunt_home() -> Path:
    raw = os.environ.get("HAUNT_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".haunt").resolve()


def models_dir() -> Path:
    raw = os.environ.get("HAUNT_MODEL_CACHE")
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
    env = os.environ.get("HAUNT_NAMESPACE")
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


def _is_user_home(path: Path) -> bool:
    try:
        return path.resolve() == Path.home().resolve()
    except OSError:
        return False


def mkdir_private(path: Path) -> Path:
    """Create *path* (and missing parents) as 0700. Never chmod the user's home."""
    path.mkdir(parents=True, exist_ok=True)
    if not _is_user_home(path):
        path.chmod(DIR_MODE)
    return path


def tighten_db_files(path: Path) -> None:
    """chmod 0600 on a sqlite db and sibling WAL/SHM, if they exist."""
    for extra in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if extra.is_file() and not _is_user_home(extra):
            extra.chmod(FILE_MODE)


def repair_private_modes(root: Path | None = None) -> list[str]:
    """Tighten HAUNT_HOME dirs to 0700 and sqlite files to 0600.

    Never chmod the user's home directory. Only touches haunt layout paths
    (home, namespaces/, bin/, models/) and sqlite db/WAL/SHM files.
    """
    home = (root or haunt_home()).resolve()
    changed: list[str] = []

    def _chmod(path: Path, mode: int) -> None:
        if not path.exists() or _is_user_home(path):
            return
        current = path.stat().st_mode & 0o777
        if current != mode:
            path.chmod(mode)
            changed.append(str(path))

    if not _is_user_home(home):
        _chmod(home, DIR_MODE)
    for d in (home / "namespaces", home / "bin", home / "models"):
        if d.is_dir():
            _chmod(d, DIR_MODE)
    registry = home / "registry.db"
    for f in (registry, Path(str(registry) + "-wal"), Path(str(registry) + "-shm")):
        _chmod(f, FILE_MODE)
    ns_dir = home / "namespaces"
    if ns_dir.is_dir():
        for f in ns_dir.iterdir():
            name = f.name
            if name.endswith(".db") or name.endswith(".db-wal") or name.endswith(".db-shm"):
                _chmod(f, FILE_MODE)
    return changed


def ensure_layout() -> Path:
    home = haunt_home()
    for p in (home, namespaces_dir(), bin_dir(), models_dir()):
        mkdir_private(p)
    return home
