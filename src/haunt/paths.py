"""Filesystem layout: ~/.haunt (or $HAUNT_HOME). Absolute paths only."""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

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


def repository_identity(remote_url: str | None) -> str | None:
    """Return host/owner/repo identity for common git remote URL forms."""
    raw = (remote_url or "").strip().rstrip("/")
    if not raw:
        return None
    host = ""
    path = ""
    if "://" in raw:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").lower()
        path = parsed.path
    else:
        scp = re.match(r"^(?:[^@/\s]+@)?([^:/\s]+):(.+)$", raw)
        if scp and not re.match(r"^[A-Za-z]:[\\/]", raw):
            host = scp.group(1).lower()
            path = scp.group(2)
    parts = [part for part in path.strip("/").split("/") if part]
    if not host or len(parts) < 2:
        return None
    parts[-1] = parts[-1].removesuffix(".git")
    if not parts[-1]:
        return None
    return "/".join([host, *(part.lower() for part in parts)])


def namespace_for_repo_identity(identity: str) -> str:
    """Turn a remote identity into a stable, collision-resistant namespace."""
    cleaned = SAFE_NS.sub("-", identity.strip()).strip("-.") or "default"
    if len(cleaned) <= 80:
        return cleaned
    digest = sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:69]}-{digest}"


def _registered_namespace_for_repo(
    *, remote_identity: str | None, repo_root: Path | None
) -> str | None:
    """Preserve an existing namespace already registered to this repository."""
    path = registry_path()
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT name, repo_path FROM namespaces").fetchall()
        conn.close()
    except sqlite3.Error:
        return None
    resolved_root = repo_root.resolve() if repo_root else None
    for row in rows:
        stored = str(row["repo_path"] or "").strip()
        if not stored:
            continue
        if remote_identity and repository_identity(stored) == remote_identity:
            return safe_name(str(row["name"]))
        if resolved_root and repository_identity(stored) is None:
            try:
                if Path(stored).expanduser().resolve() == resolved_root:
                    return safe_name(str(row["name"]))
            except OSError:
                continue
    return None


def _git_repo_context(root: Path) -> tuple[str | None, Path | None]:
    """Return origin URL and repository root without raising on non-git paths."""
    remote_url: str | None = None
    repo_root: Path | None = None
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if remote.returncode == 0 and remote.stdout.strip():
            remote_url = remote.stdout.strip()
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if top.returncode == 0 and top.stdout.strip():
            repo_root = Path(top.stdout.strip()).resolve()
    except (OSError, subprocess.SubprocessError):
        pass
    return remote_url, repo_root


def infer_namespace(cwd: Path | None = None) -> str:
    """Infer from remote identity, preserving a matching legacy registration."""
    env = os.environ.get("HAUNT_NAMESPACE")
    if env:
        return safe_name(env)
    if cwd is None:
        proj = os.environ.get("CURSOR_PROJECT_DIR") or os.environ.get("CLAUDE_PROJECT_DIR")
        root = Path(proj).expanduser().resolve() if proj else Path.cwd().resolve()
    else:
        root = cwd.resolve()
    remote_url, repo_root = _git_repo_context(root)
    identity = repository_identity(remote_url)
    registered = _registered_namespace_for_repo(
        remote_identity=identity,
        repo_root=repo_root,
    )
    if registered:
        return registered
    if identity:
        return namespace_for_repo_identity(identity)
    if repo_root:
        return safe_name(repo_root.name)
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
