"""Filesystem layout: ~/.haunt (or $HAUNT_HOME). Absolute paths only."""

from __future__ import annotations

import os
import re
import sqlite3
import stat
import subprocess
import threading
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

SAFE_NS = re.compile(r"[^a-zA-Z0-9._-]+")

DIR_MODE = 0o700
FILE_MODE = 0o600

_RegistryFingerprint = tuple[
    tuple[int, int, int, int, int] | None,
    tuple[int, int, int, int, int] | None,
]
_NAMESPACE_ALIAS_CACHE: dict[
    tuple[str, str], tuple[_RegistryFingerprint, str, Path, int, int]
] = {}
_NAMESPACE_ALIAS_CACHE_LOCK = threading.Lock()


class NamespacePathError(ValueError):
    """Raised when namespace storage could redirect outside its physical identity."""


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
    registered = _registered_db_path(name)
    if registered is not None:
        return registered
    return namespaces_dir() / f"{safe_name(name)}.db"


def safe_name(name: str) -> str:
    cleaned = SAFE_NS.sub("-", name.strip()).strip("-.")
    if not cleaned:
        return "default"
    return cleaned[:80]


def normalize_namespace_label(name: str) -> str:
    """Normalize a namespace label for unique registry lookup.

    Labels retain their display spelling, while identity comparisons are
    case-insensitive and use the same filesystem-safe/truncation rules as
    legacy namespace names.
    """
    return safe_name(name).casefold()


def validate_namespace_root(*, create: bool = False) -> Path:
    """Return the real namespace root, rejecting symlinks and non-directories."""
    root = namespaces_dir()
    if create:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise NamespacePathError(
                f"cannot create namespace database root {root}: {exc}"
            ) from exc
    try:
        info = root.lstat()
    except OSError as exc:
        raise NamespacePathError(
            f"namespace database root is missing or unreadable: {root}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise NamespacePathError(
            f"namespace database root must be a real non-symlink directory: {root}"
        )
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NamespacePathError(
            f"cannot resolve namespace database root {root}"
        ) from exc
    if create and not _is_user_home(root):
        root.chmod(DIR_MODE)
    return resolved


def validate_namespace_db_paths(
    paths: list[str],
    *,
    expected: dict[str, tuple[int | None, int | None]] | None = None,
) -> dict[str, tuple[int, int]]:
    """Validate and identify every mapped DB under the real namespace root."""
    root = namespaces_dir()
    root_resolved = validate_namespace_root()
    root_lexical = Path(os.path.abspath(os.path.normpath(str(root))))
    seen: dict[tuple[int, int], str] = {}
    identities: dict[str, tuple[int, int]] = {}
    for raw in paths:
        path = Path(raw)
        lexical = Path(os.path.abspath(os.path.normpath(str(path))))
        if not path.is_absolute() or path != lexical or lexical.parent != root_lexical:
            raise NamespacePathError(
                f"namespace database path must be a canonical direct child of {root}: {raw}"
            )
        try:
            info = path.lstat()
        except OSError as exc:
            raise NamespacePathError(
                f"namespace database is missing or unreadable: {raw}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise NamespacePathError(
                f"namespace database must be a regular non-symlink file: {raw}"
            )
        if int(info.st_nlink) != 1:
            raise NamespacePathError(
                f"namespace database must have exactly one filesystem link: {raw}"
            )
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise NamespacePathError(
                f"cannot resolve namespace database path: {raw}"
            ) from exc
        if resolved.parent != root_resolved:
            raise NamespacePathError(
                f"namespace database resolves outside {root}: {raw}"
            )
        identity = int(info.st_dev), int(info.st_ino)
        expected_identity = (expected or {}).get(raw)
        if raw in (expected or {}) and (
            expected_identity is None
            or any(value is None for value in expected_identity)
        ):
            raise NamespacePathError(
                f"namespace database has no recorded physical identity: {raw}"
            )
        if expected_identity:
            if identity != (int(expected_identity[0]), int(expected_identity[1])):
                raise NamespacePathError(
                    f"namespace database physical identity changed: {raw}"
                )
        previous = seen.get(identity)
        if previous is not None and previous != raw:
            raise NamespacePathError(
                "distinct namespace database paths identify the same physical file: "
                f"{previous!r} and {raw!r}"
            )
        seen[identity] = raw
        identities[raw] = identity
    return identities


def _alias_cache_key(name: str) -> tuple[str, str]:
    return str(registry_path()), normalize_namespace_label(name)


def _registry_fingerprint() -> _RegistryFingerprint:
    base = registry_path()

    def _stat(path: Path) -> tuple[int, int, int, int, int] | None:
        try:
            info = path.stat()
        except OSError:
            return None
        return (
            int(info.st_dev),
            int(info.st_ino),
            int(info.st_size),
            int(info.st_mtime_ns),
            int(info.st_ctime_ns),
        )

    return _stat(base), _stat(Path(str(base) + "-wal"))


def _remember_registered_alias(
    name: str,
    canonical: str,
    db_path: Path,
    db_device: int,
    db_inode: int,
    *,
    fingerprint: _RegistryFingerprint,
) -> bool:
    """Publish a query result only while its registry snapshot is unchanged."""
    if fingerprint != _registry_fingerprint():
        return False
    with _NAMESPACE_ALIAS_CACHE_LOCK:
        _NAMESPACE_ALIAS_CACHE[_alias_cache_key(name)] = (
            fingerprint,
            canonical,
            db_path,
            db_device,
            db_inode,
        )
    return True


def _forget_registered_alias(name: str) -> None:
    with _NAMESPACE_ALIAS_CACHE_LOCK:
        _NAMESPACE_ALIAS_CACHE.pop(_alias_cache_key(name), None)


def validate_registry_db_sources(
    conn: sqlite3.Connection, tables: set[str]
) -> dict[str, tuple[int, int]]:
    mapped_paths: list[str] = []
    expected_paths: dict[str, tuple[int | None, int | None]] = {}
    if "namespace_identities" in tables:
        columns = {
            str(info["name"])
            for info in conn.execute(
                "PRAGMA table_info(namespace_identities)"
            ).fetchall()
        }
        identities = conn.execute("SELECT * FROM namespace_identities").fetchall()
        mapped_paths.extend(str(identity["db_path"]) for identity in identities)
        if {"db_device", "db_inode"} <= columns:
            expected_paths.update(
                {
                    str(identity["db_path"]): (
                        identity["db_device"],
                        identity["db_inode"],
                    )
                    for identity in identities
                }
            )
    if "namespaces" in tables:
        mapped_paths.extend(
            str(legacy["db_path"])
            for legacy in conn.execute("SELECT db_path FROM namespaces").fetchall()
        )
    return validate_namespace_db_paths(mapped_paths, expected=expected_paths)


def _registered_alias(name: str) -> tuple[str, Path] | None:
    """Read an alias mapping without initializing or mutating the registry."""
    with _NAMESPACE_ALIAS_CACHE_LOCK:
        cached = _NAMESPACE_ALIAS_CACHE.get(_alias_cache_key(name))
    # The fingerprint must bracket the query. Otherwise a retirement and
    # reassignment between SELECT and cache publication can pin stale identity.
    for _attempt in range(3):
        before = _registry_fingerprint()
        path = registry_path()
        if before[0] is None or not path.is_file():
            _forget_registered_alias(name)
            return None
        conn: sqlite3.Connection | None = None
        result: tuple[str, Path, int, int] | None = None
        try:
            # A quiescent registry can be opened immutable without creating
            # WAL/SHM sidecars. If a writer appears, the bracket fingerprint
            # changes and the retry uses the WAL-aware read-only mode.
            immutable = "&immutable=1" if before[1] is None else ""
            conn = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode=ro{immutable}", uri=True
            )
            conn.row_factory = sqlite3.Row
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            physical = validate_registry_db_sources(conn, tables)
            if cached is not None and cached[0] == before:
                cached_path = str(cached[2])
                cached_physical = physical.get(cached_path)
                if cached_physical == (cached[3], cached[4]):
                    result = (
                        cached[1],
                        cached[2],
                        cached[3],
                        cached[4],
                    )
            if result is None and {"namespace_aliases", "namespace_identities"} <= tables:
                row = conn.execute(
                    """
                    SELECT i.canonical_label, i.db_path
                    FROM namespace_aliases a
                    JOIN namespace_identities i ON i.namespace_id=a.namespace_id
                    WHERE a.normalized_label=?
                    """,
                    (normalize_namespace_label(name),),
                ).fetchone()
                if row:
                    db_path = str(row["db_path"])
                    result = (
                        str(row["canonical_label"]),
                        Path(db_path),
                        physical[db_path][0],
                        physical[db_path][1],
                    )
            elif result is None:
                row = conn.execute(
                    "SELECT name, db_path FROM namespaces WHERE name=?",
                    (safe_name(name),),
                ).fetchone()
                if row:
                    db_path = str(row["db_path"])
                    result = (
                        str(row["name"]),
                        Path(db_path),
                        physical[db_path][0],
                        physical[db_path][1],
                    )
        except sqlite3.Error:
            _forget_registered_alias(name)
            return None
        finally:
            if conn is not None:
                conn.close()
        if before != _registry_fingerprint():
            continue
        if result is None:
            _forget_registered_alias(name)
            return None
        if _remember_registered_alias(name, *result, fingerprint=before):
            return result[0], result[1]
    # Repeated concurrent migrations: fail closed instead of caching or
    # returning a value from a registry snapshot already known to be stale.
    _forget_registered_alias(name)
    return None


def _registered_db_path(name: str) -> Path | None:
    registered = _registered_alias(name)
    return registered[1] if registered else None


def repository_identity(remote_url: str | None) -> str | None:
    """Return host/owner/repo identity for common git remote URL forms."""
    raw = (remote_url or "").strip().rstrip("/")
    if not raw:
        return None
    host = ""
    path = ""
    if "://" in raw:
        try:
            parsed = urlsplit(raw)
            hostname = (parsed.hostname or "").lower()
            port = parsed.port
        except ValueError:
            # urllib rejects malformed bracketed IPv6 hosts and invalid ports.
            # A malformed remote must never collapse into a valid repository
            # identity or escape into namespace inference as an exception.
            return None
        if ":" in hostname:
            hostname = f"[{hostname}]"
        host = hostname
        default_ports = {"http": 80, "https": 443, "ssh": 22, "git": 9418}
        if port is not None and port != default_ports.get(parsed.scheme.lower()):
            host = f"{host}:{port}"
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
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        validate_registry_db_sources(conn, tables)
        if {"repository_bindings", "namespace_identities"} <= tables:
            if remote_identity:
                row = conn.execute(
                    """
                    SELECT i.canonical_label
                    FROM repository_bindings b
                    JOIN namespace_identities i ON i.namespace_id=b.namespace_id
                    WHERE b.repository_identity=?
                    """,
                    (remote_identity,),
                ).fetchone()
                if row:
                    return str(row["canonical_label"])
            if repo_root:
                row = conn.execute(
                    """
                    SELECT i.canonical_label
                    FROM repository_bindings b
                    JOIN namespace_identities i ON i.namespace_id=b.namespace_id
                    WHERE b.repo_path=?
                    """,
                    (str(repo_root.resolve()),),
                ).fetchone()
                if row:
                    return str(row["canonical_label"])
        rows = conn.execute("SELECT name, repo_path FROM namespaces").fetchall()
    except sqlite3.Error:
        return None
    finally:
        if conn is not None:
            conn.close()
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
        registered = _registered_alias(env)
        return registered[0] if registered else safe_name(env)
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
        registered = _registered_alias(name)
        return registered[0] if registered else safe_name(name)
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
            if (extra.stat().st_mode & 0o777) != FILE_MODE:
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
    mkdir_private(home)
    validate_namespace_root(create=True)
    for p in (bin_dir(), models_dir()):
        mkdir_private(p)
    return home
