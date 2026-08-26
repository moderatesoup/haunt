"""SQLite store: registry + per-namespace DBs. WAL. Verbatim writes only."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import fcntl
import hashlib
import json
import sqlite3
import stat
import struct
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import get_ident
from typing import Any, Iterator

import sqlite_vec

from haunt.embed import embed_one
from haunt.embed import embed_texts
from haunt.embed import state as embed_state
from haunt.paths import (
    _forget_registered_alias,
    _git_repo_context,
    ensure_layout,
    haunt_home,
    materialize_sqlite_shadow,
    mkdir_private,
    NamespacePathError,
    namespaces_dir,
    normalize_namespace_label,
    registry_path,
    required_o_nofollow,
    readonly_sqlite_mode,
    repository_identity,
    resolve_namespace,
    safe_name,
    SQLITE_OPEN_LOCK,
    SQLitePrimaryGuard,
    SQLiteSidecarGuard,
    SQLiteStorageSnapshot,
    sqlite_storage_snapshot,
    temporary_sqlite_shadow,
    tighten_db_files,
    validate_namespace_db_paths,
    validate_namespace_root,
    validate_registry_db_sources,
    validate_sqlite_sidecars,
)
from haunt.provenance import (
    encode_json_safe_sqlite_key,
    json_safe_sqlite,
    provenance_json,
    public_provenance,
    validate_provenance,
)
from haunt.util import (
    clamp_limit,
    clock_sql_column,
    dumps,
    iso_or_now,
    loads,
    new_id,
    normalize_clock,
    now_iso,
    parse_iso,
    utc_iso,
)

ROLES = ("user", "assistant", "tool", "system")
TIERS = ("episodic", "semantic", "procedural", "coordinate")

# 1: one-time normalize of offset/naive clocks to UTC microseconds.
# 2: graph evidence tables + hook idempotency key.
# 3: durable queue for hook-deferred embeddings.
# 4: append-only correction lineage plus privacy-erasure tombstones.
# 5: privacy-safe rekeying for erased target and correction sessions.
# 6: schema-enforced normal-vs-privacy-scrubbed correction invariants.
# 7: database-enforced append-only corrections outside authorized purge.
# 8: validated, versioned source provenance on events.
SCHEMA_VERSION = 8
SCHEMA_VERSION_KEY = "schema_version"
REGISTRY_SCHEMA_VERSION = 5
REGISTRY_SCHEMA_VERSION_KEY = "schema_version"
_NAMESPACE_DB_HANDLE_LOCK = SQLITE_OPEN_LOCK
_NAMESPACE_MIGRATION_LOCK = threading.RLock()
_SQLITE_CONFIGURATION_LOCK = threading.RLock()
_SQLITE_CONFIGURATION_STATE = threading.local()


@contextmanager
def _namespace_migration_lock() -> Iterator[None]:
    """Serialize migration plan verification/apply across local processes."""
    with _NAMESPACE_MIGRATION_LOCK:
        root = haunt_home()
        mkdir_private(root)
        lock_path = root / ".namespace-migration.lock"
        nofollow = required_o_nofollow()
        flags = os.O_RDWR | nofollow | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            fd = os.open(lock_path, flags)
        try:
            held = os.fstat(fd)
            current = lock_path.lstat()
            if (
                not stat.S_ISREG(held.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or int(held.st_nlink) != 1
                or int(current.st_nlink) != 1
                or stat.S_IMODE(held.st_mode) != 0o600
                or (int(held.st_dev), int(held.st_ino))
                != (int(current.st_dev), int(current.st_ino))
            ):
                raise NamespaceMigrationError("namespace migration lock is unsafe")
            fcntl.flock(fd, fcntl.LOCK_EX)
            current = lock_path.lstat()
            if (int(current.st_dev), int(current.st_ino)) != (
                int(held.st_dev), int(held.st_ino)
            ):
                raise NamespaceMigrationError(
                    "namespace migration lock changed while acquiring it"
                )
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


@contextmanager
def _sqlite_configuration_lock() -> Iterator[None]:
    """Serialize every Haunt RW sidecar claim/open/first-PRAGMA sequence.

    The advisory lock closes accidental cross-process races between cooperating
    Haunt writers. It is deliberately not presented as kernel isolation from an
    arbitrary process running as the same account, which can ignore ``flock``.
    """
    with _SQLITE_CONFIGURATION_LOCK:
        depth = int(getattr(_SQLITE_CONFIGURATION_STATE, "depth", 0))
        if depth:
            _SQLITE_CONFIGURATION_STATE.depth = depth + 1
            try:
                yield
            finally:
                _SQLITE_CONFIGURATION_STATE.depth = depth
            return
        root = haunt_home()
        mkdir_private(root)
        lock_path = root / ".sqlite-configuration.lock"
        nofollow = required_o_nofollow()
        flags = os.O_RDWR | nofollow | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            fd = os.open(lock_path, flags)
        try:
            held = os.fstat(fd)
            current = lock_path.lstat()
            identity = int(held.st_dev), int(held.st_ino)
            if (
                not stat.S_ISREG(held.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or int(held.st_nlink) != 1
                or int(current.st_nlink) != 1
                or stat.S_IMODE(held.st_mode) != 0o600
                or identity != (int(current.st_dev), int(current.st_ino))
            ):
                raise NamespacePathError("SQLite configuration lock is unsafe")
            fcntl.flock(fd, fcntl.LOCK_EX)
            current = lock_path.lstat()
            if (
                stat.S_ISLNK(current.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or (int(current.st_dev), int(current.st_ino)) != identity
            ):
                raise NamespacePathError(
                    "SQLite configuration lock changed while acquiring it"
                )
            _SQLITE_CONFIGURATION_STATE.depth = 1
            try:
                yield
            finally:
                _SQLITE_CONFIGURATION_STATE.depth = 0
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _sqlite_configuration_lock_held() -> bool:
    return bool(getattr(_SQLITE_CONFIGURATION_STATE, "depth", 0))

_CLOCK_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sessions", ("started_at", "ended_at")),
    ("events", ("ts", "event_time")),
    ("memories", ("valid_from", "valid_to", "created_at")),
    ("entities", ("first_seen", "last_seen")),
    ("relations", ("valid_from", "valid_to")),
)

CORRECTION_KEY_MAX = 512
TOMBSTONE_SCHEMA_VERSION = 1
PURGE_SAFE_ORIGIN = "privacy-sanitized"
PURGE_SAFE_SESSION_SOURCE = "privacy-sanitized"
PURGE_SAFE_PROVENANCE = provenance_json(
    {
        "schema_version": 1,
        "kind": "native",
        "channel": "privacy_purge",
        "origin": PURGE_SAFE_ORIGIN,
    }
)


class UnknownNamespaceError(ValueError):
    """Raised when a read/mutation targets a namespace that does not exist."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"unknown namespace: {name}")


class NamespaceCollisionError(ValueError):
    """Raised when a label, repository, or target file belongs elsewhere."""


class AliasRetirementError(ValueError):
    """Raised when a live registry-owned reference blocks alias retirement."""


class NamespaceMigrationError(ValueError):
    """Raised when a migration plan, backup, or reversal is unsafe."""


def _validate_unmapped_namespace_target(
    target: Path, *, mapped_db_path: Path | None = None
) -> None:
    """Refuse a new label path that exists, escapes the DB root, or aliases a file."""
    root = namespaces_dir()
    try:
        root_resolved = validate_namespace_root()
        target_resolved = target.resolve(strict=False)
    except (NamespacePathError, OSError, RuntimeError) as exc:
        raise NamespaceCollisionError(
            f"cannot establish a safe database path for {target}"
        ) from exc
    if not target_resolved.is_relative_to(root_resolved):
        raise NamespaceCollisionError(
            f"namespace database target escapes {root}: {target}"
        )
    if mapped_db_path is not None and target == mapped_db_path:
        return
    try:
        validate_sqlite_sidecars(target, require_absent=True)
    except NamespacePathError as exc:
        raise NamespaceCollisionError(str(exc)) from exc
    try:
        target.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise NamespaceCollisionError(
            f"cannot inspect namespace database target {target}"
        ) from exc
    raise NamespaceCollisionError(
        f"target label has an unmapped filesystem entry at {target}"
    )


class _SidecarGuardedConnection(sqlite3.Connection):
    """SQLite connection that retains sidecar claims until SQLite closes."""

    _sidecar_guard: SQLiteSidecarGuard | None = None
    _primary_guard: SQLitePrimaryGuard | None = None
    _clean_unused_sidecar_claims = True
    _clean_primary_claim = True
    _zero_write_snapshot: tuple[Path, SQLiteStorageSnapshot] | None = None
    _temporary_read_dir: Any = None

    def set_sidecar_guard(
        self, guard: SQLiteSidecarGuard, *, clean_unused_claims: bool
    ) -> None:
        self._sidecar_guard = guard
        self._clean_unused_sidecar_claims = clean_unused_claims

    def set_primary_guard(self, guard: SQLitePrimaryGuard) -> None:
        self._primary_guard = guard

    def set_zero_write_snapshot(
        self, path: Path, snapshot: SQLiteStorageSnapshot
    ) -> None:
        self._zero_write_snapshot = (path, snapshot)

    def set_temporary_read_dir(self, temporary: Any) -> None:
        self._temporary_read_dir = temporary

    def verify_storage_guards(self) -> None:
        if self._primary_guard is None or self._sidecar_guard is None:
            raise NamespacePathError("SQLite connection has no held storage guards")
        self._primary_guard.verify()
        self._sidecar_guard.verify()

    def copy_primary_to_fd(self, destination_fd: int) -> None:
        """Copy this immutable, materialized main file to an already-held FD."""
        self.verify_storage_guards()
        assert self._primary_guard is not None
        nofollow = required_o_nofollow()
        source_fd = os.open(
            self._primary_guard.path,
            os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            source_info = os.fstat(source_fd)
            if (
                not stat.S_ISREG(source_info.st_mode)
                or int(source_info.st_nlink) != 1
                or (int(source_info.st_dev), int(source_info.st_ino))
                != self._primary_guard.identity
            ):
                raise NamespacePathError(
                    "SQLite source changed while creating registry backup"
                )
            os.lseek(destination_fd, 0, os.SEEK_SET)
            os.ftruncate(destination_fd, 0)
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fsync(destination_fd)
            source_after = os.fstat(source_fd)
            if (
                (int(source_after.st_dev), int(source_after.st_ino))
                != self._primary_guard.identity
                or int(source_after.st_nlink) != 1
            ):
                raise NamespacePathError(
                    "SQLite source changed while creating registry backup"
                )
            self.verify_storage_guards()
        finally:
            os.close(source_fd)

    def preserve_sidecar_claims(self) -> None:
        self._clean_unused_sidecar_claims = False
        self._clean_primary_claim = False

    def close(self) -> None:
        guard = self._sidecar_guard
        primary = self._primary_guard
        zero_write = self._zero_write_snapshot
        temporary = self._temporary_read_dir
        self._sidecar_guard = None
        self._primary_guard = None
        self._zero_write_snapshot = None
        self._temporary_read_dir = None
        with _NAMESPACE_DB_HANDLE_LOCK:
            close_error: Exception | None = None
            try:
                super().close()
                if zero_write is not None:
                    path, before = zero_write
                    if sqlite_storage_snapshot(path) != before:
                        close_error = NamespacePathError(
                            f"SQLite zero-write read observed storage drift: {path}"
                        )
            finally:
                if guard is not None:
                    guard.close(
                        clean_unused_claims=self._clean_unused_sidecar_claims
                    )
                if primary is not None:
                    primary.close(clean_claim=self._clean_primary_claim)
                if temporary is not None:
                    temporary.cleanup()
            if close_error is not None:
                raise close_error


def _raw_connect(path: Path, *, create: bool = True) -> sqlite3.Connection:
    if not create and not path.exists():
        raise FileNotFoundError(path)
    if create:
        mkdir_private(path.parent)
    # Serialize opens with namespace descriptor verification. SQLite's unix
    # VFS can retain/reuse descriptors while sibling connections hold locks.
    with _NAMESPACE_DB_HANDLE_LOCK:
        conn = sqlite3.connect(
            str(path),
            check_same_thread=False,
            factory=_SidecarGuardedConnection,
        )
    conn.row_factory = sqlite3.Row
    return conn


def _sqlite_sidecar_open_hook(_path: Path) -> None:
    """Test hook after sidecar claim/validation and before SQLite open."""


def _sqlite_sidecar_pragma_hook(_path: Path) -> None:
    """Test hook after SQLite open and before the first configuring PRAGMA."""


def _sqlite_sidecar_verified_hook(_path: Path) -> None:
    """Test hook after validation at the exact first-PRAGMA boundary."""


def _open_readonly_connection(
    path: Path,
    *,
    immutable: bool = False,
    claim_missing: bool = True,
    zero_write_snapshot: SQLiteStorageSnapshot | None = None,
) -> sqlite3.Connection:
    with _NAMESPACE_DB_HANDLE_LOCK:
        primary = SQLitePrimaryGuard.acquire(path, create_missing=False)
        try:
            sidecars = SQLiteSidecarGuard.acquire(
                path, claim_missing=claim_missing
            )
        except Exception:
            primary.close()
            raise
        conn: sqlite3.Connection | None = None
        try:
            _sqlite_sidecar_open_hook(path)
            primary.verify()
            sidecars.verify()
            before = _fd_snapshot()
            immutable_query = "&immutable=1" if immutable else ""
            conn = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode=ro{immutable_query}",
                uri=True,
                factory=_SidecarGuardedConnection,
            )
            assert isinstance(conn, _SidecarGuardedConnection)
            conn.set_primary_guard(primary)
            conn.set_sidecar_guard(sidecars, clean_unused_claims=True)
            if zero_write_snapshot is not None:
                conn.set_zero_write_snapshot(path, zero_write_snapshot)
            conn.row_factory = sqlite3.Row
            _verify_new_sqlite_fd(
                before, primary.identity, allow_verified_vfs_reuse=True,
                ignored_identities=_sidecar_identities(sidecars),
            )
            primary.verify()
            sidecars.verify()
            if claim_missing:
                conn.execute("PRAGMA schema_version").fetchone()
                validate_sqlite_sidecars(path)
            return conn
        except Exception:
            if conn is not None:
                conn.close()
            else:
                sidecars.close(clean_unused_claims=True)
                primary.close()
            raise


def _configure_connection(
    conn: sqlite3.Connection,
    path: Path,
    *,
    tighten: bool = True,
    sidecars: SQLiteSidecarGuard | None = None,
) -> sqlite3.Connection:
    if not _sqlite_configuration_lock_held():
        raise NamespacePathError(
            "SQLite write configuration requires the serialized sidecar lock"
        )
    _sqlite_sidecar_pragma_hook(path)
    if sidecars is not None:
        sidecars.verify()
    _sqlite_sidecar_verified_hook(path)
    if sidecars is not None:
        # The hook represents the exact scheduling boundary after the previous
        # validation. Recheck while the cross-process writer lock is still held
        # before SQLite receives its first write-mode pragma.
        sidecars.verify()
    conn.execute("PRAGMA journal_mode=WAL")
    validate_sqlite_sidecars(path)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    if tighten:
        tighten_db_files(path)
    from haunt.embed import fts_only

    if not fts_only():
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as exc:
            conn.close()
            raise RuntimeError(
                f"sqlite-vec failed to load: {exc}\n"
                "Set HAUNT_FTS_ONLY=1 to run without vector search."
            ) from exc
    return conn


def _connect(path: Path, *, create: bool = True) -> sqlite3.Connection:
    with _sqlite_configuration_lock():
        return _connect_with_configuration_lock(path, create=create)


def _connect_with_configuration_lock(
    path: Path, *, create: bool = True
) -> sqlite3.Connection:
    with _NAMESPACE_DB_HANDLE_LOCK:
        if create:
            mkdir_private(path.parent)
        primary = SQLitePrimaryGuard.acquire(path, create_missing=create)
        try:
            sidecars = SQLiteSidecarGuard.acquire(path, claim_missing=True)
        except Exception:
            primary.close(clean_claim=True)
            raise
        conn: sqlite3.Connection | None = None
        try:
            _sqlite_sidecar_open_hook(path)
            primary.verify()
            sidecars.verify()
            before = _fd_snapshot()
            conn = _raw_connect(path, create=False)
            assert isinstance(conn, _SidecarGuardedConnection)
            conn.set_primary_guard(primary)
            conn.set_sidecar_guard(sidecars, clean_unused_claims=True)
            _verify_new_sqlite_fd(
                before, primary.identity, allow_verified_vfs_reuse=True,
                ignored_identities=_sidecar_identities(sidecars),
            )
            primary.verify()
            sidecars.verify()
            result = _configure_connection(conn, path, sidecars=sidecars)
            conn.preserve_sidecar_claims()
            return result
        except Exception:
            if conn is not None:
                conn.close()
            else:
                sidecars.close(clean_unused_claims=True)
                primary.close(clean_claim=True)
            raise


def _fd_snapshot() -> dict[int, tuple[int, int]]:
    """Return open descriptor identities for safe SQLite-open verification."""
    for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
        try:
            names = list(directory.iterdir())
        except OSError:
            continue
        out: dict[int, tuple[int, int]] = {}
        for entry in names:
            try:
                fd = int(entry.name)
                info = os.fstat(fd)
            except (OSError, ValueError):
                continue
            out[fd] = int(info.st_dev), int(info.st_ino)
        return out
    raise NamespacePathError(
        "this platform cannot verify the physical file opened by SQLite"
    )


def _verify_new_sqlite_fd(
    before: dict[int, tuple[int, int]],
    expected: tuple[int, int],
    *,
    allow_verified_vfs_reuse: bool = False,
    ignored_identities: set[tuple[int, int]] | None = None,
) -> None:
    after = _fd_snapshot()
    before_count = sum(identity == expected for identity in before.values())
    after_count = sum(identity == expected for identity in after.values())
    if after_count > before_count:
        return
    if (
        allow_verified_vfs_reuse
        # One descriptor is the caller's held guard. A second descriptor is a
        # previously verified SQLite unix-VFS handle eligible for reuse.
        and before_count >= 2
        and after_count >= 2
    ):
        # SQLite's POSIX VFS deliberately retains an inode descriptor when a
        # connection closes while sibling connections still hold locks, then
        # reuses it for the next connection. In that case no descriptor delta
        # exists, but the retained descriptor was already verified as the
        # expected physical file. Unrelated concurrent SQLite descriptor churn
        # is irrelevant because only the held physical identity is counted.
        # The caller still brackets this with pathname and stored-inode checks.
        return
    raise NamespacePathError(
        "SQLite did not open the claimed namespace database identity"
    )


def _sidecar_identities(guard: SQLiteSidecarGuard) -> set[tuple[int, int]]:
    return {
        entry.identity for entry in guard.entries if entry.identity is not None
    }


def _mapped_namespace_open_hook(_path: Path) -> None:
    """Test hook immediately before an existing mapped DB is opened."""


def _fresh_namespace_claim_hook(_path: Path) -> None:
    """Test hook after atomic target claim and before registry publication."""


def _validate_all_registered_namespace_dbs_read_only() -> None:
    """Validate every legacy/current mapping without initializing the registry."""
    registry = registry_path()
    if not registry.is_file():
        raise NamespacePathError(f"namespace registry is missing: {registry}")
    conn = _open_readonly_connection(registry)
    try:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        validate_registry_db_sources(conn, tables)
    finally:
        conn.close()


def _open_mapped_namespace_db(
    path: Path, *, expected: tuple[int | None, int | None]
) -> sqlite3.Connection:
    with _sqlite_configuration_lock():
        return _open_mapped_namespace_db_with_configuration_lock(
            path, expected=expected
        )


def _open_mapped_namespace_db_with_configuration_lock(
    path: Path, *, expected: tuple[int | None, int | None]
) -> sqlite3.Connection:
    _validate_all_registered_namespace_dbs_read_only()
    expected_map = {str(path): expected}
    actual = validate_namespace_db_paths([str(path)], expected=expected_map)[str(path)]
    sidecars: SQLiteSidecarGuard | None = None
    primary: SQLitePrimaryGuard | None = None
    conn: sqlite3.Connection | None = None
    with _NAMESPACE_DB_HANDLE_LOCK:
        try:
            primary = SQLitePrimaryGuard.acquire(path, create_missing=False)
            if primary.identity != actual:
                raise NamespacePathError(
                    f"namespace database physical identity changed: {path}"
                )
            sidecars = SQLiteSidecarGuard.acquire(path, claim_missing=True)
            _mapped_namespace_open_hook(path)
            _sqlite_sidecar_open_hook(path)
            primary.verify()
            sidecars.verify()
            before = _fd_snapshot()
            conn = _raw_connect(path, create=False)
            assert isinstance(conn, _SidecarGuardedConnection)
            conn.set_primary_guard(primary)
            conn.set_sidecar_guard(sidecars, clean_unused_claims=True)
            _verify_new_sqlite_fd(
                before, actual, allow_verified_vfs_reuse=True,
                ignored_identities=_sidecar_identities(sidecars),
            )
            sidecars.verify()
            primary.verify()
            validate_namespace_db_paths(
                [str(path)], expected={str(path): actual}
            )
            _validate_all_registered_namespace_dbs_read_only()
            result = _configure_connection(
                conn, path, tighten=False, sidecars=sidecars
            )
            conn.preserve_sidecar_claims()
            return result
        except Exception:
            if conn is not None:
                conn.close()
            else:
                if sidecars is not None:
                    sidecars.close(clean_unused_claims=True)
                if primary is not None:
                    primary.close()
            raise


def _preflight_registry_storage_read_only() -> None:
    """Reject unsafe existing mappings before migration opens the registry RW."""
    registry = registry_path()
    if not registry.exists():
        return
    conn = _open_readonly_connection(registry)
    try:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        validate_registry_db_sources(conn, tables)
    finally:
        conn.close()


def _vec_loaded(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT vec_version()")
        return True
    except sqlite3.Error:
        return False


def init_registry() -> None:
    ensure_layout()
    _preflight_registry_storage_read_only()
    conn = _connect(registry_path())
    try:
        existing_tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        preflight_paths: list[str] = []
        preflight_expected: dict[str, tuple[int | None, int | None]] = {}
        preflight_version = None
        if "registry_meta" in existing_tables:
            preflight_version_row = conn.execute(
                "SELECT value FROM registry_meta WHERE key=?",
                (REGISTRY_SCHEMA_VERSION_KEY,),
            ).fetchone()
            preflight_version = (
                str(preflight_version_row["value"])
                if preflight_version_row
                else None
            )
        if "namespace_identities" in existing_tables:
            columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(namespace_identities)"
                ).fetchall()
            }
            identities = conn.execute(
                "SELECT * FROM namespace_identities"
            ).fetchall()
            preflight_paths.extend(str(row["db_path"]) for row in identities)
            if (
                {"db_device", "db_inode"} <= columns
                and preflight_version == str(REGISTRY_SCHEMA_VERSION)
            ):
                preflight_expected.update(
                    {
                        str(row["db_path"]): (row["db_device"], row["db_inode"])
                        for row in identities
                    }
                )
        if "namespaces" in existing_tables:
            preflight_paths.extend(
                str(row["db_path"])
                for row in conn.execute("SELECT db_path FROM namespaces").fetchall()
            )
        validate_namespace_db_paths(
            preflight_paths, expected=preflight_expected
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS namespaces (
                name TEXT PRIMARY KEY,
                repo_path TEXT,
                db_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS registry_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS namespace_identities (
                namespace_id TEXT PRIMARY KEY,
                canonical_label TEXT NOT NULL,
                canonical_label_norm TEXT NOT NULL UNIQUE,
                db_path TEXT NOT NULL UNIQUE,
                db_device INTEGER NOT NULL,
                db_inode INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS namespace_aliases (
                normalized_label TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                namespace_id TEXT NOT NULL,
                is_canonical INTEGER NOT NULL DEFAULT 0 CHECK(is_canonical IN (0,1)),
                source_alias_norm TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(namespace_id) REFERENCES namespace_identities(namespace_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_namespace_one_canonical
                ON namespace_aliases(namespace_id) WHERE is_canonical=1;
            CREATE TABLE IF NOT EXISTS repository_bindings (
                binding_id TEXT PRIMARY KEY,
                namespace_id TEXT NOT NULL,
                repository_identity TEXT,
                repo_path TEXT,
                label_norm TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK(repository_identity IS NOT NULL OR repo_path IS NOT NULL),
                FOREIGN KEY(namespace_id) REFERENCES namespace_identities(namespace_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_repository_remote_unique
                ON repository_bindings(repository_identity)
                WHERE repository_identity IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_repository_path_unique
                ON repository_bindings(repo_path) WHERE repo_path IS NOT NULL;
            CREATE TABLE IF NOT EXISTS namespace_migrations (
                migration_id TEXT PRIMARY KEY,
                namespace_id TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('alias','rename')),
                old_label TEXT NOT NULL,
                old_label_norm TEXT NOT NULL,
                new_label TEXT NOT NULL,
                new_label_norm TEXT NOT NULL,
                repository_identity TEXT,
                repository_key TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                plan_digest TEXT,
                before_state TEXT,
                after_state TEXT,
                backup_path TEXT,
                backup_sha256 TEXT,
                backup_integrity TEXT,
                undone_at TEXT,
                undo_plan_digest TEXT,
                undo_backup_path TEXT,
                undo_backup_sha256 TEXT,
                undo_backup_integrity TEXT,
                UNIQUE(namespace_id, action, old_label_norm, new_label_norm, repository_key),
                FOREIGN KEY(namespace_id) REFERENCES namespace_identities(namespace_id)
            );
            """
        )
        conn.commit()
        version_row = conn.execute(
            "SELECT value FROM registry_meta WHERE key=?",
            (REGISTRY_SCHEMA_VERSION_KEY,),
        ).fetchone()
        legacy_rows = conn.execute(
            "SELECT name,repo_path,db_path,created_at,updated_at FROM namespaces"
        ).fetchall()
        identity_rows = conn.execute(
            "SELECT * FROM namespace_identities"
        ).fetchall()
        identity_paths = [str(row["db_path"]) for row in identity_rows]
        physical_identities = validate_namespace_db_paths(
            [*identity_paths, *(str(row["db_path"]) for row in legacy_rows)]
        )
        identity_columns = {
            str(row["name"])
            for row in conn.execute(
                "PRAGMA table_info(namespace_identities)"
            ).fetchall()
        }
        has_physical_columns = {"db_device", "db_inode"} <= identity_columns
        if (
            has_physical_columns
            and version_row
            and str(version_row["value"]) == str(REGISTRY_SCHEMA_VERSION)
        ):
            validate_namespace_db_paths(
                identity_paths,
                expected={
                    str(row["db_path"]): (row["db_device"], row["db_inode"])
                    for row in identity_rows
                },
            )
        fully_projected = all(
            conn.execute(
                """SELECT 1
                   FROM namespace_aliases a
                   JOIN namespace_identities i ON i.namespace_id=a.namespace_id
                   WHERE a.normalized_label=? AND i.db_path=?""",
                (normalize_namespace_label(str(row["name"])), str(row["db_path"])),
            ).fetchone()
            is not None
            for row in legacy_rows
        )
        migration_columns = {
            str(row["name"])
            for row in conn.execute(
                "PRAGMA table_info(namespace_migrations)"
            ).fetchall()
        }
        reversible_columns = {
            "plan_digest", "before_state", "after_state", "backup_path",
            "backup_sha256", "backup_integrity", "undone_at",
            "undo_plan_digest", "undo_backup_path", "undo_backup_sha256",
            "undo_backup_integrity",
        }
        if (
            version_row
            and str(version_row["value"]) == str(REGISTRY_SCHEMA_VERSION)
            and has_physical_columns
            and fully_projected
            and reversible_columns <= migration_columns
        ):
            return
        # Additively project every legacy registry row into the identity model.
        # The legacy table and its database paths remain the compatibility view.
        conn.execute("BEGIN IMMEDIATE")
        try:
            if "db_device" not in identity_columns:
                conn.execute(
                    "ALTER TABLE namespace_identities ADD COLUMN db_device INTEGER"
                )
            if "db_inode" not in identity_columns:
                conn.execute(
                    "ALTER TABLE namespace_identities ADD COLUMN db_inode INTEGER"
                )
            for column in sorted(reversible_columns):
                if column not in migration_columns:
                    conn.execute(
                        f"ALTER TABLE namespace_migrations ADD COLUMN {column} TEXT"
                    )
            by_db: dict[str, list[sqlite3.Row]] = {}
            for row in legacy_rows:
                by_db.setdefault(str(row["db_path"]), []).append(row)
            for db_path in sorted(by_db):
                # A legacy registry can contain several labels for one file.
                # The earliest created_at wins; normalized/display labels break ties.
                group = sorted(
                    by_db[db_path],
                    key=lambda item: (
                        str(item["created_at"]),
                        normalize_namespace_label(str(item["name"])),
                        str(item["name"]),
                    ),
                )
                canonical_row = group[0]
                canonical_label = str(canonical_row["name"])
                canonical_norm = normalize_namespace_label(canonical_label)
                by_path = conn.execute(
                    """SELECT namespace_id,canonical_label_norm
                       FROM namespace_identities WHERE db_path=?""",
                    (db_path,),
                ).fetchone()
                if by_path:
                    namespace_id = str(by_path["namespace_id"])
                    effective_canonical_norm = str(by_path["canonical_label_norm"])
                else:
                    namespace_id = new_id()
                    effective_canonical_norm = canonical_norm
                    conn.execute(
                        """INSERT INTO namespace_identities(
                               namespace_id,canonical_label,canonical_label_norm,db_path,
                               db_device,db_inode,created_at,updated_at
                           ) VALUES (?,?,?,?,?,?,?,?)""",
                        (
                            namespace_id,
                            canonical_label,
                            canonical_norm,
                            db_path,
                            physical_identities[db_path][0],
                            physical_identities[db_path][1],
                            str(canonical_row["created_at"]),
                            str(canonical_row["updated_at"]),
                        ),
                    )
                for row in group:
                    label = str(row["name"])
                    norm = normalize_namespace_label(label)
                    by_alias = conn.execute(
                        "SELECT namespace_id FROM namespace_aliases WHERE normalized_label=?",
                        (norm,),
                    ).fetchone()
                    if by_alias and by_alias["namespace_id"] != namespace_id:
                        raise NamespaceCollisionError(
                            f"legacy namespace label collision after normalization: {label!r}"
                        )
                    if by_alias:
                        continue
                    conn.execute(
                        """INSERT INTO namespace_aliases(
                               normalized_label,label,namespace_id,is_canonical,created_at
                           ) VALUES (?,?,?,?,?)""",
                        (
                            norm,
                            label,
                            namespace_id,
                            int(norm == effective_canonical_norm),
                            str(row["created_at"]),
                        ),
                    )
                binding_rows = [*group[1:], canonical_row]
                for row in binding_rows:
                    legacy_repo = str(row["repo_path"] or "").strip()
                    if legacy_repo:
                        remote_identity, local_path = _repository_context(legacy_repo)
                        _bind_repository(
                            conn,
                            namespace_id=namespace_id,
                            label_norm=normalize_namespace_label(str(row["name"])),
                            repo_identity=remote_identity,
                            repo_path=local_path,
                            now=str(row["updated_at"]),
                        )
            for identity in conn.execute(
                "SELECT namespace_id,db_path FROM namespace_identities"
            ).fetchall():
                db_path = str(identity["db_path"])
                physical = physical_identities.get(db_path)
                if physical is None:
                    physical = validate_namespace_db_paths([db_path])[db_path]
                conn.execute(
                    """UPDATE namespace_identities
                       SET db_device=?,db_inode=? WHERE namespace_id=?""",
                    (physical[0], physical[1], identity["namespace_id"]),
                )
            conn.execute(
                "INSERT OR REPLACE INTO registry_meta(key,value) VALUES (?,?)",
                (REGISTRY_SCHEMA_VERSION_KEY, str(REGISTRY_SCHEMA_VERSION)),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()


def _registry() -> sqlite3.Connection:
    init_registry()
    return _connect(registry_path())


def _init_namespace_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            source TEXT,
            meta TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            idempotency_key TEXT,
            session_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            event_time TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_name TEXT,
            tool_input TEXT,
            tool_output TEXT,
            origin TEXT,
            tier TEXT NOT NULL,
            meta TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            tier TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (event_id) REFERENCES events(id)
        );
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            norm_name TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS relations (
            id TEXT PRIMARY KEY,
            src_entity TEXT NOT NULL,
            rel TEXT NOT NULL,
            dst_entity TEXT NOT NULL,
            event_id TEXT,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            weight REAL NOT NULL DEFAULT 1.0
        );
        CREATE TABLE IF NOT EXISTS entity_mentions (
            event_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            PRIMARY KEY (event_id, entity_id),
            FOREIGN KEY (event_id) REFERENCES events(id),
            FOREIGN KEY (entity_id) REFERENCES entities(id)
        );
        CREATE TABLE IF NOT EXISTS relation_evidence (
            event_id TEXT NOT NULL,
            src_entity TEXT NOT NULL,
            rel TEXT NOT NULL,
            dst_entity TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (event_id, src_entity, rel, dst_entity),
            FOREIGN KEY (event_id) REFERENCES events(id),
            FOREIGN KEY (src_entity) REFERENCES entities(id),
            FOREIGN KEY (dst_entity) REFERENCES entities(id)
        );
        CREATE TABLE IF NOT EXISTS embedding_jobs (
            memory_id TEXT PRIMARY KEY,
            queued_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
        CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time);
        CREATE INDEX IF NOT EXISTS idx_events_tier ON events(tier);
        CREATE INDEX IF NOT EXISTS idx_memories_event ON memories(event_id);
        CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(valid_from, valid_to);
        CREATE INDEX IF NOT EXISTS idx_entities_norm ON entities(norm_name, type);
        CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src_entity);
        CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst_entity);
        CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
        CREATE INDEX IF NOT EXISTS idx_relation_evidence_src ON relation_evidence(src_entity);
        CREATE INDEX IF NOT EXISTS idx_relation_evidence_dst ON relation_evidence(dst_entity);
        CREATE INDEX IF NOT EXISTS idx_embedding_jobs_queued ON embedding_jobs(queued_at);
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            id UNINDEXED,
            content,
            tokenize='porter unicode61'
        );
        """
    )
    conn.commit()


def _normalize_clock_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return text
    try:
        return utc_iso(parse_iso(text))
    except (TypeError, ValueError):
        return text


def _normalize_stored_clocks(conn: sqlite3.Connection) -> int:
    """Rewrite offset/naive timestamps to canonical UTC. Returns rows touched."""
    changed = 0
    for table, cols in _CLOCK_COLUMNS:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not exists:
            continue
        rows = conn.execute(f"SELECT rowid, {', '.join(cols)} FROM {table}").fetchall()
        for row in rows:
            sets: list[str] = []
            params: list[Any] = []
            for col in cols:
                old = row[col]
                if old is None:
                    continue
                new = _normalize_clock_value(old)
                if new != old:
                    sets.append(f"{col}=?")
                    params.append(new)
            if sets:
                params.append(row["rowid"])
                conn.execute(
                    f"UPDATE {table} SET {', '.join(sets)} WHERE rowid=?",
                    params,
                )
                changed += 1
    return changed


def _schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM meta WHERE key=?", (SCHEMA_VERSION_KEY,)
    ).fetchone()
    if not row:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def _ensure_correction_invariant_triggers(conn: sqlite3.Connection) -> None:
    """Reject malformed normal rows while allowing purge-scrubbed lineage."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='corrections'"
    ).fetchone()
    if not exists:
        return
    valid = """
        (
            NEW.target_tombstone_id IS NULL
            AND NEW.replacement_tombstone_id IS NULL
            AND NEW.origin IS NOT NULL
            AND NEW.session_id IS NOT NULL
            AND NEW.idempotency_key IS NOT NULL
            AND length(trim(
                NEW.idempotency_key,
                char(9) || char(10) || char(11) || char(12) || char(13) || ' '
            )) > 0
            AND length(NEW.idempotency_key) <= 512
            AND NEW.request_identity IS NOT NULL
            AND length(NEW.request_identity) = 71
            AND substr(NEW.request_identity, 1, 7) = 'sha256:'
            AND substr(NEW.request_identity, 8) NOT GLOB '*[^0-9a-f]*'
            AND NEW.request_payload IS NOT NULL
            AND typeof(NEW.request_payload) = 'blob'
            AND NEW.response_json IS NOT NULL
            AND json_valid(NEW.response_json) = 1
        )
        OR
        (
            (NEW.target_tombstone_id IS NOT NULL
             OR NEW.replacement_tombstone_id IS NOT NULL)
            AND NEW.origin IS NULL
            AND NEW.session_id IS NULL
            AND NEW.reason IS NULL
            AND NEW.idempotency_key IS NULL
            AND NEW.request_identity IS NULL
            AND NEW.request_payload IS NULL
            AND NEW.response_json IS NULL
        )
    """
    conn.execute(
        f"""
        CREATE TRIGGER IF NOT EXISTS corrections_invariant_insert
        BEFORE INSERT ON corrections
        WHEN NOT ({valid})
        BEGIN
            SELECT RAISE(ABORT, 'invalid correction invariant');
        END
        """
    )
    conn.execute(
        f"""
        CREATE TRIGGER IF NOT EXISTS corrections_invariant_update
        BEFORE UPDATE ON corrections
        WHEN NOT ({valid})
        BEGIN
            SELECT RAISE(ABORT, 'invalid correction invariant');
        END
        """
    )


def _ensure_correction_append_only_triggers(conn: sqlite3.Connection) -> None:
    """Block correction mutation unless this Store is in its purge transaction.

    The authorization function is registered only on Store-owned connections.
    An external SQLite connection cannot satisfy the trigger and fails closed.
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='corrections'"
    ).fetchone()
    if not exists:
        return
    for operation in ("UPDATE", "DELETE"):
        conn.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS corrections_append_only_{operation.lower()}
            BEFORE {operation} ON corrections
            WHEN haunt_privacy_purge_authorized() != 1
            BEGIN
                SELECT RAISE(ABORT, 'corrections are append-only');
            END
            """
        )


def _ensure_provenance_type_triggers(conn: sqlite3.Connection) -> None:
    """Require new structured provenance to use SQLite TEXT storage."""
    columns = {
        str(row["name"]) for row in conn.execute("PRAGMA table_info(events)").fetchall()
    }
    if "provenance" not in columns:
        return
    for operation in ("INSERT", "UPDATE OF provenance"):
        name = operation.lower().replace(" ", "_")
        conn.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS events_provenance_type_{name}
            BEFORE {operation} ON events
            WHEN NEW.provenance IS NOT NULL
                 AND typeof(NEW.provenance) != 'text'
            BEGIN
                SELECT RAISE(ABORT, 'event provenance must be text');
            END
            """
        )


def _ensure_namespace_schema(conn: sqlite3.Connection) -> None:
    """Create tables and run one-time migrations. Not invoked per query."""
    _init_namespace_schema(conn)
    current = _schema_version(conn)
    if current >= SCHEMA_VERSION:
        _ensure_correction_invariant_triggers(conn)
        _ensure_correction_append_only_triggers(conn)
        _ensure_provenance_type_triggers(conn)
        conn.commit()
        return
    if current < 1:
        _normalize_stored_clocks(conn)
    if current < 2:
        event_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        if "idempotency_key" not in event_columns:
            conn.execute("ALTER TABLE events ADD COLUMN idempotency_key TEXT")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idempotency "
            "ON events(idempotency_key) WHERE idempotency_key IS NOT NULL"
        )
    if current < 3:
        conn.execute(
            """
            INSERT OR IGNORE INTO embedding_jobs(memory_id, queued_at)
            SELECT id, created_at FROM memories
            WHERE embedding IS NULL AND TRIM(content) != ''
            """
        )
    if current < 4:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lineage_tombstones (
                schema_version INTEGER NOT NULL,
                tombstone_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK (status = 'erased'),
                erased_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                target_memory_id TEXT,
                target_tombstone_id TEXT,
                replacement_memory_id TEXT,
                replacement_tombstone_id TEXT,
                corrected_at TEXT NOT NULL,
                origin TEXT,
                session_id TEXT,
                reason TEXT,
                idempotency_key TEXT,
                request_identity TEXT,
                request_payload BLOB,
                response_json TEXT,
                CHECK ((target_memory_id IS NOT NULL) !=
                       (target_tombstone_id IS NOT NULL)),
                CHECK (replacement_memory_id IS NULL OR
                       replacement_tombstone_id IS NULL)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_idempotency
                ON corrections(idempotency_key)
                WHERE idempotency_key IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_target_memory
                ON corrections(target_memory_id)
                WHERE target_memory_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_target_tombstone
                ON corrections(target_tombstone_id)
                WHERE target_tombstone_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_replacement_memory
                ON corrections(replacement_memory_id)
                WHERE replacement_memory_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_replacement_tombstone
                ON corrections(replacement_tombstone_id)
                WHERE replacement_tombstone_id IS NOT NULL;
            """
        )
    if current < 8:
        event_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        if "provenance" not in event_columns:
            conn.execute(
                "ALTER TABLE events ADD COLUMN provenance TEXT "
                "CHECK (provenance IS NULL OR "
                "(json_valid(provenance)=1 AND json_type(provenance)='object'))"
            )
    _ensure_correction_invariant_triggers(conn)
    _ensure_correction_append_only_triggers(conn)
    _ensure_provenance_type_triggers(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (SCHEMA_VERSION_KEY, str(SCHEMA_VERSION)),
    )
    conn.commit()


@dataclass
class _FreshNamespaceClaim:
    target: Path
    temporary: Path
    claim_fd: int
    identity: tuple[int, int]
    conn: sqlite3.Connection
    sidecars: SQLiteSidecarGuard

    def verify_for_publication(self) -> None:
        """Confirm the live claim and final name still identify one safe file."""
        info = os.fstat(self.claim_fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or int(info.st_nlink) != 1
            or (int(info.st_dev), int(info.st_ino)) != self.identity
        ):
            raise NamespacePathError(
                "fresh namespace database claim changed before publication"
            )
        validate_namespace_db_paths(
            [str(self.target)], expected={str(self.target): self.identity}
        )

    def close(self, *, remove_target: bool) -> None:
        with _NAMESPACE_DB_HANDLE_LOCK:
            self.conn.close()
        self.sidecars.remove_claimed_files()
        try:
            os.close(self.claim_fd)
        except OSError:
            pass
        for extra in (self.temporary,):
            try:
                extra.unlink()
            except FileNotFoundError:
                pass
        if remove_target:
            try:
                current = self.target.lstat()
            except OSError:
                return
            if (
                stat.S_ISREG(current.st_mode)
                and (int(current.st_dev), int(current.st_ino)) == self.identity
            ):
                self.target.unlink()


def _claim_fresh_namespace_db(target: Path) -> _FreshNamespaceClaim:
    """Initialize a fresh DB privately, then atomically claim its final path."""
    with _sqlite_configuration_lock():
        return _claim_fresh_namespace_db_with_configuration_lock(target)


def _claim_fresh_namespace_db_with_configuration_lock(
    target: Path,
) -> _FreshNamespaceClaim:
    """Initialize a fresh DB while holding serialized sidecar configuration."""
    _validate_unmapped_namespace_target(target)
    root = validate_namespace_root()
    claim_fd, raw_temporary = tempfile.mkstemp(
        prefix=".haunt-claim-", suffix=".db", dir=str(root)
    )
    temporary = Path(raw_temporary)
    info = os.fstat(claim_fd)
    identity = int(info.st_dev), int(info.st_ino)
    if not stat.S_ISREG(info.st_mode) or int(info.st_nlink) != 1:
        os.close(claim_fd)
        temporary.unlink(missing_ok=True)
        raise NamespacePathError("failed to claim a unique regular namespace database")
    before = _fd_snapshot()
    conn: sqlite3.Connection | None = None
    sidecars: SQLiteSidecarGuard | None = None
    linked = False
    try:
        sidecars = SQLiteSidecarGuard.acquire(temporary, claim_missing=True)
        _sqlite_sidecar_open_hook(temporary)
        sidecars.verify()
        conn = _raw_connect(temporary, create=False)
        assert isinstance(conn, _SidecarGuardedConnection)
        conn.set_sidecar_guard(sidecars, clean_unused_claims=True)
        _verify_new_sqlite_fd(
            before, identity, ignored_identities=_sidecar_identities(sidecars)
        )
        sidecars.verify()
        validate_namespace_db_paths(
            [str(temporary)], expected={str(temporary): identity}
        )
        _configure_connection(conn, temporary, sidecars=sidecars)
        conn.preserve_sidecar_claims()
        _init_namespace_schema(conn)
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise NamespaceCollisionError(
                f"target label has an unmapped filesystem entry at {target}"
            ) from exc
        linked = True
        temporary.unlink()
        _fresh_namespace_claim_hook(target)
        validate_namespace_db_paths(
            [str(target)], expected={str(target): identity}
        )
        _verify_new_sqlite_fd(
            before, identity, ignored_identities=_sidecar_identities(sidecars)
        )
        return _FreshNamespaceClaim(
            target=target,
            temporary=temporary,
            claim_fd=claim_fd,
            identity=identity,
            conn=conn,
            sidecars=sidecars,
        )
    except Exception:
        if conn is not None:
            conn.close()
        elif sidecars is not None:
            sidecars.close(clean_unused_claims=True)
        if sidecars is not None:
            sidecars.remove_claimed_files()
        try:
            os.close(claim_fd)
        except OSError:
            pass
        for extra in (temporary,):
            try:
                extra.unlink()
            except FileNotFoundError:
                pass
        if linked:
            try:
                current = target.lstat()
            except OSError:
                current = None
            if current is not None and (
                stat.S_ISREG(current.st_mode)
                and (int(current.st_dev), int(current.st_ino)) == identity
            ):
                target.unlink()
        raise


def ensure_vec_table(
    conn: sqlite3.Connection, dim: int, *, commit: bool = True
) -> bool:
    if dim <= 0 or not _vec_loaded(conn):
        return False
    existing = conn.execute("SELECT value FROM meta WHERE key='embed_dim'").fetchone()
    if existing and int(existing["value"]) != dim:
        conn.execute("DROP TABLE IF EXISTS vec_memories")
    try:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                id TEXT PRIMARY KEY,
                embedding FLOAT[{int(dim)}] distance_metric=cosine
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_dim', ?)",
            (str(dim),),
        )
        if commit:
            conn.commit()
        return True
    except sqlite3.Error:
        return False


def _repository_context(value: str | None) -> tuple[str | None, str | None]:
    """Return normalized remote identity and local repository root, if known."""
    raw = (value or "").strip()
    if not raw:
        return None, None
    remote = repository_identity(raw)
    if remote:
        return remote, None
    path = Path(raw).expanduser().resolve()
    remote_url, root = _git_repo_context(path)
    return repository_identity(remote_url), str((root or path).resolve())


def _identity_row(conn: sqlite3.Connection, label: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT i.*
        FROM namespace_aliases a
        JOIN namespace_identities i ON i.namespace_id=a.namespace_id
        WHERE a.normalized_label=?
        """,
        (normalize_namespace_label(label),),
    ).fetchone()


def _resolve_namespace_identity_once(name: str) -> dict[str, Any] | None:
    try:
        conn = _readonly_registry()
    except FileNotFoundError:
        return None
    result: dict[str, Any] | None = None
    try:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        validate_registry_db_sources(conn, tables)
        row = _identity_row(conn, name)
        if not row:
            result = None
        else:
            aliases = conn.execute(
                """SELECT label, normalized_label, is_canonical, source_alias_norm
                   FROM namespace_aliases WHERE namespace_id=?
                   ORDER BY is_canonical DESC, normalized_label""",
                (row["namespace_id"],),
            ).fetchall()
            result = {**dict(row), "aliases": [dict(alias) for alias in aliases]}
    except sqlite3.Error as exc:
        raise NamespacePathError(f"cannot read namespace registry: {exc}") from exc
    finally:
        conn.close()
    return result


_CONCURRENT_REGISTRY_CHANGE_MARKERS = (
    "storage drift",
    "registry changed repeatedly",
    "incomplete WAL state",
    "changed while copying read snapshot",
    "changed while snapshotting",
    "sidecar appeared during safe open",
    "sidecar disappeared during safe open",
    "sidecar changed while opening",
    "sidecar physical identity changed while opening",
    "sidecar physical identity changed:",
)


def _is_concurrent_registry_change(exc: NamespacePathError) -> bool:
    message = str(exc)
    return any(marker in message for marker in _CONCURRENT_REGISTRY_CHANGE_MARKERS)


def resolve_namespace_identity(name: str) -> dict[str, Any] | None:
    """Resolve a label without writes, retrying a concurrently changing registry."""
    for attempt in range(32):
        try:
            return _resolve_namespace_identity_once(name)
        except NamespacePathError as exc:
            if not _is_concurrent_registry_change(exc) or attempt == 31:
                raise
    return None


def _resolve_namespace_id_once(namespace_id: str) -> dict[str, Any] | None:
    try:
        conn = _readonly_registry()
    except FileNotFoundError:
        return None
    result: dict[str, Any] | None = None
    try:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        validate_registry_db_sources(conn, tables)
        row = conn.execute(
            "SELECT * FROM namespace_identities WHERE namespace_id=?",
            (namespace_id,),
        ).fetchone()
        if not row:
            result = None
        else:
            aliases = conn.execute(
                """SELECT label,normalized_label,is_canonical,source_alias_norm
                   FROM namespace_aliases WHERE namespace_id=?
                   ORDER BY is_canonical DESC,normalized_label""",
                (namespace_id,),
            ).fetchall()
            result = {**dict(row), "aliases": [dict(alias) for alias in aliases]}
    except sqlite3.Error as exc:
        raise NamespacePathError(f"cannot read namespace registry: {exc}") from exc
    finally:
        conn.close()
    return result


def resolve_namespace_id(namespace_id: str) -> dict[str, Any] | None:
    """Resolve a stable ID without writes, retrying concurrent registry drift."""
    for attempt in range(32):
        try:
            return _resolve_namespace_id_once(namespace_id)
        except NamespacePathError as exc:
            if not _is_concurrent_registry_change(exc) or attempt == 31:
                raise
    return None


def _bind_repository(
    conn: sqlite3.Connection,
    *,
    namespace_id: str,
    label_norm: str,
    repo_identity: str | None,
    repo_path: str | None,
    now: str,
) -> None:
    if not repo_identity and not repo_path:
        return
    remote_row = None
    path_row = None
    if repo_identity:
        remote_row = conn.execute(
            "SELECT namespace_id,binding_id FROM repository_bindings WHERE repository_identity=?",
            (repo_identity,),
        ).fetchone()
        if remote_row and remote_row["namespace_id"] != namespace_id:
            raise NamespaceCollisionError(
                f"repository {repo_identity!r} is already bound to another namespace"
            )
    if repo_path:
        path_row = conn.execute(
            "SELECT namespace_id,binding_id FROM repository_bindings WHERE repo_path=?",
            (repo_path,),
        ).fetchone()
        if path_row and path_row["namespace_id"] != namespace_id:
            raise NamespaceCollisionError(
                f"repository path {repo_path!r} is already bound to another namespace"
            )
    if remote_row and path_row and remote_row["binding_id"] != path_row["binding_id"]:
        raise NamespaceCollisionError(
            "repository remote and path are recorded by different bindings"
        )
    row = remote_row or path_row
    if row:
        conn.execute(
            """UPDATE repository_bindings
               SET repo_path=COALESCE(?,repo_path),label_norm=?,updated_at=?
               WHERE binding_id=?""",
            (repo_path, label_norm, now, row["binding_id"]),
        )
        return
    conn.execute(
        """INSERT INTO repository_bindings(
               binding_id,namespace_id,repository_identity,repo_path,label_norm,
               created_at,updated_at
           ) VALUES (?,?,?,?,?,?,?)""",
        (new_id(), namespace_id, repo_identity, repo_path, label_norm, now, now),
    )


def register_namespace(name: str, repo_path: str | None = None) -> Path:
    label = safe_name(name)
    norm = normalize_namespace_label(label)
    now = now_iso()
    repo_identity, repo = _repository_context(repo_path)
    conn = _registry()
    claim: _FreshNamespaceClaim | None = None
    row: sqlite3.Row | None = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _identity_row(conn, label)
        if not row and (repo_identity or repo):
            if repo_identity:
                binding = conn.execute(
                    "SELECT namespace_id FROM repository_bindings WHERE repository_identity=?",
                    (repo_identity,),
                ).fetchone()
            else:
                binding = conn.execute(
                    "SELECT namespace_id FROM repository_bindings WHERE repo_path=?",
                    (repo,),
                ).fetchone()
            if binding:
                row = conn.execute(
                    "SELECT * FROM namespace_identities WHERE namespace_id=?",
                    (binding["namespace_id"],),
                ).fetchone()
                _validate_unmapped_namespace_target(
                    namespaces_dir() / f"{label}.db",
                    mapped_db_path=Path(str(row["db_path"])),
                )
                conn.execute(
                    """INSERT INTO namespace_aliases(
                           normalized_label,label,namespace_id,is_canonical,created_at
                       ) VALUES (?,?,?,?,?)""",
                    (norm, label, row["namespace_id"], 0, now),
                )
        if row:
            db = Path(str(row["db_path"]))
            namespace_id = str(row["namespace_id"])
            canonical = str(row["canonical_label"])
            if repo_identity or repo:
                conn.execute(
                    "UPDATE namespace_identities SET updated_at=? WHERE namespace_id=?",
                    (now, namespace_id),
                )
        else:
            db = namespaces_dir() / f"{label}.db"
            path_owner = conn.execute(
                "SELECT canonical_label FROM namespace_identities WHERE db_path=?",
                (str(db),),
            ).fetchone()
            if path_owner:
                raise NamespaceCollisionError(
                    f"database path {db} is already mapped to {path_owner['canonical_label']!r}"
                )
            claim = _claim_fresh_namespace_db(db)
            namespace_id = new_id()
            canonical = label
            conn.execute(
                """INSERT INTO namespace_identities(
                       namespace_id,canonical_label,canonical_label_norm,db_path,
                       db_device,db_inode,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    namespace_id,
                    label,
                    norm,
                    str(db),
                    claim.identity[0],
                    claim.identity[1],
                    now,
                    now,
                ),
            )
            conn.execute(
                """INSERT INTO namespace_aliases(
                       normalized_label,label,namespace_id,is_canonical,created_at
                   ) VALUES (?,?,?,?,?)""",
                (norm, label, namespace_id, 1, now),
            )
            conn.execute(
                """INSERT INTO namespaces(name,repo_path,db_path,created_at,updated_at)
                   VALUES (?,?,?,?,?)""",
                (canonical, repo, str(db), now, now),
            )
        _bind_repository(
            conn,
            namespace_id=namespace_id,
            label_norm=norm,
            repo_identity=repo_identity,
            repo_path=repo,
            now=now,
        )
        if repo:
            conn.execute(
                "UPDATE namespaces SET repo_path=COALESCE(?,repo_path),updated_at=? WHERE db_path=?",
                (repo, now, str(db)),
            )
        if claim is not None:
            # Registry publication is the commit below, so revalidate the
            # still-open atomic claim immediately before making it visible.
            claim.verify_for_publication()
        conn.commit()
    except Exception:
        conn.rollback()
        if claim is not None:
            claim.close(remove_target=True)
        raise
    finally:
        conn.close()
    ns = (
        claim.conn
        if claim is not None
        else _open_mapped_namespace_db(
            db, expected=(row["db_device"], row["db_inode"])
        )
    )
    try:
        if claim is None:
            _ensure_namespace_schema(ns)
        if repo:
            ns.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('repo_path', ?)",
                (repo,),
            )
            ns.commit()
    finally:
        if claim is not None:
            claim.close(remove_target=False)
        else:
            with _NAMESPACE_DB_HANDLE_LOCK:
                ns.close()
    return db


def namespace_exists(name: str) -> bool:
    conn = _registry()
    try:
        return _identity_row(conn, name) is not None
    finally:
        conn.close()


def touch_namespace(name: str, *, namespace_id: str | None = None) -> None:
    conn = _registry()
    touched_namespace_id: str | None = None
    try:
        row = (
            conn.execute(
                "SELECT * FROM namespace_identities WHERE namespace_id=?",
                (namespace_id,),
            ).fetchone()
            if namespace_id is not None
            else _identity_row(conn, name)
        )
        if not row:
            return
        touched_namespace_id = str(row["namespace_id"])
        now = now_iso()
        conn.execute("UPDATE namespace_identities SET updated_at=? WHERE namespace_id=?", (now, row["namespace_id"]))
        conn.execute("UPDATE namespaces SET updated_at=? WHERE db_path=?", (now, row["db_path"]))
        conn.commit()
    finally:
        conn.close()
    if touched_namespace_id:
        resolve_namespace_id(touched_namespace_id)


def list_namespace_rows() -> list[dict[str, Any]]:
    registry_error: str | None = None
    registry_exception: NamespacePathError | None = None
    try:
        init_registry()
    except NamespacePathError as exc:
        registry_error = str(exc)
        registry_exception = exc
    try:
        conn = _readonly_registry() if registry_error else _connect(registry_path())
    except FileNotFoundError:
        if registry_exception is not None:
            raise registry_exception
        return []
    try:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        rows = (
            conn.execute(
                """SELECT i.*
                   FROM namespace_identities i
                   ORDER BY i.canonical_label_norm"""
            ).fetchall()
            if "namespace_identities" in tables
            else []
        )
        legacy_rows = (
            conn.execute(
                """SELECT name,repo_path,db_path,created_at,updated_at
                   FROM namespaces
                   ORDER BY db_path,created_at,name"""
            ).fetchall()
            if "namespaces" in tables
            else []
        )
        legacy_by_path: dict[str, list[sqlite3.Row]] = {}
        for legacy in legacy_rows:
            legacy_by_path.setdefault(str(legacy["db_path"]), []).append(legacy)
        if registry_error is None:
            try:
                validate_namespace_db_paths(
                    [
                        *(str(row["db_path"]) for row in rows),
                        *(str(row["db_path"]) for row in legacy_rows),
                    ],
                    expected={
                        str(row["db_path"]): (
                            row["db_device"] if "db_device" in row.keys() else None,
                            row["db_inode"] if "db_inode" in row.keys() else None,
                        )
                        for row in rows
                    },
                )
            except NamespacePathError as exc:
                registry_error = str(exc)
        if not rows:
            out: list[dict[str, Any]] = []
            for db_path in sorted(legacy_by_path):
                group = sorted(
                    legacy_by_path[db_path],
                    key=lambda candidate: (
                        str(candidate["created_at"]),
                        normalize_namespace_label(str(candidate["name"])),
                        str(candidate["name"]),
                    ),
                )
                canonical = group[0]
                out.append(
                    {
                        "namespace_id": None,
                        "canonical_label": str(canonical["name"]),
                        "canonical_label_norm": normalize_namespace_label(
                            str(canonical["name"])
                        ),
                        "db_path": db_path,
                        "created_at": str(canonical["created_at"]),
                        "updated_at": str(canonical["updated_at"]),
                        "name": str(canonical["name"]),
                        "repo_path": canonical["repo_path"],
                        "aliases": [str(candidate["name"]) for candidate in group],
                        **({"error": registry_error} if registry_error else {}),
                    }
                )
            return out
        out: list[dict[str, Any]] = []
        for row in rows:
            aliases = (
                conn.execute(
                    "SELECT label FROM namespace_aliases WHERE namespace_id=? ORDER BY normalized_label",
                    (row["namespace_id"],),
                ).fetchall()
                if "namespace_aliases" in tables
                else []
            )
            legacy = legacy_by_path.get(str(row["db_path"]), [])
            ordered_legacy = sorted(
                legacy,
                key=lambda candidate: (
                    str(candidate["name"]) != str(row["canonical_label"]),
                    str(candidate["created_at"]),
                    str(candidate["name"]),
                ),
            )
            repo_path = ordered_legacy[0]["repo_path"] if ordered_legacy else None
            out.append(
                {
                    **dict(row),
                    "name": str(row["canonical_label"]),
                    "repo_path": repo_path,
                    "aliases": [str(a["label"]) for a in aliases],
                    **({"error": registry_error} if registry_error else {}),
                }
            )
        return out
    finally:
        conn.close()


def _readonly_registry() -> sqlite3.Connection:
    """Open the existing registry without creating files or setting pragmas."""
    path = registry_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    for _attempt in range(3):
        primary = SQLitePrimaryGuard.acquire(path, create_missing=False)
        try:
            sidecars = SQLiteSidecarGuard.acquire(path, claim_missing=False)
        except Exception:
            primary.close()
            raise
        try:
            immutable, snapshot = readonly_sqlite_mode(path)
            primary.verify()
            sidecars.verify()
            if immutable:
                primary.close()
                sidecars.close(clean_unused_claims=False)
                return _open_readonly_connection(
                    path,
                    immutable=True,
                    claim_missing=False,
                    zero_write_snapshot=snapshot,
                )
            temporary, shadow = temporary_sqlite_shadow(path, snapshot)
            try:
                primary.verify()
                sidecars.verify()
                if sqlite_storage_snapshot(path) != snapshot:
                    temporary.cleanup()
                    continue
                materialize_sqlite_shadow(shadow, temporary)
                conn = _open_readonly_connection(
                    shadow, immutable=True, claim_missing=False
                )
                assert isinstance(conn, _SidecarGuardedConnection)
                conn.set_zero_write_snapshot(path, snapshot)
                conn.set_temporary_read_dir(temporary)
                return conn
            except Exception:
                temporary.cleanup()
                raise
        finally:
            sidecars.close(clean_unused_claims=False)
            primary.close()
    raise NamespacePathError(
        "registry changed repeatedly while creating a zero-write read snapshot"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _registry_state(
    conn: sqlite3.Connection, *, legacy_only: bool = False, compatibility_v4: bool = False
) -> dict[str, Any]:
    """Return a deterministic logical registry snapshot without changing it."""
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    selected = ["namespaces"] if legacy_only else [
        "namespaces",
        "registry_meta",
        "namespace_identities",
        "namespace_aliases",
        "repository_bindings",
        "namespace_migrations",
    ]
    state: dict[str, Any] = {}
    for table in selected:
        if table not in tables:
            continue
        columns = [
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        ]
        if compatibility_v4 and table == "registry_meta":
            continue
        if compatibility_v4 and table == "namespace_migrations":
            columns = [
                column
                for column in columns
                if column
                in {
                    "migration_id", "namespace_id", "action", "old_label",
                    "old_label_norm", "new_label", "new_label_norm",
                    "repository_identity", "repository_key", "applied_at",
                }
            ]
        rows = [
            {column: row[column] for column in columns}
            for row in conn.execute(f"SELECT * FROM {table}").fetchall()
        ]
        rows.sort(key=_canonical_json)
        state[table] = {"columns": columns, "rows": rows}
    return state


def _state_digest(state: Any) -> str:
    return hashlib.sha256(_canonical_json(state).encode("utf-8")).hexdigest()


def _finish_plan(report: dict[str, Any], registry_state: dict[str, Any]) -> dict[str, Any]:
    report["registry_state_digest"] = _state_digest(registry_state)
    operation = {key: value for key, value in report.items() if key != "mode"}
    report["plan_digest"] = _state_digest(
        {"protocol": "haunt-namespace-migration-v1", "operation": operation}
    )
    return report


def _namespace_state(conn: sqlite3.Connection, namespace_id: str) -> dict[str, Any]:
    identity = conn.execute(
        "SELECT * FROM namespace_identities WHERE namespace_id=?", (namespace_id,)
    ).fetchone()
    if identity is None:
        raise UnknownNamespaceError(namespace_id)
    db_path = str(identity["db_path"])

    def rows(query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        values = [dict(row) for row in conn.execute(query, params).fetchall()]
        values.sort(key=_canonical_json)
        return values

    return {
        "identity": dict(identity),
        "aliases": rows(
            "SELECT * FROM namespace_aliases WHERE namespace_id=?",
            (namespace_id,),
        ),
        "bindings": rows(
            "SELECT * FROM repository_bindings WHERE namespace_id=?",
            (namespace_id,),
        ),
        "legacy": rows("SELECT * FROM namespaces WHERE db_path=?", (db_path,)),
    }


def _private_backup_root() -> tuple[Path, int]:
    """Atomically create/hold the private direct-child registry backup directory."""
    nofollow = required_o_nofollow()
    home = haunt_home()
    backup_root = home / "backups"
    for _attempt in range(3):
        try:
            info = backup_root.lstat()
        except FileNotFoundError:
            try:
                backup_root.mkdir(mode=0o700)
            except FileExistsError:
                continue
            info = backup_root.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise NamespaceMigrationError("registry backup directory is unsafe")
        try:
            fd = os.open(
                backup_root,
                os.O_RDONLY
                | nofollow
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as exc:
            raise NamespaceMigrationError(
                "registry backup directory cannot be opened safely"
            ) from exc
        try:
            os.fchmod(fd, 0o700)
            _verify_private_backup_root(backup_root, fd)
        except Exception:
            os.close(fd)
            raise
        return backup_root, fd
    raise NamespaceMigrationError("registry backup directory changed repeatedly")


def _verify_private_backup_root(backup_root: Path, fd: int) -> tuple[int, int]:
    """Reverify the private backup directory without releasing its descriptor."""
    try:
        held = os.fstat(fd)
        current = backup_root.lstat()
    except OSError as exc:
        raise NamespaceMigrationError("registry backup directory changed") from exc
    identity = int(held.st_dev), int(held.st_ino)
    if (
        not stat.S_ISDIR(held.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or identity != (int(current.st_dev), int(current.st_ino))
        or stat.S_IMODE(held.st_mode) != 0o700
        or stat.S_IMODE(current.st_mode) != 0o700
        or int(held.st_uid) != os.geteuid()
        or int(current.st_uid) != os.geteuid()
    ):
        raise NamespaceMigrationError("registry backup directory changed")
    return identity


def _registry_backup_hook(_phase: str, _backup_root: Path) -> None:
    """Test hook around descriptor-relative registry backup publication."""


def _relative_regular_file(
    directory_fd: int, name: str, held_fd: int
) -> tuple[int, int]:
    """Verify a private single-link regular file through a held directory."""
    if not name or name != Path(name).name:
        raise NamespaceMigrationError("registry backup filename is unsafe")
    try:
        held = os.fstat(held_fd)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise NamespaceMigrationError("registry backup changed") from exc
    identity = int(held.st_dev), int(held.st_ino)
    if (
        not stat.S_ISREG(held.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or int(held.st_nlink) != 1
        or int(current.st_nlink) != 1
        or identity != (int(current.st_dev), int(current.st_ino))
        or stat.S_IMODE(held.st_mode) != 0o600
        or stat.S_IMODE(current.st_mode) != 0o600
    ):
        raise NamespaceMigrationError("registry backup is not a private regular file")
    return identity


def _held_file_sha256(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _held_sqlite_integrity(fd: int, identity: tuple[int, int]) -> str:
    """Run immutable integrity_check through the already-held file descriptor."""
    errors: list[Exception] = []
    for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
        descriptor_path = directory / str(fd)
        try:
            current = descriptor_path.stat()
            # Darwin's Python 3.10 reports the /dev/fd proxy device with a
            # different signed representation; the kernel-controlled proxy
            # inode still matches the held descriptor exactly.
            if not stat.S_ISREG(current.st_mode) or int(current.st_ino) != identity[1]:
                continue
            conn = sqlite3.connect(
                f"{descriptor_path.as_uri()}?mode=ro&immutable=1", uri=True
            )
            try:
                row = conn.execute("PRAGMA integrity_check").fetchone()
                result = str(row[0]) if row else "missing"
            finally:
                conn.close()
            held = os.fstat(fd)
            current = descriptor_path.stat()
            if (
                (int(held.st_dev), int(held.st_ino)) != identity
                or not stat.S_ISREG(current.st_mode)
                or int(current.st_ino) != identity[1]
            ):
                raise NamespaceMigrationError(
                    "registry backup descriptor changed during verification"
                )
            return result
        except (OSError, sqlite3.Error, NamespaceMigrationError) as exc:
            errors.append(exc)
    raise NamespaceMigrationError(
        "registry backup cannot be verified through its held descriptor"
    ) from (errors[-1] if errors else None)


def _unlink_relative_identity(
    directory_fd: int, name: str | None, identity: tuple[int, int] | None
) -> None:
    if not name or identity is None:
        return
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and (int(current.st_dev), int(current.st_ino)) == identity
    ):
        try:
            os.unlink(name, dir_fd=directory_fd)
        except OSError:
            pass


class _VerifiedRegistryBackup(dict[str, str]):
    """Backup report retaining its parent and file identity until registry commit."""

    def __init__(
        self,
        report: dict[str, str],
        *,
        backup_root: Path,
        backup_root_fd: int,
        final_name: str,
        final_fd: int,
        identity: tuple[int, int],
    ) -> None:
        super().__init__(report)
        self._backup_root = backup_root
        self._backup_root_fd = backup_root_fd
        self._final_name = final_name
        self._final_fd = final_fd
        self._identity = identity
        self._closed = False

    def verify_for_record(self) -> None:
        if self._closed:
            raise NamespaceMigrationError("registry backup guard is closed")
        _registry_backup_hook("before_record", self._backup_root)
        _verify_private_backup_root(self._backup_root, self._backup_root_fd)
        if (
            _relative_regular_file(
                self._backup_root_fd, self._final_name, self._final_fd
            )
            != self._identity
            or _held_file_sha256(self._final_fd) != self["sha256"]
        ):
            raise NamespaceMigrationError("registry backup changed before recording")
        os.fsync(self._final_fd)
        os.fsync(self._backup_root_fd)
        _verify_private_backup_root(self._backup_root, self._backup_root_fd)
        if (
            _relative_regular_file(
                self._backup_root_fd, self._final_name, self._final_fd
            )
            != self._identity
        ):
            raise NamespaceMigrationError("registry backup changed before recording")

    def discard(self) -> None:
        if self._closed:
            return
        _unlink_relative_identity(
            self._backup_root_fd, self._final_name, self._identity
        )
        try:
            os.fsync(self._backup_root_fd)
        except OSError:
            pass
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in (self._final_fd, self._backup_root_fd):
            try:
                os.close(fd)
            except OSError:
                pass

    def __del__(self) -> None:
        self.close()


def _backup_registry(*, purpose: str) -> _VerifiedRegistryBackup:
    """Create and verify a consistent private backup of registry.db only."""
    backup_root, backup_root_fd = _private_backup_root()
    fd = -1
    final_fd = -1
    temp_name: str | None = None
    final_name: str | None = None
    backup_identity: tuple[int, int] | None = None
    source: sqlite3.Connection | None = None
    try:
        _registry_backup_hook("before_create", backup_root)
        _verify_private_backup_root(backup_root, backup_root_fd)
        temp_name = f".registry-backup-{new_id()}.db"
        fd = os.open(
            temp_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | required_o_nofollow()
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=backup_root_fd,
        )
        os.fchmod(fd, 0o600)
        created = os.fstat(fd)
        backup_identity = int(created.st_dev), int(created.st_ino)
        _relative_regular_file(backup_root_fd, temp_name, fd)
        source = _readonly_registry()
        assert isinstance(source, _SidecarGuardedConnection)
        source.copy_primary_to_fd(fd)
        source.close()
        source = None
        os.fsync(fd)
        _relative_regular_file(backup_root_fd, temp_name, fd)
        _registry_backup_hook("before_link", backup_root)
        _verify_private_backup_root(backup_root, backup_root_fd)
        _relative_regular_file(backup_root_fd, temp_name, fd)
        final_name = f"registry-{purpose}-{new_id()}.db"
        os.link(
            temp_name,
            final_name,
            src_dir_fd=backup_root_fd,
            dst_dir_fd=backup_root_fd,
            follow_symlinks=False,
        )
        os.unlink(temp_name, dir_fd=backup_root_fd)
        temp_name = None
        os.fsync(backup_root_fd)
        final_fd = os.open(
            final_name,
            os.O_RDONLY
            | required_o_nofollow()
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=backup_root_fd,
        )
        if _relative_regular_file(backup_root_fd, final_name, final_fd) != backup_identity:
            raise NamespaceMigrationError("registry backup identity changed")
        digest = _held_file_sha256(final_fd)
        os.fsync(final_fd)
        _registry_backup_hook("before_final_verify", backup_root)
        _verify_private_backup_root(backup_root, backup_root_fd)
        if _relative_regular_file(backup_root_fd, final_name, final_fd) != backup_identity:
            raise NamespaceMigrationError("registry backup identity changed")
        final = backup_root / final_name
        verified = _held_sqlite_integrity(final_fd, backup_identity)
        _verify_private_backup_root(backup_root, backup_root_fd)
        if _relative_regular_file(backup_root_fd, final_name, final_fd) != backup_identity:
            raise NamespaceMigrationError("registry backup identity changed")
        if verified != "ok" or _held_file_sha256(final_fd) != digest:
            raise NamespaceMigrationError("registry backup verification failed")
        os.fsync(final_fd)
        os.fsync(backup_root_fd)
        _verify_private_backup_root(backup_root, backup_root_fd)
        if _relative_regular_file(backup_root_fd, final_name, final_fd) != backup_identity:
            raise NamespaceMigrationError("registry backup identity changed")
        return _VerifiedRegistryBackup(
            {
                "path": str(final),
                "sha256": digest,
                "integrity": verified,
            },
            backup_root=backup_root,
            backup_root_fd=os.dup(backup_root_fd),
            final_name=final_name,
            final_fd=os.dup(final_fd),
            identity=backup_identity,
        )
    except Exception:
        _unlink_relative_identity(
            backup_root_fd, temp_name, backup_identity
        )
        _unlink_relative_identity(
            backup_root_fd, final_name, backup_identity
        )
        try:
            os.fsync(backup_root_fd)
        except OSError:
            pass
        raise
    finally:
        if final_fd >= 0:
            os.close(final_fd)
        if fd >= 0:
            os.close(fd)
        if source is not None:
            source.close()
        os.close(backup_root_fd)


def _legacy_namespace_change_source(
    conn: sqlite3.Connection, old_display: str, old_norm: str
) -> tuple[sqlite3.Row, list[sqlite3.Row], list[sqlite3.Row]]:
    rows = conn.execute(
        "SELECT name,repo_path,db_path,created_at,updated_at FROM namespaces"
    ).fetchall()
    matches = [
        row
        for row in rows
        if normalize_namespace_label(str(row["name"])) == old_norm
    ]
    if not matches:
        raise UnknownNamespaceError(old_display)
    paths = {str(row["db_path"]) for row in matches}
    if len(paths) != 1:
        raise NamespaceCollisionError(
            "legacy namespace labels collide after normalization"
        )
    db_path = next(iter(paths))
    group = sorted(
        (row for row in rows if str(row["db_path"]) == db_path),
        key=lambda row: (
            str(row["created_at"]),
            normalize_namespace_label(str(row["name"])),
            str(row["name"]),
        ),
    )
    return group[0], group, rows


def _plan_namespace_label_read_only(
    *,
    old_display: str,
    new_display: str,
    old_norm: str,
    new_norm: str,
    repository_identity_value: str | None,
    repository_path: str | None,
    action: str,
) -> dict[str, Any]:
    """Plan a label change against legacy or current schema without writes."""
    try:
        conn = _readonly_registry()
    except FileNotFoundError as exc:
        raise UnknownNamespaceError(old_display) from exc
    try:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        physical_paths: list[str] = []
        expected_paths: dict[str, tuple[int | None, int | None]] = {}
        has_physical_columns = False
        if "namespace_identities" in tables:
            columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(namespace_identities)"
                ).fetchall()
            }
            has_physical_columns = {"db_device", "db_inode"} <= columns
            identity_rows = conn.execute(
                "SELECT * FROM namespace_identities"
            ).fetchall()
            physical_paths.extend(str(row["db_path"]) for row in identity_rows)
            if has_physical_columns:
                expected_paths.update(
                    {
                        str(row["db_path"]): (row["db_device"], row["db_inode"])
                        for row in identity_rows
                    }
                )
        if "namespaces" in tables:
            physical_paths.extend(
                str(row["db_path"])
                for row in conn.execute("SELECT db_path FROM namespaces").fetchall()
            )
        validate_namespace_db_paths(physical_paths, expected=expected_paths)
        for physical_path in dict.fromkeys(physical_paths):
            validate_sqlite_sidecars(Path(physical_path))
        current_schema = {"namespace_aliases", "namespace_identities"} <= tables
        version_row = (
            conn.execute(
                "SELECT value FROM registry_meta WHERE key=?",
                (REGISTRY_SCHEMA_VERSION_KEY,),
            ).fetchone()
            if "registry_meta" in tables
            else None
        )
        registry_current = bool(
            version_row and str(version_row["value"]) == str(REGISTRY_SCHEMA_VERSION)
        )
        if "namespace_migrations" in tables:
            migration_columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(namespace_migrations)"
                ).fetchall()
            }
            registry_current = registry_current and {
                "plan_digest", "before_state", "after_state", "backup_path",
                "backup_sha256", "backup_integrity", "undone_at",
                "undo_plan_digest", "undo_backup_path", "undo_backup_sha256",
                "undo_backup_integrity",
            } <= migration_columns
        requires_registry_upgrade = (
            not current_schema or not has_physical_columns or not registry_current
        )
        source = _identity_row(conn, old_display) if current_schema else None
        if source is not None:
            namespace_id: str | None = str(source["namespace_id"])
            canonical_label = str(source["canonical_label"])
            canonical_norm = str(source["canonical_label_norm"])
            db_path = str(source["db_path"])
            target = _identity_row(conn, new_display)
            if target and str(target["namespace_id"]) != namespace_id:
                raise NamespaceCollisionError(
                    f"label {new_display!r} is already mapped to another namespace"
                )
            target_exists = target is not None
            recorded_repository = (
                repository_identity_value
                or (f"path:{repository_path}" if repository_path else None)
            )
            if recorded_repository is None and "repository_bindings" in tables:
                binding = conn.execute(
                    """SELECT repository_identity,repo_path
                       FROM repository_bindings WHERE namespace_id=?
                       ORDER BY repository_identity IS NOT NULL DESC,created_at
                       LIMIT 1""",
                    (namespace_id,),
                ).fetchone()
                if binding:
                    recorded_repository = str(
                        binding["repository_identity"]
                        or f"path:{binding['repo_path']}"
                    )
            if "repository_bindings" in tables and repository_identity_value:
                binding = conn.execute(
                    """SELECT namespace_id FROM repository_bindings
                       WHERE repository_identity=?""",
                    (repository_identity_value,),
                ).fetchone()
                if binding and str(binding["namespace_id"]) != namespace_id:
                    raise NamespaceCollisionError(
                        f"repository {repository_identity_value!r} is already bound "
                        "to another namespace"
                    )
            if "repository_bindings" in tables and repository_path:
                binding = conn.execute(
                    "SELECT namespace_id FROM repository_bindings WHERE repo_path=?",
                    (repository_path,),
                ).fetchone()
                if binding and str(binding["namespace_id"]) != namespace_id:
                    raise NamespaceCollisionError(
                        f"repository path {repository_path!r} is already bound "
                        "to another namespace"
                    )
        elif "namespaces" in tables:
            # A registry can contain the new tables before legacy projection
            # has completed. Planning still reads the authoritative legacy row.
            current_schema = False
            requires_registry_upgrade = True
            canonical, group, legacy_rows = _legacy_namespace_change_source(
                conn, old_display, old_norm
            )
            namespace_id = None
            canonical_label = str(canonical["name"])
            canonical_norm = normalize_namespace_label(canonical_label)
            db_path = str(canonical["db_path"])
            targets = [
                row
                for row in legacy_rows
                if normalize_namespace_label(str(row["name"])) == new_norm
            ]
            if any(str(row["db_path"]) != db_path for row in targets):
                raise NamespaceCollisionError(
                    f"label {new_display!r} is already mapped to another namespace"
                )
            target_exists = bool(targets)
            recorded_repository = (
                repository_identity_value
                or (f"path:{repository_path}" if repository_path else None)
            )
            if recorded_repository is None:
                for row in group:
                    stored = str(row["repo_path"] or "").strip()
                    if not stored:
                        continue
                    stored_remote, stored_path = _repository_context(stored)
                    recorded_repository = stored_remote or (
                        f"path:{stored_path}" if stored_path else None
                    )
                    if recorded_repository:
                        break
            if repository_identity_value or repository_path:
                for row in legacy_rows:
                    if str(row["db_path"]) == db_path:
                        continue
                    stored = str(row["repo_path"] or "").strip()
                    if not stored:
                        continue
                    stored_remote, stored_path = _repository_context(stored)
                    if (
                        repository_identity_value
                        and stored_remote == repository_identity_value
                    ):
                        raise NamespaceCollisionError(
                            f"repository {repository_identity_value!r} is already "
                            "bound to another namespace"
                        )
                    if repository_path and stored_path == repository_path:
                        raise NamespaceCollisionError(
                            f"repository path {repository_path!r} is already bound "
                            "to another namespace"
                        )
        else:
            raise UnknownNamespaceError(old_display)

        target_path = namespaces_dir() / f"{new_display}.db"
        if not target_exists:
            _validate_unmapped_namespace_target(
                target_path, mapped_db_path=Path(db_path)
            )
        report = {
            "action": action,
            "mode": "dry-run",
            "namespace_id": namespace_id,
            "requires_registry_upgrade": requires_registry_upgrade,
            "canonical_before": canonical_label,
            "canonical_after": new_display if action == "rename" else canonical_label,
            "old_label": old_display,
            "old_normalized": old_norm,
            "new_label": new_display,
            "new_normalized": new_norm,
            "repository_identity": recorded_repository,
            "repository_path": repository_path,
            "db_path": db_path,
            "database_operation": "none",
            "idempotent": target_exists
            and (action == "alias" or canonical_norm == new_norm),
            "registry_state_scope": (
                "legacy" if not current_schema else ("pre-v5" if not registry_current else "full")
            ),
        }
        return _finish_plan(
            report,
            _registry_state(
                conn,
                legacy_only=not current_schema,
                compatibility_v4=current_schema and not registry_current,
            ),
        )
    except sqlite3.Error as exc:
        raise NamespaceMigrationError(
            f"cannot read namespace registry safely: {exc}"
        ) from exc
    finally:
        conn.close()


def _change_namespace_label(
    old_label: str,
    new_label: str,
    *,
    repository: str | None = None,
    action: str = "rename",
    apply: bool = False,
    plan_digest: str | None = None,
) -> dict[str, Any]:
    """Plan or atomically apply a namespace alias/rename.

    ``rename`` changes the canonical display label and retains the old label as
    an alias. ``alias`` adds another label without changing the canonical one.
    Neither operation moves, copies, or renames the namespace database.
    """
    if action not in {"alias", "rename"}:
        raise ValueError("action must be 'alias' or 'rename'")
    old_display = safe_name(old_label)
    new_display = safe_name(new_label)
    old_norm = normalize_namespace_label(old_display)
    new_norm = normalize_namespace_label(new_display)
    repo_identity, repo = _repository_context(repository)

    def replay_result(
        conn: sqlite3.Connection, replay: sqlite3.Row, supplied_digest: str
    ) -> dict[str, Any]:
        if not replay["after_state"]:
            raise NamespaceMigrationError(
                "migration has no recorded target state for safe idempotent replay"
            )
        current = _namespace_state(conn, str(replay["namespace_id"]))
        expected = json.loads(str(replay["after_state"]))
        if current != expected:
            raise NamespaceMigrationError(
                "migration replay conflicts with current alias, canonical, legacy, "
                "or repository-binding state"
            )
        return {
            "action": str(replay["action"]),
            "mode": "apply",
            "namespace_id": str(replay["namespace_id"]),
            "migration_id": str(replay["migration_id"]),
            "canonical_after": current["identity"]["canonical_label"],
            "old_label": str(replay["old_label"]),
            "new_label": str(replay["new_label"]),
            "db_path": current["identity"]["db_path"],
            "database_operation": "none",
            "plan_digest": supplied_digest,
            "recorded_plan_digest": replay["plan_digest"],
            "backup": {
                "path": replay["backup_path"],
                "sha256": replay["backup_sha256"],
                "integrity": replay["backup_integrity"],
            },
            "applied": True,
            "idempotent": True,
        }

    if apply and plan_digest and registry_path().is_file():
        replay_conn = _readonly_registry()
        try:
            columns = {
                str(row["name"])
                for row in replay_conn.execute(
                    "PRAGMA table_info(namespace_migrations)"
                ).fetchall()
            }
            if "plan_digest" in columns:
                replay = replay_conn.execute(
                    """SELECT m.*,i.canonical_label,i.db_path
                       FROM namespace_migrations m
                       JOIN namespace_identities i ON i.namespace_id=m.namespace_id
                       WHERE m.plan_digest=? AND m.action=?
                         AND m.old_label_norm=? AND m.new_label_norm=?""",
                    (plan_digest, action, old_norm, new_norm),
                ).fetchone()
                explicit_repo = repo_identity or (f"path:{repo}" if repo else None)
                if replay is not None and (
                    explicit_repo is None
                    or explicit_repo == replay["repository_identity"]
                ):
                    return replay_result(replay_conn, replay, plan_digest)
        finally:
            replay_conn.close()
    plan = _plan_namespace_label_read_only(
        old_display=old_display,
        new_display=new_display,
        old_norm=old_norm,
        new_norm=new_norm,
        repository_identity_value=repo_identity,
        repository_path=repo,
        action=action,
    )
    if not apply:
        return plan
    if not plan_digest:
        raise NamespaceMigrationError(
            "apply requires the plan_digest returned by a preceding dry-run"
        )
    if plan_digest != plan["plan_digest"]:
        raise NamespaceMigrationError(
            "migration plan digest does not match the current registry state and operation"
        )
    if plan["namespace_id"] is not None:
        replay_conn = _readonly_registry()
        try:
            replay = replay_conn.execute(
                """SELECT * FROM namespace_migrations
                   WHERE namespace_id=? AND action=? AND old_label_norm=?
                     AND new_label_norm=? AND repository_key=?""",
                (
                    plan["namespace_id"], action, old_norm, new_norm,
                    plan["repository_identity"] or "",
                ),
            ).fetchone()
            if replay is not None:
                return replay_result(replay_conn, replay, plan_digest)
        finally:
            replay_conn.close()
    backup = _backup_registry(purpose="apply")
    now = now_iso()
    conn: sqlite3.Connection | None = None
    committed = False
    try:
        conn = _registry()
        conn.execute("BEGIN IMMEDIATE")
        current_state = _registry_state(
            conn,
            legacy_only=plan["registry_state_scope"] == "legacy",
            compatibility_v4=plan["registry_state_scope"] == "pre-v5",
        )
        if _state_digest(current_state) != plan["registry_state_digest"]:
            raise NamespaceMigrationError(
                "registry state changed after planning; run dry-run again"
            )
        source = _identity_row(conn, old_display)
        if not source:
            raise UnknownNamespaceError(old_display)
        namespace_id = str(source["namespace_id"])
        previous_canonical_label = str(source["canonical_label"])
        previous_canonical_norm = str(source["canonical_label_norm"])
        recorded_repo_identity = repo_identity or (f"path:{repo}" if repo else None)
        if recorded_repo_identity is None:
            existing_binding = conn.execute(
                """SELECT repository_identity,repo_path FROM repository_bindings
                   WHERE namespace_id=?
                   ORDER BY repository_identity IS NOT NULL DESC, created_at
                   LIMIT 1""",
                (namespace_id,),
            ).fetchone()
            if existing_binding:
                recorded_repo_identity = str(
                    existing_binding["repository_identity"]
                    or f"path:{existing_binding['repo_path']}"
                )
        target = _identity_row(conn, new_display)
        if target and target["namespace_id"] != namespace_id:
            raise NamespaceCollisionError(
                f"label {new_display!r} is already mapped to another namespace"
            )
        target_path = namespaces_dir() / f"{new_display}.db"
        if not target:
            _validate_unmapped_namespace_target(
                target_path, mapped_db_path=Path(str(source["db_path"]))
            )
        if repo_identity:
            binding = conn.execute(
                "SELECT namespace_id FROM repository_bindings WHERE repository_identity=?",
                (repo_identity,),
            ).fetchone()
            if binding and binding["namespace_id"] != namespace_id:
                raise NamespaceCollisionError(
                    f"repository {repo_identity!r} is already bound to another namespace"
                )
        if repo:
            binding = conn.execute(
                "SELECT namespace_id FROM repository_bindings WHERE repo_path=?",
                (repo,),
            ).fetchone()
            if binding and binding["namespace_id"] != namespace_id:
                raise NamespaceCollisionError(
                    f"repository path {repo!r} is already bound to another namespace"
                )
        result: dict[str, Any] = {
            **plan,
            "mode": "apply",
            "namespace_id": namespace_id,
            "canonical_before": previous_canonical_label,
            "canonical_after": new_display if action == "rename" else previous_canonical_label,
            "old_label": old_display,
            "old_normalized": old_norm,
            "new_label": new_display,
            "new_normalized": new_norm,
            "repository_identity": recorded_repo_identity,
            "repository_path": repo,
            "db_path": str(source["db_path"]),
            "database_operation": "none",
            "backup": backup,
        }
        before_state = _namespace_state(conn, namespace_id)
        if not target:
            conn.execute(
                """INSERT INTO namespace_aliases(
                       normalized_label,label,namespace_id,is_canonical,
                       source_alias_norm,created_at
                   ) VALUES (?,?,?,?,?,?)""",
                (
                    new_norm,
                    new_display,
                    namespace_id,
                    0,
                    old_norm if action == "alias" and old_norm != new_norm else None,
                    now,
                ),
            )
        elif new_norm == old_norm and new_display != target["canonical_label"]:
            conn.execute(
                "UPDATE namespace_aliases SET label=? WHERE normalized_label=?",
                (new_display, new_norm),
            )
        if action == "rename":
            conn.execute(
                "UPDATE namespace_aliases SET is_canonical=0 WHERE namespace_id=?",
                (namespace_id,),
            )
            conn.execute(
                "UPDATE namespace_aliases SET is_canonical=1,label=? WHERE normalized_label=?",
                (new_display, new_norm),
            )
            conn.execute(
                """UPDATE namespace_identities
                   SET canonical_label=?,canonical_label_norm=?,updated_at=?
                   WHERE namespace_id=?""",
                (new_display, new_norm, now, namespace_id),
            )
            conn.execute(
                """UPDATE namespace_aliases SET source_alias_norm=NULL
                   WHERE normalized_label IN (?,?)""",
                (old_norm, new_norm),
            )
            legacy_names = [
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM namespaces WHERE db_path=?",
                    (source["db_path"],),
                ).fetchall()
            ]
            has_new_legacy = any(
                normalize_namespace_label(name) == new_norm for name in legacy_names
            )
            for legacy_name in legacy_names:
                if normalize_namespace_label(legacy_name) != previous_canonical_norm:
                    continue
                if has_new_legacy:
                    conn.execute("DELETE FROM namespaces WHERE name=?", (legacy_name,))
                else:
                    conn.execute(
                        "UPDATE namespaces SET name=?,updated_at=? WHERE name=?",
                        (new_display, now, legacy_name),
                    )
                    has_new_legacy = True
            conn.execute(
                """UPDATE repository_bindings SET label_norm=?,updated_at=?
                   WHERE namespace_id=? AND label_norm IN (?,?)""",
                (new_norm, now, namespace_id, old_norm, previous_canonical_norm),
            )
        repo_key = recorded_repo_identity or ""
        _bind_repository(
            conn,
            namespace_id=namespace_id,
            label_norm=new_norm,
            repo_identity=repo_identity,
            repo_path=repo,
            now=now,
        )
        after_state = _namespace_state(conn, namespace_id)
        existing_migration = conn.execute(
            """SELECT * FROM namespace_migrations
               WHERE namespace_id=? AND action=? AND old_label_norm=?
                 AND new_label_norm=? AND repository_key=?""",
            (namespace_id, action, old_norm, new_norm, repo_key),
        ).fetchone()
        migration_id = (
            str(existing_migration["migration_id"])
            if existing_migration is not None
            else new_id()
        )
        if existing_migration is None:
            conn.execute(
                """INSERT INTO namespace_migrations(
                   migration_id,namespace_id,action,old_label,old_label_norm,
                   new_label,new_label_norm,repository_identity,repository_key,applied_at,
                   plan_digest,before_state,after_state,backup_path,backup_sha256,
                   backup_integrity
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    migration_id, namespace_id, action, old_display, old_norm,
                    new_display, new_norm, recorded_repo_identity, repo_key, now,
                    plan_digest, _canonical_json(before_state),
                    _canonical_json(after_state), backup["path"], backup["sha256"],
                    backup["integrity"],
                ),
            )
        backup.verify_for_record()
        conn.commit()
        committed = True
        result["applied"] = True
        result["idempotent"] = existing_migration is not None
        result["migration_id"] = migration_id
        return result
    except Exception:
        if conn is not None:
            conn.rollback()
        if not committed:
            backup.discard()
        raise
    finally:
        if conn is not None:
            conn.close()
        backup.close()


def change_namespace_label(
    old_label: str,
    new_label: str,
    *,
    repository: str | None = None,
    action: str = "rename",
    apply: bool = False,
    plan_digest: str | None = None,
) -> dict[str, Any]:
    """Plan or apply one serialized namespace label migration."""
    if not apply:
        return _change_namespace_label(
            old_label, new_label, repository=repository, action=action
        )
    if not plan_digest:
        raise NamespaceMigrationError(
            "apply requires the plan_digest returned by a preceding dry-run"
        )
    with _namespace_migration_lock():
        return _change_namespace_label(
            old_label,
            new_label,
            repository=repository,
            action=action,
            apply=True,
            plan_digest=plan_digest,
        )


def _insert_dict_row(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    columns = list(row)
    placeholders = ",".join("?" for _ in columns)
    conn.execute(
        f"INSERT INTO {table}({','.join(columns)}) VALUES ({placeholders})",
        tuple(row[column] for column in columns),
    )


def _restore_namespace_state(
    conn: sqlite3.Connection, namespace_id: str, state: dict[str, Any]
) -> None:
    identity = state["identity"]
    current = conn.execute(
        "SELECT db_path FROM namespace_identities WHERE namespace_id=?",
        (namespace_id,),
    ).fetchone()
    if current is None:
        raise NamespaceMigrationError("namespace identity no longer exists")
    db_path = str(current["db_path"])
    conn.execute("DELETE FROM repository_bindings WHERE namespace_id=?", (namespace_id,))
    conn.execute("DELETE FROM namespace_aliases WHERE namespace_id=?", (namespace_id,))
    columns = [column for column in identity if column != "namespace_id"]
    conn.execute(
        "UPDATE namespace_identities SET "
        + ",".join(f"{column}=?" for column in columns)
        + " WHERE namespace_id=?",
        (*[identity[column] for column in columns], namespace_id),
    )
    conn.execute("DELETE FROM namespaces WHERE db_path=?", (db_path,))
    for row in state["aliases"]:
        _insert_dict_row(conn, "namespace_aliases", row)
    for row in state["bindings"]:
        _insert_dict_row(conn, "repository_bindings", row)
    for row in state["legacy"]:
        _insert_dict_row(conn, "namespaces", row)


def _plan_namespace_undo_read_only(migration_id: str) -> dict[str, Any]:
    try:
        conn = _readonly_registry()
    except FileNotFoundError as exc:
        raise NamespaceMigrationError("namespace registry does not exist") from exc
    try:
        columns = {
            str(info["name"])
            for info in conn.execute(
                "PRAGMA table_info(namespace_migrations)"
            ).fetchall()
        }
        if not {"before_state", "after_state", "undone_at"} <= columns:
            raise NamespaceMigrationError(
                "registry predates reversible migration history; upgrade it before undo"
            )
        row = conn.execute(
            "SELECT * FROM namespace_migrations WHERE migration_id=?",
            (migration_id,),
        ).fetchone()
        if row is None:
            raise NamespaceMigrationError(f"unknown namespace migration: {migration_id}")
        if not row["before_state"] or not row["after_state"]:
            raise NamespaceMigrationError(
                "migration predates reversible history and cannot be undone safely"
            )
        before_state = json.loads(str(row["before_state"]))
        after_state = json.loads(str(row["after_state"]))
        current_state = _namespace_state(conn, str(row["namespace_id"]))
        undone = row["undone_at"] is not None
        expected = before_state if undone else after_state
        if current_state != expected:
            if not undone:
                raise NamespaceMigrationError(
                    "migration cannot be undone because aliases, repository bindings, "
                    "or canonical state changed; a retired alias cannot be reconstructed "
                    "without an exact recorded state"
                )
            raise NamespaceMigrationError(
                "undone migration state has drifted from its recorded result"
            )
        report: dict[str, Any] = {
            "action": "undo",
            "mode": "dry-run",
            "migration_id": migration_id,
            "namespace_id": str(row["namespace_id"]),
            "canonical_before": current_state["identity"]["canonical_label"],
            "canonical_after": before_state["identity"]["canonical_label"],
            "database_operation": "none",
            "idempotent": undone,
            "registry_state_scope": "full",
        }
        return _finish_plan(report, _registry_state(conn))
    finally:
        conn.close()


def _undo_namespace_migration(
    migration_id: str, *, apply: bool = False, plan_digest: str | None = None
) -> dict[str, Any]:
    """Plan or reverse one recorded namespace migration without moving its DB."""
    migration_id = migration_id.strip()
    if not migration_id:
        raise NamespaceMigrationError("migration_id is required")

    def replay_result(
        conn: sqlite3.Connection, replay: sqlite3.Row, supplied_digest: str
    ) -> dict[str, Any]:
        if not replay["before_state"]:
            raise NamespaceMigrationError(
                "migration has no recorded source state for safe undo replay"
            )
        current = _namespace_state(conn, str(replay["namespace_id"]))
        expected = json.loads(str(replay["before_state"]))
        if current != expected:
            raise NamespaceMigrationError(
                "undo replay conflicts with current alias, canonical, legacy, "
                "or repository-binding state"
            )
        return {
            "action": "undo",
            "mode": "apply",
            "migration_id": migration_id,
            "namespace_id": str(replay["namespace_id"]),
            "applied": True,
            "idempotent": True,
            "database_operation": "none",
            "plan_digest": supplied_digest,
            "recorded_plan_digest": replay["undo_plan_digest"],
            "backup": {
                "path": replay["undo_backup_path"],
                "sha256": replay["undo_backup_sha256"],
                "integrity": replay["undo_backup_integrity"],
            },
        }

    if apply and plan_digest:
        conn = _readonly_registry()
        try:
            replay = conn.execute(
                "SELECT * FROM namespace_migrations WHERE migration_id=?",
                (migration_id,),
            ).fetchone()
            if (
                replay is not None
                and replay["undone_at"] is not None
                and replay["undo_plan_digest"] == plan_digest
            ):
                return replay_result(conn, replay, plan_digest)
        finally:
            conn.close()
    plan = _plan_namespace_undo_read_only(migration_id)
    if not apply:
        return plan
    if not plan_digest:
        raise NamespaceMigrationError(
            "undo apply requires the plan_digest returned by a preceding dry-run"
        )
    if plan_digest != plan["plan_digest"]:
        raise NamespaceMigrationError(
            "undo plan digest does not match the current registry state"
        )
    if plan["idempotent"]:
        conn = _readonly_registry()
        try:
            replay = conn.execute(
                "SELECT * FROM namespace_migrations WHERE migration_id=?",
                (migration_id,),
            ).fetchone()
            if replay is None or replay["undone_at"] is None:
                raise NamespaceMigrationError(
                    "undo replay state changed after planning"
                )
            return replay_result(conn, replay, plan_digest)
        finally:
            conn.close()
    backup = _backup_registry(purpose="undo")
    conn: sqlite3.Connection | None = None
    committed = False
    try:
        conn = _registry()
        conn.execute("BEGIN IMMEDIATE")
        if _state_digest(_registry_state(conn)) != plan["registry_state_digest"]:
            raise NamespaceMigrationError(
                "registry state changed after undo planning; run dry-run again"
            )
        row = conn.execute(
            "SELECT * FROM namespace_migrations WHERE migration_id=?",
            (migration_id,),
        ).fetchone()
        if row is None or not row["before_state"] or not row["after_state"]:
            raise NamespaceMigrationError("migration has no reversible state")
        current = _namespace_state(conn, str(row["namespace_id"]))
        after_state = json.loads(str(row["after_state"]))
        if current != after_state:
            raise NamespaceMigrationError(
                "migration state changed after planning and cannot be undone"
            )
        before_state = json.loads(str(row["before_state"]))
        _restore_namespace_state(conn, str(row["namespace_id"]), before_state)
        undone_at = now_iso()
        conn.execute(
            """UPDATE namespace_migrations
               SET undone_at=?,undo_plan_digest=?,undo_backup_path=?,
                   undo_backup_sha256=?,undo_backup_integrity=?
               WHERE migration_id=?""",
            (
                undone_at, plan_digest, backup["path"], backup["sha256"],
                backup["integrity"], migration_id,
            ),
        )
        backup.verify_for_record()
        conn.commit()
        committed = True
        return {
            **plan,
            "mode": "apply",
            "applied": True,
            "idempotent": False,
            "backup": backup,
            "undone_at": undone_at,
        }
    except Exception:
        if conn is not None:
            conn.rollback()
        if not committed:
            backup.discard()
        raise
    finally:
        if conn is not None:
            conn.close()
        backup.close()


def undo_namespace_migration(
    migration_id: str, *, apply: bool = False, plan_digest: str | None = None
) -> dict[str, Any]:
    if not apply:
        return _undo_namespace_migration(migration_id)
    if not plan_digest:
        raise NamespaceMigrationError(
            "undo apply requires the plan_digest returned by a preceding dry-run"
        )
    with _namespace_migration_lock():
        return _undo_namespace_migration(
            migration_id, apply=True, plan_digest=plan_digest
        )


def retire_namespace_alias(label: str, *, apply: bool = False) -> dict[str, Any]:
    """Check registry-owned references and optionally retire a noncanonical alias."""
    display = safe_name(label)
    norm = normalize_namespace_label(display)
    conn = _registry()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT a.*,i.canonical_label,i.db_path
               FROM namespace_aliases a
               JOIN namespace_identities i ON i.namespace_id=a.namespace_id
               WHERE a.normalized_label=?""",
            (norm,),
        ).fetchone()
        if not row:
            raise UnknownNamespaceError(display)
        blockers: list[dict[str, str]] = []
        if row["is_canonical"] or normalize_namespace_label(str(row["canonical_label"])) == norm:
            blockers.append({"kind": "canonical-label", "label": str(row["canonical_label"])})
        for binding in conn.execute(
            """SELECT repository_identity,repo_path FROM repository_bindings
               WHERE namespace_id=? AND label_norm=?""",
            (row["namespace_id"], norm),
        ).fetchall():
            blockers.append(
                {
                    "kind": "repository-binding",
                    "reference": str(binding["repository_identity"] or binding["repo_path"]),
                }
            )
        for alias in conn.execute(
            "SELECT label FROM namespace_aliases WHERE source_alias_norm=?",
            (norm,),
        ).fetchall():
            blockers.append({"kind": "dependent-alias", "label": str(alias["label"])})
        result: dict[str, Any] = {
            "mode": "apply" if apply else "dry-run",
            "label": str(row["label"]),
            "normalized_label": norm,
            "namespace_id": str(row["namespace_id"]),
            "canonical_label": str(row["canonical_label"]),
            "db_path": str(row["db_path"]),
            "safe": not blockers,
            "blockers": blockers,
            "operator_caveat": (
                "External editor/host configuration is not recorded in the registry; "
                "inspect and update it before retiring this alias."
            ),
        }
        if blockers:
            conn.rollback()
            if apply:
                kinds = ", ".join(blocker["kind"] for blocker in blockers)
                raise AliasRetirementError(
                    f"cannot retire alias {display!r}; registry references: {kinds}"
                )
            return result
        if apply:
            for legacy in conn.execute(
                "SELECT name FROM namespaces WHERE db_path=?",
                (row["db_path"],),
            ).fetchall():
                legacy_name = str(legacy["name"])
                if normalize_namespace_label(legacy_name) == norm:
                    conn.execute("DELETE FROM namespaces WHERE name=?", (legacy_name,))
            conn.execute(
                "DELETE FROM namespace_aliases WHERE normalized_label=?",
                (norm,),
            )
            conn.commit()
            _forget_registered_alias(display)
            result["retired"] = True
        else:
            conn.rollback()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def verbatim_text(
    content: str = "",
    tool_name: str | None = None,
    tool_input: str | None = None,
    tool_output: str | None = None,
) -> str:
    """Concatenate stored fields as-is. Not a summary."""
    parts: list[str] = []
    if content:
        parts.append(content)
    if tool_name:
        parts.append(f"tool:{tool_name}")
    if tool_input:
        parts.append(tool_input)
    if tool_output:
        parts.append(tool_output)
    return "\n".join(parts)


def _correction_request_payload(
    memory_id: str,
    replacement: str | None,
    reason: str | None,
) -> bytes:
    """Length-prefix exact UTF-8 request fields; null and empty stay distinct."""
    parts: list[bytes] = []
    for value in (memory_id, replacement, reason):
        if value is None:
            parts.append(b"\x00")
            continue
        raw = value.encode("utf-8")
        parts.append(b"\x01" + struct.pack(">Q", len(raw)) + raw)
    return b"".join(parts)


def _correction_request_identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _erasure_context_values(*raw_values: object) -> set[str]:
    """Collect textual erasure markers, including structured event metadata.

    Keep target event fields routed through this hook so future provenance
    fields can extend purge sanitization without scattering privacy logic.
    Target metadata keys may themselves contain user-controlled private bytes,
    so both mapping keys and values participate in session sanitization.
    """
    values: set[str] = set()
    not_json = object()

    def add(value: object) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                add(key)
                add(child)
            return
        if isinstance(value, (list, tuple, set)):
            for child in value:
                add(child)
            return
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
            encoded = base64.b64encode(raw).decode("ascii")
            envelope = {"encoding": "base64", "data": encoded}
            values.update(
                {
                    raw.hex(),
                    encoded,
                    f"<sqlite-blob base64:{encoded}>",
                    json.dumps(envelope, ensure_ascii=False),
                    json.dumps(
                        envelope,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    encode_json_safe_sqlite_key(raw),
                }
            )
            try:
                add(raw.decode("utf-8"))
            except UnicodeDecodeError:
                pass
            return
        text = str(value)
        if not text or text in values:
            return
        values.add(text)
        if isinstance(value, str):
            parsed = loads(value, default=not_json)
            if parsed is not not_json and parsed != value:
                add(parsed)

    for raw_value in raw_values:
        add(raw_value)
    return values


def _provenance_erasure_values(raw: object) -> tuple[object, ...]:
    """Return provenance values without treating fixed schema keys as secrets."""
    parsed = loads(raw if isinstance(raw, str) else None, default={})
    if isinstance(parsed, dict):
        return tuple(parsed.values())
    return (raw,)


@dataclass
class ObserveResult:
    event_id: str
    memory_id: str
    session_id: str
    namespace: str
    tier: str
    entities: list[str] = field(default_factory=list)
    embedded: bool = False
    embedding_queued: bool = False
    deduplicated: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)


class Store:
    def __init__(self, name: str, repo_path: str | None = None, *, create: bool = True):
        requested = safe_name(name)
        if create:
            register_namespace(requested, repo_path)
        identity = None
        attempts = 8 if create else 1
        last_error: NamespacePathError | None = None
        for _attempt in range(attempts):
            try:
                identity = resolve_namespace_identity(requested)
                last_error = None
            except NamespacePathError as exc:
                if not create or not _is_concurrent_registry_change(exc):
                    raise
                last_error = exc
                continue
            if identity is not None:
                break
        if last_error is not None:
            raise last_error
        if not identity:
            raise UnknownNamespaceError(requested)
        self._initialize_identity(identity)

    @classmethod
    def _from_identity(cls, identity: dict[str, Any]) -> "Store":
        """Open an already-resolved stable identity without resolving a label."""
        store = cls.__new__(cls)
        store._initialize_identity(identity)
        return store

    def _initialize_identity(self, identity: dict[str, Any]) -> None:
        self.namespace_id = str(identity["namespace_id"])
        self.name = str(identity["canonical_label"])
        self.db_path = Path(str(identity["db_path"]))
        self._privacy_purge_thread_id: int | None = None
        self.conn = _open_mapped_namespace_db(
            self.db_path,
            expected=(identity.get("db_device"), identity.get("db_inode")),
        )
        self.conn.create_function(
            "haunt_privacy_purge_authorized",
            0,
            lambda: int(
                self._privacy_purge_thread_id == get_ident()
                and self.conn.in_transaction
            ),
        )
        _ensure_namespace_schema(self.conn)
        self._ensure_graph_evidence()

    def close(self) -> None:
        with _NAMESPACE_DB_HANDLE_LOCK:
            self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_graph_evidence(self) -> None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='graph_evidence_version'"
        ).fetchone()
        if row and str(row["value"]) == "1":
            return
        self.rebuild_graph(touch=False)

    def vec_ok(self) -> bool:
        return _vec_loaded(self.conn)

    def vec_version(self) -> str | None:
        if not self.vec_ok():
            return None
        return str(self.conn.execute("SELECT vec_version()").fetchone()[0])

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def ensure_session(
        self,
        session_id: str | None = None,
        source: str = "cli",
        meta: dict[str, Any] | None = None,
        *,
        commit: bool = True,
    ) -> str:
        if session_id:
            row = self.conn.execute(
                "SELECT id, ended_at FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if not row:
                self.conn.execute(
                    "INSERT INTO sessions(id, started_at, ended_at, source, meta) VALUES (?,?,?,?,?)",
                    (session_id, now_iso(), None, source, dumps(meta or {})),
                )
                if commit:
                    self.conn.commit()
            return session_id
        current = self.get_meta("current_session")
        if current:
            row = self.conn.execute(
                "SELECT id, ended_at FROM sessions WHERE id=?", (current,)
            ).fetchone()
            if row and not row["ended_at"]:
                return current
        sid = new_id()
        self.conn.execute(
            "INSERT INTO sessions(id, started_at, ended_at, source, meta) VALUES (?,?,?,?,?)",
            (sid, now_iso(), None, source, dumps(meta or {})),
        )
        if commit:
            self.set_meta("current_session", sid)
        else:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("current_session", sid),
            )
        return sid

    def end_session(self, session_id: str | None = None) -> dict[str, Any]:
        """Close an open session. Returns ok=False if nothing was ended."""
        sid = session_id or self.get_meta("current_session")
        if not sid:
            return {"ok": False, "error": "no open session"}
        cur = self.conn.execute(
            "UPDATE sessions SET ended_at=? WHERE id=? AND ended_at IS NULL",
            (now_iso(), sid),
        )
        if cur.rowcount == 0:
            row = self.conn.execute(
                "SELECT ended_at FROM sessions WHERE id=?", (sid,)
            ).fetchone()
            if not row:
                return {
                    "ok": False,
                    "session_id": sid,
                    "error": f"session {sid} not found",
                }
            return {
                "ok": False,
                "session_id": sid,
                "error": f"session {sid} already ended",
            }
        if self.get_meta("current_session") == sid:
            self.conn.execute("DELETE FROM meta WHERE key='current_session'")
        self.conn.commit()
        return {"ok": True, "session_id": sid}

    def observe(
        self,
        content: str = "",
        *,
        role: str = "user",
        tier: str = "episodic",
        session_id: str | None = None,
        tool_name: str | None = None,
        tool_input: str | None = None,
        tool_output: str | None = None,
        producer_call_id: str | None = None,
        event_time: str | None = None,
        origin: str = "python",
        channel: str = "python",
        meta: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        idempotency_key: str | None = None,
        defer_embedding: bool = False,
        commit: bool = True,
    ) -> ObserveResult:
        # Provenance is validated before sessions, events, embedding jobs, or
        # graph/index projections can be written.
        canonical_provenance = validate_provenance(
            provenance,
            origin=origin,
            channel=channel,
            tool_name=tool_name,
            producer_call_id=producer_call_id,
        )
        encoded_provenance = provenance_json(canonical_provenance)
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}")
        text = verbatim_text(content, tool_name, tool_input, tool_output)
        idem = (idempotency_key or "").strip() or None
        if idem and len(idem) > 512:
            raise ValueError("idempotency_key must be 512 characters or fewer")
        if idem:
            existing = self._observe_by_idempotency_key(idem, text, encoded_provenance)
            if existing is not None:
                return existing
        if commit and not defer_embedding:
            self.ensure_current_embeddings()
            self.process_embedding_jobs(limit=32)
        try:
            sid = self.ensure_session(session_id, source=origin, commit=False)
            et = iso_or_now(event_time)
            ts = now_iso()
            vf = iso_or_now(valid_from) if valid_from else et
            vt = iso_or_now(valid_to) if valid_to else None
            event_id = new_id()
            memory_id = new_id()
            self.conn.execute(
                """
                INSERT INTO events(
                    id, idempotency_key, session_id, ts, event_time, role, content,
                    tool_name, tool_input, tool_output, origin, tier, meta,
                    provenance
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id,
                    idem,
                    sid,
                    ts,
                    et,
                    role,
                    content or "",
                    tool_name,
                    tool_input,
                    tool_output,
                    origin,
                    tier,
                    dumps(meta or {}),
                    encoded_provenance,
                ),
            )
            blob = None
            embedded = False
            vec = (
                None if defer_embedding else (embed_one(text) if text.strip() else None)
            )
            if vec is not None:
                blob = sqlite_vec.serialize_float32(vec)
                ensure_vec_table(self.conn, len(vec), commit=False)
                es = embed_state()
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("embed_model", es.model_id),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("embed_dim", str(len(vec))),
                )
            self.conn.execute(
                """
                INSERT INTO memories(
                    id, event_id, tier, content, embedding, valid_from, valid_to, created_at
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (memory_id, event_id, tier, text, blob, vf, vt, ts),
            )
            self.conn.execute(
                "INSERT INTO memories_fts(id, content) VALUES (?, ?)",
                (memory_id, text),
            )
            embedding_queued = bool(blob is None and text.strip())
            if embedding_queued:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO embedding_jobs(memory_id, queued_at)
                    VALUES (?, ?)
                    """,
                    (memory_id, ts),
                )
            if blob is not None and _vec_loaded(self.conn):
                try:
                    self.conn.execute(
                        "INSERT INTO vec_memories(id, embedding) VALUES (?, ?)",
                        (memory_id, blob),
                    )
                    embedded = True
                except sqlite3.Error:
                    pass
            from haunt.graph import extract_and_store

            entity_names = extract_and_store(
                self.conn, event_id, text, et, tool_name, commit=False
            )
            if commit:
                self.conn.commit()
        except sqlite3.IntegrityError:
            self.conn.rollback()
            if idem:
                existing = self._observe_by_idempotency_key(
                    idem, text, encoded_provenance
                )
                if existing is not None:
                    return existing
            raise
        except Exception:
            self.conn.rollback()
            raise
        if commit:
            try:
                touch_namespace(self.name, namespace_id=self.namespace_id)
            except Exception:
                pass
        return ObserveResult(
            event_id=event_id,
            memory_id=memory_id,
            session_id=sid,
            namespace=self.name,
            tier=tier,
            entities=entity_names,
            embedded=embedded,
            embedding_queued=embedding_queued,
            provenance=canonical_provenance,
        )

    def _observe_by_idempotency_key(
        self,
        key: str,
        expected_text: str,
        expected_provenance: str,
    ) -> ObserveResult | None:
        row = self.conn.execute(
            """
            SELECT e.id AS event_id, e.session_id, e.tier,
                   m.id AS memory_id, m.content, m.embedding, e.provenance,
                   e.origin, e.meta, e.tool_name
            FROM events e
            JOIN memories m ON m.event_id=e.id
            WHERE e.idempotency_key=?
            ORDER BY m.rowid ASC
            LIMIT 1
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        if row["content"] != expected_text:
            raise ValueError("idempotency_key was reused with different content")
        if row["provenance"] is None:
            raise ValueError("idempotency_key replay cannot verify legacy provenance")
        stored_provenance = public_provenance(
            row["provenance"],
            origin=row["origin"],
            legacy_meta=row["meta"],
            tool_name=row["tool_name"],
        )
        if stored_provenance.get("kind") == "invalid_stored":
            raise ValueError(
                "idempotency_key replay cannot verify invalid stored provenance"
            )
        # A retry cannot silently replace source attribution. This remains
        # compatible with new native calls because their canonical envelope is
        # deterministic from the same observe inputs.
        if row["provenance"] is not None and row["provenance"] != expected_provenance:
            raise ValueError("idempotency_key was reused with different provenance")
        entities = [
            str(r["name"])
            for r in self.conn.execute(
                """
                SELECT e.name
                FROM entity_mentions em
                JOIN entities e ON e.id=em.entity_id
                WHERE em.event_id=?
                ORDER BY e.name
                """,
                (row["event_id"],),
            ).fetchall()
        ]
        return ObserveResult(
            event_id=row["event_id"],
            memory_id=row["memory_id"],
            session_id=row["session_id"],
            namespace=self.name,
            tier=row["tier"],
            entities=entities,
            embedded=row["embedding"] is not None,
            embedding_queued=self.conn.execute(
                "SELECT 1 FROM embedding_jobs WHERE memory_id=?",
                (row["memory_id"],),
            ).fetchone()
            is not None,
            deduplicated=True,
            provenance=stored_provenance,
        )

    def process_embedding_jobs(self, *, limit: int = 64) -> dict[str, Any]:
        """Embed queued hook writes in a persistent, model-owning process."""
        cap = clamp_limit(limit, default=64)
        queued = self.conn.execute(
            """
            SELECT j.memory_id, m.content
            FROM embedding_jobs j
            JOIN memories m ON m.id=j.memory_id
            ORDER BY j.queued_at ASC, j.rowid ASC
            LIMIT ?
            """,
            (cap,),
        ).fetchall()
        if not queued:
            return {"queued": 0, "processed": 0, "failed": 0}
        es = embed_state()
        if not es.available:
            return {
                "queued": len(queued),
                "processed": 0,
                "failed": 0,
                "available": False,
            }
        try:
            vectors = embed_texts(
                [row["content"] if row["content"] else " " for row in queued]
            )
        except Exception as exc:
            message = str(exc)[:1000]
            self.conn.executemany(
                """
                UPDATE embedding_jobs
                SET attempts=attempts+1, last_error=? WHERE memory_id=?
                """,
                [(message, row["memory_id"]) for row in queued],
            )
            self.conn.commit()
            return {
                "queued": len(queued),
                "processed": 0,
                "failed": len(queued),
                "error": message,
            }
        if not vectors:
            message = "embedding backend returned no vectors"
            self.conn.executemany(
                """
                UPDATE embedding_jobs
                SET attempts=attempts+1, last_error=? WHERE memory_id=?
                """,
                [(message, row["memory_id"]) for row in queued],
            )
            self.conn.commit()
            return {
                "queued": len(queued),
                "processed": 0,
                "failed": len(queued),
                "error": message,
            }

        ensure_vec_table(self.conn, es.dim, commit=False)
        processed = 0
        failed = 0
        for row, vec in zip(queued, vectors):
            memory_id = row["memory_id"]
            try:
                if len(vec) != es.dim:
                    raise ValueError(
                        f"embedding backend returned dimension {len(vec)}; "
                        f"expected {es.dim}"
                    )
                blob = sqlite_vec.serialize_float32(vec)
                self.conn.execute(
                    "UPDATE memories SET embedding=? WHERE id=?",
                    (blob, memory_id),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO vec_memories(id, embedding) VALUES (?, ?)",
                    (memory_id, blob),
                )
                self.conn.execute(
                    "DELETE FROM embedding_jobs WHERE memory_id=?",
                    (memory_id,),
                )
                processed += 1
            except (sqlite3.Error, TypeError, ValueError) as exc:
                self.conn.execute(
                    """
                    UPDATE embedding_jobs
                    SET attempts=attempts+1, last_error=? WHERE memory_id=?
                    """,
                    (str(exc)[:1000], memory_id),
                )
                failed += 1
        if len(vectors) < len(queued):
            missing = queued[len(vectors) :]
            message = "embedding backend returned fewer vectors than inputs"
            self.conn.executemany(
                """
                UPDATE embedding_jobs
                SET attempts=attempts+1, last_error=? WHERE memory_id=?
                """,
                [(message, row["memory_id"]) for row in missing],
            )
            failed += len(missing)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_model', ?)",
            (es.model_id,),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_dim', ?)",
            (str(es.dim),),
        )
        self.conn.commit()
        return {
            "queued": len(queued),
            "processed": processed,
            "failed": failed,
            "available": True,
        }

    def embeddings_stale(self) -> bool:
        """True when stored vectors do not match the currently loaded model."""
        es = embed_state()
        if not es.available:
            return False
        stored_dim = self.get_meta("embed_dim")
        stored_model = self.get_meta("embed_model")
        row = self.conn.execute(
            "SELECT embedding FROM memories WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        if row and row["embedding"]:
            n = len(row["embedding"]) // 4
            if n != es.dim:
                return True
        if stored_dim and int(stored_dim) != es.dim:
            return True
        if stored_model and stored_model != es.model_id:
            return True
        return False

    def reembed(self) -> dict[str, Any]:
        """Rebuild every memory embedding with the currently loaded model.

        ``updated`` is the number of rows that actually landed in
        ``vec_memories``, not blob writes to ``memories.embedding``.
        """
        es = embed_state()
        rows = self.conn.execute("SELECT id, content FROM memories").fetchall()
        self.conn.execute("DROP TABLE IF EXISTS vec_memories")
        if not es.available:
            self.conn.execute("UPDATE memories SET embedding=NULL")
            self.conn.execute(
                """
                INSERT OR IGNORE INTO embedding_jobs(memory_id, queued_at)
                SELECT id, created_at FROM memories WHERE TRIM(content) != ''
                """
            )
            self.conn.commit()
            return {
                "updated": 0,
                "total": len(rows),
                "model": es.model_id,
                "dim": es.dim,
                "available": False,
            }
        ensure_vec_table(self.conn, es.dim)
        ids = [r["id"] for r in rows]
        texts = [r["content"] if r["content"] else " " for r in rows]
        updated = 0
        chunk = 16
        for i in range(0, len(texts), chunk):
            vecs = embed_texts(texts[i : i + chunk])
            if not vecs:
                continue
            for mid, vec in zip(ids[i : i + chunk], vecs):
                blob = sqlite_vec.serialize_float32(vec)
                self.conn.execute(
                    "UPDATE memories SET embedding=? WHERE id=?", (blob, mid)
                )
                if self.vec_ok():
                    try:
                        self.conn.execute(
                            "INSERT INTO vec_memories(id, embedding) VALUES (?, ?)",
                            (mid, blob),
                        )
                        self.conn.execute(
                            "DELETE FROM embedding_jobs WHERE memory_id=?",
                            (mid,),
                        )
                        updated += 1
                    except sqlite3.Error:
                        pass
        self.set_meta("embed_model", es.model_id)
        self.set_meta("embed_dim", str(es.dim))
        self.conn.commit()
        return {
            "updated": updated,
            "total": len(rows),
            "model": es.model_id,
            "dim": es.dim,
            "available": True,
        }

    def ensure_current_embeddings(self) -> dict[str, Any] | None:
        """Rebuild vectors if the loaded model/dim does not match this DB."""
        if self.embeddings_stale():
            return self.reembed()
        return None

    def events(
        self,
        *,
        session_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        clock: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        col = clock_sql_column(clock, qualified=False)
        sql = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if session_id:
            sql += " AND session_id=?"
            params.append(session_id)
        if since:
            sql += f" AND {col}>=?"
            params.append(iso_or_now(since))
        if until:
            sql += f" AND {col}<=?"
            params.append(iso_or_now(until))
        if normalize_clock(clock) == "storage_time":
            sql += " ORDER BY ts DESC, event_time DESC, rowid DESC LIMIT ? OFFSET ?"
        else:
            sql += " ORDER BY event_time DESC, ts DESC, rowid DESC LIMIT ? OFFSET ?"
        params.append(clamp_limit(limit, default=100))
        try:
            off = int(offset)
        except (TypeError, ValueError):
            off = 0
        params.append(max(0, off))
        out: list[dict[str, Any]] = []
        for row in self.conn.execute(sql, params).fetchall():
            event = dict(row)
            event["provenance"] = public_provenance(
                event.get("provenance"),
                origin=event.get("origin"),
                legacy_meta=event.get("meta"),
                tool_name=event.get("tool_name"),
            )
            out.append(json_safe_sqlite(event))
        return out

    def stats(self) -> dict[str, Any]:
        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        tiers = {
            r["tier"]: r["n"]
            for r in self.conn.execute(
                "SELECT tier, COUNT(*) AS n FROM events GROUP BY tier"
            )
        }
        last = self.conn.execute(
            "SELECT ts, event_time FROM events ORDER BY ts DESC, rowid DESC LIMIT 1"
        ).fetchone()
        db = Path(self.db_path)
        size = db.stat().st_size if db.exists() else 0
        wal = db.with_suffix(db.suffix + "-wal")
        if wal.exists():
            size += wal.stat().st_size
        return json_safe_sqlite(
            {
                "namespace": self.name,
                "db_path": str(db.resolve()),
                "db_size_bytes": size,
                "events": count("events"),
                "memories": count("memories"),
                "sessions": count("sessions"),
                "entities": count("entities"),
                "relations": count("relations"),
                "embedding_jobs": count("embedding_jobs"),
                "corrections": count("corrections"),
                "lineage_tombstones": count("lineage_tombstones"),
                "tiers": tiers,
                "last_write": last["ts"] if last else None,
                "last_event_time": last["event_time"] if last else None,
                "wal": True,
            }
        )

    def top_entities(
        self,
        limit: int = 15,
        *,
        trusted_only: bool = False,
    ) -> list[dict[str, Any]]:
        trusted_clause = ""
        if trusted_only:
            trusted_clause = """
            WHERE EXISTS (
                SELECT 1
                FROM entity_mentions em
                JOIN events ev ON ev.id=em.event_id
                WHERE em.entity_id=e.id
                  AND ev.role != 'tool'
                  AND ev.tool_name IS NULL
            )
            """
        rows = self.conn.execute(
            f"""
            SELECT e.id, e.name, e.type, e.norm_name, e.first_seen, e.last_seen,
                   (SELECT COUNT(*) FROM relations r
                    WHERE r.src_entity=e.id OR r.dst_entity=e.id) AS rels
            FROM entities e
            {trusted_clause}
            ORDER BY e.last_seen DESC
            LIMIT ?
            """,
            (clamp_limit(limit, default=15),),
        ).fetchall()
        return json_safe_sqlite([dict(r) for r in rows])

    def graph(self, entity: str | None = None) -> dict[str, Any]:
        if entity:
            norm = entity.strip().lower()
            ents = [
                dict(r)
                for r in self.conn.execute(
                    "SELECT * FROM entities WHERE norm_name LIKE ? OR name LIKE ? OR id=?",
                    (f"%{norm}%", f"%{entity}%", entity),
                )
            ]
            ids = [e["id"] for e in ents]
            rels: list[dict[str, Any]] = []
            if ids:
                placeholders = ",".join("?" * len(ids))
                rels = [
                    dict(r)
                    for r in self.conn.execute(
                        f"SELECT * FROM relations WHERE src_entity IN ({placeholders}) OR dst_entity IN ({placeholders})",
                        ids + ids,
                    )
                ]
            return json_safe_sqlite({"entities": ents, "relations": rels})
        return json_safe_sqlite(
            {
                "entities": [
                    dict(r)
                    for r in self.conn.execute(
                        "SELECT * FROM entities ORDER BY last_seen DESC LIMIT 200"
                    )
                ],
                "relations": [
                    dict(r)
                    for r in self.conn.execute(
                        "SELECT * FROM relations ORDER BY valid_from DESC LIMIT 400"
                    )
                ],
            }
        )

    def rebuild_graph(self, *, touch: bool = True) -> dict[str, Any]:
        """Rebuild graph evidence and derived aggregates from stored events."""
        from haunt.graph import extract_and_store

        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        before_ents = count("entities")
        before_rels = count("relations")
        events_n = count("events")
        memories_n = count("memories")

        try:
            self.conn.execute("DELETE FROM relation_evidence")
            self.conn.execute("DELETE FROM entity_mentions")
            self.conn.execute("DELETE FROM relations")
            self.conn.execute("DELETE FROM entities")

            rows = self.conn.execute(
                """
                SELECT id, content, tool_name, tool_input, tool_output, event_time
                FROM events
                ORDER BY event_time ASC, ts ASC, rowid ASC
                """
            ).fetchall()
            for r in rows:
                text = verbatim_text(
                    r["content"] or "",
                    r["tool_name"],
                    r["tool_input"],
                    r["tool_output"],
                )
                extract_and_store(
                    self.conn,
                    r["id"],
                    text,
                    r["event_time"],
                    r["tool_name"],
                    commit=False,
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("graph_evidence_version", "1"),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        if touch:
            try:
                touch_namespace(self.name, namespace_id=self.namespace_id)
            except Exception:
                pass

        return {
            "events": events_n,
            "memories": memories_n,
            "entities_before": before_ents,
            "relations_before": before_rels,
            "entities": count("entities"),
            "relations": count("relations"),
        }

    # ------------------------------------------------------------------
    # purge: hard-delete a memory and its entire provenance chain
    # ------------------------------------------------------------------

    def purge(self, memory_id: str) -> dict[str, Any]:
        """Hard-delete a memory and clean up all associated data.

        Removes: memory row, FTS row, vec row, graph rows tied to the
        memory's event, and the event itself if no other memories reference it.
        Returns a report of what was deleted.
        """
        row = self.conn.execute(
            """
            SELECT m.id, m.event_id, m.content,
                   e.origin, e.session_id,
                   e.ts AS event_ts, e.event_time, e.role AS event_role,
                   e.tier AS event_tier,
                   e.idempotency_key AS event_idempotency_key,
                   e.content AS event_content,
                   e.tool_name, e.tool_input, e.tool_output,
                   e.meta AS event_meta, e.provenance AS event_provenance
            FROM memories m JOIN events e ON e.id=m.event_id
            WHERE m.id=?
            """,
            (memory_id,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": "memory not found"}

        event_id = row["event_id"]
        deleted: dict[str, Any] = {
            "ok": True,
            "fts_deleted": False,
            "vec_deleted": False,
            "relations_deleted": 0,
            "entities_deleted": 0,
            "event_deleted": False,
            "session_deleted": False,
        }

        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self._privacy_purge_thread_id = get_ident()
            # Privacy erasure is the sole exception to correction immutability.
            # Replace an erased chain member with a fresh opaque tombstone and
            # scrub correction request/context fields that could retain it.
            lineage_rows = self.conn.execute(
                """
                SELECT * FROM corrections
                WHERE target_memory_id=? OR replacement_memory_id=?
                """,
                (memory_id, memory_id),
            ).fetchall()
            needs_tombstone = any(
                r["replacement_memory_id"] is not None
                or r["replacement_tombstone_id"] is not None
                for r in lineage_rows
                if r["target_memory_id"] == memory_id
            ) or any(r["replacement_memory_id"] == memory_id for r in lineage_rows)
            tombstone: dict[str, Any] | None = None
            sessions_to_cleanup: dict[str, dict[str, Any]] = {}
            erased_values = _erasure_context_values(
                memory_id,
                event_id,
                row["content"],
                row["origin"],
                row["session_id"],
                row["event_idempotency_key"],
                row["event_content"],
                row["tool_name"],
                row["tool_input"],
                row["tool_output"],
                row["event_meta"],
                *_provenance_erasure_values(row["event_provenance"]),
            )

            def track_erased_session(
                session_id: object, *context_values: object
            ) -> None:
                if session_id is None:
                    return
                info = sessions_to_cleanup.setdefault(
                    str(session_id), {"sensitive_values": set(erased_values)}
                )
                info["sensitive_values"].update(
                    _erasure_context_values(*context_values)
                )

            # A target session ID is erased context even when that session
            # predates the target and still contains unrelated events.
            track_erased_session(row["session_id"], *erased_values)
            if needs_tombstone:
                tombstone = {
                    "schema_version": TOMBSTONE_SCHEMA_VERSION,
                    "tombstone_id": new_id(),
                    "status": "erased",
                    "erased_at": now_iso(),
                }
                self.conn.execute(
                    """
                    INSERT INTO lineage_tombstones(
                        schema_version, tombstone_id, status, erased_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    tuple(tombstone.values()),
                )
            for correction in lineage_rows:
                correction_session = correction["session_id"]
                track_erased_session(
                    correction_session,
                    correction["origin"],
                    correction["session_id"],
                    correction["reason"],
                    correction["idempotency_key"],
                    correction["request_identity"],
                    correction["target_tombstone_id"],
                    correction["replacement_tombstone_id"],
                )
                if correction["target_memory_id"] == memory_id:
                    self._sanitize_correction_replacement_event(
                        correction, erased_memory_id=memory_id
                    )
                    has_successor = (
                        correction["replacement_memory_id"] is not None
                        or correction["replacement_tombstone_id"] is not None
                    )
                    if not has_successor:
                        self.conn.execute(
                            "DELETE FROM corrections WHERE id=?", (correction["id"],)
                        )
                        continue
                    self.conn.execute(
                        """
                        UPDATE corrections
                        SET target_memory_id=NULL, target_tombstone_id=?,
                            origin=NULL, session_id=NULL, reason=NULL,
                            idempotency_key=NULL, request_identity=NULL,
                            request_payload=NULL, response_json=NULL
                        WHERE id=?
                        """,
                        (tombstone["tombstone_id"], correction["id"]),
                    )
                if correction["replacement_memory_id"] == memory_id:
                    self.conn.execute(
                        """
                        UPDATE corrections
                        SET replacement_memory_id=NULL, replacement_tombstone_id=?,
                            origin=NULL, session_id=NULL, reason=NULL,
                            idempotency_key=NULL, request_identity=NULL,
                            request_payload=NULL, response_json=NULL
                        WHERE id=?
                        """,
                        (tombstone["tombstone_id"], correction["id"]),
                    )

            self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))

            has_fts = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
            ).fetchone()
            if has_fts:
                self.conn.execute("DELETE FROM memories_fts WHERE id=?", (memory_id,))
                deleted["fts_deleted"] = True
            has_vec = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_memories'"
            ).fetchone()
            if has_vec:
                self.conn.execute("DELETE FROM vec_memories WHERE id=?", (memory_id,))
                deleted["vec_deleted"] = True

            other_memories = self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE event_id=?", (event_id,)
            ).fetchone()[0]
            from haunt.graph import remove_event_evidence

            rel_count, entity_count = remove_event_evidence(self.conn, event_id)
            deleted["relations_deleted"] = rel_count
            deleted["entities_deleted"] = entity_count
            if other_memories == 0:
                self.conn.execute("DELETE FROM events WHERE id=?", (event_id,))
                deleted["event_deleted"] = True
            else:
                # One event may have more than one materialized memory. The
                # survivors remain, but neither the shared event's identifier
                # nor its target-owned context may survive privacy erasure.
                safe_event_id = new_id()
                self.conn.execute(
                    """
                    INSERT INTO events(
                        id, idempotency_key, session_id, ts, event_time, role,
                        content, tool_name, tool_input, tool_output, origin,
                        tier, meta, provenance
                    ) VALUES (?, NULL, ?, ?, ?, ?, '', NULL, NULL, NULL, ?, ?, ?, ?)
                    """,
                    (
                        safe_event_id,
                        row["session_id"],
                        row["event_ts"],
                        row["event_time"],
                        row["event_role"],
                        PURGE_SAFE_ORIGIN,
                        row["event_tier"],
                        dumps({}),
                        PURGE_SAFE_PROVENANCE,
                    ),
                )
                self.conn.execute(
                    "UPDATE memories SET event_id=? WHERE event_id=?",
                    (safe_event_id, event_id),
                )
                self.conn.execute("DELETE FROM events WHERE id=?", (event_id,))
                deleted["event_deleted"] = True

            for session_id, session_info in sessions_to_cleanup.items():
                session_refs = self.conn.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM events WHERE session_id=?) +
                      (SELECT COUNT(*) FROM corrections WHERE session_id=?)
                    """,
                    (session_id, session_id),
                ).fetchone()[0]
                started_at, ended_at, safe_source, safe_meta = (
                    self._purge_safe_session_context(
                        session_id, session_info["sensitive_values"]
                    )
                )
                if session_refs > 0:
                    safe_session = self._create_purge_safe_session(
                        started_at=started_at,
                        ended_at=ended_at,
                        source=safe_source,
                        meta=safe_meta,
                    )
                    # Session IDs attached to the target or adjacent correction
                    # are erased context. Rekey every remaining reference while
                    # preserving unrelated event content and origins.
                    self.conn.execute(
                        "UPDATE events SET session_id=? WHERE session_id=?",
                        (safe_session, session_id),
                    )
                    self.conn.execute(
                        "UPDATE corrections SET session_id=? WHERE session_id=?",
                        (safe_session, session_id),
                    )
                    self.conn.execute(
                        "UPDATE meta SET value=? WHERE key='current_session' AND value=?",
                        (safe_session, session_id),
                    )
                    self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
                    deleted["session_deleted"] = True
                else:
                    self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
                    self.conn.execute(
                        "DELETE FROM meta WHERE key='current_session' AND value=?",
                        (session_id,),
                    )
                    deleted["session_deleted"] = True

            if tombstone is not None:
                deleted["lineage_tombstone"] = tombstone

            self._prune_erased_only_lineage()

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self._privacy_purge_thread_id = None
        try:
            touch_namespace(self.name, namespace_id=self.namespace_id)
        except Exception:
            pass
        return deleted

    def _sanitize_correction_replacement_event(
        self,
        correction: sqlite3.Row,
        *,
        erased_memory_id: str,
    ) -> None:
        """Remove purged correction context from its surviving replacement event.

        Only the direct replacement created by this correction is eligible.
        Content and unrelated event origins are never changed. Session rekeying
        is handled once for every target and adjacent correction session.
        """
        replacement_id = correction["replacement_memory_id"]
        if replacement_id is None or replacement_id == erased_memory_id:
            return
        event = self.conn.execute(
            """
            SELECT e.id, e.origin, e.provenance
            FROM memories m JOIN events e ON e.id=m.event_id
            WHERE m.id=?
            """,
            (replacement_id,),
        ).fetchone()
        if event is None:
            return

        correction_origin = correction["origin"]
        origin_matches = (
            correction_origin is not None and event["origin"] == correction_origin
        )
        parsed_provenance = loads(event["provenance"], default={})
        provenance_matches = bool(
            isinstance(parsed_provenance, dict)
            and correction_origin is not None
            and parsed_provenance.get("origin") == correction_origin
        )
        if not origin_matches and not provenance_matches:
            return

        updates: list[str] = []
        params: list[Any] = []
        if origin_matches:
            updates.append("origin=?")
            params.append(PURGE_SAFE_ORIGIN)
        if provenance_matches:
            updates.append("provenance=?")
            params.append(PURGE_SAFE_PROVENANCE)
        params.append(event["id"])
        self.conn.execute(
            f"UPDATE events SET {', '.join(updates)} WHERE id=?",
            params,
        )

    def _create_purge_safe_session(
        self,
        *,
        started_at: str | None = None,
        ended_at: str | None = None,
        source: Any,
        meta: Any,
    ) -> str:
        safe_session = new_id()
        self.conn.execute(
            """
            INSERT INTO sessions(id, started_at, ended_at, source, meta)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                safe_session,
                now_iso() if started_at is None else started_at,
                ended_at,
                source,
                meta,
            ),
        )
        return safe_session

    def _purge_safe_session_context(
        self, session_id: str, sensitive_values: set[str]
    ) -> tuple[str | None, str | None, Any, Any]:
        """Preserve clean session fields and remove only erased context."""
        row = self.conn.execute(
            "SELECT started_at, ended_at, source, meta FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None, None, PURGE_SAFE_SESSION_SOURCE, dumps({})
        source = row["source"]
        original_meta = row["meta"]
        dropped = object()

        def tainted(value: Any) -> bool:
            if isinstance(value, memoryview):
                value = value.tobytes()
            if isinstance(value, (bytes, bytearray)):
                raw = bytes(value)
                return any(
                    raw == token.encode("utf-8")
                    or (len(token) >= 8 and token.encode("utf-8") in raw)
                    for token in sensitive_values
                )
            if not isinstance(value, str):
                return False
            return any(
                value == token or (len(token) >= 8 and token in value)
                for token in sensitive_values
            )

        safe_source = PURGE_SAFE_SESSION_SOURCE if tainted(source) else source
        if isinstance(original_meta, memoryview):
            original_meta = original_meta.tobytes()
        if isinstance(original_meta, (bytes, bytearray)):
            try:
                meta_text = bytes(original_meta).decode("utf-8")
            except UnicodeDecodeError:
                # Opaque metadata on an affected session cannot be proven free
                # of erased context. Privacy purge therefore drops it even when
                # no plaintext marker can be found in the raw bytes.
                return row["started_at"], row["ended_at"], safe_source, dumps({})
        else:
            meta_text = original_meta
        not_json = object()
        original = loads(meta_text, default=not_json)

        def contains_tainted(value: Any) -> bool:
            if tainted(value):
                return True
            if isinstance(value, dict):
                return any(
                    contains_tainted(key) or contains_tainted(child)
                    for key, child in value.items()
                )
            if isinstance(value, list):
                return any(contains_tainted(child) for child in value)
            return False

        if not tainted(original_meta) and (
            original is not_json or not contains_tainted(original)
        ):
            return row["started_at"], row["ended_at"], safe_source, original_meta
        if original is not_json:
            return row["started_at"], row["ended_at"], safe_source, dumps({})

        def sanitize(value: Any) -> Any:
            if isinstance(value, str):
                return dropped if tainted(value) else value
            if isinstance(value, dict):
                clean: dict[Any, Any] = {}
                for key, child in value.items():
                    if isinstance(key, str) and tainted(key):
                        continue
                    sanitized = sanitize(child)
                    if sanitized is not dropped:
                        clean[key] = sanitized
                return clean
            if isinstance(value, list):
                return [
                    sanitized
                    for child in value
                    if (sanitized := sanitize(child)) is not dropped
                ]
            return value

        sanitized_meta = dumps(sanitize(original))
        if tainted(sanitized_meta):
            sanitized_meta = dumps({})
        return row["started_at"], row["ended_at"], safe_source, sanitized_meta

    def _prune_erased_only_lineage(self) -> None:
        """During purge, discard components that no surviving memory can trace."""
        rows = self.conn.execute(
            """
            SELECT id, target_memory_id, target_tombstone_id,
                   replacement_memory_id, replacement_tombstone_id
            FROM corrections
            """
        ).fetchall()

        def nodes(row: sqlite3.Row) -> list[tuple[str, str]]:
            found: list[tuple[str, str]] = []
            for prefix in ("target", "replacement"):
                if row[f"{prefix}_memory_id"] is not None:
                    found.append(("memory", str(row[f"{prefix}_memory_id"])))
                elif row[f"{prefix}_tombstone_id"] is not None:
                    found.append(("tombstone", str(row[f"{prefix}_tombstone_id"])))
            return found

        by_node: dict[tuple[str, str], set[str]] = {}
        row_nodes: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            correction_nodes = nodes(row)
            row_nodes[str(row["id"])] = correction_nodes
            for item in correction_nodes:
                by_node.setdefault(item, set()).add(str(row["id"]))

        pending = set(row_nodes)
        while pending:
            seed = pending.pop()
            component = {seed}
            frontier = [seed]
            component_nodes: set[tuple[str, str]] = set()
            while frontier:
                correction_id = frontier.pop()
                for item in row_nodes[correction_id]:
                    component_nodes.add(item)
                    for neighbor in by_node[item]:
                        if neighbor in pending:
                            pending.remove(neighbor)
                            component.add(neighbor)
                            frontier.append(neighbor)
            if any(kind == "memory" for kind, _ in component_nodes):
                continue
            self.conn.executemany(
                "DELETE FROM corrections WHERE id=?", [(item,) for item in component]
            )

        self.conn.execute(
            """
            DELETE FROM lineage_tombstones
            WHERE tombstone_id NOT IN (
                SELECT target_tombstone_id FROM corrections
                WHERE target_tombstone_id IS NOT NULL
                UNION
                SELECT replacement_tombstone_id FROM corrections
                WHERE replacement_tombstone_id IS NOT NULL
            )
            """
        )

    def trace(self, memory_id: str) -> dict[str, Any]:
        """Return the ordered correction chain containing a surviving memory.

        Correction records are append-only during ordinary operation. Purge may
        scrub/delete adjacent records and substitute allowlisted tombstones.
        """
        requested = self.conn.execute(
            "SELECT id, valid_to FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if requested is None:
            return {"ok": False, "error": "memory not found"}

        correction_rows = self.conn.execute(
            "SELECT * FROM corrections ORDER BY corrected_at, rowid"
        ).fetchall()

        def node(row: sqlite3.Row, prefix: str) -> tuple[str, str] | None:
            memory = row[f"{prefix}_memory_id"]
            if memory is not None:
                return ("memory", str(memory))
            tombstone = row[f"{prefix}_tombstone_id"]
            if tombstone is not None:
                return ("tombstone", str(tombstone))
            return None

        incoming: dict[tuple[str, str], sqlite3.Row] = {}
        outgoing: dict[tuple[str, str], sqlite3.Row] = {}
        for correction in correction_rows:
            target = node(correction, "target")
            replacement = node(correction, "replacement")
            if target is not None:
                outgoing[target] = correction
            if replacement is not None:
                incoming[replacement] = correction

        start = ("memory", memory_id)
        seen: set[tuple[str, str]] = set()
        while start in incoming and start not in seen:
            seen.add(start)
            predecessor = node(incoming[start], "target")
            if predecessor is None:
                break
            start = predecessor

        members: list[dict[str, Any]] = []
        corrections: list[dict[str, Any]] = []
        current = start
        seen.clear()
        while current not in seen:
            seen.add(current)
            if current[0] == "tombstone":
                tomb = self.conn.execute(
                    """
                    SELECT schema_version, tombstone_id, status, erased_at
                    FROM lineage_tombstones WHERE tombstone_id=?
                    """,
                    (current[1],),
                ).fetchone()
                if tomb is None:
                    break
                members.append(dict(tomb))
            else:
                memory = self.conn.execute(
                    """
                    SELECT m.id AS memory_id, m.event_id, m.content, m.tier,
                           m.valid_from, m.valid_to, m.created_at,
                           e.session_id, e.event_time, e.ts, e.role, e.origin,
                           e.tool_name, e.meta, e.provenance
                    FROM memories m JOIN events e ON e.id=m.event_id
                    WHERE m.id=?
                    """,
                    (current[1],),
                ).fetchone()
                if memory is None:
                    break
                member = dict(memory)
                member["provenance"] = public_provenance(
                    member.pop("provenance"),
                    origin=member["origin"],
                    legacy_meta=member.pop("meta"),
                    tool_name=member.get("tool_name"),
                )
                if current in outgoing:
                    member["status"] = "superseded"
                elif member["valid_to"] is not None:
                    member["status"] = "legacy_unlinked"
                else:
                    member["status"] = "current"
                members.append(member)

            correction = outgoing.get(current)
            if correction is None:
                break
            corrections.append(
                {
                    "correction_id": correction["id"],
                    "corrected_at": correction["corrected_at"],
                    "origin": correction["origin"],
                    "session_id": correction["session_id"],
                    "reason": correction["reason"],
                }
            )
            successor = node(correction, "replacement")
            if successor is None:
                break
            current = successor

        linked = bool(corrections or incoming.get(("memory", memory_id)))
        lineage_status = (
            "linked"
            if linked
            else (
                "legacy_unlinked" if requested["valid_to"] is not None else "standalone"
            )
        )
        return json_safe_sqlite(
            {
                "ok": True,
                "schema_version": 1,
                "namespace": self.name,
                "requested_memory_id": memory_id,
                "lineage_status": lineage_status,
                "members": members,
                "corrections": corrections,
            }
        )

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Retrieve full provenance detail for a single memory."""
        row = self.conn.execute(
            """
            SELECT m.id AS memory_id, m.event_id, m.tier, m.content,
                   m.valid_from, m.valid_to, m.created_at,
                   e.session_id, e.ts, e.event_time, e.role, e.content AS event_content,
                   e.tool_name, e.tool_input, e.tool_output, e.origin, e.meta,
                   e.provenance
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE m.id = ?
            """,
            (memory_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["provenance"] = public_provenance(
            d.pop("provenance"),
            origin=d["origin"],
            legacy_meta=d["meta"],
            tool_name=d["tool_name"],
        )
        d["db_path"] = str(Path(self.db_path).resolve())
        d["haunt_home"] = str(haunt_home())
        d["namespace"] = self.name
        d["has_embedding"] = self.conn.execute(
            "SELECT embedding IS NOT NULL AS has FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()["has"]

        mentions = self.conn.execute(
            """
            SELECT DISTINCT e.id, e.name, e.type
            FROM entities e
            JOIN relations r ON (r.src_entity = e.id OR r.dst_entity = e.id)
            WHERE r.event_id = ?
            """,
            (d["event_id"],),
        ).fetchall()
        d["entity_mentions"] = [dict(m) for m in mentions]

        related = self.conn.execute(
            """
            SELECT m.id AS memory_id, m.tier, m.content, m.valid_from, m.valid_to
            FROM memories m
            WHERE m.event_id IN (
                SELECT id FROM events WHERE session_id = ?
            ) AND m.id != ?
            ORDER BY m.created_at DESC, m.rowid DESC
            LIMIT 20
            """,
            (d["session_id"], memory_id),
        ).fetchall()
        d["related_memories"] = [dict(r) for r in related]
        d["trace"] = self.trace(memory_id)

        return json_safe_sqlite(d)

    def browse_memories(
        self,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        tier: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse memories with filters. Returns paginated results."""
        sql = """
            SELECT m.id AS memory_id, m.event_id, m.tier, m.content,
                   m.valid_from, m.valid_to, m.created_at,
                   e.session_id, e.event_time, e.role, e.origin, e.tool_name,
                   e.meta, e.provenance
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE 1=1
        """
        count_sql = """
            SELECT COUNT(*) FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE 1=1
        """
        params: list[Any] = []
        if session_id:
            sql += " AND e.session_id = ?"
            count_sql += " AND e.session_id = ?"
            params.append(session_id)
        if origin:
            sql += " AND e.origin = ?"
            count_sql += " AND e.origin = ?"
            params.append(origin)
        if tier:
            sql += " AND m.tier = ?"
            count_sql += " AND m.tier = ?"
            params.append(tier)
        if since:
            sql += " AND e.event_time >= ?"
            count_sql += " AND e.event_time >= ?"
            params.append(iso_or_now(since))
        if until:
            sql += " AND e.event_time <= ?"
            count_sql += " AND e.event_time <= ?"
            params.append(iso_or_now(until))

        total = self.conn.execute(count_sql, params).fetchone()[0]
        sql += " ORDER BY m.created_at DESC, m.rowid DESC LIMIT ? OFFSET ?"
        rows = self.conn.execute(sql, params + [limit, offset]).fetchall()
        return json_safe_sqlite(
            {
                "memories": [
                    {
                        **{
                            k: value
                            for k, value in dict(r).items()
                            if k not in {"provenance", "meta"}
                        },
                        "provenance": public_provenance(
                            r["provenance"],
                            origin=r["origin"],
                            legacy_meta=r["meta"],
                            tool_name=r["tool_name"],
                        ),
                    }
                    for r in rows
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    # ------------------------------------------------------------------
    # worldview: compact per-namespace briefing
    # ------------------------------------------------------------------

    def worldview(self, *, facts_cap: int = 12, names_cap: int = 12) -> dict[str, Any]:
        """Compile a structured namespace briefing from existing rows.

        No LLM. Pure read queries over stored semantic/procedural/entity data.
        """
        facts_cap = clamp_limit(facts_cap, default=12)
        names_cap = clamp_limit(names_cap, default=12)
        facts = [
            dict(r)
            for r in self.conn.execute(
                """
                SELECT m.id, m.content, m.valid_from, m.created_at
                FROM memories m
                JOIN events e ON e.id=m.event_id
                WHERE m.tier='semantic' AND m.valid_to IS NULL
                  AND e.role != 'tool' AND e.tool_name IS NULL
                ORDER BY m.created_at DESC, m.rowid DESC
                LIMIT ?
                """,
                (facts_cap,),
            ).fetchall()
        ]

        procedures = [
            dict(r)
            for r in self.conn.execute(
                """
                SELECT m.id, m.content, e.meta,
                       e.origin, e.tool_name, e.provenance
                FROM memories m
                JOIN events e ON e.id = m.event_id
                WHERE m.tier='procedural' AND m.valid_to IS NULL
                  AND CASE WHEN json_valid(e.meta)
                           THEN json_extract(e.meta, '$.kind') END = 'procedure'
                ORDER BY m.created_at DESC, m.rowid DESC
                """,
            ).fetchall()
        ]
        proc_index: list[dict[str, Any]] = []
        for p in procedures:
            emeta = loads(p.get("meta"))
            proc_index.append(
                {
                    "id": p["id"],
                    "name": emeta.get("name", ""),
                    "trigger": emeta.get("trigger", ""),
                    "provenance": public_provenance(
                        p["provenance"],
                        origin=p["origin"],
                        legacy_meta=p["meta"],
                        tool_name=p["tool_name"],
                    ),
                }
            )

        names = self.top_entities(limit=names_cap, trusted_only=True)
        name_list = [
            {"name": n["name"], "type": n["type"], "mentions": n["rels"]} for n in names
        ]

        stats = self.stats()
        counts = {
            "events": stats["events"],
            "memories": stats["memories"],
            "sessions": stats["sessions"],
        }

        return json_safe_sqlite(
            {
                "namespace": self.name,
                "facts": facts,
                "names": name_list,
                "procedures": proc_index,
                "counts": counts,
            }
        )

    # ------------------------------------------------------------------
    # procedure: named how-tos
    # ------------------------------------------------------------------

    def procedure_write(
        self,
        name: str,
        body: str,
        *,
        trigger: str = "",
        origin: str = "python",
        channel: str = "python",
        session_id: str | None = None,
    ) -> ObserveResult:
        """Store a named procedure. Verbatim body, stored as tier=procedural."""
        meta = {"kind": "procedure", "name": name, "trigger": trigger}
        return self.observe(
            body,
            role="system",
            tier="procedural",
            session_id=session_id,
            origin=origin,
            channel=channel,
            meta=meta,
        )

    def procedure_get(self, name: str) -> dict[str, Any] | None:
        """Retrieve a procedure by name. Returns newest matching row."""
        row = self.conn.execute(
            """
            SELECT m.id, m.content, m.valid_from, m.valid_to, m.created_at,
                   e.meta, e.origin, e.tool_name, e.provenance
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE m.tier='procedural'
              AND m.valid_to IS NULL
              AND CASE WHEN json_valid(e.meta)
                       THEN json_extract(e.meta, '$.kind') END = 'procedure'
              AND CASE WHEN json_valid(e.meta)
                       THEN json_extract(e.meta, '$.name') END = ?
            ORDER BY m.created_at DESC, m.rowid DESC
            LIMIT 1
            """,
            (name,),
        ).fetchone()
        if not row:
            return None
        emeta = loads(row["meta"])
        return json_safe_sqlite(
            {
                "id": row["id"],
                "name": emeta.get("name", name),
                "body": row["content"],
                "trigger": emeta.get("trigger", ""),
                "valid_from": row["valid_from"],
                "created_at": row["created_at"],
                "provenance": public_provenance(
                    row["provenance"],
                    origin=row["origin"],
                    legacy_meta=row["meta"],
                    tool_name=row["tool_name"],
                ),
            }
        )

    def procedure_list(self) -> list[dict[str, Any]]:
        """List all active procedures (valid_to IS NULL)."""
        rows = self.conn.execute(
            """
            SELECT m.id, m.content, m.created_at, e.meta,
                   e.origin, e.tool_name, e.provenance
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE m.tier='procedural'
              AND m.valid_to IS NULL
              AND CASE WHEN json_valid(e.meta)
                       THEN json_extract(e.meta, '$.kind') END = 'procedure'
            ORDER BY m.created_at DESC, m.rowid DESC
            """,
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            emeta = loads(r["meta"])
            out.append(
                {
                    "id": r["id"],
                    "name": emeta.get("name", ""),
                    "body": r["content"],
                    "trigger": emeta.get("trigger", ""),
                    "created_at": r["created_at"],
                    "provenance": public_provenance(
                        r["provenance"],
                        origin=r["origin"],
                        legacy_meta=r["meta"],
                        tool_name=r["tool_name"],
                    ),
                }
            )
        return json_safe_sqlite(out)

    # ------------------------------------------------------------------
    # contradict: supersede a memory
    # ------------------------------------------------------------------

    def _correction_replay(self, key: str, payload: bytes) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT request_payload, response_json FROM corrections WHERE idempotency_key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if row["request_payload"] != payload:
            return {
                "ok": False,
                "conflict": "idempotency_key_reused",
                "error": "idempotency_key was reused with a different correction payload",
            }
        original = loads(row["response_json"], default={})
        original["deduplicated"] = True
        return original

    def contradict(
        self,
        memory_id: str,
        *,
        idempotency_key: str,
        replacement: str | None = None,
        namespace: str | None = None,
        origin: str = "python",
        channel: str = "python",
        session_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Append a correction and update its current/as-of projection atomically."""
        if replacement is not None and not isinstance(replacement, str):
            raise ValueError("replacement must be a string or null")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string or null")
        if not isinstance(idempotency_key, str):
            raise ValueError("idempotency_key must be a string")
        key = idempotency_key
        if not key or not key.strip():
            raise ValueError("idempotency_key must be non-empty")
        if len(key) > CORRECTION_KEY_MAX:
            raise ValueError(
                f"idempotency_key must be {CORRECTION_KEY_MAX} characters or fewer"
            )

        # Null, empty, and whitespace-only replacements are distinct canonical
        # requests. An explicitly supplied string is always stored verbatim.
        payload = _correction_request_payload(memory_id, replacement, reason)
        request_identity = _correction_request_identity(payload)
        replay_result = self._correction_replay(key, payload)
        if replay_result is not None:
            return replay_result
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError("origin must be a non-empty string")
        if session_id is not None and not isinstance(session_id, str):
            raise ValueError("session_id must be a string or null")
        replacement_text = replacement

        try:
            self.conn.execute("BEGIN IMMEDIATE")
            replay_result = self._correction_replay(key, payload)
            if replay_result is not None:
                self.conn.rollback()
                return replay_result

            row = self.conn.execute(
                "SELECT id, valid_to FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if not row:
                self.conn.rollback()
                return {"ok": False, "error": f"memory {memory_id} not found"}
            if row["valid_to"] is not None:
                self.conn.rollback()
                return {
                    "ok": False,
                    "conflict": "already_superseded",
                    "error": f"memory {memory_id} already superseded",
                    "valid_to": row["valid_to"],
                }

            ts = now_iso()
            correction_id = new_id()
            sid = self.ensure_session(session_id, source=origin, commit=False)
            cur = self.conn.execute(
                "UPDATE memories SET valid_to=? WHERE id=? AND valid_to IS NULL",
                (ts, memory_id),
            )
            if cur.rowcount != 1:
                self.conn.rollback()
                again = self.conn.execute(
                    "SELECT valid_to FROM memories WHERE id=?", (memory_id,)
                ).fetchone()
                return {
                    "ok": False,
                    "conflict": "already_superseded",
                    "error": f"memory {memory_id} already superseded",
                    "valid_to": None if again is None else again["valid_to"],
                }
            result: dict[str, Any] = {
                "ok": True,
                "correction_id": correction_id,
                "superseded": memory_id,
                "valid_to": ts,
                "idempotency_key": key,
                "request_identity": request_identity,
                "deduplicated": False,
            }
            replacement_memory_id: str | None = None
            if replacement_text is not None:
                r = self.observe(
                    replacement_text,
                    role="system",
                    tier="semantic",
                    session_id=sid,
                    event_time=ts,
                    valid_from=ts,
                    origin=origin,
                    channel=channel,
                    commit=False,
                )
                replacement_memory_id = r.memory_id
                result["replacement_memory_id"] = r.memory_id
                result["replacement_event_id"] = r.event_id
            self.conn.execute(
                """
                INSERT INTO corrections(
                    id, target_memory_id, replacement_memory_id, corrected_at,
                    origin, session_id, reason, idempotency_key,
                    request_identity, request_payload, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    correction_id,
                    memory_id,
                    replacement_memory_id,
                    ts,
                    origin,
                    sid,
                    reason,
                    key,
                    request_identity,
                    payload,
                    dumps(result),
                ),
            )
            self.conn.commit()
            try:
                touch_namespace(self.name, namespace_id=self.namespace_id)
            except Exception:
                pass
            return result
        except Exception:
            self.conn.rollback()
            raise


def open_existing(name: str, repo_path: str | None = None) -> Store:
    """Open a registered namespace. Never creates a DB or registry row."""
    if not namespace_exists(name):
        raise UnknownNamespaceError(name)
    try:
        return Store(name, repo_path=repo_path, create=False)
    except FileNotFoundError as exc:
        raise UnknownNamespaceError(name) from exc


def open_namespace_identity(
    namespace_id: str,
    *,
    expected_db_path: str | None = None,
    expected_db_device: int | None = None,
    expected_db_inode: int | None = None,
) -> Store:
    """Open one stable registry identity without resolving any label again."""

    def validate_selected(identity: dict[str, Any]) -> None:
        if expected_db_path is not None and str(identity["db_path"]) != expected_db_path:
            raise NamespacePathError(
                "selected namespace database path changed before open"
            )
        actual_device = identity.get("db_device")
        if expected_db_device is not None and (
            actual_device is None
            or int(actual_device) != int(expected_db_device)
        ):
            raise NamespacePathError(
                "selected namespace database identity changed before open"
            )
        actual_inode = identity.get("db_inode")
        if expected_db_inode is not None and (
            actual_inode is None
            or int(actual_inode) != int(expected_db_inode)
        ):
            raise NamespacePathError(
                "selected namespace database identity changed before open"
            )

    with _namespace_migration_lock():
        identity = resolve_namespace_id(namespace_id)
        if identity is None:
            raise UnknownNamespaceError(namespace_id)
        validate_selected(identity)
        store = Store._from_identity(identity)
        try:
            current = resolve_namespace_id(namespace_id)
            if current is None:
                raise UnknownNamespaceError(namespace_id)
            validate_selected(current)
            stable_fields = (
                "namespace_id", "canonical_label", "canonical_label_norm",
                "db_path", "db_device", "db_inode",
            )
            if any(current[field] != identity[field] for field in stable_fields):
                raise NamespacePathError(
                    "selected namespace identity changed while opening"
                )
            if store.namespace_id != namespace_id or str(store.db_path) != str(
                current["db_path"]
            ):
                raise NamespacePathError(
                    "opened namespace does not match selected stable identity"
                )
            if isinstance(store.conn, _SidecarGuardedConnection):
                store.conn.verify_storage_guards()
            return store
        except Exception:
            store.close()
            raise


def get_store(name: str | None = None, repo_path: str | None = None) -> Store:
    ns = resolve_namespace(name)
    return Store(ns, repo_path=repo_path, create=True)


def observe(
    content: str = "",
    *,
    namespace: str | None = None,
    **kwargs: Any,
) -> ObserveResult:
    kwargs.setdefault("origin", "python")
    kwargs.setdefault("channel", "python")
    with get_store(namespace) as store:
        return store.observe(content, **kwargs)


def list_namespaces(*, only: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows = list_namespace_rows()
    if only is not None:
        requested_norm = normalize_namespace_label(only)
        rows = [
            row
            for row in rows
            if any(
                normalize_namespace_label(str(alias)) == requested_norm
                for alias in row.get("aliases", [row["name"]])
            )
        ]
    for row in rows:
        db = Path(row["db_path"])
        extra: dict[str, Any] = {"db_size_bytes": 0}
        if row.get("error"):
            extra["error"] = str(row["error"])
            out.append({**row, **extra})
            continue
        extra["db_size_bytes"] = db.stat().st_size if db.exists() else 0
        try:
            with Store(row["name"], create=False) as st:
                stats = st.stats()
                extra.update(
                    {
                        "events": stats["events"],
                        "memories": stats["memories"],
                        "sessions": stats["sessions"],
                        "entities": stats["entities"],
                        "db_size_bytes": stats["db_size_bytes"],
                    }
                )
        except (sqlite3.Error, OSError, NamespacePathError) as exc:
            extra["error"] = str(exc)
        out.append({**row, **extra})
    return json_safe_sqlite(out)


def iter_stores() -> Iterator[Store]:
    for row in list_namespace_rows():
        yield Store(row["name"], create=False)


def reembed_all_namespaces() -> list[dict[str, Any]]:
    """Rebuild embeddings in every registered namespace."""
    out: list[dict[str, Any]] = []
    for row in list_namespace_rows():
        with Store(row["name"], create=False) as st:
            report = st.reembed()
            report["namespace"] = row["name"]
            out.append(report)
    return out
