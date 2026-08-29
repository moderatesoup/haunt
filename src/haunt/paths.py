"""Filesystem layout: ~/.haunt (or $HAUNT_HOME). Absolute paths only."""

from __future__ import annotations

import os
import ipaddress
import re
import secrets
import sqlite3
import stat
import subprocess
import threading
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

SAFE_NS = re.compile(r"[^a-zA-Z0-9._-]+")

DIR_MODE = 0o700
FILE_MODE = 0o600
SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
SQLITE_OPEN_LOCK = threading.RLock()

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


def required_o_nofollow() -> int:
    """Return the mandatory no-follow flag or fail closed on this platform."""
    value = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(value, int) or value <= 0:
        raise NamespacePathError("O_NOFOLLOW is required for safe filesystem opens")
    return value


@dataclass
class SQLitePrimaryGuard:
    """Hold a SQLite main-file descriptor and its pathname identity.

    SQLite still opens the pathname through its VFS.  The caller must bracket
    that open with :meth:`verify` and verify the VFS descriptor identity; this
    guard prevents a validated name from silently becoming a different file.
    """

    path: Path
    fd: int
    identity: tuple[int, int]
    claimed: bool = False
    _closed: bool = False

    @classmethod
    def acquire(cls, path: Path, *, create_missing: bool) -> "SQLitePrimaryGuard":
        path = Path(path)
        lexical = Path(os.path.abspath(os.path.normpath(str(path))))
        if not path.is_absolute() or path != lexical:
            raise NamespacePathError(
                f"SQLite database path must be canonical and absolute: {path}"
            )
        try:
            parent = path.parent.lstat()
        except OSError as exc:
            raise NamespacePathError(
                f"SQLite database directory is missing or unreadable: {path.parent}"
            ) from exc
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            raise NamespacePathError(
                "SQLite database directory must be a real non-symlink directory: "
                f"{path.parent}"
            )
        nofollow = required_o_nofollow()
        cloexec = getattr(os, "O_CLOEXEC", 0)
        for _attempt in range(3):
            claimed = False
            try:
                before = path.lstat()
            except FileNotFoundError:
                if not create_missing:
                    raise
                try:
                    fd = os.open(
                        path,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow | cloexec,
                        FILE_MODE,
                    )
                    claimed = True
                except FileExistsError:
                    continue
            except OSError as exc:
                raise NamespacePathError(
                    f"cannot inspect SQLite database: {path}"
                ) from exc
            else:
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                    raise NamespacePathError(
                        f"SQLite database must be a regular non-symlink file: {path}"
                    )
                if int(before.st_nlink) != 1:
                    raise NamespacePathError(
                        f"SQLite database must have exactly one filesystem link: {path}"
                    )
                try:
                    fd = os.open(path, os.O_RDONLY | nofollow | cloexec)
                except OSError as exc:
                    raise NamespacePathError(
                        f"cannot safely open SQLite database: {path}"
                    ) from exc
            info = os.fstat(fd)
            identity = int(info.st_dev), int(info.st_ino)
            try:
                current = path.lstat()
            except OSError as exc:
                os.close(fd)
                raise NamespacePathError(
                    f"SQLite database disappeared while opening: {path}"
                ) from exc
            if (
                not stat.S_ISREG(info.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or int(info.st_nlink) != 1
                or int(current.st_nlink) != 1
                or identity != (int(current.st_dev), int(current.st_ino))
            ):
                os.close(fd)
                raise NamespacePathError(
                    f"SQLite database physical identity changed while opening: {path}"
                )
            if claimed:
                os.fchmod(fd, FILE_MODE)
            return cls(path, fd, identity, claimed)
        raise NamespacePathError(
            f"SQLite database changed repeatedly while claiming its name: {path}"
        )

    def verify(self) -> None:
        info = os.fstat(self.fd)
        try:
            current = self.path.lstat()
        except OSError as exc:
            raise NamespacePathError(
                f"SQLite database disappeared during safe open: {self.path}"
            ) from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or int(info.st_nlink) != 1
            or int(current.st_nlink) != 1
            or (int(info.st_dev), int(info.st_ino)) != self.identity
            or (int(current.st_dev), int(current.st_ino)) != self.identity
        ):
            raise NamespacePathError(
                f"SQLite database physical identity changed: {self.path}"
            )

    def close(self, *, clean_claim: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if clean_claim and self.claimed:
            try:
                current = self.path.lstat()
            except OSError:
                current = None
            if current is not None and (
                stat.S_ISREG(current.st_mode)
                and not stat.S_ISLNK(current.st_mode)
                and (int(current.st_dev), int(current.st_ino)) == self.identity
            ):
                try:
                    self.path.unlink()
                except OSError:
                    pass
        try:
            os.close(self.fd)
        except OSError:
            pass


def _descriptor_identities() -> dict[int, tuple[int, int]]:
    """Open descriptor identities, for verifying which file SQLite opened."""
    for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        result: dict[int, tuple[int, int]] = {}
        for entry in entries:
            try:
                fd = int(entry.name)
                info = os.fstat(fd)
            except (OSError, ValueError):
                continue
            result[fd] = int(info.st_dev), int(info.st_ino)
        return result
    raise NamespacePathError("cannot verify the physical file opened by SQLite")


def _verify_sqlite_primary_open(
    before: dict[int, tuple[int, int]],
    primary: SQLitePrimaryGuard,
    sidecars: "SQLiteSidecarGuard",
) -> None:
    after = _descriptor_identities()
    before_count = sum(
        identity == primary.identity for identity in before.values()
    )
    after_count = sum(
        identity == primary.identity for identity in after.values()
    )
    if after_count > before_count:
        return
    if before_count >= 2 and after_count >= 2:
        # SQLite may reuse its already-verified unix VFS descriptor.
        return
    raise NamespacePathError("SQLite did not open the claimed registry identity")


@dataclass
class _SQLiteSidecarEntry:
    path: Path
    fd: int | None
    identity: tuple[int, int] | None
    claimed: bool = False


class SQLiteSidecarGuard:
    """Hold and verify SQLite sidecar names across a database open/configure."""

    def __init__(self, db_path: Path, entries: list[_SQLiteSidecarEntry]):
        self.db_path = db_path
        self.entries = entries
        self._closed = False

    @classmethod
    def acquire(
        cls, db_path: Path, *, claim_missing: bool
    ) -> "SQLiteSidecarGuard":
        db_path = Path(db_path)
        lexical = Path(os.path.abspath(os.path.normpath(str(db_path))))
        if not db_path.is_absolute() or db_path != lexical:
            raise NamespacePathError(
                f"SQLite database path must be canonical and absolute: {db_path}"
            )
        parent = db_path.parent
        try:
            parent_info = parent.lstat()
        except OSError as exc:
            raise NamespacePathError(
                f"SQLite database directory is missing or unreadable: {parent}"
            ) from exc
        if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
            raise NamespacePathError(
                f"SQLite database directory must be a real non-symlink directory: {parent}"
            )

        entries: list[_SQLiteSidecarEntry] = []
        guard = cls(db_path, entries)
        try:
            for suffix in SQLITE_SIDECAR_SUFFIXES:
                sidecar = Path(str(db_path) + suffix)
                entries.append(
                    cls._acquire_one(sidecar, claim_missing=claim_missing)
                )
            guard.verify()
            return guard
        except Exception:
            guard.close(clean_unused_claims=True)
            raise

    @staticmethod
    def _acquire_one(path: Path, *, claim_missing: bool) -> _SQLiteSidecarEntry:
        nofollow = required_o_nofollow()
        cloexec = getattr(os, "O_CLOEXEC", 0)
        nonblock = getattr(os, "O_NONBLOCK", 0)
        for _attempt in range(3):
            try:
                before = path.lstat()
            except FileNotFoundError:
                if not claim_missing:
                    return _SQLiteSidecarEntry(path, None, None)
                try:
                    fd = os.open(
                        path,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow | cloexec,
                        FILE_MODE,
                    )
                except FileExistsError:
                    continue
                info = os.fstat(fd)
                return _SQLiteSidecarEntry(
                    path,
                    fd,
                    (int(info.st_dev), int(info.st_ino)),
                    claimed=True,
                )
            except OSError as exc:
                raise NamespacePathError(
                    f"cannot inspect SQLite sidecar: {path}"
                ) from exc
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise NamespacePathError(
                    f"SQLite sidecar must be a regular non-symlink file: {path}"
                )
            if int(before.st_nlink) > 1:
                raise NamespacePathError(
                    f"SQLite sidecar must have exactly one filesystem link: {path}"
                )
            if int(before.st_nlink) != 1:
                raise NamespacePathError(
                    f"SQLite sidecar changed while opening: {path}"
                )
            try:
                fd = os.open(path, os.O_RDONLY | nofollow | cloexec | nonblock)
            except FileNotFoundError as exc:
                raise NamespacePathError(
                    f"SQLite sidecar changed while opening: {path}"
                ) from exc
            except OSError as exc:
                raise NamespacePathError(
                    f"cannot safely open SQLite sidecar: {path}"
                ) from exc
            info = os.fstat(fd)
            identity = (int(info.st_dev), int(info.st_ino))
            if (
                not stat.S_ISREG(info.st_mode)
                or int(info.st_nlink) > 1
            ):
                os.close(fd)
                raise NamespacePathError(
                    "SQLite sidecar physical identity changed unsafely while opening: "
                    f"{path}"
                )
            if (
                int(info.st_nlink) != 1
                or identity != (int(before.st_dev), int(before.st_ino))
            ):
                os.close(fd)
                raise NamespacePathError(
                    f"SQLite sidecar physical identity changed while opening: {path}"
                )
            return _SQLiteSidecarEntry(path, fd, identity)
        raise NamespacePathError(
            f"SQLite sidecar changed repeatedly while claiming its name: {path}"
        )

    def verify(self) -> None:
        """Fail unless every held/absent sidecar name remains unchanged."""
        for entry in self.entries:
            if entry.fd is None:
                try:
                    entry.path.lstat()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise NamespacePathError(
                        f"cannot inspect SQLite sidecar: {entry.path}"
                    ) from exc
                raise NamespacePathError(
                    f"SQLite sidecar appeared during safe open: {entry.path}"
                )
            info = os.fstat(entry.fd)
            held = (int(info.st_dev), int(info.st_ino))
            try:
                current = entry.path.lstat()
            except OSError as exc:
                raise NamespacePathError(
                    f"SQLite sidecar disappeared during safe open: {entry.path}"
                ) from exc
            current_identity = (int(current.st_dev), int(current.st_ino))
            if (
                not stat.S_ISREG(info.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or int(info.st_nlink) > 1
                or int(current.st_nlink) > 1
            ):
                raise NamespacePathError(
                    "SQLite sidecar physical identity changed unsafely during open: "
                    f"{entry.path}"
                )
            if (
                int(info.st_nlink) != 1
                or held != entry.identity
                or current_identity != entry.identity
            ):
                raise NamespacePathError(
                    f"SQLite sidecar physical identity changed: {entry.path}"
                )

    def close(self, *, clean_unused_claims: bool) -> None:
        if self._closed:
            return
        self._closed = True
        for entry in self.entries:
            if clean_unused_claims and entry.claimed and entry.identity is not None:
                try:
                    current = entry.path.lstat()
                except OSError:
                    current = None
                if current is not None and (
                    stat.S_ISREG(current.st_mode)
                    and not stat.S_ISLNK(current.st_mode)
                    and (int(current.st_dev), int(current.st_ino)) == entry.identity
                    and int(current.st_size) == 0
                ):
                    try:
                        entry.path.unlink()
                    except OSError:
                        pass
            if entry.fd is not None:
                try:
                    os.close(entry.fd)
                except OSError:
                    pass

    def remove_claimed_files(self) -> None:
        """Remove only sidecar paths that still name this guard's claimed inode."""
        for entry in self.entries:
            if not entry.claimed or entry.identity is None:
                continue
            try:
                current = entry.path.lstat()
            except OSError:
                continue
            if (
                stat.S_ISREG(current.st_mode)
                and not stat.S_ISLNK(current.st_mode)
                and (int(current.st_dev), int(current.st_ino)) == entry.identity
            ):
                try:
                    entry.path.unlink()
                except OSError:
                    pass


def validate_sqlite_sidecars(
    db_path: Path, *, require_absent: bool = False
) -> dict[str, tuple[int, int]]:
    """Validate existing sidecars without creating, deleting, or chmodding them."""
    with SQLITE_OPEN_LOCK:
        guard = SQLiteSidecarGuard.acquire(Path(db_path), claim_missing=False)
        try:
            existing = {
                str(entry.path): entry.identity
                for entry in guard.entries
                if entry.identity is not None
            }
            if require_absent and existing:
                joined = ", ".join(sorted(existing))
                raise NamespacePathError(
                    f"unmapped SQLite sidecar already exists: {joined}"
                )
            return {path: identity for path, identity in existing.items() if identity}
        finally:
            guard.close(clean_unused_claims=False)


_SQLiteFileState = tuple[int, int, int, int, int, str]
SQLiteStorageSnapshot = dict[str, _SQLiteFileState | None]


def sqlite_storage_snapshot(db_path: Path) -> SQLiteStorageSnapshot:
    """Hash SQLite main/sidecar state through no-follow descriptors."""
    snapshot: SQLiteStorageSnapshot = {}
    nofollow = required_o_nofollow()
    cloexec = getattr(os, "O_CLOEXEC", 0)
    for candidate in (
        Path(db_path),
        *(Path(str(db_path) + suffix) for suffix in SQLITE_SIDECAR_SUFFIXES),
    ):
        key = str(candidate)
        try:
            before = candidate.lstat()
        except FileNotFoundError:
            snapshot[key] = None
            continue
        except OSError as exc:
            raise NamespacePathError(f"cannot inspect SQLite file: {candidate}") from exc
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) > 1
        ):
            raise NamespacePathError(
                f"SQLite file must be a single-link regular non-symlink: {candidate}"
            )
        if int(before.st_nlink) != 1:
            raise NamespacePathError(
                f"SQLite file changed while snapshotting: {candidate}"
            )
        try:
            fd = os.open(candidate, os.O_RDONLY | nofollow | cloexec)
        except FileNotFoundError as exc:
            raise NamespacePathError(
                f"SQLite file changed while snapshotting: {candidate}"
            ) from exc
        except OSError as exc:
            raise NamespacePathError(
                f"cannot safely snapshot SQLite file: {candidate}"
            ) from exc
        try:
            info = os.fstat(fd)
            digest = sha256()
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            try:
                current = candidate.lstat()
            except FileNotFoundError as exc:
                raise NamespacePathError(
                    f"SQLite file changed while snapshotting: {candidate}"
                ) from exc
            identity = int(info.st_dev), int(info.st_ino)
            if (
                not stat.S_ISREG(info.st_mode)
                or int(info.st_nlink) != 1
                or stat.S_ISLNK(current.st_mode)
                or identity != (int(current.st_dev), int(current.st_ino))
            ):
                raise NamespacePathError(
                    f"SQLite file changed while snapshotting: {candidate}"
                )
            snapshot[key] = (
                identity[0], identity[1], int(info.st_mode), int(info.st_size),
                int(info.st_mtime_ns), digest.hexdigest(),
            )
        finally:
            os.close(fd)
    return snapshot


def readonly_sqlite_mode(
    db_path: Path,
) -> tuple[bool, SQLiteStorageSnapshot]:
    """Choose immutable for quiescent DBs, WAL-aware only for complete WAL state."""
    snapshot = sqlite_storage_snapshot(db_path)
    wal_state = snapshot[str(db_path) + "-wal"]
    shm_state = snapshot[str(db_path) + "-shm"]
    wal_active = wal_state is not None and wal_state[3] != 0
    journal_state = snapshot[str(db_path) + "-journal"]
    if journal_state is not None and journal_state[3] != 0:
        raise NamespacePathError(
            f"cannot perform zero-write read with a rollback journal: {db_path}"
        )
    if wal_active and shm_state is None:
        raise NamespacePathError(
            f"cannot perform zero-write read with incomplete WAL state: {db_path}"
        )
    return not wal_active, snapshot


def _temporary_shadow_hook(_phase: str, _temporary: "GuardedTemporaryDirectory") -> None:
    """Test hook around private shadow creation and materialization."""


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _verify_trusted_temp_base(path: Path, fd: int) -> tuple[int, int]:
    """Verify a fixed system temporary directory through its held descriptor."""
    try:
        held = os.fstat(fd)
        current = path.lstat()
    except OSError as exc:
        raise NamespacePathError("trusted system temporary directory changed") from exc
    identity = int(held.st_dev), int(held.st_ino)
    if (
        not stat.S_ISDIR(held.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or identity != (int(current.st_dev), int(current.st_ino))
    ):
        raise NamespacePathError("trusted system temporary directory changed")
    mode = stat.S_IMODE(held.st_mode)
    if int(held.st_uid) != 0:
        raise NamespacePathError("trusted system temporary directory has unsafe owner")
    if mode & 0o022 and not mode & stat.S_ISVTX:
        raise NamespacePathError("trusted system temporary directory is not sticky")
    return identity


def _open_trusted_temp_base() -> tuple[Path, int, tuple[int, int]]:
    """Open a fixed POSIX system temp base, ignoring all temp environment vars."""
    nofollow = required_o_nofollow()
    flags = (
        os.O_RDONLY
        | nofollow
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    home = haunt_home().resolve()
    errors: list[Exception] = []
    for candidate in (Path("/private/tmp"), Path("/tmp")):
        try:
            # A trusted base is the literal fixed directory, not a symlink such as
            # /tmp -> /private/tmp on Darwin.
            current = candidate.lstat()
            if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
                continue
            resolved = candidate.resolve(strict=True)
            if resolved != candidate or _path_is_within(resolved, home):
                continue
            fd = os.open(candidate, flags)
            try:
                identity = _verify_trusted_temp_base(candidate, fd)
            except Exception:
                os.close(fd)
                raise
            return candidate, fd, identity
        except (OSError, NamespacePathError) as exc:
            errors.append(exc)
    raise NamespacePathError(
        "no safe fixed system temporary directory is available outside HAUNT_HOME"
    ) from (errors[-1] if errors else None)


@dataclass
class GuardedTemporaryDirectory:
    """A private temp directory anchored to a held trusted-base descriptor."""

    base_path: Path
    base_fd: int
    base_identity: tuple[int, int]
    child_name: str
    root_path: Path
    root_fd: int
    root_identity: tuple[int, int]
    source_primary: SQLitePrimaryGuard
    source_sidecars: SQLiteSidecarGuard
    source_snapshot: SQLiteStorageSnapshot
    shadow_name: str
    _closed: bool = False

    @property
    def name(self) -> str:
        return str(self.root_path)

    def verify_directory(self) -> None:
        if self._closed:
            raise NamespacePathError("private SQLite snapshot directory is closed")
        if _verify_trusted_temp_base(self.base_path, self.base_fd) != self.base_identity:
            raise NamespacePathError("trusted system temporary directory changed")
        try:
            held = os.fstat(self.root_fd)
            current = os.stat(
                self.child_name, dir_fd=self.base_fd, follow_symlinks=False
            )
        except OSError as exc:
            raise NamespacePathError(
                "private SQLite snapshot directory changed"
            ) from exc
        identity = int(held.st_dev), int(held.st_ino)
        if (
            not stat.S_ISDIR(held.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or identity != self.root_identity
            or (int(current.st_dev), int(current.st_ino)) != self.root_identity
            or stat.S_IMODE(held.st_mode) != DIR_MODE
            or stat.S_IMODE(current.st_mode) != DIR_MODE
        ):
            raise NamespacePathError("private SQLite snapshot directory changed")

    def verify_source(self) -> None:
        self.source_primary.verify()
        self.source_sidecars.verify()
        current = sqlite_storage_snapshot(self.source_primary.path)
        # WAL-index bytes are coordination state and can change when another
        # read-only connection attaches.  The copied transaction state is the
        # held main file plus WAL; the journal must also remain absent/empty.
        stable_paths = (
            str(self.source_primary.path),
            str(self.source_primary.path) + "-wal",
            str(self.source_primary.path) + "-journal",
        )
        if any(current[path] != self.source_snapshot[path] for path in stable_paths):
            raise NamespacePathError(
                "SQLite source changed while copying read snapshot"
            )

    def create_file(self, name: str) -> int:
        if not name or name != Path(name).name:
            raise NamespacePathError("invalid private SQLite snapshot filename")
        self.verify_directory()
        fd = os.open(
            name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | required_o_nofollow()
            | getattr(os, "O_CLOEXEC", 0),
            FILE_MODE,
            dir_fd=self.root_fd,
        )
        try:
            os.fchmod(fd, FILE_MODE)
            info = os.fstat(fd)
            current = os.stat(name, dir_fd=self.root_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(info.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or int(info.st_nlink) != 1
                or int(current.st_nlink) != 1
                or (int(info.st_dev), int(info.st_ino))
                != (int(current.st_dev), int(current.st_ino))
                or stat.S_IMODE(info.st_mode) != FILE_MODE
            ):
                raise NamespacePathError("private SQLite snapshot file changed")
            return fd
        except Exception:
            os.close(fd)
            try:
                os.unlink(name, dir_fd=self.root_fd)
            except OSError:
                pass
            raise

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            # SQLite can create any of these known siblings while recovering WAL.
            known = {
                self.shadow_name,
                *(self.shadow_name + suffix for suffix in SQLITE_SIDECAR_SUFFIXES),
            }
            try:
                entries = os.listdir(self.root_fd)
            except OSError:
                entries = []
            for name in entries:
                if name not in known:
                    continue
                try:
                    os.unlink(name, dir_fd=self.root_fd)
                except FileNotFoundError:
                    pass
                except IsADirectoryError:
                    try:
                        os.rmdir(name, dir_fd=self.root_fd)
                    except OSError:
                        pass
            try:
                os.fsync(self.root_fd)
            except OSError:
                pass
            try:
                current = os.stat(
                    self.child_name, dir_fd=self.base_fd, follow_symlinks=False
                )
            except OSError:
                current = None
            if current is not None and (
                stat.S_ISDIR(current.st_mode)
                and (int(current.st_dev), int(current.st_ino)) == self.root_identity
            ):
                try:
                    os.rmdir(self.child_name, dir_fd=self.base_fd)
                    os.fsync(self.base_fd)
                except OSError:
                    pass
        finally:
            self.source_sidecars.close(clean_unused_claims=False)
            self.source_primary.close()
            for fd in (self.root_fd, self.base_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass


def _new_guarded_temporary_directory(
    primary: SQLitePrimaryGuard,
    sidecars: SQLiteSidecarGuard,
    snapshot: SQLiteStorageSnapshot,
) -> GuardedTemporaryDirectory:
    base_fd = -1
    root_fd = -1
    child_name = ""
    try:
        base_path, base_fd, base_identity = _open_trusted_temp_base()
        nofollow = required_o_nofollow()
        for _attempt in range(8):
            child_name = f".haunt-sqlite-read-{secrets.token_hex(16)}"
            if _path_is_within(base_path / child_name, haunt_home().resolve()):
                continue
            try:
                os.mkdir(child_name, DIR_MODE, dir_fd=base_fd)
                break
            except FileExistsError:
                continue
        else:
            raise NamespacePathError("cannot claim private SQLite snapshot directory")
        root_fd = os.open(
            child_name,
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=base_fd,
        )
        os.fchmod(root_fd, DIR_MODE)
        held = os.fstat(root_fd)
        temporary = GuardedTemporaryDirectory(
            base_path=base_path,
            base_fd=base_fd,
            base_identity=base_identity,
            child_name=child_name,
            root_path=base_path / child_name,
            root_fd=root_fd,
            root_identity=(int(held.st_dev), int(held.st_ino)),
            source_primary=primary,
            source_sidecars=sidecars,
            source_snapshot=snapshot,
            shadow_name=primary.path.name,
        )
        temporary.verify_directory()
        os.fsync(base_fd)
        return temporary
    except Exception:
        if root_fd >= 0:
            try:
                os.close(root_fd)
            except OSError:
                pass
        if child_name:
            try:
                os.rmdir(child_name, dir_fd=base_fd)
            except OSError:
                pass
        if base_fd >= 0:
            os.close(base_fd)
        primary.close()
        sidecars.close(clean_unused_claims=False)
        raise


def temporary_sqlite_shadow(
    db_path: Path, snapshot: SQLiteStorageSnapshot
) -> tuple[GuardedTemporaryDirectory, Path]:
    """Copy a stable main/WAL byte snapshot outside HAUNT_HOME for safe reading."""
    primary = SQLitePrimaryGuard.acquire(db_path, create_missing=False)
    try:
        sidecars = SQLiteSidecarGuard.acquire(db_path, claim_missing=False)
    except Exception:
        primary.close()
        raise
    temporary = _new_guarded_temporary_directory(
        primary, sidecars, snapshot
    )
    shadow = temporary.root_path / db_path.name
    try:
        temporary.verify_source()
        _temporary_shadow_hook("created", temporary)
        sources: list[tuple[int, Path, str]] = [
            (primary.fd, db_path, db_path.name)
        ]
        wal_path = Path(str(db_path) + "-wal")
        wal_entry = next(
            entry for entry in sidecars.entries if entry.path == wal_path
        )
        if snapshot.get(str(wal_path)) is not None:
            if wal_entry.fd is None:
                raise NamespacePathError(
                    f"SQLite source changed while copying read snapshot: {wal_path}"
                )
            sources.append((wal_entry.fd, wal_path, db_path.name + "-wal"))
        for held_fd, source, destination_name in sources:
            expected = snapshot.get(str(source))
            if expected is None:
                raise NamespacePathError(
                    f"SQLite source changed while copying read snapshot: {source}"
                )
            source_fd = os.dup(held_fd)
            os.lseek(source_fd, 0, os.SEEK_SET)
            try:
                opened = os.fstat(source_fd)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or int(opened.st_nlink) != 1
                    or (int(opened.st_dev), int(opened.st_ino))
                    != (expected[0], expected[1])
                ):
                    raise NamespacePathError(
                        f"SQLite source changed while copying read snapshot: {source}"
                    )
                destination_fd = temporary.create_file(destination_name)
                try:
                    digest = sha256()
                    copied = 0
                    while True:
                        chunk = os.read(source_fd, 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        copied += len(chunk)
                        view = memoryview(chunk)
                        while view:
                            written = os.write(destination_fd, view)
                            view = view[written:]
                    os.fsync(destination_fd)
                    after = os.fstat(source_fd)
                    if (
                        (int(after.st_dev), int(after.st_ino))
                        != (expected[0], expected[1])
                        or int(after.st_nlink) != 1
                        or copied != expected[3]
                        or digest.hexdigest() != expected[5]
                    ):
                        raise NamespacePathError(
                            "SQLite source changed while copying read snapshot: "
                            f"{source}"
                        )
                finally:
                    os.close(destination_fd)
            finally:
                os.close(source_fd)
        os.fsync(temporary.root_fd)
        temporary.verify_source()
        temporary.verify_directory()
        _temporary_shadow_hook("copied", temporary)
        temporary.verify_source()
        temporary.verify_directory()
        return temporary, shadow
    except Exception:
        temporary.cleanup()
        raise


def materialize_sqlite_shadow(
    db_path: Path, temporary: GuardedTemporaryDirectory | None = None
) -> None:
    """Recover a private WAL snapshot into its main file through guarded opens."""
    if temporary is not None:
        temporary.verify_source()
        temporary.verify_directory()
        _temporary_shadow_hook("before_materialize", temporary)
        temporary.verify_source()
        temporary.verify_directory()
    primary = SQLitePrimaryGuard.acquire(db_path, create_missing=False)
    try:
        sidecars = SQLiteSidecarGuard.acquire(db_path, claim_missing=True)
    except Exception:
        primary.close()
        raise
    conn: sqlite3.Connection | None = None
    try:
        primary.verify()
        sidecars.verify()
        before = _descriptor_identities()
        conn = sqlite3.connect(str(db_path))
        _verify_sqlite_primary_open(before, primary, sidecars)
        primary.verify()
        sidecars.verify()
        checkpoint = conn.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
        if (
            checkpoint is None
            or int(checkpoint[0]) != 0
            or int(checkpoint[1]) <= 0
            or int(checkpoint[2]) != int(checkpoint[1])
        ):
            raise NamespacePathError(
                f"cannot materialize private SQLite WAL snapshot: {db_path}"
            )
        truncated = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if truncated is None or int(truncated[0]) != 0:
            raise NamespacePathError(
                f"cannot truncate private SQLite WAL snapshot: {db_path}"
            )
        primary.verify()
        sidecars.verify()
        if temporary is not None:
            temporary.verify_source()
            temporary.verify_directory()
            _temporary_shadow_hook("after_materialize", temporary)
            temporary.verify_source()
            temporary.verify_directory()
    finally:
        if conn is not None:
            conn.close()
        sidecars.close(clean_unused_claims=True)
        primary.close()


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
    physical = validate_namespace_db_paths(mapped_paths, expected=expected_paths)
    for mapped_path in dict.fromkeys(mapped_paths):
        validate_sqlite_sidecars(Path(mapped_path))
    return physical


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
        primary: SQLitePrimaryGuard | None = None
        sidecars: SQLiteSidecarGuard | None = None
        read_primary: SQLitePrimaryGuard | None = None
        read_sidecars: SQLiteSidecarGuard | None = None
        temporary: GuardedTemporaryDirectory | None = None
        locked = False
        storage_before: SQLiteStorageSnapshot | None = None
        storage_changed = False
        result: tuple[str, Path, int, int] | None = None
        try:
            SQLITE_OPEN_LOCK.acquire()
            locked = True
            primary = SQLitePrimaryGuard.acquire(path, create_missing=False)
            sidecars = SQLiteSidecarGuard.acquire(path, claim_missing=False)
            descriptors_before = _descriptor_identities()
            immutable_mode, storage_before = readonly_sqlite_mode(path)
            read_path = path
            read_primary = primary
            read_sidecars = sidecars
            if not immutable_mode:
                temporary, read_path = temporary_sqlite_shadow(path, storage_before)
                if sqlite_storage_snapshot(path) != storage_before:
                    storage_changed = True
                    raise NamespacePathError(
                        "registry changed while creating a zero-write read snapshot"
                    )
                materialize_sqlite_shadow(read_path, temporary)
                read_primary = SQLitePrimaryGuard.acquire(
                    read_path, create_missing=False
                )
                read_sidecars = SQLiteSidecarGuard.acquire(
                    read_path, claim_missing=False
                )
                descriptors_before = _descriptor_identities()
            conn = sqlite3.connect(
                f"{read_path.resolve().as_uri()}?mode=ro&immutable=1", uri=True
            )
            conn.row_factory = sqlite3.Row
            _verify_sqlite_primary_open(
                descriptors_before, read_primary, read_sidecars
            )
            read_primary.verify()
            read_sidecars.verify()
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
            if storage_before is not None:
                try:
                    storage_changed = sqlite_storage_snapshot(path) != storage_before
                except NamespacePathError:
                    storage_changed = True
            if sidecars is not None:
                sidecars.close(clean_unused_claims=False)
            if primary is not None:
                primary.close()
            if read_sidecars is not None and read_sidecars is not sidecars:
                read_sidecars.close(clean_unused_claims=True)
            if read_primary is not None and read_primary is not primary:
                read_primary.close()
            if temporary is not None:
                temporary.cleanup()
            if locked:
                SQLITE_OPEN_LOCK.release()
        if storage_changed or before != _registry_fingerprint():
            continue
        if result is None:
            _forget_registered_alias(name)
            return None
        if _remember_registered_alias(name, *result, fingerprint=before):
            return result[0], result[1]
    # Repeated concurrent migrations: fail closed instead of caching or
    # returning a value from a registry snapshot already known to be stale.
    _forget_registered_alias(name)
    raise NamespacePathError(
        "namespace registry changed repeatedly during alias resolution"
    )


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
        authority = parsed.netloc.rsplit("@", 1)[-1]
        if authority.startswith("["):
            if "]" not in authority:
                return None
            try:
                bracketed = ipaddress.ip_address(hostname)
            except ValueError:
                return None
            if bracketed.version != 6:
                return None
        elif "[" in authority or "]" in authority:
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


def disambiguate_namespace_label(label: str, discriminator: str) -> str:
    """Append a digest of *discriminator* to *label*.

    The digest depends on nothing but *discriminator*, so a repository that
    needs disambiguating derives the same label on every inference and on
    every machine. The result stays inside safe_name()'s 80-character budget
    so it survives registry lookup and namespace_db_path() unshortened.
    """
    digest = sha256(discriminator.encode("utf-8")).hexdigest()[:10]
    return f"{safe_name(label)[:69].rstrip('-.')}-{digest}"


@dataclass(frozen=True)
class _RepoRegistryMatch:
    """What the registry knows about a repository and the label it would mint."""

    namespace: str | None
    mint_label_taken: bool


def _mint_label_is_taken(
    mint_label: str,
    rows: list[sqlite3.Row],
    bindings: list[sqlite3.Row],
) -> bool:
    """Report that *mint_label* already belongs to a different repository.

    Only consulted once the registry has been searched for the current
    repository and come back empty, so any registration naming a repository
    here names some other one. A blank ``repo_path`` with no binding row
    names no repository at all and so is not evidence of another owner --
    the same rule, for the same reason, that keeps
    _registered_namespace_for_repo() from matching such a row.
    """
    if bindings:
        return True
    normalized = normalize_namespace_label(mint_label)
    return any(
        normalize_namespace_label(str(row["name"])) == normalized
        and str(row["repo_path"] or "").strip()
        for row in rows
    )


def _registered_namespace_for_repo(
    *,
    remote_identity: str | None,
    repo_root: Path | None,
    mint_label: str | None = None,
) -> _RepoRegistryMatch:
    """Preserve an existing namespace already registered to this repository.

    Reports separately, in ``mint_label_taken``, whether *mint_label* -- the
    label the caller would otherwise mint -- is already registered to a
    different repository, so the caller can fork instead of opening another
    repository's database.

    Never matches a registry row whose ``repo_path`` is blank, even when its
    name equals the checkout's directory basename. A blank row stores nothing
    tying it to a repository, so that would be a coincidence-of-labels guess
    (`notes`, `api`, `app` recur across unrelated clones) and a false positive
    would silently commingle two repositories' memory with no clean undo.
    The caller instead mints a fresh identity-derived namespace -- one honest
    fork per legacy blank row, not a growing one, because register_namespace()
    writes the repository_bindings row immediately. Healing the resulting
    split is backlog C3's operator-invoked, dry-run-first, reversible job.
    """
    path = registry_path()
    if not path.is_file():
        return _RepoRegistryMatch(None, False)
    conn: sqlite3.Connection | None = None
    primary: SQLitePrimaryGuard | None = None
    sidecars: SQLiteSidecarGuard | None = None
    read_primary: SQLitePrimaryGuard | None = None
    read_sidecars: SQLiteSidecarGuard | None = None
    temporary: GuardedTemporaryDirectory | None = None
    locked = False
    storage_before: SQLiteStorageSnapshot | None = None
    candidate: str | None = None
    rows: list[sqlite3.Row] = []
    bindings: list[sqlite3.Row] = []
    try:
        SQLITE_OPEN_LOCK.acquire()
        locked = True
        primary = SQLitePrimaryGuard.acquire(path, create_missing=False)
        sidecars = SQLiteSidecarGuard.acquire(path, claim_missing=False)
        descriptors_before = _descriptor_identities()
        immutable_mode, storage_before = readonly_sqlite_mode(path)
        read_path = path
        read_primary = primary
        read_sidecars = sidecars
        if not immutable_mode:
            temporary, read_path = temporary_sqlite_shadow(path, storage_before)
            if sqlite_storage_snapshot(path) != storage_before:
                raise NamespacePathError(
                    "registry changed while creating a zero-write read snapshot"
                )
            materialize_sqlite_shadow(read_path, temporary)
            read_primary = SQLitePrimaryGuard.acquire(
                read_path, create_missing=False
            )
            read_sidecars = SQLiteSidecarGuard.acquire(
                read_path, claim_missing=False
            )
            descriptors_before = _descriptor_identities()
        conn = sqlite3.connect(
            f"{read_path.resolve().as_uri()}?mode=ro&immutable=1", uri=True
        )
        conn.row_factory = sqlite3.Row
        _verify_sqlite_primary_open(descriptors_before, read_primary, read_sidecars)
        read_primary.verify()
        read_sidecars.verify()
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
                    candidate = str(row["canonical_label"])
            if candidate is None and repo_root:
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
                    candidate = str(row["canonical_label"])
        if candidate is None:
            rows = conn.execute("SELECT name, repo_path FROM namespaces").fetchall()
            if mint_label and {"namespace_aliases", "repository_bindings"} <= tables:
                bindings = conn.execute(
                    """
                    SELECT b.repository_identity, b.repo_path
                    FROM namespace_aliases a
                    JOIN repository_bindings b ON b.namespace_id=a.namespace_id
                    WHERE a.normalized_label=?
                    """,
                    (normalize_namespace_label(mint_label),),
                ).fetchall()
    except sqlite3.Error:
        return _RepoRegistryMatch(None, False)
    finally:
        if conn is not None:
            conn.close()
        try:
            storage_changed = bool(
                storage_before is not None
                and sqlite_storage_snapshot(path) != storage_before
            )
        except NamespacePathError:
            storage_changed = True
        if sidecars is not None:
            sidecars.close(clean_unused_claims=False)
        if primary is not None:
            primary.close()
        if read_sidecars is not None and read_sidecars is not sidecars:
            read_sidecars.close(clean_unused_claims=True)
        if read_primary is not None and read_primary is not primary:
            read_primary.close()
        if temporary is not None:
            temporary.cleanup()
        if locked:
            SQLITE_OPEN_LOCK.release()
    if storage_changed:
        raise NamespacePathError(
            "namespace registry changed during repository resolution"
        )
    if candidate is not None:
        return _RepoRegistryMatch(candidate, False)
    resolved_root = repo_root.resolve() if repo_root else None
    for row in rows:
        stored = str(row["repo_path"] or "").strip()
        if not stored:
            # Intentional: see this function's docstring for why a blank
            # repo_path is never treated as a match, even when only one
            # such row exists.
            continue
        if remote_identity and repository_identity(stored) == remote_identity:
            return _RepoRegistryMatch(safe_name(str(row["name"])), False)
        if resolved_root and repository_identity(stored) is None:
            try:
                if Path(stored).expanduser().resolve() == resolved_root:
                    return _RepoRegistryMatch(safe_name(str(row["name"])), False)
            except OSError:
                continue
    taken = mint_label is not None and _mint_label_is_taken(
        mint_label, rows, bindings
    )
    return _RepoRegistryMatch(None, taken)


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


def infer_namespace_context(cwd: Path | None = None) -> tuple[str, str | None]:
    """Infer the namespace together with the repository to record for it.

    Returns ``(namespace, repo_path)``. ``repo_path`` is a value safe to pass
    straight through to ``register_namespace()`` / ``Store(..., repo_path=...)``:
    the repository root when *cwd* sits inside a git working tree, or
    ``None`` when it does not. Callers that construct a ``Store`` must pass
    ``repo_path`` through even when it is ``None`` -- that is how namespace
    creation "explicitly records that there was none" instead of silently
    discarding the git context computed here (see backlog C1). Explicit
    selection (``HAUNT_NAMESPACE``) never auto-binds a repository: it is a
    deliberate override, not an inference.
    """
    env = os.environ.get("HAUNT_NAMESPACE")
    if env:
        registered = _registered_alias(env)
        return (registered[0] if registered else safe_name(env)), None
    if cwd is None:
        proj = os.environ.get("CURSOR_PROJECT_DIR") or os.environ.get("CLAUDE_PROJECT_DIR")
        root = Path(proj).expanduser().resolve() if proj else Path.cwd().resolve()
    else:
        root = cwd.resolve()
    remote_url, repo_root = _git_repo_context(root)
    identity = repository_identity(remote_url)
    # Both derivations below are lossy -- the remote one rewrites every "/"
    # to "-", the fallback keeps only a basename -- so two unrelated
    # repositories can derive one label (github.com/acme/foo-bar against
    # github.com/acme-foo/bar; any two checkouts named api). Pair the label
    # each would mint with the strongest identifier that tells this
    # repository apart from whoever else could mint it: a remote identity is
    # the same on every machine, while a remote-less checkout has nothing
    # portable but its own path.
    mint_label: str | None = None
    discriminator = ""
    if identity:
        mint_label, discriminator = namespace_for_repo_identity(identity), identity
    elif repo_root:
        mint_label, discriminator = safe_name(repo_root.name), str(repo_root)
    match = _registered_namespace_for_repo(
        remote_identity=identity,
        repo_root=repo_root,
        mint_label=mint_label,
    )
    repo_path = str(repo_root) if repo_root is not None else None
    if mint_label is not None and match.mint_label_taken:
        # The label is another repository's already. Fork: two namespaces are
        # a visible, reversible split, while commingled memory has no clean
        # undo and no signal that it happened.
        #
        # This read cannot be the ownership decision: two repositories that
        # infer before either registers both see the label free here. It is
        # the fast path, and register_namespace() repeats the test inside the
        # transaction that publishes and binds, forking to this same label
        # because the digest depends only on *discriminator*.
        #
        # The two paths agree only while this fork target is free. If a third
        # repository already owns it, arriving here (sequentially) forks again
        # off the already-forked label, whereas losing the race there has no
        # candidate left and fails closed. See register_namespace_context().
        mint_label = disambiguate_namespace_label(mint_label, discriminator)
    registered = match.namespace
    if registered:
        # _registered_namespace_for_repo() only returns a name here via an
        # exact repository_bindings match or a legacy row whose repo_path
        # is already known to correspond to this repository -- never a
        # blank-repo_path row (see that function's docstring for why a
        # blank row is never treated as a match). Returning repo_path here
        # regardless still matters: it is what lets the caller's
        # Store()/register_namespace() call create the repository_bindings
        # row the *first* time a legacy, pre-fix registration is confirmed
        # (e.g. by remote identity), and keeps it fresh afterward.
        return registered, repo_path
    if identity:
        # No exact match was found above, including among blank-repo_path
        # rows -- deliberately; see _registered_namespace_for_repo(). This
        # mints (or re-derives) the identity-formula name, which is unique
        # per repository rather than shared, so at worst this costs one
        # honest, one-time fork for a repository whose only prior
        # registration predates repo_path tracking. Healing that split is
        # backlog C3, not this function.
        return mint_label, repo_path
    # No repository identity was found. Never let a bare directory silently
    # mint (or keep re-targeting) a namespace (backlog C2):
    #  - the home directory must never become a namespace name, even if a
    #    stale one already exists for it (that is the bug, not data worth
    #    preserving going forward);
    #  - any other non-repository directory may only *reuse* a namespace
    #    already registered under its exact basename -- never mint a new
    #    one. This is what keeps a legitimately in-use, directory-derived
    #    namespace (e.g. one created before this fix) resolving to itself.
    if _is_user_home(root):
        return "default", None
    if repo_root:
        return mint_label, repo_path
    if root.name and root.name not in {".", "/", ""}:
        candidate = safe_name(root.name)
        existing = _registered_alias(candidate)
        if existing:
            return existing[0], None
    return "default", None


def infer_namespace(cwd: Path | None = None) -> str:
    """Infer from remote identity, preserving a matching legacy registration."""
    return infer_namespace_context(cwd)[0]


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
    """Tighten only verified SQLite files, never following a sidecar symlink."""
    validate_sqlite_sidecars(path)
    nofollow = required_o_nofollow()
    cloexec = getattr(os, "O_CLOEXEC", 0)
    for extra in (
        path,
        *(Path(str(path) + suffix) for suffix in SQLITE_SIDECAR_SUFFIXES),
    ):
        try:
            before = extra.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise NamespacePathError(f"cannot inspect SQLite file: {extra}") from exc
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
        ):
            raise NamespacePathError(
                f"SQLite file must be a single-link regular non-symlink: {extra}"
            )
        try:
            fd = os.open(extra, os.O_RDONLY | nofollow | cloexec)
        except OSError as exc:
            raise NamespacePathError(f"cannot safely open SQLite file: {extra}") from exc
        try:
            opened = os.fstat(fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or int(opened.st_nlink) != 1
                or (int(opened.st_dev), int(opened.st_ino))
                != (int(before.st_dev), int(before.st_ino))
            ):
                raise NamespacePathError(
                    f"SQLite file physical identity changed before chmod: {extra}"
                )
            if not _is_user_home(extra) and (opened.st_mode & 0o777) != FILE_MODE:
                try:
                    os.fchmod(fd, FILE_MODE)
                except (AttributeError, OSError) as exc:
                    raise NamespacePathError(
                        f"cannot safely tighten SQLite file mode: {extra}"
                    ) from exc
        finally:
            os.close(fd)


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
    def _tighten_database(db_path: Path) -> None:
        candidates = (
            db_path,
            *(Path(str(db_path) + suffix) for suffix in SQLITE_SIDECAR_SUFFIXES),
        )
        before_modes: dict[Path, int] = {}
        for candidate in candidates:
            try:
                info = candidate.lstat()
            except OSError:
                continue
            before_modes[candidate] = info.st_mode & 0o777
        tighten_db_files(db_path)
        for candidate, before_mode in before_modes.items():
            try:
                after_mode = candidate.lstat().st_mode & 0o777
            except OSError:
                continue
            if before_mode != after_mode:
                changed.append(str(candidate))

    registry = home / "registry.db"
    if registry.exists():
        _tighten_database(registry)
    else:
        validate_sqlite_sidecars(registry, require_absent=True)
    ns_dir = home / "namespaces"
    if ns_dir.is_dir():
        entries = list(ns_dir.iterdir())
        databases = {
            entry
            for entry in entries
            if entry.name.endswith(".db") and not entry.name.startswith(".haunt-claim-")
        }
        for entry in entries:
            for suffix in SQLITE_SIDECAR_SUFFIXES:
                if entry.name.endswith(f".db{suffix}"):
                    main = Path(str(entry)[: -len(suffix)])
                    if main not in databases:
                        raise NamespacePathError(
                            f"unmapped SQLite sidecar already exists: {entry}"
                        )
        for database in sorted(databases):
            _tighten_database(database)
    return changed


def ensure_layout() -> Path:
    home = haunt_home()
    mkdir_private(home)
    validate_namespace_root(create=True)
    for p in (bin_dir(), models_dir()):
        mkdir_private(p)
    return home
