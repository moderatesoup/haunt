"""L33: never open a descriptor on a sidecar SQLite is locking.

POSIX advisory locks are per (process, inode). Closing *any* descriptor for an
inode releases every `fcntl` lock the process holds on it, and SQLite defends
against this only for descriptors it opened itself -- it cannot see one Python
opened. Its WAL read, write and checkpoint locks are byte-range locks on
`<db>-shm`, so opening and closing haunt's own fd on that file while a
connection is live silently drops that connection's locks.

Two places did exactly that, both after `PRAGMA journal_mode=WAL` had already
put the connection into WAL mode: `validate_sqlite_sidecars` and, right after
it, `tighten_db_files`, which opens every sidecar to check and chmod it.

The tests assert the property structurally rather than trying to observe a
dropped lock. A dropped advisory lock has no direct observable in Python, and
its consequence -- `SQLITE_PROTOCOL` at commit -- needs a concurrent writer and
a large corpus to surface (that is L32, whose regression tests live next door).
What is checkable, cheaply and deterministically, is that no descriptor is
opened on those paths at all.
"""

from __future__ import annotations

import os

import pytest

LOCKED = ("-wal", "-shm")


def _watch_opens(monkeypatch, recorder):
    """Record every path os.open is called with, then delegate."""
    real = os.open

    def spy(path, flags, *args, **kwargs):
        recorder.append(str(path))
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy)


def test_the_descriptor_free_validator_opens_nothing(tmp_path, monkeypatch):
    """It is only useful if it genuinely takes no descriptor."""
    from haunt.paths import validate_sqlite_sidecars_no_descriptors

    db = tmp_path / "x.db"
    db.write_bytes(b"")
    for suffix in LOCKED + ("-journal",):
        (tmp_path / f"x.db{suffix}").write_bytes(b"")

    opened: list[str] = []
    _watch_opens(monkeypatch, opened)
    found = validate_sqlite_sidecars_no_descriptors(db)

    assert not opened, f"opened {opened} -- the point is to open nothing"
    assert len(found) == 3, "it still has to see and identify the sidecars"


def test_the_descriptor_free_validator_still_rejects_a_planted_symlink(tmp_path):
    """Dropping the descriptor must not drop the symlink defence."""
    from haunt.paths import NamespacePathError, validate_sqlite_sidecars_no_descriptors

    db = tmp_path / "x.db"
    db.write_bytes(b"")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_bytes(b"")
    (tmp_path / "x.db-wal").symlink_to(elsewhere)

    with pytest.raises(NamespacePathError, match="regular non-symlink"):
        validate_sqlite_sidecars_no_descriptors(db)


def test_the_descriptor_free_validator_rejects_a_hardlinked_sidecar(tmp_path):
    from haunt.paths import NamespacePathError, validate_sqlite_sidecars_no_descriptors

    db = tmp_path / "x.db"
    db.write_bytes(b"")
    target = tmp_path / "x.db-wal"
    target.write_bytes(b"")
    os.link(target, tmp_path / "second-name")

    with pytest.raises(NamespacePathError, match="exactly one filesystem link"):
        validate_sqlite_sidecars_no_descriptors(db)


def test_the_two_post_connect_calls_touch_no_locked_sidecar(haunt_env, monkeypatch):
    """The load-bearing one, against a genuinely live WAL connection.

    Scoped to one database on purpose. A whole `Store()` open touches several
    (registry, the fresh-namespace claim file, then the namespace itself), and
    each has its own legitimate pre-connect phase where claiming `-wal`/`-shm`
    is correct -- there are no locks to drop yet, and claiming the name is what
    stops a symlink being planted there. Watching all of it at once cannot tell
    those apart from the thing under test. So: open a store, let it settle into
    WAL, then run exactly the two calls that used to sit after
    `PRAGMA journal_mode=WAL` and watch only that database's sidecars.
    """
    from pathlib import Path

    from haunt.paths import tighten_db_files, validate_sqlite_sidecars_no_descriptors
    from haunt.store import Store

    with Store("l33-live") as store:
        store.observe("a row so the database is real", role="user", tier="episodic")
        db = Path(store.db_path)
        assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

        opened: list[str] = []
        _watch_opens(monkeypatch, opened)
        validate_sqlite_sidecars_no_descriptors(db)
        tighten_db_files(db, live_connection=True)

    mine = {str(db) + suffix for suffix in LOCKED}
    offenders = sorted(set(opened) & mine)
    assert not offenders, (
        "opened a descriptor on a sidecar this live connection holds locks "
        f"on: {offenders}"
    )


def test_tighten_skips_locked_sidecars_only_when_told_a_connection_is_live(
    tmp_path, monkeypatch
):
    """Off the connect path the stricter descriptor check must still run.

    `live_connection` is a narrowing, not a new default: a database nobody has
    open should still get the full fstat-versus-lstat treatment on every
    sidecar.
    """
    from haunt.paths import tighten_db_files

    db = tmp_path / "y.db"
    db.write_bytes(b"")
    for suffix in LOCKED + ("-journal",):
        (tmp_path / f"y.db{suffix}").write_bytes(b"")

    live: list[str] = []
    _watch_opens(monkeypatch, live)
    tighten_db_files(db, live_connection=True)
    assert not [p for p in live if p.endswith(LOCKED)], (
        "live_connection=True must not open -wal or -shm"
    )
    assert any(p.endswith("y.db") for p in live), "the main file is still tightened"

    cold: list[str] = []
    _watch_opens(monkeypatch, cold)
    tighten_db_files(db)
    assert [p for p in cold if p.endswith(LOCKED)], (
        "the default path still opens every sidecar; narrowing it everywhere "
        "would lose the fstat-versus-lstat identity check"
    )


def test_the_connect_path_uses_the_descriptor_free_validator():
    """A source guard, so the ordering cannot silently regress.

    `validate_sqlite_sidecars` after `journal_mode=WAL` is the exact shape that
    caused L33, and it reads as harmless.
    """
    import inspect

    from haunt import store as store_mod

    source = inspect.getsource(store_mod._configure_sqlite_connection) if hasattr(
        store_mod, "_configure_sqlite_connection"
    ) else inspect.getsource(store_mod)
    after_wal = source[source.index('PRAGMA journal_mode=WAL') :]
    window = after_wal[:1200]
    assert "validate_sqlite_sidecars_no_descriptors(path)" in window
    assert "live_connection=True" in window
