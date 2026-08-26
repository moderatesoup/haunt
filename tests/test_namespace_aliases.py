"""E3 canonical namespace identity, aliases, and explicit migration."""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from typer.testing import CliRunner

from haunt.cli import app
from haunt.paths import (
    NamespacePathError,
    SQLitePrimaryGuard,
    SQLiteSidecarGuard,
    infer_namespace,
    namespace_db_path,
    registry_path,
    repository_identity,
    resolve_namespace,
    sqlite_storage_snapshot,
    tighten_db_files,
)
from haunt.store import (
    AliasRetirementError,
    NamespaceCollisionError,
    NamespaceMigrationError,
    Store,
    UnknownNamespaceError,
    change_namespace_label,
    init_registry,
    list_namespace_rows,
    namespace_exists,
    open_existing,
    register_namespace,
    resolve_namespace_identity,
    retire_namespace_alias,
    undo_namespace_migration,
)


@pytest.fixture
def alias_home(tmp_path, monkeypatch):
    home = tmp_path / "haunt-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed

    embed.reset()
    init_registry()
    yield home
    embed.reset()


def _apply_change(old_label, new_label, **kwargs):
    kwargs.pop("apply", None)
    plan = change_namespace_label(old_label, new_label, apply=False, **kwargs)
    return change_namespace_label(
        old_label,
        new_label,
        apply=True,
        plan_digest=plan["plan_digest"],
        **kwargs,
    )


def test_fresh_identity_alias_rename_reuses_exact_database(alias_home):
    with Store("Original Name") as store:
        memory = store.observe("rename canary")
        original_id = store.namespace_id
        original_db = store.db_path

    dry = change_namespace_label("Original Name", "Moved Name", apply=False)
    assert dry["mode"] == "dry-run"
    assert dry["database_operation"] == "none"
    assert not namespace_exists("Moved Name")

    applied = _apply_change("Original Name", "Moved Name")
    replay = _apply_change("Original Name", "Moved Name")
    assert applied["db_path"] == replay["db_path"] == str(original_db)
    assert replay["idempotent"] is True
    assert namespace_db_path("Original Name") == original_db
    assert namespace_db_path("Moved Name") == original_db
    assert not (alias_home / "namespaces" / "Moved-Name.db").exists()
    for label in ("Original Name", "Moved Name", "moved name"):
        with open_existing(label) as store:
            assert store.namespace_id == original_id
            assert store.name == "Moved-Name"
            assert store.get_memory(memory.memory_id)["content"] == "rename canary"


def test_additive_upgrade_preserves_legacy_name_and_path(alias_home):
    old_db = alias_home / "namespaces" / "Legacy.DB.db"
    old_db.parent.mkdir(parents=True, exist_ok=True)
    old_db.touch()
    registry_path().unlink()
    conn = sqlite3.connect(registry_path())
    conn.execute(
        """CREATE TABLE namespaces(
               name TEXT PRIMARY KEY,repo_path TEXT,db_path TEXT NOT NULL,
               created_at TEXT NOT NULL,updated_at TEXT NOT NULL)"""
    )
    conn.execute(
        "INSERT INTO namespaces VALUES (?,?,?,?,?)",
        ("Legacy.DB", "/old/location", str(old_db), "t0", "t1"),
    )
    conn.commit()
    conn.close()

    init_registry()
    init_registry()
    identity = resolve_namespace_identity("legacy.db")
    assert identity is not None
    assert identity["canonical_label"] == "Legacy.DB"
    assert identity["db_path"] == str(old_db)
    conn = sqlite3.connect(registry_path())
    assert conn.execute("SELECT COUNT(*) FROM namespaces").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM namespace_identities").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM namespace_aliases").fetchone()[0] == 1
    conn.close()


def test_v3_identity_upgrade_records_physical_database_identity(tmp_path, monkeypatch):
    home = tmp_path / "v3-home"
    namespace_root = home / "namespaces"
    namespace_root.mkdir(parents=True)
    db = namespace_root / "v3.db"
    db.touch()
    registry = home / "registry.db"
    conn = sqlite3.connect(registry)
    conn.executescript(
        """
        CREATE TABLE namespaces(
            name TEXT PRIMARY KEY,repo_path TEXT,db_path TEXT NOT NULL,
            created_at TEXT NOT NULL,updated_at TEXT NOT NULL
        );
        CREATE TABLE registry_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE namespace_identities(
            namespace_id TEXT PRIMARY KEY,canonical_label TEXT NOT NULL,
            canonical_label_norm TEXT NOT NULL UNIQUE,db_path TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,updated_at TEXT NOT NULL
        );
        CREATE TABLE namespace_aliases(
            normalized_label TEXT PRIMARY KEY,label TEXT NOT NULL,
            namespace_id TEXT NOT NULL,is_canonical INTEGER NOT NULL,
            source_alias_norm TEXT,created_at TEXT NOT NULL
        );
        """
    )
    conn.execute("INSERT INTO registry_meta VALUES ('schema_version','3')")
    conn.execute(
        "INSERT INTO namespaces VALUES (?,?,?,?,?)",
        ("v3", None, str(db), "2025-01-01", "2025-01-01"),
    )
    conn.execute(
        "INSERT INTO namespace_identities VALUES (?,?,?,?,?,?)",
        ("v3-id", "v3", "v3", str(db), "2025-01-01", "2025-01-01"),
    )
    conn.execute(
        "INSERT INTO namespace_aliases VALUES (?,?,?,?,?,?)",
        ("v3", "v3", "v3-id", 1, None, "2025-01-01"),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")

    init_registry()
    conn = sqlite3.connect(registry)
    row = conn.execute(
        "SELECT db_device,db_inode FROM namespace_identities WHERE namespace_id='v3-id'"
    ).fetchone()
    version = conn.execute(
        "SELECT value FROM registry_meta WHERE key='schema_version'"
    ).fetchone()[0]
    conn.close()
    assert row == (db.stat().st_dev, db.stat().st_ino)
    assert version == "5"
    with Store("v3", create=False) as store:
        assert store.namespace_id == "v3-id"


def test_remote_forms_share_identity_but_same_leaf_other_remote_does_not(alias_home):
    first = register_namespace("first", "https://github.com/acme/api.git")
    second = register_namespace("clone", "git@github.com:acme/api.git")
    third = register_namespace("other", "ssh://git@gitlab.com/acme/api.git")
    assert first == second
    assert third != first
    assert resolve_namespace_identity("first")["namespace_id"] == resolve_namespace_identity("clone")["namespace_id"]
    assert resolve_namespace_identity("other")["namespace_id"] != resolve_namespace_identity("first")["namespace_id"]
    assert repository_identity("git@github.com:acme/api.git") == "github.com/acme/api"


def test_remote_identity_preserves_nondefault_ports_and_normalizes_defaults():
    assert repository_identity("https://git.example.com:443/acme/api.git") == (
        "git.example.com/acme/api"
    )
    assert repository_identity("http://git.example.com:80/acme/api.git") == (
        "git.example.com/acme/api"
    )
    assert repository_identity("ssh://git@git.example.com:22/acme/api.git") == (
        "git.example.com/acme/api"
    )
    assert repository_identity("git://git.example.com:9418/acme/api.git") == (
        "git.example.com/acme/api"
    )
    first = repository_identity("ssh://git@git.example.com:2222/acme/api.git")
    second = repository_identity("ssh://git@git.example.com:2223/acme/api.git")
    assert first == "git.example.com:2222/acme/api"
    assert second == "git.example.com:2223/acme/api"
    assert first != second


@pytest.mark.parametrize(
    "remote",
    [
        "https://[2001:db8::1/acme/api.git",
        "https://2001:db8::1]/acme/api.git",
        "https://[not-ipv6]/acme/api.git",
        "ssh://git@[2001:db8::1]:not-a-port/acme/api.git",
        "ssh://git@[2001:db8::1]:70000/acme/api.git",
    ],
)
def test_remote_identity_rejects_malformed_ipv6_and_ports(remote):
    assert repository_identity(remote) is None


def test_remote_identity_normalizes_valid_ipv6_ports():
    assert repository_identity(
        "https://[2001:DB8::1]:443/acme/api.git"
    ) == "[2001:db8::1]/acme/api"
    assert repository_identity(
        "ssh://git@[2001:DB8::1]:22/acme/api.git"
    ) == "[2001:db8::1]/acme/api"
    assert repository_identity(
        "ssh://git@[2001:DB8::1]:2222/acme/api.git"
    ) == "[2001:db8::1]:2222/acme/api"


def test_repository_move_binding_and_inference(alias_home, tmp_path, monkeypatch):
    old_root = tmp_path / "old" / "repo"
    new_root = tmp_path / "new" / "repo"
    old_root.mkdir(parents=True)
    new_root.mkdir(parents=True)
    register_namespace("before", str(old_root))
    _apply_change("before", "after", repository=str(new_root))

    import haunt.paths as paths

    monkeypatch.setattr(paths, "_git_repo_context", lambda root: (None, root.resolve()))
    assert infer_namespace(old_root) == "after"
    assert infer_namespace(new_root) == "after"


@pytest.mark.parametrize(
    "order",
    [
        ("late", "middle", "first"),
        ("first", "late", "middle"),
    ],
)
def test_duplicate_legacy_paths_migrate_deterministically_and_quiesce(
    tmp_path, monkeypatch, order
):
    home = tmp_path / "duplicate-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    (home / "namespaces").mkdir(parents=True)
    shared_db = home / "namespaces" / "shared-original.db"
    shared_db.touch()
    conn = sqlite3.connect(home / "registry.db")
    conn.execute(
        """CREATE TABLE namespaces(
               name TEXT PRIMARY KEY,repo_path TEXT,db_path TEXT NOT NULL,
               created_at TEXT NOT NULL,updated_at TEXT NOT NULL)"""
    )
    created = {"first": "2025-01-01", "middle": "2025-01-01", "late": "2026-01-01"}
    for label in order:
        conn.execute(
            "INSERT INTO namespaces VALUES (?,?,?,?,?)",
            (label, None, str(shared_db), created[label], created[label]),
        )
    conn.commit()
    conn.close()

    init_registry()
    identity = resolve_namespace_identity("late")
    assert identity["canonical_label"] == "first"
    assert {alias["label"] for alias in identity["aliases"]} == {
        "first", "middle", "late"
    }
    rows = list_namespace_rows()
    assert len(rows) == 1
    assert rows[0]["name"] == "first"
    assert set(rows[0]["aliases"]) == {"first", "middle", "late"}

    observer = sqlite3.connect(registry_path())
    before = observer.execute("PRAGMA data_version").fetchone()[0]
    init_registry()
    after = observer.execute("PRAGMA data_version").fetchone()[0]
    observer.close()
    assert after == before, "a completed registry migration must not write again"

    _apply_change("late", "renamed")
    assert resolve_namespace_identity("middle")["canonical_label"] == "renamed"
    retired = retire_namespace_alias("middle", apply=True)
    assert retired["retired"] is True
    init_registry()
    assert not namespace_exists("middle")
    assert len(list_namespace_rows()) == 1


def test_first_legacy_dry_run_is_byte_for_byte_read_only_then_apply_succeeds(
    tmp_path, monkeypatch
):
    home = tmp_path / "legacy-dry-home"
    namespaces = home / "namespaces"
    namespaces.mkdir(parents=True)
    legacy_db = namespaces / "Legacy.db"
    other_db = namespaces / "Other.db"
    orphan_db = namespaces / "orphan.db"
    for path in (legacy_db, other_db, orphan_db):
        path.touch()
    registry = home / "registry.db"
    conn = sqlite3.connect(registry)
    conn.execute(
        """CREATE TABLE namespaces(
               name TEXT PRIMARY KEY,repo_path TEXT,db_path TEXT NOT NULL,
               created_at TEXT NOT NULL,updated_at TEXT NOT NULL)"""
    )
    conn.executemany(
        "INSERT INTO namespaces VALUES (?,?,?,?,?)",
        [
            (
                "Legacy",
                "https://github.com/acme/legacy.git",
                str(legacy_db),
                "2025-01-01",
                "2025-01-01",
            ),
            (
                "Other",
                "https://github.com/acme/other.git",
                str(other_db),
                "2025-01-02",
                "2025-01-02",
            ),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")

    observer = sqlite3.connect(f"{registry.resolve().as_uri()}?mode=ro", uri=True)

    def logical_snapshot():
        schema = observer.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        ).fetchall()
        rows = observer.execute(
            "SELECT name,repo_path,db_path,created_at,updated_at FROM namespaces "
            "ORDER BY name"
        ).fetchall()
        return schema, rows, observer.execute("PRAGMA user_version").fetchone()[0]

    def file_snapshot():
        return {
            str(path.relative_to(home)): path.read_bytes()
            for path in sorted(home.rglob("*"))
            if path.is_file()
        }

    before_files = file_snapshot()
    before_logical = logical_snapshot()
    before_version = observer.execute("PRAGMA data_version").fetchone()[0]

    with pytest.raises(NamespaceCollisionError):
        change_namespace_label("Legacy", "Other", apply=False)
    with pytest.raises(NamespaceCollisionError):
        change_namespace_label(
            "Legacy",
            "unused",
            repository="git@github.com:acme/other.git",
            apply=False,
        )
    with pytest.raises(NamespaceCollisionError):
        change_namespace_label("Legacy", "orphan", apply=False)

    plan = change_namespace_label("Legacy", "Renamed", apply=False)
    assert plan["namespace_id"] is None
    assert plan["requires_registry_upgrade"] is True
    assert plan["canonical_before"] == "Legacy"
    assert plan["db_path"] == str(legacy_db)
    cli_plan = CliRunner().invoke(
        app, ["namespace", "migrate", "Legacy", "Renamed"]
    )
    assert cli_plan.exit_code == 0, cli_plan.output
    assert '"requires_registry_upgrade": true' in cli_plan.output

    assert file_snapshot() == before_files
    assert logical_snapshot() == before_logical
    assert observer.execute("PRAGMA data_version").fetchone()[0] == before_version
    observer.close()
    assert not Path(str(registry) + "-wal").exists()
    assert not Path(str(registry) + "-shm").exists()

    applied = change_namespace_label(
        "Legacy", "Renamed", apply=True, plan_digest=plan["plan_digest"]
    )
    assert applied["namespace_id"]
    assert applied["db_path"] == str(legacy_db)
    identity = resolve_namespace_identity("Renamed")
    assert identity is not None
    assert identity["namespace_id"] == applied["namespace_id"]
    assert identity["db_path"] == str(legacy_db)


def test_alias_and_repository_resolution_reads_committed_wal(
    alias_home, tmp_path, monkeypatch
):
    """Resolution must not use SQLite immutable mode, which ignores WAL."""
    register_namespace("wal-main")
    identity = resolve_namespace_identity("wal-main")
    writer = sqlite3.connect(registry_path())
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute(
        """INSERT INTO namespace_aliases(
               normalized_label,label,namespace_id,is_canonical,created_at
           ) VALUES (?,?,?,?,?)""",
        ("wal-alias", "wal-alias", identity["namespace_id"], 0, "t2"),
    )
    writer.execute(
        """INSERT INTO repository_bindings(
               binding_id,namespace_id,repository_identity,repo_path,label_norm,
               created_at,updated_at
           ) VALUES (?,?,?,?,?,?,?)""",
        (
            "wal-binding",
            identity["namespace_id"],
            "github.com/acme/wal",
            None,
            "wal-alias",
            "t2",
            "t2",
        ),
    )
    writer.commit()
    try:
        assert (registry_path().parent / "registry.db-wal").exists()
        assert resolve_namespace("wal-alias") == "wal-main"

        repo = tmp_path / "wal-clone"
        repo.mkdir()
        import haunt.paths as paths

        monkeypatch.setattr(
            paths,
            "_git_repo_context",
            lambda root: ("git@github.com:acme/wal.git", repo),
        )
        assert infer_namespace(repo) == "wal-main"
    finally:
        writer.close()


def test_collision_truncation_and_orphan_target_refused(alias_home):
    register_namespace("alpha")
    register_namespace("beta")
    with pytest.raises(NamespaceCollisionError):
        _apply_change("alpha", "BETA", action="alias")

    prefix = "x" * 80
    _apply_change("alpha", prefix + "one", action="alias")
    with pytest.raises(NamespaceCollisionError):
        _apply_change("beta", prefix + "two", action="alias")

    orphan = alias_home / "namespaces" / "orphan.db"
    orphan.touch()
    with pytest.raises(NamespaceCollisionError):
        _apply_change("alpha", "orphan", action="alias")


def test_retirement_checks_only_recorded_references_and_reports_caveat(alias_home):
    register_namespace("old", "https://github.com/acme/retire.git")
    _apply_change("old", "new", repository="git@github.com:acme/retire.git")
    check = retire_namespace_alias("old")
    assert check["safe"] is True
    assert "External editor/host" in check["operator_caveat"]
    retired = retire_namespace_alias("old", apply=True)
    assert retired["retired"] is True
    assert not namespace_exists("old")
    assert namespace_exists("new")
    with pytest.raises(AliasRetirementError):
        retire_namespace_alias("new", apply=True)


def test_rename_through_alias_rebinds_previous_canonical(alias_home, tmp_path, monkeypatch):
    register_namespace("old", "https://github.com/acme/through-alias.git")
    _apply_change("old", "bridge", action="alias")
    _apply_change("bridge", "new", action="rename")
    check = retire_namespace_alias("old")
    assert check["safe"] is True
    retire_namespace_alias("old", apply=True)

    repo = tmp_path / "through-alias"
    repo.mkdir()
    import haunt.paths as paths

    monkeypatch.setattr(
        paths,
        "_git_repo_context",
        lambda root: ("git@github.com:acme/through-alias.git", repo),
    )
    assert infer_namespace(repo) == "new"


def test_dependent_alias_blocks_retirement(alias_home):
    register_namespace("canonical")
    _apply_change("canonical", "bridge", action="alias")
    _apply_change("bridge", "dependent", action="alias")
    check = retire_namespace_alias("bridge")
    assert check["safe"] is False
    assert {b["kind"] for b in check["blockers"]} == {"dependent-alias"}


def test_rename_to_existing_alias_reroots_lineage_for_old_retirement(alias_home):
    register_namespace("a")
    _apply_change("a", "b", action="alias")
    before = resolve_namespace_identity("b")
    assert next(
        alias for alias in before["aliases"] if alias["normalized_label"] == "b"
    )["source_alias_norm"] == "a"

    _apply_change("a", "b", action="rename")
    after = resolve_namespace_identity("b")
    assert after["canonical_label"] == "b"
    assert next(
        alias for alias in after["aliases"] if alias["normalized_label"] == "b"
    )["source_alias_norm"] is None
    assert retire_namespace_alias("a")["safe"] is True
    assert retire_namespace_alias("a", apply=True)["retired"] is True
    assert not namespace_exists("a")
    assert namespace_exists("b")


def test_typo_read_does_not_create_registry_alias_or_database(alias_home):
    register_namespace("known")
    before = list_namespace_rows()
    with pytest.raises(UnknownNamespaceError):
        open_existing("knwon")
    assert list_namespace_rows() == before
    assert not (alias_home / "namespaces" / "knwon.db").exists()


def test_unmapped_physical_targets_cannot_gain_identity_or_authority(
    alias_home, tmp_path
):
    owner_remote = "https://github.com/acme/physical-owner.git"
    with Store("physical-owner", repo_path=owner_remote) as store:
        owner_db = store.db_path
        owner_id = store.namespace_id

    outside_db = tmp_path / "outside.db"
    outside_db.touch()
    attacks = {
        "symlink-target": lambda path: path.symlink_to(owner_db),
        "hardlink-target": lambda path: path.hardlink_to(owner_db),
        "unmapped-target": lambda path: path.touch(),
        "outside-target": lambda path: path.symlink_to(outside_db),
    }

    from haunt.mcp_server import MCPAuthority, MCPAuthorityError

    authority = MCPAuthority(
        bound_namespace="physical-owner",
        bound_namespace_id=owner_id,
    )
    assert authority.select("physical-owner") == "physical-owner"
    for label, build_attack in attacks.items():
        target = alias_home / "namespaces" / f"{label}.db"
        build_attack(target)
        with pytest.raises((NamespaceCollisionError, NamespacePathError)):
            Store(label)
        cli = CliRunner().invoke(
            app,
            [
                "namespace",
                "alias",
                "physical-owner",
                label,
                "--apply",
            ],
        )
        assert cli.exit_code == 2, cli.output
        assert "error:" in cli.output
        with pytest.raises((MCPAuthorityError, NamespacePathError)):
            authority.select(label)
        target.unlink()
        assert not namespace_exists(label)

    symlink_target = alias_home / "namespaces" / "symlink-target.db"
    symlink_target.symlink_to(owner_db)
    with pytest.raises((NamespaceCollisionError, NamespacePathError)):
        register_namespace("symlink-target", owner_remote)
    symlink_target.unlink()

    conn = sqlite3.connect(registry_path())
    assert conn.execute("SELECT COUNT(*) FROM namespace_identities").fetchone()[0] == 1
    conn.close()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "equivalent"])
def test_legacy_ambiguous_physical_database_paths_fail_closed(
    tmp_path, monkeypatch, kind
):
    home = tmp_path / f"ambiguous-{kind}"
    namespace_root = home / "namespaces"
    namespace_root.mkdir(parents=True)
    physical_db = namespace_root / "physical.db"
    physical_db.touch()
    if kind == "symlink":
        alternate = namespace_root / "alternate.db"
        alternate.symlink_to(physical_db)
        alternate_value = str(alternate)
    elif kind == "hardlink":
        alternate = namespace_root / "alternate.db"
        alternate.hardlink_to(physical_db)
        alternate_value = str(alternate)
    else:
        (namespace_root / "nested").mkdir()
        alternate_value = str(namespace_root / "nested" / ".." / "physical.db")

    registry = home / "registry.db"
    conn = sqlite3.connect(registry)
    conn.execute(
        """CREATE TABLE namespaces(
               name TEXT PRIMARY KEY,repo_path TEXT,db_path TEXT NOT NULL,
               created_at TEXT NOT NULL,updated_at TEXT NOT NULL)"""
    )
    conn.executemany(
        "INSERT INTO namespaces VALUES (?,?,?,?,?)",
        [
            ("first", None, str(physical_db), "2025-01-01", "2025-01-01"),
            ("second", None, alternate_value, "2025-01-02", "2025-01-02"),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")

    with pytest.raises(NamespacePathError):
        change_namespace_label("first", "third", apply=False)
    with pytest.raises(NamespacePathError):
        init_registry()
    conn = sqlite3.connect(registry)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "namespace_identities" not in tables
    assert "namespace_aliases" not in tables
    conn.close()


@pytest.mark.parametrize("kind", ["symlink-root", "outside-db"])
def test_unsafe_legacy_storage_dry_run_is_exactly_read_only(
    tmp_path, monkeypatch, kind
):
    home = tmp_path / f"unsafe-legacy-{kind}"
    home.mkdir()
    external = tmp_path / f"external-{kind}"
    external.mkdir()
    external_db = external / "legacy.db"
    external_db.touch()
    if kind == "symlink-root":
        (home / "namespaces").symlink_to(external, target_is_directory=True)
        stored_db = home / "namespaces" / "legacy.db"
    else:
        (home / "namespaces").mkdir()
        stored_db = external_db
    registry = home / "registry.db"
    conn = sqlite3.connect(registry)
    conn.execute(
        """CREATE TABLE namespaces(
               name TEXT PRIMARY KEY,repo_path TEXT,db_path TEXT NOT NULL,
               created_at TEXT NOT NULL,updated_at TEXT NOT NULL)"""
    )
    conn.execute(
        "INSERT INTO namespaces VALUES (?,?,?,?,?)",
        ("legacy", None, str(stored_db), "2025-01-01", "2025-01-01"),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")

    before_bytes = registry.read_bytes()
    conn = sqlite3.connect(f"{registry.resolve().as_uri()}?mode=ro", uri=True)
    before_schema = conn.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
    ).fetchall()
    before_rows = conn.execute("SELECT * FROM namespaces").fetchall()
    before_version = conn.execute("PRAGMA data_version").fetchone()[0]

    with pytest.raises(NamespacePathError):
        change_namespace_label("legacy", "new", apply=False)
    assert registry.read_bytes() == before_bytes
    assert conn.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
    ).fetchall() == before_schema
    assert conn.execute("SELECT * FROM namespaces").fetchall() == before_rows
    assert conn.execute("PRAGMA data_version").fetchone()[0] == before_version
    conn.close()
    assert not Path(str(registry) + "-wal").exists()
    assert not Path(str(registry) + "-shm").exists()
    listed = list_namespace_rows()
    assert len(listed) == 1
    assert listed[0]["namespace_id"] is None
    assert listed[0]["name"] == "legacy"
    assert listed[0]["error"]
    assert registry.read_bytes() == before_bytes
    assert not Path(str(registry) + "-wal").exists()
    assert not Path(str(registry) + "-shm").exists()
    with pytest.raises(NamespacePathError):
        init_registry()


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_atomic_fresh_claim_rejects_exact_replacement_hook(
    alias_home, monkeypatch, kind
):
    with Store("claim-owner") as owner:
        owner.observe("CLAIM-OWNER-CANARY")
        owner_db = owner.db_path
        owner_before = owner.stats()["events"]
    import haunt.store as store_module

    target = alias_home / "namespaces" / "claimed.db"

    def replace_claim(path):
        assert path == target
        path.unlink()
        if kind == "symlink":
            path.symlink_to(owner_db)
        else:
            path.hardlink_to(owner_db)

    monkeypatch.setattr(store_module, "_fresh_namespace_claim_hook", replace_claim)
    with pytest.raises(NamespacePathError):
        Store("claimed")
    assert target.is_symlink() if kind == "symlink" else target.exists()
    target.unlink()
    assert owner_db.stat().st_nlink == 1
    assert not list((alias_home / "namespaces").glob(".haunt-claim-*"))

    conn = sqlite3.connect(registry_path())
    assert conn.execute(
        "SELECT COUNT(*) FROM namespace_aliases WHERE normalized_label='claimed'"
    ).fetchone()[0] == 0
    row = conn.execute(
        "SELECT db_device,db_inode FROM namespace_identities "
        "WHERE canonical_label_norm='claim-owner'"
    ).fetchone()
    conn.close()
    assert row == (owner_db.stat().st_dev, owner_db.stat().st_ino)
    with Store("claim-owner", create=False) as owner:
        assert owner.stats()["events"] == owner_before


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_mapped_open_rejects_exact_replacement_before_sqlite_use(
    alias_home, tmp_path, monkeypatch, kind
):
    with Store("mapped-owner") as owner:
        owner.observe("MAPPED-OWNER-CANARY")
        owner_db = owner.db_path
    with Store("redirect-target") as redirect:
        redirect.observe("REDIRECT-CANARY")
        redirect_db = redirect.db_path
    redirect_before = redirect_db.read_bytes()
    backup = tmp_path / "mapped-owner-backup.db"
    import haunt.store as store_module

    def replace_before_open(path):
        assert path == owner_db
        owner_db.rename(backup)
        if kind == "symlink":
            owner_db.symlink_to(redirect_db)
        else:
            owner_db.hardlink_to(redirect_db)

    monkeypatch.setattr(
        store_module, "_mapped_namespace_open_hook", replace_before_open
    )
    with pytest.raises(NamespacePathError):
        Store("mapped-owner", create=False)
    assert redirect_db.read_bytes() == redirect_before

    owner_db.unlink()
    backup.rename(owner_db)
    assert redirect_db.stat().st_nlink == 1
    monkeypatch.setattr(store_module, "_mapped_namespace_open_hook", lambda _path: None)
    with Store("mapped-owner", create=False) as owner:
        assert owner.stats()["events"] == 1


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_mapped_open_verifies_handle_when_replacement_is_swapped_back(
    alias_home, tmp_path, monkeypatch, kind
):
    with Store("handle-owner") as owner:
        owner.observe("HANDLE-OWNER-CANARY")
        owner_db = owner.db_path
    with Store("handle-redirect") as redirect:
        redirect.observe("HANDLE-REDIRECT-CANARY")
        redirect_db = redirect.db_path
    redirect_before = redirect_db.read_bytes()
    backup = tmp_path / "handle-owner-backup.db"
    import haunt.store as store_module

    original_raw_connect = store_module._raw_connect
    replaced = False

    def replace_before_open(path):
        nonlocal replaced
        assert path == owner_db
        replaced = True
        owner_db.rename(backup)
        if kind == "symlink":
            owner_db.symlink_to(redirect_db)
        else:
            owner_db.hardlink_to(redirect_db)

    def connect_then_restore(path, *, create=True):
        conn = original_raw_connect(path, create=create)
        if replaced and path == owner_db:
            owner_db.unlink()
            backup.rename(owner_db)
        return conn

    monkeypatch.setattr(
        store_module, "_mapped_namespace_open_hook", replace_before_open
    )
    monkeypatch.setattr(store_module, "_raw_connect", connect_then_restore)
    with pytest.raises(
        NamespacePathError, match="physical identity changed|SQLite did not open"
    ):
        Store("handle-owner", create=False)
    if backup.exists():
        owner_db.unlink()
        backup.rename(owner_db)
    assert owner_db.stat().st_ino != redirect_db.stat().st_ino
    assert redirect_db.read_bytes() == redirect_before


def test_cached_and_listed_identity_reject_regular_file_replacement(
    alias_home, tmp_path
):
    with Store("replace-owner") as owner:
        owner.observe("REPLACE-OWNER-CANARY")
        owner_db = owner.db_path
    assert namespace_db_path("replace-owner") == owner_db
    backup = tmp_path / "replace-owner-backup.db"
    owner_db.rename(backup)
    owner_db.touch()

    with pytest.raises(NamespacePathError):
        namespace_db_path("replace-owner")
    with pytest.raises(NamespacePathError):
        resolve_namespace_identity("replace-owner")
    listed = next(row for row in list_namespace_rows() if row["name"] == "replace-owner")
    assert "physical identity changed" in listed["error"]

    owner_db.unlink()
    backup.rename(owner_db)
    with Store("replace-owner", create=False) as owner:
        assert owner.stats()["events"] == 1


def test_cached_resolution_validates_every_registry_database(alias_home, tmp_path):
    safe_db = register_namespace("all-source-safe")
    unsafe_db = register_namespace("all-source-unsafe")
    assert namespace_db_path("all-source-safe") == safe_db
    backup = tmp_path / "all-source-unsafe-backup.db"
    outside = tmp_path / "outside.db"
    outside.touch()
    unsafe_db.rename(backup)
    unsafe_db.symlink_to(outside)

    with pytest.raises(NamespacePathError, match="non-symlink"):
        namespace_db_path("all-source-safe")
    with pytest.raises(NamespacePathError, match="non-symlink"):
        resolve_namespace_identity("all-source-safe")
    listed = {row["name"]: row for row in list_namespace_rows()}
    assert "non-symlink" in listed["all-source-safe"]["error"]
    assert "non-symlink" in listed["all-source-unsafe"]["error"]

    unsafe_db.unlink()
    backup.rename(unsafe_db)


def test_current_registry_rejects_symlinked_namespace_root_everywhere(
    alias_home, tmp_path
):
    owner_db = register_namespace("root-owner")
    assert namespace_db_path("root-owner") == owner_db
    root = alias_home / "namespaces"
    real_root = tmp_path / "real-namespace-root"
    root.rename(real_root)
    root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(NamespacePathError, match="real non-symlink directory"):
        namespace_db_path("root-owner")
    with pytest.raises(NamespacePathError, match="real non-symlink directory"):
        resolve_namespace_identity("root-owner")
    with pytest.raises(NamespacePathError, match="real non-symlink directory"):
        Store("root-owner", create=False)
    rows = list_namespace_rows()
    assert len(rows) == 1
    assert "real non-symlink directory" in rows[0]["error"]
    cli = CliRunner().invoke(app, ["namespaces"])
    assert cli.exit_code == 0, cli.output
    assert "real non-symlink directory" in cli.output

    from haunt.mcp_server import MCPAuthority

    authority = MCPAuthority(bound_namespace="root-owner")
    with pytest.raises(NamespacePathError, match="real non-symlink directory"):
        authority.select("root-owner")


def test_registered_alias_beats_later_alias_shaped_database(alias_home):
    register_namespace("split-original")
    original = namespace_db_path("split-original")
    _apply_change("split-original", "split-new")
    impostor = alias_home / "namespaces" / "split-new.db"
    impostor.touch()

    import haunt.paths as paths

    with paths._NAMESPACE_ALIAS_CACHE_LOCK:
        paths._NAMESPACE_ALIAS_CACHE.clear()
    assert namespace_db_path("split-new") == original
    with open_existing("split-new") as store:
        assert store.db_path == original


def test_alias_cache_observes_cross_process_rename_and_retirement(alias_home):
    register_namespace("cross-old")
    assert resolve_namespace("cross-old") == "cross-old"  # populate local cache
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from haunt.store import change_namespace_label; "
                "p=change_namespace_label('cross-old','cross-new'); "
                "change_namespace_label('cross-old','cross-new',apply=True,plan_digest=p['plan_digest'])"
            ),
        ],
        env=env,
        check=True,
    )
    assert resolve_namespace("cross-old") == "cross-new"
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from haunt.store import retire_namespace_alias; "
                "retire_namespace_alias('cross-old',apply=True)"
            ),
        ],
        env=env,
        check=True,
    )
    import haunt.paths as paths

    assert paths._registered_alias("cross-old") is None


def test_alias_cache_invalidates_when_registry_is_recreated(alias_home):
    first_path = register_namespace("first-registry")
    _apply_change("first-registry", "shared-label", action="alias")
    assert namespace_db_path("shared-label") == first_path
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(registry_path()) + suffix)
        if candidate.exists():
            candidate.unlink()

    init_registry()
    second_path = register_namespace("second-registry")
    _apply_change("second-registry", "shared-label", action="alias")
    assert second_path != first_path
    assert namespace_db_path("shared-label") == second_path
    assert resolve_namespace("shared-label") == "second-registry"


def test_alias_cache_never_publishes_retired_then_reassigned_identity(
    alias_home, monkeypatch
):
    first_db = register_namespace("race-first")
    second_db = register_namespace("race-second")
    _apply_change("race-first", "race-shared", action="alias")

    import haunt.paths as paths

    with paths._NAMESPACE_ALIAS_CACHE_LOCK:
        paths._NAMESPACE_ALIAS_CACHE.clear()
    original_fingerprint = paths._registry_fingerprint
    query_finished = threading.Event()
    reassigned = threading.Event()
    call_lock = threading.Lock()
    calls = 0

    def gated_fingerprint():
        nonlocal calls
        with call_lock:
            calls += 1
            call = calls
        if call == 2:
            query_finished.set()
            assert reassigned.wait(5), "registry reassignment did not complete"
        return original_fingerprint()

    monkeypatch.setattr(paths, "_registry_fingerprint", gated_fingerprint)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(paths._registered_alias, "race-shared")
        assert query_finished.wait(5), "alias reader did not reach cache publication"
        retire_namespace_alias("race-shared", apply=True)
        _apply_change("race-second", "race-shared", action="alias")
        reassigned.set()
        assert future.result(timeout=5) == ("race-second", second_db)

    assert paths._registered_alias("race-shared") == ("race-second", second_db)
    assert paths._registered_alias("race-shared") != ("race-first", first_db)


def test_concurrent_alias_apply_is_atomic_and_idempotent(alias_home):
    register_namespace("race")
    plan = change_namespace_label("race", "raced", action="alias")

    def apply_once(_index):
        return change_namespace_label(
            "race", "raced", action="alias", apply=True,
            plan_digest=plan["plan_digest"],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        reports = list(pool.map(apply_once, range(16)))
    identity = resolve_namespace_identity("raced")
    assert identity is not None
    assert sum(a["normalized_label"] == "raced" for a in identity["aliases"]) == 1
    conn = sqlite3.connect(registry_path())
    assert conn.execute(
        "SELECT COUNT(*) FROM namespace_migrations WHERE new_label_norm='raced'"
    ).fetchone()[0] == 1
    conn.close()
    assert any(report["idempotent"] is False for report in reports)
    assert any(report["idempotent"] is True for report in reports)


def test_cross_process_apply_reuses_dry_plan_after_restart(alias_home):
    register_namespace("process-race")
    plan = change_namespace_label("process-race", "process-raced", action="alias")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    command = [
        sys.executable,
        "-c",
        (
            "import sys; from haunt.store import change_namespace_label; "
            "r=change_namespace_label('process-race','process-raced',action='alias',"
            "apply=True,plan_digest=sys.argv[1]); print(r['migration_id'])"
        ),
        plan["plan_digest"],
    ]
    processes = [
        subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for _ in range(2)
    ]
    outputs = [process.communicate(timeout=20) for process in processes]
    assert all(process.returncode == 0 for process in processes), outputs
    migration_ids = {stdout.decode().strip() for stdout, _stderr in outputs}
    assert len(migration_ids) == 1
    conn = sqlite3.connect(registry_path())
    assert conn.execute(
        "SELECT COUNT(*) FROM namespace_migrations WHERE new_label_norm='process-raced'"
    ).fetchone()[0] == 1
    conn.close()


def test_cli_defaults_to_dry_run_and_apply_records_history(alias_home):
    register_namespace("cli-old")
    runner = CliRunner()
    dry = runner.invoke(
        app,
        [
            "namespace", "migrate", "cli-old", "cli-new",
            "--repo", "https://github.com/acme/cli.git",
        ],
    )
    assert dry.exit_code == 0, dry.output
    assert '"mode": "dry-run"' in dry.output
    assert not namespace_exists("cli-new")
    digest = json.loads(dry.output)["plan_digest"]
    applied = runner.invoke(
        app,
        [
            "namespace", "migrate", "cli-old", "cli-new",
            "--repo", "https://github.com/acme/cli.git",
            "--apply", "--plan-digest", digest,
        ],
    )
    assert applied.exit_code == 0, applied.output
    assert namespace_exists("cli-old") and namespace_exists("cli-new")
    conn = sqlite3.connect(registry_path())
    row = conn.execute(
        "SELECT old_label,new_label,repository_identity FROM namespace_migrations"
    ).fetchone()
    conn.close()
    assert row == ("cli-old", "cli-new", "github.com/acme/cli")


def test_ordinary_mcp_accepts_own_alias_but_denies_other_identity(alias_home):
    from haunt.mcp_server import MCPAuthority, MCPAuthorityError

    register_namespace("alpha")
    register_namespace("beta")
    _apply_change("alpha", "alpha-old", action="alias")
    _apply_change("beta", "beta-old", action="alias")
    alpha = resolve_namespace_identity("alpha")
    authority = MCPAuthority(
        bound_namespace="alpha",
        bound_namespace_id=alpha["namespace_id"],
        admin=False,
        allow_purge=True,
    )
    assert authority.select("alpha-old") == "alpha"
    with pytest.raises(MCPAuthorityError):
        authority.select("beta-old")
    assert authority.allow_purge is True  # alias authority does not alter purge gating


def test_fresh_mcp_authority_pins_first_identity_concurrently(alias_home):
    from haunt.mcp_server import MCPAuthority, MCPAuthorityError

    authority = MCPAuthority(bound_namespace="fresh-bound")
    assert authority.bound_namespace_id is None
    with pytest.raises(FrozenInstanceError):
        authority.bound_namespace = "other"
    with pytest.raises(MCPAuthorityError):
        authority.select("unknown-cross")

    def create_and_pin(_index):
        selected = authority.select(None)
        with Store(selected) as store:
            return authority.pin_open_store(store), store.namespace_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(create_and_pin, range(16)))
    assert {namespace for namespace, _identity in results} == {"fresh-bound"}
    assert len({identity for _namespace, identity in results}) == 1

    _apply_change("fresh-bound", "fresh-renamed")
    _apply_change("fresh-renamed", "fresh-alias", action="alias")
    register_namespace("other-identity")
    assert authority.select(None) == "fresh-renamed"
    assert authority.select("fresh-bound") == "fresh-renamed"
    assert authority.select("fresh-alias") == "fresh-renamed"
    with pytest.raises(MCPAuthorityError):
        authority.select("other-identity")
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(registry_path()) + suffix)
        if candidate.exists():
            candidate.unlink()
    init_registry()
    with pytest.raises(MCPAuthorityError, match="no longer registered"):
        authority.select(None)
    assert not namespace_exists("fresh-bound")


def test_fresh_mcp_process_pins_after_first_observe_and_survives_rename(
    alias_home, monkeypatch
):
    monkeypatch.setenv("HAUNT_NAMESPACE", "mcp-fresh")
    monkeypatch.delenv("HAUNT_MCP_ADMIN", raising=False)
    import haunt.mcp_server as mcp

    mcp._MCP_AUTHORITY = None
    mcp._MCP_AUTHORITY_HOME = None
    first = json.loads(mcp.memory_observe(text="MCP-FRESH-PIN-CANARY"))
    assert first["ok"] is True
    pinned = mcp._authority()._pin.namespace_id
    assert pinned == resolve_namespace_identity("mcp-fresh")["namespace_id"]

    _apply_change("mcp-fresh", "mcp-renamed")
    _apply_change("mcp-renamed", "mcp-alias", action="alias")
    with Store("mcp-other"):
        pass
    own = json.loads(
        mcp.memory_recall(query="MCP-FRESH-PIN-CANARY", namespace="mcp-alias")
    )
    assert own["namespace"] == "mcp-renamed"
    assert own["hits"]
    denied = json.loads(mcp.memory_recall(query="secret", namespace="mcp-other"))
    assert denied["ok"] is False
    assert "denied" in denied["error"]


def test_mcp_recall_opens_selected_stable_id_after_label_reassignment(
    alias_home, monkeypatch
):
    with Store("mcp-race-original") as original:
        original.observe("MCP AUTHORITY RACE CANARY FROM ORIGINAL")
        original_id = original.namespace_id
        original_db = original.db_path
    _apply_change("mcp-race-original", "mcp-race-label", action="alias")
    monkeypatch.setenv("HAUNT_NAMESPACE", "mcp-race-label")
    monkeypatch.delenv("HAUNT_MCP_ADMIN", raising=False)
    import haunt.mcp_server as mcp

    mcp._MCP_AUTHORITY = None
    mcp._MCP_AUTHORITY_HOME = None
    selected = threading.Event()
    resume = threading.Event()
    selected_access = None

    def pause_after_selection(access):
        nonlocal selected_access
        selected_access = access
        selected.set()
        assert resume.wait(timeout=15)

    monkeypatch.setattr(mcp, "_mcp_after_selection_hook", pause_after_selection)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            mcp.memory_recall,
            query="MCP AUTHORITY RACE CANARY",
            namespace="mcp-race-label",
        )
        assert selected.wait(timeout=15)
        assert selected_access.namespace_id == original_id
        assert selected_access.db_path == str(original_db)

        _apply_change("mcp-race-original", "mcp-race-original-renamed")
        retire_namespace_alias("mcp-race-label", apply=True)
        with Store("mcp-race-replacement") as replacement:
            replacement.observe("MCP AUTHORITY RACE CANARY FROM REPLACEMENT")
            replacement_id = replacement.namespace_id
            replacement_db = replacement.db_path
        _apply_change(
            "mcp-race-replacement", "mcp-race-label", action="alias"
        )
        assert replacement_id != original_id
        assert replacement_db != original_db
        assert resolve_namespace_identity("mcp-race-label")["namespace_id"] == (
            replacement_id
        )
        resume.set()
        result = json.loads(pending.result(timeout=20))

    assert result["namespace"] == "mcp-race-original-renamed"
    contents = [hit["content"] for hit in result["hits"]]
    assert "MCP AUTHORITY RACE CANARY FROM ORIGINAL" in contents
    assert "MCP AUTHORITY RACE CANARY FROM REPLACEMENT" not in contents


def test_cli_hooks_and_dashboard_alias_routes_use_canonical_store(
    alias_home, monkeypatch
):
    with Store("route-main") as store:
        store.observe("ROUTE-ALIAS-CANARY")
        db_path = store.db_path
    _apply_change("route-main", "route-alias", action="alias")

    cli = CliRunner().invoke(
        app, ["recall", "ROUTE-ALIAS-CANARY", "--namespace", "route-alias"]
    )
    assert cli.exit_code == 0, cli.output
    assert "ROUTE-ALIAS-CANARY" in cli.output

    monkeypatch.setenv("HAUNT_NAMESPACE", "route-alias")
    from haunt.claude_hook import _hook_namespace
    from haunt.cursor_hook import hook_namespace

    for selected in (hook_namespace({}), _hook_namespace({})):
        with Store(selected) as store:
            assert store.name == "route-main"
            assert store.db_path == db_path

    from haunt.dashboard import configure_dashboard_security, reset_dashboard_security
    from tests.dashutil import TEST_DASH_TOKEN, make_dash_client

    configure_dashboard_security(token=TEST_DASH_TOKEN)
    try:
        response = make_dash_client().get(
            "/api/namespace/route-alias/recall?q=ROUTE-ALIAS-CANARY"
        )
        assert response.status_code == 200
        assert response.json()["hits"][0]["namespace"] == "route-main"
    finally:
        reset_dashboard_security()


def test_cli_dashboard_and_mcp_namespace_listings_are_portable(alias_home, monkeypatch):
    with Store("listing-main") as store:
        store.observe("LISTING-CANARY")
        db_path = store.db_path
    _apply_change("listing-main", "listing-alias", action="alias")

    rows = list_namespace_rows()
    assert [(row["name"], row["db_path"]) for row in rows] == [
        ("listing-main", str(db_path))
    ]
    cli = CliRunner().invoke(app, ["namespaces"])
    assert cli.exit_code == 0, cli.output
    assert "listing-main" in cli.output

    from haunt.dashboard import configure_dashboard_security, reset_dashboard_security
    from tests.dashutil import TEST_DASH_TOKEN, make_dash_client

    configure_dashboard_security(token=TEST_DASH_TOKEN)
    try:
        response = make_dash_client().get("/api/namespaces")
        assert response.status_code == 200
        assert [row["name"] for row in response.json()["namespaces"]] == [
            "listing-main"
        ]
    finally:
        reset_dashboard_security()

    monkeypatch.setenv("HAUNT_NAMESPACE", "listing-alias")
    monkeypatch.delenv("HAUNT_MCP_ADMIN", raising=False)
    import haunt.mcp_server as mcp

    mcp._MCP_AUTHORITY = None
    mcp._MCP_AUTHORITY_HOME = None
    payload = json.loads(mcp.memory_namespaces())
    assert payload["bound_namespace"] == "listing-main"
    assert [row["name"] for row in payload["namespaces"]] == ["listing-main"]


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_registry_primary_rejects_unsafe_file_without_touching_victim(
    tmp_path, monkeypatch, kind
):
    home = tmp_path / f"registry-{kind}"
    (home / "namespaces").mkdir(parents=True)
    victim = tmp_path / f"registry-{kind}-victim"
    victim.write_bytes(b"REGISTRY-PRIMARY-VICTIM")
    victim.chmod(0o640)
    target = home / "registry.db"
    target.symlink_to(victim) if kind == "symlink" else target.hardlink_to(victim)
    before = victim.read_bytes(), stat.S_IMODE(victim.stat().st_mode)
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    with pytest.raises(NamespacePathError):
        init_registry()
    assert (victim.read_bytes(), stat.S_IMODE(victim.stat().st_mode)) == before


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_registry_primary_swap_before_open_preserves_victim(
    alias_home, tmp_path, monkeypatch, kind
):
    victim = tmp_path / f"registry-swap-{kind}-victim"
    victim.write_bytes(b"REGISTRY-SWAP-VICTIM")
    victim.chmod(0o640)
    before = victim.read_bytes(), stat.S_IMODE(victim.stat().st_mode)
    registry = registry_path()
    saved = tmp_path / f"saved-registry-{kind}.db"
    import haunt.store as store_module

    def swap(path):
        if path == registry:
            registry.rename(saved)
            registry.symlink_to(victim) if kind == "symlink" else registry.hardlink_to(victim)

    monkeypatch.setattr(store_module, "_sqlite_sidecar_open_hook", swap)
    try:
        with pytest.raises(NamespacePathError, match="physical identity changed"):
            store_module._connect(registry, create=False)
    finally:
        if registry.exists() or registry.is_symlink():
            registry.unlink()
        saved.rename(registry)
    assert (victim.read_bytes(), stat.S_IMODE(victim.stat().st_mode)) == before


def test_migration_requires_exact_plan_and_verified_backup(alias_home):
    db = register_namespace("digest-old")
    before_db = db.read_bytes()
    plan = change_namespace_label("digest-old", "digest-new")
    assert plan["plan_digest"] == change_namespace_label(
        "digest-old", "digest-new"
    )["plan_digest"]
    with pytest.raises(NamespaceMigrationError, match="preceding dry-run"):
        change_namespace_label("digest-old", "digest-new", apply=True)
    with pytest.raises(NamespaceMigrationError, match="does not match"):
        change_namespace_label(
            "digest-old", "digest-new", apply=True, plan_digest="0" * 64
        )
    applied = change_namespace_label(
        "digest-old", "digest-new", apply=True,
        plan_digest=plan["plan_digest"],
    )
    backup = Path(applied["backup"]["path"])
    assert backup.parent == alias_home / "backups"
    assert stat.S_IMODE(backup.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == applied["backup"]["sha256"]
    check = sqlite3.connect(f"{backup.as_uri()}?mode=ro&immutable=1", uri=True)
    assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert check.execute(
        "SELECT canonical_label FROM namespace_identities"
    ).fetchone()[0] == "digest-old"
    check.close()
    assert db.read_bytes() == before_db
    conn = sqlite3.connect(registry_path())
    history = conn.execute(
        """SELECT plan_digest,backup_path,backup_sha256,backup_integrity
           FROM namespace_migrations WHERE migration_id=?""",
        (applied["migration_id"],),
    ).fetchone()
    conn.close()
    assert history == (
        plan["plan_digest"], str(backup), applied["backup"]["sha256"], "ok"
    )


def test_plan_drift_and_backup_failure_do_not_change_namespace(alias_home, monkeypatch):
    register_namespace("drift-old")
    stale = change_namespace_label("drift-old", "drift-new")
    register_namespace("unrelated-drift")
    with pytest.raises(NamespaceMigrationError, match="does not match"):
        change_namespace_label(
            "drift-old", "drift-new", apply=True,
            plan_digest=stale["plan_digest"],
        )
    fresh = change_namespace_label("drift-old", "drift-new")
    import haunt.store as store_module

    def fail_backup(*, purpose):
        raise NamespaceMigrationError(f"forced {purpose} backup failure")

    monkeypatch.setattr(store_module, "_backup_registry", fail_backup)
    with pytest.raises(NamespaceMigrationError, match="backup failure"):
        change_namespace_label(
            "drift-old", "drift-new", apply=True,
            plan_digest=fresh["plan_digest"],
        )
    assert resolve_namespace_identity("drift-old")["canonical_label"] == "drift-old"
    assert resolve_namespace_identity("drift-new") is None


def test_backup_root_symlink_is_rejected_without_touching_external_directory(
    alias_home, tmp_path
):
    register_namespace("backup-root-old")
    plan = change_namespace_label("backup-root-old", "backup-root-new")
    external = tmp_path / "external-backup-victim"
    external.mkdir(mode=0o755)
    external.chmod(0o755)
    victim = external / "keep.txt"
    victim.write_bytes(b"external-backup-victim")
    victim.chmod(0o644)
    (alias_home / "backups").symlink_to(external, target_is_directory=True)
    before = (
        victim.read_bytes(),
        stat.S_IMODE(victim.stat().st_mode),
        stat.S_IMODE(external.stat().st_mode),
        tuple(sorted(path.name for path in external.iterdir())),
    )

    with pytest.raises(NamespaceMigrationError, match="backup directory is unsafe"):
        change_namespace_label(
            "backup-root-old", "backup-root-new", apply=True,
            plan_digest=plan["plan_digest"],
        )

    after = (
        victim.read_bytes(),
        stat.S_IMODE(victim.stat().st_mode),
        stat.S_IMODE(external.stat().st_mode),
        tuple(sorted(path.name for path in external.iterdir())),
    )
    assert after == before
    assert resolve_namespace_identity("backup-root-old")["canonical_label"] == (
        "backup-root-old"
    )


def test_digest_gated_undo_restores_exact_state_and_keeps_history(alias_home):
    register_namespace("undo-old", "https://github.com/acme/undo.git")
    plan = change_namespace_label(
        "undo-old", "undo-new", repository="git@github.com:acme/undo.git"
    )
    applied = change_namespace_label(
        "undo-old", "undo-new", repository="git@github.com:acme/undo.git",
        apply=True, plan_digest=plan["plan_digest"],
    )
    undo_plan = undo_namespace_migration(applied["migration_id"])
    with pytest.raises(NamespaceMigrationError, match="does not match"):
        undo_namespace_migration(applied["migration_id"], apply=True, plan_digest="bad")
    undone = undo_namespace_migration(
        applied["migration_id"], apply=True, plan_digest=undo_plan["plan_digest"]
    )
    assert undone["backup"]["integrity"] == "ok"
    assert resolve_namespace_identity("undo-old")["canonical_label"] == "undo-old"
    assert resolve_namespace_identity("undo-new") is None
    replay = undo_namespace_migration(
        applied["migration_id"], apply=True, plan_digest=undo_plan["plan_digest"]
    )
    assert replay["idempotent"] is True
    conn = sqlite3.connect(registry_path())
    row = conn.execute(
        "SELECT undone_at,undo_plan_digest FROM namespace_migrations WHERE migration_id=?",
        (applied["migration_id"],),
    ).fetchone()
    conn.close()
    assert row[0] and row[1] == undo_plan["plan_digest"]


def test_undo_refuses_after_alias_retirement(alias_home):
    register_namespace("retire-undo-old")
    applied = _apply_change("retire-undo-old", "retire-undo-new")
    retire_namespace_alias("retire-undo-old", apply=True)
    with pytest.raises(NamespaceMigrationError, match="retired alias"):
        undo_namespace_migration(applied["migration_id"])


def test_cli_and_mcp_admin_share_digest_gated_workflow(alias_home, monkeypatch):
    register_namespace("surface-old")
    runner = CliRunner()
    dry = runner.invoke(app, ["namespace", "migrate", "surface-old", "surface-new"])
    assert dry.exit_code == 0
    digest = json.loads(dry.output)["plan_digest"]
    rejected = runner.invoke(
        app, ["namespace", "migrate", "surface-old", "surface-new", "--apply"]
    )
    assert rejected.exit_code == 2
    applied = runner.invoke(
        app,
        [
            "namespace", "migrate", "surface-old", "surface-new", "--apply",
            "--plan-digest", digest,
        ],
    )
    assert applied.exit_code == 0, applied.output

    import haunt.mcp_server as mcp

    monkeypatch.delenv("HAUNT_MCP_ADMIN", raising=False)
    mcp._MCP_AUTHORITY = None
    mcp._MCP_AUTHORITY_HOME = None
    denied = json.loads(
        mcp.memory_namespace_migrate("surface-new", "surface-alias")
    )
    assert denied["ok"] is False
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    mcp._MCP_AUTHORITY = None
    mcp._MCP_AUTHORITY_HOME = None
    mcp_plan = json.loads(
        mcp.memory_namespace_migrate("surface-new", "surface-alias", action="alias")
    )
    mcp_apply = json.loads(
        mcp.memory_namespace_migrate(
            "surface-new", "surface-alias", action="alias", apply=True,
            plan_digest=mcp_plan["plan_digest"],
        )
    )
    assert mcp_apply["applied"] is True
    undo_plan = json.loads(mcp.memory_namespace_undo(mcp_apply["migration_id"]))
    undone = json.loads(
        mcp.memory_namespace_undo(
            mcp_apply["migration_id"], apply=True,
            plan_digest=undo_plan["plan_digest"],
        )
    )
    assert undone["applied"] is True


def _exact_home_snapshot(home: Path) -> dict[str, tuple[bytes, int, int, int]]:
    return {
        str(path.relative_to(home)): (
            path.read_bytes(), stat.S_IMODE(path.stat().st_mode),
            int(path.stat().st_dev), int(path.stat().st_ino),
        )
        for path in sorted(home.rglob("*"))
        if path.is_file()
    }


def _registry_logical_snapshot() -> tuple:
    conn = sqlite3.connect(
        f"{registry_path().as_uri()}?mode=ro&immutable=1", uri=True
    )
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        schema = conn.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        ).fetchall()
        rows = tuple(
            (table, tuple(conn.execute(f"SELECT * FROM {table}").fetchall()))
            for table in tables
        )
        return (
            tuple(schema), rows,
            conn.execute("PRAGMA user_version").fetchone()[0],
            conn.execute("PRAGMA data_version").fetchone()[0],
        )
    finally:
        conn.close()


def _remove_quiescent_registry_sidecars() -> None:
    conn = sqlite3.connect(registry_path())
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(registry_path()) + suffix)
        if sidecar.exists():
            sidecar.unlink()


def test_current_registry_dry_runs_are_exactly_zero_write_without_sidecars(
    alias_home, monkeypatch
):
    register_namespace("zero-old")
    _remove_quiescent_registry_sidecars()
    before_files = _exact_home_snapshot(alias_home)
    before_logical = _registry_logical_snapshot()
    observer = sqlite3.connect(
        f"{registry_path().as_uri()}?mode=ro&immutable=1", uri=True
    )
    before_data_version = observer.execute("PRAGMA data_version").fetchone()[0]

    try:
        direct = change_namespace_label("zero-old", "zero-new")
        assert direct["plan_digest"]
        assert _exact_home_snapshot(alias_home) == before_files
        assert _registry_logical_snapshot() == before_logical
        assert observer.execute("PRAGMA data_version").fetchone()[0] == (
            before_data_version
        )

        cli = CliRunner().invoke(
            app, ["namespace", "migrate", "zero-old", "zero-new"]
        )
        assert cli.exit_code == 0, cli.output
        assert json.loads(cli.output)["plan_digest"] == direct["plan_digest"]
        assert _exact_home_snapshot(alias_home) == before_files
        assert _registry_logical_snapshot() == before_logical
        assert observer.execute("PRAGMA data_version").fetchone()[0] == (
            before_data_version
        )

        monkeypatch.setenv("HAUNT_NAMESPACE", "zero-old")
        monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
        import haunt.mcp_server as mcp

        mcp._MCP_AUTHORITY = None
        mcp._MCP_AUTHORITY_HOME = None
        mcp_plan = json.loads(mcp.memory_namespace_migrate("zero-old", "zero-new"))
        assert mcp_plan["plan_digest"] == direct["plan_digest"]
        assert _exact_home_snapshot(alias_home) == before_files
        assert _registry_logical_snapshot() == before_logical
        assert observer.execute("PRAGMA data_version").fetchone()[0] == (
            before_data_version
        )
        assert not Path(str(registry_path()) + "-wal").exists()
        assert not Path(str(registry_path()) + "-shm").exists()
    finally:
        observer.close()


def test_live_wal_dry_run_sees_commit_without_touching_registry_files(
    alias_home, monkeypatch
):
    register_namespace("wal-plan-main")
    identity = resolve_namespace_identity("wal-plan-main")
    writer = sqlite3.connect(registry_path())
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute(
        """INSERT INTO namespace_aliases(
               normalized_label,label,namespace_id,is_canonical,created_at
           ) VALUES (?,?,?,?,?)""",
        ("wal-plan-alias", "wal-plan-alias", identity["namespace_id"], 0, "t"),
    )
    writer.commit()
    try:
        before = _exact_home_snapshot(alias_home)
        plan = change_namespace_label("wal-plan-alias", "wal-plan-new")
        assert plan["namespace_id"] == identity["namespace_id"]
        assert _exact_home_snapshot(alias_home) == before
        cli = CliRunner().invoke(
            app, ["namespace", "migrate", "wal-plan-alias", "wal-plan-new"]
        )
        assert cli.exit_code == 0, cli.output
        assert json.loads(cli.output)["plan_digest"] == plan["plan_digest"]
        assert _exact_home_snapshot(alias_home) == before

        monkeypatch.setenv("HAUNT_NAMESPACE", "wal-plan-alias")
        monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
        import haunt.mcp_server as mcp

        mcp._MCP_AUTHORITY = None
        mcp._MCP_AUTHORITY_HOME = None
        mcp_plan = json.loads(
            mcp.memory_namespace_migrate("wal-plan-alias", "wal-plan-new")
        )
        assert mcp_plan["plan_digest"] == plan["plan_digest"]
        assert _exact_home_snapshot(alias_home) == before
        journal = Path(str(registry_path()) + "-journal")
        assert not journal.exists() or journal.stat().st_size == 0
    finally:
        writer.close()


def test_live_wal_shadow_ignores_temp_environment_and_cleans_every_path(
    alias_home, monkeypatch
):
    register_namespace("trusted-temp-old")
    identity = resolve_namespace_identity("trusted-temp-old")
    writer = sqlite3.connect(registry_path())
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute(
        """INSERT INTO namespace_aliases(
               normalized_label,label,namespace_id,is_canonical,created_at
           ) VALUES (?,?,?,?,?)""",
        (
            "trusted-temp-alias", "trusted-temp-alias",
            identity["namespace_id"], 0, "t",
        ),
    )
    writer.commit()
    for variable in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(variable, str(alias_home))
    monkeypatch.setattr(tempfile, "tempdir", None)
    import haunt.paths as paths_module

    before_home = _exact_home_snapshot(alias_home)
    roots: list[Path] = []
    audit_active = False
    home_mutations: list[tuple[str, str]] = []

    def filesystem_audit(event, arguments):
        if not audit_active:
            return
        if event == "open":
            flags = arguments[2] if len(arguments) > 2 else 0
            if not isinstance(flags, int) or not flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            ):
                return
        elif event not in {
            "os.mkdir", "os.rmdir", "os.remove", "os.rename", "os.link",
            "os.symlink", "os.chmod",
        }:
            return
        for argument in arguments:
            if not isinstance(argument, (str, bytes, os.PathLike)):
                continue
            candidate = Path(os.fsdecode(argument))
            if not candidate.is_absolute():
                continue
            if candidate.resolve().is_relative_to(alias_home.resolve()):
                home_mutations.append((event, str(candidate)))

    sys.addaudithook(filesystem_audit)

    def audit_shadow(phase, temporary):
        root = Path(temporary.name)
        roots.append(root)
        assert not root.resolve().is_relative_to(alias_home.resolve())
        assert root.parent in {Path("/private/tmp"), Path("/tmp")}
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        for entry in root.iterdir():
            assert entry.is_file() and not entry.is_symlink()
            assert stat.S_IMODE(entry.stat().st_mode) == 0o600
        assert _exact_home_snapshot(alias_home) == before_home

    monkeypatch.setattr(paths_module, "_temporary_shadow_hook", audit_shadow)
    try:
        audit_active = True
        plan = change_namespace_label("trusted-temp-alias", "trusted-temp-new")
        audit_active = False
        assert plan["namespace_id"] == identity["namespace_id"]
        assert roots
        assert all(not root.exists() for root in roots)
        assert _exact_home_snapshot(alias_home) == before_home
        assert home_mutations == []

        failed_roots: list[Path] = []

        def fail_after_copy(phase, temporary):
            failed_roots.append(Path(temporary.name))
            audit_shadow(phase, temporary)
            if phase == "copied":
                raise RuntimeError("forced private snapshot failure")

        monkeypatch.setattr(
            paths_module, "_temporary_shadow_hook", fail_after_copy
        )
        audit_active = True
        with pytest.raises(RuntimeError, match="forced private snapshot failure"):
            change_namespace_label("trusted-temp-alias", "trusted-temp-new")
        audit_active = False
        assert failed_roots
        assert all(not root.exists() for root in failed_roots)
        assert _exact_home_snapshot(alias_home) == before_home
        assert home_mutations == []
    finally:
        audit_active = False
        writer.close()


def test_incomplete_live_wal_dry_run_fails_without_touching_registry_files(
    alias_home,
):
    register_namespace("incomplete-wal-old")
    _remove_quiescent_registry_sidecars()
    Path(str(registry_path()) + "-wal").write_bytes(b"incomplete-live-wal")
    before = _exact_home_snapshot(alias_home)

    with pytest.raises(NamespacePathError, match="incomplete WAL state"):
        change_namespace_label("incomplete-wal-old", "incomplete-wal-new")

    assert _exact_home_snapshot(alias_home) == before
    assert not Path(str(registry_path()) + "-shm").exists()


def test_unreadable_complete_wal_dry_run_fails_without_touching_registry_files(
    alias_home,
):
    register_namespace("unreadable-wal-old")
    _remove_quiescent_registry_sidecars()
    Path(str(registry_path()) + "-wal").write_bytes(b"not-a-valid-wal" * 8)
    Path(str(registry_path()) + "-shm").write_bytes(bytes(32 * 1024))
    before = _exact_home_snapshot(alias_home)

    with pytest.raises(NamespacePathError, match="cannot materialize"):
        change_namespace_label("unreadable-wal-old", "unreadable-wal-new")

    assert _exact_home_snapshot(alias_home) == before


def test_fresh_idempotent_rename_is_no_write_and_remains_undoable(alias_home):
    register_namespace("fresh-replay-old")
    first_plan = change_namespace_label("fresh-replay-old", "fresh-replay-new")
    applied = change_namespace_label(
        "fresh-replay-old", "fresh-replay-new", apply=True,
        plan_digest=first_plan["plan_digest"],
    )
    _remove_quiescent_registry_sidecars()
    before_fresh_plan = _exact_home_snapshot(alias_home)
    before_fresh_logical = _registry_logical_snapshot()
    fresh_plan = change_namespace_label("fresh-replay-old", "fresh-replay-new")
    assert fresh_plan["idempotent"] is True
    assert _exact_home_snapshot(alias_home) == before_fresh_plan
    assert _registry_logical_snapshot() == before_fresh_logical
    before = _exact_home_snapshot(alias_home)
    replay = change_namespace_label(
        "fresh-replay-old", "fresh-replay-new", apply=True,
        plan_digest=fresh_plan["plan_digest"],
    )
    assert replay["idempotent"] is True
    assert _exact_home_snapshot(alias_home) == before
    conn = sqlite3.connect(registry_path())
    assert conn.execute(
        "SELECT name FROM namespaces"
    ).fetchall() == [("fresh-replay-new",)]
    conn.close()
    undo_plan = undo_namespace_migration(applied["migration_id"])
    undo_namespace_migration(
        applied["migration_id"], apply=True,
        plan_digest=undo_plan["plan_digest"],
    )
    assert resolve_namespace_identity("fresh-replay-old")["canonical_label"] == (
        "fresh-replay-old"
    )


def test_apply_replay_refuses_retirement_drift_without_writes(alias_home):
    register_namespace("apply-drift-old")
    plan = change_namespace_label("apply-drift-old", "apply-drift-new")
    change_namespace_label(
        "apply-drift-old", "apply-drift-new", apply=True,
        plan_digest=plan["plan_digest"],
    )
    retire_namespace_alias("apply-drift-old", apply=True)
    before = _exact_home_snapshot(alias_home)
    with pytest.raises(NamespaceMigrationError, match="replay conflicts"):
        change_namespace_label(
            "apply-drift-old", "apply-drift-new", apply=True,
            plan_digest=plan["plan_digest"],
        )
    assert _exact_home_snapshot(alias_home) == before


def test_undo_replay_refuses_later_migration_drift_without_writes(alias_home):
    register_namespace("undo-drift-old")
    applied = _apply_change("undo-drift-old", "undo-drift-new")
    undo_plan = undo_namespace_migration(applied["migration_id"])
    undo_namespace_migration(
        applied["migration_id"], apply=True,
        plan_digest=undo_plan["plan_digest"],
    )
    _apply_change("undo-drift-old", "undo-drift-later", action="alias")
    before = _exact_home_snapshot(alias_home)
    with pytest.raises(NamespaceMigrationError, match="undo replay conflicts"):
        undo_namespace_migration(
            applied["migration_id"], apply=True,
            plan_digest=undo_plan["plan_digest"],
        )
    assert _exact_home_snapshot(alias_home) == before


def test_malformed_bracketed_non_ip_host_is_rejected_uniformly():
    assert repository_identity("ssh://git@[not-ipv6]:2222/acme/api.git") is None
    assert repository_identity("https://[127.0.0.1]/acme/api.git") is None


@pytest.mark.parametrize("entry", ["primary", "sidecar", "tighten", "snapshot", "lock", "backup"])
def test_missing_o_nofollow_fails_closed_at_every_safe_open(
    alias_home, monkeypatch, entry
):
    register_namespace("nofollow-old")
    plan = change_namespace_label("nofollow-old", "nofollow-new")
    import haunt.store as store_module

    monkeypatch.delattr(os, "O_NOFOLLOW")
    action = {
        "primary": lambda: SQLitePrimaryGuard.acquire(
            registry_path(), create_missing=False
        ),
        "sidecar": lambda: SQLiteSidecarGuard.acquire(
            registry_path(), claim_missing=False
        ),
        "tighten": lambda: tighten_db_files(registry_path()),
        "snapshot": lambda: sqlite_storage_snapshot(registry_path()),
        "lock": lambda: change_namespace_label(
            "nofollow-old", "nofollow-new", apply=True,
            plan_digest=plan["plan_digest"],
        ),
        "backup": lambda: store_module._backup_registry(purpose="test"),
    }[entry]
    with pytest.raises(NamespacePathError, match="O_NOFOLLOW is required"):
        action()


@pytest.mark.parametrize(
    "phase", ["before_create", "before_link", "before_final_verify", "before_record"]
)
def test_registry_backup_directory_swap_fails_without_publication_or_mutation(
    alias_home, monkeypatch, phase
):
    register_namespace(f"backup-race-{phase}-old")
    plan = change_namespace_label(
        f"backup-race-{phase}-old", f"backup-race-{phase}-new"
    )
    import haunt.store as store_module

    moved = alias_home / f"held-backups-{phase}"
    replacement_snapshot = None
    fired = False

    def swap_backup_root(current_phase, backup_root):
        nonlocal fired, replacement_snapshot
        if fired or current_phase != phase:
            return
        fired = True
        backup_root.rename(moved)
        backup_root.mkdir(mode=0o700)
        victim = backup_root / "victim.txt"
        victim.write_bytes(b"replacement must remain untouched")
        victim.chmod(0o640)
        replacement_snapshot = (
            victim.read_bytes(),
            stat.S_IMODE(victim.stat().st_mode),
            int(victim.stat().st_dev),
            int(victim.stat().st_ino),
            stat.S_IMODE(backup_root.stat().st_mode),
            tuple(sorted(entry.name for entry in backup_root.iterdir())),
        )

    monkeypatch.setattr(store_module, "_registry_backup_hook", swap_backup_root)
    with pytest.raises(NamespaceMigrationError, match="backup directory changed"):
        change_namespace_label(
            f"backup-race-{phase}-old", f"backup-race-{phase}-new",
            apply=True, plan_digest=plan["plan_digest"],
        )

    assert fired and replacement_snapshot is not None
    backup_root = alias_home / "backups"
    victim = backup_root / "victim.txt"
    assert (
        victim.read_bytes(),
        stat.S_IMODE(victim.stat().st_mode),
        int(victim.stat().st_dev),
        int(victim.stat().st_ino),
        stat.S_IMODE(backup_root.stat().st_mode),
        tuple(sorted(entry.name for entry in backup_root.iterdir())),
    ) == replacement_snapshot
    assert list(moved.iterdir()) == []
    assert resolve_namespace_identity(
        f"backup-race-{phase}-old"
    )["canonical_label"] == f"backup-race-{phase}-old"
    assert resolve_namespace_identity(f"backup-race-{phase}-new") is None
    conn = sqlite3.connect(registry_path())
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM namespace_migrations"
        ).fetchone()[0] == 0
    finally:
        conn.close()
