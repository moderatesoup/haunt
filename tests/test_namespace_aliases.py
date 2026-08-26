"""E3 canonical namespace identity, aliases, and explicit migration."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from typer.testing import CliRunner

from haunt.cli import app
from haunt.paths import (
    infer_namespace,
    namespace_db_path,
    registry_path,
    repository_identity,
    resolve_namespace,
)
from haunt.store import (
    AliasRetirementError,
    NamespaceCollisionError,
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


def test_fresh_identity_alias_rename_reuses_exact_database(alias_home):
    with Store("Original Name") as store:
        memory = store.observe("rename canary")
        original_id = store.namespace_id
        original_db = store.db_path

    dry = change_namespace_label("Original Name", "Moved Name", apply=False)
    assert dry["mode"] == "dry-run"
    assert dry["database_operation"] == "none"
    assert not namespace_exists("Moved Name")

    applied = change_namespace_label("Original Name", "Moved Name", apply=True)
    replay = change_namespace_label("Original Name", "Moved Name", apply=True)
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
    change_namespace_label("before", "after", repository=str(new_root), apply=True)

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

    change_namespace_label("late", "renamed", apply=True)
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

    applied = change_namespace_label("Legacy", "Renamed", apply=True)
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
        change_namespace_label("alpha", "BETA", action="alias", apply=True)

    prefix = "x" * 80
    change_namespace_label("alpha", prefix + "one", action="alias", apply=True)
    with pytest.raises(NamespaceCollisionError):
        change_namespace_label("beta", prefix + "two", action="alias", apply=True)

    orphan = alias_home / "namespaces" / "orphan.db"
    orphan.touch()
    with pytest.raises(NamespaceCollisionError):
        change_namespace_label("alpha", "orphan", action="alias", apply=True)


def test_retirement_checks_only_recorded_references_and_reports_caveat(alias_home):
    register_namespace("old", "https://github.com/acme/retire.git")
    change_namespace_label(
        "old", "new", repository="git@github.com:acme/retire.git", apply=True
    )
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
    change_namespace_label("old", "bridge", action="alias", apply=True)
    change_namespace_label("bridge", "new", action="rename", apply=True)
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
    change_namespace_label("canonical", "bridge", action="alias", apply=True)
    change_namespace_label("bridge", "dependent", action="alias", apply=True)
    check = retire_namespace_alias("bridge")
    assert check["safe"] is False
    assert {b["kind"] for b in check["blockers"]} == {"dependent-alias"}


def test_rename_to_existing_alias_reroots_lineage_for_old_retirement(alias_home):
    register_namespace("a")
    change_namespace_label("a", "b", action="alias", apply=True)
    before = resolve_namespace_identity("b")
    assert next(
        alias for alias in before["aliases"] if alias["normalized_label"] == "b"
    )["source_alias_norm"] == "a"

    change_namespace_label("a", "b", action="rename", apply=True)
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

    symlink_target = alias_home / "namespaces" / "symlink-target.db"
    symlink_target.symlink_to(owner_db)
    hardlink_target = alias_home / "namespaces" / "hardlink-target.db"
    hardlink_target.hardlink_to(owner_db)
    regular_target = alias_home / "namespaces" / "unmapped-target.db"
    regular_target.touch()
    outside_db = tmp_path / "outside.db"
    outside_db.touch()
    outside_target = alias_home / "namespaces" / "outside-target.db"
    outside_target.symlink_to(outside_db)

    for label in (
        "symlink-target",
        "hardlink-target",
        "unmapped-target",
        "outside-target",
    ):
        with pytest.raises(NamespaceCollisionError):
            Store(label)
        assert not namespace_exists(label)
    with pytest.raises(NamespaceCollisionError):
        register_namespace("symlink-target", owner_remote)

    for label in ("symlink-target", "hardlink-target", "outside-target"):
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

    from haunt.mcp_server import MCPAuthority, MCPAuthorityError

    authority = MCPAuthority(
        bound_namespace="physical-owner",
        bound_namespace_id=owner_id,
    )
    assert authority.select("physical-owner") == "physical-owner"
    for label in ("symlink-target", "hardlink-target", "outside-target"):
        with pytest.raises(MCPAuthorityError):
            authority.select(label)

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

    with pytest.raises(NamespaceCollisionError, match="same file"):
        change_namespace_label("first", "third", apply=False)
    with pytest.raises(NamespaceCollisionError, match="same file"):
        init_registry()
    conn = sqlite3.connect(registry)
    assert conn.execute("SELECT COUNT(*) FROM namespace_identities").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM namespace_aliases").fetchone()[0] == 0
    conn.close()


def test_registered_alias_beats_later_alias_shaped_database(alias_home):
    register_namespace("split-original")
    original = namespace_db_path("split-original")
    change_namespace_label("split-original", "split-new", apply=True)
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
                "change_namespace_label('cross-old','cross-new',apply=True)"
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
    change_namespace_label("first-registry", "shared-label", action="alias", apply=True)
    assert namespace_db_path("shared-label") == first_path
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(registry_path()) + suffix)
        if candidate.exists():
            candidate.unlink()

    init_registry()
    second_path = register_namespace("second-registry")
    change_namespace_label("second-registry", "shared-label", action="alias", apply=True)
    assert second_path != first_path
    assert namespace_db_path("shared-label") == second_path
    assert resolve_namespace("shared-label") == "second-registry"


def test_alias_cache_never_publishes_retired_then_reassigned_identity(
    alias_home, monkeypatch
):
    first_db = register_namespace("race-first")
    second_db = register_namespace("race-second")
    change_namespace_label("race-first", "race-shared", action="alias", apply=True)

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
        change_namespace_label(
            "race-second", "race-shared", action="alias", apply=True
        )
        reassigned.set()
        assert future.result(timeout=5) == ("race-second", second_db)

    assert paths._registered_alias("race-shared") == ("race-second", second_db)
    assert paths._registered_alias("race-shared") != ("race-first", first_db)


def test_concurrent_alias_apply_is_atomic_and_idempotent(alias_home):
    register_namespace("race")

    def apply_once(_index):
        return change_namespace_label("race", "raced", action="alias", apply=True)

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


def test_cli_defaults_to_dry_run_and_apply_records_history(alias_home):
    register_namespace("cli-old")
    runner = CliRunner()
    dry = runner.invoke(app, ["namespace", "migrate", "cli-old", "cli-new"])
    assert dry.exit_code == 0, dry.output
    assert '"mode": "dry-run"' in dry.output
    assert not namespace_exists("cli-new")
    applied = runner.invoke(
        app,
        [
            "namespace", "migrate", "cli-old", "cli-new",
            "--repo", "https://github.com/acme/cli.git", "--apply",
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
    change_namespace_label("alpha", "alpha-old", action="alias", apply=True)
    change_namespace_label("beta", "beta-old", action="alias", apply=True)
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
            return authority.pin_namespace(store.name), store.namespace_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(create_and_pin, range(16)))
    assert {namespace for namespace, _identity in results} == {"fresh-bound"}
    assert len({identity for _namespace, identity in results}) == 1

    change_namespace_label("fresh-bound", "fresh-renamed", apply=True)
    change_namespace_label("fresh-renamed", "fresh-alias", action="alias", apply=True)
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

    change_namespace_label("mcp-fresh", "mcp-renamed", apply=True)
    change_namespace_label("mcp-renamed", "mcp-alias", action="alias", apply=True)
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


def test_cli_hooks_and_dashboard_alias_routes_use_canonical_store(
    alias_home, monkeypatch
):
    with Store("route-main") as store:
        store.observe("ROUTE-ALIAS-CANARY")
        db_path = store.db_path
    change_namespace_label("route-main", "route-alias", action="alias", apply=True)

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
    change_namespace_label("listing-main", "listing-alias", action="alias", apply=True)

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
