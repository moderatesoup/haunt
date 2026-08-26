"""E3 canonical namespace identity, aliases, and explicit migration."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

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


def test_dependent_alias_blocks_retirement(alias_home):
    register_namespace("canonical")
    change_namespace_label("canonical", "bridge", action="alias", apply=True)
    change_namespace_label("bridge", "dependent", action="alias", apply=True)
    check = retire_namespace_alias("bridge")
    assert check["safe"] is False
    assert {b["kind"] for b in check["blockers"]} == {"dependent-alias"}


def test_typo_read_does_not_create_registry_alias_or_database(alias_home):
    register_namespace("known")
    before = list_namespace_rows()
    with pytest.raises(UnknownNamespaceError):
        open_existing("knwon")
    assert list_namespace_rows() == before
    assert not (alias_home / "namespaces" / "knwon.db").exists()


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
