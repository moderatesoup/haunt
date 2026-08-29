"""Backlog C3: reconcile namespaces that are already split.

Scope (see `reconcile_namespaces()` in src/haunt/store.py for the full
reasoning, and the module comment above it in the same file):

  - One-directional. SOURCE is opened read-only for the whole call and is
    never written to. TARGET gains every row SOURCE has that it does not
    already have; TARGET keeps everything it already had.
  - Operator-invoked only: both namespace labels are always supplied
    explicitly by the caller. Nothing in a hook, MCP entry point, or
    bootstrap calls this.
  - Dry-run (the default) is zero-write, including no backup. Apply
    requires the exact `plan_digest` a preceding dry-run printed and
    refuses -- writing nothing -- if either namespace's content has
    changed since, including a prior successful apply of that very digest
    (idempotency is achieved by repeating the whole dry-run/apply cycle,
    which then reports/moves zero rows, not by replaying one digest twice).
  - A row present in both namespaces under the same primary key must be
    byte-identical (ignoring `memories.embedding`, which is deliberately
    dropped and re-queued rather than copied) or the whole operation
    refuses and writes nothing, anywhere.
  - The registry (namespace_identities/aliases/repository_bindings) is
    never touched. Both labels stay independently resolvable after apply;
    only TARGET's database content changes.

This does not re-test E3's own alias/rename/undo machinery (see
tests/test_namespace_aliases.py) or C1/C2's registration/inference fix (see
tests/test_repo_binding.py) -- only that this new code does not touch or
weaken them.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import stat
import threading
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import haunt.paths as paths
import haunt.store as store
from haunt.cli import app
from haunt.paths import registry_path
from haunt.store import (
    AliasRetirementError,
    NamespaceCollisionError,
    NamespaceMigrationError,
    SCHEMA_VERSION,
    Store,
    UnknownNamespaceError,
    _RECONCILE_TABLES,
    _init_namespace_schema,
    init_registry,
    reconcile_namespaces,
    register_namespace,
)


@pytest.fixture
def reconcile_home(tmp_path, monkeypatch):
    home = tmp_path / "haunt-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
    from haunt import embed

    embed.reset()
    init_registry()
    yield home
    embed.reset()


def _patch_git_context(monkeypatch, remote_url: str | None, repo_root: Path | None) -> None:
    """See tests/test_repo_binding.py: store.py binds its own copy of this."""

    def fake(_root):
        return remote_url, repo_root

    monkeypatch.setattr(paths, "_git_repo_context", fake)
    monkeypatch.setattr(store, "_git_repo_context", fake)


def _db_path(home: Path, label: str) -> Path:
    return home / "namespaces" / f"{label}.db"


def _plan_and_apply(source: str, target: str) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = reconcile_namespaces(source, target)
    applied = reconcile_namespaces(
        source, target, apply=True, plan_digest=plan["plan_digest"]
    )
    return plan, applied


def _import_envelope(fidelity: str = "lossless") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "import",
        "channel": "python",
        "source_platform": "聊天平台",
        "source_native_id": "消息-雪-🧊",
        "source_format": "vendor-json",
        "parser_version": "parser/2.4.1",
        "imported_at": "2025-03-04T05:06:07.123456-06:00",
        "fidelity": fidelity,
        "original_blob_sha256": "sha256:" + "ab" * 32,
        "transforms": ["decode:utf-8", "normalize:newlines"],
    }


def _logical_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = (
        "sessions", "events", "memories", "entities", "relations",
        "entity_mentions", "relation_evidence", "corrections",
        "lineage_tombstones",
    )
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


# ---------------------------------------------------------------------------
# The real, measured scenario as a fixture.
# ---------------------------------------------------------------------------


def test_real_measured_split_pair_reconciles_with_every_row_accounted_for(
    reconcile_home, monkeypatch
):
    """Reproduces the exact split BACKLOG.md's C3 baseline table describes:
    ironscope (blank repo_path, the pre-fix legacy registration) and
    github.com-moderatesoup-ironscope (the fork C1 mints going forward) are
    two populated, independently registered namespaces for one repository.
    After reconcile, TARGET (the fork -- what C1/C2 route new sessions to)
    holds the exact union of both, with every source row present unchanged
    and every original target row still present unchanged.
    """
    register_namespace("ironscope")
    with Store("ironscope") as st:
        source_ids = [st.observe(f"ironscope legacy memory {i}").memory_id for i in range(6)]

    project = reconcile_home / "ironscope"
    project.mkdir()
    _patch_git_context(
        monkeypatch, "git@github.com:moderatesoup/ironscope.git", project
    )
    register_namespace("github.com-moderatesoup-ironscope", str(project))
    with Store("github.com-moderatesoup-ironscope") as st:
        target_ids = [st.observe(f"fork memory {i}").memory_id for i in range(9)]

    plan = reconcile_namespaces("ironscope", "github.com-moderatesoup-ironscope")
    assert plan["mode"] == "dry-run"
    assert plan["tables"]["memories"]["insert_into_target"] == 6
    assert plan["total_rows_to_insert"] > 0

    applied = reconcile_namespaces(
        "ironscope", "github.com-moderatesoup-ironscope",
        apply=True, plan_digest=plan["plan_digest"],
    )
    assert applied["applied"] is True

    with Store("github.com-moderatesoup-ironscope", create=False) as st:
        for mid in source_ids + target_ids:
            mem = st.get_memory(mid)
            assert mem is not None, f"missing {mid} after reconcile"
        stats = st.stats()
        assert stats["memories"] == len(source_ids) + len(target_ids)

    with Store("ironscope", create=False) as st:
        assert st.stats()["memories"] == len(source_ids)
        for mid in target_ids:
            assert st.get_memory(mid) is None  # source never gains target's rows

    # Both labels remain independently resolvable; the registry itself was
    # never touched by this operation.
    assert store.namespace_exists("ironscope")
    assert store.namespace_exists("github.com-moderatesoup-ironscope")


# ---------------------------------------------------------------------------
# Dry-run is exactly zero-write.
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing_anywhere(reconcile_home):
    register_namespace("dry-src")
    register_namespace("dry-dst")
    with Store("dry-src") as st:
        st.observe("source content")
    with Store("dry-dst") as st:
        st.observe("target content")

    src_before = _db_path(reconcile_home, "dry-src").read_bytes()
    dst_before = _db_path(reconcile_home, "dry-dst").read_bytes()
    registry_before = registry_path().read_bytes()
    backups_dir = reconcile_home / "backups"
    assert not backups_dir.exists()

    plan = reconcile_namespaces("dry-src", "dry-dst")
    assert plan["mode"] == "dry-run"
    assert "plan_digest" in plan

    assert _db_path(reconcile_home, "dry-src").read_bytes() == src_before
    assert _db_path(reconcile_home, "dry-dst").read_bytes() == dst_before
    assert registry_path().read_bytes() == registry_before
    assert not backups_dir.exists()


def test_plan_digest_is_deterministic_across_repeated_dry_runs(reconcile_home):
    register_namespace("det-src")
    register_namespace("det-dst")
    with Store("det-src") as st:
        st.observe("a")
        st.observe("b")
    with Store("det-dst") as st:
        st.observe("c")

    first = reconcile_namespaces("det-src", "det-dst")
    second = reconcile_namespaces("det-src", "det-dst")
    assert first["plan_digest"] == second["plan_digest"]
    assert first["content_state_digest"] == second["content_state_digest"]


# ---------------------------------------------------------------------------
# Apply preserves exact row counts and is idempotent.
# ---------------------------------------------------------------------------


def test_apply_preserves_exact_row_counts_from_both_sides(reconcile_home):
    register_namespace("count-src")
    register_namespace("count-dst")
    with Store("count-src") as st:
        for i in range(7):
            st.observe(f"src {i}")
    with Store("count-dst") as st:
        for i in range(4):
            st.observe(f"dst {i}")

    with Store("count-src", create=False) as st:
        src_before = _logical_counts(st.conn)
    with Store("count-dst", create=False) as st:
        dst_before = _logical_counts(st.conn)

    plan, applied = _plan_and_apply("count-src", "count-dst")

    with Store("count-src", create=False) as st:
        src_after = _logical_counts(st.conn)
    with Store("count-dst", create=False) as st:
        dst_after = _logical_counts(st.conn)

    assert src_after == src_before, "source must be completely unchanged"
    for table in dst_before:
        assert dst_after[table] == dst_before[table] + src_before[table], table
    # memories/events/sessions grew by exactly the source's row count
    assert dst_after["memories"] == dst_before["memories"] + src_before["memories"]
    assert dst_after["events"] == dst_before["events"] + src_before["events"]
    assert dst_after["sessions"] == dst_before["sessions"] + src_before["sessions"]


def test_reconciliation_is_idempotent_across_full_cycles(reconcile_home):
    register_namespace("idem-src")
    register_namespace("idem-dst")
    with Store("idem-src") as st:
        for i in range(5):
            st.observe(f"idem src {i}")
    with Store("idem-dst") as st:
        st.observe("idem dst 0")

    _plan_and_apply("idem-src", "idem-dst")
    with Store("idem-dst", create=False) as st:
        after_first = st.stats()["memories"]

    plan2 = reconcile_namespaces("idem-src", "idem-dst")
    assert plan2["total_rows_to_insert"] == 0
    applied2 = reconcile_namespaces(
        "idem-src", "idem-dst", apply=True, plan_digest=plan2["plan_digest"]
    )
    assert sum(v["inserted"] for v in applied2["rows_inserted"].values()) == 0

    with Store("idem-dst", create=False) as st:
        after_second = st.stats()["memories"]
    assert after_second == after_first


def test_replaying_an_already_applied_digest_is_refused_not_silently_replayed(
    reconcile_home,
):
    register_namespace("replay-src")
    register_namespace("replay-dst")
    with Store("replay-src") as st:
        st.observe("replay content")
    register_namespace("replay-dst")

    plan = reconcile_namespaces("replay-src", "replay-dst")
    applied = reconcile_namespaces(
        "replay-src", "replay-dst", apply=True, plan_digest=plan["plan_digest"]
    )
    with pytest.raises(NamespaceMigrationError, match="does not match"):
        reconcile_namespaces(
            "replay-src", "replay-dst",
            apply=True, plan_digest=applied["plan_digest"],
        )


# ---------------------------------------------------------------------------
# Collisions are detected and refused, never guessed.
# ---------------------------------------------------------------------------


def _insert_colliding_event_and_memory(
    home: Path, target_label: str, *, memory_id: str, event_id: str,
    session_id: str, content: str,
) -> None:
    conn = sqlite3.connect(str(_db_path(home, target_label)))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    now = "2026-01-01T00:00:00.000000+00:00"
    conn.execute(
        "INSERT INTO sessions(id, started_at, ended_at, source, meta) VALUES (?,?,?,?,?)",
        (session_id, now, None, "test", "{}"),
    )
    ev_cols = [r["name"] for r in conn.execute("PRAGMA table_info(events)").fetchall()]
    values = {
        "id": event_id, "idempotency_key": None, "session_id": session_id,
        "ts": now, "event_time": now, "role": "user", "content": content,
        "tool_name": None, "tool_input": None, "tool_output": None,
        "origin": "python", "tier": "episodic", "meta": "{}",
        "recall_class": None, "provenance": None,
    }
    conn.execute(
        f"INSERT INTO events({','.join(ev_cols)}) VALUES ({','.join('?' for _ in ev_cols)})",
        tuple(values[c] for c in ev_cols),
    )
    conn.execute(
        """INSERT INTO memories(
               id, event_id, tier, content, embedding, valid_from, valid_to,
               created_at, content_hash
           ) VALUES (?,?,?,?,?,?,?,?,?)""",
        (memory_id, event_id, "episodic", content, None, now, None, now, "x" * 64),
    )
    conn.commit()
    conn.close()


def test_id_collision_with_different_content_refuses_and_writes_nothing(reconcile_home):
    register_namespace("coll-src")
    with Store("coll-src") as st:
        r = st.observe("shared content")
        shared_id = r.memory_id
        event_id = st.conn.execute(
            "SELECT event_id FROM memories WHERE id=?", (shared_id,)
        ).fetchone()[0]

    register_namespace("coll-dst")
    with Store("coll-dst"):
        pass  # initialize schema without writing content
    _insert_colliding_event_and_memory(
        reconcile_home, "coll-dst",
        memory_id=shared_id, event_id=event_id,
        session_id="coll-sess", content="DIFFERENT content",
    )

    dst_before = _db_path(reconcile_home, "coll-dst").read_bytes()
    with pytest.raises(NamespaceCollisionError, match="memories"):
        reconcile_namespaces("coll-src", "coll-dst")
    assert _db_path(reconcile_home, "coll-dst").read_bytes() == dst_before

    # Apply must also be unreachable: no valid plan_digest was ever produced
    # for this unsafe state, and a fabricated one is rejected too -- apply
    # always recomputes the plan fresh before even looking at the supplied
    # digest, so this still surfaces as the same collision refusal.
    with pytest.raises((NamespaceMigrationError, NamespaceCollisionError)):
        reconcile_namespaces(
            "coll-src", "coll-dst", apply=True, plan_digest="0" * 64
        )
    assert _db_path(reconcile_home, "coll-dst").read_bytes() == dst_before


def test_id_collision_with_identical_content_is_a_safe_noop(reconcile_home):
    """The idempotent-replay shape: the exact same row already exists on
    both sides (byte-identical). This is *not* a collision -- it must be
    silently skipped, not refused and not duplicated."""
    register_namespace("same-src")
    with Store("same-src") as st:
        r = st.observe("identical content")
        shared_id = r.memory_id
        event_id = st.conn.execute(
            "SELECT event_id FROM memories WHERE id=?", (shared_id,)
        ).fetchone()[0]
        session_id = st.conn.execute(
            "SELECT session_id FROM events WHERE id=?", (event_id,)
        ).fetchone()[0]
        row = dict(
            st.conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        )
        session_row = dict(
            st.conn.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        )
        memory_row = dict(
            st.conn.execute(
                "SELECT * FROM memories WHERE id=?", (shared_id,)
            ).fetchone()
        )

    register_namespace("same-dst")
    with Store("same-dst"):
        pass  # initialize schema without writing content
    conn = sqlite3.connect(str(_db_path(reconcile_home, "same-dst")))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    session_cols = list(session_row.keys())
    conn.execute(
        f"INSERT INTO sessions({','.join(session_cols)}) "
        f"VALUES ({','.join('?' for _ in session_cols)})",
        tuple(session_row[c] for c in session_cols),
    )
    cols = list(row.keys())
    conn.execute(
        f"INSERT INTO events({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
        tuple(row[c] for c in cols),
    )
    mem_cols = list(memory_row.keys())
    conn.execute(
        f"INSERT INTO memories({','.join(mem_cols)}) "
        f"VALUES ({','.join('?' for _ in mem_cols)})",
        tuple(memory_row[c] for c in mem_cols),
    )
    conn.commit()
    conn.close()

    plan = reconcile_namespaces("same-src", "same-dst")
    assert plan["tables"]["memories"]["already_consistent"] == 1
    assert plan["tables"]["memories"]["insert_into_target"] == 0
    assert plan["tables"]["memories"]["colliding_ids"] == []

    applied = reconcile_namespaces(
        "same-src", "same-dst", apply=True, plan_digest=plan["plan_digest"]
    )
    with Store("same-dst", create=False) as st:
        assert st.stats()["memories"] == 1


def test_secondary_key_collision_on_idempotency_key_refuses(reconcile_home):
    register_namespace("idk-src")
    with Store("idk-src") as st:
        st.observe("keyed content", idempotency_key="shared-idem-key")

    register_namespace("idk-dst")
    with Store("idk-dst") as st:
        st.observe("unrelated content", idempotency_key="shared-idem-key")

    with pytest.raises(
        NamespaceCollisionError, match="collide on a unique column other than id"
    ):
        reconcile_namespaces("idk-src", "idk-dst")


# ---------------------------------------------------------------------------
# Schema version mismatches are refused.
# ---------------------------------------------------------------------------


def test_mismatched_schema_version_refuses_in_either_direction(reconcile_home):
    register_namespace("stale-ns")
    conn = sqlite3.connect(str(_db_path(reconcile_home, "stale-ns")))
    _init_namespace_schema(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION - 1),),
    )
    conn.commit()
    conn.close()

    register_namespace("fresh-ns")
    with Store("fresh-ns") as st:
        st.observe("fresh content")

    with pytest.raises(NamespaceMigrationError, match="schema version"):
        reconcile_namespaces("stale-ns", "fresh-ns")
    with pytest.raises(NamespaceMigrationError, match="schema version"):
        reconcile_namespaces("fresh-ns", "stale-ns")


# ---------------------------------------------------------------------------
# Column ordinals may differ; column sets and declarations may not.
# ---------------------------------------------------------------------------

# The v1 table definitions verbatim, from before any column that schema
# versions 2-13 append with ALTER TABLE existed. `_init_namespace_schema`
# creates tables with IF NOT EXISTS, so laying these down first and stamping
# the database at v1 makes a later open run the real migration ladder over
# them. Nothing else reproduces the ordinals a migrated namespace carries.
_V1_NAMESPACE_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    source TEXT,
    meta TEXT
);
CREATE TABLE events (
    id TEXT PRIMARY KEY,
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
CREATE TABLE memories (
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
"""


def _seed_namespace_tables(home: Path, label: str, script: str, version: int) -> None:
    """Register `label` and pre-create some of its tables from raw DDL.

    Whatever `script` does not define is created normally on the first open;
    `version` is what that open sees, so it decides which migrations run.
    """
    register_namespace(label)
    conn = sqlite3.connect(str(_db_path(home, label)))
    try:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS meta ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL);\n" + script
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(version),),
        )
        conn.commit()
    finally:
        conn.close()


def _column_names(home: Path, label: str, table: str) -> list[str]:
    conn = sqlite3.connect(str(_db_path(home, label)))
    try:
        return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def _stored_schema_version(home: Path, label: str) -> int:
    conn = sqlite3.connect(str(_db_path(home, label)))
    try:
        return int(
            conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()[0]
        )
    finally:
        conn.close()


def test_alter_migrated_and_fresh_namespaces_reconcile_across_column_ordinals(
    reconcile_home,
):
    """A namespace that reached the current schema through ALTER TABLE
    reconciles with one created fresh at it, in both directions.

    This is the pairing reconcile exists for -- a legacy namespace and the
    fork that superseded it -- and the two ways of arriving at schema v13
    order `events` differently: `idempotency_key` is created second but
    appended thirteenth. The column sets are identical and every row is
    read, diffed, and inserted by column name, so the ordinals are not a
    difference in anything the copy depends on.
    """
    _seed_namespace_tables(reconcile_home, "migrated-ns", _V1_NAMESPACE_SCHEMA, 1)
    with Store("migrated-ns") as st:
        source_ids = [st.observe(f"migrated memory {i}").memory_id for i in range(4)]

    register_namespace("created-fresh-ns")
    with Store("created-fresh-ns") as st:
        target_ids = [st.observe(f"fresh memory {i}").memory_id for i in range(3)]

    # The divergence under test, asserted rather than assumed: both are at the
    # current schema version, `events` holds the same columns in both, and
    # they sit at different ordinals.
    assert _stored_schema_version(reconcile_home, "migrated-ns") == SCHEMA_VERSION
    assert _stored_schema_version(reconcile_home, "created-fresh-ns") == SCHEMA_VERSION
    migrated = _column_names(reconcile_home, "migrated-ns", "events")
    fresh = _column_names(reconcile_home, "created-fresh-ns", "events")
    assert sorted(migrated) == sorted(fresh)
    assert migrated != fresh
    assert migrated.index("idempotency_key") != fresh.index("idempotency_key")

    # Dry-run only, and taken before the apply below changes either side.
    reverse = reconcile_namespaces("created-fresh-ns", "migrated-ns")
    assert reverse["tables"]["memories"]["insert_into_target"] == len(target_ids)

    plan, applied = _plan_and_apply("migrated-ns", "created-fresh-ns")
    assert plan["tables"]["memories"]["insert_into_target"] == len(source_ids)
    assert applied["applied"] is True

    with Store("created-fresh-ns", create=False) as st:
        for mid in source_ids + target_ids:
            assert st.get_memory(mid) is not None, f"missing {mid} after reconcile"
        assert st.stats()["memories"] == len(source_ids) + len(target_ids)
        # Every copied row landed under the right column, not merely present:
        # a positional copy between these two tables would silently transpose
        # `idempotency_key` into `session_id`.
        for i, mid in enumerate(source_ids):
            assert st.get_memory(mid)["content"] == f"migrated memory {i}"


def test_column_present_in_only_one_namespace_still_refuses(reconcile_home):
    """An added column is a real difference and still refuses, both ways.

    ALTER TABLE does not touch `meta`, so both namespaces keep reporting the
    current schema version and this reaches the column check rather than the
    version check above it.
    """
    register_namespace("stray-src")
    register_namespace("stray-dst")
    with Store("stray-src") as st:
        st.observe("src content")
    with Store("stray-dst") as st:
        st.observe("dst content")

    conn = sqlite3.connect(str(_db_path(reconcile_home, "stray-src")))
    conn.execute("ALTER TABLE events ADD COLUMN stray_column TEXT")
    conn.commit()
    conn.close()

    assert _stored_schema_version(reconcile_home, "stray-src") == SCHEMA_VERSION
    with pytest.raises(NamespaceMigrationError, match="stray_column"):
        reconcile_namespaces("stray-src", "stray-dst")
    with pytest.raises(NamespaceMigrationError, match="stray_column"):
        reconcile_namespaces("stray-dst", "stray-src")


def _relax_events_content_not_null(home: Path, label: str) -> None:
    """Rebuild `label`'s events table with `content` nullable, rows preserved.

    The replacement is generated from the live table's own PRAGMA table_info,
    so column names, order, types, defaults, and primary key all survive
    untouched and `notnull` on `content` is the only thing that moves. CHECK
    constraints and foreign keys are lost in the rebuild; table_info reports
    neither, so neither is what the column check compares.
    """
    conn = sqlite3.connect(str(_db_path(home, label)))
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        # Keeps the RENAME below from reparsing `memories`, whose foreign key
        # still names a table that is dropped part-way through the rebuild.
        conn.execute("PRAGMA legacy_alter_table=ON")
        rows = conn.execute("PRAGMA table_info(events)").fetchall()
        parts = []
        for _cid, name, ctype, notnull, default, pk in rows:
            decl = f"{name} {ctype}"
            if pk:
                decl += " PRIMARY KEY"
            if notnull and name != "content":
                decl += " NOT NULL"
            if default is not None:
                decl += f" DEFAULT {default}"
            parts.append(decl)
        conn.executescript(
            "CREATE TABLE events_rebuilt (\n    "
            + ",\n    ".join(parts)
            + "\n);\n"
            "INSERT INTO events_rebuilt SELECT * FROM events;\n"
            "DROP TABLE events;\n"
            "ALTER TABLE events_rebuilt RENAME TO events;\n"
            "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);\n"
            "CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time);\n"
            "CREATE INDEX IF NOT EXISTS idx_events_tier ON events(tier);\n"
        )
        conn.commit()
    finally:
        conn.close()


def test_same_columns_in_same_order_with_a_changed_declaration_refuses(reconcile_home):
    """Dropping NOT NULL from `events.content` refuses even though the column
    names and their ordinal positions match exactly.

    Comparing by name is not the same as comparing names only. This pair is
    indistinguishable to a comparison that reads names out of PRAGMA
    table_info and nothing else, and merging it would push a NULL `content`
    into a table that forbids one, part-way through the copy.
    """
    register_namespace("decl-src")
    register_namespace("decl-dst")
    with Store("decl-src") as st:
        st.observe("src content")
    with Store("decl-dst") as st:
        st.observe("dst content")

    _relax_events_content_not_null(reconcile_home, "decl-src")

    # Both namespaces are otherwise untouched, so `events.content` is the only
    # thing that differs -- and it differs in nothing PRAGMA table_info reports
    # except `notnull`.
    assert _stored_schema_version(reconcile_home, "decl-src") == SCHEMA_VERSION
    assert _column_names(reconcile_home, "decl-src", "events") == _column_names(
        reconcile_home, "decl-dst", "events"
    )
    with pytest.raises(NamespaceMigrationError, match="content"):
        reconcile_namespaces("decl-src", "decl-dst")
    with pytest.raises(NamespaceMigrationError, match="content"):
        reconcile_namespaces("decl-dst", "decl-src")


# ---------------------------------------------------------------------------
# State drift between dry-run and apply is refused.
# ---------------------------------------------------------------------------


def test_state_change_after_dry_run_before_apply_is_refused(reconcile_home):
    register_namespace("drift-src")
    register_namespace("drift-dst")
    with Store("drift-src") as st:
        st.observe("original content")
    with Store("drift-dst") as st:
        st.observe("original dst content")

    plan = reconcile_namespaces("drift-src", "drift-dst")

    with Store("drift-src", create=False) as st:
        st.observe("a new row added after the dry-run")

    dst_before = _db_path(reconcile_home, "drift-dst").read_bytes()
    with pytest.raises(NamespaceMigrationError, match="does not match"):
        reconcile_namespaces(
            "drift-src", "drift-dst", apply=True, plan_digest=plan["plan_digest"]
        )
    assert _db_path(reconcile_home, "drift-dst").read_bytes() == dst_before


def test_apply_requires_plan_digest(reconcile_home):
    register_namespace("nodig-src")
    register_namespace("nodig-dst")
    with pytest.raises(NamespaceMigrationError, match="plan_digest"):
        reconcile_namespaces("nodig-src", "nodig-dst", apply=True)


def test_same_namespace_refuses(reconcile_home):
    register_namespace("solo-ns")
    with pytest.raises(NamespaceMigrationError, match="nothing to reconcile"):
        reconcile_namespaces("solo-ns", "solo-ns")


def test_unknown_namespace_refuses_on_either_side(reconcile_home):
    register_namespace("known-ns")
    with pytest.raises(UnknownNamespaceError):
        reconcile_namespaces("does-not-exist", "known-ns")
    with pytest.raises(UnknownNamespaceError):
        reconcile_namespaces("known-ns", "does-not-exist")


# ---------------------------------------------------------------------------
# Backups: created, verified, and restore the pre-merge state.
# ---------------------------------------------------------------------------


def test_backup_created_verified_permissioned_for_both_namespaces(reconcile_home):
    register_namespace("bkp-src")
    register_namespace("bkp-dst")
    with Store("bkp-src") as st:
        st.observe("backup src content")
    with Store("bkp-dst") as st:
        st.observe("backup dst content")

    plan, applied = _plan_and_apply("bkp-src", "bkp-dst")

    for key in ("source_backup", "target_backup"):
        backup = applied[key]
        path = Path(backup["path"])
        assert path.parent == reconcile_home / "backups"
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert hashlib.sha256(path.read_bytes()).hexdigest() == backup["sha256"]
        check = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        check.close()
    assert applied["source_backup"]["path"] != applied["target_backup"]["path"]


def test_source_backup_restores_the_exact_pre_merge_source_content(reconcile_home):
    register_namespace("restore-src")
    with Store("restore-src") as st:
        ids = [st.observe(f"restore src {i}").memory_id for i in range(4)]
    register_namespace("restore-dst")
    with Store("restore-dst") as st:
        st.observe("restore dst 0")

    plan, applied = _plan_and_apply("restore-src", "restore-dst")
    source_backup_path = Path(applied["source_backup"]["path"])

    # Prove the backup itself, opened directly, already contains the exact
    # pre-merge source content (source is never written to, so "restore"
    # and "current state" coincide here -- the interesting property is that
    # the backup is independently readable and correct).
    check = sqlite3.connect(f"{source_backup_path.as_uri()}?mode=ro&immutable=1", uri=True)
    check.row_factory = sqlite3.Row
    backup_ids = {
        row["id"] for row in check.execute("SELECT id FROM memories").fetchall()
    }
    check.close()
    assert backup_ids == set(ids)


def test_target_backup_restores_pre_merge_state_after_live_mutation(reconcile_home):
    """The meaningful restore case: TARGET *is* mutated by apply, so its
    backup must let an operator get back to the exact pre-merge state."""
    register_namespace("trestore-src")
    with Store("trestore-src") as st:
        st.observe("new content that will be merged in")
    register_namespace("trestore-dst")
    with Store("trestore-dst") as st:
        original_ids = [st.observe(f"trestore dst {i}").memory_id for i in range(3)]

    plan, applied = _plan_and_apply("trestore-src", "trestore-dst")
    target_backup_path = Path(applied["target_backup"]["path"])

    with Store("trestore-dst", create=False) as st:
        merged_count = st.stats()["memories"]
    assert merged_count == len(original_ids) + 1

    # Restore: stop everything referencing the namespace, drop any live WAL
    # sidecars, and copy the verified backup back over the live db path.
    target_path = _db_path(reconcile_home, "trestore-dst")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(target_path) + suffix)
        sidecar.unlink(missing_ok=True)
    shutil.copy2(target_backup_path, target_path)

    restored = sqlite3.connect(str(target_path))
    restored.row_factory = sqlite3.Row
    restored_ids = {
        row["id"] for row in restored.execute("SELECT id FROM memories").fetchall()
    }
    restored.close()
    assert restored_ids == set(original_ids)
    assert len(restored_ids) == len(original_ids)


def test_target_backup_covers_every_row_the_merge_writes_onto(
    reconcile_home, monkeypatch
):
    """A write racing the backup must not be merged-over but unbacked-up.

    The lock reconcile takes only serializes migrations, not observe(), so a
    live host keeps writing to TARGET throughout an apply. If the backup is
    taken before the write transaction, a row landing in that window is in
    the merged database and absent from the backup -- and the documented
    recovery (restore from HAUNT_HOME/backups) silently discards it. The
    invariant that closes the window: every TARGET row the merge reads is
    covered by the backup taken for that merge.
    """
    register_namespace("race-src")
    with Store("race-src") as st:
        st.observe("source row to merge in")
    register_namespace("race-dst")
    with Store("race-dst") as st:
        st.observe("dst row present before apply")

    plan = reconcile_namespaces("race-src", "race-dst")
    backed_up = threading.Event()
    raced = threading.Event()
    racing_id: list[str] = []
    racing_error: list[BaseException] = []
    # Opened up front so the thread contends only on the write itself, not
    # on writer-open configuration.
    racer_store = Store("race-dst", create=False)

    def race_the_backup() -> None:
        try:
            assert backed_up.wait(timeout=30)
            racing_id.append(racer_store.observe("write racing the merge").memory_id)
        except BaseException as exc:  # reported by the assertions below
            racing_error.append(exc)
        finally:
            raced.set()

    real_backup = store._backup_namespace_database

    def backup_then_yield(store_obj, *, purpose):
        backup = real_backup(store_obj, purpose=purpose)
        if purpose == "reconcile-target":
            backed_up.set()
            # Returns immediately when the racing write can proceed; times
            # out when the transaction correctly holds it off until commit.
            raced.wait(timeout=2.0)
        return backup

    real_execute = store._execute_reconciliation_writes
    merged_over: set[str] = set()

    def capture_target_rows(source_conn, target_conn, **kwargs):
        merged_over.update(
            str(row[0])
            for row in target_conn.execute("SELECT id FROM memories").fetchall()
        )
        return real_execute(source_conn, target_conn, **kwargs)

    monkeypatch.setattr(store, "_backup_namespace_database", backup_then_yield)
    monkeypatch.setattr(store, "_execute_reconciliation_writes", capture_target_rows)
    thread = threading.Thread(target=race_the_backup)
    thread.start()
    try:
        applied = reconcile_namespaces(
            "race-src", "race-dst", apply=True, plan_digest=plan["plan_digest"]
        )
    finally:
        backed_up.set()
        thread.join(timeout=30)
        racer_store.close()

    assert not racing_error, f"racing write failed outright: {racing_error!r}"
    assert racing_id, "fixture assumption: the racing write must actually land"

    backup_conn = sqlite3.connect(
        f"{Path(applied['target_backup']['path']).as_uri()}?mode=ro&immutable=1", uri=True
    )
    backed_up_ids = {
        str(row[0]) for row in backup_conn.execute("SELECT id FROM memories").fetchall()
    }
    backup_conn.close()

    assert merged_over, "fixture assumption: the merge must read TARGET's own rows"
    assert merged_over <= backed_up_ids, (
        "the merge wrote on top of rows the target backup does not contain; "
        "restoring that backup would discard them"
    )
    with Store("race-dst", create=False) as st:
        assert st.get_memory(racing_id[0]) is not None, "the racing write must survive"


def test_backup_failure_leaves_target_completely_unchanged(reconcile_home, monkeypatch):
    register_namespace("bfail-src")
    with Store("bfail-src") as st:
        st.observe("would-be-merged content")
    register_namespace("bfail-dst")
    with Store("bfail-dst") as st:
        st.observe("original dst content")

    plan = reconcile_namespaces("bfail-src", "bfail-dst")
    dst_before = _db_path(reconcile_home, "bfail-dst").read_bytes()
    src_before = _db_path(reconcile_home, "bfail-src").read_bytes()

    def fail_backup(store_obj, *, purpose):
        raise NamespaceMigrationError(f"forced {purpose} backup failure")

    monkeypatch.setattr(store, "_backup_namespace_database", fail_backup)
    with pytest.raises(NamespaceMigrationError, match="forced"):
        reconcile_namespaces(
            "bfail-src", "bfail-dst", apply=True, plan_digest=plan["plan_digest"]
        )

    assert _db_path(reconcile_home, "bfail-dst").read_bytes() == dst_before
    assert _db_path(reconcile_home, "bfail-src").read_bytes() == src_before
    with Store("bfail-dst", create=False) as st:
        assert st.stats()["memories"] == 1


def test_write_failure_mid_transaction_rolls_back_target_completely(
    reconcile_home, monkeypatch
):
    register_namespace("wfail-src")
    with Store("wfail-src") as st:
        for i in range(3):
            st.observe(f"wfail src {i}")
    register_namespace("wfail-dst")
    with Store("wfail-dst") as st:
        st.observe("wfail dst original")

    plan = reconcile_namespaces("wfail-src", "wfail-dst")

    real_execute = store._execute_reconciliation_writes

    def flaky_execute(source_conn, target_conn, **kwargs):
        # Let the fresh in-transaction diff run, then blow up before it
        # returns so the caller's rollback path is exercised for real.
        real_execute(source_conn, target_conn, **kwargs)
        raise RuntimeError("simulated mid-write failure")

    monkeypatch.setattr(store, "_execute_reconciliation_writes", flaky_execute)
    with pytest.raises(RuntimeError, match="simulated"):
        reconcile_namespaces(
            "wfail-src", "wfail-dst", apply=True, plan_digest=plan["plan_digest"]
        )

    with Store("wfail-dst", create=False) as st:
        assert st.stats()["memories"] == 1, "rollback must undo every insert"


# ---------------------------------------------------------------------------
# Correction lineage and structured provenance survive verbatim.
# ---------------------------------------------------------------------------


def test_correction_lineage_and_provenance_survive_verbatim(reconcile_home):
    register_namespace("lineage-src")
    with Store("lineage-src") as st:
        original = st.observe(
            "original claim", provenance=_import_envelope("lossless")
        )
        st.contradict(
            original.memory_id,
            idempotency_key="fix-1",
            replacement="corrected claim",
            reason="typo",
        )
        source_trace = st.trace(original.memory_id)
        assert source_trace["ok"] is True
        # find the replacement memory id from the trace's ordered members
        replacement_id = next(
            entry["memory_id"]
            for entry in source_trace["members"]
            if entry.get("memory_id") and entry["memory_id"] != original.memory_id
        )
        source_replacement_detail = st.get_memory(replacement_id)

    register_namespace("lineage-dst")
    with Store("lineage-dst") as st:
        st.observe("unrelated dst content")

    plan, applied = _plan_and_apply("lineage-src", "lineage-dst")
    assert plan["tables"]["corrections"]["insert_into_target"] == 1

    with Store("lineage-dst", create=False) as st:
        dst_trace = st.trace(original.memory_id)
        dst_replacement_detail = st.get_memory(replacement_id)
        original_detail = st.get_memory(original.memory_id)

    assert original_detail["content"] == "original claim"
    assert original_detail["provenance"]["source_native_id"] == "消息-雪-🧊"
    assert original_detail["provenance"]["fidelity"] == "lossless"
    assert dst_replacement_detail["content"] == "corrected claim"
    assert [entry.get("memory_id") for entry in dst_trace["members"]] == [
        entry.get("memory_id") for entry in source_trace["members"]
    ]
    assert dst_trace["members"][-1]["memory_id"] == source_replacement_detail["memory_id"]


def test_purge_tombstone_in_a_correction_chain_survives_verbatim(reconcile_home):
    """A chain A -> B -> C where A was privacy-purged leaves an allowlisted
    tombstone standing in for A (see E1). That tombstone row and the
    corrections that reference it by `target_tombstone_id` must copy into
    TARGET exactly like any other row -- lineage survives even though the
    original erased content, by design, never existed to copy."""
    register_namespace("tomb-src")
    with Store("tomb-src") as st:
        a = st.observe("original A content", session_id="tomb-sess")
        c1 = st.contradict(
            a.memory_id,
            replacement="replacement B content",
            idempotency_key="tomb-c1",
            reason="first correction",
        )
        b_id = c1["replacement_memory_id"]
        c2 = st.contradict(
            b_id,
            replacement="replacement C content",
            idempotency_key="tomb-c2",
            reason="second correction",
        )
        c_id = c2["replacement_memory_id"]
        st.purge(a.memory_id)
        source_trace = st.trace(c_id)
        assert source_trace["ok"] is True
        tombstone_ids = [
            m["tombstone_id"] for m in source_trace["members"] if "tombstone_id" in m
        ]
        assert len(tombstone_ids) == 1, "fixture assumption: purging A leaves one tombstone"

    with Store("tomb-src", create=False) as st:
        src_counts = _logical_counts(st.conn)
    assert src_counts["lineage_tombstones"] == 1
    assert src_counts["corrections"] == 2

    register_namespace("tomb-dst")
    with Store("tomb-dst") as st:
        st.observe("unrelated tomb dst content")

    plan, applied = _plan_and_apply("tomb-src", "tomb-dst")
    assert plan["tables"]["lineage_tombstones"]["insert_into_target"] == 1
    assert plan["tables"]["corrections"]["insert_into_target"] == 2

    with Store("tomb-dst", create=False) as st:
        dst_trace = st.trace(c_id)
        row = st.conn.execute(
            "SELECT tombstone_id, status FROM lineage_tombstones WHERE tombstone_id=?",
            (tombstone_ids[0],),
        ).fetchone()

    assert row is not None and row["status"] == "erased"
    assert [
        (m.get("memory_id"), m.get("tombstone_id")) for m in dst_trace["members"]
    ] == [
        (m.get("memory_id"), m.get("tombstone_id")) for m in source_trace["members"]
    ]
    assert dst_trace["lineage_status"] == source_trace["lineage_status"]


# ---------------------------------------------------------------------------
# Embeddings are stripped and re-queued; meta and vec state are never copied.
# ---------------------------------------------------------------------------


def test_embeddings_stripped_fts_rebuilt_jobs_requeued(reconcile_home):
    register_namespace("embed-src")
    with Store("embed-src") as st:
        r = st.observe("unique zzyzx searchable phrase")
        mem_id = r.memory_id
        st.conn.execute(
            "UPDATE memories SET embedding=? WHERE id=?", (b"\x01\x02\x03\x04", mem_id)
        )
        st.conn.commit()

    register_namespace("embed-dst")
    with Store("embed-dst") as st:
        st.observe("unrelated dst content")

    _plan_and_apply("embed-src", "embed-dst")

    with Store("embed-dst", create=False) as st:
        row = st.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        assert row[0] is None
        job = st.conn.execute(
            "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (mem_id,)
        ).fetchone()
        assert job is not None
        hits = st.conn.execute(
            "SELECT id FROM memories_fts WHERE memories_fts MATCH 'zzyzx'"
        ).fetchall()
        assert any(row["id"] == mem_id for row in hits)

    with Store("embed-src", create=False) as st:
        still_there = st.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (mem_id,)
        ).fetchone()[0]
        assert still_there == b"\x01\x02\x03\x04"


# ---------------------------------------------------------------------------
# C-series adversarial review defect (merge blocker): re-queueing on
# reconcile must mirror SOURCE's own embed-capture outcome, not blindly
# enqueue every copied row. Before this fix, a row that SOURCE deliberately
# captured-but-never-embedded (skip_embedding=True, the C6 policy, or
# HAUNT_EMBED_EXCLUDE_TOOLS at the hook layer) got queued in TARGET and
# silently embedded on the next drain -- exactly the tool exhaust the
# policy exists to keep out of the vector index. See
# _reconcile_requeue_embedding in src/haunt/store.py for the fix and the
# exhausted-row reasoning. The last test in this block
# (test_reconcile_tolerates_source_missing_embedding_jobs_table) instead
# guards the fix's OWN blast radius: it now reads SOURCE's embedding_jobs
# table, which a never-migrated read-only SOURCE can genuinely lack.
# ---------------------------------------------------------------------------


def test_reconcile_excludes_skip_embedding_row_from_target_queue(reconcile_home):
    """The core defect repro: a skip_embedding=True row in SOURCE (the C6
    capture policy -- what HAUNT_EMBED_EXCLUDE_TOOLS uses at the hook
    layer) must come out the other side of reconcile still unembedded AND
    still unqueued in TARGET. Queuing it would silently embed policy-
    excluded tool exhaust on the next drain."""
    register_namespace("policy-src")
    with Store("policy-src") as st:
        r = st.observe(
            "$ curl -s https://example.com/secret | jq .token",
            tool_name="Bash",
            skip_embedding=True,
        )
        mem_id = r.memory_id
        # Sanity: reproduce the precondition this defect assumes -- SOURCE
        # really did admit this with neither an embedding nor a queued job.
        src_row = st.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        assert src_row["embedding"] is None
        assert (
            st.conn.execute(
                "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (mem_id,)
            ).fetchone()
            is None
        )

    register_namespace("policy-dst")
    with Store("policy-dst") as st:
        st.observe("unrelated dst content")

    _plan_and_apply("policy-src", "policy-dst")

    with Store("policy-dst", create=False) as st:
        row = st.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        assert row is not None, "the row itself must still be copied"
        assert row["embedding"] is None
        job = st.conn.execute(
            "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (mem_id,)
        ).fetchone()
        assert job is None, (
            "skip_embedding row must not be queued for embedding in TARGET"
        )


def test_reconcile_requeues_row_actually_embedded_via_observe(
    reconcile_home, monkeypatch
):
    """The mirror-image case, driven through the real capture-policy code
    path (observe() with skip_embedding=False) rather than a raw SQL
    UPDATE: a row SOURCE genuinely embedded must be re-queued in TARGET,
    since reconcile drops the embedding itself and TARGET needs to
    recompute it under its own model. Complements
    test_embeddings_stripped_fts_rebuilt_jobs_requeued above, which
    exercises the same outcome by injecting a raw embedding blob."""
    import haunt.store as store_mod

    def fake_embed_one(text):
        return [0.1, 0.2, 0.3, 0.4] if (text or "").strip() else None

    monkeypatch.setattr(store_mod, "embed_one", fake_embed_one)

    register_namespace("embedded-src")
    with Store("embedded-src") as st:
        r = st.observe("genuinely embedded content")
        mem_id = r.memory_id
        src_row = st.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        assert src_row["embedding"] is not None  # sanity: real embed path ran

    register_namespace("embedded-dst")
    with Store("embedded-dst") as st:
        st.observe("unrelated dst content")

    _plan_and_apply("embedded-src", "embedded-dst")

    with Store("embedded-dst", create=False) as st:
        row = st.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        assert row["embedding"] is None  # dropped, per documented reconcile behavior
        job = st.conn.execute(
            "SELECT attempts FROM embedding_jobs WHERE memory_id=?", (mem_id,)
        ).fetchone()
        assert job is not None, "previously-embedded row must be re-queued in TARGET"
        assert job["attempts"] == 0


def test_reconcile_preserves_pending_embedding_job_state(reconcile_home):
    """A row SOURCE never embedded but did queue (attempts > 0 from a
    prior transient failure, still under the cap) must arrive in TARGET
    with that exact state -- attempts, last_error, and queued_at intact --
    rather than a fresh `INSERT ... attempts=0` that would discard the
    retry history and let it silently get another full retry budget."""
    register_namespace("pending-src")
    with Store("pending-src") as st:
        r = st.observe("deferred content awaiting embedding", defer_embedding=True)
        mem_id = r.memory_id
        src_memory = st.conn.execute(
            "SELECT embedding, created_at FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        assert src_memory["embedding"] is None
        # queued_at is deliberately set far from created_at: observe()
        # happens to leave them equal at admission (both come from the
        # same `ts`), which would let a reconcile that reuses
        # `created_at` (the pre-fix behavior for the *embedded* branch)
        # accidentally look right here too. Forcing them apart makes the
        # "queued_at survives verbatim" assertion below actually
        # discriminate that from "queued_at was recomputed".
        st.conn.execute(
            "UPDATE embedding_jobs SET attempts=2, last_error=?, queued_at=? "
            "WHERE memory_id=?",
            (
                "simulated transient failure",
                "2020-01-01T00:00:00.000000+00:00",
                mem_id,
            ),
        )
        st.conn.commit()
        job_before = dict(
            st.conn.execute(
                "SELECT queued_at, attempts, last_error FROM embedding_jobs "
                "WHERE memory_id=?",
                (mem_id,),
            ).fetchone()
        )
        assert job_before["queued_at"] != src_memory["created_at"]

    register_namespace("pending-dst")
    with Store("pending-dst") as st:
        st.observe("unrelated dst content")

    _plan_and_apply("pending-src", "pending-dst")

    with Store("pending-dst", create=False) as st:
        job_after = st.conn.execute(
            "SELECT queued_at, attempts, last_error FROM embedding_jobs "
            "WHERE memory_id=?",
            (mem_id,),
        ).fetchone()
        assert job_after is not None
        assert job_after["attempts"] == job_before["attempts"] == 2
        assert (
            job_after["last_error"]
            == job_before["last_error"]
            == "simulated transient failure"
        )
        assert job_after["queued_at"] == job_before["queued_at"]


def test_reconcile_does_not_reset_attempts_for_exhausted_row(reconcile_home):
    """The exhausted-row decision, stated explicitly and verified two
    ways: a SOURCE row already at HAUNT_EMBED_MAX_ATTEMPTS must land in
    TARGET with its `attempts` counter copied verbatim, not reset to 0.
    max_attempts is evaluated dynamically at drain time (never persisted
    per-row), so copying the raw count is what keeps a copied exhausted
    row excluded from TARGET's own process_embedding_jobs selection
    immediately -- resetting it would resurrect a known-bad row into a
    fresh retry loop, which this explicitly refuses to do."""
    from haunt.store import EMBED_MAX_ATTEMPTS_DEFAULT

    register_namespace("exhausted-src")
    with Store("exhausted-src") as st:
        r = st.observe("content that always fails to embed", defer_embedding=True)
        mem_id = r.memory_id
        st.conn.execute(
            "UPDATE embedding_jobs SET attempts=?, last_error=? WHERE memory_id=?",
            (EMBED_MAX_ATTEMPTS_DEFAULT, "permanent failure", mem_id),
        )
        st.conn.commit()

    register_namespace("exhausted-dst")
    with Store("exhausted-dst") as st:
        st.observe("unrelated dst content")

    _plan_and_apply("exhausted-src", "exhausted-dst")

    with Store("exhausted-dst", create=False) as st:
        # 1) The raw column value: not reset to 0.
        job = st.conn.execute(
            "SELECT attempts, last_error FROM embedding_jobs WHERE memory_id=?",
            (mem_id,),
        ).fetchone()
        assert job is not None
        assert job["attempts"] == EMBED_MAX_ATTEMPTS_DEFAULT, (
            "copying an exhausted row must not reset its attempts counter"
        )
        assert job["last_error"] == "permanent failure"
        # 2) Behaviorally: TARGET's own stats() must classify it as
        # exhausted, not pending -- i.e. it does not silently get a fresh
        # retry budget just because it was copied. The unrelated
        # "unrelated dst content" row (queued fresh, attempts=0) is the
        # only pending one.
        stats = st.stats()
        assert stats["embedding_exhausted"] == 1
        assert stats["embedding_pending"] == 1
        row = st.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        assert row["embedding"] is None


def test_reconcile_tolerates_source_missing_embedding_jobs_table(reconcile_home):
    """A blast-radius check on the fix itself, not the original defect:
    SOURCE is opened via open_existing_readonly (a ReadOnlyStore), which
    deliberately never runs schema migration on open -- see its class
    docstring: "the latter [Store(create=False)] intentionally performs
    migration/configuration work for writers" and "do not repair a
    corrupt/old database while reading it." A namespace old enough to
    predate the embedding_jobs table -- exactly the C3 motivating shape,
    a long-lived legacy database like `ironscope` that was never since
    opened by a current writer -- can genuinely lack that table on disk.

    The pre-fix code never queried SOURCE at all (it only ever wrote to
    TARGET, which open_existing always migrates), so it had no dependency
    on SOURCE's embedding_jobs existing. Naively querying it from
    _reconcile_requeue_embedding would turn a merely-old, perfectly valid
    SOURCE database into a hard `sqlite3.OperationalError: no such table:
    embedding_jobs` for the *whole* reconcile apply -- a regression this
    fix must not introduce. Missing table must be handled exactly like
    "no job row": don't enqueue, don't crash.
    """
    register_namespace("old-schema-src")
    with Store("old-schema-src") as st:
        r = st.observe(
            "row from a namespace older than embedding_jobs", defer_embedding=True
        )
        mem_id = r.memory_id
        assert (
            st.conn.execute(
                "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (mem_id,)
            ).fetchone()
            is not None
        )
        st.conn.execute("DROP TABLE embedding_jobs")
        st.conn.commit()
        assert (
            st.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='embedding_jobs'"
            ).fetchone()
            is None
        )

    register_namespace("old-schema-dst")
    with Store("old-schema-dst") as st:
        st.observe("unrelated dst content")

    # The point of the test: this must not raise.
    _plan_and_apply("old-schema-src", "old-schema-dst")

    with Store("old-schema-dst", create=False) as st:
        row = st.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        assert row is not None, "the row itself must still be copied"
        assert row["embedding"] is None
        job = st.conn.execute(
            "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (mem_id,)
        ).fetchone()
        assert job is None, (
            "no job info was available from SOURCE (table absent), so "
            "TARGET must not enqueue -- same 'no positive signal' rule as "
            "the skip_embedding case, not a fresh INSERT"
        )


def test_meta_table_never_copied(reconcile_home):
    register_namespace("meta-src")
    with Store("meta-src") as st:
        st.observe("meta src content")
        st.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_dim', '999')"
        )
        st.conn.commit()

    register_namespace("meta-dst")
    with Store("meta-dst") as st:
        st.observe("meta dst content")
        before = dict(
            st.conn.execute("SELECT key, value FROM meta").fetchall()
        )

    _plan_and_apply("meta-src", "meta-dst")

    with Store("meta-dst", create=False) as st:
        after = dict(st.conn.execute("SELECT key, value FROM meta").fetchall())
    assert after == before
    assert after.get("embed_dim") != "999"


# ---------------------------------------------------------------------------
# Graph evidence (entities/relations) merges with foreign-key integrity.
# ---------------------------------------------------------------------------


def test_graph_evidence_merges_with_foreign_key_integrity(reconcile_home):
    register_namespace("graph-src")
    with Store("graph-src") as st:
        st.observe("Aron met Priya at the office", tier="episodic")
    register_namespace("graph-dst")
    with Store("graph-dst") as st:
        st.observe("Priya met Sam at the cafe", tier="episodic")

    with Store("graph-src", create=False) as st:
        src_entities = st.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    assert src_entities > 0, "fixture assumption: entity extraction produced rows"

    _plan_and_apply("graph-src", "graph-dst")

    with Store("graph-dst", create=False) as st:
        fk_violations = st.conn.execute("PRAGMA foreign_key_check").fetchall()
        assert fk_violations == []
        orphan_mentions = st.conn.execute(
            """SELECT COUNT(*) FROM entity_mentions m
               WHERE NOT EXISTS (SELECT 1 FROM events e WHERE e.id=m.event_id)
                  OR NOT EXISTS (SELECT 1 FROM entities en WHERE en.id=m.entity_id)"""
        ).fetchone()[0]
        assert orphan_mentions == 0


# ---------------------------------------------------------------------------
# C1/C2 are unaffected: this module never imports or monkeypatches
# infer_namespace_context/_registered_namespace_for_repo, and the CLI/API
# surfaces it adds are entirely new (`namespace reconcile` /
# `reconcile_namespaces`), not modifications of existing ones. The shared
# proof is the full suite, including tests/test_repo_binding.py, passing
# unmodified; this test only pins the specific refusal C1/C2 rely on C3 to
# eventually resolve, so a future change cannot silently make inference
# start guessing again.
# ---------------------------------------------------------------------------


def test_inference_still_refuses_to_guess_between_a_split_pair(reconcile_home, monkeypatch):
    from haunt.paths import infer_namespace_context

    project = reconcile_home / "ironscope"
    project.mkdir()
    register_namespace("ironscope")
    register_namespace("github.com-moderatesoup-ironscope")
    _patch_git_context(
        monkeypatch, "git@github.com:moderatesoup/ironscope.git", project
    )
    ns, _repo_path = infer_namespace_context(project)
    assert ns == "github.com-moderatesoup-ironscope"


# ---------------------------------------------------------------------------
# CLI surface.
# ---------------------------------------------------------------------------


def test_cli_reconcile_defaults_to_dry_run_and_apply_with_digest(reconcile_home):
    register_namespace("cli-src")
    with Store("cli-src") as st:
        st.observe("cli src content")
    register_namespace("cli-dst")
    with Store("cli-dst") as st:
        st.observe("cli dst content")

    runner = CliRunner()
    dry = runner.invoke(app, ["namespace", "reconcile", "cli-src", "cli-dst"])
    assert dry.exit_code == 0, dry.output
    assert '"mode": "dry-run"' in dry.output
    import json as _json

    digest = _json.loads(dry.output)["plan_digest"]

    with Store("cli-dst", create=False) as st:
        before = st.stats()["memories"]

    applied = runner.invoke(
        app,
        [
            "namespace", "reconcile", "cli-src", "cli-dst",
            "--apply", "--plan-digest", digest,
        ],
    )
    assert applied.exit_code == 0, applied.output
    assert '"applied": true' in applied.output

    with Store("cli-dst", create=False) as st:
        after = st.stats()["memories"]
    assert after == before + 1


def test_cli_reconcile_apply_without_digest_exits_nonzero(reconcile_home):
    register_namespace("cli2-src")
    register_namespace("cli2-dst")
    runner = CliRunner()
    result = runner.invoke(
        app, ["namespace", "reconcile", "cli2-src", "cli2-dst", "--apply"]
    )
    assert result.exit_code == 2
    assert "plan_digest" in result.output


def test_cli_reconcile_unknown_namespace_exits_nonzero(reconcile_home):
    register_namespace("cli3-dst")
    runner = CliRunner()
    result = runner.invoke(
        app, ["namespace", "reconcile", "does-not-exist", "cli3-dst"]
    )
    assert result.exit_code == 2
    assert "error:" in result.output


# ---------------------------------------------------------------------------
# Scale: closer to the real measured namespaces (up to 1491 rows) than the
# small fixtures above, to catch anything that only shows up with more rows
# than a handful (quadratic diffing, a query that silently truncates, etc).
# ---------------------------------------------------------------------------


def test_apply_accounts_for_every_row_at_realistic_scale(reconcile_home):
    register_namespace("scale-src")
    with Store("scale-src") as st:
        source_ids = [st.observe(f"scale source memory {i}") for i in range(250)]
        for i in range(0, 250, 25):
            st.contradict(
                source_ids[i].memory_id,
                idempotency_key=f"scale-fix-{i}",
                replacement=f"scale corrected memory {i}",
            )

    register_namespace("scale-dst")
    with Store("scale-dst") as st:
        target_ids = [st.observe(f"scale target memory {i}") for i in range(120)]

    with Store("scale-src", create=False) as st:
        src_counts = _logical_counts(st.conn)
    with Store("scale-dst", create=False) as st:
        dst_counts_before = _logical_counts(st.conn)

    plan = reconcile_namespaces("scale-src", "scale-dst")
    assert plan["tables"]["memories"]["insert_into_target"] == src_counts["memories"]
    applied = reconcile_namespaces(
        "scale-src", "scale-dst", apply=True, plan_digest=plan["plan_digest"]
    )
    assert applied["applied"] is True

    with Store("scale-dst", create=False) as st:
        dst_counts_after = _logical_counts(st.conn)
        for source_result in source_ids:
            assert st.get_memory(source_result.memory_id) is not None
        for target_result in target_ids:
            assert st.get_memory(target_result.memory_id) is not None

    for table in dst_counts_before:
        assert dst_counts_after[table] == dst_counts_before[table] + src_counts[table]

    # Idempotent replay at this scale too: nothing left to insert, nothing
    # duplicated.
    plan2 = reconcile_namespaces("scale-src", "scale-dst")
    assert plan2["total_rows_to_insert"] == 0
    reconcile_namespaces(
        "scale-src", "scale-dst", apply=True, plan_digest=plan2["plan_digest"]
    )
    with Store("scale-dst", create=False) as st:
        dst_counts_final = _logical_counts(st.conn)
    assert dst_counts_final == dst_counts_after


def test_every_content_table_is_in_the_reconcile_copy_set(reconcile_home):
    """No content table may be silently omitted from the copy set.

    A table present in the schema but absent from _RECONCILE_TABLES is
    invisible partial data loss: reconcile reports success, the operator
    deletes the drained source, and those rows are gone. Nothing else
    catches it -- `relations`, for one, carries no foreign keys, so
    PRAGMA foreign_key_check is structurally blind to its absence.

    Derived from the live schema rather than a hand-written list, so a
    table added later fails here until it is deliberately classified.
    """
    register_namespace("coverage-ns")
    with Store("coverage-ns") as st:
        live = {
            r[0]
            for r in st.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    copied = {name for name, _pk, _ignore in _RECONCILE_TABLES}
    # Deliberately not copied, each for a stated reason:
    #   meta          - per-database identity (schema_version, embed_dim);
    #                   TARGET keeps its own or the merge would corrupt it.
    #   embedding_jobs - not row-for-row copied: re-derived per copied
    #                   memories row from SOURCE's own embed/queue state
    #                   (see _reconcile_requeue_embedding) so a
    #                   skip_embedding-excluded row stays excluded instead
    #                   of being blindly re-queued in TARGET.
    #   memories_fts* - FTS5 shadow tables, rebuilt from copied memories.
    #   vec_memories* - vec0 shadow tables; embeddings are re-derived.
    #   sqlite_*      - SQLite internals.
    excluded = {"meta", "embedding_jobs"}
    unclassified = {
        t
        for t in live - copied - excluded
        if not t.startswith(("memories_fts", "vec_memories", "sqlite_"))
    }
    assert unclassified == set(), (
        f"table(s) {sorted(unclassified)} are neither copied by reconcile nor "
        "explicitly excluded; classify them before shipping"
    )


def test_relations_rows_actually_round_trip(reconcile_home):
    """Guards the `relations` copy path with content that really produces it.

    The row-count parity tests use fixture text that extracts no entities,
    so their relations assertion is 0 == 0 and passes even if the table is
    dropped from the copy set entirely. This one asserts on a non-zero
    count and compares the rows themselves.
    """
    register_namespace("rel-src")
    with Store("rel-src") as st:
        st.observe("Aron met Priya at the office", tier="episodic")
        src_rows = {
            tuple(r)
            for r in st.conn.execute(
                "SELECT id, src_entity, rel, dst_entity FROM relations"
            ).fetchall()
        }
    assert src_rows, "fixture assumption: this content must produce relations"

    register_namespace("rel-dst")
    with Store("rel-dst") as st:
        st.observe("unrelated destination row", tier="episodic")
        dst_before = st.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]

    _plan_and_apply("rel-src", "rel-dst")

    with Store("rel-dst", create=False) as st:
        dst_rows = {
            tuple(r)
            for r in st.conn.execute(
                "SELECT id, src_entity, rel, dst_entity FROM relations"
            ).fetchall()
        }
        dst_after = st.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]

    assert src_rows <= dst_rows, "every source relation must survive the merge"
    assert dst_after == dst_before + len(src_rows)


# ---------------------------------------------------------------------------
# Retiring the drained namespace: the operator step reconcile leaves undone.
# ---------------------------------------------------------------------------


def test_retire_refuses_a_namespace_that_is_not_drained(reconcile_home):
    register_namespace("keep-src")
    with Store("keep-src") as st:
        st.observe("row that only the source has")
    register_namespace("keep-dst")
    with Store("keep-dst") as st:
        st.observe("unrelated dst row")

    report = store.retire_namespace("keep-src", into="keep-dst")
    assert report["safe"] is False
    assert [b["kind"] for b in report["blockers"]] == ["undrained-rows"]
    assert report["undrained_rows"]["memories"] == 1

    with pytest.raises(AliasRetirementError, match="undrained-rows"):
        store.retire_namespace("keep-src", into="keep-dst", apply=True)

    assert store.namespace_exists("keep-src")
    assert _db_path(reconcile_home, "keep-src").exists()


def test_retire_after_reconcile_deregisters_and_removes_the_database(reconcile_home):
    """The commit message's own final operator step, with a command behind it."""
    register_namespace("drain-src")
    with Store("drain-src") as st:
        source_ids = [st.observe(f"drained row {i}").memory_id for i in range(3)]
    register_namespace("drain-dst")
    with Store("drain-dst") as st:
        st.observe("dst row")

    _plan_and_apply("drain-src", "drain-dst")
    dry = store.retire_namespace("drain-src", into="drain-dst")
    assert dry["safe"] is True
    assert dry["undrained_rows"] == {}
    assert dry["labels"] == ["drain-src"]
    assert store.namespace_exists("drain-src"), "dry-run must not deregister"

    applied = store.retire_namespace("drain-src", into="drain-dst", apply=True)
    assert applied["retired"] is True
    assert applied["stranded_rows"] == {}
    assert applied["database_removed"] is True

    assert not store.namespace_exists("drain-src")
    assert not _db_path(reconcile_home, "drain-src").exists()
    with pytest.raises(UnknownNamespaceError):
        store.open_existing_readonly("drain-src")

    backup = Path(applied["backup"]["path"])
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == applied["backup"]["sha256"]
    check = sqlite3.connect(f"{backup.as_uri()}?mode=ro&immutable=1", uri=True)
    backup_ids = {row[0] for row in check.execute("SELECT id FROM memories").fetchall()}
    check.close()
    assert backup_ids == set(source_ids), "the removed database must be recoverable"

    # The reconciled content is untouched, and the drained label is free again.
    with Store("drain-dst", create=False) as st:
        assert st.stats()["memories"] == len(source_ids) + 1
    register_namespace("drain-src")
    with Store("drain-src", create=False) as st:
        assert st.stats()["memories"] == 0, "a retired label must not re-adopt a file"


def test_cli_retire_defaults_to_dry_run(reconcile_home):
    register_namespace("clir-src")
    with Store("clir-src") as st:
        st.observe("clir src content")
    register_namespace("clir-dst")
    with Store("clir-dst") as st:
        st.observe("clir dst content")
    _plan_and_apply("clir-src", "clir-dst")

    runner = CliRunner()
    dry = runner.invoke(app, ["namespace", "retire", "clir-src", "--into", "clir-dst"])
    assert dry.exit_code == 0, dry.output
    assert '"mode": "dry-run"' in dry.output
    assert store.namespace_exists("clir-src")

    applied = runner.invoke(
        app, ["namespace", "retire", "clir-src", "--into", "clir-dst", "--apply"]
    )
    assert applied.exit_code == 0, applied.output
    assert '"retired": true' in applied.output
    assert not store.namespace_exists("clir-src")

    blocked = runner.invoke(
        app, ["namespace", "retire", "clir-dst", "--into", "clir-src"]
    )
    assert blocked.exit_code == 2
    assert "error:" in blocked.output


def test_retire_backup_failure_leaves_the_namespace_registered(
    reconcile_home, monkeypatch
):
    """A failed backup must not leave the label gone and the file orphaned.

    Deregistration used to commit first, so a backup failure retired the
    label, discarded the report, and left the operator with no output naming
    the database that survived at its own path.
    """
    register_namespace("bufail-src")
    with Store("bufail-src") as st:
        st.observe("drained row")
    register_namespace("bufail-dst")
    with Store("bufail-dst") as st:
        st.observe("dst row")
    _plan_and_apply("bufail-src", "bufail-dst")

    real_backup = store._backup_namespace_database

    def fail_backup(store_obj, *, purpose):
        raise NamespaceMigrationError(f"forced {purpose} backup failure")

    monkeypatch.setattr(store, "_backup_namespace_database", fail_backup)
    with pytest.raises(NamespaceMigrationError, match="forced retire"):
        store.retire_namespace("bufail-src", into="bufail-dst", apply=True)

    assert store.namespace_exists("bufail-src"), (
        "a failed backup must not deregister the namespace"
    )
    assert _db_path(reconcile_home, "bufail-src").exists()
    ro = store.open_existing_readonly("bufail-src")
    try:
        assert ro.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    finally:
        ro.close()

    # And the operation still completes once the backup can be taken.
    monkeypatch.setattr(store, "_backup_namespace_database", real_backup)
    applied = store.retire_namespace("bufail-src", into="bufail-dst", apply=True)
    assert applied["retired"] is True
    assert applied["database_removed"] is True


def test_namespace_backup_carries_rows_that_are_still_only_in_the_wal(reconcile_home):
    """A backup taken while a writer holds uncheckpointed frames is complete.

    Databases run journal_mode=WAL and the backup is a byte copy of a main
    file, so this is the case where committed rows could silently go missing.
    They do not: open_existing_readonly copies main and -wal into a private
    shadow and checkpoints *that* before opening it, so the file the backup
    copies is already materialized.
    """
    register_namespace("walbackup")
    canary = b"walonlycanaryf3k8"
    with Store("walbackup") as st:
        st.observe(f"committed row {canary.decode()}")
        db_path = _db_path(reconcile_home, "walbackup")
        assert canary not in db_path.read_bytes(), (
            "fixture assumption: the row must still be WAL-only"
        )
        assert canary in Path(f"{db_path}-wal").read_bytes()

        ro = store.open_existing_readonly("walbackup")
        try:
            backup = store._backup_namespace_database(ro, purpose="waltest")
        finally:
            ro.close()

    try:
        assert backup["integrity"] == "ok"
        copied = sqlite3.connect(
            f"{Path(backup['path']).as_uri()}?mode=ro&immutable=1", uri=True
        )
        try:
            found = copied.execute(
                "SELECT COUNT(*) FROM memories WHERE content LIKE ?",
                (f"%{canary.decode()}%",),
            ).fetchone()[0]
        finally:
            copied.close()
        assert found == 1, "the backup dropped a committed row that lived in the WAL"
    finally:
        backup.close()


def test_embedding_jobs_drift_between_the_dry_run_and_the_apply_is_refused(
    reconcile_home,
):
    """The apply copies embedding_jobs rows, so the digest must cover them.

    `attempts`/`last_error` are mutable queue state a background drain moves.
    They were copied verbatim into TARGET while sitting outside
    content_state_digest, so the operator authorized one plan and the apply
    wrote values that plan never saw.
    """
    register_namespace("jobs-src")
    with Store("jobs-src") as st:
        queued = st.observe("a source row waiting to be embedded").memory_id
        assert st.conn.execute(
            "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (queued,)
        ).fetchone() is not None
    register_namespace("jobs-dst")
    with Store("jobs-dst") as st:
        st.observe("an unrelated dst row")

    plan = reconcile_namespaces("jobs-src", "jobs-dst")

    with Store("jobs-src") as st:
        st.conn.execute(
            "UPDATE embedding_jobs SET attempts=4, last_error='drift' WHERE memory_id=?",
            (queued,),
        )
        st.conn.commit()

    with pytest.raises(NamespaceMigrationError, match="digest"):
        reconcile_namespaces(
            "jobs-src", "jobs-dst", apply=True, plan_digest=plan["plan_digest"]
        )

    with Store("jobs-dst", create=False) as st:
        assert st.conn.execute(
            "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (queued,)
        ).fetchone() is None, "nothing may be written once the digest is refused"

    # A fresh cycle authorizes the drifted state and then copies it.
    replan = reconcile_namespaces("jobs-src", "jobs-dst")
    reconcile_namespaces(
        "jobs-src", "jobs-dst", apply=True, plan_digest=replan["plan_digest"]
    )
    with Store("jobs-dst", create=False) as st:
        job = st.conn.execute(
            "SELECT attempts, last_error FROM embedding_jobs WHERE memory_id=?",
            (queued,),
        ).fetchone()
        assert (job["attempts"], job["last_error"]) == (4, "drift")


def test_embedding_jobs_stays_out_of_the_plans_per_table_report(reconcile_home):
    """Digest-only means digest-only: no diff, no collision semantics."""
    register_namespace("jobs-report-src")
    with Store("jobs-report-src") as st:
        st.observe("source row")
    register_namespace("jobs-report-dst")
    with Store("jobs-report-dst") as st:
        st.observe("dst row")

    plan = reconcile_namespaces("jobs-report-src", "jobs-report-dst")
    assert set(plan["tables"]) == {table for table, _pk, _ig in _RECONCILE_TABLES}
    assert "embedding_jobs" not in plan["tables"]
