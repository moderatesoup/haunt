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
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import haunt.paths as paths
import haunt.store as store
from haunt.cli import app
from haunt.paths import registry_path
from haunt.store import (
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

    def flaky_execute(source_conn, target_conn):
        # Let the fresh in-transaction diff run, then blow up before it
        # returns so the caller's rollback path is exercised for real.
        real_execute(source_conn, target_conn)
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
    #   embedding_jobs - regenerated: copied rows are re-queued, not moved.
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
