"""Tests for Store.purge — hard-delete walks the full provenance chain."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from haunt.recall import recall
from haunt.store import Store, observe


def _canary_counts(db_path: Path, needle: bytes) -> dict[str, int]:
    """Raw occurrences of needle across the namespace file and its sidecars."""
    paths = (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm"))
    return {p.name: p.read_bytes().count(needle) for p in paths if p.exists()}


def test_purge_removes_memory_and_keeps_other(haunt_env):
    """Observe two memories, purge one, assert the other remains intact."""
    r1 = observe("alpha memory XRAY-11 unique text", namespace="default", role="user")
    r2 = observe("beta memory ZULU-22 different text", namespace="default", role="assistant")

    with Store("default") as st:
        result = st.purge(r1.memory_id)

    assert result["ok"] is True
    assert "memory_id" not in result
    assert "event_id" not in result

    with Store("default") as st:
        gone = st.conn.execute(
            "SELECT id FROM memories WHERE id=?", (r1.memory_id,)
        ).fetchone()
        assert gone is None, "purged memory should not exist"

        kept = st.conn.execute(
            "SELECT id FROM memories WHERE id=?", (r2.memory_id,)
        ).fetchone()
        assert kept is not None, "other memory should still exist"


def test_purge_removes_fts_entry(haunt_env):
    """After purge, FTS should not contain the deleted memory."""
    unique = "FTS-PURGE-CANARY-77 this phrase only appears once"
    r = observe(unique, namespace="default", role="user")

    with Store("default") as st:
        fts_before = st.conn.execute(
            "SELECT id FROM memories_fts WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert fts_before is not None

        st.purge(r.memory_id)

        fts_after = st.conn.execute(
            "SELECT id FROM memories_fts WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert fts_after is None, "FTS entry should be deleted after purge"


def test_purge_removes_vec_entry(haunt_env):
    """After purge, vec_memories should not contain the deleted memory."""
    r = observe("VEC-PURGE-CANARY vector data test", namespace="default", role="user")

    with Store("default") as st:
        if st.vec_ok():
            has_table = st.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_memories'"
            ).fetchone()
            if has_table:
                vec_before = st.conn.execute(
                    "SELECT id FROM vec_memories WHERE id=?", (r.memory_id,)
                ).fetchone()
                assert vec_before is not None

        result = st.purge(r.memory_id)
        assert result["ok"] is True

        if st.vec_ok():
            has_table = st.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_memories'"
            ).fetchone()
            if has_table:
                vec_after = st.conn.execute(
                    "SELECT id FROM vec_memories WHERE id=?", (r.memory_id,)
                ).fetchone()
                assert vec_after is None, "vec entry should be deleted after purge"


def test_purge_removes_graph_rows(haunt_env):
    """After purge, relations tied to the event should be removed."""
    text = "Alice updated src/haunt/store.py in function init_schema()"
    r = observe(text, namespace="default", role="user")

    with Store("default") as st:
        rels_before = st.conn.execute(
            "SELECT COUNT(*) FROM relations WHERE event_id=?", (r.event_id,)
        ).fetchone()[0]

        st.purge(r.memory_id)

        rels_after = st.conn.execute(
            "SELECT COUNT(*) FROM relations WHERE event_id=?", (r.event_id,)
        ).fetchone()[0]
        assert rels_after == 0, "graph relations should be deleted after purge"


def test_purge_removes_orphan_event(haunt_env):
    """If no other memories reference the event, the event itself is deleted."""
    r = observe("orphan event test CANARY-88", namespace="default", role="user")

    with Store("default") as st:
        event_before = st.conn.execute(
            "SELECT id FROM events WHERE id=?", (r.event_id,)
        ).fetchone()
        assert event_before is not None

        result = st.purge(r.memory_id)
        assert result["event_deleted"] is True

        event_after = st.conn.execute(
            "SELECT id FROM events WHERE id=?", (r.event_id,)
        ).fetchone()
        assert event_after is None, "orphan event should be deleted"


def test_purge_recall_does_not_return_deleted(haunt_env):
    """After purge, recall must not return the purged text."""
    unique = "PURGE-RECALL-CANARY-99 absolutely unique text for recall test"
    r = observe(unique, namespace="default", role="user")

    hits_before = recall(unique, namespace="default", k=8)
    assert any(unique in h.content for h in hits_before), "should find before purge"

    with Store("default") as st:
        st.purge(r.memory_id)

    hits_after = recall(unique, namespace="default", k=8)
    assert all(unique not in h.content for h in hits_after), (
        "purged text should not appear in recall"
    )


def test_purge_not_found(haunt_env):
    """Purging a nonexistent memory returns ok=False."""
    with Store("default") as st:
        result = st.purge("nonexistent-id-12345")
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_purge_overwrites_canary_bytes_on_disk(haunt_env):
    """The purged text must not survive anywhere in the raw namespace files.

    The churn between observe and purge is the point: index merges and page
    splits leave older copies of a row behind, and a purge that only unlinks
    rows leaves those copies legible.
    """
    canary = "securedeletecanaryq7x2"
    r = observe(f"leaked credential {canary} rotate it", namespace="default")
    for i in range(40):
        observe(f"filler note {i} about deployments and databases", namespace="default")

    needle = canary.encode()
    with Store("default") as st:
        db_path = st.db_path
        before = _canary_counts(db_path, needle)
        assert sum(before.values()) > 0, f"canary was never written: {before}"

        result = st.purge(r.memory_id)
        assert result["bytes_overwritten"] is True

        after = _canary_counts(db_path, needle)
    assert sum(after.values()) == 0, f"canary survives on disk: {after}"


def test_purge_restores_secure_delete(haunt_env):
    """secure_delete is scoped to the purge, not left on for ordinary writes."""
    r = observe("SECURE-DELETE-SCOPE-CANARY unique phrase", namespace="default")
    with Store("default") as st:
        before = st.conn.execute("PRAGMA secure_delete").fetchone()[0]
        st.purge(r.memory_id)
        assert st.conn.execute("PRAGMA secure_delete").fetchone()[0] == before


def test_purge_zeroes_its_own_bytes_when_the_rebuild_is_blocked(haunt_env):
    """A refused VACUUM still overwrites what this purge freed, and says so."""
    canary = "vacuumblockedcanaryk4m9"
    r = observe(f"leaked credential {canary} rotate it", namespace="default")
    needle = canary.encode()

    with Store("default") as st:
        db_path = st.db_path
        real_execute = st.conn.execute

        def refuse_vacuum(sql, *args):
            if sql.lstrip().upper().startswith("VACUUM"):
                raise sqlite3.OperationalError("database is locked")
            return real_execute(sql, *args)

        st.conn.execute = refuse_vacuum  # type: ignore[method-assign]
        try:
            result = st.purge(r.memory_id)
        finally:
            del st.conn.execute

        assert result["ok"] is True
        assert result["bytes_overwritten"] is False
        after = _canary_counts(db_path, needle)
    assert sum(after.values()) == 0, f"canary survives on disk: {after}"


def test_purge_rebuild_never_spills_plaintext_to_a_temp_directory(haunt_env, tmp_path):
    """The rebuild must not materialize the surviving corpus outside HAUNT_HOME.

    With SQLite's default temp_store the VACUUM writes the entire rebuilt
    database to a transient file under the temp directory -- a plaintext copy
    of every memory that survived the purge, outside the deliberately-0700
    HAUNT_HOME, unlinked without being zeroed. SQLite unlinks that file as
    soon as it opens it, so the evidence is the directory's own mtime: an
    entry appeared there and went. cache_size is lowered only to reach at
    test size the spill a real namespace reaches on its own.
    """
    canary = "tempspillcanaryb8n3"
    r = observe(f"leaked credential {canary} rotate it", namespace="default")
    for i in range(40):
        observe(f"filler note {i} about deployments and databases", namespace="default")

    spill_dir = tmp_path / "sqlite-temp"
    spill_dir.mkdir()
    with Store("default") as st:
        st.conn.execute("PRAGMA cache_size=-16")
        # Process-global in SQLite, so it is restored before leaving the test.
        st.conn.execute(f"PRAGMA temp_store_directory='{spill_dir}'")
        try:
            before = spill_dir.stat().st_mtime_ns
            result = st.purge(r.memory_id)
            after = spill_dir.stat().st_mtime_ns
        finally:
            st.conn.execute("PRAGMA temp_store_directory=''")

    assert result["bytes_overwritten"] is True
    assert list(spill_dir.iterdir()) == []
    assert after == before, "the rebuild created a file in the temp directory"


def test_purge_erases_the_predecessor_id_a_successor_names(haunt_env, tmp_path):
    """A successor's link to an erased session must not republish its id.

    `claude --resume` replays an ended session id, so the successor records
    which session it continues. Purging a memory erases that session; the
    link is a row the erased session does not own, and an unrekeyed one keeps
    the id both in the file and in every export bundle taken from it.
    """
    from haunt.portability import export_namespace_path

    erased_session = "purgedsessioncanaryv5"
    with Store("default") as st:
        st.ensure_session(erased_session)
        target = st.observe("secret to erase", session_id=erased_session)
        st.observe("unrelated work that survives", session_id=erased_session)
        st.end_session(erased_session)
        # Replaying the ended id is what mints the successor.
        st.observe("work after the resume", session_id=erased_session)

        db_path = st.db_path
        needle = erased_session.encode()
        assert sum(_canary_counts(db_path, needle).values()) > 0
        assert st.purge(target.memory_id)["session_deleted"] is True
        after = _canary_counts(db_path, needle)

    assert sum(after.values()) == 0, f"erased session id survives on disk: {after}"

    bundle = tmp_path / "bundle.haunt"
    export_namespace_path("default", bundle)
    assert erased_session not in bundle.read_text(errors="replace")


def test_purge_restores_a_fast_secure_delete_default(haunt_env):
    """FAST must round-trip: SQLite parses the integer 2 as boolean ON.

    On a SQLITE_SECURE_DELETE=FAST build the first purge would otherwise
    promote the connection to full secure_delete for the rest of its life.
    """
    r = observe("SECURE-DELETE-FAST-CANARY unique phrase", namespace="default")
    with Store("default") as st:
        st.conn.execute("PRAGMA secure_delete=FAST")
        assert st.conn.execute("PRAGMA secure_delete").fetchone()[0] == 2
        st.purge(r.memory_id)
        assert st.conn.execute("PRAGMA secure_delete").fetchone()[0] == 2


def test_purge_restores_secure_delete_when_the_transaction_fails(haunt_env):
    """A failed purge must not strand secure_delete on the writer connection."""
    r = observe("SECURE-DELETE-ROLLBACK-CANARY unique phrase", namespace="default")
    with Store("default") as st:
        before = st.conn.execute("PRAGMA secure_delete").fetchone()[0]
        real_execute = st.conn.execute

        def fail_the_delete(sql, *args):
            if sql.lstrip().upper().startswith("DELETE FROM MEMORIES"):
                raise sqlite3.OperationalError("forced mid-purge failure")
            return real_execute(sql, *args)

        st.conn.execute = fail_the_delete  # type: ignore[method-assign]
        try:
            with pytest.raises(sqlite3.OperationalError, match="forced"):
                st.purge(r.memory_id)
        finally:
            del st.conn.execute

        assert st.conn.execute("PRAGMA secure_delete").fetchone()[0] == before


def test_purge_keeps_the_replacement_session_in_its_succession_chain(haunt_env):
    """The stand-in session inherits the erased session's own predecessor.

    Erasing a memory written after a `claude --resume` replaces that session
    with an opaque one. Dropping its link to the session it continues would
    leave the replacement with no way to be closed with its predecessor.
    """
    predecessor = "chainpredecessorcanaryh2"
    with Store("default") as st:
        st.ensure_session(predecessor)
        st.observe("work before the end", session_id=predecessor)
        st.end_session(predecessor)
        # The replay mints the successor whose session is about to be erased.
        target = st.observe("secret written after the resume", session_id=predecessor)
        st.observe("unrelated work in the same successor", session_id=predecessor)

        assert st.purge(target.memory_id)["session_deleted"] is True
        ended = st.end_session(predecessor)

    assert ended["ok"] is True, "the replacement lost its place in the chain"
    assert len(ended["sessions_ended"]) == 1


def _backup_canary_counts(home: Path, needle: bytes) -> dict[str, int]:
    """Raw occurrences of needle across every file Haunt's backup dir holds."""
    root = home / "backups"
    if not root.is_dir():
        return {}
    return {p.name: p.read_bytes().count(needle) for p in sorted(root.iterdir())}


def test_purge_erases_the_canary_from_the_backups_haunt_wrote(haunt_env):
    """Both backup-creating paths must not outlive the erasure they predate.

    reconcile and retire each write a full plaintext copy of a namespace
    database under HAUNT_HOME/backups. Evidence is the raw bytes of those
    files, not the purge report: a copy nothing scans is exactly how the
    erasure guarantee was false while every in-database assertion passed.
    """
    from haunt.store import (
        reconcile_namespaces,
        register_namespace,
        retire_namespace,
    )

    canary = "backupleakcanaryf3w8"
    register_namespace("leak-src")
    with Store("leak-src") as st:
        memory_id = st.observe(f"leaked credential {canary} rotate it").memory_id
        for i in range(20):
            st.observe(f"filler note {i} about deployments and databases")
    register_namespace("leak-dst")
    with Store("leak-dst") as st:
        st.observe("an unrelated destination row")

    plan = reconcile_namespaces("leak-src", "leak-dst")
    reconcile_namespaces(
        "leak-src", "leak-dst", apply=True, plan_digest=plan["plan_digest"]
    )
    retired = retire_namespace("leak-src", into="leak-dst", apply=True)
    assert retired["retired"] is True

    needle = canary.encode()
    before = _backup_canary_counts(haunt_env, needle)
    assert sum(before.values()) > 0, f"no backup ever held the canary: {before}"

    with Store("leak-dst", create=False) as st:
        result = st.purge(memory_id)

    after = _backup_canary_counts(haunt_env, needle)
    assert sum(after.values()) == 0, f"canary survives in a backup: {after}"
    assert result["ok"] is True
    assert result["backups_unerased"] == []
    assert result["backups_erased"] >= 2, (
        "the reconcile-source and retire backups both held the row"
    )


def test_purge_leaves_backups_that_never_held_the_row_alone(haunt_env):
    """Sweeping is per-row, not a retention policy on the whole directory."""
    from haunt.store import reconcile_namespaces, register_namespace

    register_namespace("keep-src")
    with Store("keep-src") as st:
        kept = st.observe("a source row that survives the purge").memory_id
        doomed = st.observe("PURGED-BACKUP-SIBLING-TOKEN").memory_id
    register_namespace("keep-dst")
    with Store("keep-dst") as st:
        st.observe("an unrelated destination row")

    plan = reconcile_namespaces("keep-src", "keep-dst")
    reconcile_namespaces(
        "keep-src", "keep-dst", apply=True, plan_digest=plan["plan_digest"]
    )

    with Store("keep-dst", create=False) as st:
        result = st.purge(doomed)

    backups = sorted((haunt_env / "backups").glob("namespace-*.db"))
    assert backups, "the reconcile must have written backups to sweep"
    for backup in backups:
        check = sqlite3.connect(f"{backup.as_uri()}?mode=ro", uri=True)
        try:
            ids = {row[0] for row in check.execute("SELECT id FROM memories")}
        finally:
            check.close()
        assert doomed not in ids
        if kept in ids:
            break
    else:
        pytest.fail("the surviving row was swept out of every backup")
    assert result["backups_unerased"] == []
