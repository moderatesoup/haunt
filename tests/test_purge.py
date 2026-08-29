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


# Every content-bearing surface a purged memory can be reached through. The
# session id is one of them: callers choose it, so it carries private bytes
# exactly as often as content does.
BACKUP_CANARIES = {
    "content": "BACKUP-CONTENT-CANARY-a41f",
    "session": "BACKUP-SESSION-CANARY-b52a",
    # Only erased context is dropped from session metadata, in a backup as in
    # the live namespace, so this canary names the session it belongs to.
    "session_meta": "BACKUP-SESSION-CANARY-b52a/resumed-transcript",
    "tool_input": "BACKUP-TOOLIN-CANARY-d74c",
    "tool_output": "BACKUP-TOOLOUT-CANARY-e85d",
    "event_meta": "BACKUP-EVENTMETA-CANARY-f96e",
    "provenance": "BACKUP-PROVENANCE-CANARY-0a7f",
    "reason": "BACKUP-REASON-CANARY-1b80",
    "idempotency": "BACKUP-IDEMPOTENCY-CANARY-2c91",
}
SURVIVOR_TEXT = "an unrelated note sharing the erased session"
REPLACEMENT_TEXT = "the correction replacement that outlives the purge"


def _seed_backup_canaries(namespace: str) -> tuple[str, dict[str, int]]:
    """Plant every canary in one namespace, then return the target and its counts.

    Counts are read before any purge so a later assertion can name the exact
    surviving population rather than a plausible-looking one.
    """
    from haunt.store import Store

    with Store(namespace) as store:
        store.ensure_session(
            BACKUP_CANARIES["session"],
            meta={
                "erased": BACKUP_CANARIES["session_meta"],
                "unrelated": "kept-session-metadata",
            },
        )
        target = store.observe(
            f"leaked credential {BACKUP_CANARIES['content']} rotate it",
            session_id=BACKUP_CANARIES["session"],
            tool_name="Bash",
            tool_input=BACKUP_CANARIES["tool_input"],
            tool_output=BACKUP_CANARIES["tool_output"],
            meta={"trace": BACKUP_CANARIES["event_meta"]},
            provenance={
                "schema_version": 1,
                "kind": "import",
                "source_platform": "legacy-transcripts",
                "source_native_id": BACKUP_CANARIES["provenance"],
                "imported_at": "2026-01-01T00:00:00+00:00",
                "fidelity": "lossless",
                "original_blob_sha256": None,
            },
            defer_embedding=True,
        )
        # A second event in the same session: rekeying must move it, not drop it.
        store.observe(
            SURVIVOR_TEXT,
            session_id=BACKUP_CANARIES["session"],
            defer_embedding=True,
        )
        for i in range(5):
            store.observe(f"filler note {i} about deployments", defer_embedding=True)
        store.contradict(
            target.memory_id,
            replacement=REPLACEMENT_TEXT,
            idempotency_key=BACKUP_CANARIES["idempotency"],
            reason=BACKUP_CANARIES["reason"],
        )
        store.conn.commit()
        counts = _table_counts(store.conn)
    return target.memory_id, counts


def _table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = (
        "meta",
        "sessions",
        "events",
        "memories",
        "corrections",
        "lineage_tombstones",
        "entities",
        "relations",
        "relation_evidence",
        "entity_mentions",
    )
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in tables
    }


def _raw_hits(paths, needles) -> dict[str, dict[str, int]]:
    """Raw byte occurrences of each needle in each file that exists."""
    hits: dict[str, dict[str, int]] = {}
    for path in paths:
        if not Path(path).exists():
            continue
        data = Path(path).read_bytes()
        found = {
            name: data.count(value.encode())
            for name, value in needles.items()
            if data.count(value.encode())
        }
        if found:
            hits[Path(path).name] = found
    return hits


def _backup_files(home: Path) -> list[Path]:
    root = home / "backups"
    return sorted(root.iterdir()) if root.is_dir() else []


def _restore_backup(source: Path, namespace: str) -> Path:
    """Put a backup's bytes back as the live namespace database."""
    import shutil

    from haunt.paths import namespace_db_path

    destination = namespace_db_path(namespace)
    for suffix in ("-wal", "-shm"):
        Path(f"{destination}{suffix}").unlink(missing_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o600)
    return destination


def _swept_source_backup(home: Path, namespace: str) -> Path:
    matches = [
        path
        for path in _backup_files(home)
        if path.name.startswith(f"namespace-{namespace}-reconcile-source-")
    ]
    assert len(matches) == 1, f"expected one source backup, got {matches}"
    return matches[0]


def _reconcile_into(source: str, target: str) -> None:
    from haunt.store import reconcile_namespaces

    plan = reconcile_namespaces(source, target)
    reconcile_namespaces(source, target, apply=True, plan_digest=plan["plan_digest"])


def test_pre_purge_bundle_cannot_resurrect_a_memory_through_a_swept_backup(haunt_env):
    """Restoring a swept backup must not reopen the door a purge closed.

    Sweeping erases the row from the backup, but a bundle exported before the
    purge is an external artifact no erasure reaches. The live purge rotates
    the namespace's opaque privacy head so that bundle is refused; a backup
    that keeps its pre-purge head restores a database which accepts it, and
    the erased row walks back in.
    """
    from haunt.portability import (
        ImportConflictError,
        build_namespace_export,
        canonical_export_bytes,
        import_namespace_bytes,
    )
    from haunt.store import Store, open_existing, register_namespace

    register_namespace("resurrect-src")
    register_namespace("resurrect-dst")
    memory_id, _ = _seed_backup_canaries("resurrect-src")
    with Store("resurrect-dst") as store:
        store.observe("an unrelated destination row", defer_embedding=True)

    stale = canonical_export_bytes(build_namespace_export("resurrect-src"))
    assert BACKUP_CANARIES["content"].encode() in stale

    _reconcile_into("resurrect-src", "resurrect-dst")
    with open_existing("resurrect-dst") as store:
        assert store.purge(memory_id)["backups_unerased"] == []

    restored = _restore_backup(
        _swept_source_backup(haunt_env, "resurrect-src"), "resurrect-src"
    )
    # Only the content canary here; the full surface sweep is its own test, and
    # this one has to reach the import to say anything about resurrection.
    content_only = {"content": BACKUP_CANARIES["content"]}
    assert _raw_hits([restored], content_only) == {}
    with open_existing("resurrect-src") as store:
        assert store.get_memory(memory_id) is None

    with pytest.raises(ImportConflictError, match="privacy lineage"):
        import_namespace_bytes(stale)

    with open_existing("resurrect-src") as store:
        assert store.get_memory(memory_id) is None
    assert _raw_hits([restored], content_only) == {}


def test_purge_erases_every_reachable_canary_from_every_backup(haunt_env):
    """Raw bytes, not row counts: the leak this sweep exists for is unqueried.

    Session identifiers and metadata, tool payloads, structured provenance and
    correction request context are all reachable from the purged memory and
    all were left legible in a backup while the report said the sweep was
    complete.
    """
    from haunt.store import Store, open_existing, register_namespace

    register_namespace("canary-src")
    register_namespace("canary-dst")
    memory_id, _ = _seed_backup_canaries("canary-src")
    with Store("canary-dst") as store:
        store.observe("an unrelated destination row", defer_embedding=True)

    _reconcile_into("canary-src", "canary-dst")
    before = _raw_hits(_backup_files(haunt_env), BACKUP_CANARIES)
    assert set(before) and set(next(iter(before.values()))) == set(BACKUP_CANARIES), (
        f"the backup never held every canary to begin with: {before}"
    )

    with open_existing("canary-dst") as store:
        result = store.purge(memory_id)

    assert result["backups_unerased"] == []
    assert result["backups_erased"] >= 1
    assert _raw_hits(_backup_files(haunt_env), BACKUP_CANARIES) == {}


def test_swept_backup_erases_exactly_what_the_live_purge_erased(haunt_env):
    """The two erasures must agree row for row, not merely both look erased.

    Reconcile leaves the source untouched, so its backup and its live database
    hold the same rows going into the purge. Every table therefore has to come
    out at the same count on both sides, and the restored backup has to still
    be a usable database.
    """
    from haunt.store import Store, open_existing, register_namespace

    register_namespace("survive-src")
    register_namespace("survive-dst")
    memory_id, before = _seed_backup_canaries("survive-src")
    with Store("survive-dst") as store:
        store.observe("an unrelated destination row", defer_embedding=True)

    _reconcile_into("survive-src", "survive-dst")
    with open_existing("survive-src") as store:
        assert _table_counts(store.conn) == before, "reconcile moved the source"
        assert store.purge(memory_id)["backups_unerased"] == []
        live = _table_counts(store.conn)

    backup = _swept_source_backup(haunt_env, "survive-src")
    check = sqlite3.connect(f"{backup.as_uri()}?mode=ro", uri=True)
    check.row_factory = sqlite3.Row
    try:
        assert _table_counts(check) == live
    finally:
        check.close()
    assert live["memories"] == before["memories"] - 1
    assert live["lineage_tombstones"] == before["lineage_tombstones"] + 1

    _restore_backup(backup, "survive-src")
    with open_existing("survive-src") as store:
        conn = store.conn
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert store.get_memory(memory_id) is None
        surviving = {
            row["content"] for row in conn.execute("SELECT content FROM memories")
        }
        assert SURVIVOR_TEXT in surviving
        assert REPLACEMENT_TEXT in surviving
        assert {f"filler note {i} about deployments" for i in range(5)} <= surviving
        # The rekeyed session keeps the metadata that was never erased context.
        metas = [row["meta"] for row in conn.execute("SELECT meta FROM sessions")]
        assert any("kept-session-metadata" in (meta or "") for meta in metas)


@pytest.mark.parametrize("broken", ["privacy-lineage", "session-metadata"])
def test_backups_unerased_names_a_backup_whose_erasure_cannot_be_proven(
    haunt_env, monkeypatch, broken
):
    """The report is derived from checking the file, not asserted beside it.

    Each parameter breaks one half of the erasure a different check has to
    catch: a backup left on its pre-purge privacy head, and erased context
    copied onward into the replacement session's metadata.
    """
    from haunt import store as store_module
    from haunt.store import Store, open_existing, register_namespace

    real_rotate = store_module._rotate_privacy_lineage_head

    def unrotated(conn, namespace_id):
        if namespace_id is None:
            return "sha256:" + "0" * 64
        return real_rotate(conn, namespace_id)

    def unsanitized(conn, session_id, sensitive_values):
        row = conn.execute(
            "SELECT started_at, ended_at, source, meta FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        return row["started_at"], row["ended_at"], row["source"], row["meta"]

    register_namespace("honest-src")
    register_namespace("honest-dst")
    memory_id, _ = _seed_backup_canaries("honest-src")
    with Store("honest-dst") as store:
        store.observe("an unrelated destination row", defer_embedding=True)
    _reconcile_into("honest-src", "honest-dst")

    if broken == "privacy-lineage":
        monkeypatch.setattr(store_module, "_rotate_privacy_lineage_head", unrotated)
    else:
        monkeypatch.setattr(store_module, "_purge_safe_session_context", unsanitized)

    with open_existing("honest-dst") as store:
        result = store.purge(memory_id)

    assert result["ok"] is True
    assert result["backups_scanned"] >= 1
    assert result["backups_erased"] == 0
    assert result["backups_unerased"], "a backup left incompletely erased must be named"
