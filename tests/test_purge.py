"""Tests for Store.purge — hard-delete walks the full provenance chain."""

from __future__ import annotations

from haunt.recall import recall
from haunt.store import Store, observe


def test_purge_removes_memory_and_keeps_other(lore_env):
    """Observe two memories, purge one, assert the other remains intact."""
    r1 = observe("alpha memory XRAY-11 unique text", namespace="default", role="user")
    r2 = observe("beta memory ZULU-22 different text", namespace="default", role="assistant")

    with Store("default") as st:
        result = st.purge(r1.memory_id)

    assert result["ok"] is True
    assert result["memory_id"] == r1.memory_id

    with Store("default") as st:
        gone = st.conn.execute(
            "SELECT id FROM memories WHERE id=?", (r1.memory_id,)
        ).fetchone()
        assert gone is None, "purged memory should not exist"

        kept = st.conn.execute(
            "SELECT id FROM memories WHERE id=?", (r2.memory_id,)
        ).fetchone()
        assert kept is not None, "other memory should still exist"


def test_purge_removes_fts_entry(lore_env):
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


def test_purge_removes_vec_entry(lore_env):
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


def test_purge_removes_graph_rows(lore_env):
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


def test_purge_removes_orphan_event(lore_env):
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


def test_purge_recall_does_not_return_deleted(lore_env):
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


def test_purge_not_found(lore_env):
    """Purging a nonexistent memory returns ok=False."""
    with Store("default") as st:
        result = st.purge("nonexistent-id-12345")
    assert result["ok"] is False
    assert "not found" in result["error"]
