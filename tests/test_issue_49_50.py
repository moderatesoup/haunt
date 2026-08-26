"""#49 graph provenance and #50 atomic/idempotent observe."""

from __future__ import annotations

import json
import sqlite3

import pytest


@pytest.fixture
def graph_env(tmp_path, monkeypatch):
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "graph-atomic")
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry, register_namespace

    embed.reset()
    ensure_layout()
    init_registry()
    register_namespace("graph-atomic")
    yield home
    embed.reset()


def _counts(store):
    return {
        table: store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "sessions",
            "events",
            "memories",
            "memories_fts",
            "entities",
            "entity_mentions",
            "relations",
            "relation_evidence",
        )
    }


def test_purge_latest_shared_relation_keeps_older_evidence(graph_env):
    from haunt.store import Store

    text = "Alice updated src/haunt/store.py in function rebuild_graph()"
    with Store("graph-atomic") as store:
        older = store.observe(
            text,
            role="user",
            event_time="2026-08-20T10:00:00+00:00",
        )
        newer = store.observe(
            text,
            role="assistant",
            event_time="2026-08-21T10:00:00+00:00",
        )
        triple = store.conn.execute(
            """
            SELECT src_entity, rel, dst_entity, COUNT(*) AS n
            FROM relation_evidence
            WHERE event_id IN (?, ?)
            GROUP BY src_entity, rel, dst_entity
            HAVING COUNT(*)=2
            LIMIT 1
            """,
            (older.event_id, newer.event_id),
        ).fetchone()
        assert triple is not None

        before = store.conn.execute(
            """
            SELECT weight, event_id FROM relations
            WHERE src_entity=? AND rel=? AND dst_entity=?
            """,
            (triple["src_entity"], triple["rel"], triple["dst_entity"]),
        ).fetchone()
        assert before["weight"] == pytest.approx(2.0)
        assert before["event_id"] == newer.event_id

        result = store.purge(newer.memory_id)
        assert result["ok"] is True
        assert result["event_deleted"] is True

        after = store.conn.execute(
            """
            SELECT weight, event_id FROM relations
            WHERE src_entity=? AND rel=? AND dst_entity=?
            """,
            (triple["src_entity"], triple["rel"], triple["dst_entity"]),
        ).fetchone()
        assert after is not None
        assert after["weight"] == pytest.approx(1.0)
        assert after["event_id"] == older.event_id
        assert store.conn.execute(
            "SELECT COUNT(*) FROM relation_evidence WHERE event_id=?",
            (newer.event_id,),
        ).fetchone()[0] == 0


def test_purge_does_not_delete_unrelated_standalone_entity(graph_env):
    from haunt.store import Store

    with Store("graph-atomic") as store:
        standalone = store.observe("Alice", role="user")
        target = store.observe("Bob changed src/haunt/store.py", role="assistant")
        alice = store.conn.execute(
            "SELECT id FROM entities WHERE norm_name='alice'"
        ).fetchone()
        assert alice is not None
        assert store.conn.execute(
            "SELECT COUNT(*) FROM entity_mentions WHERE event_id=? AND entity_id=?",
            (standalone.event_id, alice["id"]),
        ).fetchone()[0] == 1

        store.purge(target.memory_id)

        assert store.conn.execute(
            "SELECT id FROM entities WHERE id=?", (alice["id"],)
        ).fetchone() is not None
        assert store.conn.execute(
            "SELECT COUNT(*) FROM entity_mentions WHERE entity_id=?",
            (alice["id"],),
        ).fetchone()[0] == 1


def test_entity_span_is_min_and_max_not_arrival_order(graph_env):
    from haunt.store import Store

    with Store("graph-atomic") as store:
        store.observe("Alice", event_time="2026-08-22T12:00:00+00:00")
        store.observe("Alice", event_time="2025-01-02T03:04:05+00:00")
        row = store.conn.execute(
            "SELECT first_seen, last_seen FROM entities WHERE norm_name='alice'"
        ).fetchone()
        assert row["first_seen"].startswith("2025-01-02T03:04:05")
        assert row["last_seen"].startswith("2026-08-22T12:00:00")


def test_evidence_rebuild_recovers_all_events(graph_env):
    from haunt.store import Store

    with Store("graph-atomic") as store:
        first = store.observe("Alice changed src/haunt/store.py")
        second = store.observe("Alice changed src/haunt/store.py")
        expected_events = {first.event_id, second.event_id}
        store.conn.execute("DELETE FROM relation_evidence")
        store.conn.execute("DELETE FROM entity_mentions")
        store.conn.execute("DELETE FROM relations")
        store.conn.execute("DELETE FROM entities")
        store.conn.execute("DELETE FROM meta WHERE key='graph_evidence_version'")
        store.conn.commit()

    with Store("graph-atomic", create=False) as migrated:
        event_ids = {
            row["event_id"]
            for row in migrated.conn.execute(
                "SELECT DISTINCT event_id FROM entity_mentions"
            ).fetchall()
        }
        assert expected_events <= event_ids
        assert migrated.get_meta("graph_evidence_version") == "1"


def test_observe_rolls_back_everything_when_graph_fails(graph_env, monkeypatch):
    import haunt.graph
    from haunt.store import Store

    with Store("graph-atomic") as store:
        before = _counts(store)

        def fail_graph(*args, **kwargs):
            raise RuntimeError("graph exploded")

        monkeypatch.setattr(haunt.graph, "extract_and_store", fail_graph)
        with pytest.raises(RuntimeError, match="graph exploded"):
            store.observe(
                "ATOMIC-OBSERVE-CANARY",
                session_id="atomic-session",
                idempotency_key="atomic-event-1",
            )

        assert _counts(store) == before
        assert store.conn.execute(
            "SELECT id FROM sessions WHERE id='atomic-session'"
        ).fetchone() is None


def test_store_idempotency_returns_original_and_rejects_reuse(graph_env):
    from haunt.store import Store

    with Store("graph-atomic") as store:
        first = store.observe(
            "IDEMPOTENT-CANARY",
            session_id="idem-session",
            idempotency_key="host-event-123",
        )
        retry = store.observe(
            "IDEMPOTENT-CANARY",
            session_id="idem-session",
            idempotency_key="host-event-123",
        )
        assert retry.deduplicated is True
        assert retry.event_id == first.event_id
        assert retry.memory_id == first.memory_id
        assert store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE idempotency_key='host-event-123'"
        ).fetchone()[0] == 1
        with pytest.raises(ValueError, match="different content"):
            store.observe(
                "DIFFERENT",
                session_id="idem-session",
                idempotency_key="host-event-123",
            )


def test_cursor_hook_retry_with_generation_id_is_idempotent(graph_env):
    from haunt.cursor_hook import run
    from haunt.store import Store

    payload = {
        "hook_event_name": "beforeSubmitPrompt",
        "conversation_id": "cursor-session",
        "generation_id": "generation-abc",
        "prompt": "HOOK-IDEMPOTENCY-CANARY",
    }
    assert run(json.dumps(payload))["continue"] is True
    assert run(json.dumps(payload))["continue"] is True

    with Store("graph-atomic", create=False) as store:
        rows = store.conn.execute(
            "SELECT id, idempotency_key FROM events WHERE content=?",
            ("HOOK-IDEMPOTENCY-CANARY",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["idempotency_key"].startswith("hook:")


def test_same_hook_text_with_distinct_generation_ids_is_not_deduped(graph_env):
    from haunt.cursor_hook import run
    from haunt.store import Store

    base = {
        "hook_event_name": "beforeSubmitPrompt",
        "conversation_id": "cursor-session",
        "prompt": "REPEATED-BUT-LEGITIMATE",
    }
    run(json.dumps({**base, "generation_id": "generation-1"}))
    run(json.dumps({**base, "generation_id": "generation-2"}))

    with Store("graph-atomic", create=False) as store:
        assert store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE content=?",
            ("REPEATED-BUT-LEGITIMATE",),
        ).fetchone()[0] == 2


def test_claude_hook_retry_with_generation_id_is_idempotent(graph_env):
    from haunt.claude_hook import run
    from haunt.store import Store

    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "claude-session",
        "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
        "prompt": "CLAUDE-HOOK-IDEMPOTENCY-CANARY",
    }
    assert "hookSpecificOutput" in run(json.dumps(payload))
    assert "hookSpecificOutput" in run(json.dumps(payload))

    with Store("graph-atomic", create=False) as store:
        rows = store.conn.execute(
            "SELECT id, idempotency_key FROM events WHERE content=?",
            ("CLAUDE-HOOK-IDEMPOTENCY-CANARY",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["idempotency_key"].startswith("hook:")


def test_claude_tool_retry_uses_official_tool_use_id(graph_env):
    from haunt.claude_hook import run
    from haunt.store import Store

    payload = {
        "hook_event_name": "PostToolUse",
        "session_id": "claude-session",
        "prompt_id": "prompt-with-two-tools",
        "tool_use_id": "toolu_01ABC123",
        "tool_name": "Read",
        "tool_input": {"file_path": "src/haunt/store.py"},
        "tool_response": "TOOL-RETRY-CANARY",
    }
    assert run(json.dumps(payload)) == {}
    assert run(json.dumps(payload)) == {}

    with Store("graph-atomic", create=False) as store:
        assert store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE tool_output=?",
            ("TOOL-RETRY-CANARY",),
        ).fetchone()[0] == 1


def test_v1_database_migrates_idempotency_and_graph_evidence(graph_env):
    from haunt.paths import namespace_db_path
    from haunt.store import SCHEMA_VERSION, Store, register_namespace

    register_namespace("legacy-v1")
    path = namespace_db_path("legacy-v1")
    original_identity = (path.stat().st_dev, path.stat().st_ino)
    legacy = sqlite3.connect(":memory:")
    legacy.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key, value) VALUES ('schema_version', '1');
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
            meta TEXT
        );
        """
    )
    conn = sqlite3.connect(path)
    legacy.backup(conn)
    conn.commit()
    conn.close()
    legacy.close()
    assert (path.stat().st_dev, path.stat().st_ino) == original_identity

    with Store("legacy-v1", create=False) as store:
        columns = {
            row["name"]
            for row in store.conn.execute("PRAGMA table_info(events)").fetchall()
        }
        assert "idempotency_key" in columns
        assert store.get_meta("schema_version") == str(SCHEMA_VERSION)
        assert store.get_meta("graph_evidence_version") == "1"
        first = store.observe(
            "MIGRATED-IDEMPOTENCY-CANARY",
            idempotency_key="legacy-host-event",
        )
        retry = store.observe(
            "MIGRATED-IDEMPOTENCY-CANARY",
            idempotency_key="legacy-host-event",
        )
        assert retry.deduplicated is True
        assert retry.event_id == first.event_id
