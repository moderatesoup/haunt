"""end_session / memory_session_end must not report ok for a session that was not ended.

The old path ran UPDATE ... WHERE id=? AND ended_at IS NULL, ignored rowcount,
and memory_session_end always returned ok:true with the requested session_id.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def fts_env(tmp_path, monkeypatch):
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield home
    embed.reset()


def _claimed_ok(result: object) -> bool:
    """Whether the caller was told the session ended.

    The old always-ok path returned the requested session id (a truthy str)
    even when no row was updated. That is a claimed success.
    """
    if isinstance(result, dict):
        return result.get("ok") is True
    return bool(result)


def test_end_session_nonexistent_is_not_ok(fts_env):
    """Published #22 falsifier at the store: missing id must not look ended."""
    from haunt.store import Store

    missing = "nonexistent-session-xyz"
    with Store("default") as st:
        result = st.end_session(missing)
        rows = st.conn.execute(
            "SELECT id FROM sessions WHERE id=?", (missing,)
        ).fetchall()

    assert not _claimed_ok(result), (
        "end_session must not succeed for a missing session "
        f"(result={result!r})"
    )
    assert isinstance(result, dict), (
        "end_session must return an honest report dict, not the requested id "
        f"(result={result!r})"
    )
    assert result.get("ok") is False
    assert result.get("error")
    assert missing in str(result.get("error"))
    assert result.get("session_id") == missing
    assert rows == []


def test_end_session_already_ended_is_not_ok(fts_env):
    from haunt.store import Store

    with Store("default") as st:
        sid = st.ensure_session("already-ended-sess")
        first = st.end_session(sid)
        assert _claimed_ok(first), f"first close of an open session must succeed: {first!r}"
        again = st.end_session(sid)
        row = st.conn.execute(
            "SELECT ended_at FROM sessions WHERE id=?", (sid,)
        ).fetchone()

    assert not _claimed_ok(again), (
        "end_session must not succeed for an already-ended session "
        f"(result={again!r})"
    )
    assert isinstance(again, dict)
    assert again.get("ok") is False
    assert again.get("error")
    assert "already ended" in str(again.get("error")).lower()
    assert row is not None and row["ended_at"]


def test_end_session_no_current_session_is_not_ok(fts_env):
    from haunt.store import Store

    with Store("default") as st:
        assert st.get_meta("current_session") is None
        result = st.end_session()

    assert not _claimed_ok(result), (
        "end_session with no current session must not look like success "
        f"(result={result!r})"
    )
    assert isinstance(result, dict)
    assert result.get("ok") is False
    assert result.get("error")


def test_end_session_open_session_succeeds(fts_env):
    from haunt.store import Store

    with Store("default") as st:
        sid = st.ensure_session("open-sess")
        result = st.end_session(sid)
        row = st.conn.execute(
            "SELECT ended_at FROM sessions WHERE id=?", (sid,)
        ).fetchone()
        current = st.get_meta("current_session")

    assert _claimed_ok(result), f"open session must end successfully: {result!r}"
    assert isinstance(result, dict)
    assert result["ok"] is True
    assert result["session_id"] == sid
    assert row is not None and row["ended_at"]
    assert current != sid


def test_memory_session_end_nonexistent_is_not_ok(fts_env):
    """Published #22 falsifier: MCP must not return ok:true for a missing session."""
    from haunt.mcp_server import memory_session_end
    from haunt.store import Store

    with Store("default") as st:
        st.ensure_session("placeholder-so-default-exists")

    missing = "nonexistent-session-xyz"
    data = json.loads(
        memory_session_end(session=missing, namespace="default")
    )

    assert data["ok"] is False, (
        "memory_session_end must not return ok:true for a session that "
        f"does not exist (payload={data!r})"
    )
    assert data.get("error"), f"failure must include error (payload={data!r})"
    assert data["namespace"] == "default"
    assert data["session_id"] == missing
    assert data["distilled"] is False

    from haunt.store import Store

    with Store("default") as st:
        rows = st.conn.execute(
            "SELECT id FROM sessions WHERE id=?", (missing,)
        ).fetchall()
    assert rows == []


def test_memory_session_end_already_ended_is_not_ok(fts_env):
    from haunt.mcp_server import memory_session_end
    from haunt.store import Store

    sid = "ended-once"
    with Store("default") as st:
        st.ensure_session(sid)
        assert _claimed_ok(st.end_session(sid))

    data = json.loads(memory_session_end(session=sid, namespace="default"))
    assert data["ok"] is False
    assert data.get("error")
    assert data["session_id"] == sid
    assert data["distilled"] is False


def test_memory_session_end_no_session_is_not_ok(fts_env):
    from haunt.mcp_server import memory_session_end
    from haunt.store import Store

    with Store("default") as st:
        st.set_meta("bootstrapped", "1")

    data = json.loads(memory_session_end(namespace="default"))
    assert data["ok"] is False, (
        "memory_session_end with no session and no current session "
        f"must not return ok:true (payload={data!r})"
    )
    assert data.get("error")
    assert data["distilled"] is False


def test_memory_session_end_open_session_succeeds(fts_env):
    from haunt.mcp_server import memory_session_end
    from haunt.store import Store

    sid = "real-open-sess"
    with Store("default") as st:
        st.ensure_session(sid)

    data = json.loads(memory_session_end(session=sid, namespace="default"))
    assert data["ok"] is True, f"open session must end: {data!r}"
    assert data["namespace"] == "default"
    assert data["session_id"] == sid
    assert data["distilled"] is False
    assert "error" not in data

    with Store("default") as st:
        row = st.conn.execute(
            "SELECT ended_at FROM sessions WHERE id=?", (sid,)
        ).fetchone()
    assert row is not None and row["ended_at"]


def test_ended_session_never_takes_another_event(fts_env):
    """A caller replaying an ended id (claude --resume, a late hook, an
    explicit --session) must not have its write attributed to the closed
    session, and must not silently clear what end_session recorded."""
    from haunt.store import Store

    with Store("default") as st:
        sid = st.ensure_session("resumed-sess")
        first = st.observe("before the end", session_id=sid)
        ended_at = st.end_session(sid)["ok"] and st.conn.execute(
            "SELECT ended_at FROM sessions WHERE id=?", (sid,)
        ).fetchone()["ended_at"]

        after = st.observe("after the end", session_id=sid)
        row = st.conn.execute(
            "SELECT ended_at FROM sessions WHERE id=?", (sid,)
        ).fetchone()
        attributed = st.conn.execute(
            "SELECT session_id FROM events WHERE id=?", (after.event_id,)
        ).fetchone()["session_id"]
        successor = st.conn.execute(
            "SELECT ended_at, meta FROM sessions WHERE id=?", (after.session_id,)
        ).fetchone()

    assert first.session_id == sid
    assert after.session_id != sid, "an ended session must not take another event"
    assert attributed == after.session_id
    assert row["ended_at"] == ended_at, "end_session's record must stand"
    assert successor["ended_at"] is None
    assert json.loads(successor["meta"])["succeeds_session"] == sid


def test_replayed_ended_session_reuses_one_successor(fts_env):
    """Every write after the end shares one successor, not a row per event."""
    from haunt.store import Store

    with Store("default") as st:
        sid = st.ensure_session("replayed-sess")
        st.observe("before", session_id=sid)
        st.end_session(sid)
        successors = {
            st.observe(f"after {i}", session_id=sid).session_id for i in range(3)
        }
        total = st.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    assert len(successors) == 1
    assert total == 2


def test_end_session_closes_the_successor_too(fts_env):
    """Otherwise ending stays advisory: the next replay refills the successor."""
    from haunt.store import Store

    with Store("default") as st:
        sid = st.ensure_session("advisory-sess")
        st.end_session(sid)
        successor = st.observe("after the end", session_id=sid).session_id

        result = st.end_session(sid)
        open_rows = st.conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL"
        ).fetchone()[0]
        again = st.observe("after the second end", session_id=sid).session_id

    assert result["ok"] is True
    assert result["sessions_ended"] == [successor]
    assert open_rows == 0
    assert again not in (sid, successor)
