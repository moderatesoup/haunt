"""C6: capture policy should skip embedding, not drop capture.

HAUNT_EXCLUDE_TOOLS (existing, unchanged) is a privacy opt-out: matching
tools are dropped before observe() is ever called -- no event, no memory
row, no FTS entry. That must keep working exactly as before.

HAUNT_EMBED_EXCLUDE_TOOLS (new) is a separate, narrower control: matching
tool rows are still captured in full (event + memory row + FTS entry) but
are never embedded and never enqueued into embedding_jobs. The record stays
complete and keyword-searchable; only vector-index capacity is saved.
Session-start ceremony and stored-thought rows get the same treatment
unconditionally.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def policy_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME, FTS-only, both hook hosts point at 'policy-test'."""
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "policy-test")
    monkeypatch.delenv("HAUNT_EXCLUDE_TOOLS", raising=False)
    monkeypatch.delenv("HAUNT_EMBED_EXCLUDE_TOOLS", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry, register_namespace

    embed.reset()
    ensure_layout()
    init_registry()
    register_namespace("policy-test")
    yield home
    embed.reset()


def _memory_row(store, memory_id):
    return store.conn.execute(
        "SELECT content, embedding FROM memories WHERE id=?", (memory_id,)
    ).fetchone()


def _fts_matches(store, token):
    """Query memories_fts the same safe way recall() does: FTS5's MATCH
    query syntax treats bare hyphens, colons, etc. specially, so a raw
    unescaped token (our tokens are hyphenated, e.g. "FOO-BAR-TOKEN") is not
    a literal string match -- it has to go through the same quoting recall()
    uses before it reaches SQLite.
    """
    from haunt.recall import _fts_match_query

    match = _fts_match_query(token)
    return store.conn.execute(
        "SELECT id FROM memories_fts WHERE memories_fts MATCH ?", (match,)
    ).fetchall()


def _queued(store, memory_id):
    return (
        store.conn.execute(
            "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (memory_id,)
        ).fetchone()
        is not None
    )


def _memory_id_for_event(store, session_id):
    row = store.conn.execute(
        """
        SELECT m.id FROM memories m JOIN events e ON e.id=m.event_id
        WHERE e.session_id=? ORDER BY m.rowid DESC LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    assert row is not None, f"no memory row for session {session_id}"
    return row["id"]


# ---------------------------------------------------------------------------
# HAUNT_EXCLUDE_TOOLS: unchanged privacy behavior (regression guard)
# ---------------------------------------------------------------------------


def test_haunt_exclude_tools_still_drops_completely(policy_env, monkeypatch):
    """The old privacy opt-out must still leave *nothing* behind: no event,
    no memory row, no FTS entry. C6 must not soften this into "captured but
    unembedded" -- that would start persisting exactly the content these
    users excluded."""
    from haunt.claude_hook import run as run_claude
    from haunt.cursor_hook import run as run_cursor
    from haunt.store import Store

    monkeypatch.setenv("HAUNT_EXCLUDE_TOOLS", "read, secret_*")

    run_cursor(
        json.dumps(
            {
                "hook_event_name": "postToolUse",
                "conversation_id": "excl-session",
                "tool_call_id": "excl-cursor",
                "tool_name": "Read",
                "tool_output": "MUST-NOT-EXIST-ANYWHERE-CURSOR",
            }
        )
    )
    run_claude(
        json.dumps(
            {
                "hook_event_name": "PostToolUse",
                "session_id": "excl-session",
                "tool_use_id": "excl-claude",
                "tool_name": "Secret_Scanner",
                "tool_response": "MUST-NOT-EXIST-ANYWHERE-CLAUDE",
            }
        )
    )

    with Store("policy-test", create=False) as store:
        assert store.events(session_id="excl-session") == []
        assert _fts_matches(store, "MUST-NOT-EXIST-ANYWHERE-CURSOR") == []
        assert _fts_matches(store, "MUST-NOT-EXIST-ANYWHERE-CLAUDE") == []
        total_memories = store.conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        assert total_memories == 0


# ---------------------------------------------------------------------------
# HAUNT_EMBED_EXCLUDE_TOOLS: capture in full, skip embedding only
# ---------------------------------------------------------------------------


def test_default_policy_captures_bash_and_read_but_never_embeds(policy_env):
    """Bash and Read are excluded from embedding by default, but the event,
    memory row, and FTS entry must all still be written verbatim."""
    from haunt.cursor_hook import run as run_cursor
    from haunt.store import Store

    cases = [
        ("Bash", "BASH-OUTPUT-POLICY-TOKEN-ONE"),
        ("Read", "READ-OUTPUT-POLICY-TOKEN-TWO"),
    ]
    for tool_name, token in cases:
        run_cursor(
            json.dumps(
                {
                    "hook_event_name": "postToolUse",
                    "conversation_id": f"policy-session-{tool_name}",
                    "tool_call_id": f"policy-call-{tool_name}",
                    "tool_name": tool_name,
                    "tool_output": token,
                }
            )
        )

    with Store("policy-test", create=False) as store:
        for tool_name, token in cases:
            rows = store.events(session_id=f"policy-session-{tool_name}")
            assert rows, f"expected a captured event for {tool_name}"
            assert rows[0]["role"] == "tool"
            assert rows[0]["tool_name"] == tool_name

            memory_id = _memory_id_for_event(store, f"policy-session-{tool_name}")
            mem = _memory_row(store, memory_id)
            assert token in mem["content"], "verbatim content must be captured"
            assert mem["embedding"] is None, "policy-excluded row must not embed"
            fts_rows = _fts_matches(store, token)
            assert any(row["id"] == memory_id for row in fts_rows), (
                "FTS entry for this exact row must still be written"
            )
            assert not _queued(store, memory_id), (
                "policy-excluded row must not be enqueued into embedding_jobs"
            )


def test_policy_excluded_row_is_findable_via_keyword_recall(policy_env):
    """The whole point of skipping embedding instead of dropping capture:
    keyword recall must still find the row."""
    from haunt.cursor_hook import run as run_cursor
    from haunt.recall import recall
    from haunt.store import Store

    token = "KEYWORD-RECALL-POLICY-EXCLUDED-CANARY"
    run_cursor(
        json.dumps(
            {
                "hook_event_name": "postToolUse",
                "conversation_id": "recall-policy-session",
                "tool_call_id": "recall-policy-call",
                "tool_name": "Bash",
                "tool_output": token,
            }
        )
    )

    with Store("policy-test", create=False) as store:
        memory_id = _memory_id_for_event(store, "recall-policy-session")
        assert _memory_row(store, memory_id)["embedding"] is None

        hits = recall(
            token,
            namespace="policy-test",
            store=store,
            use_vectors=False,
            include_untrusted=True,
        )
        assert any(h.memory_id == memory_id for h in hits), (
            "policy-excluded row must still be recallable via FTS keyword search"
        )


def test_edit_write_task_continue_to_embed_normally(policy_env):
    """Tools not on the embed-exclusion list must still be queued for
    embedding exactly as before -- the policy is per-tool, not per-category."""
    from haunt.cursor_hook import run as run_cursor
    from haunt.store import Store

    for tool_name in ("Edit", "Write", "Task"):
        run_cursor(
            json.dumps(
                {
                    "hook_event_name": "postToolUse",
                    "conversation_id": f"normal-embed-{tool_name}",
                    "tool_call_id": f"normal-embed-call-{tool_name}",
                    "tool_name": tool_name,
                    "tool_output": f"{tool_name}-NORMAL-EMBED-TOKEN",
                }
            )
        )

    with Store("policy-test", create=False) as store:
        for tool_name in ("Edit", "Write", "Task"):
            memory_id = _memory_id_for_event(store, f"normal-embed-{tool_name}")
            assert _queued(store, memory_id), (
                f"{tool_name} must still be queued for embedding by default"
            )


def test_session_start_ceremony_not_embedded_either_host(policy_env):
    """The fixed 'haunt session start' row (both hosts) must be captured
    (worldview still returned) but never embedded or enqueued."""
    from haunt.claude_hook import run as run_claude
    from haunt.cursor_hook import run as run_cursor
    from haunt.store import Store

    cursor_out = run_cursor(
        json.dumps({"hook_event_name": "sessionStart", "conversation_id": "cur-start"})
    )
    claude_out = run_claude(
        json.dumps({"hook_event_name": "SessionStart", "session_id": "cc-start"})
    )
    assert "additional_context" in cursor_out
    assert "hookSpecificOutput" in claude_out

    with Store("policy-test", create=False) as store:
        for session_id in ("cur-start", "cc-start"):
            memory_id = _memory_id_for_event(store, session_id)
            mem = _memory_row(store, memory_id)
            assert mem["content"] == "haunt session start"
            assert mem["embedding"] is None
            assert not _queued(store, memory_id), (
                "session-start ceremony row must never be enqueued for embedding"
            )
            # Still fully captured and keyword-searchable.
            assert _fts_matches(store, "haunt")


def test_shell_output_follows_the_same_embed_policy_under_both_hosts(policy_env):
    """Cursor's afterShellExecution rows are tagged "Shell" and Claude Code's
    are tagged "Bash", but both are the same raw shell output. A default that
    named only Bash embedded one host's copy and skipped the other's."""
    from haunt.claude_hook import run as run_claude
    from haunt.cursor_hook import run as run_cursor
    from haunt.store import Store

    run_cursor(
        json.dumps(
            {
                "hook_event_name": "afterShellExecution",
                "conversation_id": "shell-cursor",
                "command": "ls -la",
                "output": "SHELL-POLICY-TOKEN-CURSOR",
            }
        )
    )
    run_claude(
        json.dumps(
            {
                "hook_event_name": "PostToolUse",
                "session_id": "shell-claude",
                "tool_use_id": "shell-claude-call",
                "tool_name": "Bash",
                "tool_response": "SHELL-POLICY-TOKEN-CLAUDE",
            }
        )
    )

    with Store("policy-test", create=False) as store:
        for session_id, token in (
            ("shell-cursor", "SHELL-POLICY-TOKEN-CURSOR"),
            ("shell-claude", "SHELL-POLICY-TOKEN-CLAUDE"),
        ):
            memory_id = _memory_id_for_event(store, session_id)
            mem = _memory_row(store, memory_id)
            assert token in mem["content"], "verbatim content must be captured"
            assert mem["embedding"] is None
            assert not _queued(store, memory_id), (
                f"{session_id} shell output must not be enqueued for embedding"
            )
            assert _fts_matches(store, token)


def test_stored_thoughts_are_captured_but_not_embedded(policy_env, monkeypatch):
    """HAUNT_STORE_THOUGHTS rows are the same role=system/tier=coordinate
    shape as the session-start ceremony row, and get the same treatment:
    captured in full and keyword-searchable, never vector-indexed."""
    from haunt.cursor_hook import run as run_cursor
    from haunt.store import Store

    monkeypatch.setenv("HAUNT_STORE_THOUGHTS", "1")
    run_cursor(
        json.dumps(
            {
                "hook_event_name": "afterAgentThought",
                "conversation_id": "thought-session",
                "text": "THOUGHT-POLICY-TOKEN considering the next step",
            }
        )
    )

    with Store("policy-test", create=False) as store:
        memory_id = _memory_id_for_event(store, "thought-session")
        mem = _memory_row(store, memory_id)
        assert "THOUGHT-POLICY-TOKEN" in mem["content"]
        assert mem["embedding"] is None
        assert not _queued(store, memory_id), (
            "stored thoughts must not be enqueued for embedding"
        )
        assert _fts_matches(store, "THOUGHT-POLICY-TOKEN")


# ---------------------------------------------------------------------------
# HAUNT_EMBED_EXCLUDE_TOOLS override behavior
# ---------------------------------------------------------------------------


def test_embed_exclude_tools_can_be_emptied_to_embed_everything(
    policy_env, monkeypatch
):
    """Setting the var to "" explicitly must embed even the default-excluded
    tools -- an unset var means "use the default", not the same as empty."""
    from haunt.cursor_hook import run as run_cursor
    from haunt.store import Store

    monkeypatch.setenv("HAUNT_EMBED_EXCLUDE_TOOLS", "")
    run_cursor(
        json.dumps(
            {
                "hook_event_name": "postToolUse",
                "conversation_id": "override-empty-session",
                "tool_call_id": "override-empty-call",
                "tool_name": "Bash",
                "tool_output": "OVERRIDE-EMPTY-EMBEDS-EVERYTHING",
            }
        )
    )
    with Store("policy-test", create=False) as store:
        memory_id = _memory_id_for_event(store, "override-empty-session")
        assert _queued(store, memory_id), (
            "HAUNT_EMBED_EXCLUDE_TOOLS='' must embed even Bash"
        )


def test_embed_exclude_tools_custom_list_replaces_default(policy_env, monkeypatch):
    """A custom value replaces the default list rather than adding to it:
    Bash goes back to embedding normally, and the custom tool is excluded."""
    from haunt.cursor_hook import run as run_cursor
    from haunt.store import Store

    monkeypatch.setenv("HAUNT_EMBED_EXCLUDE_TOOLS", "Fetch")
    run_cursor(
        json.dumps(
            {
                "hook_event_name": "postToolUse",
                "conversation_id": "override-custom-bash",
                "tool_call_id": "override-custom-bash-call",
                "tool_name": "Bash",
                "tool_output": "CUSTOM-LIST-BASH-NOW-EMBEDS",
            }
        )
    )
    run_cursor(
        json.dumps(
            {
                "hook_event_name": "postToolUse",
                "conversation_id": "override-custom-fetch",
                "tool_call_id": "override-custom-fetch-call",
                "tool_name": "Fetch",
                "tool_output": "CUSTOM-LIST-FETCH-NOW-EXCLUDED",
            }
        )
    )
    with Store("policy-test", create=False) as store:
        bash_id = _memory_id_for_event(store, "override-custom-bash")
        fetch_id = _memory_id_for_event(store, "override-custom-fetch")
        assert _queued(store, bash_id), "Bash is not in the custom list, must embed"
        assert not _queued(store, fetch_id), "Fetch is in the custom list, must not"


# ---------------------------------------------------------------------------
# Store.observe() level: where the policy decision lives
# ---------------------------------------------------------------------------


def test_skip_embedding_flag_captures_but_never_queues(policy_env):
    """Direct Store.observe() proof of the mechanism the hooks rely on."""
    from haunt.store import Store

    with Store("policy-test") as store:
        r = store.observe(
            "",
            role="tool",
            tier="episodic",
            tool_name="Bash",
            tool_output="DIRECT-SKIP-EMBEDDING-TOKEN",
            defer_embedding=True,
            skip_embedding=True,
        )
        assert r.embedded is False
        assert r.embedding_queued is False
        assert not _queued(store, r.memory_id)
        fts_rows = _fts_matches(store, "DIRECT-SKIP-EMBEDDING-TOKEN")
        assert any(row["id"] == r.memory_id for row in fts_rows)


def test_mcp_style_observe_call_is_not_silently_unembedded(policy_env, monkeypatch):
    """Guards the placement decision itself: memory_observe (mcp_server.py)
    calls Store.observe() without ever passing skip_embedding, even when the
    caller sets tool_name="Bash" or "Read" by hand. Default policy tool
    names must have zero effect unless a caller explicitly opts in -- so a
    user who manually records something through MCP is never surprised by
    it silently skipping embedding just because they mentioned Bash.
    """
    from haunt.store import Store

    # Same default policy a hook would apply (Bash/Read excluded) is active
    # here, via the untouched env var -- and still must not matter, because
    # this call site (mirroring mcp_server.memory_observe) never reads it.
    monkeypatch.delenv("HAUNT_EMBED_EXCLUDE_TOOLS", raising=False)

    with Store("policy-test") as store:
        r = store.observe(
            "manually recording some Bash output on purpose",
            role="user",
            tier="episodic",
            tool_name="Bash",
            defer_embedding=True,
        )
        assert r.embedding_queued is True, (
            "a direct observe() call (as MCP's memory_observe makes) must "
            "embed normally regardless of tool_name, since it never passes "
            "skip_embedding itself"
        )
        assert _queued(store, r.memory_id)
