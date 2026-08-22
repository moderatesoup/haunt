"""Claude Code hook handler tests. Isolated temp dirs, FTS-only."""

from __future__ import annotations

import io
import json
import sys

import pytest

from haunt.store import Store


@pytest.fixture
def fts_cc_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME, FTS-only — never download BGE-M3."""
    home = tmp_path / "haunthome"
    project = tmp_path / "myproj"
    project.mkdir()
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "cctest")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))
    monkeypatch.setenv("CURSOR_HOME", str(tmp_path / "cursor-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    monkeypatch.delenv("LORE_HOME", raising=False)
    monkeypatch.delenv("LORE_NAMESPACE", raising=False)
    monkeypatch.delenv("ENGRAM_NAMESPACE", raising=False)
    monkeypatch.delenv("ENGRAM_HOME", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield {"home": home, "project": project}
    embed.reset()


def _run_hook(payload: dict, capsys, monkeypatch) -> dict:
    from haunt.claude_hook import main

    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    raw = capsys.readouterr().out
    return json.loads(raw)


def test_user_prompt_submit_observes_prompt(fts_cc_env, capsys, monkeypatch):
    unique = "Remember the ZX-CC-TOKEN lives only in this sentence."
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": unique,
        "session_id": "sess-cc-1",
        "cwd": str(fts_cc_env["project"]),
    }
    out = _run_hook(payload, capsys, monkeypatch)
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "UserPromptSubmit"
    assert "additionalContext" in hso
    assert "ZX-CC-TOKEN" in hso["additionalContext"]
    assert "[haunt ns=cctest]" in hso["additionalContext"]
    with Store("cctest") as st:
        rows = st.events(session_id="sess-cc-1")
        assert any(r["content"] == unique and r["role"] == "user" for r in rows)
        assert any(r["origin"] == "claude-code" for r in rows)
        mem = st.conn.execute("SELECT content, tier FROM memories").fetchall()
        assert any(unique in (m["content"] or "") and m["tier"] == "episodic" for m in mem)


def test_post_tool_use_skips_memory_tools(fts_cc_env, capsys, monkeypatch):
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "memory_recall",
        "tool_input": {"query": "x"},
        "tool_response": {"hits": []},
        "session_id": "sess-skip",
        "cwd": str(fts_cc_env["project"]),
    }
    out = _run_hook(payload, capsys, monkeypatch)
    assert out == {}
    with Store("cctest") as st:
        assert st.events() == []


def test_post_tool_use_stores_episodic_not_procedural(fts_cc_env, capsys, monkeypatch):
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Read",
        "tool_input": {"path": "src/haunt/store.py"},
        "tool_response": "def init_schema(conn): pass",
        "session_id": "sess-tool",
        "cwd": str(fts_cc_env["project"]),
    }
    _run_hook(payload, capsys, monkeypatch)
    with Store("cctest") as st:
        rows = st.events(session_id="sess-tool")
        assert rows
        row = rows[0]
        assert row["role"] == "tool"
        assert row["tier"] == "episodic"
        assert row["tool_name"] == "Read"
        assert row["origin"] == "claude-code"
        assert "init_schema" in (row["tool_output"] or "")


def test_stop_observes_last_assistant_message(fts_cc_env, capsys, monkeypatch):
    text = "I stored the CC-STOP-TOKEN in this assistant reply."
    payload = {
        "hook_event_name": "Stop",
        "last_assistant_message": text,
        "session_id": "sess-stop",
        "cwd": str(fts_cc_env["project"]),
    }
    out = _run_hook(payload, capsys, monkeypatch)
    assert out == {}
    with Store("cctest") as st:
        rows = st.events(session_id="sess-stop")
        assert any(r["content"] == text and r["role"] == "assistant" for r in rows)
        assert any(r["origin"] == "claude-code" for r in rows)


def test_session_start_worldview_shape(fts_cc_env, capsys, monkeypatch):
    with Store("cctest") as st:
        st.observe("The vault key is in .env", role="system", tier="semantic")
    payload = {
        "hook_event_name": "SessionStart",
        "session_id": "sess-start",
        "cwd": str(fts_cc_env["project"]),
    }
    out = _run_hook(payload, capsys, monkeypatch)
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "SessionStart"
    ctx = hso["additionalContext"]
    assert "[haunt worldview" in ctx
    assert "memory_recall" in ctx


def test_fail_open_garbage_stdin(fts_cc_env, capsys, monkeypatch):
    from haunt.claude_hook import main

    monkeypatch.setattr(sys, "stdin", io.StringIO("not-json{"))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out) == {}


def test_post_tool_failure_skips_memory_star(fts_cc_env, capsys, monkeypatch):
    payload = {
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "memory_observe",
        "tool_input": {"text": "nope"},
        "error": "tool failed",
        "session_id": "sess-fail-skip",
    }
    _run_hook(payload, capsys, monkeypatch)
    with Store("cctest") as st:
        assert st.events() == []
