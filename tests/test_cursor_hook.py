from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from haunt.recall import recall
from haunt.store import Store


@pytest.fixture
def fts_hook_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME, FTS-only — never download BGE-M3."""
    home = tmp_path / "haunthome"
    project = tmp_path / "myproj"
    project.mkdir()
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "hooktest")
    monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield {"home": home, "project": project}
    embed.reset()


def _run_hook(payload: dict, capsys, monkeypatch) -> dict:
    from haunt.cursor_hook import main

    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    raw = capsys.readouterr().out
    return json.loads(raw)


def test_before_submit_prompt_stores_verbatim_and_recalls(fts_hook_env, capsys, monkeypatch):
    project = fts_hook_env["project"]
    unique = "Remember the ZX-HOOK-TOKEN lives only in this sentence."
    payload = {
        "hook_event_name": "beforeSubmitPrompt",
        "prompt": unique,
        "conversation_id": "conv-hook-1",
        "workspace_roots": [str(project)],
        "cwd": str(project),
    }
    out = _run_hook(payload, capsys, monkeypatch)
    assert out["continue"] is True
    assert "additional_context" in out
    assert isinstance(out["additional_context"], str)
    # Just-written prompt must not appear in this turn's recall (self-hit).
    assert "ZX-HOOK-TOKEN" not in out["additional_context"]
    with Store("hooktest") as st:
        rows = st.events(session_id="conv-hook-1")
        assert any(r["content"] == unique and r["role"] == "user" for r in rows)
        mem = st.conn.execute("SELECT content, tier FROM memories").fetchall()
        assert any(unique in (m["content"] or "") and m["tier"] == "episodic" for m in mem)


def test_before_submit_recalls_history_not_just_written(
    fts_hook_env, capsys, monkeypatch
):
    """Observe-before-recall would let the new prompt outrank stored history."""
    project = fts_hook_env["project"]
    history = "The vault combination is HISTORY-ZX-991 and lives in this older turn."
    with Store("hooktest") as st:
        prior = st.observe(history, role="user", origin="test")

    prompt = "where is the vault combination? HISTORY-ZX-991 also UNIQUE-NEW-PROMPT"
    payload = {
        "hook_event_name": "beforeSubmitPrompt",
        "prompt": prompt,
        "conversation_id": "conv-hook-self",
        "workspace_roots": [str(project)],
        "cwd": str(project),
    }
    out = _run_hook(payload, capsys, monkeypatch)
    ctx = out["additional_context"]
    assert "HISTORY-ZX-991" in ctx
    assert prior.memory_id in ctx
    assert "UNIQUE-NEW-PROMPT" not in ctx
    with Store("hooktest") as st:
        rows = st.events(session_id="conv-hook-self")
        assert any(r["content"] == prompt for r in rows)


def test_post_tool_use_stores_verbatim(fts_hook_env, capsys, monkeypatch):
    project = fts_hook_env["project"]
    tool_input = {"path": "src/haunt/store.py", "offset": 1}
    tool_output = "def init_schema(conn):\n    conn.execute('create table events')"
    payload = {
        "hook_event_name": "postToolUse",
        "tool_name": "Read",
        "tool_input": tool_input,
        "tool_output": tool_output,
        "conversation_id": "conv-hook-2",
        "workspace_roots": [str(project)],
        "cwd": str(project),
    }
    out = _run_hook(payload, capsys, monkeypatch)
    assert out == {} or isinstance(out, dict)
    json.dumps(out)
    with Store("hooktest") as st:
        rows = st.events(session_id="conv-hook-2")
        assert rows, "expected a stored tool event"
        row = rows[0]
        assert row["role"] == "tool"
        assert row["tier"] == "episodic"
        assert row["tool_name"] == "Read"
        assert "src/haunt/store.py" in (row["tool_input"] or "")
        assert "init_schema" in (row["tool_output"] or "")
        mem = st.conn.execute("SELECT content FROM memories").fetchone()
        assert mem and "init_schema" in mem["content"]


def test_infer_event_without_hook_event_name(fts_hook_env, capsys, monkeypatch):
    project = fts_hook_env["project"]
    unique = "bare payload should still store PROMPT-NO-NAME-77"
    payload = {
        "prompt": unique,
        "conversation_id": "conv-infer",
        "workspace_roots": [str(project)],
    }
    out = _run_hook(payload, capsys, monkeypatch)
    assert out.get("continue") is True
    with Store("hooktest") as st:
        rows = st.events()
        assert any(unique in (r["content"] or "") for r in rows)


def test_fail_open_invalid_json(fts_hook_env, capsys, monkeypatch):
    from haunt.cursor_hook import main

    monkeypatch.setattr(sys, "stdin", io.StringIO("not-json{"))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out) == {}


def test_cursor_install_merges_without_clobber(fts_hook_env, tmp_path, monkeypatch):
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir()
    hooks_path = cursor_home / "hooks.json"
    hooks_path.write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "afterFileEdit": [{"command": "./hooks/format.sh"}],
                    "beforeSubmitPrompt": [{"command": "./hooks/audit.sh"}],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CURSOR_HOME", str(cursor_home))
    from haunt.cursor_hook import install_cursor_hooks

    report = install_cursor_hooks()
    data = json.loads(Path(report["hooks_json"]).read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["hooks"]["afterFileEdit"] == [{"command": "./hooks/format.sh"}]
    prompts = data["hooks"]["beforeSubmitPrompt"]
    assert any("audit.sh" in c["command"] for c in prompts)
    assert any("haunt-hook" in c["command"] for c in prompts)
    assert Path(report["launcher"]).is_file()
    for event in (
        "afterAgentResponse",
        "postToolUse",
        "afterShellExecution",
        "afterMCPExecution",
        "sessionStart",
        "sessionEnd",
    ):
        assert any("haunt-hook" in c["command"] for c in data["hooks"][event])


def test_cursor_install_writes_rule_file(fts_hook_env, tmp_path, monkeypatch):
    """cursor-install must write haunt.mdc to ~/.cursor/rules/ even without contrib/ on disk."""
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir()
    monkeypatch.setenv("CURSOR_HOME", str(cursor_home))
    from haunt.cursor_hook import install_cursor_hooks

    report = install_cursor_hooks()
    assert report["rule"] is not None, "rule file should be written"
    rule_path = Path(report["rule"])
    assert rule_path.exists(), f"rule file should exist at {rule_path}"
    content = rule_path.read_text(encoding="utf-8")
    assert "haunt" in content
    assert "memory_recall" in content
    assert "memory_purge" in content
    assert rule_path.name == "haunt.mdc"
    assert rule_path.parent.name == "rules"


def test_skips_memory_mcp_tools(fts_hook_env, capsys, monkeypatch):
    project = fts_hook_env["project"]
    payload = {
        "hook_event_name": "afterMCPExecution",
        "tool_name": "memory_recall",
        "tool_input": '{"query":"x"}',
        "result_json": '{"hits":[]}',
        "conversation_id": "conv-skip",
        "workspace_roots": [str(project)],
    }
    _run_hook(payload, capsys, monkeypatch)
    with Store("hooktest") as st:
        assert st.events() == []


def test_secret_redaction_in_tool_output(fts_hook_env, capsys, monkeypatch):
    project = fts_hook_env["project"]
    payload = {
        "hook_event_name": "postToolUse",
        "tool_name": "Read",
        "tool_input": '{"path": "config/.env"}',
        "tool_output": (
            "API_KEY=sk-live-abc123XYZ456def789ghi012jkl\n"
            "DB_HOST=localhost\n"
            "GITHUB_TOKEN=ghp_aAbBcCdDfFeEgGhHiIjJkKlLmMnNoOpPqQrRsS01\n"
            "AWS_KEY=AKIAIOSFODNN7EXAMPLE\n"
        ),
        "conversation_id": "conv-secret",
        "workspace_roots": [str(project)],
    }
    _run_hook(payload, capsys, monkeypatch)
    with Store("hooktest") as st:
        rows = st.events(session_id="conv-secret")
        assert rows, "expected a stored tool event"
        output = rows[0]["tool_output"] or ""
        assert "sk-live-abc123" not in output
        assert "ghp_aAbBcCdDfF" not in output
        assert "AKIAIOSFODNN7EXAMPLE" not in output
        assert "DB_HOST" in output or "localhost" in output
        assert "[REDACTED]" in output


def test_session_start_returns_worldview_card(fts_hook_env, capsys, monkeypatch):
    project = fts_hook_env["project"]
    with Store("hooktest") as st:
        st.observe("The API key is stored in .env", role="system", tier="semantic")
        st.procedure_write("deploy", "git pull && make", trigger="when shipping")
    payload = {
        "hook_event_name": "sessionStart",
        "session_id": "sess-wv",
        "workspace_roots": [str(project)],
    }
    out = _run_hook(payload, capsys, monkeypatch)
    ctx = out.get("additional_context", "")
    assert "[haunt worldview" in ctx
    assert "memory_recall" in ctx
    assert "API key" in ctx or "facts" in ctx
    assert "deploy" in ctx


def test_secret_redaction_in_tool_input(fts_hook_env, capsys, monkeypatch):
    """Secrets in tool_input (e.g. shell commands with auth headers) must be redacted."""
    project = fts_hook_env["project"]
    secret_cmd = (
        'curl -H "Authorization: Bearer sk-live-abc123XYZ456def789ghi012jkl" '
        "https://api.stripe.com/v1/charges"
    )
    payload = {
        "hook_event_name": "afterShellExecution",
        "command": secret_cmd,
        "output": '{"id": "ch_123", "amount": 5000}',
        "conversation_id": "conv-input-secret",
        "workspace_roots": [str(project)],
    }
    _run_hook(payload, capsys, monkeypatch)
    with Store("hooktest") as st:
        rows = st.events(session_id="conv-input-secret")
        assert rows, "expected a stored shell event"
        stored_input = rows[0]["tool_input"] or ""
        assert "sk-live-abc123" not in stored_input
        assert "[REDACTED]" in stored_input
        assert "api.stripe.com" in stored_input
        mem = st.conn.execute("SELECT content FROM memories").fetchone()
        assert mem and "sk-live-abc123" not in mem["content"]


def test_generic_tool_shell_mcp_io_is_episodic_not_procedural(
    fts_hook_env, capsys, monkeypatch
):
    """Published #23 falsifier: generic hook I/O is episodic, not a named how-to.

    postToolUse / afterShell / afterMCP must store as tier=episodic so
    recall(tier='episodic') hits them and recall(tier='procedural') does not.
    Fails if those handlers hardcode tier='procedural'.
    """
    project = fts_hook_env["project"]
    cases = [
        {
            "hook_event_name": "postToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "src/haunt/store.py"},
            "tool_output": "HOOK-EPISODIC-READ-ZX23 unique Read output",
            "conversation_id": "conv-tier-read",
            "workspace_roots": [str(project)],
            "cwd": str(project),
        },
        {
            "hook_event_name": "afterShellExecution",
            "command": "ls -la",
            "output": "HOOK-EPISODIC-SHELL-ZX23 unique ls output",
            "conversation_id": "conv-tier-shell",
            "workspace_roots": [str(project)],
            "cwd": str(project),
        },
        {
            "hook_event_name": "afterMCPExecution",
            "tool_name": "some_external",
            "tool_input": {"q": "ping"},
            "result_json": '{"ok": true, "token": "HOOK-EPISODIC-MCP-ZX23"}',
            "conversation_id": "conv-tier-mcp",
            "workspace_roots": [str(project)],
            "cwd": str(project),
        },
    ]
    for payload in cases:
        _run_hook(payload, capsys, monkeypatch)

    with Store("hooktest") as st:
        events = st.events()
        assert len(events) == 3
        by_tool = {r["tool_name"]: r for r in events}
        assert by_tool["Read"]["tier"] == "episodic"
        assert by_tool["Shell"]["tier"] == "episodic"
        assert by_tool["some_external"]["tier"] == "episodic"
        assert all(r["role"] == "tool" for r in events)
        mems = st.conn.execute("SELECT content, tier FROM memories").fetchall()
        assert len(mems) == 3
        assert all(m["tier"] == "episodic" for m in mems)

        named = st.procedure_write(
            "deploy-zx23",
            "git pull && make deploy HOOK-NAMED-PROC-ZX23",
            trigger="when shipping",
        )
        assert named.tier == "procedural"
        procs = st.procedure_list()
        assert any(p["name"] == "deploy-zx23" for p in procs)
        assert all(p["name"] != "Read" for p in procs)
        assert all(p["name"] != "Shell" for p in procs)
        assert all(p["name"] != "some_external" for p in procs)

        for query, token in (
            ("HOOK-EPISODIC-READ-ZX23", "HOOK-EPISODIC-READ-ZX23"),
            ("HOOK-EPISODIC-SHELL-ZX23 ls -la", "HOOK-EPISODIC-SHELL-ZX23"),
            ("HOOK-EPISODIC-MCP-ZX23 some_external", "HOOK-EPISODIC-MCP-ZX23"),
        ):
            epi = recall(query, namespace="hooktest", tier="episodic", k=8, store=st)
            assert epi, f"expected episodic hit for {token!r}"
            assert any(token in h.content and h.tier == "episodic" for h in epi)
            proc = recall(query, namespace="hooktest", tier="procedural", k=8, store=st)
            assert all(token not in h.content for h in proc)

        proc_hits = recall(
            "HOOK-NAMED-PROC-ZX23",
            namespace="hooktest",
            tier="procedural",
            k=8,
            store=st,
        )
        assert proc_hits
        assert any(
            "HOOK-NAMED-PROC-ZX23" in h.content and h.tier == "procedural"
            for h in proc_hits
        )
