from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from lore.store import Store


@pytest.fixture
def fts_hook_env(tmp_path, monkeypatch):
    """Isolated LORE_HOME, FTS-only — never download BGE-M3."""
    home = tmp_path / "lorehome"
    project = tmp_path / "myproj"
    project.mkdir()
    monkeypatch.setenv("LORE_HOME", str(home))
    monkeypatch.setenv("LORE_FTS_ONLY", "1")
    monkeypatch.setenv("LORE_EMBED_MODEL", "off")
    monkeypatch.setenv("LORE_NAMESPACE", "hooktest")
    monkeypatch.delenv("ENGRAM_NAMESPACE", raising=False)
    monkeypatch.delenv("ENGRAM_HOME", raising=False)
    monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
    from lore import embed
    from lore.paths import ensure_layout
    from lore.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield {"home": home, "project": project}
    embed.reset()


def _run_hook(payload: dict, capsys, monkeypatch) -> dict:
    from lore.cursor_hook import main

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
    assert "ZX-HOOK-TOKEN" in out["additional_context"]
    assert "episodic" in out["additional_context"]
    with Store("hooktest") as st:
        rows = st.events(session_id="conv-hook-1")
        assert any(r["content"] == unique and r["role"] == "user" for r in rows)
        mem = st.conn.execute("SELECT content, tier FROM memories").fetchall()
        assert any(unique in (m["content"] or "") and m["tier"] == "episodic" for m in mem)


def test_post_tool_use_stores_verbatim(fts_hook_env, capsys, monkeypatch):
    project = fts_hook_env["project"]
    tool_input = {"path": "src/lore/store.py", "offset": 1}
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
    json.dumps(out)  # stdout JSON is valid
    with Store("hooktest") as st:
        rows = st.events(session_id="conv-hook-2")
        assert rows, "expected a stored tool event"
        row = rows[0]
        assert row["role"] == "tool"
        assert row["tier"] == "procedural"
        assert row["tool_name"] == "Read"
        assert "src/lore/store.py" in (row["tool_input"] or "")
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
    from lore.cursor_hook import main

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
    from lore.cursor_hook import install_cursor_hooks

    report = install_cursor_hooks()
    data = json.loads(Path(report["hooks_json"]).read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["hooks"]["afterFileEdit"] == [{"command": "./hooks/format.sh"}]
    prompts = data["hooks"]["beforeSubmitPrompt"]
    assert any("audit.sh" in c["command"] for c in prompts)
    assert any("engram-hook" in c["command"] for c in prompts)
    assert Path(report["launcher"]).is_file()
    for event in (
        "afterAgentResponse",
        "postToolUse",
        "afterShellExecution",
        "afterMCPExecution",
        "sessionStart",
        "sessionEnd",
    ):
        assert any("engram-hook" in c["command"] for c in data["hooks"][event])


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
