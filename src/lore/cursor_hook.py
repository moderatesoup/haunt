"""Cursor command hooks: verbatim observe/recall. No LLM. Fail-open.

Reads one JSON event on stdin, writes JSON on stdout, always exits 0.
See https://cursor.com/docs/hooks.md
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from lore.paths import infer_namespace, safe_name
from lore.recall import Hit, recall
from lore.store import Store
from lore.util import snippet

ORIGIN = "cursor-hook"
HOOK_EVENTS = (
    "beforeSubmitPrompt",
    "afterAgentResponse",
    "postToolUse",
    "afterShellExecution",
    "afterMCPExecution",
    "sessionStart",
    "sessionEnd",
)
STORE_THOUGHTS_ENV = ("ENGRAM_STORE_THOUGHTS", "LORE_STORE_THOUGHTS")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def cursor_dir() -> Path:
    raw = os.environ.get("CURSOR_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".cursor"


def cursor_hooks_json() -> Path:
    override = os.environ.get("CURSOR_HOOKS_JSON")
    if override:
        return Path(override).expanduser()
    return cursor_dir() / "hooks.json"


def detect_event(payload: dict[str, Any]) -> str:
    """Use hook_event_name, or infer from payload keys if Cursor omits it."""
    name = (
        payload.get("hook_event_name")
        or payload.get("hook_event")
        or payload.get("event")
        or ""
    )
    if name:
        return str(name)
    if payload.get("reason") and (
        "session_id" in payload or "final_status" in payload
    ) and "prompt" not in payload and "text" not in payload:
        return "sessionEnd"
    if "composer_mode" in payload or (
        "is_background_agent" in payload
        and "session_id" in payload
        and "command" not in payload
        and "prompt" not in payload
    ):
        return "sessionStart"
    if "prompt" in payload:
        return "beforeSubmitPrompt"
    if "result_json" in payload:
        return "afterMCPExecution"
    if "tool_output" in payload or (
        payload.get("tool_name") and "tool_input" in payload and "output" not in payload
    ):
        return "postToolUse"
    if "command" in payload and "output" in payload:
        return "afterShellExecution"
    if "text" in payload and "duration_ms" in payload:
        return "afterAgentThought"
    if "text" in payload:
        return "afterAgentResponse"
    if "session_id" in payload and "reason" in payload:
        return "sessionEnd"
    if "session_id" in payload:
        return "sessionStart"
    return ""


def hook_cwd(payload: dict[str, Any]) -> Path | None:
    project = os.environ.get("CURSOR_PROJECT_DIR") or os.environ.get("CLAUDE_PROJECT_DIR")
    if project:
        return Path(project)
    roots = payload.get("workspace_roots") or []
    if roots:
        return Path(str(roots[0]))
    cwd = payload.get("cwd")
    if cwd:
        return Path(str(cwd))
    return None


def hook_namespace(payload: dict[str, Any]) -> str:
    env = os.environ.get("LORE_NAMESPACE") or os.environ.get("ENGRAM_NAMESPACE")
    if env:
        return safe_name(env)
    return infer_namespace(hook_cwd(payload))


def hook_session(payload: dict[str, Any]) -> str | None:
    sid = payload.get("conversation_id") or payload.get("session_id")
    if sid:
        return str(sid)
    return None


def _is_memory_tool(name: str) -> bool:
    n = (name or "").strip()
    if n.startswith("memory_"):
        return True
    leaf = n.split(":")[-1]
    return leaf.startswith("memory_")


def format_recall_block(hits: list[Hit], namespace: str) -> str:
    lines = [f"[engram ns={namespace}]"]
    if not hits:
        lines.append("(no memories)")
        return "\n".join(lines)
    for i, h in enumerate(hits, 1):
        lines.append(
            f"{i}  {h.score:.4f}  {h.tier}  {h.memory_id}  {snippet(h.content, 160)}"
        )
    return "\n".join(lines)


def format_timeline_block(rows: list[dict[str, Any]], namespace: str) -> str:
    lines = [f"[engram recent ns={namespace}]"]
    if not rows:
        lines.append("(no memories)")
        return "\n".join(lines)
    for i, r in enumerate(rows, 1):
        body = r.get("content") or ""
        if r.get("tool_name"):
            body = f"[tool:{r['tool_name']}] {body}".strip()
        mid = r.get("id") or ""
        lines.append(f"{i}  {r.get('tier', '')}  {mid}  {snippet(str(body), 160)}")
    return "\n".join(lines)


def _observe(store: Store, payload: dict[str, Any], **kwargs: Any) -> None:
    store.observe(
        kwargs.pop("content", ""),
        session_id=hook_session(payload),
        origin=ORIGIN,
        meta={"hook": detect_event(payload)},
        **kwargs,
    )


def _handle_before_submit(store: Store, payload: dict[str, Any], ns: str) -> dict[str, Any]:
    prompt = _as_text(payload.get("prompt"))
    if prompt.strip():
        _observe(store, payload, content=prompt, role="user", tier="episodic")
    hits: list[Hit] = []
    if prompt.strip():
        try:
            hits = recall(prompt, namespace=ns, k=8, store=store)
        except Exception:
            hits = []
    return {
        "continue": True,
        "additional_context": format_recall_block(hits, ns),
    }


def _handle_after_response(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    text = _as_text(payload.get("text"))
    if text.strip():
        _observe(store, payload, content=text, role="assistant", tier="episodic")
    return {}


def _handle_after_thought(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    if not any(_truthy(os.environ.get(k)) for k in STORE_THOUGHTS_ENV):
        return {}
    text = _as_text(payload.get("text"))
    if text.strip():
        _observe(store, payload, content=text, role="system", tier="coordinate")
    return {}


def _handle_post_tool(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    name = _as_text(payload.get("tool_name")) or "tool"
    if _is_memory_tool(name):
        return {}
    _observe(
        store,
        payload,
        content="",
        role="tool",
        tier="procedural",
        tool_name=name,
        tool_input=_as_text(payload.get("tool_input")),
        tool_output=_as_text(payload.get("tool_output")),
    )
    return {}


def _handle_after_shell(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    _observe(
        store,
        payload,
        content="",
        role="tool",
        tier="procedural",
        tool_name="Shell",
        tool_input=_as_text(payload.get("command")),
        tool_output=_as_text(payload.get("output")),
    )
    return {}


def _handle_after_mcp(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    name = _as_text(payload.get("tool_name"))
    if _is_memory_tool(name):
        return {}
    _observe(
        store,
        payload,
        content="",
        role="tool",
        tier="procedural",
        tool_name=name or "mcp",
        tool_input=_as_text(payload.get("tool_input")),
        tool_output=_as_text(payload.get("result_json")),
    )
    return {}


def format_worldview_card(wv: dict[str, Any]) -> str:
    """Render a worldview dict as a compact text card for additional_context."""
    lines = [f"[engram worldview ns={wv['namespace']}]"]
    counts = wv.get("counts", {})
    lines.append(
        f"events={counts.get('events', 0)} memories={counts.get('memories', 0)} "
        f"sessions={counts.get('sessions', 0)}"
    )
    facts = wv.get("facts", [])
    if facts:
        lines.append(f"facts ({len(facts)}):")
        for f in facts[:12]:
            lines.append(f"  {snippet(f.get('content', ''), 140)}")
    names = wv.get("names", [])
    if names:
        lines.append(f"entities ({len(names)}):")
        for n in names[:12]:
            lines.append(f"  {n['name']} ({n['type']})")
    procs = wv.get("procedures", [])
    if procs:
        lines.append(f"procedures ({len(procs)}):")
        for p in procs:
            trigger = f" — when: {p['trigger']}" if p.get("trigger") else ""
            lines.append(f"  {p['name']}{trigger}")
    return "\n".join(lines)


def _handle_session_start(store: Store, payload: dict[str, Any], ns: str) -> dict[str, Any]:
    _observe(
        store,
        payload,
        content="engram session start",
        role="system",
        tier="coordinate",
    )
    wv = store.worldview()
    card = format_worldview_card(wv)
    intro = (
        "You have persistent local memory via engram (MCP tools "
        "memory_recall / memory_observe / memory_worldview / memory_procedure). "
        "Before acting on a user request, call memory_recall with their wording. "
        "Store new facts with memory_observe (tier=semantic). "
        f"Namespace: {ns}."
    )
    return {"additional_context": f"{intro}\n\n{card}"}


def _handle_session_end(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    store.end_session(hook_session(payload))
    return {}


def handle_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one Cursor hook payload. Raises on store errors (caller fail-opens)."""
    event = detect_event(payload)
    if event == "afterAgentThought" and not any(
        _truthy(os.environ.get(k)) for k in STORE_THOUGHTS_ENV
    ):
        return {}
    ns = hook_namespace(payload)
    with Store(ns) as store:
        if event == "beforeSubmitPrompt":
            return _handle_before_submit(store, payload, ns)
        if event == "afterAgentResponse":
            return _handle_after_response(store, payload)
        if event == "afterAgentThought":
            return _handle_after_thought(store, payload)
        if event == "postToolUse":
            return _handle_post_tool(store, payload)
        if event == "afterShellExecution":
            return _handle_after_shell(store, payload)
        if event == "afterMCPExecution":
            return _handle_after_mcp(store, payload)
        if event == "sessionStart":
            return _handle_session_start(store, payload, ns)
        if event == "sessionEnd":
            return _handle_session_end(store, payload)
    return {}


def run(raw: str) -> dict[str, Any]:
    """Parse stdin JSON and handle it. Fail-open to {}."""
    try:
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            return {}
        return handle_event(payload)
    except Exception:
        return {}


def _is_engram_command(command: str) -> bool:
    name = command.replace("\\", "/").rstrip("/").split("/")[-1]
    return name in {"engram-hook", "lore-hook"}


def merge_hooks_json(path: Path, command: str) -> dict[str, Any]:
    """Merge engram hook entries into a Cursor hooks.json. Do not clobber others."""
    existing: dict[str, Any] = {"version": 1, "hooks": {}}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {"version": 1, "hooks": {}}
    existing.setdefault("version", 1)
    hooks = existing.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        existing["hooks"] = hooks
    for event in HOOK_EVENTS:
        entries = hooks.get(event)
        if not isinstance(entries, list):
            entries = []
            hooks[event] = entries
        updated = False
        for item in entries:
            if isinstance(item, dict) and _is_engram_command(str(item.get("command", ""))):
                item["command"] = command
                updated = True
        if not updated:
            entries.append({"command": command})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return existing


def install_cursor_hooks() -> dict[str, Any]:
    """Write ~/.lore/bin/engram-hook and merge ~/.cursor/hooks.json."""
    from lore.bootstrap import write_hook_launcher, write_launcher
    from lore.paths import ensure_layout

    home = ensure_layout()
    write_launcher()
    launcher = write_hook_launcher()
    hooks_path = cursor_hooks_json()
    command = str(launcher)
    merge_hooks_json(hooks_path, command)
    return {
        "lore_home": str(home),
        "launcher": command,
        "hooks_json": str(hooks_path),
        "events": list(HOOK_EVENTS),
    }


def main() -> None:
    try:
        raw = sys.stdin.read()
        out = run(raw)
        if not isinstance(out, dict):
            out = {}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    except Exception:
        sys.stdout.write("{}\n")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
