"""Cursor command hooks: verbatim observe/recall. No LLM. Fail-open.

Reads one JSON event on stdin, writes JSON on stdout, always exits 0.
See https://cursor.com/docs/hooks.md
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from haunt.paths import infer_namespace, safe_name
from haunt.recall import Hit, recall
from haunt.store import Store
from haunt.util import snippet

ORIGIN = "cursor-hook"

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"""(?i)"""
        r"""(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token"""
        r"""|auth[_-]?token|bearer|password|passwd|private[_-]?key"""
        r"""|client[_-]?secret|webhook[_-]?secret|signing[_-]?secret"""
        r"""|database[_-]?url|connection[_-]?string)"""
        r"""[\s]*[=:]\s*["']?([^\s"']{8,})"""
    ),
    re.compile(r"""(?:sk|pk)[-_](?:live|test|prod)[A-Za-z0-9_\-]{16,}"""),
    re.compile(r"""ghp_[A-Za-z0-9]{36,}"""),
    re.compile(r"""glpat-[A-Za-z0-9\-_]{20,}"""),
    re.compile(r"""xox[bsrap]-[A-Za-z0-9\-]{10,}"""),
    re.compile(r"""eyJ[A-Za-z0-9_\-]{20,}\.eyJ[A-Za-z0-9_\-]{20,}"""),
    re.compile(r"""AKIA[0-9A-Z]{16}"""),
]


def _redact_secrets(text: str) -> str:
    """Best-effort redaction of obvious secret patterns. Not exhaustive."""
    if not text:
        return text
    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


HOOK_EVENTS = (
    "beforeSubmitPrompt",
    "afterAgentResponse",
    "postToolUse",
    "afterShellExecution",
    "afterMCPExecution",
    "sessionStart",
    "sessionEnd",
)
STORE_THOUGHTS_ENV = ("HAUNT_STORE_THOUGHTS", "ENGRAM_STORE_THOUGHTS", "LORE_STORE_THOUGHTS")


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
    env = (
        os.environ.get("HAUNT_NAMESPACE")
        or os.environ.get("LORE_NAMESPACE")
        or os.environ.get("ENGRAM_NAMESPACE")
    )
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
    lines = [f"[haunt ns={namespace}]"]
    if not hits:
        lines.append("(no memories)")
        return "\n".join(lines)
    for i, h in enumerate(hits, 1):
        lines.append(
            f"{i}  {h.score:.4f}  {h.tier}  {h.memory_id}  {snippet(h.content, 160)}"
        )
    return "\n".join(lines)


def format_timeline_block(rows: list[dict[str, Any]], namespace: str) -> str:
    lines = [f"[haunt recent ns={namespace}]"]
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
    # NOTE: Cursor's beforeSubmitPrompt output schema is {continue, user_message}
    # only.  additional_context is NOT honored here (silently dropped).  We still
    # return it so a future Cursor build or third-party runner *could* use it, but
    # agents must not assume per-turn recall is injected into the model.
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
    # TODO: tier="procedural" is a lane mix — generic tool I/O is episodic,
    # not a named how-to.  Should be "episodic" unless meta.kind=procedure.
    _observe(
        store,
        payload,
        content="",
        role="tool",
        tier="procedural",
        tool_name=name,
        tool_input=_redact_secrets(_as_text(payload.get("tool_input"))),
        tool_output=_redact_secrets(_as_text(payload.get("tool_output"))),
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
        tool_input=_redact_secrets(_as_text(payload.get("command"))),
        tool_output=_redact_secrets(_as_text(payload.get("output"))),
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
        tool_input=_redact_secrets(_as_text(payload.get("tool_input"))),
        tool_output=_redact_secrets(_as_text(payload.get("result_json"))),
    )
    return {}


def format_worldview_card(wv: dict[str, Any]) -> str:
    """Render a worldview dict as a compact text card for additional_context."""
    lines = [f"[haunt worldview ns={wv['namespace']}]"]
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
        content="haunt session start",
        role="system",
        tier="coordinate",
    )
    wv = store.worldview()
    card = format_worldview_card(wv)
    intro = (
        "You have persistent local memory via haunt (MCP server haunt). "
        "Hooks store turns automatically — do not double-observe what hooks already log. "
        "Hooks do NOT inject recall into your context on beforeSubmitPrompt "
        "(additional_context there is unproven). You MUST call memory_recall "
        "with the user's wording yourself unless a [haunt ns=…] block is "
        "already visible in this context. "
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


def _is_haunt_command(command: str) -> bool:
    name = command.replace("\\", "/").rstrip("/").split("/")[-1]
    return name in {"haunt-hook", "engram-hook", "lore-hook"}


def merge_hooks_json(path: Path, command: str) -> dict[str, Any]:
    """Merge haunt hook entries into a Cursor hooks.json. Do not clobber others.

    Delegates to the Cursor host adapter.
    """
    from haunt.hosts.cursor import _merge_hooks_json

    return _merge_hooks_json(path, command)


_HAUNT_MDC = """\
---
description: haunt local memory — hooks store, agents must recall
alwaysApply: true
---

# haunt

Local-first verbatim memory. MCP server name is `haunt`; tools are
prefixed `memory_`.

## What hooks do (automatic)

Cursor hooks automatically LOG turns: prompts, replies, tool calls,
shell output, MCP results. This happens without agent action.
Do not double-observe what hooks already store.

## What hooks do NOT do

Hooks do NOT reliably inject recall context into your prompt.
`beforeSubmitPrompt` returns `additional_context` but Cursor's official
output schema for that hook is only `{continue, user_message}` —
injection is unproven and may not arrive.

## Agent responsibility: recall

If no `[haunt ns=…]` block is visible in your current context, you MUST
call `memory_recall` with the user's exact wording before acting.
Do not assume prior context was injected.

## Observe rules

- Hooks handle episodic logging. Only call `memory_observe` manually
  when hooks are absent (e.g. Grok Bot).
- tier=semantic for durable facts. tier=episodic for chat.
- Always pass `origin` (e.g. "cursor", "cli") and `session` id when
  available.
- Never summarize. Never distill. Store verbatim or don't store.

## Skip list (never observe)

- Secrets, tokens, API keys, passwords
- Acks: "ok", "got it", "sure", empty turns
- `memory_*` tool inputs/outputs (hooks already skip these)
- Entire READMEs or large file dumps — store a pointer, not the blob

## Worldview

Call `memory_worldview` when you need the full namespace briefing.
The sessionStart hook tries to inject a compact worldview card, but
verify it arrived (look for `[haunt worldview ns=…]`).

## Procedures

Use `memory_procedure` action=write only when deliberately promoting a
how-to the user wants remembered. Include `name`, verbatim `body`, and
optional `trigger`. Do not auto-extract procedures from every turn.

## Contradict / purge

Use `memory_contradict` with the `memory_id` of a now-wrong fact to
supersede it (sets valid_to). Optionally pass `replacement` to store
the correction.

Use `memory_purge` to permanently hard-delete a memory and its entire
provenance chain (FTS, embedding, graph, orphaned event). Data is gone
after purge — not just marked superseded.

## Recall scores

RRF scores are rank-normalized, not relevance scores. A hit with
score 0.03 is not "3% relevant" — it just ranked lower. Ignore hits
that are clearly off-corpus rather than trusting the number.

## Namespace

Inferred from `CURSOR_PROJECT_DIR` / git / cwd. Do not invent
namespaces unless the user asks.

## No hooks environment (Grok Bot)

When hooks are unavailable, the agent must both observe AND recall
manually. Observe each user turn (tier=episodic) and each durable
fact (tier=semantic). Recall before acting.
"""


def _install_rule_file() -> Path | None:
    """Write haunt.mdc into .cursor/rules/."""
    rules_dir = cursor_dir() / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    dest = rules_dir / "haunt.mdc"
    dest.write_text(_HAUNT_MDC, encoding="utf-8")
    old = rules_dir / "engram.mdc"
    if old.exists():
        old.unlink()
    return dest


def install_cursor_hooks() -> dict[str, Any]:
    """Write ~/.haunt/bin/haunt-hook, merge ~/.cursor/hooks.json + mcp.json, install rule.

    Delegates to the Cursor host adapter for the full bind.
    """
    from haunt.bootstrap import write_hook_launcher, write_launcher
    from haunt.hosts.cursor import install as cursor_install
    from haunt.paths import bin_dir, ensure_layout

    home = ensure_layout()
    write_launcher()
    launcher = write_hook_launcher()
    hook_cmd = str(launcher)
    mcp_cmd = str(bin_dir() / "haunt-mcp")
    report = cursor_install(str(home), hook_cmd, mcp_cmd)
    return {
        "haunt_home": str(home),
        "lore_home": str(home),
        "launcher": hook_cmd,
        "hooks_json": report.hooks_path,
        "mcp_json": report.mcp_path,
        "events": report.events,
        "rule": report.rule_path,
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
