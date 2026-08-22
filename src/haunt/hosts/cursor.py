"""Cursor host adapter: hooks.json + mcp.json + haunt.mdc rule."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from haunt.hosts import HostReport, HostStatus

HOST_NAME = "cursor"

HOOK_EVENTS = (
    "beforeSubmitPrompt",
    "afterAgentResponse",
    "postToolUse",
    "afterShellExecution",
    "afterMCPExecution",
    "sessionStart",
    "sessionEnd",
)


def _cursor_dir() -> Path:
    raw = os.environ.get("CURSOR_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".cursor"


def _hooks_json_path() -> Path:
    override = os.environ.get("CURSOR_HOOKS_JSON")
    if override:
        return Path(override).expanduser()
    return _cursor_dir() / "hooks.json"


def _mcp_json_path() -> Path:
    return _cursor_dir() / "mcp.json"


def _is_haunt_command(command: str) -> bool:
    name = command.replace("\\", "/").rstrip("/").split("/")[-1]
    return name in {"haunt-hook", "engram-hook", "lore-hook"}


def _merge_hooks_json(path: Path, command: str) -> dict[str, Any]:
    """Merge haunt hook entries into a Cursor hooks.json. Do not clobber others."""
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
            if isinstance(item, dict) and _is_haunt_command(
                str(item.get("command", ""))
            ):
                item["command"] = command
                updated = True
        if not updated:
            entries.append({"command": command})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return existing


def _is_haunt_mcp(key: str, entry: dict[str, Any]) -> bool:
    if key == "haunt":
        return True
    cmd = str(entry.get("command", ""))
    name = cmd.replace("\\", "/").rstrip("/").split("/")[-1]
    return name in {"haunt-mcp", "engram-mcp", "lore-mcp"}


def _merge_mcp_json(path: Path, mcp_cmd: str) -> dict[str, Any]:
    """Merge haunt MCP server into Cursor's mcp.json. Do not clobber others."""
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            pass
    servers = existing.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        servers = {}
        existing["mcpServers"] = servers

    found_key: str | None = None
    for key, entry in list(servers.items()):
        if isinstance(entry, dict) and _is_haunt_mcp(key, entry):
            found_key = key
            break

    haunt_entry = {"command": mcp_cmd}
    if found_key:
        servers[found_key] = haunt_entry
    else:
        servers["haunt"] = haunt_entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return existing


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


def _install_rule(cursor_dir: Path) -> Path:
    rules_dir = cursor_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    dest = rules_dir / "haunt.mdc"
    dest.write_text(_HAUNT_MDC, encoding="utf-8")
    old = rules_dir / "engram.mdc"
    if old.exists():
        old.unlink()
    return dest


def install(haunt_home: str, hook_cmd: str, mcp_cmd: str) -> HostReport:
    """Bind Cursor: hooks.json + mcp.json + haunt.mdc."""
    cdir = _cursor_dir()
    seeded = not cdir.exists()

    hooks_path = _hooks_json_path()
    _merge_hooks_json(hooks_path, hook_cmd)

    mcp_path = _mcp_json_path()
    _merge_mcp_json(mcp_path, mcp_cmd)

    rule_path = _install_rule(cdir)

    return HostReport(
        host=HOST_NAME,
        hooks_path=str(hooks_path),
        mcp_path=str(mcp_path),
        rule_path=str(rule_path),
        events=list(HOOK_EVENTS),
        seeded=seeded,
    )


def doctor(haunt_home: str, hook_cmd: str, mcp_cmd: str) -> HostStatus:
    """Check Cursor bindings."""
    status = HostStatus(host=HOST_NAME)

    hooks_path = _hooks_json_path()
    status.hooks_path = str(hooks_path)
    if hooks_path.exists():
        try:
            data = json.loads(hooks_path.read_text(encoding="utf-8"))
            hooks = data.get("hooks", {})
            has_haunt = False
            for event in HOOK_EVENTS:
                entries = hooks.get(event, [])
                if any(
                    isinstance(e, dict) and _is_haunt_command(str(e.get("command", "")))
                    for e in entries
                ):
                    has_haunt = True
            status.hooks_present = has_haunt
            if not has_haunt:
                status.issues.append("haunt hook entries missing from hooks.json")
        except (json.JSONDecodeError, KeyError, TypeError):
            status.issues.append("hooks.json malformed")
    else:
        status.issues.append("hooks.json not found")

    mcp_path = _mcp_json_path()
    status.mcp_path = str(mcp_path)
    if mcp_path.exists():
        try:
            data = json.loads(mcp_path.read_text(encoding="utf-8"))
            servers = data.get("mcpServers", {})
            if not isinstance(servers, dict):
                servers = {}
            status.mcp_present = any(
                _is_haunt_mcp(k, v)
                for k, v in servers.items()
                if isinstance(v, dict)
            )
            if not status.mcp_present:
                status.issues.append("haunt MCP server missing from mcp.json")
        except (json.JSONDecodeError, KeyError, TypeError):
            status.issues.append("mcp.json malformed")
    else:
        status.issues.append("mcp.json not found")

    cdir = _cursor_dir()
    rule = cdir / "rules" / "haunt.mdc"
    status.rule_path = str(rule)
    status.rule_present = rule.exists()
    if not status.rule_present:
        status.issues.append("haunt.mdc rule not found")

    return status
