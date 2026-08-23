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

MCP server `haunt`. Tools are `memory_*`. Hooks store; recall is not automatic.

## Automatic vs not

Hooks auto-log prompts, replies, tool I/O, shell, MCP (skip `memory_*`). Do not double-observe.
`sessionStart` may inject `[haunt worldview ns=…]` — do not assume it arrived.
If no `[haunt ns=…]` block is visible, call `memory_recall` with the user's exact wording.

## Temporal

`compile() runs automatically on memory_recall`. Pass the user's wording. Do not compute `since`/`until`. Clock is `event_time` (including speech verbs). Do not filter on storage `ts`.
Default recall hides superseded rows unless you pass `as_of`.

## Call

- `memory_recall` `query`=user wording. Optional `as_of`, `since`, `until`, `clock`, `k`.
- `memory_observe` only if hooks are off. `text`, `tier`, `origin`, `session`. Never summarize.
- `memory_worldview` if no worldview card is in context.
- `memory_procedure` `action`=`write`/`get`/`list`. Write needs `name`, `body`.
- `memory_contradict` `memory_id`, optional `replacement`. Supersedes. Does not delete.
- `memory_purge` `memory_id`. Hard delete.
- `memory_timeline` only with ISO `since`/`until` or `session`. No NL compile.
- `memory_session_end` `session`. `ok: false` if nothing ended.
- `memory_health` / `memory_namespaces` when you need counts or the namespace list.

## Namespace

Inferred. Do not invent. When building haunt, namespace is `haunt`.
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
