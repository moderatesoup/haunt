"""Claude Code command hooks: verbatim observe/recall. No LLM. Fail-open.

Reads one JSON event on stdin, writes JSON on stdout, always exits 0.
Claude Code hook events:
  UserPromptSubmit, Stop, SessionStart, SessionEnd,
  PostToolUse, PostToolUseFailure

Payload shape differs from Cursor — this module translates CC payloads
into the same Store.observe calls. origin=claude-code.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from haunt.cursor_hook import (
    _as_text,
    _embed_excluded,
    _is_memory_tool,
    _prepare_tool_io,
    _tool_excluded,
    format_recall_block,
    format_worldview_card,
    hook_idempotency_key,
)
from haunt.paths import infer_namespace, infer_namespace_context, safe_name
from haunt.recall import recall
from haunt.store import Store

ORIGIN = "claude-code"

CC_EVENTS = (
    "UserPromptSubmit",
    "Stop",
    "SessionStart",
    "SessionEnd",
    "PostToolUse",
    "PostToolUseFailure",
)


def _hook_cwd(payload: dict[str, Any]) -> Path | None:
    project = os.environ.get("CLAUDE_PROJECT_DIR") or os.environ.get("CURSOR_PROJECT_DIR")
    if project:
        return Path(project)
    cwd = payload.get("cwd")
    if cwd:
        return Path(str(cwd))
    return None


def _hook_namespace(payload: dict[str, Any]) -> str:
    env = os.environ.get("HAUNT_NAMESPACE")
    if env:
        return safe_name(env)
    return infer_namespace(_hook_cwd(payload))


def _hook_namespace_context(payload: dict[str, Any]) -> tuple[str, str | None]:
    """Like _hook_namespace, but also returns the repository to register.

    An explicit HAUNT_NAMESPACE is a deliberate override, not an inference,
    so it never auto-binds a repository -- matching _hook_namespace above.
    """
    env = os.environ.get("HAUNT_NAMESPACE")
    if env:
        return safe_name(env), None
    return infer_namespace_context(_hook_cwd(payload))


def _hook_session(payload: dict[str, Any]) -> str | None:
    sid = payload.get("session_id") or payload.get("conversation_id")
    if sid:
        return str(sid)
    return None


def _hook_specific_output(event: str, additional_context: str) -> dict[str, Any]:
    """Claude Code only honors additionalContext inside hookSpecificOutput."""
    return {
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": additional_context,
        }
    }


def _observe(store: Store, payload: dict[str, Any], **kwargs: Any) -> None:
    from haunt.provenance import native_provenance

    event = detect_event(payload)
    session_id = _hook_session(payload)
    tool_name = kwargs.get("tool_name")
    call_id = None
    if tool_name:
        call_id = (
            payload.get("tool_use_id")
            or payload.get("tool_call_id")
            or payload.get("call_id")
        )
    store.observe(
        kwargs.pop("content", ""),
        session_id=session_id,
        origin=ORIGIN,
        channel="claude_code_hook",
        meta={"hook": event},
        idempotency_key=hook_idempotency_key(
            payload,
            origin=ORIGIN,
            event=event,
            session_id=session_id,
        ),
        defer_embedding=True,
        producer_call_id=None if call_id is None else str(call_id),
        provenance=native_provenance(
            channel="claude_code_hook",
            origin=ORIGIN,
            tool=tool_name,
            call_id=None if call_id is None else str(call_id),
        ),
        **kwargs,
    )


def _handle_user_prompt_submit(
    store: Store, payload: dict[str, Any], ns: str
) -> dict[str, Any]:
    """UserPromptSubmit: recall first, then observe the prompt."""
    prompt = _as_text(payload.get("prompt") or payload.get("user_prompt", ""))
    hits = []
    if prompt.strip():
        try:
            hits = recall(
                prompt,
                namespace=ns,
                k=8,
                store=store,
                include_untrusted=False,
                use_vectors=False,
            )
        except Exception:
            hits = []
        _observe(store, payload, content=prompt, role="user", tier="episodic")

    return _hook_specific_output("UserPromptSubmit", format_recall_block(hits, ns))


def _handle_stop(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    """Stop: observe assistant reply from last_assistant_message. Never exit 2."""
    text = _as_text(
        payload.get("last_assistant_message")
        or payload.get("assistant_message", "")
    )
    if text.strip():
        _observe(store, payload, content=text, role="assistant", tier="episodic")
    return {}


def _handle_session_start(
    store: Store, payload: dict[str, Any], ns: str
) -> dict[str, Any]:
    """SessionStart: log coordinate event, return worldview (same JSON shape)."""
    _observe(
        store,
        payload,
        content="haunt session start",
        role="system",
        tier="coordinate",
        # SessionStart is host lifecycle residue, not a text-derived guess.
        recall_class="task",
        # Fixed ceremony row, not user content: keyed on role/tier (both
        # already literal right above), not the "haunt session start"
        # string -- see cursor_hook._handle_session_start for the full
        # rationale (same policy, same shape, mirrored across both hosts).
        skip_embedding=True,
    )
    wv = store.worldview()
    card = format_worldview_card(wv)
    intro = (
        "You have persistent local memory via haunt (MCP server haunt). "
        "Hooks store turns automatically — do not double-observe what hooks already log. "
        "You MUST call memory_recall with the user's wording yourself unless a "
        "[haunt ns=…] block is already visible in this context. "
        f"Namespace: {ns}."
    )
    return _hook_specific_output("SessionStart", f"{intro}\n\n{card}")


def _handle_session_end(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    """SessionEnd: close session."""
    store.end_session(_hook_session(payload))
    return {}


def _tool_output_text(payload: dict[str, Any]) -> str:
    return _as_text(
        payload.get("tool_output")
        or payload.get("tool_response")
        or payload.get("error")
        or ""
    )


def _handle_post_tool_use(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    """PostToolUse / PostToolUseFailure: log tool I/O as episodic, skip memory_*."""
    name = _as_text(payload.get("tool_name")) or "tool"
    if _is_memory_tool(name) or _tool_excluded(name):
        return {}
    tool_input, tool_output = _prepare_tool_io(
        _as_text(payload.get("tool_input")),
        _tool_output_text(payload),
    )
    _observe(
        store,
        payload,
        content="",
        role="tool",
        tier="episodic",
        tool_name=name,
        tool_input=tool_input,
        tool_output=tool_output,
        skip_embedding=_embed_excluded(name),
    )
    return {}


def detect_event(payload: dict[str, Any]) -> str:
    """Detect the CC event from the payload."""
    name = (
        payload.get("hook_event_name")
        or payload.get("hookEventName")
        or payload.get("event")
        or ""
    )
    return str(name)


def handle_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one Claude Code hook payload. Fail-open on errors."""
    event = detect_event(payload)
    ns, repo_path = _hook_namespace_context(payload)
    with Store(ns, repo_path) as store:
        if event == "UserPromptSubmit":
            return _handle_user_prompt_submit(store, payload, ns)
        if event == "Stop":
            return _handle_stop(store, payload)
        if event == "SessionStart":
            return _handle_session_start(store, payload, ns)
        if event == "SessionEnd":
            return _handle_session_end(store, payload)
        if event in ("PostToolUse", "PostToolUseFailure"):
            return _handle_post_tool_use(store, payload)
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
