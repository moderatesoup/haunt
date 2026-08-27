"""Cursor command hooks: verbatim observe/recall. No LLM. Fail-open.

Reads one JSON event on stdin, writes JSON on stdout, always exits 0.
See https://cursor.com/docs/hooks.md
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from haunt.paths import infer_namespace, infer_namespace_context, safe_name
from haunt.recall import Hit, recall
from haunt.store import Store
from haunt.util import snippet

ORIGIN = "cursor-hook"

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)authorization\s*:\s*bearer\s+\S+"),
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
STORE_THOUGHTS_ENV = ("HAUNT_STORE_THOUGHTS",)
TOOL_IO_MAX_CHARS_DEFAULT = 12_000
# C11: format_recall_block() renders the [haunt ns=...] block both hosts
# inject as automatic per-prompt additional_context. Each line is already
# a single ~160-char snippet, but the block as a whole had no ceiling on
# how many such lines it could emit. Both current callers pass a fixed
# k=8, so this budget is a no-op for them today (8 lines is nowhere near
# 4,000 chars); it exists so format_recall_block stays safe as a general
# function, since this block runs on every prompt submission -- far more
# often than an explicit memory_recall call -- so it should stay lean.
RECALL_BLOCK_MAX_CHARS_DEFAULT = 4_000


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
    env = os.environ.get("HAUNT_NAMESPACE")
    if env:
        return safe_name(env)
    return infer_namespace(hook_cwd(payload))


def hook_namespace_context(payload: dict[str, Any]) -> tuple[str, str | None]:
    """Like hook_namespace, but also returns the repository to register.

    An explicit HAUNT_NAMESPACE is a deliberate override, not an inference,
    so it never auto-binds a repository -- matching hook_namespace above.
    """
    env = os.environ.get("HAUNT_NAMESPACE")
    if env:
        return safe_name(env), None
    return infer_namespace_context(hook_cwd(payload))


def hook_session(payload: dict[str, Any]) -> str | None:
    sid = payload.get("conversation_id") or payload.get("session_id")
    if sid:
        return str(sid)
    return None


def hook_idempotency_key(
    payload: dict[str, Any],
    *,
    origin: str,
    event: str,
    session_id: str | None,
) -> str | None:
    """Build a retry key only when the host supplied an event-specific ID."""
    id_fields = (
        "hook_event_id",
        "event_id",
        "generation_id",
        "prompt_id",
        "message_id",
        "tool_use_id",
        "tool_call_id",
        "call_id",
        "request_id",
        "response_id",
    )
    identifiers = {
        key: str(payload[key])
        for key in id_fields
        if payload.get(key) not in (None, "")
    }
    if not identifiers:
        return None
    material = json.dumps(
        {
            "origin": origin,
            "event": event,
            "session": session_id,
            "ids": identifiers,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"hook:{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


def _is_memory_tool(name: str) -> bool:
    n = (name or "").strip()
    if n.startswith("memory_"):
        return True
    leaf = n.split(":")[-1]
    return leaf.startswith("memory_")


def _tool_excluded(name: str) -> bool:
    """Match comma-separated, case-insensitive tool globs from the environment."""
    raw = os.environ.get("HAUNT_EXCLUDE_TOOLS") or ""
    patterns = [part.strip().casefold() for part in raw.split(",") if part.strip()]
    candidate = (name or "").strip().casefold()
    return any(fnmatchcase(candidate, pattern) for pattern in patterns)


# C6 capture policy: EMBED_EXCLUDE_TOOLS is a *separate* control from
# HAUNT_EXCLUDE_TOOLS / _tool_excluded above. HAUNT_EXCLUDE_TOOLS is a
# privacy opt-out -- matching tools are dropped before observe() is ever
# called, so there is no event, no memory row, no FTS entry, nothing. That
# is deliberate: users exclude a tool because its output holds something
# they do not want persisted at all, and this code must never soften that
# into "persisted but not embedded".
#
# HAUNT_EMBED_EXCLUDE_TOOLS controls something narrower: matching tool rows
# are still captured in full (event + memory + FTS, via Store.observe's
# skip_embedding=True), just never embedded or enqueued into
# embedding_jobs. The record stays complete and keyword-searchable; only
# vector-index capacity is saved.
EMBED_EXCLUDE_TOOLS_DEFAULT = "Bash,Read"


def _embed_excluded(name: str) -> bool:
    """Match comma-separated, case-insensitive tool globs deciding embed policy.

    Same glob syntax as _tool_excluded, but a different default: an unset
    HAUNT_EMBED_EXCLUDE_TOOLS means "use EMBED_EXCLUDE_TOOLS_DEFAULT", not
    "exclude nothing" -- a user has to set it to "" explicitly to embed
    every tool. The default excludes Bash and Read because, measured on a
    dogfooded corpus, tool rows are ~80% of all memory and Bash alone is
    ~76% of those (Read ~14%) -- almost entirely raw shell/file output that
    FTS keyword search already covers as well as a vector index would.
    """
    raw = os.environ.get("HAUNT_EMBED_EXCLUDE_TOOLS")
    if raw is None:
        raw = EMBED_EXCLUDE_TOOLS_DEFAULT
    patterns = [part.strip().casefold() for part in raw.split(",") if part.strip()]
    candidate = (name or "").strip().casefold()
    return any(fnmatchcase(candidate, pattern) for pattern in patterns)


def _tool_io_cap() -> int:
    raw = (os.environ.get("HAUNT_TOOL_IO_MAX_CHARS") or "").strip()
    try:
        value = int(raw) if raw else TOOL_IO_MAX_CHARS_DEFAULT
    except ValueError:
        value = TOOL_IO_MAX_CHARS_DEFAULT
    return max(256, min(value, 100_000))


def _cap_tool_io(text: str) -> str:
    cap = _tool_io_cap()
    if len(text) <= cap:
        return text
    omitted = len(text) - cap
    return f"{text[:cap]}\n… [truncated by haunt: {omitted} chars omitted]"


def _prepare_tool_io(tool_input: str, tool_output: str) -> tuple[str, str]:
    return (
        _redact_secrets(_cap_tool_io(tool_input)),
        _redact_secrets(_cap_tool_io(tool_output)),
    )


def _recall_block_cap() -> int:
    """HAUNT_RECALL_BLOCK_MAX_CHARS, clamped. Same parse/fallback/clamp
    idiom as HAUNT_TOOL_IO_MAX_CHARS just above (_tool_io_cap): parse, fall
    back to the default on anything unparsable, then clamp so a bad env
    value can't disable the budget or set it below what one header plus
    one hit line needs (each line is already bounded to roughly 200 chars
    by the fixed 160-char snippet() call below, so 500 always leaves room
    for at least one full line and the dropped-count marker).
    """
    raw = (os.environ.get("HAUNT_RECALL_BLOCK_MAX_CHARS") or "").strip()
    try:
        value = int(raw) if raw else RECALL_BLOCK_MAX_CHARS_DEFAULT
    except ValueError:
        value = RECALL_BLOCK_MAX_CHARS_DEFAULT
    return max(500, min(value, 100_000))


def _drop_marker(dropped: int, cap: int) -> str:
    return f"… [truncated by haunt: {dropped} more hit(s) omitted, block budget {cap} chars]"


def _truncate_header(header: str, cap: int, reason: str) -> str:
    """Last-resort path: `header` cannot coexist with anything else --
    not a body line, not a drop marker -- within `cap` chars. Truncates
    the header text itself with its own inline marker, guaranteed <= cap
    BY CONSTRUCTION: plain Python string slicing costs exactly one char
    per char kept, unlike JSON serialization (see _truncate_hit_content
    in mcp_server.py for the escape-expansion trap that bites when that
    assumption is false there), so no measure-after-the-fact check is
    needed here -- the arithmetic below is exact, not an estimate.

    Both real call sites (cursor_hook, claude_hook) clamp namespace to 80
    chars via safe_name(), so `len(header) > cap` cannot happen with the
    default 4,000-char block cap or any HAUNT_RECALL_BLOCK_MAX_CHARS an
    operator would plausibly set (floor 500). This function exists so
    format_recall_block honestly handles the degenerate case anyway,
    instead of falling through to an unconditional block[:cap] slice --
    which can land anywhere, including through the header, eating
    whatever marker would have reported the omission along with it and
    leaving zero indication anything was cut. That silent failure is
    exactly what every call site of this helper replaces.
    """
    marker = f"… [truncated by haunt: {reason}, block budget {cap} chars]"
    sep = "\n"
    if len(marker) + len(sep) >= cap:
        # cap floor is 500 (_recall_block_cap) and marker is a short
        # fixed-ish string, so unreachable in practice -- still never
        # return something longer than cap regardless.
        return marker[:cap]
    keep = cap - len(marker) - len(sep)
    return f"{header[:keep]}{sep}{marker}"


def format_recall_block(hits: list[Hit], namespace: str) -> str:
    cap = _recall_block_cap()
    header = f"[haunt ns={namespace}]"

    # Degenerate configuration: the header alone (namespace far longer
    # than the configured cap) cannot coexist with anything else at all
    # -- not "(no memories)", not a single hit line, not the marker that
    # would normally report an omission. Handled explicitly, before
    # anything else, rather than falling into the packing logic below and
    # its blind block[:cap] backstop -- the shape that used to slice
    # straight through the header and eat the marker along with it,
    # dropping a real hit with zero indication anything was omitted.
    if len(header) > cap:
        return _truncate_header(header, cap, "namespace exceeds cap")

    # Defense in depth: automatic hook context must never render raw tool I/O,
    # even if a future caller forgets recall(include_untrusted=False).
    safe_hits = [hit for hit in hits if hit.trusted]
    if not safe_hits:
        block = f"{header}\n(no memories)"
        if len(block) <= cap:
            return block
        # header alone fit (just checked above) but cap sits between
        # len(header) and len(header) + len("\n(no memories)") -- still
        # must never silently exceed cap. No hit was actually omitted
        # (there are none), but the namespace text itself has to be cut
        # to say so honestly rather than exceeding cap silently.
        return _truncate_header(header, cap, "no room for body")

    hit_lines = [
        f"{i}  rrf={h.score:.4f}  {h.tier}  {h.memory_id}  {snippet(h.content, 160)}"
        for i, h in enumerate(safe_hits, 1)
    ]
    total = len(header) + sum(len(line) + 1 for line in hit_lines)
    if total <= cap:
        # Common case (both hosts call recall() with a fixed k=8 today):
        # byte-for-byte the same block this function has always produced.
        return "\n".join([header, *hit_lines])
    # Over budget: this block is injected on every prompt submission, so
    # prefer fewer complete lines over a block mangled all the way through
    # -- keep whole lines in their existing rank order until the next one
    # would overflow, then say plainly how many were left out rather than
    # silently cutting the block short (mirrors _cap_tool_io's explicit
    # marker convention). Ordering/ranking is untouched: this only ever
    # drops a suffix of the already-ranked line list, never reorders it.
    #
    # The drop-count marker line is itself appended to this same block, so
    # its own cost has to be reserved *inside* the packing loop below, not
    # added after -- otherwise the marker that reports the overage can
    # itself push the block over `cap` (the bug this fixes: a cap=600
    # block came back 644 chars, the marker line pushing it 44 over).
    # Reserve the worst case: the marker text grows only with `dropped`'s
    # digit count, and `dropped` can be at most len(hit_lines) (if zero
    # lines end up kept) -- so sizing the reservation off len(hit_lines)
    # is always >= the real marker's eventual size, never less, regardless
    # of how many lines actually get kept below.
    marker_reserve = len(_drop_marker(len(hit_lines), cap)) + 1
    available = cap - len(header) - marker_reserve
    if available < 0:
        # header fits alone (checked above) but not alongside even a
        # zero-hit-lines marker -- the same degenerate shape as the
        # header-only check above, just only visible once a marker is
        # actually needed. Handle it the same explicit, marker-safe way
        # rather than let the packing loop's "keep nothing" result reach
        # a blind slice.
        return _truncate_header(header, cap, "no room for any hits")

    used = 0
    kept = 0
    for line in hit_lines:
        cost = len(line) + 1
        if used + cost > available:
            break
        used += cost
        kept += 1
    lines = [header, *hit_lines[:kept]]
    dropped = len(hit_lines) - kept
    if dropped:
        lines.append(_drop_marker(dropped, cap))
    block = "\n".join(lines)
    # Provably <= cap given `available >= 0` above: `used` never exceeds
    # `available` (the loop breaks before adding a line that would push
    # it over), and the real marker for the actual `dropped` count is
    # never longer than the worst-case `marker_reserve` already reserved
    # for it (dropped <= len(hit_lines), and the marker's length tracks
    # only `dropped`'s digit count, which only shrinks as `dropped`
    # shrinks) -- so len(header) + used + 1 + len(real marker) <= cap
    # always holds by construction. Still verified by measurement, not
    # trusted on arithmetic alone: if this is ever wrong (e.g. a future
    # edit to the reservation math above), the fallback is the same
    # explicit, marker-preserving truncation used for the degenerate
    # cases above -- never a blind block[:cap] slice through arbitrary
    # content, which is exactly how this bug slipped through before: it
    # can land anywhere, including through the header, eating the marker
    # along with it and leaving zero indication anything was omitted.
    if len(block) <= cap:
        return block
    return _truncate_header(header, cap, "no room for any hits")


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
    from haunt.provenance import native_provenance

    event = detect_event(payload)
    session_id = hook_session(payload)
    tool_name = kwargs.get("tool_name")
    call_id = None
    if tool_name:
        call_id = (
            payload.get("tool_call_id")
            or payload.get("tool_use_id")
            or payload.get("call_id")
        )
    store.observe(
        kwargs.pop("content", ""),
        session_id=session_id,
        origin=ORIGIN,
        channel="cursor_hook",
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
            channel="cursor_hook",
            origin=ORIGIN,
            tool=tool_name,
            call_id=None if call_id is None else str(call_id),
        ),
        **kwargs,
    )


def _handle_before_submit(store: Store, payload: dict[str, Any], ns: str) -> dict[str, Any]:
    prompt = _as_text(payload.get("prompt"))
    # Recall first so the just-written prompt cannot outrank history.
    hits: list[Hit] = []
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
    # NOTE: Cursor's beforeSubmitPrompt output schema is {continue, user_message}
    # only.  additional_context is NOT honored here (silently dropped).  We still
    # return it so a future Cursor build or third-party runner *could* use it, but
    # agents must not assume per-turn recall is injected into the model.
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
    if _is_memory_tool(name) or _tool_excluded(name):
        return {}
    tool_input, tool_output = _prepare_tool_io(
        _as_text(payload.get("tool_input")),
        _as_text(payload.get("tool_output")),
    )
    # Generic tool I/O is episodic, not a named how-to (meta.kind=procedure).
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


def _handle_after_shell(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    if _tool_excluded("Shell"):
        return {}
    tool_input, tool_output = _prepare_tool_io(
        _as_text(payload.get("command")),
        _as_text(payload.get("output")),
    )
    _observe(
        store,
        payload,
        content="",
        role="tool",
        tier="episodic",
        tool_name="Shell",
        tool_input=tool_input,
        tool_output=tool_output,
        skip_embedding=_embed_excluded("Shell"),
    )
    return {}


def _handle_after_mcp(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    name = _as_text(payload.get("tool_name"))
    if _is_memory_tool(name) or _tool_excluded(name or "mcp"):
        return {}
    tool_input, tool_output = _prepare_tool_io(
        _as_text(payload.get("tool_input")),
        _as_text(payload.get("result_json")),
    )
    _observe(
        store,
        payload,
        content="",
        role="tool",
        tier="episodic",
        tool_name=name or "mcp",
        tool_input=tool_input,
        tool_output=tool_output,
        skip_embedding=_embed_excluded(name or "mcp"),
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
        # This entry point is lifecycle residue by definition; do not classify
        # ordinary prompts/replies from their text.
        recall_class="task",
        # Fixed ceremony row, not user content: keyed on role/tier (both
        # already literal right above) rather than matching the "haunt
        # session start" string, so a future wording tweak to this message
        # can't silently start embedding it again -- and so this decision
        # can't accidentally fire on unrelated content that happens to
        # share the same text. One real namespace held 58 of these rows,
        # all byte-identical, 55 embedded: pure vector-index waste.
        skip_embedding=True,
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
    ns, repo_path = hook_namespace_context(payload)
    with Store(ns, repo_path) as store:
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
    return name in {"haunt-hook"}


def merge_hooks_json(path: Path, command: str) -> dict[str, Any]:
    """Merge haunt hook entries into a Cursor hooks.json. Do not clobber others.

    Delegates to the Cursor host adapter.
    """
    from haunt.hosts.cursor import _merge_hooks_json

    return _merge_hooks_json(path, command)


def _install_rule_file() -> Path | None:
    """Write haunt.mdc into .cursor/rules/."""
    from haunt.hosts.cursor import _HAUNT_MDC

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
    from haunt.bootstrap import bind_launchers
    from haunt.hosts.cursor import install as cursor_install

    home, hook_cmd, mcp_cmd = bind_launchers()
    report = cursor_install(str(home), hook_cmd, mcp_cmd)
    return {
        "haunt_home": str(home),
        "launcher": hook_cmd,
        "hooks_json": report.hooks_path,
        "mcp_json": report.mcp_path,
        "events": report.events,
        "rule": report.rule_path,
        "skill": report.skill_path,
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
