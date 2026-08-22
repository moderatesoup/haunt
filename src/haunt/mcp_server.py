"""MCP stdio server. Tools are verbatim store/recall — haunt never calls an LLM."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from mcp.server import MCPServer

from haunt.paths import resolve_namespace
from haunt.recall import recall
from haunt.store import Store, list_namespaces

server = MCPServer(
    name="haunt",
    version="0.1.0",
    instructions=(
        "haunt is local-first verbatim agent memory. "
        "If hooks are active (Cursor or Claude Code), they log turns "
        "automatically — do NOT also call memory_observe (that would "
        "double-store). Only call memory_observe when hooks are absent "
        "(e.g. Grok Bot). "
        "Call memory_recall to fetch prior context. Never summarize on write."
    ),
)


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


@server.tool(description="Store a verbatim agent turn or tool call. No summarization.")
def memory_observe(
    text: str = "",
    namespace: Optional[str] = None,
    tier: str = "episodic",
    session: Optional[str] = None,
    role: str = "user",
    tool_name: Optional[str] = None,
    tool_input: Optional[str] = None,
    tool_output: Optional[str] = None,
    event_time: Optional[str] = None,
    origin: str = "mcp",
) -> str:
    ns = resolve_namespace(namespace)
    with Store(ns) as st:
        r = st.observe(
            text,
            role=role,
            tier=tier,
            session_id=session,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            event_time=event_time,
            origin=origin,
        )
    return _json(
        {
            "ok": True,
            "event_id": r.event_id,
            "memory_id": r.memory_id,
            "session_id": r.session_id,
            "namespace": r.namespace,
            "tier": r.tier,
            "embedded": r.embedded,
            "entities": r.entities,
        }
    )


@server.tool(description="Hybrid recall over verbatim memories (vec + FTS5 + RRF). Scores are rank-normalized (not relevance probabilities).")
def memory_recall(
    query: str,
    namespace: Optional[str] = None,
    as_of: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    tier: Optional[str] = None,
    k: int = 8,
) -> str:
    ns = resolve_namespace(namespace)
    with Store(ns) as st:
        hits = recall(
            query,
            namespace=ns,
            as_of=as_of,
            since=since,
            until=until,
            tier=tier,
            k=k,
            store=st,
        )
    return _json({"namespace": ns, "query": query, "hits": [h.as_dict() for h in hits]})


@server.tool(description="List stored events in time order.")
def memory_timeline(
    namespace: Optional[str] = None,
    session: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 50,
) -> str:
    ns = resolve_namespace(namespace)
    with Store(ns) as st:
        rows = st.events(session_id=session, since=since, until=until, limit=limit)
    return _json({"namespace": ns, "events": rows})


@server.tool(description="Health and counts for a namespace.")
def memory_health(namespace: Optional[str] = None) -> str:
    from haunt.embed import state as embed_state
    from haunt.paths import haunt_home

    ns = resolve_namespace(namespace)
    es = embed_state()
    with Store(ns) as st:
        stats = st.stats()
        vec_info: dict = {"ok": st.vec_ok()}
        ver = st.vec_version()
        if ver:
            vec_info["version"] = ver
    return _json(
        {
            "namespace": ns,
            "db_path": stats.get("db_path", ""),
            "haunt_home": str(haunt_home()),
            "sqlite_vec": vec_info,
            "embed": {
                "loaded": es.model_id,
                "dim": es.dim,
                "available": es.available,
                "requested": es.requested,
                "fallback": es.fallback,
            },
            "stats": stats,
        }
    )


@server.tool(description="List all memory namespaces.")
def memory_namespaces() -> str:
    return _json({"namespaces": list_namespaces()})


@server.tool(description="Mark a session ended. No distillation — just close it.")
def memory_session_end(
    namespace: Optional[str] = None,
    session: Optional[str] = None,
) -> str:
    ns = resolve_namespace(namespace)
    with Store(ns) as st:
        result = st.end_session(session)
    payload = {
        "ok": bool(result.get("ok")),
        "namespace": ns,
        "session_id": result.get("session_id"),
        "distilled": False,
    }
    if not payload["ok"]:
        payload["error"] = result.get("error") or "session was not ended"
    return _json(payload)


@server.tool(
    description=(
        "Compact per-namespace briefing for session start. Returns current facts "
        "(semantic memories), top entity names, procedure index, and counts."
    )
)
def memory_worldview(
    namespace: Optional[str] = None,
    facts_cap: int = 12,
    names_cap: int = 12,
) -> str:
    ns = resolve_namespace(namespace)
    with Store(ns) as st:
        wv = st.worldview(facts_cap=facts_cap, names_cap=names_cap)
    return _json(wv)


@server.tool(
    description=(
        "Named how-to procedures (verbatim steps). "
        "action=write: store a named procedure. "
        "action=get: retrieve by name. "
        "action=list: list all active procedures."
    )
)
def memory_procedure(
    action: str = "list",
    name: Optional[str] = None,
    body: Optional[str] = None,
    trigger: Optional[str] = None,
    namespace: Optional[str] = None,
    origin: str = "mcp",
) -> str:
    valid_actions = ("write", "get", "list")
    if action not in valid_actions:
        return _json({"ok": False, "error": f"unknown action '{action}', must be one of: {', '.join(valid_actions)}"})
    ns = resolve_namespace(namespace)
    with Store(ns) as st:
        if action == "write":
            if not name:
                return _json({"ok": False, "error": "name is required for write"})
            if not body:
                return _json({"ok": False, "error": "body is required for write"})
            r = st.procedure_write(name, body, trigger=trigger or "", origin=origin)
            return _json({
                "ok": True,
                "action": "write",
                "memory_id": r.memory_id,
                "event_id": r.event_id,
                "namespace": ns,
                "name": name,
            })
        elif action == "get":
            if not name:
                return _json({"ok": False, "error": "name is required for get"})
            proc = st.procedure_get(name)
            if not proc:
                return _json({"ok": False, "error": f"procedure '{name}' not found"})
            return _json({"ok": True, "action": "get", "namespace": ns, "procedure": proc})
        else:
            procs = st.procedure_list()
            return _json({"ok": True, "action": "list", "namespace": ns, "procedures": procs})


@server.tool(
    description=(
        "Permanently delete a memory and its entire provenance chain: "
        "FTS index, vector embedding, graph relations/entities tied to the event, "
        "and the event itself if no other memories reference it. "
        "This is a hard purge — the data is gone, not just superseded. "
        "Use memory_contradict to supersede (set valid_to) without deleting."
    )
)
def memory_purge(
    memory_id: str,
    namespace: Optional[str] = None,
) -> str:
    ns = resolve_namespace(namespace)
    with Store(ns) as st:
        result = st.purge(memory_id)
    result["namespace"] = ns
    return _json(result)


@server.tool(
    description=(
        "Mark a memory superseded: sets valid_to=now on the old row. "
        "Optionally store a replacement as a new semantic memory."
    )
)
def memory_contradict(
    memory_id: str,
    replacement: Optional[str] = None,
    namespace: Optional[str] = None,
    origin: str = "mcp",
) -> str:
    ns = resolve_namespace(namespace)
    with Store(ns) as st:
        result = st.contradict(memory_id, replacement=replacement, origin=origin)
    result["namespace"] = ns
    return _json(result)


def main() -> None:
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
