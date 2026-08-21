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
        "Call memory_observe on every user/assistant/tool turn. "
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


@server.tool(description="Hybrid recall over verbatim memories (vec + FTS5 + RRF).")
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
    from haunt.bootstrap import probe_sqlite_vec
    from haunt.embed import state as embed_state
    from haunt.paths import haunt_home

    ns = resolve_namespace(namespace)
    es = embed_state()
    with Store(ns) as st:
        stats = st.stats()
    return _json(
        {
            "haunt_home": str(haunt_home()),
            "sqlite_vec": probe_sqlite_vec(),
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
        sid = st.end_session(session)
    return _json({"ok": True, "namespace": ns, "session_id": sid, "distilled": False})


def main() -> None:
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
