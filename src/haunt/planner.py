"""Retrieval planner for a compiled TemporalQuery.

compile() does language only. plan() chooses timeline | recall | union.
When TemporalQuery.clock is unresolved (mixed speech vs occurrence cues),
retrieval applies the compiled [start, end] to event_time OR write_time
and unions the rows. We do not pick a single clock.

Default for temporal=true is union (C): leftover words must not force
recall-only. Non-temporal queries never reach this module's window path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from haunt.recall import Hit, recall
from haunt.store import Store
from haunt.temporal import TemporalQuery, compile
from haunt.util import iso_or_now

Plan = Literal["timeline", "recall", "union"]

# Documented fallback when compile() leaves clock unresolved.
UNRESOLVED_CLOCK_FALLBACK = (
    "unresolved clock: apply [start, end] to event_time OR write_time "
    "(union). Do not guess a single clock."
)


def plan(tq: TemporalQuery) -> Plan:
    """Choose a retrieval strategy from a TemporalQuery.

    Does not inspect leftover words to force recall. v1 default for any
    temporal window is union so timeline can surface rows FTS misses.
    """
    if not tq.temporal:
        return "recall"
    return "union"


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def _clocks(clock: str) -> tuple[str, ...]:
    if clock == "unresolved":
        return ("event_time", "write_time")
    if clock in ("event_time", "write_time"):
        return (clock,)
    raise ValueError(f"clock must be event_time, write_time, or unresolved, got {clock!r}")


def _hits_from_events(store: Store, events: list[dict], *, limit: int) -> list[Hit]:
    hits: list[Hit] = []
    seen: set[str] = set()
    for ev in events:
        row = store.conn.execute(
            """
            SELECT m.id, m.event_id, m.tier, m.content, m.valid_from, m.valid_to,
                   e.role, e.event_time, e.ts, e.tool_name, e.origin
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE m.event_id=? AND m.valid_to IS NULL
            ORDER BY m.created_at DESC
            LIMIT 1
            """,
            (ev["id"],),
        ).fetchone()
        if not row or row["id"] in seen:
            continue
        seen.add(row["id"])
        hits.append(
            Hit(
                memory_id=row["id"],
                event_id=row["event_id"],
                score=0.0,
                tier=row["tier"],
                content=row["content"],
                role=row["role"],
                event_time=row["event_time"],
                ts=row["ts"],
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
                tool_name=row["tool_name"],
                origin=row["origin"],
            )
        )
        if len(hits) >= limit:
            break
    return hits


def run_timeline(
    tq: TemporalQuery,
    store: Store,
    *,
    session_id: str | None = None,
    limit: int = 50,
    clock: str | None = None,
) -> list[Hit]:
    """A: events in [start, end] on the chosen clock(s)."""
    since, until = _iso(tq.start), _iso(tq.end)
    chosen = clock or tq.clock
    merged: dict[str, Hit] = {}
    for clk in _clocks(chosen):
        rows = store.events(
            session_id=session_id,
            since=since,
            until=until,
            clock=clk,
            limit=limit,
        )
        for h in _hits_from_events(store, rows, limit=limit):
            merged.setdefault(h.memory_id, h)
    return list(merged.values())[:limit]


def run_recall(
    tq: TemporalQuery,
    store: Store,
    *,
    as_of: str | None = None,
    tier: str | None = None,
    k: int = 8,
    clock: str | None = None,
    namespace: str | None = None,
) -> list[Hit]:
    """B: recall(cleaned_query, window, clock)."""
    since, until = _iso(tq.start), _iso(tq.end)
    chosen = clock or tq.clock
    merged: dict[str, Hit] = {}
    for clk in _clocks(chosen):
        hits = recall(
            tq.cleaned_query,
            namespace=namespace,
            as_of=as_of,
            since=since,
            until=until,
            clock=clk,
            tier=tier,
            k=k,
            store=store,
        )
        for h in hits:
            prev = merged.get(h.memory_id)
            if prev is None or h.score > prev.score:
                merged[h.memory_id] = h
    ranked = sorted(merged.values(), key=lambda h: h.score, reverse=True)
    return ranked[:k]


def run_union(
    tq: TemporalQuery,
    store: Store,
    *,
    as_of: str | None = None,
    tier: str | None = None,
    k: int = 8,
    clock: str | None = None,
    namespace: str | None = None,
    session_id: str | None = None,
    timeline_limit: int = 50,
) -> list[Hit]:
    """C: union of timeline(window, clock) and windowed recall."""
    by_id: dict[str, Hit] = {}
    for h in run_timeline(
        tq, store, session_id=session_id, limit=timeline_limit, clock=clock
    ):
        by_id[h.memory_id] = h
    for h in run_recall(
        tq, store, as_of=as_of, tier=tier, k=k, clock=clock, namespace=namespace
    ):
        prev = by_id.get(h.memory_id)
        if prev is None or h.score > prev.score:
            by_id[h.memory_id] = h
    ranked = sorted(by_id.values(), key=lambda h: h.score, reverse=True)
    return ranked[: max(k, len(by_id))]


def execute(
    tq: TemporalQuery,
    store: Store,
    *,
    strategy: Plan | None = None,
    as_of: str | None = None,
    tier: str | None = None,
    k: int = 8,
    clock: str | None = None,
    namespace: str | None = None,
    session_id: str | None = None,
) -> list[Hit]:
    chosen = strategy or plan(tq)
    if chosen == "timeline":
        return run_timeline(tq, store, session_id=session_id, limit=max(k, 50), clock=clock)
    if chosen == "recall":
        return run_recall(
            tq, store, as_of=as_of, tier=tier, k=k, clock=clock, namespace=namespace
        )
    if chosen == "union":
        return run_union(
            tq,
            store,
            as_of=as_of,
            tier=tier,
            k=k,
            clock=clock,
            namespace=namespace,
            session_id=session_id,
        )
    raise ValueError(f"unknown plan {chosen!r}")


def planned_recall(
    query: str,
    *,
    now: datetime | None = None,
    namespace: str | None = None,
    as_of: str | None = None,
    since: str | None = None,
    until: str | None = None,
    clock: str | None = None,
    tier: str | None = None,
    k: int = 8,
    store: Store | None = None,
    strategy: Plan | None = None,
) -> list[Hit]:
    """Recall entry used by CLI/MCP.

    Structural invariant: if compile(query).temporal is false, the existing
    recall() path runs unchanged (no compiler window, no compiler clock).
    Caller-supplied since/until/clock stay as they were before this module.
    """
    if clock is not None and clock not in ("event_time", "write_time"):
        raise ValueError(f"clock must be event_time or write_time, got {clock!r}")
    if since:
        iso_or_now(since)
    if until:
        iso_or_now(until)

    tq = compile(query, now)
    if not tq.temporal:
        return recall(
            query,
            namespace=namespace,
            as_of=as_of,
            since=since,
            until=until,
            clock=clock,
            tier=tier,
            k=k,
            store=store,
        )

    # Temporal: do not silently drop the compiled window. Explicit since/until
    # from the caller are an override of the bounds only.
    if since is not None or until is not None:
        tq = TemporalQuery(
            temporal=True,
            cleaned_query=tq.cleaned_query,
            start=datetime.fromisoformat(iso_or_now(since)) if since else tq.start,
            end=datetime.fromisoformat(iso_or_now(until)) if until else tq.end,
            clock=clock or tq.clock,
            granularity=tq.granularity,
            certainty=tq.certainty,
            confidence=tq.confidence,
        )
    elif clock is not None:
        tq = TemporalQuery(
            temporal=True,
            cleaned_query=tq.cleaned_query,
            start=tq.start,
            end=tq.end,
            clock=clock,
            granularity=tq.granularity,
            certainty=tq.certainty,
            confidence=tq.confidence,
        )
    own = store is None
    store = store or Store(namespace or "default")
    try:
        return execute(
            tq,
            store,
            strategy=strategy,
            as_of=as_of,
            tier=tier,
            k=k,
            namespace=namespace,
        )
    finally:
        if own:
            store.close()
