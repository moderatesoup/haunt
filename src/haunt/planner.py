"""Retrieval planner for a compiled TemporalQuery.

compile() does language only. plan() chooses timeline or recall.
union remains available for tests/experiments via execute(strategy="union")
and run_union(); it is not the default.

When TemporalQuery.clock is unresolved (mixed speech vs occurrence cues),
retrieval applies the compiled [start, end] to event_time only. Unresolved
must not apply a storage_time / events.ts filter.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from haunt.recall import Hit, recall
from haunt.store import Store, open_existing
from haunt.temporal import TemporalQuery, compile
from haunt.util import clamp_k, iso_or_now, normalize_clock, utc_iso

Plan = Literal["timeline", "recall", "union"]

# Documented fallback when compile() leaves clock unresolved.
UNRESOLVED_CLOCK_FALLBACK = (
    "unresolved clock: apply [start, end] to event_time only. "
    "Do not apply a storage_time / events.ts filter."
)

# Leftover words that are not a topic. Pure "what happened two weeks ago"
# must plan as timeline, not recall.
_RESIDUE_STOP = frozenset(
    {
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "which",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "was",
        "were",
        "is",
        "are",
        "am",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "i",
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "you",
        "your",
        "they",
        "them",
        "their",
        "he",
        "she",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "the",
        "a",
        "an",
        "about",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "with",
        "from",
        "by",
        "into",
        "over",
        "during",
        "as",
        "please",
        "just",
        "also",
        "any",
        "anything",
        "something",
        "things",
        "thing",
        "there",
        "mention",
        "mentioned",
        "mentioning",
        "say",
        "said",
        "tell",
        "told",
        "telling",
        "talk",
        "talking",
        "discuss",
        "discussed",
        "discussing",
        "happen",
        "happened",
        "happening",
        "occur",
        "occurred",
        "occurring",
        "went",
        "going",
        "go",
        "haunt",
        "ingest",
        "ingested",
        "ingesting",
        "store",
        "stored",
    }
)
_RESIDUE_TOKEN = re.compile(r"[A-Za-z0-9_./+-]+")


def has_topical_residue(cleaned_query: str) -> bool:
    """True when compile leftover contains a topic, not just function words."""
    for tok in _RESIDUE_TOKEN.findall(cleaned_query or ""):
        if tok.lower() not in _RESIDUE_STOP:
            return True
    return False


def plan(tq: TemporalQuery) -> Plan:
    """Choose a retrieval strategy from a TemporalQuery.

    Topical residue after compile → recall (cleaned query + event_time window).
    Bare temporal ("what happened two weeks ago") → timeline.
    union is not returned by plan().
    """
    if not tq.temporal:
        return "recall"
    if has_topical_residue(tq.cleaned_query):
        return "recall"
    return "timeline"


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return utc_iso(dt)


def _clocks(clock: str) -> tuple[str, ...]:
    c = normalize_clock(clock, allow_unresolved=True)
    if c == "unresolved":
        return ("event_time",)
    return (c,)


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
            ORDER BY m.created_at DESC, m.rowid DESC
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
    """A: events in [start, end] on the chosen clock(s).

    Fetch enough events to fill ``limit`` *current* memories. Superseded
    rows (valid_to IS NOT NULL) are skipped by ``_hits_from_events``; a
    short prefix of recent superseded events must not starve k.
    """
    since, until = _iso(tq.start), _iso(tq.end)
    chosen = clock or tq.clock
    merged: dict[str, Hit] = {}
    for clk in _clocks(chosen):
        # Page through events. store.events clamps LIMIT to 100; doubling
        # fetch_n past that made len(rows) < fetch_n and starved refill.
        batch = max(int(limit), 1)
        offset = 0
        while True:
            rows = store.events(
                session_id=session_id,
                since=since,
                until=until,
                clock=clk,
                limit=batch,
                offset=offset,
            )
            for h in _hits_from_events(store, rows, limit=limit):
                merged.setdefault(h.memory_id, h)
            if len(merged) >= limit or len(rows) < batch:
                break
            nxt = offset + len(rows)
            if nxt <= offset:
                break
            offset = nxt
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
    k = clamp_k(k)
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
    k = clamp_k(k)
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
    return ranked[:k]


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
    k = clamp_k(k)
    chosen = strategy or plan(tq)
    if chosen == "timeline":
        return run_timeline(tq, store, session_id=session_id, limit=k, clock=clock)
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
    k = clamp_k(k)
    if clock is not None:
        normalize_clock(clock)
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
    # from the caller are an override of the bounds only. write_time is
    # canonicalized to storage_time (ingest time, not source time).
    chosen_clock = normalize_clock(clock) if clock is not None else tq.clock
    if since is not None or until is not None:
        tq = TemporalQuery(
            temporal=True,
            cleaned_query=tq.cleaned_query,
            start=datetime.fromisoformat(iso_or_now(since)) if since else tq.start,
            end=datetime.fromisoformat(iso_or_now(until)) if until else tq.end,
            clock=chosen_clock,
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
            clock=chosen_clock,
            granularity=tq.granularity,
            certainty=tq.certainty,
            confidence=tq.confidence,
        )
    own = store is None
    store = store or open_existing(namespace or "default")
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
