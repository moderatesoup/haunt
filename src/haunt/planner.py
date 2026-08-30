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
import sqlite3
from datetime import datetime
from typing import Literal

from haunt.recall import (
    Hit,
    RecallResult,
    classify_recall_residue,
    execution_metadata,
    recall,
)
from haunt.store import ReadOnlyStore, Store, open_existing_readonly
from haunt.temporal import TemporalQuery, compile
from haunt.util import LIMIT_MAX, clamp_k, iso_or_now, normalize_clock, utc_iso

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


def _recall_order(hit: Hit) -> tuple[int, float, str, str]:
    """Sort key preserving the order recall() returned a hit in.

    recall() owns ranking and an enabled rerank stage replaces RRF order
    with its own, so re-deriving order from ``score`` here would silently
    undo it. With no such stage final_rank is assigned in exactly
    ``(-score, content_hash, memory_id)`` order, so this key reproduces it. A
    Hit built by hand rather than by recall() carries no final_rank and falls
    back to that score order.

    run_ranked merges hits from several recall runs, so two hits routinely
    share a final_rank and the tie is decided here rather than inside recall.
    content_hash leads memory_id for the same reason it does there: the id is
    a fresh uuid4 per write and would re-roll this order on every ingest.
    """
    return (
        hit.final_rank if hit.final_rank is not None else 0,
        -hit.score,
        hit.content_hash or hit.memory_id,
        hit.memory_id,
    )


def _aggregate_execution(
    strategy: str,
    runs: list[tuple[str, list[Hit]]],
) -> dict[str, object] | None:
    """Combine only known stage evidence from one or more clock executions."""
    known = [(clock, execution_metadata(hits)) for clock, hits in runs]
    if not known or any(execution is None for _, execution in known):
        # Tests and external integrations may provide a plain list of Hits.
        # It carries no execution provenance, so do not synthesize any.
        return None

    sources = ("vector", "fts")
    modalities: dict[str, dict[str, str]] = {}
    for source in sources:
        stages = [
            execution["modalities"][source]  # type: ignore[index]
            for _, execution in known
        ]
        if all(stage == stages[0] for stage in stages[1:]):
            modalities[source] = dict(stages[0])
        elif any(stage.get("state") == "candidate" for stage in stages):
            modalities[source] = {
                "state": "candidate",
                "reason": "candidate_in_one_or_more_clocks",
            }
        elif any(stage.get("state") == "ran_not_candidate" for stage in stages):
            modalities[source] = {
                "state": "ran_not_candidate",
                "reason": "no_candidates_in_one_or_more_clocks",
            }
        else:
            modalities[source] = {
                "state": "not_run",
                "reason": "not_run_in_all_clocks",
            }

    pending_by_run = {
        clock: execution.get("pending_embedding_jobs")
        for clock, execution in known
        if execution.get("pending_embedding_jobs") is not None
    }
    pending_values = list(pending_by_run.values())
    if not pending_values:
        pending: object | None = None
    elif all(value == pending_values[0] for value in pending_values[1:]):
        pending = pending_values[0]
    else:
        # A concurrent writer can enqueue between two independently planned
        # legs. Do not hide that observation behind the first leg's count.
        pending = {
            "state": "observed_not_drained",
            "count": None,
            "by_run": pending_by_run,
        }

    def combined(field: str) -> object | None:
        values = [execution.get(field) for _, execution in known]
        return values[0] if all(value == values[0] for value in values[1:]) else "mixed"

    return {
        "version": 1,
        "strategy": strategy,
        "modalities": modalities,
        "read_only": all(bool(execution.get("read_only")) for _, execution in known),
        "maintenance_performed": False,
        "pending_embedding_jobs": pending,
        "residue_filter": combined("residue_filter"),
        "residue_filter_source": combined("residue_filter_source"),
        "recall_class_capability": combined("recall_class_capability"),
        "clock_runs": [
            {"clock": clock, "modalities": execution["modalities"]}
            for clock, execution in known
        ],
    }


def _hits_from_events(
    store: Store,
    events: list[dict],
    *,
    limit: int,
    filter_context: dict[str, object],
) -> list[Hit]:
    hits: list[Hit] = []
    seen: set[str] = set()
    recall_class_select = (
        "e.recall_class AS recall_class"
        if bool(getattr(store, "recall_class_available", False))
        else "NULL AS recall_class"
    )
    for ev in events:
        row = store.conn.execute(
            f"""
            SELECT m.id, m.event_id, m.tier, m.content, m.valid_from, m.valid_to,
                   e.role, e.event_time, e.ts, e.tool_name, e.tool_input,
                   e.tool_output, e.origin, {recall_class_select}
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE m.event_id=? AND m.valid_to IS NULL
            ORDER BY m.created_at DESC, m.rowid DESC, m.id ASC
            LIMIT 1
            """,
            (ev["id"],),
        ).fetchone()
        if not row or row["id"] in seen:
            continue
        seen.add(row["id"])
        raw_tool, classification_source = classify_recall_residue(
            recall_class=row["recall_class"],
            role=row["role"],
            tool_name=row["tool_name"],
            tool_input=row["tool_input"],
            tool_output=row["tool_output"],
        )
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
                filter_context=filter_context,
                vector_stage={"state": "not_run", "reason": "timeline_time_order"},
                fts_stage={"state": "not_run", "reason": "timeline_time_order"},
                recall_class=row["recall_class"],
                classification_source=classification_source,
                raw_tool_structure=raw_tool,
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

    Direct callers share the public ``clamp_k`` contract, so ``limit`` is
    always in [1, K_MAX]. Fetch enough bounded event pages to fill that many
    *current* memories. Superseded rows (valid_to IS NOT NULL) are skipped by
    ``_hits_from_events``; a short prefix of recent superseded events must not
    starve the requested count.
    """
    limit = clamp_k(limit)
    since, until = _iso(tq.start), _iso(tq.end)
    chosen = clock or tq.clock
    merged: dict[str, Hit] = {}
    for clk in _clocks(chosen):
        filter_context: dict[str, object] = {
            "validity": "current",
            "as_of": None,
            "clock": clk,
            "since": since,
            "until": until,
            "tier": None,
            "include_untrusted": None,
            "include_residue": None,
            "residue_filter": "not_applicable",
            "residue_filter_source": "not_applicable",
            "recall_class_capability": "not_applicable",
            "maintenance_performed": False,
            "session_id": session_id,
        }
        # Page through events at the store's hard page ceiling. ``limit`` is
        # already clamped, but retaining this min keeps pagination correct if
        # the two public bounds ever diverge.
        batch = min(limit, LIMIT_MAX)
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
            for h in _hits_from_events(
                store, rows, limit=limit, filter_context=filter_context
            ):
                merged.setdefault(h.memory_id, h)
            if len(merged) >= limit or len(rows) < batch:
                break
            nxt = offset + len(rows)
            if nxt <= offset:
                break
            offset = nxt
    # Preserve chronological order. Only exact values on the selected clock
    # fall back to the content key; never sort timeline hits by it alone. The
    # memory id is NOT stable across ingests -- it is a fresh uuid4 per write
    # -- so content_hash leads and the id only settles byte-identical content.
    hits = list(merged.values())
    hits.sort(key=lambda hit: (hit.content_hash or hit.memory_id, hit.memory_id))
    time_attr = "ts" if _clocks(chosen)[0] == "storage_time" else "event_time"
    hits.sort(key=lambda hit: getattr(hit, time_attr) or "", reverse=True)
    hits = hits[:limit]
    references = store.recall_references_many([hit.memory_id for hit in hits])
    for hit in hits:
        hit.references = references.get(hit.memory_id)
    for final_rank, hit in enumerate(hits, start=1):
        hit.final_rank = final_rank
    try:
        pending_jobs = int(
            store.conn.execute("SELECT COUNT(*) FROM embedding_jobs").fetchone()[0]
        )
    except sqlite3.Error:
        pending_jobs = None
    return RecallResult(
        hits,
        execution={
            "version": 1,
            "strategy": "timeline",
            "modalities": {
                "vector": {"state": "not_run", "reason": "timeline_time_order"},
                "fts": {"state": "not_run", "reason": "timeline_time_order"},
            },
            "read_only": bool(getattr(store, "read_only", False)),
            "maintenance_performed": False,
            "pending_embedding_jobs": {
                "state": "observed_not_drained",
                "count": pending_jobs,
            },
            "residue_filter": "not_applicable",
            "residue_filter_source": "not_applicable",
            "recall_class_capability": "not_applicable",
        },
    )


def run_recall(
    tq: TemporalQuery,
    store: Store,
    *,
    as_of: str | None = None,
    tier: str | None = None,
    k: int = 8,
    clock: str | None = None,
    namespace: str | None = None,
    include_residue: bool | None = None,
    include_untrusted: bool | None = None,
) -> list[Hit]:
    """B: recall(cleaned_query, window, clock)."""
    k = clamp_k(k)
    since, until = _iso(tq.start), _iso(tq.end)
    chosen = clock or tq.clock
    merged: dict[str, Hit] = {}
    recall_runs: list[tuple[str, list[Hit]]] = []
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
            include_residue=include_residue,
            include_untrusted=include_untrusted,
        )
        recall_runs.append((clk, hits))
        for h in hits:
            prev = merged.get(h.memory_id)
            if prev is None or h.score > prev.score:
                merged[h.memory_id] = h
    ranked = sorted(merged.values(), key=_recall_order)
    hits = ranked[:k]
    for final_rank, hit in enumerate(hits, start=1):
        hit.final_rank = final_rank
    execution = _aggregate_execution("recall", recall_runs)
    if execution is None:
        return hits
    return RecallResult(hits, execution=execution)


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
    include_residue: bool | None = None,
    include_untrusted: bool | None = None,
) -> list[Hit]:
    """C: union of timeline(window, clock) and windowed recall."""
    k = clamp_k(k)
    by_id: dict[str, Hit] = {}
    timeline = run_timeline(
        tq, store, session_id=session_id, limit=timeline_limit, clock=clock
    )
    for h in timeline:
        by_id[h.memory_id] = h
    recalled = run_recall(
        tq,
        store,
        as_of=as_of,
        tier=tier,
        k=k,
        clock=clock,
        namespace=namespace,
        include_residue=include_residue,
        include_untrusted=include_untrusted,
    )
    for h in recalled:
        prev = by_id.get(h.memory_id)
        if prev is None or h.score > prev.score:
            by_id[h.memory_id] = h
    ranked = sorted(
        (hit for hit in by_id.values() if hit.vec_rank is not None or hit.fts_rank is not None),
        key=_recall_order,
    )
    # Unranked timeline rows remain in their already chronological order.
    timeline_hits = [
        hit
        for hit in timeline
        if by_id.get(hit.memory_id) is hit
        and hit.vec_rank is None
        and hit.fts_rank is None
    ]
    hits = (ranked + timeline_hits)[:k]
    for final_rank, hit in enumerate(hits, start=1):
        hit.final_rank = final_rank
    timeline_execution = execution_metadata(timeline)
    recall_execution = execution_metadata(recalled)
    if timeline_execution is None or recall_execution is None:
        return hits
    aggregate = _aggregate_execution(
        "union", [("timeline", timeline), ("recall", recalled)]
    )
    if aggregate is None:
        return hits
    execution = {
        "version": 1,
        "strategy": "union",
        "modalities": aggregate["modalities"],
        "read_only": aggregate["read_only"],
        "maintenance_performed": False,
        "pending_embedding_jobs": aggregate["pending_embedding_jobs"],
        # The legs intentionally differ: ranked recall filters residue while
        # a timeline is an audit/navigation view where it is not applicable.
        "residue_filter": aggregate["residue_filter"],
        "residue_filter_source": aggregate["residue_filter_source"],
        "recall_class_capability": aggregate["recall_class_capability"],
        "components": {
            "timeline": timeline_execution,
            "recall": recall_execution,
        },
    }
    return RecallResult(hits, execution=execution)


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
    include_residue: bool | None = None,
    include_untrusted: bool | None = None,
) -> list[Hit]:
    k = clamp_k(k)
    chosen = strategy or plan(tq)
    if chosen == "timeline":
        return run_timeline(tq, store, session_id=session_id, limit=k, clock=clock)
    if chosen == "recall":
        return run_recall(
            tq,
            store,
            as_of=as_of,
            tier=tier,
            k=k,
            clock=clock,
            namespace=namespace,
            include_residue=include_residue,
            include_untrusted=include_untrusted,
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
            include_residue=include_residue,
            include_untrusted=include_untrusted,
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
    store: Store | ReadOnlyStore | None = None,
    strategy: Plan | None = None,
    include_residue: bool | None = None,
    include_untrusted: bool | None = None,
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
            include_residue=include_residue,
            include_untrusted=include_untrusted,
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
    store = store or open_existing_readonly(namespace or "default")
    try:
        return execute(
            tq,
            store,
            strategy=strategy,
            as_of=as_of,
            tier=tier,
            k=k,
            namespace=namespace,
            include_residue=include_residue,
            include_untrusted=include_untrusted,
        )
    finally:
        if own:
            store.close()
