"""Hybrid recall: sqlite-vec ANN + FTS5 BM25 + Reciprocal Rank Fusion.

No reader-LLM. Optional cross-encoder is not wired (off).
"""

from __future__ import annotations

import math
import re
import sqlite3
import struct
from dataclasses import dataclass
from typing import Any

import sqlite_vec

from haunt.embed import available as embed_available
from haunt.embed import embed_one
from haunt.provenance import json_safe_sqlite
from haunt.store import Store, open_existing
from haunt.util import clamp_k, clock_sql_column, iso_or_now, normalize_clock, snippet

# Match FTS5 unicode61 word characters (letters/digits) plus the same
# extras the previous ASCII class allowed. ASCII-only [A-Za-z0-9_...]
# drops non-Latin queries while the index still tokenizes them.
_FTS_TOKEN = re.compile(r"[\w./+-]+", re.UNICODE)

RRF_K = 60
CANDIDATES = 40


@dataclass
class Hit:
    memory_id: str
    event_id: str
    score: float
    tier: str
    content: str
    role: str
    event_time: str
    valid_from: str
    valid_to: str | None
    tool_name: str | None
    ts: str | None = None
    origin: str | None = None
    vec_rank: int | None = None
    fts_rank: int | None = None
    vec_distance: float | None = None
    fts_rank_raw: float | None = None
    filter_context: dict[str, Any] | None = None
    final_rank: int | None = None

    @property
    def trusted(self) -> bool:
        """False for raw tool I/O. Recalled text is always data, not authority."""
        return self.role != "tool" and self.tool_name is None

    @property
    def trust_reason(self) -> str:
        return "ordinary-memory" if self.trusted else "untrusted-tool-io"

    def as_dict(self) -> dict[str, Any]:
        """Serialize a hit without changing the established result fields.

        ``score`` remains the legacy RRF score.  ``explanation`` is additive
        provenance for callers that need to understand how that rank signal was
        produced; it deliberately does not present retrieval as confidence.
        """
        vector = (
            {"rank": self.vec_rank, "distance": self.vec_distance}
            if self.vec_rank is not None
            else None
        )
        fts = (
            {"rank": self.fts_rank, "raw_score": self.fts_rank_raw}
            if self.fts_rank is not None
            else None
        )
        contributions: list[dict[str, Any]] = []
        for source, rank in (("vector", self.vec_rank), ("fts", self.fts_rank)):
            if rank is not None:
                contributions.append(
                    {
                        "source": source,
                        "rank": rank,
                        "value": round(1.0 / (RRF_K + rank), 6),
                    }
                )
        is_rrf = bool(contributions)
        explanation = {
            "version": 1,
            "retrieval_method": (
                "hybrid_rrf" if len(contributions) == 2
                else f"{contributions[0]['source']}_rrf" if contributions
                else "timeline"
            ),
            "score_semantics": (
                "rrf_rank_signal_not_confidence" if is_rrf else "not_ranked"
            ),
            "final_rank": self.final_rank,
            "rrf_score": round(self.score, 6) if is_rrf else None,
            "rrf_k": RRF_K if is_rrf else None,
            "rrf_contributions": contributions,
            "vector": vector,
            "fts": fts,
            "filters": self.filter_context,
            "trust": {
                "trusted": self.trusted,
                "reason": self.trust_reason,
            },
        }
        return json_safe_sqlite({
            "memory_id": self.memory_id,
            "event_id": self.event_id,
            "score": round(self.score, 6),
            "tier": self.tier,
            "content": self.content,
            "snippet": snippet(self.content, 200),
            "role": self.role,
            "origin": self.origin,
            "event_time": self.event_time,
            "ts": self.ts,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "tool_name": self.tool_name,
            "trusted": self.trusted,
            "trust_reason": self.trust_reason,
            "vec_rank": self.vec_rank,
            "fts_rank": self.fts_rank,
            "explanation": explanation,
        })


def _fts_match_query(q: str) -> str | None:
    tokens = _FTS_TOKEN.findall(q)
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    parts = []
    for t in tokens[:24]:
        esc = t.replace('"', '""')
        parts.append(f'"{esc}"')
    return " OR ".join(parts)


def _filters(
    as_of: str | None,
    since: str | None,
    until: str | None,
    tier: str | None,
    clock: str | None = None,
    include_untrusted: bool = True,
) -> tuple[str, list[Any]]:
    clauses = ["1=1"]
    params: list[Any] = []
    if as_of:
        t = iso_or_now(as_of)
        clauses.append("m.valid_from <= ? AND (m.valid_to IS NULL OR m.valid_to > ?)")
        params.extend([t, t])
    else:
        # Current slice: contradict/supersede writes valid_to, so hide those
        # rows unless the caller asked for an explicit as_of snapshot.
        clauses.append("m.valid_to IS NULL")
    col = clock_sql_column(clock, qualified=True)
    if since:
        clauses.append(f"{col} >= ?")
        params.append(iso_or_now(since))
    if until:
        clauses.append(f"{col} <= ?")
        params.append(iso_or_now(until))
    if tier:
        clauses.append("m.tier = ?")
        params.append(tier)
    if not include_untrusted:
        clauses.append("e.role != 'tool' AND e.tool_name IS NULL")
    return " AND ".join(clauses), params


def _filter_context(
    *,
    as_of: str | None,
    since: str | None,
    until: str | None,
    tier: str | None,
    clock: str | None,
    include_untrusted: bool,
) -> dict[str, Any]:
    """Return the resolved, user-visible filters applied by recall.

    This is intentionally limited to inputs that the retrieval path actually
    applies.  In particular, current recall is represented as ``valid_to IS
    NULL`` semantically, rather than pretending there is a relevance or
    confidence threshold.
    """
    return {
        "validity": "as_of" if as_of else "current",
        "as_of": iso_or_now(as_of) if as_of else None,
        "clock": normalize_clock(clock),
        "since": iso_or_now(since) if since else None,
        "until": iso_or_now(until) if until else None,
        "tier": tier,
        "include_untrusted": include_untrusted,
    }


def _deserialize(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _fts_hits(
    conn: sqlite3.Connection,
    query: str,
    where: str,
    params: list[Any],
    limit: int,
) -> list[tuple[str, int, float]]:
    match = _fts_match_query(query)
    if not match:
        return []
    sql = f"""
        SELECT f.id AS mid, f.rank AS rnk
        FROM memories_fts f
        JOIN memories m ON m.id = f.id
        JOIN events e ON e.id = m.event_id
        WHERE memories_fts MATCH ?
          AND {where}
        ORDER BY f.rank
        LIMIT ?
    """
    try:
        rows = conn.execute(sql, [match, *params, limit]).fetchall()
    except sqlite3.Error:
        return []
    return [(r["mid"], i + 1, float(r["rnk"])) for i, r in enumerate(rows)]


def _vec_hits(
    store: Store,
    query_vec: list[float],
    where: str,
    params: list[Any],
    limit: int,
) -> list[tuple[str, int, float]]:
    blob = sqlite_vec.serialize_float32(query_vec)
    conn = store.conn
    if store.vec_ok():
        has = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_memories'"
        ).fetchone()
        if has:
            sql = f"""
                SELECT v.id AS mid, v.distance AS dist
                FROM vec_memories v
                JOIN memories m ON m.id = v.id
                JOIN events e ON e.id = m.event_id
                WHERE v.embedding MATCH ?
                  AND k = ?
                  AND {where}
                ORDER BY v.distance
            """
            try:
                rows = conn.execute(sql, [blob, limit, *params]).fetchall()
                return [(r["mid"], i + 1, float(r["dist"])) for i, r in enumerate(rows)]
            except sqlite3.Error:
                pass
    sql = f"""
        SELECT m.id AS mid, m.embedding
        FROM memories m
        JOIN events e ON e.id = m.event_id
        WHERE m.embedding IS NOT NULL AND {where}
    """
    scored: list[tuple[str, float]] = []
    for r in conn.execute(sql, params):
        vec = _deserialize(r["embedding"])
        if len(vec) != len(query_vec):
            continue
        scored.append((r["mid"], _l2(query_vec, vec)))
    scored.sort(key=lambda x: x[1])
    return [(mid, i + 1, dist) for i, (mid, dist) in enumerate(scored[:limit])]


def recall(
    query: str,
    *,
    namespace: str | None = None,
    as_of: str | None = None,
    since: str | None = None,
    until: str | None = None,
    clock: str | None = None,
    tier: str | None = None,
    k: int = 8,
    store: Store | None = None,
    include_untrusted: bool = True,
    use_vectors: bool = True,
) -> list[Hit]:
    k = clamp_k(k)
    own = store is None
    store = store or open_existing(namespace or "default")
    try:
        if use_vectors:
            store.ensure_current_embeddings()
            store.process_embedding_jobs(limit=64)
        where, params = _filters(
            as_of,
            since,
            until,
            tier,
            clock,
            include_untrusted=include_untrusted,
        )
        filter_context = _filter_context(
            as_of=as_of,
            since=since,
            until=until,
            tier=tier,
            clock=clock,
            include_untrusted=include_untrusted,
        )
        fts = _fts_hits(store.conn, query, where, params, CANDIDATES)
        vec: list[tuple[str, int, float]] = []
        if use_vectors and embed_available():
            qv = embed_one(query)
            if qv:
                vec = _vec_hits(store, qv, where, params, CANDIDATES)

        rrf: dict[str, float] = {}
        vec_rank: dict[str, tuple[int, float]] = {}
        fts_rank: dict[str, tuple[int, float]] = {}
        for mid, rank, raw in vec:
            rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (RRF_K + rank)
            vec_rank[mid] = (rank, raw)
        for mid, rank, raw in fts:
            rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (RRF_K + rank)
            fts_rank[mid] = (rank, raw)

        ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[: int(k)]
        hits: list[Hit] = []
        for final_rank, (mid, score) in enumerate(ranked, start=1):
            row = store.conn.execute(
                """
                SELECT m.id, m.event_id, m.tier, m.content, m.valid_from, m.valid_to,
                       e.role, e.event_time, e.ts, e.tool_name, e.origin
                FROM memories m
                JOIN events e ON e.id = m.event_id
                WHERE m.id=?
                """,
                (mid,),
            ).fetchone()
            if not row:
                continue
            vr = vec_rank.get(mid)
            fr = fts_rank.get(mid)
            hits.append(
                Hit(
                    memory_id=row["id"],
                    event_id=row["event_id"],
                    score=score,
                    tier=row["tier"],
                    content=row["content"],
                    role=row["role"],
                    event_time=row["event_time"],
                    ts=row["ts"],
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                    tool_name=row["tool_name"],
                    origin=row["origin"],
                    vec_rank=vr[0] if vr else None,
                    fts_rank=fr[0] if fr else None,
                    vec_distance=vr[1] if vr else None,
                    fts_rank_raw=fr[1] if fr else None,
                    filter_context=filter_context,
                    final_rank=final_rank,
                )
            )
        return hits
    finally:
        if own:
            store.close()
