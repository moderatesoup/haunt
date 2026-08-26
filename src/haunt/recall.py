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
BACKEND_ERROR_CODE = "retrieval_backend_error"


class RetrievalBackendError(RuntimeError):
    """A sqlite/vector retrieval failure that machine surfaces can classify.

    Python callers still receive an exception (with the sqlite error chained),
    while CLI/MCP/dashboard adapters can return one stable error code.
    """

    code = BACKEND_ERROR_CODE


def is_retrieval_backend_error(exc: BaseException) -> bool:
    """True for a retrieval backend failure, including an unwrapped DB error."""
    return isinstance(exc, (RetrievalBackendError, sqlite3.Error)) or (
        isinstance(exc, RuntimeError)
        and str(exc).startswith("sqlite-vec failed to load:")
    )


class RecallResult(list["Hit"]):
    """List-compatible result with optional, versioned execution evidence."""

    def __init__(
        self,
        hits: list["Hit"],
        *,
        modalities: dict[str, dict[str, str]] | None = None,
        strategy: str = "recall",
        execution: dict[str, Any] | None = None,
    ):
        super().__init__(hits)
        if execution is None and modalities is not None:
            execution = {
                "version": 1,
                "strategy": strategy,
                "modalities": modalities,
            }
        self.execution = execution
        # Kept as a convenient compatibility alias for internal callers.
        self.modalities = (
            execution.get("modalities") if execution is not None else None
        )


def execution_metadata(hits: object) -> dict[str, Any] | None:
    """Return structured execution evidence, never inventing it for old lists."""
    execution = getattr(hits, "execution", None)
    if not isinstance(execution, dict) or execution.get("version") != 1:
        return None
    modalities = execution.get("modalities")
    if not isinstance(modalities, dict):
        return None
    return execution


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
    vec_metric: str | None = None
    fts_rank_raw: float | None = None
    filter_context: dict[str, Any] | None = None
    final_rank: int | None = None
    vector_stage: dict[str, str] | None = None
    fts_stage: dict[str, str] | None = None

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
        vector = _modality_explanation(
            self.vector_stage,
            candidate=self.vec_rank is not None,
            candidate_reason="returned_vector_candidate",
            fields={
                "rank": self.vec_rank,
                "distance": self.vec_distance,
                "metric": self.vec_metric,
                "lower_is_better": True if self.vec_rank is not None else None,
            },
        )
        fts = _modality_explanation(
            self.fts_stage,
            candidate=self.fts_rank is not None,
            candidate_reason="returned_fts_candidate",
            fields={
                "rank": self.fts_rank,
                "raw_score": self.fts_rank_raw,
                "metric": "fts5_bm25" if self.fts_rank is not None else None,
                "lower_is_better": True if self.fts_rank is not None else None,
            },
        )
        contributions: list[dict[str, Any]] = []
        for source, rank in (("vector", self.vec_rank), ("fts", self.fts_rank)):
            if rank is not None:
                contributions.append(
                    {
                        "source": source,
                        "rank": rank,
                        "value": 1.0 / (RRF_K + rank),
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
            # Keep the unrounded contributions and their sum together so a
            # consumer can reproduce this serialized score without a rounding
            # discrepancy. The legacy top-level ``score`` remains rounded.
            "rrf_score": (
                sum(item["value"] for item in contributions) if is_rrf else None
            ),
            "rrf_k": RRF_K if is_rrf else None,
            "rrf_contributions": contributions,
            "vector": vector,
            "fts": fts,
            "filters": self.filter_context,
            "references": {
                "correction_lineage": None,
                "correction_lineage_status": "unavailable_legacy",
                "provenance": None,
                "provenance_status": "legacy_unstructured",
            },
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


def _stage(state: str, reason: str) -> dict[str, str]:
    """Build one of the explicit per-modality execution states."""
    return {"state": state, "reason": reason}


def _modality_explanation(
    stage: dict[str, str] | None,
    *,
    candidate: bool,
    candidate_reason: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """Expose a candidate, a run without this candidate, or a non-run.

    Legacy callers may construct ``Hit`` directly. Those synthetic hits retain
    their evidence when present; otherwise they explicitly say that execution
    provenance was not structured in the legacy object.
    """
    if candidate:
        state = _stage("candidate", candidate_reason)
    elif stage is None:
        state = _stage("not_run", "legacy_unstructured")
    elif stage["state"] == "candidate":
        # A different candidate was returned by this stage, but not this hit.
        state = _stage("ran_not_candidate", "candidate_not_returned_for_hit")
    else:
        state = stage
    return {**state, **fields}


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
        ORDER BY f.rank, f.id
        LIMIT ?
    """
    rows = conn.execute(sql, [match, *params, limit]).fetchall()
    return [(r["mid"], i + 1, float(r["rnk"])) for i, r in enumerate(rows)]


def _vec_hits(
    store: Store,
    query_vec: list[float],
    where: str,
    params: list[Any],
    limit: int,
) -> list[tuple[str, int, float, str]]:
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
                ORDER BY distance
            """
            # vec0 accepts KNN queries ordered by distance alone. Sort the
            # returned candidate set in Python to settle exact distance ties;
            # do not treat a malformed native KNN query as an L2 fallback.
            rows = conn.execute(sql, [blob, limit, *params]).fetchall()
            candidates = sorted(
                ((r["mid"], float(r["dist"])) for r in rows),
                key=lambda item: (item[1], item[0]),
            )
            return [
                (mid, i + 1, distance, "cosine_distance")
                for i, (mid, distance) in enumerate(candidates)
            ]
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
    scored.sort(key=lambda x: (x[1], x[0]))
    return [
        (mid, i + 1, dist, "l2_distance")
        for i, (mid, dist) in enumerate(scored[:limit])
    ]


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
        match = _fts_match_query(query)
        if match is None:
            fts = []
            fts_execution = _stage("not_run", "query_has_no_fts_tokens")
        else:
            fts = _fts_hits(store.conn, query, where, params, CANDIDATES)
            fts_execution = _stage(
                "ran_not_candidate" if not fts else "candidate",
                "no_fts_candidates" if not fts else "returned_fts_candidates",
            )
        vec: list[tuple[str, int, float, str]] = []
        if not use_vectors:
            vector_execution = _stage("not_run", "disabled_by_caller")
        elif not embed_available():
            vector_execution = _stage("not_run", "embedding_unavailable")
        else:
            qv = embed_one(query)
            if qv:
                vec = _vec_hits(store, qv, where, params, CANDIDATES)
                vector_execution = _stage(
                    "ran_not_candidate" if not vec else "candidate",
                    "no_vector_candidates"
                    if not vec
                    else "returned_vector_candidates",
                )
            else:
                # No candidate search can run without a query vector. Keep
                # ran_not_candidate for the branch that actually calls
                # _vec_hits above.
                vector_execution = _stage("not_run", "query_embedding_empty")

        rrf: dict[str, float] = {}
        vec_rank: dict[str, tuple[int, float, str]] = {}
        fts_rank: dict[str, tuple[int, float]] = {}
        for mid, rank, raw, metric in vec:
            rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (RRF_K + rank)
            vec_rank[mid] = (rank, raw, metric)
        for mid, rank, raw in fts:
            rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (RRF_K + rank)
            fts_rank[mid] = (rank, raw)

        ranked = sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))[: int(k)]
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
                    vec_metric=vr[2] if vr else None,
                    fts_rank_raw=fr[1] if fr else None,
                    filter_context=filter_context,
                    final_rank=final_rank,
                    vector_stage=vector_execution,
                    fts_stage=fts_execution,
                )
            )
        return RecallResult(
            hits,
            modalities={"vector": vector_execution, "fts": fts_execution},
        )
    except sqlite3.Error as exc:
        raise RetrievalBackendError(str(exc)) from exc
    finally:
        if own:
            store.close()
