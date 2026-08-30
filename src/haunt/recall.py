"""Hybrid recall: sqlite-vec ANN + FTS5 BM25 + Reciprocal Rank Fusion.

No reader-LLM. Optional cross-encoder is not wired (off). An optional
lexical diversity rerank (haunt.rerank) sits between fusion and the
returned k; it is off unless HAUNT_RERANK_ENABLED is set.
"""

from __future__ import annotations

import math
import re
import sqlite3
import struct
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import sqlite_vec

from haunt.embed import available as embed_available
from haunt.embed import embed_one
from haunt.embed import offline as embed_offline
from haunt.provenance import json_safe_sqlite
from haunt.store import ReadOnlyStore, Store, open_existing_readonly
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
        self.execution = deepcopy(execution) if execution is not None else None
        # Kept as a convenient compatibility alias for internal callers.
        self.modalities = (
            deepcopy(self.execution.get("modalities"))
            if self.execution is not None
            else None
        )


def execution_metadata(hits: object) -> dict[str, Any] | None:
    """Return structured execution evidence, never inventing it for old lists."""
    execution = getattr(hits, "execution", None)
    if not isinstance(execution, dict) or execution.get("version") != 1:
        return None
    modalities = execution.get("modalities")
    if not isinstance(modalities, dict):
        return None
    return deepcopy(execution)


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
    # Set only when a post-fusion stage reordered the fused list, so
    # final_rank no longer equals the RRF rank the hit came from. None --
    # the default -- means final_rank IS the fusion rank.
    rerank_stage: dict[str, Any] | None = None
    references: dict[str, Any] | None = None
    recall_class: str | None = None
    classification_source: str = "legacy_unknown"
    # Keep structured tool detection internal: callers still receive only the
    # established public tool_name field, while trust cannot be bypassed by a
    # tool_input-only or tool_output-only event.
    raw_tool_structure: bool | None = None

    @property
    def trusted(self) -> bool:
        """False for raw tool I/O. Recalled text is always data, not authority."""
        raw_tool = (
            self.raw_tool_structure
            if self.raw_tool_structure is not None
            else (
                self.recall_class == "tool"
                or self.role == "tool"
                or self.tool_name is not None
            )
        )
        return not raw_tool

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
            # Only when a rerank stage moved this hit: without it final_rank
            # is the fusion rank and repeating it would be noise.
            **(
                {"rrf_rank": self.rerank_stage["rrf_rank"]}
                if self.rerank_stage is not None
                else {}
            ),
            # Keep the unrounded contributions and their sum together so a
            # consumer can reproduce this serialized score without a rounding
            # discrepancy. The legacy top-level ``score`` remains rounded.
            "rrf_score": (
                sum(item["value"] for item in contributions) if is_rrf else None
            ),
            "rrf_k": RRF_K if is_rrf else None,
            "rrf_contributions": contributions,
            "ordering": _ordering_explanation(self, is_rrf=is_rrf),
            "vector": vector,
            "fts": fts,
            "filters": self.filter_context,
            "references": self.references
            if self.references is not None
            else {
                "correction_lineage": None,
                "correction_lineage_status": "unavailable_legacy",
                "provenance": None,
                "provenance_status": "legacy_unstructured",
            },
            "trust": {
                "trusted": self.trusted,
                "reason": self.trust_reason,
            },
            "residue": {
                "recall_class": self.recall_class,
                "classification_source": self.classification_source,
                "filter": (
                    None
                    if self.filter_context is None
                    else self.filter_context.get("residue_filter")
                ),
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
            "recall_class": self.recall_class,
            "classification_source": self.classification_source,
            "trusted": self.trusted,
            "trust_reason": self.trust_reason,
            "vec_rank": self.vec_rank,
            "fts_rank": self.fts_rank,
            "explanation": explanation,
        })


def _stage(state: str, reason: str) -> dict[str, str]:
    """Build one of the explicit per-modality execution states."""
    return {"state": state, "reason": reason}


def classify_recall_residue(
    *,
    recall_class: Any,
    role: Any,
    tool_name: Any,
    tool_input: Any,
    tool_output: Any,
) -> tuple[bool, str]:
    """Return raw-tool status and the honest stored/structural class source.

    A legacy v8 source has no ``recall_class`` column, while structurally raw
    tool rows still need both the ranked filter and untrusted label.  Keeping
    the calculation here makes ranked and timeline hits agree without exposing
    raw input/output bytes on ``Hit``.
    """
    raw_tool = role == "tool" or any(
        value is not None for value in (tool_name, tool_input, tool_output)
    )
    classification_source = (
        "events.recall_class"
        if recall_class is not None
        else "raw_tool_structure"
        if raw_tool
        else "legacy_unknown"
    )
    return raw_tool, classification_source


def _ordering_explanation(hit: Hit, *, is_rrf: bool) -> dict[str, str]:
    """Describe only ordering evidence this Hit actually carries.

    A post-fusion stage owns the order it produced: reporting fusion
    ordering for a list some later stage reordered would describe an order
    the caller was not given.
    """
    if hit.rerank_stage is not None:
        method = str(hit.rerank_stage["method"])
        return {
            "primary": f"{method}_desc",
            "ties": "content_hash_asc_then_memory_id_asc",
            "stage": method,
            "reordered_from": "rrf_score_desc",
        }
    if is_rrf:
        return {
            "primary": "rrf_score_desc",
            "ties": "content_hash_asc_then_memory_id_asc",
        }
    if (
        hit.vector_stage is not None
        and hit.fts_stage is not None
        and hit.vector_stage.get("reason") == "timeline_time_order"
        and hit.fts_stage.get("reason") == "timeline_time_order"
    ):
        return {
            "primary": "selected_clock_desc",
            "ties": (
                "event_id_asc_at_bounded_event_selection_then_"
                "memory_id_asc_after_materialization"
            ),
        }
    return {"primary": "not_recorded", "ties": "not_recorded_legacy"}


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
    include_residue: bool = False,
    recall_class_available: bool = False,
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
    if not include_residue:
        # This raw-tool fence works on pre-v9 files too. It intentionally uses
        # structured columns/role only; Haunt never guesses a class from text.
        clauses.append(
            "e.role != 'tool' AND e.tool_name IS NULL "
            "AND e.tool_input IS NULL AND e.tool_output IS NULL"
        )
        if recall_class_available:
            clauses.append("e.recall_class IS NULL")
    return " AND ".join(clauses), params


def _filter_context(
    *,
    as_of: str | None,
    since: str | None,
    until: str | None,
    tier: str | None,
    clock: str | None,
    include_residue: bool,
    include_untrusted: bool | None,
    residue_filter_source: str,
    recall_class_available: bool,
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
        "include_residue": include_residue,
        "include_untrusted": include_untrusted,
        # The compatibility flag is intentionally observable. A caller can
        # distinguish the new default from an old include_untrusted request,
        # and an explicit include_residue always wins in recall().
        "residue_filter_source": residue_filter_source,
        "residue_filter": (
            "bypassed"
            if include_residue
            else "applied"
            if recall_class_available
            else "unavailable"
        ),
        "recall_class_capability": (
            "available" if recall_class_available else "unavailable"
        ),
        "maintenance_performed": False,
    }


def _deserialize(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _content_keys(conn: sqlite3.Connection, ids: list[str]) -> dict[str, str]:
    """Stable per-row sort keys for a fused candidate set.

    memory_id is a fresh uuid4 per write, so ordering tied rows by it is total
    but re-randomized on every ingest: the same corpus scored twice put the
    same two exactly-tied documents in either order. content_hash is a pure
    function of the stored text, so it settles the tie the same way every time.
    Rows written before schema v10 and not yet backfilled hold NULL and fall
    back to the id (register item R7).
    """
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, COALESCE(content_hash, '') AS chash "
        f"FROM memories WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {str(row["id"]): str(row["chash"]) for row in rows}


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
        ORDER BY f.rank, COALESCE(m.content_hash, ''), f.id
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
                SELECT v.id AS mid, v.distance AS dist,
                       COALESCE(m.content_hash, '') AS chash
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
                ((r["mid"], float(r["dist"]), str(r["chash"])) for r in rows),
                key=lambda item: (item[1], item[2], item[0]),
            )
            return [
                (mid, i + 1, distance, "cosine_distance")
                for i, (mid, distance, _chash) in enumerate(candidates)
            ]
    sql = f"""
        SELECT m.id AS mid, m.embedding,
               COALESCE(m.content_hash, '') AS chash
        FROM memories m
        JOIN events e ON e.id = m.event_id
        WHERE m.embedding IS NOT NULL AND {where}
    """
    scored: list[tuple[str, float, str]] = []
    for r in conn.execute(sql, params):
        vec = _deserialize(r["embedding"])
        if len(vec) != len(query_vec):
            continue
        scored.append((r["mid"], _l2(query_vec, vec), str(r["chash"])))
    scored.sort(key=lambda x: (x[1], x[2], x[0]))
    return [
        (mid, i + 1, dist, "l2_distance")
        for i, (mid, dist, _chash) in enumerate(scored[:limit])
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
    store: Store | ReadOnlyStore | None = None,
    include_residue: bool | None = None,
    include_untrusted: bool | None = None,
    use_vectors: bool = True,
) -> list[Hit]:
    k = clamp_k(k)
    own = store is None
    store = store or open_existing_readonly(namespace or "default")
    try:
        # The deprecated trust flag remains an alias only when the modern flag
        # is omitted. Explicit ``include_residue`` always has precedence.
        resolved_residue = (
            bool(include_residue)
            if include_residue is not None
            else bool(include_untrusted)
            if include_untrusted is not None
            else False
        )
        residue_filter_source = (
            "include_residue"
            if include_residue is not None
            else "deprecated_include_untrusted"
            if include_untrusted is not None
            else "default"
        )
        recall_class_available = bool(
            getattr(store, "recall_class_available", False)
        )
        where, params = _filters(
            as_of,
            since,
            until,
            tier,
            clock,
            include_residue=resolved_residue,
            recall_class_available=recall_class_available,
        )
        filter_context = _filter_context(
            as_of=as_of,
            since=since,
            until=until,
            tier=tier,
            clock=clock,
            include_residue=resolved_residue,
            include_untrusted=include_untrusted,
            residue_filter_source=residue_filter_source,
            recall_class_available=recall_class_available,
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
        elif embed_offline():
            vector_execution = _stage("not_run", "offline_mode")
        elif not embed_available():
            vector_execution = _stage("not_run", "embedding_unavailable")
        else:
            qv = embed_one(query)
            if qv:
                vec = _vec_hits(store, qv, where, params, CANDIDATES)
                vec_reason = (
                    "no_vector_candidates"
                    if not vec
                    else "returned_persisted_embedding_candidates"
                    if any(metric == "l2_distance" for _, _, _, metric in vec)
                    else "returned_native_vec_candidates"
                )
                vector_execution = _stage(
                    "ran_not_candidate" if not vec else "candidate",
                    vec_reason,
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

        # haunt.rerank consumes this module's Hit/RecallResult, so importing
        # it back at module scope would be a cycle. The optional stage sits
        # between fusion and the returned k, so it sizes the slice it trims:
        # exactly k while disabled, wide enough for MMR to have somewhere to
        # promote from while enabled.
        from haunt import rerank

        # Only exactly-equal fused scores can consult the stable key, so pay
        # for the lookup only when two of them collide. FTS-only recall feeds
        # fusion a dense 1..N rank, which cannot produce equal sums, so that
        # path never pays at all.
        ordered_scores = sorted(rrf.values(), reverse=True)
        fused_tie = any(
            left == right for left, right in zip(ordered_scores, ordered_scores[1:])
        )
        stable = _content_keys(store.conn, list(rrf)) if fused_tie else {}
        ranked = sorted(
            rrf.items(), key=lambda kv: (-kv[1], stable.get(kv[0], ""), kv[0])
        )[: rerank.candidate_pool(k)]
        hits: list[Hit] = []
        recall_class_select = (
            "e.recall_class AS recall_class"
            if recall_class_available
            else "NULL AS recall_class"
        )
        for final_rank, (mid, score) in enumerate(ranked, start=1):
            row = store.conn.execute(
                f"""
                SELECT m.id, m.event_id, m.tier, m.content, m.valid_from, m.valid_to,
                       e.role, e.event_time, e.ts, e.tool_name, e.tool_input,
                       e.tool_output, e.origin,
                       {recall_class_select}
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
                    recall_class=row["recall_class"],
                    classification_source=classification_source,
                    raw_tool_structure=raw_tool,
                )
            )
        references = store.recall_references_many([hit.memory_id for hit in hits])
        for hit in hits:
            hit.references = references.get(hit.memory_id)
        try:
            pending_jobs = int(
                store.conn.execute("SELECT COUNT(*) FROM embedding_jobs").fetchone()[0]
            )
        except sqlite3.Error:
            pending_jobs = None
        result = RecallResult(
            hits,
            execution={
                "version": 1,
                "strategy": "recall",
                "modalities": {"vector": vector_execution, "fts": fts_execution},
                "read_only": bool(getattr(store, "read_only", False)),
                "maintenance_performed": False,
                "pending_embedding_jobs": {
                    "state": "observed_not_drained",
                    "count": pending_jobs,
                },
                "residue_filter": filter_context["residue_filter"],
                "residue_filter_source": filter_context["residue_filter_source"],
                "recall_class_capability": filter_context[
                    "recall_class_capability"
                ],
            },
        )
        # Returns `result` itself while disabled; reorders and truncates the
        # widened pool to k, restamping rank provenance, while enabled.
        return rerank.apply(result, k=int(k))
    except sqlite3.Error as exc:
        raise RetrievalBackendError(str(exc)) from exc
    finally:
        if own:
            store.close()
