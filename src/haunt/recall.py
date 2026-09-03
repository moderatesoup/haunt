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
from typing import Any, Callable

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
# Ceiling on the span KNN width. The span table is keyed by span, not by
# memory, so asking it for `limit` rows can come back as far fewer distinct
# memories -- measured at 21 of a possible 40 on the dogfooded corpus, where
# one memory holds up to 23 spans. Asking for `limit * max_spans` makes the
# worst case exact (a memory cannot contribute more rows than it has spans);
# this bounds that product so a large HAUNT_EMBED_MAX_SPANS cannot turn one
# recall into an unbounded scan. Above the ceiling the leg is best-effort
# again, which is the pre-existing behavior rather than a new failure.
SPAN_KNN_MAX = 2048
# Ceiling on how wide a vector KNN may be re-asked when the eligibility filter
# eats the budget. See `_knn_eligible`: vec0 returns the k nearest rows and the
# validity / residue predicates discard some afterwards, so a corpus with many
# superseded or raw-tool rows nearer to the query than the live answer can
# leave nothing. Escalating is bounded here so one recall cannot turn into a
# full scan of a large index.
VEC_KNN_MAX = 4096
# Multiplier per escalation. Four keeps the number of round trips small (40 ->
# 160 -> 640 -> 2560) while not over-fetching hugely on the common case, which
# is satisfied by the very first pass.
VEC_KNN_GROWTH = 4
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
    # Set only when the reported vector distance came from a tail span rather
    # than the memory's head window (schema v14, `haunt.spans`). None means
    # the head vector matched, which is every memory short enough to fit one
    # embedding pass. Provenance, not metric: vec_metric is cosine distance
    # either way.
    vec_span_ord: int | None = None
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
    # The reproducible tie key this hit was ordered by. Falls back to the
    # memory id on a database whose content_hash column predates v10, so it is
    # always a usable sort key. Deliberately absent from as_dict(): it is
    # ordering machinery, not part of the public response.
    content_hash: str | None = None
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
                # Present only when a tail window produced this distance.
                # Omitted rather than set to null on a head match, so the
                # serialized vector explanation is byte-identical to the
                # pre-v14 one for every memory that fits a single pass.
                **(
                    {"matched_span_ord": self.vec_span_ord}
                    if self.vec_span_ord is not None
                    else {}
                ),
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


def _tie_key_name(hit: Hit) -> str:
    """The tie key this hit was actually ordered by.

    Not cosmetic. content_hash arrives with the v10 migration and recall opens
    read-only by default, so on a database no writer has migrated
    _stable_tie_key degrades to m.id -- and a row whose hash is still NULL
    degrades the same way through COALESCE. In both cases the hit's key IS its
    memory id, and claiming otherwise would put a mechanism the database cannot
    supply next to score_semantics, which is this module's honesty surface.
    """
    if hit.content_hash and hit.content_hash != hit.memory_id:
        return "content_hash_asc_then_memory_id_asc"
    return "memory_id_asc"


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
            "ties": _tie_key_name(hit),
            "stage": method,
            "reordered_from": "rrf_score_desc",
        }
    if is_rrf:
        return {"primary": "rrf_score_desc", "ties": _tie_key_name(hit)}
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


def _stable_tie_key(conn: sqlite3.Connection) -> str:
    """SQL for the reproducible tie key, degrading where the column is absent.

    content_hash arrives with the v10 migration, which only a writer performs;
    ReadOnlyStore never migrates (see its class docstring) and recall opens
    read-only by default. On a database no writer has opened at this code
    version the column does not exist, and naming it would turn every recall
    into "no such column". Store.stats() already guards the same way.

    The fallback is m.id, not a constant: falling back to '' would sort every
    unhashed row ahead of every hashed one, which is a systematic reordering
    rather than a settled tie. m.id reduces exactly to the previous
    (rank, id) behaviour for rows the key cannot cover.
    """
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(memories)").fetchall()
    }
    return "COALESCE(m.content_hash, m.id)" if "content_hash" in columns else "m.id"


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
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(memories)").fetchall()
    }
    if "content_hash" not in columns:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, COALESCE(content_hash, id) AS chash "
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
    tie = _stable_tie_key(conn)
    sql = f"""
        SELECT f.id AS mid, f.rank AS rnk
        FROM memories_fts f
        JOIN memories m ON m.id = f.id
        JOIN events e ON e.id = m.event_id
        WHERE memories_fts MATCH ?
          AND {where}
        ORDER BY f.rank, {tie}, f.id
        LIMIT ?
    """
    rows = conn.execute(sql, [match, *params, limit]).fetchall()
    return [(r["mid"], i + 1, float(r["rnk"])) for i, r in enumerate(rows)]


def _knn_eligible(
    conn: sqlite3.Connection,
    body: str,
    frm: str,
    frm_wide: str,
    blob: bytes,
    where: str,
    params: list[Any],
    fetch: int,
    ceiling: int,
    satisfied: "Callable[[list[sqlite3.Row]], bool]",
) -> list[sqlite3.Row]:
    """Nearest eligible rows, re-asking wider only when the filter bites.

    vec0 answers a KNN by returning the `k` nearest rows; Haunt's validity and
    residue predicates are then applied in the join. Rows the filter will throw
    away therefore spend candidate slots. On a corpus where superseded rows or
    raw tool I/O sit nearer to the query than the live answer, a `k` of 40 can
    come back with **nothing** eligible -- measured at 45 hidden rows -- and
    recall then reports `no_vector_candidates` when the index in fact returned
    a full 40. That is L30, and it predates tail spans.

    Two shapes, deliberately, and `frm` must be byte-identical to the joins
    this always used -- inner, not outer. The first pass is exactly the query
    that ran before, so a healthy corpus (nearly every recall) pays nothing for
    this fix. Measured on a 4,884-memory namespace: an unconditional wider pass
    tripled median recall latency, and so did merely widening the fast path's
    joins to LEFT, which changes the plan SQLite picks.
    Only when that pass comes back short does it re-ask with the predicate
    projected as a flag over LEFT JOINs, which is the shape that can report
    both signals the escalation needs: how many rows the index actually
    returned (is it exhausted?) and how many are eligible (is that enough?).

    Backend errors propagate. A dimension mismatch or a corrupt index is a
    `retrieval_backend_error` the caller must surface, not an empty result --
    only the span leg tolerates a failure here, and it catches at its own call
    site because a namespace mid-migration legitimately lacks the table.
    """
    fast = f"""
        SELECT {body}
        {frm}
        WHERE v.embedding MATCH ?
          AND k = ?
          AND {where}
        ORDER BY distance
    """
    rows = conn.execute(fast, [blob, fetch, *params]).fetchall()
    if satisfied(rows) or fetch >= ceiling:
        return rows

    k = fetch
    seen_eligible = len(rows)
    while k < ceiling:
        # Ratio-guided, not blindly geometric. The first pass already measured
        # how dense eligible rows are near this query, so aim straight at the
        # width that density implies and double it for headroom, instead of
        # creeping up by a fixed factor. On the dogfooded corpus -- 92% raw
        # tool residue -- this converges in one extra query where a fixed x4
        # took several. Falls back to the fixed factor when the first pass
        # found nothing, because then there is no density to extrapolate from.
        if seen_eligible > 0:
            implied = -(-fetch * k // max(1, seen_eligible)) * 2
            k = min(max(k * 2, implied), ceiling)
        else:
            k = min(k * VEC_KNN_GROWTH, ceiling)
        wide = f"""
            SELECT * FROM (
                SELECT {body},
                       CASE WHEN {where} THEN 1 ELSE 0 END AS eligible
                {frm_wide}
                WHERE v.embedding MATCH ?
                  AND k = ?
                ORDER BY distance
            )
        """
        probed = conn.execute(wide, [*params, blob, k]).fetchall()
        rows = [r for r in probed if r["eligible"]]
        seen_eligible = len(rows)
        if satisfied(rows):
            return rows
        if len(probed) < k:
            # The index handed back fewer rows than asked: it is exhausted,
            # and no wider request can find more.
            return rows
    return rows


def _span_hits(
    conn: sqlite3.Connection,
    blob: bytes,
    where: str,
    params: list[Any],
    limit: int,
    tie: str = "m.id",
) -> dict[str, tuple[float, int, str]]:
    """Nearest tail spans, collapsed to `memory_id -> (distance, ord, tie)`.

    `tie` is the stable tie-break expression the caller sorts by, carried
    through so a span-matched memory settles an exact distance tie by the same
    rule as a head-matched one.

    A memory longer than the embedding window has its head in `vec_memories`
    and the rest in `vec_memory_spans` (schema v14, see `haunt.spans`).
    Searching only the first table makes everything past the window
    unreachable by vector search -- on the live corpora that was around two
    thirds of all tokens.

    A memory can match on several spans at once. Only its best one is kept:
    the fused rank is a rank of memories, and letting one long memory occupy
    several candidate slots would crowd out short ones purely for being long.
    """
    if not _table_exists(conn, "vec_memory_spans"):
        return {}
    from haunt import spans as _spans

    # Two separate reasons to ask wider than `limit`, multiplied together.
    # `limit` nearest *spans* is not `limit` nearest memories (L23), and rows
    # the eligibility filter discards still spend slots (L30).
    want = min(max(limit, limit * _spans.max_spans()), SPAN_KNN_MAX)
    try:
        rows = _knn_eligible(
            conn,
            f"s.memory_id AS mid, s.ord AS ord, v.distance AS dist, {tie} AS chash",
            """
            FROM vec_memory_spans v
            JOIN memory_spans s ON s.id = v.id
            JOIN memories m ON m.id = s.memory_id
            JOIN events e ON e.id = m.event_id
            """,
            """
            FROM vec_memory_spans v
            LEFT JOIN memory_spans s ON s.id = v.id
            LEFT JOIN memories m ON m.id = s.memory_id
            LEFT JOIN events e ON e.id = m.event_id
            """,
            blob,
            where,
            params,
            want,
            max(SPAN_KNN_MAX, VEC_KNN_MAX),
            lambda rs: len({r["mid"] for r in rs}) >= limit,
        )
    except sqlite3.Error:
        # A namespace mid-migration can have the span table without its
        # vector table, or vice versa. The head vectors still answer; a
        # missing tail index degrades coverage, not correctness. The head leg
        # deliberately does not do this -- there, a backend error is real.
        return {}
    best: dict[str, tuple[float, int, str]] = {}
    for row in rows:
        mid = row["mid"]
        entry = (float(row["dist"]), int(row["ord"]), str(row["chash"]))
        current = best.get(mid)
        if current is None or entry[:2] < current[:2]:
            best[mid] = entry
    # Rows arrive ordered by distance, so the first `limit` distinct memories
    # are the nearest `limit`; trimming here keeps the leg the same width as
    # the head leg it is merged with.
    if len(best) <= limit:
        return best
    ordered = sorted(
        best.items(), key=lambda kv: (kv[1][0], kv[1][2], kv[0])
    )[:limit]
    return dict(ordered)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _vec_hits(
    store: Store,
    query_vec: list[float],
    where: str,
    params: list[Any],
    limit: int,
    span_origins: dict[str, int] | None = None,
) -> list[tuple[str, int, float, str]]:
    """Nearest memories by vector, over head vectors and tail spans alike.

    The returned tuple shape and its metric label are deliberately unchanged.
    A span vector is produced by the same model, in the same vec0 table
    configuration, under the same `distance_metric=cosine`: it is the same
    metric measured against a different window of the same memory, so calling
    it anything else would misreport the metric and would move E6's pinned
    profile identity (`haunt.abstention_eval`) for a reason that is not a
    profile change.

    Which window matched is provenance, not metric. When `span_origins` is
    supplied it is filled with `memory_id -> span ord` for every memory whose
    best distance came from a tail span, and the caller attaches that to the
    hit's explanation.
    """
    blob = sqlite_vec.serialize_float32(query_vec)
    conn = store.conn
    tie = _stable_tie_key(conn)
    if store.vec_ok():
        has = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_memories'"
        ).fetchone()
        if has:
            # vec0 accepts KNN queries ordered by distance alone. Sort the
            # returned candidate set in Python to settle exact distance ties;
            # do not treat a malformed native KNN query as an L2 fallback.
            # Asked through _knn_eligible so superseded rows and raw tool I/O
            # nearer than the live answer cannot spend the whole budget (L30).
            rows = _knn_eligible(
                conn,
                f"v.id AS mid, v.distance AS dist, {tie} AS chash",
                """
                FROM vec_memories v
                JOIN memories m ON m.id = v.id
                JOIN events e ON e.id = m.event_id
                """,
                """
                FROM vec_memories v
                LEFT JOIN memories m ON m.id = v.id
                LEFT JOIN events e ON e.id = m.event_id
                """,
                blob,
                where,
                params,
                limit,
                VEC_KNN_MAX,
                lambda rs: len(rs) >= limit,
            )
            # (distance, tie key). The tie key is #87's content hash, not
            # the metric name -- sorting on a constant would silently collapse
            # that tie-break back to memory id.
            merged: dict[str, tuple[float, str]] = {}
            for r in rows:
                merged[r["mid"]] = (float(r["dist"]), str(r["chash"]))
            # Both legs are asked for `limit` candidates and then merged, so a
            # memory reachable only by its tail competes on equal terms with
            # one reachable by its head. Nearest wins when both legs return
            # the same memory: the head and a tail window are two views of one
            # row, not two pieces of evidence.
            for mid, (distance, ord_, chash) in _span_hits(
                conn, blob, where, params, limit, tie
            ).items():
                current = merged.get(mid)
                if current is None or distance < current[0]:
                    merged[mid] = (distance, chash)
                    if span_origins is not None:
                        span_origins[mid] = ord_
                elif span_origins is not None:
                    # The head was at least as close. Drop any earlier span
                    # claim so the explanation never names a window that did
                    # not produce the reported distance.
                    span_origins.pop(mid, None)
            candidates = sorted(
                (
                    (mid, dist, chash)
                    for mid, (dist, chash) in merged.items()
                ),
                key=lambda item: (item[1], item[2], item[0]),
            )[:limit]
            if span_origins is not None:
                kept = {mid for mid, _, _ in candidates}
                for mid in [m for m in span_origins if m not in kept]:
                    del span_origins[mid]
            return [
                (mid, i + 1, distance, "cosine_distance")
                for i, (mid, distance, _chash) in enumerate(candidates)
            ]
    sql = f"""
        SELECT m.id AS mid, m.embedding,
               {tie} AS chash
        FROM memories m
        JOIN events e ON e.id = m.event_id
        WHERE m.embedding IS NOT NULL AND {where}
    """
    best_l2: dict[str, tuple[float, str]] = {}
    for r in conn.execute(sql, params):
        vec = _deserialize(r["embedding"])
        if len(vec) != len(query_vec):
            continue
        best_l2[r["mid"]] = (_l2(query_vec, vec), str(r["chash"]))
    # The native path searches head and tail vectors; this one has to as well,
    # or a namespace whose sqlite-vec extension failed to load is quietly back
    # to head-only retrieval while `haunt health` still reports tail coverage.
    # memory_spans.embedding exists for exactly this path.
    if _table_exists(conn, "memory_spans"):
        span_sql = f"""
            SELECT s.memory_id AS mid, s.embedding AS embedding,
                   {tie} AS chash
            FROM memory_spans s
            JOIN memories m ON m.id = s.memory_id
            JOIN events e ON e.id = m.event_id
            WHERE s.embedding IS NOT NULL AND {where}
        """
        try:
            span_rows = conn.execute(span_sql, params).fetchall()
        except sqlite3.Error:
            span_rows = []
        for r in span_rows:
            vec = _deserialize(r["embedding"])
            if len(vec) != len(query_vec):
                continue
            distance = _l2(query_vec, vec)
            current = best_l2.get(r["mid"])
            if current is None or distance < current[0]:
                best_l2[r["mid"]] = (distance, str(r["chash"]))
    scored = sorted(
        ((mid, dist, chash) for mid, (dist, chash) in best_l2.items()),
        key=lambda x: (x[1], x[2], x[0]),
    )
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
        # memory_id -> tail span ord, for hits whose reported distance came
        # from a window past the embedding head. Empty on every other path.
        span_origins: dict[str, int] = {}
        if not use_vectors:
            vector_execution = _stage("not_run", "disabled_by_caller")
        elif embed_offline():
            vector_execution = _stage("not_run", "offline_mode")
        elif not embed_available():
            vector_execution = _stage("not_run", "embedding_unavailable")
        else:
            qv = embed_one(query)
            if qv:
                vec = _vec_hits(
                    store, qv, where, params, CANDIDATES, span_origins
                )
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
            rrf.items(), key=lambda kv: (-kv[1], stable.get(kv[0], kv[0]), kv[0])
        )[: rerank.candidate_pool(k)]
        hits: list[Hit] = []
        materialize_tie = _stable_tie_key(store.conn)
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
                       {materialize_tie} AS content_hash,
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
                    content_hash=row["content_hash"],
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
                    vec_span_ord=span_origins.get(mid) if vr else None,
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
