"""C12: an off-by-default diversity rerank pass over haunt.recall.recall().

Measure-first (BACKLOG.md, C12 -- "Measure before adopting."). recall.py
documents the optional cross-encoder as deliberately not wired
(``src/haunt/recall.py:3``). This module does not wire one either: there is
no cross-encoder dependency anywhere in this project (see pyproject.toml --
sqlite-vec, fastembed, onnxruntime, tokenizers, huggingface_hub, and no
cross-encoder / sentence-transformers / transformers package), and loading
one would mean adding a new heavy dependency plus a new model download --
exactly the kind of unmeasured commitment C12 says not to make.

Instead this is a deterministic, non-model reranker: lexical Maximal
Marginal Relevance (MMR; Carbonell & Goldstein 1998). Relevance for each
candidate reuses its *existing* RRF ``score`` from recall() (min-max
normalized across the candidate pool) rather than re-deriving relevance from
scratch -- recall.py's hybrid vector+FTS fusion is a tested signal; a fresh
lexical-only relevance estimate would only be a weaker, unvetted substitute
for it. Diversity is plain token-set Jaccard overlap between candidate
``content`` strings. No embeddings, no model, no network: it needs nothing
beyond what a pure-FTS installation already has, which matches
frozen_retrieval_eval.py's own reason for excluding vectors from its golden
lock (model/host portability), and it is bitwise reproducible on any
machine.

Off by default, gated by HAUNT_RERANK_ENABLED (see rerank_enabled()). This
module never imports anything that touches recall.py's internals and never
modifies recall.py, mcp_server.py, cursor_hook.py, planner.py, dashboard.py,
or bootstrap.py -- it only consumes the public ``Hit``/``RecallResult``
objects recall.recall() already returns. No production call path imports
this module yet: see rerank_eval.py for the measurement this capability
exists to support, and the C12 backlog entry for the adopt-on-evidence bar.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Sequence

from haunt.recall import Hit, RecallResult, recall as _recall

_TOKEN = re.compile(r"[\w./+-]+", re.UNICODE)

RERANK_ENABLED_ENV = "HAUNT_RERANK_ENABLED"
RERANK_LAMBDA_ENV = "HAUNT_RERANK_LAMBDA"
RERANK_LAMBDA_DEFAULT = 0.5
RERANK_METHOD = "lexical_mmr"
# Mirrors recall.py's own CANDIDATES = 40: an algorithm-internal constant,
# not an env var. How many already RRF-ranked candidates recall_with_rerank()
# asks for before MMR trims the pool down to the caller's requested k. MMR
# can only trade off a redundant top candidate for a more distinct one that
# is already somewhere in this pool -- it can never promote a memory recall()
# did not return as a candidate at all.
RERANK_POOL = 40


def rerank_enabled() -> bool:
    """HAUNT_RERANK_ENABLED. Off unless explicitly turned on.

    Same boolean idiom as haunt.embed.offline()/fts_only(): '1'/'true'/'yes'
    (case-insensitive) is on, everything else -- unset, empty, 'false', '0',
    garbage -- is off.
    """
    raw = (os.environ.get(RERANK_ENABLED_ENV) or "").strip().lower()
    return raw in {"1", "true", "yes"}


def rerank_lambda() -> float:
    """HAUNT_RERANK_LAMBDA, clamped to [0, 1]. 1.0 = relevance only (no
    diversity pressure); 0.0 = diversity only (ignores relevance).

    Same parse -> fallback-on-garbage -> clamp idiom as HAUNT_TOOL_IO_MAX_CHARS
    (cursor_hook.py's _tool_io_cap): parse, fall back to the default on
    anything unparsable, then clamp so a bad env value cannot push the
    tradeoff outside its valid range.
    """
    raw = (os.environ.get(RERANK_LAMBDA_ENV) or "").strip()
    try:
        value = float(raw) if raw else RERANK_LAMBDA_DEFAULT
    except ValueError:
        value = RERANK_LAMBDA_DEFAULT
    return max(0.0, min(value, 1.0))


def _tokens(text: str | None) -> frozenset[str]:
    return frozenset(match.lower() for match in _TOKEN.findall(text or ""))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def mmr_rerank(
    hits: Sequence[Hit],
    *,
    k: int,
    lambda_: float = RERANK_LAMBDA_DEFAULT,
) -> list[Hit]:
    """Deterministic lexical MMR over already-ranked hits.

    Greedily selects, one at a time, the remaining hit maximizing::

        lambda_ * relevance(hit) - (1 - lambda_) * max_similarity_to_selected

    where ``relevance`` is the hit's existing RRF ``score``, min-max
    normalized across ``hits`` (so it lives on the same [0, 1] scale as the
    similarity term), and ``similarity`` is token-set Jaccard overlap between
    ``content`` strings. Ties break on ``memory_id`` ascending, so output is
    a pure function of the (score, content, memory_id) of each hit -- it
    never depends on dict/set iteration order, nor on the order ``hits``
    itself was passed in.

    Never mutates a Hit and never invents fields: this only reorders and
    truncates the sequence it is given. Every returned object is one of the
    input Hit instances, so trusted/trust_reason (computed properties) and
    every other field are exactly what recall() produced.
    """
    pool = list(hits)
    if k <= 0 or not pool:
        return []
    scores = [h.score for h in pool]
    lo, hi = min(scores), max(scores)
    spread = hi - lo

    def relevance(h: Hit) -> float:
        return (h.score - lo) / spread if spread > 0 else 1.0

    tokens = [_tokens(h.content) for h in pool]
    remaining = list(range(len(pool)))
    selected: list[int] = []
    while remaining and len(selected) < k:
        best_i: int | None = None
        best_key: tuple[float, str] | None = None
        for i in remaining:
            sim = (
                max(_jaccard(tokens[i], tokens[j]) for j in selected)
                if selected
                else 0.0
            )
            mmr_score = lambda_ * relevance(pool[i]) - (1.0 - lambda_) * sim
            # Sort ascending on (-mmr_score, memory_id): the smallest key is
            # the best candidate, and the tuple comparison is total, so this
            # is deterministic -- independent of both dict/set iteration
            # order and the order `hits` was passed in -- even when
            # mmr_score ties exactly.
            key = (-mmr_score, pool[i].memory_id)
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
        assert best_i is not None  # remaining is non-empty in this loop
        selected.append(best_i)
        remaining.remove(best_i)
    return [pool[i] for i in selected]


def apply(hits: Sequence[Hit], *, k: int) -> list[Hit]:
    """The one seam a future production caller would use. Off by default.

    When rerank_enabled() is false, returns ``hits`` completely unchanged --
    the identical object when it is already a list -- so a caller that
    unconditionally routes recall() output through here sees byte-identical
    behavior to calling recall.recall() directly.

    When enabled, reorders/truncates to ``k`` via mmr_rerank() and, if the
    input carried RecallResult.execution evidence, returns a new
    RecallResult with that same evidence plus an additive "rerank" entry --
    it never edits or drops existing execution/modalities evidence. Per-hit
    fields (including final_rank and the RRF-stage explanation.ordering
    computed by Hit.as_dict()) are left exactly as recall() produced them:
    they describe fusion provenance, not this module's post-fusion list
    order, and this module does not claim otherwise.
    """
    if not rerank_enabled():
        return hits if isinstance(hits, list) else list(hits)
    lambda_ = rerank_lambda()
    pool_size = len(hits)
    reranked = mmr_rerank(hits, k=k, lambda_=lambda_)
    execution = getattr(hits, "execution", None)
    if execution is not None:
        merged = deepcopy(execution)
        merged["rerank"] = {
            "enabled": True,
            "method": RERANK_METHOD,
            "lambda": lambda_,
            "pool": pool_size,
            "selected": len(reranked),
        }
        return RecallResult(reranked, execution=merged)
    return reranked


def recall_with_rerank(query: str, *, k: int = 8, **recall_kwargs: object) -> list[Hit]:
    """Convenience wrapper: recall() then apply(). Never modifies recall.py.

    Disabled (the default): calls ``recall.recall(query, k=k,
    **recall_kwargs)`` directly and returns it unchanged -- byte-identical to
    calling recall() yourself.

    Enabled: requests a wider RERANK_POOL candidate pool from recall() (so
    MMR has room to trade a redundant top candidate for a distinct one
    further down), then trims to ``k`` via apply().
    """
    if not rerank_enabled():
        return _recall(query, k=k, **recall_kwargs)
    pool_k = max(int(k), RERANK_POOL)
    hits = _recall(query, k=pool_k, **recall_kwargs)
    return apply(hits, k=k)
