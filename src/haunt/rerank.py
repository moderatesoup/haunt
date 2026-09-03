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

Off by default, gated by HAUNT_RERANK_ENABLED (see rerank_enabled()). It
consumes only the public ``Hit``/``RecallResult`` objects recall.recall()
already returns. recall() is the single wiring point -- it sizes its fused
candidate slice with candidate_pool() and trims that slice to the caller's
``k`` with apply() -- so CLI, MCP, dashboard and hooks either all rerank or
all do not. See rerank_eval.py for the measurement this capability exists
to support, and the C12 backlog entry for the adopt-on-evidence bar.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from dataclasses import replace
from typing import Sequence

from haunt.recall import Hit, RecallResult
from haunt.util import env_flag

_TOKEN = re.compile(r"[\w./+-]+", re.UNICODE)

RERANK_ENABLED_ENV = "HAUNT_RERANK_ENABLED"
RERANK_LAMBDA_ENV = "HAUNT_RERANK_LAMBDA"
RERANK_LAMBDA_DEFAULT = 0.3
RERANK_METHOD = "lexical_mmr"
# Mirrors recall.py's own CANDIDATES = 40: an algorithm-internal constant,
# not an env var. How many already RRF-ranked candidates candidate_pool()
# asks recall() to materialize before MMR trims them down to the caller's
# requested k. MMR can only trade off a redundant top candidate for a more
# distinct one that is already somewhere in this pool -- it can never
# promote a memory recall() did not fuse as a candidate at all.
RERANK_POOL = 40


def rerank_enabled() -> bool:
    """HAUNT_RERANK_ENABLED. Off unless explicitly turned on."""
    return env_flag(RERANK_ENABLED_ENV)


def rerank_lambda() -> float:
    """HAUNT_RERANK_LAMBDA, clamped to [0, 1]. 1.0 = relevance only (no
    diversity pressure); 0.0 = diversity only (ignores relevance). Falls back
    to the default on anything unparsable."""
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
    ``content`` strings. Ties break on ``content_hash`` then ``memory_id``
    ascending, matching what recall() advertises, so output is a pure
    function of the (score, content, content_hash, memory_id) of each hit -- it
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
            # Sort ascending on (-mmr_score, content_hash, memory_id): the
            # smallest key is the best candidate, and the tuple comparison is
            # total, so this is deterministic -- independent of both dict/set
            # iteration order and the order `hits` was passed in -- even when
            # mmr_score ties exactly. content_hash comes first among the tie
            # keys because memory_id is a fresh uuid4 per write, so keying on
            # it alone reproduced within a run but not across ingests, which
            # is what recall() advertises as its tie order.
            key = (
                -mmr_score,
                pool[i].content_hash or pool[i].memory_id,
                pool[i].memory_id,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
        assert best_i is not None  # remaining is non-empty in this loop
        selected.append(best_i)
        remaining.remove(best_i)
    return [pool[i] for i in selected]


def candidate_pool(k: int) -> int:
    """How many fused candidates recall() should materialize for a caller's ``k``.

    ``k`` itself while disabled, so a default recall still reads exactly the
    rows it returns. MMR can only trade a redundant top candidate for a more
    distinct one already in the pool, so the wider pool is worth its extra
    row reads only when something will use it.
    """
    return max(int(k), RERANK_POOL) if rerank_enabled() else int(k)


def apply(hits: Sequence[Hit], *, k: int) -> list[Hit]:
    """The seam recall() routes its fused candidate slice through.

    When rerank_enabled() is false, returns ``hits`` completely unchanged --
    the identical object when it is already a list -- so recall() behaves
    exactly as it did before this stage existed.

    When enabled, reorders and truncates to ``k`` via mmr_rerank(), then
    restamps ranking provenance so it describes the order actually returned:
    ``final_rank`` becomes the post-rerank position and ``rerank_stage``
    names the stage that chose it plus the RRF fusion rank the hit moved
    from. Restamping builds copies, so the caller's own Hit objects are never
    mutated. When the input carried RecallResult.execution evidence the
    result carries that same evidence plus an additive "rerank" entry;
    existing execution/modalities evidence is never edited or dropped.
    """
    if not rerank_enabled():
        return hits if isinstance(hits, list) else list(hits)
    lambda_ = rerank_lambda()
    pool_size = len(hits)
    reranked: list[Hit] = [
        replace(
            hit,
            final_rank=position,
            rerank_stage={"method": RERANK_METHOD, "rrf_rank": hit.final_rank},
        )
        for position, hit in enumerate(mmr_rerank(hits, k=k, lambda_=lambda_), start=1)
    ]
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
