"""C12 measurement harness: recall quality with vs. without rerank.mmr_rerank.

BACKLOG.md C12 is explicitly measure-first ("Measure before adopting"). This
module exists to answer one question with real numbers: does the lexical MMR
diversity pass in haunt.rerank change recall@k / MRR, and does the answer
differ between a query with one unambiguous right answer ("clear_top1") and
a query with several genuinely relevant candidates where a near-duplicate
cluster can crowd a distinct one out of a small k ("ambiguous")?

Shape follows haunt.frozen_retrieval_eval, which this file does not modify
and does not compete with: real haunt.store.Store instances are seeded via
Store.observe, and the public haunt.recall.recall entry point is called
un-modified and un-monkeypatched. FTS-only and vector-free for the same
reason frozen_retrieval_eval.py gives for excluding vectors from its own
lock -- "model availability and numeric results are not [...] portable" --
which also matches rerank.py's own choice of a lexical, model-free
reranker. This is deliberately simpler than frozen_retrieval_eval.py in one
respect: it has no baseline.json / relock-guard machinery, because that
mechanism exists there to catch drift against a *committed* CI lock, and
this harness is not a CI gate (E0 already owns the frozen FTS gate; pinned
hybrid evaluation is E6's). It is a re-runnable measurement tool: the
determinism contract it does keep is that evaluate() called twice on the
same corpus returns exactly the same numbers (see
tests/test_rerank_eval.py).

Both arms are computed from a single recall() call per case: recall() is
requested at a wide pool (haunt.rerank.RERANK_POOL) and RRF's own ordering
is a stable prefix (``sorted(...)[:pool][:k] == sorted(...)[:k]``), so
slicing that pool's first ``k`` is exactly what plain ``recall(query,
k=k)`` returns today -- the "baseline" arm is not an approximation.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from haunt.recall import Hit, recall
from haunt.rerank import (
    RERANK_ENABLED_ENV,
    RERANK_LAMBDA_DEFAULT,
    RERANK_POOL,
    mmr_rerank,
)
from haunt.store import Store

SCHEMA_VERSION = 1
K = 3
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = _ROOT / "tests" / "fixtures" / "rerank_eval" / "corpus.json"
QUERY_CLASSES = ("clear_top1", "ambiguous")


@dataclass(frozen=True)
class Evaluation:
    k: int
    lambda_: float
    pool: int
    cases: dict[str, dict[str, Any]]
    report: dict[str, dict[str, dict[str, Any]]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "k": self.k,
            "lambda": self.lambda_,
            "pool": self.pool,
            "cases": self.cases,
            "report": self.report,
        }


def load_corpus(path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if corpus.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported rerank eval corpus schema")
    for case in corpus["cases"]:
        if case.get("class") not in QUERY_CLASSES:
            raise ValueError(
                f"case {case.get('id')!r} has class {case.get('class')!r}, "
                f"expected one of {QUERY_CLASSES}"
            )
    return corpus


def _seed(corpus: dict[str, Any]) -> dict[str, str]:
    """Seed the fixture through Store.observe, retaining fixture-id mapping."""
    stores: dict[str, Store] = {}
    memory_ids: dict[str, str] = {}
    try:
        for record in corpus["records"]:
            namespace = record["namespace"]
            store = stores.get(namespace)
            if store is None:
                store = Store(namespace)
                stores[namespace] = store
            result = store.observe(
                record["content"],
                role=record.get("role", "user"),
                tier=record.get("tier", "episodic"),
                origin="rerank-eval",
                channel="rerank_eval",
                event_time=record["event_time"],
                valid_from=record.get("valid_from", record["event_time"]),
                # Never invoke embedding code while seeding: this harness
                # (and the rerank it measures) is deliberately vector-free.
                defer_embedding=True,
            )
            memory_ids[record["id"]] = result.memory_id
        return memory_ids
    finally:
        for store in stores.values():
            store.close()


def _redundancy_rate(hits: list[Hit], *, threshold: float = 0.6) -> float:
    """Diagnostic only, not one of the two required metrics: the fraction of
    returned hits (after the first) whose content is a lexical near-duplicate
    (token Jaccard >= threshold) of an earlier hit in the same list. Explains
    *why* recall@k/MRR moved, if it did -- it does not replace either metric.
    """
    if len(hits) < 2:
        return 0.0
    from haunt.rerank import _jaccard, _tokens

    seen: list[frozenset[str]] = []
    redundant = 0
    for hit in hits:
        tok = _tokens(hit.content)
        if any(_jaccard(tok, prior) >= threshold for prior in seen):
            redundant += 1
        seen.append(tok)
    return round(redundant / len(hits), 6)


def _run_case(
    case: dict[str, Any],
    by_memory_id: dict[str, str],
    *,
    k: int,
    lambda_: float,
    pool: int,
) -> dict[str, Any]:
    options = dict(case.get("options", {}))
    namespace = case["namespace"]
    candidates = recall(
        case["query"], namespace=namespace, k=pool, use_vectors=False, **options
    )
    baseline_hits = list(candidates[:k])
    reranked_hits = mmr_rerank(candidates, k=k, lambda_=lambda_)

    def _summarize(hits: list[Hit]) -> dict[str, Any]:
        returned = [by_memory_id[hit.memory_id] for hit in hits]
        expected = list(case.get("relevant", []))
        expected_set = set(expected)
        rank = next(
            (i for i, value in enumerate(returned, start=1) if value in expected_set),
            None,
        )
        return {
            "returned": returned,
            "relevant": expected,
            "first_relevant_rank": rank,
            "empty": not returned,
            "redundancy_rate": _redundancy_rate(hits),
        }

    return {
        "class": case["class"],
        "query": case["query"],
        "candidate_pool_size": len(candidates),
        "baseline": _summarize(baseline_hits),
        "reranked": _summarize(reranked_hits),
    }


def _metrics_for(entries: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [e for e in entries if e["relevant"]]
    if not positive:
        return {
            "cases": len(entries),
            "positive_cases": 0,
            "recall_at_k": None,
            "mrr": None,
            "mean_redundancy_rate": None,
        }
    recalls = [
        len(set(e["returned"]) & set(e["relevant"])) / len(e["relevant"])
        for e in positive
    ]
    reciprocal_ranks = [
        0.0 if e["first_relevant_rank"] is None else 1.0 / e["first_relevant_rank"]
        for e in positive
    ]
    redundancy = [e["redundancy_rate"] for e in entries]
    return {
        "cases": len(entries),
        "positive_cases": len(positive),
        "recall_at_k": round(sum(recalls) / len(recalls), 6),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6),
        "mean_redundancy_rate": round(sum(redundancy) / len(redundancy), 6),
    }


def _report(cases: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("baseline", "reranked"):
        by_class: dict[str, dict[str, Any]] = {
            "overall": _metrics_for([v[arm] for v in cases.values()])
        }
        for cls in QUERY_CLASSES:
            by_class[cls] = _metrics_for(
                [v[arm] for v in cases.values() if v["class"] == cls]
            )
        report[arm] = by_class
    return report


def _evaluate_unsafe(
    corpus: dict[str, Any], *, k: int, lambda_: float, pool: int
) -> Evaluation:
    """Run one evaluation inside an already-isolated FTS-only environment."""
    if os.environ.get("HAUNT_FTS_ONLY") not in {"1", "true", "yes"}:
        raise RuntimeError("rerank eval worker requires HAUNT_FTS_ONLY=1")
    memory_ids = _seed(corpus)
    by_memory_id = {memory_id: fixture_id for fixture_id, memory_id in memory_ids.items()}
    cases = {
        case["id"]: _run_case(case, by_memory_id, k=k, lambda_=lambda_, pool=pool)
        for case in corpus["cases"]
    }
    return Evaluation(k, lambda_, pool, cases, _report(cases))


def evaluate(
    corpus_path: Path = DEFAULT_CORPUS,
    *,
    k: int = K,
    lambda_: float = RERANK_LAMBDA_DEFAULT,
    pool: int = RERANK_POOL,
) -> Evaluation:
    """Evaluate ``corpus_path`` in a temporary, deterministic FTS-only home.

    This public entry point owns and restores the relevant environment, so it
    cannot create namespaces or embedding state in the caller's Haunt home --
    same isolation contract as frozen_retrieval_eval.evaluate().
    """
    corpus = load_corpus(corpus_path)
    from haunt import embed

    saved_env = {
        key: os.environ.get(key)
        for key in (
            "HAUNT_HOME",
            "HAUNT_FTS_ONLY",
            "HAUNT_EMBED_MODEL",
            RERANK_ENABLED_ENV,
        )
    }
    with TemporaryDirectory(prefix="haunt-rerank-eval-") as tmp:
        try:
            os.environ.update(
                {
                    "HAUNT_HOME": tmp,
                    "HAUNT_FTS_ONLY": "1",
                    "HAUNT_EMBED_MODEL": "off",
                }
            )
            # Both arms come from one recall() call, and recall() honours
            # HAUNT_RERANK_ENABLED -- left set, the baseline arm would
            # silently become a second reranked arm.
            os.environ.pop(RERANK_ENABLED_ENV, None)
            embed.reset()
            return _evaluate_unsafe(corpus, k=k, lambda_=lambda_, pool=pool)
        finally:
            embed.reset()
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="measure haunt.rerank.mmr_rerank against plain recall()"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=RERANK_LAMBDA_DEFAULT)
    parser.add_argument("--pool", type=int, default=RERANK_POOL)
    args = parser.parse_args(argv)
    result = evaluate(args.corpus, k=args.k, lambda_=args.lambda_, pool=args.pool).as_dict()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual measurement entry point
    raise SystemExit(_main())
