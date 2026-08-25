"""Frozen, FTS-only retrieval evaluation used by the CI regression gate.

The harness intentionally creates real :class:`haunt.store.Store` instances
and invokes the public recall/planner entry points.  Its corpus is small by
design: it is a stable regression signal, not a benchmark claim.  Vector
retrieval is excluded because model availability and numeric results are not
portable enough to make a cross-platform golden lock useful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from haunt.planner import planned_recall
from haunt.recall import recall
from haunt.store import Store

SCHEMA_VERSION = 1
K = 3
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = _ROOT / "tests" / "fixtures" / "retrieval_eval" / "corpus.json"
DEFAULT_BASELINE = (
    _ROOT / "tests" / "fixtures" / "retrieval_eval" / "baseline.json"
)

# This is both a declaration of what the lock measures and input to its hash.
# Keep it deliberately free of host-specific library/model version strings.
CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "backend": "sqlite_fts5_porter_unicode61",
    "vectors": "disabled",
    "k": K,
    "positive_metrics": ["recall_at_k", "mrr", "false_negative_rate"],
    "negative_metrics": ["empty_result_rate", "false_positive_rate"],
}


@dataclass(frozen=True)
class Evaluation:
    dataset_sha256: str
    config_sha256: str
    cases: dict[str, dict[str, Any]]
    metrics: dict[str, float | int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_sha256": self.dataset_sha256,
            "config_sha256": self.config_sha256,
            "cases": self.cases,
            "metrics": self.metrics,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def load_corpus(path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if corpus.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported frozen retrieval corpus schema")
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
            supersedes = record.get("supersedes")
            if supersedes:
                old_memory_id = memory_ids.get(supersedes)
                if old_memory_id is None:
                    raise ValueError(f"unknown superseded fixture id: {supersedes}")
                # Store.contradict assigns both the old row's valid_to and the
                # replacement's clocks. Patch those two time sources so the
                # complete production supersession path is repeatable.
                at = record["superseded_at"]
                with patch("haunt.store.now_iso", return_value=at), patch(
                    "haunt.util.now_iso", return_value=at
                ):
                    contradicted = store.contradict(
                        old_memory_id,
                        replacement=record["content"],
                        origin="frozen-retrieval-eval",
                    )
                replacement_id = contradicted.get("replacement_memory_id")
                if not contradicted.get("ok") or not replacement_id:
                    raise RuntimeError(
                        f"fixture supersession failed for {record['id']}: {contradicted}"
                    )
                memory_ids[record["id"]] = str(replacement_id)
                continue
            result = store.observe(
                record["content"],
                role=record.get("role", "user"),
                tier=record.get("tier", "episodic"),
                origin="frozen-retrieval-eval",
                event_time=record["event_time"],
                valid_from=record.get("valid_from", record["event_time"]),
                meta=record.get("meta"),
                # Never invoke embedding code while constructing the lock.
                defer_embedding=True,
            )
            memory_ids[record["id"]] = result.memory_id
        return memory_ids
    finally:
        for store in stores.values():
            store.close()


def _run_case(
    case: dict[str, Any],
    memory_ids: dict[str, str],
) -> dict[str, Any]:
    options = dict(case.get("options", {}))
    namespace = case["namespace"]
    if case.get("entrypoint") == "planned_recall":
        now = datetime.fromisoformat(case["now"])
        hits = planned_recall(
            case["query"], now=now, namespace=namespace, k=K, **options
        )
    else:
        hits = recall(
            case["query"], namespace=namespace, k=K, use_vectors=False, **options
        )

    by_memory_id = {
        memory_id: fixture_id for fixture_id, memory_id in memory_ids.items()
    }
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
    }


def _metrics(cases: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    positive = [case for case in cases.values() if case["relevant"]]
    negative = [case for case in cases.values() if not case["relevant"]]
    if not positive or not negative:
        raise ValueError("frozen retrieval corpus needs positive and negative cases")

    recalls = [
        len(set(case["returned"]) & set(case["relevant"])) / len(case["relevant"])
        for case in positive
    ]
    reciprocal_ranks = [
        0.0
        if case["first_relevant_rank"] is None
        else 1.0 / case["first_relevant_rank"]
        for case in positive
    ]
    return {
        "cases": len(cases),
        "positive_cases": len(positive),
        "negative_cases": len(negative),
        "recall_at_3": round(sum(recalls) / len(recalls), 6),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6),
        "empty_result_rate": round(
            sum(1 for case in cases.values() if case["empty"]) / len(cases), 6
        ),
        "false_negative_rate": round(
            sum(1 for case in positive if case["first_relevant_rank"] is None)
            / len(positive),
            6,
        ),
        "false_positive_rate": round(
            sum(1 for case in negative if not case["empty"]) / len(negative), 6
        ),
    }


def _evaluate_unsafe(corpus: dict[str, Any]) -> Evaluation:
    """Run one evaluation inside an already-isolated FTS-only environment."""
    if os.environ.get("HAUNT_FTS_ONLY") not in {"1", "true", "yes"}:
        raise RuntimeError("frozen retrieval worker requires HAUNT_FTS_ONLY=1")
    memory_ids = _seed(corpus)
    cases = {
        case["id"]: _run_case(case, memory_ids) for case in corpus["cases"]
    }
    return Evaluation(_sha256(corpus), _sha256(CONFIG), cases, _metrics(cases))


def evaluate(corpus_path: Path = DEFAULT_CORPUS) -> Evaluation:
    """Evaluate ``corpus_path`` in a temporary, deterministic FTS-only home.

    This public entry point owns and restores the relevant environment, so it
    cannot create namespaces or embedding state in the caller's Haunt home.
    """
    corpus = load_corpus(corpus_path)
    from haunt import embed

    saved_env = {
        key: os.environ.get(key)
        for key in ("HAUNT_HOME", "HAUNT_FTS_ONLY", "HAUNT_EMBED_MODEL")
    }
    with TemporaryDirectory(prefix="haunt-frozen-retrieval-") as tmp:
        try:
            os.environ.update(
                {
                    "HAUNT_HOME": tmp,
                    "HAUNT_FTS_ONLY": "1",
                    "HAUNT_EMBED_MODEL": "off",
                }
            )
            # A prior caller may have cached an embedding backend. Resetting
            # makes planned_recall's normal vector check resolve to off.
            embed.reset()
            return _evaluate_unsafe(corpus)
        finally:
            embed.reset()
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="run Haunt frozen retrieval evaluation"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)
    if args.write_baseline and os.environ.get("HAUNT_ALLOW_FROZEN_RETRIEVAL_RELOCK") != "1":
        parser.error(
            "relocking is guarded; set HAUNT_ALLOW_FROZEN_RETRIEVAL_RELOCK=1 "
            "after reviewing intended retrieval changes"
        )
    result = evaluate(args.corpus).as_dict()
    if args.write_baseline:
        args.baseline.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the gate via evaluate()
    raise SystemExit(_main())
