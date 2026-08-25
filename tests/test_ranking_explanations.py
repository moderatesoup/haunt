"""Recall explanations expose rank provenance without changing score semantics."""

from __future__ import annotations

import importlib

from haunt.store import Store, observe


def test_fts_only_explanation_preserves_legacy_fields_and_marks_tool_io(haunt_env):
    """FTS-only output keeps old keys while adding inspectable provenance."""
    stored = observe(
        "EXPLAIN-FTS-91 deployment trace",
        namespace="default",
        role="tool",
        tool_name="Read",
    )
    recall_module = importlib.import_module("haunt.recall")

    hits = recall_module.recall(
        "EXPLAIN-FTS-91",
        namespace="default",
        use_vectors=False,
        as_of="2030-01-02T03:04:05+00:00",
        since="2025-01-01T00:00:00+00:00",
        tier="episodic",
    )

    hit = next(hit for hit in hits if hit.memory_id == stored.memory_id)
    payload = hit.as_dict()
    # Existing consumer fields remain available with their original values.
    assert payload["memory_id"] == stored.memory_id
    assert payload["score"] == round(hit.score, 6)
    assert payload["fts_rank"] == 1
    assert payload["trusted"] is False
    assert payload["trust_reason"] == "untrusted-tool-io"

    explanation = payload["explanation"]
    assert explanation == {
        "version": 1,
        "retrieval_method": "fts_rrf",
        "score_semantics": "rrf_rank_signal_not_confidence",
        "final_rank": 1,
        "rrf_score": payload["score"],
        "rrf_k": 60,
        "rrf_contributions": [
            {"source": "fts", "rank": 1, "value": round(1 / 61, 6)}
        ],
        "vector": None,
        "fts": {"rank": 1, "raw_score": hit.fts_rank_raw},
        "filters": {
            "validity": "as_of",
            "as_of": "2030-01-02T03:04:05.000000+00:00",
            "clock": "event_time",
            "since": "2025-01-01T00:00:00.000000+00:00",
            "until": None,
            "tier": "episodic",
            "include_untrusted": True,
        },
        "trust": {"trusted": False, "reason": "untrusted-tool-io"},
    }
    assert explanation["fts"]["raw_score"] is not None


def test_hybrid_explanation_reports_each_rrf_contribution(haunt_env, monkeypatch):
    """Mock ranks make vector + FTS explanation deterministic and exact."""
    first = observe("EXPLAIN-HYBRID-ONE", namespace="default")
    second = observe("EXPLAIN-HYBRID-TWO", namespace="default")
    recall_module = importlib.import_module("haunt.recall")

    monkeypatch.setattr(Store, "ensure_current_embeddings", lambda self: None)
    monkeypatch.setattr(
        Store, "process_embedding_jobs", lambda self, *, limit=64: {"processed": 0}
    )
    monkeypatch.setattr(recall_module, "embed_available", lambda: True)
    monkeypatch.setattr(recall_module, "embed_one", lambda query: [0.0])
    monkeypatch.setattr(
        recall_module,
        "_vec_hits",
        lambda store, query_vec, where, params, limit: [
            (first.memory_id, 1, 0.125),
            (second.memory_id, 2, 0.75),
        ],
    )
    monkeypatch.setattr(
        recall_module,
        "_fts_hits",
        lambda conn, query, where, params, limit: [
            (second.memory_id, 1, -3.0),
            (first.memory_id, 3, -1.5),
        ],
    )

    hits = recall_module.recall("EXPLAIN-HYBRID", namespace="default", k=2)
    by_id = {hit.memory_id: hit.as_dict()["explanation"] for hit in hits}

    first_explanation = by_id[first.memory_id]
    assert first_explanation["retrieval_method"] == "hybrid_rrf"
    assert first_explanation["final_rank"] == 2
    assert first_explanation["vector"] == {"rank": 1, "distance": 0.125}
    assert first_explanation["fts"] == {"rank": 3, "raw_score": -1.5}
    assert first_explanation["rrf_contributions"] == [
        {"source": "vector", "rank": 1, "value": round(1 / 61, 6)},
        {"source": "fts", "rank": 3, "value": round(1 / 63, 6)},
    ]
    assert first_explanation["rrf_score"] == round(1 / 61 + 1 / 63, 6)
    assert first_explanation["filters"]["validity"] == "current"
    assert first_explanation["filters"]["include_untrusted"] is True
