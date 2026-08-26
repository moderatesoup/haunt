"""Recall explanations expose rank provenance without changing score semantics."""

from __future__ import annotations

import importlib
import json
import struct
from dataclasses import dataclass

from typer.testing import CliRunner

from haunt.recall import Hit
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
        "rrf_score": 1 / 61,
        "rrf_k": 60,
        "rrf_contributions": [
            {"source": "fts", "rank": 1, "value": 1 / 61}
        ],
        "vector": None,
        "fts": {
            "rank": 1,
            "raw_score": hit.fts_rank_raw,
            "metric": "fts5_bm25",
            "lower_is_better": True,
        },
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
    assert sum(item["value"] for item in explanation["rrf_contributions"]) == explanation["rrf_score"]


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
            (first.memory_id, 1, 0.125, "cosine_distance"),
            (second.memory_id, 2, 0.75, "cosine_distance"),
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
    assert first_explanation["vector"] == {
        "rank": 1,
        "distance": 0.125,
        "metric": "cosine_distance",
        "lower_is_better": True,
    }
    assert first_explanation["fts"] == {
        "rank": 3,
        "raw_score": -1.5,
        "metric": "fts5_bm25",
        "lower_is_better": True,
    }
    assert first_explanation["rrf_contributions"] == [
        {"source": "vector", "rank": 1, "value": 1 / 61},
        {"source": "fts", "rank": 3, "value": 1 / 63},
    ]
    assert first_explanation["rrf_score"] == 1 / 61 + 1 / 63
    assert sum(item["value"] for item in first_explanation["rrf_contributions"]) == first_explanation["rrf_score"]
    assert first_explanation["filters"]["validity"] == "current"
    assert first_explanation["filters"]["include_untrusted"] is True


def test_vector_only_explanation_reports_vector_evidence(haunt_env, monkeypatch):
    """A vector-only hit says exactly that—there is no fabricated FTS evidence."""
    stored = observe("EXPLAIN-VECTOR-ONLY", namespace="default")
    recall_module = importlib.import_module("haunt.recall")

    monkeypatch.setattr(Store, "ensure_current_embeddings", lambda self: None)
    monkeypatch.setattr(
        Store, "process_embedding_jobs", lambda self, *, limit=64: {"processed": 0}
    )
    monkeypatch.setattr(recall_module, "embed_available", lambda: True)
    monkeypatch.setattr(recall_module, "embed_one", lambda query: [0.0])
    monkeypatch.setattr(recall_module, "_fts_hits", lambda *args: [])
    monkeypatch.setattr(
        recall_module,
        "_vec_hits",
        lambda *args: [(stored.memory_id, 1, 0.25, "cosine_distance")],
    )

    hit = recall_module.recall("EXPLAIN-VECTOR", namespace="default", k=1)[0]
    explanation = hit.as_dict()["explanation"]
    assert explanation["retrieval_method"] == "vector_rrf"
    assert explanation["fts"] is None
    assert explanation["rrf_contributions"] == [
        {"source": "vector", "rank": 1, "value": 1 / 61}
    ]
    assert explanation["vector"] == {
        "rank": 1,
        "distance": 0.25,
        "metric": "cosine_distance",
        "lower_is_better": True,
    }


def test_empty_token_query_returns_no_fabricated_hit_or_explanation(haunt_env):
    """A query with no FTS tokens is an empty result, not a scored fallback."""
    observe("EXPLAIN-NO-CANDIDATE", namespace="default")
    recall_module = importlib.import_module("haunt.recall")

    hits = recall_module.recall("!!!", namespace="default", use_vectors=False)
    assert hits == []


def test_rank_one_rrf_contributions_sum_to_serialized_score():
    """No independent rounding makes a pair of rank-one sources disagree."""
    hit = Hit(
        memory_id="memory",
        event_id="event",
        score=2 / 61,
        tier="episodic",
        content="content",
        role="user",
        event_time="2026-08-25T00:00:00+00:00",
        valid_from="2026-08-25T00:00:00+00:00",
        valid_to=None,
        tool_name=None,
        vec_rank=1,
        vec_distance=0.1,
        vec_metric="cosine_distance",
        fts_rank=1,
        fts_rank_raw=-0.5,
    )

    explanation = json.loads(json.dumps(hit.as_dict()))["explanation"]
    assert explanation["rrf_contributions"] == [
        {"source": "vector", "rank": 1, "value": 1 / 61},
        {"source": "fts", "rank": 1, "value": 1 / 61},
    ]
    assert sum(item["value"] for item in explanation["rrf_contributions"]) == explanation["rrf_score"]


@dataclass
class _Rows:
    rows: list[dict]

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def __iter__(self):
        return iter(self.rows)


class _NativeVecConnection:
    def execute(self, sql, params=None):
        if "sqlite_master" in sql:
            return _Rows([{"name": "vec_memories"}])
        assert "FROM vec_memories" in sql
        return _Rows([{"mid": "native", "dist": 0.25}])


class _FallbackVecConnection:
    def execute(self, sql, params=None):
        assert "FROM memories" in sql
        return _Rows([{"mid": "fallback", "embedding": struct.pack("1f", 1.0)}])


class _NativeVecStore:
    conn = _NativeVecConnection()

    @staticmethod
    def vec_ok():
        return True


class _FallbackVecStore:
    conn = _FallbackVecConnection()

    @staticmethod
    def vec_ok():
        return False


def test_vector_explanation_identifies_native_cosine_and_fallback_l2():
    """Raw vector distances are interpretable across both retrieval paths."""
    recall_module = importlib.import_module("haunt.recall")

    native = recall_module._vec_hits(_NativeVecStore(), [0.0], "1=1", [], 8)
    fallback = recall_module._vec_hits(_FallbackVecStore(), [0.0], "1=1", [], 8)

    assert native == [("native", 1, 0.25, "cosine_distance")]
    assert fallback == [("fallback", 1, 1.0, "l2_distance")]


def test_temporal_timeline_surfaces_not_ranked_and_cli_uses_time_order(haunt_env, monkeypatch):
    """A bare temporal query has timeline semantics, never an RRF label."""
    from haunt import cli
    from haunt.planner import planned_recall
    from haunt.temporal import compile
    from tests.test_temporal_planner import NOW

    with Store("default") as store:
        stored = store.observe("timeline event", event_time="2026-08-08T12:00:00+00:00")
        query = compile("what happened two weeks ago", NOW)
        hits = planned_recall("what happened two weeks ago", now=NOW, store=store)

    hit = next(hit for hit in hits if hit.memory_id == stored.memory_id)
    explanation = hit.as_dict()["explanation"]
    assert explanation["retrieval_method"] == "timeline"
    assert explanation["score_semantics"] == "not_ranked"
    assert explanation["rrf_score"] is None
    assert hit.score == 0.0
    assert query.temporal is True

    monkeypatch.setattr(cli, "planned_recall", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(cli, "_existing", lambda namespace: Store("default"))
    result = CliRunner().invoke(cli.app, ["recall", "what happened two weeks ago"])
    assert result.exit_code == 0
    assert "signal" in result.stdout
    assert "time-order" in result.stdout
    assert "rrf=" not in result.stdout

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    monkeypatch.setattr(mcp_server, "planned_recall", lambda *args, **kwargs: [hit])
    payload = json.loads(mcp_server.memory_recall(query="what happened two weeks ago"))
    assert payload["hits"][0]["score"] == 0.0
    assert payload["hits"][0]["explanation"]["score_semantics"] == "not_ranked"
    assert payload["hits"][0]["explanation"]["rrf_score"] is None


def test_cli_json_serializes_ranked_and_timeline_explanations(haunt_env, monkeypatch):
    """--json exposes Hit.as_dict while default output remains human-readable."""
    from haunt import cli

    ranked = Hit(
        memory_id="ranked-memory",
        event_id="ranked-event",
        score=1 / 61,
        tier="episodic",
        content="ranked content",
        role="user",
        event_time="2026-08-08T12:00:00+00:00",
        valid_from="2026-08-08T12:00:00+00:00",
        valid_to=None,
        tool_name=None,
        fts_rank=1,
        fts_rank_raw=-1.0,
        final_rank=1,
    )
    timeline = Hit(
        memory_id="timeline-memory",
        event_id="timeline-event",
        score=0.0,
        tier="episodic",
        content="timeline content",
        role="user",
        event_time="2026-08-08T12:00:00+00:00",
        valid_from="2026-08-08T12:00:00+00:00",
        valid_to=None,
        tool_name=None,
        final_rank=1,
    )
    monkeypatch.setattr(cli, "_existing", lambda namespace: Store("default"))
    runner = CliRunner()

    monkeypatch.setattr(cli, "planned_recall", lambda *args, **kwargs: [ranked])
    ranked_json = runner.invoke(cli.app, ["recall", "ranked", "--json"])
    assert ranked_json.exit_code == 0
    ranked_payload = json.loads(ranked_json.stdout)
    assert ranked_payload["hits"][0]["memory_id"] == "ranked-memory"
    assert ranked_payload["hits"][0]["explanation"]["retrieval_method"] == "fts_rrf"
    assert ranked_payload["hits"][0]["explanation"]["rrf_score"] == 1 / 61

    human = runner.invoke(cli.app, ["recall", "ranked"])
    assert human.exit_code == 0
    assert "signal" in human.stdout
    assert not human.stdout.lstrip().startswith("{")

    monkeypatch.setattr(cli, "planned_recall", lambda *args, **kwargs: [timeline])
    timeline_json = runner.invoke(cli.app, ["recall", "what happened", "--json"])
    assert timeline_json.exit_code == 0
    timeline_payload = json.loads(timeline_json.stdout)
    explanation = timeline_payload["hits"][0]["explanation"]
    assert explanation["retrieval_method"] == "timeline"
    assert explanation["score_semantics"] == "not_ranked"
    assert explanation["rrf_score"] is None
