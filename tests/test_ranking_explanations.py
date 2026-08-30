"""Recall explanations expose rank provenance without changing score semantics."""

from __future__ import annotations

import importlib
import json
import struct
from dataclasses import dataclass

import pytest
import sqlite_vec
from typer.testing import CliRunner

from haunt.recall import Hit
from haunt.store import Store, ensure_vec_table, observe


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
        include_residue=True,
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
        "ordering": {"primary": "rrf_score_desc", "ties": "content_hash_asc_then_memory_id_asc"},
        "vector": {
            "state": "not_run",
            "reason": "disabled_by_caller",
            "rank": None,
            "distance": None,
            "metric": None,
            "lower_is_better": None,
        },
        "fts": {
            "state": "candidate",
            "reason": "returned_fts_candidate",
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
            "include_residue": True,
            "include_untrusted": None,
            "residue_filter_source": "include_residue",
            "residue_filter": "bypassed",
            "recall_class_capability": "available",
            "maintenance_performed": False,
        },
        "references": {
            "correction_lineage": {"status": "standalone"},
            "correction_lineage_status": "standalone",
            "provenance": stored.provenance,
            "provenance_status": "native",
        },
        "trust": {"trusted": False, "reason": "untrusted-tool-io"},
        "residue": {
            "recall_class": "tool",
            "classification_source": "events.recall_class",
            "filter": "bypassed",
        },
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
        "state": "candidate",
        "reason": "returned_vector_candidate",
        "rank": 1,
        "distance": 0.125,
        "metric": "cosine_distance",
        "lower_is_better": True,
    }
    assert first_explanation["fts"] == {
        "state": "candidate",
        "reason": "returned_fts_candidate",
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
    assert first_explanation["filters"]["include_untrusted"] is None
    assert first_explanation["filters"]["include_residue"] is False


def test_recall_references_use_e1_lineage_and_e2_public_provenance(
    haunt_env, monkeypatch
):
    """Recall attaches only safe lineage and structured-provenance references."""
    recall_module = importlib.import_module("haunt.recall")
    imported_provenance = {
        "schema_version": 1,
        "kind": "import",
        "channel": "python",
        "source_platform": "reference-test",
        "source_native_id": "safe-native-id",
        "source_format": "json",
        "parser_version": "v1",
        "imported_at": "2026-08-08T12:00:00+00:00",
        "fidelity": "lossless",
        "original_blob_sha256": "sha256:" + "ab" * 32,
        "transforms": ["parse"],
    }
    erased_canary = "EXPLAIN-REFERENCE-ERASED-CANARY"
    with Store("default") as store:
        imported = store.observe(
            "EXPLAIN-REFERENCE-IMPORT",
            provenance=imported_provenance,
            defer_embedding=True,
        )
        legacy = store.observe("EXPLAIN-REFERENCE-LEGACY", defer_embedding=True)
        invalid = store.observe("EXPLAIN-REFERENCE-INVALID", defer_embedding=True)
        store.conn.execute(
            "UPDATE events SET provenance=NULL WHERE id=?", (legacy.event_id,)
        )
        store.conn.execute(
            "UPDATE events SET provenance='{\"kind\":\"bogus\"}' WHERE id=?",
            (invalid.event_id,),
        )

        original = store.observe("EXPLAIN-REFERENCE-ORIGINAL", defer_embedding=True)
        correction = store.contradict(
            original.memory_id,
            replacement="EXPLAIN-REFERENCE-LINKED",
            idempotency_key="explanation-linked",
        )

        first = store.observe("EXPLAIN-REFERENCE-FIRST", defer_embedding=True)
        into_erased = store.contradict(
            first.memory_id,
            replacement=erased_canary,
            idempotency_key="explanation-erased-1",
        )
        erased_id = into_erased["replacement_memory_id"]
        erased_event_id = into_erased["replacement_event_id"]
        survivor = store.contradict(
            erased_id,
            replacement="EXPLAIN-REFERENCE-SURVIVOR",
            idempotency_key="explanation-erased-2",
        )["replacement_memory_id"]
        store.purge(erased_id)

        def recalled(query: str):
            return recall_module.recall(query, store=store, use_vectors=False)[0]

        import_references = recalled("EXPLAIN-REFERENCE-IMPORT").as_dict()[
            "explanation"
        ]["references"]
        legacy_references = recalled("EXPLAIN-REFERENCE-LEGACY").as_dict()[
            "explanation"
        ]["references"]
        invalid_references = recalled("EXPLAIN-REFERENCE-INVALID").as_dict()[
            "explanation"
        ]["references"]
        linked_references = recalled("EXPLAIN-REFERENCE-LINKED").as_dict()[
            "explanation"
        ]["references"]
        survivor_hit = recalled("EXPLAIN-REFERENCE-SURVIVOR")
        survivor_references = survivor_hit.as_dict()["explanation"]["references"]

    assert import_references == {
        "correction_lineage": {"status": "standalone"},
        "correction_lineage_status": "standalone",
        "provenance": imported.provenance,
        "provenance_status": "import",
    }
    assert legacy_references["provenance"] is None
    assert legacy_references["provenance_status"] == "legacy_unstructured"
    assert invalid_references["provenance"] is None
    assert invalid_references["provenance_status"] == "invalid_stored"
    assert linked_references["correction_lineage"] == {
        "status": "linked",
        "correction_ids": [correction["correction_id"]],
    }
    assert survivor_hit.memory_id == survivor
    assert survivor_references["correction_lineage"] == {
        "status": "privacy_tombstone"
    }
    serialized = json.dumps(survivor_hit.as_dict(), allow_nan=False)
    assert erased_canary not in serialized
    assert erased_id not in serialized
    assert erased_event_id not in serialized

    from haunt import cli, mcp_server
    from tests.dashutil import make_dash_client

    cli_result = CliRunner().invoke(
        cli.app,
        ["recall", "EXPLAIN-REFERENCE-LINKED", "-n", "default", "--json"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_payload = json.loads(cli_result.stdout)
    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    mcp_server._MCP_AUTHORITY = None
    mcp_payload = json.loads(
        mcp_server.memory_recall(
            query="EXPLAIN-REFERENCE-LINKED", namespace="default"
        )
    )
    dashboard_response = make_dash_client().get(
        "/api/namespace/default/recall?q=EXPLAIN-REFERENCE-LINKED"
    )
    assert dashboard_response.status_code == 200, dashboard_response.text
    expected_reference = linked_references
    for payload in (cli_payload, mcp_payload, dashboard_response.json()):
        hit = next(
            hit
            for hit in payload["hits"]
            if hit["memory_id"] == correction["replacement_memory_id"]
        )
        assert hit["explanation"]["references"] == expected_reference


def test_recall_reference_batch_avoids_unrelated_correction_graph_scans(
    haunt_env, monkeypatch
):
    """One result batch uses one chain-local correction query, not O(k * N)."""
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    with Store("default") as store:
        root = store.observe("REFERENCE-BATCH-ROOT", defer_embedding=True)
        linked = store.contradict(
            root.memory_id,
            replacement="REFERENCE-BATCH-LINKED",
            idempotency_key="reference-batch-root",
        )
        for index in range(128):
            unrelated = store.observe(
                f"REFERENCE-BATCH-UNRELATED-{index}", defer_embedding=True
            )
            store.contradict(
                unrelated.memory_id,
                replacement=f"REFERENCE-BATCH-REPLACEMENT-{index}",
                idempotency_key=f"reference-batch-{index}",
            )

        statements: list[str] = []
        store.conn.set_trace_callback(statements.append)
        try:
            references = store.recall_references_many(
                [linked["replacement_memory_id"]]
            )
        finally:
            store.conn.set_trace_callback(None)

    correction_reads = [
        statement for statement in statements if "corrections" in statement.lower()
    ]
    assert len(correction_reads) == 1
    assert "ORDER BY corrected_at, rowid" not in correction_reads[0]
    assert references[linked["replacement_memory_id"]]["correction_lineage"] == {
        "status": "linked",
        "correction_ids": [linked["correction_id"]],
    }


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
    assert explanation["fts"] == {
        "state": "ran_not_candidate",
        "reason": "no_fts_candidates",
        "rank": None,
        "raw_score": None,
        "metric": None,
        "lower_is_better": None,
    }
    assert explanation["rrf_contributions"] == [
        {"source": "vector", "rank": 1, "value": 1 / 61}
    ]
    assert explanation["vector"] == {
        "state": "candidate",
        "reason": "returned_vector_candidate",
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
    assert hits.modalities == {
        "vector": {"state": "not_run", "reason": "disabled_by_caller"},
        "fts": {"state": "not_run", "reason": "query_has_no_fts_tokens"},
    }


def test_vector_stage_reports_unavailable_and_tokenless_fts_honestly(
    haunt_env, monkeypatch
):
    """A query vector cannot imply that tokenless FTS was executed."""
    stored = observe("EXPLAIN-TOKENLESS-VECTOR", namespace="default")
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
        lambda *args: [(stored.memory_id, 1, 0.0, "cosine_distance")],
    )
    monkeypatch.setattr(
        recall_module,
        "_fts_hits",
        lambda *args: (_ for _ in ()).throw(AssertionError("FTS must not run")),
    )

    hit = recall_module.recall("!!!", namespace="default", k=1)[0]
    explanation = hit.as_dict()["explanation"]
    assert explanation["vector"]["state"] == "candidate"
    assert explanation["fts"] == {
        "state": "not_run",
        "reason": "query_has_no_fts_tokens",
        "rank": None,
        "raw_score": None,
        "metric": None,
        "lower_is_better": None,
    }

    monkeypatch.setattr(recall_module, "_fts_hits", lambda *args: [])
    monkeypatch.setattr(recall_module, "embed_available", lambda: False)
    unavailable = recall_module.recall(
        "EXPLAIN-TOKENLESS-VECTOR", namespace="default", use_vectors=True
    )
    assert unavailable.modalities["vector"] == {
        "state": "not_run",
        "reason": "embedding_unavailable",
    }


def test_zero_candidates_records_a_ran_stage_without_inventing_a_hit(haunt_env):
    """Empty FTS output is execution evidence on RecallResult, not a Hit."""
    recall_module = importlib.import_module("haunt.recall")

    hits = recall_module.recall(
        "EXPLAIN-ZERO-CANDIDATES-UNMATCHED", namespace="default", use_vectors=False
    )
    assert hits == []
    assert hits.modalities == {
        "vector": {"state": "not_run", "reason": "disabled_by_caller"},
        "fts": {"state": "ran_not_candidate", "reason": "no_fts_candidates"},
    }


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
    def __init__(self):
        self.calls: list[str] = []

    def execute(self, sql, params=None):
        self.calls.append(sql)
        if "PRAGMA table_info" in sql:
            # recall probes for content_hash before naming it: ReadOnlyStore
            # never migrates, so on a pre-v10 file the column is absent.
            return _Rows([{"name": "id"}, {"name": "content_hash"}])
        if "sqlite_master" in sql:
            return _Rows([{"name": "vec_memories"}])
        assert "FROM vec_memories" in sql
        assert "ORDER BY distance" in sql
        assert "ORDER BY distance," not in sql
        # vec0 may choose an arbitrary order for exact equal distances; the
        # caller settles that returned candidate set deterministically. The
        # content hashes here deliberately contradict the memory-id order, so
        # this asserts the tie breaks on content and not on the id.
        return _Rows([
            {"mid": "native-z", "dist": 0.25, "chash": "aaa"},
            {"mid": "native-a", "dist": 0.25, "chash": "bbb"},
        ])


class _FallbackVecConnection:
    def execute(self, sql, params=None):
        if "PRAGMA table_info" in sql:
            return _Rows([{"name": "id"}, {"name": "content_hash"}])
        assert "FROM memories" in sql
        return _Rows([
            {"mid": "fallback", "embedding": struct.pack("1f", 1.0), "chash": "aaa"}
        ])


class _NativeVecStore:
    def __init__(self):
        self.conn = _NativeVecConnection()

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

    native_store = _NativeVecStore()
    native = recall_module._vec_hits(native_store, [0.0], "1=1", [], 8)
    fallback = recall_module._vec_hits(_FallbackVecStore(), [0.0], "1=1", [], 8)

    assert native == [
        ("native-z", 1, 0.25, "cosine_distance"),
        ("native-a", 2, 0.25, "cosine_distance"),
    ]
    assert fallback == [("fallback", 1, 1.0, "l2_distance")]
    assert all("FROM memories m" not in sql for sql in native_store.conn.calls)


def test_native_sqlite_vec_knn_uses_cosine_without_l2_fallback(tmp_path, monkeypatch):
    """The actual vec0 KNN query accepts ORDER BY distance and stays native."""
    recall_module = importlib.import_module("haunt.recall")
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunt-home"))
    monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)

    with Store("default") as store:
        # Do not involve the embedding model: vec0 itself is under test and
        # we supply its vectors below.
        first = store.observe("NATIVE-VEC-COSINE-FIRST", defer_embedding=True)
        second = store.observe("NATIVE-VEC-COSINE-SECOND", defer_embedding=True)
        if not store.vec_ok():
            pytest.skip("sqlite-vec extension is unavailable in this environment")
        if not ensure_vec_table(store.conn, 2):
            pytest.skip("sqlite-vec could not create a vec0 table")
        store.conn.execute("DELETE FROM vec_memories")
        store.conn.execute(
            "INSERT INTO vec_memories(id, embedding) VALUES (?, ?)",
            (first.memory_id, sqlite_vec.serialize_float32([1.0, 0.0])),
        )
        store.conn.execute(
            "INSERT INTO vec_memories(id, embedding) VALUES (?, ?)",
            (second.memory_id, sqlite_vec.serialize_float32([0.0, 1.0])),
        )
        store.conn.commit()
        monkeypatch.setattr(
            recall_module,
            "_deserialize",
            lambda blob: (_ for _ in ()).throw(AssertionError("L2 fallback used")),
        )
        hits = recall_module._vec_hits(store, [1.0, 0.0], "1=1", [], 2)

    assert [hit[0] for hit in hits] == [first.memory_id, second.memory_id]
    assert [hit[3] for hit in hits] == ["cosine_distance", "cosine_distance"]
    assert hits[0][2] < hits[1][2]


def test_temporal_timeline_surfaces_not_ranked_and_cli_uses_time_order(haunt_env, monkeypatch):
    """A bare temporal query has timeline semantics, never an RRF label."""
    from haunt import cli
    from haunt.planner import planned_recall
    from haunt.temporal import compile
    from tests.test_temporal_planner import NOW

    with Store("default") as store:
        stored = store.observe(
            "timeline event",
            event_time="2026-08-08T12:00:00+00:00",
            defer_embedding=True,
        )
        query = compile("what happened two weeks ago", NOW)
        hits = planned_recall("what happened two weeks ago", now=NOW, store=store)

    hit = next(hit for hit in hits if hit.memory_id == stored.memory_id)
    explanation = hit.as_dict()["explanation"]
    assert explanation["retrieval_method"] == "timeline"
    assert explanation["score_semantics"] == "not_ranked"
    assert explanation["rrf_score"] is None
    assert explanation["vector"]["state"] == "not_run"
    assert explanation["vector"]["reason"] == "timeline_time_order"
    assert explanation["fts"]["state"] == "not_run"
    assert explanation["fts"]["reason"] == "timeline_time_order"
    assert hit.score == 0.0
    assert query.temporal is True

    monkeypatch.setattr(cli, "planned_recall", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(cli, "_existing_readonly", lambda namespace: Store("default"))
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
    monkeypatch.setattr(cli, "_existing_readonly", lambda namespace: Store("default"))
    monkeypatch.setattr(
        cli, "open_existing_readonly", lambda namespace: Store("default")
    )
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


def test_cli_json_errors_are_machine_readable_and_nonzero(haunt_env):
    """JSON mode never mixes an invalid filter diagnostic with human text."""
    from haunt import cli

    result = CliRunner().invoke(
        cli.app,
        ["recall", "anything", "-n", "default", "--json", "--clock", "not-a-clock"],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["query"] == "anything"
    assert "clock must be" in payload["error"]


def test_synthetic_hit_references_remain_explicitly_legacy_and_unscored():
    """A bare compatibility Hit does not fabricate Store-derived references."""
    hit = Hit(
        memory_id="legacy-memory",
        event_id="legacy-event",
        score=0.0,
        tier="episodic",
        content="legacy content",
        role="user",
        event_time="2026-08-08T12:00:00+00:00",
        valid_from="2026-08-08T12:00:00+00:00",
        valid_to=None,
        tool_name=None,
    )
    explanation = hit.as_dict()["explanation"]
    assert explanation["references"] == {
        "correction_lineage": None,
        "correction_lineage_status": "unavailable_legacy",
        "provenance": None,
        "provenance_status": "legacy_unstructured",
    }
    assert "confidence" not in explanation


def test_synthetic_tool_class_is_untrusted_without_internal_raw_marker():
    """Compatibility-created Hits cannot accidentally launder tool residue."""
    hit = Hit(
        memory_id="tool-memory",
        event_id="tool-event",
        score=0.0,
        tier="episodic",
        content="tool residue",
        role="user",
        event_time="2026-08-08T12:00:00+00:00",
        valid_from="2026-08-08T12:00:00+00:00",
        valid_to=None,
        tool_name=None,
        recall_class="tool",
    )

    assert hit.trusted is False
    assert hit.trust_reason == "untrusted-tool-io"
    assert hit.as_dict()["explanation"]["trust"] == {
        "trusted": False,
        "reason": "untrusted-tool-io",
    }
