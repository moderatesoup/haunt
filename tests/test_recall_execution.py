"""Machine-surface backend errors and zero-hit recall execution evidence."""

from __future__ import annotations

import importlib
import json

import pytest
import sqlite_vec
from typer.testing import CliRunner

from haunt.paths import ensure_layout, namespace_db_path
from haunt.store import Store, ensure_vec_table, init_registry


@pytest.fixture
def fts_recall_env(tmp_path, monkeypatch):
    """A no-model namespace for recall serialization tests."""
    home = tmp_path / "haunt-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    from haunt import embed

    embed.reset()
    ensure_layout()
    init_registry()
    yield home
    embed.reset()


@pytest.fixture
def native_vec_recall_env(tmp_path, monkeypatch):
    """A real vec0 namespace without loading an embedding model."""
    home = tmp_path / "haunt-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)
    # Store only loads sqlite-vec here; recall's embedding calls are mocked.
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    from haunt import embed

    embed.reset()
    ensure_layout()
    init_registry()
    yield home
    embed.reset()


def _surface_payloads(query: str, namespace: str = "default") -> list[dict]:
    """Call each machine surface with the same request."""
    from haunt import cli
    from haunt.mcp_server import memory_recall
    from tests.dashutil import make_dash_client

    cli_result = CliRunner().invoke(
        cli.app, ["recall", query, "-n", namespace, "--json"]
    )
    assert cli_result.exit_code == 0, cli_result.output
    dashboard_result = make_dash_client().get(
        f"/api/namespace/{namespace}/recall", params={"q": query}
    )
    assert dashboard_result.status_code == 200, dashboard_result.text
    return [
        json.loads(cli_result.stdout),
        json.loads(memory_recall(query=query, namespace=namespace)),
        dashboard_result.json(),
    ]


def test_execution_metadata_survives_zero_and_nonzero_recall_surfaces(fts_recall_env):
    """CLI, MCP, and dashboard keep stage evidence even with no serialized hit."""
    with Store("default") as store:
        store.observe(
            "PRESENT-EXECUTION-CANARY",
            event_time="2026-08-08T12:00:00+00:00",
            defer_embedding=True,
        )

    expected = {
        "!!!": ("recall", "not_run", "query_has_no_fts_tokens"),
        "MISSING-EXECUTION-CANARY": (
            "recall",
            "ran_not_candidate",
            "no_fts_candidates",
        ),
        "NORESULTTOPIC what happened on 2026-08-08": (
            "recall",
            "ran_not_candidate",
            "no_fts_candidates",
        ),
        "what happened on 2026-08-08": (
            "timeline",
            "not_run",
            "timeline_time_order",
        ),
        "PRESENT-EXECUTION-CANARY": (
            "recall",
            "candidate",
            "returned_fts_candidates",
        ),
    }
    for query, (strategy, fts_state, fts_reason) in expected.items():
        payloads = _surface_payloads(query)
        executions = [payload["execution"] for payload in payloads]
        assert all(execution["version"] == 1 for execution in executions)
        assert all(execution["strategy"] == strategy for execution in executions)
        assert all(
            execution["modalities"]["fts"]
            == {"state": fts_state, "reason": fts_reason}
            for execution in executions
        ), (query, executions)
        assert all(
            execution["modalities"]["vector"]
            == {"state": "not_run", "reason": "embedding_unavailable"}
            if strategy == "recall"
            else execution["modalities"]["vector"]
            == {"state": "not_run", "reason": "timeline_time_order"}
            for execution in executions
        )
        if query == "PRESENT-EXECUTION-CANARY":
            assert all(
                payload["hits"][0]["explanation"]["ordering"]
                == {"primary": "rrf_score_desc", "ties": "content_hash_asc_then_memory_id_asc"}
                for payload in payloads
            )


def test_execution_metadata_covers_disabled_vectors_and_legacy_lists(fts_recall_env):
    """Disabled is distinct from unavailable; an old plain list gains no metadata."""
    recall_module = importlib.import_module("haunt.recall")
    with Store("default") as store:
        store.observe("DISABLED-EXECUTION-CANARY", defer_embedding=True)
        disabled = recall_module.recall(
            "DISABLED-EXECUTION-CANARY", store=store, use_vectors=False
        )
    assert disabled.execution["modalities"]["vector"] == {
        "state": "not_run",
        "reason": "disabled_by_caller",
    }
    assert recall_module.execution_metadata([disabled[0]]) is None
    copied = recall_module.execution_metadata(disabled)
    assert copied is not None
    copied["modalities"]["vector"]["reason"] = "tampered"
    assert recall_module.execution_metadata(disabled)["modalities"]["vector"] == {
        "state": "not_run",
        "reason": "disabled_by_caller",
    }


def test_union_execution_is_explicit_and_keeps_component_evidence(fts_recall_env):
    """Union does not discard the timeline/recall execution distinction."""
    from haunt.planner import run_union
    from haunt.temporal import compile as compile_temporal

    with Store("default") as store:
        store.observe(
            "union timeline event",
            event_time="2026-08-08T12:00:00+00:00",
            defer_embedding=True,
        )
        union = run_union(
            compile_temporal("NORESULTTOPIC what happened on 2026-08-08"),
            store,
        )
    execution = union.execution
    assert execution["version"] == 1
    assert execution["strategy"] == "union"
    assert execution["components"]["timeline"]["strategy"] == "timeline"
    assert execution["components"]["recall"]["strategy"] == "recall"
    assert execution["components"]["recall"]["modalities"]["fts"] == {
        "state": "ran_not_candidate",
        "reason": "no_fts_candidates",
    }


def test_all_namespace_groups_keep_per_namespace_execution(fts_recall_env):
    """Every registered namespace gets local zero-hit execution evidence."""
    from tests.dashutil import make_dash_client

    with Store("alpha") as store:
        store.observe("ALPHA-EXECUTION-CANARY", defer_embedding=True)
    with Store("beta") as store:
        store.observe("BETA-EXECUTION-CANARY", defer_embedding=True)
    # Register a namespace with no events. It still runs the planned path and
    # therefore has a truthful v1 execution record rather than being skipped.
    with Store("empty"):
        pass

    response = make_dash_client().get("/api/recall?q=MISSING-EXECUTION-CANARY")
    assert response.status_code == 200
    groups = response.json()["namespace_groups"]
    assert [group["namespace"] for group in groups] == ["alpha", "beta", "empty"]
    for group in groups:
        assert group["hits"] == []
        assert group["execution"]["version"] == 1
        assert group["execution"]["modalities"]["fts"] == {
            "state": "ran_not_candidate",
            "reason": "no_fts_candidates",
        }


def test_timeline_bounded_ties_select_events_then_order_materialized_memories(
    fts_recall_env,
):
    """A bounded equal-time page cannot claim it chose all ties by memory ID."""
    from haunt.planner import run_timeline
    from haunt.temporal import compile as compile_temporal

    with Store("default") as store:
        tied = [
            store.observe(
                f"bounded tie {index}",
                event_time="2026-08-08T12:00:00+00:00",
                defer_embedding=True,
            )
            for index in range(5)
        ]
        later = store.observe(
            "bounded later",
            event_time="2026-08-08T13:00:00+00:00",
            defer_embedding=True,
        )
        hits = run_timeline(
            compile_temporal("what happened on 2026-08-08"), store, limit=3
        )

    # The SQL event page contains the later event plus the first two event IDs
    # among the five equal-time events. Only that selected set is subsequently
    # sorted by memory ID after materialization.
    selected_ties = sorted(tied, key=lambda result: result.event_id)[:2]
    assert [hit.memory_id for hit in hits] == [
        later.memory_id,
        *sorted(result.memory_id for result in selected_ties),
    ]
    ordering = hits[1].as_dict()["explanation"]["ordering"]
    assert ordering == {
        "primary": "selected_clock_desc",
        "ties": "event_id_asc_at_bounded_event_selection_then_memory_id_asc_after_materialization",
    }


def _prepare_dimension_mismatch(monkeypatch) -> None:
    """Make a real vec0 table whose dimension disagrees with the query vector."""
    recall_module = importlib.import_module("haunt.recall")
    with Store("default") as store:
        result = store.observe("DIMENSION-MISMATCH-CANARY", defer_embedding=True)
        if not store.vec_ok() or not ensure_vec_table(store.conn, 2):
            pytest.skip("sqlite-vec extension is unavailable in this environment")
        store.conn.execute(
            "INSERT INTO vec_memories(id, embedding) VALUES (?, ?)",
            (result.memory_id, sqlite_vec.serialize_float32([1.0, 0.0])),
        )
        store.conn.commit()
    monkeypatch.setattr(Store, "ensure_current_embeddings", lambda self: None)
    monkeypatch.setattr(
        Store, "process_embedding_jobs", lambda self, *, limit=64: {"processed": 0}
    )
    monkeypatch.setattr(recall_module, "embed_available", lambda: True)
    monkeypatch.setattr(recall_module, "embed_one", lambda query: [1.0, 0.0, 0.0])


def test_real_vec_dimension_mismatch_is_structured_on_machine_surfaces(
    native_vec_recall_env, monkeypatch
):
    """A real sqlite-vec failure remains a Python raise but serializes elsewhere."""
    from haunt import cli
    from haunt.mcp_server import memory_recall
    from haunt.recall import BACKEND_ERROR_CODE, RetrievalBackendError, recall
    from tests.dashutil import make_dash_client

    _prepare_dimension_mismatch(monkeypatch)
    with Store("default") as store:
        with pytest.raises(RetrievalBackendError):
            recall("DIMENSION-MISMATCH-CANARY", store=store)

    cli_result = CliRunner().invoke(
        cli.app, ["recall", "DIMENSION-MISMATCH-CANARY", "-n", "default", "--json"]
    )
    assert cli_result.exit_code != 0
    assert len(cli_result.stdout.strip().splitlines()) == 1
    assert "Traceback" not in cli_result.stdout
    cli_payload = json.loads(cli_result.stdout)
    mcp_payload = json.loads(memory_recall(query="DIMENSION-MISMATCH-CANARY"))
    dashboard_result = make_dash_client().get(
        "/api/namespace/default/recall?q=DIMENSION-MISMATCH-CANARY"
    )
    assert dashboard_result.status_code == 500
    dashboard_payload = dashboard_result.json()
    for payload in (cli_payload, mcp_payload, dashboard_payload):
        assert payload["ok"] is False
        assert payload["code"] == BACKEND_ERROR_CODE
        assert payload["query"] == "DIMENSION-MISMATCH-CANARY"


def test_corrupt_db_is_structured_on_cli_mcp_and_dashboard(fts_recall_env):
    """Registered but corrupt databases never make JSON clients parse a traceback."""
    from haunt import cli
    from haunt.mcp_server import memory_recall
    from haunt.recall import BACKEND_ERROR_CODE
    from tests.dashutil import make_dash_client

    with Store("corrupt") as store:
        store.observe("CORRUPT-RECALL-CANARY", defer_embedding=True)
    namespace_db_path("corrupt").write_text("GARBAGE")

    cli_result = CliRunner().invoke(
        cli.app, ["recall", "CORRUPT-RECALL-CANARY", "-n", "corrupt", "--json"]
    )
    assert cli_result.exit_code != 0
    cli_payload = json.loads(cli_result.stdout)
    mcp_payload = json.loads(
        memory_recall(query="CORRUPT-RECALL-CANARY", namespace="corrupt")
    )
    dashboard_result = make_dash_client().get(
        "/api/namespace/corrupt/recall?q=CORRUPT-RECALL-CANARY"
    )
    assert dashboard_result.status_code == 500
    dashboard_payload = dashboard_result.json()
    for payload in (cli_payload, mcp_payload, dashboard_payload):
        assert payload["ok"] is False
        assert payload["code"] == BACKEND_ERROR_CODE
        assert payload["namespace"] == "corrupt"
