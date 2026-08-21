"""Tests for dashboard routes — read and mutation (delete hits the store)."""

from __future__ import annotations

import pytest

from haunt.store import Store, observe


@pytest.fixture
def dash_client(lore_env):
    """HTTPX test client for the dashboard app."""
    from starlette.testclient import TestClient
    from haunt.dashboard import app

    observe("dashboard test memory DASH-CANARY-42", namespace="default", role="user")
    return TestClient(app)


def test_index_returns_html(dash_client):
    r = dash_client.get("/")
    assert r.status_code == 200
    assert "haunt" in r.text


def test_api_namespaces(dash_client):
    r = dash_client.get("/api/namespaces")
    assert r.status_code == 200
    data = r.json()
    assert "namespaces" in data
    assert "haunt_home" in data


def test_api_namespace(dash_client):
    r = dash_client.get("/api/namespace/default")
    assert r.status_code == 200
    data = r.json()
    assert "stats" in data
    assert "events" in data
    assert "entities" in data
    assert "health" in data


def test_api_recall(dash_client):
    r = dash_client.get("/api/namespace/default/recall?q=DASH-CANARY-42")
    assert r.status_code == 200
    data = r.json()
    assert "hits" in data
    assert any("DASH-CANARY-42" in (h.get("content", "") or "") for h in data["hits"])


def test_api_browse(dash_client):
    r = dash_client.get("/api/namespace/default/browse?limit=10")
    assert r.status_code == 200
    data = r.json()
    assert "memories" in data
    assert "total" in data
    assert len(data["memories"]) > 0


def test_api_memory_detail(dash_client):
    r = observe("detail test memory UNIQUE-DETAIL-77", namespace="default", role="user")
    resp = dash_client.get(f"/api/namespace/default/memory/{r.memory_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["memory_id"] == r.memory_id
    assert data["namespace"] == "default"
    assert "db_path" in data
    assert "haunt_home" in data
    assert "entity_mentions" in data
    assert "related_memories" in data


def test_api_memory_detail_not_found(dash_client):
    resp = dash_client.get("/api/namespace/default/memory/nonexistent-id")
    assert resp.status_code == 404


def test_api_delete_memory(dash_client):
    """DELETE route must actually purge the memory from the store."""
    r = observe("DELETE-ME-CANARY-55 this must be gone", namespace="default", role="user")

    with Store("default") as st:
        mem = st.conn.execute("SELECT id FROM memories WHERE id=?", (r.memory_id,)).fetchone()
        assert mem is not None

    resp = dash_client.request("DELETE", f"/api/namespace/default/memory/{r.memory_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["fts_deleted"] is True

    with Store("default") as st:
        mem = st.conn.execute("SELECT id FROM memories WHERE id=?", (r.memory_id,)).fetchone()
        assert mem is None, "memory should be gone after DELETE"


def test_api_delete_not_found(dash_client):
    resp = dash_client.request("DELETE", "/api/namespace/default/memory/nonexistent-id")
    assert resp.status_code == 404


def test_api_procedures(dash_client):
    with Store("default") as st:
        st.procedure_write("test-proc", "step 1\nstep 2", trigger="when testing")
    r = dash_client.get("/api/namespace/default/procedures")
    assert r.status_code == 200
    data = r.json()
    assert any(p["name"] == "test-proc" for p in data["procedures"])


def test_api_worldview(dash_client):
    r = dash_client.get("/api/namespace/default/worldview")
    assert r.status_code == 200
    data = r.json()
    assert "facts" in data
    assert "names" in data
    assert "procedures" in data


def test_api_health(dash_client):
    r = dash_client.get("/api/namespace/default/health")
    assert r.status_code == 200
    data = r.json()
    assert "sqlite_vec" in data
    assert "embed" in data
    assert "stats" in data
    assert "db_path" in data
    assert "namespace" in data
