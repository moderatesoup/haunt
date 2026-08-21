"""Tests for dashboard routes — read and mutation (delete hits the store)."""

from __future__ import annotations

import inspect

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


def test_no_open_still_serves(dash_client):
    """dash --no-open must still serve HTTP; the test client exercises this."""
    r = dash_client.get("/")
    assert r.status_code == 200
    assert "haunt" in r.text.lower()


def test_run_dashboard_opens_browser_by_default():
    """Mutation guard: run_dashboard must default open_browser=True.

    If someone removes browser-open, this test fails — ensuring the
    first-run experience keeps working.
    """
    from haunt.dashboard import run_dashboard

    sig = inspect.signature(run_dashboard)
    param = sig.parameters.get("open_browser")
    assert param is not None, "run_dashboard must accept open_browser parameter"
    assert param.default is True, (
        f"open_browser default must be True (got {param.default!r}); "
        "haunt dash opens the browser by default for first-run UX"
    )


def test_run_dashboard_no_open_skips_browser(lore_env, monkeypatch):
    """open_browser=False must not attempt webbrowser.open."""
    import threading

    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))

    from haunt.dashboard import run_dashboard

    original_run = None
    try:
        import uvicorn
        original_run = uvicorn.run

        def fake_run(*a, **kw):
            pass

        monkeypatch.setattr("uvicorn.run", fake_run)
        run_dashboard(open_browser=False)
        assert opened == [], "webbrowser.open should not be called when open_browser=False"
    finally:
        pass


def test_api_namespace_nonexistent_returns_404(dash_client, lore_env):
    """GET /api/namespace/{name} must return 404 for unknown namespaces, not auto-create them."""
    from haunt.store import list_namespaces

    names_before = {ns["name"] for ns in list_namespaces()}
    assert "XYZZY_NEVER_EXISTS_99" not in names_before

    r = dash_client.get("/api/namespace/XYZZY_NEVER_EXISTS_99")
    assert r.status_code == 404, (
        f"Nonexistent namespace should return 404, got {r.status_code}. "
        "A GET must not auto-create namespaces."
    )

    names_after = {ns["name"] for ns in list_namespaces()}
    created = names_after - names_before
    assert "xyzzy-never-exists-99" not in created and "XYZZY_NEVER_EXISTS_99" not in created, (
        f"GET auto-created namespace(s): {created}. "
        "Reading a nonexistent namespace must not have write side-effects."
    )


GHOST_NS = "GHOST_NEVER_EXISTS_77"

@pytest.mark.parametrize("path_suffix", [
    "",
    "/recall?q=test",
    "/browse?limit=1",
    "/memory/nonexistent-id",
    "/event/nonexistent-id/memories",
    "/procedures",
    "/worldview",
    "/health",
])
def test_all_namespace_routes_return_404_for_missing_ns(dash_client, lore_env, path_suffix):
    """Every GET route under /api/namespace/{name} must return 404 — not
    auto-create — when the namespace does not exist."""
    from haunt.store import list_namespaces, namespace_exists

    assert not namespace_exists(GHOST_NS)

    r = dash_client.get(f"/api/namespace/{GHOST_NS}{path_suffix}")
    assert r.status_code == 404, (
        f"GET /api/namespace/{GHOST_NS}{path_suffix} returned {r.status_code}, "
        "expected 404 for nonexistent namespace."
    )

    assert not namespace_exists(GHOST_NS), (
        f"GET /api/namespace/{GHOST_NS}{path_suffix} auto-created the namespace. "
        "Read-only routes must not have write side-effects."
    )


def test_delete_on_missing_namespace_returns_404(dash_client, lore_env):
    """DELETE /api/namespace/{name}/memory/{id} must 404 on missing namespace."""
    from haunt.store import namespace_exists

    assert not namespace_exists(GHOST_NS)

    r = dash_client.request("DELETE", f"/api/namespace/{GHOST_NS}/memory/nonexistent-id")
    assert r.status_code == 404

    assert not namespace_exists(GHOST_NS)
