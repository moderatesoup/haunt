"""Tests for dashboard routes — read and mutation (delete hits the store)."""

from __future__ import annotations

import inspect

import pytest

from haunt.store import Store, observe


@pytest.fixture
def dash_client(haunt_env):
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


def test_run_dashboard_no_open_skips_browser(haunt_env, monkeypatch):
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


# --------------------------------------------------------------------------
# Regression: switchView must NOT destroy DOM with innerHTML
# --------------------------------------------------------------------------


class TestSwitchViewNoInnerHTMLDestruction:
    """Guard against the dead-click bug where switchView replaced view
    innerHTML with a hint string, destroying inputs/tables/ids and making
    every subsequent click throw."""

    def test_switchview_does_not_replace_view_innerhtml(self, dash_client):
        """switchView must not replace a view container's innerHTML with a
        hint string — that destroys inputs, tables, and element ids."""
        from haunt.dashboard import HTML

        assert "$('view-'+v).innerHTML=" not in HTML

    def test_html_uses_allns_hint_overlay(self, dash_client):
        """The fix should use an overlay approach (allns-hint) instead of
        replacing innerHTML."""
        from haunt.dashboard import HTML

        assert "allns-hint" in HTML
        assert "showAllNsHint" in HTML
        assert "hideAllNsHint" in HTML


# --------------------------------------------------------------------------
# Unit tests for pick_default_namespace
# --------------------------------------------------------------------------


class TestPickDefaultNamespace:
    def test_empty_list_returns_default(self):
        from haunt.dashboard import pick_default_namespace

        assert pick_default_namespace([]) == "default"

    def test_prefers_most_events(self):
        from haunt.dashboard import pick_default_namespace

        ns = [
            {"name": "aronriley", "events": 0},
            {"name": "haunt", "events": 42},
            {"name": "work", "events": 10},
        ]
        assert pick_default_namespace(ns) == "haunt"

    def test_prefers_higher_event_count(self):
        from haunt.dashboard import pick_default_namespace

        ns = [
            {"name": "alpha", "events": 5},
            {"name": "beta", "events": 100},
            {"name": "gamma", "events": 50},
        ]
        assert pick_default_namespace(ns) == "beta"

    def test_skips_zero_event_ns(self):
        from haunt.dashboard import pick_default_namespace

        ns = [
            {"name": "aronriley", "events": 0},
            {"name": "empty-too", "events": 0},
            {"name": "real-data", "events": 3},
        ]
        assert pick_default_namespace(ns) == "real-data"

    def test_all_zero_prefers_haunt(self):
        from haunt.dashboard import pick_default_namespace

        ns = [
            {"name": "aronriley", "events": 0},
            {"name": "haunt", "events": 0},
            {"name": "work", "events": 0},
        ]
        assert pick_default_namespace(ns) == "haunt"

    def test_all_zero_no_haunt_returns_first(self):
        from haunt.dashboard import pick_default_namespace

        ns = [
            {"name": "alpha", "events": 0},
            {"name": "beta", "events": 0},
        ]
        assert pick_default_namespace(ns) == "alpha"

    def test_none_events_treated_as_zero(self):
        from haunt.dashboard import pick_default_namespace

        ns = [
            {"name": "empty", "events": None},
            {"name": "real", "events": 5},
        ]
        assert pick_default_namespace(ns) == "real"


class TestApiNamespacesDefault:
    def test_api_namespaces_includes_default(self, dash_client):
        """The /api/namespaces response must include a 'default' field
        recommending which namespace to load on boot."""
        r = dash_client.get("/api/namespaces")
        assert r.status_code == 200
        data = r.json()
        assert "default" in data
        assert isinstance(data["default"], str)
        assert len(data["default"]) > 0
