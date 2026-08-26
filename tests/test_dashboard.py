"""Tests for dashboard routes — read and mutation (delete hits the store)."""

from __future__ import annotations

import inspect

import pytest

from haunt.store import Store, observe


@pytest.fixture
def dash_client(haunt_env):
    """HTTPX test client for the dashboard app."""
    from tests.dashutil import make_dash_client

    observe("dashboard test memory DASH-CANARY-42", namespace="default", role="user")
    return make_dash_client()


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


def test_api_recall_uses_planner_for_bare_temporal_query(dash_client, monkeypatch):
    """Dashboard matches CLI/MCP timeline semantics instead of raw recall."""
    from haunt import dashboard
    from haunt.recall import Hit

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
        vector_stage={"state": "not_run", "reason": "timeline_time_order"},
        fts_stage={"state": "not_run", "reason": "timeline_time_order"},
    )
    calls: list[str] = []

    def planned(query, **kwargs):
        calls.append(query)
        return [timeline]

    monkeypatch.setattr(dashboard, "planned_recall", planned)
    response = dash_client.get(
        "/api/namespace/default/recall?q=what+happened+two+weeks+ago"
    )
    assert response.status_code == 200
    assert calls == ["what happened two weeks ago"]
    explanation = response.json()["hits"][0]["explanation"]
    assert explanation["retrieval_method"] == "timeline"
    assert explanation["score_semantics"] == "not_ranked"


def test_api_recall_temporal_surface_matches_planner_and_mcp(dash_client, monkeypatch):
    """A real endpoint shares CLI/MCP's planned timeline semantics."""
    import json

    from haunt.mcp_server import memory_recall
    from haunt.planner import planned_recall

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    with Store("default") as store:
        stored = store.observe(
            "dashboard temporal parity",
            event_time="2026-08-08T12:00:00+00:00",
            defer_embedding=True,
        )
        expected = planned_recall("what happened on 2026-08-08", store=store)

    dashboard_result = dash_client.get(
        "/api/namespace/default/recall?q=what+happened+on+2026-08-08"
    )
    assert dashboard_result.status_code == 200
    dashboard_hits = dashboard_result.json()["hits"]
    mcp_hits = json.loads(memory_recall(query="what happened on 2026-08-08"))["hits"]

    expected_ids = [hit.memory_id for hit in expected]
    assert stored.memory_id in expected_ids
    assert [hit["memory_id"] for hit in dashboard_hits] == expected_ids
    assert [hit["memory_id"] for hit in mcp_hits] == expected_ids
    assert dashboard_hits[0]["explanation"]["score_semantics"] == "not_ranked"


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
    err = (resp.json().get("error") or "")
    assert "memory" in err and "not found" in err
    assert "unknown namespace" not in err


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
