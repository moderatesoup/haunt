"""Tests for the all-namespaces recall API: GET /api/recall?q=&k=&tier="""

from __future__ import annotations

import pytest

from haunt.paths import ensure_layout, namespace_db_path
from haunt.store import Store, init_registry, observe


@pytest.fixture
def recall_all_env(tmp_path, monkeypatch):
    """Exercise dashboard fan-out without downloading an embedding model."""
    home = tmp_path / "haunt-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed

    embed.reset()
    ensure_layout()
    init_registry()
    yield home
    embed.reset()


@pytest.fixture
def multi_ns_client(recall_all_env):
    """Set up two namespaces with distinct memories, return test client."""
    from tests.dashutil import make_dash_client

    observe(
        "the quantum flux capacitor enables time travel",
        namespace="alpha",
        role="user",
        origin="cli",
        tier="episodic",
    )
    observe(
        "alpha-only memory about neural networks",
        namespace="alpha",
        role="assistant",
        origin="mcp",
        tier="semantic",
    )
    observe(
        "the quantum flux capacitor was invented by Doc Brown",
        namespace="beta",
        role="user",
        origin="cursor-hook",
        tier="episodic",
    )
    observe(
        "beta-only memory about photosynthesis",
        namespace="beta",
        role="assistant",
        origin="cli",
        tier="semantic",
    )
    return make_dash_client()


def test_all_ns_recall_returns_hits_from_both(multi_ns_client):
    """Searching across all namespaces returns hits tagged with correct namespace."""
    r = multi_ns_client.get("/api/recall?q=quantum+flux+capacitor")
    assert r.status_code == 200
    data = r.json()
    assert "hits" in data
    hits = data["hits"]
    assert len(hits) >= 2

    namespaces_seen = {h["namespace"] for h in hits}
    assert "alpha" in namespaces_seen
    assert "beta" in namespaces_seen

    for h in hits:
        assert "namespace" in h
        assert "memory_id" in h
        assert "score" in h
        assert "tier" in h


def test_all_ns_recall_namespace_field_correct(multi_ns_client):
    """Each hit's namespace field matches where it actually lives."""
    r = multi_ns_client.get("/api/recall?q=neural+networks")
    data = r.json()
    hits = data["hits"]
    neural_hits = [h for h in hits if "neural" in (h.get("content") or "").lower()]
    assert len(neural_hits) >= 1
    for h in neural_hits:
        assert h["namespace"] == "alpha"


def test_all_ns_recall_empty_query_safe(multi_ns_client):
    """Empty query returns empty hits, no error."""
    r = multi_ns_client.get("/api/recall?q=")
    assert r.status_code == 200
    data = r.json()
    assert data["hits"] == []


def test_all_ns_recall_no_query_param(multi_ns_client):
    """Missing q param returns empty hits."""
    r = multi_ns_client.get("/api/recall")
    assert r.status_code == 200
    data = r.json()
    assert data["hits"] == []


def test_all_ns_recall_unknown_tier_safe(multi_ns_client):
    """Unknown tier filter returns empty hits (no crash)."""
    r = multi_ns_client.get("/api/recall?q=quantum&tier=nonexistent")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["hits"], list)


def test_all_ns_recall_tier_filter(multi_ns_client):
    """Tier filter limits results to matching tier."""
    r = multi_ns_client.get("/api/recall?q=quantum+flux&tier=episodic")
    assert r.status_code == 200
    data = r.json()
    for h in data["hits"]:
        assert h["tier"] == "episodic"


def test_all_ns_recall_k_param(multi_ns_client):
    """k is a per-namespace limit; endpoint does not invent a global rank."""
    r = multi_ns_client.get("/api/recall?q=quantum&k=1")
    assert r.status_code == 200
    data = r.json()
    assert data["ranking_scope"] == "per_namespace"
    assert all(len(group["hits"]) <= 1 for group in data["namespace_groups"])
    assert len(data["hits"]) <= len(data["namespace_groups"])


def test_all_ns_recall_includes_origin(multi_ns_client):
    """Hits include the origin field."""
    r = multi_ns_client.get("/api/recall?q=quantum+flux+capacitor")
    assert r.status_code == 200
    data = r.json()
    for h in data["hits"]:
        assert "origin" in h


def test_per_ns_recall_includes_namespace(multi_ns_client):
    """Per-namespace recall also includes namespace in response."""
    r = multi_ns_client.get("/api/namespace/alpha/recall?q=quantum")
    assert r.status_code == 200
    data = r.json()
    for h in data["hits"]:
        assert h["namespace"] == "alpha"
    assert data["ranking_scope"] == "namespace"


def test_dashboard_recall_rejects_invalid_clock_and_temporal_query(multi_ns_client):
    """Bad recall input has the same JSON error envelope as MCP recall."""
    bad_clock = multi_ns_client.get(
        "/api/namespace/alpha/recall?q=quantum&clock=wrong-clock"
    )
    assert bad_clock.status_code == 400
    assert bad_clock.json() == {
        "ok": False,
        "code": "invalid_recall_request",
        "error": bad_clock.json()["error"],
        "query": "quantum",
        "namespace": "alpha",
    }
    assert "clock must be" in bad_clock.json()["error"]

    bad_date = multi_ns_client.get("/api/recall?q=what+happened+on+2026-02-30")
    assert bad_date.status_code == 400
    assert bad_date.json()["ok"] is False
    assert bad_date.json()["query"] == "what happened on 2026-02-30"
    assert "invalid date" in bad_date.json()["error"]


def test_all_ns_recall_surfaces_corrupt_namespace_errors(recall_all_env):
    """#55: one good ns + one corrupt registered DB is not a clean hits-only 200.

    Partial success is ok: hits from the good namespace stay, but the payload
    must include a non-empty errors list with namespace + error string.
    Deleting the errors field from the JSON fails this test.
    """
    from tests.dashutil import make_dash_client

    canary = "CANARY-RECALL-ALL-55"
    observe(canary, namespace="goodns", role="user")
    observe("this db will be overwritten with garbage", namespace="badns", role="user")
    db = namespace_db_path("badns")
    assert db.exists(), f"expected registered db at {db}"
    db.write_text("GARBAGE")

    client = make_dash_client()
    r = client.get(f"/api/recall?q={canary}")
    assert r.status_code == 200
    data = r.json()
    assert "hits" in data
    assert "errors" in data, (
        "GET /api/recall must include an errors field when a namespace fails; "
        "a hits-only body looks like success"
    )
    assert data["errors"], (
        "errors must be non-empty when a registered DB is corrupt "
        f"(body={data!r})"
    )
    for err in data["errors"]:
        assert err.get("namespace"), f"error entry missing namespace: {err!r}"
        assert err.get("code") == "retrieval_backend_error", err
        assert isinstance(err.get("error"), str) and err["error"].strip(), (
            f"error entry missing error string: {err!r}"
        )
    err_ns = {err["namespace"] for err in data["errors"]}
    assert "badns" in err_ns
    assert "goodns" not in err_ns
    groups = {group["namespace"]: group for group in data["namespace_groups"]}
    assert groups["badns"]["hits"] == []
    assert groups["badns"]["error"]["code"] == "retrieval_backend_error"
    assert any(canary in (h.get("content") or "") for h in data["hits"])
    assert all(h.get("namespace") != "badns" for h in data["hits"])


def test_do_recall_surfaces_errors_in_recall_meta():
    """#55: recallMeta must not only say 'N hits (all namespaces)'."""
    from haunt.dashboard import HTML

    assert 'recallMeta").textContent=(data.hits||[]).length+" hits"+(ALL_NS?" (all namespaces)":"")' not in HTML
    assert "data.errors" in HTML
    assert "failed" in HTML
    assert "recallMeta" in HTML
