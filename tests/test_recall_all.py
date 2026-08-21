"""Tests for the all-namespaces recall API: GET /api/recall?q=&k=&tier="""

from __future__ import annotations

import pytest

from haunt.store import Store, observe


@pytest.fixture
def multi_ns_client(lore_env):
    """Set up two namespaces with distinct memories, return test client."""
    from starlette.testclient import TestClient
    from haunt.dashboard import app

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
    return TestClient(app)


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
    """k parameter limits total results."""
    r = multi_ns_client.get("/api/recall?q=quantum&k=1")
    assert r.status_code == 200
    data = r.json()
    assert len(data["hits"]) <= 1


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
