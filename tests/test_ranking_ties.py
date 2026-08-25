"""Canonical ordering for equal recall and timeline signals."""

from __future__ import annotations

import importlib

from haunt.recall import Hit
from haunt.store import Store, observe


def _hit(memory_id: str, *, score: float = 0.0, fts_rank: int | None = None) -> Hit:
    return Hit(
        memory_id=memory_id,
        event_id=f"event-{memory_id}",
        score=score,
        tier="episodic",
        content=memory_id,
        role="user",
        event_time="2026-08-08T12:00:00+00:00",
        valid_from="2026-08-08T12:00:00+00:00",
        valid_to=None,
        tool_name=None,
        fts_rank=fts_rank,
        fts_rank_raw=-1.0 if fts_rank is not None else None,
    )


def test_recall_ties_ignore_candidate_arrival_order(haunt_env, monkeypatch):
    """Equal RRF scores use memory IDs even when modality input is reversed."""
    first = observe("RECALL-TIE-FIRST", namespace="default")
    second = observe("RECALL-TIE-SECOND", namespace="default")
    recall_module = importlib.import_module("haunt.recall")
    candidates = [
        (second.memory_id, 1, -1.0),
        (first.memory_id, 1, -1.0),
    ]
    monkeypatch.setattr(
        recall_module,
        "_fts_hits",
        lambda conn, query, where, params, limit: candidates,
    )

    reversed_hits = recall_module.recall(
        "RECALL-TIE", namespace="default", k=2, use_vectors=False
    )
    candidates.reverse()
    forward_hits = recall_module.recall(
        "RECALL-TIE", namespace="default", k=2, use_vectors=False
    )

    expected = sorted([first.memory_id, second.memory_id])
    assert [hit.memory_id for hit in reversed_hits] == expected
    assert [hit.memory_id for hit in forward_hits] == expected
    assert [hit.final_rank for hit in forward_hits] == [1, 2]


def test_timeline_ties_use_memory_id_without_losing_time_order(haunt_env):
    """IDs settle exact timestamps only; newer events still appear first."""
    from haunt.planner import run_timeline
    from haunt.temporal import compile
    from tests.test_temporal_planner import NOW

    with Store("default") as store:
        tied_a = store.observe(
            "TIMELINE-TIE-A", event_time="2026-08-08T12:00:00+00:00"
        )
        tied_b = store.observe(
            "TIMELINE-TIE-B", event_time="2026-08-08T12:00:00+00:00"
        )
        later = store.observe(
            "TIMELINE-LATER", event_time="2026-08-08T13:00:00+00:00"
        )
        hits = run_timeline(compile("what happened two weeks ago", NOW), store)

    ids = [hit.memory_id for hit in hits]
    assert ids[0] == later.memory_id
    assert ids[1:] == sorted([tied_a.memory_id, tied_b.memory_id])
    assert [hit.final_rank for hit in hits] == [1, 2, 3]


def test_union_ties_sort_ranked_hits_but_keep_timeline_order(monkeypatch):
    """Union sorts equal ranked signals canonically and keeps chronology for time rows."""
    planner = importlib.import_module("haunt.planner")
    timeline = [_hit("timeline-later"), _hit("timeline-earlier")]
    ranked = [
        _hit("rank-z", score=1 / 61, fts_rank=1),
        _hit("rank-a", score=1 / 61, fts_rank=1),
    ]
    monkeypatch.setattr(planner, "run_timeline", lambda *args, **kwargs: timeline)
    monkeypatch.setattr(planner, "run_recall", lambda *args, **kwargs: ranked)

    hits = planner.run_union(object(), Store.__new__(Store), k=8)

    assert [hit.memory_id for hit in hits] == [
        "rank-a",
        "rank-z",
        "timeline-later",
        "timeline-earlier",
    ]
    assert [hit.final_rank for hit in hits] == [1, 2, 3, 4]


def test_dashboard_all_namespace_ties_use_namespace_then_memory_id(haunt_env, monkeypatch):
    """Cross-namespace ties do not depend on registry or recall arrival order."""
    from haunt import dashboard
    from tests.dashutil import make_dash_client

    observe("DASHBOARD-TIE-ALPHA", namespace="alpha")
    observe("DASHBOARD-TIE-BETA", namespace="beta")
    by_namespace = {
        "alpha": [
            _hit("alpha-z", score=1 / 61, fts_rank=1),
            _hit("alpha-a", score=1 / 61, fts_rank=1),
        ],
        "beta": [_hit("beta-a", score=1 / 61, fts_rank=1)],
    }
    monkeypatch.setattr(
        dashboard,
        "recall",
        lambda query, namespace, **kwargs: by_namespace[namespace],
    )

    response = make_dash_client().get("/api/recall?q=DASHBOARD-TIE")
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert [(hit["namespace"], hit["memory_id"]) for hit in hits] == [
        ("alpha", "alpha-a"),
        ("alpha", "alpha-z"),
        ("beta", "beta-a"),
    ]
    assert [hit["explanation"]["final_rank"] for hit in hits] == [1, 2, 3]
