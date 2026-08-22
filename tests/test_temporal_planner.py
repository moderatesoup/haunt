"""Planner / clock filter tests on a throwaway Store (not LME)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from haunt.planner import (
    UNRESOLVED_CLOCK_FALLBACK,
    execute,
    has_topical_residue,
    plan,
    planned_recall,
    run_recall,
    run_timeline,
    run_union,
)
from haunt.recall import recall
from haunt.store import Store
from haunt.temporal import compile

NOW = datetime(2026, 8, 22, 15, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def fts_env(tmp_path, monkeypatch):
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("LORE_HOME", raising=False)
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield home
    embed.reset()


def _set_ts(store: Store, event_id: str, ts: str) -> None:
    store.conn.execute("UPDATE events SET ts=? WHERE id=?", (ts, event_id))
    store.conn.commit()


def test_plan_topical_is_recall_bare_temporal_is_timeline():
    assert compile("Azure architecture", NOW).temporal is False
    assert plan(compile("Azure architecture", NOW)) == "recall"
    tq = compile("Azure two weeks ago", NOW)
    assert tq.temporal
    assert "Azure" in tq.cleaned_query
    assert has_topical_residue(tq.cleaned_query)
    assert plan(tq) == "recall"
    bare = compile("two weeks ago", NOW)
    assert bare.temporal
    assert plan(bare) == "timeline"
    happened = compile("what happened two weeks ago", NOW)
    assert happened.temporal
    assert not has_topical_residue(happened.cleaned_query)
    assert plan(happened) == "timeline"
    assert plan(compile("during the past month", NOW)) == "timeline"


def test_write_time_vs_event_time_select_different_rows(fts_env):
    with Store("clocks") as st:
        wrote = st.observe(
            "I told you about the lighthouse lamp",
            event_time="2026-01-15T12:00:00+00:00",
            origin="test",
        )
        _set_ts(st, wrote.event_id, "2026-08-20T09:00:00+00:00")
        happened = st.observe(
            "the lighthouse lamp failed during the storm",
            event_time="2026-08-20T18:00:00+00:00",
            origin="test",
        )
        _set_ts(st, happened.event_id, "2026-08-21T09:00:00+00:00")

        win_since, win_until = "2026-08-20T00:00:00+00:00", "2026-08-20T23:59:59+00:00"
        by_event = st.events(since=win_since, until=win_until, clock="event_time")
        by_storage = st.events(since=win_since, until=win_until, clock="storage_time")
        by_write_alias = st.events(since=win_since, until=win_until, clock="write_time")
        event_ids = {r["id"] for r in by_event}
        storage_ids = {r["id"] for r in by_storage}
        alias_ids = {r["id"] for r in by_write_alias}
        assert happened.event_id in event_ids
        assert wrote.event_id not in event_ids
        assert wrote.event_id in storage_ids
        assert happened.event_id not in storage_ids
        assert alias_ids == storage_ids

        rec_event = recall(
            "lighthouse",
            since=win_since,
            until=win_until,
            clock="event_time",
            store=st,
            k=8,
        )
        rec_storage = recall(
            "lighthouse",
            since=win_since,
            until=win_until,
            clock="storage_time",
            store=st,
            k=8,
        )
        rec_write_alias = recall(
            "lighthouse",
            since=win_since,
            until=win_until,
            clock="write_time",
            store=st,
            k=8,
        )
        assert {h.event_id for h in rec_event} == {happened.event_id}
        assert {h.event_id for h in rec_storage} == {wrote.event_id}
        assert {h.event_id for h in rec_write_alias} == {wrote.event_id}


def test_nontemporal_planned_recall_matches_bare_recall(fts_env):
    with Store("plain") as st:
        st.observe("Azure architecture decision record", origin="test")
        st.observe("unrelated grocery list", origin="test")
        q = "Azure architecture"
        tq = compile(q, NOW)
        assert tq.temporal is False
        bare = recall(q, store=st, k=8)
        planned = planned_recall(q, now=NOW, store=st, k=8)
        assert [h.memory_id for h in planned] == [h.memory_id for h in bare]
        assert all(h.event_time for h in planned)


def test_windowed_recall_excludes_rows_outside_compiled_interval(fts_env):
    with Store("window") as st:
        inside = st.observe(
            "Azure region failover notes",
            event_time="2026-08-08T12:00:00+00:00",
            origin="test",
        )
        _set_ts(st, inside.event_id, "2026-08-08T12:00:00+00:00")
        outside = st.observe(
            "Azure region failover notes from last year",
            event_time="2025-08-08T12:00:00+00:00",
            origin="test",
        )
        _set_ts(st, outside.event_id, "2025-08-08T12:00:00+00:00")
        tq = compile("Azure two weeks ago", NOW)
        assert tq.temporal
        assert (NOW.date() - tq.start.date()).days == 14
        hits = run_recall(tq, st, k=8)
        ids = {h.event_id for h in hits}
        assert inside.event_id in ids
        assert outside.event_id not in ids
        for h in hits:
            assert tq.start.isoformat(timespec="seconds") <= h.event_time
            assert h.event_time <= tq.end.isoformat(timespec="seconds")


def test_union_includes_timeline_row_that_windowed_fts_misses(fts_env):
    """Audit failure mode: FTS has no overlap; timeline still lists the row."""
    with Store("union") as st:
        gold = st.observe(
            "Deployed the billing hotfix to production",
            event_time="2026-03-04T16:00:00+00:00",
            origin="test",
        )
        _set_ts(st, gold.event_id, "2026-03-04T16:00:00+00:00")
        other = st.observe(
            "Azure architecture notes from winter",
            event_time="2026-01-10T12:00:00+00:00",
            origin="test",
        )
        _set_ts(st, other.event_id, "2026-01-10T12:00:00+00:00")
        tq = compile("Azure on March 4", NOW)
        assert tq.temporal
        assert "Azure" in tq.cleaned_query
        assert tq.start.date().isoformat() == "2026-03-04"

        fts_hits = run_recall(tq, st, k=8)
        assert gold.event_id not in {h.event_id for h in fts_hits}

        tl_hits = run_timeline(tq, st, limit=20)
        assert gold.event_id in {h.event_id for h in tl_hits}

        union = run_union(tq, st, k=8)
        assert gold.event_id in {h.event_id for h in union}
        assert other.event_id not in {h.event_id for h in union}

        # union is opt-in only; topical residue defaults to recall.
        assert plan(tq) == "recall"
        default = execute(tq, st, k=8)
        assert gold.event_id not in {h.event_id for h in default}
        opt_in = execute(tq, st, strategy="union", k=8)
        assert gold.event_id in {h.event_id for h in opt_in}


def test_unresolved_clock_does_not_apply_storage_filter(fts_env):
    """Mixed say+happened stays unresolved and must not filter events.ts."""
    assert "storage_time" in UNRESOLVED_CLOCK_FALLBACK or "events.ts" in UNRESOLVED_CLOCK_FALLBACK
    with Store("unresolved") as st:
        by_event = st.observe(
            "I said the outage happened then",
            event_time="2026-08-08T12:00:00+00:00",
            origin="test",
        )
        _set_ts(st, by_event.event_id, "2026-08-22T12:00:00+00:00")
        by_storage = st.observe(
            "I said the outage happened then (ingested later)",
            event_time="2025-01-01T12:00:00+00:00",
            origin="test",
        )
        _set_ts(st, by_storage.event_id, "2026-08-08T12:00:00+00:00")
        tq = compile("What did I say happened two weeks ago?", NOW)
        assert tq.clock == "unresolved"
        hits = run_timeline(tq, st, limit=20)
        ids = {h.event_id for h in hits}
        assert by_event.event_id in ids
        assert by_storage.event_id not in ids


def test_mentioned_two_weeks_ago_filters_event_time_not_storage(fts_env):
    with Store("mentioned") as st:
        conversation = st.observe(
            "I mentioned the lighthouse lamp",
            event_time="2026-08-08T12:00:00+00:00",
            origin="test",
        )
        _set_ts(st, conversation.event_id, "2026-08-22T09:00:00+00:00")
        ingested = st.observe(
            "I mentioned the lighthouse lamp at ingest",
            event_time="2025-01-01T12:00:00+00:00",
            origin="test",
        )
        _set_ts(st, ingested.event_id, "2026-08-08T12:00:00+00:00")
        tq = compile("I mentioned the lighthouse two weeks ago", NOW)
        assert tq.clock == "event_time"
        assert plan(tq) == "recall"
        hits = run_recall(tq, st, k=8)
        ids = {h.event_id for h in hits}
        assert conversation.event_id in ids
        assert ingested.event_id not in ids


def test_default_clock_is_event_time_when_unspecified(fts_env):
    with Store("defaultclk") as st:
        row = st.observe(
            "unique-token ZXCLK",
            event_time="2024-03-10T10:00:00+00:00",
            origin="test",
        )
        _set_ts(st, row.event_id, "2026-08-22T10:00:00+00:00")
        # since/until without clock must keep pre-compiler event_time behavior
        hits = recall(
            "ZXCLK",
            since="2024-03-10T00:00:00+00:00",
            until="2024-03-10T23:59:59+00:00",
            store=st,
        )
        assert hits and hits[0].event_id == row.event_id
        none = recall(
            "ZXCLK",
            since="2026-08-22T00:00:00+00:00",
            until="2026-08-22T23:59:59+00:00",
            store=st,
        )
        assert all(h.event_id != row.event_id for h in none)
