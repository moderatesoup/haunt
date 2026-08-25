"""Mutation-quality tests for retrieval correctness bugs.

Not LME. Does not assert ranking weights — only k, FTS tokens, UTC order,
and exact procedure names.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from haunt.planner import execute, plan
from haunt.recall import _fts_match_query, recall
from haunt.store import Store
from haunt.temporal import compile
from haunt.util import clamp_k, format_iso, iso_or_now, now_iso, utc_iso

NOW = datetime(2026, 8, 22, 15, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def fts_env(tmp_path, monkeypatch):
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield home
    embed.reset()


def test_clamp_k_bounds():
    assert clamp_k(2) == 2
    assert clamp_k(0) == 1
    assert clamp_k(-3) == 1
    assert clamp_k(200) == 100
    assert clamp_k("8") == 8
    assert clamp_k(None) == 8


def test_execute_temporal_honors_k(fts_env):
    """Bare temporal compile + execute(k=2) must not inflate to max(k, 50)."""
    with Store("klimit") as st:
        for i in range(8):
            st.observe(
                f"standup notes day slot {i}",
                event_time=f"2026-08-08T{10 + i:02d}:00:00+00:00",
                origin="test",
            )
        tq = compile("what happened two weeks ago", NOW)
        assert tq.temporal
        assert plan(tq) == "timeline"
        hits = execute(tq, st, k=2)
        assert len(hits) <= 2
        assert len(hits) == 2


def test_fts_query_tokenizes_unicode_like_index(fts_env):
    """Query-side tokenize must keep Unicode word chars the FTS index stores."""
    text = "Пароль от хранилища лежит в сейфе TOK-UNI-77"
    # Exact stored form — porter does not stem Russian.
    query = "хранилища"
    match = _fts_match_query(query)
    assert match is not None
    assert query in match
    assert _fts_match_query("東京サーバー") is not None
    assert _fts_match_query("only-ascii") is not None

    with Store("unicode") as st:
        wrote = st.observe(text, role="user", origin="test")
        hits = recall(query, store=st, k=8)
        assert hits, "non-Latin query must hit a matching observed memory"
        assert any(h.memory_id == wrote.memory_id for h in hits)
        assert any(query in h.content for h in hits)

        jp = st.observe("東京 サーバー の設定", role="user", origin="test")
        jp_hits = recall("東京 サーバー", store=st, k=8)
        assert jp_hits, "CJK query must hit a matching observed memory"
        assert any(h.memory_id == jp.memory_id for h in jp_hits)


def test_utc_iso_keeps_microseconds():
    """#67: new clocks retain microseconds and stay UTC."""
    with_us = datetime(2026, 8, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
    whole = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert utc_iso(with_us) == "2026-08-01T12:00:00.123456+00:00"
    assert utc_iso(whole) == "2026-08-01T12:00:00.000000+00:00"
    offset = datetime(2026, 8, 1, 10, 0, 0, 1, tzinfo=timezone.utc).astimezone(
        timezone.utc
    )
    assert utc_iso(offset).endswith("+00:00")
    stamped = now_iso()
    assert stamped.endswith("+00:00")
    parsed = datetime.fromisoformat(stamped)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_offset_timestamps_sort_by_utc_not_text(fts_env):
    """Offsets that mis-order as text must store/order as UTC."""
    later_raw = "2026-08-01T10:00:00-08:00"  # 18:00 UTC
    earlier_raw = "2026-08-01T15:00:00+00:00"  # 15:00 UTC
    assert later_raw < earlier_raw, "precondition: text sort is wrong"

    assert iso_or_now(later_raw) == "2026-08-01T18:00:00.000000+00:00"
    assert iso_or_now(earlier_raw) == "2026-08-01T15:00:00.000000+00:00"
    assert format_iso(later_raw) == "2026-08-01T18:00:00.000000+00:00"

    with Store("tzorder") as st:
        later = st.observe("later west-coast event", event_time=later_raw, origin="test")
        earlier = st.observe("earlier utc event", event_time=earlier_raw, origin="test")
        rows = st.events()
        ids = [r["id"] for r in rows]
        assert ids[0] == later.event_id
        assert ids[1] == earlier.event_id
        by_id = {r["id"]: r for r in rows}
        assert by_id[later.event_id]["event_time"] == "2026-08-01T18:00:00.000000+00:00"
        assert by_id[earlier.event_id]["event_time"] == "2026-08-01T15:00:00.000000+00:00"
        assert by_id[later.event_id]["event_time"] > by_id[earlier.event_id]["event_time"]


def test_procedure_get_is_exact_name_not_like(fts_env):
    """A procedure named deploy must not match query de% (LIKE wildcard)."""
    with Store("procs") as st:
        st.procedure_write("deploy", "git pull && make deploy", origin="test")
        hit = st.procedure_get("deploy")
        assert hit is not None
        assert hit["name"] == "deploy"
        assert st.procedure_get("de%") is None
        assert st.procedure_get("de") is None
        assert st.procedure_get("%deploy%") is None
