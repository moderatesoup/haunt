"""Adversarial compile() suite. Language only — do not import the planner."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from haunt.temporal import TemporalParseError, TemporalQuery, compile

NOW = datetime(2026, 8, 22, 15, 30, 0, tzinfo=timezone.utc)


def test_compile_module_does_not_import_planner():
    src = Path(__file__).resolve().parents[1] / "src" / "haunt" / "temporal.py"
    tree = ast.parse(src.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "planner" not in alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "planner" not in node.module


def test_compile_returns_temporal_query_shape():
    tq = compile("two weeks ago", NOW)
    assert isinstance(tq, TemporalQuery)
    assert not hasattr(tq, "strategy_hint") or tq.__dict__.get("strategy_hint") in (None,)
    fields = set(tq.as_dict())
    assert "strategy_hint" not in fields
    assert "precision" not in fields
    assert fields == {
        "temporal",
        "cleaned_query",
        "start",
        "end",
        "clock",
        "granularity",
        "certainty",
        "confidence",
    }


def _day(tq: TemporalQuery):
    assert tq.start is not None and tq.end is not None
    return tq.start.date(), tq.end.date()


def test_two_weeks_ago_is_offset_day_not_range_to_now():
    tq = compile("two weeks ago", NOW)
    assert tq.temporal is True
    start_d, end_d = _day(tq)
    assert start_d == end_d == (NOW.date().fromordinal(NOW.date().toordinal() - 14))
    assert (NOW.date() - start_d).days == 14
    assert tq.granularity == "day"
    assert tq.certainty == "exact"
    assert tq.clock == "event_time"


@pytest.mark.parametrize(
    "query",
    [
        "tow weeks ago",
        "2 weeks ago",
        "two wks ago",
        "2 wks ago",
    ],
)
def test_ago_aliases_and_fuzzy_match_same_offset_day(query):
    expected = compile("two weeks ago", NOW)
    got = compile(query, NOW)
    assert got.temporal
    assert _day(got) == _day(expected)
    assert got.granularity == "day"
    assert got.certainty == "exact"


def test_tow_is_fuzzy_two_wks_is_alias():
    fuzzy = compile("tow weeks ago", NOW)
    alias = compile("2 wks ago", NOW)
    exact = compile("two weeks ago", NOW)
    assert _day(fuzzy) == _day(alias) == _day(exact)
    assert fuzzy.confidence <= 0.8
    assert alias.confidence >= 0.9


def test_last_two_weeks_is_range_to_now():
    for query in ("last two weeks", "in the past two weeks"):
        tq = compile(query, NOW)
        assert tq.temporal
        start_d, end_d = _day(tq)
        assert start_d != end_d, f"{query} must not collapse to a single day"
        assert (NOW.date() - start_d).days == 14
        assert end_d == NOW.date()
        assert tq.granularity == "week"


def test_about_two_weeks_ago_is_approximate_centered_minus_14d():
    tq = compile("about two weeks ago", NOW)
    assert tq.temporal
    assert tq.certainty == "approximate"
    assert tq.granularity == "day"
    start_d, end_d = _day(tq)
    assert start_d == end_d
    assert (NOW.date() - start_d).days == 14


def test_on_march_4_in_march_before_after():
    on = compile("on March 4", NOW)
    assert on.temporal
    assert on.granularity == "day"
    assert on.certainty == "exact"
    assert _day(on) == (
        datetime(2026, 3, 4, tzinfo=timezone.utc).date(),
        datetime(2026, 3, 4, tzinfo=timezone.utc).date(),
    )

    inn = compile("in March", NOW)
    assert inn.temporal
    assert inn.granularity == "month"
    assert inn.certainty == "exact"
    assert inn.start is not None and inn.end is not None
    assert inn.start.date() == datetime(2026, 3, 1).date()
    assert inn.end.date() == datetime(2026, 3, 31).date()

    before = compile("before March", NOW)
    assert before.temporal
    assert before.start is None
    assert before.end is not None
    assert before.end < inn.start

    after = compile("after March", NOW)
    assert after.temporal
    assert after.end is None
    assert after.start is not None
    assert after.start > inn.end


def test_iso_and_us_and_month_day_year():
    iso = compile("2024-03-04", NOW)
    us = compile("3/4/2024", NOW)
    named = compile("March 4, 2024", NOW)
    assert iso.temporal and us.temporal and named.temporal
    assert _day(iso) == _day(us) == _day(named)
    assert iso.granularity == "day"


def test_today_yesterday_this_last_units():
    today = compile("today", NOW)
    assert _day(today) == (NOW.date(), NOW.date())

    yest = compile("yesterday", NOW)
    assert (NOW.date() - yest.start.date()).days == 1

    last_week = compile("last week", NOW)
    assert last_week.granularity == "week"
    assert last_week.start.date().weekday() == 0
    assert (last_week.end.date() - last_week.start.date()).days == 6
    assert last_week.end.date() < NOW.date()

    this_month = compile("this month", NOW)
    assert this_month.granularity == "month"
    assert this_month.start.date() == datetime(2026, 8, 1).date()

    last_year = compile("last year", NOW)
    assert last_year.granularity == "year"
    assert last_year.start.year == 2025
    assert last_year.end.year == 2025


def test_say_is_write_time_happened_is_event_time():
    said = compile("what did I say last week", NOW)
    assert said.temporal
    assert said.clock == "write_time"
    assert "say" in said.cleaned_query.lower()
    assert "week" not in said.cleaned_query.lower()

    happened = compile("what happened last week", NOW)
    assert happened.temporal
    assert happened.clock == "event_time"
    assert "happened" in happened.cleaned_query.lower()


def test_mixed_say_happened_is_unresolved_clock():
    tq = compile("What did I say happened two weeks ago?", NOW)
    assert tq.temporal
    assert tq.clock == "unresolved"
    assert (NOW.date() - tq.start.date()).days == 14


def test_azure_two_weeks_ago_keeps_residue():
    tq = compile("Azure two weeks ago", NOW)
    assert tq.temporal
    assert "Azure" in tq.cleaned_query
    assert "week" not in tq.cleaned_query.lower()
    assert "ago" not in tq.cleaned_query.lower()


def test_does_not_spellcheck_non_grammar_tokens():
    tq = compile("Azre two weeks ago", NOW)
    assert tq.temporal
    assert "Azre" in tq.cleaned_query


@pytest.mark.parametrize(
    "query",
    [
        "two week Azure architecture",
        "two week sprint",
        "three month project",
        "one year contract",
        "May architecture",
        "March release process",
        "three days before that",
        "two weeks before that",
    ],
)
def test_false_positive_negatives_are_not_temporal(query):
    tq = compile(query, NOW)
    assert tq.temporal is False, query
    assert tq.start is None and tq.end is None


def test_last_week_metrics_is_temporal():
    tq = compile("last week metrics", NOW)
    assert tq.temporal
    assert tq.granularity == "week"
    assert "metrics" in tq.cleaned_query.lower()
    assert "week" not in tq.cleaned_query.lower()


def test_last_week_hyphen_is_temporal():
    tq = compile("last-week metrics", NOW)
    assert tq.temporal
    assert tq.granularity == "week"
    assert "metrics" in tq.cleaned_query.lower()


def test_since_before_after_between():
    since = compile("since March 4", NOW)
    assert since.temporal
    assert since.start.date() == datetime(2026, 3, 4).date()
    assert since.end.date() == NOW.date()

    before = compile("before March 4", NOW)
    assert before.start is None
    assert before.end.date() == datetime(2026, 3, 3).date()

    after = compile("after March 4", NOW)
    assert after.end is None
    assert after.start.date() == datetime(2026, 3, 5).date()

    between = compile("between March 4 and March 10", NOW)
    assert between.start.date() == datetime(2026, 3, 4).date()
    assert between.end.date() == datetime(2026, 3, 10).date()


def test_invalid_date_fails_loud():
    with pytest.raises(TemporalParseError):
        compile("on March 32", NOW)
    with pytest.raises(TemporalParseError):
        compile("2024-02-30", NOW)


def test_nontemporal_query_is_unchanged_residue():
    q = "Azure architecture notes"
    tq = compile(q, NOW)
    assert tq.temporal is False
    assert tq.cleaned_query == q
    assert tq.start is None
    assert tq.end is None
