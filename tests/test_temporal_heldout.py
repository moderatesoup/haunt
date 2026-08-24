"""Held-out temporal compile cases. These are new wordings, not the old 8."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from haunt.temporal import compile

NOW = datetime(2026, 8, 22, 15, 30, 0, tzinfo=timezone.utc)


def test_this_module_imports_compile_only():
    tree = ast.parse(Path(__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("haunt"):
            assert node.module == "haunt.temporal"


# ---------------------------------------------------------------------------
# Offsets the grammar already compiles (new wordings).
# ---------------------------------------------------------------------------

OFFSET_PASSES = (
    # query, granularity, expected cleaned_query
    ("four days ago", "day", ""),
    ("six months ago", "day", ""),  # N-unit ago is a day offset
    ("five yrs ago", "day", ""),
    ("3 mo ago", "day", ""),
    ("a couple days ago", "day", ""),
    ("roughly three weeks ago", "day", ""),
    ("around 8 days ago", "day", ""),
    ("past five weeks", "week", ""),
    ("in the last three days", "day", ""),
    ("during the past two years", "year", ""),
    ("over the last 4 wks", "week", ""),  # wks is a v1 alias
    ("10 days ago", "day", ""),
    ("last 3 days", "day", ""),
)


@pytest.mark.parametrize("query,granularity,cleaned", OFFSET_PASSES)
def test_heldout_offset_compiles(query, granularity, cleaned):
    tq = compile(query, NOW)
    assert tq.temporal is True
    assert tq.clock == "event_time"
    assert tq.granularity == granularity
    assert tq.cleaned_query == cleaned


# ---------------------------------------------------------------------------
# Calendar phrases the grammar already compiles (new wordings).
# ---------------------------------------------------------------------------

CALENDAR_PASSES = (
    ("on April 12", "day", ""),
    ("in October 2024", "month", ""),
    ("this year", "year", ""),
    ("last month", "month", ""),
    ("this week", "week", ""),
    ("September 22, 2025", "day", ""),
    ("between January 8 and January 20", "day", ""),
    ("since July 1", "day", ""),
    ("before November 2023", "month", ""),
    ("until August 1", "day", ""),
    ("on 7/4/2025", "day", ""),
    ("after 2024-06-01", "day", ""),
)


@pytest.mark.parametrize("query,granularity,cleaned", CALENDAR_PASSES)
def test_heldout_calendar_compiles(query, granularity, cleaned):
    tq = compile(query, NOW)
    assert tq.temporal is True
    assert tq.clock == "event_time"
    assert tq.granularity == granularity
    assert tq.cleaned_query == cleaned


# ---------------------------------------------------------------------------
# Speech + time: clock stays event_time; time phrase is stripped.
# ---------------------------------------------------------------------------

SPEECH_PASSES = (
    ("what did I mention last week", "week", "what did I mention"),
    ("remind me what I told you yesterday", "day", "remind me what I told you"),
    ("what were we discussing in June", "month", "what were we discussing"),
    ("what did I talk about on April 12", "day", "what did I talk about"),
    ("did I mention the kiln last month", "month", "did I mention the kiln"),
    ("notes we discussed four days ago", "day", "notes we discussed"),
    ("what did I say this year", "year", "what did I say"),
)


@pytest.mark.parametrize("query,granularity,cleaned", SPEECH_PASSES)
def test_heldout_speech_plus_time_is_event_time(query, granularity, cleaned):
    tq = compile(query, NOW)
    assert tq.temporal is True
    assert tq.clock == "event_time"
    assert tq.granularity == granularity
    assert tq.cleaned_query == cleaned


# ---------------------------------------------------------------------------
# Non-temporal lookalikes: duration labels, proper nouns, idioms.
# ---------------------------------------------------------------------------

LOOKALIKES = (
    "four week onboarding cohort",
    "six month runway forecast",
    "two day workshop agenda",
    "five year plan memo",
    "one day conference badge",
    "March Madness bracket",
    "May Day parade notes",
    "August Wilson play notes",
    "April Fools prank list",
    "Tuesday standup template",
    "last mile delivery notes",
    "past due invoice",
    "quarterly OKR review",
    "weekend hackathon writeup",
    "week of code curriculum",
    "spring cleaning checklist",
    "since forever backlog",
    "until further notice policy",
    "before and after photos",
)


@pytest.mark.parametrize("query", LOOKALIKES)
def test_heldout_lookalikes_are_not_temporal(query):
    tq = compile(query, NOW)
    assert tq.temporal is False
    assert tq.clock == "unresolved"
    assert tq.granularity == "day"
    assert tq.cleaned_query == query


# ---------------------------------------------------------------------------
# Generalization gaps: grammar *should* cover these, but v1 does not.
# Honest xfail — do not widen the grammar just to green this file.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="generalization gap: v1 offset adverb is 'ago', not 'back'",
)
def test_weeks_back_offset():
    tq = compile("five weeks back", NOW)
    assert tq.temporal is True
    assert tq.clock == "event_time"
    assert tq.granularity == "day"
    assert tq.cleaned_query == ""


@pytest.mark.xfail(
    strict=True,
    reason="generalization gap: v1 offset adverb is 'ago', not 'back'",
)
def test_couple_days_back_offset():
    tq = compile("a couple days back", NOW)
    assert tq.temporal is True
    assert tq.clock == "event_time"
    assert tq.granularity == "day"
    assert tq.cleaned_query == ""


@pytest.mark.xfail(
    strict=True,
    reason="generalization gap: fortnight is a new unit, not a wks-style spelling alias",
)
def test_past_fortnight_range():
    tq = compile("during the past fortnight", NOW)
    assert tq.temporal is True
    assert tq.clock == "event_time"
    assert tq.granularity == "week"
    assert tq.cleaned_query == ""


@pytest.mark.xfail(
    strict=True,
    reason="generalization gap: 'early' + month + 'last year' is not composed (v1 takes last year alone)",
)
def test_early_month_last_year():
    tq = compile("in early April last year", NOW)
    assert tq.temporal is True
    assert tq.clock == "event_time"
    assert tq.granularity == "month"
    cleaned = tq.cleaned_query.lower()
    assert "april" not in cleaned
    assert "year" not in cleaned
    assert "early" not in cleaned


@pytest.mark.xfail(
    strict=True,
    reason="generalization gap: no week-of-Nth grammar",
)
def test_week_of_the_nth():
    tq = compile("the week of the 12th", NOW)
    assert tq.temporal is True
    assert tq.clock == "event_time"
    assert tq.granularity == "week"
    assert tq.cleaned_query == ""


@pytest.mark.xfail(
    strict=True,
    reason="generalization gap: weekday names are not in v1",
)
def test_mention_last_weekday():
    tq = compile("what did I mention last Wednesday", NOW)
    assert tq.temporal is True
    assert tq.clock == "event_time"
    assert tq.granularity == "day"
    assert tq.cleaned_query == "what did I mention"


@pytest.mark.xfail(
    strict=True,
    reason="precision gap: idiomatic 'last week of the quarter …' is a doc name, not a calendar query",
)
def test_last_week_of_quarter_doc_is_not_temporal():
    query = "last week of the quarter planning doc"
    tq = compile(query, NOW)
    assert tq.temporal is False
    assert tq.clock == "unresolved"
    assert tq.cleaned_query == query
