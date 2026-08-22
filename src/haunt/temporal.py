"""Query-time temporal compiler.

Language → TemporalQuery only. No retrieval policy, no ±tolerance, no
strategy field. Restricted aliases first; edit-distance-1 fuzzy only after
aliases fail, and only on grammar number/unit tokens.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal

Clock = Literal["event_time", "write_time", "unresolved"]
Granularity = Literal["day", "week", "month", "year"]
Certainty = Literal["exact", "approximate"]

NUMBER_WORDS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
}

# Canonical unit aliases. These are deterministic, not typos.
UNIT_ALIASES: dict[str, Granularity] = {
    "d": "day",
    "day": "day",
    "days": "day",
    "wk": "week",
    "wks": "week",
    "week": "week",
    "weeks": "week",
    "mo": "month",
    "mos": "month",
    "month": "month",
    "months": "month",
    "yr": "year",
    "yrs": "year",
    "year": "year",
    "years": "year",
}

MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_APPROX = frozenset({"about", "around", "approximately", "approx", "roughly"})
_PREP = frozenset({"since", "before", "after", "until", "on", "in"})

# Discourse-relative: v1 must not pretend to resolve these.
_DISCOURSE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|a|an)\s+"
    r"(?:d|day|days|wk|wks|week|weeks|mo|mos|month|months|yr|yrs|year|years)"
    r"s?\s+(?:before|after)\s+(?:that|this|it)\b",
    re.IGNORECASE,
)

_WRITE_RE = re.compile(
    r"\b(say|said|tell|told|telling|talking|talk|discuss(?:ed|ing)?|"
    r"mention(?:ed|ing)?)\b",
    re.IGNORECASE,
)
_EVENT_RE = re.compile(
    r"\b(happen(?:ed|ing)?|occur(?:red|ring)?|went|going|\bgo\b|when\s+was|\bdo\b)\b",
    re.IGNORECASE,
)

_TOKEN = re.compile(r"[A-Za-z0-9./:+-]+")
_ISO_DATE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})(?:T[\d:.+-]+)?$"
)
_US_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?$")
_ORDINAL = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?$", re.IGNORECASE)
_YEAR = re.compile(r"^\d{4}$")


class TemporalParseError(ValueError):
    """A date phrase was recognized but is not a real calendar date."""


@dataclass(frozen=True)
class TemporalQuery:
    temporal: bool
    cleaned_query: str
    start: datetime | None
    end: datetime | None
    clock: Clock
    granularity: Granularity
    certainty: Certainty
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "temporal": self.temporal,
            "cleaned_query": self.cleaned_query,
            "start": self.start.isoformat(timespec="seconds") if self.start else None,
            "end": self.end.isoformat(timespec="seconds") if self.end else None,
            "clock": self.clock,
            "granularity": self.granularity,
            "certainty": self.certainty,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class _Match:
    lo: int
    hi: int
    start: datetime | None
    end: datetime | None
    granularity: Granularity
    certainty: Certainty
    confidence: float
    fuzzy: bool


def compile(query: str, now: datetime | None = None) -> TemporalQuery:
    """Compile a user question into a TemporalQuery. No retrieval policy."""
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    raw = (query or "").strip()
    if not raw:
        return _none(raw)

    if _DISCOURSE.search(raw):
        return _none(raw)

    working = _split_word_hyphens(raw)
    tokens = [m.group() for m in _TOKEN.finditer(working)]
    if not tokens:
        return _none(raw)

    match = _best_match(tokens, now)
    if match is None:
        return _none(raw)

    kept = tokens[: match.lo] + tokens[match.hi :]
    cleaned = " ".join(kept).strip()
    conf = match.confidence
    if match.fuzzy:
        conf = min(conf, 0.8)

    return TemporalQuery(
        temporal=True,
        cleaned_query=cleaned,
        start=match.start,
        end=match.end,
        clock=_infer_clock(raw),
        granularity=match.granularity,
        certainty=match.certainty,
        confidence=conf,
    )


def _none(query: str) -> TemporalQuery:
    return TemporalQuery(
        temporal=False,
        cleaned_query=query,
        start=None,
        end=None,
        clock="unresolved",
        granularity="day",
        certainty="exact",
        confidence=1.0,
    )


def _split_word_hyphens(text: str) -> str:
    """Turn last-week into last week. Leave ISO dates and numeric stamps alone."""
    protected: list[str] = []

    def hold(m: re.Match[str]) -> str:
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    held = re.sub(r"\d{4}-\d{2}-\d{2}(?:T[^\s]+)?", hold, text)
    held = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", held)
    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], held)


def _edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein capped at 2. Adjacent transposition counts as 1 (tow→two)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return 2
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if not diffs:
            return 0
        if len(diffs) == 1:
            return 1
        if len(diffs) == 2:
            i, j = diffs
            if j == i + 1 and a[i] == b[j] and a[j] == b[i]:
                return 1
        return 2
    longer, shorter = (a, b) if la > lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(longer) and j < len(shorter):
        if longer[i] == shorter[j]:
            i += 1
            j += 1
            continue
        if skipped:
            return 2
        skipped = True
        i += 1
    return 1


def _number(token: str) -> tuple[int, bool] | None:
    """Return (value, used_fuzzy). Digits and aliases first; then edit-distance 1."""
    t = token.lower().rstrip(".,;:?!")
    if t.isdigit():
        n = int(t)
        if n < 1:
            return None
        return n, False
    if t in NUMBER_WORDS:
        return NUMBER_WORDS[t], False
    if len(t) < 3:
        return None
    for word, n in NUMBER_WORDS.items():
        if word in ("a", "an"):
            continue
        if len(word) >= 3 and _edit_distance(t, word) == 1:
            return n, True
    return None


def _unit(token: str) -> tuple[Granularity, bool] | None:
    t = token.lower().rstrip(".,;:?!")
    if t in UNIT_ALIASES:
        return UNIT_ALIASES[t], False
    if len(t) < 3:
        return None
    for alias, gran in UNIT_ALIASES.items():
        if len(alias) >= 3 and _edit_distance(t, alias) == 1:
            return gran, True
    return None


def _month(token: str) -> int | None:
    t = token.lower().rstrip(".,;:?!")
    return MONTHS.get(t)


def _tz(now: datetime) -> timezone | None:
    tz = now.tzinfo
    if tz is None:
        return timezone.utc
    return tz  # type: ignore[return-value]


def _day_start(d: date, now: datetime) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=_tz(now))


def _day_end(d: date, now: datetime) -> datetime:
    return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=_tz(now))


def _day_window(d: date, now: datetime) -> tuple[datetime, datetime]:
    return _day_start(d, now), _day_end(d, now)


def _month_window(year: int, month: int, now: datetime) -> tuple[datetime, datetime]:
    last = calendar.monthrange(year, month)[1]
    return _day_start(date(year, month, 1), now), _day_end(date(year, month, last), now)


def _year_window(year: int, now: datetime) -> tuple[datetime, datetime]:
    return _day_start(date(year, 1, 1), now), _day_end(date(year, 12, 31), now)


def _week_window(monday: date, now: datetime) -> tuple[datetime, datetime]:
    sunday = monday + timedelta(days=6)
    return _day_start(monday, now), _day_end(sunday, now)


def _this_monday(now: datetime) -> date:
    return now.date() - timedelta(days=now.weekday())


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last))


def _add_years(d: date, years: int) -> date:
    try:
        return date(d.year + years, d.month, d.day)
    except ValueError:
        return date(d.year + years, d.month, calendar.monthrange(d.year + years, d.month)[1])


def _offset_ago(now: datetime, n: int, gran: Granularity) -> tuple[datetime, datetime]:
    """N units ago = that calendar day (OFFSET), not a range to now."""
    if gran == "day":
        target = now.date() - timedelta(days=n)
    elif gran == "week":
        target = now.date() - timedelta(days=7 * n)
    elif gran == "month":
        target = _add_months(now.date(), -n)
    else:
        target = _add_years(now.date(), -n)
    return _day_window(target, now)


def _last_n_range(now: datetime, n: int, gran: Granularity) -> tuple[datetime, datetime]:
    """last N / in the past N = range [now - N, now], not a single day."""
    if gran == "day":
        start_d = now.date() - timedelta(days=n)
    elif gran == "week":
        start_d = now.date() - timedelta(days=7 * n)
    elif gran == "month":
        start_d = _add_months(now.date(), -n)
    else:
        start_d = _add_years(now.date(), -n)
    return _day_start(start_d, now), now


def _valid_ymd(year: int, month: int, day: int) -> date:
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise TemporalParseError(f"invalid date {year:04d}-{month:02d}-{day:02d}") from exc


def _year_from_yy(yy: int, now: datetime) -> int:
    if yy >= 100:
        return yy
    return 2000 + yy if yy < 70 else 1900 + yy


@dataclass
class _Date:
    start: datetime
    end: datetime
    granularity: Granularity
    tokens: int
    fuzzy: bool = False


def _parse_date(tokens: list[str], i: int, now: datetime, *, allow_us_no_year: bool) -> _Date | None:
    if i >= len(tokens):
        return None
    tok = tokens[i]

    iso = _ISO_DATE.match(tok)
    if iso:
        d = _valid_ymd(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        start, end = _day_window(d, now)
        return _Date(start, end, "day", 1)

    us = _US_DATE.match(tok)
    if us:
        month, day = int(us.group(1)), int(us.group(2))
        if us.group(3) is None and not allow_us_no_year:
            return None
        year = _year_from_yy(int(us.group(3)), now) if us.group(3) else now.year
        d = _valid_ymd(year, month, day)
        start, end = _day_window(d, now)
        return _Date(start, end, "day", 1)

    month = _month(tok)
    if month is None:
        return None

    # Month day [year] / Month year. Bare month is only a date with a cue
    # (handled by the caller via allow_bare_month).
    used = 1
    day_n: int | None = None
    year_n: int | None = None
    if i + 1 < len(tokens):
        nxt = tokens[i + 1].rstrip(",")
        ord_m = _ORDINAL.match(nxt)
        if ord_m:
            day_n = int(ord_m.group(1))
            if day_n < 1 or day_n > 31:
                raise TemporalParseError(f"invalid day {day_n} after {tok}")
            used = 2
            if i + 2 < len(tokens) and _YEAR.match(tokens[i + 2].rstrip(",")):
                year_n = int(tokens[i + 2].rstrip(","))
                used = 3
        elif _YEAR.match(tokens[i + 1].rstrip(",")):
            year_n = int(tokens[i + 1].rstrip(","))
            used = 2

    year = year_n if year_n is not None else now.year
    if day_n is not None:
        d = _valid_ymd(year, month, day_n)
        start, end = _day_window(d, now)
        return _Date(start, end, "day", used)
    if year_n is not None:
        start, end = _month_window(year, month, now)
        return _Date(start, end, "month", used)
    return None


def _parse_date_or_month(
    tokens: list[str], i: int, now: datetime, *, allow_us_no_year: bool
) -> _Date | None:
    parsed = _parse_date(tokens, i, now, allow_us_no_year=allow_us_no_year)
    if parsed:
        return parsed
    if i >= len(tokens):
        return None
    month = _month(tokens[i])
    if month is None:
        return None
    # Bare month after a temporal preposition: "in March", "before March".
    if i + 1 < len(tokens) and _YEAR.match(tokens[i + 1].rstrip(",")):
        year = int(tokens[i + 1].rstrip(","))
        start, end = _month_window(year, month, now)
        return _Date(start, end, "month", 2)
    start, end = _month_window(now.year, month, now)
    return _Date(start, end, "month", 1)


def _best_match(tokens: list[str], now: datetime) -> _Match | None:
    found: list[_Match] = []
    for i in range(len(tokens)):
        for fn in (
            _try_between,
            _try_in_the_past,
            _try_last_n,
            _try_ago,
            _try_preposition,
            _try_last_this_unit,
            _try_today_yesterday,
            _try_bare_date,
        ):
            m = fn(tokens, i, now)
            if m:
                found.append(m)
    if not found:
        return None
    found.sort(key=lambda m: (m.hi - m.lo, -m.lo), reverse=True)
    return found[0]


def _try_between(tokens: list[str], i: int, now: datetime) -> _Match | None:
    if tokens[i].lower().rstrip(".,") != "between":
        return None
    left = _parse_date_or_month(tokens, i + 1, now, allow_us_no_year=True)
    if not left:
        return None
    j = i + 1 + left.tokens
    if j >= len(tokens) or tokens[j].lower().rstrip(".,") != "and":
        return None
    right = _parse_date_or_month(tokens, j + 1, now, allow_us_no_year=True)
    if not right:
        return None
    if left.start > right.end:
        raise TemporalParseError("between-range start is after end")
    gran: Granularity = (
        left.granularity
        if left.granularity == right.granularity
        else _coarser(left.granularity, right.granularity)
    )
    return _Match(
        i,
        j + 1 + right.tokens,
        left.start,
        right.end,
        gran,
        "exact",
        0.95,
        left.fuzzy or right.fuzzy,
    )


def _coarser(a: Granularity, b: Granularity) -> Granularity:
    order = {"day": 0, "week": 1, "month": 2, "year": 3}
    return a if order[a] >= order[b] else b


def _try_in_the_past(tokens: list[str], i: int, now: datetime) -> _Match | None:
    if tokens[i].lower() != "in":
        return None
    if i + 4 > len(tokens):
        return None
    if tokens[i + 1].lower() != "the":
        return None
    if tokens[i + 2].lower() not in ("past", "last"):
        return None
    num = _number(tokens[i + 3])
    unit = _unit(tokens[i + 4]) if i + 4 < len(tokens) else None
    if not num or not unit:
        return None
    n, nf = num
    gran, uf = unit
    start, end = _last_n_range(now, n, gran)
    return _Match(i, i + 5, start, end, gran, "exact", 0.95, nf or uf)


def _try_last_n(tokens: list[str], i: int, now: datetime) -> _Match | None:
    if tokens[i].lower() not in ("last", "past"):
        return None
    if i + 2 >= len(tokens):
        return None
    num = _number(tokens[i + 1])
    unit = _unit(tokens[i + 2])
    if not num or not unit:
        return None
    n, nf = num
    gran, uf = unit
    start, end = _last_n_range(now, n, gran)
    return _Match(i, i + 3, start, end, gran, "exact", 0.95, nf or uf)


def _try_ago(tokens: list[str], i: int, now: datetime) -> _Match | None:
    start_i = i
    certainty: Certainty = "exact"
    conf = 0.95
    if tokens[i].lower() in _APPROX:
        certainty = "approximate"
        conf = 0.75
        i += 1
        if i >= len(tokens):
            return None
    if i + 2 >= len(tokens):
        return None
    num = _number(tokens[i])
    unit = _unit(tokens[i + 1])
    if not num or not unit:
        return None
    if tokens[i + 2].lower().rstrip(".,;?!") != "ago":
        return None
    n, nf = num
    gran, uf = unit
    # Meaning is a day offset, even when the unit is week/month/year.
    start, end = _offset_ago(now, n, gran)
    return _Match(start_i, i + 3, start, end, "day", certainty, conf, nf or uf)


def _try_preposition(tokens: list[str], i: int, now: datetime) -> _Match | None:
    prep = tokens[i].lower().rstrip(".,")
    if prep not in _PREP:
        return None
    parsed = _parse_date_or_month(tokens, i + 1, now, allow_us_no_year=True)
    if not parsed:
        return None
    start, end = _apply_prep(prep, parsed, now)
    return _Match(
        i,
        i + 1 + parsed.tokens,
        start,
        end,
        parsed.granularity,
        "exact",
        0.95,
        parsed.fuzzy,
    )


def _apply_prep(
    prep: str, parsed: _Date, now: datetime
) -> tuple[datetime | None, datetime | None]:
    if prep in ("on", "in"):
        return parsed.start, parsed.end
    if prep == "since":
        return parsed.start, now
    if prep == "until":
        return None, parsed.end
    if prep == "before":
        # Open/exclusive at the start of the referenced unit.
        return None, parsed.start - timedelta(seconds=1)
    # after: open at the end of the referenced unit
    return parsed.end + timedelta(seconds=1), None


def _try_last_this_unit(tokens: list[str], i: int, now: datetime) -> _Match | None:
    word = tokens[i].lower()
    if word not in ("last", "this"):
        return None
    if i + 1 >= len(tokens):
        return None
    unit = _unit(tokens[i + 1])
    if not unit:
        return None
    gran, uf = unit
    if gran == "day":
        return None
    if word == "last":
        start, end = _last_calendar_unit(now, gran)
    else:
        start, end = _this_calendar_unit(now, gran)
    return _Match(i, i + 2, start, end, gran, "exact", 0.95, uf)


def _last_calendar_unit(now: datetime, gran: Granularity) -> tuple[datetime, datetime]:
    if gran == "week":
        monday = _this_monday(now) - timedelta(days=7)
        return _week_window(monday, now)
    if gran == "month":
        first = date(now.year, now.month, 1)
        prev = first - timedelta(days=1)
        return _month_window(prev.year, prev.month, now)
    return _year_window(now.year - 1, now)


def _this_calendar_unit(now: datetime, gran: Granularity) -> tuple[datetime, datetime]:
    if gran == "week":
        return _week_window(_this_monday(now), now)
    if gran == "month":
        return _month_window(now.year, now.month, now)
    return _year_window(now.year, now)


def _try_today_yesterday(tokens: list[str], i: int, now: datetime) -> _Match | None:
    w = tokens[i].lower().rstrip(".,;?!")
    if w == "today":
        start, end = _day_window(now.date(), now)
        return _Match(i, i + 1, start, end, "day", "exact", 0.99, False)
    if w == "yesterday":
        start, end = _day_window(now.date() - timedelta(days=1), now)
        return _Match(i, i + 1, start, end, "day", "exact", 0.99, False)
    return None


def _try_bare_date(tokens: list[str], i: int, now: datetime) -> _Match | None:
    # Bare month names are NOT dates ("May architecture", "March release").
    parsed = _parse_date(tokens, i, now, allow_us_no_year=False)
    if not parsed:
        return None
    return _Match(
        i,
        i + parsed.tokens,
        parsed.start,
        parsed.end,
        parsed.granularity,
        "exact",
        0.93,
        parsed.fuzzy,
    )


def _infer_clock(query: str) -> Clock:
    """Conservative clock. Mixed speech+occurrence cues stay unresolved."""
    write = bool(_WRITE_RE.search(query))
    event = bool(_EVENT_RE.search(query))
    if re.search(r"\bdid\b", query, re.IGNORECASE):
        # Auxiliary "did I say/tell" is not an occurrence cue by itself.
        if not write:
            event = True
    if write and event:
        return "unresolved"
    if write:
        return "write_time"
    return "event_time"
