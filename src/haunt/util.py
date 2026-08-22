"""Shared helpers: ids, timestamps, diagnostics. No I/O beyond stderr."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from typing import Any


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def iso_or_now(value: str | None) -> str:
    if not value:
        return now_iso()
    return parse_iso(value).isoformat(timespec="seconds")


CLOCKS = ("event_time", "storage_time")
# write_time is a deprecated alias: ingest/storage time (events.ts), not source time.
CLOCK_ALIASES = {"write_time": "storage_time"}


def normalize_clock(clock: str | None, *, allow_unresolved: bool = False) -> str:
    """Canonicalize a clock token.

    storage_time is events.ts (ingest/storage time, not conversation/source time).
    write_time is accepted as a deprecated alias for storage_time.
    Default is event_time so existing since/until callers stay unchanged.
    """
    if clock is None:
        return "event_time"
    c = CLOCK_ALIASES.get(clock, clock)
    allowed = CLOCKS + (("unresolved",) if allow_unresolved else ())
    if c not in allowed:
        extra = ", write_time (deprecated alias for storage_time)"
        if allow_unresolved:
            extra += ", unresolved"
        raise ValueError(
            f"clock must be event_time or storage_time{extra}, got {clock!r}"
        )
    return c


def clock_sql_column(clock: str | None, *, qualified: bool = True) -> str:
    """Map clock=event_time|storage_time to the events column.

    storage_time is events.ts (ingest/storage time, not source time).
    write_time is a deprecated alias for storage_time.
    event_time is events.event_time. Default is event_time.
    """
    c = normalize_clock(clock)
    if c == "event_time":
        return "e.event_time" if qualified else "event_time"
    if c == "storage_time":
        return "e.ts" if qualified else "ts"
    raise ValueError(f"clock must be event_time or storage_time, got {clock!r}")


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return {} if default is None else default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {} if default is None else default


def diag(msg: str, **fields: Any) -> None:
    """Structured diagnostics on stderr (stdout stays human-readable)."""
    payload = {"msg": msg, **fields}
    print(dumps(payload), file=sys.stderr, flush=True)


def snippet(text: str, n: int = 160) -> str:
    one = " ".join((text or "").split())
    if len(one) <= n:
        return one
    return one[: n - 1] + "…"
