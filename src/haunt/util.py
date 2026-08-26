"""Shared helpers: ids, timestamps, diagnostics. No I/O beyond stderr."""

from __future__ import annotations

import base64
import json
import math
import sys
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any


def new_id() -> str:
    return str(uuid.uuid4())


K_MIN = 1
K_MAX = 100


def clamp_k(k: Any, *, default: int = 8) -> int:
    """Clamp a retrieval k to a safe range. Callers share this bound."""
    if k is None:
        n = default
    else:
        try:
            n = int(k)
        except (TypeError, ValueError):
            n = default
    return max(K_MIN, min(K_MAX, n))


def utc_iso(dt: datetime) -> str:
    """Canonical storage/order form: UTC, microseconds, explicit +00:00."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def now_iso() -> str:
    return utc_iso(datetime.now(timezone.utc))


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
    return utc_iso(parse_iso(value))


def format_iso(value: Any) -> str:
    """Display form of a stored timestamp (always UTC)."""
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        return human_display(value, limit=80, sqlite_scalar=True)
    try:
        return utc_iso(parse_iso(value))
    except ValueError:
        return human_display(value, limit=80, sqlite_scalar=True)


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


def loads(text: Any, default: Any = None) -> Any:
    """Parse stored JSON without treating opaque legacy SQLite values as text."""
    if not text:
        return {} if default is None else default
    try:
        return json.loads(text)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return {} if default is None else default


def diag(msg: str, **fields: Any) -> None:
    """Structured diagnostics on stderr (stdout stays human-readable)."""
    payload = {"msg": msg, **fields}
    print(dumps(payload), file=sys.stderr, flush=True)


LIMIT_MIN = 1
LIMIT_MAX = 100


def clamp_limit(
    value: Any, default: int = 10, *, lo: int = LIMIT_MIN, hi: int = LIMIT_MAX
) -> int:
    """Clamp k/limit so negative never becomes an unbounded SQLite LIMIT."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if n < lo:
        return lo
    if n > hi:
        return hi
    return n


def _bounded_human_text(text: str, limit: int) -> str:
    try:
        n = max(1, int(limit))
    except (TypeError, ValueError):
        n = 160
    if len(text) <= n:
        return text
    if n == 1:
        return "…"
    return text[: n - 1] + "…"


def _escape_human_controls(text: str, *, preserve_layout: bool) -> str:
    out: list[str] = []
    for char in text:
        if preserve_layout and char in {"\n", "\t"}:
            out.append(char)
            continue
        if unicodedata.category(char).startswith("C"):
            code = ord(char)
            escape = f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"
            out.append(escape)
        else:
            out.append(char)
    return "".join(out)


def human_display(
    value: Any,
    *,
    limit: int = 160,
    collapse_whitespace: bool = False,
    preserve_layout: bool = False,
    sqlite_scalar: bool = False,
) -> str:
    """Render an already-serialized value safely for bounded human output.

    Ordinary strings retain their text. Public SQLite BLOB/non-finite REAL
    envelopes get explicit markers; other JSON-safe values use stable JSON.
    Control characters cannot inject terminal control sequences.
    """
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        data = base64.b64encode(bytes(value)).decode("ascii")
        text = f"<sqlite-blob base64:{data}>"
    elif isinstance(value, str):
        text = value
    elif value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        if math.isfinite(value):
            text = str(value)
        else:
            token = (
                "nan"
                if math.isnan(value)
                else ("+infinity" if value > 0 else "-infinity")
            )
            text = f"<sqlite-real {token}>"
    elif (
        sqlite_scalar
        and isinstance(value, dict)
        and set(value) == {"encoding", "data"}
        and value.get("encoding") == "base64"
        and isinstance(value.get("data"), str)
        and _is_canonical_base64(value["data"])
    ):
        text = f"<sqlite-blob base64:{value['data']}>"
    elif (
        sqlite_scalar
        and isinstance(value, dict)
        and set(value) == {"encoding", "data"}
        and value.get("encoding") == "sqlite-real"
        and isinstance(value.get("data"), str)
        and value.get("data") in {"nan", "+infinity", "-infinity"}
    ):
        text = f"<sqlite-real {value['data']}>"
    elif isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            text = f"<{type(value).__name__}>"
    else:
        text = f"<{type(value).__name__}>"

    if collapse_whitespace:
        text = " ".join(text.split())
    elif preserve_layout:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _escape_human_controls(text, preserve_layout=preserve_layout)
    return _bounded_human_text(text, limit)


def _is_canonical_base64(value: str) -> bool:
    """True only for the exact RFC 4648 form emitted by json_safe_sqlite."""
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError:
        return False
    return base64.b64encode(raw).decode("ascii") == value


def snippet(text: Any, n: int = 160) -> str:
    return human_display(
        text,
        limit=n,
        collapse_whitespace=True,
        sqlite_scalar=True,
    )
