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
