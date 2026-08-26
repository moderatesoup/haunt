"""Validation and public serialization for source provenance envelopes."""

from __future__ import annotations

import base64
import json
import math
import re
import struct
from datetime import datetime
from typing import Any, Mapping

from haunt.util import utc_iso

PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_KINDS = ("native", "import")
IMPORT_FIDELITIES = ("lossless", "lossy", "reconstructed", "derived")

_TEXT_MAX = 2048
_TRANSFORM_MAX = 256
_TRANSFORM_COUNT_MAX = 128
_SERIALIZED_MAX = 32768
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
SQLITE_KEY_PREFIX = "$haunt.sqlite-key:v1:"
_COMMON = {
    "schema_version",
    "kind",
    "channel",
    "origin",
    "producer_tool",
    "producer_call_id",
}
_IMPORT = {
    "source_platform",
    "source_native_id",
    "source_format",
    "parser_version",
    "imported_at",
    "fidelity",
    "original_blob_sha256",
    "transforms",
}


def json_safe_sqlite(value: Any) -> Any:
    """Losslessly encode SQLite values for public JSON surfaces.

    TEXT/NULL/integer/finite REAL values retain their existing JSON shape.
    SQLite BLOB values are never guessed as UTF-8; bytes and memoryviews use an
    explicit base64 envelope. Non-finite REAL values also need an explicit
    envelope because strict JSON cannot represent them.
    """
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return {
            "encoding": "base64",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            token = "nan"
        elif value > 0:
            token = "+infinity"
        else:
            token = "-infinity"
        return {"encoding": "sqlite-real", "data": token}
    if isinstance(value, Mapping):
        encoded: dict[str, Any] = {}
        for key, item in value.items():
            public_key = encode_json_safe_sqlite_key(key)
            if public_key in encoded:
                raise ValueError("public SQLite mapping key collision")
            encoded[public_key] = json_safe_sqlite(item)
        return encoded
    if isinstance(value, (list, tuple)):
        return [json_safe_sqlite(item) for item in value]
    raise TypeError(f"unsupported public SQLite value type: {type(value).__name__}")


def encode_json_safe_sqlite_key(value: Any) -> str:
    """Encode a SQLite mapping key without JSON string-key collisions."""
    if isinstance(value, str):
        if not value.startswith(SQLITE_KEY_PREFIX):
            return value
        data = base64.b64encode(value.encode("utf-8")).decode("ascii")
        return f"{SQLITE_KEY_PREFIX}text:{data}"
    if value is None:
        return f"{SQLITE_KEY_PREFIX}null"
    if isinstance(value, bool):
        return f"{SQLITE_KEY_PREFIX}bool:{int(value)}"
    if isinstance(value, int):
        return f"{SQLITE_KEY_PREFIX}integer:{value}"
    if isinstance(value, float):
        bits = struct.pack(">d", value).hex()
        return f"{SQLITE_KEY_PREFIX}real:{bits}"
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        data = base64.b64encode(value).decode("ascii")
        return f"{SQLITE_KEY_PREFIX}blob:{data}"
    raise TypeError(f"unsupported public SQLite mapping key: {type(value).__name__}")


def decode_json_safe_sqlite_key(value: str) -> Any:
    """Reverse a key produced by :func:`encode_json_safe_sqlite_key`."""
    if not isinstance(value, str):
        raise TypeError("public SQLite mapping key must be a string")
    if not value.startswith(SQLITE_KEY_PREFIX):
        return value
    payload = value[len(SQLITE_KEY_PREFIX) :]
    if payload == "null":
        return None
    if payload == "bool:0":
        return False
    if payload == "bool:1":
        return True
    if payload.startswith("integer:"):
        try:
            return int(payload.removeprefix("integer:"))
        except ValueError as exc:
            raise ValueError("invalid encoded SQLite integer key") from exc
    if payload.startswith("real:"):
        raw = payload.removeprefix("real:")
        if len(raw) != 16 or not re.fullmatch(r"[0-9a-f]{16}", raw):
            raise ValueError("invalid encoded SQLite REAL key")
        return struct.unpack(">d", bytes.fromhex(raw))[0]
    for kind, decoder in (
        ("text:", lambda raw: raw.decode("utf-8")),
        ("blob:", lambda raw: raw),
    ):
        if payload.startswith(kind):
            encoded = payload.removeprefix(kind)
            try:
                raw = base64.b64decode(encoded, validate=True)
                return decoder(raw)
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError(f"invalid encoded SQLite {kind[:-1]} key") from exc
    raise ValueError("unknown encoded SQLite mapping key")


def _optional_text(value: Any, field: str, *, limit: int = _TEXT_MAX) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"provenance.{field} must be a string or null")
    if not value:
        raise ValueError(f"provenance.{field} must be nonempty or null")
    if len(value.encode("utf-8")) > limit:
        raise ValueError(f"provenance.{field} must be {limit} UTF-8 bytes or fewer")
    return value


def _actual_text(value: Any, field: str) -> str:
    checked = _optional_text(value, field)
    if checked is None:
        raise ValueError(f"{field} must be a nonempty string")
    return checked


def _canonical_time(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"provenance.{field} must be a nonempty ISO timestamp")
    try:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone required")
        return utc_iso(parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"provenance.{field} must be a valid ISO timestamp") from exc


def validate_provenance(
    value: Mapping[str, Any] | None,
    *,
    origin: str,
    channel: str,
    tool_name: str | None = None,
    producer_call_id: str | None = None,
) -> dict[str, Any]:
    """Return a canonical v1 envelope without guessing absent source fields.

    An omitted envelope is a native observation. Its origin and producer tool
    are actual Store.observe inputs, not inferred source claims.
    """
    actual_origin = _actual_text(origin, "origin")
    actual_channel = _actual_text(channel, "channel")
    actual_tool = None if tool_name is None else _actual_text(tool_name, "tool_name")
    actual_call = (
        None
        if producer_call_id is None
        else _actual_text(producer_call_id, "producer_call_id")
    )
    if actual_call is not None and actual_tool is None:
        raise ValueError("producer_call_id requires tool_name")
    if value is None:
        envelope: dict[str, Any] = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "kind": "native",
            "channel": actual_channel,
            "origin": actual_origin,
        }
        if actual_tool is not None:
            envelope["producer_tool"] = actual_tool
        if actual_call is not None:
            envelope["producer_call_id"] = actual_call
        return envelope
    if not isinstance(value, Mapping):
        raise ValueError("provenance must be an object")
    envelope = dict(value)
    if any(not isinstance(key, str) for key in envelope):
        raise ValueError("provenance field names must be strings")
    version = envelope.get("schema_version")
    if type(version) is not int or version != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"provenance.schema_version must be {PROVENANCE_SCHEMA_VERSION}"
        )
    kind = envelope.get("kind")
    if kind not in PROVENANCE_KINDS:
        raise ValueError(f"provenance.kind must be one of {PROVENANCE_KINDS}")
    allowed = _COMMON | (_IMPORT if kind == "import" else set())
    extras = sorted(set(envelope) - allowed)
    if extras:
        raise ValueError(f"unsupported provenance fields: {', '.join(extras)}")

    for field in sorted(_COMMON - {"schema_version", "kind"}):
        if field in envelope:
            envelope[field] = _optional_text(envelope[field], field)
    supplied_channel = envelope.get("channel")
    if "channel" in envelope and supplied_channel != actual_channel:
        raise ValueError("provenance.channel must match the observation channel")
    envelope["channel"] = actual_channel
    supplied_origin = envelope.get("origin")
    if "origin" in envelope and supplied_origin != actual_origin:
        raise ValueError("provenance.origin must match the observation origin")
    envelope["origin"] = actual_origin
    supplied_tool = envelope.get("producer_tool")
    if supplied_tool is not None and supplied_tool != actual_tool:
        raise ValueError(
            "provenance.producer_tool must match the observation tool_name"
        )
    if actual_tool is not None:
        envelope["producer_tool"] = actual_tool
    supplied_call_id = envelope.get("producer_call_id")
    if supplied_call_id is not None and supplied_call_id != actual_call:
        raise ValueError(
            "provenance.producer_call_id must match the observation producer_call_id"
        )
    if actual_call is not None:
        envelope["producer_call_id"] = actual_call

    if kind == "import":
        for field in (
            "source_platform",
            "source_native_id",
            "source_format",
            "parser_version",
        ):
            if field in envelope:
                envelope[field] = _optional_text(envelope[field], field)
        if "imported_at" not in envelope:
            raise ValueError("provenance.imported_at is required for imports")
        envelope["imported_at"] = _canonical_time(
            envelope["imported_at"], "imported_at"
        )
        fidelity = envelope.get("fidelity")
        if fidelity not in IMPORT_FIDELITIES:
            raise ValueError(f"provenance.fidelity must be one of {IMPORT_FIDELITIES}")
        if "original_blob_sha256" not in envelope:
            raise ValueError(
                "provenance.original_blob_sha256 is required (use null when absent)"
            )
        blob_hash = envelope["original_blob_sha256"]
        if blob_hash is not None and (
            not isinstance(blob_hash, str) or not _SHA256_RE.fullmatch(blob_hash)
        ):
            raise ValueError(
                "provenance.original_blob_sha256 must be null or canonical sha256:<hex>"
            )
        if "transforms" in envelope:
            transforms = envelope["transforms"]
            if transforms is not None:
                if not isinstance(transforms, list):
                    raise ValueError("provenance.transforms must be an array or null")
                if len(transforms) > _TRANSFORM_COUNT_MAX:
                    raise ValueError(
                        "provenance.transforms may contain at most "
                        f"{_TRANSFORM_COUNT_MAX} items"
                    )
                envelope["transforms"] = [
                    _optional_text(
                        item,
                        f"transforms[{index}]",
                        limit=_TRANSFORM_MAX,
                    )
                    for index, item in enumerate(transforms)
                ]
                if any(item is None for item in envelope["transforms"]):
                    raise ValueError("provenance.transforms items must be strings")

    encoded = json.dumps(
        envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if len(encoded.encode("utf-8")) > _SERIALIZED_MAX:
        raise ValueError(
            f"provenance envelope must be {_SERIALIZED_MAX} UTF-8 bytes or fewer"
        )
    return envelope


def provenance_json(envelope: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(envelope), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def public_provenance(
    stored: Any,
    *,
    origin: Any,
    legacy_meta: Any,
    tool_name: Any = None,
) -> dict[str, Any]:
    """Serialize structured rows or label untouched pre-v8 rows honestly."""
    if stored is not None:
        parsed = None
        if isinstance(stored, str):
            try:
                parsed = json.loads(stored)
            except json.JSONDecodeError:
                pass
        if isinstance(parsed, dict):
            try:
                validated = validate_provenance(
                    parsed,
                    origin=origin,
                    channel=parsed.get("channel"),
                    tool_name=tool_name,
                    producer_call_id=parsed.get("producer_call_id"),
                )
            except ValueError:
                validated = None
            if validated == parsed:
                return parsed
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "kind": "invalid_stored",
            "origin": json_safe_sqlite(origin),
        }
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": "legacy_unstructured",
        "origin": json_safe_sqlite(origin),
        "meta": json_safe_sqlite(legacy_meta),
    }


def native_provenance(
    *,
    channel: str,
    origin: str,
    tool: str | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Convenience constructor for host integrations."""
    value: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": "native",
    }
    if channel is not None:
        value["channel"] = channel
    if tool is not None:
        value["producer_tool"] = tool
    if call_id is not None:
        value["producer_call_id"] = call_id
    return validate_provenance(
        value,
        origin=origin,
        channel=channel,
        tool_name=tool,
        producer_call_id=call_id,
    )
