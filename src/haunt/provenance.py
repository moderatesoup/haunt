"""Validation and public serialization for source provenance envelopes."""

from __future__ import annotations

import json
import re
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


def _optional_text(value: Any, field: str, *, limit: int = _TEXT_MAX) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"provenance.{field} must be a string or null")
    if not value:
        raise ValueError(f"provenance.{field} must be nonempty or null")
    if len(value) > limit:
        raise ValueError(f"provenance.{field} must be {limit} characters or fewer")
    return value


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
    tool_name: str | None = None,
    producer_call_id: str | None = None,
) -> dict[str, Any]:
    """Return a canonical v1 envelope without guessing absent source fields.

    An omitted envelope is a native observation. Its origin and producer tool
    are actual Store.observe inputs, not inferred source claims.
    """
    if not isinstance(origin, str) or not origin:
        raise ValueError("origin must be a nonempty string")
    if value is None:
        envelope: dict[str, Any] = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "kind": "native",
            "origin": origin,
        }
        if tool_name is not None:
            envelope["producer_tool"] = _optional_text(tool_name, "producer_tool")
        if producer_call_id is not None:
            if tool_name is None:
                raise ValueError("producer_call_id requires tool_name")
            envelope["producer_call_id"] = _optional_text(
                producer_call_id, "producer_call_id"
            )
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
    supplied_origin = envelope.get("origin")
    if supplied_origin is not None and supplied_origin != origin:
        raise ValueError("provenance.origin must match the observation origin")
    envelope["origin"] = origin
    supplied_tool = envelope.get("producer_tool")
    if supplied_tool is not None and supplied_tool != tool_name:
        raise ValueError(
            "provenance.producer_tool must match the observation tool_name"
        )
    if tool_name is not None:
        envelope["producer_tool"] = tool_name
    supplied_call_id = envelope.get("producer_call_id")
    if supplied_call_id is not None and supplied_call_id != producer_call_id:
        raise ValueError(
            "provenance.producer_call_id must match the observation producer_call_id"
        )
    if producer_call_id is not None:
        if tool_name is None:
            raise ValueError("producer_call_id requires tool_name")
        envelope["producer_call_id"] = _optional_text(
            producer_call_id, "producer_call_id"
        )

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
            if not isinstance(transforms, list):
                raise ValueError("provenance.transforms must be an array")
            if len(transforms) > _TRANSFORM_COUNT_MAX:
                raise ValueError(
                    f"provenance.transforms may contain at most {_TRANSFORM_COUNT_MAX} items"
                )
            envelope["transforms"] = [
                _optional_text(item, f"transforms[{index}]", limit=_TRANSFORM_MAX)
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
        try:
            parsed = json.loads(stored)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            try:
                validated = validate_provenance(
                    parsed,
                    origin=origin,
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
            "origin": origin,
        }
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": "legacy_unstructured",
        "origin": origin,
        "meta": legacy_meta,
    }


def native_provenance(
    *,
    channel: str | None,
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
        tool_name=tool,
        producer_call_id=call_id,
    )
