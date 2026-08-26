"""Canonical, bounded namespace export/import.

The bundle is a transfer format for one namespace's durable semantics.  It is
not a SQLite backup: local paths, indexes, embeddings, jobs, WAL state, and
other rebuildable projections never enter the semantic payload.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from haunt.paths import (
    NamespacePathError,
    normalize_namespace_label,
    repository_identity,
    safe_name,
)
from haunt.provenance import validate_provenance
from haunt.graph import _refresh_relation
from haunt.store import (
    RECALL_CLASSES,
    SCHEMA_VERSION,
    _claim_fresh_namespace_db_with_configuration_lock,
    _ensure_namespace_schema,
    _init_namespace_schema,
    _namespace_migration_lock,
    _readonly_registry,
    _registry,
    _sqlite_configuration_lock,
    _validate_unmapped_namespace_target,
    is_concurrent_registry_change,
    namespaces_dir,
    open_namespace_identity,
    open_namespace_identity_readonly,
    resolve_namespace_id,
    resolve_namespace_identity,
)
from haunt.util import now_iso, parse_iso, utc_iso

FORMAT_NAME = "haunt.namespace-export"
FORMAT_MAJOR = 1
FORMAT_MINOR = 0
MEDIA_TYPE = "application/vnd.haunt.namespace-export+json;version=1"

_DIGEST_PREFIX = "sha256:"
_RECEIPT_PREFIX = "import_receipt:"


class ExportError(ValueError):
    """The selected namespace cannot be represented honestly."""


class ImportBundleError(ValueError):
    """A bundle is malformed, unsupported, over budget, or conflicts."""


class ImportLimitError(ImportBundleError):
    """Actual bundle usage exceeded a resolved finite import limit."""


class ImportConflictError(ImportBundleError):
    """Bundle identity or durable records conflict with the destination."""


@dataclass(frozen=True)
class ImportLimits:
    input_bytes: int = 64 * 1024 * 1024
    decompressed_bytes: int = 64 * 1024 * 1024
    records: int = 100_000
    record_bytes: int = 1024 * 1024
    json_depth: int = 32
    collection_items: int = 10_000
    timeout_seconds: float = 30.0


_LIMIT_MAX = ImportLimits(
    input_bytes=256 * 1024 * 1024,
    decompressed_bytes=256 * 1024 * 1024,
    records=1_000_000,
    record_bytes=8 * 1024 * 1024,
    json_depth=64,
    collection_items=100_000,
    timeout_seconds=300.0,
)


def resolve_import_limits(**overrides: int | float | None) -> ImportLimits:
    """Resolve positive limits and clamp caller requests to reviewed maxima."""
    defaults = ImportLimits()
    resolved: dict[str, Any] = {}
    for field in asdict(defaults):
        requested = overrides.get(field)
        if requested is None:
            requested = getattr(defaults, field)
        if field == "timeout_seconds":
            if (
                isinstance(requested, bool)
                or not isinstance(requested, (int, float))
                or not math.isfinite(float(requested))
                or float(requested) <= 0
            ):
                raise ImportLimitError("timeout_seconds must be finite and positive")
            resolved[field] = min(float(requested), _LIMIT_MAX.timeout_seconds)
            continue
        if type(requested) is not int or requested <= 0:
            raise ImportLimitError(f"{field} must be a positive integer")
        resolved[field] = min(requested, getattr(_LIMIT_MAX, field))
    return ImportLimits(**resolved)


_TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "sessions": ("id", "started_at", "ended_at", "source", "meta"),
    "events": (
        "id",
        "idempotency_key",
        "session_id",
        "ts",
        "event_time",
        "role",
        "content",
        "tool_name",
        "tool_input",
        "tool_output",
        "origin",
        "tier",
        "meta",
        "provenance",
        "recall_class",
    ),
    "memories": (
        "id",
        "event_id",
        "tier",
        "content",
        "valid_from",
        "valid_to",
        "created_at",
    ),
    "lineage_tombstones": (
        "schema_version",
        "tombstone_id",
        "status",
        "erased_at",
    ),
    "corrections": (
        "id",
        "target_memory_id",
        "target_tombstone_id",
        "replacement_memory_id",
        "replacement_tombstone_id",
        "corrected_at",
        "origin",
        "session_id",
        "reason",
        "idempotency_key",
        "request_identity",
        "request_payload",
        "response_json",
    ),
    "entities": (
        "id",
        "name",
        "type",
        "norm_name",
        "first_seen",
        "last_seen",
    ),
    "entity_mentions": ("event_id", "entity_id", "observed_at"),
    "relation_evidence": (
        "event_id",
        "src_entity",
        "rel",
        "dst_entity",
        "observed_at",
        "weight",
    ),
}

_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "sessions": ("id",),
    "events": ("id",),
    "memories": ("id",),
    "lineage_tombstones": ("tombstone_id",),
    "corrections": ("id",),
    "entities": ("id",),
    "entity_mentions": ("event_id", "entity_id"),
    "relation_evidence": ("event_id", "src_entity", "rel", "dst_entity"),
}

_IMPORT_ORDER = tuple(_TABLE_FIELDS)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (UnicodeEncodeError, TypeError, ValueError) as exc:
        raise ImportBundleError("bundle contains a non-canonical JSON value") from exc


def canonical_export_bytes(bundle: Mapping[str, Any]) -> bytes:
    """Return the strict UTF-8 canonical serialization of a complete bundle."""
    return _canonical_bytes(dict(bundle))


def _digest(value: Any) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest_with_deadline(value: Any, check_deadline: Callable[[], None]) -> str:
    """Hash canonical JSON incrementally so timeout checks cover digest work."""
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        for chunk in encoder.iterencode(value):
            check_deadline()
            digest.update(chunk.encode("utf-8", errors="strict"))
    except (UnicodeEncodeError, TypeError, ValueError) as exc:
        raise ImportBundleError("bundle contains a non-canonical JSON value") from exc
    check_deadline()
    return _DIGEST_PREFIX + digest.hexdigest()


def _encode_sqlite(value: Any) -> Any:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return {
            "$haunt_sqlite": "blob",
            "base64": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "$haunt_sqlite": "real",
            "bits": value.hex(),
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ExportError(f"unsupported SQLite value type: {type(value).__name__}")


def _decode_sqlite(value: Any) -> Any:
    if not isinstance(value, dict):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise ImportBundleError("record values must be JSON scalars or SQLite tags")
    if set(value) == {"$haunt_sqlite", "base64"} and value["$haunt_sqlite"] == "blob":
        encoded = value["base64"]
        if not isinstance(encoded, str):
            raise ImportBundleError("SQLite BLOB base64 must be text")
        try:
            return base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ImportBundleError("invalid SQLite BLOB base64") from exc
    if set(value) == {"$haunt_sqlite", "bits"} and value["$haunt_sqlite"] == "real":
        bits = value["bits"]
        if not isinstance(bits, str):
            raise ImportBundleError("SQLite REAL bits must be text")
        try:
            result = float.fromhex(bits)
        except ValueError as exc:
            raise ImportBundleError("invalid SQLite REAL representation") from exc
        if math.isfinite(result):
            raise ImportBundleError("finite SQLite REAL values must use JSON numbers")
        return result
    raise ImportBundleError("unknown SQLite value representation")


def _encode_row(row: sqlite3.Row, fields: Sequence[str]) -> dict[str, Any]:
    return {field: _encode_sqlite(row[field]) for field in fields}


def _record_key(record: Mapping[str, Any], fields: Sequence[str]) -> bytes:
    return _canonical_bytes([record[field] for field in fields])


def _parse_cut(value: str | None) -> str:
    if value is None:
        return now_iso()
    try:
        return utc_iso(parse_iso(value))
    except (TypeError, ValueError) as exc:
        raise ExportError("cut must be a valid ISO timestamp") from exc


_EMPTY_TEMPORAL_CUT = "1970-01-01T00:00:00.000000+00:00"


def _default_temporal_cut(conn: sqlite3.Connection) -> str:
    """Return a stable high-water mark over durable write/audit clocks."""
    clocks: tuple[tuple[str, str], ...] = (
        ("sessions", "started_at"),
        ("sessions", "ended_at"),
        ("events", "ts"),
        ("memories", "created_at"),
        ("lineage_tombstones", "erased_at"),
        ("corrections", "corrected_at"),
        ("entity_mentions", "observed_at"),
        ("relation_evidence", "observed_at"),
    )
    latest = parse_iso(_EMPTY_TEMPORAL_CUT)
    for table, field in clocks:
        for row in conn.execute(
            f"SELECT {field} AS value FROM {table} WHERE {field} IS NOT NULL"
        ).fetchall():
            value = row["value"]
            if not isinstance(value, str):
                raise ExportError(
                    f"cannot derive temporal cut from legacy non-text {table}.{field}"
                )
            try:
                parsed = parse_iso(value)
            except (TypeError, ValueError) as exc:
                raise ExportError(
                    f"cannot derive temporal cut from invalid {table}.{field}"
                ) from exc
            latest = max(latest, parsed)
    return utc_iso(latest)


def _at_or_before(value: Any, cut: str, field: str) -> bool:
    if not isinstance(value, str):
        raise ExportError(f"cannot order legacy non-text timestamp {field}")
    try:
        return parse_iso(value) <= parse_iso(cut)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"cannot order invalid timestamp {field}") from exc


def _portable_registry_identity(namespace_id: str) -> dict[str, Any]:
    conn = _readonly_registry()
    try:
        identity = conn.execute(
            "SELECT namespace_id,canonical_label FROM namespace_identities "
            "WHERE namespace_id=?",
            (namespace_id,),
        ).fetchone()
        if identity is None:
            raise ExportError("namespace identity disappeared during export")
        aliases = [
            {
                "label": str(row["label"]),
                "is_canonical": bool(row["is_canonical"]),
                "source_alias_norm": (
                    None
                    if row["source_alias_norm"] is None
                    else str(row["source_alias_norm"])
                ),
            }
            for row in conn.execute(
                "SELECT label,is_canonical,source_alias_norm FROM namespace_aliases "
                "WHERE namespace_id=?",
                (namespace_id,),
            ).fetchall()
        ]
        aliases.sort(key=_alias_order_key)
        repositories = [
            str(row["repository_identity"])
            for row in conn.execute(
                "SELECT DISTINCT repository_identity FROM repository_bindings "
                "WHERE namespace_id=? AND repository_identity IS NOT NULL "
                "ORDER BY repository_identity",
                (namespace_id,),
            ).fetchall()
        ]
        return {
            "namespace_id": str(identity["namespace_id"]),
            "canonical_label": str(identity["canonical_label"]),
            "aliases": aliases,
            "repository_identities": repositories,
        }
    finally:
        conn.close()


def _alias_order_key(alias: Mapping[str, Any]) -> tuple[bool, str]:
    """One canonical alias order for both emission and strict validation."""
    return (
        not bool(alias["is_canonical"]),
        normalize_namespace_label(str(alias["label"])),
    )


def _select_export_records(conn: sqlite3.Connection, cut: str) -> dict[str, list[dict[str, Any]]]:
    version = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()
    if version is None or int(version["value"]) < SCHEMA_VERSION:
        raise ExportError(
            f"namespace must be upgraded to schema {SCHEMA_VERSION} before export"
        )

    event_rows = [
        row
        for row in conn.execute("SELECT * FROM events").fetchall()
        if _at_or_before(row["ts"], cut, "events.ts")
    ]
    event_ids = {row["id"] for row in event_rows}
    memory_rows = [
        row
        for row in conn.execute("SELECT * FROM memories").fetchall()
        if row["event_id"] in event_ids
        and _at_or_before(row["created_at"], cut, "memories.created_at")
    ]
    all_correction_rows = conn.execute("SELECT * FROM corrections").fetchall()
    future_replacements = {
        row["replacement_memory_id"]
        for row in all_correction_rows
        if row["replacement_memory_id"] is not None
        and not _at_or_before(row["corrected_at"], cut, "corrections.corrected_at")
    }
    # Correction plus replacement is one logical commit. The timestamps are
    # recorded within that transaction and can differ by microseconds; a cut
    # between them must not expose the replacement as an unrelated current row.
    memory_rows = [row for row in memory_rows if row["id"] not in future_replacements]
    retained_event_ids = {row["event_id"] for row in memory_rows}
    event_rows = [row for row in event_rows if row["id"] in retained_event_ids]
    event_ids = {row["id"] for row in event_rows}
    memory_ids = {row["id"] for row in memory_rows}
    correction_rows = [
        row
        for row in all_correction_rows
        if _at_or_before(row["corrected_at"], cut, "corrections.corrected_at")
    ]
    # A correction is atomic with its replacement.  A corrupt/incomplete
    # source cannot be exported as if the edge still existed.
    correction_rows = [
        row
        for row in correction_rows
        if (row["target_memory_id"] is None or row["target_memory_id"] in memory_ids)
        and (
            row["replacement_memory_id"] is None
            or row["replacement_memory_id"] in memory_ids
        )
    ]
    tombstone_ids = {
        value
        for row in correction_rows
        for value in (row["target_tombstone_id"], row["replacement_tombstone_id"])
        if value is not None
    }
    tombstone_rows = [
        row
        for row in conn.execute("SELECT * FROM lineage_tombstones").fetchall()
        if row["tombstone_id"] in tombstone_ids
    ]
    session_rows = [
        row
        for row in conn.execute("SELECT * FROM sessions").fetchall()
        if _at_or_before(row["started_at"], cut, "sessions.started_at")
    ]

    # Project the durable state as it existed at the cut without mutating the
    # source snapshot. A later session close/correction must not leak through a
    # historical export merely because the row is read today.
    projected_sessions: list[dict[str, Any]] = []
    for row in session_rows:
        projected = dict(row)
        if projected["ended_at"] is not None and not _at_or_before(
            projected["ended_at"], cut, "sessions.ended_at"
        ):
            projected["ended_at"] = None
        projected_sessions.append(projected)
    projected_memories: list[dict[str, Any]] = []
    for row in memory_rows:
        projected = dict(row)
        if projected["valid_to"] is not None and not _at_or_before(
            projected["valid_to"], cut, "memories.valid_to"
        ):
            projected["valid_to"] = None
        projected_memories.append(projected)

    mention_rows = [
        row
        for row in conn.execute("SELECT * FROM entity_mentions").fetchall()
        if row["event_id"] in event_ids
        and _at_or_before(row["observed_at"], cut, "entity_mentions.observed_at")
    ]
    evidence_rows = [
        row
        for row in conn.execute("SELECT * FROM relation_evidence").fetchall()
        if row["event_id"] in event_ids
        and _at_or_before(row["observed_at"], cut, "relation_evidence.observed_at")
    ]
    entity_ids = {row["entity_id"] for row in mention_rows}
    entity_ids.update(row["src_entity"] for row in evidence_rows)
    entity_ids.update(row["dst_entity"] for row in evidence_rows)
    entity_clocks: dict[Any, list[str]] = {entity_id: [] for entity_id in entity_ids}
    for row in mention_rows:
        entity_clocks[row["entity_id"]].append(str(row["observed_at"]))
    for row in evidence_rows:
        observed_at = str(row["observed_at"])
        entity_clocks[row["src_entity"]].append(observed_at)
        entity_clocks[row["dst_entity"]].append(observed_at)

    def ordered_clock(values: list[str], *, latest: bool) -> str:
        if not values:
            raise ExportError("entity has no retained observation at temporal cut")
        try:
            return sorted(values, key=lambda value: (parse_iso(value), value))[
                -1 if latest else 0
            ]
        except (TypeError, ValueError) as exc:
            raise ExportError("entity observation has invalid timestamp") from exc

    entity_rows: list[dict[str, Any]] = []
    for row in conn.execute("SELECT * FROM entities").fetchall():
        if row["id"] not in entity_ids:
            continue
        projected = dict(row)
        clocks = entity_clocks[row["id"]]
        projected["first_seen"] = ordered_clock(clocks, latest=False)
        projected["last_seen"] = ordered_clock(clocks, latest=True)
        entity_rows.append(projected)

    raw: dict[str, list[Mapping[str, Any]]] = {
        "sessions": projected_sessions,
        "events": event_rows,
        "memories": projected_memories,
        "lineage_tombstones": tombstone_rows,
        "corrections": correction_rows,
        "entities": entity_rows,
        "entity_mentions": mention_rows,
        "relation_evidence": evidence_rows,
    }
    encoded: dict[str, list[dict[str, Any]]] = {}
    for table, fields in _TABLE_FIELDS.items():
        records = [_encode_row(row, fields) for row in raw[table]]
        records.sort(key=lambda item: _record_key(item, _PRIMARY_KEYS[table]))
        encoded[table] = records
    return encoded


def _export_after_cut_hook() -> None:
    """Deterministic concurrency-test hook after the read snapshot is pinned."""


def _build_namespace_export_snapshot(
    namespace: str, cut: str | None
) -> tuple[str, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Read one registry-bracketed, zero-write namespace snapshot."""
    with _namespace_migration_lock():
        identity_before = resolve_namespace_identity(namespace)
        if identity_before is None:
            raise ExportError(f"unknown namespace: {namespace}")
        namespace_id = str(identity_before["namespace_id"])
        registry_before = _portable_registry_identity(namespace_id)
        store = open_namespace_identity_readonly(
            namespace_id,
            expected_db_path=str(identity_before["db_path"]),
            expected_db_device=int(identity_before["db_device"]),
            expected_db_inode=int(identity_before["db_inode"]),
        )
        try:
            # The guarded read-only connection already pins a physical SQLite
            # snapshot. BEGIN additionally makes the multi-statement semantic
            # boundary explicit; ROLLBACK changes connection state only.
            store.conn.execute("BEGIN")
            try:
                resolved_cut = (
                    _parse_cut(cut)
                    if cut is not None
                    else _default_temporal_cut(store.conn)
                )
                _export_after_cut_hook()
                records = _select_export_records(store.conn, resolved_cut)
            finally:
                store.conn.rollback()
        finally:
            # A guarded zero-write snapshot compares the durable primary/WAL
            # state at close. Concurrent observe/correct/purge therefore fails
            # this attempt instead of publishing stale or cross-statement data.
            store.close()

        identity_after = resolve_namespace_id(namespace_id)
        if identity_after is None:
            raise ExportError("namespace identity disappeared during export")
        registry_after = _portable_registry_identity(namespace_id)
        stable_fields = (
            "namespace_id",
            "canonical_label",
            "canonical_label_norm",
            "db_path",
            "db_device",
            "db_inode",
            "updated_at",
        )
        if any(
            identity_before[field] != identity_after[field] for field in stable_fields
        ) or registry_before != registry_after:
            raise ExportError("namespace registry identity changed during export")
        return resolved_cut, registry_before, records


def build_namespace_export(
    namespace: str,
    *,
    cut: str | None = None,
    exported_at: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic semantic export from zero-write snapshots."""
    created = _parse_cut(exported_at)
    for attempt in range(3):
        try:
            resolved_cut, registry_identity, records = _build_namespace_export_snapshot(
                namespace, cut
            )
            break
        except NamespacePathError as exc:
            if not is_concurrent_registry_change(exc) or attempt == 2:
                raise
    else:  # pragma: no cover - loop always breaks or raises
        raise AssertionError("unreachable export snapshot retry exhaustion")

    semantic = {
        "format": FORMAT_NAME,
        "version": {"major": FORMAT_MAJOR, "minor": FORMAT_MINOR},
        "temporal_cut": resolved_cut,
        "namespace": registry_identity,
        "records": records,
    }
    counts = {table: len(rows) for table, rows in records.items()}
    return {
        **semantic,
        "creation": {
            "exported_at": created,
            "media_type": MEDIA_TYPE,
            "volatile_fields": ["creation.exported_at"],
        },
        "manifest": {
            "semantic_digest": _digest(semantic),
            "record_counts": counts,
            "total_records": sum(counts.values()),
        },
    }


_NUMBER_RE = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


class _StrictJSONParser:
    """Small incremental tokenizer/parser with allocation-time limit checks.

    It consumes the byte stream token by token. Nesting, per-collection items,
    total record count, and per-record bytes are charged while parsing, before
    a later validation pass can trust any declared manifest values.
    """

    def __init__(
        self,
        raw: bytes,
        limits: ImportLimits,
        check_deadline: Callable[[], None],
    ):
        self.raw = raw
        self.limits = limits
        self.pos = 0
        self.record_count = 0
        self._record_start: int | None = None
        self._check_deadline = check_deadline
        self._next_deadline_check = 0

    def parse(self) -> Any:
        value = self._value(0, ())
        self._space()
        if self.pos != len(self.raw):
            raise ImportBundleError("trailing bytes after JSON value")
        return value

    def _check_record_bytes(self) -> None:
        if (
            self._record_start is not None
            and self.pos - self._record_start > self.limits.record_bytes
        ):
            raise ImportLimitError(
                f"actual record bytes exceed {self.limits.record_bytes}"
            )

    def _take(self) -> int:
        if self.pos >= len(self.raw):
            raise ImportBundleError("unexpected end of JSON")
        byte = self.raw[self.pos]
        self.pos += 1
        if self.pos >= self._next_deadline_check:
            self._check_deadline()
            self._next_deadline_check = self.pos + 4096
        self._check_record_bytes()
        return byte

    def _space(self) -> None:
        while self.pos < len(self.raw) and self.raw[self.pos] in b" \t\r\n":
            self._take()

    def _value(self, depth: int, path: tuple[str, ...]) -> Any:
        self._space()
        if self.pos >= len(self.raw):
            raise ImportBundleError("unexpected end of JSON")
        byte = self.raw[self.pos]
        if byte == ord("{"):
            return self._object(depth + 1, path)
        if byte == ord("["):
            return self._array(depth + 1, path)
        if byte == ord('"'):
            return self._string()
        for token, value in ((b"true", True), (b"false", False), (b"null", None)):
            if self.raw.startswith(token, self.pos):
                for _ in token:
                    self._take()
                return value
        if byte == ord("-") or ord("0") <= byte <= ord("9"):
            return self._number()
        raise ImportBundleError(f"invalid JSON token at byte {self.pos}")

    def _enter(self, depth: int) -> None:
        if depth > self.limits.json_depth:
            raise ImportLimitError(
                f"actual JSON depth exceeds {self.limits.json_depth}"
            )

    def _object(self, depth: int, path: tuple[str, ...]) -> dict[str, Any]:
        self._enter(depth)
        if self._take() != ord("{"):
            raise AssertionError("object parser desynchronized")
        out: dict[str, Any] = {}
        self._space()
        if self.pos < len(self.raw) and self.raw[self.pos] == ord("}"):
            self._take()
            return out
        while True:
            self._space()
            if self.pos >= len(self.raw) or self.raw[self.pos] != ord('"'):
                raise ImportBundleError("JSON object key must be a string")
            key = self._string()
            if key in out:
                raise ImportBundleError(f"duplicate JSON object key: {key}")
            if len(out) + 1 > self.limits.collection_items:
                raise ImportLimitError(
                    f"actual collection items exceed {self.limits.collection_items}"
                )
            self._space()
            if self._take() != ord(":"):
                raise ImportBundleError("JSON object key must be followed by ':'")
            out[key] = self._value(depth, path + (key,))
            self._space()
            delimiter = self._take()
            if delimiter == ord("}"):
                return out
            if delimiter != ord(","):
                raise ImportBundleError("JSON object items must be comma-separated")

    def _array(self, depth: int, path: tuple[str, ...]) -> list[Any]:
        self._enter(depth)
        if self._take() != ord("["):
            raise AssertionError("array parser desynchronized")
        out: list[Any] = []
        self._space()
        if self.pos < len(self.raw) and self.raw[self.pos] == ord("]"):
            self._take()
            return out
        records_array = len(path) == 2 and path[0] == "records" and path[1] in _TABLE_FIELDS
        while True:
            if records_array:
                self.record_count += 1
                if self.record_count > self.limits.records:
                    raise ImportLimitError(
                        f"actual record count exceeds {self.limits.records}"
                    )
                old_start = self._record_start
                self._record_start = self.pos
                try:
                    item = self._value(depth, path + (str(len(out)),))
                finally:
                    self._record_start = old_start
            else:
                if len(out) + 1 > self.limits.collection_items:
                    raise ImportLimitError(
                        f"actual collection items exceed {self.limits.collection_items}"
                    )
                item = self._value(depth, path + (str(len(out)),))
            out.append(item)
            self._space()
            delimiter = self._take()
            if delimiter == ord("]"):
                return out
            if delimiter != ord(","):
                raise ImportBundleError("JSON array items must be comma-separated")

    def _string(self) -> str:
        start = self.pos
        if self._take() != ord('"'):
            raise AssertionError("string parser desynchronized")
        escaped = False
        while True:
            byte = self._take()
            if escaped:
                escaped = False
                continue
            if byte == ord("\\"):
                escaped = True
                continue
            if byte == ord('"'):
                break
            if byte < 0x20:
                raise ImportBundleError("unescaped control character in JSON string")
        token = self.raw[start:self.pos]
        try:
            result = json.loads(token.decode("utf-8", errors="strict"))
            result.encode("utf-8", errors="strict")
        except (UnicodeDecodeError, UnicodeEncodeError, json.JSONDecodeError) as exc:
            raise ImportBundleError("invalid UTF-8 JSON string") from exc
        if not isinstance(result, str):
            raise AssertionError("JSON string decoder returned non-text")
        return result

    def _number(self) -> int | float:
        start = self.pos
        allowed = b"-+0123456789.eE"
        while self.pos < len(self.raw) and self.raw[self.pos] in allowed:
            self._take()
        token = self.raw[start:self.pos]
        if not _NUMBER_RE.fullmatch(token):
            raise ImportBundleError("invalid JSON number")
        try:
            if b"." not in token and b"e" not in token.lower():
                return int(token)
            value = float(token)
        except (OverflowError, ValueError) as exc:
            raise ImportBundleError("invalid JSON number") from exc
        if not math.isfinite(value):
            raise ImportBundleError("non-finite JSON number is forbidden")
        return value


def _load_json(
    raw: bytes, limits: ImportLimits, check_deadline: Callable[[], None]
) -> dict[str, Any]:
    check_deadline()
    if len(raw) > limits.input_bytes:
        raise ImportLimitError(f"actual input bytes exceed {limits.input_bytes}")
    # v1 intentionally has no compression container.  Charge both limits so a
    # later minor that adds compression cannot accidentally ignore expansion.
    if len(raw) > limits.decompressed_bytes:
        raise ImportLimitError(
            f"actual decompressed bytes exceed {limits.decompressed_bytes}"
        )
    if raw.startswith((b"\x1f\x8b", b"PK\x03\x04", b"BZh")):
        raise ImportBundleError("compressed input is not supported by format v1.0")
    try:
        value = _StrictJSONParser(raw, limits, check_deadline).parse()
    except ImportBundleError:
        raise
    except (RecursionError, UnicodeError) as exc:
        raise ImportBundleError("malformed export JSON") from exc
    if not isinstance(value, dict):
        raise ImportBundleError("export root must be an object")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ImportBundleError(f"invalid {where} fields ({'; '.join(detail)})")


def _collection_size(
    value: Any, check_deadline: Callable[[], None] | None = None
) -> int:
    if check_deadline is not None:
        check_deadline()
    if isinstance(value, dict):
        return len(value) + sum(
            _collection_size(item, check_deadline) for item in value.values()
        )
    if isinstance(value, list):
        return len(value) + sum(
            _collection_size(item, check_deadline) for item in value
        )
    return 0


def _semantic_from_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": bundle["format"],
        "version": bundle["version"],
        "temporal_cut": bundle["temporal_cut"],
        "namespace": bundle["namespace"],
        "records": bundle["records"],
    }


@dataclass(frozen=True)
class _ValidatedBundle:
    raw: dict[str, Any]
    semantic_digest: str
    namespace: dict[str, Any]
    records: dict[str, list[dict[str, Any]]]
    decoded: dict[str, list[dict[str, Any]]]
    limits: ImportLimits


def _validate_bundle(
    bundle: dict[str, Any], limits: ImportLimits, check_deadline: Callable[[], None]
) -> _ValidatedBundle:
    _require_keys(
        bundle,
        {"format", "version", "temporal_cut", "namespace", "records", "creation", "manifest"},
        "export root",
    )
    if bundle["format"] != FORMAT_NAME:
        raise ImportBundleError(f"unsupported export format: {bundle['format']!r}")
    version = bundle["version"]
    if not isinstance(version, dict):
        raise ImportBundleError("version must be an object")
    _require_keys(version, {"major", "minor"}, "version")
    if type(version["major"]) is not int or type(version["minor"]) is not int:
        raise ImportBundleError("version major/minor must be integers")
    if version["major"] != FORMAT_MAJOR:
        raise ImportBundleError(
            f"unsupported export major {version['major']}; supported major is {FORMAT_MAJOR}"
        )
    if version["minor"] != FORMAT_MINOR:
        raise ImportBundleError(
            f"unsupported export minor {version['minor']}; supported minor is {FORMAT_MINOR}"
        )
    _parse_cut(bundle["temporal_cut"])

    creation = bundle["creation"]
    if not isinstance(creation, dict):
        raise ImportBundleError("creation must be an object")
    _require_keys(creation, {"exported_at", "media_type", "volatile_fields"}, "creation")
    _parse_cut(creation["exported_at"])
    if creation["media_type"] != MEDIA_TYPE:
        raise ImportBundleError("unsupported export media type")
    if creation["volatile_fields"] != ["creation.exported_at"]:
        raise ImportBundleError("invalid volatile field declaration")

    namespace = bundle["namespace"]
    if not isinstance(namespace, dict):
        raise ImportBundleError("namespace must be an object")
    _require_keys(
        namespace,
        {"namespace_id", "canonical_label", "aliases", "repository_identities"},
        "namespace",
    )
    namespace_id = namespace["namespace_id"]
    canonical = namespace["canonical_label"]
    if not isinstance(namespace_id, str) or not namespace_id or len(namespace_id.encode()) > 256:
        raise ImportBundleError("namespace_id must be nonempty and at most 256 UTF-8 bytes")
    if not isinstance(canonical, str) or canonical != safe_name(canonical):
        raise ImportBundleError("canonical_label is not a safe exact namespace label")
    aliases = namespace["aliases"]
    if not isinstance(aliases, list) or not aliases:
        raise ImportBundleError("namespace aliases must be a nonempty array")
    alias_norms: set[str] = set()
    canonical_count = 0
    for alias in aliases:
        check_deadline()
        if not isinstance(alias, dict):
            raise ImportBundleError("namespace alias must be an object")
        _require_keys(alias, {"label", "is_canonical", "source_alias_norm"}, "namespace alias")
        label = alias["label"]
        if not isinstance(label, str) or label != safe_name(label):
            raise ImportBundleError("namespace alias is not a safe exact label")
        if type(alias["is_canonical"]) is not bool:
            raise ImportBundleError("namespace alias is_canonical must be boolean")
        source = alias["source_alias_norm"]
        if source is not None and (not isinstance(source, str) or not source):
            raise ImportBundleError("source_alias_norm must be nonempty text or null")
        norm = normalize_namespace_label(label)
        if norm in alias_norms:
            raise ImportBundleError("namespace aliases collide after normalization")
        alias_norms.add(norm)
        if alias["is_canonical"]:
            canonical_count += 1
            if label != canonical:
                raise ImportBundleError("canonical alias does not match canonical_label")
    if canonical_count != 1:
        raise ImportBundleError("namespace must contain exactly one canonical alias")
    expected_alias_order = sorted(aliases, key=_alias_order_key)
    if aliases != expected_alias_order:
        raise ImportBundleError("namespace aliases are not canonically ordered")
    alias_sources = {
        normalize_namespace_label(str(alias["label"])): alias["source_alias_norm"]
        for alias in aliases
    }
    for norm, source in alias_sources.items():
        check_deadline()
        if source is None:
            continue
        if source != normalize_namespace_label(source):
            raise ImportBundleError("source_alias_norm must itself be normalized")
        if source not in alias_norms:
            raise ImportBundleError("source_alias_norm references a missing alias")
        if source == norm:
            raise ImportBundleError("namespace alias lineage cannot reference itself")
    canonical_norm = normalize_namespace_label(canonical)
    if alias_sources[canonical_norm] is not None:
        raise ImportBundleError("canonical namespace alias cannot depend on another alias")
    for start in alias_sources:
        check_deadline()
        seen_lineage: set[str] = set()
        current: str | None = start
        while current is not None:
            check_deadline()
            if current in seen_lineage:
                raise ImportBundleError("namespace alias lineage contains a cycle")
            seen_lineage.add(current)
            current = alias_sources[current]
    repositories = namespace["repository_identities"]
    if not isinstance(repositories, list):
        raise ImportBundleError("repository_identities must be an array")
    if repositories != sorted(set(repositories)):
        raise ImportBundleError("repository_identities must be unique and sorted")
    for repository in repositories:
        check_deadline()
        if (
            not isinstance(repository, str)
            or repository_identity("https://" + repository) != repository
        ):
            raise ImportBundleError("repository identity is not canonical")

    records = bundle["records"]
    if not isinstance(records, dict):
        raise ImportBundleError("records must be an object")
    _require_keys(records, set(_TABLE_FIELDS), "records")
    decoded: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for table in _IMPORT_ORDER:
        check_deadline()
        rows = records[table]
        if not isinstance(rows, list):
            raise ImportBundleError(f"records.{table} must be an array")
        total += len(rows)
        if total > limits.records:
            raise ImportLimitError(f"actual record count exceeds {limits.records}")
        seen: set[bytes] = set()
        prior_key: bytes | None = None
        decoded_rows: list[dict[str, Any]] = []
        for record in rows:
            check_deadline()
            if not isinstance(record, dict):
                raise ImportBundleError(f"records.{table} item must be an object")
            _require_keys(record, set(_TABLE_FIELDS[table]), f"records.{table} item")
            record_size = len(_canonical_bytes(record))
            check_deadline()
            if record_size > limits.record_bytes:
                raise ImportLimitError(
                    f"records.{table} item exceeds {limits.record_bytes} bytes"
                )
            if _collection_size(record, check_deadline) > limits.collection_items:
                raise ImportLimitError(
                    f"records.{table} item exceeds {limits.collection_items} collection items"
                )
            key = _record_key(record, _PRIMARY_KEYS[table])
            if key in seen:
                raise ImportBundleError(f"duplicate records.{table} identity")
            if prior_key is not None and key <= prior_key:
                raise ImportBundleError(f"records.{table} are not canonically ordered")
            seen.add(key)
            prior_key = key
            decoded_rows.append(
                {field: _decode_sqlite(record[field]) for field in _TABLE_FIELDS[table]}
            )
        decoded[table] = decoded_rows

    manifest = bundle["manifest"]
    if not isinstance(manifest, dict):
        raise ImportBundleError("manifest must be an object")
    _require_keys(manifest, {"semantic_digest", "record_counts", "total_records"}, "manifest")
    counts = {table: len(records[table]) for table in _IMPORT_ORDER}
    if manifest["record_counts"] != counts or manifest["total_records"] != total:
        raise ImportBundleError("manifest record counts do not match actual records")
    expected_digest = _digest_with_deadline(
        _semantic_from_bundle(bundle), check_deadline
    )
    supplied_digest = manifest["semantic_digest"]
    if supplied_digest != expected_digest:
        raise ImportBundleError("semantic digest mismatch")
    check_deadline()
    _validate_records_in_scratch(decoded, check_deadline)
    return _ValidatedBundle(
        raw=bundle,
        semantic_digest=expected_digest,
        namespace=namespace,
        records=records,
        decoded=decoded,
        limits=limits,
    )


def _validate_records_in_scratch(
    records: dict[str, list[dict[str, Any]]],
    check_deadline: Callable[[], None],
) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.create_function("haunt_privacy_purge_authorized", 0, lambda: 0)
    deadline_error: list[ImportLimitError] = []

    def progress() -> int:
        try:
            check_deadline()
        except ImportLimitError as exc:
            deadline_error.append(exc)
            return 1
        return 0

    conn.set_progress_handler(progress, 1_000)
    try:
        check_deadline()
        conn.execute("PRAGMA foreign_keys=ON")
        _init_namespace_schema(conn)
        _ensure_namespace_schema(conn)
        check_deadline()
        _validate_structured_provenance(records["events"], check_deadline)
        _validate_record_references(records, check_deadline)
        _apply_records(
            conn,
            records,
            semantic_digest=None,
            check_deadline=check_deadline,
        )
        # SQLite affinity may silently coerce JSON booleans/numbers inserted
        # into TEXT columns. The scratch store must preserve every decoded
        # scalar and tagged type exactly before a destination can be touched.
        for table in _IMPORT_ORDER:
            for record in records[table]:
                check_deadline()
                if _existing_row_matches(conn, table, record) is not True:
                    raise ImportBundleError(
                        f"records.{table} changes SQLite value or type on insert"
                    )
        check_deadline()
        conn.rollback()
    except ImportBundleError:
        raise
    except (sqlite3.Error, ValueError, TypeError, OverflowError) as exc:
        if deadline_error:
            raise deadline_error[0] from exc
        raise ImportBundleError(f"invalid durable record set: {exc}") from exc
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()


def _identity_token(value: Any) -> bytes:
    if value is None:
        raise ImportBundleError("durable record identity cannot be null")
    try:
        return _canonical_bytes(_encode_sqlite(value))
    except ExportError as exc:
        raise ImportBundleError("invalid durable record identity") from exc


def _validate_record_references(
    records: dict[str, list[dict[str, Any]]],
    check_deadline: Callable[[], None],
) -> None:
    """Prove bundle closure even where legacy SQLite tables lack FKs."""
    for table, rows in records.items():
        for row in rows:
            check_deadline()
            for field in _PRIMARY_KEYS[table]:
                _identity_token(row[field])

    def identities(table: str, field: str) -> set[bytes]:
        result: set[bytes] = set()
        for row in records[table]:
            check_deadline()
            result.add(_identity_token(row[field]))
        return result

    sessions = identities("sessions", "id")
    events = identities("events", "id")
    memories = identities("memories", "id")
    tombstones = identities("lineage_tombstones", "tombstone_id")
    entities = identities("entities", "id")

    def require(value: Any, available: set[bytes], message: str) -> None:
        if _identity_token(value) not in available:
            raise ImportBundleError(message)

    for event in records["events"]:
        check_deadline()
        require(event["session_id"], sessions, "event references missing session")
    for memory in records["memories"]:
        check_deadline()
        require(memory["event_id"], events, "memory references missing event")
    for correction in records["corrections"]:
        check_deadline()
        if correction["target_memory_id"] is not None:
            require(
                correction["target_memory_id"], memories,
                "correction references missing target memory",
            )
        else:
            require(
                correction["target_tombstone_id"], tombstones,
                "correction references missing target tombstone",
            )
        if correction["replacement_memory_id"] is not None:
            require(
                correction["replacement_memory_id"], memories,
                "correction references missing replacement memory",
            )
        if correction["replacement_tombstone_id"] is not None:
            require(
                correction["replacement_tombstone_id"], tombstones,
                "correction references missing replacement tombstone",
            )
        if correction["session_id"] is not None:
            require(
                correction["session_id"], sessions,
                "correction references missing session",
            )
    for mention in records["entity_mentions"]:
        check_deadline()
        require(mention["event_id"], events, "mention references missing event")
        require(mention["entity_id"], entities, "mention references missing entity")
    for evidence in records["relation_evidence"]:
        check_deadline()
        require(evidence["event_id"], events, "relation evidence references missing event")
        require(evidence["src_entity"], entities, "relation evidence references missing source")
        require(evidence["dst_entity"], entities, "relation evidence references missing destination")


def _validate_structured_provenance(
    events: Iterable[Mapping[str, Any]],
    check_deadline: Callable[[], None],
) -> None:
    for event in events:
        check_deadline()
        provenance = event["provenance"]
        if provenance is None:
            continue
        if not isinstance(provenance, str):
            raise ImportBundleError("non-null structured provenance must be SQLite TEXT")
        try:
            parsed = json.loads(provenance)
        except json.JSONDecodeError as exc:
            raise ImportBundleError("invalid stored structured provenance JSON") from exc
        check_deadline()
        if not isinstance(parsed, dict):
            raise ImportBundleError("structured provenance must be an object")
        origin = event["origin"]
        tool_name = event["tool_name"]
        if not isinstance(origin, str):
            raise ImportBundleError("structured provenance requires text origin")
        if tool_name is not None and not isinstance(tool_name, str):
            raise ImportBundleError("structured provenance requires text tool_name")
        try:
            validated = validate_provenance(
                parsed,
                origin=origin,
                channel=parsed.get("channel"),
                tool_name=tool_name,
                producer_call_id=parsed.get("producer_call_id"),
            )
        except ValueError as exc:
            raise ImportBundleError(f"invalid stored structured provenance: {exc}") from exc
        canonical = json.dumps(validated, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if provenance != canonical:
            raise ImportBundleError("structured provenance is not canonical v1 TEXT")
        recall_class = event["recall_class"]
        if recall_class is not None and recall_class not in RECALL_CLASSES:
            raise ImportBundleError("invalid recall_class")


def _same_sqlite_value(left: Any, right: Any) -> bool:
    try:
        return _canonical_bytes(_encode_sqlite(left)) == _canonical_bytes(_encode_sqlite(right))
    except ExportError:
        return False


def _existing_row_matches(
    conn: sqlite3.Connection, table: str, record: Mapping[str, Any]
) -> bool | None:
    keys = _PRIMARY_KEYS[table]
    where = " AND ".join(f"{field} IS ?" for field in keys)
    row = conn.execute(
        f"SELECT {', '.join(_TABLE_FIELDS[table])} FROM {table} WHERE {where}",
        tuple(record[field] for field in keys),
    ).fetchone()
    if row is None:
        return None
    return all(_same_sqlite_value(row[field], record[field]) for field in _TABLE_FIELDS[table])


def _apply_records(
    conn: sqlite3.Connection,
    records: dict[str, list[dict[str, Any]]],
    *,
    semantic_digest: str | None,
    check_deadline: Callable[[], None],
) -> dict[str, int]:
    receipt_key = None if semantic_digest is None else _RECEIPT_PREFIX + semantic_digest
    receipt_value = None if semantic_digest is None else json.dumps(
        {"format": FORMAT_NAME, "semantic_digest": semantic_digest},
        sort_keys=True,
        separators=(",", ":"),
    )
    has_receipt = False
    if receipt_key is not None:
        receipt = conn.execute("SELECT value FROM meta WHERE key=?", (receipt_key,)).fetchone()
        if receipt is not None:
            if str(receipt["value"]) != receipt_value:
                raise ImportConflictError("import receipt conflicts with bundle identity")
            has_receipt = True

    missing: dict[str, list[dict[str, Any]]] = {table: [] for table in _IMPORT_ORDER}
    for table in _IMPORT_ORDER:
        for record in records[table]:
            check_deadline()
            match = _existing_row_matches(conn, table, record)
            if match is False:
                key = tuple(record[field] for field in _PRIMARY_KEYS[table])
                raise ImportConflictError(f"records.{table} identity conflicts: {key!r}")
            if match is None:
                missing[table].append(record)

    if has_receipt:
        if any(missing.values()):
            raise ImportConflictError(
                "import receipt exists but durable bundle records are missing"
            )
        return {table: 0 for table in _IMPORT_ORDER}

    for table in _IMPORT_ORDER:
        fields = _TABLE_FIELDS[table]
        placeholders = ",".join("?" for _ in fields)
        for record in missing[table]:
            check_deadline()
            conn.execute(
                f"INSERT INTO {table}({', '.join(fields)}) VALUES ({placeholders})",
                tuple(record[field] for field in fields),
            )

    # Rebuild only destination-local projections. Source embeddings and jobs
    # are absent by construction. Only destination-embeddable SQLite TEXT is
    # indexed and queued; legacy BLOB content remains durable and exact without
    # being mislabeled as text through ``str(blob)``.
    for record in records["memories"]:
        check_deadline()
        conn.execute("DELETE FROM memories_fts WHERE id=?", (record["id"],))
        content = record["content"]
        if isinstance(content, str):
            conn.execute(
                "INSERT INTO memories_fts(id,content) VALUES (?,?)",
                (record["id"], content),
            )
        if isinstance(content, str) and content.strip():
            conn.execute(
                "INSERT OR IGNORE INTO embedding_jobs(memory_id,queued_at) VALUES (?,?)",
                (record["id"], record["created_at"]),
            )
    triples: set[tuple[Any, Any, Any]] = set()
    for record in records["relation_evidence"]:
        check_deadline()
        triples.add((record["src_entity"], record["rel"], record["dst_entity"]))
    for src, rel, dst in triples:
        check_deadline()
        _refresh_relation(conn, src, rel, dst)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES ('graph_evidence_version','1')"
    )
    if receipt_key is not None:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES (?,?)", (receipt_key, receipt_value)
        )
    return {table: len(missing[table]) for table in _IMPORT_ORDER}


def _registry_preflight(
    namespace: Mapping[str, Any], check_deadline: Callable[[], None]
) -> dict[str, Any] | None:
    namespace_id = str(namespace["namespace_id"])
    check_deadline()
    existing = resolve_namespace_id(namespace_id)
    for alias in namespace["aliases"]:
        check_deadline()
        owner = resolve_namespace_identity(str(alias["label"]))
        if owner is not None and str(owner["namespace_id"]) != namespace_id:
            raise ImportConflictError(
                f"namespace alias collision: {alias['label']!r} belongs elsewhere"
            )
    if existing is None:
        return None
    expected_aliases = [
        (str(item["label"]), bool(item["is_canonical"]), item["source_alias_norm"])
        for item in namespace["aliases"]
    ]
    check_deadline()
    actual_aliases = [
        (str(item["label"]), bool(item["is_canonical"]), item["source_alias_norm"])
        for item in existing["aliases"]
    ]
    if (
        str(existing["canonical_label"]) != str(namespace["canonical_label"])
        or actual_aliases != expected_aliases
    ):
        raise ImportConflictError("existing namespace identity or aliases differ")
    conn = _readonly_registry()
    try:
        check_deadline()
        repositories = [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT repository_identity FROM repository_bindings "
                "WHERE namespace_id=? AND repository_identity IS NOT NULL "
                "ORDER BY repository_identity",
                (namespace_id,),
            ).fetchall()
        ]
        check_deadline()
    finally:
        conn.close()
    if repositories != list(namespace["repository_identities"]):
        raise ImportConflictError("existing namespace repository identity differs")
    return existing


def _require_same_selected_storage(
    current: Mapping[str, Any], selected: Mapping[str, Any]
) -> None:
    for field in ("namespace_id", "db_path", "db_device", "db_inode"):
        if current[field] != selected[field]:
            raise ImportConflictError(
                "existing namespace storage identity changed during import"
            )


def _existing_import_after_preflight_hook(_existing: Mapping[str, Any]) -> None:
    """Deterministic concurrency-test hook before exact-identity writer open."""


def _publish_new_import(
    validated: _ValidatedBundle, check_deadline: Callable[[], None]
) -> dict[str, Any]:
    namespace = validated.namespace
    label = str(namespace["canonical_label"])
    namespace_id = str(namespace["namespace_id"])
    target = namespaces_dir() / f"{label}.db"
    now = now_iso()
    with _namespace_migration_lock():
        with _sqlite_configuration_lock():
            # Repeat every collision check while holding both writer locks.
            if _registry_preflight(namespace, check_deadline) is not None:
                raise ImportConflictError("namespace appeared during import")
            registry = _registry()
            claim = None
            try:
                registry.execute("BEGIN IMMEDIATE")
                _validate_unmapped_namespace_target(target)
                for alias in namespace["aliases"]:
                    check_deadline()
                    collision = registry.execute(
                        "SELECT namespace_id FROM namespace_aliases WHERE normalized_label=?",
                        (normalize_namespace_label(str(alias["label"])),),
                    ).fetchone()
                    if collision is not None:
                        raise ImportConflictError("namespace alias appeared during import")
                for repository in namespace["repository_identities"]:
                    check_deadline()
                    collision = registry.execute(
                        "SELECT namespace_id FROM repository_bindings WHERE repository_identity=?",
                        (repository,),
                    ).fetchone()
                    if collision is not None:
                        raise ImportConflictError("repository identity belongs elsewhere")
                inserted: dict[str, int] = {}

                def prepare(staged: sqlite3.Connection) -> None:
                    nonlocal inserted
                    _ensure_namespace_schema(staged)
                    staged.execute("BEGIN IMMEDIATE")
                    try:
                        inserted = _apply_records(
                            staged,
                            validated.decoded,
                            semantic_digest=validated.semantic_digest,
                            check_deadline=check_deadline,
                        )
                        check_deadline()
                        staged.commit()
                    except Exception:
                        staged.rollback()
                        raise

                claim = _claim_fresh_namespace_db_with_configuration_lock(
                    target, prepare=prepare
                )
                claim.verify_for_publication()
                registry.execute(
                    "INSERT INTO namespace_identities(namespace_id,canonical_label,"
                    "canonical_label_norm,db_path,db_device,db_inode,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        namespace_id,
                        label,
                        normalize_namespace_label(label),
                        str(target),
                        claim.identity[0],
                        claim.identity[1],
                        now,
                        now,
                    ),
                )
                for alias in namespace["aliases"]:
                    check_deadline()
                    registry.execute(
                        "INSERT INTO namespace_aliases(normalized_label,label,namespace_id,"
                        "is_canonical,source_alias_norm,created_at) VALUES (?,?,?,?,?,?)",
                        (
                            normalize_namespace_label(str(alias["label"])),
                            alias["label"],
                            namespace_id,
                            int(alias["is_canonical"]),
                            alias["source_alias_norm"],
                            now,
                        ),
                    )
                registry.execute(
                    "INSERT INTO namespaces(name,repo_path,db_path,created_at,updated_at) "
                    "VALUES (?,?,?,?,?)",
                    (label, None, str(target), now, now),
                )
                for repository in namespace["repository_identities"]:
                    check_deadline()
                    registry.execute(
                        "INSERT INTO repository_bindings(binding_id,namespace_id,"
                        "repository_identity,repo_path,label_norm,created_at,updated_at) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (
                            hashlib.sha256(
                                f"{namespace_id}\0{repository}".encode("utf-8")
                            ).hexdigest(),
                            namespace_id,
                            repository,
                            None,
                            normalize_namespace_label(label),
                            now,
                            now,
                        ),
                    )
                check_deadline()
                registry.commit()
            except Exception:
                registry.rollback()
                if claim is not None:
                    claim.close(remove_target=True)
                raise
            finally:
                registry.close()
            assert claim is not None
            claim.close(remove_target=False)
    return {
        "namespace": label,
        "namespace_id": namespace_id,
        "semantic_digest": validated.semantic_digest,
        "deduplicated": False,
        "created_namespace": True,
        "inserted": inserted,
        "limits": asdict(validated.limits),
    }


def _apply_existing_import(
    validated: _ValidatedBundle, existing: Mapping[str, Any], check_deadline: Callable[[], None]
) -> dict[str, Any]:
    namespace_id = str(validated.namespace["namespace_id"])
    with _namespace_migration_lock():
        current = _registry_preflight(validated.namespace, check_deadline)
        if current is None:
            raise ImportConflictError("existing namespace disappeared during import")
        _require_same_selected_storage(current, existing)
        # open_namespace_identity re-enters the already-held migration lock and
        # selects only the stable ID/path/device/inode; no presentation label
        # is re-resolved between preflight and the writer transaction.
        store = open_namespace_identity(
            namespace_id,
            expected_db_path=str(existing["db_path"]),
            expected_db_device=int(existing["db_device"]),
            expected_db_inode=int(existing["db_inode"]),
        )
        try:
            store.conn.execute("BEGIN IMMEDIATE")
            receipt_key = _RECEIPT_PREFIX + validated.semantic_digest
            already = store.conn.execute(
                "SELECT value FROM meta WHERE key=?", (receipt_key,)
            ).fetchone()
            try:
                inserted = _apply_records(
                    store.conn,
                    validated.decoded,
                    semantic_digest=validated.semantic_digest,
                    check_deadline=check_deadline,
                )
                check_deadline()
                current_after = _registry_preflight(
                    validated.namespace, check_deadline
                )
                if current_after is None:
                    raise ImportConflictError(
                        "existing namespace disappeared during import"
                    )
                _require_same_selected_storage(current_after, existing)
                store.conn.commit()
            except Exception:
                store.conn.rollback()
                raise
        finally:
            store.close()
    return {
        "namespace": str(validated.namespace["canonical_label"]),
        "namespace_id": namespace_id,
        "semantic_digest": validated.semantic_digest,
        "deduplicated": already is not None,
        "created_namespace": False,
        "inserted": inserted,
        "limits": asdict(validated.limits),
    }


def _resolved_limits(
    limits: ImportLimits | None, *, timeout_seconds: float | None
) -> ImportLimits:
    supplied = limits or ImportLimits()
    if not isinstance(supplied, ImportLimits):
        raise TypeError("limits must be ImportLimits")
    values = asdict(supplied)
    if timeout_seconds is not None:
        values["timeout_seconds"] = timeout_seconds
    return resolve_import_limits(**values)


def _import_namespace_bytes_with_deadline(
    raw: bytes,
    *,
    resolved: ImportLimits,
    check_deadline: Callable[[], None],
) -> dict[str, Any]:
    bundle = _load_json(raw, resolved, check_deadline)
    validated = _validate_bundle(bundle, resolved, check_deadline)
    check_deadline()
    existing = _registry_preflight(validated.namespace, check_deadline)
    check_deadline()
    try:
        if existing is None:
            return _publish_new_import(validated, check_deadline)
        _existing_import_after_preflight_hook(existing)
        check_deadline()
        return _apply_existing_import(validated, existing, check_deadline)
    except (ImportBundleError, sqlite3.Error):
        raise
    except Exception as exc:
        raise ImportBundleError(f"import failed without committing: {exc}") from exc


def import_namespace_bytes(
    raw: bytes,
    *,
    limits: ImportLimits | None = None,
    timeout_seconds: float | None = None,
    _clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Validate, then transactionally import a strict UTF-8 v1 bundle."""
    if not isinstance(raw, bytes):
        raise TypeError("raw bundle must be bytes")
    resolved = _resolved_limits(limits, timeout_seconds=timeout_seconds)
    deadline = _clock() + resolved.timeout_seconds

    def check_deadline() -> None:
        if _clock() > deadline:
            raise ImportLimitError("import timeout exceeded")

    return _import_namespace_bytes_with_deadline(
        raw, resolved=resolved, check_deadline=check_deadline
    )


def import_namespace_path(
    path: Path,
    *,
    limits: ImportLimits | None = None,
    timeout_seconds: float | None = None,
    _clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Read one regular file within the input budget and import it."""
    resolved = _resolved_limits(limits, timeout_seconds=timeout_seconds)
    deadline = _clock() + resolved.timeout_seconds

    def check_deadline() -> None:
        if _clock() > deadline:
            raise ImportLimitError("import timeout exceeded")

    check_deadline()
    try:
        info = path.stat()
    except OSError as exc:
        raise ImportBundleError(f"cannot inspect bundle: {exc}") from exc
    if not path.is_file():
        raise ImportBundleError("bundle path must be a regular file")
    if info.st_size > resolved.input_bytes:
        raise ImportLimitError(f"declared input bytes exceed {resolved.input_bytes}")
    try:
        chunks: list[bytes] = []
        total = 0
        with path.open("rb") as handle:
            while True:
                check_deadline()
                chunk = handle.read(min(64 * 1024, resolved.input_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > resolved.input_bytes:
                    raise ImportLimitError(
                        f"actual input bytes exceed {resolved.input_bytes}"
                    )
        check_deadline()
        raw = b"".join(chunks)
        check_deadline()
    except OSError as exc:
        raise ImportBundleError(f"cannot read bundle: {exc}") from exc
    return _import_namespace_bytes_with_deadline(
        raw, resolved=resolved, check_deadline=check_deadline
    )


def export_namespace_path(
    namespace: str,
    path: Path,
    *,
    cut: str | None = None,
) -> dict[str, Any]:
    """Write a mode-0600 canonical bundle without following a destination link."""
    bundle = build_namespace_export(namespace, cut=cut)
    raw = canonical_export_bytes(bundle)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ExportError(f"cannot create export file: {exc}") from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return {
        "namespace": bundle["namespace"]["canonical_label"],
        "namespace_id": bundle["namespace"]["namespace_id"],
        "path": str(path),
        "bytes": len(raw),
        "semantic_digest": bundle["manifest"]["semantic_digest"],
        "temporal_cut": bundle["temporal_cut"],
        "warning": "Export contains potentially sensitive verbatim namespace data.",
    }
