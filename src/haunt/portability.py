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
import secrets
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from haunt.paths import (
    NamespacePathError,
    SQLITE_SIDECAR_SUFFIXES,
    haunt_home,
    mkdir_private,
    normalize_namespace_label,
    repository_identity,
    required_o_nofollow,
    safe_name,
)
from haunt.provenance import validate_provenance
from haunt.graph import ENTITY_TYPES, _refresh_relation
from haunt.store import (
    RECALL_CLASSES,
    TIERS,
    SCHEMA_VERSION,
    PRIVACY_LINEAGE_KEY,
    _claim_fresh_namespace_db_with_configuration_lock,
    _content_hash,
    _ensure_namespace_schema,
    _init_namespace_schema,
    _namespace_migration_lock,
    _readonly_registry,
    _registry,
    _sqlite_configuration_lock,
    _validate_unmapped_namespace_target,
    is_concurrent_registry_change,
    namespace_privacy_lineage_head,
    namespaces_dir,
    open_namespace_identity_readonly,
    _open_namespace_identity_unmaintained,
    resolve_namespace_id,
    resolve_namespace_identity,
)
from haunt.util import now_iso, parse_iso, utc_iso

FORMAT_NAME = "haunt.namespace-export"
FORMAT_MAJOR = 1
FORMAT_MINOR = 1
# Media type is major-scoped: a compatible minor adds fields an older reader
# would refuse, not a different container, so the negotiated type is unchanged.
MEDIA_TYPE = "application/vnd.haunt.namespace-export+json;version=1"

_DIGEST_PREFIX = "sha256:"
_RECEIPT_PREFIX = "import_receipt:"
_INTENT_VERSION = 1
_INTENT_DIRECTORY = "import-intents"


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
    "sessions": ("id", "started_at", "ended_at", "source", "meta", "succeeds_session"),
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
        "skip_embedding",
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

# Record fields a compatible minor added, keyed by the minor that introduced
# them. Additive only: a field may join this table, never leave it, and no
# entry here may change how a field an older minor already emitted is read.
_MINOR_ADDED_FIELDS: dict[int, dict[str, tuple[str, ...]]] = {
    1: {"sessions": ("succeeds_session",), "memories": ("skip_embedding",)},
}

# The value an older bundle's silence means. Both are the destination column's
# own default, so an accepted older bundle lands exactly as it always did and
# the added fields change what a bundle can say, not what one already said.
_MINOR_FIELD_DEFAULTS: dict[str, dict[str, Any]] = {
    "sessions": {"succeeds_session": None},
    "memories": {"skip_embedding": 0},
}


def _fields_by_minor() -> dict[int, dict[str, tuple[str, ...]]]:
    """Exact accepted field set per supported minor, newest backwards."""
    table: dict[int, dict[str, tuple[str, ...]]] = {}
    fields = dict(_TABLE_FIELDS)
    for minor in range(FORMAT_MINOR, -1, -1):
        table[minor] = dict(fields)
        removed = _MINOR_ADDED_FIELDS.get(minor, {})
        fields = {
            name: tuple(f for f in columns if f not in removed.get(name, ()))
            for name, columns in fields.items()
        }
    return table


_TABLE_FIELDS_BY_MINOR = _fields_by_minor()


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


def _decoded_at_current_minor(
    table: str,
    record: Mapping[str, Any],
    bundle_fields: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Decode one record and fill the fields its minor could not carry.

    The upgrade lands only in the decoded row used for writing. The bundle's
    own bytes are left alone, so an older bundle still hashes to the digest
    its exporter published and its import receipt stays replayable.
    """
    present = set(bundle_fields[table])
    return {
        field: (
            _decode_sqlite(record[field])
            if field in present
            else _MINOR_FIELD_DEFAULTS[table][field]
        )
        for field in _TABLE_FIELDS[table]
    }


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
                privacy_lineage_head = namespace_privacy_lineage_head(
                    store.conn, namespace_id
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
        return (
            resolved_cut,
            {**registry_before, "privacy_lineage_head": privacy_lineage_head},
            records,
        )


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
    # Older minors are read with their own field set and upgraded below; an
    # unknown newer one fails closed, because its added fields carry meaning
    # this reader cannot supply a default for.
    if not 0 <= version["minor"] <= FORMAT_MINOR:
        raise ImportBundleError(
            f"unsupported export minor {version['minor']}; "
            f"supported minors are 0 through {FORMAT_MINOR}"
        )
    bundle_fields = _TABLE_FIELDS_BY_MINOR[version["minor"]]
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
        {
            "namespace_id",
            "canonical_label",
            "aliases",
            "repository_identities",
            "privacy_lineage_head",
        },
        "namespace",
    )
    namespace_id = namespace["namespace_id"]
    canonical = namespace["canonical_label"]
    if not isinstance(namespace_id, str) or not namespace_id or len(namespace_id.encode()) > 256:
        raise ImportBundleError("namespace_id must be nonempty and at most 256 UTF-8 bytes")
    if not isinstance(canonical, str) or canonical != safe_name(canonical):
        raise ImportBundleError("canonical_label is not a safe exact namespace label")
    privacy_head = namespace["privacy_lineage_head"]
    if (
        not isinstance(privacy_head, str)
        or len(privacy_head) != 71
        or not privacy_head.startswith(_DIGEST_PREFIX)
        or any(ch not in "0123456789abcdef" for ch in privacy_head[7:])
    ):
        raise ImportBundleError("privacy_lineage_head must be a sha256 digest")
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
            _require_keys(record, set(bundle_fields[table]), f"records.{table} item")
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
            decoded_rows.append(_decoded_at_current_minor(table, record, bundle_fields))
        decoded[table] = decoded_rows

    manifest = bundle["manifest"]
    if not isinstance(manifest, dict):
        raise ImportBundleError("manifest must be an object")
    _require_keys(manifest, {"semantic_digest", "record_counts", "total_records"}, "manifest")
    counts = {table: len(records[table]) for table in _IMPORT_ORDER}
    supplied_counts = manifest["record_counts"]
    if not isinstance(supplied_counts, dict):
        raise ImportBundleError("manifest record_counts must be an object")
    _require_keys(supplied_counts, set(_IMPORT_ORDER), "manifest record_counts")
    for table in _IMPORT_ORDER:
        value = supplied_counts[table]
        if type(value) is not int or value < 0 or value > limits.records:
            raise ImportBundleError(
                f"manifest record_counts.{table} must be a bounded nonnegative integer"
            )
    supplied_total = manifest["total_records"]
    if (
        type(supplied_total) is not int
        or supplied_total < 0
        or supplied_total > limits.records
        or supplied_total != sum(supplied_counts.values())
    ):
        raise ImportBundleError(
            "manifest total_records must be the exact bounded count sum"
        )
    if supplied_counts != counts or supplied_total != total:
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
        _validate_enumerated_columns(records, check_deadline)
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

    # sessions.succeeds_session carries no SQLite foreign key, so the scratch
    # insert cannot catch a dangling successor link; this is the only gate.
    for session in records["sessions"]:
        check_deadline()
        predecessor = session["succeeds_session"]
        if predecessor is None:
            continue
        require(predecessor, sessions, "session references missing predecessor")
        if _identity_token(predecessor) == _identity_token(session["id"]):
            raise ImportBundleError("session cannot succeed itself")
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


def _validate_enumerated_columns(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    check_deadline: Callable[[], None],
) -> None:
    """Reject enum values no destination CHECK constraint would catch.

    ``events.tier``, ``memories.tier`` and ``entities.type`` are plain TEXT,
    so an import is the only gate between a crafted bundle and every reader
    that treats those columns as a known vocabulary.
    ``memories.skip_embedding`` is a plain INTEGER with no CHECK, and every
    reader of it (``reembed``, the queue drain) tests it as a two-valued
    flag, so anything but 0 or 1 would silently read as excluded.
    """
    for table in ("events", "memories"):
        for record in records[table]:
            check_deadline()
            if record["tier"] not in TIERS:
                raise ImportBundleError(f"invalid {table}.tier")
    for memory in records["memories"]:
        check_deadline()
        skip = memory["skip_embedding"]
        if type(skip) is not int or skip not in (0, 1):
            raise ImportBundleError("invalid memories.skip_embedding")
    for entity in records["entities"]:
        check_deadline()
        if entity["type"] not in ENTITY_TYPES:
            raise ImportBundleError("invalid entities.type")


def _same_sqlite_value(left: Any, right: Any) -> bool:
    try:
        return _canonical_bytes(_encode_sqlite(left)) == _canonical_bytes(_encode_sqlite(right))
    except ExportError:
        return False


@contextmanager
def _sqlite_deadline(
    conn: sqlite3.Connection, check_deadline: Callable[[], None]
) -> Iterator[None]:
    """Convert SQLite progress cancellation back to the resolved timeout error."""
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
        yield
    except sqlite3.OperationalError:
        if deadline_error:
            raise deadline_error[0]
        raise
    finally:
        conn.set_progress_handler(None, 0)


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
    has_receipt, missing = _preflight_record_state(
        conn,
        records,
        semantic_digest=semantic_digest,
        check_deadline=check_deadline,
    )
    if has_receipt:
        return {table: 0 for table in _IMPORT_ORDER}

    receipt_key = None if semantic_digest is None else _RECEIPT_PREFIX + semantic_digest
    receipt_value = None if semantic_digest is None else json.dumps(
        {"format": FORMAT_NAME, "semantic_digest": semantic_digest},
        sort_keys=True,
        separators=(",", ":"),
    )

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
            # content_hash is a pure function of stored content, so it is
            # recomputed rather than carried: the bundle already determines it,
            # and a second copy on the wire would only be a value validation
            # would have to reconcile against this one. Non-TEXT content has no
            # defined hash -- _content_hash takes str, and the store's own
            # backfill cannot hash a legacy BLOB either -- so it stays NULL.
            conn.execute(
                "UPDATE memories SET content_hash=? WHERE id=?",
                (_content_hash(content), record["id"]),
            )
            conn.execute(
                "INSERT INTO memories_fts(id,content) VALUES (?,?)",
                (record["id"], content),
            )
        # FTS is unconditional but the queue is not, exactly as observe()
        # writes them: the persisted capture-policy exclusion keeps the row
        # keyword-searchable and out of the vector index.
        if isinstance(content, str) and content.strip() and not record["skip_embedding"]:
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


def _preflight_record_state(
    conn: sqlite3.Connection,
    records: dict[str, list[dict[str, Any]]],
    *,
    semantic_digest: str | None,
    check_deadline: Callable[[], None],
) -> tuple[bool, dict[str, list[dict[str, Any]]]]:
    """Check receipts and every durable row without changing the connection."""
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
    return has_receipt, missing


def _require_privacy_lineage_match(
    conn: sqlite3.Connection,
    namespace: Mapping[str, Any],
) -> None:
    namespace_id = str(namespace["namespace_id"])
    try:
        current = namespace_privacy_lineage_head(conn, namespace_id)
    except NamespacePathError as exc:
        raise ImportConflictError("existing privacy lineage head is malformed") from exc
    if current != str(namespace["privacy_lineage_head"]):
        raise ImportConflictError(
            "bundle privacy lineage does not match the destination"
        )


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


def _import_publication_phase_hook(
    _phase: str, _intent: Mapping[str, Any]
) -> None:
    """Deterministic crash-test hook; production behavior is intentionally empty."""


def _intent_directory(*, create: bool) -> Path:
    root = haunt_home()
    if create:
        mkdir_private(root)
    directory = root / _INTENT_DIRECTORY
    if create:
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
    try:
        info = directory.lstat()
    except FileNotFoundError:
        if not create:
            return directory
        raise ImportBundleError("cannot create import-intent directory")
    except OSError as exc:
        raise ImportBundleError(f"cannot inspect import-intent directory: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ImportBundleError("import-intent path must be a real directory")
    if create:
        os.chmod(directory, 0o700, follow_symlinks=False)
    return directory


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | required_o_nofollow()
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_exact_owned_file(path: Path, *, max_bytes: int = 64 * 1024) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ImportBundleError(f"cannot inspect import intent {path.name}: {exc}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or int(before.st_size) > max_bytes
    ):
        raise ImportBundleError("import intent is not an owned bounded regular file")
    fd = os.open(
        path,
        os.O_RDONLY | required_o_nofollow() | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or (int(opened.st_dev), int(opened.st_ino))
            != (int(before.st_dev), int(before.st_ino))
        ):
            raise ImportBundleError("import intent changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(8192, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ImportBundleError("import intent exceeds its internal budget")
        after = os.fstat(fd)
        current = path.lstat()
        identity = (int(opened.st_dev), int(opened.st_ino))
        if (
            (int(after.st_dev), int(after.st_ino)) != identity
            or (int(current.st_dev), int(current.st_ino)) != identity
            or int(after.st_nlink) != 1
            or int(current.st_nlink) != 1
        ):
            raise ImportBundleError("import intent changed while reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _decode_import_intent(path: Path) -> dict[str, Any]:
    raw = _read_exact_owned_file(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON number: {value}")

    try:
        parsed = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ImportBundleError("import intent is malformed") from exc
    if not isinstance(parsed, dict) or raw != _canonical_bytes(parsed):
        raise ImportBundleError("import intent is not canonical")
    _require_keys(
        parsed,
        {
            "version",
            "token",
            "semantic_digest",
            "namespace_id",
            "canonical_label",
            "privacy_lineage_head",
            "target_name",
            "temporary_name",
            "primary",
            "sidecars",
        },
        "import intent",
    )
    token = parsed["token"]
    digest = parsed["semantic_digest"]
    namespace_id = parsed["namespace_id"]
    label = parsed["canonical_label"]
    head = parsed["privacy_lineage_head"]
    if type(parsed["version"]) is not int or parsed["version"] != _INTENT_VERSION:
        raise ImportBundleError("import intent version is unsupported")
    if (
        not isinstance(token, str)
        or len(token) != 64
        or any(ch not in "0123456789abcdef" for ch in token)
        or path.name != f"{token}.json"
    ):
        raise ImportBundleError("import intent token is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 71
        or not digest.startswith(_DIGEST_PREFIX)
        or any(ch not in "0123456789abcdef" for ch in digest[7:])
    ):
        raise ImportBundleError("import intent digest is invalid")
    if not isinstance(namespace_id, str) or not namespace_id:
        raise ImportBundleError("import intent namespace ID is invalid")
    if not isinstance(label, str) or label != safe_name(label):
        raise ImportBundleError("import intent namespace label is invalid")
    if (
        not isinstance(head, str)
        or len(head) != 71
        or not head.startswith(_DIGEST_PREFIX)
        or any(ch not in "0123456789abcdef" for ch in head[7:])
    ):
        raise ImportBundleError("import intent privacy lineage is invalid")
    target_name = parsed["target_name"]
    temporary_name = parsed["temporary_name"]
    if target_name != f"{label}.db":
        raise ImportBundleError("import intent target does not match its label")
    if (
        not isinstance(temporary_name, str)
        or Path(temporary_name).name != temporary_name
        or not temporary_name.startswith(".haunt-claim-")
        or not temporary_name.endswith(".db")
    ):
        raise ImportBundleError("import intent temporary name is invalid")

    def identity(value: Any, field: str) -> tuple[int, int]:
        if not isinstance(value, dict):
            raise ImportBundleError(f"import intent {field} identity is invalid")
        _require_keys(value, {"device", "inode"}, f"import intent {field}")
        device, inode = value["device"], value["inode"]
        if type(device) is not int or type(inode) is not int or device < 0 or inode <= 0:
            raise ImportBundleError(f"import intent {field} identity is invalid")
        return device, inode

    identity(parsed["primary"], "primary")
    sidecars = parsed["sidecars"]
    if not isinstance(sidecars, list) or len(sidecars) != len(SQLITE_SIDECAR_SUFFIXES):
        raise ImportBundleError("import intent sidecar identities are invalid")
    seen: set[str] = set()
    for sidecar in sidecars:
        if not isinstance(sidecar, dict):
            raise ImportBundleError("import intent sidecar identity is invalid")
        _require_keys(sidecar, {"name", "device", "inode"}, "import intent sidecar")
        name = sidecar["name"]
        if not isinstance(name, str) or name in seen:
            raise ImportBundleError("import intent sidecar name is invalid")
        seen.add(name)
        identity(
            {"device": sidecar["device"], "inode": sidecar["inode"]},
            "sidecar",
        )
    expected_names = {temporary_name + suffix for suffix in SQLITE_SIDECAR_SUFFIXES}
    if seen != expected_names:
        raise ImportBundleError("import intent sidecar names do not match the claim")
    return parsed


def _create_import_intent(intent: Mapping[str, Any]) -> Path:
    directory = _intent_directory(create=True)
    token = str(intent["token"])
    final = directory / f"{token}.json"
    pending = directory / f".{token}.{secrets.token_hex(16)}.pending"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | required_o_nofollow()
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(pending, flags, 0o600)
    try:
        raw = _canonical_bytes(intent)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(pending, final, follow_symlinks=False)
        pending.unlink()
        _fsync_directory(directory)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    return final


def _file_identity(
    path: Path,
    expected: tuple[int, int],
    *,
    allowed_links: set[int],
) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ImportBundleError(f"cannot inspect recovery-owned file: {path.name}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) not in allowed_links
        or (int(info.st_dev), int(info.st_ino)) != expected
    ):
        raise ImportBundleError(
            f"refusing to remove changed or unrelated recovery file: {path.name}"
        )
    return info


def _unlink_exact(path: Path, expected: tuple[int, int], *, allowed_links: set[int]) -> None:
    info = _file_identity(path, expected, allowed_links=allowed_links)
    if info is None:
        return
    fd = os.open(
        path,
        os.O_RDONLY | required_o_nofollow() | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(fd)
        current = path.lstat()
        if (
            (int(opened.st_dev), int(opened.st_ino)) != expected
            or (int(current.st_dev), int(current.st_ino)) != expected
            or int(opened.st_nlink) not in allowed_links
            or int(current.st_nlink) not in allowed_links
        ):
            raise ImportBundleError(
                f"refusing to remove changed recovery file: {path.name}"
            )
        path.unlink()
    finally:
        os.close(fd)


def _verify_staged_receipt(
    path: Path,
    intent: Mapping[str, Any],
    check_deadline: Callable[[], None],
) -> None:
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        with _sqlite_deadline(conn, check_deadline):
            receipt = conn.execute(
                "SELECT value FROM meta WHERE key=?",
                (_RECEIPT_PREFIX + str(intent["semantic_digest"]),),
            ).fetchone()
            schema = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            privacy = conn.execute(
                "SELECT value FROM meta WHERE key=?", (PRIVACY_LINEAGE_KEY,)
            ).fetchone()
    except sqlite3.Error as exc:
        raise ImportBundleError("cannot verify recovery-owned staged database") from exc
    finally:
        if "conn" in locals():
            conn.close()
    expected_receipt = json.dumps(
        {"format": FORMAT_NAME, "semantic_digest": intent["semantic_digest"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    if (
        receipt is None
        or receipt["value"] != expected_receipt
        or schema is None
        or str(schema["value"]) != str(SCHEMA_VERSION)
        or privacy is None
        or privacy["value"] != intent["privacy_lineage_head"]
    ):
        raise ImportBundleError("recovery-owned staged database does not match its intent")


def _remove_import_intent(path: Path, intent: Mapping[str, Any]) -> None:
    current = _decode_import_intent(path)
    if current != intent:
        raise ImportBundleError("import intent changed before cleanup")
    info = path.lstat()
    _unlink_exact(
        path,
        (int(info.st_dev), int(info.st_ino)),
        allowed_links={1},
    )
    _fsync_directory(path.parent)


def _recover_one_import_intent(path: Path, check_deadline: Callable[[], None]) -> None:
    check_deadline()
    intent = _decode_import_intent(path)
    root = namespaces_dir()
    temporary = root / str(intent["temporary_name"])
    target = root / str(intent["target_name"])
    primary = (
        int(intent["primary"]["device"]),
        int(intent["primary"]["inode"]),
    )
    temp_info = _file_identity(temporary, primary, allowed_links={1, 2})
    target_info = _file_identity(target, primary, allowed_links={1, 2})
    if temp_info is not None and target_info is not None:
        if int(temp_info.st_nlink) != 2 or int(target_info.st_nlink) != 2:
            raise ImportBundleError("recovery-owned primary has unsafe link topology")
    elif temp_info is not None and int(temp_info.st_nlink) != 1:
        raise ImportBundleError("recovery-owned temporary has an unexpected hardlink")
    elif target_info is not None and int(target_info.st_nlink) != 1:
        raise ImportBundleError("recovery-owned target has an unexpected hardlink")
    sidecar_paths: list[tuple[Path, tuple[int, int]]] = []
    for sidecar in intent["sidecars"]:
        check_deadline()
        sidecar_path = root / str(sidecar["name"])
        expected = int(sidecar["device"]), int(sidecar["inode"])
        _file_identity(sidecar_path, expected, allowed_links={1})
        sidecar_paths.append((sidecar_path, expected))

    mapped: sqlite3.Row | None = None
    registry_file = haunt_home() / "registry.db"
    if registry_file.is_file():
        registry = _readonly_registry()
        try:
            with _sqlite_deadline(registry, check_deadline):
                mapped = registry.execute(
                    "SELECT namespace_id,canonical_label,db_path,db_device,db_inode "
                    "FROM namespace_identities WHERE namespace_id=?",
                    (intent["namespace_id"],),
                ).fetchone()
                by_target = registry.execute(
                    "SELECT namespace_id FROM namespace_identities WHERE db_path=?",
                    (str(target),),
                ).fetchone()
        finally:
            registry.close()
        if by_target is not None and str(by_target["namespace_id"]) != str(
            intent["namespace_id"]
        ):
            raise ImportBundleError("recovery target belongs to another namespace")

    committed = mapped is not None
    if committed:
        if (
            str(mapped["canonical_label"]) != str(intent["canonical_label"])
            or str(mapped["db_path"]) != str(target)
            or int(mapped["db_device"]) != primary[0]
            or int(mapped["db_inode"]) != primary[1]
            or target_info is None
        ):
            raise ImportBundleError("committed import intent does not match registry identity")
        _verify_staged_receipt(target, intent, check_deadline)

    # Validate every owned name before the first unlink. Unknown replacements,
    # symlinks, and hardlinks fail closed and are never removed.
    check_deadline()
    if not committed and target_info is not None:
        _unlink_exact(target, primary, allowed_links={1, 2})
    if temp_info is not None:
        _unlink_exact(temporary, primary, allowed_links={1, 2})
    for sidecar_path, expected in sidecar_paths:
        _unlink_exact(sidecar_path, expected, allowed_links={1})
    if committed:
        _file_identity(target, primary, allowed_links={1})
    _fsync_directory(root)
    _remove_import_intent(path, intent)


def _recover_import_intents(check_deadline: Callable[[], None]) -> None:
    directory = _intent_directory(create=False)
    if not directory.exists():
        return
    with _namespace_migration_lock():
        with _sqlite_configuration_lock():
            for path in sorted(directory.iterdir(), key=lambda item: item.name):
                check_deadline()
                if path.name.endswith(".json"):
                    _recover_one_import_intent(path, check_deadline)


def _publish_new_import(
    validated: _ValidatedBundle, check_deadline: Callable[[], None]
) -> dict[str, Any]:
    namespace = validated.namespace
    label = str(namespace["canonical_label"])
    namespace_id = str(namespace["namespace_id"])
    target = namespaces_dir() / f"{label}.db"
    now = now_iso()
    intent_token = secrets.token_hex(32)
    intent_path: Path | None = None
    intent_data: dict[str, Any] | None = None

    def publication_hook(
        phase: str,
        temporary: Path,
        identity: tuple[int, int],
        sidecars: list[tuple[str, int, int]],
    ) -> None:
        nonlocal intent_path, intent_data
        check_deadline()
        if phase == "claimed":
            intent_data = {
                "version": _INTENT_VERSION,
                "token": intent_token,
                "semantic_digest": validated.semantic_digest,
                "namespace_id": namespace_id,
                "canonical_label": label,
                "privacy_lineage_head": namespace["privacy_lineage_head"],
                "target_name": target.name,
                "temporary_name": temporary.name,
                "primary": {"device": identity[0], "inode": identity[1]},
                "sidecars": [
                    {"name": name, "device": device, "inode": inode}
                    for name, device, inode in sorted(sidecars)
                ],
            }
            intent_path = _create_import_intent(intent_data)
            _import_publication_phase_hook("intent-created", intent_data)
        else:
            if intent_data is None:
                raise ImportBundleError("fresh import publication has no recovery intent")
            _import_publication_phase_hook(phase, intent_data)
        check_deadline()

    with _namespace_migration_lock():
        with _sqlite_configuration_lock():
            # Repeat every collision check while holding both writer locks.
            if _registry_preflight(namespace, check_deadline) is not None:
                raise ImportConflictError("namespace appeared during import")
            registry = _registry()
            claim = None
            registry_committed = False
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
                        staged.execute(
                            "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                            (
                                PRIVACY_LINEAGE_KEY,
                                namespace["privacy_lineage_head"],
                            ),
                        )
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
                    target,
                    prepare=prepare,
                    publication_hook=publication_hook,
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
                assert intent_data is not None
                _import_publication_phase_hook("registry-precommit", intent_data)
                registry.commit()
                registry_committed = True
                _import_publication_phase_hook("registry-committed", intent_data)
            except Exception:
                registry.rollback()
                if claim is not None:
                    claim.close(remove_target=not registry_committed)
                if intent_path is not None and intent_data is not None:
                    _remove_import_intent(intent_path, intent_data)
                raise
            finally:
                registry.close()
            assert claim is not None
            claim.close(remove_target=False)
            assert intent_path is not None and intent_data is not None
            _remove_import_intent(intent_path, intent_data)
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
        # Complete the conflict/receipt/privacy check through a physically
        # read-only exact-ID handle. A rejected bundle therefore cannot run
        # schema migration, graph repair, or writer configuration first.
        readonly = open_namespace_identity_readonly(
            namespace_id,
            expected_db_path=str(existing["db_path"]),
            expected_db_device=int(existing["db_device"]),
            expected_db_inode=int(existing["db_inode"]),
        )
        try:
            if readonly.schema_version != SCHEMA_VERSION:
                raise ImportConflictError(
                    f"existing namespace must already be schema {SCHEMA_VERSION}"
                )
            with _sqlite_deadline(readonly.conn, check_deadline):
                readonly.conn.execute("BEGIN")
                try:
                    _require_privacy_lineage_match(
                        readonly.conn, validated.namespace
                    )
                    _preflight_record_state(
                        readonly.conn,
                        validated.decoded,
                        semantic_digest=validated.semantic_digest,
                        check_deadline=check_deadline,
                    )
                finally:
                    readonly.conn.rollback()
        finally:
            readonly.close()

        # The exact-ID writer deliberately skips schema/projection maintenance.
        # Recheck all preflight facts inside its transaction to close the gap
        # between the read-only snapshot and the write lock.
        store = _open_namespace_identity_unmaintained(
            namespace_id,
            expected_db_path=str(existing["db_path"]),
            expected_db_device=int(existing["db_device"]),
            expected_db_inode=int(existing["db_inode"]),
        )
        try:
            with _sqlite_deadline(store.conn, check_deadline):
                store.conn.execute("BEGIN IMMEDIATE")
                receipt_key = _RECEIPT_PREFIX + validated.semantic_digest
                already = store.conn.execute(
                    "SELECT value FROM meta WHERE key=?", (receipt_key,)
                ).fetchone()
                try:
                    schema = store.conn.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()
                    if schema is None or str(schema["value"]) != str(SCHEMA_VERSION):
                        raise ImportConflictError(
                            f"existing namespace must remain schema {SCHEMA_VERSION}"
                        )
                    _require_privacy_lineage_match(
                        store.conn, validated.namespace
                    )
                    store.conn.execute(
                        "INSERT OR IGNORE INTO meta(key,value) VALUES (?,?)",
                        (
                            PRIVACY_LINEAGE_KEY,
                            validated.namespace["privacy_lineage_head"],
                        ),
                    )
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
    _recover_import_intents(check_deadline)
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
