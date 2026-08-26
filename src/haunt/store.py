"""SQLite store: registry + per-namespace DBs. WAL. Verbatim writes only."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from threading import get_ident
from typing import Any, Iterator

import sqlite_vec

from haunt.embed import embed_one
from haunt.embed import embed_texts
from haunt.embed import state as embed_state
from haunt.paths import (
    ensure_layout,
    haunt_home,
    mkdir_private,
    namespace_db_path,
    registry_path,
    resolve_namespace,
    safe_name,
    tighten_db_files,
)
from haunt.util import (
    clamp_limit,
    clock_sql_column,
    dumps,
    iso_or_now,
    loads,
    new_id,
    normalize_clock,
    now_iso,
    parse_iso,
    utc_iso,
)

ROLES = ("user", "assistant", "tool", "system")
TIERS = ("episodic", "semantic", "procedural", "coordinate")

# 1: one-time normalize of offset/naive clocks to UTC microseconds.
# 2: graph evidence tables + hook idempotency key.
# 3: durable queue for hook-deferred embeddings.
# 4: append-only correction lineage plus privacy-erasure tombstones.
# 5: privacy-safe rekeying for erased target and correction sessions.
# 6: schema-enforced normal-vs-privacy-scrubbed correction invariants.
# 7: database-enforced append-only corrections outside authorized purge.
SCHEMA_VERSION = 7
SCHEMA_VERSION_KEY = "schema_version"

_CLOCK_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sessions", ("started_at", "ended_at")),
    ("events", ("ts", "event_time")),
    ("memories", ("valid_from", "valid_to", "created_at")),
    ("entities", ("first_seen", "last_seen")),
    ("relations", ("valid_from", "valid_to")),
)

CORRECTION_KEY_MAX = 512
TOMBSTONE_SCHEMA_VERSION = 1
PURGE_SAFE_ORIGIN = "privacy-sanitized"
PURGE_SAFE_SESSION_SOURCE = "privacy-sanitized"


class UnknownNamespaceError(ValueError):
    """Raised when a read/mutation targets a namespace that does not exist."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"unknown namespace: {name}")


def _connect(path: Path, *, create: bool = True) -> sqlite3.Connection:
    if not create and not path.exists():
        raise FileNotFoundError(path)
    mkdir_private(path.parent)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    tighten_db_files(path)
    from haunt.embed import fts_only

    if not fts_only():
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as exc:
            conn.close()
            raise RuntimeError(
                f"sqlite-vec failed to load: {exc}\n"
                "Set HAUNT_FTS_ONLY=1 to run without vector search."
            ) from exc
    return conn


def _vec_loaded(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT vec_version()")
        return True
    except sqlite3.Error:
        return False


def init_registry() -> None:
    ensure_layout()
    conn = _connect(registry_path())
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS namespaces (
                name TEXT PRIMARY KEY,
                repo_path TEXT,
                db_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _registry() -> sqlite3.Connection:
    init_registry()
    return _connect(registry_path())


def _init_namespace_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            source TEXT,
            meta TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            idempotency_key TEXT,
            session_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            event_time TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_name TEXT,
            tool_input TEXT,
            tool_output TEXT,
            origin TEXT,
            tier TEXT NOT NULL,
            meta TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            tier TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (event_id) REFERENCES events(id)
        );
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            norm_name TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS relations (
            id TEXT PRIMARY KEY,
            src_entity TEXT NOT NULL,
            rel TEXT NOT NULL,
            dst_entity TEXT NOT NULL,
            event_id TEXT,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            weight REAL NOT NULL DEFAULT 1.0
        );
        CREATE TABLE IF NOT EXISTS entity_mentions (
            event_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            PRIMARY KEY (event_id, entity_id),
            FOREIGN KEY (event_id) REFERENCES events(id),
            FOREIGN KEY (entity_id) REFERENCES entities(id)
        );
        CREATE TABLE IF NOT EXISTS relation_evidence (
            event_id TEXT NOT NULL,
            src_entity TEXT NOT NULL,
            rel TEXT NOT NULL,
            dst_entity TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (event_id, src_entity, rel, dst_entity),
            FOREIGN KEY (event_id) REFERENCES events(id),
            FOREIGN KEY (src_entity) REFERENCES entities(id),
            FOREIGN KEY (dst_entity) REFERENCES entities(id)
        );
        CREATE TABLE IF NOT EXISTS embedding_jobs (
            memory_id TEXT PRIMARY KEY,
            queued_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
        CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time);
        CREATE INDEX IF NOT EXISTS idx_events_tier ON events(tier);
        CREATE INDEX IF NOT EXISTS idx_memories_event ON memories(event_id);
        CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(valid_from, valid_to);
        CREATE INDEX IF NOT EXISTS idx_entities_norm ON entities(norm_name, type);
        CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src_entity);
        CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst_entity);
        CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
        CREATE INDEX IF NOT EXISTS idx_relation_evidence_src ON relation_evidence(src_entity);
        CREATE INDEX IF NOT EXISTS idx_relation_evidence_dst ON relation_evidence(dst_entity);
        CREATE INDEX IF NOT EXISTS idx_embedding_jobs_queued ON embedding_jobs(queued_at);
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            id UNINDEXED,
            content,
            tokenize='porter unicode61'
        );
        """
    )
    conn.commit()


def _normalize_clock_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return text
    try:
        return utc_iso(parse_iso(text))
    except (TypeError, ValueError):
        return text


def _normalize_stored_clocks(conn: sqlite3.Connection) -> int:
    """Rewrite offset/naive timestamps to canonical UTC. Returns rows touched."""
    changed = 0
    for table, cols in _CLOCK_COLUMNS:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not exists:
            continue
        rows = conn.execute(
            f"SELECT rowid, {', '.join(cols)} FROM {table}"
        ).fetchall()
        for row in rows:
            sets: list[str] = []
            params: list[Any] = []
            for col in cols:
                old = row[col]
                if old is None:
                    continue
                new = _normalize_clock_value(old)
                if new != old:
                    sets.append(f"{col}=?")
                    params.append(new)
            if sets:
                params.append(row["rowid"])
                conn.execute(
                    f"UPDATE {table} SET {', '.join(sets)} WHERE rowid=?",
                    params,
                )
                changed += 1
    return changed


def _schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM meta WHERE key=?", (SCHEMA_VERSION_KEY,)
    ).fetchone()
    if not row:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def _ensure_correction_invariant_triggers(conn: sqlite3.Connection) -> None:
    """Reject malformed normal rows while allowing purge-scrubbed lineage."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='corrections'"
    ).fetchone()
    if not exists:
        return
    valid = """
        (
            NEW.target_tombstone_id IS NULL
            AND NEW.replacement_tombstone_id IS NULL
            AND NEW.origin IS NOT NULL
            AND NEW.session_id IS NOT NULL
            AND NEW.idempotency_key IS NOT NULL
            AND length(trim(
                NEW.idempotency_key,
                char(9) || char(10) || char(11) || char(12) || char(13) || ' '
            )) > 0
            AND length(NEW.idempotency_key) <= 512
            AND NEW.request_identity IS NOT NULL
            AND length(NEW.request_identity) = 71
            AND substr(NEW.request_identity, 1, 7) = 'sha256:'
            AND substr(NEW.request_identity, 8) NOT GLOB '*[^0-9a-f]*'
            AND NEW.request_payload IS NOT NULL
            AND typeof(NEW.request_payload) = 'blob'
            AND NEW.response_json IS NOT NULL
            AND json_valid(NEW.response_json) = 1
        )
        OR
        (
            (NEW.target_tombstone_id IS NOT NULL
             OR NEW.replacement_tombstone_id IS NOT NULL)
            AND NEW.origin IS NULL
            AND NEW.session_id IS NULL
            AND NEW.reason IS NULL
            AND NEW.idempotency_key IS NULL
            AND NEW.request_identity IS NULL
            AND NEW.request_payload IS NULL
            AND NEW.response_json IS NULL
        )
    """
    conn.execute(
        f"""
        CREATE TRIGGER IF NOT EXISTS corrections_invariant_insert
        BEFORE INSERT ON corrections
        WHEN NOT ({valid})
        BEGIN
            SELECT RAISE(ABORT, 'invalid correction invariant');
        END
        """
    )
    conn.execute(
        f"""
        CREATE TRIGGER IF NOT EXISTS corrections_invariant_update
        BEFORE UPDATE ON corrections
        WHEN NOT ({valid})
        BEGIN
            SELECT RAISE(ABORT, 'invalid correction invariant');
        END
        """
    )


def _ensure_correction_append_only_triggers(conn: sqlite3.Connection) -> None:
    """Block correction mutation unless this Store is in its purge transaction.

    The authorization function is registered only on Store-owned connections.
    An external SQLite connection cannot satisfy the trigger and fails closed.
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='corrections'"
    ).fetchone()
    if not exists:
        return
    for operation in ("UPDATE", "DELETE"):
        conn.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS corrections_append_only_{operation.lower()}
            BEFORE {operation} ON corrections
            WHEN haunt_privacy_purge_authorized() != 1
            BEGIN
                SELECT RAISE(ABORT, 'corrections are append-only');
            END
            """
        )


def _ensure_namespace_schema(conn: sqlite3.Connection) -> None:
    """Create tables and run one-time migrations. Not invoked per query."""
    _init_namespace_schema(conn)
    current = _schema_version(conn)
    if current >= SCHEMA_VERSION:
        _ensure_correction_invariant_triggers(conn)
        _ensure_correction_append_only_triggers(conn)
        conn.commit()
        return
    if current < 1:
        _normalize_stored_clocks(conn)
    if current < 2:
        event_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        if "idempotency_key" not in event_columns:
            conn.execute("ALTER TABLE events ADD COLUMN idempotency_key TEXT")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idempotency "
            "ON events(idempotency_key) WHERE idempotency_key IS NOT NULL"
        )
    if current < 3:
        conn.execute(
            """
            INSERT OR IGNORE INTO embedding_jobs(memory_id, queued_at)
            SELECT id, created_at FROM memories
            WHERE embedding IS NULL AND TRIM(content) != ''
            """
        )
    if current < 4:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lineage_tombstones (
                schema_version INTEGER NOT NULL,
                tombstone_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK (status = 'erased'),
                erased_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                target_memory_id TEXT,
                target_tombstone_id TEXT,
                replacement_memory_id TEXT,
                replacement_tombstone_id TEXT,
                corrected_at TEXT NOT NULL,
                origin TEXT,
                session_id TEXT,
                reason TEXT,
                idempotency_key TEXT,
                request_identity TEXT,
                request_payload BLOB,
                response_json TEXT,
                CHECK ((target_memory_id IS NOT NULL) !=
                       (target_tombstone_id IS NOT NULL)),
                CHECK (replacement_memory_id IS NULL OR
                       replacement_tombstone_id IS NULL)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_idempotency
                ON corrections(idempotency_key)
                WHERE idempotency_key IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_target_memory
                ON corrections(target_memory_id)
                WHERE target_memory_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_target_tombstone
                ON corrections(target_tombstone_id)
                WHERE target_tombstone_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_replacement_memory
                ON corrections(replacement_memory_id)
                WHERE replacement_memory_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_replacement_tombstone
                ON corrections(replacement_tombstone_id)
                WHERE replacement_tombstone_id IS NOT NULL;
            """
        )
    _ensure_correction_invariant_triggers(conn)
    _ensure_correction_append_only_triggers(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (SCHEMA_VERSION_KEY, str(SCHEMA_VERSION)),
    )
    conn.commit()


def ensure_vec_table(conn: sqlite3.Connection, dim: int, *, commit: bool = True) -> bool:
    if dim <= 0 or not _vec_loaded(conn):
        return False
    existing = conn.execute(
        "SELECT value FROM meta WHERE key='embed_dim'"
    ).fetchone()
    if existing and int(existing["value"]) != dim:
        conn.execute("DROP TABLE IF EXISTS vec_memories")
    try:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                id TEXT PRIMARY KEY,
                embedding FLOAT[{int(dim)}] distance_metric=cosine
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_dim', ?)",
            (str(dim),),
        )
        if commit:
            conn.commit()
        return True
    except sqlite3.Error:
        return False


def register_namespace(name: str, repo_path: str | None = None) -> Path:
    name = safe_name(name)
    db = namespace_db_path(name)
    now = now_iso()
    repo = str(Path(repo_path).expanduser().resolve()) if repo_path else None
    conn = _registry()
    try:
        row = conn.execute("SELECT name FROM namespaces WHERE name=?", (name,)).fetchone()
        if row:
            conn.execute(
                "UPDATE namespaces SET repo_path=COALESCE(?, repo_path), db_path=?, updated_at=? WHERE name=?",
                (repo, str(db), now, name),
            )
        else:
            conn.execute(
                "INSERT INTO namespaces(name, repo_path, db_path, created_at, updated_at) VALUES (?,?,?,?,?)",
                (name, repo, str(db), now, now),
            )
        conn.commit()
    finally:
        conn.close()
    ns = _connect(db)
    try:
        _ensure_namespace_schema(ns)
        if repo:
            ns.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('repo_path', ?)",
                (repo,),
            )
            ns.commit()
    finally:
        ns.close()
    return db


def namespace_exists(name: str) -> bool:
    name = safe_name(name)
    conn = _registry()
    try:
        row = conn.execute("SELECT 1 FROM namespaces WHERE name=?", (name,)).fetchone()
        return bool(row)
    finally:
        conn.close()


def touch_namespace(name: str) -> None:
    conn = _registry()
    try:
        conn.execute(
            "UPDATE namespaces SET updated_at=? WHERE name=?",
            (now_iso(), safe_name(name)),
        )
        conn.commit()
    finally:
        conn.close()


def list_namespace_rows() -> list[dict[str, Any]]:
    init_registry()
    conn = _registry()
    try:
        rows = conn.execute(
            "SELECT name, repo_path, db_path, created_at, updated_at FROM namespaces ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def verbatim_text(
    content: str = "",
    tool_name: str | None = None,
    tool_input: str | None = None,
    tool_output: str | None = None,
) -> str:
    """Concatenate stored fields as-is. Not a summary."""
    parts: list[str] = []
    if content:
        parts.append(content)
    if tool_name:
        parts.append(f"tool:{tool_name}")
    if tool_input:
        parts.append(tool_input)
    if tool_output:
        parts.append(tool_output)
    return "\n".join(parts)


def _correction_request_payload(
    memory_id: str,
    replacement: str | None,
    reason: str | None,
) -> bytes:
    """Length-prefix exact UTF-8 request fields; null and empty stay distinct."""
    parts: list[bytes] = []
    for value in (memory_id, replacement, reason):
        if value is None:
            parts.append(b"\x00")
            continue
        raw = value.encode("utf-8")
        parts.append(b"\x01" + struct.pack(">Q", len(raw)) + raw)
    return b"".join(parts)


def _correction_request_identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _erasure_context_values(*raw_values: object) -> set[str]:
    """Collect textual erasure markers, including structured event metadata.

    Keep target event fields routed through this hook so future provenance
    fields can extend purge sanitization without scattering privacy logic.
    Target metadata keys may themselves contain user-controlled private bytes,
    so both mapping keys and values participate in session sanitization.
    """
    values: set[str] = set()
    not_json = object()

    def add(value: object) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                add(key)
                add(child)
            return
        if isinstance(value, (list, tuple, set)):
            for child in value:
                add(child)
            return
        if isinstance(value, bytes):
            values.add(value.hex())
            try:
                add(value.decode("utf-8"))
            except UnicodeDecodeError:
                pass
            return
        text = str(value)
        if not text or text in values:
            return
        values.add(text)
        if isinstance(value, str):
            parsed = loads(value, default=not_json)
            if parsed is not not_json and parsed != value:
                add(parsed)

    for raw_value in raw_values:
        add(raw_value)
    return values


@dataclass
class ObserveResult:
    event_id: str
    memory_id: str
    session_id: str
    namespace: str
    tier: str
    entities: list[str] = field(default_factory=list)
    embedded: bool = False
    embedding_queued: bool = False
    deduplicated: bool = False


class Store:
    def __init__(self, name: str, repo_path: str | None = None, *, create: bool = True):
        self.name = safe_name(name)
        if create:
            register_namespace(self.name, repo_path)
        self.db_path = namespace_db_path(self.name)
        self._privacy_purge_thread_id: int | None = None
        self.conn = _connect(self.db_path, create=create)
        self.conn.create_function(
            "haunt_privacy_purge_authorized",
            0,
            lambda: int(
                self._privacy_purge_thread_id == get_ident()
                and self.conn.in_transaction
            ),
        )
        _ensure_namespace_schema(self.conn)
        self._ensure_graph_evidence()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_graph_evidence(self) -> None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='graph_evidence_version'"
        ).fetchone()
        if row and str(row["value"]) == "1":
            return
        self.rebuild_graph(touch=False)

    def vec_ok(self) -> bool:
        return _vec_loaded(self.conn)

    def vec_version(self) -> str | None:
        if not self.vec_ok():
            return None
        return str(self.conn.execute("SELECT vec_version()").fetchone()[0])

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def ensure_session(
        self,
        session_id: str | None = None,
        source: str = "cli",
        meta: dict[str, Any] | None = None,
        *,
        commit: bool = True,
    ) -> str:
        if session_id:
            row = self.conn.execute(
                "SELECT id, ended_at FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if not row:
                self.conn.execute(
                    "INSERT INTO sessions(id, started_at, ended_at, source, meta) VALUES (?,?,?,?,?)",
                    (session_id, now_iso(), None, source, dumps(meta or {})),
                )
                if commit:
                    self.conn.commit()
            return session_id
        current = self.get_meta("current_session")
        if current:
            row = self.conn.execute(
                "SELECT id, ended_at FROM sessions WHERE id=?", (current,)
            ).fetchone()
            if row and not row["ended_at"]:
                return current
        sid = new_id()
        self.conn.execute(
            "INSERT INTO sessions(id, started_at, ended_at, source, meta) VALUES (?,?,?,?,?)",
            (sid, now_iso(), None, source, dumps(meta or {})),
        )
        if commit:
            self.set_meta("current_session", sid)
        else:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("current_session", sid),
            )
        return sid

    def end_session(self, session_id: str | None = None) -> dict[str, Any]:
        """Close an open session. Returns ok=False if nothing was ended."""
        sid = session_id or self.get_meta("current_session")
        if not sid:
            return {"ok": False, "error": "no open session"}
        cur = self.conn.execute(
            "UPDATE sessions SET ended_at=? WHERE id=? AND ended_at IS NULL",
            (now_iso(), sid),
        )
        if cur.rowcount == 0:
            row = self.conn.execute(
                "SELECT ended_at FROM sessions WHERE id=?", (sid,)
            ).fetchone()
            if not row:
                return {
                    "ok": False,
                    "session_id": sid,
                    "error": f"session {sid} not found",
                }
            return {
                "ok": False,
                "session_id": sid,
                "error": f"session {sid} already ended",
            }
        if self.get_meta("current_session") == sid:
            self.conn.execute("DELETE FROM meta WHERE key='current_session'")
        self.conn.commit()
        return {"ok": True, "session_id": sid}

    def observe(
        self,
        content: str = "",
        *,
        role: str = "user",
        tier: str = "episodic",
        session_id: str | None = None,
        tool_name: str | None = None,
        tool_input: str | None = None,
        tool_output: str | None = None,
        event_time: str | None = None,
        origin: str = "cli",
        meta: dict[str, Any] | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        idempotency_key: str | None = None,
        defer_embedding: bool = False,
        commit: bool = True,
    ) -> ObserveResult:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}")
        text = verbatim_text(content, tool_name, tool_input, tool_output)
        idem = (idempotency_key or "").strip() or None
        if idem and len(idem) > 512:
            raise ValueError("idempotency_key must be 512 characters or fewer")
        if idem:
            existing = self._observe_by_idempotency_key(idem, text)
            if existing is not None:
                return existing
        if commit and not defer_embedding:
            self.ensure_current_embeddings()
            self.process_embedding_jobs(limit=32)
        try:
            sid = self.ensure_session(session_id, source=origin, commit=False)
            et = iso_or_now(event_time)
            ts = now_iso()
            vf = iso_or_now(valid_from) if valid_from else et
            vt = iso_or_now(valid_to) if valid_to else None
            event_id = new_id()
            memory_id = new_id()
            self.conn.execute(
                """
                INSERT INTO events(
                    id, idempotency_key, session_id, ts, event_time, role, content,
                    tool_name, tool_input, tool_output, origin, tier, meta
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id,
                    idem,
                    sid,
                    ts,
                    et,
                    role,
                    content or "",
                    tool_name,
                    tool_input,
                    tool_output,
                    origin,
                    tier,
                    dumps(meta or {}),
                ),
            )
            blob = None
            embedded = False
            vec = (
                None
                if defer_embedding
                else (embed_one(text) if text.strip() else None)
            )
            if vec is not None:
                blob = sqlite_vec.serialize_float32(vec)
                ensure_vec_table(self.conn, len(vec), commit=False)
                es = embed_state()
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("embed_model", es.model_id),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("embed_dim", str(len(vec))),
                )
            self.conn.execute(
                """
                INSERT INTO memories(
                    id, event_id, tier, content, embedding, valid_from, valid_to, created_at
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (memory_id, event_id, tier, text, blob, vf, vt, ts),
            )
            self.conn.execute(
                "INSERT INTO memories_fts(id, content) VALUES (?, ?)",
                (memory_id, text),
            )
            embedding_queued = bool(blob is None and text.strip())
            if embedding_queued:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO embedding_jobs(memory_id, queued_at)
                    VALUES (?, ?)
                    """,
                    (memory_id, ts),
                )
            if blob is not None and _vec_loaded(self.conn):
                try:
                    self.conn.execute(
                        "INSERT INTO vec_memories(id, embedding) VALUES (?, ?)",
                        (memory_id, blob),
                    )
                    embedded = True
                except sqlite3.Error:
                    pass
            from haunt.graph import extract_and_store

            entity_names = extract_and_store(
                self.conn, event_id, text, et, tool_name, commit=False
            )
            if commit:
                self.conn.commit()
        except sqlite3.IntegrityError:
            self.conn.rollback()
            if idem:
                existing = self._observe_by_idempotency_key(idem, text)
                if existing is not None:
                    return existing
            raise
        except Exception:
            self.conn.rollback()
            raise
        if commit:
            try:
                touch_namespace(self.name)
            except Exception:
                pass
        return ObserveResult(
            event_id=event_id,
            memory_id=memory_id,
            session_id=sid,
            namespace=self.name,
            tier=tier,
            entities=entity_names,
            embedded=embedded,
            embedding_queued=embedding_queued,
        )

    def _observe_by_idempotency_key(
        self,
        key: str,
        expected_text: str,
    ) -> ObserveResult | None:
        row = self.conn.execute(
            """
            SELECT e.id AS event_id, e.session_id, e.tier,
                   m.id AS memory_id, m.content, m.embedding
            FROM events e
            JOIN memories m ON m.event_id=e.id
            WHERE e.idempotency_key=?
            ORDER BY m.rowid ASC
            LIMIT 1
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        if row["content"] != expected_text:
            raise ValueError("idempotency_key was reused with different content")
        entities = [
            str(r["name"])
            for r in self.conn.execute(
                """
                SELECT e.name
                FROM entity_mentions em
                JOIN entities e ON e.id=em.entity_id
                WHERE em.event_id=?
                ORDER BY e.name
                """,
                (row["event_id"],),
            ).fetchall()
        ]
        return ObserveResult(
            event_id=row["event_id"],
            memory_id=row["memory_id"],
            session_id=row["session_id"],
            namespace=self.name,
            tier=row["tier"],
            entities=entities,
            embedded=row["embedding"] is not None,
            embedding_queued=self.conn.execute(
                "SELECT 1 FROM embedding_jobs WHERE memory_id=?",
                (row["memory_id"],),
            ).fetchone()
            is not None,
            deduplicated=True,
        )


    def process_embedding_jobs(self, *, limit: int = 64) -> dict[str, Any]:
        """Embed queued hook writes in a persistent, model-owning process."""
        cap = clamp_limit(limit, default=64)
        queued = self.conn.execute(
            """
            SELECT j.memory_id, m.content
            FROM embedding_jobs j
            JOIN memories m ON m.id=j.memory_id
            ORDER BY j.queued_at ASC, j.rowid ASC
            LIMIT ?
            """,
            (cap,),
        ).fetchall()
        if not queued:
            return {"queued": 0, "processed": 0, "failed": 0}
        es = embed_state()
        if not es.available:
            return {
                "queued": len(queued),
                "processed": 0,
                "failed": 0,
                "available": False,
            }
        try:
            vectors = embed_texts(
                [row["content"] if row["content"] else " " for row in queued]
            )
        except Exception as exc:
            message = str(exc)[:1000]
            self.conn.executemany(
                """
                UPDATE embedding_jobs
                SET attempts=attempts+1, last_error=? WHERE memory_id=?
                """,
                [(message, row["memory_id"]) for row in queued],
            )
            self.conn.commit()
            return {
                "queued": len(queued),
                "processed": 0,
                "failed": len(queued),
                "error": message,
            }
        if not vectors:
            message = "embedding backend returned no vectors"
            self.conn.executemany(
                """
                UPDATE embedding_jobs
                SET attempts=attempts+1, last_error=? WHERE memory_id=?
                """,
                [(message, row["memory_id"]) for row in queued],
            )
            self.conn.commit()
            return {
                "queued": len(queued),
                "processed": 0,
                "failed": len(queued),
                "error": message,
            }

        ensure_vec_table(self.conn, es.dim, commit=False)
        processed = 0
        failed = 0
        for row, vec in zip(queued, vectors):
            memory_id = row["memory_id"]
            try:
                if len(vec) != es.dim:
                    raise ValueError(
                        f"embedding backend returned dimension {len(vec)}; "
                        f"expected {es.dim}"
                    )
                blob = sqlite_vec.serialize_float32(vec)
                self.conn.execute(
                    "UPDATE memories SET embedding=? WHERE id=?",
                    (blob, memory_id),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO vec_memories(id, embedding) VALUES (?, ?)",
                    (memory_id, blob),
                )
                self.conn.execute(
                    "DELETE FROM embedding_jobs WHERE memory_id=?",
                    (memory_id,),
                )
                processed += 1
            except (sqlite3.Error, TypeError, ValueError) as exc:
                self.conn.execute(
                    """
                    UPDATE embedding_jobs
                    SET attempts=attempts+1, last_error=? WHERE memory_id=?
                    """,
                    (str(exc)[:1000], memory_id),
                )
                failed += 1
        if len(vectors) < len(queued):
            missing = queued[len(vectors) :]
            message = "embedding backend returned fewer vectors than inputs"
            self.conn.executemany(
                """
                UPDATE embedding_jobs
                SET attempts=attempts+1, last_error=? WHERE memory_id=?
                """,
                [(message, row["memory_id"]) for row in missing],
            )
            failed += len(missing)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_model', ?)",
            (es.model_id,),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_dim', ?)",
            (str(es.dim),),
        )
        self.conn.commit()
        return {
            "queued": len(queued),
            "processed": processed,
            "failed": failed,
            "available": True,
        }


    def embeddings_stale(self) -> bool:
        """True when stored vectors do not match the currently loaded model."""
        es = embed_state()
        if not es.available:
            return False
        stored_dim = self.get_meta("embed_dim")
        stored_model = self.get_meta("embed_model")
        row = self.conn.execute(
            "SELECT embedding FROM memories WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        if row and row["embedding"]:
            n = len(row["embedding"]) // 4
            if n != es.dim:
                return True
        if stored_dim and int(stored_dim) != es.dim:
            return True
        if stored_model and stored_model != es.model_id:
            return True
        return False

    def reembed(self) -> dict[str, Any]:
        """Rebuild every memory embedding with the currently loaded model.

        ``updated`` is the number of rows that actually landed in
        ``vec_memories``, not blob writes to ``memories.embedding``.
        """
        es = embed_state()
        rows = self.conn.execute("SELECT id, content FROM memories").fetchall()
        self.conn.execute("DROP TABLE IF EXISTS vec_memories")
        if not es.available:
            self.conn.execute("UPDATE memories SET embedding=NULL")
            self.conn.execute(
                """
                INSERT OR IGNORE INTO embedding_jobs(memory_id, queued_at)
                SELECT id, created_at FROM memories WHERE TRIM(content) != ''
                """
            )
            self.conn.commit()
            return {
                "updated": 0,
                "total": len(rows),
                "model": es.model_id,
                "dim": es.dim,
                "available": False,
            }
        ensure_vec_table(self.conn, es.dim)
        ids = [r["id"] for r in rows]
        texts = [r["content"] if r["content"] else " " for r in rows]
        updated = 0
        chunk = 16
        for i in range(0, len(texts), chunk):
            vecs = embed_texts(texts[i : i + chunk])
            if not vecs:
                continue
            for mid, vec in zip(ids[i : i + chunk], vecs):
                blob = sqlite_vec.serialize_float32(vec)
                self.conn.execute(
                    "UPDATE memories SET embedding=? WHERE id=?", (blob, mid)
                )
                if self.vec_ok():
                    try:
                        self.conn.execute(
                            "INSERT INTO vec_memories(id, embedding) VALUES (?, ?)",
                            (mid, blob),
                        )
                        self.conn.execute(
                            "DELETE FROM embedding_jobs WHERE memory_id=?",
                            (mid,),
                        )
                        updated += 1
                    except sqlite3.Error:
                        pass
        self.set_meta("embed_model", es.model_id)
        self.set_meta("embed_dim", str(es.dim))
        self.conn.commit()
        return {
            "updated": updated,
            "total": len(rows),
            "model": es.model_id,
            "dim": es.dim,
            "available": True,
        }

    def ensure_current_embeddings(self) -> dict[str, Any] | None:
        """Rebuild vectors if the loaded model/dim does not match this DB."""
        if self.embeddings_stale():
            return self.reembed()
        return None

    def events(
        self,
        *,
        session_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        clock: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        col = clock_sql_column(clock, qualified=False)
        sql = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if session_id:
            sql += " AND session_id=?"
            params.append(session_id)
        if since:
            sql += f" AND {col}>=?"
            params.append(iso_or_now(since))
        if until:
            sql += f" AND {col}<=?"
            params.append(iso_or_now(until))
        if normalize_clock(clock) == "storage_time":
            sql += " ORDER BY ts DESC, event_time DESC, rowid DESC LIMIT ? OFFSET ?"
        else:
            sql += " ORDER BY event_time DESC, ts DESC, rowid DESC LIMIT ? OFFSET ?"
        params.append(clamp_limit(limit, default=100))
        try:
            off = int(offset)
        except (TypeError, ValueError):
            off = 0
        params.append(max(0, off))
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def stats(self) -> dict[str, Any]:
        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        tiers = {
            r["tier"]: r["n"]
            for r in self.conn.execute(
                "SELECT tier, COUNT(*) AS n FROM events GROUP BY tier"
            )
        }
        last = self.conn.execute(
            "SELECT ts, event_time FROM events ORDER BY ts DESC, rowid DESC LIMIT 1"
        ).fetchone()
        db = Path(self.db_path)
        size = db.stat().st_size if db.exists() else 0
        wal = db.with_suffix(db.suffix + "-wal")
        if wal.exists():
            size += wal.stat().st_size
        return {
            "namespace": self.name,
            "db_path": str(db.resolve()),
            "db_size_bytes": size,
            "events": count("events"),
            "memories": count("memories"),
            "sessions": count("sessions"),
            "entities": count("entities"),
            "relations": count("relations"),
            "embedding_jobs": count("embedding_jobs"),
            "corrections": count("corrections"),
            "lineage_tombstones": count("lineage_tombstones"),
            "tiers": tiers,
            "last_write": last["ts"] if last else None,
            "last_event_time": last["event_time"] if last else None,
            "wal": True,
        }

    def top_entities(
        self,
        limit: int = 15,
        *,
        trusted_only: bool = False,
    ) -> list[dict[str, Any]]:
        trusted_clause = ""
        if trusted_only:
            trusted_clause = """
            WHERE EXISTS (
                SELECT 1
                FROM entity_mentions em
                JOIN events ev ON ev.id=em.event_id
                WHERE em.entity_id=e.id
                  AND ev.role != 'tool'
                  AND ev.tool_name IS NULL
            )
            """
        rows = self.conn.execute(
            f"""
            SELECT e.id, e.name, e.type, e.norm_name, e.first_seen, e.last_seen,
                   (SELECT COUNT(*) FROM relations r
                    WHERE r.src_entity=e.id OR r.dst_entity=e.id) AS rels
            FROM entities e
            {trusted_clause}
            ORDER BY e.last_seen DESC
            LIMIT ?
            """,
            (clamp_limit(limit, default=15),),
        ).fetchall()
        return [dict(r) for r in rows]

    def graph(self, entity: str | None = None) -> dict[str, Any]:
        if entity:
            norm = entity.strip().lower()
            ents = [
                dict(r)
                for r in self.conn.execute(
                    "SELECT * FROM entities WHERE norm_name LIKE ? OR name LIKE ? OR id=?",
                    (f"%{norm}%", f"%{entity}%", entity),
                )
            ]
            ids = [e["id"] for e in ents]
            rels: list[dict[str, Any]] = []
            if ids:
                placeholders = ",".join("?" * len(ids))
                rels = [
                    dict(r)
                    for r in self.conn.execute(
                        f"SELECT * FROM relations WHERE src_entity IN ({placeholders}) OR dst_entity IN ({placeholders})",
                        ids + ids,
                    )
                ]
            return {"entities": ents, "relations": rels}
        return {
            "entities": [dict(r) for r in self.conn.execute("SELECT * FROM entities ORDER BY last_seen DESC LIMIT 200")],
            "relations": [dict(r) for r in self.conn.execute("SELECT * FROM relations ORDER BY valid_from DESC LIMIT 400")],
        }

    def rebuild_graph(self, *, touch: bool = True) -> dict[str, Any]:
        """Rebuild graph evidence and derived aggregates from stored events."""
        from haunt.graph import extract_and_store

        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        before_ents = count("entities")
        before_rels = count("relations")
        events_n = count("events")
        memories_n = count("memories")

        try:
            self.conn.execute("DELETE FROM relation_evidence")
            self.conn.execute("DELETE FROM entity_mentions")
            self.conn.execute("DELETE FROM relations")
            self.conn.execute("DELETE FROM entities")

            rows = self.conn.execute(
                """
                SELECT id, content, tool_name, tool_input, tool_output, event_time
                FROM events
                ORDER BY event_time ASC, ts ASC, rowid ASC
                """
            ).fetchall()
            for r in rows:
                text = verbatim_text(
                    r["content"] or "",
                    r["tool_name"],
                    r["tool_input"],
                    r["tool_output"],
                )
                extract_and_store(
                    self.conn,
                    r["id"],
                    text,
                    r["event_time"],
                    r["tool_name"],
                    commit=False,
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("graph_evidence_version", "1"),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        if touch:
            try:
                touch_namespace(self.name)
            except Exception:
                pass

        return {
            "events": events_n,
            "memories": memories_n,
            "entities_before": before_ents,
            "relations_before": before_rels,
            "entities": count("entities"),
            "relations": count("relations"),
        }


    # ------------------------------------------------------------------
    # purge: hard-delete a memory and its entire provenance chain
    # ------------------------------------------------------------------

    def purge(self, memory_id: str) -> dict[str, Any]:
        """Hard-delete a memory and clean up all associated data.

        Removes: memory row, FTS row, vec row, graph rows tied to the
        memory's event, and the event itself if no other memories reference it.
        Returns a report of what was deleted.
        """
        row = self.conn.execute(
            """
            SELECT m.id, m.event_id, m.content,
                   e.origin, e.session_id,
                   e.ts AS event_ts, e.event_time, e.role AS event_role,
                   e.tier AS event_tier,
                   e.idempotency_key AS event_idempotency_key,
                   e.content AS event_content,
                   e.tool_name, e.tool_input, e.tool_output,
                   e.meta AS event_meta
            FROM memories m JOIN events e ON e.id=m.event_id
            WHERE m.id=?
            """,
            (memory_id,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": "memory not found"}

        event_id = row["event_id"]
        deleted: dict[str, Any] = {
            "ok": True,
            "fts_deleted": False,
            "vec_deleted": False,
            "relations_deleted": 0,
            "entities_deleted": 0,
            "event_deleted": False,
            "session_deleted": False,
        }

        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self._privacy_purge_thread_id = get_ident()
            # Privacy erasure is the sole exception to correction immutability.
            # Replace an erased chain member with a fresh opaque tombstone and
            # scrub correction request/context fields that could retain it.
            lineage_rows = self.conn.execute(
                """
                SELECT * FROM corrections
                WHERE target_memory_id=? OR replacement_memory_id=?
                """,
                (memory_id, memory_id),
            ).fetchall()
            needs_tombstone = any(
                r["replacement_memory_id"] is not None
                or r["replacement_tombstone_id"] is not None
                for r in lineage_rows
                if r["target_memory_id"] == memory_id
            ) or any(r["replacement_memory_id"] == memory_id for r in lineage_rows)
            tombstone: dict[str, Any] | None = None
            sessions_to_cleanup: dict[str, dict[str, Any]] = {}
            erased_values = _erasure_context_values(
                memory_id,
                event_id,
                row["content"],
                row["origin"],
                row["session_id"],
                row["event_idempotency_key"],
                row["event_content"],
                row["tool_name"],
                row["tool_input"],
                row["tool_output"],
                row["event_meta"],
            )

            def track_erased_session(
                session_id: object, *context_values: object
            ) -> None:
                if session_id is None:
                    return
                info = sessions_to_cleanup.setdefault(
                    str(session_id), {"sensitive_values": set(erased_values)}
                )
                info["sensitive_values"].update(
                    _erasure_context_values(*context_values)
                )

            # A target session ID is erased context even when that session
            # predates the target and still contains unrelated events.
            track_erased_session(row["session_id"], *erased_values)
            if needs_tombstone:
                tombstone = {
                    "schema_version": TOMBSTONE_SCHEMA_VERSION,
                    "tombstone_id": new_id(),
                    "status": "erased",
                    "erased_at": now_iso(),
                }
                self.conn.execute(
                    """
                    INSERT INTO lineage_tombstones(
                        schema_version, tombstone_id, status, erased_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    tuple(tombstone.values()),
                )
            for correction in lineage_rows:
                correction_session = correction["session_id"]
                track_erased_session(
                    correction_session,
                    correction["origin"],
                    correction["session_id"],
                    correction["reason"],
                    correction["idempotency_key"],
                    correction["request_identity"],
                    correction["target_tombstone_id"],
                    correction["replacement_tombstone_id"],
                )
                if correction["target_memory_id"] == memory_id:
                    self._sanitize_correction_replacement_event(
                        correction, erased_memory_id=memory_id
                    )
                    has_successor = (
                        correction["replacement_memory_id"] is not None
                        or correction["replacement_tombstone_id"] is not None
                    )
                    if not has_successor:
                        self.conn.execute(
                            "DELETE FROM corrections WHERE id=?", (correction["id"],)
                        )
                        continue
                    self.conn.execute(
                        """
                        UPDATE corrections
                        SET target_memory_id=NULL, target_tombstone_id=?,
                            origin=NULL, session_id=NULL, reason=NULL,
                            idempotency_key=NULL, request_identity=NULL,
                            request_payload=NULL, response_json=NULL
                        WHERE id=?
                        """,
                        (tombstone["tombstone_id"], correction["id"]),
                    )
                if correction["replacement_memory_id"] == memory_id:
                    self.conn.execute(
                        """
                        UPDATE corrections
                        SET replacement_memory_id=NULL, replacement_tombstone_id=?,
                            origin=NULL, session_id=NULL, reason=NULL,
                            idempotency_key=NULL, request_identity=NULL,
                            request_payload=NULL, response_json=NULL
                        WHERE id=?
                        """,
                        (tombstone["tombstone_id"], correction["id"]),
                    )

            self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))

            has_fts = self.conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='memories_fts'"
            ).fetchone()
            if has_fts:
                self.conn.execute(
                    "DELETE FROM memories_fts WHERE id=?", (memory_id,)
                )
                deleted["fts_deleted"] = True
            has_vec = self.conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='vec_memories'"
            ).fetchone()
            if has_vec:
                self.conn.execute(
                    "DELETE FROM vec_memories WHERE id=?", (memory_id,)
                )
                deleted["vec_deleted"] = True

            other_memories = self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE event_id=?", (event_id,)
            ).fetchone()[0]
            from haunt.graph import remove_event_evidence

            rel_count, entity_count = remove_event_evidence(self.conn, event_id)
            deleted["relations_deleted"] = rel_count
            deleted["entities_deleted"] = entity_count
            if other_memories == 0:
                self.conn.execute("DELETE FROM events WHERE id=?", (event_id,))
                deleted["event_deleted"] = True
            else:
                # One event may have more than one materialized memory. The
                # survivors remain, but neither the shared event's identifier
                # nor its target-owned context may survive privacy erasure.
                safe_event_id = new_id()
                self.conn.execute(
                    """
                    INSERT INTO events(
                        id, idempotency_key, session_id, ts, event_time, role,
                        content, tool_name, tool_input, tool_output, origin,
                        tier, meta
                    ) VALUES (?, NULL, ?, ?, ?, ?, '', NULL, NULL, NULL, ?, ?, ?)
                    """,
                    (
                        safe_event_id,
                        row["session_id"],
                        row["event_ts"],
                        row["event_time"],
                        row["event_role"],
                        PURGE_SAFE_ORIGIN,
                        row["event_tier"],
                        dumps({}),
                    ),
                )
                self.conn.execute(
                    "UPDATE memories SET event_id=? WHERE event_id=?",
                    (safe_event_id, event_id),
                )
                self.conn.execute("DELETE FROM events WHERE id=?", (event_id,))
                deleted["event_deleted"] = True

            for session_id, session_info in sessions_to_cleanup.items():
                session_refs = self.conn.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM events WHERE session_id=?) +
                      (SELECT COUNT(*) FROM corrections WHERE session_id=?)
                    """,
                    (session_id, session_id),
                ).fetchone()[0]
                started_at, ended_at, safe_source, safe_meta = (
                    self._purge_safe_session_context(
                        session_id, session_info["sensitive_values"]
                    )
                )
                if session_refs > 0:
                    safe_session = self._create_purge_safe_session(
                        started_at=started_at,
                        ended_at=ended_at,
                        source=safe_source,
                        meta=safe_meta,
                    )
                    # Session IDs attached to the target or adjacent correction
                    # are erased context. Rekey every remaining reference while
                    # preserving unrelated event content and origins.
                    self.conn.execute(
                        "UPDATE events SET session_id=? WHERE session_id=?",
                        (safe_session, session_id),
                    )
                    self.conn.execute(
                        "UPDATE corrections SET session_id=? WHERE session_id=?",
                        (safe_session, session_id),
                    )
                    self.conn.execute(
                        "UPDATE meta SET value=? WHERE key='current_session' AND value=?",
                        (safe_session, session_id),
                    )
                    self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
                    deleted["session_deleted"] = True
                else:
                    self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
                    self.conn.execute(
                        "DELETE FROM meta WHERE key='current_session' AND value=?",
                        (session_id,),
                    )
                    deleted["session_deleted"] = True

            if tombstone is not None:
                deleted["lineage_tombstone"] = tombstone

            self._prune_erased_only_lineage()

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self._privacy_purge_thread_id = None
        try:
            touch_namespace(self.name)
        except Exception:
            pass
        return deleted

    def _sanitize_correction_replacement_event(
        self,
        correction: sqlite3.Row,
        *,
        erased_memory_id: str,
    ) -> None:
        """Remove purged correction context from its surviving replacement event.

        Only the direct replacement created by this correction is eligible.
        Content and unrelated event origins are never changed. Session rekeying
        is handled once for every target and adjacent correction session.
        """
        replacement_id = correction["replacement_memory_id"]
        if replacement_id is None or replacement_id == erased_memory_id:
            return
        event = self.conn.execute(
            """
            SELECT e.id, e.origin
            FROM memories m JOIN events e ON e.id=m.event_id
            WHERE m.id=?
            """,
            (replacement_id,),
        ).fetchone()
        if event is None:
            return

        correction_origin = correction["origin"]
        origin_matches = (
            correction_origin is not None and event["origin"] == correction_origin
        )
        if not origin_matches:
            return

        updates: list[str] = []
        params: list[Any] = []
        if origin_matches:
            updates.append("origin=?")
            params.append(PURGE_SAFE_ORIGIN)
        params.append(event["id"])
        self.conn.execute(
            f"UPDATE events SET {', '.join(updates)} WHERE id=?",
            params,
        )

    def _create_purge_safe_session(
        self,
        *,
        started_at: str | None = None,
        ended_at: str | None = None,
        source: Any,
        meta: Any,
    ) -> str:
        safe_session = new_id()
        self.conn.execute(
            """
            INSERT INTO sessions(id, started_at, ended_at, source, meta)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                safe_session,
                now_iso() if started_at is None else started_at,
                ended_at,
                source,
                meta,
            ),
        )
        return safe_session

    def _purge_safe_session_context(
        self, session_id: str, sensitive_values: set[str]
    ) -> tuple[str | None, str | None, Any, Any]:
        """Preserve clean session fields and remove only erased context."""
        row = self.conn.execute(
            "SELECT started_at, ended_at, source, meta FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None, None, PURGE_SAFE_SESSION_SOURCE, dumps({})
        source = row["source"]
        original_meta = row["meta"]
        dropped = object()

        def tainted(value: Any) -> bool:
            if isinstance(value, bytes):
                return any(
                    value == token.encode("utf-8")
                    or (len(token) >= 8 and token.encode("utf-8") in value)
                    for token in sensitive_values
                )
            if not isinstance(value, str):
                return False
            return any(
                value == token or (len(token) >= 8 and token in value)
                for token in sensitive_values
            )

        safe_source = PURGE_SAFE_SESSION_SOURCE if tainted(source) else source
        if isinstance(original_meta, bytes):
            try:
                meta_text = original_meta.decode("utf-8")
            except UnicodeDecodeError:
                # Opaque metadata on an affected session cannot be proven free
                # of erased context. Privacy purge therefore drops it even when
                # no plaintext marker can be found in the raw bytes.
                return row["started_at"], row["ended_at"], safe_source, dumps({})
        else:
            meta_text = original_meta
        not_json = object()
        original = loads(meta_text, default=not_json)

        def contains_tainted(value: Any) -> bool:
            if tainted(value):
                return True
            if isinstance(value, dict):
                return any(
                    contains_tainted(key) or contains_tainted(child)
                    for key, child in value.items()
                )
            if isinstance(value, list):
                return any(contains_tainted(child) for child in value)
            return False

        if not tainted(original_meta) and (
            original is not_json or not contains_tainted(original)
        ):
            return row["started_at"], row["ended_at"], safe_source, original_meta
        if original is not_json:
            return row["started_at"], row["ended_at"], safe_source, dumps({})

        def sanitize(value: Any) -> Any:
            if isinstance(value, str):
                return dropped if tainted(value) else value
            if isinstance(value, dict):
                clean: dict[Any, Any] = {}
                for key, child in value.items():
                    if isinstance(key, str) and tainted(key):
                        continue
                    sanitized = sanitize(child)
                    if sanitized is not dropped:
                        clean[key] = sanitized
                return clean
            if isinstance(value, list):
                return [
                    sanitized
                    for child in value
                    if (sanitized := sanitize(child)) is not dropped
                ]
            return value

        sanitized_meta = dumps(sanitize(original))
        if tainted(sanitized_meta):
            sanitized_meta = dumps({})
        return row["started_at"], row["ended_at"], safe_source, sanitized_meta

    def _prune_erased_only_lineage(self) -> None:
        """During purge, discard components that no surviving memory can trace."""
        rows = self.conn.execute(
            """
            SELECT id, target_memory_id, target_tombstone_id,
                   replacement_memory_id, replacement_tombstone_id
            FROM corrections
            """
        ).fetchall()

        def nodes(row: sqlite3.Row) -> list[tuple[str, str]]:
            found: list[tuple[str, str]] = []
            for prefix in ("target", "replacement"):
                if row[f"{prefix}_memory_id"] is not None:
                    found.append(("memory", str(row[f"{prefix}_memory_id"])))
                elif row[f"{prefix}_tombstone_id"] is not None:
                    found.append(("tombstone", str(row[f"{prefix}_tombstone_id"])))
            return found

        by_node: dict[tuple[str, str], set[str]] = {}
        row_nodes: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            correction_nodes = nodes(row)
            row_nodes[str(row["id"])] = correction_nodes
            for item in correction_nodes:
                by_node.setdefault(item, set()).add(str(row["id"]))

        pending = set(row_nodes)
        while pending:
            seed = pending.pop()
            component = {seed}
            frontier = [seed]
            component_nodes: set[tuple[str, str]] = set()
            while frontier:
                correction_id = frontier.pop()
                for item in row_nodes[correction_id]:
                    component_nodes.add(item)
                    for neighbor in by_node[item]:
                        if neighbor in pending:
                            pending.remove(neighbor)
                            component.add(neighbor)
                            frontier.append(neighbor)
            if any(kind == "memory" for kind, _ in component_nodes):
                continue
            self.conn.executemany(
                "DELETE FROM corrections WHERE id=?", [(item,) for item in component]
            )

        self.conn.execute(
            """
            DELETE FROM lineage_tombstones
            WHERE tombstone_id NOT IN (
                SELECT target_tombstone_id FROM corrections
                WHERE target_tombstone_id IS NOT NULL
                UNION
                SELECT replacement_tombstone_id FROM corrections
                WHERE replacement_tombstone_id IS NOT NULL
            )
            """
        )

    def trace(self, memory_id: str) -> dict[str, Any]:
        """Return the ordered correction chain containing a surviving memory.

        Correction records are append-only during ordinary operation. Purge may
        scrub/delete adjacent records and substitute allowlisted tombstones.
        """
        requested = self.conn.execute(
            "SELECT id, valid_to FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if requested is None:
            return {"ok": False, "error": "memory not found"}

        correction_rows = self.conn.execute(
            "SELECT * FROM corrections ORDER BY corrected_at, rowid"
        ).fetchall()

        def node(row: sqlite3.Row, prefix: str) -> tuple[str, str] | None:
            memory = row[f"{prefix}_memory_id"]
            if memory is not None:
                return ("memory", str(memory))
            tombstone = row[f"{prefix}_tombstone_id"]
            if tombstone is not None:
                return ("tombstone", str(tombstone))
            return None

        incoming: dict[tuple[str, str], sqlite3.Row] = {}
        outgoing: dict[tuple[str, str], sqlite3.Row] = {}
        for correction in correction_rows:
            target = node(correction, "target")
            replacement = node(correction, "replacement")
            if target is not None:
                outgoing[target] = correction
            if replacement is not None:
                incoming[replacement] = correction

        start = ("memory", memory_id)
        seen: set[tuple[str, str]] = set()
        while start in incoming and start not in seen:
            seen.add(start)
            predecessor = node(incoming[start], "target")
            if predecessor is None:
                break
            start = predecessor

        members: list[dict[str, Any]] = []
        corrections: list[dict[str, Any]] = []
        current = start
        seen.clear()
        while current not in seen:
            seen.add(current)
            if current[0] == "tombstone":
                tomb = self.conn.execute(
                    """
                    SELECT schema_version, tombstone_id, status, erased_at
                    FROM lineage_tombstones WHERE tombstone_id=?
                    """,
                    (current[1],),
                ).fetchone()
                if tomb is None:
                    break
                members.append(dict(tomb))
            else:
                memory = self.conn.execute(
                    """
                    SELECT m.id AS memory_id, m.event_id, m.content, m.tier,
                           m.valid_from, m.valid_to, m.created_at,
                           e.session_id, e.event_time, e.ts, e.role, e.origin
                    FROM memories m JOIN events e ON e.id=m.event_id
                    WHERE m.id=?
                    """,
                    (current[1],),
                ).fetchone()
                if memory is None:
                    break
                member = dict(memory)
                if current in outgoing:
                    member["status"] = "superseded"
                elif member["valid_to"] is not None:
                    member["status"] = "legacy_unlinked"
                else:
                    member["status"] = "current"
                members.append(member)

            correction = outgoing.get(current)
            if correction is None:
                break
            corrections.append(
                {
                    "correction_id": correction["id"],
                    "corrected_at": correction["corrected_at"],
                    "origin": correction["origin"],
                    "session_id": correction["session_id"],
                    "reason": correction["reason"],
                }
            )
            successor = node(correction, "replacement")
            if successor is None:
                break
            current = successor

        linked = bool(corrections or incoming.get(("memory", memory_id)))
        lineage_status = "linked" if linked else (
            "legacy_unlinked" if requested["valid_to"] is not None else "standalone"
        )
        return {
            "ok": True,
            "schema_version": 1,
            "namespace": self.name,
            "requested_memory_id": memory_id,
            "lineage_status": lineage_status,
            "members": members,
            "corrections": corrections,
        }

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Retrieve full provenance detail for a single memory."""
        row = self.conn.execute(
            """
            SELECT m.id AS memory_id, m.event_id, m.tier, m.content,
                   m.valid_from, m.valid_to, m.created_at,
                   e.session_id, e.ts, e.event_time, e.role, e.content AS event_content,
                   e.tool_name, e.tool_input, e.tool_output, e.origin, e.meta
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE m.id = ?
            """,
            (memory_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["db_path"] = str(Path(self.db_path).resolve())
        d["haunt_home"] = str(haunt_home())
        d["namespace"] = self.name
        d["has_embedding"] = self.conn.execute(
            "SELECT embedding IS NOT NULL AS has FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()["has"]

        mentions = self.conn.execute(
            """
            SELECT DISTINCT e.id, e.name, e.type
            FROM entities e
            JOIN relations r ON (r.src_entity = e.id OR r.dst_entity = e.id)
            WHERE r.event_id = ?
            """,
            (d["event_id"],),
        ).fetchall()
        d["entity_mentions"] = [dict(m) for m in mentions]

        related = self.conn.execute(
            """
            SELECT m.id AS memory_id, m.tier, m.content, m.valid_from, m.valid_to
            FROM memories m
            WHERE m.event_id IN (
                SELECT id FROM events WHERE session_id = ?
            ) AND m.id != ?
            ORDER BY m.created_at DESC, m.rowid DESC
            LIMIT 20
            """,
            (d["session_id"], memory_id),
        ).fetchall()
        d["related_memories"] = [dict(r) for r in related]
        d["trace"] = self.trace(memory_id)

        return d

    def browse_memories(
        self,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        tier: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse memories with filters. Returns paginated results."""
        sql = """
            SELECT m.id AS memory_id, m.event_id, m.tier, m.content,
                   m.valid_from, m.valid_to, m.created_at,
                   e.session_id, e.event_time, e.role, e.origin, e.tool_name
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE 1=1
        """
        count_sql = """
            SELECT COUNT(*) FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE 1=1
        """
        params: list[Any] = []
        if session_id:
            sql += " AND e.session_id = ?"
            count_sql += " AND e.session_id = ?"
            params.append(session_id)
        if origin:
            sql += " AND e.origin = ?"
            count_sql += " AND e.origin = ?"
            params.append(origin)
        if tier:
            sql += " AND m.tier = ?"
            count_sql += " AND m.tier = ?"
            params.append(tier)
        if since:
            sql += " AND e.event_time >= ?"
            count_sql += " AND e.event_time >= ?"
            params.append(iso_or_now(since))
        if until:
            sql += " AND e.event_time <= ?"
            count_sql += " AND e.event_time <= ?"
            params.append(iso_or_now(until))

        total = self.conn.execute(count_sql, params).fetchone()[0]
        sql += " ORDER BY m.created_at DESC, m.rowid DESC LIMIT ? OFFSET ?"
        rows = self.conn.execute(sql, params + [limit, offset]).fetchall()
        return {
            "memories": [dict(r) for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # worldview: compact per-namespace briefing
    # ------------------------------------------------------------------

    def worldview(self, *, facts_cap: int = 12, names_cap: int = 12) -> dict[str, Any]:
        """Compile a structured namespace briefing from existing rows.

        No LLM. Pure read queries over stored semantic/procedural/entity data.
        """
        facts_cap = clamp_limit(facts_cap, default=12)
        names_cap = clamp_limit(names_cap, default=12)
        facts = [
            dict(r)
            for r in self.conn.execute(
                """
                SELECT m.id, m.content, m.valid_from, m.created_at
                FROM memories m
                JOIN events e ON e.id=m.event_id
                WHERE m.tier='semantic' AND m.valid_to IS NULL
                  AND e.role != 'tool' AND e.tool_name IS NULL
                ORDER BY m.created_at DESC, m.rowid DESC
                LIMIT ?
                """,
                (facts_cap,),
            ).fetchall()
        ]

        procedures = [
            dict(r)
            for r in self.conn.execute(
                """
                SELECT m.id, m.content, e.meta
                FROM memories m
                JOIN events e ON e.id = m.event_id
                WHERE m.tier='procedural' AND m.valid_to IS NULL AND e.meta LIKE '%"kind": "procedure"%'
                ORDER BY m.created_at DESC, m.rowid DESC
                """,
            ).fetchall()
        ]
        proc_index: list[dict[str, Any]] = []
        for p in procedures:
            emeta = loads(p.get("meta"))
            proc_index.append({
                "id": p["id"],
                "name": emeta.get("name", ""),
                "trigger": emeta.get("trigger", ""),
            })

        names = self.top_entities(limit=names_cap, trusted_only=True)
        name_list = [{"name": n["name"], "type": n["type"], "mentions": n["rels"]} for n in names]

        stats = self.stats()
        counts = {
            "events": stats["events"],
            "memories": stats["memories"],
            "sessions": stats["sessions"],
        }

        return {
            "namespace": self.name,
            "facts": facts,
            "names": name_list,
            "procedures": proc_index,
            "counts": counts,
        }

    # ------------------------------------------------------------------
    # procedure: named how-tos
    # ------------------------------------------------------------------

    def procedure_write(
        self,
        name: str,
        body: str,
        *,
        trigger: str = "",
        origin: str = "cli",
        session_id: str | None = None,
    ) -> ObserveResult:
        """Store a named procedure. Verbatim body, stored as tier=procedural."""
        meta = {"kind": "procedure", "name": name, "trigger": trigger}
        return self.observe(
            body,
            role="system",
            tier="procedural",
            session_id=session_id,
            origin=origin,
            meta=meta,
        )

    def procedure_get(self, name: str) -> dict[str, Any] | None:
        """Retrieve a procedure by name. Returns newest matching row."""
        row = self.conn.execute(
            """
            SELECT m.id, m.content, m.valid_from, m.valid_to, m.created_at, e.meta
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE m.tier='procedural'
              AND m.valid_to IS NULL
              AND json_extract(e.meta, '$.kind') = 'procedure'
              AND json_extract(e.meta, '$.name') = ?
            ORDER BY m.created_at DESC, m.rowid DESC
            LIMIT 1
            """,
            (name,),
        ).fetchone()
        if not row:
            return None
        emeta = loads(row["meta"])
        return {
            "id": row["id"],
            "name": emeta.get("name", name),
            "body": row["content"],
            "trigger": emeta.get("trigger", ""),
            "valid_from": row["valid_from"],
            "created_at": row["created_at"],
        }

    def procedure_list(self) -> list[dict[str, Any]]:
        """List all active procedures (valid_to IS NULL)."""
        rows = self.conn.execute(
            """
            SELECT m.id, m.content, m.created_at, e.meta
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE m.tier='procedural'
              AND m.valid_to IS NULL
              AND e.meta LIKE '%"kind": "procedure"%'
            ORDER BY m.created_at DESC, m.rowid DESC
            """,
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            emeta = loads(r["meta"])
            out.append({
                "id": r["id"],
                "name": emeta.get("name", ""),
                "body": r["content"],
                "trigger": emeta.get("trigger", ""),
                "created_at": r["created_at"],
            })
        return out

    # ------------------------------------------------------------------
    # contradict: supersede a memory
    # ------------------------------------------------------------------

    def _correction_replay(
        self, key: str, payload: bytes
    ) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT request_payload, response_json FROM corrections WHERE idempotency_key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if row["request_payload"] != payload:
            return {
                "ok": False,
                "conflict": "idempotency_key_reused",
                "error": "idempotency_key was reused with a different correction payload",
            }
        original = loads(row["response_json"], default={})
        original["deduplicated"] = True
        return original

    def contradict(
        self,
        memory_id: str,
        *,
        idempotency_key: str,
        replacement: str | None = None,
        namespace: str | None = None,
        origin: str = "cli",
        session_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Append a correction and update its current/as-of projection atomically."""
        if replacement is not None and not isinstance(replacement, str):
            raise ValueError("replacement must be a string or null")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string or null")
        if not isinstance(idempotency_key, str):
            raise ValueError("idempotency_key must be a string")
        key = idempotency_key
        if not key or not key.strip():
            raise ValueError("idempotency_key must be non-empty")
        if len(key) > CORRECTION_KEY_MAX:
            raise ValueError(
                f"idempotency_key must be {CORRECTION_KEY_MAX} characters or fewer"
            )

        # Null, empty, and whitespace-only replacements are distinct canonical
        # requests. An explicitly supplied string is always stored verbatim.
        payload = _correction_request_payload(memory_id, replacement, reason)
        request_identity = _correction_request_identity(payload)
        replay_result = self._correction_replay(key, payload)
        if replay_result is not None:
            return replay_result
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError("origin must be a non-empty string")
        if session_id is not None and not isinstance(session_id, str):
            raise ValueError("session_id must be a string or null")
        replacement_text = replacement

        try:
            self.conn.execute("BEGIN IMMEDIATE")
            replay_result = self._correction_replay(key, payload)
            if replay_result is not None:
                self.conn.rollback()
                return replay_result

            row = self.conn.execute(
                "SELECT id, valid_to FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if not row:
                self.conn.rollback()
                return {"ok": False, "error": f"memory {memory_id} not found"}
            if row["valid_to"] is not None:
                self.conn.rollback()
                return {
                    "ok": False,
                    "conflict": "already_superseded",
                    "error": f"memory {memory_id} already superseded",
                    "valid_to": row["valid_to"],
                }

            ts = now_iso()
            correction_id = new_id()
            sid = self.ensure_session(session_id, source=origin, commit=False)
            cur = self.conn.execute(
                "UPDATE memories SET valid_to=? WHERE id=? AND valid_to IS NULL",
                (ts, memory_id),
            )
            if cur.rowcount != 1:
                self.conn.rollback()
                again = self.conn.execute(
                    "SELECT valid_to FROM memories WHERE id=?", (memory_id,)
                ).fetchone()
                return {
                    "ok": False,
                    "conflict": "already_superseded",
                    "error": f"memory {memory_id} already superseded",
                    "valid_to": None if again is None else again["valid_to"],
                }
            result: dict[str, Any] = {
                "ok": True,
                "correction_id": correction_id,
                "superseded": memory_id,
                "valid_to": ts,
                "idempotency_key": key,
                "request_identity": request_identity,
                "deduplicated": False,
            }
            replacement_memory_id: str | None = None
            if replacement_text is not None:
                r = self.observe(
                    replacement_text,
                    role="system",
                    tier="semantic",
                    session_id=sid,
                    event_time=ts,
                    valid_from=ts,
                    origin=origin,
                    commit=False,
                )
                replacement_memory_id = r.memory_id
                result["replacement_memory_id"] = r.memory_id
                result["replacement_event_id"] = r.event_id
            self.conn.execute(
                """
                INSERT INTO corrections(
                    id, target_memory_id, replacement_memory_id, corrected_at,
                    origin, session_id, reason, idempotency_key,
                    request_identity, request_payload, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    correction_id,
                    memory_id,
                    replacement_memory_id,
                    ts,
                    origin,
                    sid,
                    reason,
                    key,
                    request_identity,
                    payload,
                    dumps(result),
                ),
            )
            self.conn.commit()
            try:
                touch_namespace(self.name)
            except Exception:
                pass
            return result
        except Exception:
            self.conn.rollback()
            raise


def open_existing(name: str, repo_path: str | None = None) -> Store:
    """Open a registered namespace. Never creates a DB or registry row."""
    name = safe_name(name)
    if not namespace_exists(name):
        raise UnknownNamespaceError(name)
    try:
        return Store(name, repo_path=repo_path, create=False)
    except FileNotFoundError as exc:
        raise UnknownNamespaceError(name) from exc


def get_store(name: str | None = None, repo_path: str | None = None) -> Store:
    ns = resolve_namespace(name)
    return Store(ns, repo_path=repo_path, create=True)


def observe(
    content: str = "",
    *,
    namespace: str | None = None,
    **kwargs: Any,
) -> ObserveResult:
    with get_store(namespace) as store:
        return store.observe(content, **kwargs)


def list_namespaces(*, only: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows = list_namespace_rows()
    if only is not None:
        selected = safe_name(only)
        rows = [row for row in rows if row["name"] == selected]
    for row in rows:
        db = Path(row["db_path"])
        extra: dict[str, Any] = {
            "db_size_bytes": db.stat().st_size if db.exists() else 0,
        }
        try:
            with Store(row["name"], create=False) as st:
                stats = st.stats()
                extra.update(
                    {
                        "events": stats["events"],
                        "memories": stats["memories"],
                        "sessions": stats["sessions"],
                        "entities": stats["entities"],
                        "db_size_bytes": stats["db_size_bytes"],
                    }
                )
        except (sqlite3.Error, OSError) as exc:
            extra["error"] = str(exc)
        out.append({**row, **extra})
    return out


def iter_stores() -> Iterator[Store]:
    for row in list_namespace_rows():
        yield Store(row["name"], create=False)

def reembed_all_namespaces() -> list[dict[str, Any]]:
    """Rebuild embeddings in every registered namespace."""
    out: list[dict[str, Any]] = []
    for row in list_namespace_rows():
        with Store(row["name"], create=False) as st:
            report = st.reembed()
            report["namespace"] = row["name"]
            out.append(report)
    return out
