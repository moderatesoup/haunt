"""Deterministic entity/relation extraction. No LLM NER."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from haunt.util import new_id

# Every type this module can emit. Import validation checks entities.type
# against it because the column has no CHECK constraint.
ENTITY_TYPES = (
    "env", "file", "function", "identifier",
    "proper", "repo", "symbol", "tool", "url",
)

FILE_RE = re.compile(
    r"(?<![\w/])("
    r"(?:[A-Za-z]:)?(?:\.{1,2}/|/)?"
    r"(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,10}"
    r"|[\w.-]+\.(?:py|ts|tsx|js|jsx|mjs|cjs|rs|go|java|kt|rb|php|md|toml|json|ya?ml|sh|sql|css|html|txt|lock)"
    r")\b"
)

FUNC_RE = re.compile(
    r"(?:(?:async\s+)?(?:def|function|fn|fun|func)\s+)?\b([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\s*\("
)

TICK_RE = re.compile(r"`([^`\n]{1,80})`")

QUOTE_RE = re.compile(r"""['"]([A-Za-z0-9_./:-]{2,80})['"]""")

CAMEL_RE = re.compile(
    r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]+)+)\b"
)

PROPER_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

ENV_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")

DOTTED_RE = re.compile(r"\b[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+\b")

SNAKE_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")

URL_RE = re.compile(
    r"(?:[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"`<>\[\]{}]+)"
    r"|(?:\b[A-Za-z][A-Za-z0-9_.-]*:\d{2,5}(?:/[A-Za-z0-9._~/-]*)?)"
)

REPO_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_.-]+/[A-Za-z][A-Za-z0-9_.-]+)\b")

PATH_PREFIXES = {
    "src", "lib", "bin", "app", "pkg", "config", "configs", "test", "tests",
    "docs", "scripts", "internal", "cmd", "dist", "build", "vendor",
    "include", "share", "etc", "opt", "home", "usr", "var", "tmp",
    "node_modules", "target", "out",
}

START_STOP = {
    "the", "this", "that", "these", "those", "a", "an", "i", "we", "you",
    "he", "she", "it", "they", "when", "after", "before", "if", "for",
    "in", "on", "at", "to", "from", "and", "or", "but", "with", "without",
    "please", "then", "else", "so", "as", "of", "by", "ok", "okay", "yes",
    "no", "hi", "hello", "thanks", "thank", "update", "updated", "create",
    "created", "read", "write", "run", "running", "using", "use", "used",
    "call", "called", "function", "class", "file", "path", "error",
    "warning", "info", "debug", "true", "false", "none", "null", "also",
    "next", "last", "first", "noted", "confirmed", "remember", "remembered",
    "user", "assistant", "system", "tool", "here", "there", "today",
    "tomorrow", "yesterday", "now", "later", "done", "todo", "note",
    "notes", "production", "prod", "dev", "test", "testing", "looking",
    "look", "open", "opened", "check", "checked", "see", "seen", "let",
    "lets", "make", "made", "get", "got", "set", "put", "take", "primary",
    "secondary", "new", "old", "latest", "previous", "deploy", "deployment",
    "migrate", "migration", "offline", "online", "stop", "start", "bring",
    "back", "must", "should", "will", "ill", "holds", "asked", "anyway",
    "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "besides", "meanwhile", "already", "still", "just", "really",
    "actually", "basically", "maybe", "perhaps", "sure", "right", "well",
    "again", "once", "twice", "during", "while", "into", "onto", "over",
    "under", "about", "via", "db", "api", "url", "http", "https", "cli",
    "src", "both", "each", "every", "all", "some", "any", "more", "most",
    "other", "another", "same", "such", "own", "than", "too", "very",
    "can", "could", "would", "may", "might", "do", "does", "did", "doing",
    "be", "am", "is", "are", "was", "were", "been", "being", "have", "has",
    "had", "having", "not", "only", "because", "until", "unless", "though",
    "although", "whether", "how", "what", "which", "who", "whom", "why",
    "where", "writers", "writer", "timeout", "connection", "fails",
    "failed", "fail", "midnight", "runbook",
}

FUNC_STOP = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "print",
    "len",
    "str",
    "int",
    "list",
    "dict",
    "set",
    "open",
    "type",
    "super",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "isinstance",
    "hasattr",
}

ENV_STOP = {
    "GET", "PUT", "POST", "HTTP", "HTTPS", "JSON", "UTF", "SQL", "AND",
    "THE", "FOR", "NOT", "NULL", "TRUE", "FALSE", "URL", "URI", "API",
    "CLI", "SRC", "DB",
}


@dataclass(frozen=True)
class Extracted:
    name: str
    type: str
    norm: str


def norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


def _strip_punct(name: str) -> str:
    return name.strip().strip("\"'.,;:()[]{}")


def extract_entities(text: str) -> list[Extracted]:
    found: dict[tuple[str, str], Extracted] = {}

    def add(name: str, typ: str, *, keep_internal: bool = False) -> None:
        if keep_internal:
            name = name.strip().rstrip(".,;!?)]}>\"'")
            name = name.lstrip("\"'([{<")
        else:
            name = _strip_punct(name)
        if not name or len(name) < 2:
            return
        n = norm_name(name)
        if not n or n in START_STOP:
            return
        if re.fullmatch(r"\d{2,5}/[\w.-]+", name) and typ != "url":
            return
        found[(n, typ)] = Extracted(name=name, type=typ, norm=n)

    for m in URL_RE.finditer(text):
        raw = m.group(0).rstrip(".,;!?)]}>\"'")
        if "://" in raw or ":" in raw:
            add(raw, "url", keep_internal=True)
    for m in FILE_RE.finditer(text):
        add(m.group(1), "file")
    for m in TICK_RE.finditer(text):
        raw = m.group(1)
        if FILE_RE.search(raw):
            add(raw, "file")
        elif "(" in raw:
            add(raw.split("(")[0], "function")
        elif "://" in raw or re.search(r":\d{2,5}", raw):
            add(raw, "url", keep_internal=True)
        else:
            add(raw, "symbol")
    for m in QUOTE_RE.finditer(text):
        raw = m.group(1)
        if FILE_RE.search(raw):
            add(raw, "file")
        elif "://" in raw or re.search(r":\d{2,5}", raw):
            add(raw, "url", keep_internal=True)
        elif "." in raw:
            add(raw, "symbol")
        else:
            add(raw, "symbol")
    for m in FUNC_RE.finditer(text):
        fn = m.group(1)
        leaf = fn.split(".")[-1]
        if leaf.lower() not in FUNC_STOP and not leaf[0].isdigit():
            add(fn, "function")
    for m in DOTTED_RE.finditer(text):
        raw = m.group(0)
        if FILE_RE.search(raw):
            continue
        add(raw, "symbol")
    for m in SNAKE_RE.finditer(text):
        raw = m.group(0)
        if raw.lower() in START_STOP:
            continue
        add(raw, "identifier")
    for m in CAMEL_RE.finditer(text):
        add(m.group(1), "identifier")
    for m in ENV_RE.finditer(text):
        tok = m.group(1)
        if tok not in ENV_STOP:
            add(tok, "env")
    for m in PROPER_RE.finditer(text):
        phrase = m.group(1)
        words = phrase.split()
        if words[0].lower() in START_STOP:
            continue
        if all(w.lower() in START_STOP for w in words):
            continue
        if any(ch in phrase for ch in "./\\"):
            continue
        add(phrase, "proper")
    for m in REPO_RE.finditer(text):
        raw = m.group(1)
        if FILE_RE.search(raw) or raw.startswith("http"):
            continue
        head = raw.split("/")[0].lower()
        if head in PATH_PREFIXES or head in START_STOP:
            continue
        if "." not in raw.split("/")[-1] or raw.endswith(".git"):
            add(raw, "repo")

    return list(found.values())


def extract_and_store(
    conn: sqlite3.Connection,
    event_id: str,
    text: str,
    event_time: str,
    tool_name: str | None = None,
    *,
    commit: bool = True,
) -> list[str]:
    ents = extract_entities(text)
    if tool_name:
        ents.append(Extracted(name=tool_name, type="tool", norm=norm_name(tool_name)))
        for m in FILE_RE.finditer(text):
            ents.append(Extracted(name=m.group(1), type="file", norm=norm_name(m.group(1))))

    uniq: dict[tuple[str, str], Extracted] = {}
    for e in ents:
        uniq.setdefault((e.norm, e.type), e)
    ents = list(uniq.values())

    ids: dict[tuple[str, str], str] = {}
    names: list[str] = []
    seen_names: set[str] = set()
    for e in ents:
        row = conn.execute(
            "SELECT id FROM entities WHERE norm_name=? AND type=?",
            (e.norm, e.type),
        ).fetchone()
        if row:
            eid = row["id"]
            conn.execute(
                """
                UPDATE entities
                SET first_seen=MIN(first_seen, ?),
                    last_seen=MAX(last_seen, ?),
                    name=?
                WHERE id=?
                """,
                (event_time, event_time, e.name, eid),
            )
        else:
            eid = new_id()
            conn.execute(
                """
                INSERT INTO entities(id, name, type, norm_name, first_seen, last_seen)
                VALUES (?,?,?,?,?,?)
                """,
                (eid, e.name, e.type, e.norm, event_time, event_time),
            )
        ids[(e.norm, e.type)] = eid
        if e.name not in seen_names:
            seen_names.add(e.name)
            names.append(e.name)

    for eid in ids.values():
        conn.execute(
            """
            INSERT OR IGNORE INTO entity_mentions(event_id, entity_id, observed_at)
            VALUES (?,?,?)
            """,
            (event_id, eid, event_time),
        )

    emitted: set[tuple[str, str, str]] = set()

    def rel(src: str, kind: str, dst: str, weight: float) -> None:
        key = (src, kind, dst)
        if key in emitted:
            return
        emitted.add(key)
        _add_relation_evidence(
            conn, src, kind, dst, event_id, event_time, weight
        )

    id_list = list(ids.values())
    for i, src in enumerate(id_list[:8]):
        for dst in id_list[i + 1 : 8]:
            rel(src, "CO_OCCURS", dst, 1.0)

    if tool_name:
        tool_id = ids.get((norm_name(tool_name), "tool"))
        files = [ids[k] for k in ids if k[1] == "file"]
        kind = "READS"
        low = tool_name.lower()
        if any(w in low for w in ("write", "edit", "patch", "create", "save", "put")):
            kind = "WRITES"
        elif any(w in low for w in ("shell", "bash", "run", "exec")):
            kind = "RUNS"
        if tool_id:
            for fid in files:
                rel(tool_id, kind, fid, 1.5)

    funcs = [ids[k] for k in ids if k[1] == "function"]
    files = [ids[k] for k in ids if k[1] == "file"]
    for fn in funcs:
        for fid in files:
            rel(fn, "LOCATED_IN", fid, 1.0)

    if commit:
        conn.commit()
    return names


def _add_relation_evidence(
    conn: sqlite3.Connection,
    src: str,
    rel: str,
    dst: str,
    event_id: str,
    valid_from: str,
    weight: float,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO relation_evidence(
            event_id, src_entity, rel, dst_entity, observed_at, weight
        ) VALUES (?,?,?,?,?,?)
        """,
        (event_id, src, rel, dst, valid_from, weight),
    )
    _refresh_relation(conn, src, rel, dst)


def _refresh_relation(
    conn: sqlite3.Connection,
    src: str,
    rel: str,
    dst: str,
) -> None:
    aggregate = conn.execute(
        """
        SELECT SUM(weight) AS weight, MIN(observed_at) AS valid_from
        FROM relation_evidence
        WHERE src_entity=? AND rel=? AND dst_entity=?
        """,
        (src, rel, dst),
    ).fetchone()
    rows = conn.execute(
        """
        SELECT id FROM relations
        WHERE src_entity=? AND rel=? AND dst_entity=? AND valid_to IS NULL
        ORDER BY rowid ASC
        """,
        (src, rel, dst),
    ).fetchall()
    if not aggregate or aggregate["weight"] is None:
        for row in rows:
            conn.execute("DELETE FROM relations WHERE id=?", (row["id"],))
        return
    latest = conn.execute(
        """
        SELECT event_id FROM relation_evidence
        WHERE src_entity=? AND rel=? AND dst_entity=?
        ORDER BY observed_at DESC, rowid DESC
        LIMIT 1
        """,
        (src, rel, dst),
    ).fetchone()
    if rows:
        keep = rows[0]["id"]
        conn.execute(
            """
            UPDATE relations
            SET event_id=?, valid_from=?, valid_to=NULL, weight=?
            WHERE id=?
            """,
            (
                latest["event_id"],
                aggregate["valid_from"],
                float(aggregate["weight"]),
                keep,
            ),
        )
        for duplicate in rows[1:]:
            conn.execute("DELETE FROM relations WHERE id=?", (duplicate["id"],))
        return
    conn.execute(
        """
        INSERT INTO relations(
            id, src_entity, rel, dst_entity, event_id, valid_from, valid_to, weight
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            new_id(),
            src,
            rel,
            dst,
            latest["event_id"],
            aggregate["valid_from"],
            None,
            float(aggregate["weight"]),
        ),
    )


def remove_event_evidence(
    conn: sqlite3.Connection,
    event_id: str,
) -> tuple[int, int]:
    """Delete one event's evidence and refresh only affected aggregates."""
    triples = conn.execute(
        """
        SELECT src_entity, rel, dst_entity
        FROM relation_evidence
        WHERE event_id=?
        """,
        (event_id,),
    ).fetchall()
    entity_rows = conn.execute(
        "SELECT entity_id FROM entity_mentions WHERE event_id=?",
        (event_id,),
    ).fetchall()
    conn.execute("DELETE FROM relation_evidence WHERE event_id=?", (event_id,))
    for row in triples:
        _refresh_relation(
            conn, row["src_entity"], row["rel"], row["dst_entity"]
        )
    conn.execute("DELETE FROM entity_mentions WHERE event_id=?", (event_id,))
    entities_deleted = 0
    for row in entity_rows:
        entity_id = row["entity_id"]
        span = conn.execute(
            """
            SELECT MIN(observed_at) AS first_seen, MAX(observed_at) AS last_seen
            FROM entity_mentions WHERE entity_id=?
            """,
            (entity_id,),
        ).fetchone()
        if not span or span["first_seen"] is None:
            conn.execute("DELETE FROM entities WHERE id=?", (entity_id,))
            entities_deleted += 1
        else:
            conn.execute(
                "UPDATE entities SET first_seen=?, last_seen=? WHERE id=?",
                (span["first_seen"], span["last_seen"], entity_id),
            )
    return len(triples), entities_deleted
