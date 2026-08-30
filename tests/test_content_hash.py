"""C7 phase 1: measure byte-identical duplicate content without changing
what gets stored.

Store.observe() computes a SHA-256 content_hash over the exact stored
`content` bytes at admission (no normalization -- see
haunt.store._content_hash's docstring for why). A schema v10 migration adds
the column, indexes it, and backfills it for rows written under any older
schema. Store.stats() exposes duplicate_memories / duplicate_content_values
counts derived from it.

This is a measurement primitive only: nothing here suppresses, collapses,
or otherwise changes which rows a write produces -- that is enforced
directly by test_identical_content_writes_both_produce_independent_rows and
test_58_identical_rows_stay_58_independent_rows below. The pre-existing
idempotency-key mechanism (idx_events_idempotency /
Store._observe_by_idempotency_key) solves a different problem -- replay of
one hook delivery -- and must keep behaving exactly as before; several
tests below exercise it side by side with content_hash duplication to prove
the two mechanisms do not interact.
"""

from __future__ import annotations

import hashlib
import sqlite3

import pytest


@pytest.fixture
def dup_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME, FTS-only -- same pattern as
    tests/test_capture_policy.py::policy_env and
    tests/test_embed_drain.py::drain_env. No real embed model is ever
    loaded; content_hash is orthogonal to embedding, so FTS-only keeps
    these tests fast and independent of the model cache.
    """
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "dup-test")
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry, register_namespace

    embed.reset()
    ensure_layout()
    init_registry()
    register_namespace("dup-test")
    yield home
    embed.reset()


def _content_hashes(store):
    return {
        r["id"]: r["content_hash"]
        for r in store.conn.execute("SELECT id, content_hash FROM memories").fetchall()
    }


# ---------------------------------------------------------------------------
# Hash function: exact-byte, no normalization
# ---------------------------------------------------------------------------


def test_content_hash_is_sha256_hex_of_exact_utf8_bytes():
    """Locks in the hash algorithm itself, independent of Store: a SHA-256
    hex digest over the UTF-8 bytes of the given string, nothing else."""
    from haunt.store import _content_hash

    assert _content_hash("hello") == hashlib.sha256(b"hello").hexdigest()
    assert _content_hash("") == hashlib.sha256(b"").hexdigest()
    snowman = "snowman ☃"
    assert _content_hash(snowman) == hashlib.sha256(snowman.encode("utf-8")).hexdigest()


def test_byte_identical_content_hashes_equal_and_near_misses_differ(dup_env):
    """Core no-normalization guarantee: identical bytes hash identically;
    a one-byte change, a case change, or a whitespace change must each
    produce a DIFFERENT hash. Locks in that normalization / case-folding /
    whitespace-stripping is deliberately absent."""
    from haunt.store import Store

    with Store("dup-test") as store:
        exact_1 = store.observe("The quick brown fox", session_id="s")
        exact_2 = store.observe("The quick brown fox", session_id="s")
        one_byte = store.observe("The quick brown foxx", session_id="s")
        case_changed = store.observe("the quick brown fox", session_id="s")
        trailing_space = store.observe("The quick brown fox ", session_id="s")
        extra_space = store.observe("The quick brown  fox", session_id="s")

        hashes = _content_hashes(store)

    expected = hashlib.sha256(b"The quick brown fox").hexdigest()
    assert hashes[exact_1.memory_id] == expected
    assert hashes[exact_2.memory_id] == expected
    assert hashes[one_byte.memory_id] != expected
    assert hashes[case_changed.memory_id] != expected
    assert hashes[trailing_space.memory_id] != expected
    assert hashes[extra_space.memory_id] != expected
    # Every near-miss is pairwise distinct too, not just distinct from
    # "expected" -- each kind of change must produce its own hash.
    near_misses = {
        hashes[one_byte.memory_id],
        hashes[case_changed.memory_id],
        hashes[trailing_space.memory_id],
        hashes[extra_space.memory_id],
    }
    assert len(near_misses) == 4


# ---------------------------------------------------------------------------
# No suppression: phase 1 changes no write behavior
# ---------------------------------------------------------------------------


def test_identical_content_writes_both_produce_independent_rows(dup_env):
    """The key phase-1 guard: two distinct observe() calls with byte-
    identical content and no idempotency_key must both fully commit --
    two events, two memory rows, two FTS entries -- sharing a content_hash
    but nothing else. Nothing about content_hash may suppress or collapse
    a write."""
    from haunt.store import Store

    with Store("dup-test") as store:
        first = store.observe("haunt session start", session_id="s1")
        second = store.observe("haunt session start", session_id="s2")

        assert first.event_id != second.event_id
        assert first.memory_id != second.memory_id
        assert first.deduplicated is False
        assert second.deduplicated is False

        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM events WHERE content='haunt session start'"
            ).fetchone()[0]
            == 2
        )
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content='haunt session start'"
            ).fetchone()[0]
            == 2
        )
        fts_count = store.conn.execute(
            "SELECT COUNT(*) FROM memories_fts WHERE id IN (?, ?)",
            (first.memory_id, second.memory_id),
        ).fetchone()[0]
        assert fts_count == 2

        hashes = _content_hashes(store)
        assert hashes[first.memory_id] == hashes[second.memory_id]


def test_58_identical_rows_stay_58_independent_rows(dup_env):
    """Direct regression for the measured corpus example in the C7
    background: one namespace held 58 rows that were all the identical
    string "haunt session start". Phase 1 must not collapse this -- 58
    writes must still be 58 rows, just now measurably duplicated."""
    from haunt.store import Store

    with Store("dup-test") as store:
        ids = [
            store.observe("haunt session start", session_id=f"s{i}").memory_id
            for i in range(58)
        ]
        assert len(set(ids)) == 58
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content='haunt session start'"
            ).fetchone()[0]
            == 58
        )

        stats = store.stats()

    assert stats["memories"] == 58
    assert stats["duplicate_content_values"] == 1
    assert stats["duplicate_memories"] == 57


# ---------------------------------------------------------------------------
# Duplicate statistics
# ---------------------------------------------------------------------------


def test_stats_duplicate_counts_zero_on_all_distinct_content(dup_env):
    from haunt.store import Store

    with Store("dup-test") as store:
        for i in range(10):
            store.observe(f"distinct row number {i}", session_id="s")
        stats = store.stats()

    assert stats["memories"] == 10
    assert stats["duplicate_memories"] == 0
    assert stats["duplicate_content_values"] == 0


def test_stats_duplicate_counts_on_a_known_repeat_pattern(dup_env):
    """3x "A", 2x "B", 1x "C", 1x "D": 2 distinct values are duplicated
    (A, B); rows beyond each duplicated group's first sum to
    (3-1) + (2-1) = 3."""
    from haunt.store import Store

    with Store("dup-test") as store:
        for _ in range(3):
            store.observe("A", session_id="s")
        for _ in range(2):
            store.observe("B", session_id="s")
        store.observe("C", session_id="s")
        store.observe("D", session_id="s")
        stats = store.stats()

    assert stats["memories"] == 7
    assert stats["duplicate_content_values"] == 2
    assert stats["duplicate_memories"] == 3


def test_stats_preserves_all_c4_and_earlier_keys(dup_env):
    """C7 adds duplicate_memories/duplicate_content_values to stats() but
    must never remove or rename an existing key -- mirrors
    tests/test_embed_drain.py::test_stats_preserves_all_existing_keys,
    extended with the C4 embedding-coverage keys already present on this
    branch."""
    from haunt.store import Store

    pre_existing_keys = {
        "namespace",
        "db_path",
        "db_size_bytes",
        "events",
        "memories",
        "sessions",
        "entities",
        "relations",
        "embedding_jobs",
        "corrections",
        "lineage_tombstones",
        "tiers",
        "last_write",
        "last_event_time",
        "wal",
        "memories_embedded",
        "embedding_pending",
        "embedding_exhausted",
        "vector_index",
        "vector_index_version",
    }
    with Store("dup-test") as store:
        store.observe("shape check row", session_id="s")
        stats = store.stats()

    missing = pre_existing_keys - stats.keys()
    assert not missing, f"stats() dropped pre-existing keys: {missing}"
    assert "duplicate_memories" in stats
    assert "duplicate_content_values" in stats


def test_haunt_health_shows_the_duplicate_and_coverage_counts(dup_env):
    """stats() computing a number nothing prints is a measurement the user
    cannot act on. `haunt health` is the CLI surface for stats(), and MCP
    memory_health and the dashboard already return the whole dict."""
    from typer.testing import CliRunner

    from haunt import cli
    from haunt.store import Store

    with Store("dup-test") as store:
        for i in range(3):
            store.observe("haunt session start", session_id=f"s{i}")

    result = CliRunner().invoke(cli.app, ["health", "-n", "dup-test"])
    assert result.exit_code == 0, result.output
    assert "duplicates    memories=2 content=1" in result.output
    # Every embedding number stats() computes has to reach this surface,
    # including the denominator and the queue age added for the drain
    # decision -- the whole point of this test.
    assert "embedding     embedded=0/3" in result.output
    assert "pending=" in result.output
    assert "exhausted=" in result.output
    assert "index=False" in result.output
    # None, not 0.0: this namespace is FTS-only, and "0% coverage" would read
    # as unhealthy when there is simply no vector index to be covered.
    assert "coverage=n/a" in result.output
    assert "oldest queued " in result.output


# ---------------------------------------------------------------------------
# Schema / index
# ---------------------------------------------------------------------------


def test_fresh_database_has_content_hash_column_and_index_at_v10(dup_env):
    from haunt.store import SCHEMA_VERSION, Store

    assert SCHEMA_VERSION == 13
    with Store("dup-test") as store:
        assert store.get_meta("schema_version") == "13"
        columns = {
            row["name"]
            for row in store.conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        assert "content_hash" in columns
        index = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_memories_content_hash'"
        ).fetchone()
        assert index is not None


# ---------------------------------------------------------------------------
# Migration: upgrade a populated pre-existing (v9) database
# ---------------------------------------------------------------------------


def test_migration_backfills_hashes_idempotently_without_touching_content(dup_env):
    """Simulates a genuinely pre-v10 database (content_hash column
    physically absent, not merely NULL) the same way
    tests/test_structured_provenance.py::
    test_provenance_migration_preserves_legacy_origin_and_meta_bytes
    simulates a pre-v8 one: write real rows, force schema_version back
    down, then physically drop the new column/index before reopening.
    """
    from haunt.paths import namespace_db_path
    from haunt.store import SCHEMA_VERSION, Store

    with Store("dup-test") as store:
        store.observe("haunt session start", session_id="a")
        store.observe("haunt session start", session_id="b")
        store.observe("distinct content one", session_id="c")
        store.observe("distinct content two", session_id="d")
        before_content = sorted(
            (r["id"], r["content"])
            for r in store.conn.execute("SELECT id, content FROM memories").fetchall()
        )
        store.conn.execute("UPDATE meta SET value='9' WHERE key='schema_version'")
        store.conn.commit()

    db_path = namespace_db_path("dup-test")
    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX IF EXISTS idx_memories_content_hash")
    conn.execute("ALTER TABLE memories DROP COLUMN content_hash")
    conn.commit()
    conn.close()

    with Store("dup-test", create=False) as migrated:
        assert migrated.get_meta("schema_version") == str(SCHEMA_VERSION)

        after_content = sorted(
            (r["id"], r["content"])
            for r in migrated.conn.execute("SELECT id, content FROM memories").fetchall()
        )
        assert after_content == before_content, "migration must not rewrite content"

        stored = migrated.conn.execute(
            "SELECT id, content, content_hash FROM memories"
        ).fetchall()
        assert len(stored) == 4
        for row in stored:
            assert row["content_hash"] == hashlib.sha256(
                row["content"].encode("utf-8")
            ).hexdigest()

        index = migrated.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_memories_content_hash'"
        ).fetchone()
        assert index is not None

        stats = migrated.stats()
        assert stats["duplicate_content_values"] == 1
        assert stats["duplicate_memories"] == 1

        first_pass_hashes = _content_hashes(migrated)

    # Idempotent re-run: reopening again must not change any hash, error,
    # or touch content.
    with Store("dup-test", create=False) as reopened:
        second_pass_hashes = _content_hashes(reopened)
        after_content_again = sorted(
            (r["id"], r["content"])
            for r in reopened.conn.execute("SELECT id, content FROM memories").fetchall()
        )

    assert second_pass_hashes == first_pass_hashes
    assert after_content_again == before_content


def test_migration_running_twice_in_place_is_a_no_op(dup_env):
    """Calling the migration entry point directly a second time against an
    already-migrated connection must not change anything or error --
    exercises _ensure_namespace_schema's own idempotent early-return path
    (current >= SCHEMA_VERSION) rather than only reopening a fresh Store."""
    from haunt.store import Store, _ensure_namespace_schema

    with Store("dup-test") as store:
        store.observe("haunt session start", session_id="a")
        store.observe("haunt session start", session_id="b")
        before = _content_hashes(store)

        _ensure_namespace_schema(store.conn)
        _ensure_namespace_schema(store.conn)

        after = _content_hashes(store)

    assert after == before


def test_readonly_store_reports_honest_zero_on_pre_v10_database(dup_env):
    """Mirrors tests/test_recall_release_gate.py::
    test_alias_and_old_schema_read_only_recall_never_repairs_source:
    ReadOnlyStore never runs migration (see its class docstring), so
    stats() must degrade gracefully -- an honest zero, not a crash --
    against a namespace no writer has migrated at this code version yet.
    """
    from haunt.paths import namespace_db_path
    from haunt.store import Store, open_existing_readonly

    with Store("dup-test") as store:
        store.observe("some content", session_id="s")

    db_path = namespace_db_path("dup-test")
    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX IF EXISTS idx_memories_content_hash")
    conn.execute("ALTER TABLE memories DROP COLUMN content_hash")
    conn.execute("UPDATE meta SET value='9' WHERE key='schema_version'")
    conn.commit()
    conn.close()

    with open_existing_readonly("dup-test") as store:
        assert store.read_only is True
        stats = store.stats()

    assert stats["duplicate_memories"] == 0
    assert stats["duplicate_content_values"] == 0
    assert stats["memories"] == 1


# ---------------------------------------------------------------------------
# content_hash duplication vs. idempotency-key replay: two distinct
# mechanisms that must not interact.
# ---------------------------------------------------------------------------


def test_idempotency_key_replay_is_unaffected_by_content_hash(dup_env):
    """The pre-existing idempotency-key path (a hook retrying the *same*
    delivery) must behave exactly as before: same key + same content
    returns the original row (deduplicated=True, one row in events/
    memories); same key + different content still raises. content_hash
    plays no role in this decision."""
    from haunt.store import Store

    with Store("dup-test") as store:
        first = store.observe(
            "IDEMPOTENT-CANARY", session_id="s", idempotency_key="host-event-1"
        )
        retry = store.observe(
            "IDEMPOTENT-CANARY", session_id="s", idempotency_key="host-event-1"
        )
        assert retry.deduplicated is True
        assert retry.event_id == first.event_id
        assert retry.memory_id == first.memory_id
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM events WHERE idempotency_key='host-event-1'"
            ).fetchone()[0]
            == 1
        )
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE event_id=?", (first.event_id,)
            ).fetchone()[0]
            == 1
        )

        with pytest.raises(ValueError, match="different content"):
            store.observe(
                "DIFFERENT-CONTENT", session_id="s", idempotency_key="host-event-1"
            )

        # The one physical row still gets a content_hash like any other
        # row -- idempotency replay short-circuits before reaching the
        # insert, so only the ORIGINAL write's hash exists.
        row = store.conn.execute(
            "SELECT content_hash FROM memories WHERE id=?", (first.memory_id,)
        ).fetchone()
        assert row["content_hash"] == hashlib.sha256(b"IDEMPOTENT-CANARY").hexdigest()

        stats = store.stats()
        # One logical write replayed twice is one row -- not a duplicate.
        assert stats["duplicate_memories"] == 0
        assert stats["duplicate_content_values"] == 0


def test_content_duplicate_and_idempotency_duplicate_are_independent_axes(dup_env):
    """Four observe() calls with the SAME content: two share an
    idempotency_key (one logical write, replayed -- collapses to one row),
    two do not (two distinct hook deliveries that happen to produce
    identical text -- two independent rows). Only the latter pair counts
    toward content_hash duplicate stats; idempotency replay must not
    inflate it, and content-hash duplication must not affect the
    idempotency replay's row count."""
    from haunt.store import Store

    with Store("dup-test") as store:
        a = store.observe("SAME TEXT", session_id="s", idempotency_key="replayed-key")
        a_retry = store.observe(
            "SAME TEXT", session_id="s", idempotency_key="replayed-key"
        )
        b = store.observe("SAME TEXT", session_id="s")  # no idempotency_key
        c = store.observe("SAME TEXT", session_id="s")  # no idempotency_key

        assert a_retry.deduplicated is True
        assert a_retry.memory_id == a.memory_id
        assert b.memory_id not in (a.memory_id, c.memory_id)
        assert c.memory_id != a.memory_id

        # 3 physical memory rows total: a (its replay adds no new row), b, c.
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content='SAME TEXT'"
            ).fetchone()[0]
            == 3
        )

        stats = store.stats()

    # One content value ("SAME TEXT") shared by 3 rows -> 1 distinct
    # duplicated value, 2 rows beyond the first.
    assert stats["duplicate_content_values"] == 1
    assert stats["duplicate_memories"] == 2


def test_null_content_hash_rows_are_never_counted_as_duplicates(dup_env):
    """NULL hashes must be excluded from grouping, not clustered together.

    A row can legitimately carry a NULL content_hash: anything that
    raw-inserts into `memories` instead of going through observe() (one
    pre-existing test helper does exactly this). SQLite's GROUP BY treats
    all NULLs as one group, so dropping the IS NOT NULL filter in stats()
    silently reports N unrelated NULL rows as N-1 duplicates -- corrupting
    the very metric phase 2 will decide against. The three rows below share
    nothing but their NULL hash, so a zero count here proves the filter is
    doing the work rather than the fixture happening to be unique.
    """
    from haunt.store import Store

    with Store("dup-test") as store:
        seed = store.observe("a genuinely unique row", session_id="s")
        event_id = store.conn.execute(
            "SELECT event_id FROM memories WHERE id=?", (seed.memory_id,)
        ).fetchone()[0]
        for i in range(3):
            store.conn.execute(
                """
                INSERT INTO memories(
                    id, event_id, tier, content, valid_from, created_at,
                    content_hash
                ) VALUES (?,?,?,?,?,?,NULL)
                """,
                (
                    f"null-hash-{i}",
                    event_id,
                    "episodic",
                    f"distinct unhashed content {i}",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
        store.conn.commit()

        null_rows = store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content_hash IS NULL"
        ).fetchone()[0]
        stats = store.stats()

    assert null_rows == 3, "fixture must actually produce NULL-hash rows"
    assert stats["duplicate_memories"] == 0
    assert stats["duplicate_content_values"] == 0


class _BatchCountingConn:
    """sqlite3.Connection stand-in that records each UPDATE batch's size."""

    def __init__(self, conn):
        self._conn = conn
        self.batches: list[int] = []

    def execute(self, sql, params=()):
        return self._conn.execute(sql, params)

    def executemany(self, sql, seq):
        rows = list(seq)
        self.batches.append(len(rows))
        return self._conn.executemany(sql, rows)


def test_backfill_pages_instead_of_loading_a_whole_namespace_at_once(
    dup_env, monkeypatch
):
    """The one-shot v9 -> v10 backfill must not materialize every row's
    full content in a single SELECT. Its semantics (content untouched,
    hashes identical to observe()'s) stay covered by
    test_migration_backfills_hashes_idempotently_without_touching_content.
    """
    from haunt import store as store_mod
    from haunt.store import Store, _backfill_content_hashes

    monkeypatch.setattr(store_mod, "CONTENT_HASH_BACKFILL_BATCH", 10)

    with Store("dup-test") as store:
        for i in range(25):
            store.observe(f"paged backfill row {i}", session_id=f"page-{i}")
        before = sorted(
            (r["id"], r["content"])
            for r in store.conn.execute("SELECT id, content FROM memories").fetchall()
        )
        assert len(before) == 25
        store.conn.execute("UPDATE memories SET content_hash=NULL")
        store.conn.commit()

        counting = _BatchCountingConn(store.conn)
        assert _backfill_content_hashes(counting) == 25
        assert counting.batches == [10, 10, 5], counting.batches

        rows = store.conn.execute(
            "SELECT id, content, content_hash FROM memories"
        ).fetchall()
        assert sorted((r["id"], r["content"]) for r in rows) == before
        for row in rows:
            assert row["content_hash"] == hashlib.sha256(
                row["content"].encode("utf-8")
            ).hexdigest()

        # Nothing left to fill costs one empty page and no UPDATE at all.
        again = _BatchCountingConn(store.conn)
        assert _backfill_content_hashes(again) == 0
        assert again.batches == []
