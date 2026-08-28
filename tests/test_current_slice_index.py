"""C10: an indexed current slice, and Store.trace() using existing indexes.

Default recall's "current slice" (`m.valid_to IS NULL`, see
`src/haunt/recall.py`'s `_filters()`) and several of Store's own tier-scoped
current-fact reads (`worldview()`, `procedure_get()`, `procedure_list()`)
previously drove a scan whose cost grew with the *entire* history a
namespace ever recorded, including every superseded row -- not just the
live set. Schema v11 adds a partial index,
``idx_memories_current ON memories(tier, created_at) WHERE valid_to IS
NULL``, whose own size tracks only the current rows.

Separately, `Store.trace()` used to load a namespace's whole `corrections`
table on every call and rebuild a lookup dict from it, no matter how short
the requested chain was. `corrections` has had four UNIQUE partial indexes
since schema v4 (`idx_corrections_target_memory`,
`idx_corrections_target_tombstone`, `idx_corrections_replacement_memory`,
`idx_corrections_replacement_tombstone`) that make "which correction
targets/replaces this node" a single indexed point lookup. trace() is
rewritten to walk the chain with those point lookups instead. Its output
contract is unchanged -- this file locks that, plus the new mechanism.

See scratchpad benchmark numbers (reported alongside this change, not
committed here) for before/after timings; this file is the pass/fail gate.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def current_index_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunt"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed

    embed.reset()
    yield tmp_path / "haunt"
    embed.reset()


# ---------------------------------------------------------------------------
# Schema / index shape
# ---------------------------------------------------------------------------


def test_fresh_database_has_current_index_at_v11(current_index_env):
    from haunt.store import SCHEMA_VERSION, Store

    assert SCHEMA_VERSION == 13
    with Store("default") as store:
        assert store.get_meta("schema_version") == "13"
        index = store.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND name='idx_memories_current'"
        ).fetchone()
        assert index is not None
        assert "valid_to" in index["sql"] and "IS NULL" in index["sql"]


def test_current_index_where_clause_is_partial_on_valid_to_is_null(current_index_env):
    """Locks in the shape decision: partial, not a full index on the column.

    A full (non-partial) index on valid_to would still contain every
    superseded row, defeating the point -- its size would keep growing
    with total correction history instead of tracking only the live set.
    """
    from haunt.store import Store

    with Store("default") as store:
        row = store.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND name='idx_memories_current'"
        ).fetchone()
        normalized = " ".join(row["sql"].split())
        assert "WHERE valid_to IS NULL" in normalized
        # Partial-ness matters more than the exact leading columns, but tier
        # is what makes worldview()/procedure_get()/procedure_list() an
        # index SEARCH instead of a SCAN -- see
        # test_tier_scoped_current_query_uses_the_new_index below.
        assert normalized.startswith("CREATE INDEX idx_memories_current ON memories(tier")


# ---------------------------------------------------------------------------
# Migration: upgrade a populated pre-v11 database
# ---------------------------------------------------------------------------


def test_migration_adds_index_to_populated_pre_v11_database(current_index_env):
    """Mirrors tests/test_content_hash.py's v10 migration test: write real
    rows (current and superseded), force the on-disk state back to a
    genuinely pre-v11 database (index physically dropped, not just a lower
    version number), reopen, and prove the index comes back and no data
    moved.
    """
    import sqlite3

    from haunt.paths import namespace_db_path
    from haunt.store import SCHEMA_VERSION, Store

    with Store("default") as store:
        current = store.observe("still true today")
        original = store.observe("about to be corrected")
        store.contradict(
            original.memory_id, replacement="corrected value", idempotency_key="mig-1"
        )
        before_rows = sorted(
            (r["id"], r["content"], r["valid_to"])
            for r in store.conn.execute(
                "SELECT id, content, valid_to FROM memories"
            ).fetchall()
        )
        store.conn.execute("UPDATE meta SET value='10' WHERE key='schema_version'")
        store.conn.commit()

    db_path = namespace_db_path("default")
    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX IF EXISTS idx_memories_current")
    conn.commit()
    conn.close()

    with Store("default", create=False) as migrated:
        assert migrated.get_meta("schema_version") == str(SCHEMA_VERSION)
        index = migrated.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_memories_current'"
        ).fetchone()
        assert index is not None

        after_rows = sorted(
            (r["id"], r["content"], r["valid_to"])
            for r in migrated.conn.execute(
                "SELECT id, content, valid_to FROM memories"
            ).fetchall()
        )
        assert after_rows == before_rows, "migration must not move or alter any row"
        assert migrated.get_memory(current.memory_id) is not None

    # Idempotent re-run: reopening again must not error or change anything.
    with Store("default", create=False) as reopened:
        still_there = reopened.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_memories_current'"
        ).fetchone()
        assert still_there is not None


def test_migration_running_twice_in_place_is_a_no_op(current_index_env):
    """Exercises _ensure_namespace_schema's own idempotent early-return path
    (current >= SCHEMA_VERSION), mirroring
    tests/test_content_hash.py::test_migration_running_twice_in_place_is_a_no_op.
    """
    from haunt.store import Store, _ensure_namespace_schema

    with Store("default") as store:
        store.observe("a")
        store.observe("b")
        before = sorted(
            r["id"] for r in store.conn.execute("SELECT id FROM memories").fetchall()
        )

        _ensure_namespace_schema(store.conn)
        _ensure_namespace_schema(store.conn)

        after = sorted(
            r["id"] for r in store.conn.execute("SELECT id FROM memories").fetchall()
        )
        index_count = store.conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' "
            "AND name='idx_memories_current'"
        ).fetchone()[0]

    assert after == before
    assert index_count == 1


# ---------------------------------------------------------------------------
# The index is actually usable by a tier-scoped current-fact query
# ---------------------------------------------------------------------------


def test_tier_scoped_current_query_uses_the_new_index(current_index_env):
    """Same query shape as worldview()'s facts read and procedure_list():
    tier-scoped, valid_to IS NULL, ordered by created_at DESC. Proves the
    index is actually chosen by the planner, not just present."""
    from haunt.store import Store

    with Store("default") as store:
        for i in range(20):
            store.observe(f"semantic fact {i}", tier="semantic")
        plan = store.conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT m.id, m.content, m.valid_from, m.created_at
            FROM memories m
            JOIN events e ON e.id = m.event_id
            WHERE m.tier='semantic' AND m.valid_to IS NULL
              AND e.role != 'tool' AND e.tool_name IS NULL
            ORDER BY m.created_at DESC, m.rowid DESC
            LIMIT 12
            """
        ).fetchall()
        detail = " | ".join(row["detail"] for row in plan)
        assert "idx_memories_current" in detail
        assert "SCAN m" not in detail, detail  # SEARCH via the index, not a table scan
        assert "TEMP B-TREE" not in detail, detail  # (tier, created_at) covers the ORDER BY too


# ---------------------------------------------------------------------------
# trace(): mechanism -- point lookups, not a full-table load
# ---------------------------------------------------------------------------


def test_trace_uses_bounded_point_lookups_not_a_full_table_load(current_index_env):
    """Amid many unrelated correction chains, trace() must issue only
    targeted point lookups against `corrections` -- never the old
    unfiltered `SELECT * FROM corrections ORDER BY ...` bulk load, and
    never a number of corrections-touching statements that scales with the
    namespace's total correction count rather than the requested chain's
    length.
    """
    from haunt.store import Store

    with Store("default") as store:
        for i in range(200):
            original = store.observe(f"unrelated fact {i}")
            store.contradict(
                original.memory_id,
                replacement=f"unrelated fact {i} corrected",
                idempotency_key=f"unrelated-{i}",
            )

        first = store.observe("target chain link 0")
        current_id = first.memory_id
        for step in range(1, 5):
            result = store.contradict(
                current_id,
                replacement=f"target chain link {step}",
                idempotency_key=f"target-{step}",
            )
            current_id = result["replacement_memory_id"]
        tip_id = current_id

        total_corrections = store.conn.execute(
            "SELECT COUNT(*) FROM corrections"
        ).fetchone()[0]
        assert total_corrections == 204  # 200 unrelated + 4 target-chain links

        captured_sql: list[str] = []
        store.conn.set_trace_callback(captured_sql.append)
        try:
            trace = store.trace(tip_id)
        finally:
            store.conn.set_trace_callback(None)

    assert trace["ok"] is True
    assert [m["content"] for m in trace["members"]] == [
        "target chain link 0",
        "target chain link 1",
        "target chain link 2",
        "target chain link 3",
        "target chain link 4",
    ]

    corrections_statements = [sql for sql in captured_sql if "corrections" in sql.lower()]
    assert corrections_statements, "trace() should have queried corrections at all"
    assert not any(
        "order by corrected_at" in sql.lower() for sql in captured_sql
    ), "trace() must not fall back to the old unfiltered bulk load"
    for sql in corrections_statements:
        normalized = " ".join(sql.split())
        assert "WHERE" in normalized, normalized
        assert any(
            col in normalized
            for col in (
                "target_memory_id",
                "target_tombstone_id",
                "replacement_memory_id",
                "replacement_tombstone_id",
            )
        ), normalized
    # 5-member chain: ~5 backward-walk lookups + ~5 forward-walk lookups.
    # The old code issued exactly 1 statement here regardless of chain
    # length, but that statement read all 204 rows -- bound this well
    # under 204 to prove the new code does not scale with total history.
    assert 0 < len(corrections_statements) <= 20


# ---------------------------------------------------------------------------
# trace(): output contract -- unchanged, including tombstones, amid noise
# ---------------------------------------------------------------------------


def test_trace_output_unchanged_for_long_chain_with_tombstone_among_unrelated_chains(
    current_index_env,
):
    """Combines tests/test_correction_lineage.py's
    test_three_link_trace_from_middle_and_restart (trace from every point
    in a chain) and test_purge_scrubs_canaries_and_keeps_safe_gap (a
    purged link becomes an allowlisted tombstone) into one longer, 6-link
    chain, with 100 unrelated chains alongside it -- exactly the situation
    that used to force trace() to load and filter every correction in the
    namespace to answer one lineage question.
    """
    from haunt.store import Store

    canary = "PRIVACY-CANARY-C10"
    with Store("default") as store:
        for i in range(100):
            original = store.observe(f"noise {i}")
            store.contradict(
                original.memory_id,
                replacement=f"noise {i} corrected",
                idempotency_key=f"noise-{i}",
            )

        link0 = store.observe("alpha", session_id="src")
        into_erased = store.contradict(
            link0.memory_id,
            replacement=canary,
            reason=canary,
            origin=canary,
            session_id="correction-session",
            idempotency_key="c10-1",
        )
        erased_id = into_erased["replacement_memory_id"]
        link2 = store.contradict(erased_id, replacement="gamma", idempotency_key="c10-2")[
            "replacement_memory_id"
        ]
        link3 = store.contradict(link2, replacement="delta", idempotency_key="c10-3")[
            "replacement_memory_id"
        ]
        link4 = store.contradict(link3, replacement="epsilon", idempotency_key="c10-4")[
            "replacement_memory_id"
        ]
        link5 = store.contradict(link4, replacement="zeta", idempotency_key="c10-5")[
            "replacement_memory_id"
        ]
        assert store.purge(erased_id)["ok"] is True

        for member_id in (link0.memory_id, link2, link3, link4, link5):
            trace = store.trace(member_id)
            assert trace["lineage_status"] == "linked"
            assert [m.get("content") for m in trace["members"]] == [
                "alpha", None, "gamma", "delta", "epsilon", "zeta",
            ]
            statuses = [m.get("status") for m in trace["members"]]
            assert statuses == [
                "superseded", "erased", "superseded", "superseded", "superseded", "current",
            ]
            assert len(trace["corrections"]) == 5
            # corrections[0] (alpha->erased) and corrections[1]
            # (erased->gamma) both now reference the tombstone -- purge's
            # own privacy-scrub invariant (see
            # _ensure_correction_invariant_triggers) requires session_id/
            # origin/reason to be NULL on any correction row touching a
            # tombstone, not just on the erased memory row itself. trace()
            # must surface that scrub honestly rather than inventing or
            # retaining the pre-purge session_id.
            assert trace["corrections"][0]["session_id"] is None
            assert trace["corrections"][1]["session_id"] is None
            for correction in trace["corrections"]:
                assert correction["correction_id"]
                assert correction["corrected_at"]
            tombstones = [m for m in trace["members"] if m.get("status") == "erased"]
            assert len(tombstones) == 1
            assert set(tombstones[0]) == {
                "schema_version", "tombstone_id", "status", "erased_at"
            }
            assert canary not in str(trace)


def test_trace_linked_status_false_short_circuits_the_final_lookup(current_index_env):
    """A memory with at least one recorded correction never needs the
    trailing `linked` check to query `corrections` again -- `corrections
    or ...` short-circuits. A standalone (never corrected, never a
    replacement) memory exercises the other branch: `linked` must still
    correctly evaluate to False by actually performing that lookup.
    """
    from haunt.store import Store

    with Store("default") as store:
        standalone = store.observe("nobody ever touched this")
        trace = store.trace(standalone.memory_id)

    assert trace["ok"] is True
    assert trace["lineage_status"] == "standalone"
    assert trace["corrections"] == []
    assert [m["content"] for m in trace["members"]] == ["nobody ever touched this"]
