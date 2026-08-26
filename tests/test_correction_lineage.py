"""Append-only correction lineage, replay, migration, and erasure evidence."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest


@pytest.fixture
def lineage_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunt"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed

    embed.reset()
    yield tmp_path / "haunt"
    embed.reset()


def _assert_tokens_absent_from_tables(store, tokens):
    for table_row in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        table = table_row["name"]
        for row in store.conn.execute(f'SELECT * FROM "{table}"').fetchall():
            for token in tokens:
                for value in row:
                    if isinstance(value, bytes):
                        assert token.encode("utf-8") not in value, table
                    else:
                        assert token not in json.dumps(value, ensure_ascii=False), table


def _assert_tokens_absent_from_payload(payload, tokens):
    serialized = json.dumps(payload, ensure_ascii=False)
    for token in tokens:
        assert token not in serialized


def _all_table_snapshot(store):
    snapshot = {}
    for table_row in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        table = table_row["name"]
        rows = [tuple(row) for row in store.conn.execute(f'SELECT * FROM "{table}"')]
        snapshot[table] = sorted(rows, key=repr)
    return snapshot


def test_v3_migration_is_additive_idempotent_and_survives_restart(lineage_env):
    from haunt.store import SCHEMA_VERSION, Store

    with Store("default") as st:
        original = st.observe("legacy database memory")
        st.conn.execute("DROP TABLE corrections")
        st.conn.execute("DROP TABLE lineage_tombstones")
        st.conn.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
        st.conn.commit()

    with Store("default") as st:
        assert st.get_meta("schema_version") == str(SCHEMA_VERSION)
        result = st.contradict(
            original.memory_id,
            replacement="replacement after migration",
            idempotency_key="migration-key",
        )
        replacement = result["replacement_memory_id"]

    with Store("default") as st:
        trace = st.trace(replacement)
        assert [m.get("content") for m in trace["members"]] == [
            "legacy database memory",
            "replacement after migration",
        ]
        assert st.get_meta("schema_version") == str(SCHEMA_VERSION)


def test_v5_upgrade_installs_correction_invariants_and_survives_restart(lineage_env):
    from haunt.store import SCHEMA_VERSION, Store
    from haunt.util import new_id, now_iso

    with Store("default") as st:
        target = st.observe("schema upgrade target")
        result = st.contradict(
            target.memory_id,
            replacement="schema upgrade replacement",
            idempotency_key="schema-upgrade-valid",
        )
        st.conn.execute("DROP TRIGGER corrections_invariant_insert")
        st.conn.execute("DROP TRIGGER corrections_invariant_update")
        st.conn.execute("UPDATE meta SET value='5' WHERE key='schema_version'")
        st.conn.commit()

    with Store("default") as st:
        assert st.get_meta("schema_version") == str(SCHEMA_VERSION)
        assert st.trace(result["replacement_memory_id"])["lineage_status"] == "linked"
        triggers = {
            row["name"]
            for row in st.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        assert {
            "corrections_invariant_insert",
            "corrections_invariant_update",
        } <= triggers
        with pytest.raises(sqlite3.IntegrityError, match="correction invariant"):
            st.conn.execute(
                """
                INSERT INTO corrections(
                    id, target_memory_id, corrected_at, origin, session_id,
                    idempotency_key, request_identity, request_payload, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id(),
                    new_id(),
                    now_iso(),
                    "test",
                    "session",
                    None,
                    "sha256:" + "0" * 64,
                    b"payload",
                    "{}",
                ),
            )
def test_three_link_trace_from_middle_and_restart(lineage_env):
    from haunt.store import Store

    with Store("default") as st:
        first = st.observe("one", session_id="source-session")
        first_correction = st.contradict(
            first.memory_id,
            replacement="two",
            reason="first reason",
            origin="unit-test",
            session_id="correction-session",
            idempotency_key="three-1",
        )
        middle = first_correction["replacement_memory_id"]
        last_correction = st.contradict(
            middle, replacement="three", idempotency_key="three-2"
        )
        last = last_correction["replacement_memory_id"]

    with Store("default") as st:
        for member_id in (first.memory_id, middle, last):
            trace = st.trace(member_id)
            assert trace["lineage_status"] == "linked"
            assert [m["content"] for m in trace["members"]] == ["one", "two", "three"]
            assert len(trace["corrections"]) == 2
            assert trace["members"][0]["event_id"] == first.event_id
            assert trace["corrections"][0]["session_id"] == "correction-session"


def test_correction_without_replacement_and_legacy_unlinked(lineage_env):
    from haunt.store import Store
    from haunt.util import now_iso

    with Store("default") as st:
        linked = st.observe("withdraw this")
        result = st.contradict(linked.memory_id, reason="withdrawn", idempotency_key="none")
        trace = st.trace(linked.memory_id)
        assert result["ok"] is True
        assert len(trace["members"]) == 1
        assert trace["members"][0]["status"] == "superseded"
        assert trace["corrections"][0]["reason"] == "withdrawn"

        legacy = st.observe("old valid_to only")
        st.conn.execute("UPDATE memories SET valid_to=? WHERE id=?", (now_iso(), legacy.memory_id))
        st.conn.commit()
        legacy_trace = st.trace(legacy.memory_id)
        assert legacy_trace["lineage_status"] == "legacy_unlinked"
        assert legacy_trace["members"][0]["status"] == "legacy_unlinked"
        assert legacy_trace["corrections"] == []


def test_replay_and_conflict_before_and_after_restart(lineage_env):
    from haunt.store import Store

    with Store("default") as st:
        original = st.observe("before")
        first = st.contradict(
            original.memory_id,
            replacement=" exact replacement ",
            reason=" exact reason ",
            idempotency_key="retry-key",
        )
        changes_before_replay = st.conn.total_changes
        replay = st.contradict(
            original.memory_id,
            replacement=" exact replacement ",
            reason=" exact reason ",
            idempotency_key="retry-key",
        )
        assert replay == {**first, "deduplicated": True}
        assert st.conn.total_changes == changes_before_replay
        assert st.conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0] == 1
        assert st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2

    with Store("default") as st:
        replay = st.contradict(
            original.memory_id,
            replacement=" exact replacement ",
            reason=" exact reason ",
            idempotency_key="retry-key",
        )
        assert replay["correction_id"] == first["correction_id"]
        assert replay["deduplicated"] is True
        conflict = st.contradict(
            original.memory_id,
            replacement="exact replacement",
            reason=" exact reason ",
            idempotency_key="retry-key",
        )
        assert conflict["conflict"] == "idempotency_key_reused"
        assert st.conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0] == 1


def test_canonical_payload_distinguishes_null_empty_whitespace_and_reason(lineage_env):
    from haunt.store import Store

    with Store("default") as st:
        null_target = st.observe("null target")
        first = st.contradict(
            null_target.memory_id,
            replacement=None,
            reason=None,
            origin="first-origin",
            session_id="first-session",
            idempotency_key="canonical-null",
        )
        changes = st.conn.total_changes
        replay = st.contradict(
            null_target.memory_id,
            replacement=None,
            reason=None,
            origin="",
            session_id=object(),
            idempotency_key="canonical-null",
        )
        assert replay == {**first, "deduplicated": True}
        assert st.conn.total_changes == changes

        conflict = st.contradict(
            null_target.memory_id,
            replacement="",
            reason=None,
            origin="",
            session_id=object(),
            idempotency_key="canonical-null",
        )
        assert conflict["conflict"] == "idempotency_key_reused"

        reason_conflict = st.contradict(
            null_target.memory_id,
            replacement=None,
            reason="",
            origin="",
            session_id=object(),
            idempotency_key="canonical-null",
        )
        assert reason_conflict["conflict"] == "idempotency_key_reused"

        empty_target = st.observe("empty target")
        empty = st.contradict(
            empty_target.memory_id,
            replacement="",
            reason="",
            idempotency_key="canonical-empty",
        )
        whitespace_target = st.observe("whitespace target")
        whitespace = st.contradict(
            whitespace_target.memory_id,
            replacement="   ",
            idempotency_key="canonical-whitespace",
        )
        assert st.get_memory(empty["replacement_memory_id"])["content"] == ""
        assert st.get_memory(whitespace["replacement_memory_id"])["content"] == "   "
        assert st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 5

        invalid_target = st.observe("invalid metadata target")
        with pytest.raises(ValueError, match="origin"):
            st.contradict(
                invalid_target.memory_id,
                origin="",
                idempotency_key="new-invalid-origin",
            )
        with pytest.raises(ValueError, match="session_id"):
            st.contradict(
                invalid_target.memory_id,
                session_id=object(),
                idempotency_key="new-invalid-session",
            )
        assert st.get_memory(invalid_target.memory_id)["valid_to"] is None


def test_canonical_payload_preserves_unicode_nul_and_newline_identity(lineage_env):
    from haunt.store import Store

    nfc = "caf\u00e9\nline\x00end"
    nfd = "cafe\u0301\nline\x00end"
    reason = "reason\x00with\ncontrols"
    with Store("default") as st:
        first_target = st.observe("unicode target one")
        first = st.contradict(
            first_target.memory_id,
            replacement=nfc,
            reason=reason,
            idempotency_key="unicode-exact",
        )
        replay = st.contradict(
            first_target.memory_id,
            replacement=nfc,
            reason=reason,
            idempotency_key="unicode-exact",
        )
        conflict = st.contradict(
            first_target.memory_id,
            replacement=nfd,
            reason=reason,
            idempotency_key="unicode-exact",
        )
        second_target = st.observe("unicode target two")
        second = st.contradict(
            second_target.memory_id,
            replacement=nfd,
            reason=reason,
            idempotency_key="unicode-decomposed",
        )

        assert replay == {**first, "deduplicated": True}
        assert conflict["conflict"] == "idempotency_key_reused"
        assert st.get_memory(first["replacement_memory_id"])["content"] == nfc
        assert st.get_memory(second["replacement_memory_id"])["content"] == nfd
        assert nfc != nfd


@pytest.mark.parametrize(
    ("same_key", "same_payload", "expected_conflict"),
    [
        (True, True, None),
        (True, False, "idempotency_key_reused"),
        (False, False, "already_superseded"),
    ],
)
def test_concurrent_corrections_do_not_fork(
    lineage_env, same_key, same_payload, expected_conflict
):
    from haunt.store import Store

    with Store("default") as st:
        original = st.observe("race target")
    barrier = Barrier(2)

    def correct(index):
        barrier.wait()
        with Store("default", create=False) as worker:
            return worker.contradict(
                original.memory_id,
                replacement="same" if same_payload else f"value-{index}",
                idempotency_key="shared" if same_key else f"key-{index}",
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(correct, (1, 2)))

    with Store("default") as st:
        assert st.conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0] == 1
        assert st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
    if expected_conflict is None:
        assert all(r["ok"] for r in results)
        assert sorted(r["deduplicated"] for r in results) == [False, True]
        assert len({r["correction_id"] for r in results}) == 1
    else:
        assert sum(bool(r["ok"]) for r in results) == 1
        assert next(r for r in results if not r["ok"])["conflict"] == expected_conflict


def test_replacement_failure_rolls_back_projection_and_lineage(lineage_env, monkeypatch):
    from haunt.store import Store

    with Store("default") as st:
        original = st.observe("atomic original")

        def fail(*args, **kwargs):
            raise RuntimeError("injected replacement failure")

        monkeypatch.setattr(Store, "observe", fail)
        with pytest.raises(RuntimeError, match="injected replacement"):
            st.contradict(
                original.memory_id,
                replacement="must not commit",
                idempotency_key="rollback",
            )
        assert st.conn.execute(
            "SELECT valid_to FROM memories WHERE id=?", (original.memory_id,)
        ).fetchone()[0] is None
        assert st.conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0] == 0
        assert st.trace(original.memory_id)["lineage_status"] == "standalone"


def test_ineligible_corrections_do_not_touch_embedding_or_job_state(
    lineage_env, monkeypatch
):
    import haunt.store as store_module
    from haunt.store import Store

    with Store("default") as st:
        already = st.observe("already superseded target")
        st.contradict(already.memory_id, idempotency_key="eligible-first")
        keyed = st.observe("idempotency conflict target")
        st.contradict(keyed.memory_id, idempotency_key="eligibility-conflict")
        pending = st.observe("pending embedding remains", defer_embedding=True)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("embedding work ran before correction eligibility")

        monkeypatch.setattr(Store, "ensure_current_embeddings", forbidden)
        monkeypatch.setattr(Store, "process_embedding_jobs", forbidden)
        monkeypatch.setattr(store_module, "embed_one", forbidden)

        before = _all_table_snapshot(st)
        missing = st.contradict(
            "missing-correction-target",
            replacement="must not embed",
            idempotency_key="eligibility-missing",
        )
        assert missing["ok"] is False and "not found" in missing["error"]
        assert _all_table_snapshot(st) == before

        superseded = st.contradict(
            already.memory_id,
            replacement="must not embed",
            idempotency_key="eligibility-already",
        )
        assert superseded["conflict"] == "already_superseded"
        assert _all_table_snapshot(st) == before

        conflict = st.contradict(
            keyed.memory_id,
            replacement="different canonical payload",
            idempotency_key="eligibility-conflict",
        )
        assert conflict["conflict"] == "idempotency_key_reused"
        assert _all_table_snapshot(st) == before
        assert st.conn.execute(
            "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (pending.memory_id,)
        ).fetchone() is not None


def test_replacement_failure_rolls_back_fts_jobs_sessions_and_projection(
    lineage_env, monkeypatch
):
    import haunt.graph as graph
    import haunt.store as store_module
    from haunt.store import Store

    with Store("default") as st:
        original = st.observe("deep rollback original")
        before = _all_table_snapshot(st)

        monkeypatch.setattr(store_module, "embed_one", lambda _text: None)

        def fail_after_derived_writes(*_args, **_kwargs):
            raise RuntimeError("failure after derived writes")

        monkeypatch.setattr(graph, "extract_and_store", fail_after_derived_writes)
        with pytest.raises(RuntimeError, match="after derived writes"):
            st.contradict(
                original.memory_id,
                replacement="derived rollback replacement canary",
                origin="derived-rollback-origin",
                session_id="derived-rollback-session",
                idempotency_key="derived-rollback-key",
            )

        assert _all_table_snapshot(st) == before
        assert st.get_memory(original.memory_id)["valid_to"] is None


def test_current_and_as_of_recall_follow_projection(lineage_env):
    from haunt.recall import recall
    from haunt.store import Store

    with Store("default") as st:
        old = st.observe("temporal shared-token old", valid_from="2025-01-01T00:00:00Z")
        result = st.contradict(
            old.memory_id,
            replacement="temporal shared-token new",
            idempotency_key="temporal",
        )
        corrected_at = result["valid_to"]
    current = recall("shared-token", namespace="default", k=8)
    historical = recall(
        "shared-token", namespace="default", as_of="2025-06-01T00:00:00Z", k=8
    )
    exact = recall("shared-token", namespace="default", as_of=corrected_at, k=8)
    assert [h.memory_id for h in current] == [result["replacement_memory_id"]]
    assert [h.memory_id for h in historical] == [old.memory_id]
    assert [h.memory_id for h in exact] == [result["replacement_memory_id"]]


@pytest.mark.parametrize("derived_table", ["memories_fts", "vec_memories"])
def test_purge_derived_delete_failure_rolls_back_everything(
    lineage_env, derived_table
):
    from haunt.store import Store

    canary = f"PURGE-{derived_table}-ROLLBACK-CANARY"
    with Store("default") as st:
        target = st.observe(
            canary,
            origin=canary + "-ORIGIN",
            session_id=canary + "-SESSION",
        )
        if derived_table == "vec_memories":
            st.conn.execute(
                "CREATE TABLE vec_memories(id TEXT PRIMARY KEY, embedding BLOB)"
            )
            st.conn.execute(
                "INSERT INTO vec_memories(id, embedding) VALUES (?, ?)",
                (target.memory_id, canary.encode("utf-8")),
            )
            st.conn.commit()
        before = _all_table_snapshot(st)

        def deny_delete(action, arg1, _arg2, _db, _trigger):
            if action == sqlite3.SQLITE_DELETE and arg1 == derived_table:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        st.conn.set_authorizer(deny_delete)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                st.purge(target.memory_id)
        finally:
            st.conn.set_authorizer(None)

        assert _all_table_snapshot(st) == before
        assert st.get_memory(target.memory_id)["content"] == canary
        assert st.conn.execute(
            f'SELECT 1 FROM "{derived_table}" WHERE id=?', (target.memory_id,)
        ).fetchone() is not None
        assert st.conn.execute(
            "SELECT COUNT(*) FROM lineage_tombstones"
        ).fetchone()[0] == 0


def test_purge_allows_deliberately_missing_derived_tables(lineage_env):
    from haunt.store import Store

    with Store("default") as st:
        target = st.observe("missing derived tables purge")
        st.conn.execute("DROP TABLE memories_fts")
        st.conn.commit()
        result = st.purge(target.memory_id)
        assert result["ok"] is True
        assert result["fts_deleted"] is False
        assert result["vec_deleted"] is False
        assert st.get_memory(target.memory_id) is None


def test_purge_sanitizes_shared_event_and_preserves_survivor_on_all_surfaces(
    lineage_env, monkeypatch
):
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    from haunt.mcp_server import memory_trace
    from haunt.store import PURGE_SAFE_ORIGIN, Store
    from haunt.util import new_id, now_iso
    from tests.dashutil import make_dash_client

    tokens = {
        "content": "SHARED-EVENT-TARGET-CONTENT-CANARY",
        "origin": "SHARED-EVENT-TARGET-ORIGIN-CANARY",
        "session": "SHARED-EVENT-TARGET-SESSION-CANARY",
        "event_key": "SHARED-EVENT-IDEMPOTENCY-CANARY",
        "tool_name": "SHARED-EVENT-TOOL-NAME-CANARY",
        "tool_input": "SHARED-EVENT-TOOL-INPUT-CANARY",
        "tool_output": "SHARED-EVENT-TOOL-OUTPUT-CANARY",
        "event_meta": "SHARED-EVENT-META-CANARY",
    }
    survivor_content = "SHARED-EVENT-SURVIVOR-CONTENT"
    with Store("default") as st:
        target = st.observe(
            tokens["content"],
            role="tool",
            tier="semantic",
            session_id=tokens["session"],
            origin=tokens["origin"],
            idempotency_key=tokens["event_key"],
            tool_name=tokens["tool_name"],
            tool_input=tokens["tool_input"],
            tool_output=tokens["tool_output"],
            meta={"source": tokens["event_meta"]},
        )
        survivor_id = new_id()
        ts = now_iso()
        st.conn.execute(
            """
            INSERT INTO memories(
                id, event_id, tier, content, embedding,
                valid_from, valid_to, created_at
            ) VALUES (?, ?, 'semantic', ?, NULL, ?, NULL, ?)
            """,
            (survivor_id, target.event_id, survivor_content, ts, ts),
        )
        st.conn.execute(
            "INSERT INTO memories_fts(id, content) VALUES (?, ?)",
            (survivor_id, survivor_content),
        )
        st.conn.execute(
            "INSERT INTO embedding_jobs(memory_id, queued_at) VALUES (?, ?)",
            (survivor_id, ts),
        )
        st.set_meta("current_session", tokens["session"])

        result = st.purge(target.memory_id)

        assert result["ok"] is True
        assert result["event_deleted"] is True
        survivor = st.get_memory(survivor_id)
        assert survivor["content"] == survivor_content
        assert survivor["event_id"] != target.event_id
        assert survivor["origin"] == PURGE_SAFE_ORIGIN
        assert survivor["session_id"] != tokens["session"]
        event = st.conn.execute(
            """
            SELECT idempotency_key, content, tool_name, tool_input,
                   tool_output, origin, meta, session_id
            FROM events WHERE id=?
            """,
            (survivor["event_id"],),
        ).fetchone()
        assert tuple(event) == (
            None,
            "",
            None,
            None,
            None,
            PURGE_SAFE_ORIGIN,
            "{}",
            survivor["session_id"],
        )
        assert st.get_meta("current_session") == survivor["session_id"]
        assert st.conn.execute(
            "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (survivor_id,)
        ).fetchone() is not None
        absent = tuple(tokens.values()) + (target.memory_id, target.event_id)
        _assert_tokens_absent_from_tables(st, absent)
        _assert_tokens_absent_from_payload(st.trace(survivor_id), absent)

    detail = make_dash_client().get(
        f"/api/namespace/default/memory/{survivor_id}"
    )
    assert detail.status_code == 200
    _assert_tokens_absent_from_payload(detail.json(), absent)
    _assert_tokens_absent_from_payload(
        json.loads(memory_trace(survivor_id, namespace="default")), absent
    )


def test_purge_scrubs_canaries_and_keeps_safe_gap(lineage_env):
    from haunt.store import Store

    canary = "ERASE-ME-PRIVATE-CANARY"
    with Store("default") as st:
        first = st.observe("surviving first")
        into_erased = st.contradict(
            first.memory_id,
            replacement=canary,
            reason=canary,
            origin=canary,
            session_id=canary,
            idempotency_key=canary,
        )
        erased_id = into_erased["replacement_memory_id"]
        erased_event = into_erased["replacement_event_id"]
        out = st.contradict(
            erased_id,
            replacement="surviving last",
            reason=canary,
            origin=canary,
            session_id=canary,
            idempotency_key=canary + "-2",
        )
        last = out["replacement_memory_id"]
        st.purge(erased_id)
        trace = st.trace(last)

        tombstones = [m for m in trace["members"] if m.get("status") == "erased"]
        assert len(tombstones) == 1
        assert set(tombstones[0]) == {
            "schema_version", "tombstone_id", "status", "erased_at"
        }
        assert [m.get("content") for m in trace["members"]] == [
            "surviving first", None, "surviving last"
        ]
        serialized_trace = json.dumps(trace)
        assert canary not in serialized_trace
        assert erased_id not in serialized_trace
        assert erased_event not in serialized_trace

        _assert_tokens_absent_from_tables(st, (canary, erased_id, erased_event))


@pytest.mark.parametrize("purge_position", ["first", "middle", "last", "all"])
def test_purge_privacy_matrix_scans_tables_trace_and_api(
    lineage_env, monkeypatch, purge_position
):
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    from haunt.mcp_server import memory_trace
    from haunt.store import Store
    from tests.dashutil import make_dash_client

    tokens = {
        "a_content": "PURGE-A-CONTENT-CANARY",
        "a_origin": "PURGE-A-ORIGIN-CANARY",
        "a_session": "PURGE-A-SESSION-CANARY",
        "c1_origin": "PURGE-C1-ORIGIN-CANARY",
        "c1_session": "PURGE-C1-SESSION-CANARY",
        "c1_reason": "PURGE-C1-理由-CANARY",
        "c1_key": "PURGE-C1-IDEMPOTENCY-CANARY",
        "b_content": "PURGE-B-REPLACEMENT-CANARY",
        "c2_origin": "PURGE-C2-ORIGIN-CANARY",
        "c2_session": "PURGE-C2-SESSION-CANARY",
        "c2_reason": "PURGE-C2-REASON-CANARY",
        "c2_key": "PURGE-C2-IDEMPOTENCY-CANARY",
        "c_content": "PURGE-C-REPLACEMENT-CANARY",
    }
    with Store("default") as st:
        a = st.observe(
            tokens["a_content"],
            origin=tokens["a_origin"],
            session_id=tokens["a_session"],
        )
        c1 = st.contradict(
            a.memory_id,
            replacement=tokens["b_content"],
            origin=tokens["c1_origin"],
            session_id=tokens["c1_session"],
            reason=tokens["c1_reason"],
            idempotency_key=tokens["c1_key"],
        )
        b_id = c1["replacement_memory_id"]
        b_event = c1["replacement_event_id"]
        c2 = st.contradict(
            b_id,
            replacement=tokens["c_content"],
            origin=tokens["c2_origin"],
            session_id=tokens["c2_session"],
            reason=tokens["c2_reason"],
            idempotency_key=tokens["c2_key"],
        )
        c_id = c2["replacement_memory_id"]
        c_event = c2["replacement_event_id"]

        if purge_position == "first":
            st.purge(a.memory_id)
            surviving_id = c_id
            absent = (
                tokens["a_content"], tokens["a_origin"], tokens["a_session"],
                tokens["c1_origin"], tokens["c1_session"], tokens["c1_reason"],
                tokens["c1_key"], a.memory_id, a.event_id,
            )
        elif purge_position == "middle":
            st.purge(b_id)
            surviving_id = c_id
            absent = (
                tokens["b_content"], tokens["c1_origin"], tokens["c1_session"],
                tokens["c1_reason"], tokens["c1_key"], tokens["c2_origin"],
                tokens["c2_session"], tokens["c2_reason"], tokens["c2_key"],
                b_id, b_event,
            )
        elif purge_position == "last":
            st.purge(c_id)
            surviving_id = b_id
            absent = (
                tokens["c_content"], tokens["c2_origin"], tokens["c2_session"],
                tokens["c2_reason"], tokens["c2_key"], c_id, c_event,
            )
        else:
            st.purge(a.memory_id)
            st.purge(b_id)
            st.purge(c_id)
            surviving_id = None
            absent = tuple(tokens.values()) + (
                a.memory_id, a.event_id, b_id, b_event, c_id, c_event,
            )

        _assert_tokens_absent_from_tables(st, absent)
        store_trace = st.trace(surviving_id) if surviving_id else st.trace(c_id)
        _assert_tokens_absent_from_payload(store_trace, absent)

    client = make_dash_client()
    detail_id = surviving_id if surviving_id else c_id
    detail_response = client.get(f"/api/namespace/default/memory/{detail_id}")
    _assert_tokens_absent_from_payload(detail_response.json(), absent)
    _assert_tokens_absent_from_payload(
        json.loads(memory_trace(detail_id, namespace="default")), absent
    )


def test_purge_sanitizes_shared_correction_session_without_deleting_unrelated(
    lineage_env, monkeypatch
):
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    from haunt.mcp_server import memory_trace
    from haunt.store import PURGE_SAFE_SESSION_SOURCE, Store
    from tests.dashutil import make_dash_client

    canary = "SHARED-CORRECTION-CONTEXT-CANARY"
    shared_session = canary + "-SESSION-ID"
    unrelated_content = "UNRELATED-SHARED-SESSION-CONTENT-SURVIVES"
    with Store("default") as st:
        original = st.observe("shared session purge target")
        correction = st.contradict(
            original.memory_id,
            replacement="surviving replacement content",
            origin=canary,
            session_id=shared_session,
            reason=canary,
            idempotency_key=canary,
        )
        replacement_id = correction["replacement_memory_id"]
        unrelated = st.observe(
            unrelated_content,
            origin="safe-unrelated-origin",
            session_id=shared_session,
        )
        st.conn.execute(
            "UPDATE sessions SET meta=? WHERE id=?",
            (
                json.dumps(
                    {
                        "correction_context": canary,
                        "keep": "unrelated-session-metadata",
                    }
                ),
                shared_session,
            ),
        )
        st.set_meta("current_session", shared_session)
        old_session = st.conn.execute(
            "SELECT started_at, ended_at FROM sessions WHERE id=?",
            (shared_session,),
        ).fetchone()

        st.purge(original.memory_id)

        replacement = st.get_memory(replacement_id)
        unrelated_after = st.get_memory(unrelated.memory_id)
        assert replacement["origin"] == PURGE_SAFE_SESSION_SOURCE
        assert replacement["session_id"] != shared_session
        assert unrelated_after["content"] == unrelated_content
        assert unrelated_after["origin"] == "safe-unrelated-origin"
        assert unrelated_after["session_id"] == replacement["session_id"]
        assert st.conn.execute(
            "SELECT 1 FROM sessions WHERE id=?", (shared_session,)
        ).fetchone() is None
        session = st.conn.execute(
            "SELECT started_at, ended_at, source, meta FROM sessions WHERE id=?",
            (replacement["session_id"],),
        ).fetchone()
        assert session["started_at"] == old_session["started_at"]
        assert session["ended_at"] == old_session["ended_at"]
        assert session["source"] == PURGE_SAFE_SESSION_SOURCE
        assert json.loads(session["meta"]) == {
            "keep": "unrelated-session-metadata"
        }
        assert st.get_meta("current_session") == replacement["session_id"]
        absent = (canary, original.memory_id, original.event_id)
        _assert_tokens_absent_from_tables(st, absent)
        _assert_tokens_absent_from_payload(st.trace(replacement_id), absent)

    detail = make_dash_client().get(
        f"/api/namespace/default/memory/{replacement_id}"
    )
    assert detail.status_code == 200
    _assert_tokens_absent_from_payload(detail.json(), absent)
    _assert_tokens_absent_from_payload(
        json.loads(memory_trace(replacement_id, namespace="default")), absent
    )


def test_purge_rekeys_preexisting_shared_correction_session_and_preserves_clean_bytes(
    lineage_env,
):
    from haunt.store import PURGE_SAFE_ORIGIN, Store

    session_id = "preexisting-shared-session"
    session_origin = "safe-client-origin"
    session_meta = '{  "keep": "safe client metadata", "nested": [1, 2]  }'
    with Store("default") as st:
        unrelated = st.observe(
            "preexisting unrelated event survives",
            origin=session_origin,
            session_id=session_id,
        )
        st.conn.execute(
            "UPDATE sessions SET meta=? WHERE id=?",
            (session_meta, session_id),
        )
        original = st.observe("a")
        correction = st.contradict(
            original.memory_id,
            replacement="replacement leaves preexisting session",
            origin="cli",
            session_id=session_id,
            idempotency_key="preexisting-session-correction",
        )
        st.set_meta("current_session", session_id)
        old_session = st.conn.execute(
            "SELECT started_at, ended_at FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        st.purge(original.memory_id)

        assert st.conn.execute(
            "SELECT 1 FROM sessions WHERE id=?", (session_id,)
        ).fetchone() is None
        unrelated_after = st.get_memory(unrelated.memory_id)
        replacement = st.get_memory(correction["replacement_memory_id"])
        assert unrelated_after["content"] == "preexisting unrelated event survives"
        assert unrelated_after["origin"] == session_origin
        assert unrelated_after["session_id"] != session_id
        assert replacement["origin"] == PURGE_SAFE_ORIGIN
        assert replacement["session_id"] == unrelated_after["session_id"]
        session = st.conn.execute(
            "SELECT started_at, ended_at, source, meta FROM sessions WHERE id=?",
            (replacement["session_id"],),
        ).fetchone()
        assert session["started_at"] == old_session["started_at"]
        assert session["ended_at"] == old_session["ended_at"]
        assert session["source"] == session_origin
        assert session["meta"] == session_meta
        assert st.get_meta("current_session") == replacement["session_id"]


def test_purge_rekeys_target_shared_session_and_scrubs_full_event_context(
    lineage_env, monkeypatch
):
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    from haunt.mcp_server import memory_trace
    from haunt.store import PURGE_SAFE_ORIGIN, PURGE_SAFE_SESSION_SOURCE, Store
    from tests.dashutil import make_dash_client

    tokens = {
        "content": "TARGET-CONTENT-PRIVACY-CANARY",
        "origin": "TARGET-ORIGIN-PRIVACY-CANARY",
        "session": "TARGET-SESSION-PRIVACY-CANARY",
        "event_key": "TARGET-EVENT-KEY-PRIVACY-CANARY",
        "tool_name": "TARGET-TOOL-NAME-PRIVACY-CANARY",
        "tool_input": "TARGET-TOOL-INPUT-PRIVACY-CANARY",
        "tool_output": "TARGET-TOOL-OUTPUT-PRIVACY-CANARY",
        "event_meta": "TARGET-EVENT-META-PRIVACY-CANARY",
        "correction_origin": "TARGET-CORRECTION-ORIGIN-CANARY",
        "correction_session": "TARGET-CORRECTION-SESSION-CANARY",
        "correction_reason": "TARGET-CORRECTION-REASON-CANARY",
        "correction_key": "TARGET-CORRECTION-KEY-CANARY",
    }
    clean_meta = "TARGET-SHARED-CLEAN-METADATA-SURVIVES"
    unrelated_content = "TARGET-SHARED-UNRELATED-CONTENT-SURVIVES"
    with Store("default") as st:
        target = st.observe(
            tokens["content"],
            role="tool",
            origin=tokens["origin"],
            session_id=tokens["session"],
            idempotency_key=tokens["event_key"],
            tool_name=tokens["tool_name"],
            tool_input=tokens["tool_input"],
            tool_output=tokens["tool_output"],
            meta={"provenance": tokens["event_meta"]},
        )
        unrelated = st.observe(
            unrelated_content,
            origin="safe-unrelated-target-origin",
            session_id=tokens["session"],
        )
        st.conn.execute(
            "UPDATE sessions SET meta=? WHERE id=?",
            (
                json.dumps(
                    {
                        "copied_tool_context": tokens["tool_input"],
                        "copied_event_meta": tokens["event_meta"],
                        "keep": clean_meta,
                    }
                ),
                tokens["session"],
            ),
        )
        old_target_session = st.conn.execute(
            "SELECT started_at, ended_at FROM sessions WHERE id=?",
            (tokens["session"],),
        ).fetchone()
        st.set_meta("current_session", tokens["session"])
        correction = st.contradict(
            target.memory_id,
            replacement="safe surviving target replacement",
            origin=tokens["correction_origin"],
            session_id=tokens["correction_session"],
            reason=tokens["correction_reason"],
            idempotency_key=tokens["correction_key"],
        )
        replacement_id = correction["replacement_memory_id"]
        request_payload = st.conn.execute(
            "SELECT request_payload FROM corrections WHERE id=?",
            (correction["correction_id"],),
        ).fetchone()["request_payload"]
        assert tokens["correction_reason"].encode("utf-8") in request_payload

        purge_result = st.purge(target.memory_id)

        assert "memory_id" not in purge_result
        assert "event_id" not in purge_result
        assert set(purge_result["lineage_tombstone"]) == {
            "schema_version",
            "tombstone_id",
            "status",
            "erased_at",
        }
        unrelated_after = st.get_memory(unrelated.memory_id)
        replacement = st.get_memory(replacement_id)
        assert unrelated_after["content"] == unrelated_content
        assert unrelated_after["origin"] == "safe-unrelated-target-origin"
        assert unrelated_after["session_id"] != tokens["session"]
        assert replacement["origin"] == PURGE_SAFE_ORIGIN
        assert replacement["session_id"] != tokens["correction_session"]
        assert st.conn.execute(
            "SELECT 1 FROM sessions WHERE id IN (?, ?)",
            (tokens["session"], tokens["correction_session"]),
        ).fetchone() is None
        target_session = st.conn.execute(
            "SELECT started_at, ended_at, source, meta FROM sessions WHERE id=?",
            (unrelated_after["session_id"],),
        ).fetchone()
        assert target_session["started_at"] == old_target_session["started_at"]
        assert target_session["ended_at"] == old_target_session["ended_at"]
        assert target_session["source"] == PURGE_SAFE_SESSION_SOURCE
        assert json.loads(target_session["meta"]) == {"keep": clean_meta}
        assert st.get_meta("current_session") == unrelated_after["session_id"]
        assert st.conn.execute(
            "SELECT request_payload FROM corrections WHERE id=?",
            (correction["correction_id"],),
        ).fetchone()["request_payload"] is None

        absent = tuple(tokens.values()) + (
            target.memory_id,
            target.event_id,
        )
        _assert_tokens_absent_from_tables(st, absent)
        _assert_tokens_absent_from_payload(st.trace(replacement_id), absent)

    detail = make_dash_client().get(
        f"/api/namespace/default/memory/{replacement_id}"
    )
    assert detail.status_code == 200
    _assert_tokens_absent_from_payload(detail.json(), absent)
    _assert_tokens_absent_from_payload(
        json.loads(memory_trace(replacement_id, namespace="default")), absent
    )


def test_schema_has_one_correction_per_target_and_canonical_identity(lineage_env):
    from haunt.store import Store

    with Store("default") as st:
        original = st.observe("identity")
        st.contradict(original.memory_id, replacement="new", idempotency_key="identity-key")
        row = st.conn.execute(
            "SELECT request_identity, request_payload FROM corrections"
        ).fetchone()
        assert row["request_identity"].startswith("sha256:")
        assert isinstance(row["request_payload"], bytes)
        indexes = {
            r["name"] for r in st.conn.execute("PRAGMA index_list(corrections)").fetchall()
        }
        assert "idx_corrections_target_memory" in indexes


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("idempotency_key", None),
        ("idempotency_key", ""),
        ("idempotency_key", " \t\n"),
        ("idempotency_key", "k" * 513),
        ("request_identity", None),
        ("request_identity", "sha256:short"),
        ("request_identity", "sha256:" + "z" * 64),
        ("request_payload", None),
        ("request_payload", "not-a-blob"),
        ("response_json", None),
        ("response_json", "not-json"),
    ],
)
def test_schema_rejects_malformed_normal_corrections(lineage_env, field, invalid):
    from haunt.store import Store
    from haunt.util import new_id, now_iso

    values = {
        "id": new_id(),
        "target_memory_id": new_id(),
        "corrected_at": now_iso(),
        "origin": "direct-sql-test",
        "session_id": "direct-sql-session",
        "idempotency_key": "direct-sql-key",
        "request_identity": "sha256:" + "0" * 64,
        "request_payload": b"payload",
        "response_json": "{}",
    }
    values[field] = invalid
    with Store("default") as st:
        with pytest.raises(sqlite3.IntegrityError, match="correction invariant"):
            st.conn.execute(
                """
                INSERT INTO corrections(
                    id, target_memory_id, corrected_at, origin, session_id,
                    idempotency_key, request_identity, request_payload, response_json
                ) VALUES (:id, :target_memory_id, :corrected_at, :origin, :session_id,
                          :idempotency_key, :request_identity, :request_payload,
                          :response_json)
                """,
                values,
            )


def test_schema_enforces_key_uniqueness_and_allows_scrubbed_tombstones(lineage_env):
    from haunt.store import Store
    from haunt.util import new_id, now_iso

    with Store("default") as st:
        def insert_normal(correction_id, target_id):
            st.conn.execute(
                """
                INSERT INTO corrections(
                    id, target_memory_id, corrected_at, origin, session_id,
                    idempotency_key, request_identity, request_payload, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    correction_id,
                    target_id,
                    now_iso(),
                    "direct-sql-test",
                    "direct-sql-session",
                    "duplicate-schema-key",
                    "sha256:" + "1" * 64,
                    b"payload",
                    "{}",
                ),
            )

        insert_normal(new_id(), new_id())
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            insert_normal(new_id(), new_id())

        tombstone_id = new_id()
        st.conn.execute(
            """
            INSERT INTO lineage_tombstones(schema_version, tombstone_id, status, erased_at)
            VALUES (1, ?, 'erased', ?)
            """,
            (tombstone_id, now_iso()),
        )
        st.conn.execute(
            """
            INSERT INTO corrections(id, target_tombstone_id, corrected_at)
            VALUES (?, ?, ?)
            """,
            (new_id(), tombstone_id, now_iso()),
        )
        invalid_tombstone_id = new_id()
        st.conn.execute(
            """
            INSERT INTO lineage_tombstones(schema_version, tombstone_id, status, erased_at)
            VALUES (1, ?, 'erased', ?)
            """,
            (invalid_tombstone_id, now_iso()),
        )
        with pytest.raises(sqlite3.IntegrityError, match="correction invariant"):
            st.conn.execute(
                """
                INSERT INTO corrections(
                    id, target_tombstone_id, corrected_at, idempotency_key
                ) VALUES (?, ?, ?, ?)
                """,
                (new_id(), invalid_tombstone_id, now_iso(), "must-be-scrubbed"),
            )


def test_idempotency_key_omission_is_rejected_without_writes_on_all_surfaces(
    lineage_env,
):
    from haunt.cli import app
    from haunt.mcp_server import memory_contradict, server
    from haunt.store import Store
    from tests.dashutil import make_dash_client
    from typer.testing import CliRunner

    with Store("default") as st:
        target = st.observe("required key target")
        baseline_changes = st.conn.total_changes
        with pytest.raises(TypeError, match="idempotency_key"):
            st.contradict(target.memory_id)
        assert st.conn.total_changes == baseline_changes

    cli = CliRunner().invoke(
        app,
        ["correct", target.memory_id, "--replacement", "must not land", "-n", "default"],
    )
    assert cli.exit_code == 2
    assert "idempotency" in cli.output.lower()

    dashboard = make_dash_client().post(
        f"/api/namespace/default/memory/{target.memory_id}/contradict",
        json={"replacement": "must not land"},
    )
    assert dashboard.status_code == 400
    assert dashboard.json()["error"] == "idempotency_key is required"

    with pytest.raises(TypeError, match="idempotency_key"):
        memory_contradict(target.memory_id)
    tool = next(
        tool
        for tool in server._tool_manager.list_tools()
        if tool.name == "memory_contradict"
    )
    assert "idempotency_key" in tool.parameters["required"]

    with Store("default") as st:
        assert st.get_memory(target.memory_id)["valid_to"] is None
        assert st.conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0] == 0
        assert st.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content='must not land'"
        ).fetchone()[0] == 0


def test_purge_responses_never_echo_erased_ids_on_store_cli_mcp_or_dashboard(
    lineage_env, monkeypatch
):
    import haunt.mcp_server as mcp
    from haunt.cli import app
    from haunt.store import Store
    from tests.dashutil import make_dash_client
    from typer.testing import CliRunner

    with Store("default") as st:
        store_target = st.observe("store purge response target")
        st.contradict(
            store_target.memory_id,
            replacement="store purge response successor",
            idempotency_key="store-purge-response",
        )
        store_result = st.purge(store_target.memory_id)
        cli_target = st.observe("cli purge response target")
        dashboard_target = st.observe("dashboard purge response target")
        mcp_target = st.observe("mcp purge response target")

    assert store_result["ok"] is True
    assert "memory_id" not in store_result
    assert "event_id" not in store_result
    assert set(store_result["lineage_tombstone"]) == {
        "schema_version",
        "tombstone_id",
        "status",
        "erased_at",
    }
    _assert_tokens_absent_from_payload(
        store_result, (store_target.memory_id, store_target.event_id)
    )

    cli = CliRunner().invoke(
        app,
        ["delete", cli_target.memory_id, "--yes", "-n", "default"],
    )
    assert cli.exit_code == 0, cli.output
    _assert_tokens_absent_from_payload(
        cli.output, (cli_target.memory_id, cli_target.event_id)
    )

    dashboard = make_dash_client().delete(
        f"/api/namespace/default/memory/{dashboard_target.memory_id}"
    )
    assert dashboard.status_code == 200
    assert "memory_id" not in dashboard.json()
    assert "event_id" not in dashboard.json()
    _assert_tokens_absent_from_payload(
        dashboard.json(), (dashboard_target.memory_id, dashboard_target.event_id)
    )

    monkeypatch.setenv("HAUNT_MCP_ALLOW_PURGE", "1")
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    mcp._MCP_AUTHORITY = None
    mcp._MCP_AUTHORITY_HOME = None
    mcp_result = json.loads(
        mcp.memory_purge(mcp_target.memory_id, namespace="default")
    )
    assert mcp_result["ok"] is True, mcp_result
    assert "memory_id" not in mcp_result
    assert "event_id" not in mcp_result
    _assert_tokens_absent_from_payload(
        mcp_result, (mcp_target.memory_id, mcp_target.event_id)
    )


def test_dashboard_detail_and_mutation_expose_lineage_and_idempotency(lineage_env):
    from haunt.store import Store
    from tests.dashutil import make_dash_client

    with Store("default") as st:
        original = st.observe("dashboard lineage")
    client = make_dash_client()
    path = f"/api/namespace/default/memory/{original.memory_id}/contradict"
    body = {
        "replacement": "dashboard replacement",
        "reason": "dashboard reason",
        "idempotency_key": "dashboard-retry",
    }
    first = client.post(path, json=body)
    replay = client.post(path, json=body)
    assert first.status_code == replay.status_code == 200
    assert replay.json()["deduplicated"] is True
    assert replay.json()["correction_id"] == first.json()["correction_id"]

    detail = client.get(
        f"/api/namespace/default/memory/{first.json()['replacement_memory_id']}"
    )
    assert detail.status_code == 200
    assert detail.json()["trace"]["lineage_status"] == "linked"
    assert len(detail.json()["trace"]["members"]) == 2


def test_dashboard_json_preserves_replacement_bytes_and_replay_order(lineage_env):
    from haunt.dashboard import HTML
    from haunt.store import Store
    from tests.dashutil import make_dash_client

    with Store("default") as st:
        targets = [st.observe(f"dashboard canonical {i}") for i in range(3)]
    client = make_dash_client()

    null_path = f"/api/namespace/default/memory/{targets[0].memory_id}/contradict"
    first = client.post(null_path, json={"idempotency_key": "dash-null"})
    replay = client.post(
        null_path,
        json={
            "replacement": None,
            "idempotency_key": "dash-null",
            "session_id": {"invalid": True},
        },
    )
    conflict = client.post(
        null_path,
        json={
            "replacement": "",
            "idempotency_key": "dash-null",
            "session_id": {"invalid": True},
        },
    )
    assert first.status_code == replay.status_code == 200
    assert replay.json()["deduplicated"] is True
    assert conflict.status_code == 409
    assert conflict.json()["conflict"] == "idempotency_key_reused"

    results = []
    for target, value, key in (
        (targets[1], "", "dash-empty"),
        (targets[2], "   ", "dash-whitespace"),
    ):
        response = client.post(
            f"/api/namespace/default/memory/{target.memory_id}/contradict",
            json={"replacement": value, "idempotency_key": key},
        )
        assert response.status_code == 200
        results.append(response.json())
    with Store("default") as st:
        assert st.get_memory(results[0]["replacement_memory_id"])["content"] == ""
        assert st.get_memory(results[1]["replacement_memory_id"])["content"] == "   "
        invalid_target = st.observe("dashboard invalid metadata")

    invalid = client.post(
        f"/api/namespace/default/memory/{invalid_target.memory_id}/contradict",
        json={"idempotency_key": "dash-invalid", "session_id": {"invalid": True}},
    )
    assert invalid.status_code == 400
    assert "session_id" in invalid.json()["error"]

    assert '<textarea id="contradictReplacement"' in HTML
    assert 'if($("contradictHasReplacement").checked)' in HTML
    assert 'body.replacement=$("contradictReplacement").value;' in HTML
    assert '$("contradictReplacement").value.trim()' not in HTML
    assert "const body={idempotency_key:" in HTML


def test_cli_correct_and_trace_surfaces(lineage_env):
    from haunt.cli import app
    from haunt.store import Store
    from typer.testing import CliRunner

    with Store("default") as st:
        original = st.observe("cli lineage")
    runner = CliRunner()
    corrected = runner.invoke(
        app,
        [
            "correct",
            original.memory_id,
            "--replacement",
            "cli replacement",
            "--idempotency-key",
            "cli-retry",
            "-n",
            "default",
        ],
    )
    assert corrected.exit_code == 0, corrected.output
    result = json.loads(corrected.stdout)
    traced = runner.invoke(
        app, ["trace", result["replacement_memory_id"], "-n", "default"]
    )
    assert traced.exit_code == 0, traced.output
    assert json.loads(traced.stdout)["lineage_status"] == "linked"


def test_cli_preserves_null_empty_whitespace_and_replay_order(lineage_env):
    from haunt.cli import app
    from haunt.store import Store
    from typer.testing import CliRunner

    with Store("default") as st:
        targets = [st.observe(f"cli canonical {i}") for i in range(3)]
    runner = CliRunner()

    base = ["correct", targets[0].memory_id, "--idempotency-key", "cli-null", "-n", "default"]
    first = runner.invoke(app, base)
    replay = runner.invoke(app, [*base, "--origin", ""])
    conflict = runner.invoke(app, [*base, "--replacement", "", "--origin", ""])
    assert first.exit_code == replay.exit_code == 0
    assert json.loads(replay.stdout)["deduplicated"] is True
    assert conflict.exit_code == 1
    assert json.loads(conflict.stdout)["conflict"] == "idempotency_key_reused"

    stored_ids = []
    for target, value, key in (
        (targets[1], "", "cli-empty"),
        (targets[2], "   ", "cli-whitespace"),
    ):
        response = runner.invoke(
            app,
            [
                "correct",
                target.memory_id,
                "--replacement",
                value,
                "--idempotency-key",
                key,
                "-n",
                "default",
            ],
        )
        assert response.exit_code == 0, response.output
        stored_ids.append(json.loads(response.stdout)["replacement_memory_id"])
    with Store("default") as st:
        assert st.get_memory(stored_ids[0])["content"] == ""
        assert st.get_memory(stored_ids[1])["content"] == "   "
        invalid_target = st.observe("cli invalid metadata")
    invalid = runner.invoke(
        app,
        [
            "correct",
            invalid_target.memory_id,
            "--idempotency-key",
            "cli-invalid",
            "--origin",
            "",
            "-n",
            "default",
        ],
    )
    assert invalid.exit_code == 2
    assert "origin" in invalid.output


def test_mcp_preserves_null_empty_whitespace_and_replay_order(lineage_env, monkeypatch):
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    from haunt.mcp_server import memory_contradict
    from haunt.store import Store

    with Store("default") as st:
        targets = [st.observe(f"mcp canonical {i}") for i in range(3)]

    first = json.loads(
        memory_contradict(
            targets[0].memory_id,
            replacement=None,
            namespace="default",
            origin="first-origin",
            session_id="first-session",
            idempotency_key="mcp-null",
        )
    )
    replay = json.loads(
        memory_contradict(
            targets[0].memory_id,
            replacement=None,
            namespace="default",
            origin="",
            session_id=object(),
            idempotency_key="mcp-null",
        )
    )
    conflict = json.loads(
        memory_contradict(
            targets[0].memory_id,
            replacement="",
            namespace="default",
            origin="",
            session_id=object(),
            idempotency_key="mcp-null",
        )
    )
    assert first["ok"] is True
    assert replay["deduplicated"] is True
    assert conflict["conflict"] == "idempotency_key_reused"

    stored_ids = []
    for target, value, key in (
        (targets[1], "", "mcp-empty"),
        (targets[2], "   ", "mcp-whitespace"),
    ):
        response = json.loads(
            memory_contradict(
                target.memory_id,
                replacement=value,
                namespace="default",
                idempotency_key=key,
            )
        )
        stored_ids.append(response["replacement_memory_id"])
    with Store("default") as st:
        assert st.get_memory(stored_ids[0])["content"] == ""
        assert st.get_memory(stored_ids[1])["content"] == "   "
        invalid_target = st.observe("mcp invalid metadata")
    invalid = json.loads(
        memory_contradict(
            invalid_target.memory_id,
            namespace="default",
            origin="",
            session_id=object(),
            idempotency_key="mcp-invalid",
        )
    )
    assert invalid["ok"] is False
    assert "origin" in invalid["error"]
