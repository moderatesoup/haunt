"""Append-only correction lineage, replay, migration, and erasure evidence."""

from __future__ import annotations

import json
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
            origin="safe-successor-origin",
            session_id="safe-successor-session",
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

        for table_row in st.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            table = table_row["name"]
            for row in st.conn.execute(f'SELECT * FROM "{table}"').fetchall():
                serialized = json.dumps(
                    [value.hex() if isinstance(value, bytes) else value for value in row]
                )
                assert canary not in serialized, table
                assert erased_id not in serialized, table
                assert erased_event not in serialized, table


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
