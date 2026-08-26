"""E2: structured source/import provenance without truth scoring."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import pytest
from typer.testing import CliRunner

from haunt.cli import app
from haunt.provenance import IMPORT_FIDELITIES, native_provenance
from haunt.store import SCHEMA_VERSION, Store


@pytest.fixture
def provenance_env(tmp_path, monkeypatch):
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import embed
    from haunt.bootstrap import bootstrap
    from haunt.paths import ensure_layout

    embed.reset()
    ensure_layout()
    bootstrap("default")
    yield home
    embed.reset()


def _import_envelope(fidelity: str = "lossless") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "import",
        "channel": "archive-upload",
        "source_platform": "聊天平台",
        "source_native_id": "消息-雪-🧊",
        "source_format": "vendor-json",
        "parser_version": "parser/2.4.1",
        "imported_at": "2025-03-04T05:06:07.123456-06:00",
        "fidelity": fidelity,
        "original_blob_sha256": "sha256:" + "ab" * 32,
        "transforms": ["decode:utf-8", "normalize:newlines"],
    }


@pytest.mark.parametrize("fidelity", IMPORT_FIDELITIES)
def test_import_fidelity_round_trip_and_canonical_time(provenance_env, fidelity):
    with Store("default") as st:
        result = st.observe(
            f"imported {fidelity}",
            origin="import-cli",
            provenance=_import_envelope(fidelity),
        )
        detail = st.get_memory(result.memory_id)
        browse = st.browse_memories(limit=10)["memories"]
        trace = st.trace(result.memory_id)

    expected = dict(_import_envelope(fidelity), origin="import-cli")
    expected["imported_at"] = "2025-03-04T11:06:07.123456+00:00"
    assert result.provenance == expected
    assert detail["provenance"] == expected
    assert browse[0]["provenance"] == expected
    assert trace["members"][0]["provenance"] == expected


def test_unknowns_remain_absent_and_original_blob_absence_is_explicit(provenance_env):
    provenance = {
        "schema_version": 1,
        "kind": "import",
        "imported_at": "2025-01-01T00:00:00Z",
        "fidelity": "reconstructed",
        "original_blob_sha256": None,
    }
    with Store("default") as st:
        result = st.observe("sparse import", origin="batch", provenance=provenance)
        stored = st.get_memory(result.memory_id)["provenance"]
    assert stored == {
        **provenance,
        "origin": "batch",
        "imported_at": "2025-01-01T00:00:00.000000+00:00",
    }
    for absent in (
        "source_platform",
        "source_native_id",
        "source_format",
        "parser_version",
        "transforms",
        "producer_tool",
        "producer_call_id",
    ):
        assert absent not in stored


def test_native_tool_and_call_are_actual_observe_inputs(provenance_env):
    provenance = native_provenance(
        channel="cursor_hook",
        origin="cursor",
        tool="Shell",
        call_id="调用-α-42",
    )
    with Store("default") as st:
        result = st.observe(
            "",
            role="tool",
            tool_name="Shell",
            tool_output="ok",
            producer_call_id="调用-α-42",
            origin="cursor",
            provenance=provenance,
        )
        stored = st.get_memory(result.memory_id)["provenance"]
    assert stored["channel"] == "cursor_hook"
    assert stored["producer_tool"] == "Shell"
    assert stored["producer_call_id"] == "调用-α-42"


def test_native_producer_claims_must_match_actual_observe_inputs(provenance_env):
    claimed = {
        "schema_version": 1,
        "kind": "native",
        "producer_tool": "Shell",
        "producer_call_id": "call-7",
    }
    with Store("default") as st:
        with pytest.raises(ValueError, match="producer_tool"):
            st.observe("fake tool", provenance=claimed, defer_embedding=True)
        with pytest.raises(ValueError, match="producer_call_id"):
            st.observe(
                "wrong call",
                tool_name="Shell",
                producer_call_id="call-8",
                provenance=claimed,
                defer_embedding=True,
            )
        assert st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


@pytest.mark.parametrize(
    "module_name, channel, call_key",
    [
        ("haunt.cursor_hook", "cursor_hook", "tool_call_id"),
        ("haunt.claude_hook", "claude_code_hook", "tool_use_id"),
    ],
)
def test_hooks_capture_only_supplied_tool_call_ids(
    provenance_env, module_name, channel, call_key
):
    import importlib

    hook = importlib.import_module(module_name)
    payload = {
        "hook_event_name": "PostToolUse",
        "session_id": f"{channel}-session",
        call_key: f"{channel}-调用-42",
    }
    with Store("default") as st:
        hook._observe(
            st,
            payload,
            role="tool",
            tier="episodic",
            tool_name="Shell",
            tool_output="ok",
        )
        row = st.conn.execute(
            "SELECT id FROM memories ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        provenance = st.get_memory(row["id"])["provenance"]
    assert provenance["channel"] == channel
    assert provenance["producer_tool"] == "Shell"
    assert provenance["producer_call_id"] == f"{channel}-调用-42"


@pytest.mark.parametrize(
    "bad, message",
    [
        ({**_import_envelope(), "schema_version": 2}, "schema_version"),
        ({**_import_envelope(), "fidelity": "probably"}, "fidelity"),
        ({**_import_envelope(), "parser_version": 7}, "parser_version"),
        ({**_import_envelope(), "imported_at": "2025-01-01T00:00:00"}, "timestamp"),
        ({**_import_envelope(), "original_blob_sha256": "ABC"}, "sha256"),
        ({**_import_envelope(), "source_native_id": "x" * 2049}, "2048"),
        ({**_import_envelope(), "transforms": ["x"] * 129}, "128"),
        ({**_import_envelope(), "transforms": [""]}, "nonempty"),
        ({**_import_envelope(), "confidence": 0.9}, "unsupported"),
    ],
)
def test_invalid_provenance_rejected_before_any_logical_write(
    provenance_env, bad, message
):
    with Store("default") as st:
        tables = (
            "sessions",
            "events",
            "memories",
            "memories_fts",
            "embedding_jobs",
            "entities",
            "relations",
            "entity_mentions",
            "relation_evidence",
        )
        before = {
            table: st.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        }
        with pytest.raises(ValueError, match=message):
            st.observe("must not land", provenance=bad, defer_embedding=True)
        after = {
            table: st.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        }
    assert after == before


def test_provenance_migration_preserves_legacy_origin_and_meta_bytes(provenance_env):
    meta = '{  "原文": "snowman ☃", "order": [2, 1]  }'
    with Store("default") as st:
        old = st.observe("legacy row", origin="legacy/源")
        st.conn.execute(
            "UPDATE events SET meta=?, provenance=NULL WHERE id=?",
            (meta, old.event_id),
        )
        st.conn.execute("UPDATE meta SET value='7' WHERE key='schema_version'")
        st.conn.commit()
        before = st.conn.execute(
            "SELECT origin, meta, hex(origin), hex(meta) FROM events WHERE id=?",
            (old.event_id,),
        ).fetchone()

    # Simulate the exact pre-v8 shape rather than merely a null v8 value.
    db_path = provenance_env / "namespaces" / "default.db"
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE events DROP COLUMN provenance")
    conn.commit()
    conn.close()

    with Store("default") as migrated:
        after = migrated.conn.execute(
            "SELECT origin, meta, hex(origin), hex(meta) FROM events WHERE id=?",
            (old.event_id,),
        ).fetchone()
        detail = migrated.get_memory(old.memory_id)
        assert migrated.get_meta("schema_version") == str(SCHEMA_VERSION)
    assert tuple(after) == tuple(before)
    assert detail["provenance"] == {
        "schema_version": 1,
        "kind": "legacy_unstructured",
        "origin": "legacy/源",
        "meta": meta,
    }


def test_legacy_idempotency_retry_remains_compatible_and_honest(provenance_env):
    with Store("default") as st:
        old = st.observe(
            "legacy retry",
            origin="old-hook",
            meta={"old": "bytes"},
            idempotency_key="old-hook-key",
            defer_embedding=True,
        )
        st.conn.execute("UPDATE events SET provenance=NULL WHERE id=?", (old.event_id,))
        st.conn.commit()
        retry = st.observe(
            "legacy retry",
            origin="old-hook",
            idempotency_key="old-hook-key",
            defer_embedding=True,
        )
        assert st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert retry.deduplicated is True
    assert retry.provenance == {
        "schema_version": 1,
        "kind": "legacy_unstructured",
        "origin": "old-hook",
        "meta": '{"old": "bytes"}',
    }


def test_corrupt_or_unsupported_stored_envelope_fails_honest(provenance_env):
    with Store("default") as st:
        result = st.observe("tampered provenance")
        st.conn.execute(
            "UPDATE events SET provenance=? WHERE id=?",
            ('{"schema_version":99,"kind":"native","confidence":1}', result.event_id),
        )
        st.conn.commit()
        detail = st.get_memory(result.memory_id)
    assert detail["provenance"] == {
        "schema_version": 1,
        "kind": "invalid_stored",
        "origin": "cli",
    }
    assert not _contains_key(detail, "confidence")


def test_idempotency_replays_exact_provenance_and_conflicts_on_change(provenance_env):
    first_provenance = _import_envelope()
    with Store("default") as st:
        first = st.observe(
            "retry import",
            origin="importer",
            provenance=first_provenance,
            idempotency_key="import-42",
            defer_embedding=True,
        )
        replay = st.observe(
            "retry import",
            origin="importer",
            provenance=first_provenance,
            idempotency_key="import-42",
            defer_embedding=True,
        )
        changed = {**first_provenance, "source_native_id": "different"}
        with pytest.raises(ValueError, match="different provenance"):
            st.observe(
                "retry import",
                origin="importer",
                provenance=changed,
                idempotency_key="import-42",
                defer_embedding=True,
            )
        assert st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert replay.deduplicated is True
    assert replay.event_id == first.event_id
    assert replay.provenance == first.provenance


def test_concurrent_idempotent_import_commits_one_exact_envelope(provenance_env):
    barrier = Barrier(2)
    envelope = _import_envelope("lossy")

    def worker():
        with Store("default") as st:
            barrier.wait()
            return st.observe(
                "concurrent import",
                origin="batch",
                provenance=envelope,
                idempotency_key="concurrent-import-key",
                defer_embedding=True,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result() for future in [pool.submit(worker), pool.submit(worker)]
        ]
    assert results[0].event_id == results[1].event_id
    assert sorted(result.deduplicated for result in results) == [False, True]
    assert results[0].provenance == results[1].provenance
    with Store("default") as st:
        assert st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_corrected_import_trace_reaches_lineage_and_both_source_envelopes(
    provenance_env,
):
    with Store("default") as st:
        imported = st.observe(
            "old imported statement", origin="archive", provenance=_import_envelope()
        )
        correction = st.contradict(
            imported.memory_id,
            replacement="corrected statement",
            origin="reviewer",
            reason="source amended",
            idempotency_key="correct-import-1",
        )
        trace = st.trace(correction["replacement_memory_id"])
    assert trace["lineage_status"] == "linked"
    assert trace["members"][0]["provenance"]["kind"] == "import"
    assert trace["members"][0]["provenance"]["source_native_id"] == "消息-雪-🧊"
    assert trace["members"][1]["provenance"] == {
        "schema_version": 1,
        "kind": "native",
        "origin": "reviewer",
    }
    assert trace["corrections"][0]["reason"] == "source amended"


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


def test_no_confidence_field_in_schema_or_public_provenance(provenance_env):
    with Store("default") as st:
        result = st.observe("no fake truth score", provenance=_import_envelope())
        outputs = [
            result.provenance,
            st.get_memory(result.memory_id),
            st.browse_memories(),
            st.trace(result.memory_id),
        ]
        schema = "\n".join(
            str(row[0])
            for row in st.conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
            ).fetchall()
        ).lower()
    assert "confidence" not in schema
    assert all(not _contains_key(output, "confidence") for output in outputs)


def test_purge_removes_provenance_canaries_from_all_tables_and_surfaces(provenance_env):
    canary = "PROVENANCE-ERASURE-雪-7f83a9"
    blob_hash = "sha256:" + "7f83a9" * 10 + "7f83"
    envelope = {
        **_import_envelope(),
        "source_platform": canary + "-platform",
        "source_native_id": canary + "-native",
        "source_format": canary + "-format",
        "parser_version": canary + "-parser",
        "producer_tool": "ImporterTool",
        "producer_call_id": canary + "-call",
        "original_blob_sha256": blob_hash,
        "transforms": [canary + "-transform"],
    }
    with Store("default") as st:
        target = st.observe(
            "erase provenance",
            origin="archive",
            provenance=envelope,
            session_id="shared-provenance-session",
            tool_name="ImporterTool",
            producer_call_id=canary + "-call",
        )
        unrelated = st.observe(
            "unrelated shared-session memory",
            origin="safe-origin",
            session_id="shared-provenance-session",
        )
        st.conn.execute(
            "UPDATE sessions SET meta=? WHERE id=?",
            (
                '{"kind":"keep","origin":"clean","safe":"yes"}',
                "shared-provenance-session",
            ),
        )
        st.conn.commit()
        corrected = st.contradict(
            target.memory_id,
            replacement="survivor",
            origin=canary + "-correction-origin",
            idempotency_key="purge-provenance-key",
        )
        survivor_id = corrected["replacement_memory_id"]
        st.purge(target.memory_id)
        raw = []
        for table_row in st.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            name = table_row["name"]
            if not name.replace("_", "").isalnum():
                continue
            raw.extend(tuple(row) for row in st.conn.execute(f'SELECT * FROM "{name}"'))
        surfaces = {
            "detail": st.get_memory(survivor_id),
            "browse": st.browse_memories(),
            "trace": st.trace(survivor_id),
        }
        unrelated_after = st.get_memory(unrelated.memory_id)
        session = st.conn.execute(
            "SELECT meta FROM sessions WHERE id=?",
            (unrelated_after["session_id"],),
        ).fetchone()
    serialized = json.dumps(
        {"raw": raw, "surfaces": surfaces}, ensure_ascii=False, default=str
    )
    assert canary not in serialized
    assert blob_hash not in serialized
    assert surfaces["detail"]["provenance"] == {
        "schema_version": 1,
        "kind": "native",
        "origin": "privacy-sanitized",
    }
    assert json.loads(session["meta"]) == {
        "kind": "keep",
        "origin": "clean",
        "safe": "yes",
    }


def test_cli_mcp_and_dashboard_share_the_same_envelope(provenance_env, monkeypatch):
    envelope = _import_envelope("derived")
    runner = CliRunner()
    cli = runner.invoke(
        app,
        [
            "observe",
            "cli imported",
            "-n",
            "default",
            "--origin",
            "importer",
            "--provenance-json",
            json.dumps(envelope, ensure_ascii=False),
        ],
    )
    assert cli.exit_code == 0, cli.output
    cli_provenance = json.loads(cli.stdout.split("provenance ", 1)[1])

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_payload = json.loads(
        mcp_server.memory_observe(
            "mcp imported",
            namespace="default",
            origin="importer",
            provenance=envelope,
        )
    )
    assert mcp_payload["ok"] is True

    from tests.dashutil import make_dash_client

    client = make_dash_client()
    detail = client.get(
        f"/api/namespace/default/memory/{mcp_payload['memory_id']}"
    ).json()
    browse = client.get("/api/namespace/default/browse?limit=100").json()
    browsed = next(
        row
        for row in browse["memories"]
        if row["memory_id"] == mcp_payload["memory_id"]
    )
    assert cli_provenance == mcp_payload["provenance"]
    assert detail["provenance"] == mcp_payload["provenance"]
    assert browsed["provenance"] == mcp_payload["provenance"]


def test_cli_and_mcp_invalid_parser_version_write_nothing(provenance_env, monkeypatch):
    bad = {**_import_envelope(), "parser_version": {"not": "text"}}
    runner = CliRunner()
    cli = runner.invoke(
        app,
        [
            "observe",
            "bad cli",
            "-n",
            "default",
            "--provenance-json",
            json.dumps(bad),
        ],
    )
    assert cli.exit_code == 2

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp = json.loads(
        mcp_server.memory_observe("bad mcp", namespace="default", provenance=bad)
    )
    assert mcp["ok"] is False
    with Store("default") as st:
        assert st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
