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
from haunt.store import SCHEMA_VERSION, Store, observe as public_observe


LOGICAL_TABLES = (
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


def _import_envelope(
    fidelity: str = "lossless", *, channel: str = "python"
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "import",
        "channel": channel,
        "source_platform": "聊天平台",
        "source_native_id": "消息-雪-🧊",
        "source_format": "vendor-json",
        "parser_version": "parser/2.4.1",
        "imported_at": "2025-03-04T05:06:07.123456-06:00",
        "fidelity": fidelity,
        "original_blob_sha256": "sha256:" + "ab" * 32,
        "transforms": ["decode:utf-8", "normalize:newlines"],
    }


def _logical_counts(store: Store) -> dict[str, int]:
    return {
        table: store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in LOGICAL_TABLES
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
        timeline = st.events(limit=10)
        trace = st.trace(result.memory_id)

    expected = dict(_import_envelope(fidelity), origin="import-cli")
    expected["imported_at"] = "2025-03-04T11:06:07.123456+00:00"
    assert result.provenance == expected
    assert detail["provenance"] == expected
    assert browse[0]["provenance"] == expected
    assert timeline[0]["provenance"] == expected
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
        "channel": "python",
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
            channel="cursor_hook",
        )
        stored = st.get_memory(result.memory_id)["provenance"]
    assert stored["channel"] == "cursor_hook"
    assert stored["producer_tool"] == "Shell"
    assert stored["producer_call_id"] == "调用-α-42"


def test_direct_python_write_defaults_are_honest_across_public_apis(
    provenance_env,
):
    with Store("default") as st:
        direct = st.observe("direct Store observation")
        procedure = st.procedure_write("direct procedure", "python steps")
        target = st.observe("direct correction target")
        correction = st.contradict(
            target.memory_id,
            replacement="direct correction replacement",
            idempotency_key="direct-python-correction",
        )
        direct_detail = st.get_memory(direct.memory_id)
        procedure_detail = st.procedure_get("direct procedure")
        replacement_detail = st.get_memory(correction["replacement_memory_id"])
        correction_origin = st.conn.execute(
            "SELECT origin FROM corrections WHERE id=?",
            (correction["correction_id"],),
        ).fetchone()["origin"]

    helper = public_observe(
        "top-level Python helper observation",
        namespace="default",
        defer_embedding=True,
    )
    with Store("default") as st:
        helper_detail = st.get_memory(helper.memory_id)

    expected = {
        "schema_version": 1,
        "kind": "native",
        "channel": "python",
        "origin": "python",
    }
    assert direct.provenance == expected
    assert direct_detail["origin"] == "python"
    assert direct_detail["provenance"] == expected
    assert procedure.provenance == expected
    assert procedure_detail is not None
    assert procedure_detail["provenance"] == expected
    assert replacement_detail["origin"] == "python"
    assert replacement_detail["provenance"] == expected
    assert correction_origin == "python"
    assert helper.provenance == expected
    assert helper_detail["origin"] == "python"
    assert helper_detail["provenance"] == expected


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
    "kwargs, message",
    [
        ({"origin": ""}, "origin"),
        ({"origin": 0}, "origin"),
        ({"origin": "雪" * 683}, "2048 UTF-8 bytes"),
        ({"channel": ""}, "channel"),
        ({"channel": 0}, "channel"),
        ({"channel": "雪" * 683}, "2048 UTF-8 bytes"),
        ({"tool_name": ""}, "tool_name"),
        ({"tool_name": 0}, "tool_name"),
        ({"tool_name": "雪" * 683}, "2048 UTF-8 bytes"),
        ({"producer_call_id": ""}, "producer_call_id"),
        ({"producer_call_id": 0}, "producer_call_id"),
        (
            {"tool_name": "Shell", "producer_call_id": "雪" * 683},
            "2048 UTF-8 bytes",
        ),
        ({"producer_call_id": "call-without-tool"}, "requires tool_name"),
    ],
)
def test_actual_attribution_is_strictly_validated_before_store_writes(
    provenance_env, kwargs, message
):
    with Store("default") as st:
        before = _logical_counts(st)
        with pytest.raises(ValueError, match=message):
            st.observe("rejected actual attribution", defer_embedding=True, **kwargs)
        assert _logical_counts(st) == before


@pytest.mark.parametrize(
    "provenance, kwargs, message",
    [
        (
            {"schema_version": 1, "kind": "native", "channel": "mcp"},
            {},
            "channel",
        ),
        (
            {"schema_version": 1, "kind": "native", "origin": "claimed"},
            {},
            "origin",
        ),
        (
            {"schema_version": 1, "kind": "native", "producer_tool": 0},
            {},
            "producer_tool",
        ),
        (
            {"schema_version": 1, "kind": "native", "producer_call_id": 0},
            {},
            "producer_call_id",
        ),
    ],
)
def test_supplied_native_attribution_must_match_actual_entry_inputs(
    provenance_env, provenance, kwargs, message
):
    with Store("default") as st:
        before = _logical_counts(st)
        with pytest.raises(ValueError, match=message):
            st.observe(
                "rejected claimed attribution",
                provenance=provenance,
                defer_embedding=True,
                **kwargs,
            )
        assert _logical_counts(st) == before


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
        before = _logical_counts(st)
        with pytest.raises(ValueError, match=message):
            st.observe("must not land", provenance=bad, defer_embedding=True)
        after = _logical_counts(st)
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


def test_legacy_idempotency_retry_fails_closed_when_attribution_is_unknown(
    provenance_env,
):
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
        with pytest.raises(ValueError, match="cannot verify legacy provenance"):
            st.observe(
                "legacy retry",
                origin="old-hook",
                idempotency_key="old-hook-key",
                defer_embedding=True,
            )
        assert st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert st.get_memory(old.memory_id)["provenance"] == {
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
        "origin": "python",
    }
    assert not _contains_key(detail, "confidence")


def test_invalid_stored_idempotency_retry_fails_closed(provenance_env):
    with Store("default") as st:
        result = st.observe(
            "tampered retry",
            idempotency_key="tampered-key",
            defer_embedding=True,
        )
        st.conn.execute(
            "UPDATE events SET provenance=? WHERE id=?",
            ('{"schema_version":99,"kind":"native"}', result.event_id),
        )
        st.conn.commit()
        with pytest.raises(ValueError, match="cannot verify invalid stored provenance"):
            st.observe(
                "tampered retry",
                idempotency_key="tampered-key",
                defer_embedding=True,
            )
        assert st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


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
            channel="review_workflow",
        )
        trace = st.trace(correction["replacement_memory_id"])
    assert trace["lineage_status"] == "linked"
    assert trace["members"][0]["provenance"]["kind"] == "import"
    assert trace["members"][0]["provenance"]["source_native_id"] == "消息-雪-🧊"
    assert trace["members"][1]["provenance"] == {
        "schema_version": 1,
        "kind": "native",
        "channel": "review_workflow",
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
        "channel": "privacy_purge",
        "origin": "privacy-sanitized",
    }
    assert json.loads(session["meta"]) == {
        "kind": "keep",
        "origin": "clean",
        "safe": "yes",
    }


def test_cli_mcp_and_dashboard_share_the_same_envelope_schema(
    provenance_env, monkeypatch
):
    cli_envelope = _import_envelope("derived", channel="cli")
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
            json.dumps(cli_envelope, ensure_ascii=False),
        ],
    )
    assert cli.exit_code == 0, cli.output
    cli_provenance = json.loads(cli.stdout.split("provenance ", 1)[1])

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_envelope = _import_envelope("derived", channel="mcp")
    mcp_payload = json.loads(
        mcp_server.memory_observe(
            "mcp imported",
            namespace="default",
            origin="importer",
            provenance=mcp_envelope,
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
    assert cli_provenance["channel"] == "cli"
    assert mcp_payload["provenance"]["channel"] == "mcp"
    assert {k: v for k, v in cli_provenance.items() if k != "channel"} == {
        k: v for k, v in mcp_payload["provenance"].items() if k != "channel"
    }
    assert detail["provenance"] == mcp_payload["provenance"]
    assert browsed["provenance"] == mcp_payload["provenance"]


def test_cli_timeline_json_and_human_outputs_surface_all_provenance_states(
    provenance_env, monkeypatch
):
    with Store("default") as st:
        native = st.observe("timeline native")
        imported = st.observe(
            "timeline import",
            origin="archive",
            provenance=_import_envelope("reconstructed"),
        )
        legacy = st.observe("timeline legacy", origin="legacy-source")
        invalid = st.observe("timeline invalid")
        st.conn.execute(
            "UPDATE events SET provenance=NULL WHERE id=?",
            (legacy.event_id,),
        )
        st.conn.execute(
            "UPDATE events SET provenance=? WHERE id=?",
            ('{"schema_version":99,"kind":"native"}', invalid.event_id),
        )
        st.conn.commit()
        store_rows = st.events(limit=20)

    runner = CliRunner()
    machine = runner.invoke(
        app, ["timeline", "-n", "default", "--limit", "20", "--json"]
    )
    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.stdout)
    assert payload["namespace"] == "default"
    assert payload["events"] == store_rows
    by_content = {row["content"]: row for row in payload["events"]}
    assert by_content["timeline native"]["provenance"] == native.provenance
    assert by_content["timeline import"]["provenance"] == imported.provenance
    assert by_content["timeline legacy"]["provenance"] == {
        "schema_version": 1,
        "kind": "legacy_unstructured",
        "origin": "legacy-source",
        "meta": "{}",
    }
    assert by_content["timeline invalid"]["provenance"] == {
        "schema_version": 1,
        "kind": "invalid_stored",
        "origin": "python",
    }

    human = runner.invoke(app, ["timeline", "-n", "default", "--limit", "20"])
    assert human.exit_code == 0, human.output
    assert "source=python/python" in human.stdout
    assert "source=python/archive" in human.stdout
    assert "source=unknown/legacy-source" in human.stdout

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_payload = json.loads(
        mcp_server.memory_timeline(namespace="default", limit=20)
    )
    assert mcp_payload["events"] == payload["events"]

    from tests.dashutil import make_dash_client

    dashboard = make_dash_client().get(
        "/api/namespace/default/timeline?limit=20"
    )
    assert dashboard.status_code == 200, dashboard.text
    assert dashboard.json()["events"] == payload["events"]


@pytest.mark.parametrize(
    "args, expected_error",
    [
        (["-n", "default", "--clock", "not-a-clock"], "clock"),
        (["-n", "missing-timeline-namespace"], "unknown namespace"),
    ],
)
def test_cli_timeline_json_errors_use_stable_envelope(
    provenance_env, args, expected_error
):
    result = CliRunner().invoke(app, ["timeline", *args, "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.stderr or result.output)
    assert payload["ok"] is False
    assert payload["namespace"]
    assert expected_error in payload["error"]


@pytest.mark.parametrize(
    "path, namespace, expected_error",
    [
        (
            "/api/namespace/default/timeline?clock=not-a-clock",
            "default",
            "clock must be",
        ),
        (
            "/api/namespace/default/timeline?since=not-a-time",
            "default",
            "not-a-time",
        ),
        (
            "/api/namespace/missing-dashboard-timeline/timeline",
            "missing-dashboard-timeline",
            "unknown namespace",
        ),
    ],
)
def test_dashboard_timeline_errors_are_exact_json_400_envelopes(
    provenance_env, path, namespace, expected_error
):
    from tests.dashutil import make_dash_client

    response = make_dash_client().get(path)
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    payload = response.json()
    assert set(payload) == {"ok", "error", "namespace"}
    assert payload["ok"] is False
    assert payload["namespace"] == namespace
    assert expected_error in payload["error"]
    assert "Internal Server Error" not in response.text


def test_procedure_get_and_list_keep_provenance_across_store_mcp_and_cli(
    provenance_env, monkeypatch
):
    with Store("default") as st:
        native = st.procedure_write("native procedure", "native steps")
        imported = st.observe(
            "imported steps",
            role="system",
            tier="procedural",
            origin="archive",
            meta={
                "kind": "procedure",
                "name": "imported procedure",
                "trigger": "when imported",
            },
            provenance=_import_envelope("lossless"),
        )
        legacy = st.procedure_write(
            "legacy procedure", "legacy steps", origin="legacy-procedure-source"
        )
        invalid = st.procedure_write("invalid procedure", "invalid steps")
        st.conn.execute(
            "UPDATE events SET provenance=NULL WHERE id=?",
            (legacy.event_id,),
        )
        st.conn.execute(
            "UPDATE events SET provenance=? WHERE id=?",
            ('{"schema_version":99,"kind":"native"}', invalid.event_id),
        )
        st.conn.commit()

        native_get = st.procedure_get("native procedure")
        imported_get = st.procedure_get("imported procedure")
        legacy_get = st.procedure_get("legacy procedure")
        invalid_get = st.procedure_get("invalid procedure")
        listed = {row["name"]: row for row in st.procedure_list()}
        worldview = {row["name"]: row for row in st.worldview()["procedures"]}

    assert native_get is not None and native_get["provenance"] == native.provenance
    assert imported_get is not None
    assert imported_get["provenance"] == imported.provenance
    assert legacy_get is not None
    assert legacy_get["provenance"]["kind"] == "legacy_unstructured"
    assert legacy_get["provenance"]["origin"] == "legacy-procedure-source"
    assert invalid_get is not None
    assert invalid_get["provenance"] == {
        "schema_version": 1,
        "kind": "invalid_stored",
        "origin": "python",
    }
    for name, expected in (
        ("native procedure", native_get),
        ("imported procedure", imported_get),
        ("legacy procedure", legacy_get),
        ("invalid procedure", invalid_get),
    ):
        assert listed[name]["provenance"] == expected["provenance"]
        assert worldview[name]["provenance"] == expected["provenance"]

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_get = json.loads(
        mcp_server.memory_procedure(
            "get", name="imported procedure", namespace="default"
        )
    )
    assert mcp_get["ok"] is True
    assert mcp_get["procedure"]["provenance"] == imported.provenance
    mcp_list = json.loads(mcp_server.memory_procedure("list", namespace="default"))
    assert mcp_list["ok"] is True
    mcp_listed = {row["name"]: row for row in mcp_list["procedures"]}
    assert mcp_listed["native procedure"]["provenance"] == native.provenance
    assert mcp_listed["legacy procedure"]["provenance"][
        "kind"
    ] == "legacy_unstructured"
    assert mcp_listed["invalid procedure"]["provenance"]["kind"] == "invalid_stored"
    mcp_worldview = json.loads(mcp_server.memory_worldview(namespace="default"))
    mcp_worldview_procedures = {
        row["name"]: row for row in mcp_worldview["procedures"]
    }
    for name, expected in (
        ("native procedure", native_get),
        ("imported procedure", imported_get),
        ("legacy procedure", legacy_get),
        ("invalid procedure", invalid_get),
    ):
        assert mcp_worldview_procedures[name]["provenance"] == expected[
            "provenance"
        ]

    runner = CliRunner()
    cli_get = runner.invoke(
        app, ["procedure", "get", "native procedure", "-n", "default"]
    )
    assert cli_get.exit_code == 0, cli_get.output
    native_json = json.dumps(native.provenance, ensure_ascii=False, sort_keys=True)
    assert f"provenance {native_json}" in cli_get.stdout
    cli_legacy = runner.invoke(
        app, ["procedure", "get", "legacy procedure", "-n", "default"]
    )
    assert cli_legacy.exit_code == 0, cli_legacy.output
    assert '"kind": "legacy_unstructured"' in cli_legacy.stdout
    cli_invalid = runner.invoke(
        app, ["procedure", "get", "invalid procedure", "-n", "default"]
    )
    assert cli_invalid.exit_code == 0, cli_invalid.output
    assert '"kind": "invalid_stored"' in cli_invalid.stdout
    cli_list = runner.invoke(app, ["procedure", "list", "-n", "default"])
    assert cli_list.exit_code == 0, cli_list.output
    assert f"provenance {native_json}" in cli_list.stdout
    cli_worldview = runner.invoke(
        app, ["worldview", "-n", "default", "--json"]
    )
    assert cli_worldview.exit_code == 0, cli_worldview.output
    cli_worldview_procedures = {
        row["name"]: row
        for row in json.loads(cli_worldview.stdout)["procedures"]
    }
    assert cli_worldview_procedures["legacy procedure"]["provenance"][
        "kind"
    ] == "legacy_unstructured"


def test_cli_and_mcp_invalid_parser_version_write_nothing(provenance_env, monkeypatch):
    cli_bad = {
        **_import_envelope(channel="cli"),
        "parser_version": {"not": "text"},
    }
    runner = CliRunner()
    cli = runner.invoke(
        app,
        [
            "observe",
            "bad cli",
            "-n",
            "default",
            "--provenance-json",
            json.dumps(cli_bad),
        ],
    )
    assert cli.exit_code == 2

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_bad = {
        **_import_envelope(channel="mcp"),
        "parser_version": {"not": "text"},
    }
    mcp = json.loads(
        mcp_server.memory_observe("bad mcp", namespace="default", provenance=mcp_bad)
    )
    assert mcp["ok"] is False
    with Store("default") as st:
        assert st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_cli_rejects_invalid_actual_and_claimed_attribution_without_writes(
    provenance_env,
):
    runner = CliRunner()
    oversized = "雪" * 683
    naive_import = {
        **_import_envelope(channel="cli"),
        "imported_at": "2025-01-01T00:00:00",
    }
    cases = [
        ["--origin", ""],
        ["--origin", oversized],
        ["--tool-name", ""],
        ["--tool-name", oversized],
        ["--producer-call-id", "call-without-tool"],
        ["--tool-name", "Shell", "--producer-call-id", oversized],
        [
            "--provenance-json",
            json.dumps({"schema_version": 1, "kind": "native", "channel": 0}),
        ],
        [
            "--provenance-json",
            json.dumps({"schema_version": 1, "kind": "native", "channel": "mcp"}),
        ],
        ["--provenance-json", json.dumps(naive_import)],
    ]
    with Store("default") as st:
        before = _logical_counts(st)
    for index, options in enumerate(cases):
        result = runner.invoke(
            app,
            ["observe", f"invalid cli {index}", "-n", "default", *options],
        )
        assert result.exit_code == 2, (options, result.output)
        assert "error:" in result.output
    with Store("default") as st:
        assert _logical_counts(st) == before


def test_mcp_rejects_invalid_actual_and_claimed_attribution_without_writes(
    provenance_env, monkeypatch
):
    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    oversized = "雪" * 683
    naive_import = {
        **_import_envelope(channel="mcp"),
        "imported_at": "2025-01-01T00:00:00",
    }
    cases = [
        {"origin": ""},
        {"origin": 0},
        {"origin": oversized},
        {"tool_name": ""},
        {"tool_name": 0},
        {"tool_name": oversized},
        {"producer_call_id": "call-without-tool"},
        {"tool_name": "Shell", "producer_call_id": 0},
        {"tool_name": "Shell", "producer_call_id": oversized},
        {
            "provenance": {
                "schema_version": 1,
                "kind": "native",
                "channel": 0,
            }
        },
        {
            "provenance": {
                "schema_version": 1,
                "kind": "native",
                "channel": "cli",
            }
        },
        {"provenance": naive_import},
    ]
    with Store("default") as st:
        before = _logical_counts(st)
    for index, kwargs in enumerate(cases):
        payload = json.loads(
            mcp_server.memory_observe(
                f"invalid mcp {index}", namespace="default", **kwargs
            )
        )
        assert payload["ok"] is False, kwargs
        assert payload["namespace"] == "default"
        assert payload["error"]
    invalid_procedure = json.loads(
        mcp_server.memory_procedure(
            "write",
            name="bad attribution",
            body="must not land",
            namespace="default",
            origin=0,
        )
    )
    assert invalid_procedure["ok"] is False
    assert invalid_procedure["namespace"] == "default"
    assert "origin" in invalid_procedure["error"]
    with Store("default") as st:
        assert _logical_counts(st) == before


def test_cli_and_mcp_fail_closed_on_unverifiable_idempotency_replays(
    provenance_env, monkeypatch
):
    runner = CliRunner()
    cli_args = [
        "observe",
        "legacy cli replay",
        "-n",
        "default",
        "--idempotency-key",
        "legacy-cli-key",
    ]
    first_cli = runner.invoke(app, cli_args)
    assert first_cli.exit_code == 0, first_cli.output
    with Store("default") as st:
        cli_event = st.conn.execute(
            "SELECT id FROM events WHERE idempotency_key='legacy-cli-key'"
        ).fetchone()["id"]
        st.conn.execute("UPDATE events SET provenance=NULL WHERE id=?", (cli_event,))
        st.conn.commit()
        count_after_cli = st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    cli_replay = runner.invoke(app, cli_args)
    assert cli_replay.exit_code == 2
    assert "cannot verify legacy provenance" in cli_replay.output

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    first_mcp = json.loads(
        mcp_server.memory_observe(
            "invalid mcp replay",
            namespace="default",
            idempotency_key="invalid-mcp-key",
        )
    )
    assert first_mcp["ok"] is True
    with Store("default") as st:
        st.conn.execute(
            "UPDATE events SET provenance=? WHERE id=?",
            ('{"schema_version":99,"kind":"native"}', first_mcp["event_id"]),
        )
        st.conn.commit()
        count_after_mcp = st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    mcp_replay = json.loads(
        mcp_server.memory_observe(
            "invalid mcp replay",
            namespace="default",
            idempotency_key="invalid-mcp-key",
        )
    )
    assert mcp_replay["ok"] is False
    assert "cannot verify invalid stored provenance" in mcp_replay["error"]
    with Store("default") as st:
        assert st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == count_after_mcp
        assert count_after_mcp == count_after_cli + 1


def test_cli_mcp_and_dashboard_corrections_record_actual_replacement_channel(
    provenance_env, monkeypatch
):
    with Store("default") as st:
        targets = [st.observe(f"target {index}").memory_id for index in range(3)]

    runner = CliRunner()
    cli_result = runner.invoke(
        app,
        [
            "correct",
            targets[0],
            "-n",
            "default",
            "--replacement",
            "cli replacement",
            "--idempotency-key",
            "cli-correction-channel",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_payload = json.loads(cli_result.stdout)

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_payload = json.loads(
        mcp_server.memory_contradict(
            targets[1],
            "mcp-correction-channel",
            replacement="mcp replacement",
            namespace="default",
        )
    )
    assert mcp_payload["ok"] is True

    from tests.dashutil import make_dash_client

    client = make_dash_client()
    dashboard_response = client.post(
        f"/api/namespace/default/memory/{targets[2]}/contradict",
        json={
            "replacement": "dashboard replacement",
            "idempotency_key": "dashboard-correction-channel",
        },
    )
    assert dashboard_response.status_code == 200, dashboard_response.text
    dashboard_payload = dashboard_response.json()

    with Store("default") as st:
        assert st.get_memory(cli_payload["replacement_memory_id"])["provenance"][
            "channel"
        ] == "cli"
        assert st.get_memory(mcp_payload["replacement_memory_id"])["provenance"][
            "channel"
        ] == "mcp"
        assert st.get_memory(dashboard_payload["replacement_memory_id"])[
            "provenance"
        ]["channel"] == "dashboard"
