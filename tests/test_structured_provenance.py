"""E2: structured source/import provenance without truth scoring."""

from __future__ import annotations

import base64
import json
import math
import sqlite3
import struct
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import pytest
from typer.testing import CliRunner

from haunt.cli import app
from haunt.provenance import (
    IMPORT_FIDELITIES,
    SQLITE_KEY_PREFIX,
    decode_json_safe_sqlite_key,
    encode_json_safe_sqlite_key,
    json_safe_sqlite,
    native_provenance,
    public_provenance,
)
from haunt.recall import Hit, recall
from haunt.store import SCHEMA_VERSION, Store, observe as public_observe
from haunt.util import format_iso, human_display, loads


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


def _decode_public_blob(value: Any) -> bytes:
    assert value.keys() == {"encoding", "data"}
    assert value["encoding"] == "base64"
    return base64.b64decode(value["data"], validate=True)


@pytest.mark.parametrize("value", [None, 0, 17, -2.5, "meta text", True])
def test_json_safe_sqlite_preserves_json_safe_sqlite_scalars(value):
    assert json_safe_sqlite(value) == value
    legacy = public_provenance(
        None,
        origin="legacy",
        legacy_meta=value,
    )
    assert legacy["meta"] == value


def test_json_safe_sqlite_losslessly_encodes_blob_memoryview_and_nonfinite_real():
    raw = b"\x00\xffblob</script>\x80"
    assert _decode_public_blob(json_safe_sqlite(raw)) == raw
    assert _decode_public_blob(json_safe_sqlite(bytearray(raw))) == raw
    assert _decode_public_blob(json_safe_sqlite(memoryview(raw))) == raw
    assert json_safe_sqlite(math.inf) == {
        "encoding": "sqlite-real",
        "data": "+infinity",
    }
    fallback = object()
    assert loads(raw, default=fallback) is fallback
    assert loads(memoryview(raw), default=fallback) is fallback
    json.dumps(
        json_safe_sqlite({"blob": memoryview(raw), "real": math.nan}), allow_nan=False
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ordinary text 雪", "ordinary text 雪"),
        (None, "null"),
        (7, "7"),
        (2.5, "2.5"),
        (math.inf, "<sqlite-real +infinity>"),
        (
            {"encoding": "sqlite-real", "data": "nan"},
            '{"encoding": "sqlite-real", "data": "nan"}',
        ),
        (
            {"encoding": "sqlite-real", "data": []},
            '{"encoding": "sqlite-real", "data": []}',
        ),
        ([1, None, "x"], '[1, null, "x"]'),
    ],
)
def test_human_display_is_stable_for_serialized_sqlite_types(value, expected):
    assert human_display(value) == expected


def test_human_display_marks_blobs_escapes_controls_and_bounds_output():
    raw = b"\x00\xff</script>\x1b[31m" * 200
    encoded = base64.b64encode(raw).decode("ascii")
    assert human_display(b"abc") == "<sqlite-blob base64:YWJj>"
    assert human_display(memoryview(b"abc")) == "<sqlite-blob base64:YWJj>"
    assert (
        human_display({"encoding": "base64", "data": "YWJj"}, sqlite_scalar=True)
        == "<sqlite-blob base64:YWJj>"
    )
    assert human_display({"encoding": "base64", "data": "YWJj"}) == (
        '{"encoding": "base64", "data": "YWJj"}'
    )
    assert (
        human_display(
            {"encoding": "base64", "data": "YWJj", "extra": True},
            sqlite_scalar=True,
        )
        == '{"encoding": "base64", "data": "YWJj", "extra": true}'
    )
    assert (
        human_display(
            {"encoding": "base64", "data": "***not-base64***"},
            sqlite_scalar=True,
        )
        == '{"encoding": "base64", "data": "***not-base64***"}'
    )
    assert (
        human_display({"encoding": "sqlite-real", "data": "nan"}, sqlite_scalar=True)
        == "<sqlite-real nan>"
    )
    assert human_display("normal\x1b[31m") == "normal\\u001b[31m"
    bounded = human_display(
        {"encoding": "base64", "data": encoded},
        limit=80,
        sqlite_scalar=True,
    )
    assert bounded.startswith("<sqlite-blob base64:")
    assert bounded.endswith("…")
    assert len(bounded) == 80
    assert "</script>" not in bounded
    assert format_iso(None) == ""
    assert format_iso(0) == "0"


def test_human_display_preserves_safe_layout_but_blocks_terminal_controls():
    value = "line1\r\nline2\rline3\tok\x1b]0;owned\x07\u202e"
    rendered = human_display(value, limit=200, preserve_layout=True)
    assert rendered == ("line1\nline2\nline3\tok\\u001b]0;owned\\u0007\\u202e")
    assert "\r" not in rendered
    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "\u202e" not in rendered


def _key_signature(value: Any) -> tuple[str, Any]:
    if isinstance(value, float):
        return ("real", struct.pack(">d", value))
    if isinstance(value, bytes):
        return ("blob", value)
    return (type(value).__name__, value)


def test_sqlite_mapping_key_codec_is_reversible_and_collision_free():
    reserved_text = SQLITE_KEY_PREFIX + "integer:7"
    keys = [
        "ordinary",
        "雪",
        reserved_text,
        None,
        False,
        7,
        -9,
        2.5,
        math.inf,
        -math.inf,
        float("nan"),
        b"\x00\xffblob",
        memoryview(b"memoryview-key"),
    ]
    encoded = [encode_json_safe_sqlite_key(key) for key in keys]
    assert encoded[0] == "ordinary"
    assert encoded[1] == "雪"
    assert encoded[2] != reserved_text
    assert len(encoded) == len(set(encoded))
    decoded = [decode_json_safe_sqlite_key(key) for key in encoded]
    assert [_key_signature(value) for value in decoded] == [
        _key_signature(key.tobytes() if isinstance(key, memoryview) else key)
        for key in keys
    ]

    mapping = {
        "1": "ordinary-string",
        1: "integer",
        reserved_text: "escaped-reserved-string",
        b"1": "blob",
    }
    public = json_safe_sqlite(mapping)
    assert public["1"] == "ordinary-string"
    assert len(public) == len(mapping)
    assert {
        _key_signature(decode_json_safe_sqlite_key(key)): value
        for key, value in public.items()
    } == {_key_signature(key): value for key, value in mapping.items()}
    json.dumps(public, ensure_ascii=False, allow_nan=False)

    with pytest.raises(TypeError, match="unsupported public SQLite mapping key"):
        json_safe_sqlite({("tuple",): "unsupported"})
    first_nan = float("nan")
    second_nan = float("nan")
    with pytest.raises(ValueError, match="mapping key collision"):
        json_safe_sqlite({first_nan: "one", second_nan: "two"})


def test_stats_tier_keys_round_trip_all_sqlite_types_across_public_surfaces(
    provenance_env, monkeypatch
):
    db_path = provenance_env / "namespaces" / "default.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("DROP TABLE events")
    conn.execute(
        """
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            idempotency_key,
            session_id,
            ts,
            event_time,
            role,
            content,
            tool_name,
            tool_input,
            tool_output,
            origin,
            tier,
            meta,
            provenance
        )
        """
    )
    tiers = [
        None,
        "episodic",
        7,
        2.5,
        math.inf,
        sqlite3.Binary(b"\x00\xfftier-blob"),
        memoryview(b"tier-memoryview"),
    ]
    for index, tier in enumerate(tiers):
        conn.execute(
            """
            INSERT INTO events(
                id, session_id, ts, event_time, role, content,
                origin, tier, meta, provenance
            ) VALUES (?, 'dynamic-session', ?, ?, 'user', ?, 'legacy', ?, '{}', NULL)
            """,
            (
                f"dynamic-tier-{index}",
                f"2026-01-01T00:00:{index:02d}+00:00",
                f"2026-01-01T00:00:{index:02d}+00:00",
                f"dynamic tier {index}",
                tier,
            ),
        )
    conn.commit()
    conn.close()

    with Store("default") as st:
        stats = st.stats()
    assert stats["tiers"]["episodic"] == 1
    decoded = {
        _key_signature(decode_json_safe_sqlite_key(key)): count
        for key, count in stats["tiers"].items()
    }
    expected_tiers = [
        None,
        "episodic",
        7,
        2.5,
        math.inf,
        b"\x00\xfftier-blob",
        b"tier-memoryview",
    ]
    assert decoded == {_key_signature(value): 1 for value in expected_tiers}
    json.dumps(stats, ensure_ascii=False, allow_nan=False)

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp = json.loads(mcp_server.memory_health(namespace="default"))
    assert mcp["stats"] == stats

    from tests.dashutil import make_dash_client

    client = make_dash_client()
    health = client.get("/api/namespace/default/health")
    namespace = client.get("/api/namespace/default")
    assert health.status_code == 200, health.text
    assert namespace.status_code == 200, namespace.text
    assert health.json()["stats"] == stats
    assert namespace.json()["stats"] == stats
    json.dumps(namespace.json(), ensure_ascii=False, allow_nan=False)


def test_recall_hit_dynamic_sqlite_values_are_exact_on_every_public_surface(
    provenance_env, monkeypatch
):
    blob = b"\x00\xffrecall-blob</script>\x80"
    direct = Hit(
        memory_id=blob,
        event_id=memoryview(blob),
        score=0.5,
        tier=math.inf,
        content=memoryview(blob),
        role=7,
        event_time=None,
        valid_from=2.5,
        valid_to=-math.inf,
        tool_name=sqlite3.Binary(blob),
        ts=float("nan"),
        origin=bytearray(blob),
    ).as_dict()
    assert _decode_public_blob(direct["memory_id"]) == blob
    assert _decode_public_blob(direct["event_id"]) == blob
    assert _decode_public_blob(direct["content"]) == blob
    assert _decode_public_blob(direct["tool_name"]) == blob
    assert _decode_public_blob(direct["origin"]) == blob
    assert direct["tier"] == {"encoding": "sqlite-real", "data": "+infinity"}
    assert direct["ts"] == {"encoding": "sqlite-real", "data": "nan"}
    json.dumps(direct, ensure_ascii=False, allow_nan=False)

    query = "DYNAMIC-RECALL-CANARY-5f17"
    with Store("default") as st:
        observed = st.observe(query, defer_embedding=True)
        st.conn.execute(
            """
            UPDATE memories
               SET content=?, tier=?, valid_from=?
             WHERE id=?
            """,
            (sqlite3.Binary(blob), math.inf, sqlite3.Binary(blob), observed.memory_id),
        )
        st.conn.execute(
            """
            UPDATE events
               SET role=?, event_time=?, ts=?, tool_name=?, origin=?
             WHERE id=?
            """,
            (
                7,
                sqlite3.Binary(blob),
                math.inf,
                sqlite3.Binary(blob),
                sqlite3.Binary(blob),
                observed.event_id,
            ),
        )
        st.conn.commit()
        # Match public recall surfaces so E5's honest vector-stage evidence
        # describes one execution mode on every compared payload.
        python_hits = recall(query, store=st, include_residue=True)
    assert len(python_hits) == 1
    expected = python_hits[0].as_dict()
    assert _decode_public_blob(expected["content"]) == blob
    assert _decode_public_blob(expected["valid_from"]) == blob
    assert _decode_public_blob(expected["event_time"]) == blob
    assert _decode_public_blob(expected["tool_name"]) == blob
    assert _decode_public_blob(expected["origin"]) == blob
    # These columns have TEXT affinity, so SQLite canonically stores the
    # non-finite values as text.  The direct Hit above covers REAL values.
    assert expected["tier"] == "Inf"
    assert expected["ts"] == "Inf"
    json.dumps(expected, ensure_ascii=False, allow_nan=False)

    runner = CliRunner()
    cli_json = runner.invoke(
        app, ["recall", query, "--namespace", "default", "--include-residue", "--json"]
    )
    cli_human = runner.invoke(
        app, ["recall", query, "--namespace", "default", "--include-residue"]
    )
    assert cli_json.exit_code == 0, cli_json.output
    assert cli_human.exit_code == 0, cli_human.output
    assert json.loads(cli_json.stdout)["hits"] == [expected]
    assert "<sqlite-blob base64:" in cli_human.stdout
    assert "</script>" not in cli_human.stdout

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp = json.loads(
        mcp_server.memory_recall(query, namespace="default", include_residue=True)
    )
    assert mcp["hits"] == [expected]

    from tests.dashutil import make_dash_client

    client = make_dash_client()
    single = client.get(
        f"/api/namespace/default/recall?q={query}&include_residue=true"
    )
    all_namespaces = client.get(f"/api/recall?q={query}&include_residue=true")
    assert single.status_code == 200, single.text
    assert all_namespaces.status_code == 200, all_namespaces.text
    assert single.json()["hits"] == [{**expected, "namespace": "default"}]
    assert all_namespaces.json()["hits"] == [{**expected, "namespace": "default"}]
    payloads = [
        direct,
        expected,
        json.loads(cli_json.stdout),
        mcp,
        single.json(),
        all_namespaces.json(),
    ]
    assert not _contains_key(payloads, "confidence")
    json.dumps(payloads, ensure_ascii=False, allow_nan=False)


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


def test_null_transforms_round_trip_and_idempotency_distinguishes_omission(
    provenance_env,
):
    explicit_null = {**_import_envelope(), "transforms": None}
    omitted = dict(explicit_null)
    omitted.pop("transforms")
    with Store("default") as st:
        first = st.observe(
            "null transform import",
            origin="archive",
            provenance=explicit_null,
            idempotency_key="null-transforms-key",
            defer_embedding=True,
        )
        replay = st.observe(
            "null transform import",
            origin="archive",
            provenance=explicit_null,
            idempotency_key="null-transforms-key",
            defer_embedding=True,
        )
        with pytest.raises(ValueError, match="different provenance"):
            st.observe(
                "null transform import",
                origin="archive",
                provenance=omitted,
                idempotency_key="null-transforms-key",
                defer_embedding=True,
            )
        with pytest.raises(ValueError, match="different provenance"):
            st.observe(
                "null transform import",
                origin="archive",
                provenance={**explicit_null, "transforms": []},
                idempotency_key="null-transforms-key",
                defer_embedding=True,
            )
        outputs = [
            first.provenance,
            replay.provenance,
            st.get_memory(first.memory_id)["provenance"],
            st.browse_memories()["memories"][0]["provenance"],
            st.events()[0]["provenance"],
            st.trace(first.memory_id)["members"][0]["provenance"],
        ]
        assert st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        empty = st.observe(
            "known empty transform import",
            origin="archive",
            provenance={**explicit_null, "transforms": []},
            defer_embedding=True,
        )
    assert replay.deduplicated is True
    assert all(output["transforms"] is None for output in outputs)
    assert empty.provenance["transforms"] == []
    assert all(not _contains_key(output, "confidence") for output in outputs)


def test_null_transforms_round_trip_cli_mcp_and_dashboard(provenance_env, monkeypatch):
    cli_envelope = {**_import_envelope(channel="cli"), "transforms": None}
    cli_result = CliRunner().invoke(
        app,
        [
            "observe",
            "CLI null transforms",
            "-n",
            "default",
            "--origin",
            "archive",
            "--provenance-json",
            json.dumps(cli_envelope),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_provenance = json.loads(cli_result.stdout.split("provenance ", 1)[1])
    assert "transforms" in cli_provenance
    assert cli_provenance["transforms"] is None

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_envelope = {**_import_envelope(channel="mcp"), "transforms": None}
    mcp_payload = json.loads(
        mcp_server.memory_observe(
            "MCP null transforms",
            namespace="default",
            origin="archive",
            provenance=mcp_envelope,
        )
    )
    assert mcp_payload["ok"] is True
    assert "transforms" in mcp_payload["provenance"]
    assert mcp_payload["provenance"]["transforms"] is None

    from tests.dashutil import make_dash_client

    detail = make_dash_client().get(
        f"/api/namespace/default/memory/{mcp_payload['memory_id']}"
    )
    assert detail.status_code == 200, detail.text
    dashboard_provenance = detail.json()["provenance"]
    assert "transforms" in dashboard_provenance
    assert dashboard_provenance["transforms"] is None
    assert not _contains_key(dashboard_provenance, "confidence")


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
        ({**_import_envelope(), "transforms": "decode:utf-8"}, "array or null"),
        ({**_import_envelope(), "transforms": 0}, "array or null"),
        ({**_import_envelope(), "transforms": {}}, "array or null"),
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
    conn.execute("DROP TRIGGER IF EXISTS events_provenance_type_insert")
    conn.execute("DROP TRIGGER IF EXISTS events_provenance_type_update_of_provenance")
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


def test_v7_blob_origin_and_meta_migrate_losslessly_across_public_surfaces(
    provenance_env, monkeypatch
):
    blob_origin = b"\xff\x00legacy-origin</script><img src=x onerror=alert(1)>\x80"
    blob_meta = b"\x00\xfelegacy-meta</script><svg onload=alert(2)>\x81"
    procedure_origin = b"\x80\x00procedure-origin</script>\xff"
    procedure_meta = json.dumps(
        {
            "kind": "procedure",
            "name": "blob procedure",
            "trigger": "when testing blobs",
        }
    ).encode("utf-8")
    with Store("default") as st:
        legacy = st.observe("legacy blob event")
        procedure = st.procedure_write("blob procedure", "blob procedure body")
        st.conn.execute(
            "UPDATE events SET origin=?, meta=?, provenance=NULL WHERE id=?",
            (sqlite3.Binary(blob_origin), sqlite3.Binary(blob_meta), legacy.event_id),
        )
        st.conn.execute(
            "UPDATE events SET origin=?, meta=?, provenance=NULL WHERE id=?",
            (
                sqlite3.Binary(procedure_origin),
                sqlite3.Binary(procedure_meta),
                procedure.event_id,
            ),
        )
        st.conn.execute("UPDATE meta SET value='7' WHERE key='schema_version'")
        st.conn.commit()

    db_path = provenance_env / "namespaces" / "default.db"
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TRIGGER IF EXISTS events_provenance_type_insert")
    conn.execute("DROP TRIGGER IF EXISTS events_provenance_type_update_of_provenance")
    conn.execute("ALTER TABLE events DROP COLUMN provenance")
    conn.commit()
    conn.close()

    with Store("default") as st:
        detail = st.get_memory(legacy.memory_id)
        browse = st.browse_memories(limit=20)
        events = st.events(limit=20)
        trace = st.trace(legacy.memory_id)
        procedure_get = st.procedure_get("blob procedure")
        procedure_list = st.procedure_list()
        worldview = st.worldview()
        raw_rows = {
            row["id"]: row
            for row in st.conn.execute(
                "SELECT id, typeof(origin) AS origin_type, origin, "
                "typeof(meta) AS meta_type, meta FROM events "
                "WHERE id IN (?, ?)",
                (legacy.event_id, procedure.event_id),
            ).fetchall()
        }

    assert detail is not None
    legacy_provenance = detail["provenance"]
    assert legacy_provenance["kind"] == "legacy_unstructured"
    assert _decode_public_blob(legacy_provenance["origin"]) == blob_origin
    assert _decode_public_blob(legacy_provenance["meta"]) == blob_meta
    assert _decode_public_blob(detail["origin"]) == blob_origin
    assert _decode_public_blob(detail["meta"]) == blob_meta

    browsed = next(
        row for row in browse["memories"] if row["memory_id"] == legacy.memory_id
    )
    event = next(row for row in events if row["id"] == legacy.event_id)
    member = trace["members"][0]
    for output in (browsed, event, member):
        assert _decode_public_blob(output["origin"]) == blob_origin
        assert _decode_public_blob(output["provenance"]["origin"]) == blob_origin
        assert _decode_public_blob(output["provenance"]["meta"]) == blob_meta
    assert _decode_public_blob(event["meta"]) == blob_meta

    assert procedure_get is not None
    listed_procedure = next(
        row for row in procedure_list if row["name"] == "blob procedure"
    )
    worldview_procedure = next(
        row for row in worldview["procedures"] if row["name"] == "blob procedure"
    )
    for output in (procedure_get, listed_procedure, worldview_procedure):
        assert output["provenance"]["kind"] == "legacy_unstructured"
        assert _decode_public_blob(output["provenance"]["origin"]) == procedure_origin
        assert _decode_public_blob(output["provenance"]["meta"]) == procedure_meta

    for event_id, origin_bytes, meta_bytes in (
        (legacy.event_id, blob_origin, blob_meta),
        (procedure.event_id, procedure_origin, procedure_meta),
    ):
        raw = raw_rows[event_id]
        assert raw["origin_type"] == "blob"
        assert bytes(raw["origin"]) == origin_bytes
        assert raw["meta_type"] == "blob"
        assert bytes(raw["meta"]) == meta_bytes

    runner = CliRunner()
    cli_timeline = runner.invoke(
        app, ["timeline", "-n", "default", "--limit", "20", "--json"]
    )
    cli_trace = runner.invoke(app, ["trace", legacy.memory_id, "-n", "default"])
    cli_worldview = runner.invoke(app, ["worldview", "-n", "default", "--json"])
    cli_procedure = runner.invoke(
        app, ["procedure", "get", "blob procedure", "-n", "default"]
    )
    for result in (cli_timeline, cli_trace, cli_worldview, cli_procedure):
        assert result.exit_code == 0, result.output
        assert "</script>" not in result.output
        assert "<img" not in result.output
    cli_outputs = {
        "timeline": json.loads(cli_timeline.stdout),
        "trace": json.loads(cli_trace.stdout),
        "worldview": json.loads(cli_worldview.stdout),
        "procedure": cli_procedure.stdout,
    }

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_outputs = [
        json.loads(mcp_server.memory_timeline(namespace="default", limit=20)),
        json.loads(mcp_server.memory_trace(legacy.memory_id, namespace="default")),
        json.loads(mcp_server.memory_worldview(namespace="default")),
        json.loads(
            mcp_server.memory_procedure(
                "get", name="blob procedure", namespace="default"
            )
        ),
        json.loads(mcp_server.memory_procedure("list", namespace="default")),
    ]

    from tests.dashutil import make_dash_client

    client = make_dash_client()
    dashboard_responses = [
        client.get(f"/api/namespace/default/memory/{legacy.memory_id}"),
        client.get("/api/namespace/default/browse?limit=20"),
        client.get("/api/namespace/default/timeline?limit=20"),
        client.get("/api/namespace/default/worldview"),
        client.get("/api/namespace/default/procedures"),
    ]
    for response in dashboard_responses:
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/json"
        assert "</script>" not in response.text
        assert "<img" not in response.text
    index = client.get("/")
    assert index.status_code == 200
    assert "esc(JSON.stringify(d.provenance" in index.text
    assert "esc(r.origin" in index.text

    all_outputs = {
        "detail": detail,
        "browse": browse,
        "events": events,
        "trace": trace,
        "procedure_get": procedure_get,
        "procedure_list": procedure_list,
        "worldview": worldview,
        "cli": cli_outputs,
        "mcp": mcp_outputs,
        "dashboard": [response.json() for response in dashboard_responses],
    }
    serialized_outputs = json.dumps(all_outputs, ensure_ascii=False, allow_nan=False)
    for exact_bytes in (blob_origin, blob_meta, procedure_origin, procedure_meta):
        assert base64.b64encode(exact_bytes).decode("ascii") in serialized_outputs
    assert not _contains_key(all_outputs, "confidence")

    with Store("default") as st:
        after = st.conn.execute(
            "SELECT origin, meta FROM events WHERE id=?",
            (legacy.event_id,),
        ).fetchone()
    assert bytes(after["origin"]) == blob_origin
    assert bytes(after["meta"]) == blob_meta


def test_non_utf8_legacy_procedure_meta_fails_honest_without_read_surface_crash(
    provenance_env, monkeypatch
):
    opaque_meta = b"\x00\xffprocedure-meta</script><img onerror=alert(4)>\x80"
    with Store("default") as st:
        corrupt = st.procedure_write("corrupt procedure", "opaque procedure body")
        st.conn.execute(
            "UPDATE events SET meta=?, provenance=NULL WHERE id=?",
            (sqlite3.Binary(opaque_meta), corrupt.event_id),
        )
        st.conn.commit()

        detail = st.get_memory(corrupt.memory_id)
        trace = st.trace(corrupt.memory_id)
        procedures = st.procedure_list()
        worldview = st.worldview()
        assert st.procedure_get("corrupt procedure") is None
        raw_meta = st.conn.execute(
            "SELECT meta FROM events WHERE id=?", (corrupt.event_id,)
        ).fetchone()["meta"]

    assert detail is not None
    assert _decode_public_blob(detail["meta"]) == opaque_meta
    assert _decode_public_blob(detail["provenance"]["meta"]) == opaque_meta
    assert _decode_public_blob(trace["members"][0]["provenance"]["meta"]) == opaque_meta
    assert all(row["id"] != corrupt.memory_id for row in procedures)
    assert all(row["id"] != corrupt.memory_id for row in worldview["procedures"])

    runner = CliRunner()
    cli_get = runner.invoke(
        app, ["procedure", "get", "corrupt procedure", "-n", "default"]
    )
    assert cli_get.exit_code == 1, cli_get.output
    assert "not found: corrupt procedure" in cli_get.stdout
    assert "Traceback" not in cli_get.output
    cli_trace = runner.invoke(app, ["trace", corrupt.memory_id, "-n", "default"])
    assert cli_trace.exit_code == 0, cli_trace.output
    assert (
        _decode_public_blob(
            json.loads(cli_trace.stdout)["members"][0]["provenance"]["meta"]
        )
        == opaque_meta
    )

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_get = json.loads(
        mcp_server.memory_procedure(
            "get", name="corrupt procedure", namespace="default"
        )
    )
    assert mcp_get == {"ok": False, "error": "procedure 'corrupt procedure' not found"}
    mcp_list = json.loads(mcp_server.memory_procedure("list", namespace="default"))
    assert mcp_list["ok"] is True
    assert mcp_list["procedures"] == []
    assert (
        json.loads(mcp_server.memory_worldview(namespace="default"))["procedures"] == []
    )

    from tests.dashutil import make_dash_client

    client = make_dash_client()
    dashboard_detail = client.get(f"/api/namespace/default/memory/{corrupt.memory_id}")
    dashboard_namespace = client.get("/api/namespace/default")
    dashboard_procedures = client.get("/api/namespace/default/procedures")
    dashboard_worldview = client.get("/api/namespace/default/worldview")
    for response in (
        dashboard_detail,
        dashboard_namespace,
        dashboard_procedures,
        dashboard_worldview,
    ):
        assert response.status_code == 200, response.text
        assert "</script>" not in response.text
        assert "<img" not in response.text
    assert _decode_public_blob(dashboard_detail.json()["meta"]) == opaque_meta
    assert dashboard_procedures.json()["procedures"] == []
    assert dashboard_worldview.json()["procedures"] == []

    outputs = {
        "detail": detail,
        "trace": trace,
        "procedures": procedures,
        "worldview": worldview,
        "mcp": [mcp_get, mcp_list],
        "dashboard": [
            dashboard_detail.json(),
            dashboard_namespace.json(),
            dashboard_procedures.json(),
            dashboard_worldview.json(),
        ],
    }
    json.dumps(outputs, ensure_ascii=False, allow_nan=False)
    assert not _contains_key(outputs, "confidence")
    assert bytes(raw_meta) == opaque_meta


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


def test_provenance_storage_is_text_only_and_blob_replay_fails_closed(
    provenance_env,
):
    with Store("default") as st:
        result = st.observe(
            "typed provenance",
            idempotency_key="typed-provenance-key",
            defer_embedding=True,
        )
        row = st.conn.execute(
            "SELECT provenance, session_id, ts, event_time FROM events WHERE id=?",
            (result.event_id,),
        ).fetchone()
        encoded = row["provenance"].encode("utf-8")
        before_count = st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        with pytest.raises(sqlite3.IntegrityError, match="provenance must be text"):
            st.conn.execute(
                "UPDATE events SET provenance=? WHERE id=?",
                (sqlite3.Binary(encoded), result.event_id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="provenance must be text"):
            st.conn.execute(
                """
                INSERT INTO events(
                    id, session_id, ts, event_time, role, content,
                    origin, tier, meta, provenance
                ) VALUES (?, ?, ?, ?, 'user', '', 'python', 'episodic', '{}', ?)
                """,
                (
                    "blob-provenance-insert",
                    row["session_id"],
                    row["ts"],
                    row["event_time"],
                    sqlite3.Binary(encoded),
                ),
            )
        assert (
            st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == before_count
        )
        assert (
            st.conn.execute(
                "SELECT typeof(provenance) FROM events WHERE id=?", (result.event_id,)
            ).fetchone()[0]
            == "text"
        )

        # A triggerless/corrupt database remains readable, but bytes are never
        # guessed to be text even when they contain exact valid JSON.
        st.conn.execute("DROP TRIGGER events_provenance_type_update_of_provenance")
        st.conn.execute(
            "UPDATE events SET provenance=? WHERE id=?",
            (sqlite3.Binary(encoded), result.event_id),
        )
        st.conn.commit()
        detail = st.get_memory(result.memory_id)
        assert detail["provenance"] == {
            "schema_version": 1,
            "kind": "invalid_stored",
            "origin": "python",
        }
        assert (
            public_provenance(memoryview(encoded), origin="python", legacy_meta="{}")[
                "kind"
            ]
            == "invalid_stored"
        )
        with pytest.raises(ValueError, match="invalid stored provenance"):
            st.observe(
                "typed provenance",
                idempotency_key="typed-provenance-key",
                defer_embedding=True,
            )
        raw = st.conn.execute(
            "SELECT typeof(provenance), provenance FROM events WHERE id=?",
            (result.event_id,),
        ).fetchone()
        assert raw[0] == "blob" and bytes(raw[1]) == encoded

    with Store("default") as reopened:
        assert (
            reopened.get_memory(result.memory_id)["provenance"]["kind"]
            == "invalid_stored"
        )
        with pytest.raises(sqlite3.IntegrityError, match="provenance must be text"):
            reopened.conn.execute(
                "UPDATE events SET provenance=? WHERE id=?",
                (sqlite3.Binary(encoded), result.event_id),
            )


def test_invalid_stored_provenance_with_blob_origin_and_meta_is_json_safe(
    provenance_env, monkeypatch
):
    blob_origin = b"\xff\x00invalid-origin</script>\x80"
    blob_meta = b"\x00\xfeinvalid-meta<img src=x onerror=alert(9)>\x81"
    blob_tool = b"\x80\x00invalid-tool</script>\xff"
    invalid_provenance = b'{"schema_version":99,"kind":"native"}'
    with Store("default") as st:
        result = st.observe("invalid stored blob row")
        st.conn.execute("DROP TRIGGER events_provenance_type_update_of_provenance")
        st.conn.execute(
            "UPDATE events SET origin=?, meta=?, tool_name=?, provenance=? WHERE id=?",
            (
                sqlite3.Binary(blob_origin),
                sqlite3.Binary(blob_meta),
                sqlite3.Binary(blob_tool),
                sqlite3.Binary(invalid_provenance),
                result.event_id,
            ),
        )
        st.conn.commit()
        outputs = {
            "detail": st.get_memory(result.memory_id),
            "browse": st.browse_memories(),
            "events": st.events(),
            "trace": st.trace(result.memory_id),
        }
        raw = st.conn.execute(
            "SELECT origin, meta, tool_name, provenance FROM events WHERE id=?",
            (result.event_id,),
        ).fetchone()

    detail = outputs["detail"]
    assert detail["provenance"]["kind"] == "invalid_stored"
    assert _decode_public_blob(detail["provenance"]["origin"]) == blob_origin
    assert _decode_public_blob(detail["origin"]) == blob_origin
    assert _decode_public_blob(detail["meta"]) == blob_meta
    assert _decode_public_blob(detail["tool_name"]) == blob_tool
    json.dumps(outputs, ensure_ascii=False, allow_nan=False)
    assert not _contains_key(outputs, "confidence")

    cli = CliRunner().invoke(
        app, ["timeline", "-n", "default", "--limit", "20", "--json"]
    )
    assert cli.exit_code == 0, cli.output
    cli_event = next(
        row for row in json.loads(cli.stdout)["events"] if row["id"] == result.event_id
    )
    assert _decode_public_blob(cli_event["origin"]) == blob_origin
    assert _decode_public_blob(cli_event["meta"]) == blob_meta
    assert _decode_public_blob(cli_event["tool_name"]) == blob_tool

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_event = next(
        row
        for row in json.loads(
            mcp_server.memory_timeline(namespace="default", limit=20)
        )["events"]
        if row["id"] == result.event_id
    )
    assert mcp_event == cli_event

    from tests.dashutil import make_dash_client

    client = make_dash_client()
    for path in (
        f"/api/namespace/default/memory/{result.memory_id}",
        "/api/namespace/default/browse",
        "/api/namespace/default/timeline",
    ):
        response = client.get(path)
        assert response.status_code == 200, response.text
        assert "</script>" not in response.text
        assert "<img" not in response.text
    assert bytes(raw["origin"]) == blob_origin
    assert bytes(raw["meta"]) == blob_meta
    assert bytes(raw["tool_name"]) == blob_tool
    assert bytes(raw["provenance"]) == invalid_provenance


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


def test_purge_erases_blob_provenance_canary_and_its_public_base64_form(
    provenance_env, monkeypatch
):
    blob_canary = b"BLOB-PURGE-CANARY-9d71\x00\xff</script><img onerror=alert(7)>"
    encoded_canary = base64.b64encode(blob_canary).decode("ascii")
    public_blob = json_safe_sqlite(blob_canary)
    human_blob = human_display(public_blob, sqlite_scalar=True)
    spaced_envelope = json.dumps(public_blob, ensure_ascii=False)
    compact_envelope = json.dumps(
        public_blob, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    encoded_key = encode_json_safe_sqlite_key(blob_canary)
    with Store("default") as st:
        target = st.observe("blob purge target", session_id="shared-blob-purge-session")
        unrelated = st.observe(
            "unrelated blob purge memory must survive",
            session_id="shared-blob-purge-session",
            origin="unrelated-safe-origin",
        )
        correction = st.contradict(
            target.memory_id,
            replacement="blob purge survivor",
            idempotency_key="blob-purge-correction",
            session_id="shared-blob-purge-session",
        )
        survivor_id = correction["replacement_memory_id"]
        st.conn.execute(
            "UPDATE sessions SET source=?, meta=? WHERE id=?",
            (
                "unrelated-safe-source",
                json.dumps(
                    {
                        "keep": "unrelated-safe-meta",
                        "raw_hex": blob_canary.hex(),
                        "base64": encoded_canary,
                        "human_marker": human_blob,
                        "spaced_envelope": spaced_envelope,
                        "compact_envelope": compact_envelope,
                        "nested_public_envelope": public_blob,
                        encoded_key: "encoded-key-secret",
                    },
                    ensure_ascii=False,
                ),
                "shared-blob-purge-session",
            ),
        )
        st.conn.execute(
            "UPDATE events SET origin=?, meta=?, provenance=NULL WHERE id=?",
            (
                sqlite3.Binary(blob_canary),
                sqlite3.Binary(blob_canary),
                target.event_id,
            ),
        )
        st.conn.commit()
        before = st.get_memory(target.memory_id)
        assert before is not None
        assert before["provenance"]["origin"]["data"] == encoded_canary
        purge = st.purge(target.memory_id)
        assert purge["ok"] is True
        unrelated_after = st.get_memory(unrelated.memory_id)
        assert unrelated_after["content"] == "unrelated blob purge memory must survive"
        assert unrelated_after["origin"] == "unrelated-safe-origin"
        safe_session = st.conn.execute(
            "SELECT source, meta FROM sessions WHERE id=?",
            (unrelated_after["session_id"],),
        ).fetchone()
        assert safe_session["source"] == "unrelated-safe-source"
        safe_meta = json.loads(safe_session["meta"])
        assert safe_meta["keep"] == "unrelated-safe-meta"
        surfaces = {
            "detail": st.get_memory(survivor_id),
            "unrelated": unrelated_after,
            "browse": st.browse_memories(),
            "events": st.events(),
            "trace": st.trace(survivor_id),
            "worldview": st.worldview(),
            "session_meta": safe_meta,
        }
        raw_values: list[Any] = []
        for table_row in st.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            table = table_row["name"]
            if table.replace("_", "").isalnum():
                raw_values.extend(
                    value
                    for row in st.conn.execute(f'SELECT * FROM "{table}"')
                    for value in tuple(row)
                )

    serialized = json.dumps(surfaces, ensure_ascii=False, allow_nan=False)
    assert encoded_canary not in serialized
    assert blob_canary.hex() not in serialized
    assert human_blob not in serialized
    assert spaced_envelope not in serialized
    assert compact_envelope not in serialized
    assert encoded_key not in serialized
    assert "BLOB-PURGE-CANARY-9d71" not in serialized
    for value in raw_values:
        if isinstance(value, bytes):
            assert blob_canary not in value
            assert encoded_canary.encode("ascii") not in value
        elif isinstance(value, str):
            assert "BLOB-PURGE-CANARY-9d71" not in value
            assert encoded_canary not in value
    assert not _contains_key(surfaces, "confidence")

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_surfaces = json.dumps(
        [
            json.loads(mcp_server.memory_trace(survivor_id, namespace="default")),
            json.loads(mcp_server.memory_timeline(namespace="default")),
            json.loads(mcp_server.memory_worldview(namespace="default")),
        ],
        ensure_ascii=False,
        allow_nan=False,
    )
    assert encoded_canary not in mcp_surfaces
    assert blob_canary.hex() not in mcp_surfaces
    assert encoded_key not in mcp_surfaces
    from tests.dashutil import make_dash_client

    client = make_dash_client()
    dashboard_responses = [
        client.get(f"/api/namespace/default/memory/{survivor_id}"),
        client.get(f"/api/namespace/default/memory/{unrelated.memory_id}"),
        client.get("/api/namespace/default/browse"),
        client.get("/api/namespace/default/timeline"),
        client.get("/api/namespace/default/worldview"),
    ]
    assert all(response.status_code == 200 for response in dashboard_responses)
    dashboard_surfaces = json.dumps(
        [response.json() for response in dashboard_responses],
        ensure_ascii=False,
        allow_nan=False,
    )
    assert encoded_canary not in dashboard_surfaces
    assert blob_canary.hex() not in dashboard_surfaces
    assert encoded_key not in dashboard_surfaces


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
    mcp_payload = json.loads(mcp_server.memory_timeline(namespace="default", limit=20))
    assert mcp_payload["events"] == payload["events"]

    from tests.dashutil import make_dash_client

    dashboard = make_dash_client().get("/api/namespace/default/timeline?limit=20")
    assert dashboard.status_code == 200, dashboard.text
    assert dashboard.json()["events"] == payload["events"]


def test_cli_human_timeline_handles_legacy_blob_fields_and_stays_bounded(
    provenance_env,
):
    ordinary_time = "2026-01-02T03:04:05.000000+00:00"
    blob_event_time = b"\xff\x00time</script>\x1b[2J\x80"
    blob_content = b"\x00\xfecontent</script>\x1b[31m\x81" * 2000
    memoryview_origin = b"\x80\x00origin</script>\x1b[5m\xff"
    with Store("default") as st:
        ordinary = st.observe(
            "ordinary timeline text",
            event_time=ordinary_time,
            defer_embedding=True,
        )
        dynamic = st.observe("dynamic timeline placeholder", defer_embedding=True)
        control = st.observe("text </script> \x1b[33m", defer_embedding=True)
        st.conn.execute(
            "UPDATE events SET event_time=?, role=?, content=?, tool_name=?, "
            "origin=?, tier=?, provenance=NULL WHERE id=?",
            (
                sqlite3.Binary(blob_event_time),
                7,
                sqlite3.Binary(blob_content),
                math.inf,
                memoryview(memoryview_origin),
                2.5,
                dynamic.event_id,
            ),
        )
        st.conn.execute(
            "UPDATE events SET role=? WHERE id=?",
            ("assistant\x1b[31m", control.event_id),
        )
        st.conn.commit()
        store_rows = st.events(limit=20)
        raw = st.conn.execute(
            "SELECT typeof(event_time), event_time, typeof(content), content, "
            "typeof(origin), origin, typeof(tool_name), tool_name "
            "FROM events WHERE id=?",
            (dynamic.event_id,),
        ).fetchone()

    runner = CliRunner()
    human = runner.invoke(app, ["timeline", "-n", "default", "--limit", "20"])
    assert human.exit_code == 0, human.output
    assert "ordinary timeline text" in human.stdout
    ordinary_line = (
        f"{format_iso(ordinary_time)}  {'user':<10} {'episodic':<12} "
        f"{ordinary.event_id}  source=python/python  ordinary timeline text"
    )
    assert ordinary_line in human.stdout
    assert "<sqlite-blob base64:" in human.stdout
    assert "[tool:Inf]" in human.stdout
    assert "7          2.5" in human.stdout
    assert "\\u001b" in human.stdout
    assert "\x1b" not in human.stdout
    assert len(human.stdout) < 2500

    machine = runner.invoke(
        app, ["timeline", "-n", "default", "--limit", "20", "--json"]
    )
    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.stdout)
    assert payload == {"namespace": "default", "events": store_rows}
    dynamic_event = next(
        event for event in payload["events"] if event["id"] == dynamic.event_id
    )
    assert _decode_public_blob(dynamic_event["event_time"]) == blob_event_time
    assert dynamic_event["role"] == "7"
    assert _decode_public_blob(dynamic_event["content"]) == blob_content
    # TEXT affinity honestly preserves SQLite's coercion of a bound infinity.
    assert dynamic_event["tool_name"] == "Inf"
    assert _decode_public_blob(dynamic_event["origin"]) == memoryview_origin
    assert dynamic_event["tier"] == "2.5"
    ordinary_event = next(
        event for event in payload["events"] if event["id"] == ordinary.event_id
    )
    assert ordinary_event["tool_name"] is None
    assert not _contains_key(payload, "confidence")

    assert raw[0] == "blob" and bytes(raw[1]) == blob_event_time
    assert raw[2] == "blob" and bytes(raw[3]) == blob_content
    assert raw[4] == "blob" and bytes(raw[5]) == memoryview_origin
    assert raw[6] == "text" and raw[7] == "Inf"


def test_cli_human_timeline_accepts_all_json_safe_sqlite_value_shapes(
    provenance_env, monkeypatch
):
    blob = b"\x00\xff</script>\x1b[2J\x80" * 500
    blob_envelope = json_safe_sqlite(blob)
    memoryview_envelope = json_safe_sqlite(memoryview(b"memoryview\x00\xff"))
    values = [
        None,
        "ordinary text",
        7,
        2.5,
        {"encoding": "sqlite-real", "data": "+infinity"},
        blob_envelope,
        memoryview_envelope,
        {"nested": [1, None, "text"]},
        "control\x1b[31m",
    ]
    rows = [
        {
            "id": f"event-{index}",
            "event_time": value,
            "role": value,
            "tier": value,
            "content": value,
            "tool_name": None,
            "provenance": {
                "schema_version": 1,
                "kind": "legacy_unstructured",
                "origin": value,
                "meta": None,
            },
        }
        for index, value in enumerate(values)
    ]

    class FakeStore:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def events(self, **_kwargs):
            return rows

    import haunt.cli as cli_module

    monkeypatch.setattr(cli_module, "open_existing", lambda _namespace: FakeStore())
    runner = CliRunner()
    human = runner.invoke(app, ["timeline", "-n", "default", "--limit", "20"])
    assert human.exit_code == 0, human.output
    assert "ordinary text" in human.stdout
    assert "null" in human.stdout
    assert "7" in human.stdout
    assert "2.5" in human.stdout
    assert "<sqlite-real +infinity>" in human.stdout
    assert "<sqlite-blob base64:" in human.stdout
    assert "\\u001b" in human.stdout
    assert "\x1b" not in human.stdout
    assert "</script>" not in human.stdout
    assert len(human.stdout) < 5000

    machine = runner.invoke(
        app, ["timeline", "-n", "default", "--limit", "20", "--json"]
    )
    assert machine.exit_code == 0, machine.output
    assert json.loads(machine.stdout) == {"namespace": "default", "events": rows}


def test_cli_human_worldview_and_procedure_accept_serialized_blob_fields(
    provenance_env,
):
    fact_id_blob = b"\xff\x00fact-id</script>\x80"
    fact_content_blob = b"\x00\xfefact-content</script>\x1b[2J\x81" * 1000
    procedure_id_blob = b"\x80\x00procedure-id</script>\xff"
    procedure_body_blob = b"\x00\xffprocedure-body</script>\x1b[31m\x80" * 1000
    with Store("default") as st:
        fact = st.observe("fact placeholder", tier="semantic", defer_embedding=True)
        procedure = st.procedure_write("blob body procedure", "body placeholder")
        st.conn.execute(
            "DELETE FROM embedding_jobs WHERE memory_id IN (?, ?)",
            (fact.memory_id, procedure.memory_id),
        )
        st.conn.execute(
            "UPDATE memories SET id=?, content=? WHERE id=?",
            (
                memoryview(fact_id_blob),
                sqlite3.Binary(fact_content_blob),
                fact.memory_id,
            ),
        )
        st.conn.execute(
            "UPDATE memories SET id=?, content=? WHERE id=?",
            (
                sqlite3.Binary(procedure_id_blob),
                memoryview(procedure_body_blob),
                procedure.memory_id,
            ),
        )
        st.conn.commit()
        worldview = st.worldview()
        procedure_get = st.procedure_get("blob body procedure")
        raw_rows = st.conn.execute(
            "SELECT id, content FROM memories WHERE typeof(id)='blob' ORDER BY rowid"
        ).fetchall()

    assert procedure_get is not None
    runner = CliRunner()
    human_worldview = runner.invoke(app, ["worldview", "-n", "default"])
    machine_worldview = runner.invoke(app, ["worldview", "-n", "default", "--json"])
    human_get = runner.invoke(
        app, ["procedure", "get", "blob body procedure", "-n", "default"]
    )
    human_list = runner.invoke(app, ["procedure", "list", "-n", "default"])
    for result in (human_worldview, machine_worldview, human_get, human_list):
        assert result.exit_code == 0, result.output
        assert "\x1b" not in result.output
    assert "<sqlite-" in human_worldview.stdout
    assert "<sqlite-blob base64:" in human_get.stdout
    assert "<sqlite-blob" in human_list.stdout
    assert len(human_worldview.stdout) < 1500
    assert len(human_get.stdout) < 9000
    assert len(human_list.stdout) < 5000

    machine = json.loads(machine_worldview.stdout)
    assert machine == worldview
    fact_output = next(row for row in machine["facts"] if isinstance(row["id"], dict))
    assert _decode_public_blob(fact_output["id"]) == fact_id_blob
    assert _decode_public_blob(fact_output["content"]) == fact_content_blob
    procedure_output = next(
        row for row in machine["procedures"] if row["name"] == "blob body procedure"
    )
    assert _decode_public_blob(procedure_output["id"]) == procedure_id_blob
    assert _decode_public_blob(procedure_get["body"]) == procedure_body_blob
    assert not _contains_key(machine, "confidence")

    assert bytes(raw_rows[0]["id"]) == fact_id_blob
    assert bytes(raw_rows[0]["content"]) == fact_content_blob
    assert bytes(raw_rows[1]["id"]) == procedure_id_blob
    assert bytes(raw_rows[1]["content"]) == procedure_body_blob


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
    assert mcp_listed["legacy procedure"]["provenance"]["kind"] == "legacy_unstructured"
    assert mcp_listed["invalid procedure"]["provenance"]["kind"] == "invalid_stored"
    mcp_worldview = json.loads(mcp_server.memory_worldview(namespace="default"))
    mcp_worldview_procedures = {row["name"]: row for row in mcp_worldview["procedures"]}
    for name, expected in (
        ("native procedure", native_get),
        ("imported procedure", imported_get),
        ("legacy procedure", legacy_get),
        ("invalid procedure", invalid_get),
    ):
        assert mcp_worldview_procedures[name]["provenance"] == expected["provenance"]

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
    cli_worldview = runner.invoke(app, ["worldview", "-n", "default", "--json"])
    assert cli_worldview.exit_code == 0, cli_worldview.output
    cli_worldview_procedures = {
        row["name"]: row for row in json.loads(cli_worldview.stdout)["procedures"]
    }
    assert (
        cli_worldview_procedures["legacy procedure"]["provenance"]["kind"]
        == "legacy_unstructured"
    )


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


@pytest.mark.parametrize("invalid_transforms", ["decode:utf-8", 0, {}])
def test_cli_and_mcp_reject_non_list_non_null_transforms_without_writes(
    provenance_env, monkeypatch, invalid_transforms
):
    cli_bad = {
        **_import_envelope(channel="cli"),
        "transforms": invalid_transforms,
    }
    cli = CliRunner().invoke(
        app,
        [
            "observe",
            "bad CLI transforms",
            "-n",
            "default",
            "--provenance-json",
            json.dumps(cli_bad),
        ],
    )
    assert cli.exit_code == 2
    assert "array or null" in cli.output

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    from haunt import mcp_server

    mcp_server._MCP_AUTHORITY = None
    mcp_bad = {
        **_import_envelope(channel="mcp"),
        "transforms": invalid_transforms,
    }
    mcp = json.loads(
        mcp_server.memory_observe(
            "bad MCP transforms",
            namespace="default",
            provenance=mcp_bad,
        )
    )
    assert mcp["ok"] is False
    assert "array or null" in mcp["error"]
    with Store("default") as st:
        assert _logical_counts(st) == {table: 0 for table in LOGICAL_TABLES}


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
        assert (
            st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            == count_after_mcp
        )
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
        assert (
            st.get_memory(cli_payload["replacement_memory_id"])["provenance"]["channel"]
            == "cli"
        )
        assert (
            st.get_memory(mcp_payload["replacement_memory_id"])["provenance"]["channel"]
            == "mcp"
        )
        assert (
            st.get_memory(dashboard_payload["replacement_memory_id"])["provenance"][
                "channel"
            ]
            == "dashboard"
        )
