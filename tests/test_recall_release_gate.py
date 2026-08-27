"""Adversarial release checks for read-only, residue-safe recall.

These are intentionally FTS-only fixtures: CI must not require a model cache,
network access, or a platform-specific embedding backend to prove recall's
storage and selection contracts.
"""

from __future__ import annotations

import importlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from haunt import embed
from haunt.cli import app
from haunt.paths import NamespacePathError, registry_path
from haunt.recall import recall
from haunt.planner import planned_recall
from haunt.store import (
    Store,
    change_namespace_label,
    ensure_vec_table,
    namespace_exists_readonly,
    open_existing_readonly,
    open_namespace_identity_readonly,
)


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int, int, int]]:
    """Exact source-file state, including fields a read must not touch."""
    out: dict[str, tuple[bytes, int, int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            out[str(path.relative_to(root))] = (
                path.read_bytes(),
                stat.st_mode & 0o777,
                stat.st_ino,
                stat.st_mtime_ns,
            )
    return out


def _data_version(path: Path) -> int:
    """Use Haunt's zero-write opener even while measuring the fixture."""
    from haunt.store import _open_zero_write_sqlite_snapshot

    conn = _open_zero_write_sqlite_snapshot(path)
    try:
        return int(conn.execute("PRAGMA data_version").fetchone()[0])
    finally:
        conn.close()


@pytest.fixture
def recall_gate_home(tmp_path, monkeypatch):
    home = tmp_path / "haunt-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "release-gate")
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    monkeypatch.delenv("HAUNT_OFFLINE", raising=False)
    embed.reset()
    with Store("release-gate") as store:
        eligible = store.observe("ELIGIBLE-RECALL-CANARY orbital archive", defer_embedding=True)
        tool = store.observe(
            "TOOL-RECALL-CANARY orbital archive",
            tool_name="shell",
            tool_input="orbital archive",
            defer_embedding=True,
        )
        task = store.observe(
            "TASK-RECALL-CANARY orbital archive",
            recall_class="task",
            defer_embedding=True,
        )
        legacy = store.observe("LEGACY-RECALL-CANARY orbital archive", defer_embedding=True)
    yield home, eligible, tool, task, legacy
    embed.reset()


def test_ranked_recall_filters_classes_and_correction_preserves_task_class(
    recall_gate_home,
):
    _, eligible, tool, task, legacy = recall_gate_home
    default = recall("orbital archive", namespace="release-gate", use_vectors=False)
    assert {hit.memory_id for hit in default} == {eligible.memory_id, legacy.memory_id}
    assert all(hit.recall_class is None for hit in default)
    assert default.execution["residue_filter"] == "applied"
    assert default.execution["read_only"] is True
    assert default.execution["maintenance_performed"] is False
    assert all(hit.filter_context["residue_filter"] == "applied" for hit in default)

    included = recall(
        "orbital archive", namespace="release-gate", use_vectors=False,
        include_residue=True,
    )
    classes = {hit.memory_id: hit.recall_class for hit in included}
    assert classes[tool.memory_id] == "tool"
    assert classes[task.memory_id] == "task"
    assert classes[eligible.memory_id] is None
    assert all(hit.filter_context["residue_filter"] == "bypassed" for hit in included)

    # Compatibility is explicit, and the modern flag has unambiguous priority.
    old_alias = recall(
        "orbital archive", namespace="release-gate", use_vectors=False,
        include_untrusted=True,
    )
    assert {hit.memory_id for hit in old_alias} == set(classes)
    assert old_alias[0].filter_context["residue_filter_source"] == "deprecated_include_untrusted"
    modern_wins = recall(
        "orbital archive", namespace="release-gate", use_vectors=False,
        include_residue=False, include_untrusted=True,
    )
    assert {hit.memory_id for hit in modern_wins} == {eligible.memory_id, legacy.memory_id}
    assert modern_wins[0].filter_context["residue_filter_source"] == "include_residue"

    with Store("release-gate", create=False) as store:
        correction = store.contradict(
            task.memory_id,
            idempotency_key="task-class-preservation",
            replacement="TASK-REPLACEMENT-CANARY orbital archive",
        )
        replacement_id = correction["replacement_memory_id"]
        row = store.get_memory(replacement_id)
        assert row is not None
        assert row["recall_class"] == "task"
    after_correction = recall(
        "TASK-REPLACEMENT-CANARY", namespace="release-gate", use_vectors=False,
        include_residue=True,
    )
    assert [hit.memory_id for hit in after_correction] == [replacement_id]
    assert after_correction[0].recall_class == "task"
    assert not recall(
        "TASK-REPLACEMENT-CANARY", namespace="release-gate", use_vectors=False
    )


def test_raw_tool_class_validation_is_before_writes(recall_gate_home):
    home, *_ = recall_gate_home
    with Store("release-gate", create=False) as store:
        before_rejection = _tree_snapshot(home)
        with pytest.raises(ValueError, match="raw tool structure"):
            store.observe(
                "contradictory class", tool_name="shell", recall_class="task"
            )
        assert _tree_snapshot(home) == before_rejection
        result = store.observe("auto tool", tool_output="ok", defer_embedding=True)
        assert result.recall_class == "tool"
        role_only = store.observe("tool role only", role="tool", defer_embedding=True)
        assert role_only.recall_class == "tool"
    # The valid observation changed the store; the contradiction itself created
    # no stray session/event/job, as there is exactly one additional event.
    with Store("release-gate", create=False) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 6

    cli = CliRunner().invoke(
        app,
        [
            "observe", "surface tool", "-n", "release-gate", "--tool-output", "ok",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert "recall_class=tool" in cli.output

    try:
        from haunt.mcp_server import memory_observe
    except ImportError:
        memory_observe = None
    if memory_observe is not None:
        auto_tool = json.loads(
            memory_observe(text="mcp auto tool", tool_output="ok")
        )
        explicit_task = json.loads(
            memory_observe(text="mcp actual task", recall_class="task")
        )
        assert auto_tool["ok"] is True and auto_tool["recall_class"] == "tool"
        assert explicit_task["ok"] is True and explicit_task["recall_class"] == "task"


def test_no_text_guessing_and_only_host_lifecycle_stamps_task(recall_gate_home):
    """Words that look like residue stay eligible; known lifecycle seams do not."""
    _, _, _, _, _ = recall_gate_home
    with Store("release-gate", create=False) as store:
        literal = store.observe(
            "TASK SessionStart tool call text is ordinary quoted prose",
            defer_embedding=True,
        )
        assert literal.recall_class is None

    # These are the two host entry points that have actual lifecycle knowledge.
    from haunt.claude_hook import run as run_claude
    from haunt.cursor_hook import run as run_cursor

    assert run_cursor(
        json.dumps(
            {
                "hook_event_name": "sessionStart",
                "conversation_id": "cursor-lifecycle-class",
            }
        )
    )
    assert run_claude(
        json.dumps(
            {
                "hook_event_name": "SessionStart",
                "session_id": "claude-lifecycle-class",
            }
        )
    )
    with Store("release-gate", create=False) as store:
        classes = store.conn.execute(
            """
            SELECT session_id, recall_class FROM events
            WHERE session_id IN ('cursor-lifecycle-class', 'claude-lifecycle-class')
            ORDER BY session_id
            """
        ).fetchall()
        assert [(row["session_id"], row["recall_class"]) for row in classes] == [
            ("claude-lifecycle-class", "task"),
            ("cursor-lifecycle-class", "task"),
        ]
    assert literal.memory_id in {
        hit.memory_id
        for hit in recall(
            "TASK SessionStart tool call text", namespace="release-gate", use_vectors=False
        )
    }
    assert not recall(
        "haunt session start", namespace="release-gate", use_vectors=False
    )


def test_legacy_raw_tool_correction_preserves_effective_tool_class(recall_gate_home):
    """Migrating v8 must not let correcting raw tool data launder it into recall."""
    home, _, tool, _, _ = recall_gate_home
    db = home / "namespaces" / "release-gate.db"
    # A historical v8 source had raw tool structure but no recall_class column.
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE events DROP COLUMN recall_class")
        conn.execute("UPDATE meta SET value='8' WHERE key='schema_version'")
        conn.commit()
    with Store("release-gate", create=False) as store:
        corrected = store.contradict(
            tool.memory_id,
            idempotency_key="legacy-raw-tool-correction",
            replacement="LEGACY-RAW-TOOL-REPLACEMENT-CANARY orbital archive",
        )
        replacement = corrected["replacement_memory_id"]
        assert store.get_memory(replacement)["recall_class"] == "tool"
    assert not recall(
        "LEGACY-RAW-TOOL-REPLACEMENT-CANARY",
        namespace="release-gate",
        use_vectors=False,
    )
    included = recall(
        "LEGACY-RAW-TOOL-REPLACEMENT-CANARY",
        namespace="release-gate",
        use_vectors=False,
        include_residue=True,
    )
    assert [hit.memory_id for hit in included] == [replacement]
    assert included[0].recall_class == "tool"


def test_cli_dashboard_and_detail_expose_residue_controls(recall_gate_home):
    _, eligible, tool, task, legacy = recall_gate_home
    default = CliRunner().invoke(
        app, ["recall", "orbital archive", "-n", "release-gate", "--json"]
    )
    assert default.exit_code == 0, default.output
    assert {item["memory_id"] for item in json.loads(default.stdout)["hits"]} == {
        eligible.memory_id, legacy.memory_id
    }
    included = CliRunner().invoke(
        app,
        ["recall", "orbital archive", "-n", "release-gate", "--include-residue", "--json"],
    )
    assert included.exit_code == 0, included.output
    by_id = {item["memory_id"]: item for item in json.loads(included.stdout)["hits"]}
    assert by_id[tool.memory_id]["recall_class"] == "tool"
    assert by_id[task.memory_id]["recall_class"] == "task"

    from tests.dashutil import make_dash_client

    client = make_dash_client()
    assert 'id="includeResidue"' in client.get("/").text
    dashboard = client.get(
        "/api/namespace/release-gate/recall",
        params={"q": "orbital archive", "include_residue": "true"},
    )
    assert dashboard.status_code == 200
    dashboard_ids = {item["memory_id"] for item in dashboard.json()["hits"]}
    assert {tool.memory_id, task.memory_id} <= dashboard_ids
    detail = client.get(f"/api/namespace/release-gate/memory/{tool.memory_id}")
    assert detail.status_code == 200
    assert detail.json()["recall_class"] == "tool"

    try:
        from haunt.mcp_server import memory_recall
    except ImportError:
        memory_recall = None
    if memory_recall is not None:
        payload = json.loads(
            memory_recall(query="orbital archive", include_residue=True)
        )
        returned = {item["memory_id"]: item for item in payload["hits"]}
        assert returned[tool.memory_id]["recall_class"] == "tool"
        assert returned[task.memory_id]["recall_class"] == "task"


def test_temporal_timeline_and_union_label_residue_semantics(recall_gate_home):
    _, _, tool, _, _ = recall_gate_home
    with open_existing_readonly("release-gate") as store:
        bare = planned_recall(
            "what happened on 2026-01-01", now=None, store=store, k=8
        )
        # The fixture's current timestamps are outside this window, but the
        # top-level contract is present even for a zero-hit timeline.
        assert bare.execution["strategy"] == "timeline"
        assert bare.execution["residue_filter"] == "not_applicable"

        union = planned_recall(
            "orbital archive on 2026-01-01", now=None, store=store, k=8,
            strategy="union",
        )
        assert union.execution["strategy"] == "union"
        assert union.execution["residue_filter"] == "mixed"
        assert union.execution["components"]["timeline"]["residue_filter"] == "not_applicable"
        assert union.execution["components"]["recall"]["residue_filter"] == "applied"
        assert tool.memory_id not in {hit.memory_id for hit in union}


def test_residue_remains_reachable_in_events_timeline_trace_and_detail(
    recall_gate_home,
):
    """Filtering affects ranked recall only; audit/readback paths stay intact."""
    _, _, _, _, _ = recall_gate_home
    with Store("release-gate", create=False) as store:
        raw_tool = store.observe(
            "AUDIT-TIMELINE-TOOL-CANARY",
            role="tool",
            tool_name="shell",
            event_time="2026-01-15T12:00:00+00:00",
            defer_embedding=True,
        )
        task = store.observe(
            "AUDIT-TIMELINE-TASK-CANARY",
            recall_class="task",
            event_time="2026-01-15T13:00:00+00:00",
            defer_embedding=True,
        )
        events = store.events(
            since="2026-01-15T00:00:00+00:00",
            until="2026-01-15T23:59:59+00:00",
        )
        event_classes = {row["id"]: row["recall_class"] for row in events}
        assert event_classes[raw_tool.event_id] == "tool"
        assert event_classes[task.event_id] == "task"
        detail = store.get_memory(raw_tool.memory_id)
        assert detail is not None and detail["recall_class"] == "tool"
        trace = store.trace(task.memory_id)
        assert trace["ok"] is True
        assert trace["members"][0]["memory_id"] == task.memory_id

    with open_existing_readonly("release-gate") as store:
        timeline = planned_recall(
            "what happened on 2026-01-15", now=None, store=store, k=8
        )
    timeline_by_id = {hit.memory_id: hit for hit in timeline}
    assert set(timeline_by_id) >= {raw_tool.memory_id, task.memory_id}
    assert timeline_by_id[raw_tool.memory_id].recall_class == "tool"
    assert timeline_by_id[raw_tool.memory_id].classification_source == "events.recall_class"
    assert timeline_by_id[raw_tool.memory_id].trusted is False
    assert timeline_by_id[task.memory_id].recall_class == "task"
    assert timeline_by_id[task.memory_id].classification_source == "events.recall_class"
    assert timeline.execution["residue_filter"] == "not_applicable"
    assert timeline.execution["residue_filter_source"] == "not_applicable"
    assert all(hit.filter_context["residue_filter"] == "not_applicable" for hit in timeline)

    cli = CliRunner().invoke(
        app, ["recall", "what happened on 2026-01-15", "-n", "release-gate", "--json"]
    )
    assert cli.exit_code == 0, cli.output
    cli_by_id = {hit["memory_id"]: hit for hit in json.loads(cli.stdout)["hits"]}
    assert cli_by_id[raw_tool.memory_id]["recall_class"] == "tool"
    assert cli_by_id[raw_tool.memory_id]["classification_source"] == "events.recall_class"
    assert cli_by_id[raw_tool.memory_id]["trusted"] is False
    assert cli_by_id[task.memory_id]["recall_class"] == "task"

    from tests.dashutil import make_dash_client

    dashboard = make_dash_client().get(
        "/api/namespace/release-gate/recall", params={"q": "what happened on 2026-01-15"}
    )
    assert dashboard.status_code == 200, dashboard.text
    dashboard_by_id = {hit["memory_id"]: hit for hit in dashboard.json()["hits"]}
    assert dashboard_by_id[raw_tool.memory_id]["recall_class"] == "tool"
    assert dashboard_by_id[raw_tool.memory_id]["trusted"] is False
    assert dashboard_by_id[task.memory_id]["recall_class"] == "task"

    try:
        from haunt.mcp_server import memory_recall
    except ImportError:
        memory_recall = None
    if memory_recall is not None:
        mcp = json.loads(memory_recall(query="what happened on 2026-01-15"))
        mcp_by_id = {hit["memory_id"]: hit for hit in mcp["hits"]}
        assert mcp_by_id[raw_tool.memory_id]["recall_class"] == "tool"
        assert mcp_by_id[raw_tool.memory_id]["trusted"] is False
        assert mcp_by_id[task.memory_id]["recall_class"] == "task"


def test_input_only_and_output_only_tool_residue_stays_untrusted_when_opted_in(
    recall_gate_home,
):
    """All raw tool shapes use the same trust boundary in ranking and timeline."""
    _, _, _, _, _ = recall_gate_home
    with Store("release-gate", create=False) as store:
        input_only = store.observe(
            "INPUT-ONLY-TOOL-RESIDUE-CANARY",
            tool_input="input-only payload",
            event_time="2026-01-17T12:00:00+00:00",
            defer_embedding=True,
        )
        output_only = store.observe(
            "OUTPUT-ONLY-TOOL-RESIDUE-CANARY",
            tool_output="output-only payload",
            event_time="2026-01-17T13:00:00+00:00",
            defer_embedding=True,
        )
    assert not recall(
        "INPUT-ONLY-TOOL-RESIDUE-CANARY", namespace="release-gate", use_vectors=False
    )
    assert not recall(
        "OUTPUT-ONLY-TOOL-RESIDUE-CANARY", namespace="release-gate", use_vectors=False
    )
    ranked = recall(
        "TOOL-RESIDUE-CANARY",
        namespace="release-gate",
        use_vectors=False,
        include_residue=True,
    )
    ranked_by_id = {hit.memory_id: hit for hit in ranked}
    for memory_id in (input_only.memory_id, output_only.memory_id):
        hit = ranked_by_id[memory_id]
        assert hit.recall_class == "tool"
        assert hit.trusted is False
        assert hit.trust_reason == "untrusted-tool-io"

    with open_existing_readonly("release-gate") as store:
        timeline = planned_recall(
            "what happened on 2026-01-17", now=None, store=store, k=8
        )
    timeline_by_id = {hit.memory_id: hit for hit in timeline}
    for memory_id in (input_only.memory_id, output_only.memory_id):
        hit = timeline_by_id[memory_id]
        assert hit.recall_class == "tool"
        assert hit.trusted is False
        assert hit.trust_reason == "untrusted-tool-io"


def test_repeated_python_cli_dashboard_and_mcp_recall_are_source_immutable(
    recall_gate_home, monkeypatch,
):
    home, *_ = recall_gate_home
    before = _tree_snapshot(home)
    registry_before = _data_version(registry_path())
    namespace_before = _data_version(home / "namespaces" / "release-gate.db")

    # A regression to either old recall maintenance call fails before a write.
    monkeypatch.setattr(
        Store, "ensure_current_embeddings",
        lambda self: (_ for _ in ()).throw(AssertionError("recall ran upgrade")),
    )
    monkeypatch.setattr(
        Store, "process_embedding_jobs",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("recall drained jobs")),
    )
    assert recall("ELIGIBLE-RECALL-CANARY", namespace="release-gate", use_vectors=False)
    assert recall("NO-HIT-RECALL-CANARY", namespace="release-gate", use_vectors=False) == []

    cli = CliRunner().invoke(
        app, ["recall", "ELIGIBLE-RECALL-CANARY", "-n", "release-gate", "--json"]
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.stdout)["execution"]["read_only"] is True

    from tests.dashutil import make_dash_client

    dashboard = make_dash_client().get(
        "/api/namespace/release-gate/recall", params={"q": "ELIGIBLE-RECALL-CANARY"}
    )
    assert dashboard.status_code == 200
    assert dashboard.json()["execution"]["maintenance_performed"] is False

    # The repository requires MCP >=2. This suite still runs in stripped
    # developer environments, while a dependency-correct CI exercises MCP.
    try:
        from haunt.mcp_server import memory_recall
    except ImportError:
        memory_recall = None
    if memory_recall is not None:
        payload = json.loads(memory_recall(query="ELIGIBLE-RECALL-CANARY"))
        assert payload["execution"]["read_only"] is True

    assert _data_version(registry_path()) == registry_before
    assert _data_version(home / "namespaces" / "release-gate.db") == namespace_before
    assert _tree_snapshot(home) == before


def test_alias_and_old_schema_read_only_recall_never_repairs_source(recall_gate_home):
    home, eligible, tool, *_ = recall_gate_home
    plan = change_namespace_label("release-gate", "renamed-gate", apply=False)
    change_namespace_label(
        "release-gate", "renamed-gate", apply=True, plan_digest=plan["plan_digest"]
    )
    with Store("renamed-gate", create=False) as store:
        legacy_raw = store.observe(
            "V8-RAW-TOOL-TIMELINE-CANARY",
            tool_input="historical raw tool input",
            event_time="2026-01-18T12:00:00+00:00",
            defer_embedding=True,
        )
    db = home / "namespaces" / "release-gate.db"
    # Model a pre-v9 source. It is opened only through the read-only path below.
    conn = sqlite3.connect(db)
    try:
        conn.execute("ALTER TABLE events DROP COLUMN recall_class")
        conn.execute("UPDATE meta SET value='8' WHERE key='schema_version'")
        conn.commit()
    finally:
        conn.close()
    before = _tree_snapshot(home)
    with open_existing_readonly("release-gate") as store:
        assert store.name == "renamed-gate"
        assert store.recall_class_available is False
        hits = recall("ELIGIBLE-RECALL-CANARY", store=store, use_vectors=False)
        assert [hit.memory_id for hit in hits] == [eligible.memory_id]
        assert hits.execution["recall_class_capability"] == "unavailable"
        assert not recall("TOOL-RECALL-CANARY", store=store, use_vectors=False)
        timeline = planned_recall(
            "what happened on 2026-01-18", now=None, store=store, k=8
        )
        legacy_timeline = next(
            hit for hit in timeline if hit.memory_id == legacy_raw.memory_id
        )
        assert legacy_timeline.recall_class is None
        assert legacy_timeline.classification_source == "raw_tool_structure"
        assert legacy_timeline.trusted is False
    assert _tree_snapshot(home) == before

    cli = CliRunner().invoke(
        app, ["recall", "what happened on 2026-01-18", "-n", "release-gate", "--json"]
    )
    assert cli.exit_code == 0, cli.output
    cli_legacy = next(
        hit
        for hit in json.loads(cli.stdout)["hits"]
        if hit["memory_id"] == legacy_raw.memory_id
    )
    assert cli_legacy["recall_class"] is None
    assert cli_legacy["classification_source"] == "raw_tool_structure"
    assert cli_legacy["trusted"] is False

    from tests.dashutil import make_dash_client

    dashboard = make_dash_client().get(
        "/api/namespace/release-gate/recall", params={"q": "what happened on 2026-01-18"}
    )
    assert dashboard.status_code == 200, dashboard.text
    dashboard_legacy = next(
        hit
        for hit in dashboard.json()["hits"]
        if hit["memory_id"] == legacy_raw.memory_id
    )
    assert dashboard_legacy["recall_class"] is None
    assert dashboard_legacy["classification_source"] == "raw_tool_structure"
    assert dashboard_legacy["trusted"] is False

    try:
        from haunt.mcp_server import memory_recall
    except ImportError:
        memory_recall = None
    if memory_recall is not None:
        mcp_legacy = next(
            hit
            for hit in json.loads(
                memory_recall(query="what happened on 2026-01-18", namespace="release-gate")
            )["hits"]
            if hit["memory_id"] == legacy_raw.memory_id
        )
        assert mcp_legacy["recall_class"] is None
        assert mcp_legacy["classification_source"] == "raw_tool_structure"
        assert mcp_legacy["trusted"] is False

    # A normal writer may upgrade on its explicit lifecycle, but does not
    # guess/backfill classes for historical v8 events. Restart remains at
    # the current SCHEMA_VERSION (v12 as of the succeeds_session column).
    with Store("renamed-gate", create=False) as store:
        assert store.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0] == "12"
        assert store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE recall_class IS NOT NULL"
        ).fetchone()[0] == 0
    with Store("renamed-gate", create=False) as restarted:
        assert restarted.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0] == "12"


def test_stable_id_readonly_open_survives_label_change(recall_gate_home):
    """MCP-style stable IDs resolve the selected database, not a stale label."""
    _, eligible, *_ = recall_gate_home
    with Store("release-gate", create=False) as store:
        namespace_id = store.namespace_id
        db_path = str(store.db_path)
        db_device, db_inode = store.db_path.stat().st_dev, store.db_path.stat().st_ino
    # This is the interleaving an MCP caller sees if a label changes after it
    # selected an authority-pinned namespace ID but before recall opens it.
    plan = change_namespace_label("release-gate", "relabelled-gate", apply=False)
    change_namespace_label(
        "release-gate", "relabelled-gate", apply=True, plan_digest=plan["plan_digest"]
    )
    with open_namespace_identity_readonly(
        namespace_id,
        expected_db_path=db_path,
        expected_db_device=db_device,
        expected_db_inode=db_inode,
    ) as store:
        assert store.name == "relabelled-gate"
        hits = recall("ELIGIBLE-RECALL-CANARY", store=store, use_vectors=False)
        assert [hit.memory_id for hit in hits] == [eligible.memory_id]


def test_offline_fts_never_initializes_network_backend(recall_gate_home, monkeypatch):
    home, eligible, *_ = recall_gate_home
    before = _tree_snapshot(home)
    monkeypatch.setenv("HAUNT_OFFLINE", "1")
    monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "BAAI/bge-m3")
    embed.reset()

    def no_socket(*args, **kwargs):
        raise AssertionError("offline recall attempted a socket")

    monkeypatch.setattr(socket, "socket", no_socket)
    hits = recall("ELIGIBLE-RECALL-CANARY", namespace="release-gate")
    assert [hit.memory_id for hit in hits] == [eligible.memory_id]
    assert hits.execution["modalities"]["vector"] == {
        "state": "not_run", "reason": "offline_mode"
    }
    assert _tree_snapshot(home) == before


def test_offline_subprocess_denies_sockets_with_ambient_provider_keys(
    recall_gate_home, tmp_path,
):
    """Strict offline mode must short-circuit before backend import/download."""
    home, eligible, *_ = recall_gate_home
    cold_cache = tmp_path / "cold-model-cache"
    before = _tree_snapshot(home)
    source_root = Path(__file__).resolve().parents[1] / "src"
    child = r'''
import json
import socket
import sys

def deny_socket(*args, **kwargs):
    raise AssertionError("offline child attempted to create a socket")

socket.socket = deny_socket
from haunt.recall import recall
result = recall("ELIGIBLE-RECALL-CANARY", namespace="release-gate")
print(json.dumps({
    "ids": [hit.memory_id for hit in result],
    "vector": result.execution["modalities"]["vector"],
    "optional_embedding_modules": {
        name: name in sys.modules
        for name in ("fastembed", "huggingface_hub", "onnxruntime", "transformers")
    },
}))
'''
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(source_root),
            "HAUNT_HOME": str(home),
            "HAUNT_OFFLINE": "1",
            "HAUNT_EMBED_MODEL": "BAAI/bge-m3",
            "HAUNT_MODEL_CACHE": str(cold_cache),
            "OPENAI_API_KEY": "ambient-openai-key",
            "ANTHROPIC_API_KEY": "ambient-anthropic-key",
            "HF_TOKEN": "ambient-hf-token",
            "HUGGINGFACE_HUB_TOKEN": "ambient-hf-hub-token",
        }
    )
    env.pop("HAUNT_FTS_ONLY", None)
    result = subprocess.run(
        [sys.executable, "-c", child],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ids"] == [eligible.memory_id]
    assert payload["vector"] == {"state": "not_run", "reason": "offline_mode"}
    assert not any(payload["optional_embedding_modules"].values())
    assert not cold_cache.exists()
    assert _tree_snapshot(home) == before


def test_offline_toggle_is_not_sticky_and_normal_backend_path_is_reachable(
    recall_gate_home, monkeypatch,
):
    monkeypatch.delenv("HAUNT_OFFLINE", raising=False)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    assert embed.offline() is False

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    )
    assert isinstance(embed._load_fastembed("normal-mode-test"), FakeTextEmbedding)
    assert not any(
        key in os.environ
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    )

    monkeypatch.setenv("HAUNT_OFFLINE", "1")
    assert embed.offline() is True
    monkeypatch.delenv("HAUNT_OFFLINE")
    assert embed.offline() is False


def test_maintenance_does_not_create_unknown_namespace_and_clamps_limit(recall_gate_home):
    home, *_ = recall_gate_home
    before = _tree_snapshot(home)
    result = CliRunner().invoke(
        app, ["maintenance", "-n", "never-created-maintenance-typo", "--limit", "-1", "--json"]
    )
    assert result.exit_code == 2
    assert not namespace_exists_readonly("never-created-maintenance-typo")
    assert _tree_snapshot(home) == before


def test_explicit_maintenance_owns_the_only_job_drain(recall_gate_home, monkeypatch):
    home, *_ = recall_gate_home
    db = home / "namespaces" / "release-gate.db"
    with Store("release-gate", create=False) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM embedding_jobs").fetchone()[0] > 0

    def upgrade(self):
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('maintenance-test','1')")
        self.conn.commit()
        return {"updated": 0, "available": True}

    def drain(self, *, limit):
        self.conn.execute("DELETE FROM embedding_jobs")
        self.conn.commit()
        return {"queued": 4, "processed": 4, "failed": 0, "limit": limit}

    monkeypatch.setattr(Store, "ensure_current_embeddings", upgrade)
    monkeypatch.setattr(Store, "process_embedding_jobs", drain)
    result = CliRunner().invoke(
        app, ["maintenance", "-n", "release-gate", "--limit", "-1", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["maintenance_performed"] is True
    assert payload["embedding_jobs"]["limit"] == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM embedding_jobs").fetchone()[0] == 0
        assert conn.execute("SELECT value FROM meta WHERE key='maintenance-test'").fetchone()[0] == "1"


def test_quiescent_and_live_wal_recall_leave_registry_and_namespace_exact(
    recall_gate_home,
):
    home, eligible, *_ = recall_gate_home
    namespace_db = home / "namespaces" / "release-gate.db"

    # Quiescent, sidecar-free sources exercise immutable read mode.
    for source in (registry_path(), namespace_db):
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = Path(str(source) + suffix)
            if sidecar.exists():
                sidecar.unlink()
    quiescent = _tree_snapshot(home)
    assert recall("ELIGIBLE-RECALL-CANARY", namespace="release-gate", use_vectors=False)
    assert _tree_snapshot(home) == quiescent

    # Keep both sources live in WAL while recall resolves the registry and
    # reads the namespace. The read path must use its private shadow and leave
    # every source byte/identity/mtime untouched.
    registry_writer = sqlite3.connect(registry_path())
    namespace_writer = sqlite3.connect(namespace_db)
    try:
        for writer in (registry_writer, namespace_writer):
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
        registry_writer.execute(
            "UPDATE namespace_identities SET updated_at='live-wal-read' "
            "WHERE canonical_label='release-gate'"
        )
        namespace_writer.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES ('live-wal-read','1')"
        )
        registry_writer.commit()
        namespace_writer.commit()
        assert Path(str(registry_path()) + "-wal").stat().st_size > 0
        assert Path(str(namespace_db) + "-wal").stat().st_size > 0
        live_before = _tree_snapshot(home)
        hits = recall("ELIGIBLE-RECALL-CANARY", namespace="release-gate", use_vectors=False)
        assert [hit.memory_id for hit in hits] == [eligible.memory_id]
        assert _tree_snapshot(home) == live_before
    finally:
        namespace_writer.close()
        registry_writer.close()


def test_native_vec_recall_filters_residue_before_vector_candidate_selection(
    recall_gate_home, monkeypatch,
):
    """When vec0 is present, raw/classed residue never enters default KNN."""
    import sqlite_vec
    from haunt.store import _SidecarGuardedConnection

    if not hasattr(_SidecarGuardedConnection, "enable_load_extension"):
        pytest.skip("this Python SQLite connection cannot load sqlite-vec")

    home, *_ = recall_gate_home
    monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed.reset()
    with Store("release-gate", create=False) as store:
        eligible = store.observe(
            "NATIVE-VEC-RESIDUE-CANARY", defer_embedding=True
        )
        raw_tool = store.observe(
            "NATIVE-VEC-RESIDUE-CANARY", tool_name="shell", defer_embedding=True
        )
        task = store.observe(
            "NATIVE-VEC-RESIDUE-CANARY", recall_class="task", defer_embedding=True
        )
        if not store.vec_ok() or not ensure_vec_table(store.conn, 2):
            pytest.skip("sqlite-vec is unavailable in this Python/SQLite build")
        for result, vector in (
            (eligible, [1.0, 0.0]),
            (raw_tool, [0.99, 0.01]),
            (task, [0.98, 0.02]),
        ):
            blob = sqlite_vec.serialize_float32(vector)
            store.conn.execute(
                "UPDATE memories SET embedding=? WHERE id=?", (blob, result.memory_id)
            )
            store.conn.execute(
                "INSERT INTO vec_memories(id, embedding) VALUES (?, ?)",
                (result.memory_id, blob),
            )
        store.conn.commit()
    recall_module = importlib.import_module("haunt.recall")
    monkeypatch.setattr(recall_module, "embed_available", lambda: True)
    monkeypatch.setattr(recall_module, "embed_one", lambda _query: [1.0, 0.0])
    before = _tree_snapshot(home)
    default = recall("NATIVE-VEC-RESIDUE-CANARY", namespace="release-gate")
    assert [hit.memory_id for hit in default] == [eligible.memory_id]
    assert default.execution["modalities"]["vector"] == {
        "state": "candidate", "reason": "returned_native_vec_candidates"
    }
    assert default[0].vec_metric == "cosine_distance"
    included = recall(
        "NATIVE-VEC-RESIDUE-CANARY",
        namespace="release-gate",
        include_residue=True,
    )
    assert {hit.memory_id for hit in included} >= {
        eligible.memory_id, raw_tool.memory_id, task.memory_id
    }
    assert _tree_snapshot(home) == before


def test_incomplete_wal_and_rollback_journal_fail_honestly_without_source_writes(
    recall_gate_home,
):
    """Unsafe sources are rejected instead of risking a portable 'read' write."""
    home, *_ = recall_gate_home
    namespace_db = home / "namespaces" / "release-gate.db"

    # Registry resolution itself must reject a live WAL missing its matching
    # SHM sidecar.  Snapshot after creating the adversarial state: recall may
    # neither repair it nor append a new sidecar.
    registry_writer = sqlite3.connect(registry_path())
    try:
        registry_writer.execute("PRAGMA journal_mode=WAL")
        registry_writer.execute("PRAGMA wal_autocheckpoint=0")
        registry_writer.execute(
            "UPDATE namespace_identities SET updated_at='incomplete-registry-wal' "
            "WHERE canonical_label='release-gate'"
        )
        registry_writer.commit()
        registry_shm = Path(str(registry_path()) + "-shm")
        assert registry_shm.exists()
        registry_shm.unlink()
        before_registry = _tree_snapshot(home)
        with pytest.raises(NamespacePathError, match="incomplete WAL state"):
            recall("ELIGIBLE-RECALL-CANARY", namespace="release-gate", use_vectors=False)
        assert _tree_snapshot(home) == before_registry
    finally:
        registry_writer.close()

    # The same rule applies after a safe registry lookup when the namespace
    # database is incomplete.
    namespace_writer = sqlite3.connect(namespace_db)
    try:
        namespace_writer.execute("PRAGMA journal_mode=WAL")
        namespace_writer.execute("PRAGMA wal_autocheckpoint=0")
        namespace_writer.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES ('incomplete-wal','1')"
        )
        namespace_writer.commit()
        namespace_shm = Path(str(namespace_db) + "-shm")
        assert namespace_shm.exists()
        namespace_shm.unlink()
        before_namespace = _tree_snapshot(home)
        with pytest.raises(NamespacePathError, match="incomplete WAL state"):
            recall("ELIGIBLE-RECALL-CANARY", namespace="release-gate", use_vectors=False)
        assert _tree_snapshot(home) == before_namespace
    finally:
        namespace_writer.close()

    # A non-empty rollback journal is likewise a hard, side-effect-free error.
    rollback_writer = sqlite3.connect(namespace_db)
    try:
        rollback_writer.execute("PRAGMA journal_mode=DELETE")
        rollback_writer.execute("BEGIN IMMEDIATE")
        rollback_writer.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES ('rollback-live','1')"
        )
        journal = Path(str(namespace_db) + "-journal")
        assert journal.exists() and journal.stat().st_size > 0
        before_journal = _tree_snapshot(home)
        with pytest.raises(NamespacePathError, match="rollback journal"):
            recall("ELIGIBLE-RECALL-CANARY", namespace="release-gate", use_vectors=False)
        assert _tree_snapshot(home) == before_journal
    finally:
        rollback_writer.rollback()
        rollback_writer.close()
