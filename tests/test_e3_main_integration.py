"""Integration locks for E3 rebased over correction and provenance schemas."""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import pytest

from haunt.paths import registry_path
from haunt.store import (
    REGISTRY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Store,
    change_namespace_label,
    init_registry,
    resolve_namespace_identity,
)


@pytest.fixture
def integrated_home(tmp_path, monkeypatch):
    home = tmp_path / "haunt-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    monkeypatch.delenv("HAUNT_MCP_ADMIN", raising=False)
    monkeypatch.delenv("HAUNT_MCP_ALLOW_PURGE", raising=False)
    monkeypatch.setenv("CURSOR_HOME", str(tmp_path / "cursor"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    from haunt import embed

    embed.reset()
    init_registry()
    yield home
    embed.reset()


def _rename(old_label: str, new_label: str) -> dict:
    plan = change_namespace_label(old_label, new_label, apply=False)
    return change_namespace_label(
        old_label,
        new_label,
        apply=True,
        plan_digest=plan["plan_digest"],
    )


def _reset_mcp() -> None:
    import haunt.mcp_server as mcp

    mcp._MCP_AUTHORITY = None
    mcp._MCP_AUTHORITY_HOME = None


def test_namespace_v9_registry_v5_upgrade_restart_through_alias(integrated_home):
    with Store("integrated-v7") as store:
        namespace_id = store.namespace_id
        db_path = store.db_path
        event = store.observe("schema upgrade provenance canary")

    # Model an E2 v7 namespace while retaining the upgraded E3 registry.
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE meta SET value='7' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    _rename("integrated-v7", "integrated-v8")

    for label in ("integrated-v7", "integrated-v8"):
        init_registry()
        with Store(label, create=False) as reopened:
            assert reopened.namespace_id == namespace_id
            assert reopened.db_path == db_path
            assert int(reopened.get_meta("schema_version")) == SCHEMA_VERSION == 10
            detail = reopened.get_memory(event.memory_id)
            assert detail is not None
            assert detail["provenance"]["schema_version"] == 1
            trigger_names = {
                row[0]
                for row in reopened.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
            assert "corrections_append_only_update" in trigger_names
            assert "events_provenance_type_insert" in trigger_names

        registry = sqlite3.connect(registry_path())
        version = registry.execute(
            "SELECT value FROM registry_meta WHERE key='schema_version'"
        ).fetchone()[0]
        registry.close()
        assert int(version) == REGISTRY_SCHEMA_VERSION == 5


def test_correction_and_privacy_purge_survive_alias_and_stable_mcp_identity(
    integrated_home, monkeypatch
):
    import haunt.mcp_server as mcp

    signature = inspect.signature(mcp.memory_contradict)
    assert signature.parameters["idempotency_key"].default is inspect.Parameter.empty

    with Store("lineage-original") as store:
        namespace_id = store.namespace_id
        target = store.observe("alias correction target secret")
    _rename("lineage-original", "lineage-renamed")
    monkeypatch.setenv("HAUNT_NAMESPACE", "lineage-original")
    monkeypatch.setenv("HAUNT_MCP_ALLOW_PURGE", "1")
    _reset_mcp()

    corrected = json.loads(
        mcp.memory_contradict(
            target.memory_id,
            idempotency_key="alias-correction-required-key",
            replacement="alias correction survivor",
            namespace="lineage-original",
        )
    )
    assert corrected["ok"] is True
    replacement_id = corrected["replacement_memory_id"]
    assert mcp._authority()._pin.namespace_id == namespace_id

    _rename("lineage-renamed", "lineage-final")
    purged = json.loads(
        mcp.memory_purge(target.memory_id, namespace="lineage-original")
    )
    assert purged["ok"] is True
    assert purged["namespace"] == "lineage-final"
    assert purged["lineage_tombstone"]["status"] == "erased"
    with Store("lineage-original", create=False) as reopened:
        assert reopened.namespace_id == namespace_id
        assert reopened.get_memory(target.memory_id) is None
        assert reopened.get_memory(replacement_id)["content"] == (
            "alias correction survivor"
        )


def test_provenance_recall_timeline_and_trace_follow_renamed_alias_stable_id(
    integrated_home, monkeypatch
):
    import haunt.mcp_server as mcp

    with Store("provenance-original") as store:
        namespace_id = store.namespace_id
    monkeypatch.setenv("HAUNT_NAMESPACE", "provenance-original")
    _reset_mcp()
    provenance = {
        "schema_version": 1,
        "kind": "native",
        "channel": "mcp",
        "origin": "integration-test",
    }
    observed = json.loads(
        mcp.memory_observe(
            text="RENAMED ALIAS PROVENANCE CANARY",
            namespace="provenance-original",
            origin="integration-test",
            provenance=provenance,
        )
    )
    assert observed["ok"] is True
    assert observed["provenance"] == provenance
    _rename("provenance-original", "provenance-renamed")

    recalled = json.loads(
        mcp.memory_recall(
            query="RENAMED ALIAS PROVENANCE CANARY",
            namespace="provenance-original",
        )
    )
    timeline = json.loads(mcp.memory_timeline(namespace="provenance-original"))
    assert recalled["namespace"] == "provenance-renamed"
    assert recalled["hits"][0]["origin"] == "integration-test"
    assert timeline["namespace"] == "provenance-renamed"
    event = next(row for row in timeline["events"] if row["id"] == observed["event_id"])
    assert event["provenance"] == provenance

    opened_ids: list[str | None] = []
    real_open = mcp._open_mcp_store

    def capture_stable_open(access, *, create):
        opened_ids.append(access.namespace_id)
        return real_open(access, create=create)

    monkeypatch.setattr(mcp, "_open_mcp_store", capture_stable_open)
    traced = json.loads(
        mcp.memory_trace(observed["memory_id"], namespace="provenance-original")
    )
    assert opened_ids == [namespace_id]
    assert traced["members"][0]["provenance"] == provenance


def test_alias_registry_does_not_change_host_install_or_doctor_contract(
    integrated_home
):
    from haunt.bootstrap import bind_launchers
    from haunt.hosts import doctor_all_hosts, install_all_hosts

    with Store("host-contract-original"):
        pass
    _rename("host-contract-original", "host-contract-renamed")
    _home, hook_cmd, mcp_cmd = bind_launchers()
    install_all_hosts(
        str(integrated_home),
        hook_cmd,
        mcp_cmd,
    )
    statuses = doctor_all_hosts(
        str(integrated_home),
        hook_cmd,
        mcp_cmd,
    )
    assert statuses
    assert all(status.hooks_present and status.mcp_present for status in statuses)
    assert resolve_namespace_identity("host-contract-original")["canonical_label"] == (
        "host-contract-renamed"
    )


def test_frozen_e0_baseline_is_unchanged_by_alias_registry(integrated_home):
    from haunt.frozen_retrieval_eval import DEFAULT_BASELINE, evaluate

    with Store("frozen-alias-original"):
        pass
    _rename("frozen-alias-original", "frozen-alias-renamed")
    expected = json.loads(Path(DEFAULT_BASELINE).read_text(encoding="utf-8"))
    assert evaluate().as_dict() == expected
