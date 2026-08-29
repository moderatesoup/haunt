"""E4 canonical namespace export/import contract and release evidence."""

from __future__ import annotations

import base64
import copy
import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tests.dashutil import make_dash_client

from haunt.cli import app
from haunt.embed import EmbedState
from haunt.paths import NamespacePathError
from haunt.portability import (
    FORMAT_MAJOR,
    FORMAT_MINOR,
    FORMAT_NAME,
    ExportError,
    ImportBundleError,
    ImportConflictError,
    ImportLimitError,
    ImportLimits,
    _MINOR_ADDED_FIELDS,
    _canonical_bytes,
    _digest,
    _semantic_from_bundle,
    build_namespace_export,
    canonical_export_bytes,
    export_namespace_path,
    import_namespace_bytes,
    resolve_import_limits,
)
from haunt.recall import recall
from haunt.store import (
    PRIVACY_LINEAGE_KEY,
    SCHEMA_VERSION,
    Store,
    _content_hash,
    change_namespace_label,
    namespace_exists_readonly,
    open_existing,
    privacy_lineage_genesis,
    retire_namespace_alias,
    resolve_namespace_identity,
)


@pytest.fixture
def portable_home(tmp_path, monkeypatch):
    home = tmp_path / "source"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed

    embed.reset()
    yield home
    embed.reset()


def _switch_home(monkeypatch, home: Path) -> None:
    monkeypatch.setenv("HAUNT_HOME", str(home))
    from haunt import embed

    embed.reset()


def _redigest(bundle: dict) -> bytes:
    bundle["manifest"]["semantic_digest"] = _digest(_semantic_from_bundle(bundle))
    bundle["manifest"]["record_counts"] = {
        name: len(rows) for name, rows in bundle["records"].items()
    }
    bundle["manifest"]["total_records"] = sum(
        bundle["manifest"]["record_counts"].values()
    )
    return canonical_export_bytes(bundle)


def _durable_snapshot(store: Store) -> dict[str, list[tuple]]:
    tables = (
        "meta",
        "sessions",
        "events",
        "memories",
        "lineage_tombstones",
        "corrections",
        "entities",
        "entity_mentions",
        "relation_evidence",
        "relations",
        "memories_fts",
        "embedding_jobs",
    )
    return {
        table: sorted(
            [tuple(row) for row in store.conn.execute(f'SELECT * FROM "{table}"')],
            key=repr,
        )
        for table in tables
    }


def _rejection_snapshot(store: Store) -> dict[str, object]:
    return {
        "durable": _durable_snapshot(store),
        "schema": sorted(
            tuple(row)
            for row in store.conn.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
            )
        ),
        "data_version": store.conn.execute("PRAGMA data_version").fetchone()[0],
        "schema_version": store.conn.execute("PRAGMA schema_version").fetchone()[0],
    }


def _seed_bundle() -> tuple[dict, str]:
    with Store("portable") as store:
        observed = store.observe(
            "Unicode café ☃ canonical transfer",
            event_time="2026-01-01T00:00:00Z",
            defer_embedding=True,
        )
    return build_namespace_export(
        "portable", exported_at="2026-02-01T00:00:00Z"
    ), observed.memory_id


def test_default_cut_digest_is_stable_and_round_trips_without_reusing_cut(
    portable_home, tmp_path, monkeypatch
):
    first, memory_id = _seed_bundle()
    second = build_namespace_export(
        "portable", exported_at="2030-01-01T00:00:00Z"
    )
    assert first["temporal_cut"] == second["temporal_cut"]
    assert first["manifest"]["semantic_digest"] == second["manifest"]["semantic_digest"]
    assert canonical_export_bytes(first) != canonical_export_bytes(second)

    raw = canonical_export_bytes(first)
    destination = tmp_path / "destination"
    _switch_home(monkeypatch, destination)
    imported = import_namespace_bytes(raw)
    reexport = build_namespace_export("portable")
    replay = import_namespace_bytes(canonical_export_bytes(second))

    assert imported["created_namespace"] is True
    assert replay["deduplicated"] is True
    assert sum(replay["inserted"].values()) == 0
    assert reexport["manifest"]["semantic_digest"] == first["manifest"]["semantic_digest"]
    assert reexport["namespace"] == first["namespace"]
    assert reexport["records"] == first["records"]
    with open_existing("portable") as store:
        assert store.get_memory(memory_id)["content"] == "Unicode café ☃ canonical transfer"
        assert store.get_meta("schema_version") == str(SCHEMA_VERSION)


def test_v1_golden_bundle_is_canonical_and_round_trips(
    portable_home, tmp_path, monkeypatch
):
    fixture = Path(__file__).parent / "fixtures" / "export" / "v1" / "golden.json"
    raw = fixture.read_bytes()
    bundle = json.loads(raw)
    assert canonical_export_bytes(bundle) + b"\n" == raw
    assert bundle["version"] == {"major": FORMAT_MAJOR, "minor": 0}

    _switch_home(monkeypatch, tmp_path / "golden-destination")
    report = import_namespace_bytes(raw)
    assert report["semantic_digest"] == bundle["manifest"]["semantic_digest"]
    # A v1.0 bundle re-exports at the current minor, so its digest moves with
    # the declared version. Everything else the bundle said is unchanged.
    reexport = build_namespace_export("golden")
    assert reexport["version"] == {"major": FORMAT_MAJOR, "minor": FORMAT_MINOR}
    upgraded = _semantic_from_bundle(bundle)
    upgraded["version"] = {"major": FORMAT_MAJOR, "minor": FORMAT_MINOR}
    assert reexport["manifest"]["semantic_digest"] == _digest(upgraded)


def test_export_excludes_local_and_derived_state_and_import_rebuilds_destination_state(
    portable_home, tmp_path, monkeypatch
):
    with Store("derived", repo_path=str(tmp_path / "local-secret-repository")) as store:
        result = store.observe(
            "AlphaService writes src/alpha.py for BetaService",
            defer_embedding=True,
        )
        store.conn.execute(
            "UPDATE memories SET embedding=? WHERE id=?", (b"SOURCE-EMBEDDING-CANARY", result.memory_id)
        )
        store.conn.execute("DELETE FROM embedding_jobs WHERE memory_id=?", (result.memory_id,))
        store.conn.commit()
        source_entities = store.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        source_evidence = store.conn.execute("SELECT COUNT(*) FROM relation_evidence").fetchone()[0]

    bundle = build_namespace_export("derived")
    raw = canonical_export_bytes(bundle)
    text = raw.decode("utf-8")
    for forbidden in (
        "SOURCE-EMBEDDING-CANARY",
        str(tmp_path),
        "db_path",
        "db_device",
        "db_inode",
        "memories_fts",
        "vec_memories",
        "embedding_jobs",
        "repo_path",
    ):
        assert forbidden not in text
    assert len(bundle["records"]["entities"]) == source_entities
    assert len(bundle["records"]["relation_evidence"]) == source_evidence

    _switch_home(monkeypatch, tmp_path / "derived-destination")
    import_namespace_bytes(raw)
    with open_existing("derived") as store:
        row = store.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (result.memory_id,)
        ).fetchone()
        assert row["embedding"] is None
        assert store.conn.execute(
            "SELECT COUNT(*) FROM embedding_jobs WHERE memory_id=?", (result.memory_id,)
        ).fetchone()[0] == 1
        assert store.conn.execute(
            "SELECT content FROM memories_fts WHERE id=?", (result.memory_id,)
        ).fetchone()[0] == "AlphaService writes src/alpha.py for BetaService"
        assert store.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == source_entities
        assert store.conn.execute("SELECT COUNT(*) FROM relation_evidence").fetchone()[0] == source_evidence
        assert store.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0] > 0


def test_alias_and_remote_identity_round_trip_without_local_repository_paths(
    portable_home, tmp_path, monkeypatch
):
    with Store("Old Label", repo_path="git@github.com:Example/Portable.git") as store:
        original_id = store.namespace_id
        store.observe("alias migration survives export", defer_embedding=True)
    plan = change_namespace_label("Old Label", "New Label", action="rename", apply=False)
    change_namespace_label(
        "Old Label",
        "New Label",
        action="rename",
        apply=True,
        plan_digest=plan["plan_digest"],
    )
    bundle = build_namespace_export("Old Label")
    assert bundle["namespace"] == {
        "namespace_id": original_id,
        "canonical_label": "New-Label",
        "aliases": [
            {"label": "New-Label", "is_canonical": True, "source_alias_norm": None},
            {"label": "Old-Label", "is_canonical": False, "source_alias_norm": None},
        ],
        "repository_identities": ["github.com/example/portable"],
        "privacy_lineage_head": privacy_lineage_genesis(original_id),
    }
    assert str(tmp_path) not in canonical_export_bytes(bundle).decode()

    _switch_home(monkeypatch, tmp_path / "alias-destination")
    import_namespace_bytes(canonical_export_bytes(bundle))
    for label in ("Old Label", "New Label"):
        identity = resolve_namespace_identity(label)
        assert identity is not None
        assert identity["namespace_id"] == original_id
        assert identity["canonical_label"] == "New-Label"


def test_canonical_alias_order_and_dependent_lineage_round_trip(
    portable_home, tmp_path, monkeypatch
):
    with Store("zeta") as store:
        original_id = store.namespace_id
        store.observe("alias lineage", defer_embedding=True)
    first = change_namespace_label("zeta", "alpha", action="alias", apply=False)
    change_namespace_label(
        "zeta",
        "alpha",
        action="alias",
        apply=True,
        plan_digest=first["plan_digest"],
    )
    second = change_namespace_label("alpha", "beta", action="alias", apply=False)
    change_namespace_label(
        "alpha",
        "beta",
        action="alias",
        apply=True,
        plan_digest=second["plan_digest"],
    )

    bundle = build_namespace_export("zeta")
    assert bundle["namespace"]["aliases"] == [
        {"label": "zeta", "is_canonical": True, "source_alias_norm": None},
        {"label": "alpha", "is_canonical": False, "source_alias_norm": "zeta"},
        {"label": "beta", "is_canonical": False, "source_alias_norm": "alpha"},
    ]

    _switch_home(monkeypatch, tmp_path / "ordered-alias-destination")
    import_namespace_bytes(canonical_export_bytes(bundle))
    for label in ("zeta", "alpha", "beta"):
        assert resolve_namespace_identity(label)["namespace_id"] == original_id


@pytest.mark.parametrize("failure", ["missing", "cycle", "canonical-source"])
def test_invalid_alias_lineage_is_rejected_before_namespace_creation(
    portable_home, tmp_path, monkeypatch, failure
):
    bundle, _ = _seed_bundle()
    aliases = bundle["namespace"]["aliases"]
    if failure == "missing":
        aliases[0]["source_alias_norm"] = "not-present"
    elif failure == "canonical-source":
        aliases.append(
            {"label": "upstream", "is_canonical": False, "source_alias_norm": None}
        )
        aliases[0]["source_alias_norm"] = "upstream"
    else:
        aliases.extend(
            [
                {"label": "alpha", "is_canonical": False, "source_alias_norm": "beta"},
                {"label": "beta", "is_canonical": False, "source_alias_norm": "alpha"},
            ]
        )
    aliases[:] = sorted(
        aliases,
        key=lambda alias: (
            not alias["is_canonical"],
            alias["label"].casefold(),
        ),
    )
    raw = _redigest(bundle)
    _switch_home(monkeypatch, tmp_path / f"bad-alias-{failure}")
    with pytest.raises(ImportBundleError, match="alias"):
        import_namespace_bytes(raw)
    assert not namespace_exists_readonly("portable")


def test_sqlite_blob_and_nonfinite_legacy_values_round_trip_exactly(
    portable_home, tmp_path, monkeypatch
):
    with Store("legacy-values") as store:
        observed = store.observe(
            "LegacyValueOne and LegacyValueTwo", defer_embedding=True
        )
        store.conn.execute(
            "UPDATE events SET provenance=NULL,origin=?,meta=?,tool_input=? WHERE id=?",
            (b"\x00\xfflegacy-origin", b"\x80opaque-meta", b"\x81tool-input", observed.event_id),
        )
        store.conn.execute(
            "UPDATE relation_evidence SET weight=? WHERE event_id=?",
            (float("inf"), observed.event_id),
        )
        store.conn.execute(
            "UPDATE memories SET content=? WHERE id=?",
            (b"\x82opaque-memory", observed.memory_id),
        )
        store.conn.commit()
    bundle = build_namespace_export("legacy-values")
    event = bundle["records"]["events"][0]
    assert event["origin"] == {
        "$haunt_sqlite": "blob",
        "base64": base64.b64encode(b"\x00\xfflegacy-origin").decode(),
    }
    assert event["tool_input"] == {
        "$haunt_sqlite": "blob",
        "base64": base64.b64encode(b"\x81tool-input").decode(),
    }
    assert bundle["records"]["relation_evidence"][0]["weight"] == {
        "$haunt_sqlite": "real", "bits": "inf"
    }
    assert bundle["records"]["memories"][0]["content"] == {
        "$haunt_sqlite": "blob",
        "base64": base64.b64encode(b"\x82opaque-memory").decode(),
    }

    _switch_home(monkeypatch, tmp_path / "legacy-destination")
    import_namespace_bytes(canonical_export_bytes(bundle))
    with open_existing("legacy-values") as store:
        row = store.conn.execute(
            "SELECT origin,meta,tool_input,provenance FROM events WHERE id=?",
            (observed.event_id,),
        ).fetchone()
        assert row["origin"] == b"\x00\xfflegacy-origin"
        assert row["meta"] == b"\x80opaque-meta"
        assert row["tool_input"] == b"\x81tool-input"
        assert row["provenance"] is None
        assert store.conn.execute(
            "SELECT weight FROM relation_evidence WHERE event_id=?",
            (observed.event_id,),
        ).fetchone()[0] == float("inf")
        memory = store.conn.execute(
            "SELECT content,typeof(content) AS content_type FROM memories WHERE id=?",
            (observed.memory_id,),
        ).fetchone()
        assert memory["content"] == b"\x82opaque-memory"
        assert memory["content_type"] == "blob"
        assert store.conn.execute(
            "SELECT COUNT(*) FROM embedding_jobs WHERE memory_id=?",
            (observed.memory_id,),
        ).fetchone()[0] == 0
        assert store.conn.execute(
            "SELECT COUNT(*) FROM memories_fts WHERE id=?",
            (observed.memory_id,),
        ).fetchone()[0] == 0


@pytest.mark.parametrize("mutation", ["bool-text", "int-id", "real-text", "null-text"])
def test_sqlite_affinity_coercions_are_rejected_before_destination_mutation(
    portable_home, tmp_path, monkeypatch, mutation
):
    bundle, _ = _seed_bundle()
    if mutation == "bool-text":
        bundle["records"]["sessions"][0]["source"] = True
    elif mutation == "int-id":
        old = bundle["records"]["sessions"][0]["id"]
        bundle["records"]["sessions"][0]["id"] = 7
        for event in bundle["records"]["events"]:
            if event["session_id"] == old:
                event["session_id"] = 7
    elif mutation == "real-text":
        bundle["records"]["memories"][0]["content"] = 1.25
    else:
        bundle["records"]["events"][0]["content"] = None

    _switch_home(monkeypatch, tmp_path / f"affinity-{mutation}")
    with pytest.raises(ImportBundleError):
        import_namespace_bytes(_redigest(bundle))
    assert not namespace_exists_readonly("portable")


@pytest.mark.parametrize("value", [2**63, -(2**63) - 1])
def test_out_of_range_sqlite_integers_fail_in_scratch_without_mutation(
    portable_home, tmp_path, monkeypatch, value
):
    bundle, _ = _seed_bundle()
    bundle["records"]["events"][0]["content"] = value
    _switch_home(monkeypatch, tmp_path / f"integer-{value > 0}")
    with pytest.raises(ImportBundleError):
        import_namespace_bytes(_redigest(bundle))
    assert not namespace_exists_readonly("portable")


def test_temporal_cut_keeps_correction_atomic_and_referentially_closed(
    portable_home, tmp_path, monkeypatch
):
    import haunt.store as store_module

    monkeypatch.setattr(
        store_module, "now_iso", lambda: "2026-01-01T00:00:00.000000+00:00"
    )
    with Store("cut") as store:
        target = store.observe(
            "temporal target token",
            event_time="2026-01-01T00:00:00Z",
            defer_embedding=True,
        )
        monkeypatch.setattr(
            store_module, "now_iso", lambda: "2026-01-03T00:00:00.000000+00:00"
        )
        corrected = store.contradict(
            target.memory_id,
            replacement="temporal replacement token",
            idempotency_key="cut-correction",
        )
        replacement_id = corrected["replacement_memory_id"]
        monkeypatch.setattr(
            store_module, "now_iso", lambda: "2026-01-05T00:00:00.000000+00:00"
        )
        after = store.observe("written after cut token", defer_embedding=True)

    before = build_namespace_export(
        "cut",
        cut="2026-01-02T00:00:00Z",
        exported_at="2026-02-01T00:00:00Z",
    )
    at = build_namespace_export(
        "cut",
        cut="2026-01-03T00:00:00Z",
        exported_at="2026-02-01T00:00:00Z",
    )
    after_ids = {row["id"] for row in at["records"]["memories"]}
    before_ids = {row["id"] for row in before["records"]["memories"]}
    assert before_ids == {target.memory_id}
    assert before["records"]["memories"][0]["valid_to"] is None
    assert before["records"]["corrections"] == []
    assert target.memory_id in after_ids and replacement_id in after_ids
    assert after.memory_id not in after_ids
    assert len(at["records"]["corrections"]) == 1

    for label, bundle, expected_current in (
        ("before", before, target.memory_id),
        ("at", at, replacement_id),
    ):
        # Preserve exact namespace identity but use isolated homes, one per cut.
        destination = tmp_path / f"cut-{label}"
        _switch_home(monkeypatch, destination)
        import_namespace_bytes(canonical_export_bytes(bundle))
        with open_existing("cut") as imported:
            current = recall("temporal token", store=imported, use_vectors=False)
            assert [hit.memory_id for hit in current] == [expected_current]
            historical = recall(
                "temporal token",
                as_of="2026-01-02T00:00:00Z",
                store=imported,
                use_vectors=False,
            )
            assert [hit.memory_id for hit in historical] == [target.memory_id]
            if label == "at":
                trace = imported.trace(replacement_id)
                assert trace["lineage_status"] == "linked"
                assert [item["memory_id"] for item in trace["members"]] == [
                    target.memory_id,
                    replacement_id,
                ]


def test_historical_cut_projects_entity_clocks_without_future_observations(
    portable_home, tmp_path, monkeypatch
):
    import haunt.store as store_module

    monkeypatch.setattr(
        store_module, "now_iso", lambda: "2026-01-01T00:00:00.000000+00:00"
    )
    with Store("graph-cut") as store:
        store.observe(
            "TemporalGraphNode works with AlphaService",
            event_time="2026-01-01T00:00:00Z",
            defer_embedding=True,
        )
        monkeypatch.setattr(
            store_module, "now_iso", lambda: "2026-01-05T00:00:00.000000+00:00"
        )
        store.observe(
            "TemporalGraphNode works with BetaService",
            event_time="2026-01-05T00:00:00Z",
            defer_embedding=True,
        )

    before = build_namespace_export("graph-cut", cut="2025-12-31T00:00:00Z")
    at = build_namespace_export("graph-cut", cut="2026-01-01T00:00:00Z")
    between = build_namespace_export("graph-cut", cut="2026-01-03T00:00:00Z")
    after = build_namespace_export("graph-cut", cut="2026-01-06T00:00:00Z")
    assert before["records"]["entities"] == []
    at_entity = next(
        row for row in at["records"]["entities"] if row["norm_name"] == "temporalgraphnode"
    )
    between_entity = next(
        row
        for row in between["records"]["entities"]
        if row["norm_name"] == "temporalgraphnode"
    )
    after_entity = next(
        row
        for row in after["records"]["entities"]
        if row["norm_name"] == "temporalgraphnode"
    )
    expected_first = "2026-01-01T00:00:00.000000+00:00"
    assert at_entity["first_seen"] == at_entity["last_seen"] == expected_first
    assert between_entity["last_seen"] == expected_first
    assert after_entity["last_seen"] == "2026-01-05T00:00:00.000000+00:00"

    _switch_home(monkeypatch, tmp_path / "graph-cut-destination")
    import_namespace_bytes(canonical_export_bytes(between))
    reexport = build_namespace_export("graph-cut", cut=between["temporal_cut"])
    assert reexport["manifest"]["semantic_digest"] == between["manifest"]["semantic_digest"]


@pytest.mark.parametrize("mutation", ["observe", "correct"])
def test_export_retries_concurrent_write_and_returns_one_post_write_snapshot(
    portable_home, monkeypatch, mutation
):
    import haunt.portability as portability

    writer = Store("concurrent-export")
    target = writer.observe("snapshot target", defer_embedding=True)
    pinned = threading.Event()
    resume = threading.Event()
    paused_once = False

    def pause_after_cut():
        nonlocal paused_once
        if paused_once:
            return
        paused_once = True
        pinned.set()
        assert resume.wait(5), "concurrent export was not resumed"

    monkeypatch.setattr(portability, "_export_after_cut_hook", pause_after_cut)
    result: list[dict] = []
    failures: list[BaseException] = []

    def run_export():
        try:
            result.append(build_namespace_export("concurrent-export"))
        except BaseException as exc:  # surfaced in the test thread below
            failures.append(exc)

    thread = threading.Thread(target=run_export)
    thread.start()
    assert pinned.wait(5), "export did not pin its read snapshot"
    try:
        if mutation == "observe":
            added = writer.observe("concurrent addition", defer_embedding=True)
        else:
            corrected = writer.contradict(
                target.memory_id,
                replacement="concurrent replacement",
                idempotency_key="concurrent-correction",
            )
            added = type("Added", (), {"memory_id": corrected["replacement_memory_id"]})()
    finally:
        resume.set()
        thread.join(10)
        writer.close()
    assert not thread.is_alive()
    assert failures == []
    memory_ids = {row["id"] for row in result[0]["records"]["memories"]}
    assert added.memory_id in memory_ids
    if mutation == "correct":
        assert len(result[0]["records"]["corrections"]) == 1
        target_row = next(
            row for row in result[0]["records"]["memories"] if row["id"] == target.memory_id
        )
        assert target_row["valid_to"] is not None


def test_export_never_returns_pre_purge_canary_after_concurrent_purge(
    portable_home, monkeypatch
):
    import haunt.portability as portability

    canary = "CONCURRENT-PURGE-E4-CANARY"
    writer = Store("concurrent-purge")
    target = writer.observe(canary, origin=canary, defer_embedding=True)
    pinned = threading.Event()
    resume = threading.Event()
    paused_once = False

    def pause_after_cut():
        nonlocal paused_once
        if paused_once:
            return
        paused_once = True
        pinned.set()
        assert resume.wait(5), "concurrent purge export was not resumed"

    monkeypatch.setattr(portability, "_export_after_cut_hook", pause_after_cut)
    result: list[dict] = []
    failures: list[BaseException] = []

    def run_export():
        try:
            result.append(build_namespace_export("concurrent-purge"))
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run_export)
    thread.start()
    assert pinned.wait(5), "export did not pin its pre-purge snapshot"
    try:
        assert writer.purge(target.memory_id)["ok"] is True
    finally:
        resume.set()
        thread.join(10)
        writer.close()
    assert not thread.is_alive()
    assert failures == []
    raw = canonical_export_bytes(result[0])
    assert canary.encode() not in raw
    assert base64.b64encode(canary.encode()) not in raw


def test_purged_canaries_are_absent_raw_and_encoded_and_cannot_reappear(
    portable_home, tmp_path, monkeypatch
):
    canary = "PURGED-E4-SECRET-9b7b65"
    with Store("purged") as store:
        first = store.observe(
            canary,
            origin=canary,
            meta={canary: canary},
            defer_embedding=True,
        )
        corrected = store.contradict(
            first.memory_id,
            replacement="surviving replacement",
            reason=canary,
            origin=canary,
            session_id=canary,
            idempotency_key=canary,
        )
        replacement_id = corrected["replacement_memory_id"]
        assert store.purge(first.memory_id)["ok"] is True
    bundle = build_namespace_export("purged")
    raw = canonical_export_bytes(bundle)
    assert canary.encode() not in raw
    assert base64.b64encode(canary.encode()) not in raw
    assert len(bundle["records"]["lineage_tombstones"]) == 1
    assert set(bundle["records"]["lineage_tombstones"][0]) == {
        "schema_version", "tombstone_id", "status", "erased_at"
    }

    _switch_home(monkeypatch, tmp_path / "purge-destination")
    import_namespace_bytes(raw)
    reexport = canonical_export_bytes(build_namespace_export("purged"))
    assert canary.encode() not in reexport
    assert base64.b64encode(canary.encode()) not in reexport
    with open_existing("purged") as store:
        trace = store.trace(replacement_id)
        assert trace["lineage_status"] == "linked"
        assert trace["members"][0]["status"] == "erased"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda b: b["version"].update(major=FORMAT_MAJOR + 1), "unsupported export major"),
        (lambda b: b["version"].update(minor=FORMAT_MINOR + 1), "unsupported export minor"),
        (lambda b: b["manifest"].update(semantic_digest="sha256:" + "0" * 64), "semantic digest mismatch"),
    ],
)
def test_version_and_digest_skew_fail_before_namespace_mutation(
    portable_home, tmp_path, monkeypatch, mutation, message
):
    bundle, _ = _seed_bundle()
    mutation(bundle)
    raw = canonical_export_bytes(bundle)
    _switch_home(monkeypatch, tmp_path / message.replace(" ", "-"))
    with pytest.raises(ImportBundleError, match=message):
        import_namespace_bytes(raw)
    assert not namespace_exists_readonly("portable")


@pytest.mark.parametrize("table", [
    "sessions", "events", "memories", "lineage_tombstones",
    "corrections", "entities", "entity_mentions", "relation_evidence",
])
def test_each_corrupt_record_class_rolls_back_without_namespace_or_jobs(
    portable_home, tmp_path, monkeypatch, table
):
    # Seed every class, including a surviving privacy tombstone and graph data.
    with Store("all-records") as store:
        first = store.observe("AlphaService and BetaService", defer_embedding=True)
        correction = store.contradict(
            first.memory_id,
            replacement="GammaService and DeltaService survive",
            idempotency_key="all-records-correction",
        )
        store.purge(first.memory_id)
        assert correction["replacement_memory_id"]
    bundle = build_namespace_export("all-records")
    assert bundle["records"][table], table
    record = bundle["records"][table][0]
    field, value = {
        "sessions": ("id", None),
        "events": ("id", None),
        "memories": ("id", None),
        "lineage_tombstones": ("schema_version", None),
        "corrections": ("corrected_at", None),
        "entities": ("id", None),
        "entity_mentions": ("entity_id", "missing-entity"),
        "relation_evidence": ("src_entity", "missing-entity"),
    }[table]
    record[field] = value
    raw = _redigest(bundle)

    _switch_home(monkeypatch, tmp_path / f"corrupt-{table}")
    with pytest.raises(ImportBundleError):
        import_namespace_bytes(raw)
    assert not namespace_exists_readonly("all-records")
    namespace_root = tmp_path / f"corrupt-{table}" / "namespaces"
    assert not namespace_root.exists() or not list(namespace_root.glob("*.db"))


def test_duplicate_json_keys_non_utf8_nonfinite_and_compression_are_rejected(
    portable_home, tmp_path, monkeypatch
):
    bundle, _ = _seed_bundle()
    raw = canonical_export_bytes(bundle)
    cases = (
        raw.replace(b'"format":', b'"format":"duplicate","format":', 1),
        raw[:-1] + b',"x":NaN}',
        b'{"x":"\xff"}',
        b"\x1f\x8b" + raw,
    )
    for index, malformed in enumerate(cases):
        _switch_home(monkeypatch, tmp_path / f"malformed-{index}")
        with pytest.raises(ImportBundleError):
            import_namespace_bytes(malformed)
        assert not namespace_exists_readonly("portable")


def test_streaming_limits_enforce_actual_usage_and_clamp_requests(
    portable_home, tmp_path, monkeypatch
):
    bundle, _ = _seed_bundle()
    raw = canonical_export_bytes(bundle)
    total_records = bundle["manifest"]["total_records"]
    largest_record = max(
        len(_canonical_bytes(record))
        for records in bundle["records"].values()
        for record in records
    )
    clamped = resolve_import_limits(
        input_bytes=10**12,
        decompressed_bytes=10**12,
        records=10**12,
        record_bytes=10**12,
        json_depth=10**12,
        collection_items=10**12,
        timeout_seconds=10**12,
    )
    assert clamped == ImportLimits(
        input_bytes=256 * 1024 * 1024,
        decompressed_bytes=256 * 1024 * 1024,
        records=1_000_000,
        record_bytes=8 * 1024 * 1024,
        json_depth=64,
        collection_items=100_000,
        timeout_seconds=300.0,
    )

    failure_limits = (
        ImportLimits(input_bytes=len(raw) - 1),
        ImportLimits(decompressed_bytes=len(raw) - 1),
        ImportLimits(records=total_records - 1),
        ImportLimits(record_bytes=largest_record - 1),
        ImportLimits(json_depth=3),
        ImportLimits(collection_items=14),
    )
    for index, limits in enumerate(failure_limits):
        _switch_home(monkeypatch, tmp_path / f"limit-{index}")
        with pytest.raises(ImportLimitError):
            import_namespace_bytes(raw, limits=limits)
        assert not namespace_exists_readonly("portable")

    _switch_home(monkeypatch, tmp_path / "exact-boundaries")
    exact = ImportLimits(
        input_bytes=len(raw),
        decompressed_bytes=len(raw),
        records=total_records,
        record_bytes=largest_record,
        json_depth=8,
        collection_items=32,
    )
    report = import_namespace_bytes(raw, limits=exact)
    assert report["limits"] == {
        **exact.__dict__,
        "timeout_seconds": exact.timeout_seconds,
    }


def test_injected_timeout_covers_whitespace_parser_and_leaves_no_temp_or_jobs(
    portable_home, tmp_path, monkeypatch
):
    bundle, _ = _seed_bundle()
    raw = b" " * 8192 + canonical_export_bytes(bundle)
    destination = tmp_path / "timeout"
    _switch_home(monkeypatch, destination)
    ticks = iter([0.0, 0.0, 0.0, 31.0])

    def clock() -> float:
        return next(ticks, 31.0)

    with pytest.raises(ImportLimitError, match="timeout"):
        import_namespace_bytes(raw, _clock=clock)
    assert not namespace_exists_readonly("portable")
    namespace_root = destination / "namespaces"
    assert not namespace_root.exists() or list(namespace_root.iterdir()) == []


def test_injected_timeout_inside_scratch_validation_leaves_no_namespace(
    portable_home, tmp_path, monkeypatch
):
    import haunt.portability as portability

    bundle, _ = _seed_bundle()
    raw = canonical_export_bytes(bundle)
    destination = tmp_path / "scratch-timeout"
    _switch_home(monkeypatch, destination)
    entered_scratch = False
    original = portability._validate_records_in_scratch

    def wrapped(records, check_deadline):
        nonlocal entered_scratch
        entered_scratch = True
        return original(records, check_deadline)

    monkeypatch.setattr(portability, "_validate_records_in_scratch", wrapped)

    def clock() -> float:
        return 31.0 if entered_scratch else 0.0

    with pytest.raises(ImportLimitError, match="timeout"):
        import_namespace_bytes(raw, _clock=clock)
    assert entered_scratch is True
    assert not namespace_exists_readonly("portable")
    namespace_root = destination / "namespaces"
    assert not namespace_root.exists() or list(namespace_root.iterdir()) == []


def test_sqlite_progress_timeout_inside_scratch_rolls_back_and_cleans_up(
    portable_home, tmp_path, monkeypatch
):
    import haunt.portability as portability

    bundle, _ = _seed_bundle()
    raw = canonical_export_bytes(bundle)
    destination = tmp_path / "sqlite-progress-timeout"
    _switch_home(monkeypatch, destination)
    in_sqlite = False
    original_init = portability._init_namespace_schema

    def slow_init(conn):
        nonlocal in_sqlite
        original_init(conn)
        in_sqlite = True
        try:
            conn.execute(
                "WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL "
                "SELECT x+1 FROM n WHERE x<1000000) SELECT sum(x) FROM n"
            ).fetchone()
        finally:
            in_sqlite = False

    monkeypatch.setattr(portability, "_init_namespace_schema", slow_init)

    def clock() -> float:
        return 31.0 if in_sqlite else 0.0

    with pytest.raises(ImportLimitError, match="timeout"):
        import_namespace_bytes(raw, _clock=clock)
    assert not namespace_exists_readonly("portable")
    namespace_root = destination / "namespaces"
    assert not namespace_root.exists() or list(namespace_root.iterdir()) == []


def test_same_namespace_id_different_label_and_alias_collision_fail_before_mutation(
    portable_home, tmp_path, monkeypatch
):
    bundle, _ = _seed_bundle()
    destination = tmp_path / "identity-conflicts"
    _switch_home(monkeypatch, destination)
    with Store("different-label") as store:
        existing_id = store.namespace_id
        store.observe("destination canary", defer_embedding=True)
        before = _durable_snapshot(store)

    wrong_identity = copy.deepcopy(bundle)
    wrong_identity["namespace"]["namespace_id"] = existing_id
    with pytest.raises(ImportConflictError, match="identity or aliases"):
        import_namespace_bytes(_redigest(wrong_identity))
    with open_existing("different-label") as store:
        assert _durable_snapshot(store) == before

    # A different source ID cannot claim an already-owned label either.
    alias_collision = copy.deepcopy(bundle)
    alias_collision["namespace"]["canonical_label"] = "different-label"
    alias_collision["namespace"]["aliases"] = [
        {"label": "different-label", "is_canonical": True, "source_alias_norm": None}
    ]
    with pytest.raises(ImportConflictError, match="alias collision"):
        import_namespace_bytes(_redigest(alias_collision))
    with open_existing("different-label") as store:
        assert _durable_snapshot(store) == before


def test_existing_import_label_reassignment_never_writes_the_new_label_owner(
    portable_home, tmp_path, monkeypatch
):
    import haunt.portability as portability

    initial_bundle, memory_id = _seed_bundle()
    _switch_home(monkeypatch, tmp_path / "import-reassignment")
    import_namespace_bytes(canonical_export_bytes(initial_bundle))
    renamed = change_namespace_label(
        "portable", "shared", action="rename", apply=False
    )
    change_namespace_label(
        "portable",
        "shared",
        action="rename",
        apply=True,
        plan_digest=renamed["plan_digest"],
    )
    # The migrated canonical label intentionally differs from the unchanged
    # physical portable.db path, so it can later be retired and reassigned.
    bundle = build_namespace_export("shared")
    raw = canonical_export_bytes(bundle)
    with Store("other") as other:
        other.observe("other namespace canary", defer_embedding=True)
        other_id = other.namespace_id
        other_before = _durable_snapshot(other)

    selected = threading.Event()
    reassigned = threading.Event()

    def pause_after_preflight(_existing):
        selected.set()
        assert reassigned.wait(5), "label was not reassigned"

    monkeypatch.setattr(
        portability, "_existing_import_after_preflight_hook", pause_after_preflight
    )
    failures: list[BaseException] = []

    def run_import():
        try:
            import_namespace_bytes(raw)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run_import)
    thread.start()
    assert selected.wait(5), "import did not complete initial preflight"
    first = change_namespace_label("shared", "moved", action="rename", apply=False)
    change_namespace_label(
        "shared",
        "moved",
        action="rename",
        apply=True,
        plan_digest=first["plan_digest"],
    )
    retire_namespace_alias("shared", apply=True)
    second = change_namespace_label("other", "shared", action="alias", apply=False)
    change_namespace_label(
        "other",
        "shared",
        action="alias",
        apply=True,
        plan_digest=second["plan_digest"],
    )
    reassigned.set()
    thread.join(10)
    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], ImportConflictError)
    assert resolve_namespace_identity("shared")["namespace_id"] == other_id
    with open_existing("other") as other:
        assert _durable_snapshot(other) == other_before
        assert other.get_memory(memory_id) is None


def test_duplicate_record_id_with_different_bytes_rolls_back_existing_store(
    portable_home, tmp_path, monkeypatch
):
    bundle, memory_id = _seed_bundle()
    destination = tmp_path / "record-conflict"
    _switch_home(monkeypatch, destination)
    import_namespace_bytes(canonical_export_bytes(bundle))
    with open_existing("portable") as store:
        before = _durable_snapshot(store)

    conflict = copy.deepcopy(bundle)
    conflict["records"]["memories"][0]["content"] = "different bytes"
    with pytest.raises(ImportConflictError, match="memories identity conflicts"):
        import_namespace_bytes(_redigest(conflict))
    with open_existing("portable") as store:
        assert store.get_memory(memory_id)["content"] == "Unicode café ☃ canonical transfer"
        assert _durable_snapshot(store) == before


@pytest.mark.parametrize("drift", ["tamper", "delete"])
def test_import_receipt_never_hides_durable_record_drift(
    portable_home, tmp_path, monkeypatch, drift
):
    bundle, memory_id = _seed_bundle()
    raw = canonical_export_bytes(bundle)
    _switch_home(monkeypatch, tmp_path / f"receipt-{drift}")
    import_namespace_bytes(raw)
    with open_existing("portable") as store:
        if drift == "tamper":
            store.conn.execute(
                "UPDATE memories SET content='receipt drift' WHERE id=?", (memory_id,)
            )
        else:
            store.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        store.conn.commit()
        before = _durable_snapshot(store)

    with pytest.raises(ImportConflictError):
        import_namespace_bytes(raw)
    with open_existing("portable") as store:
        assert _durable_snapshot(store) == before


def test_invalid_structured_provenance_fails_before_any_destination_mutation(
    portable_home, tmp_path, monkeypatch
):
    bundle, _ = _seed_bundle()
    bundle["records"]["events"][0]["provenance"] = json.dumps(
        {"schema_version": 99, "kind": "native", "channel": "python", "origin": "python"}
    )
    raw = _redigest(bundle)
    _switch_home(monkeypatch, tmp_path / "bad-provenance")
    with pytest.raises(ImportBundleError, match="provenance"):
        import_namespace_bytes(raw)
    assert not namespace_exists_readonly("portable")


def test_export_file_is_exclusive_mode_0600_and_cli_reports_resolved_limits(
    portable_home, tmp_path, monkeypatch
):
    _seed_bundle()
    output = tmp_path / "portable.json"
    report = export_namespace_path("portable", output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert report["warning"].startswith("Export contains potentially sensitive")
    with pytest.raises(ExportError):
        export_namespace_path("portable", output)

    destination = tmp_path / "cli-destination"
    _switch_home(monkeypatch, destination)
    result = CliRunner().invoke(
        app,
        ["import", str(output), "--json", "--records", "999999999", "--timeout", "9999"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["limits"]["records"] == 1_000_000
    assert payload["limits"]["timeout_seconds"] == 300.0


def test_mcp_export_import_are_admin_only_and_round_trip(portable_home, tmp_path, monkeypatch):
    import haunt.mcp_server as mcp

    bundle, _ = _seed_bundle()
    mcp._MCP_AUTHORITY = None
    monkeypatch.delenv("HAUNT_MCP_ADMIN", raising=False)
    denied = json.loads(mcp.memory_export_bundle("portable"))
    assert denied["ok"] is False

    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    mcp._MCP_AUTHORITY = None
    exported = json.loads(mcp.memory_export_bundle("portable"))
    assert exported["ok"] is True
    assert exported["semantic_digest"] == bundle["manifest"]["semantic_digest"]

    _switch_home(monkeypatch, tmp_path / "mcp-destination")
    mcp._MCP_AUTHORITY = None
    imported = json.loads(mcp.memory_import_bundle(exported["bundle_json"]))
    assert imported["ok"] is True
    assert imported["created_namespace"] is True
    replay = json.loads(mcp.memory_import_bundle(exported["bundle_json"]))
    assert replay["deduplicated"] is True


def test_dashboard_export_is_authenticated_unknown_safe_and_header_safe(
    portable_home,
):
    with Store('evil"><script>alert(1)</script>') as store:
        store.observe("dashboard export", defer_embedding=True)
        canonical = store.name
    client = make_dash_client()
    missing = client.get("/api/namespace/definitely-missing/export")
    assert missing.status_code == 404
    assert not namespace_exists_readonly("definitely-missing")
    unauthorized = make_dash_client(token=None).get(
        f"/api/namespace/{canonical}/export"
    )
    assert unauthorized.status_code == 401
    response = client.get(f"/api/namespace/{canonical}/export")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    disposition = response.headers["content-disposition"]
    assert "attachment" in disposition
    assert all(token not in disposition for token in ('<', '>', '"<', "\r", "\n"))
    bundle = json.loads(response.content)
    assert bundle["format"] == FORMAT_NAME
    assert response.headers["x-haunt-semantic-digest"] == bundle["manifest"]["semantic_digest"]


def test_dashboard_import_requires_token_origin_media_type_and_bounded_body(
    portable_home, tmp_path, monkeypatch
):
    bundle, _ = _seed_bundle()
    raw = canonical_export_bytes(bundle)
    _switch_home(monkeypatch, tmp_path / "dashboard-destination")
    client = make_dash_client()
    path = "/api/import"
    headers = {
        "Origin": "http://127.0.0.1:7340",
        "Content-Type": "application/vnd.haunt.namespace-export+json",
    }
    assert make_dash_client(token=None).post(path, content=raw, headers=headers).status_code == 401
    assert client.post(
        path,
        content=raw,
        headers={"Content-Type": "application/vnd.haunt.namespace-export+json"},
    ).status_code == 403
    assert client.post(
        path,
        content=raw,
        headers={**headers, "Origin": "http://evil.example"},
    ).status_code == 403
    assert client.post(
        path,
        content=raw,
        headers={**headers, "Content-Type": "text/plain"},
    ).status_code == 415
    too_large = client.post(path + "?input_bytes=1", content=raw, headers=headers)
    assert too_large.status_code == 413
    assert not namespace_exists_readonly("portable")

    imported = client.post(path, content=raw, headers=headers)
    assert imported.status_code == 200, imported.text
    assert imported.json()["created_namespace"] is True
    assert imported.json()["limits"]["timeout_seconds"] == 30.0


def test_dashboard_import_conflict_is_409_and_preserves_destination(
    portable_home, tmp_path, monkeypatch
):
    bundle, _ = _seed_bundle()
    _switch_home(monkeypatch, tmp_path / "dashboard-conflict")
    raw = canonical_export_bytes(bundle)
    headers = {
        "Origin": "http://127.0.0.1:7340",
        "Content-Type": "application/json",
    }
    client = make_dash_client()
    assert client.post("/api/import", content=raw, headers=headers).status_code == 200
    with open_existing("portable") as store:
        before = _durable_snapshot(store)
    conflict = copy.deepcopy(bundle)
    conflict["records"]["memories"][0]["content"] = "dashboard conflict"
    response = client.post(
        "/api/import", content=_redigest(conflict), headers=headers
    )
    assert response.status_code == 409
    with open_existing("portable") as store:
        assert _durable_snapshot(store) == before


def test_empty_namespace_has_deterministic_sentinel_cut(portable_home):
    with Store("empty"):
        pass
    bundle = build_namespace_export("empty", exported_at="2030-01-01T00:00:00Z")
    assert bundle["format"] == FORMAT_NAME
    assert bundle["version"] == {"major": FORMAT_MAJOR, "minor": FORMAT_MINOR}
    assert bundle["temporal_cut"] == "1970-01-01T00:00:00.000000+00:00"
    assert bundle["manifest"]["total_records"] == 0


def test_session_without_events_is_portable_and_keeps_default_digest(
    portable_home, tmp_path, monkeypatch
):
    with Store("session-only") as store:
        session_id = store.ensure_session("standalone-session", source="test")
        assert store.end_session(session_id)["ok"] is True
    bundle = build_namespace_export("session-only")
    assert [row["id"] for row in bundle["records"]["sessions"]] == [session_id]
    assert all(not rows for name, rows in bundle["records"].items() if name != "sessions")

    _switch_home(monkeypatch, tmp_path / "session-only-destination")
    import_namespace_bytes(canonical_export_bytes(bundle))
    reexport = build_namespace_export("session-only")
    assert reexport["manifest"]["semantic_digest"] == bundle["manifest"]["semantic_digest"]


def test_pre_purge_bundle_cannot_resurrect_raw_or_blob_canaries_after_restart(
    portable_home, tmp_path, monkeypatch
):
    raw_canary = "RAW-PURGE-CANARY-7fbe2dd3"
    blob_canary = b"\x00BLOB-PURGE-CANARY-9a5c\xff"
    blob_token = base64.b64encode(blob_canary).decode("ascii")
    with Store("purge-watermark") as store:
        observed = store.observe(raw_canary, defer_embedding=True)
        store.conn.execute(
            "UPDATE events SET tool_output=? WHERE id=?",
            (blob_canary, observed.event_id),
        )
        store.conn.commit()
    stale = build_namespace_export("purge-watermark")
    stale_raw = canonical_export_bytes(stale)
    assert raw_canary.encode() in stale_raw
    assert blob_token.encode() in stale_raw

    with open_existing("purge-watermark") as store:
        purged = store.purge(observed.memory_id)
        assert purged["ok"] is True
        assert "privacy_lineage_head" not in purged
    current = build_namespace_export("purge-watermark")
    current_raw = canonical_export_bytes(current)
    assert current["namespace"]["privacy_lineage_head"] != stale["namespace"][
        "privacy_lineage_head"
    ]
    assert raw_canary.encode() not in current_raw
    assert blob_token.encode() not in current_raw

    # A bundle at the current privacy head is valid and exactly idempotent.
    first = import_namespace_bytes(current_raw)
    second = import_namespace_bytes(current_raw)
    assert first["created_namespace"] is False
    assert second["deduplicated"] is True
    with open_existing("purge-watermark") as store:
        before = _rejection_snapshot(store)
    with pytest.raises(ImportConflictError, match="privacy lineage"):
        import_namespace_bytes(stale_raw)
    with open_existing("purge-watermark") as store:
        assert _rejection_snapshot(store) == before
        assert store.get_memory(observed.memory_id) is None

    # The same old bundle remains valid for a genuinely fresh home, where its
    # opaque lineage head is preserved exactly.
    _switch_home(monkeypatch, tmp_path / "fresh-pre-purge")
    imported = import_namespace_bytes(stale_raw)
    assert imported["created_namespace"] is True
    reexport = build_namespace_export("purge-watermark")
    assert reexport["manifest"]["semantic_digest"] == stale["manifest"][
        "semantic_digest"
    ]
    with open_existing("purge-watermark") as store:
        event = store.conn.execute(
            "SELECT tool_output FROM events WHERE id=?", (observed.event_id,)
        ).fetchone()
        assert event["tool_output"] == blob_canary


@pytest.mark.parametrize("purged_member", ["first", "middle", "last", "all"])
def test_stale_bundle_cannot_restore_any_purged_correction_chain_member(
    portable_home, monkeypatch, purged_member
):
    contents = {
        "first": "CHAIN-PURGE-FIRST-4e21",
        "middle": "CHAIN-PURGE-MIDDLE-5f32",
        "last": "CHAIN-PURGE-LAST-6a43",
    }
    with Store("purge-chain") as store:
        first = store.observe(contents["first"], defer_embedding=True)
        second = store.contradict(
            first.memory_id,
            replacement=contents["middle"],
            idempotency_key="purge-chain-second",
        )
        third = store.contradict(
            second["replacement_memory_id"],
            replacement=contents["last"],
            idempotency_key="purge-chain-third",
        )
    memory_ids = {
        "first": first.memory_id,
        "middle": second["replacement_memory_id"],
        "last": third["replacement_memory_id"],
    }
    stale_raw = canonical_export_bytes(build_namespace_export("purge-chain"))
    targets = list(memory_ids) if purged_member == "all" else [purged_member]
    with open_existing("purge-chain") as store:
        for target in targets:
            assert store.purge(memory_ids[target])["ok"] is True

    after_raw = canonical_export_bytes(build_namespace_export("purge-chain"))
    for target in targets:
        assert contents[target].encode() not in after_raw
    # Close/reopen before replay so the lineage protection is demonstrably
    # durable rather than process-local.
    with open_existing("purge-chain") as store:
        before = _rejection_snapshot(store)
    with pytest.raises(ImportConflictError, match="privacy lineage"):
        import_namespace_bytes(stale_raw)
    with open_existing("purge-chain") as store:
        assert _rejection_snapshot(store) == before
        for target in targets:
            assert store.get_memory(memory_ids[target]) is None


def test_privacy_lineage_forks_diverge_and_post_purge_bundle_round_trips_fresh(
    portable_home, tmp_path, monkeypatch
):
    bundle, memory_id = _seed_bundle()
    raw = canonical_export_bytes(bundle)
    heads: dict[str, str] = {}
    post_bundle: dict | None = None
    for fork in ("a", "b"):
        _switch_home(monkeypatch, tmp_path / f"fork-{fork}")
        import_namespace_bytes(raw)
        with open_existing("portable") as store:
            assert store.purge(memory_id)["ok"] is True
        exported = build_namespace_export("portable")
        heads[fork] = exported["namespace"]["privacy_lineage_head"]
        if fork == "a":
            post_bundle = exported
    assert heads["a"] != heads["b"]
    assert post_bundle is not None

    _switch_home(monkeypatch, tmp_path / "fork-b")
    with open_existing("portable") as store:
        before = _rejection_snapshot(store)
    with pytest.raises(ImportConflictError, match="privacy lineage"):
        import_namespace_bytes(canonical_export_bytes(post_bundle))
    with open_existing("portable") as store:
        assert _rejection_snapshot(store) == before

    _switch_home(monkeypatch, tmp_path / "fork-fresh")
    import_namespace_bytes(canonical_export_bytes(post_bundle))
    reexport = build_namespace_export("portable")
    assert reexport["manifest"]["semantic_digest"] == post_bundle["manifest"][
        "semantic_digest"
    ]
    assert reexport["namespace"]["privacy_lineage_head"] == heads["a"]


def test_malformed_privacy_head_fails_closed_and_purge_rolls_back(
    portable_home
):
    with Store("bad-privacy-head") as store:
        observed = store.observe("must survive failed purge", defer_embedding=True)
        store.conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
            (PRIVACY_LINEAGE_KEY, "malformed-private-state"),
        )
        store.conn.commit()
        before = _rejection_snapshot(store)
        with pytest.raises(NamespacePathError, match="privacy lineage head is malformed"):
            store.purge(observed.memory_id)
        assert _rejection_snapshot(store) == before
        assert store.get_memory(observed.memory_id) is not None
    with pytest.raises(NamespacePathError, match="privacy lineage head is malformed"):
        build_namespace_export("bad-privacy-head")


@pytest.mark.parametrize(
    "mutation",
    [
        "count-bool",
        "count-float",
        "count-string",
        "count-negative",
        "count-huge",
        "total-bool",
        "total-float",
        "total-string",
        "total-negative",
        "total-huge",
        "total-sum",
        "missing-key",
        "extra-key",
    ],
)
def test_manifest_counts_require_exact_bounded_nonnegative_integers(
    portable_home, tmp_path, monkeypatch, mutation
):
    bundle, _ = _seed_bundle()
    counts = bundle["manifest"]["record_counts"]
    if mutation.startswith("count-"):
        counts["sessions"] = {
            "count-bool": True,
            "count-float": 1.0,
            "count-string": "1",
            "count-negative": -1,
            "count-huge": 100_001,
        }[mutation]
    elif mutation == "missing-key":
        counts.pop("sessions")
    elif mutation == "extra-key":
        counts["derived_jobs"] = 0
    elif mutation == "total-sum":
        bundle["manifest"]["total_records"] += 1
    else:
        bundle["manifest"]["total_records"] = {
            "total-bool": True,
            "total-float": float(bundle["manifest"]["total_records"]),
            "total-string": str(bundle["manifest"]["total_records"]),
            "total-negative": -1,
            "total-huge": 100_001,
        }[mutation]
    destination = tmp_path / f"manifest-{mutation}"
    _switch_home(monkeypatch, destination)
    with pytest.raises(ImportBundleError, match="manifest"):
        import_namespace_bytes(canonical_export_bytes(bundle))
    assert not namespace_exists_readonly("portable")
    root = destination / "namespaces"
    assert not root.exists() or not list(root.glob("*.db"))


def test_rejected_existing_conflict_runs_no_schema_graph_or_meta_maintenance(
    portable_home, tmp_path, monkeypatch
):
    bundle, memory_id = _seed_bundle()
    raw = canonical_export_bytes(bundle)
    _switch_home(monkeypatch, tmp_path / "zero-write-existing-conflict")
    import_namespace_bytes(raw)
    with open_existing("portable") as store:
        # Recreate the state that the old normal Store opener repaired before
        # discovering a conflict.
        store.conn.execute("DELETE FROM meta WHERE key='graph_evidence_version'")
        store.conn.execute("DELETE FROM relations")
        store.conn.execute(
            "UPDATE memories SET content='destination conflict' WHERE id=?",
            (memory_id,),
        )
        store.conn.commit()
        before = _rejection_snapshot(store)
        with pytest.raises(ImportConflictError, match="memories identity conflicts"):
            import_namespace_bytes(raw)
        assert _rejection_snapshot(store) == before


def test_existing_zero_write_preflight_sqlite_timeout_preserves_every_state(
    portable_home, tmp_path, monkeypatch
):
    import haunt.portability as portability

    bundle, _ = _seed_bundle()
    raw = canonical_export_bytes(bundle)
    _switch_home(monkeypatch, tmp_path / "existing-progress-timeout")
    import_namespace_bytes(raw)
    with open_existing("portable") as store:
        before = _rejection_snapshot(store)

    in_sqlite = False
    original_match = portability._existing_row_matches

    def slow_match(conn, table, record):
        nonlocal in_sqlite
        in_sqlite = True
        try:
            conn.execute(
                "WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL "
                "SELECT x+1 FROM n WHERE x<1000000) SELECT sum(x) FROM n"
            ).fetchone()
        finally:
            in_sqlite = False
        return original_match(conn, table, record)

    def arm_after_validation(_existing):
        monkeypatch.setattr(portability, "_existing_row_matches", slow_match)

    monkeypatch.setattr(
        portability, "_existing_import_after_preflight_hook", arm_after_validation
    )

    def clock() -> float:
        return 31.0 if in_sqlite else 0.0

    with pytest.raises(ImportLimitError, match="timeout"):
        import_namespace_bytes(raw, _clock=clock)
    with open_existing("portable") as store:
        assert _rejection_snapshot(store) == before


def test_existing_import_requires_current_schema_without_running_migration(
    portable_home, tmp_path, monkeypatch
):
    bundle, _ = _seed_bundle()
    raw = canonical_export_bytes(bundle)
    _switch_home(monkeypatch, tmp_path / "existing-old-schema")
    import_namespace_bytes(raw)
    with open_existing("portable") as store:
        store.conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'",
            (str(SCHEMA_VERSION - 1),),
        )
        store.conn.commit()
        before = _rejection_snapshot(store)
        with pytest.raises(ImportConflictError, match="already be schema"):
            import_namespace_bytes(raw)
        assert _rejection_snapshot(store) == before


def test_valid_existing_import_still_rebuilds_destination_projections(
    portable_home, tmp_path, monkeypatch
):
    with Store("existing-rebuild") as store:
        observed = store.observe(
            "AlphaService works with BetaService", defer_embedding=True
        )
    empty = build_namespace_export(
        "existing-rebuild", cut="1970-01-01T00:00:00Z"
    )
    full = build_namespace_export("existing-rebuild")
    _switch_home(monkeypatch, tmp_path / "existing-rebuild-destination")
    import_namespace_bytes(canonical_export_bytes(empty))
    report = import_namespace_bytes(canonical_export_bytes(full))
    assert report["created_namespace"] is False
    assert report["inserted"]["memories"] == 1
    with open_existing("existing-rebuild") as store:
        assert store.get_memory(observed.memory_id) is not None
        assert store.conn.execute(
            "SELECT content FROM memories_fts WHERE id=?", (observed.memory_id,)
        ).fetchone()[0] == "AlphaService works with BetaService"
        assert store.conn.execute(
            "SELECT COUNT(*) FROM embedding_jobs WHERE memory_id=?",
            (observed.memory_id,),
        ).fetchone()[0] == 1
        assert store.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0] > 0


@pytest.mark.parametrize(
    "phase",
    [
        "intent-created",
        "staged",
        "linked",
        "named",
        "registry-precommit",
        "registry-committed",
    ],
)
def test_fresh_import_crash_recovery_has_no_orphan_and_retry_succeeds(
    portable_home, tmp_path, monkeypatch, phase
):
    bundle, memory_id = _seed_bundle()
    bundle_path = tmp_path / f"crash-{phase}.json"
    bundle_path.write_bytes(canonical_export_bytes(bundle))
    destination = tmp_path / f"crash-{phase}-home"
    script = """
import os
from pathlib import Path
import haunt.portability as portability

phase = os.environ["HAUNT_TEST_CRASH_PHASE"]
def crash(actual, _intent):
    if actual == phase:
        os._exit(73)
portability._import_publication_phase_hook = crash
portability.import_namespace_bytes(Path(os.environ["HAUNT_TEST_BUNDLE"]).read_bytes())
"""
    env = os.environ.copy()
    env.update(
        {
            "HAUNT_HOME": str(destination),
            "HAUNT_FTS_ONLY": "1",
            "HAUNT_EMBED_MODEL": "off",
            "HAUNT_TEST_CRASH_PHASE": phase,
            "HAUNT_TEST_BUNDLE": str(bundle_path),
            "PYTHONPATH": str(Path(__file__).parents[1] / "src")
            + os.pathsep
            + env.get("PYTHONPATH", ""),
        }
    )
    crashed = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert crashed.returncode == 73, crashed.stderr
    crashed_intent_dir = destination / "import-intents"
    crashed_intents = list(crashed_intent_dir.glob("*.json"))
    assert len(crashed_intents) == 1
    assert stat.S_IMODE(crashed_intent_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(crashed_intents[0].stat().st_mode) == 0o600

    _switch_home(monkeypatch, destination)
    report = import_namespace_bytes(bundle_path.read_bytes())
    assert report["semantic_digest"] == bundle["manifest"]["semantic_digest"]
    with open_existing("portable") as store:
        assert store.get_memory(memory_id) is not None
    intent_dir = destination / "import-intents"
    assert intent_dir.is_dir()
    assert list(intent_dir.iterdir()) == []
    namespace_files = list((destination / "namespaces").iterdir())
    assert [path.name for path in namespace_files if path.suffix == ".db"] == [
        "portable.db"
    ]
    assert not [path for path in destination.rglob("*") if ".haunt-claim-" in path.name]


@pytest.mark.parametrize("replacement", ["symlink", "hardlink", "owned-hardlink"])
def test_crash_recovery_never_deletes_replaced_or_unrelated_target(
    portable_home, tmp_path, monkeypatch, replacement
):
    bundle, _ = _seed_bundle()
    raw_path = tmp_path / f"attacker-{replacement}.json"
    raw_path.write_bytes(canonical_export_bytes(bundle))
    destination = tmp_path / f"attacker-{replacement}-home"
    script = """
import os
from pathlib import Path
import haunt.portability as portability
def crash(actual, _intent):
    if actual == "named":
        os._exit(74)
portability._import_publication_phase_hook = crash
portability.import_namespace_bytes(Path(os.environ["HAUNT_TEST_BUNDLE"]).read_bytes())
"""
    env = os.environ.copy()
    env.update(
        {
            "HAUNT_HOME": str(destination),
            "HAUNT_FTS_ONLY": "1",
            "HAUNT_EMBED_MODEL": "off",
            "HAUNT_TEST_BUNDLE": str(raw_path),
            "PYTHONPATH": str(Path(__file__).parents[1] / "src")
            + os.pathsep
            + env.get("PYTHONPATH", ""),
        }
    )
    crashed = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert crashed.returncode == 74, crashed.stderr
    target = destination / "namespaces" / "portable.db"
    owned = target.with_name("owned-import-primary.db")
    unrelated = tmp_path / f"unrelated-{replacement}.txt"
    unrelated.write_bytes(b"UNRELATED-RECOVERY-CANARY")
    if replacement == "owned-hardlink":
        os.link(target, owned)
    else:
        target.rename(owned)
    if replacement == "symlink":
        target.symlink_to(unrelated)
    elif replacement == "hardlink":
        os.link(unrelated, target)

    _switch_home(monkeypatch, destination)
    with pytest.raises(
        ImportBundleError, match="refusing to remove|unexpected hardlink"
    ):
        import_namespace_bytes(raw_path.read_bytes())
    assert unrelated.read_bytes() == b"UNRELATED-RECOVERY-CANARY"
    assert target.exists()
    assert owned.exists()


_FIXTURES = Path(__file__).parent / "fixtures" / "export" / "v1"
_LEGACY_V1_0 = _FIXTURES / "legacy-v1.0.json"
_GOLDEN_V1_1 = _FIXTURES / "golden-v1.1.json"
_FAKE_DIM = 384
_FAKE_STATE = EmbedState(
    model_id="test-portability-model",
    requested="test-portability-model",
    dim=_FAKE_DIM,
    available=True,
    fallback=False,
)


def _fake_embed_texts(texts):
    return [[0.1] * _FAKE_DIM for _ in texts]


def _plain_vec_table(conn, dim, commit=True):
    """Stand-in for ensure_vec_table: portable_home runs without sqlite-vec."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vec_memories (id TEXT PRIMARY KEY, embedding BLOB)"
    )
    if commit:
        conn.commit()
    return True


def _seed_durable_fields(namespace: str = "durable") -> dict[str, str]:
    """Seed one namespace carrying every field the v1.1 minor added."""
    with Store(namespace) as store:
        host = store.ensure_session("host-session-alpha")
        ordinary = store.observe(
            "an ordinary note worth embedding",
            session_id=host,
            event_time="2026-01-01T00:00:00Z",
            defer_embedding=True,
        )
        excluded = store.observe(
            "",
            role="tool",
            tool_name="Bash",
            tool_output="ROUND-TRIP-EXCLUSION-TOKEN",
            session_id=host,
            event_time="2026-01-01T00:01:00Z",
            defer_embedding=True,
            skip_embedding=True,
        )
        store.end_session(host)
        successor = store.ensure_session("host-session-alpha")
        return {
            "host": host,
            "successor": successor,
            "ordinary": ordinary.memory_id,
            "excluded": excluded.memory_id,
        }


def _queued(store, memory_id) -> bool:
    return (
        store.conn.execute(
            "SELECT 1 FROM embedding_jobs WHERE memory_id=?", (memory_id,)
        ).fetchone()
        is not None
    )


def test_capture_policy_exclusion_survives_round_trip_and_reembed(
    portable_home, tmp_path, monkeypatch
):
    """The v13 exclusion is durable, so a bundle that drops it is lossy.

    Proves the transferred flag still does its whole job at the destination:
    no queue row at import, and no resurrection by a later full rebuild.
    """
    seeded = _seed_durable_fields()
    raw = canonical_export_bytes(build_namespace_export("durable"))

    memories = json.loads(raw)["records"]["memories"]
    assert {row["id"]: row["skip_embedding"] for row in memories} == {
        seeded["ordinary"]: 0,
        seeded["excluded"]: 1,
    }

    _switch_home(monkeypatch, tmp_path / "exclusion-destination")
    import_namespace_bytes(raw)
    with open_existing("durable") as store:
        flags = dict(
            store.conn.execute("SELECT id, skip_embedding FROM memories").fetchall()
        )
        assert flags[seeded["excluded"]] == 1
        assert flags[seeded["ordinary"]] == 0
        assert not _queued(store, seeded["excluded"])
        assert _queued(store, seeded["ordinary"])
        # Excluded rows stay fully keyword-searchable; only the vector index
        # is withheld, exactly as observe() writes them.
        assert store.conn.execute(
            "SELECT COUNT(*) FROM memories_fts WHERE id=?", (seeded["excluded"],)
        ).fetchone()[0] == 1

        with (
            patch("haunt.store.embed_state", return_value=_FAKE_STATE),
            patch("haunt.store.embed_texts", side_effect=_fake_embed_texts),
            patch("haunt.store.ensure_vec_table", side_effect=_plain_vec_table),
            patch.object(store, "vec_ok", return_value=True),
        ):
            rebuilt = store.reembed()

        assert rebuilt["skipped"] == 1
        vectored = {
            row["id"]
            for row in store.conn.execute("SELECT id FROM vec_memories").fetchall()
        }
        assert seeded["excluded"] not in vectored
        assert seeded["ordinary"] in vectored
        assert not _queued(store, seeded["excluded"])


def test_successor_linkage_survives_round_trip_and_does_not_fork_on_resume(
    portable_home, tmp_path, monkeypatch
):
    """A dropped successor link is invisible until the next `claude --resume`.

    The destination must reuse the imported successor rather than mint a
    second one, which is the only observable difference a lost link makes.
    """
    seeded = _seed_durable_fields()
    raw = canonical_export_bytes(build_namespace_export("durable"))
    with open_existing("durable") as store:
        source_sessions = sorted(
            tuple(row)
            for row in store.conn.execute(
                "SELECT id, succeeds_session FROM sessions"
            )
        )
    assert (seeded["successor"], seeded["host"]) in source_sessions

    _switch_home(monkeypatch, tmp_path / "successor-destination")
    import_namespace_bytes(raw)
    with open_existing("durable") as store:
        assert sorted(
            tuple(row)
            for row in store.conn.execute(
                "SELECT id, succeeds_session FROM sessions"
            )
        ) == source_sessions

        before = store.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        resumed = store.ensure_session("host-session-alpha")
        assert resumed == seeded["successor"]
        assert store.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == before


def test_imported_memories_carry_the_stores_own_content_hash(
    portable_home, tmp_path, monkeypatch
):
    """content_hash is reconstructed, not carried, so it is never NULL again.

    Compared against `_content_hash` itself rather than a second SHA-256 here,
    so a change to the store's hashing would fail this rather than agree with
    a copy of the old rule.
    """
    seeded = _seed_durable_fields()
    raw = canonical_export_bytes(build_namespace_export("durable"))
    with open_existing("durable") as store:
        source_hashes = dict(
            store.conn.execute("SELECT id, content_hash FROM memories").fetchall()
        )
    assert "content_hash" not in json.loads(raw)["records"]["memories"][0]

    _switch_home(monkeypatch, tmp_path / "hash-destination")
    import_namespace_bytes(raw)
    with open_existing("durable") as store:
        rows = store.conn.execute(
            "SELECT id, content, content_hash FROM memories"
        ).fetchall()
        assert len(rows) == len(source_hashes)
        for row in rows:
            assert row["content_hash"] == _content_hash(row["content"])
            assert row["content_hash"] == source_hashes[row["id"]]
        assert store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content_hash IS NULL"
        ).fetchone()[0] == 0
    assert seeded["excluded"] in source_hashes


def test_v1_0_bundles_import_under_documented_defaults(
    portable_home, tmp_path, monkeypatch
):
    """v1.1 is additive, so v1.0 keeps meaning exactly what it always meant.

    The two fixtures are one namespace exported twice, by the pre-change
    exporter and by this one, so the pair pins the superset relationship
    rather than asserting it against a hand-written expectation.
    """
    legacy_raw = _LEGACY_V1_0.read_bytes()
    legacy = json.loads(legacy_raw)
    current = json.loads(_GOLDEN_V1_1.read_bytes())
    assert legacy["version"] == {"major": FORMAT_MAJOR, "minor": 0}
    assert current["version"] == {"major": FORMAT_MAJOR, "minor": FORMAT_MINOR}
    assert canonical_export_bytes(legacy) + b"\n" == legacy_raw

    added = _MINOR_ADDED_FIELDS[1]
    assert {
        table: [
            {k: v for k, v in row.items() if k not in added.get(table, ())}
            for row in rows
        ]
        for table, rows in current["records"].items()
    } == legacy["records"]
    assert [row["succeeds_session"] for row in current["records"]["sessions"]] != [
        None,
        None,
    ]
    assert 1 in [row["skip_embedding"] for row in current["records"]["memories"]]

    _switch_home(monkeypatch, tmp_path / "legacy-destination")
    import_namespace_bytes(legacy_raw)
    with open_existing("golden") as store:
        # Documented v1.0 defaults: the destination column default for both,
        # which is exactly what a v1.0 bundle already imported as.
        assert store.conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE succeeds_session IS NOT NULL"
        ).fetchone()[0] == 0
        assert store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE skip_embedding!=0"
        ).fetchone()[0] == 0
        # No row is excluded, so every non-blank memory is queued: a legacy
        # bundle cannot silently lose its rows out of the vector index.
        assert store.conn.execute(
            "SELECT COUNT(*) FROM embedding_jobs"
        ).fetchone()[0] == len(legacy["records"]["memories"])
        # Reconstruction is version-independent, so v1.0 gains it too.
        for row in store.conn.execute("SELECT content, content_hash FROM memories"):
            assert row["content_hash"] == _content_hash(row["content"])


def test_v1_1_golden_bundle_is_canonical_and_reexports_identically(
    portable_home, tmp_path, monkeypatch
):
    """Export -> import -> re-export is stable under the volatile-field rule.

    `creation.exported_at` is the only declared volatile field, so the check
    is byte equality everywhere else, not just digest equality.
    """
    raw = _GOLDEN_V1_1.read_bytes()
    bundle = json.loads(raw)
    assert canonical_export_bytes(bundle) + b"\n" == raw
    assert bundle["creation"]["volatile_fields"] == ["creation.exported_at"]

    _switch_home(monkeypatch, tmp_path / "reexport-destination")
    report = import_namespace_bytes(raw)
    assert report["semantic_digest"] == bundle["manifest"]["semantic_digest"]

    reexport = build_namespace_export("golden", exported_at="2099-12-31T23:59:59Z")
    assert reexport["creation"]["exported_at"] != bundle["creation"]["exported_at"]
    assert canonical_export_bytes(reexport) != canonical_export_bytes(bundle)
    reexport["creation"]["exported_at"] = bundle["creation"]["exported_at"]
    assert canonical_export_bytes(reexport) + b"\n" == raw


@pytest.mark.parametrize(
    ("version", "message"),
    [
        ({"major": FORMAT_MAJOR, "minor": FORMAT_MINOR + 1}, "unsupported export minor"),
        ({"major": FORMAT_MAJOR, "minor": -1}, "unsupported export minor"),
        ({"major": FORMAT_MAJOR + 1, "minor": 0}, "unsupported export major"),
        ({"major": 0, "minor": FORMAT_MINOR}, "unsupported export major"),
    ],
)
def test_unknown_versions_fail_closed_before_namespace_mutation(
    portable_home, tmp_path, monkeypatch, version, message
):
    """An unreadable version must never reach a destination.

    A newer minor's added fields carry meaning this reader has no default
    for, so guessing at them is exactly the silent loss this format
    version exists to prevent.
    """
    _seed_durable_fields()
    bundle = build_namespace_export("durable")
    bundle["version"] = version
    raw = _redigest(bundle)

    _switch_home(monkeypatch, tmp_path / f"closed-{version['major']}-{version['minor']}")
    with pytest.raises(ImportBundleError, match=message):
        import_namespace_bytes(raw)
    assert not namespace_exists_readonly("durable")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda b: b["records"]["memories"][0].update(skip_embedding=2),
            "invalid memories.skip_embedding",
        ),
        (
            lambda b: b["records"]["memories"][0].update(skip_embedding=True),
            "invalid memories.skip_embedding",
        ),
        (
            lambda b: b["records"]["memories"][0].update(skip_embedding="1"),
            "invalid memories.skip_embedding",
        ),
        (
            lambda b: b["records"]["sessions"][0].update(succeeds_session="no-such-session"),
            "session references missing predecessor",
        ),
        (
            lambda b: b["records"]["sessions"][0].update(
                succeeds_session=b["records"]["sessions"][0]["id"]
            ),
            "session cannot succeed itself",
        ),
        (
            lambda b: [
                row.pop("skip_embedding") for row in b["records"]["memories"]
            ],
            r"invalid records.memories item",
        ),
        (
            lambda b: [
                row.update(unexpected_field=1) for row in b["records"]["sessions"]
            ],
            r"invalid records.sessions item",
        ),
    ],
)
def test_added_field_violations_fail_before_namespace_mutation(
    portable_home, tmp_path, monkeypatch, mutation, message
):
    """The added fields are validated, not merely copied.

    Neither has a destination CHECK or foreign key, so import is the only
    gate between a crafted bundle and a store that reads them as trusted.
    """
    _seed_durable_fields()
    bundle = build_namespace_export("durable")
    mutation(bundle)
    raw = _redigest(bundle)

    _switch_home(monkeypatch, tmp_path / f"reject-{abs(hash(message))}")
    with pytest.raises(ImportBundleError, match=message):
        import_namespace_bytes(raw)
    assert not namespace_exists_readonly("durable")


def test_declared_minor_pins_the_exact_record_field_set(
    portable_home, tmp_path, monkeypatch
):
    """A bundle is read at the minor it declares, never at a guessed one.

    Without this a v1.0 bundle could smuggle v1.1 fields past the defaults
    its own version promises, and a v1.1 bundle could omit them silently.
    """
    _seed_durable_fields()
    bundle = build_namespace_export("durable")
    downlevel = copy.deepcopy(bundle)
    downlevel["version"] = {"major": FORMAT_MAJOR, "minor": 0}

    _switch_home(monkeypatch, tmp_path / "smuggled")
    with pytest.raises(ImportBundleError, match=r"invalid records\.\w+ item"):
        import_namespace_bytes(_redigest(downlevel))
    assert not namespace_exists_readonly("durable")

    stripped = copy.deepcopy(bundle)
    stripped["version"] = {"major": FORMAT_MAJOR, "minor": 0}
    for table, fields in _MINOR_ADDED_FIELDS[1].items():
        for row in stripped["records"][table]:
            for field in fields:
                row.pop(field)
    _switch_home(monkeypatch, tmp_path / "downgraded")
    import_namespace_bytes(_redigest(stripped))
    with open_existing("durable") as store:
        assert store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE skip_embedding!=0"
        ).fetchone()[0] == 0


def test_v1_0_bundle_cannot_silently_unset_a_stored_added_field(
    portable_home, tmp_path, monkeypatch
):
    """A 1.0 default is what its silence means, not a value it may impose.

    Replaying the same namespace's older export onto rows that already carry
    the added fields is a real disagreement, so it conflicts rather than
    un-excluding rows and dropping successor links back out.
    """
    _switch_home(monkeypatch, tmp_path / "mixed-minor")
    import_namespace_bytes(_GOLDEN_V1_1.read_bytes())
    with pytest.raises(ImportConflictError, match="records.sessions identity conflicts"):
        import_namespace_bytes(_LEGACY_V1_0.read_bytes())
    with open_existing("golden") as store:
        assert store.conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE succeeds_session IS NOT NULL"
        ).fetchone()[0] == 1
        assert store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE skip_embedding=1"
        ).fetchone()[0] == 1
