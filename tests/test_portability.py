"""E4 canonical namespace export/import contract and release evidence."""

from __future__ import annotations

import base64
import copy
import json
import stat
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.dashutil import make_dash_client

from haunt.cli import app
from haunt.portability import (
    FORMAT_MAJOR,
    FORMAT_MINOR,
    FORMAT_NAME,
    ExportError,
    ImportBundleError,
    ImportConflictError,
    ImportLimitError,
    ImportLimits,
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
    SCHEMA_VERSION,
    Store,
    change_namespace_label,
    namespace_exists_readonly,
    open_existing,
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

    _switch_home(monkeypatch, tmp_path / "golden-destination")
    report = import_namespace_bytes(raw)
    assert report["semantic_digest"] == bundle["manifest"]["semantic_digest"]
    reexport = build_namespace_export("golden")
    assert reexport["manifest"]["semantic_digest"] == bundle["manifest"]["semantic_digest"]


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
