"""Canonical ordering for equal recall and timeline signals."""

from __future__ import annotations

import importlib

from haunt.recall import Hit
from haunt.store import Store, observe


def _hit(memory_id: str, *, score: float = 0.0, fts_rank: int | None = None) -> Hit:
    return Hit(
        memory_id=memory_id,
        event_id=f"event-{memory_id}",
        score=score,
        tier="episodic",
        content=memory_id,
        role="user",
        event_time="2026-08-08T12:00:00+00:00",
        valid_from="2026-08-08T12:00:00+00:00",
        valid_to=None,
        tool_name=None,
        fts_rank=fts_rank,
        fts_rank_raw=-1.0 if fts_rank is not None else None,
    )


def test_recall_ties_ignore_candidate_arrival_order(haunt_env, monkeypatch):
    """Equal RRF scores settle on content even when modality input is reversed.

    Arrival order must not decide the result. The key is the content hash
    rather than the memory id, because the id is a fresh uuid4 per write and
    re-randomizes the answer on every ingest of the same corpus.
    """
    first = observe("RECALL-TIE-FIRST", namespace="default")
    second = observe("RECALL-TIE-SECOND", namespace="default")
    recall_module = importlib.import_module("haunt.recall")
    candidates = [
        (second.memory_id, 1, -1.0),
        (first.memory_id, 1, -1.0),
    ]
    monkeypatch.setattr(
        recall_module,
        "_fts_hits",
        lambda conn, query, where, params, limit: candidates,
    )

    reversed_hits = recall_module.recall(
        "RECALL-TIE", namespace="default", k=2, use_vectors=False
    )
    candidates.reverse()
    forward_hits = recall_module.recall(
        "RECALL-TIE", namespace="default", k=2, use_vectors=False
    )

    import hashlib

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    expected = [
        memory_id
        for _hash, memory_id in sorted(
            [
                (digest("RECALL-TIE-FIRST"), first.memory_id),
                (digest("RECALL-TIE-SECOND"), second.memory_id),
            ]
        )
    ]
    assert [hit.memory_id for hit in reversed_hits] == expected
    assert [hit.memory_id for hit in forward_hits] == expected
    assert [hit.final_rank for hit in forward_hits] == [1, 2]


def test_timeline_ties_use_memory_id_without_losing_time_order(haunt_env):
    """IDs settle exact timestamps only; newer events still appear first."""
    from haunt.planner import run_timeline
    from haunt.temporal import compile
    from tests.test_temporal_planner import NOW

    with Store("default") as store:
        tied_a = store.observe(
            "TIMELINE-TIE-A", event_time="2026-08-08T12:00:00+00:00"
        )
        tied_b = store.observe(
            "TIMELINE-TIE-B", event_time="2026-08-08T12:00:00+00:00"
        )
        later = store.observe(
            "TIMELINE-LATER", event_time="2026-08-08T13:00:00+00:00"
        )
        hits = run_timeline(compile("what happened two weeks ago", NOW), store)

    ids = [hit.memory_id for hit in hits]
    assert ids[0] == later.memory_id
    assert ids[1:] == sorted([tied_a.memory_id, tied_b.memory_id])
    assert [hit.final_rank for hit in hits] == [1, 2, 3]


def test_union_ties_sort_ranked_hits_but_keep_timeline_order(monkeypatch):
    """Union sorts equal ranked signals canonically and keeps chronology for time rows."""
    planner = importlib.import_module("haunt.planner")
    timeline = [_hit("timeline-later"), _hit("timeline-earlier")]
    ranked = [
        _hit("rank-z", score=1 / 61, fts_rank=1),
        _hit("rank-a", score=1 / 61, fts_rank=1),
    ]
    monkeypatch.setattr(planner, "run_timeline", lambda *args, **kwargs: timeline)
    monkeypatch.setattr(planner, "run_recall", lambda *args, **kwargs: ranked)

    hits = planner.run_union(object(), Store.__new__(Store), k=8)

    assert [hit.memory_id for hit in hits] == [
        "rank-a",
        "rank-z",
        "timeline-later",
        "timeline-earlier",
    ]
    assert [hit.final_rank for hit in hits] == [1, 2, 3, 4]


def test_dashboard_all_namespace_groups_preserve_local_ranks(tmp_path, monkeypatch):
    """Namespace groups are deterministic without claiming cross-namespace RRF."""
    from haunt import dashboard
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry
    from tests.dashutil import make_dash_client

    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunt-home"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    embed.reset()
    ensure_layout()
    init_registry()
    with Store("alpha") as store:
        store.observe("DASHBOARD-TIE-ALPHA", defer_embedding=True)
    with Store("beta") as store:
        store.observe("DASHBOARD-TIE-BETA", defer_embedding=True)
    alpha_z = _hit("alpha-z", score=1 / 61, fts_rank=1)
    alpha_a = _hit("alpha-a", score=1 / 61, fts_rank=1)
    beta_a = _hit("beta-a", score=9 / 61, fts_rank=1)
    alpha_a.final_rank = 1
    alpha_z.final_rank = 2
    beta_a.final_rank = 1
    by_namespace = {
        "alpha": [
            alpha_z,
            alpha_a,
        ],
        "beta": [beta_a],
    }
    monkeypatch.setattr(
        dashboard,
        "planned_recall",
        lambda query, namespace, **kwargs: by_namespace[namespace],
    )

    response = make_dash_client().get("/api/recall?q=DASHBOARD-TIE")
    assert response.status_code == 200
    data = response.json()
    assert data["ranking_scope"] == "per_namespace"
    assert [group["namespace"] for group in data["namespace_groups"]] == [
        "alpha",
        "beta",
    ]
    hits = data["hits"]
    assert [(hit["namespace"], hit["memory_id"]) for hit in hits] == [
        ("alpha", "alpha-a"),
        ("alpha", "alpha-z"),
        ("beta", "beta-a"),
    ]
    assert [hit["explanation"]["final_rank"] for hit in hits] == [1, 2, 1]
    # beta's larger local score does not move it ahead of the alpha group.
    assert hits[-1]["score"] > hits[0]["score"]
    embed.reset()


# --- reproducibility of tie order across ingests ---------------------------
# memory_id is a fresh uuid4 per write. Ordering exactly-tied rows by it is
# total within a run but re-rolled on the next one, so the same corpus scored
# twice returned the same two documents in either order. One LongMemEval
# question oscillated between gold rank 5 and 6 across ten runs of identical
# trees for exactly this reason: bm25 equal to the last bit, order decided by
# whichever uuid4 happened to sort first.

TIE_GOLD = "zeta alpha"
TIE_DECOY = "zeta bravo"


def _tied_pair_order(store, first: str, second: str) -> list[str]:
    """Ingest two documents that tie exactly on bm25 and return recall order."""
    store.observe(first, defer_embedding=True)
    store.observe(second, defer_embedding=True)
    from haunt.recall import recall

    hits = recall("zeta", namespace=store.name, k=2, use_vectors=False)
    return [hit.content for hit in hits]


def test_tied_documents_order_identically_across_repeated_ingests(
    haunt_env, monkeypatch
):
    """The same corpus must rank the same way every time it is ingested.

    Fresh namespace per trial, so every row gets a brand new uuid4. Under the
    old id-keyed tie-break each trial was an independent coin flip, so the
    trial count is the guard strength: 6 trials would let a regression through
    once in 32 runs. 20 puts that at about two in a million. It can only ever
    fail open, never flake red.
    """
    from haunt.store import Store

    orders = []
    for trial in range(20):
        with Store(f"tie-repeat-{trial}", create=True) as store:
            orders.append(_tied_pair_order(store, TIE_GOLD, TIE_DECOY))
    assert all(order == orders[0] for order in orders), orders


def test_tie_order_follows_content_not_the_random_memory_id(haunt_env, monkeypatch):
    """Force the id order to contradict the content order; content must win.

    Deterministic where the trial-repetition test above is probabilistic: the
    ids are chosen so an id-keyed tie-break returns the opposite list.
    """
    import hashlib
    import itertools

    from haunt import store as store_module
    from haunt.store import Store

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    lower, higher = sorted([TIE_GOLD, TIE_DECOY], key=digest)
    # The document with the HIGHER content hash gets the LOWER memory id, so
    # id-ordering and content-ordering disagree on every field.
    # Monotonic, so the first row written always gets the lower memory id.
    # observe() mints several ids per call, so this must not be a fixed list.
    counter = itertools.count()
    monkeypatch.setattr(
        store_module, "new_id", lambda: f"{next(counter):08d}-0000-0000-0000-0000"
    )

    with Store("tie-forced", create=True) as store:
        order = _tied_pair_order(store, higher, lower)

    assert order == [lower, higher], (
        f"tie broke on the memory id, not on content: {order}"
    )


def test_recall_works_on_a_database_no_writer_has_migrated(haunt_env):
    """content_hash arrives with v10, and recall opens read-only by default.

    ReadOnlyStore never migrates -- that is its documented contract -- so a
    namespace file no writer has opened at this code version still has no
    content_hash column. Naming it unguarded in the candidate queries turned
    every recall against such a file into `no such column: m.content_hash`.
    Store.stats() already guards the same way for the same reason.
    """
    import sqlite3

    from haunt.paths import namespace_db_path
    from haunt.recall import recall
    from haunt.store import Store

    with Store("prev10", create=True) as store:
        store.observe(TIE_GOLD, defer_embedding=True)
        store.observe(TIE_DECOY, defer_embedding=True)

    db = namespace_db_path("prev10")
    raw = sqlite3.connect(str(db))
    raw.execute("DROP INDEX IF EXISTS idx_memories_content_hash")
    raw.execute("ALTER TABLE memories DROP COLUMN content_hash")
    raw.execute("UPDATE meta SET value='9' WHERE key='schema_version'")
    raw.commit()
    columns = {row[1] for row in raw.execute("PRAGMA table_info(memories)")}
    raw.close()
    assert "content_hash" not in columns, "setup failed to remove the column"

    hits = recall("zeta", namespace="prev10", k=5, use_vectors=False)
    assert sorted(hit.content for hit in hits) == sorted([TIE_GOLD, TIE_DECOY])


def test_an_unhashed_row_is_not_promoted_above_hashed_rows(haunt_env):
    """The fallback is the memory id, not the empty string.

    COALESCE(content_hash, '') sorts every unhashed row ahead of every hashed
    one at equal score -- a systematic reordering, not a settled tie, and at a
    k=3 cut it turns a mixed draw into a guaranteed all-unhashed top three.
    Falling back to the id reduces exactly to the previous (rank, id)
    behaviour for rows the key cannot cover.

    Asserted across trials rather than once: with the id fallback three of six
    rows still land in the top three about one run in twenty, so a single
    trial would be flaky. With the '' fallback they do so every time, which is
    exactly the difference being pinned.
    """
    import sqlite3

    from haunt.paths import namespace_db_path
    from haunt.recall import recall
    from haunt.store import Store

    swept = 0
    trials = 6
    for trial in range(trials):
        namespace = f"mixed-{trial}"
        with Store(namespace, create=True) as store:
            for index in range(6):
                store.observe(f"zeta doc{index}", defer_embedding=True)

        raw = sqlite3.connect(str(namespace_db_path(namespace)))
        raw.row_factory = sqlite3.Row
        ids = [r["id"] for r in raw.execute("SELECT id FROM memories ORDER BY rowid")]
        nulled = set(ids[:3])
        raw.executemany(
            "UPDATE memories SET content_hash=NULL WHERE id=?",
            [(i,) for i in nulled],
        )
        raw.commit()
        raw.close()

        hits = recall("zeta", namespace=namespace, k=6, use_vectors=False)
        positions = [i for i, hit in enumerate(hits) if hit.memory_id in nulled]
        swept += positions == [0, 1, 2]

    assert swept < trials, (
        f"unhashed rows swept the top three in all {trials} trials; "
        "the fallback is promoting them rather than settling a tie"
    )
