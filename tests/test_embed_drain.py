"""C4: bounded, out-of-band embedding-queue drain + coverage reporting.

Store.observe() called from a hook always passes defer_embedding=True (see
src/haunt/store.py's C6 capture-policy comment), so a hook-driven write is
captured but never reaches the drain gated on `commit and not
defer_embedding` inside observe() itself. Before this change the only other
caller of process_embedding_jobs was recall() -- so a namespace that is
written to but rarely searched grew an unbounded backlog with no way to
clear it out-of-band, and stats() gave no signal distinguishing "healthy,
nothing queued" from "no vector index exists at all".

This file exercises:
  - Store.drain_embedding_queue(): the bounded loop itself (happy path,
    bound respected + honest reporting, exhausted rows excluded and not
    spun on, backend-unavailable rows excluded and not spun on).
  - bootstrap()'s per-namespace loop now calling that drain and folding the
    result into reembed_report, with no recall() call anywhere in sight.
  - Store.stats()'s new embedding-coverage fields, including the FTS-only
    "no vector index at all" case that must not look like "fully healthy".
  - format_report()'s rendering of drain-only and reembed+drain rows.
"""

from __future__ import annotations

import pytest

from haunt.embed import EmbedState

FAKE_DIM = 4
FAKE_STATE = EmbedState(
    model_id="c4-drain-test-model",
    requested="c4-drain-test-model",
    dim=FAKE_DIM,
    available=True,
    fallback=False,
)
UNAVAILABLE_STATE = EmbedState(
    model_id="off",
    requested="off",
    dim=0,
    available=False,
    fallback=False,
)


def _fake_embed_texts(texts):
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _fake_embed_one(text):
    return [0.1, 0.2, 0.3, 0.4] if (text or "").strip() else None


@pytest.fixture
def drain_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME, FTS-only -- same pattern as
    tests/test_embedding_isolation.py::iso_env. Never downloads or loads a
    real embed model; process_embedding_jobs / drain_embedding_queue are
    exercised entirely through monkeypatched haunt.store.embed_state /
    embed_texts / ensure_vec_table where a test needs a working fake
    backend, or through the real (unavailable) embed.embed_one where a
    test wants genuine FTS-only behavior.
    """
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    monkeypatch.delenv("HAUNT_EMBED_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("HAUNT_EMBED_DRAIN_LIMIT", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield home
    embed.reset()


@pytest.fixture
def real_vec_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME with the *real* sqlite-vec extension loaded (the
    test runner's Python can load it -- see the repo's test instructions),
    but the embed model faked: haunt.store.embed_one / embed_texts /
    embed_state get monkeypatched per test, so no real model download or
    inference ever happens. This lets stats()'s vec_ok()-based fields
    (memories_embedded, vector_index) be exercised against a genuine vec0
    virtual table instead of the FTS-only stand-in used elsewhere in this
    file.
    """
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)
    monkeypatch.delenv("HAUNT_EMBED_MODEL", raising=False)
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield home
    embed.reset()


def _wire_fake_backend(store, monkeypatch, embed_texts_fn=_fake_embed_texts):
    """Point Store at a fake embedding backend without needing sqlite-vec
    (same helper as tests/test_embedding_isolation.py::_wire_fake_backend)."""
    import haunt.store as store_mod

    store.conn.execute(
        "CREATE TABLE IF NOT EXISTS vec_memories(id TEXT PRIMARY KEY, embedding BLOB)"
    )
    store.conn.commit()
    monkeypatch.setattr(store_mod, "embed_state", lambda: FAKE_STATE)
    monkeypatch.setattr(store_mod, "ensure_vec_table", lambda conn, dim, commit=False: True)
    monkeypatch.setattr(store_mod, "embed_texts", embed_texts_fn)


def _wire_fake_backend_real_vec(monkeypatch):
    """Fake the embed model but let sqlite-vec / ensure_vec_table run for
    real. observe()'s synchronous immediate-embed path calls embed_one
    directly (not embed_texts), so that must be patched too -- embed_texts
    / embed_state alone only cover process_embedding_jobs's path for
    deferred rows.
    """
    import haunt.store as store_mod

    monkeypatch.setattr(store_mod, "embed_state", lambda: FAKE_STATE)
    monkeypatch.setattr(store_mod, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(store_mod, "embed_one", _fake_embed_one)


# ---------------------------------------------------------------------------
# HAUNT_EMBED_DRAIN_LIMIT parsing
# ---------------------------------------------------------------------------


def test_embed_drain_limit_env_var_clamped(monkeypatch):
    """HAUNT_EMBED_DRAIN_LIMIT follows the same clamped-int-from-env style
    as HAUNT_EMBED_MAX_ATTEMPTS / _tool_io_cap: parse, default on garbage,
    clamp."""
    from haunt.store import EMBED_DRAIN_LIMIT_DEFAULT, _embed_drain_limit

    monkeypatch.delenv("HAUNT_EMBED_DRAIN_LIMIT", raising=False)
    assert _embed_drain_limit() == EMBED_DRAIN_LIMIT_DEFAULT

    monkeypatch.setenv("HAUNT_EMBED_DRAIN_LIMIT", "0")
    assert _embed_drain_limit() == 1  # clamped up, never disables the bound

    monkeypatch.setenv("HAUNT_EMBED_DRAIN_LIMIT", "-5")
    assert _embed_drain_limit() == 1

    monkeypatch.setenv("HAUNT_EMBED_DRAIN_LIMIT", "not-a-number")
    assert _embed_drain_limit() == EMBED_DRAIN_LIMIT_DEFAULT

    monkeypatch.setenv("HAUNT_EMBED_DRAIN_LIMIT", "50")
    assert _embed_drain_limit() == 50

    monkeypatch.setenv("HAUNT_EMBED_DRAIN_LIMIT", "99999999")
    assert _embed_drain_limit() == 100_000  # clamped down to the ceiling


# ---------------------------------------------------------------------------
# Store.drain_embedding_queue()
# ---------------------------------------------------------------------------


def test_drain_clears_a_deferred_backlog(drain_env, monkeypatch):
    """Basic happy path: several defer_embedding=True writes (what a hook
    produces) all get embedded and removed from embedding_jobs by one
    drain_embedding_queue() call."""
    from haunt.store import Store

    with Store("drain-basic") as store:
        ids = [
            store.observe(f"deferred row {i}", defer_embedding=True).memory_id
            for i in range(5)
        ]
        _wire_fake_backend(store, monkeypatch)

        queued_before = store.conn.execute(
            "SELECT COUNT(*) FROM embedding_jobs"
        ).fetchone()[0]
        assert queued_before == 5

        result = store.drain_embedding_queue()

        assert result["processed"] == 5
        assert result["failed"] == 0
        assert result["remaining"] == 0
        assert result["exhausted"] == 0
        assert result["stop_reason"] == "drained"
        assert result["stopped_early"] is False

        queued_after = store.conn.execute(
            "SELECT COUNT(*) FROM embedding_jobs"
        ).fetchone()[0]
        assert queued_after == 0
        for mid in ids:
            row = store.conn.execute(
                "SELECT embedding FROM memories WHERE id=?", (mid,)
            ).fetchone()
            assert row["embedding"] is not None


def test_drain_respects_bound_and_reports_remaining_honestly(drain_env, monkeypatch):
    """A bound smaller than the backlog must stop the drain early and
    report the honest split -- "drained 3, 7 still queued" -- rather than
    silently claiming completion."""
    from haunt.store import Store

    with Store("drain-bound") as store:
        for i in range(10):
            store.observe(f"deferred row {i}", defer_embedding=True)
        _wire_fake_backend(store, monkeypatch)

        result = store.drain_embedding_queue(max_rows=3)

        assert result["processed"] == 3
        assert result["failed"] == 0
        assert result["remaining"] == 7
        assert result["bound"] == 3
        assert result["stop_reason"] == "bound"
        assert result["stopped_early"] is True

        still_queued = store.conn.execute(
            "SELECT COUNT(*) FROM embedding_jobs"
        ).fetchone()[0]
        assert still_queued == 7


def test_drain_exhausted_row_excluded_and_does_not_spin(drain_env, monkeypatch):
    """A row that hits HAUNT_EMBED_MAX_ATTEMPTS must be reported via
    `exhausted`, not `remaining`, and must not make the drain spin toward
    a large bound re-selecting it forever."""
    monkeypatch.setenv("HAUNT_EMBED_MAX_ATTEMPTS", "1")
    from haunt.store import Store

    def always_fails(texts):
        raise RuntimeError("permanently broken row")

    with Store("drain-exhausted") as store:
        store.observe("ALWAYS-FAILS-CANARY", defer_embedding=True)
        _wire_fake_backend(store, monkeypatch, always_fails)

        result = store.drain_embedding_queue(max_rows=1000)

        assert result["processed"] == 0
        assert result["failed"] == 1
        assert result["exhausted"] == 1
        assert result["remaining"] == 0
        assert result["stop_reason"] == "drained"
        assert result["stopped_early"] is False
        # Proves it did not spin re-selecting the exhausted row up toward
        # max_rows=1000: one batch fails it (attempts -> cap), one more
        # confirms the SELECT now comes back empty.
        assert result["batches"] <= 2


def test_drain_stops_immediately_when_no_backend_available(drain_env, monkeypatch):
    """Distinct failure mode from the attempts cap: no embed backend at
    all. process_embedding_jobs reports queued>0, processed=0, failed=0,
    available=False, and never touches `attempts` in that branch -- so the
    drain must recognize zero progress and stop after one batch rather
    than re-selecting the same rows until it exhausts a huge bound."""
    from haunt.store import Store
    import haunt.store as store_mod

    with Store("drain-blocked") as store:
        for i in range(4):
            store.observe(f"stuck row {i}", defer_embedding=True)

        store.conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_memories(id TEXT PRIMARY KEY, embedding BLOB)"
        )
        store.conn.commit()
        monkeypatch.setattr(store_mod, "embed_state", lambda: UNAVAILABLE_STATE)

        result = store.drain_embedding_queue(max_rows=1000)

        assert result["processed"] == 0
        assert result["failed"] == 0
        assert result["available"] is False
        assert result["remaining"] == 4
        assert result["stop_reason"] == "blocked"
        assert result["stopped_early"] is True
        assert result["batches"] == 1


# ---------------------------------------------------------------------------
# bootstrap() wiring -- drain runs out-of-band, independent of recall()
# ---------------------------------------------------------------------------


def test_bootstrap_drains_backlog_with_no_recall_call(drain_env, monkeypatch):
    """C4's core requirement: a backlog left behind by deferred (hook-style)
    writes is cleared by `haunt bootstrap` alone. This test never imports
    or calls haunt.recall -- the backlog empties purely as a side effect of
    the bootstrap path, proving the drain no longer depends on read/recall
    traffic to make progress."""
    from haunt.store import Store

    with Store("bootstrap-drain") as store:
        ids = [
            store.observe(f"hook-deferred row {i}", defer_embedding=True).memory_id
            for i in range(6)
        ]
        _wire_fake_backend(store, monkeypatch)
        queued = store.conn.execute(
            "SELECT COUNT(*) FROM embedding_jobs"
        ).fetchone()[0]
        assert queued == 6

    from haunt.bootstrap import bootstrap

    report = bootstrap("bootstrap-drain")

    with Store("bootstrap-drain") as store:
        remaining = store.conn.execute(
            "SELECT COUNT(*) FROM embedding_jobs"
        ).fetchone()[0]
        assert remaining == 0
        for mid in ids:
            row = store.conn.execute(
                "SELECT embedding FROM memories WHERE id=?", (mid,)
            ).fetchone()
            assert row["embedding"] is not None

    entries = [r for r in report["reembed"] if r.get("namespace") == "bootstrap-drain"]
    assert len(entries) == 1
    drain = entries[0]["drain"]
    assert drain["processed"] == 6
    assert drain["remaining"] == 0
    assert drain["stopped_early"] is False
    assert entries[0]["auto"] is True


def test_bootstrap_reembed_report_omits_healthy_namespaces(drain_env):
    """A namespace with nothing ever queued must not grow a no-op entry in
    reembed_report on every `haunt bootstrap` call."""
    from haunt.bootstrap import bootstrap

    report = bootstrap("always-healthy")
    names = [r.get("namespace") for r in report["reembed"]]
    assert "always-healthy" not in names


# ---------------------------------------------------------------------------
# Store.stats() embedding-coverage fields
# ---------------------------------------------------------------------------


def test_stats_fully_embedded(real_vec_env, monkeypatch):
    """Every memory embedded: memories_embedded == memories, nothing
    pending or exhausted, a real usable vector index."""
    from haunt.store import Store

    _wire_fake_backend_real_vec(monkeypatch)
    with Store("stats-full") as st:
        if not st.vec_ok():
            pytest.skip("sqlite-vec extension not loadable in this environment")
        for i in range(3):
            st.observe(f"embedded row {i}", role="user")
        stats = st.stats()

    assert stats["memories"] == 3
    assert stats["memories_embedded"] == 3
    assert stats["embedding_pending"] == 0
    assert stats["embedding_exhausted"] == 0
    assert stats["vector_index"] is True
    assert stats["vector_index_version"] is not None


def test_stats_partially_embedded_reports_split(real_vec_env, monkeypatch):
    """A mix of already-embedded and still-queued rows must be split
    correctly across memories_embedded / embedding_pending."""
    from haunt.store import Store

    _wire_fake_backend_real_vec(monkeypatch)
    with Store("stats-partial") as st:
        if not st.vec_ok():
            pytest.skip("sqlite-vec extension not loadable in this environment")
        for i in range(2):
            st.observe(f"embedded row {i}", role="user")
        for i in range(3):
            st.observe(f"deferred row {i}", role="user", defer_embedding=True)
        stats = st.stats()

    assert stats["memories"] == 5
    assert stats["memories_embedded"] == 2
    assert stats["embedding_pending"] == 3
    assert stats["embedding_exhausted"] == 0
    assert stats["vector_index"] is True
    assert (
        stats["embedding_jobs"]
        == stats["embedding_pending"] + stats["embedding_exhausted"]
    )


def test_stats_no_vector_index_reports_honestly(drain_env):
    """Real FTS-only: vec_ok() is False, so there is no usable vector index
    at all. stats() must say so explicitly (vector_index=False) instead of
    reporting memories_embedded=0 the same indistinguishable way a fully
    healthy, fully-embedded namespace would report zero pending."""
    from haunt.store import Store

    with Store("stats-no-index") as st:
        assert st.vec_ok() is False
        st.observe("row one", role="user")
        st.observe("row two", role="user")
        stats = st.stats()

    assert stats["vector_index"] is False
    assert stats["vector_index_version"] is None
    assert stats["memories_embedded"] == 0
    assert stats["memories"] == 2
    # Rows are still fully captured and queued -- real FTS-only mode still
    # enqueues embedding_jobs rows on write (see observe()'s
    # embedding_queued logic) even though they can never drain without a
    # backend. That backlog must stay visible, not silently zeroed.
    assert stats["embedding_pending"] == 2
    assert stats["embedding_exhausted"] == 0


def test_stats_exhausted_rows_counted_separately_from_pending(drain_env, monkeypatch):
    """A permanently-failed row and a merely-not-yet-tried row must land in
    different stats() buckets -- an exhausted row is not "still pending"."""
    monkeypatch.setenv("HAUNT_EMBED_MAX_ATTEMPTS", "1")
    from haunt.store import Store

    def poison_only(texts):
        raise RuntimeError("permanently broken row")

    with Store("stats-mixed") as store:
        store.observe("ALWAYS-FAILS-CANARY", defer_embedding=True)
        _wire_fake_backend(store, monkeypatch, poison_only)
        first = store.process_embedding_jobs(limit=8)
        assert first["failed"] == 1  # now at attempts == max_attempts (1)

        store.observe("freshly queued, never attempted", defer_embedding=True)

        stats = store.stats()

    assert stats["embedding_exhausted"] == 1
    assert stats["embedding_pending"] == 1
    assert stats["embedding_jobs"] == 2


def test_stats_preserves_all_existing_keys(drain_env):
    """C4 adds embedding-coverage keys to stats() but must never remove or
    rename an existing one -- stats() has other consumers (cli.py,
    dashboard.py, mcp_server.py) plus at least one pre-existing test that
    asserts on its shape."""
    from haunt.store import Store

    pre_existing_keys = {
        "namespace",
        "db_path",
        "db_size_bytes",
        "events",
        "memories",
        "sessions",
        "entities",
        "relations",
        "embedding_jobs",
        "corrections",
        "lineage_tombstones",
        "tiers",
        "last_write",
        "last_event_time",
        "wal",
    }
    with Store("stats-shape") as st:
        st.observe("shape check row", role="user", defer_embedding=True)
        stats = st.stats()

    missing = pre_existing_keys - stats.keys()
    assert not missing, f"stats() dropped pre-existing keys: {missing}"


# ---------------------------------------------------------------------------
# bootstrap.format_report() rendering
# ---------------------------------------------------------------------------


def test_format_report_renders_drain_only_and_combined_rows():
    """Pure formatting test: a drain-only reembed_report row must not print
    "updated=None/None", and the human-readable text must make "fully
    drained" and "stopped early" visibly distinct."""
    from haunt.bootstrap import format_report

    report = {
        "haunt_home": "/tmp/home",
        "launcher": "/tmp/home/bin/haunt-mcp",
        "hook_launcher": "/tmp/home/bin/haunt-hook",
        "desktop_icon": None,
        "python": "/usr/bin/python3",
        "sqlite_vec": {"ok": True, "version": "v0"},
        "embed": {
            "requested": "off",
            "loaded": "off",
            "dim": 0,
            "available": False,
            "fallback": False,
            "error": None,
        },
        "default_namespace": "default",
        "default_db": "/tmp/home/namespaces/default.db",
        "reembed": [
            {
                "namespace": "drain-only-ns",
                "auto": True,
                "drain": {
                    "processed": 5,
                    "failed": 0,
                    "batches": 2,
                    "remaining": 0,
                    "exhausted": 0,
                    "available": True,
                    "bound": 500,
                    "stop_reason": "drained",
                    "stopped_early": False,
                },
            },
            {
                "namespace": "bound-hit-ns",
                "auto": True,
                "drain": {
                    "processed": 3,
                    "failed": 0,
                    "batches": 1,
                    "remaining": 7,
                    "exhausted": 1,
                    "available": True,
                    "bound": 3,
                    "stop_reason": "bound",
                    "stopped_early": True,
                },
            },
            {
                "namespace": "reembedded-ns",
                "auto": True,
                "updated": 4,
                "total": 4,
                "dim": 4,
                "model": "fake-model",
                "available": True,
            },
        ],
        "hosts": [],
    }

    text = format_report(report)

    assert "ns=drain-only-ns" in text
    assert "fully drained" in text
    assert "ns=bound-hit-ns" in text
    assert "stopped early (bound)" in text
    assert "remaining=7" in text
    assert "exhausted=1" in text
    assert "ns=reembedded-ns" in text
    assert "updated=4/4" in text
    assert "updated=None" not in text
