"""C5: per-row failure isolation in Store.process_embedding_jobs.

Two real defects fixed here:

1. Batch-level failure — embed_texts() ran once for the whole queued batch;
   if it raised, every row in the batch got attempts+1 and none were
   processed. One malformed row poisoned the whole batch.
2. No attempts cap — the SELECT had no WHERE attempts < N, so a
   permanently-failing row at the head of the queue (ordered by queued_at)
   was re-selected forever, blocking every job behind it.
"""

from __future__ import annotations

import pytest

from haunt.embed import EmbedState

FAKE_DIM = 4
FAKE_STATE = EmbedState(
    model_id="iso-test-model",
    requested="iso-test-model",
    dim=FAKE_DIM,
    available=True,
    fallback=False,
)


@pytest.fixture
def iso_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME, FTS-only — never download a real embed model.

    process_embedding_jobs is exercised entirely through monkeypatched
    haunt.store.embed_state / embed_texts / ensure_vec_table, same pattern
    as tests/test_issue_51_52.py::test_persistent_process_drains_embedding_queue.
    """
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "iso-test")
    monkeypatch.delenv("HAUNT_EMBED_MAX_ATTEMPTS", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry, register_namespace

    embed.reset()
    ensure_layout()
    init_registry()
    register_namespace("iso-test")
    yield home
    embed.reset()


def _wire_fake_backend(store, monkeypatch, embed_texts_fn):
    """Point Store at a fake embedding backend without needing sqlite-vec."""
    import haunt.store as store_mod

    store.conn.execute(
        "CREATE TABLE IF NOT EXISTS vec_memories(id TEXT PRIMARY KEY, embedding BLOB)"
    )
    store.conn.commit()
    monkeypatch.setattr(store_mod, "embed_state", lambda: FAKE_STATE)
    monkeypatch.setattr(store_mod, "ensure_vec_table", lambda conn, dim, commit=False: True)
    monkeypatch.setattr(store_mod, "embed_texts", embed_texts_fn)


def test_poison_row_does_not_block_batch_siblings(iso_env, monkeypatch):
    """A single row whose content crashes the batch embed call must not
    prevent its batch-mates from embedding successfully."""
    from haunt.store import Store

    with Store("iso-test") as store:
        good1 = store.observe("GOOD-ROW-ONE", defer_embedding=True)
        poison = store.observe("POISON-ROW-CONTENT", defer_embedding=True)
        good2 = store.observe("GOOD-ROW-TWO", defer_embedding=True)

        def fake_embed_texts(texts):
            if any("POISON-ROW-CONTENT" in t for t in texts):
                raise RuntimeError("boom: cannot embed poison content")
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        _wire_fake_backend(store, monkeypatch, fake_embed_texts)

        report = store.process_embedding_jobs(limit=8)

        # The fast batch call (all 3 rows) raises because the poison row is
        # in it. Isolation means the fallback still embeds the two good
        # rows individually.
        assert report["queued"] == 3
        assert report["processed"] == 2
        assert report["failed"] == 1

        def embedding_of(memory_id: str):
            row = store.conn.execute(
                "SELECT embedding FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            return row["embedding"]

        assert embedding_of(good1.memory_id) is not None
        assert embedding_of(good2.memory_id) is not None
        assert embedding_of(poison.memory_id) is None

        # Good rows are fully drained from the queue...
        remaining = {
            r["memory_id"]
            for r in store.conn.execute(
                "SELECT memory_id FROM embedding_jobs"
            ).fetchall()
        }
        assert good1.memory_id not in remaining
        assert good2.memory_id not in remaining

        # ...the poison row stays queued, with its own attempts/last_error,
        # not the good rows'.
        job = store.conn.execute(
            "SELECT attempts, last_error FROM embedding_jobs WHERE memory_id=?",
            (poison.memory_id,),
        ).fetchone()
        assert job is not None
        assert job["attempts"] == 1
        assert "boom" in (job["last_error"] or "")


def test_poison_row_isolated_even_when_batch_returns_wrong_count(iso_env, monkeypatch):
    """A backend that returns a mismatched vector count (rather than
    raising) must also fall back to per-row isolation, not silently
    misalign vectors to the wrong rows via zip().

    The bad row is queued *first* on purpose: naive `zip(queued, vectors)`
    positional pairing (the pre-fix approach) would pair the bad row with
    the one vector that did come back — reporting the bad row as embedded
    and the good row as the failure. Isolation must get this right by
    content, not by position.
    """
    from haunt.store import Store

    with Store("iso-test") as store:
        bad = store.observe("SHORT-COUNT-BAD-ROW", defer_embedding=True)
        good = store.observe("SHORT-COUNT-GOOD-ROW", defer_embedding=True)

        def fake_embed_texts(texts):
            if len(texts) > 1:
                # Simulate a backend that drops one row instead of raising.
                return [[0.1, 0.2, 0.3, 0.4]]
            if any("BAD-ROW" in t for t in texts):
                raise RuntimeError("bad row, individually, also fails")
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        _wire_fake_backend(store, monkeypatch, fake_embed_texts)

        report = store.process_embedding_jobs(limit=8)
        assert report["processed"] == 1
        assert report["failed"] == 1

        good_row = store.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (good.memory_id,)
        ).fetchone()
        bad_row = store.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (bad.memory_id,)
        ).fetchone()
        assert good_row["embedding"] is not None
        assert bad_row["embedding"] is None


def test_row_exceeding_max_attempts_stops_being_retried_and_is_reported(
    iso_env, monkeypatch
):
    """Once a row's attempts reach HAUNT_EMBED_MAX_ATTEMPTS, it must be
    excluded from selection (so it stops blocking the queue) and counted
    via the `exhausted` key (so it stays discoverable)."""
    from haunt.store import Store

    monkeypatch.setenv("HAUNT_EMBED_MAX_ATTEMPTS", "2")

    def always_fails(texts):
        raise RuntimeError("permanently broken row")

    with Store("iso-test") as store:
        doomed = store.observe("ALWAYS-FAILS-CANARY", defer_embedding=True)
        _wire_fake_backend(store, monkeypatch, always_fails)

        report1 = store.process_embedding_jobs(limit=8)
        assert report1["queued"] == 1
        assert report1["failed"] == 1
        assert report1["exhausted"] == 0
        attempts1 = store.conn.execute(
            "SELECT attempts FROM embedding_jobs WHERE memory_id=?",
            (doomed.memory_id,),
        ).fetchone()["attempts"]
        assert attempts1 == 1

        report2 = store.process_embedding_jobs(limit=8)
        assert report2["queued"] == 1
        assert report2["failed"] == 1
        attempts2 = store.conn.execute(
            "SELECT attempts FROM embedding_jobs WHERE memory_id=?",
            (doomed.memory_id,),
        ).fetchone()["attempts"]
        assert attempts2 == 2

        # attempts is now == max_attempts (2): excluded from the next SELECT.
        report3 = store.process_embedding_jobs(limit=8)
        assert report3["queued"] == 0
        assert report3["exhausted"] == 1

        # The job row itself is untouched (still attempts=2) -- it was never
        # selected again, so it was never retried a third time.
        final = store.conn.execute(
            "SELECT attempts FROM embedding_jobs WHERE memory_id=?",
            (doomed.memory_id,),
        ).fetchone()
        assert final["attempts"] == 2


def test_other_rows_keep_draining_around_an_exhausted_row(iso_env, monkeypatch):
    """An exhausted row must not block unrelated rows queued after it, even
    though it sits at the head of the queue (queued_at ASC)."""
    from haunt.store import Store

    monkeypatch.setenv("HAUNT_EMBED_MAX_ATTEMPTS", "1")

    with Store("iso-test") as store:
        doomed = store.observe("HEAD-OF-QUEUE-POISON", defer_embedding=True)

        def always_fails(texts):
            raise RuntimeError("stuck forever")

        _wire_fake_backend(store, monkeypatch, always_fails)
        report1 = store.process_embedding_jobs(limit=8)
        assert report1["failed"] == 1
        # attempts is now 1 == max_attempts(1): excluded going forward.

        # A healthy row queued *after* the poison row must still embed.
        newcomer = store.observe("QUEUED-AFTER-POISON-ROW", defer_embedding=True)

        def succeeds(texts):
            return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

        import haunt.store as store_mod

        monkeypatch.setattr(store_mod, "embed_texts", succeeds)

        report2 = store.process_embedding_jobs(limit=8)
        assert report2["queued"] == 1
        assert report2["processed"] == 1
        assert report2["exhausted"] == 1

        newcomer_row = store.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (newcomer.memory_id,)
        ).fetchone()
        assert newcomer_row["embedding"] is not None
        doomed_row = store.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (doomed.memory_id,)
        ).fetchone()
        assert doomed_row["embedding"] is None


def test_embed_max_attempts_env_var_clamped(monkeypatch):
    """HAUNT_EMBED_MAX_ATTEMPTS follows the same clamped-int-from-env style
    as _tool_io_cap / embed._max_len: parse, default on garbage, clamp."""
    from haunt.store import EMBED_MAX_ATTEMPTS_DEFAULT, _embed_max_attempts

    monkeypatch.delenv("HAUNT_EMBED_MAX_ATTEMPTS", raising=False)
    assert _embed_max_attempts() == EMBED_MAX_ATTEMPTS_DEFAULT

    monkeypatch.setenv("HAUNT_EMBED_MAX_ATTEMPTS", "0")
    assert _embed_max_attempts() == 1  # clamped up, never disables the cap

    monkeypatch.setenv("HAUNT_EMBED_MAX_ATTEMPTS", "-5")
    assert _embed_max_attempts() == 1

    monkeypatch.setenv("HAUNT_EMBED_MAX_ATTEMPTS", "not-a-number")
    assert _embed_max_attempts() == EMBED_MAX_ATTEMPTS_DEFAULT

    monkeypatch.setenv("HAUNT_EMBED_MAX_ATTEMPTS", "3")
    assert _embed_max_attempts() == 3

    monkeypatch.setenv("HAUNT_EMBED_MAX_ATTEMPTS", "99999")
    assert _embed_max_attempts() == 1000  # clamped down to the ceiling
