"""L32: never call the embedding model while a write transaction is open.

Embedding a batch of tail spans takes minutes. `process_embedding_jobs` used
to do it after its head-vector writes had already opened a write transaction,
so the transaction stayed open for the whole model call. On a namespace with a
live concurrent writer every commit then failed with
`sqlite3.OperationalError: locking protocol` (SQLITE_PROTOCOL, not the
`SQLITE_BUSY` that `PRAGMA busy_timeout` would have retried).

The symptom was badly misleading. Four quieter namespaces drained without a
single retry while a 1.08 GB one failed 400 consecutive batches, so it read as
corruption or as something specific to that file. It was neither: `quick_check`
returned `ok`, a plain `sqlite3` connection committed on the same file at the
same moment, and a *small* write through `Store` committed fine too. Only the
long transaction failed, and only where something else was writing.

These tests assert the invariant directly rather than the symptom, because the
symptom needs a concurrent writer and a large corpus to appear at all.
"""

from __future__ import annotations

import os

import pytest

requires_model = pytest.mark.skipif(
    bool(os.environ.get("HAUNT_FTS_ONLY"))
    or os.environ.get("HAUNT_EMBED_MODEL") == "off",
    reason="span embedding requires a model",
)

LONG = " ".join(f"token{i:05d}" for i in range(1500))


def test_prepare_memory_spans_takes_no_connection():
    """The slow half cannot hold a lock it has no handle to.

    A signature check, deliberately: it is the structural guarantee. If a
    connection is ever threaded back into the embedding half, this fails
    before anyone has to reproduce a locking error on a live namespace.
    """
    import inspect

    from haunt.store import prepare_memory_spans

    params = set(inspect.signature(prepare_memory_spans).parameters)
    assert "conn" not in params, (
        "prepare_memory_spans embeds; giving it a connection reopens L32"
    )


@requires_model
def test_the_drain_never_embeds_inside_a_write_transaction(haunt_env, monkeypatch):
    """The load-bearing assertion: no open transaction across a model call."""
    from haunt import embed, store as store_mod
    from haunt.store import Store

    if not embed.available():
        pytest.skip("no embedding backend")

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()

    with Store("l32-drain") as store:
        for i in range(4):
            store.observe(f"{LONG} row {i}", role="user", tier="episodic",
                          defer_embedding=True)
        store.conn.commit()

        seen: list[bool] = []
        real = store_mod.embed_texts

        def watched(texts):
            seen.append(bool(store.conn.in_transaction))
            return real(texts)

        monkeypatch.setattr(store_mod, "embed_texts", watched)
        report = store.process_embedding_jobs(limit=8)

    assert seen, "the model was never called; this proves nothing"
    assert not any(seen), (
        f"embed_texts ran inside an open write transaction on "
        f"{sum(seen)} of {len(seen)} calls -- that is L32"
    )
    assert report["processed"] == 4
    assert report["spans"]["spans"] > 0


@requires_model
def test_head_vectors_are_durable_before_span_work_begins(haunt_env, monkeypatch):
    """A span failure must not cost the head vectors that already succeeded.

    Committing the head work first is what makes the model call safe, and it
    also makes the durability boundary honest: a tail is an additional index
    over content whose vector is already correct.
    """
    from haunt import embed, store as store_mod
    from haunt.store import Store

    if not embed.available():
        pytest.skip("no embedding backend")

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()

    with Store("l32-durable") as store:
        written = [
            store.observe(f"{LONG} row {i}", role="user", tier="episodic",
                          defer_embedding=True).memory_id
            for i in range(3)
        ]
        store.conn.commit()

        # Head embedding succeeds; the span pass blows up.
        real = store_mod.embed_texts
        calls = {"n": 0}

        def flaky(texts):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("span embedding exploded")
            return real(texts)

        monkeypatch.setattr(store_mod, "embed_texts", flaky)
        report = store.process_embedding_jobs(limit=8)

        assert report["processed"] == 3
        assert report["spans"]["spans"] == 0
        embedded = store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL AND id IN "
            f"({','.join('?' * len(written))})",
            written,
        ).fetchone()[0]

    assert embedded == 3, (
        "head vectors were rolled back by a span failure; they are durable "
        "work that had already succeeded"
    )


@requires_model
def test_reembed_also_keeps_the_model_call_out_of_the_transaction(
    haunt_env, monkeypatch
):
    """reembed() rebuilds spans per chunk and had the same shape."""
    from haunt import embed, store as store_mod
    from haunt.store import Store

    if not embed.available():
        pytest.skip("no embedding backend")

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()

    with Store("l32-reembed") as store:
        for i in range(3):
            store.observe(f"{LONG} r{i}", role="user", tier="episodic")
        store.conn.commit()

        seen: list[bool] = []
        real = store_mod.embed_texts

        def watched(texts):
            seen.append(bool(store.conn.in_transaction))
            return real(texts)

        monkeypatch.setattr(store_mod, "embed_texts", watched)
        store.reembed()

    assert seen
    assert not any(seen), (
        f"reembed embedded inside an open transaction on {sum(seen)} of "
        f"{len(seen)} calls"
    )
