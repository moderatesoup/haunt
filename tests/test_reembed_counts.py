"""Tests that Store.reembed reports updated= only for rows that actually
land in vec_memories — never for skipped or swallowed inserts.

If ``updated += 1`` is moved back outside the successful INSERT, these
fail (vec_ok False and INSERT-raise both report updated > 0).
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from haunt.embed import EmbedState


FAKE_DIM = 384
FAKE_VEC = [0.1] * FAKE_DIM
FAKE_STATE = EmbedState(
    model_id="test-reembed-model",
    requested="test-reembed-model",
    dim=FAKE_DIM,
    available=True,
    fallback=False,
)


def _fts_only_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)

    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    return embed


def _vec_count(conn: sqlite3.Connection) -> int:
    try:
        return int(conn.execute("SELECT COUNT(*) FROM vec_memories").fetchone()[0])
    except sqlite3.Error:
        return 0


def _fake_embed_texts(texts):
    return [list(FAKE_VEC) for _ in texts]


def _plain_vec_table(conn, dim):
    """Stand-in for ensure_vec_table that works without sqlite-vec loaded."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vec_memories (id TEXT PRIMARY KEY, embedding BLOB)"
    )
    conn.commit()
    return True


def test_reembed_updated_zero_when_vec_ok_false(tmp_path, monkeypatch):
    """vec_ok patched False: blobs may still write, but updated must be 0
    and vec_memories must stay empty. This is the published #17 falsifier.
    """
    embed = _fts_only_home(tmp_path, monkeypatch)
    from haunt.store import Store

    with Store("reembed-vec-skip") as st:
        st.observe("memory one", role="user")
        st.observe("memory two", role="user")
        n_mem = int(st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        assert n_mem == 2

        with (
            patch("haunt.store.embed_state", return_value=FAKE_STATE),
            patch("haunt.store.embed_texts", side_effect=_fake_embed_texts),
            patch.object(st, "vec_ok", return_value=False),
        ):
            result = st.reembed()

        vec_n = _vec_count(st.conn)
        blobs = int(
            st.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        )
        assert blobs == 2, (
            "embedding blobs should still be written when vec insert is skipped"
        )
        assert vec_n == 0
        assert result["updated"] == 0, (
            "updated must not count memories.embedding writes when vec insert is skipped"
        )
        assert result["updated"] == vec_n
        assert result["total"] == 2

    embed.reset()


def test_reembed_updated_zero_when_insert_raises(tmp_path, monkeypatch):
    """vec_ok True but table missing: INSERT raises, updated must stay 0."""
    embed = _fts_only_home(tmp_path, monkeypatch)
    from haunt.store import Store

    with Store("reembed-vec-raise") as st:
        st.observe("memory one", role="user")
        st.observe("memory two", role="user")

        with (
            patch("haunt.store.embed_state", return_value=FAKE_STATE),
            patch("haunt.store.embed_texts", side_effect=_fake_embed_texts),
            patch.object(st, "vec_ok", return_value=True),
            patch("haunt.store.ensure_vec_table", return_value=False),
        ):
            result = st.reembed()

        vec_n = _vec_count(st.conn)
        assert vec_n == 0
        assert result["updated"] == 0, (
            "updated must be 0 when vec_memories INSERT raises"
        )
        assert result["updated"] == vec_n
        assert result["total"] == 2

    embed.reset()


def test_reembed_updated_matches_vec_rows(tmp_path, monkeypatch):
    """Happy path (FTS-only + stand-in vec table): updated equals vec row count."""
    embed = _fts_only_home(tmp_path, monkeypatch)
    from haunt.store import Store

    with Store("reembed-happy") as st:
        st.observe("memory one", role="user")
        st.observe("memory two", role="user")

        with (
            patch("haunt.store.embed_state", return_value=FAKE_STATE),
            patch("haunt.store.embed_texts", side_effect=_fake_embed_texts),
            patch.object(st, "vec_ok", return_value=True),
            patch("haunt.store.ensure_vec_table", side_effect=_plain_vec_table),
        ):
            result = st.reembed()

        vec_n = _vec_count(st.conn)
        assert vec_n == 2
        assert result["updated"] == vec_n
        assert result["total"] == 2
        assert result["available"] is True

    embed.reset()


def test_reembed_fts_only_available_false(tmp_path, monkeypatch):
    """Legitimate FTS-only path: no embed model, updated=0, available=False."""
    embed = _fts_only_home(tmp_path, monkeypatch)
    from haunt.store import Store

    with Store("reembed-fts-honest") as st:
        st.observe("some text", role="user")
        result = st.reembed()

    assert result["updated"] == 0
    assert result["available"] is False
    assert result["total"] == 1

    embed.reset()


def test_reembed_updated_matches_vec_rows_live(haunt_env):
    """Live path when sqlite-vec and the embed model are actually loaded."""
    from haunt.embed import embed_one as real_embed_one
    from haunt.store import Store, _vec_loaded

    if real_embed_one("test reembed content") is None:
        pytest.skip("embedding model not available")

    with Store("default") as st:
        if not _vec_loaded(st.conn):
            pytest.skip("sqlite-vec not loaded")
        st.observe("alpha memory for reembed", role="user")
        st.observe("beta memory for reembed", role="user")
        result = st.reembed()
        vec_n = int(st.conn.execute("SELECT COUNT(*) FROM vec_memories").fetchone()[0])
        mem_n = int(st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        assert vec_n >= 2
        assert result["updated"] == vec_n
        assert result["total"] == mem_n
        assert result["available"] is True


def test_reembed_all_namespaces_isolates_a_sibling_registry_error(
    tmp_path, monkeypatch
):
    """One unopenable namespace must not end the walk over the others.

    UnknownNamespaceError, NamespaceCollisionError and NamespaceMigrationError
    are what a namespace deregistered, remapped, or left mid-migration since
    the registry was listed raises; none of them is a sqlite3.Error or OSError.
    """
    _fts_only_home(tmp_path, monkeypatch)
    from haunt import store as store_module

    with store_module.Store("reembed-broken") as st:
        st.observe("row in the namespace that will not open")
    with store_module.Store("reembed-intact") as st:
        st.observe("row that must still be reembedded")

    real_store = store_module.Store

    def flaky_store(name, **kwargs):
        if name == "reembed-broken":
            raise store_module.UnknownNamespaceError(name)
        return real_store(name, **kwargs)

    monkeypatch.setattr(store_module, "Store", flaky_store)
    reports = {r["namespace"]: r for r in store_module.reembed_all_namespaces()}

    assert "unknown namespace" in reports["reembed-broken"]["error"]
    assert "error" not in reports["reembed-intact"], reports["reembed-intact"]
