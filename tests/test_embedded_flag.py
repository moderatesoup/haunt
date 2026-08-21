"""Tests that ObserveResult.embedded is True only when the row actually
lands in vec_memories — never when the insert is skipped or raises."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest


def test_embedded_false_when_vec_not_loaded(tmp_path, monkeypatch):
    """If _vec_loaded returns False (e.g. FTS-only mode), embedded must be
    False even when embed_one produces a vector."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("LORE_HOME", raising=False)
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)

    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import Store, _vec_loaded, init_registry

    embed.reset()
    ensure_layout()
    init_registry()

    fake_vec = [0.1] * 384

    with Store("vec-skip-test") as st:
        assert not _vec_loaded(st.conn), "precondition: vec must not be loaded"
        with patch("haunt.store.embed_one", return_value=fake_vec):
            r = st.observe("hello world", role="user")
        assert r.embedded is False, (
            "embedded must be False when vec_memories insert is skipped"
        )

    embed.reset()


def test_embedded_false_when_vec_insert_raises(tmp_path, monkeypatch):
    """If the INSERT INTO vec_memories raises sqlite3.Error,
    embedded must be False.  We patch _vec_loaded→True so the code
    enters the INSERT branch, but ensure_vec_table→False so the table
    does not exist, producing a real sqlite3.OperationalError."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("LORE_HOME", raising=False)
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)

    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import Store, init_registry

    embed.reset()
    ensure_layout()
    init_registry()

    fake_vec = [0.1] * 384

    with Store("vec-raise-test") as st:
        with (
            patch("haunt.store.embed_one", return_value=fake_vec),
            patch("haunt.store._vec_loaded", return_value=True),
            patch("haunt.store.ensure_vec_table", return_value=False),
        ):
            r = st.observe("hello world", role="user")
        assert r.embedded is False, (
            "embedded must be False when vec_memories insert raises"
        )

    embed.reset()


def test_embedded_true_when_vec_insert_succeeds(lore_env):
    """When vec is loaded and insert succeeds, embedded must be True."""
    from haunt.embed import embed_one as real_embed_one
    from haunt.store import Store, _vec_loaded

    vec = real_embed_one("test embedding content")
    if vec is None:
        pytest.skip("embedding model not available")

    with Store("default") as st:
        if not _vec_loaded(st.conn):
            pytest.skip("sqlite-vec not loaded")
        r = st.observe("test embedding content", role="user")
    assert r.embedded is True, (
        "embedded must be True when vec_memories insert succeeds"
    )


def test_fts_only_embedded_always_false(tmp_path, monkeypatch):
    """The legitimate FTS-only path: embed_one returns None, so
    embedded is False. Sanity check that FTS-only is honest."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("LORE_HOME", raising=False)
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)

    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import Store, init_registry

    embed.reset()
    ensure_layout()
    init_registry()

    with Store("fts-honest-test") as st:
        r = st.observe("some text", role="user")
    assert r.embedded is False, (
        "FTS-only mode must report embedded=False"
    )

    embed.reset()


def test_embedded_false_not_vacuous(tmp_path, monkeypatch):
    """The False result must not come from a vacuous empty check — the
    embedding blob must actually have been computed (embed_one returned
    a vector) but the vec insert was blocked."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("LORE_HOME", raising=False)
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)

    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import Store, _vec_loaded, init_registry

    embed.reset()
    ensure_layout()
    init_registry()

    fake_vec = [0.1] * 384
    embed_one_called = False
    original_embed_one = None

    def tracking_embed_one(text):
        nonlocal embed_one_called
        embed_one_called = True
        return fake_vec

    with Store("vacuous-test") as st:
        assert not _vec_loaded(st.conn)
        with patch("haunt.store.embed_one", side_effect=tracking_embed_one):
            r = st.observe("non-empty text", role="user")

        assert embed_one_called, "embed_one must have been called"
        row = st.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert row["embedding"] is not None, (
            "embedding blob should be stored in memories even without vec table"
        )
        assert r.embedded is False, (
            "embedded must be False: embed_one ran but vec insert was skipped"
        )

    embed.reset()
