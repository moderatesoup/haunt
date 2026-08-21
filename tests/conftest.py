from __future__ import annotations

import os
from pathlib import Path

import pytest

MODEL_CACHE = Path("/workspace/lore/.model-cache")


@pytest.fixture
def lore_env(tmp_path, monkeypatch):
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.delenv("LORE_HOME", raising=False)
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    monkeypatch.delenv("LORE_NAMESPACE", raising=False)
    monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)
    monkeypatch.delenv("LORE_FTS_ONLY", raising=False)
    if MODEL_CACHE.exists():
        monkeypatch.setenv("HAUNT_MODEL_CACHE", str(MODEL_CACHE))
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    from haunt import embed
    from haunt.bootstrap import bootstrap
    from haunt.paths import ensure_layout

    embed.reset()
    ensure_layout()
    bootstrap("default")
    yield home
    embed.reset()
