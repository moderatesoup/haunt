from __future__ import annotations

import os
from pathlib import Path

import pytest

MODEL_CACHE = Path("/workspace/lore/.model-cache")


@pytest.fixture
def lore_env(tmp_path, monkeypatch):
    home = tmp_path / "lorehome"
    monkeypatch.setenv("LORE_HOME", str(home))
    monkeypatch.delenv("LORE_NAMESPACE", raising=False)
    monkeypatch.delenv("LORE_FTS_ONLY", raising=False)
    if MODEL_CACHE.exists():
        monkeypatch.setenv("LORE_MODEL_CACHE", str(MODEL_CACHE))
    monkeypatch.setenv("LORE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    from lore import embed
    from lore.bootstrap import bootstrap
    from lore.paths import ensure_layout

    embed.reset()
    ensure_layout()
    bootstrap("default")
    yield home
    embed.reset()
