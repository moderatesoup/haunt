from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest

from tests.dashutil import TEST_DASH_TOKEN


@lru_cache(maxsize=1)
def _host_model_cache() -> Path | None:
    """The model cache this host already has, resolved from the real environment.

    Must be read before `haunt_env` repoints HAUNT_HOME, because `models_dir()`
    follows HAUNT_HOME: unpinned, every test that embeds re-downloads the model
    into a tmp directory pytest then deletes. An explicit HAUNT_MODEL_CACHE is
    taken as given; the derived default (`~/.haunt/models` unless the host set
    HAUNT_HOME) only counts when it already holds something.
    """
    from haunt.paths import models_dir

    cache = models_dir()
    if os.environ.get("HAUNT_MODEL_CACHE"):
        return cache
    if cache.is_dir() and any(cache.iterdir()):
        return cache
    return None


@pytest.fixture(autouse=True)
def _dashboard_security_defaults():
    """Every test starts with a configured launch token and loopback bind host."""
    from haunt.dashboard import configure_dashboard_security, reset_dashboard_security

    configure_dashboard_security(token=TEST_DASH_TOKEN, bind_host="127.0.0.1")
    yield
    reset_dashboard_security()


@pytest.fixture(autouse=True)
def isolate_host_homes(tmp_path, monkeypatch):
    """Never write Cursor/Claude configs into the real (or cloud-agent) home.

    Tests that need a specific layout set CURSOR_HOME / CLAUDE_CONFIG_DIR
    themselves; those override this fixture.
    """
    if not os.environ.get("CURSOR_HOME"):
        monkeypatch.setenv("CURSOR_HOME", str(tmp_path / "cursor-home"))
    if not os.environ.get("CLAUDE_CONFIG_DIR"):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))


@pytest.fixture
def haunt_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME reusing the host model cache.

    HAUNT_FTS_ONLY and HAUNT_EMBED_MODEL are left alone when the caller set
    them: CI runs the suite with HAUNT_FTS_ONLY=1, and clearing it here would
    make that run exercise the embedding path it claims to skip. With neither
    set the run is still correct, only slower.
    """
    model_cache = _host_model_cache()
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    if model_cache is not None:
        monkeypatch.setenv("HAUNT_MODEL_CACHE", str(model_cache))
    if not os.environ.get("HAUNT_EMBED_MODEL"):
        # Smallest model that still exercises the vector path. The bge-m3
        # default is 2.1 GB.
        monkeypatch.setenv("HAUNT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    from haunt import embed
    from haunt.bootstrap import bootstrap
    from haunt.paths import ensure_layout

    embed.reset()
    ensure_layout()
    bootstrap("default")
    yield home
    embed.reset()
