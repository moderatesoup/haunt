"""Tests for the hardening changes: bootstrap fail-loud, health db_path,
default embed model, --version."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_version_prints_and_exits_zero():
    """haunt --version prints a semver string and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "haunt", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    version = result.stdout.strip()
    parts = version.split(".")
    assert len(parts) >= 2, f"expected semver, got {version!r}"
    assert all(p.isdigit() for p in parts), f"non-numeric semver parts in {version!r}"


def test_bootstrap_exits_1_when_vec_probe_fails(tmp_path, monkeypatch):
    """If sqlite-vec cannot load, bootstrap must exit 1 — not silently continue."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.delenv("LORE_HOME", raising=False)

    from haunt import embed
    from haunt.bootstrap import BootstrapError, bootstrap

    embed.reset()

    with patch(
        "haunt.bootstrap.probe_sqlite_vec",
        return_value={"ok": False, "error": "mocked: extension load failed"},
    ):
        with pytest.raises(BootstrapError) as exc_info:
            bootstrap()
        assert exc_info.value.code == 1
        assert "sqlite-vec" in exc_info.value.message.lower()

    embed.reset()


def test_health_includes_db_path(tmp_path, monkeypatch):
    """haunt health and memory_health must include the absolute db_path."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "healthtest")
    monkeypatch.delenv("LORE_HOME", raising=False)
    monkeypatch.delenv("LORE_NAMESPACE", raising=False)

    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import Store, init_registry

    embed.reset()
    ensure_layout()
    init_registry()

    with Store("healthtest") as st:
        stats = st.stats()

    assert "db_path" in stats
    db_path = stats["db_path"]
    assert os.path.isabs(db_path), f"db_path should be absolute, got {db_path!r}"
    assert "healthtest" in db_path

    assert "namespace" in stats
    assert stats["namespace"] == "healthtest"

    embed.reset()


def test_health_mcp_includes_db_path_and_namespace(tmp_path, monkeypatch):
    """memory_health MCP tool must have namespace and db_path at the top level."""
    import json

    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("LORE_HOME", raising=False)
    monkeypatch.delenv("LORE_NAMESPACE", raising=False)
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)

    from haunt import embed
    from haunt.mcp_server import memory_health
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()

    raw = memory_health(namespace="mcp-health-test")
    data = json.loads(raw)

    assert "namespace" in data
    assert data["namespace"] == "mcp-health-test"
    assert "db_path" in data
    assert os.path.isabs(data["db_path"]), f"expected absolute path, got {data['db_path']!r}"
    assert "mcp-health-test" in data["db_path"]

    embed.reset()


def test_default_embed_model_is_bge_m3():
    """The product default embed model must be BAAI/bge-m3."""
    from haunt.embed import DEFAULT_REQUESTED

    assert DEFAULT_REQUESTED == "BAAI/bge-m3"


def test_bootstrap_report_includes_desktop_icon(tmp_path, monkeypatch):
    """bootstrap() report must include a desktop_icon path when HOME is a temp dir."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.delenv("LORE_HOME", raising=False)

    from haunt import embed
    from haunt.bootstrap import bootstrap

    embed.reset()
    report = bootstrap()
    assert "desktop_icon" in report, "bootstrap report must include 'desktop_icon' key"
    embed.reset()


def test_fts_only_env_disables_embeddings(tmp_path, monkeypatch):
    """HAUNT_FTS_ONLY=1 must disable embeddings entirely."""
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")

    from haunt import embed

    embed.reset()

    st = embed.state()
    assert not st.available
    assert st.dim == 0
    assert st.model_id == "off"

    embed.reset()
