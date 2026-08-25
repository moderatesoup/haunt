"""Tests for the hardening changes: bootstrap fail-loud, health db_path,
default embed model, --version, and sqlite-vec split-brain prevention."""

from __future__ import annotations

import json
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
    """If sqlite-vec cannot load, bootstrap must exit 1 — not silently continue.

    Quality path: FTS-only unset. HAUNT_FTS_ONLY=1 is the opt-out (#64).
    """
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)
    monkeypatch.delenv("HAUNT_EMBED_MODEL", raising=False)

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

    home = tmp_path / "haunthome"
    assert not (home / "registry.db").exists(), (
        "bootstrap must not write registry.db when sqlite-vec fails"
    )
    assert not (home / "namespaces" / "default.db").exists(), (
        "bootstrap must not write a namespace store when sqlite-vec fails"
    )

    embed.reset()


def test_health_includes_db_path(tmp_path, monkeypatch):
    """haunt health and memory_health must include the absolute db_path."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "healthtest")

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
    monkeypatch.setenv("HAUNT_NAMESPACE", "mcp-health-test")

    from haunt import embed
    from haunt.mcp_server import memory_health
    from haunt.paths import ensure_layout
    from haunt.store import Store, init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    with Store("mcp-health-test") as st:
        st.observe("health path canary", origin="test")

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


def test_memory_procedure_rejects_invalid_action(tmp_path, monkeypatch):
    """MCP memory_procedure must return ok=False for unknown action strings."""
    import json

    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "proc-test")

    from haunt import embed
    from haunt.mcp_server import memory_procedure
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()

    for bad_action in ("delete", "update", "remove", "supersede", ""):
        raw = memory_procedure(action=bad_action, name="x", namespace="proc-test")
        data = json.loads(raw)
        assert data["ok"] is False, f"action={bad_action!r} should fail, got ok=True"
        assert "unknown action" in data["error"].lower() or "must be" in data["error"].lower()

    embed.reset()


# ------------------------------------------------------------------
# Split-brain prevention: vec load failure vs. health reporting
# ------------------------------------------------------------------


def test_vec_load_failure_is_fatal_without_fts_only(tmp_path, monkeypatch):
    """When HAUNT_FTS_ONLY is not set and sqlite-vec fails to load,
    _connect must raise — not silently continue with a vec-less connection."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)
    monkeypatch.delenv("HAUNT_EMBED_MODEL", raising=False)

    from haunt import embed
    from haunt.paths import ensure_layout, namespace_db_path
    from haunt.store import _connect

    embed.reset()
    ensure_layout()

    db_path = namespace_db_path("splitbrain-test")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with patch("sqlite_vec.load", side_effect=OSError("mocked vec load failure")):
        with pytest.raises(RuntimeError, match="sqlite-vec failed to load"):
            _connect(db_path)

    embed.reset()


def test_fts_only_skips_vec_load_entirely(tmp_path, monkeypatch):
    """With HAUNT_FTS_ONLY=1, _connect must not attempt to load sqlite-vec
    and the resulting connection must report vec_ok()=False."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")

    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import Store, _vec_loaded, init_registry

    embed.reset()
    ensure_layout()
    init_registry()

    load_called = False
    _original_load = __import__("sqlite_vec").load

    def spy_load(conn):
        nonlocal load_called
        load_called = True
        return _original_load(conn)

    with patch("sqlite_vec.load", side_effect=spy_load):
        with Store("fts-skip-test") as st:
            assert not st.vec_ok(), "vec_ok() must be False in FTS-only mode"
            assert st.vec_version() is None
            assert not _vec_loaded(st.conn)

    assert not load_called, "sqlite_vec.load should not be called in FTS-only mode"
    embed.reset()


def test_no_split_brain_health_matches_store(tmp_path, monkeypatch):
    """Health endpoints must reflect the actual Store connection's vec status.
    When FTS-only is set, health must say sqlite_vec.ok=False — not probe
    a separate in-memory connection that could disagree."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "split-brain-test")

    from haunt import embed
    from haunt.mcp_server import memory_health
    from haunt.paths import ensure_layout
    from haunt.store import Store, init_registry

    embed.reset()
    ensure_layout()
    init_registry()

    with Store("split-brain-test") as st:
        store_vec_ok = st.vec_ok()

    raw = memory_health(namespace="split-brain-test")
    data = json.loads(raw)

    assert data["sqlite_vec"]["ok"] is store_vec_ok, (
        f"health reported sqlite_vec.ok={data['sqlite_vec']['ok']} "
        f"but Store.vec_ok()={store_vec_ok} — split brain!"
    )
    assert data["sqlite_vec"]["ok"] is False, (
        "In FTS-only mode, sqlite_vec.ok must be False"
    )

    embed.reset()


def test_split_brain_impossible_with_sabotaged_load(tmp_path, monkeypatch):
    """Even if sqlite-vec can be probed in-memory (probe_sqlite_vec would
    return ok=True), a Store whose _connect skipped vec must not
    claim vec is OK in any health report.

    This reproduces the exact split-brain scenario from the issue."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "sabotage-test")

    from haunt.bootstrap import probe_sqlite_vec
    from haunt import embed
    from haunt.mcp_server import memory_health
    from haunt.paths import ensure_layout
    from haunt.store import Store, init_registry

    embed.reset()
    ensure_layout()
    init_registry()

    probe_result = probe_sqlite_vec()
    probe_says_ok = probe_result.get("ok", False)

    with Store("sabotage-test") as st:
        store_says_ok = st.vec_ok()

    raw = memory_health(namespace="sabotage-test")
    health = json.loads(raw)
    health_says_ok = health["sqlite_vec"]["ok"]

    assert health_says_ok == store_says_ok, (
        f"SPLIT BRAIN: health says ok={health_says_ok}, "
        f"store says ok={store_says_ok}, probe says ok={probe_says_ok}"
    )

    if probe_says_ok and not store_says_ok:
        assert not health_says_ok, (
            "probe_sqlite_vec says ok=True but Store has no vec; "
            "health must agree with Store, not with probe"
        )

    embed.reset()
