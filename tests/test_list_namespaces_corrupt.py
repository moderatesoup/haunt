"""list_namespaces must not report zeros for a corrupt or unreadable DB.

The old path defaulted events/memories/sessions/entities to 0, then
swallowed sqlite3.Error on Store open. A garbage file looked empty.

These tests fail if that swallow-to-zero returns.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from haunt.paths import namespace_db_path
from haunt.store import Store, list_namespaces, observe


def _corrupt_ns_db(name: str) -> Path:
    db = namespace_db_path(name)
    assert db.exists(), f"expected namespace db at {db}"
    db.write_text("GARBAGE")
    return db


def test_list_namespaces_corrupt_db_is_not_empty_zeros(lore_env):
    """Published #18 falsifier: overwrite the DB with GARBAGE; listing
    must surface an error and must not report events=0, memories=0.
    """
    observe("corrupt-ns canary phrase CORRUPT-NS-18", namespace="corrupt-ns", role="user")
    before = {r["name"]: r for r in list_namespaces()}
    assert before["corrupt-ns"]["events"] >= 1
    assert before["corrupt-ns"]["memories"] >= 1
    assert not before["corrupt-ns"].get("error")

    _corrupt_ns_db("corrupt-ns")
    rows = {r["name"]: r for r in list_namespaces()}
    row = rows["corrupt-ns"]

    assert row.get("error"), (
        "corrupt DB must surface error; got no error key "
        f"(row={row!r})"
    )
    assert row.get("events") != 0, (
        "corrupt DB must not look empty: events was 0 "
        f"(row={row!r})"
    )
    assert row.get("memories") != 0, (
        "corrupt DB must not look empty: memories was 0 "
        f"(row={row!r})"
    )
    assert row.get("sessions") != 0
    assert row.get("entities") != 0


def test_list_namespaces_empty_healthy_still_zeros(lore_env):
    """A real empty namespace still reports zero counts and no error."""
    with Store("empty-ns") as st:
        stats = st.stats()
        assert stats["events"] == 0
        assert stats["memories"] == 0

    rows = {r["name"]: r for r in list_namespaces()}
    row = rows["empty-ns"]
    assert not row.get("error")
    assert row["events"] == 0
    assert row["memories"] == 0
    assert row["sessions"] == 0
    assert row["entities"] == 0
    assert "events" in row
    assert "memories" in row


def test_list_namespaces_healthy_keeps_count_keys(lore_env):
    observe("healthy namespace phrase HEALTHY-NS-18", namespace="healthy-ns", role="user")
    rows = {r["name"]: r for r in list_namespaces()}
    row = rows["healthy-ns"]
    assert not row.get("error")
    assert row["events"] >= 1
    assert row["memories"] >= 1
    assert isinstance(row["sessions"], int)
    assert isinstance(row["entities"], int)
    assert isinstance(row["db_size_bytes"], int)
    assert row["db_size_bytes"] > 0


def test_list_namespaces_missing_db_is_not_empty_zeros(lore_env):
    """Registered namespace whose file is gone is unreadable, not empty."""
    with Store("missing-ns"):
        pass
    db = namespace_db_path("missing-ns")
    db.unlink()
    rows = {r["name"]: r for r in list_namespaces()}
    row = rows["missing-ns"]
    assert row.get("error"), f"missing DB must surface error (row={row!r})"
    assert row.get("events") != 0
    assert row.get("memories") != 0


def test_cli_namespaces_prints_error_not_zeros(lore_env):
    observe("cli corrupt canary CLI-CORRUPT-18", namespace="cli-corrupt", role="user")
    _corrupt_ns_db("cli-corrupt")
    env = os.environ.copy()
    env["HAUNT_HOME"] = str(lore_env)
    env["HAUNT_FTS_ONLY"] = "1"
    env["HAUNT_EMBED_MODEL"] = "off"
    p = subprocess.run(
        [sys.executable, "-m", "haunt", "namespaces"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    out = p.stdout
    assert "cli-corrupt" in out
    assert "error:" in out.lower()
    # Old swallow printed the name with zero counts in the numeric columns.
    assert "cli-corrupt" in out and "error" in out
    for line in out.splitlines():
        if "cli-corrupt" in line:
            assert "error" in line.lower()
            assert not line.split()[1:2] == ["0"]


def test_mcp_memory_namespaces_surfaces_error(lore_env):
    observe("mcp corrupt canary MCP-CORRUPT-18", namespace="mcp-corrupt", role="user")
    _corrupt_ns_db("mcp-corrupt")
    from haunt.mcp_server import memory_namespaces

    data = json.loads(memory_namespaces())
    row = next(r for r in data["namespaces"] if r["name"] == "mcp-corrupt")
    assert row.get("error"), f"MCP must surface error (row={row!r})"
    assert row.get("events") != 0
    assert row.get("memories") != 0


def test_dash_api_namespaces_surfaces_error(lore_env):
    observe("dash corrupt canary DASH-CORRUPT-18", namespace="dash-corrupt", role="user")
    _corrupt_ns_db("dash-corrupt")
    from starlette.testclient import TestClient
    from haunt.dashboard import app

    client = TestClient(app)
    r = client.get("/api/namespaces")
    assert r.status_code == 200
    row = next(n for n in r.json()["namespaces"] if n["name"] == "dash-corrupt")
    assert row.get("error"), f"dash API must surface error (row={row!r})"
    assert row.get("events") != 0
    assert row.get("memories") != 0
