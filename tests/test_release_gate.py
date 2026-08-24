"""Release-gate holes: honesty, modes, MCP pin, bind, XSS, GET/DELETE-create, limits, redaction.

Each test is a mutation check — revert the corresponding fix and this file fails.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME, FTS-only — no model download, no host bind."""
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry, register_namespace

    embed.reset()
    ensure_layout()
    init_registry()
    register_namespace("default")
    yield home
    embed.reset()


# ---------------------------------------------------------------------------
# 1. SECURITY.md honesty
# ---------------------------------------------------------------------------


def test_security_md_file_per_namespace_is_storage_not_auth():
    """File-per-namespace is storage isolation, not authorization / a kernel."""
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    low = text.lower()
    assert "storage isolation" in low
    assert "not authorization" in low
    assert "not a security kernel" in low
    assert "mcp" in low
    assert "every namespace" in low or "any namespace" in low


# ---------------------------------------------------------------------------
# 2. Filesystem modes
# ---------------------------------------------------------------------------


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_fresh_haunt_home_is_private(tmp_path, monkeypatch):
    """A newly created HAUNT_HOME must be 0700; new db/WAL/SHM 0600."""
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)

    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import Store, init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    with Store("privtest") as st:
        st.observe("private mode canary", role="user")

    assert home.is_dir()
    assert _mode(home) == 0o700, f"HAUNT_HOME mode {_mode(home):o}, want 0700"
    assert _mode(home / "namespaces") == 0o700
    db = home / "namespaces" / "privtest.db"
    assert db.is_file()
    assert _mode(db) == 0o600, f"db mode {_mode(db):o}, want 0600"
    for extra in (Path(str(db) + "-wal"), Path(str(db) + "-shm")):
        if extra.exists():
            assert _mode(extra) == 0o600, f"{extra.name} mode {_mode(extra):o}, want 0600"
    embed.reset()


def test_bootstrap_and_doctor_repair_world_readable_modes(tmp_path, monkeypatch):
    """bootstrap + doctor must tighten existing 0755/0644 under HAUNT_HOME only."""
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)

    from haunt import embed
    from haunt.paths import ensure_layout, repair_private_modes
    from haunt.store import Store, init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    with Store("repairme") as st:
        st.observe("repair canary", role="user")

    home.chmod(0o755)
    (home / "namespaces").chmod(0o755)
    db = home / "namespaces" / "repairme.db"
    db.chmod(0o644)
    wal = Path(str(db) + "-wal")
    if wal.exists():
        wal.chmod(0o644)

    user_home = Path.home()
    home_mode_before = user_home.stat().st_mode

    repair_private_modes()

    assert _mode(home) == 0o700
    assert _mode(home / "namespaces") == 0o700
    assert _mode(db) == 0o600
    if wal.exists():
        assert _mode(wal) == 0o600
    assert user_home.stat().st_mode == home_mode_before, (
        "repair_private_modes must not chmod the user's whole home"
    )
    embed.reset()


def test_bootstrap_and_doctor_invoke_repair():
    from haunt.bootstrap import bootstrap
    from haunt.cli import doctor_cmd

    assert "repair_private_modes" in inspect.getsource(bootstrap)
    assert "repair_private_modes" in inspect.getsource(doctor_cmd)


def test_repair_private_modes_skips_user_home(tmp_path, monkeypatch):
    """Even if HAUNT_HOME were confused, never chmod Path.home()."""
    from haunt.paths import repair_private_modes

    user_home = Path.home()
    before = user_home.stat().st_mode
    # Point HAUNT_HOME at a temp dir — repair must still refuse Path.home().
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    (tmp_path / "haunthome").mkdir()
    repair_private_modes()
    assert user_home.stat().st_mode == before


# ---------------------------------------------------------------------------
# 3. MCP pin
# ---------------------------------------------------------------------------


def test_pyproject_requires_mcp_v2():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "mcp>=2" in text
    assert "mcp>=1.0.0" not in text
    assert "mcp>=1." not in text


def test_mcp_server_import_rejects_mcp_1x(monkeypatch):
    """Import assertion must refuse MCP 1.x so it cannot silently install."""
    import haunt.mcp_server as ms

    assert hasattr(ms, "_require_mcp_v2"), (
        "mcp_server must expose _require_mcp_v2 so MCP 1.x cannot slip through"
    )
    monkeypatch.setattr(ms, "_mcp_package_version", lambda: "1.13.0")
    with pytest.raises((RuntimeError, ImportError), match=r"mcp>=2|MCP 1"):
        ms._require_mcp_v2()


def test_mcp_server_imports_mcpserver():
    from mcp.server import MCPServer
    from haunt.mcp_server import server

    assert isinstance(server, MCPServer)


# ---------------------------------------------------------------------------
# 4. Dashboard bind
# ---------------------------------------------------------------------------


def test_run_dashboard_defaults_to_loopback():
    from haunt.dashboard import run_dashboard

    sig = inspect.signature(run_dashboard)
    assert sig.parameters["host"].default == "127.0.0.1"


def test_dashboard_rejects_non_loopback_without_allow_remote():
    from haunt.dashboard import check_dashboard_bind

    with pytest.raises((ValueError, SystemExit), match=r"allow-remote|loopback|127\.0\.0\.1"):
        check_dashboard_bind("0.0.0.0", allow_remote=False)


def test_dashboard_allows_non_loopback_with_allow_remote(capsys):
    from haunt.dashboard import check_dashboard_bind

    check_dashboard_bind("0.0.0.0", allow_remote=True)
    err = capsys.readouterr().err
    assert "WARNING" in err or "warning" in err.lower() or "remote" in err.lower()


def test_cli_dash_rejects_0_0_0_0_without_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    from typer.testing import CliRunner
    from haunt.cli import app

    result = CliRunner().invoke(app, ["dash", "--host", "0.0.0.0", "--no-open"])
    assert result.exit_code != 0
    blob = (result.stdout or "") + (result.stderr or "")
    assert "allow-remote" in blob.lower() or "loopback" in blob.lower()


# ---------------------------------------------------------------------------
# 5. Dashboard XSS
# ---------------------------------------------------------------------------


XSS_ORIGIN = '"><img src=x onerror=alert(1)><script>alert(1)</script>'


def test_dashboard_js_does_not_interpolate_origin_raw():
    """Stored origin/tool_name/text must go through esc() / textContent."""
    from haunt.dashboard import HTML

    assert "${r.origin||\"\"}" not in HTML
    assert "${h.origin||''}" not in HTML
    assert "${m.origin||\"\"}" not in HTML
    assert '["origin",d.origin]' not in HTML
    assert "esc(r.origin" in HTML or "esc(r.origin||" in HTML
    assert "esc(h.origin" in HTML or "esc(h.origin||" in HTML
    assert "esc(d.origin" in HTML or "esc(d.origin||" in HTML
    assert "esc(d.tool_name" in HTML or "esc(d.tool_name||" in HTML


def test_stored_xss_origin_does_not_appear_raw_in_html(gate_env):
    from starlette.testclient import TestClient
    from haunt.dashboard import app
    from haunt.store import observe

    observe("xss canary body", namespace="default", role="user", origin=XSS_ORIGIN)
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert XSS_ORIGIN not in page.text
    assert "<script>alert(1)</script>" not in page.text
    assert '"><img src=x onerror=alert(1)>' not in page.text

    data = client.get("/api/namespace/default").json()
    origins = [e.get("origin") or "" for e in data.get("events") or []]
    assert any(XSS_ORIGIN in o for o in origins), "fixture must actually store the payload"


# ---------------------------------------------------------------------------
# 6. Dashboard GET must not create namespaces
# ---------------------------------------------------------------------------


def test_get_unknown_namespace_is_404_and_does_not_create(gate_env):
    from starlette.testclient import TestClient
    from haunt.dashboard import app
    from haunt.store import namespace_exists

    client = TestClient(app)
    mystery = "never-created-ns-xyz"
    assert not namespace_exists(mystery)
    r = client.get(f"/api/namespace/{mystery}")
    assert r.status_code == 404
    assert not namespace_exists(mystery)
    db = gate_env / "namespaces" / f"{mystery}.db"
    assert not db.exists(), "GET must not create the namespace db"


def test_get_unknown_namespace_uses_create_false():
    import haunt.dashboard as dash

    src = inspect.getsource(dash.api_namespace)
    helper = inspect.getsource(dash._missing_namespace)
    assert "create=False" in src
    assert "404" in helper
    assert "namespace_exists" in helper


# ---------------------------------------------------------------------------
# 6b. Dashboard DELETE / contradict must not create namespaces (#56)
# ---------------------------------------------------------------------------


def test_delete_unknown_namespace_is_404_and_does_not_create(gate_env):
    """#56: DELETE on a typo'd ns must not create the DB then 404 memory-not-found."""
    from starlette.testclient import TestClient
    from haunt.dashboard import app
    from haunt.store import namespace_exists

    client = TestClient(app)
    mystery = "typo-ns-delete-never-created"
    assert not namespace_exists(mystery)
    r = client.request("DELETE", f"/api/namespace/{mystery}/memory/does-not-exist")
    assert r.status_code == 404
    err = (r.json().get("error") or "")
    assert "unknown namespace" in err, f"expected unknown namespace, got {r.json()!r}"
    assert "memory" not in err, f"must not imply the namespace exists: {r.json()!r}"
    assert not namespace_exists(mystery)
    db = gate_env / "namespaces" / f"{mystery}.db"
    assert not db.exists(), "DELETE must not create the namespace db"


def test_contradict_unknown_namespace_is_404_and_does_not_create(gate_env):
    """#56: POST contradict on a typo'd ns must not create the DB."""
    from starlette.testclient import TestClient
    from haunt.dashboard import app
    from haunt.store import namespace_exists

    client = TestClient(app)
    mystery = "typo-ns-contradict-never-created"
    assert not namespace_exists(mystery)
    r = client.post(
        f"/api/namespace/{mystery}/memory/does-not-exist/contradict",
        json={},
    )
    assert r.status_code == 404
    err = (r.json().get("error") or "")
    assert "unknown namespace" in err, f"expected unknown namespace, got {r.json()!r}"
    assert "memory" not in err, f"must not imply the namespace exists: {r.json()!r}"
    assert not namespace_exists(mystery)
    db = gate_env / "namespaces" / f"{mystery}.db"
    assert not db.exists(), "POST contradict must not create the namespace db"


def test_delete_existing_ns_missing_memory_is_404_memory_not_found(gate_env):
    """#56: existing ns + missing memory still 404s memory-not-found, no extra ns."""
    from starlette.testclient import TestClient
    from haunt.dashboard import app
    from haunt.store import list_namespaces, namespace_exists

    client = TestClient(app)
    assert namespace_exists("default")
    before = {ns["name"] for ns in list_namespaces()}
    r = client.request("DELETE", "/api/namespace/default/memory/does-not-exist")
    assert r.status_code == 404
    err = (r.json().get("error") or "")
    assert "memory" in err and "not found" in err, f"expected memory not found, got {r.json()!r}"
    assert "unknown namespace" not in err
    after = {ns["name"] for ns in list_namespaces()}
    assert after == before


def test_contradict_existing_ns_missing_memory_is_404_memory_not_found(gate_env):
    """#56: existing ns + missing memory still 404s memory-not-found, no extra ns."""
    from starlette.testclient import TestClient
    from haunt.dashboard import app
    from haunt.store import list_namespaces, namespace_exists

    client = TestClient(app)
    assert namespace_exists("default")
    before = {ns["name"] for ns in list_namespaces()}
    r = client.post(
        "/api/namespace/default/memory/does-not-exist/contradict",
        json={},
    )
    assert r.status_code == 404
    err = (r.json().get("error") or "")
    assert "memory" in err and "not found" in err, f"expected memory not found, got {r.json()!r}"
    assert "unknown namespace" not in err
    after = {ns["name"] for ns in list_namespaces()}
    assert after == before


def test_delete_and_contradict_use_create_false():
    """#56 mutation: restoring Store(name) default create=True fails this test."""
    import haunt.dashboard as dash

    delete_src = inspect.getsource(dash.api_memory_delete)
    contradict_src = inspect.getsource(dash.api_contradict)
    helper = inspect.getsource(dash._missing_namespace)
    assert "Store(name, create=False)" in delete_src
    assert "Store(name, create=False)" in contradict_src
    assert "_missing_namespace" in delete_src
    assert "_missing_namespace" in contradict_src
    assert "namespace_exists" in helper
    assert "404" in helper
    # Explicit create=True (or dropping the kwarg) must not sneak back in.
    assert "create=True" not in delete_src
    assert "create=True" not in contradict_src


# ---------------------------------------------------------------------------
# 7. Negative / huge limits
# ---------------------------------------------------------------------------


def test_clamp_limit_bounds():
    from haunt.util import clamp_limit

    assert clamp_limit(-1) == 1
    assert clamp_limit(0) == 1
    assert clamp_limit(1) == 1
    assert clamp_limit(100) == 100
    assert clamp_limit(10_000) == 100
    assert clamp_limit("nope", default=8) == 8


def test_negative_timeline_limit_is_not_unbounded(gate_env):
    """SQLite LIMIT -1 means no limit. Negative must clamp, not dump the table."""
    from starlette.testclient import TestClient
    from haunt.dashboard import app
    from haunt.store import Store, observe

    for i in range(12):
        observe(f"limit-canary-{i}", namespace="default", role="user")

    client = TestClient(app)
    r = client.get("/api/namespace/default/timeline?limit=-1")
    assert r.status_code == 200
    events = r.json()["events"]
    with Store("default") as st:
        total = st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert total >= 12
    assert len(events) == 1, (
        f"limit=-1 must clamp to 1, not return {len(events)} of {total} (unbounded SQLite LIMIT)"
    )


def test_huge_k_is_clamped_on_dashboard_and_mcp(gate_env):
    from starlette.testclient import TestClient
    from haunt.dashboard import app
    from haunt.mcp_server import memory_recall, memory_timeline
    from haunt.store import observe

    observe("huge-limit-canary UNIQUE-CLAMP-99", namespace="default", role="user")
    client = TestClient(app)
    r = client.get("/api/namespace/default/recall?q=UNIQUE-CLAMP-99&k=99999")
    assert r.status_code == 200
    assert len(r.json()["hits"]) <= 100

    rec = json.loads(memory_recall(query="UNIQUE-CLAMP-99", namespace="default", k=-5))
    assert rec.get("ok") is not False
    assert len(rec["hits"]) <= 100

    tl = json.loads(memory_timeline(namespace="default", limit=-1))
    assert tl.get("ok") is not False
    assert len(tl["events"]) == 1


def test_mcp_and_dashboard_call_clamp_limit():
    from haunt import dashboard, mcp_server

    assert "clamp_limit" in inspect.getsource(mcp_server.memory_recall)
    assert "clamp_limit" in inspect.getsource(mcp_server.memory_timeline)
    assert "clamp_limit" in inspect.getsource(dashboard.api_recall)
    assert "clamp_limit" in inspect.getsource(dashboard.api_timeline)
    assert "clamp_limit" in inspect.getsource(dashboard.api_recall_all)


# ---------------------------------------------------------------------------
# 8. Hook redaction — Authorization: Bearer header form
# ---------------------------------------------------------------------------


def test_redact_authorization_bearer_header_form():
    """Ordinary `Authorization: Bearer …` must redact, not only JSON-ish tokens."""
    from haunt.cursor_hook import _redact_secrets

    token = "ordinheader.token.value_XYZ987654321"
    raw = f"Authorization: Bearer {token}\nOK 200"
    out = _redact_secrets(raw)
    assert token not in out
    assert "REDACTED" in out
    assert "OK 200" in out


def test_hook_stores_redacted_authorization_bearer(tmp_path, monkeypatch, capsys):
    home = tmp_path / "haunthome"
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "hooktest")

    from haunt import embed
    from haunt.cursor_hook import main
    from haunt.paths import ensure_layout
    from haunt.store import Store, init_registry
    import io
    import sys

    embed.reset()
    ensure_layout()
    init_registry()

    token = "ordinheader.token.value_XYZ987654321"
    payload = {
        "hook_event_name": "afterShellExecution",
        "command": f'curl -H "Authorization: Bearer {token}" https://example.test/v1',
        "output": "ok",
        "conversation_id": "conv-bearer-header",
        "workspace_roots": [str(project)],
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    capsys.readouterr()
    with Store("hooktest") as st:
        rows = st.events(session_id="conv-bearer-header")
        assert rows
        stored = (rows[0]["tool_input"] or "") + (rows[0]["tool_output"] or "")
        assert token not in stored
        assert "REDACTED" in stored
    embed.reset()
