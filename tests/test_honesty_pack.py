"""Honesty pack (#61): XSS leftovers, install wipe, atomic contradict,
timeline k underfill, wrapper HAUNT_HOME, public-size clamps.

Each test is a mutation check — revert the corresponding fix and this file fails.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

NOW = datetime(2026, 8, 22, 15, 30, 0, tzinfo=timezone.utc)

XSS_SESSION = '"><img src=x onerror=alert(1)><script>alert(1)</script>'
XSS_HOME = '"><img src=x onerror=alert(2)>'


@pytest.fixture
def honesty_env(tmp_path, monkeypatch):
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


@pytest.fixture
def host_env(tmp_path, monkeypatch):
    haunt_home = tmp_path / "haunthome"
    cursor_home = tmp_path / "cursor"
    claude_dir = tmp_path / "claude-config"
    monkeypatch.setenv("HAUNT_HOME", str(haunt_home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("CURSOR_HOME", str(cursor_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    monkeypatch.delenv("CURSOR_HOOKS_JSON", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield {
        "haunt_home": haunt_home,
        "cursor_home": cursor_home,
        "claude_dir": claude_dir,
        "hook_cmd": str(haunt_home / "bin" / "haunt-hook"),
        "mcp_cmd": str(haunt_home / "bin" / "haunt-mcp"),
    }
    embed.reset()


def _js_esc(s) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _open_detail_src() -> str:
    from haunt.dashboard import HTML

    m = re.search(r"async function openDetail\(.*?\nfunction ", HTML, re.S)
    assert m, "openDetail missing from dashboard HTML"
    return m.group(0)


def _render_open_detail_vals(detail: dict, fields: list[str]) -> str:
    """Replay openDetail's row interpolations against planted values.

    If a field expression does not call esc() and the rows.map template does
    not wrap ${v} in esc(), the raw payload lands in .val — the mutation.
    """
    src = _open_detail_src()
    map_escapes = bool(re.search(r"rows\.map\(\(\[l,v\]\)=>`[^`]*\$\{esc\(v", src))
    parts: list[str] = []
    for field in fields:
        m = re.search(rf'\["{re.escape(field)}",([^\]]+)\]', src)
        assert m, f"{field} is not interpolated in openDetail"
        expr = m.group(1)
        raw = detail.get(field)
        text = "" if raw is None else str(raw)
        if map_escapes or re.search(r"\besc\s*\(", expr):
            val = _js_esc(text)
        else:
            val = text
        parts.append(f'<span class="val">{val}</span>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# 1. Stored XSS leftover (dashboard detail)
# ---------------------------------------------------------------------------


def test_open_detail_escapes_session_id_and_haunt_home(tmp_path, monkeypatch):
    """Plant XSS in session_id and haunt_home; rendered .val must be escaped.

    Mutation: leave session_id as d.session_id (no esc) and this fails.
    """
    home = tmp_path / f"home{XSS_HOME}"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import Store, init_registry, register_namespace

    embed.reset()
    ensure_layout()
    init_registry()
    register_namespace("default")
    with Store("default") as st:
        r = st.observe(
            "xss detail canary",
            role="user",
            session_id=XSS_SESSION,
        )
        detail = st.get_memory(r.memory_id)
    embed.reset()

    assert detail is not None
    assert detail["session_id"] == XSS_SESSION
    assert XSS_HOME in str(detail["haunt_home"])

    src = _open_detail_src()
    assert '["session_id",d.session_id]' not in src
    assert "esc(d.session_id" in src
    assert "esc(d.haunt_home" in src

    rendered = _render_open_detail_vals(
        detail, ["session_id", "haunt_home", "memory_id", "namespace", "db_path"]
    )
    assert XSS_SESSION not in rendered
    assert XSS_HOME not in rendered
    assert _js_esc(XSS_SESSION) in rendered
    assert _js_esc(XSS_HOME) in rendered


# ---------------------------------------------------------------------------
# 2. Install wipes malformed editor JSON
# ---------------------------------------------------------------------------


def test_malformed_cursor_hooks_json_is_not_replaced(host_env):
    """A broken hooks.json must stay broken. Install must not write version/hooks-only."""
    from haunt.hosts import HostConfigError
    from haunt.hosts.cursor import install as cursor_install

    env = host_env
    env["cursor_home"].mkdir(parents=True)
    path = env["cursor_home"] / "hooks.json"
    broken = "{not json, leftover user hooks"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(HostConfigError, match="malformed"):
        cursor_install(str(env["haunt_home"]), env["hook_cmd"], env["mcp_cmd"])

    text = path.read_text(encoding="utf-8")
    assert text == broken
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_malformed_claude_settings_json_is_not_replaced(host_env):
    from haunt.hosts import HostConfigError
    from haunt.hosts.claude import install as claude_install

    env = host_env
    env["claude_dir"].mkdir(parents=True)
    path = env["claude_dir"] / "settings.json"
    broken = "{not json, leftover claude settings"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(HostConfigError, match="malformed"):
        claude_install(str(env["haunt_home"]), env["hook_cmd"], env["mcp_cmd"])

    text = path.read_text(encoding="utf-8")
    assert text == broken
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_cursor_merge_hooks_does_not_blank_on_json_decode_error():
    """Source mutation: except JSONDecodeError: existing = {version, hooks} fails this."""
    from haunt.hosts import cursor

    src = inspect.getsource(cursor._merge_hooks_json)
    assert "JSONDecodeError" not in src
    assert "read_json_object" in src or "HostConfigError" in src


# ---------------------------------------------------------------------------
# 3. contradict is non-atomic
# ---------------------------------------------------------------------------


def test_contradict_raising_replacement_leaves_original_current(honesty_env, monkeypatch):
    from haunt.store import Store

    with Store("default") as st:
        r = st.observe("original fact stays current", role="system", tier="semantic")

        def boom(*_a, **_k):
            raise RuntimeError("replacement exploded")

        monkeypatch.setattr(Store, "observe", boom)
        with pytest.raises(RuntimeError, match="replacement exploded"):
            st.contradict(r.memory_id, replacement="should not land")
        row = st.conn.execute(
            "SELECT valid_to, content FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert row["valid_to"] is None, "raising replacement must not supersede the original"
        assert "original fact stays current" in row["content"]


def test_contradict_already_superseded_is_ok_false_and_keeps_valid_to(honesty_env):
    from haunt.store import Store

    with Store("default") as st:
        r = st.observe("first fact", role="system", tier="semantic")
        first = st.contradict(r.memory_id, replacement="second fact")
        assert first["ok"] is True
        vt = first["valid_to"]
        stored = st.conn.execute(
            "SELECT valid_to FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()["valid_to"]
        assert stored == vt

        again = st.contradict(r.memory_id, replacement="third fact")
        assert again["ok"] is False
        assert "superseded" in (again.get("error") or "")
        stored2 = st.conn.execute(
            "SELECT valid_to FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()["valid_to"]
        assert stored2 == vt, "re-contradict must not move valid_to forward"
        n = st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert n == 2, "already-superseded contradict must not store another replacement"


def test_dashboard_contradict_already_superseded_is_not_500(honesty_env):
    from starlette.testclient import TestClient
    from haunt.dashboard import app
    from haunt.store import Store

    with Store("default") as st:
        r = st.observe("dash contradict once", role="system", tier="semantic")
        mid = r.memory_id
        vt = st.contradict(mid)["valid_to"]

    client = TestClient(app)
    resp = client.post(
        f"/api/namespace/default/memory/{mid}/contradict",
        json={"replacement": "must not apply"},
    )
    assert resp.status_code != 500
    data = resp.json()
    assert data["ok"] is False
    with Store("default") as st:
        stored = st.conn.execute(
            "SELECT valid_to FROM memories WHERE id=?", (mid,)
        ).fetchone()["valid_to"]
        assert stored == vt


# ---------------------------------------------------------------------------
# 4. Temporal timeline k under-fills
# ---------------------------------------------------------------------------


def test_timeline_k_fills_current_after_recent_superseded(honesty_env):
    """Two recent superseded + two older current and k=2 must return the two current."""
    from haunt.planner import execute
    from haunt.store import Store
    from haunt.temporal import TemporalQuery

    with Store("default") as st:
        older1 = st.observe(
            "current older ALPHA",
            event_time="2026-01-01T12:00:00+00:00",
        )
        older2 = st.observe(
            "current older BETA",
            event_time="2026-01-02T12:00:00+00:00",
        )
        recent1 = st.observe(
            "superseded recent ONE",
            event_time="2026-08-20T12:00:00+00:00",
        )
        recent2 = st.observe(
            "superseded recent TWO",
            event_time="2026-08-21T12:00:00+00:00",
        )
        st.contradict(recent1.memory_id)
        st.contradict(recent2.memory_id)

        tq = TemporalQuery(
            temporal=True,
            cleaned_query="",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            clock="event_time",
            granularity="year",
            certainty="exact",
            confidence=1.0,
        )
        hits = execute(tq, st, strategy="timeline", k=2)
        ids = [h.memory_id for h in hits]
        assert len(hits) == 2
        assert older1.memory_id in ids
        assert older2.memory_id in ids
        assert recent1.memory_id not in ids
        assert recent2.memory_id not in ids


def test_nontemporal_planned_recall_still_matches_bare_recall(honesty_env):
    from haunt.planner import planned_recall
    from haunt.recall import recall
    from haunt.store import Store

    with Store("default") as st:
        st.observe("Azure architecture decision record", origin="test")
        st.observe("unrelated grocery list", origin="test")
        q = "Azure architecture"
        bare = recall(q, store=st, k=8)
        planned = planned_recall(q, now=NOW, store=st, k=8)
        assert [h.memory_id for h in planned] == [h.memory_id for h in bare]


# ---------------------------------------------------------------------------
# 5. Generated wrapper HAUNT_HOME command substitution
# ---------------------------------------------------------------------------


def test_wrapper_crafted_haunt_home_does_not_execute(tmp_path, monkeypatch):
    """A HAUNT_HOME containing $(touch …) must not create the marker file."""
    from haunt.bootstrap import write_hook_launcher
    from haunt.paths import ensure_layout

    marker = tmp_path / "pwned-from-wrapper"
    crafted = tmp_path / f"home$(touch {marker})"
    monkeypatch.setenv("HAUNT_HOME", str(crafted))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    ensure_layout()
    dest = write_hook_launcher()
    body = dest.read_text(encoding="utf-8")
    export_line = next(line for line in body.splitlines() if "HAUNT_HOME" in line)
    assert "touch" in export_line
    env = {k: v for k, v in os.environ.items() if k != "HAUNT_HOME"}
    subprocess.run(["/bin/sh", "-c", export_line], env=env, check=True)
    assert not marker.exists(), f"export line executed; marker at {marker}"
    subprocess.run(
        ["/bin/sh", str(dest)],
        env=env,
        input=b"",
        capture_output=True,
        timeout=15,
    )
    assert not marker.exists(), f"wrapper executed crafted HAUNT_HOME; marker at {marker}"


def test_wrapper_backtick_haunt_home_does_not_execute(tmp_path, monkeypatch):
    from haunt.bootstrap import write_hook_launcher
    from haunt.paths import ensure_layout

    marker = tmp_path / "pwned-from-backtick"
    crafted = tmp_path / f"home`touch {marker}`"
    monkeypatch.setenv("HAUNT_HOME", str(crafted))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    ensure_layout()
    dest = write_hook_launcher()
    export_line = next(
        line for line in dest.read_text(encoding="utf-8").splitlines() if "HAUNT_HOME" in line
    )
    env = {k: v for k, v in os.environ.items() if k != "HAUNT_HOME"}
    subprocess.run(["/bin/sh", "-c", export_line], env=env, check=True)
    assert not marker.exists()


# ---------------------------------------------------------------------------
# 6. Cheap clamps
# ---------------------------------------------------------------------------


def test_cli_timeline_negative_limit_does_not_dump(honesty_env):
    from haunt.cli import app
    from haunt.store import Store

    with Store("default") as st:
        for i in range(12):
            st.observe(f"cli-limit-canary-{i}", role="user")
        total = st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert total >= 12

    result = CliRunner().invoke(app, ["timeline", "--limit", "-1"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    lines = [ln for ln in result.stdout.splitlines() if ln.strip() and ln != "no events"]
    assert len(lines) == 1, (
        f"limit=-1 must clamp, not dump {len(lines)} of {total} events: {result.stdout!r}"
    )


def test_worldview_negative_caps_are_not_unlimited(honesty_env):
    from haunt.mcp_server import memory_worldview
    from haunt.store import Store

    with Store("default") as st:
        for i in range(8):
            st.observe(
                f"semantic fact UNIQUE-WV-{i}",
                role="system",
                tier="semantic",
            )
        wv = st.worldview(facts_cap=-1, names_cap=-1)
        assert len(wv["facts"]) == 1, (
            f"facts_cap=-1 must clamp to 1, not return {len(wv['facts'])}"
        )

    mcp = json.loads(memory_worldview(namespace="default", facts_cap=-1, names_cap=-1))
    assert len(mcp["facts"]) == 1

    from haunt.cli import app

    result = CliRunner().invoke(
        app,
        ["worldview", "--facts-cap", "-1", "--names-cap", "-1", "--json"],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    cli_wv = json.loads(result.stdout)
    assert len(cli_wv["facts"]) == 1


def test_cli_and_mcp_worldview_call_clamp_helpers():
    from haunt import cli, mcp_server

    assert "clamp_limit" in inspect.getsource(cli.timeline_cmd)
    assert "clamp_limit" in inspect.getsource(cli.worldview_cmd)
    assert "clamp_limit" in inspect.getsource(mcp_server.memory_worldview)
