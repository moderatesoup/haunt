"""#67 timestamps + #68 typo namespaces.

Mutation-sensitive. Run under HAUNT_FTS_ONLY=1.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from haunt.cli import app
from haunt.paths import namespace_db_path
from haunt.store import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    Store,
    UnknownNamespaceError,
    namespace_exists,
    open_existing,
)
from haunt.util import utc_iso

TYPO = "typo-ns-never"
FROZEN = "2026-08-25T12:00:00.000000+00:00"
runner = CliRunner()


@pytest.fixture
def fts_env(tmp_path, monkeypatch):
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    # These legacy cross-namespace MCP assertions exercise explicit admin mode.
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield home
    embed.reset()


def _home_db(home, name: str):
    return home / "namespaces" / f"{name}.db"


def _assert_no_typo(home):
    assert not namespace_exists(TYPO)
    assert not _home_db(home, TYPO).exists()
    assert not namespace_db_path(TYPO).exists()


# ---------------------------------------------------------------------------
# #67 — microseconds + UTC + one-time migrate + procedure tie-breaker
# ---------------------------------------------------------------------------


def test_now_iso_and_utc_iso_keep_microseconds_utc():
    src = inspect.getsource(utc_iso)
    assert "microseconds" in src
    assert "seconds" not in src.replace("microseconds", "")
    assert utc_iso(datetime(2026, 8, 1, 7, 0, 0, 42, tzinfo=timezone.utc)) == (
        "2026-08-01T07:00:00.000042+00:00"
    )


def test_new_observe_timestamps_have_microseconds(fts_env):
    with Store("fresh-clocks") as st:
        r = st.observe("microsecond canary", origin="test")
        row = st.conn.execute(
            "SELECT ts, event_time FROM events WHERE id=?", (r.event_id,)
        ).fetchone()
        mem = st.conn.execute(
            "SELECT created_at FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()
    for value in (row["ts"], row["event_time"], mem["created_at"]):
        assert value.endswith("+00:00")
        dt = datetime.fromisoformat(value)
        assert "." in value
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timezone.utc.utcoffset(dt)
        assert dt.microsecond >= 0


def test_legacy_offset_fixture_rewritten_once_and_ordered(fts_env):
    """Pre-upgrade -05:00 / naive clocks normalize on first open only."""
    later_raw = "2026-08-01T10:00:00-05:00"  # 15:00 UTC
    earlier_raw = "2026-08-01T14:00:00+00:00"  # 14:00 UTC
    naive_raw = "2026-08-01T16:00:00"
    assert later_raw < earlier_raw, "precondition: text sort is wrong"

    with Store("legacy-clocks") as st:
        sid = "legacy-sess"
        st.conn.execute(
            "INSERT INTO sessions(id, started_at, ended_at, source, meta) VALUES (?,?,?,?,?)",
            (sid, later_raw, None, "test", "{}"),
        )
        st.conn.execute(
            """
            INSERT INTO events(
                id, session_id, ts, event_time, role, content,
                tool_name, tool_input, tool_output, origin, tier, meta
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "evt-later",
                sid,
                later_raw,
                later_raw,
                "user",
                "later offset event",
                None,
                None,
                None,
                "test",
                "episodic",
                "{}",
            ),
        )
        st.conn.execute(
            """
            INSERT INTO events(
                id, session_id, ts, event_time, role, content,
                tool_name, tool_input, tool_output, origin, tier, meta
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "evt-earlier",
                sid,
                earlier_raw,
                earlier_raw,
                "user",
                "earlier utc event",
                None,
                None,
                None,
                "test",
                "episodic",
                "{}",
            ),
        )
        st.conn.execute(
            """
            INSERT INTO events(
                id, session_id, ts, event_time, role, content,
                tool_name, tool_input, tool_output, origin, tier, meta
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "evt-naive",
                sid,
                naive_raw,
                naive_raw,
                "user",
                "naive event",
                None,
                None,
                None,
                "test",
                "episodic",
                "{}",
            ),
        )
        st.conn.execute(
            "DELETE FROM meta WHERE key=?",
            (SCHEMA_VERSION_KEY,),
        )
        st.conn.commit()
        assert st.get_meta(SCHEMA_VERSION_KEY) is None

    with Store("legacy-clocks", create=False) as st:
        assert st.get_meta(SCHEMA_VERSION_KEY) == str(SCHEMA_VERSION)
        rows = {r["id"]: r for r in st.events(limit=10)}
        assert rows["evt-later"]["event_time"] == "2026-08-01T15:00:00.000000+00:00"
        assert rows["evt-earlier"]["event_time"] == "2026-08-01T14:00:00.000000+00:00"
        assert rows["evt-naive"]["event_time"] == "2026-08-01T16:00:00.000000+00:00"
        ordered = [r["id"] for r in st.events(limit=10)]
        assert ordered[0] == "evt-naive"
        assert ordered[1] == "evt-later"
        assert ordered[2] == "evt-earlier"
        first_pass = {
            r["id"]: (r["event_time"], r["ts"]) for r in st.events(limit=10)
        }

    with Store("legacy-clocks", create=False) as st:
        assert st.get_meta(SCHEMA_VERSION_KEY) == str(SCHEMA_VERSION)
        second_pass = {
            r["id"]: (r["event_time"], r["ts"]) for r in st.events(limit=10)
        }
        assert second_pass == first_pass
        leftover = "2026-08-02T09:00:00-05:00"
        st.conn.execute(
            """
            INSERT INTO events(
                id, session_id, ts, event_time, role, content,
                tool_name, tool_input, tool_output, origin, tier, meta
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "evt-post",
                "legacy-sess",
                leftover,
                leftover,
                "user",
                "inserted after migrate",
                None,
                None,
                None,
                "test",
                "episodic",
                "{}",
            ),
        )
        st.conn.commit()
        post = st.conn.execute(
            "SELECT event_time FROM events WHERE id='evt-post'"
        ).fetchone()["event_time"]
        assert post == leftover, "queries must not rewrite clocks"

    with Store("legacy-clocks", create=False) as st:
        post = st.conn.execute(
            "SELECT event_time FROM events WHERE id='evt-post'"
        ).fetchone()["event_time"]
        assert post == leftover, "reopen must not migrate again when version is current"


def test_schema_migrate_is_version_gated():
    import haunt.store as store_mod

    src = inspect.getsource(store_mod._ensure_namespace_schema)
    assert "SCHEMA_VERSION" in src
    assert "_normalize_stored_clocks" in src
    assert "if current >= SCHEMA_VERSION" in src


def test_procedure_get_same_second_returns_later_write(fts_env, monkeypatch):
    monkeypatch.setattr("haunt.store.now_iso", lambda: FROZEN)
    monkeypatch.setattr("haunt.store.iso_or_now", lambda value=None: FROZEN)
    with Store("proc-tie") as st:
        first = st.procedure_write("deploy", "first body", origin="test")
        second = st.procedure_write("deploy", "second body", origin="test")
        c1 = st.conn.execute(
            "SELECT created_at FROM memories WHERE id=?", (first.memory_id,)
        ).fetchone()["created_at"]
        c2 = st.conn.execute(
            "SELECT created_at FROM memories WHERE id=?", (second.memory_id,)
        ).fetchone()["created_at"]
        assert c1 == c2 == FROZEN
        got = st.procedure_get("deploy")
        assert got is not None
        assert got["id"] == second.memory_id
        assert got["body"] == "second body"
        listed = st.procedure_list()
        assert listed[0]["id"] == second.memory_id


def test_latest_row_picks_use_rowid_tiebreak():
    from haunt import planner
    from haunt.store import Store as StoreCls

    assert "rowid DESC" in inspect.getsource(StoreCls.procedure_get)
    assert "rowid DESC" in inspect.getsource(StoreCls.procedure_list)
    assert "rowid DESC" in inspect.getsource(StoreCls.stats)
    assert "rowid DESC" in inspect.getsource(planner._hits_from_events)


# ---------------------------------------------------------------------------
# #68 — typo namespaces must not auto-create
# ---------------------------------------------------------------------------


def test_open_existing_is_the_read_helper():
    src = inspect.getsource(open_existing)
    assert "create=False" in src
    assert "namespace_exists" in src
    assert "UnknownNamespaceError" in src


def test_open_existing_unknown_raises_and_creates_nothing(fts_env):
    _assert_no_typo(fts_env)
    try:
        open_existing(TYPO)
        raise AssertionError("open_existing must raise on unknown ns")
    except UnknownNamespaceError as exc:
        assert TYPO in str(exc)
        assert "unknown namespace" in str(exc)
    _assert_no_typo(fts_env)


def test_mcp_recall_purge_contradict_unknown_ns(fts_env):
    from haunt.mcp_server import memory_contradict, memory_purge, memory_recall

    _assert_no_typo(fts_env)
    rec = json.loads(memory_recall(query="anything", namespace=TYPO))
    assert rec.get("ok") is False
    assert "unknown namespace" in rec.get("error", "")
    _assert_no_typo(fts_env)

    purged = json.loads(memory_purge(memory_id="x", namespace=TYPO))
    assert purged.get("ok") is False
    assert "unknown namespace" in purged.get("error", "")
    _assert_no_typo(fts_env)

    contradicted = json.loads(
        memory_contradict(
            memory_id="x", idempotency_key="unknown-namespace", namespace=TYPO
        )
    )
    assert contradicted.get("ok") is False
    assert "unknown namespace" in contradicted.get("error", "")
    _assert_no_typo(fts_env)


def test_mcp_read_tools_fail_loud_on_typo_ns(fts_env):
    from haunt.mcp_server import (
        memory_health,
        memory_procedure,
        memory_session_end,
        memory_timeline,
        memory_worldview,
    )

    for raw in (
        memory_timeline(namespace=TYPO),
        memory_health(namespace=TYPO),
        memory_worldview(namespace=TYPO),
        memory_session_end(namespace=TYPO),
        memory_procedure(action="get", name="deploy", namespace=TYPO),
        memory_procedure(action="list", namespace=TYPO),
    ):
        data = json.loads(raw)
        assert data.get("ok") is False, data
        assert "unknown namespace" in data.get("error", ""), data
        _assert_no_typo(fts_env)


def test_cli_recall_purge_unknown_ns(fts_env):
    rec = runner.invoke(app, ["recall", "anything", "-n", TYPO])
    combined = f"{rec.stdout}{rec.stderr}{rec.output}"
    assert rec.exit_code != 0
    assert "unknown namespace" in combined
    _assert_no_typo(fts_env)

    purged = runner.invoke(app, ["delete", "fake-memory-id", "--yes", "-n", TYPO])
    combined = f"{purged.stdout}{purged.stderr}{purged.output}"
    assert purged.exit_code != 0
    assert "unknown namespace" in combined
    _assert_no_typo(fts_env)


def test_cli_reads_fail_loud_on_typo_ns(fts_env):
    for args in (
        ["timeline", "-n", TYPO],
        ["health", "-n", TYPO],
        ["worldview", "-n", TYPO],
        ["graph", "-n", TYPO],
        ["procedure", "get", "deploy", "-n", TYPO],
        ["procedure", "list", "-n", TYPO],
    ):
        result = runner.invoke(app, args)
        combined = f"{result.stdout}{result.stderr}{result.output}"
        assert result.exit_code != 0, args
        assert "unknown namespace" in combined, (args, combined)
        _assert_no_typo(fts_env)


def test_observe_and_init_still_create(fts_env):
    from haunt.mcp_server import memory_observe, memory_procedure

    obs = runner.invoke(app, ["observe", "first write creates", "-n", "brand-new-obs"])
    assert obs.exit_code == 0, obs.output
    assert namespace_exists("brand-new-obs")
    assert _home_db(fts_env, "brand-new-obs").exists()

    init = runner.invoke(app, ["init", "brand-new-init"])
    assert init.exit_code == 0, init.output
    assert namespace_exists("brand-new-init")
    assert _home_db(fts_env, "brand-new-init").exists()

    raw = memory_observe(text="mcp observe creates", namespace="mcp-new-obs")
    data = json.loads(raw)
    assert data.get("ok") is True
    assert namespace_exists("mcp-new-obs")

    proc = json.loads(
        memory_procedure(
            action="write",
            name="onboard",
            body="step one",
            namespace="mcp-new-proc",
        )
    )
    assert proc.get("ok") is True
    assert namespace_exists("mcp-new-proc")


def test_mcp_cli_read_paths_use_identity_openers_not_create():
    import haunt.cli as cli
    import haunt.mcp_server as mcp

    for fn in (
        mcp.memory_recall,
        mcp.memory_timeline,
        mcp.memory_health,
        mcp.memory_session_end,
        mcp.memory_worldview,
        mcp.memory_purge,
        mcp.memory_contradict,
    ):
        src = inspect.getsource(fn)
        assert "_open_mcp_store" in src, fn.__name__
        assert "with Store(ns)" not in src, fn.__name__

    proc_src = inspect.getsource(mcp.memory_procedure)
    assert "_open_mcp_store" in proc_src
    assert "with Store(ns)" not in proc_src

    obs_src = inspect.getsource(mcp.memory_observe)
    assert "_open_mcp_store" in obs_src
    assert "with Store(ns)" not in obs_src

    for fn in (
        cli.recall_cmd,
        cli.timeline_cmd,
        cli.health_cmd,
        cli.graph_cmd,
        cli.worldview_cmd,
        cli.delete_cmd,
        cli.procedure_get_cmd,
        cli.procedure_list_cmd,
    ):
        src = inspect.getsource(fn)
        assert "_existing" in src, fn.__name__
        assert "with Store(ns)" not in src, fn.__name__

    assert "with Store(ns)" in inspect.getsource(cli.observe_cmd)
    assert "with Store(ns)" in inspect.getsource(cli.procedure_write_cmd)
    assert "open_existing" in inspect.getsource(cli._existing)
