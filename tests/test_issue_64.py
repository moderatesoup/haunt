"""#64 leftovers: contradict 400/415 no-mutation + FTS-only bootstrap honesty.

Each test is mutation-sensitive — revert the corresponding fix and this fails.
Run under HAUNT_FTS_ONLY=1.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest
from tests.dashutil import make_dash_client


@pytest.fixture
def leftover_env(tmp_path, monkeypatch):
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


def _valid_to(memory_id: str):
    from haunt.store import Store

    with Store("default", create=False) as st:
        row = st.conn.execute(
            "SELECT valid_to FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        assert row is not None
        return row["valid_to"]


def _observe(content: str = "original fact stays current"):
    from haunt.store import Store

    with Store("default") as st:
        return st.observe(content, role="system", tier="semantic")


def _post(path: str, **kwargs):
    return make_dash_client().post(path, **kwargs)


# ---------------------------------------------------------------------------
# 1. Dashboard contradict malformed payload — 400/415, no mutation
# ---------------------------------------------------------------------------


def test_contradict_wrong_type_replacement_is_400_and_keeps_valid_to(leftover_env):
    r = _observe("wrong-type payload must not supersede")
    resp = _post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        json={"replacement": {"not": "a string"}},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "replacement must be a string or null"
    assert _valid_to(r.memory_id) is None


def test_contradict_form_post_is_415_and_keeps_valid_to(leftover_env):
    r = _observe("form POST must not supersede")
    resp = _post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        data={"replacement": "should-not-apply"},
    )
    assert resp.status_code == 415
    assert resp.json()["error"] == "content-type must be application/json"
    assert _valid_to(r.memory_id) is None


def test_contradict_missing_content_type_is_415_and_keeps_valid_to(leftover_env):
    r = _observe("missing content-type must not supersede")
    resp = _post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        content=b'{"replacement":"nope"}',
    )
    assert resp.status_code == 415
    assert resp.json()["error"] == "content-type must be application/json"
    assert _valid_to(r.memory_id) is None


def test_contradict_invalid_json_is_400_and_keeps_valid_to(leftover_env):
    r = _observe("invalid JSON must not supersede")
    resp = _post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid JSON body"
    assert _valid_to(r.memory_id) is None


def test_contradict_unicode_decode_error_is_400_and_keeps_valid_to(leftover_env):
    r = _observe("bad utf-8 must not supersede")
    resp = _post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        content=b"\xff\xfe not utf-8",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid JSON body"
    assert _valid_to(r.memory_id) is None


def test_contradict_non_object_body_is_400_and_keeps_valid_to(leftover_env):
    r = _observe("array body must not supersede")
    resp = _post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        json=["not", "an", "object"],
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "JSON body must be an object"
    assert _valid_to(r.memory_id) is None


def test_contradict_null_body_is_400_and_keeps_valid_to(leftover_env):
    r = _observe("null body must not supersede")
    resp = _post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        content=b"null",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "JSON body must be an object"
    assert _valid_to(r.memory_id) is None


def test_contradict_whitespace_replacement_is_stored_verbatim(leftover_env):
    from haunt.store import Store

    r = _observe("whitespace replacement is intentional")
    resp = _post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        json={"replacement": "   "},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "replacement_memory_id" in data
    assert _valid_to(r.memory_id) is not None
    with Store("default", create=False) as st:
        n = st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        replacement = st.conn.execute(
            "SELECT content FROM memories WHERE id=?",
            (data["replacement_memory_id"],),
        ).fetchone()["content"]
    assert n == 2
    assert replacement == "   "


def test_contradict_valid_after_bad_payloads_still_works(leftover_env):
    r = _observe("survive bad payloads then contradict")
    url = f"/api/namespace/default/memory/{r.memory_id}/contradict"
    assert _post(url, json={"replacement": {"not": "a string"}}).status_code == 400
    assert _post(url, data={"replacement": "nope"}).status_code == 415
    assert _valid_to(r.memory_id) is None
    ok = _post(url, json={"replacement": "corrected fact"})
    assert ok.status_code == 200
    assert ok.json()["ok"] is True
    assert _valid_to(r.memory_id) is not None


def test_store_contradict_non_string_replacement_raises_and_keeps_valid_to(leftover_env):
    from haunt.store import Store

    with Store("default") as st:
        r = st.observe("store ValueError must not mutate", role="system", tier="semantic")
        with pytest.raises(ValueError, match="replacement must be a string or null"):
            st.contradict(r.memory_id, replacement={"not": "a string"})
        row = st.conn.execute(
            "SELECT valid_to FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert row["valid_to"] is None


def test_contradict_handler_source_mentions_status_and_errors():
    """Revert the 400/415 checks and this source lock fails."""
    from haunt import dashboard

    src = inspect.getsource(dashboard.api_contradict)
    assert "import json" in inspect.getsource(dashboard) or "json.loads" in src
    assert "json.loads" in src
    assert "415" in src
    assert "content-type must be application/json" in src
    assert "invalid JSON body" in src
    assert "JSON body must be an object" in src
    assert "replacement must be a string or null" in src
    assert "UnicodeDecodeError" in src
    assert "JSONDecodeError" in src


def test_store_contradict_source_raises_valueerror_for_non_str():
    from haunt.store import Store

    src = inspect.getsource(Store.contradict)
    assert "ValueError" in src
    assert "isinstance(replacement, str)" in src


# ---------------------------------------------------------------------------
# 2. HAUNT_FTS_ONLY=1 bootstrap honesty
# ---------------------------------------------------------------------------


def _fresh_home(tmp_path, monkeypatch, *, fts: bool):
    home = tmp_path / "fresh-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    if fts:
        monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
        monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    else:
        monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)
        monkeypatch.delenv("HAUNT_EMBED_MODEL", raising=False)
    from haunt import embed

    embed.reset()
    return home


def test_fts_only_bootstrap_succeeds_when_vec_probe_fails(tmp_path, monkeypatch):
    """HAUNT_FTS_ONLY=1 must create layout + default ns even if vec cannot load."""
    home = _fresh_home(tmp_path, monkeypatch, fts=True)
    from haunt import embed
    from haunt.bootstrap import bootstrap
    from haunt.paths import namespace_db_path, registry_path
    from haunt.store import Store

    download_called = False

    def boom(*_a, **_k):
        nonlocal download_called
        download_called = True
        raise AssertionError("BGE-M3 download must not run under HAUNT_FTS_ONLY=1")

    with patch(
        "haunt.bootstrap.probe_sqlite_vec",
        return_value={"ok": False, "error": "mocked: extension load failed"},
    ), patch("haunt.embed._download_bge_m3", side_effect=boom):
        report = bootstrap()

    assert not download_called
    assert report["sqlite_vec"]["ok"] is False
    assert report["fts_only"] is True
    assert report["embed"]["available"] is False
    assert report["embed"]["loaded"] == "off"
    assert registry_path().exists()
    assert namespace_db_path("default").exists()

    with Store("default", create=False) as st:
        r = st.observe("fts-only bootstrap canary", role="user", tier="episodic")
        assert r.memory_id
        row = st.conn.execute(
            "SELECT content, valid_to FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert row["content"] == "fts-only bootstrap canary"
        assert row["valid_to"] is None
        assert st.vec_ok() is False

    text = __import__("haunt.bootstrap", fromlist=["format_report"]).format_report(report)
    assert "skipped (FTS-only)" in text
    assert (home / "namespaces" / "default.db").exists()
    embed.reset()


def test_embed_model_off_bootstrap_succeeds_when_vec_probe_fails(tmp_path, monkeypatch):
    """HAUNT_EMBED_MODEL=off is the same FTS gate as HAUNT_FTS_ONLY=1."""
    home = tmp_path / "off-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed
    from haunt.bootstrap import bootstrap
    from haunt.paths import namespace_db_path, registry_path

    embed.reset()
    with patch(
        "haunt.bootstrap.probe_sqlite_vec",
        return_value={"ok": False, "error": "mocked: extension load failed"},
    ), patch(
        "haunt.embed._download_bge_m3",
        side_effect=AssertionError("must not download BGE-M3 when EMBED_MODEL=off"),
    ):
        report = bootstrap()
    assert report["fts_only"] is True
    assert registry_path().exists()
    assert namespace_db_path("default").exists()
    embed.reset()


def test_bootstrap_without_fts_only_still_fatals_on_vec_fail(tmp_path, monkeypatch):
    home = _fresh_home(tmp_path, monkeypatch, fts=False)
    from haunt import embed
    from haunt.bootstrap import BootstrapError, bootstrap
    from haunt.paths import namespace_db_path, registry_path

    with patch(
        "haunt.bootstrap.probe_sqlite_vec",
        return_value={"ok": False, "error": "mocked: extension load failed"},
    ):
        with pytest.raises(BootstrapError) as exc:
            bootstrap()
        assert "sqlite-vec" in exc.value.message.lower()
        assert exc.value.code == 1

    assert not registry_path().exists()
    assert not namespace_db_path("default").exists()
    embed.reset()


def test_bootstrap_source_skips_vec_fatal_only_when_fts_only():
    from haunt.bootstrap import bootstrap

    src = inspect.getsource(bootstrap)
    assert "fts_only()" in src
    assert "probe_sqlite_vec" in src
    assert "BootstrapError" in src
    # Must gate the fatal on the existing FTS helper, not a raw env check
    # that forgets HAUNT_EMBED_MODEL=off.
    assert "not fts_only()" in src or "fts_only()" in src and "if not vec" in src


def test_doctor_sqlite_vec_ok_when_fts_only_and_probe_fails(leftover_env):
    from haunt.doctor import REQUIRED_CHECKS, _check_mcp_python, _check_sqlite_vec

    with patch(
        "haunt.doctor.probe_sqlite_vec",
        return_value={"ok": False, "error": "mocked: extension load failed"},
    ):
        check = _check_sqlite_vec()
    assert check.ok is True
    assert "FTS-only" in check.detail
    assert "sqlite-vec not required" in check.detail

    src = inspect.getsource(_check_sqlite_vec)
    assert "fts_only()" in src
    mcp_src = inspect.getsource(_check_mcp_python)
    assert "fts_only()" in mcp_src
    assert "sqlite-vec" in REQUIRED_CHECKS


def test_readme_advertises_fts_only_bootstrap_without_requiring_vec():
    from pathlib import Path

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "HAUNT_FTS_ONLY=1 haunt bootstrap" in readme
    assert "does not fail if sqlite-vec cannot load" in readme
    assert "does not download BGE-M3" in readme
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    assert "#64" in changelog
