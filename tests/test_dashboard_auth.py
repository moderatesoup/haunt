"""#66 dashboard Host/Origin/token. Mutation-sensitive. Run under HAUNT_FTS_ONLY=1."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dashutil import TEST_DASH_TOKEN, make_dash_client


@pytest.fixture
def auth_env(tmp_path, monkeypatch):
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


def _observe(content: str):
    from haunt.store import Store

    with Store("default") as st:
        return st.observe(content, role="user", tier="semantic")


def _memory_row(memory_id: str):
    from haunt.store import Store

    with Store("default", create=False) as st:
        return st.conn.execute(
            "SELECT id, valid_to, content FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()


def test_index_loads_locally_without_token(auth_env):
    client = make_dash_client(token=None)
    r = client.get("/")
    assert r.status_code == 200
    assert "haunt" in r.text.lower()


def test_index_injects_launch_token_into_html(auth_env):
    client = make_dash_client()
    r = client.get("/")
    assert r.status_code == 200
    assert TEST_DASH_TOKEN in r.text
    assert "X-Haunt-Token" in r.text
    assert "__HAUNT_LAUNCH_TOKEN__" not in r.text


def test_host_evil_example_is_rejected_and_does_not_delete(auth_env):
    r = _observe("HOST-EVIL-DELETE-CANARY must stay")
    client = make_dash_client(host="evil.example")
    resp = client.request("DELETE", f"/api/namespace/default/memory/{r.memory_id}")
    assert resp.status_code in (400, 403)
    assert _memory_row(r.memory_id) is not None


def test_host_evil_example_is_rejected_and_does_not_contradict(auth_env):
    r = _observe("HOST-EVIL-CONTRADICT-CANARY must stay current")
    client = make_dash_client(host="evil.example")
    resp = client.post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        json={},
    )
    assert resp.status_code in (400, 403)
    row = _memory_row(r.memory_id)
    assert row is not None
    assert row["valid_to"] is None


def test_host_evil_example_rejects_reads(auth_env):
    client = make_dash_client(host="evil.example")
    assert client.get("/api/namespaces").status_code in (400, 403)
    assert client.get("/").status_code in (400, 403)


def test_missing_token_is_401_and_does_not_delete(auth_env):
    r = _observe("MISSING-TOKEN-DELETE-CANARY must stay")
    client = make_dash_client(token=None)
    resp = client.request("DELETE", f"/api/namespace/default/memory/{r.memory_id}")
    assert resp.status_code == 401
    assert _memory_row(r.memory_id) is not None


def test_wrong_token_is_401_and_does_not_contradict(auth_env):
    r = _observe("WRONG-TOKEN-CONTRADICT-CANARY must stay current")
    client = make_dash_client(token="definitely-not-the-token")
    resp = client.post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        json={"replacement": "attacker"},
    )
    assert resp.status_code == 401
    row = _memory_row(r.memory_id)
    assert row is not None
    assert row["valid_to"] is None
    assert row["content"] == "WRONG-TOKEN-CONTRADICT-CANARY must stay current"


def test_missing_token_rejects_get_api(auth_env):
    client = make_dash_client(token=None)
    assert client.get("/api/namespaces").status_code == 401
    assert client.get("/api/namespace/default").status_code == 401


def test_query_token_still_reads(auth_env):
    client = make_dash_client(token=None)
    r = client.get(f"/api/namespaces?token={TEST_DASH_TOKEN}")
    assert r.status_code == 200
    assert "namespaces" in r.json()


def test_unconfigured_token_is_401_on_every_api_route(auth_env):
    from haunt.dashboard import configure_dashboard_security

    configure_dashboard_security(token=None, bind_host="127.0.0.1")
    client = make_dash_client(token=None)
    assert client.get("/").status_code == 200
    assert client.get("/api/namespaces").status_code == 401
    assert client.get("/api/recall?q=x").status_code == 401


def test_cross_origin_contradict_without_token_is_rejected(auth_env):
    r = _observe("CSRF-CONTRADICT-CANARY must stay current")
    client = make_dash_client(token=None)
    resp = client.post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        data={"replacement": "forged"},
        headers={"Origin": "http://evil.example"},
    )
    assert resp.status_code in (401, 403)
    row = _memory_row(r.memory_id)
    assert row is not None
    assert row["valid_to"] is None


def test_cross_origin_contradict_with_token_is_still_rejected(auth_env):
    r = _observe("CROSS-ORIGIN-TOKEN-CANARY must stay current")
    client = make_dash_client()
    resp = client.post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        json={"replacement": "forged"},
        headers={"Origin": "http://evil.example"},
    )
    assert resp.status_code == 403
    row = _memory_row(r.memory_id)
    assert row is not None
    assert row["valid_to"] is None


def test_honest_127_and_token_reads_and_contradicts(auth_env):
    r = _observe("HONEST-LOCAL-CONTRADICT-CANARY unique fact")
    client = make_dash_client()
    read = client.get("/api/namespace/default/recall?q=HONEST-LOCAL-CONTRADICT-CANARY")
    assert read.status_code == 200
    assert any(
        "HONEST-LOCAL-CONTRADICT-CANARY" in (h.get("content") or "")
        for h in read.json()["hits"]
    )
    resp = client.post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        json={"replacement": "corrected locally"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    row = _memory_row(r.memory_id)
    assert row is not None
    assert row["valid_to"] is not None


def test_same_origin_origin_header_still_contradicts(auth_env):
    r = _observe("SAME-ORIGIN-CONTRADICT-CANARY")
    client = make_dash_client()
    resp = client.post(
        f"/api/namespace/default/memory/{r.memory_id}/contradict",
        json={},
        headers={"Origin": "http://127.0.0.1:7340"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert _memory_row(r.memory_id)["valid_to"] is not None


def test_missing_origin_from_local_testclient_still_deletes(auth_env):
    r = _observe("MISSING-ORIGIN-DELETE-CANARY")
    client = make_dash_client()
    resp = client.request("DELETE", f"/api/namespace/default/memory/{r.memory_id}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert _memory_row(r.memory_id) is None


def test_run_dashboard_prints_launch_token(capsys, monkeypatch):
    import uvicorn
    from haunt.dashboard import dashboard_token, run_dashboard

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    run_dashboard(open_browser=False, token="fixed-launch-token")
    out = capsys.readouterr().out
    assert "fixed-launch-token" in out
    assert "X-Haunt-Token" in out
    assert dashboard_token() == "fixed-launch-token"


def test_run_dashboard_mints_token_when_omitted(capsys, monkeypatch):
    import uvicorn
    from haunt.dashboard import dashboard_token, run_dashboard

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    run_dashboard(open_browser=False)
    token = dashboard_token()
    assert token
    assert len(token) >= 16
    assert token in capsys.readouterr().out


def test_allow_remote_without_token_refuses_to_start(monkeypatch):
    import uvicorn
    from haunt.dashboard import run_dashboard

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    with pytest.raises(ValueError, match=r"allow-remote|token"):
        run_dashboard(host="0.0.0.0", allow_remote=True, token="", open_browser=False)


def test_docs_say_allow_remote_is_unsafe_without_token():
    readme = Path("README.md").read_text(encoding="utf-8")
    security = Path("SECURITY.md").read_text(encoding="utf-8")
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    blob = readme + "\n" + security
    assert "--allow-remote" in blob
    assert "unsafe without" in blob.lower()
    assert "launch token" in blob.lower() or "X-Haunt-Token" in blob
    assert "namespaces" in security.lower()
    assert "not authorization" in security.lower() or "not auth" in blob.lower()
    assert "#66" in changelog
