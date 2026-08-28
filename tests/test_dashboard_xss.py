"""Stored XSS chain: import validation, rendered markup, and the document CSP."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.dashutil import TEST_DASH_TOKEN, make_dash_client

from haunt import dashboard
from haunt.graph import ENTITY_TYPES
from haunt.portability import (
    ImportBundleError,
    _digest,
    _semantic_from_bundle,
    build_namespace_export,
    canonical_export_bytes,
    import_namespace_bytes,
)
from haunt.store import Store, namespace_exists_readonly

TAG_PAYLOAD = "<img src=x onerror=alert(1)>"
ATTR_PAYLOAD = 'episodic" onmouseover="alert(1)'

# Tags the dashboard never emits: one in rendered output means a stored value
# was parsed as markup instead of text.
_EXECUTABLE_TAG = re.compile(
    r"<\s*(?:img|svg|script|iframe|object|embed|link|form|a)\b", re.I
)
# An on*= handler inside a tag. `[^<>]*` cannot cross a tag boundary, so
# escaped text such as `&lt;img src=x onerror=alert(1)&gt;` does not match.
_HANDLER_ATTR = re.compile(r"<[a-z]+[^<>]*\son[a-z]+\s*=", re.I)

_RICH = (
    "Ada Lovelace ran install_hook() from src/haunt/store.py in haunt/store "
    "with HAUNT_HOME set, see https://example.test/docs for MyWidget and "
    "some_value details"
)


def _dashboard_script() -> str:
    match = re.search(r"<script[^>]*>\n(.*)\n</script>", dashboard.HTML, re.S)
    assert match, "dashboard template no longer has a single inline script"
    return match.group(1).replace(dashboard._HTML_TOKEN_PLACEHOLDER, '""')


@pytest.fixture
def poisoned_namespace(haunt_env):
    """Stored rows carrying markup in every column the dashboard renders."""
    with Store("default") as st:
        st.procedure_write(
            TAG_PAYLOAD, f"procedure body {TAG_PAYLOAD}", trigger=TAG_PAYLOAD
        )
        observed = st.observe(f"XSS-PROBE canary {_RICH}", defer_embedding=True)
        st.conn.execute(
            "UPDATE memories SET tier=? WHERE id=?", (ATTR_PAYLOAD, observed.memory_id)
        )
        st.conn.execute(
            "UPDATE events SET tier=?, role=?, origin=? WHERE id=?",
            (TAG_PAYLOAD, TAG_PAYLOAD, TAG_PAYLOAD, observed.event_id),
        )
        st.conn.execute("UPDATE entities SET type=?, name=?", (TAG_PAYLOAD, TAG_PAYLOAD))
        st.conn.commit()
    return observed.memory_id


def _capture_api(memory_id: str) -> dict:
    """Real dashboard responses for every route the client script fetches."""
    client = make_dash_client()
    paths = [
        "/api/namespaces",
        "/api/namespace/default",
        "/api/namespace/default/browse",
        "/api/namespace/default/timeline",
        "/api/namespace/default/procedures",
        "/api/namespace/default/worldview",
        "/api/namespace/default/health",
        f"/api/namespace/default/memory/{memory_id}",
        "/api/namespace/default/recall",
    ]
    captured: dict = {}
    for path in paths:
        params = {"q": "XSS-PROBE"} if path.endswith("/recall") else None
        response = client.get(path, params=params)
        assert response.status_code == 200, (path, response.text)
        captured[path] = response.json()
    captured["__namespace__"] = "default"
    captured["__memory_id__"] = memory_id
    captured["__query__"] = "XSS-PROBE"
    return captured


def _render_with_node(captured: dict, tmp_path: Path) -> list[dict]:
    harness = (Path(__file__).parent / "dashboard_render_harness.js").read_text()
    prologue, epilogue = harness.split("//__DASHBOARD_SCRIPT__\n")
    bundle = tmp_path / "render.js"
    bundle.write_text(prologue + _dashboard_script() + epilogue)
    payloads = tmp_path / "payloads.json"
    payloads.write_text(json.dumps(captured))
    proc = subprocess.run(
        ["node", str(bundle), str(payloads)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_dashboard_api_serves_the_stored_payload(poisoned_namespace):
    """The sinks are genuinely fed; escaping is what has to stop the payload."""
    client = make_dash_client()
    overview = client.get("/api/namespace/default").json()
    browse = client.get("/api/namespace/default/browse").json()
    assert any(e["tier"] == TAG_PAYLOAD for e in overview["events"])
    assert any(e["type"] == TAG_PAYLOAD for e in overview["entities"])
    assert any(m["tier"] == ATTR_PAYLOAD for m in browse["memories"])


@pytest.mark.skipif(
    shutil.which("node") is None, reason="node runs the dashboard client script"
)
def test_stored_payload_never_renders_as_executable_markup(
    poisoned_namespace, tmp_path
):
    rendered = _render_with_node(_capture_api(poisoned_namespace), tmp_path)
    joined = "".join(block["html"] for block in rendered)
    assert "&lt;img src=x" in joined, "harness never rendered the stored payload"
    for block in rendered:
        assert not _EXECUTABLE_TAG.search(block["html"]), block
        assert not _HANDLER_ATTR.search(block["html"]), block
        assert '" onmouseover' not in block["html"], block


def test_served_document_has_no_inline_event_handlers(haunt_env):
    """Inline handlers would need script-src 'unsafe-inline' to keep working."""
    assert not _HANDLER_ATTR.search(make_dash_client().get("/").text)


def test_index_ships_a_nonce_csp_that_matches_its_script(haunt_env):
    client = make_dash_client()
    first = client.get("/")
    policy = first.headers["content-security-policy"]
    nonce = re.search(r"script-src 'nonce-([A-Za-z0-9_-]+)'", policy)
    assert nonce, policy
    assert f'<script nonce="{nonce.group(1)}">' in first.text
    assert "default-src 'none'" in policy
    assert "'unsafe-inline'" not in policy.split("style-src")[0]
    assert "'unsafe-eval'" not in policy
    assert dashboard._HTML_NONCE_PLACEHOLDER not in first.text
    assert client.get("/").headers["content-security-policy"] != policy


def test_api_responses_are_not_documents(haunt_env):
    response = make_dash_client().get("/api/namespaces")
    assert response.headers["content-security-policy"] == dashboard._API_CSP
    assert response.headers["x-content-type-options"] == "nosniff"


def test_unhandled_errors_still_carry_the_api_hardening_headers(haunt_env, monkeypatch):
    """Starlette builds ServerErrorMiddleware outside `user_middleware`, so a
    500 replied through the raw ASGI send and never reached the guard
    middleware that sets these -- leaving the response easiest to induce
    post-auth as the one with no CSP, no nosniff and a text/plain body."""
    from starlette.testclient import TestClient

    def boom():
        raise RuntimeError("induced-failure-detail")

    monkeypatch.setattr(dashboard, "list_namespaces", boom)
    client = TestClient(
        dashboard.app,
        base_url="http://127.0.0.1:7340",
        headers={"X-Haunt-Token": TEST_DASH_TOKEN},
        raise_server_exceptions=False,
    )
    response = client.get("/api/namespaces")

    assert response.status_code == 500
    assert response.headers["content-security-policy"] == dashboard._API_CSP
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-type"].startswith("application/json")
    assert "induced-failure-detail" not in response.text


@pytest.fixture
def portable_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "source"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed

    embed.reset()
    yield
    embed.reset()


def _crafted_bundle(mutations: dict[tuple[str, str], object] | None = None) -> bytes:
    with Store("portable") as st:
        st.observe(_RICH, event_time="2026-01-01T00:00:00Z", defer_embedding=True)
        st.observe(
            "ran the build",
            role="tool",
            tool_name="Bash",
            event_time="2026-01-01T00:00:01Z",
            defer_embedding=True,
        )
    bundle = build_namespace_export("portable", exported_at="2026-02-01T00:00:00Z")
    for (table, field), value in (mutations or {}).items():
        assert bundle["records"][table], f"nothing seeded in {table}"
        for row in bundle["records"][table]:
            row[field] = value
    bundle["manifest"]["semantic_digest"] = _digest(_semantic_from_bundle(bundle))
    return canonical_export_bytes(bundle)


def _switch_home(monkeypatch, path: Path) -> None:
    monkeypatch.setenv("HAUNT_HOME", str(path))
    from haunt import embed

    embed.reset()


@pytest.mark.parametrize(
    "table,field,value",
    [
        ("events", "tier", TAG_PAYLOAD),
        ("memories", "tier", TAG_PAYLOAD),
        ("memories", "tier", ATTR_PAYLOAD),
        ("entities", "type", TAG_PAYLOAD),
        ("entities", "type", "unknown-type"),
    ],
)
def test_import_rejects_values_outside_an_enumerated_column(
    portable_home, tmp_path, monkeypatch, table, field, value
):
    raw = _crafted_bundle({(table, field): value})
    _switch_home(monkeypatch, tmp_path / "destination")
    with pytest.raises(ImportBundleError):
        import_namespace_bytes(raw)
    assert not namespace_exists_readonly("portable")


def test_clean_bundle_still_imports_with_the_real_type_vocabulary(
    portable_home, tmp_path, monkeypatch
):
    """Guards the whitelists against rejecting what the extractor really emits."""
    raw = _crafted_bundle()
    _switch_home(monkeypatch, tmp_path / "destination")
    assert import_namespace_bytes(raw)["created_namespace"] is True
    with Store("portable", create=False) as st:
        seen = {row[0] for row in st.conn.execute("SELECT DISTINCT type FROM entities")}
    assert len(seen) > 3, seen
    assert seen <= set(ENTITY_TYPES)
