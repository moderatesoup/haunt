"""Tests for console features: timeline, as_of/since/until recall, contradict."""

from __future__ import annotations

import shutil

import pytest

from haunt.store import Store, observe


@pytest.fixture
def dash_client(haunt_env):
    """HTTPX test client with pre-populated data for console feature tests."""
    from tests.dashutil import make_dash_client

    observe(
        "early event about database migration",
        namespace="default",
        role="user",
        event_time="2024-03-10T10:00:00+00:00",
        origin="cli",
    )
    observe(
        "mid event about API deployment",
        namespace="default",
        role="assistant",
        event_time="2024-03-12T14:00:00+00:00",
        origin="mcp",
    )
    observe(
        "late event about monitoring alerts",
        namespace="default",
        role="user",
        event_time="2024-03-15T09:00:00+00:00",
        origin="cli",
    )
    return make_dash_client()


# --------------------------------------------------------------------------
# Feature 1: Timeline / "what changed"
# --------------------------------------------------------------------------


class TestTimeline:
    def test_timeline_returns_events(self, dash_client):
        r = dash_client.get("/api/namespace/default/timeline")
        assert r.status_code == 200
        data = r.json()
        assert "events" in data
        assert data["namespace"] == "default"
        assert len(data["events"]) >= 3

    def test_timeline_since_filter(self, dash_client):
        r = dash_client.get(
            "/api/namespace/default/timeline?since=2024-03-12T00:00:00%2B00:00"
        )
        assert r.status_code == 200
        events = r.json()["events"]
        assert len(events) >= 2
        for ev in events:
            assert ev["event_time"] >= "2024-03-12T00:00:00"

    def test_timeline_until_filter(self, dash_client):
        r = dash_client.get(
            "/api/namespace/default/timeline?until=2024-03-11T23:59:59%2B00:00"
        )
        assert r.status_code == 200
        events = r.json()["events"]
        assert len(events) >= 1
        for ev in events:
            assert ev["event_time"] <= "2024-03-11T23:59:59"

    def test_timeline_since_until_day_window(self, dash_client):
        r = dash_client.get(
            "/api/namespace/default/timeline"
            "?since=2024-03-12T00:00:00%2B00:00"
            "&until=2024-03-12T23:59:59%2B00:00"
        )
        assert r.status_code == 200
        events = r.json()["events"]
        assert len(events) >= 1
        for ev in events:
            assert ev["event_time"] >= "2024-03-12T00:00:00"
            assert ev["event_time"] <= "2024-03-12T23:59:59"

    def test_timeline_limit(self, dash_client):
        r = dash_client.get("/api/namespace/default/timeline?limit=1")
        assert r.status_code == 200
        events = r.json()["events"]
        assert len(events) == 1

    def test_timeline_events_have_required_fields(self, dash_client):
        r = dash_client.get("/api/namespace/default/timeline")
        assert r.status_code == 200
        for ev in r.json()["events"]:
            assert "id" in ev
            assert "event_time" in ev
            assert "role" in ev
            assert "tier" in ev
            assert "origin" in ev


# --------------------------------------------------------------------------
# Feature 2: as_of / since / until on recall
# --------------------------------------------------------------------------


class TestRecallTemporalFilters:
    def test_recall_since_filters_results(self, dash_client):
        r = dash_client.get(
            "/api/namespace/default/recall"
            "?q=event&since=2024-03-14T00:00:00%2B00:00"
        )
        assert r.status_code == 200
        hits = r.json()["hits"]
        for h in hits:
            assert h["event_time"] >= "2024-03-14T00:00:00"

    def test_recall_until_filters_results(self, dash_client):
        r = dash_client.get(
            "/api/namespace/default/recall"
            "?q=event&until=2024-03-11T00:00:00%2B00:00"
        )
        assert r.status_code == 200
        hits = r.json()["hits"]
        for h in hits:
            assert h["event_time"] <= "2024-03-11T00:00:00"

    def test_recall_as_of_excludes_superseded(self, dash_client):
        """as_of=now (and default current slice) hide valid_to-set rows.

        An as_of *before* valid_from also hides the row — that tests
        valid_from, not contradict. Use a query unique to this memory
        and as_of after valid_from.
        """
        with Store("default") as st:
            r = st.observe(
                "the port is 8080 UNIQUE-DASH-8080",
                role="system",
                tier="semantic",
                event_time="2024-03-10T10:00:00+00:00",
            )
            st.contradict(
                r.memory_id,
                replacement="the port is 9090 UNIQUE-DASH-9090",
                idempotency_key="console-as-of",
            )

        current = dash_client.get(
            "/api/namespace/default/recall?q=UNIQUE-DASH"
        )
        assert current.status_code == 200
        contents = [h.get("content", "") for h in current.json()["hits"]]
        assert not any("8080" in c for c in contents)
        assert any("9090" in c for c in contents)

        past = dash_client.get(
            "/api/namespace/default/recall"
            "?q=UNIQUE-DASH&as_of=2024-03-11T00:00:00%2B00:00"
        )
        assert past.status_code == 200
        past_contents = [h.get("content", "") for h in past.json()["hits"]]
        assert any("8080" in c for c in past_contents)

    def test_all_ns_recall_since(self, dash_client):
        r = dash_client.get(
            "/api/recall?q=event&since=2024-03-14T00:00:00%2B00:00"
        )
        assert r.status_code == 200
        hits = r.json()["hits"]
        for h in hits:
            assert h["event_time"] >= "2024-03-14T00:00:00"

    def test_all_ns_recall_until(self, dash_client):
        r = dash_client.get(
            "/api/recall?q=event&until=2024-03-11T00:00:00%2B00:00"
        )
        assert r.status_code == 200
        hits = r.json()["hits"]
        for h in hits:
            assert h["event_time"] <= "2024-03-11T00:00:00"


# --------------------------------------------------------------------------
# Feature 3: Contradict from the console
# --------------------------------------------------------------------------


class TestContradictRoute:
    def test_contradict_sets_valid_to(self, dash_client):
        """POST contradict sets valid_to on the old memory (not a purge)."""
        r = observe(
            "the sky is green CONTRADICT-TEST-1",
            namespace="default",
            role="system",
            tier="semantic",
        )
        resp = dash_client.post(
            f"/api/namespace/default/memory/{r.memory_id}/contradict",
            json={"idempotency_key": "console-no-replacement"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["superseded"] == r.memory_id
        assert "valid_to" in data

        with Store("default") as st:
            mem = st.conn.execute(
                "SELECT id, valid_to FROM memories WHERE id=?", (r.memory_id,)
            ).fetchone()
            assert mem is not None, "contradict must NOT delete the row"
            assert mem["valid_to"] is not None, "valid_to must be set"

    def test_contradict_with_replacement(self, dash_client):
        r = observe(
            "the database host is db.old.example.com",
            namespace="default",
            role="system",
            tier="semantic",
        )
        resp = dash_client.post(
            f"/api/namespace/default/memory/{r.memory_id}/contradict",
            json={
                "replacement": "the database host is db.new.example.com",
                "idempotency_key": "console-with-replacement",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "replacement_memory_id" in data

        with Store("default") as st:
            old = st.conn.execute(
                "SELECT valid_to FROM memories WHERE id=?", (r.memory_id,)
            ).fetchone()
            assert old["valid_to"] is not None

            new = st.conn.execute(
                "SELECT content FROM memories WHERE id=?",
                (data["replacement_memory_id"],),
            ).fetchone()
            assert "db.new.example.com" in new["content"]

    def test_contradict_not_found(self, dash_client):
        resp = dash_client.post(
            "/api/namespace/default/memory/nonexistent-id-999/contradict",
            json={"idempotency_key": "console-not-found"},
        )
        assert resp.status_code == 404
        data = resp.json()
        assert data["ok"] is False
        assert "not found" in data["error"]
        assert "memory" in data["error"]
        assert "unknown namespace" not in data["error"]

    def test_contradict_does_not_purge(self, dash_client):
        """Contradict MUST NOT delete the memory row — that's purge's job."""
        r = observe(
            "contradict keeps data CANARY-KEEP-ME",
            namespace="default",
            role="user",
            tier="episodic",
        )
        resp = dash_client.post(
            f"/api/namespace/default/memory/{r.memory_id}/contradict",
            json={"idempotency_key": "console-does-not-purge"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        with Store("default") as st:
            mem = st.conn.execute(
                "SELECT id, content, valid_to FROM memories WHERE id=?",
                (r.memory_id,),
            ).fetchone()
            assert mem is not None, "contradict must not delete the memory"
            assert "CANARY-KEEP-ME" in mem["content"]
            assert mem["valid_to"] is not None

    def test_contradict_includes_namespace(self, dash_client):
        r = observe("namespace field test", namespace="default", role="user")
        resp = dash_client.post(
            f"/api/namespace/default/memory/{r.memory_id}/contradict",
            json={"idempotency_key": "console-namespace"},
        )
        assert resp.status_code == 200
        assert resp.json()["namespace"] == "default"


# --------------------------------------------------------------------------
# The console client script itself, run under stubbed browser globals
# --------------------------------------------------------------------------


def _dashboard_script() -> str:
    import re

    from haunt import dashboard

    match = re.search(r"<script[^>]*>\n(.*)\n</script>", dashboard.HTML, re.S)
    assert match, "dashboard template no longer has a single inline script"
    return match.group(1).replace(dashboard._HTML_TOKEN_PLACEHOLDER, '""')


def _run_console(tmp_path, *, purge=None, acts=(), purge_response=None) -> dict:
    """Boot the shipped client script in node against real API payloads."""
    import json
    import subprocess
    from pathlib import Path

    from tests.dashutil import make_dash_client

    client = make_dash_client()
    responses = {
        path: client.get(path).json()
        for path in ("/api/namespaces", "/api/namespace/default")
    }
    if purge is not None:
        responses[
            f"/api/namespace/{purge['namespace']}/memory/{purge['memory_id']}"
        ] = purge_response

    harness = (Path(__file__).parent / "dashboard_console_harness.js").read_text()
    prologue, epilogue = harness.split("//__DASHBOARD_SCRIPT__\n")
    bundle = tmp_path / "console.js"
    bundle.write_text(prologue + _dashboard_script() + epilogue)
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps({"responses": responses, "purge": purge, "acts": list(acts)})
    )
    proc = subprocess.run(
        ["node", str(bundle), str(case)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


_PURGED = {"namespace": "default", "memory_id": "mem-console-probe"}


@pytest.mark.skipif(
    shutil.which("node") is None, reason="node runs the dashboard client script"
)
class TestConsoleClientScript:
    def test_purge_tells_the_operator_when_free_pages_were_not_overwritten(
        self, dash_client, tmp_path
    ):
        """_overwrite_erased_pages returns False whenever a concurrent reader
        blocks the rebuild, and the API still answers 200 {ok:true}. CLI, MCP
        and SECURITY.md all surface that; the console said only "deleted"."""
        result = _run_console(
            tmp_path,
            purge=_PURGED,
            purge_response={"ok": True, "bytes_overwritten": False},
        )
        assert any("bytes_overwritten" in message for message in result["alerts"]), (
            f"purge reported nothing about unerased free pages: {result['alerts']}"
        )

    def test_a_fully_erased_purge_stays_quiet(self, dash_client, tmp_path):
        """The warning must mean something -- it cannot fire on every purge."""
        result = _run_console(
            tmp_path,
            purge=_PURGED,
            purge_response={"ok": True, "bytes_overwritten": True},
        )
        assert result["alerts"] == []

    def test_prototype_keys_never_dispatch_as_actions(self, dash_client, tmp_path):
        """ACTIONS[el.dataset.act] was an unguarded lookup: "__proto__" gave a
        truthy non-callable and "valueOf" an inherited Object.prototype
        method, both reached from inside the one global click handler."""
        result = _run_console(
            tmp_path,
            acts=["__proto__", "valueOf", "toString", "constructor", "purge-cancel"],
        )
        assert result["errors"] == []
        assert result["called"] == ["purge-cancel"]
