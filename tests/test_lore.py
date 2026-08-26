from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from haunt.embed import available as embed_available
from haunt.graph import extract_entities
from haunt.recall import recall
from haunt.store import Store, list_namespaces, observe


def test_observe_recall_verbatim_rank1(haunt_env):
    unique = "The purple widget token ZX-991 lives only in this sentence."
    observe(unique, namespace="default", role="user")
    hits = recall(unique, namespace="default", k=5)
    assert hits, "expected at least one hit"
    assert unique in hits[0].content
    assert hits[0].score > 0


def test_paraphrase_or_fts_only(haunt_env):
    text = (
        "The production deployment failed because the primary database "
        "connection timed out during the midnight migration."
    )
    observe(text, namespace="default", role="assistant")
    if embed_available():
        hits = recall("why did the prod db go down last night", namespace="default", k=5)
        assert hits, "paraphrase recall returned nothing"
        assert "database" in hits[0].content.lower() or "migration" in hits[0].content.lower()
    else:
        pytest.skip("embeddings unavailable — FTS-only path covered by verbatim test")
        hits = recall("database connection timed out", namespace="default", k=5)
        assert hits
        assert "database" in hits[0].content.lower()


def test_namespace_isolation(haunt_env):
    secret = "namespace-A-only secret phrase QUOKKA-77"
    observe(secret, namespace="alpha", role="user")
    observe("harmless note in the other silo", namespace="beta", role="user")
    hits_b = recall("QUOKKA-77", namespace="beta", k=8)
    assert all(secret not in h.content for h in hits_b)
    hits_a = recall("QUOKKA-77", namespace="alpha", k=8)
    assert hits_a and secret in hits_a[0].content
    names = {r["name"] for r in list_namespaces()}
    assert "alpha" in names and "beta" in names


def test_temporal_as_of(haunt_env):
    past = "ancient ritual: the brass compass was stored in vault 4"
    observe(past, namespace="default", event_time="2020-01-15T12:00:00+00:00")
    before = recall(past, namespace="default", as_of="2020-01-01T00:00:00+00:00", k=8)
    assert all(past not in h.content for h in before)
    after = recall(past, namespace="default", as_of="2020-02-01T00:00:00+00:00", k=8)
    assert after and past in after[0].content


def test_tool_call_verbatim(haunt_env):
    inp = '{"path": "src/haunt/store.py", "offset": 1}'
    out = "def init_schema(conn):\n    conn.execute('create table events')"
    r = observe(
        "",
        namespace="default",
        role="tool",
        tier="procedural",
        tool_name="Read",
        tool_input=inp,
        tool_output=out,
    )
    with Store("default") as st:
        row = st.conn.execute("SELECT * FROM events WHERE id=?", (r.event_id,)).fetchone()
        assert row["tool_name"] == "Read"
        assert row["tool_input"] == inp
        assert row["tool_output"] == out
        assert row["content"] == ""
    hits = recall(
        "init_schema store.py", namespace="default", k=8, include_residue=True
    )
    assert hits
    assert "init_schema" in hits[0].content or "store.py" in hits[0].content


def test_graph_extract_entities(haunt_env):
    sentence = "Alice updated src/haunt/store.py in function init_schema() after reviewing Store."
    ents = extract_entities(sentence)
    types = {e.type for e in ents}
    names = {e.name for e in ents}
    assert any("store.py" in n for n in names)
    assert any("init_schema" in n for n in names)
    assert any(n == "Alice" or n.startswith("Alice") for n in names)
    observe(sentence, namespace="default")
    with Store("default") as st:
        g = st.graph()
        stored = {e["name"] for e in g["entities"]}
        assert stored
        assert any("store.py" in n or "init_schema" in n or n == "Alice" for n in stored)


def test_cli_smoke(haunt_env):
    env = os.environ.copy()
    env["HAUNT_HOME"] = str(haunt_env)
    env["HAUNT_EMBED_MODEL"] = "BAAI/bge-small-en-v1.5"
    if Path("/workspace/lore/.model-cache").exists():
        env["HAUNT_MODEL_CACHE"] = "/workspace/lore/.model-cache"
    py = sys.executable
    def run(*args: str) -> str:
        p = subprocess.run(
            [py, "-m", "haunt", *args],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return p.stdout
    out = run("init", "cli-smoke")
    assert "cli-smoke" in out
    out = run("observe", "CLI smoke phrase ORBIT-42", "--namespace", "cli-smoke")
    assert "ok" in out and "event=" in out
    out = run("recall", "ORBIT-42", "--namespace", "cli-smoke")
    assert "ORBIT-42" in out
    out = run("namespaces")
    assert "cli-smoke" in out

def test_graph_drops_discourse_keeps_identifiers():
    """Noted/Confirmed are not entities; URLs, files, and functions are."""
    noted = (
        "Noted. Production deploy runbook: stop writers, migrate "
        "postgres://prod-db:5432/app, then bring writers back. "
        "Last timeout was during the midnight migration."
    )
    ents = extract_entities(noted)
    names = {e.name for e in ents}
    norms = {e.norm for e in ents}
    assert "Noted" not in names
    assert "noted" not in norms
    assert not any(n.lower() == "confirmed" for n in names)
    assert not any(n.lower() == "also" for n in names)
    assert not any(n == "5432/app" or n.endswith("5432/app") and "postgres" not in n for n in names)
    assert any(
        "postgres://" in n or "prod-db:5432" in n for n in names
    ), names

    confirmed = (
        "Confirmed. config/secrets.toml holds stripe.webhook_secret. "
        "Next I'll open src/billing/webhooks.py. "
        "The handler is handle_stripe_event()."
    )
    ents2 = extract_entities(confirmed)
    names2 = {e.name for e in ents2}
    assert "Confirmed" not in names2
    assert "Noted" not in names2
    assert "Next" not in names2
    assert any("secrets.toml" in n for n in names2), names2
    assert any("handle_stripe_event" in n for n in names2), names2
    assert any("webhooks.py" in n for n in names2), names2


def test_graph_no_duplicate_triples_same_event(haunt_env):
    text = (
        "Noted. handle_stripe_event reads config/secrets.toml "
        "and config/secrets.toml again."
    )
    r = observe(text, namespace="default")
    with Store("default") as st:
        rows = st.conn.execute(
            """
            SELECT src_entity, rel, dst_entity, COUNT(*) AS n
            FROM relations
            WHERE event_id=?
            GROUP BY src_entity, rel, dst_entity
            """,
            (r.event_id,),
        ).fetchall()
        assert rows, "expected at least one relation"
        assert all(row["n"] == 1 for row in rows)
        stored = {e["name"] for e in st.graph()["entities"]}
        assert "Noted" not in stored
        assert any("secrets.toml" in n for n in stored)
        assert any("handle_stripe_event" in n for n in stored)


# --------------------------------------------------------------------------
# worldview
# --------------------------------------------------------------------------


def test_worldview_lists_semantic_facts_and_procedures(haunt_env):
    """Write a semantic fact + a named procedure; worldview lists both."""
    with Store("default") as st:
        st.observe(
            "The API base URL is https://api.example.com/v2",
            role="system",
            tier="semantic",
            origin="test",
        )
        st.procedure_write(
            "deploy",
            "1. git pull\n2. docker compose up -d\n3. curl health",
            trigger="when deploying to production",
            origin="test",
        )
        wv = st.worldview()

    assert wv["namespace"] == "default"
    assert wv["counts"]["memories"] >= 2
    assert len(wv["facts"]) >= 1
    assert any("api.example.com" in f["content"] for f in wv["facts"])
    assert len(wv["procedures"]) >= 1
    assert any(p["name"] == "deploy" for p in wv["procedures"])
    assert any("deploying to production" in p.get("trigger", "") for p in wv["procedures"])


def test_worldview_excludes_episodic_from_facts(haunt_env):
    """Chat-only episodic turns do NOT show up as facts or procedures."""
    with Store("default") as st:
        st.observe("user asked about the weather", role="user", tier="episodic")
        st.observe(
            "The database host is db.internal:5432",
            role="system",
            tier="semantic",
        )
        wv = st.worldview()

    for f in wv["facts"]:
        assert "weather" not in f["content"]
    assert any("db.internal" in f["content"] for f in wv["facts"])


# --------------------------------------------------------------------------
# procedure
# --------------------------------------------------------------------------


def test_procedure_write_and_get_verbatim(haunt_env):
    """procedure get returns verbatim body."""
    body = "1. Stop the server\n2. Run migrations\n3. Restart"
    with Store("default") as st:
        st.procedure_write("maintenance", body, trigger="when doing maintenance")
        proc = st.procedure_get("maintenance")

    assert proc is not None
    assert proc["name"] == "maintenance"
    assert proc["body"] == body
    assert proc["trigger"] == "when doing maintenance"


def test_procedure_list(haunt_env):
    with Store("default") as st:
        st.procedure_write("deploy", "git pull && make deploy", origin="test")
        st.procedure_write("rollback", "git revert HEAD && make deploy", trigger="when deploy fails", origin="test")
        procs = st.procedure_list()

    names = [p["name"] for p in procs]
    assert "deploy" in names
    assert "rollback" in names


def test_episodic_not_in_procedure_list(haunt_env):
    """Chat-only episodic turns do NOT show up as procedures."""
    with Store("default") as st:
        st.observe("just chatting about code", role="user", tier="episodic")
        st.observe(
            "",
            role="tool",
            tier="procedural",
            tool_name="Read",
            tool_input='{"path": "foo.py"}',
            tool_output="print('hello')",
            origin="test",
        )
        st.procedure_write("real-proc", "do the thing", origin="test")
        procs = st.procedure_list()

    names = [p["name"] for p in procs]
    assert "real-proc" in names
    assert len(procs) == 1


# --------------------------------------------------------------------------
# contradict
# --------------------------------------------------------------------------


def test_contradict_supersedes_memory(haunt_env):
    with Store("default") as st:
        r = st.observe("the port is 8080", role="system", tier="semantic")
        result = st.contradict(
            r.memory_id,
            replacement="the port is 9090",
            idempotency_key="lore-basic",
        )

    assert result["ok"] is True
    assert result["superseded"] == r.memory_id
    assert "replacement_memory_id" in result

    with Store("default") as st:
        old = st.conn.execute(
            "SELECT valid_to FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert old["valid_to"] is not None

        wv = st.worldview()
        contents = [f["content"] for f in wv["facts"]]
        assert any("9090" in c for c in contents)
        assert not any("8080" in c for c in contents)


def test_contradict_not_found(haunt_env):
    with Store("default") as st:
        result = st.contradict(
            "nonexistent-id-12345", idempotency_key="lore-not-found"
        )
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_default_recall_excludes_superseded(haunt_env):
    """Current recall (no as_of) must hide valid_to-set rows.

    Dashboard / contradict claim 'excluded from current recall'. That is the
    current slice (valid_to IS NULL), not an implicit time-phrase parser.
    """
    with Store("default") as st:
        r = st.observe(
            "the port is 8080 UNIQUE-PORT-8080",
            role="system",
            tier="semantic",
            event_time="2024-06-01T12:00:00+00:00",
        )
        st.contradict(
            r.memory_id,
            replacement="the port is 9090 UNIQUE-PORT-9090",
            idempotency_key="lore-current",
        )

    current = recall("UNIQUE-PORT", namespace="default", k=8)
    assert current, "replacement should still recall"
    assert all("8080" not in h.content for h in current)
    assert any("9090" in h.content for h in current)


def test_as_of_past_still_returns_later_superseded(haunt_env):
    """as_of is a valid_from/valid_to snapshot, not an event_time window."""
    with Store("default") as st:
        r = st.observe(
            "the port is 8080 UNIQUE-ASOF-8080",
            role="system",
            tier="semantic",
            event_time="2024-06-01T12:00:00+00:00",
        )
        st.contradict(
            r.memory_id,
            replacement="the port is 9090 UNIQUE-ASOF-9090",
            idempotency_key="lore-as-of",
        )

    past = recall(
        "UNIQUE-ASOF",
        namespace="default",
        as_of="2024-06-15T00:00:00+00:00",
        k=8,
    )
    assert past and any("8080" in h.content for h in past)
    now = recall(
        "UNIQUE-ASOF",
        namespace="default",
        as_of="2099-01-01T00:00:00+00:00",
        k=8,
    )
    assert all("8080" not in h.content for h in now)
    assert any("9090" in h.content for h in now)


def test_reembed_rebuilds_on_dim_mismatch(haunt_env):
    if not embed_available():
        pytest.skip("embeddings unavailable — cannot test dim-mismatch reembed")
    observe("stripe webhook key lives in config/secrets.toml", namespace="default")
    with Store("default") as st:
        before = st.get_meta("embed_dim")
        assert before is not None
        st.set_meta("embed_dim", "999")
        st.set_meta("embed_model", "fake-old-model")
        assert st.embeddings_stale()
        report = st.ensure_current_embeddings()
        assert report is not None
        assert report["updated"] >= 1
        assert st.get_meta("embed_dim") == before
        assert st.get_meta("embed_model") != "fake-old-model"


def test_rebuild_graph_rewrites_entities(haunt_env):
    text = (
        "Noted. config/secrets.toml holds stripe.webhook_secret. "
        "The handler is handle_stripe_event()."
    )
    observe(text, namespace="default")
    with Store("default") as st:
        ev = int(st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        mem = int(st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        st.conn.execute(
            "INSERT INTO entities(id, name, type, norm_name, first_seen, last_seen) "
            "VALUES ('junk-noted', 'Noted', 'proper', 'noted', '2020-01-01', '2020-01-01')"
        )
        st.conn.execute(
            "INSERT INTO entities(id, name, type, norm_name, first_seen, last_seen) "
            "VALUES ('junk-port', '5432/app', 'repo', '5432/app', '2020-01-01', '2020-01-01')"
        )
        st.conn.commit()
        report = st.rebuild_graph()
        assert report["events"] == ev
        assert report["memories"] == mem
        assert int(st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]) == ev
        assert int(st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]) == mem
        names = {e["name"] for e in st.graph()["entities"]}
        assert "Noted" not in names
        assert "5432/app" not in names
        assert any("secrets.toml" in n for n in names)
        assert any("handle_stripe_event" in n for n in names)
        assert report["entities_before"] >= 2
        assert report["entities"] < report["entities_before"]
