from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lore.embed import available as embed_available
from lore.graph import extract_entities
from lore.recall import recall
from lore.store import Store, list_namespaces, observe


def test_observe_recall_verbatim_rank1(lore_env):
    unique = "The purple widget token ZX-991 lives only in this sentence."
    observe(unique, namespace="default", role="user")
    hits = recall(unique, namespace="default", k=5)
    assert hits, "expected at least one hit"
    assert unique in hits[0].content
    assert hits[0].score > 0


def test_paraphrase_or_fts_only(lore_env):
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


def test_namespace_isolation(lore_env):
    secret = "namespace-A-only secret phrase QUOKKA-77"
    observe(secret, namespace="alpha", role="user")
    observe("harmless note in the other silo", namespace="beta", role="user")
    hits_b = recall("QUOKKA-77", namespace="beta", k=8)
    assert all(secret not in h.content for h in hits_b)
    hits_a = recall("QUOKKA-77", namespace="alpha", k=8)
    assert hits_a and secret in hits_a[0].content
    names = {r["name"] for r in list_namespaces()}
    assert "alpha" in names and "beta" in names


def test_temporal_as_of(lore_env):
    past = "ancient ritual: the brass compass was stored in vault 4"
    observe(past, namespace="default", event_time="2020-01-15T12:00:00+00:00")
    before = recall(past, namespace="default", as_of="2020-01-01T00:00:00+00:00", k=8)
    assert all(past not in h.content for h in before)
    after = recall(past, namespace="default", as_of="2020-02-01T00:00:00+00:00", k=8)
    assert after and past in after[0].content


def test_tool_call_verbatim(lore_env):
    inp = '{"path": "src/lore/store.py", "offset": 1}'
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
    hits = recall("init_schema store.py", namespace="default", k=8)
    assert hits
    assert "init_schema" in hits[0].content or "store.py" in hits[0].content


def test_graph_extract_entities(lore_env):
    sentence = "Alice updated src/lore/store.py in function init_schema() after reviewing Store."
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


def test_cli_smoke(lore_env):
    env = os.environ.copy()
    env["LORE_HOME"] = str(lore_env)
    env["LORE_EMBED_MODEL"] = "BAAI/bge-small-en-v1.5"
    if Path("/workspace/lore/.model-cache").exists():
        env["LORE_MODEL_CACHE"] = "/workspace/lore/.model-cache"
    py = sys.executable
    def run(*args: str) -> str:
        p = subprocess.run(
            [py, "-m", "lore", *args],
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


def test_graph_no_duplicate_triples_same_event(lore_env):
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


def test_reembed_rebuilds_on_dim_mismatch(lore_env):
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


def test_rebuild_graph_rewrites_entities(lore_env):
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
