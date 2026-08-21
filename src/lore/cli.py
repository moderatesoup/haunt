"""lore CLI — human-readable stdout, JSON diagnostics on stderr."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from lore import __version__
from lore.bootstrap import bootstrap, format_report
from lore.embed import state as embed_state
from lore.paths import lore_home, resolve_namespace
from lore.recall import recall
from lore.store import Store, list_namespaces, register_namespace
from lore.util import snippet

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="engram (lore) — local-first verbatim memory for AI agents",
)


def _ns(name: Optional[str]) -> str:
    return resolve_namespace(name)


@app.command("bootstrap")
def bootstrap_cmd(
    reembed: bool = typer.Option(
        False,
        "--reembed",
        help="Rebuild embeddings in every namespace for the loaded model (required after a dim change).",
    ),
) -> None:
    """Create ~/.lore, probe sqlite-vec, download the embed model, init default."""
    report = bootstrap(reembed=reembed)
    typer.echo(format_report(report))


@app.command("init")
def init_cmd(
    name: Optional[str] = typer.Argument(None, help="Namespace name (default: inferred)"),
    repo: Optional[Path] = typer.Option(None, "--repo", help="Repo path recorded in the registry"),
) -> None:
    """Create a namespace (one SQLite file)."""
    ns = name or resolve_namespace(None, cwd=repo)
    db = register_namespace(ns, repo_path=str(repo) if repo else None)
    with Store(ns) as st:
        stats = st.stats()
    typer.echo(f"namespace  {ns}")
    typer.echo(f"db         {db}")
    typer.echo(f"events     {stats['events']}")


@app.command("observe")
def observe_cmd(
    text: str = typer.Argument("", help="Verbatim content (empty ok for tool-only events)"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    tier: str = typer.Option("episodic", "--tier"),
    session: Optional[str] = typer.Option(None, "--session"),
    role: str = typer.Option("user", "--role"),
    tool_name: Optional[str] = typer.Option(None, "--tool-name"),
    tool_input: Optional[str] = typer.Option(None, "--tool-input"),
    tool_output: Optional[str] = typer.Option(None, "--tool-output"),
    event_time: Optional[str] = typer.Option(None, "--event-time"),
    origin: str = typer.Option("cli", "--origin"),
) -> None:
    """Store a chat turn or tool call as-is. No summarization."""
    ns = _ns(namespace)
    with Store(ns) as st:
        result = st.observe(
            text,
            role=role,
            tier=tier,
            session_id=session,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            event_time=event_time,
            origin=origin,
        )
    ents = ",".join(result.entities[:8]) if result.entities else "-"
    typer.echo(
        f"ok  event={result.event_id}  memory={result.memory_id}  "
        f"ns={result.namespace}  tier={result.tier}  "
        f"session={result.session_id}  embedded={int(result.embedded)}  entities={ents}"
    )


@app.command("recall")
def recall_cmd(
    query: str = typer.Argument(..., help="Search query (verbatim or paraphrase)"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    as_of: Optional[str] = typer.Option(None, "--as-of"),
    since: Optional[str] = typer.Option(None, "--since"),
    until: Optional[str] = typer.Option(None, "--until"),
    tier: Optional[str] = typer.Option(None, "--tier"),
    k: int = typer.Option(8, "--k"),
) -> None:
    """Hybrid recall (vec + FTS5 + RRF). Prints score, tier, id, snippet."""
    ns = _ns(namespace)
    with Store(ns) as st:
        hits = recall(
            query,
            namespace=ns,
            as_of=as_of,
            since=since,
            until=until,
            tier=tier,
            k=k,
            store=st,
        )
    if not hits:
        typer.echo("no hits")
        return
    typer.echo(f"{'#':<3} {'score':<8} {'tier':<12} {'id':<36} snippet")
    for i, h in enumerate(hits, 1):
        typer.echo(
            f"{i:<3} {h.score:<8.4f} {h.tier:<12} {h.memory_id:<36} {snippet(h.content, 140)}"
        )


@app.command("timeline")
def timeline_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    session: Optional[str] = typer.Option(None, "--session"),
    since: Optional[str] = typer.Option(None, "--since"),
    until: Optional[str] = typer.Option(None, "--until"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """Events in event_time order (newest first)."""
    ns = _ns(namespace)
    with Store(ns) as st:
        rows = st.events(session_id=session, since=since, until=until, limit=limit)
    if not rows:
        typer.echo("no events")
        return
    for r in rows:
        body = r["content"] or ""
        if r["tool_name"]:
            body = f"[tool:{r['tool_name']}] {body}".strip()
        typer.echo(
            f"{r['event_time']}  {r['role']:<10} {r['tier']:<12} {r['id']}  {snippet(body, 120)}"
        )


@app.command("namespaces")
def namespaces_cmd() -> None:
    """List namespaces with counts."""
    rows = list_namespaces()
    if not rows:
        typer.echo("no namespaces (run: lore bootstrap)")
        return
    typer.echo(f"{'name':<24} {'events':>7} {'mem':>7} {'sess':>6} {'ents':>6}  db")
    for r in rows:
        typer.echo(
            f"{r['name']:<24} {r['events']:>7} {r['memories']:>7} {r['sessions']:>6} {r['entities']:>6}  {r['db_path']}"
        )


@app.command("health")
def health_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
) -> None:
    """Store, vec, and embed status."""
    from lore.bootstrap import probe_sqlite_vec
    from lore.paths import registry_path

    es = embed_state()
    vec = probe_sqlite_vec()
    typer.echo(f"lore          v{__version__}")
    typer.echo(f"LORE_HOME     {lore_home()}")
    typer.echo(f"registry      {registry_path()}  exists={registry_path().exists()}")
    typer.echo(
        f"sqlite-vec    {'ok ' + str(vec.get('version')) if vec.get('ok') else 'FAIL'}"
    )
    typer.echo(
        f"embed         loaded={es.model_id} dim={es.dim} requested={es.requested} "
        f"available={es.available} fallback={es.fallback} backend={getattr(es, 'backend', '?')}"
    )
    if es.error:
        typer.echo(f"embed error   {es.error}")
    ns = _ns(namespace)
    with Store(ns) as st:
        s = st.stats()
        typer.echo(f"namespace     {s['namespace']}")
        typer.echo(f"db            {s['db_path']}  bytes={s['db_size_bytes']}  wal={s['wal']}")
        typer.echo(
            f"counts        events={s['events']} memories={s['memories']} "
            f"sessions={s['sessions']} entities={s['entities']} relations={s['relations']}"
        )
        typer.echo(f"tiers         {s['tiers']}")
        typer.echo(f"last write    {s['last_write']}")


@app.command("graph")
def graph_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    entity: Optional[str] = typer.Option(None, "--entity"),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Wipe entities/relations and re-extract from stored events (events/memories kept).",
    ),
) -> None:
    """Entities and typed relations (deterministic extract)."""
    ns = _ns(namespace)
    with Store(ns) as st:
        if rebuild:
            report = st.rebuild_graph()
            typer.echo(
                f"rebuilt graph  ns={ns}  events={report['events']} memories={report['memories']}  "
                f"entities={report['entities_before']}→{report['entities']}  "
                f"relations={report['relations_before']}→{report['relations']}"
            )
        g = st.graph(entity)
        names = {e["id"]: e for e in g["entities"]}
        typer.echo(f"entities ({len(g['entities'])})")
        for e in g["entities"][:40]:
            typer.echo(f"  {e['name']:<32} {e['type']:<12} last={e['last_seen']}")
        typer.echo(f"relations ({len(g['relations'])})")
        for r in g["relations"][:50]:
            src = names.get(r["src_entity"], {}).get("name", r["src_entity"][:8])
            dst = names.get(r["dst_entity"], {}).get("name", r["dst_entity"][:8])
            typer.echo(f"  {src}  --{r['rel']}-->  {dst}  w={r['weight']}")


@app.command("worldview")
def worldview_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    facts_cap: int = typer.Option(12, "--facts-cap"),
    names_cap: int = typer.Option(12, "--names-cap"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Compact namespace briefing: facts, entities, procedures, counts."""
    import json as _json

    ns = _ns(namespace)
    with Store(ns) as st:
        wv = st.worldview(facts_cap=facts_cap, names_cap=names_cap)
    if json_out:
        typer.echo(_json.dumps(wv, ensure_ascii=False, default=str, indent=2))
        return
    typer.echo(f"namespace  {wv['namespace']}")
    typer.echo(f"counts     events={wv['counts']['events']}  memories={wv['counts']['memories']}  sessions={wv['counts']['sessions']}")
    typer.echo("")
    typer.echo(f"facts ({len(wv['facts'])})")
    for f in wv["facts"]:
        typer.echo(f"  {f['id'][:8]}  {snippet(f['content'], 120)}")
    typer.echo("")
    typer.echo(f"names ({len(wv['names'])})")
    for n in wv["names"]:
        typer.echo(f"  {n['name']:<28} {n['type']:<12} mentions={n['mentions']}")
    typer.echo("")
    typer.echo(f"procedures ({len(wv['procedures'])})")
    for p in wv["procedures"]:
        trigger = f"  when: {p['trigger']}" if p.get("trigger") else ""
        typer.echo(f"  {p['name']:<28} {p['id'][:8]}{trigger}")


procedure_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Named how-to procedures (verbatim steps).",
)
app.add_typer(procedure_app, name="procedure")


@procedure_app.command("write")
def procedure_write_cmd(
    name: str = typer.Argument(..., help="Procedure name"),
    body: str = typer.Option(..., "--body", "-b", help="Verbatim step-by-step body"),
    when: Optional[str] = typer.Option(None, "--when", "-w", help="Trigger description"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
) -> None:
    """Store a named procedure."""
    ns = _ns(namespace)
    with Store(ns) as st:
        r = st.procedure_write(name, body, trigger=when or "", origin="cli")
    typer.echo(f"ok  procedure={name}  memory={r.memory_id}  ns={r.namespace}")


@procedure_app.command("get")
def procedure_get_cmd(
    name: str = typer.Argument(..., help="Procedure name"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
) -> None:
    """Retrieve a named procedure."""
    ns = _ns(namespace)
    with Store(ns) as st:
        proc = st.procedure_get(name)
    if not proc:
        typer.echo(f"not found: {name}")
        raise typer.Exit(1)
    typer.echo(f"name     {proc['name']}")
    if proc.get("trigger"):
        typer.echo(f"trigger  {proc['trigger']}")
    typer.echo(f"id       {proc['id']}")
    typer.echo(f"created  {proc['created_at']}")
    typer.echo(f"---")
    typer.echo(proc["body"])


@procedure_app.command("list")
def procedure_list_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
) -> None:
    """List all active procedures."""
    ns = _ns(namespace)
    with Store(ns) as st:
        procs = st.procedure_list()
    if not procs:
        typer.echo("no procedures")
        return
    typer.echo(f"{'name':<28} {'trigger':<32} id")
    for p in procs:
        typer.echo(f"{p['name']:<28} {p.get('trigger', ''):<32} {p['id'][:12]}")


@app.command("cursor-install")
def cursor_install_cmd() -> None:
    """Merge Cursor hooks at ~/.cursor/hooks.json (engram auto-memory)."""
    from lore.cursor_hook import install_cursor_hooks

    report = install_cursor_hooks()
    typer.echo(f"hooks     {report['hooks_json']}")
    typer.echo(f"launcher  {report['launcher']}")
    typer.echo(f"events    {', '.join(report['events'])}")
    typer.echo("merged existing hooks; other commands were kept")
    typer.echo(f"home      {report['lore_home']}  (LORE_HOME / ENGRAM_HOME)")


@app.command("dash")
def dash_cmd(
    port: int = typer.Option(7340, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Start the local metrics dashboard (127.0.0.1)."""
    from lore.dashboard import run_dashboard

    typer.echo(f"lore dash  http://{host}:{port}  home={lore_home()}")
    run_dashboard(host=host, port=port)


@app.callback()
def _root(
    version: bool = typer.Option(False, "--version", help="Print version and exit"),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
