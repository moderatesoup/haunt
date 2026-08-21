"""Local-only metrics dashboard. 127.0.0.1. No React build."""

from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from haunt.embed import state as embed_state
from haunt.paths import haunt_home, resolve_namespace
from haunt.recall import recall
from haunt.store import Store, list_namespaces, namespace_exists

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>haunt</title>
<style>
:root {
  --bg:#090a0d; --panel:#10131a; --panel2:#161a22; --line:#242833;
  --tx:#d7dde8; --mut:#7b8494; --acc:#d6f26a; --acc2:#7dd3fc;
  --ep:#7dd3fc; --se:#c4b5fd; --pr:#fbbf24; --co:#fb7185;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sans: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
}
* { box-sizing:border-box; }
html,body { margin:0; background:var(--bg); color:var(--tx); font:13px/1.45 var(--sans); }
body { min-height:100vh; }
a { color:var(--acc2); text-decoration:none; }
button,input { font:inherit; }
#app { display:grid; grid-template-columns:220px 1fr; min-height:100vh; }
aside {
  border-right:1px solid var(--line); padding:16px 12px; background:var(--panel);
  display:flex; flex-direction:column; gap:12px;
}
.brand { font-family:var(--mono); font-weight:700; letter-spacing:.08em; font-size:14px; }
.brand span { color:var(--acc); }
.sub { color:var(--mut); font-size:11px; font-family:var(--mono); }
.ns-list { display:flex; flex-direction:column; gap:4px; overflow:auto; }
.ns {
  display:flex; justify-content:space-between; gap:8px;
  padding:6px 8px; border:1px solid transparent; border-radius:4px;
  cursor:pointer; font-family:var(--mono); font-size:12px;
}
.ns:hover { background:var(--panel2); }
.ns.on { border-color:var(--acc); background:#161c10; }
.ns b { font-weight:600; }
.ns i { color:var(--mut); font-style:normal; }
main { padding:18px 22px 40px; display:flex; flex-direction:column; gap:16px; }
header.top { display:flex; justify-content:space-between; align-items:baseline; gap:12px; }
h1 { font-size:15px; margin:0; font-family:var(--mono); font-weight:600; }
.pills { display:flex; gap:6px; flex-wrap:wrap; }
.pill {
  font-family:var(--mono); font-size:11px; padding:2px 7px; border-radius:99px;
  border:1px solid var(--line); color:var(--mut);
}
.pill.ok { color:var(--acc); border-color:#3a4a18; }
.grid { display:grid; grid-template-columns:repeat(6,1fr); gap:8px; }
.card {
  background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px 12px;
}
.card .k { color:var(--mut); font-size:10px; letter-spacing:.08em; text-transform:uppercase; font-family:var(--mono); }
.card .v { font-family:var(--mono); font-size:20px; margin-top:4px; }
.card .s { color:var(--mut); font-size:11px; font-family:var(--mono); margin-top:2px; }
.row { display:grid; grid-template-columns:1.4fr .8fr; gap:12px; }
.box { background:var(--panel); border:1px solid var(--line); border-radius:6px; overflow:hidden; }
.box h2 {
  margin:0; padding:8px 12px; font-size:11px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--mut); font-family:var(--mono); border-bottom:1px solid var(--line);
  display:flex; justify-content:space-between; align-items:center;
}
.search { display:flex; gap:6px; padding:8px 12px; border-bottom:1px solid var(--line); }
.search input {
  flex:1; background:var(--panel2); border:1px solid var(--line); color:var(--tx);
  padding:6px 8px; border-radius:4px; font-family:var(--mono);
}
.search button {
  background:var(--acc); color:#111; border:0; padding:6px 10px; border-radius:4px;
  font-family:var(--mono); font-weight:700; cursor:pointer;
}
table { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:12px; }
th { text-align:left; color:var(--mut); font-weight:500; padding:6px 10px; border-bottom:1px solid var(--line); }
td { padding:6px 10px; border-bottom:1px solid #1a1e27; vertical-align:top; }
td.snip { color:#c5cbd6; max-width:520px; word-break:break-word; }
.t-episodic{color:var(--ep)} .t-semantic{color:var(--se)} .t-procedural{color:var(--pr)} .t-coordinate{color:var(--co)}
.empty { color:var(--mut); padding:16px; font-family:var(--mono); }
.ent { display:flex; justify-content:space-between; padding:6px 12px; border-bottom:1px solid #1a1e27; font-family:var(--mono); font-size:12px; }
.ent .ty { color:var(--mut); }
@media (max-width:1100px) {
  #app { grid-template-columns:1fr; }
  aside { border-right:0; border-bottom:1px solid var(--line); }
  .grid { grid-template-columns:repeat(3,1fr); }
  .row { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<div id="app">
  <aside>
    <div>
      <div class="brand">hau<span>nt</span></div>
      <div class="sub" id="home"></div>
    </div>
    <div class="sub">namespaces</div>
    <div class="ns-list" id="nsList"></div>
  </aside>
  <main>
    <header class="top">
      <h1 id="title">—</h1>
      <div class="pills" id="pills"></div>
    </header>
    <section class="grid" id="stats"></section>
    <section class="row">
      <div class="box">
        <h2>recall <span id="recallMeta"></span></h2>
        <div class="search">
          <input id="q" placeholder="paraphrase or verbatim query" />
          <button id="go">recall</button>
        </div>
        <div id="hits"><div class="empty">type a query</div></div>
      </div>
      <div class="box">
        <h2>entities</h2>
        <div id="ents"><div class="empty">none</div></div>
      </div>
    </section>
    <div class="box">
      <h2>recent events</h2>
      <div id="events"><div class="empty">none</div></div>
    </div>
  </main>
</div>
<script>
const $ = (id) => document.getElementById(id);
let NS = null;
function fmtBytes(n){
  if(n<1024) return n+" B";
  if(n<1024*1024) return (n/1024).toFixed(1)+" KB";
  return (n/1024/1024).toFixed(2)+" MB";
}
function tierCls(t){ return "t-"+(t||""); }
async function j(url){
  const r = await fetch(url);
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
function setPills(h){
  const e = h.embed||{};
  const v = h.sqlite_vec||{};
  $("pills").innerHTML = [
    `<span class="pill ok">local</span>`,
    `<span class="pill ${v.ok?'ok':''}">vec ${v.version||'off'}</span>`,
    `<span class="pill ${e.available?'ok':''}">${e.loaded||'fts'} · ${e.dim||0}d</span>`,
    `<span class="pill">verbatim</span>`,
  ].join("");
}
function renderNs(list, current){
  $("nsList").innerHTML = list.map(n =>
    `<div class="ns ${n.name===current?'on':''}" data-ns="${n.name}">
      <b>${n.name}</b><i>${n.events}</i></div>`
  ).join("") || `<div class="empty">none</div>`;
  $("nsList").querySelectorAll(".ns").forEach(el => {
    el.onclick = () => loadNs(el.dataset.ns);
  });
}
function statCards(s){
  const tiers = s.tiers||{};
  const items = [
    ["events", s.events, ""],
    ["memories", s.memories, ""],
    ["sessions", s.sessions, ""],
    ["entities", s.entities, s.relations+" rels"],
    ["db", fmtBytes(s.db_size_bytes||0), s.db_path?s.db_path.split("/").slice(-2).join("/"):""],
    ["last write", (s.last_write||"—").replace("T"," ").replace("+00:00","Z"), ""],
  ];
  $("stats").innerHTML = items.map(([k,v,sub]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${sub}</div></div>`
  ).join("");
}
function eventsTable(rows){
  if(!rows.length){ $("events").innerHTML='<div class="empty">none</div>'; return; }
  $("events").innerHTML = `<table><thead><tr>
    <th>event_time</th><th>role</th><th>tier</th><th>snippet</th></tr></thead><tbody>` +
    rows.map(r => `<tr>
      <td>${(r.event_time||"").replace("T"," ").slice(0,19)}</td>
      <td>${r.role||""}</td>
      <td class="${tierCls(r.tier)}">${r.tier}</td>
      <td class="snip">${esc((r.content|| (r.tool_name?("tool:"+r.tool_name):"")).slice(0,220))}</td>
    </tr>`).join("") + "</tbody></table>";
}
function hitsTable(hits){
  if(!hits.length){ $("hits").innerHTML='<div class="empty">no hits</div>'; return; }
  $("hits").innerHTML = `<table><thead><tr><th>#</th><th>score</th><th>tier</th><th>snippet</th></tr></thead><tbody>` +
    hits.map((h,i) => `<tr>
      <td>${i+1}</td><td>${(h.score||0).toFixed(4)}</td>
      <td class="${tierCls(h.tier)}">${h.tier}</td>
      <td class="snip">${esc(h.snippet||h.content||"")}</td>
    </tr>`).join("") + "</tbody></table>";
}
function entsList(ents){
  if(!ents.length){ $("ents").innerHTML='<div class="empty">none</div>'; return; }
  $("ents").innerHTML = ents.map(e =>
    `<div class="ent"><span>${esc(e.name)}</span><span class="ty">${e.type}</span></div>`
  ).join("");
}
function esc(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function loadNs(name){
  NS = name;
  const data = await j("/api/namespace/"+encodeURIComponent(name));
  $("title").textContent = name;
  $("home").textContent = data.haunt_home || "";
  setPills(data.health||{});
  renderNs(data.namespaces||[], name);
  statCards(data.stats||{});
  eventsTable(data.events||[]);
  entsList(data.entities||[]);
}
async function doRecall(){
  const q = $("q").value.trim();
  if(!q || !NS) return;
  $("recallMeta").textContent = "…";
  const data = await j("/api/namespace/"+encodeURIComponent(NS)+"/recall?q="+encodeURIComponent(q));
  $("recallMeta").textContent = (data.hits||[]).length + " hits";
  hitsTable(data.hits||[]);
}
$("go").onclick = doRecall;
$("q").addEventListener("keydown", e => { if(e.key==="Enter") doRecall(); });
(async () => {
  const boot = await j("/api/namespaces");
  $("home").textContent = boot.haunt_home || "";
  const first = (boot.namespaces[0]||{}).name || "default";
  await loadNs(first);
})();
</script>
</body>
</html>
"""


def _health(ns: str | None = None) -> dict[str, Any]:
    from haunt.bootstrap import probe_sqlite_vec

    es = embed_state()
    payload: dict[str, Any] = {
        "haunt_home": str(haunt_home()),
        "sqlite_vec": probe_sqlite_vec(),
        "embed": {
            "loaded": es.model_id,
            "dim": es.dim,
            "available": es.available,
            "requested": es.requested,
            "fallback": es.fallback,
        },
    }
    return payload


async def index(_request: Request) -> HTMLResponse:
    return HTMLResponse(HTML)


async def api_namespaces(_request: Request) -> JSONResponse:
    return JSONResponse({"haunt_home": str(haunt_home()), "namespaces": list_namespaces()})


async def api_namespace(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    if not namespace_exists(name):
        # still openable via create? dashboard is read-mostly; init empty view
        pass
    with Store(name) as st:
        stats = st.stats()
        events = st.events(limit=40)
        entities = st.top_entities(20)
    return JSONResponse(
        {
            "haunt_home": str(haunt_home()),
            "health": _health(name),
            "namespaces": list_namespaces(),
            "stats": stats,
            "events": events,
            "entities": entities,
        }
    )


async def api_recall(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    q = request.query_params.get("q") or ""
    k = int(request.query_params.get("k") or 8)
    with Store(name) as st:
        hits = recall(q, namespace=name, k=k, store=st)
    return JSONResponse({"query": q, "hits": [h.as_dict() for h in hits]})


routes = [
    Route("/", index),
    Route("/api/namespaces", api_namespaces),
    Route("/api/namespace/{name}", api_namespace),
    Route("/api/namespace/{name}/recall", api_recall),
]

app = Starlette(debug=False, routes=routes)


def run_dashboard(host: str = "127.0.0.1", port: int = 7340) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()
