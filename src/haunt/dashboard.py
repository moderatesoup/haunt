"""Local-only memory management dashboard. 127.0.0.1. No React build."""

from __future__ import annotations

import json
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from haunt.embed import state as embed_state
from haunt.paths import haunt_home, resolve_namespace
from haunt.recall import recall, Hit, RRF_K
from haunt.store import Store, list_namespaces, list_namespace_rows, namespace_exists

def pick_default_namespace(namespaces: list[dict[str, Any]]) -> str:
    """Choose the best namespace to display on boot.

    Priority: namespace with the most events, then one named ``haunt``,
    then any namespace with data, then the first namespace, then ``"default"``.
    Never prefer a 0-event namespace when one with data exists.
    """
    if not namespaces:
        return "default"
    with_events = [ns for ns in namespaces if (ns.get("events") or 0) > 0]
    if with_events:
        return max(with_events, key=lambda ns: ns.get("events", 0))["name"]
    haunt = next((ns for ns in namespaces if ns["name"] == "haunt"), None)
    if haunt:
        return haunt["name"]
    return namespaces[0]["name"]


# ---------------------------------------------------------------------------
# HTML: single-file memory management console
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>haunt · memory console</title>
<style>
:root {
  --bg:#090a0d; --panel:#10131a; --panel2:#161a22; --line:#242833;
  --tx:#d7dde8; --mut:#7b8494; --acc:#d6f26a; --acc2:#7dd3fc;
  --ep:#7dd3fc; --se:#c4b5fd; --pr:#fbbf24; --co:#fb7185;
  --red:#ef4444; --green:#22c55e; --amber:#f59e0b;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sans: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
}
*{box-sizing:border-box;margin:0;}
html,body{background:var(--bg);color:var(--tx);font:13px/1.45 var(--sans);min-height:100vh;}
a{color:var(--acc2);text-decoration:none;}
button,input,select{font:inherit;color:var(--tx);}
input,select{background:var(--panel2);border:1px solid var(--line);padding:6px 8px;border-radius:4px;font-family:var(--mono);}
button{background:var(--panel2);border:1px solid var(--line);padding:6px 12px;border-radius:4px;cursor:pointer;font-family:var(--mono);}
button:hover{border-color:var(--acc);color:var(--acc);}
button.danger{color:var(--red);border-color:#5c1919;}
button.danger:hover{background:#2a0e0e;border-color:var(--red);}
button.primary{background:#2a3a10;color:var(--acc);border-color:#3a4a18;}
button.primary:hover{background:#354a14;}

#app{display:grid;grid-template-columns:220px 1fr;min-height:100vh;}
aside{border-right:1px solid var(--line);padding:16px 12px;background:var(--panel);display:flex;flex-direction:column;gap:12px;overflow:auto;}
.brand{font-family:var(--mono);font-weight:700;letter-spacing:.08em;font-size:14px;}
.brand span{color:var(--acc);}
.sub{color:var(--mut);font-size:11px;font-family:var(--mono);}
.nav-section{font-size:10px;color:var(--mut);letter-spacing:.1em;text-transform:uppercase;font-family:var(--mono);margin-top:8px;}
.ns-list{display:flex;flex-direction:column;gap:4px;overflow:auto;}
.ns{display:flex;justify-content:space-between;gap:8px;padding:6px 8px;border:1px solid transparent;border-radius:4px;cursor:pointer;font-family:var(--mono);font-size:12px;}
.ns:hover{background:var(--panel2);}
.ns.on{border-color:var(--acc);background:#161c10;}
.ns b{font-weight:600;} .ns i{color:var(--mut);font-style:normal;}
.nav-btn{display:block;width:100%;text-align:left;padding:6px 8px;border:1px solid transparent;border-radius:4px;cursor:pointer;font-family:var(--mono);font-size:12px;color:var(--tx);background:none;}
.nav-btn:hover{background:var(--panel2);}
.nav-btn.on{border-color:var(--acc2);background:#0e1a22;color:var(--acc2);}

main{padding:18px 22px 40px;display:flex;flex-direction:column;gap:16px;overflow:auto;}
header.top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;}
h1{font-size:15px;margin:0;font-family:var(--mono);font-weight:600;}
h2.section{font-size:12px;color:var(--mut);letter-spacing:.08em;text-transform:uppercase;font-family:var(--mono);margin:0;}
.pills{display:flex;gap:6px;flex-wrap:wrap;}
.pill{font-family:var(--mono);font-size:11px;padding:2px 7px;border-radius:99px;border:1px solid var(--line);color:var(--mut);}
.pill.ok{color:var(--acc);border-color:#3a4a18;}
.pill.fail{color:var(--red);border-color:#5c1919;}
.pill.warn{color:var(--amber);border-color:#5c3e00;}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 12px;}
.card .k{color:var(--mut);font-size:10px;letter-spacing:.08em;text-transform:uppercase;font-family:var(--mono);}
.card .v{font-family:var(--mono);font-size:20px;margin-top:4px;}
.card .s{color:var(--mut);font-size:11px;font-family:var(--mono);margin-top:2px;}

.box{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden;}
.box-header{margin:0;padding:8px 12px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);font-family:var(--mono);border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;}
.search{display:flex;gap:6px;padding:8px 12px;border-bottom:1px solid var(--line);flex-wrap:wrap;}
.search input{flex:1;min-width:180px;}
.search select{min-width:100px;}
.filters{display:flex;gap:6px;padding:8px 12px;border-bottom:1px solid var(--line);flex-wrap:wrap;align-items:center;}
.filters label{font-size:11px;color:var(--mut);font-family:var(--mono);}

table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px;}
th{text-align:left;color:var(--mut);font-weight:500;padding:6px 10px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel);}
td{padding:6px 10px;border-bottom:1px solid #1a1e27;vertical-align:top;}
td.snip{color:#c5cbd6;max-width:400px;word-break:break-word;}
tr.clickable{cursor:pointer;} tr.clickable:hover{background:var(--panel2);}
.t-episodic{color:var(--ep)} .t-semantic{color:var(--se)} .t-procedural{color:var(--pr)} .t-coordinate{color:var(--co)}
.empty{color:var(--mut);padding:16px;font-family:var(--mono);}
.ent{display:flex;justify-content:space-between;padding:6px 12px;border-bottom:1px solid #1a1e27;font-family:var(--mono);font-size:12px;}
.ent .ty{color:var(--mut);}

.detail-panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px;display:none;}
.detail-panel.open{display:block;}
.detail-row{display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #1a1e27;font-family:var(--mono);font-size:12px;}
.detail-row .lbl{color:var(--mut);min-width:120px;flex-shrink:0;}
.detail-row .val{word-break:break-all;}
.detail-content{background:var(--panel2);padding:12px;border-radius:4px;font-family:var(--mono);font-size:12px;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow:auto;margin:8px 0;}

.health-strip{display:flex;gap:8px;flex-wrap:wrap;padding:8px 12px;}
.health-item{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px;}
.pulse{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.pulse.ok{background:var(--green);box-shadow:0 0 6px var(--green);}
.pulse.fail{background:var(--red);box-shadow:0 0 6px var(--red);}
.pulse.warn{background:var(--amber);box-shadow:0 0 6px var(--amber);}

.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;display:none;align-items:center;justify-content:center;}
.modal-bg.open{display:flex;}
.modal{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:20px;max-width:440px;width:90%;}
.modal h3{font-family:var(--mono);font-size:14px;margin-bottom:12px;}
.modal p{font-size:12px;color:var(--mut);margin-bottom:16px;font-family:var(--mono);}
.modal .actions{display:flex;gap:8px;justify-content:flex-end;}

.tab-bar{display:flex;gap:0;border-bottom:1px solid var(--line);}
.tab{padding:8px 16px;font-family:var(--mono);font-size:12px;cursor:pointer;border-bottom:2px solid transparent;color:var(--mut);}
.tab:hover{color:var(--tx);}
.tab.on{color:var(--acc2);border-color:var(--acc2);}
.tab-content{display:none;} .tab-content.on{display:block;}

.page-nav{display:flex;gap:8px;padding:8px 12px;align-items:center;font-family:var(--mono);font-size:11px;color:var(--mut);}
.ns-badge{font-family:var(--mono);font-size:10px;padding:1px 5px;border-radius:3px;background:#1a2030;border:1px solid var(--line);color:var(--acc2);}
.persistent-health{border-bottom:1px solid var(--line);background:var(--panel);padding:0;}
.persistent-health .health-strip{padding:6px 12px;}
.ns.all-ns{color:var(--acc2);font-style:italic;}

@media(max-width:1100px){
  #app{grid-template-columns:1fr;}
  aside{border-right:0;border-bottom:1px solid var(--line);}
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
    <div class="nav-section">namespaces</div>
    <div class="ns-list" id="nsList"></div>
    <div class="nav-section" style="margin-top:16px">views</div>
    <button class="nav-btn on" data-view="overview" onclick="switchView('overview')">overview</button>
    <button class="nav-btn" data-view="timeline" onclick="switchView('timeline')">timeline</button>
    <button class="nav-btn" data-view="browse" onclick="switchView('browse')">browse memories</button>
    <button class="nav-btn" data-view="search" onclick="switchView('search')">search / recall</button>
    <button class="nav-btn" data-view="procedures" onclick="switchView('procedures')">procedures</button>
    <button class="nav-btn" data-view="worldview" onclick="switchView('worldview')">worldview</button>
    <button class="nav-btn" data-view="health" onclick="switchView('health')">health</button>
  </aside>
  <main>
    <div class="persistent-health" id="persistentHealth">
      <div class="health-strip" id="healthStripGlobal"></div>
    </div>
    <header class="top">
      <h1 id="title">—</h1>
      <div class="pills" id="pills"></div>
    </header>

    <!-- OVERVIEW VIEW -->
    <div id="view-overview" class="tab-content on">
      <section class="grid" id="stats"></section>
      <div class="box">
        <div class="box-header">health <span id="healthAge"></span></div>
        <div class="health-strip" id="healthStrip"></div>
      </div>
      <div class="box">
        <div class="box-header">recent events</div>
        <div id="events"><div class="empty">none</div></div>
      </div>
      <div class="box">
        <div class="box-header">top entities</div>
        <div id="ents"><div class="empty">none</div></div>
      </div>
    </div>

    <!-- TIMELINE VIEW -->
    <div id="view-timeline" class="tab-content">
      <div class="box">
        <div class="box-header">timeline — what changed</div>
        <div class="filters">
          <label>since</label><input id="tlSince" type="date"/>
          <label>until</label><input id="tlUntil" type="date"/>
          <label>limit</label><input id="tlLimit" type="number" value="200" style="width:70px"/>
          <button class="primary" onclick="loadTimeline()">filter</button>
        </div>
        <div id="timelineResults"><div class="empty">pick a namespace and click filter</div></div>
      </div>
    </div>

    <!-- BROWSE VIEW -->
    <div id="view-browse" class="tab-content">
      <div class="box">
        <div class="box-header">browse memories</div>
        <div class="filters">
          <label>tier</label><select id="bTier"><option value="">all</option><option>episodic</option><option>semantic</option><option>procedural</option><option>coordinate</option></select>
          <label>origin</label><select id="bOrigin"><option value="">all</option><option>cli</option><option>mcp</option><option>cursor-hook</option><option>test</option></select>
          <label>session</label><input id="bSession" placeholder="session id" style="width:180px"/>
          <label>since</label><input id="bSince" type="date"/>
          <label>until</label><input id="bUntil" type="date"/>
          <button class="primary" onclick="doBrowse(0)">filter</button>
        </div>
        <div id="browseResults"><div class="empty">apply filters or click filter to load</div></div>
        <div class="page-nav" id="browseNav"></div>
      </div>
    </div>

    <!-- SEARCH VIEW -->
    <div id="view-search" class="tab-content">
      <div class="box">
        <div class="box-header">recall <span id="recallMeta"></span></div>
        <div class="search">
          <input id="q" placeholder="paraphrase or verbatim query"/>
          <select id="sTier"><option value="">all tiers</option><option>episodic</option><option>semantic</option><option>procedural</option><option>coordinate</option></select>
          <button class="primary" id="go">recall</button>
        </div>
        <div class="filters">
          <label>as_of</label><input id="sAsOf" type="date" title="point-in-time snapshot: only memories valid at this date"/>
          <label>since</label><input id="sSince" type="date" title="events from this date onward"/>
          <label>until</label><input id="sUntil" type="date" title="events up to this date"/>
        </div>
        <div id="hits"><div class="empty">type a query</div></div>
      </div>
    </div>

    <!-- PROCEDURES VIEW -->
    <div id="view-procedures" class="tab-content">
      <div class="box">
        <div class="box-header">procedures</div>
        <div id="procList"><div class="empty">loading…</div></div>
      </div>
    </div>

    <!-- WORLDVIEW VIEW -->
    <div id="view-worldview" class="tab-content">
      <div class="box">
        <div class="box-header">worldview — semantic facts</div>
        <div id="wvFacts"><div class="empty">loading…</div></div>
      </div>
      <div class="box" style="margin-top:12px">
        <div class="box-header">worldview — top names</div>
        <div id="wvNames"><div class="empty">loading…</div></div>
      </div>
      <div class="box" style="margin-top:12px">
        <div class="box-header">worldview — procedure index</div>
        <div id="wvProcs"><div class="empty">loading…</div></div>
      </div>
    </div>

    <!-- HEALTH VIEW -->
    <div id="view-health" class="tab-content">
      <div class="box">
        <div class="box-header">system health</div>
        <div id="healthDetail"><div class="empty">loading…</div></div>
      </div>
    </div>

    <!-- MEMORY DETAIL PANEL (overlay) -->
    <div id="detailPanel" class="detail-panel">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px;">
        <h2 class="section">memory detail</h2>
        <div style="display:flex;gap:8px;">
          <button style="color:var(--amber);border-color:#5c3e00;" onclick="confirmContradict()">supersede</button>
          <button class="danger" onclick="confirmPurge()">permanently delete</button>
          <button onclick="closeDetail()">close</button>
        </div>
      </div>
      <div id="detailBody"></div>
    </div>
  </main>
</div>

<!-- CONFIRM MODAL -->
<div class="modal-bg" id="confirmModal">
  <div class="modal">
    <h3>permanently delete memory</h3>
    <p id="confirmText">This will permanently delete the memory, its FTS index, vector embedding, and associated graph data. This cannot be undone. To keep the data but mark it outdated, use <b>supersede</b> instead.</p>
    <div class="actions">
      <button onclick="closeModal()">cancel</button>
      <button class="danger" id="confirmBtn" onclick="doPurge()">delete permanently</button>
    </div>
  </div>
</div>

<!-- SUPERSEDE MODAL -->
<div class="modal-bg" id="contradictModal">
  <div class="modal">
    <h3>supersede memory</h3>
    <p>This marks the memory as superseded (sets valid_to = now). The original data is <b>kept</b> but excluded from current recall. This is NOT a delete.</p>
    <div style="margin-bottom:12px;">
      <label style="font-size:11px;color:var(--mut);font-family:var(--mono);display:block;margin-bottom:4px;">optional replacement text</label>
      <input id="contradictReplacement" style="width:100%;" placeholder="new corrected fact (leave blank to just supersede)"/>
    </div>
    <div class="actions">
      <button onclick="closeContradictModal()">cancel</button>
      <button style="background:#3a2800;color:var(--amber);border-color:#5c3e00;" onclick="doContradict()">supersede</button>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let NS=null, DETAIL_MID=null, DETAIL_NS=null, BROWSE_PAGE=0, ALL_NS=false;

function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmtBytes(n){if(n<1024)return n+" B";if(n<1048576)return(n/1024).toFixed(1)+" KB";return(n/1048576).toFixed(2)+" MB";}
function tierCls(t){return "t-"+(t||"");}
function fmtTime(s){return(s||"—").replace("T"," ").replace(/\+00:00$/,"Z").slice(0,19);}
function snip(s,n){s=(s||"").replace(/\s+/g," ");return s.length<=n?s:s.slice(0,n-1)+"…";}
async function j(url,opts){const r=await fetch(url,opts);if(!r.ok)throw new Error(await r.text());return r.json();}

function showAllNsHint(container){
  Array.from(container.children).forEach(el=>{
    if(!el.classList.contains('allns-hint')) el.dataset.hiddenByAllns='1', el.style.display='none';
  });
  let hint=container.querySelector('.allns-hint');
  if(!hint){
    hint=document.createElement('div');
    hint.className='empty allns-hint';
    hint.textContent='pick a namespace to use this view';
    container.appendChild(hint);
  }
  hint.style.display='';
}

function hideAllNsHint(container){
  const hint=container.querySelector('.allns-hint');
  if(hint) hint.style.display='none';
  Array.from(container.children).forEach(el=>{
    if(el.dataset.hiddenByAllns){delete el.dataset.hiddenByAllns; el.style.display='';}
  });
}

function switchView(v){
  document.querySelectorAll('.tab-content').forEach(el=>el.classList.remove('on'));
  document.querySelectorAll('.nav-btn').forEach(el=>el.classList.remove('on'));
  $('view-'+v).classList.add('on');
  document.querySelector(`.nav-btn[data-view="${v}"]`).classList.add('on');
  if(ALL_NS && v!=='search' && v!=='overview'){
    showAllNsHint($('view-'+v));
    return;
  }
  hideAllNsHint($('view-'+v));
  if(v==='timeline')loadTimeline();
  if(v==='browse')doBrowse(0);
  if(v==='procedures')loadProcedures();
  if(v==='worldview')loadWorldview();
  if(v==='health')loadHealth();
}

function setPills(h){
  const e=h.embed||{},v=h.sqlite_vec||{};
  $("pills").innerHTML=[
    `<span class="pill ok">local · 127.0.0.1</span>`,
    `<span class="pill ${v.ok?'ok':'fail'}">vec ${v.version||'off'}</span>`,
    `<span class="pill ${e.available?'ok':'warn'}">${e.loaded||'fts'} · ${e.dim||0}d</span>`,
    `<span class="pill">verbatim</span>`,
  ].join("");
}

function renderNs(list,current){
  const allOn=ALL_NS;
  let html=`<div class="ns all-ns ${allOn?'on':''}" data-ns="__all__"><b>all namespaces</b><i>${list.reduce((s,n)=>s+(n.events||0),0)}</i></div>`;
  html+=list.map(n=>`<div class="ns ${!allOn&&n.name===current?'on':''}" data-ns="${esc(n.name)}"><b>${esc(n.name)}</b><i>${n.error?'error':n.events}</i></div>`).join("");
  $("nsList").innerHTML=html||`<div class="empty">none</div>`;
  $("nsList").querySelectorAll(".ns").forEach(el=>{
    el.onclick=()=>{
      if(el.dataset.ns==='__all__') selectAllNs();
      else loadNs(el.dataset.ns);
    };
  });
}

function renderHealthGlobal(h,stats){
  const e=h.embed||{},v=h.sqlite_vec||{};
  let lastWrite=stats.last_write;
  let age="unknown";
  if(lastWrite){
    const ms=Date.now()-new Date(lastWrite).getTime();
    if(ms<60000)age=Math.round(ms/1000)+"s ago";
    else if(ms<3600000)age=Math.round(ms/60000)+"m ago";
    else if(ms<86400000)age=Math.round(ms/3600000)+"h ago";
    else age=Math.round(ms/86400000)+"d ago";
  }
  let items;
  if(ALL_NS){
    items=[
      `<div class="health-item"><span class="pulse ${v.ok?'ok':'fail'}"></span>sqlite-vec ${v.ok?v.version:'off'}</div>`,
      `<div class="health-item"><span class="pulse ${e.available?'ok':'warn'}"></span>embed ${e.loaded||'none'} ${e.dim||0}d</div>`,
      `<div class="health-item"><span class="pulse ok"></span>${stats.namespace_count||0} namespaces</div>`,
      `<div class="health-item"><span class="pulse ok"></span>${stats.haunt_home||''}</div>`,
    ];
  }else{
    items=[
      `<div class="health-item"><span class="pulse ${v.ok?'ok':'fail'}"></span>sqlite-vec ${v.ok?v.version:'off'}</div>`,
      `<div class="health-item"><span class="pulse ${e.available?'ok':'warn'}"></span>embed ${e.loaded||'none'} ${e.dim||0}d</div>`,
      `<div class="health-item"><span class="pulse ok"></span>ns: ${esc(stats.namespace||NS)}</div>`,
      `<div class="health-item"><span class="pulse ok"></span>${fmtBytes(stats.db_size_bytes||0)}</div>`,
      `<div class="health-item"><span class="pulse ${lastWrite?'ok':'warn'}"></span>write ${age}</div>`,
      `<div class="health-item"><span class="pulse ok"></span>${stats.events||0} events</div>`,
    ];
  }
  $("healthStripGlobal").innerHTML=items.join("");
}

function statCards(s){
  const items=[
    ["events",s.events,""],["memories",s.memories,""],["sessions",s.sessions,""],
    ["entities",s.entities,s.relations+" rels"],
    ["db",fmtBytes(s.db_size_bytes||0),s.db_path?s.db_path.split("/").slice(-2).join("/"):""],
    ["last write",fmtTime(s.last_write),""],
  ];
  $("stats").innerHTML=items.map(([k,v,sub])=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${sub}</div></div>`).join("");
}

function renderHealthStrip(h,stats){
  const e=h.embed||{},v=h.sqlite_vec||{};
  let lastWrite=stats.last_write;
  let age="unknown";
  if(lastWrite){
    const ms=Date.now()-new Date(lastWrite).getTime();
    if(ms<60000)age=Math.round(ms/1000)+"s ago";
    else if(ms<3600000)age=Math.round(ms/60000)+"m ago";
    else if(ms<86400000)age=Math.round(ms/3600000)+"h ago";
    else age=Math.round(ms/86400000)+"d ago";
  }
  $("healthAge").textContent=age;
  $("healthStrip").innerHTML=[
    `<div class="health-item"><span class="pulse ${v.ok?'ok':'fail'}"></span>sqlite-vec ${v.ok?v.version:'off'}</div>`,
    `<div class="health-item"><span class="pulse ${e.available?'ok':'warn'}"></span>embed ${e.loaded||'none'} ${e.dim||0}d</div>`,
    `<div class="health-item"><span class="pulse ok"></span>namespace ${esc(stats.namespace||NS)}</div>`,
    `<div class="health-item"><span class="pulse ok"></span>${fmtBytes(stats.db_size_bytes||0)}</div>`,
    `<div class="health-item"><span class="pulse ${lastWrite?'ok':'warn'}"></span>last write ${age}</div>`,
    `<div class="health-item"><span class="pulse ok"></span>${stats.events||0} events</div>`,
    `<div class="health-item"><span class="pulse ok"></span>${stats.db_path||''}</div>`,
  ].join("");
  renderHealthGlobal(h,stats);
}

function eventsTable(rows){
  if(!rows.length){$("events").innerHTML='<div class="empty">none</div>';return;}
  $("events").innerHTML=`<table><thead><tr><th>event_time</th><th>role</th><th>tier</th><th>origin</th><th>snippet</th></tr></thead><tbody>`+
    rows.map(r=>`<tr class="clickable" onclick="openEventMemory('${esc(r.id||"")}')">
      <td>${fmtTime(r.event_time)}</td><td>${r.role||""}</td>
      <td class="${tierCls(r.tier)}">${r.tier}</td><td>${r.origin||""}</td>
      <td class="snip">${esc(snip(r.content||(r.tool_name?"tool:"+r.tool_name:""),180))}</td>
    </tr>`).join("")+"</tbody></table>";
}

function hitsTable(hits){
  if(!hits.length){$("hits").innerHTML='<div class="empty">no hits</div>';return;}
  $("hits").innerHTML=`<table><thead><tr><th>#</th><th>score</th><th>tier</th><th>origin</th>${ALL_NS?'<th>namespace</th>':''}<th>memory_id</th><th>snippet</th><th></th></tr></thead><tbody>`+
    hits.map((h,i)=>`<tr class="clickable" onclick="openDetail('${esc(h.memory_id)}','${esc(h.namespace||NS)}')">
      <td>${i+1}</td><td>${(h.score||0).toFixed(4)}</td>
      <td class="${tierCls(h.tier)}">${h.tier}</td>
      <td style="font-size:11px;color:var(--mut)">${h.origin||''}</td>
      ${ALL_NS?`<td><span class="ns-badge">${esc(h.namespace||'')}</span></td>`:''}
      <td style="font-size:11px;color:var(--mut)">${(h.memory_id||"").slice(0,12)}</td>
      <td class="snip">${esc(snip(h.content||h.snippet||"",200))}</td>
      <td><button style="font-size:11px;padding:2px 8px" onclick="event.stopPropagation();openDetail('${esc(h.memory_id)}','${esc(h.namespace||NS)}')">detail</button></td>
    </tr>`).join("")+"</tbody></table>";
}

function entsList(ents){
  if(!ents.length){$("ents").innerHTML='<div class="empty">none</div>';return;}
  $("ents").innerHTML=ents.map(e=>`<div class="ent"><span>${esc(e.name)}</span><span class="ty">${e.type}</span></div>`).join("");
}

async function openEventMemory(eventId){
  if(!NS)return;
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/event/${encodeURIComponent(eventId)}/memories`);
  const mems=data.memories||[];
  if(mems.length>0)openDetail(mems[0],NS);
  else alert("No memories for this event");
}

async function openDetail(memId,ns){
  ns=ns||NS;
  DETAIL_MID=memId;
  DETAIL_NS=ns;
  const d=await j(`/api/namespace/${encodeURIComponent(ns)}/memory/${encodeURIComponent(memId)}`);
  const dp=$("detailPanel");
  dp.classList.add('open');
  const rows=[
    ["memory_id",d.memory_id],["event_id",d.event_id],["session_id",d.session_id],
    ["namespace",d.namespace],["tier",`<span class="${tierCls(d.tier)}">${d.tier}</span>`],
    ["role",d.role],["origin",d.origin],
    ["event_time",fmtTime(d.event_time)],["valid_from",fmtTime(d.valid_from)],
    ["valid_to",d.valid_to?fmtTime(d.valid_to):"<em>current</em>"],
    ["created_at",fmtTime(d.created_at)],
    ["has_embedding",d.has_embedding?"yes":"no"],
    ["db_path",d.db_path],["haunt_home",d.haunt_home],
  ];
  if(d.tool_name)rows.push(["tool_name",d.tool_name]);
  let html=rows.map(([l,v])=>`<div class="detail-row"><span class="lbl">${l}</span><span class="val">${v}</span></div>`).join("");
  html+=`<h2 class="section" style="margin-top:12px;">content</h2><div class="detail-content">${esc(d.content||d.event_content||"(empty)")}</div>`;
  if(d.tool_input)html+=`<h2 class="section">tool input</h2><div class="detail-content">${esc(d.tool_input)}</div>`;
  if(d.tool_output)html+=`<h2 class="section">tool output</h2><div class="detail-content">${esc(d.tool_output)}</div>`;
  if(d.entity_mentions&&d.entity_mentions.length){
    html+=`<h2 class="section" style="margin-top:12px;">entity mentions (${d.entity_mentions.length})</h2>`;
    html+=d.entity_mentions.map(e=>`<div class="ent"><span>${esc(e.name)}</span><span class="ty">${e.type}</span></div>`).join("");
  }
  if(d.related_memories&&d.related_memories.length){
    html+=`<h2 class="section" style="margin-top:12px;">related memories (same session)</h2>`;
    html+=`<table><thead><tr><th>id</th><th>tier</th><th>snippet</th></tr></thead><tbody>`;
    html+=d.related_memories.map(r=>`<tr class="clickable" onclick="openDetail('${esc(r.memory_id)}','${esc(ns)}')">
      <td style="font-size:11px">${(r.memory_id||"").slice(0,12)}</td>
      <td class="${tierCls(r.tier)}">${r.tier}</td>
      <td class="snip">${esc(snip(r.content||"",160))}</td>
    </tr>`).join("");
    html+="</tbody></table>";
  }
  $("detailBody").innerHTML=html;
  dp.scrollIntoView({behavior:'smooth',block:'start'});
}

function closeDetail(){$("detailPanel").classList.remove('open');DETAIL_MID=null;DETAIL_NS=null;}

function confirmPurge(){
  if(!DETAIL_MID)return;
  $("confirmText").textContent=`Permanently delete memory ${DETAIL_MID.slice(0,12)}…? This removes the memory, its FTS index, vector embedding, graph data, and orphaned events. Cannot be undone.`;
  $("confirmModal").classList.add('open');
}
function closeModal(){$("confirmModal").classList.remove('open');}

async function doPurge(){
  if(!DETAIL_MID)return;
  const ns=DETAIL_NS||NS;
  if(!ns)return;
  closeModal();
  const r=await j(`/api/namespace/${encodeURIComponent(ns)}/memory/${encodeURIComponent(DETAIL_MID)}`,{method:'DELETE'});
  if(r.ok){
    closeDetail();
    if(!ALL_NS)loadNs(NS);
    if($('view-browse').classList.contains('on'))doBrowse(BROWSE_PAGE);
  }else{
    alert("Delete failed: "+(r.error||"unknown error"));
  }
}

async function doBrowse(page){
  if(!NS||ALL_NS)return;
  BROWSE_PAGE=page;
  const limit=50;
  const params=new URLSearchParams({limit,offset:page*limit});
  const tier=$("bTier").value;if(tier)params.set("tier",tier);
  const origin=$("bOrigin").value;if(origin)params.set("origin",origin);
  const session=$("bSession").value.trim();if(session)params.set("session",session);
  const since=$("bSince").value;if(since)params.set("since",since+"T00:00:00+00:00");
  const until=$("bUntil").value;if(until)params.set("until",until+"T23:59:59+00:00");
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/browse?${params}`);
  const mems=data.memories||[];
  if(!mems.length){$("browseResults").innerHTML='<div class="empty">no memories match</div>';$("browseNav").innerHTML="";return;}
  $("browseResults").innerHTML=`<table><thead><tr><th>created</th><th>tier</th><th>role</th><th>origin</th><th>session</th><th>snippet</th><th></th></tr></thead><tbody>`+
    mems.map(m=>`<tr class="clickable" onclick="openDetail('${esc(m.memory_id)}','${esc(NS)}')">
      <td>${fmtTime(m.created_at)}</td>
      <td class="${tierCls(m.tier)}">${m.tier}</td>
      <td>${m.role||""}</td><td>${m.origin||""}</td>
      <td style="font-size:11px;color:var(--mut)">${(m.session_id||"").slice(0,8)}</td>
      <td class="snip">${esc(snip(m.content||"",160))}</td>
      <td><button style="font-size:11px;padding:2px 8px" onclick="event.stopPropagation();openDetail('${esc(m.memory_id)}','${esc(NS)}')">detail</button></td>
    </tr>`).join("")+"</tbody></table>";
  const total=data.total||0;
  const pages=Math.ceil(total/limit);
  let nav=`<span>${total} memories · page ${page+1}/${pages}</span>`;
  if(page>0)nav+=` <button onclick="doBrowse(${page-1})">← prev</button>`;
  if(page<pages-1)nav+=` <button onclick="doBrowse(${page+1})">next →</button>`;
  $("browseNav").innerHTML=nav;
}

async function loadProcedures(){
  if(!NS||ALL_NS)return;
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/procedures`);
  const procs=data.procedures||[];
  if(!procs.length){$("procList").innerHTML='<div class="empty">no procedures</div>';return;}
  $("procList").innerHTML=`<table><thead><tr><th>name</th><th>trigger</th><th>id</th><th>body</th></tr></thead><tbody>`+
    procs.map(p=>`<tr class="clickable" onclick="openDetail('${esc(p.id)}','${esc(NS)}')">
      <td>${esc(p.name)}</td><td>${esc(p.trigger||"")}</td>
      <td style="font-size:11px;color:var(--mut)">${(p.id||"").slice(0,12)}</td>
      <td class="snip">${esc(snip(p.body||"",200))}</td>
    </tr>`).join("")+"</tbody></table>";
}

async function loadWorldview(){
  if(!NS||ALL_NS)return;
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/worldview`);
  const facts=data.facts||[];
  $("wvFacts").innerHTML=facts.length?facts.map(f=>`<div class="ent"><span>${esc(snip(f.content,200))}</span><span class="ty">${(f.id||"").slice(0,8)}</span></div>`).join(""):'<div class="empty">no semantic facts</div>';
  const names=data.names||[];
  $("wvNames").innerHTML=names.length?names.map(n=>`<div class="ent"><span>${esc(n.name)}</span><span class="ty">${n.type} · ${n.mentions} mentions</span></div>`).join(""):'<div class="empty">no entities</div>';
  const procs=data.procedures||[];
  $("wvProcs").innerHTML=procs.length?procs.map(p=>`<div class="ent"><span>${esc(p.name)}</span><span class="ty">${esc(p.trigger||"")}</span></div>`).join(""):'<div class="empty">no procedures</div>';
}

async function loadHealth(){
  if(!NS||ALL_NS)return;
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/health`);
  const items=[
    ["haunt_home",data.haunt_home],
    ["namespace",data.namespace],
    ["db_path",data.db_path],
    ["sqlite_vec",data.sqlite_vec?.ok?`ok · ${data.sqlite_vec.version}`:`FAIL · ${data.sqlite_vec?.error||"unknown"}`],
    ["embed model",data.embed?.loaded||"none"],
    ["embed dim",data.embed?.dim||0],
    ["embed available",data.embed?.available?"yes":"no"],
    ["embed requested",data.embed?.requested||""],
    ["events",data.stats?.events||0],
    ["memories",data.stats?.memories||0],
    ["sessions",data.stats?.sessions||0],
    ["entities",data.stats?.entities||0],
    ["relations",data.stats?.relations||0],
    ["db size",fmtBytes(data.stats?.db_size_bytes||0)],
    ["last write",fmtTime(data.stats?.last_write)],
  ];
  $("healthDetail").innerHTML=items.map(([k,v])=>`<div class="detail-row"><span class="lbl">${k}</span><span class="val">${v}</span></div>`).join("");
}

async function loadTimeline(){
  if(!NS||ALL_NS)return;
  const since=$("tlSince").value;
  const until=$("tlUntil").value;
  const limit=$("tlLimit").value||200;
  const params=new URLSearchParams({limit});
  if(since)params.set("since",since+"T00:00:00+00:00");
  if(until)params.set("until",until+"T23:59:59+00:00");
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/timeline?${params}`);
  const rows=data.events||[];
  if(!rows.length){$("timelineResults").innerHTML='<div class="empty">no events match</div>';return;}
  $("timelineResults").innerHTML=`<table><thead><tr><th>event_time</th><th>origin</th><th>role</th><th>tier</th><th>snippet</th></tr></thead><tbody>`+
    rows.map(r=>`<tr class="clickable" onclick="openEventMemory('${esc(r.id||"")}')">
      <td>${fmtTime(r.event_time)}</td><td>${r.origin||""}</td>
      <td>${r.role||""}</td>
      <td class="${tierCls(r.tier)}">${r.tier}</td>
      <td class="snip">${esc(snip(r.content||(r.tool_name?"tool:"+r.tool_name:""),180))}</td>
    </tr>`).join("")+"</tbody></table>";
}

function confirmContradict(){
  if(!DETAIL_MID)return;
  $("contradictReplacement").value="";
  $("contradictModal").classList.add('open');
}
function closeContradictModal(){$("contradictModal").classList.remove('open');}

async function doContradict(){
  if(!DETAIL_MID)return;
  const ns=DETAIL_NS||NS;
  if(!ns)return;
  closeContradictModal();
  const replacement=$("contradictReplacement").value.trim()||null;
  const body={};
  if(replacement)body.replacement=replacement;
  const r=await j(`/api/namespace/${encodeURIComponent(ns)}/memory/${encodeURIComponent(DETAIL_MID)}/contradict`,{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)
  });
  if(r.ok){
    closeDetail();
    if(!ALL_NS)loadNs(NS);
    if($('view-browse').classList.contains('on'))doBrowse(BROWSE_PAGE);
    if($('view-timeline').classList.contains('on'))loadTimeline();
  }else{
    alert("Supersede failed: "+(r.error||"unknown error"));
  }
}

async function doRecall(){
  const q=$("q").value.trim();if(!q)return;
  if(!ALL_NS&&!NS)return;
  $("recallMeta").textContent="…";
  const tier=$("sTier").value;
  const asOf=$("sAsOf").value;
  const since=$("sSince").value;
  const until=$("sUntil").value;
  let url,data;
  if(ALL_NS){
    url=`/api/recall?q=${encodeURIComponent(q)}`;
    if(tier)url+=`&tier=${encodeURIComponent(tier)}`;
    if(asOf)url+=`&as_of=${encodeURIComponent(asOf+"T23:59:59+00:00")}`;
    if(since)url+=`&since=${encodeURIComponent(since+"T00:00:00+00:00")}`;
    if(until)url+=`&until=${encodeURIComponent(until+"T23:59:59+00:00")}`;
    data=await j(url);
  }else{
    url=`/api/namespace/${encodeURIComponent(NS)}/recall?q=${encodeURIComponent(q)}`;
    if(tier)url+=`&tier=${encodeURIComponent(tier)}`;
    if(asOf)url+=`&as_of=${encodeURIComponent(asOf+"T23:59:59+00:00")}`;
    if(since)url+=`&since=${encodeURIComponent(since+"T00:00:00+00:00")}`;
    if(until)url+=`&until=${encodeURIComponent(until+"T23:59:59+00:00")}`;
    data=await j(url);
  }
  $("recallMeta").textContent=(data.hits||[]).length+" hits"+(ALL_NS?" (all namespaces)":"");
  hitsTable(data.hits||[]);
}

$("go").onclick=doRecall;
$("q").addEventListener("keydown",e=>{if(e.key==="Enter")doRecall();});

async function selectAllNs(){
  ALL_NS=true;
  NS=null;
  $("title").textContent="all namespaces";
  const boot=await j("/api/namespaces");
  renderNs(boot.namespaces||[],null);
  const h=_health_cache||{};
  renderHealthGlobal(h,{namespace_count:(boot.namespaces||[]).length,haunt_home:boot.haunt_home||''});
  $("stats").innerHTML='<div class="empty">pick a namespace for overview stats</div>';
  $("events").innerHTML='<div class="empty">pick a namespace</div>';
  $("ents").innerHTML='<div class="empty">pick a namespace</div>';
  $("healthStrip").innerHTML='';
  $("healthAge").textContent='';
  switchView('search');
}
let _health_cache={};

async function loadNs(name){
  ALL_NS=false;
  NS=name;
  document.querySelectorAll('.tab-content').forEach(v=>hideAllNsHint(v));
  const data=await j("/api/namespace/"+encodeURIComponent(name));
  $("title").textContent=name;
  $("home").textContent=data.haunt_home||"";
  setPills(data.health||{});
  _health_cache=data.health||{};
  renderNs(data.namespaces||[],name);
  statCards(data.stats||{});
  eventsTable(data.events||[]);
  entsList(data.entities||[]);
  renderHealthStrip(data.health||{},data.stats||{});
}

(async()=>{
  const boot=await j("/api/namespaces");
  $("home").textContent=boot.haunt_home||"";
  const first=boot.default||(boot.namespaces[0]||{}).name||"default";
  await loadNs(first);
})();

setInterval(async()=>{
  try{
    if(ALL_NS){
      const boot=await j("/api/namespaces");
      const h=_health_cache.embed?_health_cache:{embed:{},sqlite_vec:{}};
      renderHealthGlobal(h,{namespace_count:(boot.namespaces||[]).length,haunt_home:boot.haunt_home||''});
    }else if(NS){
      const data=await j(`/api/namespace/${encodeURIComponent(NS)}/health`);
      _health_cache={embed:data.embed,sqlite_vec:data.sqlite_vec};
      renderHealthStrip({embed:data.embed,sqlite_vec:data.sqlite_vec},data.stats||{});
    }
  }catch(e){}
},15000);
</script>
</body>
</html>
"""


def _health_from_store(st: Store) -> dict[str, Any]:
    es = embed_state()
    vec_info: dict[str, Any] = {"ok": st.vec_ok()}
    ver = st.vec_version()
    if ver:
        vec_info["version"] = ver
    return {
        "haunt_home": str(haunt_home()),
        "sqlite_vec": vec_info,
        "embed": {
            "loaded": es.model_id,
            "dim": es.dim,
            "available": es.available,
            "requested": es.requested,
            "fallback": es.fallback,
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

async def index(_request: Request) -> HTMLResponse:
    return HTMLResponse(HTML)


async def api_namespaces(_request: Request) -> JSONResponse:
    ns_list = list_namespaces()
    return JSONResponse({
        "haunt_home": str(haunt_home()),
        "namespaces": ns_list,
        "default": pick_default_namespace(ns_list),
    })


async def api_namespace(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    if not namespace_exists(name):
        pass
    with Store(name) as st:
        stats = st.stats()
        events = st.events(limit=40)
        entities = st.top_entities(20)
        health = _health_from_store(st)
    return JSONResponse(
        {
            "haunt_home": str(haunt_home()),
            "health": health,
            "namespaces": list_namespaces(),
            "stats": stats,
            "events": events,
            "entities": entities,
        }
    )


async def api_recall_all(request: Request) -> JSONResponse:
    """Cross-namespace recall: fan out to every registered namespace, merge via RRF."""
    q = request.query_params.get("q") or ""
    if not q.strip():
        return JSONResponse({"query": q, "hits": []})
    k = int(request.query_params.get("k") or 10)
    tier = request.query_params.get("tier") or None
    as_of = request.query_params.get("as_of") or None
    since = request.query_params.get("since") or None
    until = request.query_params.get("until") or None

    ns_rows = list_namespace_rows()
    all_hits: list[tuple[Hit, str]] = []
    for row in ns_rows:
        ns_name = row["name"]
        try:
            with Store(ns_name, create=False) as st:
                if st.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0:
                    continue
                hits = recall(q, namespace=ns_name, k=k, tier=tier,
                              as_of=as_of, since=since, until=until, store=st)
                for h in hits:
                    all_hits.append((h, ns_name))
        except (FileNotFoundError, Exception):
            continue

    rrf: dict[str, float] = {}
    hit_map: dict[str, tuple[Hit, str]] = {}
    for h, ns_name in all_hits:
        key = f"{ns_name}:{h.memory_id}"
        hit_map[key] = (h, ns_name)
        rrf[key] = h.score

    ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    results = []
    for key, score in ranked:
        h, ns_name = hit_map[key]
        d = h.as_dict()
        d["namespace"] = ns_name
        d["score"] = round(score, 6)
        results.append(d)

    return JSONResponse({"query": q, "hits": results})


async def api_recall(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    q = request.query_params.get("q") or ""
    k = int(request.query_params.get("k") or 8)
    tier = request.query_params.get("tier") or None
    as_of = request.query_params.get("as_of") or None
    since = request.query_params.get("since") or None
    until = request.query_params.get("until") or None
    with Store(name) as st:
        hits = recall(q, namespace=name, k=k, tier=tier,
                      as_of=as_of, since=since, until=until, store=st)
    results = []
    for h in hits:
        d = h.as_dict()
        d["namespace"] = name
        results.append(d)
    return JSONResponse({"query": q, "hits": results})


async def api_browse(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    params = request.query_params
    with Store(name) as st:
        result = st.browse_memories(
            session_id=params.get("session") or None,
            origin=params.get("origin") or None,
            tier=params.get("tier") or None,
            since=params.get("since") or None,
            until=params.get("until") or None,
            limit=int(params.get("limit") or 100),
            offset=int(params.get("offset") or 0),
        )
    return JSONResponse(result)


async def api_memory_detail(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    memory_id = request.path_params["memory_id"]
    with Store(name) as st:
        detail = st.get_memory(memory_id)
    if not detail:
        return JSONResponse({"error": f"memory {memory_id} not found"}, status_code=404)
    return JSONResponse(detail)


async def api_memory_delete(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    memory_id = request.path_params["memory_id"]
    with Store(name) as st:
        result = st.purge(memory_id)
    status = 200 if result.get("ok") else 404
    return JSONResponse(result, status_code=status)


async def api_event_memories(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    event_id = request.path_params["event_id"]
    with Store(name) as st:
        rows = st.conn.execute(
            "SELECT id FROM memories WHERE event_id=? ORDER BY created_at DESC",
            (event_id,),
        ).fetchall()
    mids = [r["id"] for r in rows]
    return JSONResponse({"memories": mids})


async def api_procedures(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    with Store(name) as st:
        procs = st.procedure_list()
    return JSONResponse({"procedures": procs})


async def api_worldview(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    with Store(name) as st:
        wv = st.worldview()
    return JSONResponse(wv)


async def api_health(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    with Store(name) as st:
        health = _health_from_store(st)
        stats = st.stats()
    health["namespace"] = name
    health["db_path"] = stats.get("db_path", "")
    health["stats"] = stats
    return JSONResponse(health)


async def api_timeline(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    params = request.query_params
    since = params.get("since") or None
    until = params.get("until") or None
    limit = int(params.get("limit") or 200)
    with Store(name) as st:
        events = st.events(since=since, until=until, limit=limit)
    return JSONResponse({"namespace": name, "events": events})


async def api_contradict(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    memory_id = request.path_params["memory_id"]
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    replacement = body.get("replacement") or None
    with Store(name) as st:
        result = st.contradict(memory_id, replacement=replacement, origin="dashboard")
    result["namespace"] = name
    status = 200 if result.get("ok") else 404
    return JSONResponse(result, status_code=status)


routes = [
    Route("/", index),
    Route("/api/namespaces", api_namespaces),
    Route("/api/recall", api_recall_all),
    Route("/api/namespace/{name}", api_namespace),
    Route("/api/namespace/{name}/recall", api_recall),
    Route("/api/namespace/{name}/browse", api_browse),
    Route("/api/namespace/{name}/timeline", api_timeline),
    Route("/api/namespace/{name}/memory/{memory_id}", api_memory_detail),
    Route("/api/namespace/{name}/memory/{memory_id}", api_memory_delete, methods=["DELETE"]),
    Route("/api/namespace/{name}/memory/{memory_id}/contradict", api_contradict, methods=["POST"]),
    Route("/api/namespace/{name}/event/{event_id}/memories", api_event_memories),
    Route("/api/namespace/{name}/procedures", api_procedures),
    Route("/api/namespace/{name}/worldview", api_worldview),
    Route("/api/namespace/{name}/health", api_health),
]

app = Starlette(debug=False, routes=routes)


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 7340,
    open_browser: bool = True,
) -> None:
    import threading
    import time
    import socket
    import uvicorn
    import webbrowser

    url = f"http://{host}:{port}"

    if open_browser:
        def _open_when_ready() -> None:
            for _ in range(40):
                try:
                    with socket.create_connection((host, port), timeout=0.5):
                        webbrowser.open(url)
                        return
                except OSError:
                    time.sleep(0.25)

        threading.Thread(target=_open_when_ready, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()
