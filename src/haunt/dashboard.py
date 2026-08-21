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
from haunt.recall import recall
from haunt.store import Store, list_namespaces, namespace_exists

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
    <button class="nav-btn" data-view="browse" onclick="switchView('browse')">browse memories</button>
    <button class="nav-btn" data-view="search" onclick="switchView('search')">search / recall</button>
    <button class="nav-btn" data-view="procedures" onclick="switchView('procedures')">procedures</button>
    <button class="nav-btn" data-view="worldview" onclick="switchView('worldview')">worldview</button>
    <button class="nav-btn" data-view="health" onclick="switchView('health')">health</button>
  </aside>
  <main>
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
          <button class="danger" onclick="confirmPurge()">delete</button>
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
    <h3>delete memory</h3>
    <p id="confirmText">This will permanently delete the memory, its FTS index, vector embedding, and associated graph data. This cannot be undone.</p>
    <div class="actions">
      <button onclick="closeModal()">cancel</button>
      <button class="danger" id="confirmBtn" onclick="doPurge()">delete permanently</button>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let NS=null, DETAIL_MID=null, BROWSE_PAGE=0;

function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmtBytes(n){if(n<1024)return n+" B";if(n<1048576)return(n/1024).toFixed(1)+" KB";return(n/1048576).toFixed(2)+" MB";}
function tierCls(t){return "t-"+(t||"");}
function fmtTime(s){return(s||"—").replace("T"," ").replace(/\+00:00$/,"Z").slice(0,19);}
function snip(s,n){s=(s||"").replace(/\s+/g," ");return s.length<=n?s:s.slice(0,n-1)+"…";}
async function j(url,opts){const r=await fetch(url,opts);if(!r.ok)throw new Error(await r.text());return r.json();}

function switchView(v){
  document.querySelectorAll('.tab-content').forEach(el=>el.classList.remove('on'));
  document.querySelectorAll('.nav-btn').forEach(el=>el.classList.remove('on'));
  $('view-'+v).classList.add('on');
  document.querySelector(`.nav-btn[data-view="${v}"]`).classList.add('on');
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
  $("nsList").innerHTML=list.map(n=>`<div class="ns ${n.name===current?'on':''}" data-ns="${n.name}"><b>${esc(n.name)}</b><i>${n.events}</i></div>`).join("")||`<div class="empty">none</div>`;
  $("nsList").querySelectorAll(".ns").forEach(el=>{el.onclick=()=>loadNs(el.dataset.ns);});
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
  $("hits").innerHTML=`<table><thead><tr><th>#</th><th>score</th><th>tier</th><th>memory_id</th><th>snippet</th><th></th></tr></thead><tbody>`+
    hits.map((h,i)=>`<tr>
      <td>${i+1}</td><td>${(h.score||0).toFixed(4)}</td>
      <td class="${tierCls(h.tier)}">${h.tier}</td>
      <td style="font-size:11px;color:var(--mut)">${(h.memory_id||"").slice(0,12)}</td>
      <td class="snip">${esc(snip(h.content||h.snippet||"",200))}</td>
      <td><button style="font-size:11px;padding:2px 8px" onclick="openDetail('${esc(h.memory_id)}')">detail</button></td>
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
  if(mems.length>0)openDetail(mems[0]);
  else alert("No memories for this event");
}

async function openDetail(memId){
  DETAIL_MID=memId;
  const d=await j(`/api/namespace/${encodeURIComponent(NS)}/memory/${encodeURIComponent(memId)}`);
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
    html+=d.related_memories.map(r=>`<tr class="clickable" onclick="openDetail('${esc(r.memory_id)}')">
      <td style="font-size:11px">${(r.memory_id||"").slice(0,12)}</td>
      <td class="${tierCls(r.tier)}">${r.tier}</td>
      <td class="snip">${esc(snip(r.content||"",160))}</td>
    </tr>`).join("");
    html+="</tbody></table>";
  }
  $("detailBody").innerHTML=html;
}

function closeDetail(){$("detailPanel").classList.remove('open');DETAIL_MID=null;}

function confirmPurge(){
  if(!DETAIL_MID)return;
  $("confirmText").textContent=`Permanently delete memory ${DETAIL_MID.slice(0,12)}…? This removes the memory, its FTS index, vector embedding, graph data, and orphaned events. Cannot be undone.`;
  $("confirmModal").classList.add('open');
}
function closeModal(){$("confirmModal").classList.remove('open');}

async function doPurge(){
  if(!DETAIL_MID||!NS)return;
  closeModal();
  const r=await j(`/api/namespace/${encodeURIComponent(NS)}/memory/${encodeURIComponent(DETAIL_MID)}`,{method:'DELETE'});
  if(r.ok){
    closeDetail();
    loadNs(NS);
    if($('view-browse').classList.contains('on'))doBrowse(BROWSE_PAGE);
  }else{
    alert("Delete failed: "+(r.error||"unknown error"));
  }
}

async function doBrowse(page){
  if(!NS)return;
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
    mems.map(m=>`<tr>
      <td>${fmtTime(m.created_at)}</td>
      <td class="${tierCls(m.tier)}">${m.tier}</td>
      <td>${m.role||""}</td><td>${m.origin||""}</td>
      <td style="font-size:11px;color:var(--mut)">${(m.session_id||"").slice(0,8)}</td>
      <td class="snip">${esc(snip(m.content||"",160))}</td>
      <td><button style="font-size:11px;padding:2px 8px" onclick="openDetail('${esc(m.memory_id)}')">detail</button></td>
    </tr>`).join("")+"</tbody></table>";
  const total=data.total||0;
  const pages=Math.ceil(total/limit);
  let nav=`<span>${total} memories · page ${page+1}/${pages}</span>`;
  if(page>0)nav+=` <button onclick="doBrowse(${page-1})">← prev</button>`;
  if(page<pages-1)nav+=` <button onclick="doBrowse(${page+1})">next →</button>`;
  $("browseNav").innerHTML=nav;
}

async function loadProcedures(){
  if(!NS)return;
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/procedures`);
  const procs=data.procedures||[];
  if(!procs.length){$("procList").innerHTML='<div class="empty">no procedures</div>';return;}
  $("procList").innerHTML=`<table><thead><tr><th>name</th><th>trigger</th><th>id</th><th>body</th></tr></thead><tbody>`+
    procs.map(p=>`<tr class="clickable" onclick="openDetail('${esc(p.id)}')">
      <td>${esc(p.name)}</td><td>${esc(p.trigger||"")}</td>
      <td style="font-size:11px;color:var(--mut)">${(p.id||"").slice(0,12)}</td>
      <td class="snip">${esc(snip(p.body||"",200))}</td>
    </tr>`).join("")+"</tbody></table>";
}

async function loadWorldview(){
  if(!NS)return;
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/worldview`);
  const facts=data.facts||[];
  $("wvFacts").innerHTML=facts.length?facts.map(f=>`<div class="ent"><span>${esc(snip(f.content,200))}</span><span class="ty">${(f.id||"").slice(0,8)}</span></div>`).join(""):'<div class="empty">no semantic facts</div>';
  const names=data.names||[];
  $("wvNames").innerHTML=names.length?names.map(n=>`<div class="ent"><span>${esc(n.name)}</span><span class="ty">${n.type} · ${n.mentions} mentions</span></div>`).join(""):'<div class="empty">no entities</div>';
  const procs=data.procedures||[];
  $("wvProcs").innerHTML=procs.length?procs.map(p=>`<div class="ent"><span>${esc(p.name)}</span><span class="ty">${esc(p.trigger||"")}</span></div>`).join(""):'<div class="empty">no procedures</div>';
}

async function loadHealth(){
  if(!NS)return;
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

async function doRecall(){
  const q=$("q").value.trim();if(!q||!NS)return;
  $("recallMeta").textContent="…";
  const tier=$("sTier").value;
  let url=`/api/namespace/${encodeURIComponent(NS)}/recall?q=${encodeURIComponent(q)}`;
  if(tier)url+=`&tier=${encodeURIComponent(tier)}`;
  const data=await j(url);
  $("recallMeta").textContent=(data.hits||[]).length+" hits";
  hitsTable(data.hits||[]);
}

$("go").onclick=doRecall;
$("q").addEventListener("keydown",e=>{if(e.key==="Enter")doRecall();});

async function loadNs(name){
  NS=name;
  const data=await j("/api/namespace/"+encodeURIComponent(name));
  $("title").textContent=name;
  $("home").textContent=data.haunt_home||"";
  setPills(data.health||{});
  renderNs(data.namespaces||[],name);
  statCards(data.stats||{});
  eventsTable(data.events||[]);
  entsList(data.entities||[]);
  renderHealthStrip(data.health||{},data.stats||{});
}

(async()=>{
  const boot=await j("/api/namespaces");
  $("home").textContent=boot.haunt_home||"";
  const first=(boot.namespaces[0]||{}).name||"default";
  await loadNs(first);
})();

setInterval(async()=>{
  if(!NS)return;
  try{
    const data=await j(`/api/namespace/${encodeURIComponent(NS)}/health`);
    renderHealthStrip({embed:data.embed,sqlite_vec:data.sqlite_vec},data.stats||{});
  }catch(e){}
},15000);
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _resolve_or_404(request: Request) -> str | JSONResponse:
    """Resolve namespace name from the request; return a 404 JSONResponse if it
    does not exist so that read-only GETs never auto-create databases."""
    name = resolve_namespace(request.path_params["name"])
    if not namespace_exists(name):
        return JSONResponse({"error": f"namespace '{name}' not found"}, status_code=404)
    return name


async def index(_request: Request) -> HTMLResponse:
    return HTMLResponse(HTML)


async def api_namespaces(_request: Request) -> JSONResponse:
    return JSONResponse({"haunt_home": str(haunt_home()), "namespaces": list_namespaces()})


async def api_namespace(request: Request) -> JSONResponse:
    result = _resolve_or_404(request)
    if isinstance(result, JSONResponse):
        return result
    name = result
    try:
        with Store(name, create=False) as st:
            stats = st.stats()
            events = st.events(limit=40)
            entities = st.top_entities(20)
    except FileNotFoundError:
        return JSONResponse({"error": f"namespace '{name}' registered but database missing"}, status_code=404)
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
    result = _resolve_or_404(request)
    if isinstance(result, JSONResponse):
        return result
    name = result
    q = request.query_params.get("q") or ""
    k = int(request.query_params.get("k") or 8)
    tier = request.query_params.get("tier") or None
    with Store(name, create=False) as st:
        hits = recall(q, namespace=name, k=k, tier=tier, store=st)
    return JSONResponse({"query": q, "hits": [h.as_dict() for h in hits]})


async def api_browse(request: Request) -> JSONResponse:
    result = _resolve_or_404(request)
    if isinstance(result, JSONResponse):
        return result
    name = result
    params = request.query_params
    with Store(name, create=False) as st:
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
    result = _resolve_or_404(request)
    if isinstance(result, JSONResponse):
        return result
    name = result
    memory_id = request.path_params["memory_id"]
    with Store(name, create=False) as st:
        detail = st.get_memory(memory_id)
    if not detail:
        return JSONResponse({"error": f"memory {memory_id} not found"}, status_code=404)
    return JSONResponse(detail)


async def api_memory_delete(request: Request) -> JSONResponse:
    result = _resolve_or_404(request)
    if isinstance(result, JSONResponse):
        return result
    name = result
    memory_id = request.path_params["memory_id"]
    with Store(name, create=False) as st:
        result = st.purge(memory_id)
    status = 200 if result.get("ok") else 404
    return JSONResponse(result, status_code=status)


async def api_event_memories(request: Request) -> JSONResponse:
    result = _resolve_or_404(request)
    if isinstance(result, JSONResponse):
        return result
    name = result
    event_id = request.path_params["event_id"]
    with Store(name, create=False) as st:
        rows = st.conn.execute(
            "SELECT id FROM memories WHERE event_id=? ORDER BY created_at DESC",
            (event_id,),
        ).fetchall()
    mids = [r["id"] for r in rows]
    return JSONResponse({"memories": mids})


async def api_procedures(request: Request) -> JSONResponse:
    result = _resolve_or_404(request)
    if isinstance(result, JSONResponse):
        return result
    name = result
    with Store(name, create=False) as st:
        procs = st.procedure_list()
    return JSONResponse({"procedures": procs})


async def api_worldview(request: Request) -> JSONResponse:
    result = _resolve_or_404(request)
    if isinstance(result, JSONResponse):
        return result
    name = result
    with Store(name, create=False) as st:
        wv = st.worldview()
    return JSONResponse(wv)


async def api_health(request: Request) -> JSONResponse:
    result = _resolve_or_404(request)
    if isinstance(result, JSONResponse):
        return result
    name = result
    health = _health(name)
    with Store(name, create=False) as st:
        stats = st.stats()
    health["namespace"] = name
    health["db_path"] = stats.get("db_path", "")
    health["stats"] = stats
    return JSONResponse(health)


routes = [
    Route("/", index),
    Route("/api/namespaces", api_namespaces),
    Route("/api/namespace/{name}", api_namespace),
    Route("/api/namespace/{name}/recall", api_recall),
    Route("/api/namespace/{name}/browse", api_browse),
    Route("/api/namespace/{name}/memory/{memory_id}", api_memory_detail),
    Route("/api/namespace/{name}/memory/{memory_id}", api_memory_delete, methods=["DELETE"]),
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
