"""Local-only memory management dashboard. 127.0.0.1. No React build."""

from __future__ import annotations

import hmac
import ipaddress
import json
import secrets
import sys
from typing import Any
from urllib.parse import urlencode, urlparse

from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from haunt.budget import apply_recall_budget
from haunt.embed import state as embed_state
from haunt.paths import haunt_home, resolve_namespace
from haunt.planner import planned_recall
from haunt.portability import (
    MEDIA_TYPE,
    ExportError,
    ImportBundleError,
    ImportConflictError,
    ImportLimitError,
    build_namespace_export,
    canonical_export_bytes,
    import_namespace_bytes,
    resolve_import_limits,
)
from haunt.recall import BACKEND_ERROR_CODE, Hit, execution_metadata, is_retrieval_backend_error
from haunt.store import (
    Store,
    UnknownNamespaceError,
    list_namespaces,
    list_namespace_rows_readonly,
    namespace_exists_readonly,
    open_existing,
    open_existing_readonly,
)
from haunt.temporal import TemporalParseError, compile as compile_temporal
from haunt.util import clamp_limit, iso_or_now, normalize_clock

TOKEN_HEADER = "X-Haunt-Token"
TOKEN_QUERY = "token"
_HTML_TOKEN_PLACEHOLDER = "__HAUNT_LAUNCH_TOKEN_JSON__"
_HTML_NONCE_PLACEHOLDER = "__HAUNT_CSP_NONCE__"
# style-src stays 'unsafe-inline': a nonce cannot cover the template's style=
# attributes, and script-src is where injected markup is actually stopped.
_HTML_CSP = (
    "default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)
# Nothing outside GET / is a document, so no source list needs to be relaxed.
_API_CSP = "default-src 'none'; base-uri 'none'; frame-ancestors 'none'"
_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})

_dash_token: str | None = None
_dash_bind_host: str = "127.0.0.1"
_dash_allow_remote: bool = False


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


# HTML: single-file memory management console

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
    <button class="nav-btn on" data-view="overview" data-act="view">overview</button>
    <button class="nav-btn" data-view="timeline" data-act="view">timeline</button>
    <button class="nav-btn" data-view="browse" data-act="view">browse memories</button>
    <button class="nav-btn" data-view="search" data-act="view">search / recall</button>
    <button class="nav-btn" data-view="procedures" data-act="view">procedures</button>
    <button class="nav-btn" data-view="worldview" data-act="view">worldview</button>
    <button class="nav-btn" data-view="health" data-act="view">health</button>
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
          <button class="primary" data-act="timeline-filter">filter</button>
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
          <button class="primary" data-act="browse-filter">filter</button>
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
          <label title="Include raw tool and task/session residue for audit search"><input id="includeResidue" type="checkbox"/> include residue</label>
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
          <button style="color:var(--amber);border-color:#5c3e00;" data-act="contradict-open">supersede</button>
          <button class="danger" data-act="purge-open">permanently delete</button>
          <button data-act="detail-close">close</button>
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
      <button data-act="purge-cancel">cancel</button>
      <button class="danger" id="confirmBtn" data-act="purge-confirm">delete permanently</button>
    </div>
  </div>
</div>

<!-- SUPERSEDE MODAL -->
<div class="modal-bg" id="contradictModal">
  <div class="modal">
    <h3>supersede memory</h3>
    <p>This marks the memory as superseded (sets valid_to = now). The original data is <b>kept</b> but excluded from current recall. This is NOT a delete.</p>
    <div style="margin-bottom:12px;">
      <label style="font-size:11px;color:var(--mut);font-family:var(--mono);display:block;margin-bottom:4px;">
        <input id="contradictHasReplacement" type="checkbox"/> attach a verbatim replacement
      </label>
      <textarea id="contradictReplacement" style="width:100%;" disabled placeholder="empty and whitespace-only text are intentional"></textarea>
    </div>
    <div class="actions">
      <button data-act="contradict-cancel">cancel</button>
      <button style="background:#3a2800;color:var(--amber);border-color:#5c3e00;" data-act="contradict-confirm">supersede</button>
    </div>
  </div>
</div>

<script nonce="__HAUNT_CSP_NONCE__">
const $=id=>document.getElementById(id);
const HAUNT_TOKEN=__HAUNT_LAUNCH_TOKEN_JSON__;
let NS=null, DETAIL_MID=null, DETAIL_NS=null, BROWSE_PAGE=0, ALL_NS=false;

function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
const TIERS=["episodic","semantic","procedural","coordinate"];
function fmtBytes(n){if(n<1024)return n+" B";if(n<1048576)return(n/1024).toFixed(1)+" KB";return(n/1048576).toFixed(2)+" MB";}
// Stored tier reaches a class attribute; only the four styled tiers may
// produce one, so an unknown value styles nothing instead of escaping it.
function tierCls(t){return TIERS.includes(t)?"t-"+t:"";}
function fmtTime(s){return(s||"—").replace("T"," ").replace(/\+00:00$/,"Z").slice(0,19);}
function snip(s,n){s=(s||"").replace(/\s+/g," ");return s.length<=n?s:s.slice(0,n-1)+"…";}
async function j(url,opts){
  opts=opts||{};
  const headers=Object.assign({},opts.headers||{});
  const tok=HAUNT_TOKEN||new URLSearchParams(location.search).get("token")||"";
  if(tok)headers["X-Haunt-Token"]=tok;
  opts.headers=headers;
  const r=await fetch(url,opts);
  if(!r.ok)throw new Error(await r.text());
  return r.json();
}

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
    `<span class="pill ${v.ok?'ok':'fail'}">vec ${esc(v.version||'off')}</span>`,
    `<span class="pill ${e.available?'ok':'warn'}">${esc(e.loaded||'fts')} · ${esc(e.dim||0)}d</span>`,
    `<span class="pill">verbatim</span>`,
  ].join("");
}

function renderNs(list,current){
  const allOn=ALL_NS;
  let html=`<div class="ns all-ns ${allOn?'on':''}" data-ns="__all__"><b>all namespaces</b><i>${list.reduce((s,n)=>s+(n.events||0),0)}</i></div>`;
  html+=list.map(n=>`<div class="ns ${!allOn&&n.name===current?'on':''}" data-ns="${esc(n.name)}"><b>${esc(n.name)}</b><i>${n.error?'error':esc(n.events)}</i></div>`).join("");
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
      `<div class="health-item"><span class="pulse ${v.ok?'ok':'fail'}"></span>sqlite-vec ${v.ok?esc(v.version):'off'}</div>`,
      `<div class="health-item"><span class="pulse ${e.available?'ok':'warn'}"></span>embed ${esc(e.loaded||'none')} ${esc(e.dim||0)}d</div>`,
      `<div class="health-item"><span class="pulse ok"></span>${esc(stats.namespace_count||0)} namespaces</div>`,
      `<div class="health-item"><span class="pulse ok"></span>${esc(stats.haunt_home||'')}</div>`,
    ];
  }else{
    items=[
      `<div class="health-item"><span class="pulse ${v.ok?'ok':'fail'}"></span>sqlite-vec ${v.ok?esc(v.version):'off'}</div>`,
      `<div class="health-item"><span class="pulse ${e.available?'ok':'warn'}"></span>embed ${esc(e.loaded||'none')} ${esc(e.dim||0)}d</div>`,
      `<div class="health-item"><span class="pulse ok"></span>ns: ${esc(stats.namespace||NS)}</div>`,
      `<div class="health-item"><span class="pulse ok"></span>${esc(fmtBytes(stats.db_size_bytes||0))}</div>`,
      `<div class="health-item"><span class="pulse ${lastWrite?'ok':'warn'}"></span>write ${esc(age)}</div>`,
      `<div class="health-item"><span class="pulse ok"></span>${esc(stats.events||0)} events</div>`,
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
  $("stats").innerHTML=items.map(([k,v,sub])=>`<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(sub)}</div></div>`).join("");
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
    `<div class="health-item"><span class="pulse ${v.ok?'ok':'fail'}"></span>sqlite-vec ${v.ok?esc(v.version):'off'}</div>`,
    `<div class="health-item"><span class="pulse ${e.available?'ok':'warn'}"></span>embed ${esc(e.loaded||'none')} ${esc(e.dim||0)}d</div>`,
    `<div class="health-item"><span class="pulse ok"></span>namespace ${esc(stats.namespace||NS)}</div>`,
    `<div class="health-item"><span class="pulse ok"></span>${esc(fmtBytes(stats.db_size_bytes||0))}</div>`,
    `<div class="health-item"><span class="pulse ${lastWrite?'ok':'warn'}"></span>last write ${esc(age)}</div>`,
    `<div class="health-item"><span class="pulse ok"></span>${esc(stats.events||0)} events</div>`,
    `<div class="health-item"><span class="pulse ok"></span>${esc(stats.db_path||'')}</div>`,
  ].join("");
  renderHealthGlobal(h,stats);
}

function eventsTable(rows){
  if(!rows.length){$("events").innerHTML='<div class="empty">none</div>';return;}
  $("events").innerHTML=`<table><thead><tr><th>event_time</th><th>role</th><th>tier</th><th>origin</th><th>snippet</th></tr></thead><tbody>`+
    rows.map(r=>`<tr class="clickable" data-act="event" data-eid="${esc(r.id||"")}">
      <td>${esc(fmtTime(r.event_time))}</td><td>${esc(r.role||"")}</td>
      <td class="${tierCls(r.tier)}">${esc(r.tier||"")}</td><td>${esc(r.origin||"")}</td>
      <td class="snip">${esc(snip(r.content||(r.tool_name?"tool:"+r.tool_name:""),180))}</td>
    </tr>`).join("")+"</tbody></table>";
}

function hitsTable(hits){
  if(!hits.length){$("hits").innerHTML='<div class="empty">no hits</div>';return;}
  $("hits").innerHTML=`<table><thead><tr><th>rank</th><th title="RRF rank signal for ranked hits; timeline hits are time-ordered">signal</th><th>tier</th><th>origin</th>${ALL_NS?'<th>namespace</th>':''}<th>memory_id</th><th>snippet</th><th></th></tr></thead><tbody>`+
    hits.map((h,i)=>`<tr class="clickable" data-act="detail" data-mid="${esc(h.memory_id)}" data-ns="${esc(h.namespace||NS)}">
      <td>${esc(h.explanation?.final_rank??i+1)}</td><td>${h.explanation?.score_semantics==='rrf_rank_signal_not_confidence'?'rrf='+esc((h.score||0).toFixed(4)):'time-order'}</td>
      <td class="${tierCls(h.tier)}">${esc(h.tier||"")}</td>
      <td style="font-size:11px;color:var(--mut)">${esc(h.origin||'')}</td>
      ${ALL_NS?`<td><span class="ns-badge">${esc(h.namespace||'')}</span></td>`:''}
      <td style="font-size:11px;color:var(--mut)">${esc((h.memory_id||"").slice(0,12))}</td>
      <td class="snip">${esc(snip(h.content||h.snippet||"",200))}</td>
      <td><button style="font-size:11px;padding:2px 8px" data-act="detail" data-mid="${esc(h.memory_id)}" data-ns="${esc(h.namespace||NS)}">detail</button></td>
    </tr>`).join("")+"</tbody></table>";
}

function entsList(ents){
  if(!ents.length){$("ents").innerHTML='<div class="empty">none</div>';return;}
  $("ents").innerHTML=ents.map(e=>`<div class="ent"><span>${esc(e.name)}</span><span class="ty">${esc(e.type)}</span></div>`).join("");
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
    ["memory_id",esc(d.memory_id||"")],["event_id",esc(d.event_id||"")],["session_id",esc(d.session_id||"")],
    ["namespace",esc(d.namespace||"")],["tier",`<span class="${esc(tierCls(d.tier))}">${esc(d.tier||"")}</span>`],
    ["role",esc(d.role||"")],["origin",esc(d.origin||"")],
    ["event_time",esc(fmtTime(d.event_time))],["valid_from",esc(fmtTime(d.valid_from))],
    ["valid_to",d.valid_to?esc(fmtTime(d.valid_to)):"<em>current</em>"],
    ["created_at",esc(fmtTime(d.created_at))],
    ["has_embedding",esc(d.has_embedding?"yes":"no")],
    ["db_path",esc(d.db_path||"")],["haunt_home",esc(d.haunt_home||"")],
  ];
  if(d.tool_name)rows.push(["tool_name",esc(d.tool_name)]);
  let html=rows.map(([l,v])=>`<div class="detail-row"><span class="lbl">${l}</span><span class="val">${v}</span></div>`).join("");
  html+=`<h2 class="section" style="margin-top:12px;">source provenance</h2><div class="detail-content">${esc(JSON.stringify(d.provenance||{},null,2))}</div>`;
  html+=`<h2 class="section" style="margin-top:12px;">content</h2><div class="detail-content">${esc(d.content||d.event_content||"(empty)")}</div>`;
  if(d.tool_input)html+=`<h2 class="section">tool input</h2><div class="detail-content">${esc(d.tool_input)}</div>`;
  if(d.tool_output)html+=`<h2 class="section">tool output</h2><div class="detail-content">${esc(d.tool_output)}</div>`;
  if(d.entity_mentions&&d.entity_mentions.length){
    html+=`<h2 class="section" style="margin-top:12px;">entity mentions (${d.entity_mentions.length})</h2>`;
    html+=d.entity_mentions.map(e=>`<div class="ent"><span>${esc(e.name)}</span><span class="ty">${esc(e.type)}</span></div>`).join("");
  }
  if(d.related_memories&&d.related_memories.length){
    html+=`<h2 class="section" style="margin-top:12px;">related memories (same session)</h2>`;
    html+=`<table><thead><tr><th>id</th><th>tier</th><th>snippet</th></tr></thead><tbody>`;
    html+=d.related_memories.map(r=>`<tr class="clickable" data-act="detail" data-mid="${esc(r.memory_id)}" data-ns="${esc(ns)}">
      <td style="font-size:11px">${esc((r.memory_id||"").slice(0,12))}</td>
      <td class="${esc(tierCls(r.tier))}">${esc(r.tier||"")}</td>
      <td class="snip">${esc(snip(r.content||"",160))}</td>
    </tr>`).join("");
    html+="</tbody></table>";
  }
  if(d.trace&&d.trace.members){
    html+=`<h2 class="section" style="margin-top:12px;">correction lineage (${esc(d.trace.lineage_status||"")})</h2>`;
    html+=`<div class="detail-content">${esc(JSON.stringify(d.trace,null,2))}</div>`;
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
  $("browseResults").innerHTML=`<table><thead><tr><th>created</th><th>tier</th><th>role</th><th>origin</th><th>source</th><th>session</th><th>snippet</th><th></th></tr></thead><tbody>`+
    mems.map(m=>`<tr class="clickable" data-act="detail" data-mid="${esc(m.memory_id)}" data-ns="${esc(NS)}">
      <td>${esc(fmtTime(m.created_at))}</td>
      <td class="${tierCls(m.tier)}">${esc(m.tier||"")}</td>
      <td>${esc(m.role||"")}</td><td>${esc(m.origin||"")}</td>
      <td>${esc((m.provenance||{}).kind||"unknown")}</td>
      <td style="font-size:11px;color:var(--mut)">${esc((m.session_id||"").slice(0,8))}</td>
      <td class="snip">${esc(snip(m.content||"",160))}</td>
      <td><button style="font-size:11px;padding:2px 8px" data-act="detail" data-mid="${esc(m.memory_id)}" data-ns="${esc(NS)}">detail</button></td>
    </tr>`).join("")+"</tbody></table>";
  const total=data.total||0;
  const pages=Math.ceil(total/limit);
  let nav=`<span>${esc(total)} memories · page ${page+1}/${pages}</span>`;
  if(page>0)nav+=` <button data-act="page" data-page="${page-1}">← prev</button>`;
  if(page<pages-1)nav+=` <button data-act="page" data-page="${page+1}">next →</button>`;
  $("browseNav").innerHTML=nav;
}

async function loadProcedures(){
  if(!NS||ALL_NS)return;
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/procedures`);
  const procs=data.procedures||[];
  if(!procs.length){$("procList").innerHTML='<div class="empty">no procedures</div>';return;}
  $("procList").innerHTML=`<table><thead><tr><th>name</th><th>trigger</th><th>id</th><th>body</th></tr></thead><tbody>`+
    procs.map(p=>`<tr class="clickable" data-act="detail" data-mid="${esc(p.id)}" data-ns="${esc(NS)}">
      <td>${esc(p.name)}</td><td>${esc(p.trigger||"")}</td>
      <td style="font-size:11px;color:var(--mut)">${esc((p.id||"").slice(0,12))}</td>
      <td class="snip">${esc(snip(p.body||"",200))}</td>
    </tr>`).join("")+"</tbody></table>";
}

async function loadWorldview(){
  if(!NS||ALL_NS)return;
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/worldview`);
  const facts=data.facts||[];
  $("wvFacts").innerHTML=facts.length?facts.map(f=>`<div class="ent"><span>${esc(snip(f.content,200))}</span><span class="ty">${esc((f.id||"").slice(0,8))}</span></div>`).join(""):'<div class="empty">no semantic facts</div>';
  const names=data.names||[];
  $("wvNames").innerHTML=names.length?names.map(n=>`<div class="ent"><span>${esc(n.name)}</span><span class="ty">${esc(n.type)} · ${esc(n.mentions)} mentions</span></div>`).join(""):'<div class="empty">no entities</div>';
  const procs=data.procedures||[];
  $("wvProcs").innerHTML=procs.length?procs.map(p=>`<div class="ent"><span>${esc(p.name)}</span><span class="ty">${esc(p.trigger||"")}</span></div>`).join(""):'<div class="empty">no procedures</div>';
}

async function loadHealth(){
  if(!NS||ALL_NS)return;
  const data=await j(`/api/namespace/${encodeURIComponent(NS)}/health`);
  const items=[
    ["haunt_home",esc(data.haunt_home||"")],
    ["namespace",esc(data.namespace||"")],
    ["db_path",esc(data.db_path||"")],
    ["sqlite_vec",esc(data.sqlite_vec?.ok?`ok · ${data.sqlite_vec.version}`:`FAIL · ${data.sqlite_vec?.error||"unknown"}`)],
    ["embed model",esc(data.embed?.loaded||"none")],
    ["embed dim",esc(data.embed?.dim||0)],
    ["embed available",esc(data.embed?.available?"yes":"no")],
    ["embed requested",esc(data.embed?.requested||"")],
    ["events",esc(data.stats?.events||0)],
    ["memories",esc(data.stats?.memories||0)],
    ["sessions",esc(data.stats?.sessions||0)],
    ["entities",esc(data.stats?.entities||0)],
    ["relations",esc(data.stats?.relations||0)],
    ["db size",esc(fmtBytes(data.stats?.db_size_bytes||0))],
    ["last write",esc(fmtTime(data.stats?.last_write))],
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
    rows.map(r=>`<tr class="clickable" data-act="event" data-eid="${esc(r.id||"")}">
      <td>${esc(fmtTime(r.event_time))}</td><td>${esc(r.origin||"")}</td>
      <td>${esc(r.role||"")}</td>
      <td class="${tierCls(r.tier)}">${esc(r.tier||"")}</td>
      <td class="snip">${esc(snip(r.content||(r.tool_name?"tool:"+r.tool_name:""),180))}</td>
    </tr>`).join("")+"</tbody></table>";
}

function confirmContradict(){
  if(!DETAIL_MID)return;
  $("contradictHasReplacement").checked=false;
  $("contradictReplacement").value="";
  $("contradictReplacement").disabled=true;
  $("contradictModal").classList.add('open');
}
function closeContradictModal(){$("contradictModal").classList.remove('open');}
function toggleContradictReplacement(){
  $("contradictReplacement").disabled=!$("contradictHasReplacement").checked;
}

async function doContradict(){
  if(!DETAIL_MID)return;
  const ns=DETAIL_NS||NS;
  if(!ns)return;
  closeContradictModal();
  const body={idempotency_key:crypto.randomUUID()};
  if($("contradictHasReplacement").checked){
    body.replacement=$("contradictReplacement").value;
  }
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
  const includeResidue=$("includeResidue").checked;
  let url,data;
  if(ALL_NS){
    url=`/api/recall?q=${encodeURIComponent(q)}`;
    if(tier)url+=`&tier=${encodeURIComponent(tier)}`;
    if(asOf)url+=`&as_of=${encodeURIComponent(asOf+"T23:59:59+00:00")}`;
    if(since)url+=`&since=${encodeURIComponent(since+"T00:00:00+00:00")}`;
    if(until)url+=`&until=${encodeURIComponent(until+"T23:59:59+00:00")}`;
    if(includeResidue)url+="&include_residue=true";
    data=await j(url);
  }else{
    url=`/api/namespace/${encodeURIComponent(NS)}/recall?q=${encodeURIComponent(q)}`;
    if(tier)url+=`&tier=${encodeURIComponent(tier)}`;
    if(asOf)url+=`&as_of=${encodeURIComponent(asOf+"T23:59:59+00:00")}`;
    if(since)url+=`&since=${encodeURIComponent(since+"T00:00:00+00:00")}`;
    if(until)url+=`&until=${encodeURIComponent(until+"T23:59:59+00:00")}`;
    if(includeResidue)url+="&include_residue=true";
    data=await j(url);
  }
  const errs=data.errors||[];
  let meta=data.ok===false?(data.error||"recall failed"):(data.hits||[]).length+" hits"+(ALL_NS?" (all namespaces; ranked per namespace)":"");
  if(errs.length){
    const names=errs.map(e=>e.namespace||"?").join(", ");
    meta+=" — "+errs.length+" namespace"+(errs.length===1?"":"s")+" failed ("+names+")";
  }
  $("recallMeta").textContent=meta;
  $("recallMeta").style.color=(errs.length||data.ok===false)?"var(--red)":"";
  hitsTable(data.hits||[]);
}

$("go").onclick=doRecall;
$("q").addEventListener("keydown",e=>{if(e.key==="Enter")doRecall();});
$("contradictHasReplacement").addEventListener("change",toggleContradictReplacement);

// Delegated dispatch, not onclick="…": an inline handler would need
// script-src 'unsafe-inline', and esc()'s &#39; is decoded by the HTML parser
// before the JS parser sees it, so it never protected a handler argument.
const ACTIONS={
  view:el=>switchView(el.dataset.view),
  detail:el=>openDetail(el.dataset.mid,el.dataset.ns),
  event:el=>openEventMemory(el.dataset.eid),
  page:el=>doBrowse(Number(el.dataset.page)),
  "timeline-filter":()=>loadTimeline(),
  "browse-filter":()=>doBrowse(0),
  "detail-close":()=>closeDetail(),
  "purge-open":()=>confirmPurge(),
  "purge-cancel":()=>closeModal(),
  "purge-confirm":()=>doPurge(),
  "contradict-open":()=>confirmContradict(),
  "contradict-cancel":()=>closeContradictModal(),
  "contradict-confirm":()=>doContradict(),
};
document.addEventListener("click",ev=>{
  const el=ev.target.closest("[data-act]");
  if(!el)return;
  const run=ACTIONS[el.dataset.act];
  if(run)run(el);
});

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


# Routes

async def index(_request: Request) -> HTMLResponse:
    token = (_dash_token or "") if embed_launch_token_in_html() else ""
    token_json = (
        json.dumps(token)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    nonce = secrets.token_urlsafe(16)
    body = HTML.replace(_HTML_TOKEN_PLACEHOLDER, token_json).replace(
        _HTML_NONCE_PLACEHOLDER, nonce
    )
    return HTMLResponse(
        body, headers={"Content-Security-Policy": _HTML_CSP.format(nonce=nonce)}
    )


async def api_namespaces(_request: Request) -> JSONResponse:
    ns_list = list_namespaces()
    return JSONResponse({
        "haunt_home": str(haunt_home()),
        "namespaces": ns_list,
        "default": pick_default_namespace(ns_list),
    })


def _missing_namespace(name: str) -> JSONResponse | None:
    if namespace_exists_readonly(name):
        return None
    return JSONResponse(
        {"ok": False, "error": f"unknown namespace: {name}", "namespace": name},
        status_code=404,
    )


def _recall_error(
    exc: Exception,
    *,
    query: str,
    namespace: str | None = None,
    status_code: int = 400,
) -> JSONResponse:
    """Use the MCP-style error envelope for dashboard recall endpoints."""
    payload: dict[str, Any] = {
        "ok": False,
        "code": (
            BACKEND_ERROR_CODE
            if is_retrieval_backend_error(exc)
            else "invalid_recall_request"
        ),
        "error": str(exc),
        "query": query,
    }
    if namespace is not None:
        payload["namespace"] = namespace
    return JSONResponse(payload, status_code=status_code)


def _validate_recall_request(
    query: str,
    *,
    as_of: str | None,
    since: str | None,
    until: str | None,
    clock: str | None,
) -> None:
    """Reject malformed temporal/filter inputs before a namespace fan-out.

    The planner performs the same validation during a normal recall. Doing it
    here makes all-namespace requests atomic with respect to bad input instead
    of returning a misleading collection of per-namespace failures.
    """
    if clock is not None:
        normalize_clock(clock)
    for value in (as_of, since, until):
        if value:
            iso_or_now(value)
    compile_temporal(query)


def _local_recall_order(hit: Hit) -> tuple[int, int, float, str]:
    """Order only within one namespace; never compare RRF across namespaces."""
    if hit.final_rank is not None:
        return (0, hit.final_rank, 0.0, hit.memory_id)
    # Defensive ordering for synthetic/custom callers that did not set a rank.
    # It stays local to the namespace and does not rewrite that hit's rank.
    return (1, 0, -hit.score, hit.memory_id)


async def api_namespace(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    missing = _missing_namespace(name)
    if missing:
        return missing
    with Store(name, create=False) as st:
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
    """Recall each namespace independently; RRF values are not global scores."""
    q = request.query_params.get("q") or ""
    k = clamp_limit(request.query_params.get("k") or 10, default=10)
    tier = request.query_params.get("tier") or None
    as_of = request.query_params.get("as_of") or None
    since = request.query_params.get("since") or None
    until = request.query_params.get("until") or None
    clock = request.query_params.get("clock") or None
    include_residue = (request.query_params.get("include_residue") or "").lower() in {
        "1", "true", "yes"
    }

    try:
        _validate_recall_request(
            q, as_of=as_of, since=since, until=until, clock=clock
        )
    except (TemporalParseError, ValueError) as exc:
        return _recall_error(exc, query=q)
    ns_rows = sorted(list_namespace_rows_readonly(), key=lambda row: row["name"])
    namespace_groups: list[dict[str, Any]] = []
    flattened: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in ns_rows:
        ns_name = row["name"]
        # Every registered namespace gets a deterministic group, even if it
        # has no events. A corrupt namespace cannot honestly expose execution
        # evidence, so its group carries the same structured error as errors.
        group: dict[str, Any] = {"namespace": ns_name, "hits": []}
        try:
            with open_existing_readonly(ns_name) as st:
                hits = planned_recall(
                    q,
                    namespace=ns_name,
                    k=k,
                    tier=tier,
                    as_of=as_of,
                    since=since,
                    until=until,
                    clock=clock,
                    store=st,
                    include_residue=include_residue,
                )
                results: list[dict[str, Any]] = []
                for h in sorted(hits, key=_local_recall_order):
                    result = h.as_dict()
                    result["namespace"] = ns_name
                    results.append(result)
                group["hits"] = results
                execution = execution_metadata(hits)
                if execution is not None:
                    group["execution"] = execution
                flattened.extend(results)
        except Exception as exc:
            error = {
                "namespace": ns_name,
                "code": (
                    BACKEND_ERROR_CODE
                    if is_retrieval_backend_error(exc)
                    else "retrieval_namespace_error"
                ),
                "error": str(exc),
            }
            errors.append(error)
            group["error"] = error
        namespace_groups.append(group)

    # One budget for the whole fan-out rather than one per namespace: this
    # endpoint answers from every registered namespace at once, so a
    # per-namespace cap would multiply by however many exist. Groups are
    # rebuilt from what survived so a group can never disagree with `hits`.
    bounded_hits, recall_budget = apply_recall_budget(flattened, k=k)
    kept: dict[str, list[dict[str, Any]]] = {}
    for hit in bounded_hits:
        kept.setdefault(str(hit.get("namespace")), []).append(hit)
    for group in namespace_groups:
        group["hits"] = kept.get(group["namespace"], [])

    return JSONResponse(
        {
            "query": q,
            "ranking_scope": "per_namespace",
            "k_per_namespace": k,
            "namespace_groups": namespace_groups,
            # Kept for the existing UI/API shape. This is namespace-grouped,
            # not a global result ranking, and each final_rank remains local.
            "hits": bounded_hits,
            "recall_budget": recall_budget,
            "errors": errors,
        }
    )


async def api_recall(request: Request) -> JSONResponse:
    q = request.query_params.get("q") or ""
    try:
        name = resolve_namespace(request.path_params["name"])
    except ValueError as exc:
        return _recall_error(exc, query=q)
    missing = _missing_namespace(name)
    if missing:
        return missing
    k = clamp_limit(request.query_params.get("k") or 8, default=8)
    tier = request.query_params.get("tier") or None
    as_of = request.query_params.get("as_of") or None
    since = request.query_params.get("since") or None
    until = request.query_params.get("until") or None
    clock = request.query_params.get("clock") or None
    include_residue = (request.query_params.get("include_residue") or "").lower() in {
        "1", "true", "yes"
    }
    try:
        _validate_recall_request(
            q, as_of=as_of, since=since, until=until, clock=clock
        )
        with open_existing_readonly(name) as st:
            hits = planned_recall(
                q,
                namespace=name,
                k=k,
                tier=tier,
                as_of=as_of,
                since=since,
                until=until,
                clock=clock,
                store=st,
                include_residue=include_residue,
            )
    except (TemporalParseError, ValueError) as exc:
        return _recall_error(exc, query=q, namespace=name)
    except Exception as exc:
        return _recall_error(exc, query=q, namespace=name, status_code=500)
    results = []
    for h in hits:
        d = h.as_dict()
        d["namespace"] = name
        results.append(d)
    bounded_hits, recall_budget = apply_recall_budget(results, k=k)
    payload: dict[str, Any] = {
        "query": q,
        "namespace": name,
        "ranking_scope": "namespace",
        "hits": bounded_hits,
        "recall_budget": recall_budget,
    }
    execution = execution_metadata(hits)
    if execution is not None:
        payload["execution"] = execution
    return JSONResponse(payload)


async def api_browse(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    missing = _missing_namespace(name)
    if missing:
        return missing
    params = request.query_params
    with Store(name, create=False) as st:
        result = st.browse_memories(
            session_id=params.get("session") or None,
            origin=params.get("origin") or None,
            tier=params.get("tier") or None,
            since=params.get("since") or None,
            until=params.get("until") or None,
            limit=clamp_limit(params.get("limit") or 100, default=100),
            offset=max(0, int(params.get("offset") or 0)),
        )
    return JSONResponse(result)


async def api_memory_detail(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    missing = _missing_namespace(name)
    if missing:
        return missing
    memory_id = request.path_params["memory_id"]
    with Store(name, create=False) as st:
        detail = st.get_memory(memory_id)
    if not detail:
        return JSONResponse({"error": "memory not found"}, status_code=404)
    return JSONResponse(detail)


async def api_memory_delete(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    missing = _missing_namespace(name)
    if missing:
        return missing
    memory_id = request.path_params["memory_id"]
    with Store(name, create=False) as st:
        result = st.purge(memory_id)
    status = 200 if result.get("ok") else 404
    return JSONResponse(result, status_code=status)


async def api_event_memories(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    missing = _missing_namespace(name)
    if missing:
        return missing
    event_id = request.path_params["event_id"]
    with Store(name, create=False) as st:
        rows = st.conn.execute(
            "SELECT id FROM memories WHERE event_id=? ORDER BY created_at DESC, rowid DESC",
            (event_id,),
        ).fetchall()
    mids = [r["id"] for r in rows]
    return JSONResponse({"memories": mids})


async def api_procedures(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    missing = _missing_namespace(name)
    if missing:
        return missing
    with Store(name, create=False) as st:
        procs = st.procedure_list()
    return JSONResponse({"procedures": procs})


async def api_worldview(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    missing = _missing_namespace(name)
    if missing:
        return missing
    with Store(name, create=False) as st:
        wv = st.worldview()
    return JSONResponse(wv)


async def api_health(request: Request) -> JSONResponse:
    name = resolve_namespace(request.path_params["name"])
    missing = _missing_namespace(name)
    if missing:
        return missing
    with Store(name, create=False) as st:
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
    clock = params.get("clock") or None
    limit = clamp_limit(params.get("limit") or 100, default=100)
    try:
        with open_existing(name) as st:
            events = st.events(
                since=since,
                until=until,
                clock=clock,
                limit=limit,
            )
    except (UnknownNamespaceError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc), "namespace": name},
            status_code=400,
        )
    return JSONResponse({"namespace": name, "events": events})


async def api_namespace_export(request: Request) -> Response:
    """Authenticated download; the read path never creates a namespace."""
    name = resolve_namespace(request.path_params["name"])
    missing = _missing_namespace(name)
    if missing:
        return missing
    try:
        bundle = build_namespace_export(
            name, cut=request.query_params.get("cut") or None
        )
        raw = canonical_export_bytes(bundle)
    except (ExportError, UnknownNamespaceError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc), "namespace": name},
            status_code=400,
        )
    # Canonical labels are already safe-name normalized, but strip header
    # metacharacters again at the response boundary. Never reflect raw path
    # parameters into Content-Disposition.
    canonical = str(bundle["namespace"]["canonical_label"])
    filename = "".join(
        char for char in canonical if char.isalnum() or char in "-_."
    )[:80] or "namespace"
    return Response(
        raw,
        media_type=MEDIA_TYPE.split(";", 1)[0],
        headers={
            "Content-Disposition": f'attachment; filename="{filename}.haunt.json"',
            "X-Haunt-Semantic-Digest": str(bundle["manifest"]["semantic_digest"]),
            "Cache-Control": "no-store",
        },
    )


def _dashboard_import_limits(request: Request):
    params = request.query_params

    def integer(name: str) -> int | None:
        value = params.get(name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError as exc:
            raise ImportLimitError(f"{name} must be a positive integer") from exc

    timeout_raw = params.get("timeout_seconds")
    try:
        timeout = None if timeout_raw is None else float(timeout_raw)
    except ValueError as exc:
        raise ImportLimitError("timeout_seconds must be finite and positive") from exc
    return resolve_import_limits(
        input_bytes=integer("input_bytes"),
        decompressed_bytes=integer("decompressed_bytes"),
        records=integer("records"),
        record_bytes=integer("record_bytes"),
        json_depth=integer("json_depth"),
        collection_items=integer("collection_items"),
        timeout_seconds=timeout,
    )


async def api_namespace_import(request: Request) -> JSONResponse:
    """Launch-token admin mutation with strict Origin, media type, and size."""
    origin = request.headers.get("origin")
    if not origin or not request_origin_is_trusted(origin, request):
        return JSONResponse(
            {"ok": False, "error": "trusted Origin header is required"},
            status_code=403,
        )
    content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if content_type not in {
        "application/json",
        MEDIA_TYPE.split(";", 1)[0].lower(),
    }:
        return JSONResponse(
            {"ok": False, "error": "content-type must be a Haunt namespace JSON bundle"},
            status_code=415,
        )
    try:
        limits = _dashboard_import_limits(request)
    except ImportLimitError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limits.input_bytes:
                raise ImportLimitError(
                    f"declared input bytes exceed {limits.input_bytes}"
                )
        except ImportLimitError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=413)
        except ValueError:
            return JSONResponse(
                {"ok": False, "error": "invalid content-length"}, status_code=400
            )
    chunks: list[bytes] = []
    actual = 0
    async for chunk in request.stream():
        actual += len(chunk)
        if actual > limits.input_bytes:
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"actual input bytes exceed {limits.input_bytes}",
                },
                status_code=413,
            )
        chunks.append(bytes(chunk))
    try:
        report = import_namespace_bytes(b"".join(chunks), limits=limits)
    except ImportConflictError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    except ImportLimitError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=413)
    except ImportBundleError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse({"ok": True, **report})


async def api_contradict(request: Request) -> JSONResponse:
    ct = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if ct != "application/json":
        return JSONResponse(
            {"error": "content-type must be application/json"},
            status_code=415,
        )
    try:
        body = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON body must be an object"}, status_code=400)
    if "replacement" in body:
        replacement = body["replacement"]
        if replacement is not None and not isinstance(replacement, str):
            return JSONResponse(
                {"error": "replacement must be a string or null"},
                status_code=400,
            )
    else:
        replacement = None
    reason = body.get("reason")
    if reason is not None and not isinstance(reason, str):
        return JSONResponse({"error": "reason must be a string or null"}, status_code=400)
    if "idempotency_key" not in body:
        return JSONResponse(
            {"error": "idempotency_key is required"}, status_code=400
        )
    idempotency_key = body["idempotency_key"]
    if not isinstance(idempotency_key, str):
        return JSONResponse(
            {"error": "idempotency_key must be a string"}, status_code=400
        )
    session_id = body.get("session_id")

    name = resolve_namespace(request.path_params["name"])
    missing = _missing_namespace(name)
    if missing:
        return missing
    memory_id = request.path_params["memory_id"]
    try:
        with Store(name, create=False) as st:
            result = st.contradict(
                memory_id,
                replacement=replacement,
                origin="dashboard",
                session_id=session_id,
                reason=reason,
                idempotency_key=idempotency_key,
                channel="dashboard",
            )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    result["namespace"] = name
    if result.get("ok"):
        status = 200
    elif "not found" in (result.get("error") or ""):
        status = 404
    elif result.get("conflict") in {
        "idempotency_key_reused",
        "already_superseded",
    }:
        status = 409
    else:
        status = 200
    return JSONResponse(result, status_code=status)


routes = [
    Route("/", index),
    Route("/api/namespaces", api_namespaces),
    Route("/api/recall", api_recall_all),
    Route("/api/namespace/{name}", api_namespace),
    Route("/api/namespace/{name}/recall", api_recall),
    Route("/api/namespace/{name}/browse", api_browse),
    Route("/api/namespace/{name}/timeline", api_timeline),
    Route("/api/namespace/{name}/export", api_namespace_export),
    Route("/api/import", api_namespace_import, methods=["POST"]),
    Route("/api/namespace/{name}/memory/{memory_id}", api_memory_detail),
    Route("/api/namespace/{name}/memory/{memory_id}", api_memory_delete, methods=["DELETE"]),
    Route("/api/namespace/{name}/memory/{memory_id}/contradict", api_contradict, methods=["POST"]),
    Route("/api/namespace/{name}/event/{event_id}/memories", api_event_memories),
    Route("/api/namespace/{name}/procedures", api_procedures),
    Route("/api/namespace/{name}/worldview", api_worldview),
    Route("/api/namespace/{name}/health", api_health),
]


def normalize_host_header(host: str) -> str:
    """Strip port and brackets from a Host header or bind address."""
    h = (host or "").strip().lower()
    if not h:
        return ""
    if h.startswith("["):
        end = h.find("]")
        if end != -1:
            return h[1:end]
        return h.strip("[]")
    if h.count(":") == 1:
        return h.rsplit(":", 1)[0]
    return h


def configure_dashboard_security(
    *,
    token: str | None,
    bind_host: str = "127.0.0.1",
    allow_remote: bool = False,
) -> None:
    """Set Host/token policy for this dashboard process (or a test)."""
    global _dash_token, _dash_bind_host, _dash_allow_remote
    _dash_token = token
    _dash_bind_host = bind_host
    _dash_allow_remote = allow_remote


def reset_dashboard_security() -> None:
    configure_dashboard_security(token=None, bind_host="127.0.0.1", allow_remote=False)


def dashboard_token() -> str | None:
    return _dash_token


def embed_launch_token_in_html() -> bool:
    """Publish the token in GET / only for a loopback, non-remote bind.

    ``--allow-remote`` / a non-loopback bind must not put X-Haunt-Token in the
    unauthenticated HTML. The operator still sees it on haunt dash stdout.
    """
    if _dash_allow_remote:
        return False
    return is_loopback_host(normalize_host_header(_dash_bind_host))


def mint_dashboard_token() -> str:
    return secrets.token_urlsafe(32)


def host_name_is_trusted(name: str) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return False
    if n in _LOOPBACK_NAMES or is_loopback_host(n):
        return True
    bind = normalize_host_header(_dash_bind_host)
    try:
        bind_ip = ipaddress.ip_address(bind)
    except ValueError:
        bind_ip = None
    if bind_ip is not None and bind_ip.is_unspecified:
        try:
            ipaddress.ip_address(n)
            return True
        except ValueError:
            return False
    return bool(bind) and n == bind


def request_host_is_trusted(host_header: str) -> bool:
    return host_name_is_trusted(normalize_host_header(host_header))


def _effective_port(scheme: str, port: int | None) -> int | None:
    if port is not None:
        return port
    return {"http": 80, "https": 443}.get(scheme)


def request_origin_is_trusted(origin: str, request: Request) -> bool:
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    if not host:
        return False
    request_host = request.url.hostname
    if not request_host or not host_name_is_trusted(host.lower()):
        return False
    return (
        parsed.scheme == request.url.scheme
        and host.lower() == request_host.lower()
        and _effective_port(parsed.scheme, parsed.port)
        == _effective_port(request.url.scheme, request.url.port)
    )


def _tokens_match(provided: str, expected: str) -> bool:
    a = provided.encode("utf-8")
    b = expected.encode("utf-8")
    if len(a) != len(b):
        return False
    return hmac.compare_digest(a, b)


def request_token(request: Request) -> str | None:
    header = request.headers.get(TOKEN_HEADER)
    if header:
        return header
    return request.query_params.get(TOKEN_QUERY)


def guard_dashboard_request(request: Request) -> JSONResponse | None:
    """Host on every request; token on /api; Origin on cookie-less mutations."""
    host = request.headers.get("host") or ""
    if not request_host_is_trusted(host):
        return JSONResponse({"error": "untrusted host"}, status_code=400)

    path = request.url.path
    if not path.startswith("/api"):
        return None

    expected = _dash_token
    provided = request_token(request)
    if not expected or not provided or not _tokens_match(provided, expected):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    if request.method in {"POST", "DELETE", "PUT", "PATCH"}:
        origin = request.headers.get("origin")
        if origin and not request_origin_is_trusted(origin, request):
            return JSONResponse({"error": "untrusted origin"}, status_code=403)
    return None


class DashboardGuardMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)

        async def send_hardened(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("content-security-policy", _API_CSP)
                headers.setdefault("x-content-type-options", "nosniff")
            await send(message)

        denied = guard_dashboard_request(request)
        if denied is not None:
            await denied(scope, receive, send_hardened)
            return
        await self.app(scope, receive, send_hardened)


app = Starlette(
    debug=False,
    routes=routes,
    middleware=[Middleware(DashboardGuardMiddleware)],
)


def is_loopback_host(host: str) -> bool:
    h = (host or "").strip().lower()
    if h in {"127.0.0.1", "::1", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def check_dashboard_bind(host: str, allow_remote: bool = False) -> None:
    """Refuse non-loopback binds unless --allow-remote is explicit."""
    if is_loopback_host(host):
        return
    if not allow_remote:
        raise ValueError(
            f"refusing to bind dashboard to {host!r} (not loopback). "
            "Pass --allow-remote to expose the local memory console on the network."
        )
    print(
        f"WARNING: binding haunt dashboard to {host} — "
        "local memories are reachable beyond loopback. "
        "--allow-remote is unsafe without the launch token; "
        "namespaces are not authorization.",
        file=sys.stderr,
    )


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 7340,
    open_browser: bool = True,
    allow_remote: bool = False,
    token: str | None = None,
) -> None:
    import threading
    import time
    import socket
    import uvicorn
    import webbrowser

    check_dashboard_bind(host, allow_remote=allow_remote)
    launch_token = mint_dashboard_token() if token is None else token
    if allow_remote and not (launch_token or "").strip():
        raise ValueError(
            "refusing --allow-remote without a dashboard token. "
            "An empty token would expose an unauthenticated admin API."
        )
    configure_dashboard_security(
        token=launch_token or None,
        bind_host=host,
        allow_remote=allow_remote,
    )
    if launch_token:
        print(f"haunt dash token  {launch_token}")
        print("  send as X-Haunt-Token or ?token= on every /api route")
        if not embed_launch_token_in_html():
            print("  not embedded in HTML (--allow-remote / non-loopback bind)")
    else:
        print(
            "WARNING: dashboard launch token is empty — every /api route returns 401.",
            file=sys.stderr,
        )

    url = f"http://{host}:{port}"
    open_url = (
        f"{url}/?{urlencode({TOKEN_QUERY: launch_token})}"
        if launch_token and embed_launch_token_in_html()
        else url
    )

    if open_browser:
        def _open_when_ready() -> None:
            for _ in range(40):
                try:
                    with socket.create_connection((host, port), timeout=0.5):
                        webbrowser.open(open_url)
                        return
                except OSError:
                    time.sleep(0.25)

        threading.Thread(target=_open_when_ready, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()
