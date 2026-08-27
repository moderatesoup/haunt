// Renders the dashboard's own client script against captured API responses, so
// a stored payload is judged on the markup the browser would parse rather than
// on esc() in isolation. argv: <payloads.json>; prints innerHTML writes as JSON.
const fs = require("fs");
const PAYLOADS = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const CAPTURED = [];
const NODES = new Map();

function node(id) {
  if (NODES.has(id)) return NODES.get(id);
  const el = {
    id: id,
    value: "",
    checked: false,
    disabled: false,
    style: {},
    dataset: {},
    children: [],
    classList: { add() {}, remove() {}, contains() { return false; } },
    addEventListener() {},
    appendChild() {},
    scrollIntoView() {},
    querySelector() { return node(id + " >"); },
    querySelectorAll() { return []; },
    set innerHTML(v) { CAPTURED.push({ id: id, html: String(v) }); },
    get innerHTML() { return ""; },
    set textContent(v) {},
    get textContent() { return ""; },
  };
  NODES.set(id, el);
  return el;
}

const document = {
  getElementById: node,
  querySelector() { return node("querySelector"); },
  querySelectorAll() { return []; },
  addEventListener() {},
  createElement() { return node("createElement"); },
};
const location = { search: "" };
const alert = () => {};
const setInterval = () => 0;
const fetch = async (url) => {
  const path = String(url).split("?")[0];
  if (!(path in PAYLOADS)) throw new Error("no captured response for " + path);
  return { ok: true, json: async () => PAYLOADS[path], text: async () => "" };
};

//__DASHBOARD_SCRIPT__

(async () => {
  const ns = PAYLOADS.__namespace__;
  node("q").value = PAYLOADS.__query__;
  await loadNs(ns);
  await doBrowse(0);
  await loadTimeline();
  await loadProcedures();
  await loadWorldview();
  await loadHealth();
  await openDetail(PAYLOADS.__memory_id__, ns);
  await doRecall();
  process.stdout.write(JSON.stringify(CAPTURED));
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
