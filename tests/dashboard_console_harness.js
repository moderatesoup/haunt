// Runs the dashboard's own client script under stubbed browser globals, so
// console behaviour -- what a purge tells the operator, and what a data-act
// value dispatches -- is judged on the shipped script rather than a copy of
// it. argv: <case.json>; prints {alerts, errors, called, requests} as JSON.
const fs = require("fs");
const CASE = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const ALERTS = [];
const CALLED = [];
const REQUESTS = [];
const CLICK_HANDLERS = [];
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
    set innerHTML(v) {},
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
  addEventListener(type, handler) {
    if (type === "click") CLICK_HANDLERS.push(handler);
  },
  createElement() { return node("createElement"); },
};
const location = { search: "" };
const alert = (message) => ALERTS.push(String(message));
const setInterval = () => 0;
const fetch = async (url, opts) => {
  const path = String(url).split("?")[0];
  REQUESTS.push({ path: path, method: (opts && opts.method) || "GET" });
  if (!(path in CASE.responses)) throw new Error("no captured response for " + path);
  return { ok: true, json: async () => CASE.responses[path], text: async () => "" };
};

//__DASHBOARD_SCRIPT__

(async () => {
  const errors = [];
  // Wrap the real handlers in place so a dispatch that reaches one is
  // visible, and one that resolves to an inherited Object.prototype member
  // is visibly not in this list.
  for (const key of Object.keys(ACTIONS)) {
    const original = ACTIONS[key];
    ACTIONS[key] = (el) => {
      CALLED.push(key);
      return original(el);
    };
  }
  if (CASE.purge) {
    DETAIL_MID = CASE.purge.memory_id;
    DETAIL_NS = CASE.purge.namespace;
    NS = CASE.purge.namespace;
    ALL_NS = true;
    await doPurge();
  }
  for (const act of CASE.acts || []) {
    const el = { dataset: { act: act }, closest() { return el; } };
    for (const handler of CLICK_HANDLERS) {
      try {
        handler({ target: el });
      } catch (err) {
        errors.push(act + ": " + String(err));
      }
    }
  }
  process.stdout.write(
    JSON.stringify({ alerts: ALERTS, errors: errors, called: CALLED, requests: REQUESTS })
  );
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
