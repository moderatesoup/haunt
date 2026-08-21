# Silent-Failure Sweep 2 — Findings Report

**Measured at:** `e58dbf25f48c3d423f8e773a89c520a79107c875` (current `main`)  
**Diff reviewed:** `d0e12ab3..e58dbf25` (PR #13 first-run UX + PR #14 memory_procedure reject)  
**Date:** 2026-08-21

---

## PR #14 Verification (CONFIRMED WORKING)

The merged fix correctly rejects unknown actions:

```
$ python3 -c "from haunt.mcp_server import memory_procedure; print(memory_procedure(action='delete', name='x', namespace='default'))"
{"ok": false, "error": "unknown action 'delete', must be one of: write, get, list"}
```

All six tested invalid actions (`delete`, `update`, `remove`, `supersede`, `""`, `DROP TABLE`) return `ok=False` with a clear error. Valid actions (`write`, `get`, `list`) remain functional.

---

## NEW Candidates (not already filed)

### CANDIDATE 1: `GET /api/namespace/{ANYTHING}` auto-creates namespace (200 OK)

**File:** `src/haunt/dashboard.py` lines 579–596  
**Severity:** Medium (data pollution, namespace squatting)  
**Classification:** BUG — false-green on nonexistent resource

The `api_namespace` dashboard route calls `namespace_exists(name)` but ignores the result (`pass` on line 582). It then opens `Store(name)` which auto-creates the namespace including a ~106KB SQLite database file.

**Reproduction:**

```python
from starlette.testclient import TestClient
from haunt.dashboard import app

client = TestClient(app)
r = client.get('/api/namespace/DOES_NOT_EXIST')
# status=200, creates /namespaces/DOES_NOT_EXIST.db (106KB)
```

**Evidence:**
- Before call: namespace "ghost_namespace_xyz" not in registry
- After call: namespace registered, 106KB `.db` file created on disk
- Response: `200 OK` with `{"stats": {"events": 0, ...}}`

**Expected behavior:** Return `404` with `{"error": "namespace 'X' not found"}` or at minimum don't auto-create the DB.

---

### CANDIDATE 2: `memory_session_end` returns `ok: True` for nonexistent sessions

**File:** `src/haunt/mcp_server.py` line 145–152, `src/haunt/store.py` lines 356–367  
**Severity:** Low-Medium (false green, no way for caller to distinguish success from no-op)  
**Classification:** CANDIDATE — silent false-green

`Store.end_session(session_id)` runs `UPDATE sessions SET ended_at=? WHERE id=? AND ended_at IS NULL` which matches 0 rows when the session doesn't exist, but still returns the passed-in `session_id`. The MCP wrapper then returns `{"ok": true, "session_id": "bogus-id", "distilled": false}`.

**Reproduction:**

```python
from haunt.mcp_server import memory_session_end
import json
raw = memory_session_end(namespace='default', session='BOGUS-NEVER-EXISTED')
print(json.loads(raw))
# {"ok": true, "namespace": "default", "session_id": "BOGUS-NEVER-EXISTED", "distilled": false}
```

**Evidence:**
- DB query: `SELECT * FROM sessions WHERE id='BOGUS-NEVER-EXISTED'` → 0 rows
- Return value: `ok=True` despite nothing being ended

**Could be by-design:** If the design intent is "fire-and-forget, idempotent close", this is acceptable. But it differs from `memory_purge`/`memory_contradict` which properly return `ok=False` on missing IDs.

---

### CANDIDATE 3: `memory_observe(text="")` stores ghost memories

**File:** `src/haunt/mcp_server.py` lines 33–69, `src/haunt/store.py` lines 369–456  
**Severity:** Low (data pollution, worldview noise)  
**Classification:** NIT — by-design for tool events, problematic for pure-text observe

When `memory_observe(text="", tier="semantic")` is called with no tool fields, it:
1. Returns `{"ok": true, "embedded": false, ...}`
2. Stores an event with `content=""`
3. Inserts an empty FTS row
4. Creates a memory row with `content=""`

These ghost memories then appear in `worldview` as empty facts (`content: ""`).

**Reproduction:**

```python
from haunt.mcp_server import memory_observe, memory_worldview
import json

memory_observe(text='', namespace='default', tier='semantic')
wv = json.loads(memory_worldview(namespace='default'))
# facts includes: {"id": "...", "content": "", ...}
```

**Note:** Empty content is LEGITIMATE for tool events (`postToolUse`, `afterShellExecution`, etc.) where `tool_input`/`tool_output` carry the data and `verbatim_text()` concatenates them. The issue is specifically MCP `memory_observe(text="")` with no tool fields.

**Expected behavior:** Either reject with `ok=False` and `"error": "text is required when no tool fields provided"`, or at minimum skip FTS/memory row creation for empty verbatim text.

---

## Not NEW / By-Design / Already Filed (confirmed still present)

| Issue | Status | Note |
|-------|--------|------|
| `bootstrap.sh \|\| true` | Rejected | Fail-open by design |
| Hook fail-open (`run()` returns `{}`) | By design | Documented at top of `cursor_hook.py` |
| `#15` vec swallow in `_connect` | Filed | Still present in `store.py:39` area |
| `#16` observe `embedded=True` without vec insert | Filed | `store.py:448-455` still swallows `sqlite3.Error` |
| `#17` reembed `updated=` counter | Filed | Unchanged |
| `#18` `list_namespaces` zeros | Filed | `list_namespaces` still swallows `sqlite3.Error` at line 1097 |
| `purge ok=True` with individual deleted flags | Downgraded | Struct is intentional |
| PR #14 `memory_procedure` invalid action | Merged | Verified working |

---

## CI Pass-Condition Analysis

| Step | Assertion type | Adequate? |
|------|---------------|-----------|
| `haunt --version \| grep -qE '^[0-9]+'` | grep exit code | Yes |
| `haunt --help` | exit code only | Marginal but intentional (exercises imports) |
| `haunt observe + recall \| grep -q` | grep on content | Yes — fails if recall returns no hits |
| `haunt health` | exit code only | Marginal — crashes on real errors, but "healthy but empty" passes |
| `dash --install-icon + test -f` | file existence | Yes |

No new CI step passes on literally empty output (the `grep -q` assertions are correct).

---

## Summary

| # | Finding | Class | Risk |
|---|---------|-------|------|
| 1 | Dashboard namespace auto-create on GET | BUG | Medium |
| 2 | `session_end` ok=True on nonexistent session | CANDIDATE | Low-Med |
| 3 | `observe(text="")` stores ghost memories | NIT | Low |

Candidates 1 and 2 are distinct from the sibling sweep (bc-7bc0564e). Candidate 1 is the most actionable — it's a clear 200-on-nonexistent-resource with a side effect (DB creation).
