# Silent Failure Audit — `moderatesoup/haunt` main

**SHA:** `2e6890a6deb678665b522f9a0ac6ea9bf060d057`
**Date:** 2026-08-21
**Scope:** Surfaces NOT already filed as #15–#18 or rejected by VL6 seat.

---

## FILE-candidates (new, for VL6 review)

### F1: `delete --event-id` requires a dummy `memory_id` positional (silent ignore)

**File:** `src/haunt/cli.py:327-333`

`memory_id` is `typer.Argument(...)` (required), but when `--event-id` is used,
the positional is silently ignored. A user who wants to delete by event has no
way to skip the positional:

```
$ haunt delete --event-id some-event-id --yes
Usage: haunt delete [OPTIONS] {memory_id}
Try 'haunt delete --help' for help.
╭─ Error ──────────────────────────────────────────────────────────────────╮
│ Missing argument 'memory_id'.                                            │
╰──────────────────────────────────────────────────────────────────────────╯
EXIT=2
```

They must supply a junk positional that is silently discarded:

```
$ haunt delete THIS-IS-IGNORED --event-id some-event-id --yes
no memories for event some-event-id
EXIT=1
```

**Impact:** UX confusion; the required `memory_id` is dead when `--event-id`
is present. Could also mislead: `haunt delete <real-id> --event-id <eid>`
ignores the positional `<real-id>` entirely — no warning that the two
arguments disagree.

**Fix:** Make `memory_id` optional (`typer.Argument(None, ...)`), require
exactly one of `memory_id` or `--event-id`, error on both.

---

### F2: Dashboard GET routes silently create non-existent namespaces

**File:** `src/haunt/dashboard.py:579-596`

Every read-only dashboard route (`/api/namespace/{name}`, `.../recall`,
`.../browse`, `.../health`, `.../procedures`, `.../worldview`) creates
the namespace as a side effect because `Store(name)` calls
`register_namespace()` with `create=True` (the default).

Line 581–582 has a guard that never fires:

```python
if not namespace_exists(name):
    pass          # ← dead code: should be 404 or at least skip create
```

**Evidence (python):**

```
Before GET: namespace_exists('ghost-ns'): False
GET /api/namespace/ghost-ns → 200
After GET:  namespace_exists('ghost-ns'): True
```

Every route does it — recall, browse, health, procedures, worldview all
auto-create via `Store(name)`:

```
Namespaces before: ['default']
GET /api/namespace/phantom/recall?q=test  → 200  → namespaces: ['default', 'phantom']
GET /api/namespace/phantom2/browse        → 200  → namespaces: [..., 'phantom2']
GET /api/namespace/phantom3/health        → 200  → namespaces: [..., 'phantom3']
```

**Impact:** Typos, crawlers, or bookmarks to stale namespace URLs silently
create empty junk namespaces that pollute `haunt namespaces` and the
dashboard sidebar.

**Fix:** Dashboard routes should use `Store(name, create=False)` and return
404 if the namespace doesn't exist, or remove the dead `if not
namespace_exists(name): pass` guard and document auto-create.

---

### F3: `memory_session_end` returns `ok: True` for non-existent sessions

**File:** `src/haunt/mcp_server.py:144-152` + `src/haunt/store.py:356-367`

When called with a session that doesn't exist, `end_session()` runs
`UPDATE sessions SET ended_at=? WHERE id=? AND ended_at IS NULL` which
matches zero rows, then returns the session_id anyway. The MCP wrapper
returns `ok: True`:

```json
// memory_session_end(namespace="default", session=None) — no session active
{"ok": true, "namespace": "default", "session_id": null, "distilled": false}

// memory_session_end(namespace="default", session="nonexistent-session-id")
{"ok": true, "namespace": "default", "session_id": "nonexistent-session-id", "distilled": false}
```

**Impact:** Callers cannot distinguish "session ended" from "no session
to end" or "session doesn't exist". An agent that calls
`memory_session_end(session="typo")` gets `ok: True` and believes it
succeeded.

**Fix:** Return `ok: False` (or at least `ended: false`) when the UPDATE
matched zero rows. Check `cursor.rowcount` or re-query.

---

### F4: `haunt procedure` no-args exits 2 (inconsistent with root)

**File:** `src/haunt/cli.py:268-273`

```
$ haunt
[shows help]
EXIT=0

$ haunt procedure
[shows help]
EXIT=2
```

The root `app` uses `invoke_without_command=True` + a manual help
callback (fixed in PR #2/commit 2e6890a). The `procedure_app` subcommand
still uses `no_args_is_help=True` which Typer/Click renders as help
text but exits 2 — the "error" exit code. Any script that runs
`haunt procedure` to check if procedures are available sees exit 2 as a
failure.

**Impact:** Inconsistent CLI contract. Scripts/CI steps that check
`haunt procedure || echo "no procedure support"` will false-alarm.

**Fix:** Mirror the root pattern: add `invoke_without_command=True`
callback on `procedure_app`, remove `no_args_is_help=True`, return
help + exit 0 when invoked without a subcommand.

---

### F5: Hook `postToolUse` / `afterShellExecution` / `afterMCPExecution` stores generic tool I/O as `tier=procedural` — invisible to worldview

**File:** `src/haunt/cursor_hook.py:242-289`

Generic tool calls (Read, Shell, MCP) are stored as `tier="procedural"`.
The code has a TODO comment on line 248:

```python
# TODO: tier="procedural" is a lane mix — generic tool I/O is episodic,
# not a named how-to.  Should be "episodic" unless meta.kind=procedure.
```

These memories are invisible to every aggregation view:
- Not in `worldview.facts` (requires `tier='semantic'`)
- Not in `worldview.procedures` (requires `meta LIKE '%"kind": "procedure"%'`)
- Not in `procedure_list` (same filter)
- Not in `worldview.names` (entity extraction still works, but the
  memories themselves are orphaned)

**Evidence:**

```
postToolUse  tool_name=Read  → stored as tier=procedural
procedure_list:      0 procedures  (no kind=procedure metadata)
worldview.facts:     0 items       (not semantic)
worldview.procedures: 0 items      (no kind=procedure)
memories by tier:    procedural=1  (exists but invisible)
```

**Impact:** Tool I/O from hook ingestion (the majority of memories in a
Cursor session) is stored but invisible to every user-facing summary.
`worldview` under-reports memory content because it never queries
`tier='procedural'` memories that lack `kind: procedure`.

**Fix:** Store as `tier="episodic"` unless the tool is explicitly a
procedure operation.

---

## Nits (lower priority)

### N1: `list_namespaces()` calls `st.stats()` 5× per namespace

**File:** `src/haunt/store.py:1086-1096`

Each call to `st.stats()` runs 5 `COUNT(*)` queries. Called 5 times
per namespace row. Total: 25 queries per namespace instead of 5.

```python
extra.update({
    "events": st.stats()["events"],
    "memories": st.stats()["memories"],
    "sessions": st.stats()["sessions"],
    "entities": st.stats()["entities"],
    "db_size_bytes": st.stats()["db_size_bytes"],
})
```

**Fix:** `s = st.stats(); extra.update({k: s[k] for k in [...]})`.

### N2: `memory_session_end` returns `distilled: False` — feature doesn't exist

**File:** `src/haunt/mcp_server.py:152`

The response includes `"distilled": False` suggesting a future
distillation feature. No such feature exists. The field is always
False.

### N3: CI runs all tests `HAUNT_FTS_ONLY=1` — embed/vec paths untested

**File:** `.github/workflows/ci.yml:26`

`HAUNT_FTS_ONLY: "1"` is set for the entire pytest step. All
embedding and sqlite-vec query paths (`_vec_hits`, `embed_one`,
`ensure_vec_table`, `ensure_current_embeddings`) are never exercised
in CI. The conftest fixture does set `HAUNT_EMBED_MODEL` to
bge-small for `lore_env`, but the CI-level `HAUNT_FTS_ONLY=1`
overrides it.

### N4: `recall._fts_hits` and `recall._vec_hits` swallow `sqlite3.Error`

**File:** `src/haunt/recall.py:128,160`

Same pattern as #15 (`_connect` swallows sqlite-vec load). These are
in the recall hot path: if FTS or vec queries fail, the error is
swallowed and results silently degrade to partial or empty. Distinct
file from #15 but same family of issue.

---

## Surfaces examined — no issues found

| Surface | Result |
|---------|--------|
| `haunt --version` | Exits 0, prints semver. CI validates with grep. |
| `haunt --help` | Exits 0. All 14 commands listed. |
| All CLI commands wired | All 14 commands resolve to implementation. No unwired/dead commands. |
| All 10 MCP tools wired | Verified via `server.list_tools()`. |
| `cursor-install` | Writes hooks.json, launcher, rule file. Merge-without-clobber tested. |
| Desktop icon (Linux/macOS/Win) | All 3 platform paths write files. Tests verify content. |
| `recall` empty query | Returns 0 hits. No crash, no random results. |
| `embed fallback` lock-in | Fallback is logged via `diag()`, visible in `haunt health`. Auto-reembed on dim mismatch is documented behavior. |
| `haunt health` | Always produces output. Shows vec, embed, namespace, counts. |
| `bootstrap.sh` | `haunt dash --install-icon || true` already rejected by VL6 as by-design under `set -e`. |
| Hook fail-open | Already rejected by VL6 as product design. |

---

## Already-filed items verified still present

| Issue | Status at SHA 2e6890a |
|-------|----------------------|
| #15 `_connect` swallows sqlite-vec load | `store.py:43-48` — `except Exception: pass` still present |
| #16 observe `embedded=True` without vec_memories insert | `store.py:448-455` — `except sqlite3.Error: pass` still present |
| #17 reembed `updated=` does not mean vec indexed | `store.py:521-528` — same pattern |
| #18 list_namespaces zeros on corrupt DB | `store.py:1095-1098` — `except sqlite3.Error: pass` still present |
| PR #14 memory_procedure invalid action → list | `mcp_server.py:212-213` — `else: procs = st.procedure_list()` |

---

## Method

1. Read every tracked file in the repo (18 source, 6 test, 2 config, 3 doc)
2. Installed `haunt` (`pip install -e ".[dev]"`) on Ubuntu + Python 3.12.3
3. Ran `pytest tests/ -v` → 63 pass
4. Reproduced each finding with isolated `HAUNT_HOME` in `mktemp -d`
5. Verified with `HAUNT_FTS_ONLY=1` (matching CI) and `HAUNT_EMBED_MODEL=off`
6. Checked MCP tool list via `asyncio.run(server.list_tools())`
7. Used Starlette `TestClient` for dashboard route tests
