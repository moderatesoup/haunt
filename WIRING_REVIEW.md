# Adversarial Wiring Review — `cursor/rebrand-haunt-01d0` (PR #2)

Reviewed: commit `f1bce5f` (branch tip at time of review).
Also read: PR #1 (`cursor/memory-tools-mvp-6af2`, tip `6a7d5c8`).

## Summary

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | `haunt --help` / `observe` / `recall` / `bootstrap` entry points | **PASS** | All resolve via pip `[project.scripts]`. `lore` and `engram` aliases also work. |
| 2 | MCP tools match README | **PASS** | Server exposes exactly 6 tools: `memory_observe`, `memory_recall`, `memory_timeline`, `memory_health`, `memory_namespaces`, `memory_session_end`. README lists the same 6. `worldview`/`procedure`/`contradict` exist on PR #1 only — README on this branch does not claim them. |
| 3 | Cursor hook installer writes executable, merges hooks.json | **PASS** | `haunt cursor-install` writes `~/.haunt/bin/haunt-hook` (chmod +x), merges existing hooks without clobbering (verified: `afterFileEdit` and prior `beforeSubmitPrompt` entries preserved). |
| 4 | Hook handlers fail-open (exit 0 / `{}`) on bad JSON | **PASS** | Piping `not-json{` into `haunt-hook` returns `{}` on stdout, exit 0. |
| 5 | Namespace isolation | **PASS** | Observing into `ns-a` and recalling from `ns-b` returns no hits; recalling from `ns-a` returns the correct hit. SQLite file-per-namespace design enforces this. |
| 6 | pytest passes | **PASS** | 17/17 pass (FTS-only mode; model-download not required). |
| 7 | Dashboard routes exist | **PASS** | 4 routes: `/`, `/api/namespaces`, `/api/namespace/{name}`, `/api/namespace/{name}/recall`. All return 200 via Starlette TestClient. |
| 8 | Embed dim mismatch / reembed path | **PASS (conditional)** | When embeddings are loaded, `embeddings_stale()` detects dim/model mismatch and `ensure_current_embeddings()` triggers a full rebuild. When embeddings are unavailable (model failed to load / FTS-only), `embeddings_stale()` returns False by design — this is intentional, not a silent no-op. |

## Bugs Found (fixed in this PR)

### BUG: `haunt --version` exits with code 2 "Missing command"

The `--help` output advertises `--version`, but running `haunt --version` errors out:

```
Usage: haunt [OPTIONS] COMMAND [ARGS]...
╭─ Error ──────────────────────────────────────────────────────────────────────╮
│ Missing command.                                                             │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Root cause:** Typer's `no_args_is_help=True` requires a command before it processes callback options. The `--version` option in `@app.callback()` never fires because the "missing command" error occurs first.

**Fix:** Set `invoke_without_command=True` on the callback, add `is_eager=True` to the `--version` option, and handle no-args-help manually in the callback. Removes `no_args_is_help=True` from the Typer constructor.

## Minor Observations (not bugs)

1. **`afterAgentThought` is not installed by `cursor-install`** — The README documents it as "skipped by default" with a `HAUNT_STORE_THOUGHTS=1` opt-in, but `cursor-install` does not register it in hooks.json. The handler code _does_ process it if Cursor delivers it. This is consistent with the README's stated behavior, not a bug.

2. **Hook launcher uses `-m haunt.cursor_hook` fallback** — When the console script `haunt-hook` is not a sibling of `sys.executable` (e.g., user-install to `~/.local/bin` but python at `/usr/bin/python3`), the launcher falls back to `python -m haunt.cursor_hook`. This works as long as the `haunt` package is importable by the target Python. Not a bug, but worth noting for virtual-env-less installs.

3. **`contrib/cursor/hooks.json` uses `~/` tilde** — The example file uses `~/.haunt/bin/haunt-hook` which may not be resolved by Cursor. But this is just a documentation example; `cursor-install` writes the real absolute path. Not a bug.

4. **PR #1 and PR #2 are divergent branches** — PR #1 (`cursor/memory-tools-mvp-6af2`) adds `memory_worldview`, `memory_procedure`, `memory_contradict` tools plus `SKILL.md`, but uses `src/lore/`. PR #2 rebrands to `src/haunt/` but doesn't include those three tools. These branches haven't been merged; the rebrand needs to be rebased on or merged with PR #1 to get those tools.

## Test Method

- Installed via `pip install -e ".[dev]"` (installs all console scripts)
- Ran `haunt --help`, `observe`, `recall`, `bootstrap`, `cursor-install`, `dash --help` via CLI
- Piped JSON payloads to the hook launcher directly
- Used Starlette TestClient for dashboard routes
- Programmatically listed MCP tools via `asyncio.run(server.list_tools())`
- Ran `pytest tests/ -v` (17 pass)
- Tested reembed path with fastembed BAAI/bge-small-en-v1.5 (384-d) loaded
- Verified namespace isolation via CLI (observe in ns-a, recall in ns-b returns nothing)
