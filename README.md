# haunt

Local-first **verbatim** memory for AI agents. One SQLite file per namespace. No cloud, no distillation, no reader-LLM.

[Apache-2.0 License](LICENSE)

## 60-second start

```bash
# Option A: pip install from git (no clone needed)
# PyPI name `haunt` is taken (mikepqr's stow); git install is the path.
pip install git+https://github.com/moderatesoup/haunt.git
haunt bootstrap          # creates ~/.haunt; probes sqlite-vec + downloads embed model unless HAUNT_FTS_ONLY=1
haunt dash               # opens the memory console in your browser → http://127.0.0.1:7340

# Option B: one-command bootstrap (git clone)
scripts/bootstrap.sh

# Option C: manual (git clone)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
haunt bootstrap          # creates ~/.haunt; probes sqlite-vec + downloads BAAI/bge-m3 (~2.28 GB) unless HAUNT_FTS_ONLY=1
haunt dash               # opens the memory console in your browser → http://127.0.0.1:7340
```

To wire up all supported hosts (Cursor + Claude Code) in one command:

```bash
haunt install            # hooks + MCP + rules for Cursor and Claude Code
```

This registers haunt hooks, MCP server, agent rules, and the haunt skill for
each host — even if the host is not installed yet (config dirs are pre-seeded).
Re-run `haunt install` or `haunt doctor` after adding a new editor, or if an
installer rewrote a config. `haunt doctor` checks the files install wrote,
the `haunt-mcp` wrapper (not a PATH name), sqlite-vec load (or explicit FTS-only), and embed model
presence (or explicit FTS-only) and exits 1 if any check fails.

To wire up Cursor only:

```bash
haunt cursor-install     # hooks.json + mcp.json + haunt.mdc (Cursor only)
```

> **Note:** `haunt-mcp` is a stdio server — do not run it with `--help` or directly in a terminal; it reads JSON on stdin and will hang. Use it only as an MCP server command.

### macOS: use Homebrew Python, not pyenv

pyenv-compiled CPython typically lacks `--enable-loadable-sqlite-extensions`, which means `sqlite-vec` cannot load (`enable_load_extension` is missing or disabled). `haunt bootstrap` will correctly fail loud if this happens, unless `HAUNT_FTS_ONLY=1` (or `HAUNT_EMBED_MODEL=off`) is set — then bootstrap still creates the layout and a usable FTS-only default namespace.

The working pattern on macOS:

```bash
# Install Homebrew Python (has loadable-extension support out of the box)
brew install python@3.14      # or python@3.12, python@3.13

# Create a venv from Homebrew Python — NOT from pyenv Python
/opt/homebrew/bin/python3 -m venv ~/.haunt/venv
source ~/.haunt/venv/bin/activate
pip install -e .              # or: pip install git+https://github.com/moderatesoup/haunt.git
haunt bootstrap               # should succeed — sqlite-vec loads

# ~/.haunt/bin/haunt-mcp execs the venv Python, so MCP works everywhere
```

If you **must** use pyenv, rebuild with:

```bash
PYTHON_CONFIGURE_OPTS="--enable-loadable-sqlite-extensions" pyenv install 3.12
```

### What is and isn't automatic

- **Hooks store automatically.** Once `haunt install` runs, every prompt, response, and tool call is stored verbatim via hooks in both Cursor and Claude Code.
- **Recall is NOT automatic.** Neither Cursor nor Claude Code reliably inject recall into the model context. Agents must call `memory_recall` explicitly via MCP unless a `[haunt ns=…]` block is already visible.
- **sessionStart / SessionStart** may return a worldview card — this may appear as context but is not a guaranteed recall path and is not a kernel.
- **No-hook IDEs** (Grok Bot, Codex, etc.): call `memory_observe` and `memory_recall` manually via MCP.

## Memory console (dashboard)

`haunt dash` serves a local-only memory management UI at `http://127.0.0.1:7340` and opens it in your default browser once the server is ready. Pass `--no-open` to suppress the browser (useful for CI or scripting). No React, no npm — single-file inline HTML.

Features:

- **Browse** memories with filters: tier, origin, session, time range. Paginated.
- **Search / recall** with hybrid vec+FTS. Results link to detail view. Clicking any result row opens the detail panel.
- **All-namespaces search**: select "all namespaces" in the sidebar to fan out a single query across every registered namespace. Results are merged by score and each hit shows a namespace badge. API: `GET /api/recall?q=&k=&tier=`.
- **Memory detail** with full provenance: origin, session_id, event_id, memory_id, role, tier, event_time, valid_from/valid_to, db_path (absolute), haunt_home, tool name/input/output, related memories from the same session, entity mentions.
- **Timeline** view: events in time order with since/until day filters. Rows click through to memory detail.
- **Time-bounded search**: `as_of`, `since`, `until` date filters on recall (both per-namespace and all-namespaces).
- **Supersede** a memory from the detail panel (sets `valid_to=now`, optional replacement text). Keeps original data — distinct from delete.
- **Delete** a memory from the UI (with confirmation). Hard purge removes the memory, FTS index, vector embedding, graph rows, and orphan events. Recall will not return deleted content.
- **Procedures and worldview** visible (browse, view detail; write stays CLI/MCP).
- **Health** strip: sqlite-vec status/version, embed model+dim+availability, last write age, event count, namespace, absolute db_path. Always visible (persistent header), live-updating (15s poll).

### Desktop shortcut

`haunt bootstrap` automatically installs a desktop shortcut. You can also install or reinstall it manually:

```bash
haunt dash --install-icon
```

Writes a real shortcut: Linux `.desktop` file (`Haunt Memories` → `haunt dash`), macOS `.command` script, or Windows `.bat`. The shortcut launches `haunt dash`, which starts the server and opens the browser.

## Embeddings

Default is **`BAAI/bge-m3`** dense (1024-d, ~2.28 GB ONNX). haunt loads it locally via onnxruntime (no API key). First `haunt bootstrap` downloads the model from Hugging Face.

Namespaces lock to the embed model/dimension they were written with. Switching models later requires a full re-embed (`haunt bootstrap --reembed`). A namespace switch = full re-embed of that namespace.

For machines that cannot spare 2 GB, use the small model as an explicit opt-in:

```bash
HAUNT_EMBED_MODEL=BAAI/bge-small-en-v1.5 haunt bootstrap
```

For FTS-only (no embeddings, sqlite-vec not required):

```bash
HAUNT_FTS_ONLY=1 haunt bootstrap
```

This still creates `~/.haunt` and the default namespace. It does not download BGE-M3 and does not fail if sqlite-vec cannot load. `HAUNT_EMBED_MODEL=off` is the same FTS gate. CI uses this to avoid the 2 GB download.

## Hooks

Auto-store every prompt, reply, and tool call. Verbatim only — no LLM, no summaries. Fail-open (`{}` + exit 0) so a hook never blocks the agent.

`haunt install` binds all known hosts (mkdir parents even if the app is not installed). `haunt cursor-install` binds Cursor only. Each bind:

1. Merges capture hooks (preserving foreign hooks).
2. Merges the `haunt` MCP stdio server (preserving other servers) as an absolute `~/.haunt/bin/haunt-mcp` command — not a PATH lookup.
3. Writes a small haunt-owned rule so agents still `memory_recall` if no `[haunt ns=…]` block is visible.
4. Writes `skills/haunt/SKILL.md` into the host config dir.

**Secret redaction:** Hook-stored tool input and output are run through a best-effort denylist (API keys, bearer tokens, AWS keys, GitHub PATs, JWTs, etc.). This is **not** a security boundary — see [SECURITY.md](SECURITY.md).

### Cursor hooks

| hook | what |
|---|---|
| `beforeSubmitPrompt` | observe user prompt, recall k=8 (results returned in `additional_context` but Cursor does not reliably inject this into the model context) |
| `afterAgentResponse` | observe assistant text |
| `afterAgentThought` | skipped by default (`HAUNT_STORE_THOUGHTS=1` to store as system/coordinate) |
| `postToolUse` | observe tool_name + input + output |
| `afterShellExecution` | observe command + output |
| `afterMCPExecution` | observe MCP call (skips `memory_*` to avoid recursion) |
| `sessionStart` | session-open coordinate event + worldview card (`additional_context`); not a reliable recall path |
| `sessionEnd` | close session; no summary |

Cursor MCP is merged into `~/.cursor/mcp.json` (honors `CURSOR_HOME`) as an absolute `~/.haunt/bin/haunt-mcp` command.

### Claude Code hooks

| hook | what |
|---|---|
| `UserPromptSubmit` | observe `prompt`; return `additionalContext` inside `hookSpecificOutput` with `hookEventName` |
| `Stop` | observe `last_assistant_message` (never exit 2) |
| `SessionStart` | session-open coordinate event + worldview card (same JSON shape) |
| `SessionEnd` | close session |
| `PostToolUse` | observe tool I/O as episodic (skips `memory_*`) |
| `PostToolUseFailure` | same as PostToolUse for failed tool calls |

Claude hooks live in `~/.claude/settings.json` (nested matcher-group schema, absolute `haunt-hook-claude`). User-scope MCP **must** live in `~/.claude.json` (or `$CLAUDE_CONFIG_DIR/.claude.json`) — `settings.json` silently ignores `mcpServers`. The rule is `~/.claude/rules/haunt.md` (does not overwrite `CLAUDE.md`).

## CLI

| command | what |
|---|---|
| `haunt bootstrap [--reembed]` | first-run setup; installs desktop shortcut; exits 1 if sqlite-vec fails (unless `HAUNT_FTS_ONLY=1`) |
| `haunt init [name] [--repo PATH]` | create a namespace |
| `haunt observe TEXT ...` | store a turn / tool call verbatim |
| `haunt recall QUERY [--as-of --since --until --clock --tier --k]` | hybrid recall (vec + FTS5 + RRF); query-time temporal compile |
| `haunt delete MEMORY_ID [-y]` / `haunt delete --event-id EVENT_ID [-y]` | hard-delete a memory (or all memories for an event) and its provenance chain |
| `haunt timeline [--since --until --clock]` | events by `event_time` or `storage_time` (`ts` ingest time; `write_time` is a deprecated alias) |
| `haunt namespaces` | list + counts |
| `haunt health [-n NAMESPACE]` | vec / embed / counts / db path |
| `haunt worldview [-n NAMESPACE]` | compact namespace briefing: facts, entities, procedures |
| `haunt procedure write NAME --body BODY` | store a named procedure |
| `haunt procedure get NAME` | retrieve a named procedure |
| `haunt procedure list` | list all active procedures |
| `haunt graph [--entity] [--rebuild]` | entities + relations |
| `haunt dash [--port 7340] [--install-icon] [--no-open]` | local memory console (127.0.0.1); opens browser automatically (use `--no-open` for CI/scripts) |
| `haunt install` | bind all known hosts (Cursor, Claude Code): hooks + MCP + rules + skill |
| `haunt doctor` | check sqlite-vec (or FTS-only), haunt-mcp wrapper/python, embed (or FTS-only), and host files; rematch host files if missing; exit 1 if any check fails |
| `haunt cursor-install` | bind Cursor only: hooks.json + mcp.json + haunt.mdc + skill |

## MCP

haunt is its own MCP server — it runs alongside any other servers you already have (IronRecall, etc.) without interfering.

`haunt install` (or `haunt bootstrap`) automatically registers the MCP server in both Cursor (`~/.cursor/mcp.json`) and Claude Code (`~/.claude.json`). Merge only — other servers are kept. No manual JSON paste required.

`haunt-mcp` is a stdio server. Do not run it directly in a terminal — it reads JSON on stdin. Use it only as an MCP server command in your client config.

Tools: `memory_observe`, `memory_recall`, `memory_purge`, `memory_worldview`, `memory_procedure`, `memory_contradict`, `memory_timeline`, `memory_health`, `memory_namespaces`, `memory_session_end`.

## Environment variables

| variable | default | what |
|---|---|---|
| `HAUNT_HOME` | `~/.haunt` | data directory |
| `HAUNT_EMBED_MODEL` | `BAAI/bge-m3` | embedding model (set to `BAAI/bge-small-en-v1.5` for smaller; `off` for none) |
| `HAUNT_FTS_ONLY` | unset | set to `1` for FTS-only (no embeddings; sqlite-vec not required) |
| `HAUNT_NAMESPACE` | inferred from git | override namespace |
| `HAUNT_MODEL_CACHE` | `$HAUNT_HOME/models` | model download directory |

## Layout

```
~/.haunt/
├── registry.db
├── namespaces/<name>.db
├── bin/haunt-mcp
├── bin/haunt-hook
├── bin/haunt-hook-claude
└── models/
```

Default home is `~/.haunt` (`HAUNT_HOME`).

## What v1 does / does not

**Does:** verbatim events + memories, bi-temporal filtering, namespaces as isolated SQLite files, deterministic graph extract, hybrid recall, hard purge (delete with full provenance chain cleanup), CLI, MCP, Cursor hooks, Claude Code hooks, local memory management dashboard.

**Does not:** summarize or distill, call any LLM, talk to the network at query time, require Docker/Postgres, expose a team HTTP API, auto-recall into the model context.
