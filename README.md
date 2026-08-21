# haunt

Local-first **verbatim** memory for AI agents. One SQLite file per namespace. No cloud, no distillation, no reader-LLM.

[Apache-2.0 License](LICENSE)

## 60-second start

```bash
# Option A: pip install (no clone needed)
pip install haunt
haunt bootstrap          # creates ~/.haunt, probes sqlite-vec, downloads embed model, installs desktop shortcut
haunt dash               # opens the memory console in your browser → http://127.0.0.1:7340

# Option B: one-command bootstrap (git clone)
scripts/bootstrap.sh

# Option C: manual (git clone)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
haunt bootstrap          # creates ~/.haunt, probes sqlite-vec, downloads BAAI/bge-m3 (~2.28 GB), installs desktop shortcut
haunt dash               # opens the memory console in your browser → http://127.0.0.1:7340
```

To wire up Cursor hooks:

```bash
haunt cursor-install     # merges hooks at ~/.cursor/hooks.json + writes ~/.cursor/rules/haunt.mdc
```

Then add haunt as an MCP server (it runs alongside any other servers you have — it does not replace them):

```json
{
  "mcpServers": {
    "haunt": {
      "command": "~/.haunt/bin/haunt-mcp"
    }
  }
}
```

> **Note:** `haunt-mcp` is a stdio server — do not run it with `--help` or directly in a terminal; it reads JSON on stdin and will hang. Use it only as an MCP server command.

### macOS: use Homebrew Python, not pyenv

pyenv-compiled CPython typically lacks `--enable-loadable-sqlite-extensions`, which means `sqlite-vec` cannot load (`enable_load_extension` is missing or disabled). `haunt bootstrap` will correctly fail loud if this happens.

The working pattern on macOS:

```bash
# Install Homebrew Python (has loadable-extension support out of the box)
brew install python@3.14      # or python@3.12, python@3.13

# Create a venv from Homebrew Python — NOT from pyenv Python
/opt/homebrew/bin/python3 -m venv ~/.haunt/venv
source ~/.haunt/venv/bin/activate
pip install -e .              # or: pip install haunt
haunt bootstrap               # should succeed — sqlite-vec loads

# ~/.haunt/bin/haunt-mcp execs the venv Python, so MCP works everywhere
```

If you **must** use pyenv, rebuild with:

```bash
PYTHON_CONFIGURE_OPTS="--enable-loadable-sqlite-extensions" pyenv install 3.12
```

### What is and isn't automatic

- **Hooks store automatically.** Once `haunt cursor-install` runs, every prompt, response, and tool call is stored verbatim via Cursor hooks.
- **Recall is NOT automatic.** Cursor does not reliably inject `additional_context` from `beforeSubmitPrompt` into the model context. Agents must call `memory_recall` explicitly via MCP.
- **sessionStart** injects a worldview card via `additional_context` — this may appear as context but is not a guaranteed recall path.
- **No-hook IDEs** (Grok Bot, Claude Code, Codex, etc.): call `memory_observe` and `memory_recall` manually via MCP. There is no Cursor-style hook integration for these environments — they have no auto-store hooks.

## Memory console (dashboard)

`haunt dash` serves a local-only memory management UI at `http://127.0.0.1:7340` and opens it in your default browser once the server is ready. Pass `--no-open` to suppress the browser (useful for CI or scripting). No React, no npm — single-file inline HTML.

Features:

- **Browse** memories with filters: tier, origin, session, time range. Paginated.
- **Search / recall** with hybrid vec+FTS. Results link to detail view. Clicking any result row opens the detail panel.
- **All-namespaces search**: select "all namespaces" in the sidebar to fan out a single query across every registered namespace. Results are merged by score and each hit shows a namespace badge. API: `GET /api/recall?q=&k=&tier=`.
- **Memory detail** with full provenance: origin, session_id, event_id, memory_id, role, tier, event_time, valid_from/valid_to, db_path (absolute), haunt_home, tool name/input/output, related memories from the same session, entity mentions.
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

For FTS-only (no embeddings at all):

```bash
HAUNT_FTS_ONLY=1 haunt bootstrap
```

CI runs FTS-only to avoid the 2 GB download.

## Cursor hooks

Auto-store every prompt, reply, and tool call. Verbatim only — no LLM, no summaries. Fail-open (`{}` + exit 0) so a hook never blocks the agent.

`haunt cursor-install` does two things:
1. Merges haunt hook entries into `~/.cursor/hooks.json` (preserving your existing hooks).
2. Writes `~/.cursor/rules/haunt.mdc` — a Cursor rule file that tells agents how to use haunt (recall responsibility, observe rules, skip list).

**Secret redaction:** Hook-stored tool input and output are run through a best-effort denylist (API keys, bearer tokens, AWS keys, GitHub PATs, JWTs, etc.). This is **not** a security boundary — see [SECURITY.md](SECURITY.md).

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

## CLI

| command | what |
|---|---|
| `haunt bootstrap [--reembed]` | first-run setup; installs desktop shortcut; exits 1 if sqlite-vec fails |
| `haunt init [name] [--repo PATH]` | create a namespace |
| `haunt observe TEXT ...` | store a turn / tool call verbatim |
| `haunt recall QUERY [--as-of --since --until --tier --k]` | hybrid recall (vec + FTS5 + RRF) |
| `haunt delete MEMORY_ID [--event-id] [-y]` | hard-delete a memory and its provenance chain |
| `haunt timeline` | events by `event_time` |
| `haunt namespaces` | list + counts |
| `haunt health [-n NAMESPACE]` | vec / embed / counts / db path |
| `haunt worldview [-n NAMESPACE]` | compact namespace briefing: facts, entities, procedures |
| `haunt procedure write NAME --body BODY` | store a named procedure |
| `haunt procedure get NAME` | retrieve a named procedure |
| `haunt procedure list` | list all active procedures |
| `haunt graph [--entity] [--rebuild]` | entities + relations |
| `haunt dash [--port 7340] [--install-icon] [--no-open]` | local memory console (127.0.0.1); opens browser automatically (use `--no-open` for CI/scripts) |
| `haunt cursor-install` | merge Cursor user hooks + install haunt.mdc rule |

## MCP

haunt is its own MCP server — it runs alongside any other servers you already have (IronRecall, etc.) without interfering.

After bootstrap, add haunt to your MCP config:

```json
{
  "mcpServers": {
    "haunt": {
      "command": "~/.haunt/bin/haunt-mcp"
    }
  }
}
```

`haunt-mcp` is a stdio server. Do not run it directly in a terminal — it reads JSON on stdin. Use it only as an MCP server command in your client config.

Tools: `memory_observe`, `memory_recall`, `memory_purge`, `memory_worldview`, `memory_procedure`, `memory_contradict`, `memory_timeline`, `memory_health`, `memory_namespaces`, `memory_session_end`.

## Environment variables

| variable | default | what |
|---|---|---|
| `HAUNT_HOME` | `~/.haunt` | data directory |
| `HAUNT_EMBED_MODEL` | `BAAI/bge-m3` | embedding model (set to `BAAI/bge-small-en-v1.5` for smaller; `off` for none) |
| `HAUNT_FTS_ONLY` | unset | set to `1` for FTS-only (no embeddings) |
| `HAUNT_NAMESPACE` | inferred from git | override namespace |
| `HAUNT_MODEL_CACHE` | `$HAUNT_HOME/models` | model download directory |

Legacy aliases `LORE_HOME`, `LORE_EMBED_MODEL`, etc. are accepted. CLI aliases `lore` and `engram` still work.

## Layout

```
~/.haunt/
├── registry.db
├── namespaces/<name>.db
├── bin/haunt-mcp
├── bin/haunt-hook
└── models/
```

Default home is `~/.haunt`. If `~/.haunt` does not exist and `~/.lore` does, haunt falls back to `~/.lore` (no silent migration — existing data stays in place).

## What v1 does / does not

**Does:** verbatim events + memories, bi-temporal filtering, namespaces as isolated SQLite files, deterministic graph extract, hybrid recall, hard purge (delete with full provenance chain cleanup), CLI, MCP, Cursor hooks, local memory management dashboard.

**Does not:** summarize or distill, call any LLM, talk to the network at query time, require Docker/Postgres, expose a team HTTP API, auto-recall into the model context.
