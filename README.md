# engram

Local-first **verbatim** memory for AI agents. Cursor first. One SQLite file per namespace. No cloud, no distillation, no reader-LLM.

Public name is **engram**. The Python package, CLI entry `lore`, and home directory stay `lore` / `~/.lore` so existing venvs keep working. `lore` is a drop-in alias for `engram`.

v1 stores every chat turn and tool call as-is, embeds on-device, and recalls with sqlite-vec + FTS5 + RRF. It does **not** summarize, does **not** talk to a model, and has **no** team/org HTTP tier.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
engram bootstrap
engram observe "the auth token lives in src/auth/session.py"
engram recall "where is the auth token"
engram dash --port 7340
engram cursor-install
```

`lore` still works (`lore observe`, `lore-mcp`, …). `engram bootstrap` creates `~/.lore/{namespaces,bin,models}`, writes space-free launchers at `~/.lore/bin/lore-mcp` and `~/.lore/bin/engram-hook`, probes sqlite-vec, downloads the embedding model, and inits the `default` namespace.

Env: `LORE_HOME` (default `~/.lore`; `ENGRAM_HOME` is an alias), `LORE_EMBED_MODEL`, `LORE_NAMESPACE` (`ENGRAM_NAMESPACE` alias), `LORE_FTS_ONLY=1`, `LORE_MODEL_CACHE`.

## Cursor hooks

Auto-store every prompt, reply, and tool call. Verbatim only — no LLM, no summaries. Fail-open (`{}` + exit 0) so a hook never blocks the agent.

```bash
engram cursor-install
```

(`lore cursor-install` is the same command.) This:

1. Publishes `~/.lore/bin/engram-hook` (a `/bin/sh` wrapper to the venv `engram-hook` script; also writes `lore-hook`).
2. Merges user hooks at `~/.cursor/hooks.json` (version 1). Existing hook commands are kept. Adds:
   `beforeSubmitPrompt`, `afterAgentResponse`, `postToolUse`, `afterShellExecution`, `afterMCPExecution`, `sessionStart`, `sessionEnd`.

Project-level example: [`contrib/cursor/hooks.json`](contrib/cursor/hooks.json). Agent rule: [`contrib/cursor/engram.mdc`](contrib/cursor/engram.mdc) — copy into `.cursor/rules/` if you want the agent to `memory_recall` when hook context is missing.

| hook | what |
|---|---|
| `beforeSubmitPrompt` | observe user prompt (episodic), recall k=8, return `additional_context` |
| `afterAgentResponse` | observe assistant text |
| `afterAgentThought` | skipped by default (`ENGRAM_STORE_THOUGHTS=1` to store as system/coordinate) |
| `postToolUse` | observe tool_name + input + output (procedural) |
| `afterShellExecution` | observe command + output (`tool_name=Shell`) |
| `afterMCPExecution` | observe MCP call; skips `memory_*` to avoid recursion |
| `sessionStart` | tiny session-open event + last 5 memories + MCP reminder |
| `sessionEnd` | close session; no summary |

Namespace is inferred from `CURSOR_PROJECT_DIR` / git / cwd (same as the rest of engram). Session id is `conversation_id` or `session_id`.

## Embeddings

Default is **`BAAI/bge-m3`** dense (1024-d). engram loads it locally via **onnxruntime + the model's tokenizer** (no API key):

1. Official ONNX from Hugging Face (`BAAI/bge-m3` `onnx/model.onnx` + `onnx/model.onnx_data` + `onnx/tokenizer.json`) into `~/.lore/models/BAAI-bge-m3` (~2.28 GB).
2. If that download fails, a community INT8 ONNX of the same model.
3. If a newer fastembed build lists `BAAI/bge-m3`, that path is tried next.
4. Automatic fallback: `BAAI/bge-small-en-v1.5` (384-d ONNX via fastembed, ~67 MB).

`LORE_EMBED_MODEL` is the switch (`BAAI/bge-m3` by default; set `BAAI/bge-small-en-v1.5` to force the small model; `off` / `LORE_FTS_ONLY=1` for FTS-only). Embeddings are never faked.

Namespace DBs store `embed_model` / `embed_dim` in `meta`. A dim or model change drops `vec_memories` and rebuilds every row — run **`engram bootstrap --reembed`** after switching models (observe/recall also auto-rebuild a stale namespace so 384-d demo data does not silently mix with 1024-d queries).

## CLI

| command | what |
|---|---|
| `engram init [name] [--repo PATH]` | create a namespace |
| `engram observe TEXT ...` | store a turn / tool call verbatim |
| `engram recall QUERY [--as-of --since --until --tier --k]` | hybrid recall |
| `engram timeline` | events by `event_time` |
| `engram namespaces` | list + counts |
| `engram health` | vec / embed / counts |
| `engram graph [--entity] [--rebuild]` | entities + relations; `--rebuild` wipes graph and re-extracts from events |
| `engram dash [--port 7340]` | local dashboard (127.0.0.1) |
| `engram bootstrap [--reembed]` | first-run setup; `--reembed` rebuilds all namespace vectors |
| `engram cursor-install` | merge Cursor user hooks → `~/.lore/bin/engram-hook` |

Every command also works as `lore …`. Observe flags: `--namespace --tier --session --role --tool-name --tool-input --tool-output --event-time`.

## Cursor MCP

After `engram bootstrap` (and `pip install -e .` so `python -m lore.mcp_server` imports):

```json
{
  "mcpServers": {
    "lore": {
      "command": "/home/box/.lore/bin/lore-mcp",
      "env": {
        "LORE_HOME": "/home/box/.lore",
        "LORE_EMBED_MODEL": "BAAI/bge-m3",
        "LORE_MODEL_CACHE": "/home/box/.lore/models"
      }
    }
  }
}
```

`engram bootstrap` prints the absolute launcher path. The file is a `/bin/sh` wrapper around the venv `lore-mcp` console script (stdio). `~/.lore/bin/engram-mcp` is the same server. Tools: `memory_observe`, `memory_recall`, `memory_timeline`, `memory_health`, `memory_namespaces`, `memory_session_end` (closes a session; **no** distillation).

## What v1 does / does not

**Does:** verbatim events + memories, bi-temporal `event_time` / `valid_from` / `valid_to`, namespaces as isolated SQLite files, deterministic graph extract (regex / noun-phrase / paths), hybrid recall, CLI, MCP, Cursor hooks, local dashboard.

**Does not:** summarize or distill, call any reader-LLM, talk to the network at query time, require Docker/Postgres, expose a team HTTP API, support Claude Code / Grok / Codex as first-class clients yet (MCP is generic; Cursor is documented first).

## Layout

```
~/.lore/registry.db
~/.lore/namespaces/<name>.db
~/.lore/bin/lore-mcp
~/.lore/bin/engram-mcp
~/.lore/bin/engram-hook
~/.lore/bin/lore-hook
~/.lore/models/
```

Default namespace is inferred from `CURSOR_PROJECT_DIR` / `git remote` / repo folder / cwd, else `default`. Home dir stays `~/.lore` (`LORE_HOME`; `ENGRAM_HOME` accepted as an alias).
