# haunt

Local-first **verbatim** memory for AI agents. Cursor first. One SQLite file per namespace. No cloud, no distillation, no reader-LLM.

> Renamed from *engram* (2025) to avoid collisions with other agent-memory products.

v1 stores every chat turn and tool call as-is, embeds on-device, and recalls with sqlite-vec + FTS5 + RRF. Recall scores are RRF rank-normalized (not relevance probabilities). It does **not** summarize, does **not** talk to a model, and has **no** team/org HTTP tier.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
haunt bootstrap
haunt observe "the auth token lives in src/auth/session.py"
haunt recall "where is the auth token"
haunt dash --port 7340
haunt cursor-install
```

`lore` and `engram` still work as deprecated CLI aliases (`lore observe`, `engram-mcp`, …). `haunt bootstrap` creates `~/.haunt/{namespaces,bin,models}`, writes launchers at `~/.haunt/bin/haunt-mcp` and `~/.haunt/bin/haunt-hook`, probes sqlite-vec, downloads the embedding model, and inits the `default` namespace.

Env: `HAUNT_HOME` (default `~/.haunt`; falls back to `~/.lore` if it exists; `LORE_HOME` / `ENGRAM_HOME` accepted as legacy aliases), `HAUNT_EMBED_MODEL`, `HAUNT_NAMESPACE` (`LORE_NAMESPACE` / `ENGRAM_NAMESPACE` aliases), `HAUNT_FTS_ONLY=1`, `HAUNT_MODEL_CACHE`.

## Cursor hooks

Auto-store every prompt, reply, and tool call. Verbatim only — no LLM, no summaries. Fail-open (`{}` + exit 0) so a hook never blocks the agent.

**Hooks store; recall is not automatic.** The `beforeSubmitPrompt` hook observes the user prompt and runs recall, but Cursor does not inject the results into the model (`additional_context` is not in that hook's output schema). Agents must call `memory_recall` explicitly. When hooks are active, do **not** call `memory_observe` on content the hooks already stored (prompts, responses, tool calls) — that creates duplicates.

```bash
haunt cursor-install
```

This:

1. Publishes `~/.haunt/bin/haunt-hook` (a `/bin/sh` wrapper to the venv `haunt-hook` script; also writes `lore-hook` and `engram-hook` for backwards compat).
2. Merges user hooks at `~/.cursor/hooks.json` (version 1). Existing hook commands are kept. Adds:
   `beforeSubmitPrompt`, `afterAgentResponse`, `postToolUse`, `afterShellExecution`, `afterMCPExecution`, `sessionStart`, `sessionEnd`.

Project-level example: [`contrib/cursor/hooks.json`](contrib/cursor/hooks.json). Agent rule: [`contrib/cursor/haunt.mdc`](contrib/cursor/haunt.mdc) — copy into `.cursor/rules/` if you want the agent to `memory_recall` when hook context is missing.

| hook | what |
|---|---|
| `beforeSubmitPrompt` | observe user prompt (episodic), recall k=8. **Note:** Cursor's output schema for this hook is `{continue, user_message}` only — `additional_context` is returned optimistically but silently ignored. Per-turn recall is **not** injected into the model. |
| `afterAgentResponse` | observe assistant text |
| `afterAgentThought` | skipped by default (`HAUNT_STORE_THOUGHTS=1` to store as system/coordinate) |
| `postToolUse` | observe tool_name + input + output (tier=procedural; see **tier caveat** below) |
| `afterShellExecution` | observe command + output (`tool_name=Shell`) |
| `afterMCPExecution` | observe MCP call; skips `memory_*` to avoid recursion |
| `sessionStart` | tiny session-open event + last 5 memories + MCP reminder |
| `sessionEnd` | close session; no summary |

**Secret redaction:** Hook-stored tool input and output are run through a best-effort denylist that redacts common secret patterns (API keys, bearer tokens, AWS access keys, GitHub PATs, JWTs, etc.). This is **not** exhaustive — do not rely on it as a security boundary. If a tool returns sensitive material you must not persist, avoid passing secrets through `memory_observe`.

Namespace is inferred from `CURSOR_PROJECT_DIR` / git / cwd. Session id is `conversation_id` or `session_id`.

**Tier caveat:** `postToolUse`, `afterShellExecution`, and `afterMCPExecution` currently store all tool I/O as `tier=procedural`. This is a lane mix — `procedural` is meant for named how-tos (`meta.kind=procedure`), not generic tool-call logs. A future version should store hook-originated tool events as `episodic` (or `coordinate`) unless the event carries `meta.kind=procedure`. This does not affect recall correctness but pollutes the procedural tier's semantic meaning.

## Embeddings

Default is **`BAAI/bge-m3`** dense (1024-d). haunt loads it locally via **onnxruntime + the model's tokenizer** (no API key):

1. Official ONNX from Hugging Face (`BAAI/bge-m3` `onnx/model.onnx` + `onnx/model.onnx_data` + `onnx/tokenizer.json`) into `~/.haunt/models/BAAI-bge-m3` (~2.28 GB).
2. If that download fails, a community INT8 ONNX of the same model.
3. If a newer fastembed build lists `BAAI/bge-m3`, that path is tried next.
4. Automatic fallback: `BAAI/bge-small-en-v1.5` (384-d ONNX via fastembed, ~67 MB).

`HAUNT_EMBED_MODEL` is the switch (`BAAI/bge-m3` by default; set `BAAI/bge-small-en-v1.5` to force the small model; `off` / `HAUNT_FTS_ONLY=1` for FTS-only). Embeddings are never faked.

Namespace DBs store `embed_model` / `embed_dim` in `meta`. A dim or model change drops `vec_memories` and rebuilds every row — run **`haunt bootstrap --reembed`** after switching models (observe/recall also auto-rebuild a stale namespace so 384-d demo data does not silently mix with 1024-d queries).

## CLI

| command | what |
|---|---|
| `haunt init [name] [--repo PATH]` | create a namespace |
| `haunt observe TEXT ...` | store a turn / tool call verbatim |
| `haunt recall QUERY [--as-of --since --until --tier --k]` | hybrid recall |
| `haunt timeline` | events by `event_time` |
| `haunt namespaces` | list + counts |
| `haunt health` | vec / embed / counts |
| `haunt graph [--entity] [--rebuild]` | entities + relations; `--rebuild` wipes graph and re-extracts from events |
| `haunt dash [--port 7340]` | local dashboard (127.0.0.1) |
| `haunt bootstrap [--reembed]` | first-run setup; `--reembed` rebuilds all namespace vectors |
| `haunt cursor-install` | merge Cursor user hooks → `~/.haunt/bin/haunt-hook` |

Deprecated aliases `lore` and `engram` still work for all commands. Observe flags: `--namespace --tier --session --role --tool-name --tool-input --tool-output --event-time`.

## Cursor MCP

After `haunt bootstrap` (and `pip install -e .` so `python -m haunt.mcp_server` imports):

```json
{
  "mcpServers": {
    "haunt": {
      "command": "/home/box/.haunt/bin/haunt-mcp",
      "env": {
        "HAUNT_HOME": "/home/box/.haunt",
        "HAUNT_EMBED_MODEL": "BAAI/bge-m3",
        "HAUNT_MODEL_CACHE": "/home/box/.haunt/models"
      }
    }
  }
}
```

`haunt bootstrap` prints the absolute launcher path. The file is a `/bin/sh` wrapper around the venv `haunt-mcp` console script (stdio). `~/.haunt/bin/lore-mcp` and `~/.haunt/bin/engram-mcp` are legacy aliases for the same server. Tools: `memory_observe`, `memory_recall`, `memory_timeline`, `memory_health`, `memory_namespaces`, `memory_session_end` (closes a session; **no** distillation).

## What v1 does / does not

**Does:** verbatim events + memories, bi-temporal `event_time` / `valid_from` / `valid_to`, namespaces as isolated SQLite files, deterministic graph extract (regex / noun-phrase / paths), hybrid recall, CLI, MCP, Cursor hooks, local dashboard.

**Does not:** summarize or distill, call any reader-LLM, talk to the network at query time, require Docker/Postgres, expose a team HTTP API, auto-store in non-Cursor environments. Claude Code, Grok Bot, and Codex have no Cursor-style hooks — in those environments the agent must call `memory_observe` and `memory_recall` manually via MCP. (MCP is generic; Cursor is the only documented hooks integration.)

## Layout

```
~/.haunt/registry.db
~/.haunt/namespaces/<name>.db
~/.haunt/bin/haunt-mcp
~/.haunt/bin/haunt-hook
~/.haunt/bin/lore-mcp      (legacy alias)
~/.haunt/bin/engram-mcp    (legacy alias)
~/.haunt/bin/engram-hook   (legacy alias)
~/.haunt/bin/lore-hook     (legacy alias)
~/.haunt/models/
```

Default namespace is inferred from `CURSOR_PROJECT_DIR` / `git remote` / repo folder / cwd, else `default`. Home dir is `~/.haunt` (`HAUNT_HOME`); falls back to `~/.lore` if it exists and `~/.haunt` does not. `LORE_HOME` / `ENGRAM_HOME` accepted as legacy aliases.
