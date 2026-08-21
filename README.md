# haunt

Local-first **verbatim** memory for AI agents. One SQLite file per namespace. No cloud, no distillation, no reader-LLM.

[Apache-2.0 License](LICENSE)

## 60-second start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
haunt bootstrap          # creates ~/.haunt, probes sqlite-vec, downloads BAAI/bge-m3 (~2.28 GB)
haunt observe "the auth token lives in src/auth/session.py"
haunt recall "where is the auth token"
```

To wire up Cursor hooks:

```bash
haunt cursor-install     # merges hooks at ~/.cursor/hooks.json
```

Then point your MCP client at `~/.haunt/bin/haunt-mcp`.

Or run `scripts/bootstrap.sh` for a one-command clone-and-go setup.

### What is and isn't automatic

- **Hooks store automatically.** Once `haunt cursor-install` runs, every prompt, response, and tool call is stored verbatim via Cursor hooks.
- **Recall is NOT automatic.** Cursor ignores `additional_context` from `beforeSubmitPrompt`. Agents must call `memory_recall` explicitly via MCP.
- **No-hook IDEs** (Claude Code, Codex, etc.): call `memory_observe` and `memory_recall` manually via MCP. There is no Cursor-style hook integration for other environments.

## Embeddings

Default is **`BAAI/bge-m3`** dense (1024-d, ~2.28 GB). haunt loads it locally via onnxruntime (no API key). First `haunt bootstrap` downloads the model from Hugging Face.

Namespaces lock to the embed model/dimension they were written with. Switching models later requires a full re-embed (`haunt bootstrap --reembed`).

For machines that cannot spare 2 GB, use the small model as an explicit opt-in:

```bash
HAUNT_EMBED_MODEL=BAAI/bge-small-en-v1.5 haunt bootstrap
```

For FTS-only (no embeddings at all):

```bash
HAUNT_FTS_ONLY=1 haunt bootstrap
```

## Cursor hooks

Auto-store every prompt, reply, and tool call. Verbatim only — no LLM, no summaries. Fail-open (`{}` + exit 0) so a hook never blocks the agent.

**Secret redaction:** Hook-stored tool input and output are run through a best-effort denylist (API keys, bearer tokens, AWS keys, GitHub PATs, JWTs, etc.). This is **not** a security boundary — see [SECURITY.md](SECURITY.md).

| hook | what |
|---|---|
| `beforeSubmitPrompt` | observe user prompt, recall k=8 (results returned but Cursor ignores `additional_context`) |
| `afterAgentResponse` | observe assistant text |
| `postToolUse` | observe tool_name + input + output |
| `afterShellExecution` | observe command + output |
| `afterMCPExecution` | observe MCP call (skips `memory_*` to avoid recursion) |
| `sessionStart` | session-open event + last 5 memories + MCP reminder |
| `sessionEnd` | close session; no summary |

## CLI

| command | what |
|---|---|
| `haunt bootstrap [--reembed]` | first-run setup; exits 1 if sqlite-vec fails |
| `haunt init [name] [--repo PATH]` | create a namespace |
| `haunt observe TEXT ...` | store a turn / tool call verbatim |
| `haunt recall QUERY [--as-of --since --until --tier --k]` | hybrid recall (vec + FTS5 + RRF) |
| `haunt timeline` | events by `event_time` |
| `haunt namespaces` | list + counts |
| `haunt health [-n NAMESPACE]` | vec / embed / counts / db path |
| `haunt graph [--entity] [--rebuild]` | entities + relations |
| `haunt dash [--port 7340]` | local dashboard (127.0.0.1) |
| `haunt cursor-install` | merge Cursor user hooks |

## MCP

After bootstrap, configure your MCP client:

```json
{
  "mcpServers": {
    "haunt": {
      "command": "~/.haunt/bin/haunt-mcp"
    }
  }
}
```

Tools: `memory_observe`, `memory_recall`, `memory_timeline`, `memory_health`, `memory_namespaces`, `memory_session_end`.

## Environment variables

| variable | default | what |
|---|---|---|
| `HAUNT_HOME` | `~/.haunt` | data directory |
| `HAUNT_EMBED_MODEL` | `BAAI/bge-m3` | embedding model |
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

## What v1 does / does not

**Does:** verbatim events + memories, bi-temporal filtering, namespaces as isolated SQLite files, deterministic graph extract, hybrid recall, CLI, MCP, Cursor hooks, local dashboard.

**Does not:** summarize or distill, call any LLM, talk to the network at query time, require Docker/Postgres, expose a team HTTP API, auto-recall into the model context.
