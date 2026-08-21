# haunt — local-first agent memory

## What it is

haunt (Python package `lore`) gives AI agents persistent verbatim memory
backed by SQLite. No cloud, no API keys, no LLM summarization. One file
per namespace, embeddings computed on-device.

## MCP tools (server name: `lore`)

| Tool | Purpose |
|---|---|
| `memory_recall` | Hybrid search (vec + FTS5 + RRF) over stored memories |
| `memory_observe` | Store a verbatim turn or tool call |
| `memory_worldview` | Compact namespace briefing: facts, entities, procedures, counts |
| `memory_procedure` | Named how-to procedures — action: `write` / `get` / `list` |
| `memory_contradict` | Mark a memory superseded, optionally store replacement |
| `memory_timeline` | List events in time order |
| `memory_health` | Namespace health and counts |
| `memory_namespaces` | List all namespaces |
| `memory_session_end` | Close a session (no distillation) |

## Usage rules

### Every turn
1. **Recall first.** Call `memory_recall` with the user's exact wording
   before acting. If worldview context is already present from session
   start, you may skip the redundant recall.
2. **Observe after.** Let hooks auto-store. If hooks are absent, call
   `memory_observe` with verbatim text. Use `tier=episodic` for chat,
   `tier=semantic` for durable facts.
3. **Never summarize.** Pass text as-is into memory.

### Worldview
- Call `memory_worldview` when you need the full namespace briefing.
- The `sessionStart` hook returns a compact worldview card automatically
  via `additional_context`.

### Procedures
- Use `memory_procedure` action=write only when deliberately promoting a
  how-to the user wants remembered.
- Provide: `name` (short identifier), `body` (verbatim steps), optional
  `trigger` (one-liner: "when X happens").
- Do NOT auto-extract procedures from every turn.

### Contradict
- Use `memory_contradict` when a stored fact is now wrong.
- Pass the `memory_id` of the old row. Optionally pass `replacement`
  text to store the corrected fact as semantic.

### Don'ts
- Do not observe `memory_*` tool calls (hooks already skip them).
- Do not invent namespaces. Namespace is inferred from the project.
- Do not call any external API — haunt is fully local.

## CLI quick reference

```bash
engram worldview                          # namespace briefing
engram procedure write NAME --body "..."  # store a procedure
engram procedure get NAME                 # retrieve by name
engram recall "search query"              # hybrid search
engram observe "fact text" --tier semantic # store a fact
```

`lore` is an alias for `engram` (all commands work with either name).

## Architecture

- Home: `~/.lore` (env `LORE_HOME` / `ENGRAM_HOME`)
- Registry: `~/.lore/registry.db`
- Namespace DBs: `~/.lore/namespaces/<name>.db`
- Embeddings: on-device ONNX (BAAI/bge-m3 default, bge-small fallback)
- No network calls at query time
