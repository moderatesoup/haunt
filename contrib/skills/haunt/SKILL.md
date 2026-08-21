# haunt — local-first agent memory

## What it is

haunt gives AI agents persistent verbatim memory
backed by SQLite. No cloud, no API keys, no LLM summarization. One file
per namespace, embeddings computed on-device.

## Core principle

**Hooks store; agents must recall unless context was injected.**

Cursor hooks automatically log every turn. But recall is NOT automatic —
the agent must call `memory_recall` unless a `[haunt ns=…]` block is
already visible in context.

## MCP tools (server name: `haunt`)

| Tool | Purpose |
|---|---|
| `memory_recall` | Hybrid search (vec + FTS5 + RRF) over stored memories |
| `memory_observe` | Store a verbatim turn or tool call |
| `memory_worldview` | Compact namespace briefing: facts, entities, procedures, counts |
| `memory_procedure` | Named how-to procedures — action: `write` / `get` / `list` |
| `memory_contradict` | Mark a memory superseded, optionally store replacement |
| `memory_purge` | Permanently hard-delete a memory and its provenance chain |
| `memory_timeline` | List events in time order |
| `memory_health` | Namespace health and counts |
| `memory_namespaces` | List all namespaces |
| `memory_session_end` | Close a session (no distillation) |

## Hooks vs. agent actions

| Layer | What it does | Agent action needed? |
|---|---|---|
| Cursor hooks | Auto-log prompts, replies, tool calls, shell, MCP | No — automatic |
| `beforeSubmitPrompt` recall | Attempts `additional_context` injection | Unproven — do not rely on it |
| `sessionStart` worldview | Injects `[haunt worldview ns=…]` card | Check if present; if not, call `memory_worldview` |
| Per-turn recall | Fetch relevant memories | YES — agent must call `memory_recall` |
| Manual observe | Store facts when hooks absent | Only when hooks unavailable |

## Usage rules

### Recall
1. If no `[haunt ns=…]` block is visible in context, call
   `memory_recall` with the user's exact wording before acting.
2. RRF scores are rank-normalized, not relevance. A score of 0.03 means
   "ranked low", not "3% relevant." Ignore hits that are clearly
   off-corpus rather than trusting the number.

### Observe (when hooks are absent)
- tier=episodic for chat turns.
- tier=semantic for durable facts.
- Always pass `origin` (e.g. "cursor", "cli", "grok") and `session` id.
- Never summarize or distill. Store verbatim or don't store.

### Do NOT observe (skip list)
- Secrets, tokens, API keys, passwords.
- Acks and empty turns: "ok", "got it", "sure", "hey".
- `memory_*` tool inputs/outputs (hooks already skip these).
- Entire READMEs or large file contents — store a pointer, not the blob.
- Never paste whole files into memory.

### Do NOT double-observe
When hooks are active (Cursor with haunt installed), they log turns
automatically. Do not also call `memory_observe` on the same content.

### Worldview
- Call `memory_worldview` for the full namespace briefing.
- `sessionStart` hook tries to inject a compact card — verify it arrived
  by looking for `[haunt worldview ns=…]` in context.

### Procedures
- `memory_procedure` action=write only when deliberately promoting a
  specific how-to the user wants remembered.
- Provide: `name` (short identifier), `body` (verbatim steps), optional
  `trigger` (one-liner: "when X happens").
- Do NOT auto-extract procedures from every turn.

### Contradict
- `memory_contradict` with the `memory_id` of the old row.
- Optionally pass `replacement` to store the corrected fact as semantic.

### Purge (hard delete)
- `memory_purge` with `memory_id` to permanently delete a memory.
- Removes the memory, FTS index, vector embedding, graph relations,
  orphaned entities, and the event if nothing else references it.
- Data is gone — not just superseded. Use contradict to supersede.

## No-hooks environments (Grok Bot, Claude Code, etc.)

These clients have no hook support. The agent must:
1. Observe each user turn (tier=episodic) and durable facts (tier=semantic).
2. Recall before acting.
3. Pass `origin` to identify the client.

## CLI quick reference

```bash
haunt worldview                          # namespace briefing
haunt procedure write NAME --body "..."  # store a procedure
haunt procedure get NAME                 # retrieve by name
haunt recall "search query"              # hybrid search
haunt observe "fact text" --tier semantic # store a fact
haunt delete MEMORY_ID -y                # hard-delete a memory
haunt dash                               # open local memory console
```

`lore` and `engram` are aliases for `haunt` (all commands work with any name).

## Architecture

- Home: `~/.haunt` (env `HAUNT_HOME` / `LORE_HOME` / `ENGRAM_HOME`)
- Registry: `~/.haunt/registry.db`
- Namespace DBs: `~/.haunt/namespaces/<name>.db`
- Embeddings: on-device ONNX (BAAI/bge-m3 default, bge-small fallback)
- No network calls at query time
- No API keys required
