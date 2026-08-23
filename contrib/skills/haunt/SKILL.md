# haunt — local-first verbatim memory

MCP server name is `haunt`. Store is verbatim. Never summarize. Never distill.

## Automatic vs not

| Automatic | Not automatic |
|---|---|
| Hooks store prompts, replies, tool I/O (skip `memory_*`) | **Recall** — you must call `memory_recall` |
| `sessionEnd` / `SessionEnd` close the session | `sessionStart` worldview — may or may not inject `[haunt worldview ns=…]` |

If no `[haunt ns=…]` block is visible, call `memory_recall` with the user's exact wording before acting. Recall is not automatic.

## Temporal — one path

**`compile() runs automatically on memory_recall`.** Pass the user's wording. Do not compute `since`/`until` yourself. Do not do date arithmetic.

- Clock is `event_time` for normal user-facing time (including speech verbs: said, mentioned, told).
- Do not filter on storage `ts`. `storage_time` is ingest time only (operational "what did haunt ingest…").
- Default recall hides superseded rows (`valid_to IS NULL`) unless you pass `as_of`.
- Topical leftover after compile → hybrid recall. Bare time ("what happened last week") → timeline internally. `union` is experimental — do not request it.
- `memory_timeline` does **not** compile natural language. Use it only with ISO `since`/`until` or a session dump.

## Tools

### `memory_recall`
`query` (user wording), optional `as_of`, `since`, `until`, `clock`, `k`
Call before acting unless a `[haunt ns=…]` block is already visible. RRF scores are rank-normalized, not relevance — ignore off-corpus hits.

### `memory_observe`
`text`, `tier` (`episodic` chat / `semantic` durable fact), `origin`, `session`
Call only when hooks are absent. Never summarize. Never double-observe what hooks already stored.

### `memory_worldview`
optional `facts_cap`, `names_cap`
Call when no `[haunt worldview ns=…]` card is in context.

### `memory_procedure`
`action` = `write` / `get` / `list`. Write needs `name`, `body`, optional `trigger`.
Write only when the user wants a named how-to remembered. Do not auto-extract.

### `memory_contradict`
`memory_id`, optional `replacement`
Supersede a wrong fact (`valid_to=now`). Does not delete. Optional `replacement` is stored as semantic.

### `memory_purge`
`memory_id`
Hard-delete the memory and its provenance. Data is gone. Use contradict to supersede.

### `memory_timeline`
optional `session`, `since`, `until`, `clock`, `limit`
ISO bounds or a session dump. No natural-language compile. Clock default is `event_time`.

### `memory_session_end`
optional `session`
Close a session. No distillation. If nothing was ended (missing, already ended, no current session), returns `ok: false` — treat that as failure.

### `memory_health`
optional `namespace`
Counts, sqlite-vec, embed status.

### `memory_namespaces`
No args. List namespaces.

## Observe skip list

- Secrets, tokens, API keys, passwords
- Acks: "ok", "got it", "sure", empty turns
- `memory_*` tool inputs/outputs (hooks already skip these)
- Entire files / READMEs — store a pointer, not the blob

## Namespace

Inferred from git / `CURSOR_PROJECT_DIR` / `CLAUDE_PROJECT_DIR` / cwd. Do not invent. When building haunt itself, namespace is `haunt`.

## No-hooks hosts (Grok Bot, Codex, …)

Observe each user turn (`tier=episodic`) and each durable fact (`tier=semantic`). Recall before acting. Pass `origin`.

## CLI

```bash
haunt recall "user wording"          # compile() runs automatically
haunt timeline --since ISO --until ISO --clock event_time
haunt observe "fact" --tier semantic --origin cli
haunt worldview
haunt procedure write NAME --body "..."
haunt delete MEMORY_ID -y            # purge; no CLI contradict / session-end
haunt health / namespaces / dash
```

`lore` and `engram` are aliases for `haunt`.
