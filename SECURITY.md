# Security

## Architecture

haunt is **local-only**. All data (SQLite databases, embeddings, models) stays on your machine under `~/.haunt/` (or `$HAUNT_HOME`). haunt never phones home, never opens a port (except the optional local dashboard on 127.0.0.1), and never sends data to any remote service. The only network call is the one-time model download from Hugging Face during `haunt bootstrap`.

## Secret redaction

Cursor hook input and output are run through a best-effort denylist that redacts common secret patterns (API keys, bearer tokens, AWS access keys, GitHub PATs, JWTs, Slack tokens, etc.).

**This is not a security boundary.** The denylist is pattern-based and cannot catch every secret format. If a tool returns sensitive material you must not persist, do not pass it through hook-stored events.

Guidance:

- Do not point `haunt observe` at `.env` files, credential stores, or token outputs.
- Treat `~/.haunt/namespaces/*.db` files as sensitive — they contain verbatim agent conversation history.
- The redaction layer is defense-in-depth, not a guarantee.

## Fail-open hooks

Cursor hooks are **fail-open**: if a hook errors, it prints `{}` and exits 0. A hook will never block your agent or prevent a prompt from being submitted. This is a deliberate trade-off — reliability of the agent takes priority over memory completeness.

## File-per-namespace isolation

Each namespace is a **separate SQLite file** under `~/.haunt/namespaces/`. That is **storage isolation** (separate files, tables, connections, and queries) — **not authorization**. A `recall` in namespace A cannot return rows from namespace B's file.

This is not a security kernel. The same local user can open every namespace file via MCP (`memory_recall` with any `namespace`) or the CLI (`haunt recall -n …`). haunt does not authenticate callers or enforce per-namespace access control.

## Reporting a vulnerability

Open an issue at <https://github.com/moderatesoup/haunt/issues> or email the maintainer directly. There is no bug bounty program.
