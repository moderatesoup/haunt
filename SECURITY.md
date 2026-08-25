# Security

## Architecture

haunt is **local-only**. All data (SQLite databases, embeddings, models) stays on your machine under `~/.haunt/` (or `$HAUNT_HOME`). haunt never phones home, never opens a port (except the optional local dashboard on 127.0.0.1), and never sends data to any remote service. The only network call is the one-time model download from Hugging Face during `haunt bootstrap`.

## Dashboard bind, Host, Origin, and launch token

`haunt dash` binds **127.0.0.1** by default. Loopback bind is not enough on its own:

- Requests whose `Host` is not a trusted loopback name (`127.0.0.1`, `localhost`, `::1`), the configured bind host, or a literal IP reached through an explicit wildcard bind are rejected (400). Arbitrary DNS names remain rejected, which blocks DNS rebinding while keeping `0.0.0.0 --allow-remote` usable from a LAN IP.
- Every `/api` route, including GET, requires the launch token (`X-Haunt-Token` header or `?token=`). Missing or wrong token is 401. The HTML index can still load without the token; the API is gated.
- Loopback bind (default 127.0.0.1) injects the token into the console HTML so the local UI works. **`--allow-remote` / a non-loopback bind does not embed the token in HTML.** GET `/` with a trusted Host is not enough to obtain `X-Haunt-Token`. The token is printed only on `haunt dash` stdout for the operator who launched it.
- Cookie-less mutation routes (`DELETE` memory, `POST` contradict) also validate `Origin` when present. Same-origin and missing-Origin local TestClient requests still work. Cross-origin form posts are rejected.
- `haunt dash` mints a random launch token at start and prints it. `--allow-remote` without that token configured refuses to start (or every `/api` route returns 401).
- **`--allow-remote` is unsafe without the token.** It exposes the local memory admin API on the network. Anyone who has the token can read and mutate every namespace. **Namespaces are still not authorization** — see below.

Do not add Docker, Postgres, or HTTP team-tier auth. This is a local-first console.

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

This is not a security kernel. An ordinary MCP process is blast-radius-limited to one immutable namespace, inferred from the full git remote identity (`host/owner/repo`) or set with `HAUNT_NAMESPACE`; cross-namespace requests are denied and namespace listing is filtered. An intentionally separate process launched with `HAUNT_MCP_ADMIN=1` may cross/list namespaces. MCP hard purge is disabled unless `HAUNT_MCP_ALLOW_PURGE=1` is also set.

Those process capabilities are guardrails, not operating-system authorization. The same local user can still open every SQLite file directly or use the CLI (`haunt recall -n …`, confirmed `haunt delete`). A compromised same-user process can do the same. Do not present namespace binding as protection from the local account that owns `HAUNT_HOME`.

## Reporting a vulnerability

Open an issue at <https://github.com/moderatesoup/haunt/issues> or email the maintainer directly. There is no bug bounty program.
