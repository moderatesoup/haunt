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

The canonical export download is covered by the same token gate. Dashboard
import is an administrative mutation and additionally requires a trusted
`Origin`, an accepted JSON media type, and a bounded body. The HTML console has
no import/paste control in v1; callers use the authenticated API deliberately.

## Secret redaction

Cursor hook input and output are run through a best-effort denylist that redacts common secret patterns (API keys, bearer tokens, AWS access keys, GitHub PATs, JWTs, Slack tokens, etc.).

**This is not a security boundary.** The denylist is pattern-based and cannot catch every secret format. If a tool returns sensitive material you must not persist, do not pass it through hook-stored events.

Guidance:

- Do not point `haunt observe` at `.env` files, credential stores, or token outputs.
- Treat `~/.haunt/namespaces/*.db` files as sensitive — they contain verbatim agent conversation history.
- Treat `haunt.namespace-export` JSON files as equally sensitive. They contain
  verbatim surviving history and audit metadata. Mode 0600 is defense in depth;
  it is not encryption.
- The redaction layer is defense-in-depth, not a guarantee.

Canonical export digests detect corruption and make exact import replay
identifiable. They are unkeyed hashes, not signatures or authenticity claims.
Export excludes local paths, embeddings, indexes, and previously purged bytes,
but anyone who can read a bundle can read its surviving plaintext content.
An opaque namespace history head rotates in the hard-purge transaction, so an
older or independently diverged bundle cannot be replayed into that namespace
to restore erased rows. The head is not a secret, signature, or defense against
a same-user attacker who can directly rewrite Haunt's files. Interrupted fresh
imports recover only files still matching the fsynced intent's exact token,
digest, device, and inode ownership; replacement links fail closed.
MCP export/import therefore require `HAUNT_MCP_ADMIN=1`; the dashboard launch
token similarly grants administrative transfer access to every namespace.

## Persistent recalled content and prompt injection

Stored text is untrusted data. In particular, tool input/output can contain hostile instructions copied from files, web pages, command output, or another MCP server.

- Hooks exclude raw tool I/O from automatically injected recall and worldview content. Hook recall is FTS-only and never initializes the embedding model.
- Ranked recall excludes raw tool structure and explicitly classified `tool`/`task` residue by default. An operator must opt in with `--include-residue` / `include_residue=true` for audit/search. Timeline, trace, and detail remain audit surfaces and do not pretend this ranked filter applies to them.
- Explicitly included tool I/O is marked `trusted=false` with `trust_reason=untrusted-tool-io`; task residue remains data and is never an instruction.
- Recalled content cannot authorize `observe`, `contradict`, `purge`, shell commands, or any other mutation. MCP purge has its own launch-time capability and remains off by default.
- New hook tool-input/output fields are capped at 12,000 characters by default. Set `HAUNT_TOOL_IO_MAX_CHARS` to a smaller value, and use comma-separated `HAUNT_EXCLUDE_TOOLS` globs for tools whose output must never be stored.

These controls reduce persistent prompt-injection exposure. They do not make arbitrary recalled text safe, and they do not rewrite or delete tool rows stored by older versions.

## Read-only recall and offline operation

Recall opens the E3-selected identity through a guarded zero-write snapshot.
It does not migrate schemas, configure WAL, rebuild graph evidence, tighten
permissions, write the registry, or drain embedding jobs. `haunt maintenance`
is the separately named and intentionally mutating embedding upgrade/job-drain
surface. Its use is visible in its own output, never hidden in a query.

`HAUNT_OFFLINE=1` stops before optional embedding/model libraries are imported,
preventing model initialization/download and their network paths even when
ambient provider tokens are present. FTS recall remains local; vector execution
is reported honestly as not run.

## Fail-open hooks

Cursor hooks are **fail-open**: if a hook errors, it prints `{}` and exits 0. A hook will never block your agent or prevent a prompt from being submitted. This is a deliberate trade-off — reliability of the agent takes priority over memory completeness.

## File-per-namespace isolation

Each namespace is a **separate SQLite file** under `~/.haunt/namespaces/`. That is **storage isolation** (separate files, tables, connections, and queries) — **not authorization**. A `recall` in namespace A cannot return rows from namespace B's file.

This is not a security kernel. An ordinary MCP process is blast-radius-limited to one immutable canonical namespace identity, inferred from the full git remote identity (`host/owner/repo`) or set with `HAUNT_NAMESPACE`; a proven alias of that identity is accepted, but aliases never grant access to another identity. Cross-namespace requests are denied and namespace listing is filtered. An intentionally separate process launched with `HAUNT_MCP_ADMIN=1` may cross/list namespaces. MCP hard purge is disabled unless `HAUNT_MCP_ALLOW_PURGE=1` is also set.

Those process capabilities are guardrails, not operating-system authorization. The same local user can still open every SQLite file directly or use the CLI (`haunt recall -n …`, confirmed `haunt delete`). A compromised same-user process can do the same. Do not present namespace binding as protection from the local account that owns `HAUNT_HOME`.

Haunt serializes its own writable SQLite opens across processes and brackets the first write-mode pragma with held primary/sidecar identity checks. This prevents cooperating Haunt writers from accidentally replacing WAL/SHM paths during configuration. The lock is advisory: an arbitrary same-user process can ignore it and rename filesystem entries in the remaining system-call-sized check/use interval. That case is outside Haunt's practical corruption guard and belongs to operating-system account isolation.

## Reporting a vulnerability

Open an issue at <https://github.com/moderatesoup/haunt/issues> or email the maintainer directly. There is no bug bounty program.
