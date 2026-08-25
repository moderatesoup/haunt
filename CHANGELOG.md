# Changelog

## Unreleased

New timestamps keep microseconds (or finer) and stay UTC. Opening a namespace runs a schema-versioned one-time rewrite of offset/naive clocks; queries do not rewrite history. Newest-procedure / latest-row picks break ties with `rowid`, not second-resolution `created_at` alone. (#67)

MCP and CLI reads/mutations use `open_existing`: unknown namespaces fail loud and do not create a `*.db`. Only `init`, first `observe`, procedure write, and bootstrap create. (#68)

Dashboard POST contradict rejects non-JSON (415), invalid/non-object JSON and non-string `replacement` (400) without setting `valid_to`. `Store.contradict` raises `ValueError` if `replacement` is not a string or null. `HAUNT_FTS_ONLY=1` / `HAUNT_EMBED_MODEL=off` bootstrap no longer fatals on a failed sqlite-vec probe; layout + default namespace still come up FTS-only, without downloading BGE-M3. Doctor treats sqlite-vec as optional in that mode. (#64)

Honesty pack (#61): dashboard `openDetail` escapes every `.val` field (including `session_id`); host hook/settings merge fail-closed on malformed JSON; `Store.contradict` is one transaction and refuses already-superseded rows; timeline fills `k` current memories; generated wrappers quote `HAUNT_HOME`; CLI timeline / worldview size params use the same clamp helpers as recall `k`.

Doctor honesty: `{host}.hooks` fails when the planted haunt-hook path is missing or is not the expected wrapper. Leaf name alone is not enough. Runtime hook fail-open is unchanged.

DELETE `/api/namespace/{name}/memory/{id}` and POST contradict 404 `unknown namespace` on a missing namespace (`Store(..., create=False)`). A typo no longer creates an empty DB, then 404s `memory not found`.

GET `/api/recall` (all-namespaces) attaches a non-empty `errors` list when a registered namespace fails to open or search, and the console recall meta surfaces those failures instead of only `N hits (all namespaces)`.

Release-gate hardening (storage isolation honesty, private HAUNT_HOME modes, mcp>=2 pin, loopback dashboard bind, dashboard XSS/GET-create/limit clamps, Authorization Bearer hook redaction).

## 0.2.0 — 2026-08-24

Naming cleanup. Product name is **haunt** only.

- Removed `lore` / `engram` console scripts and `~/.haunt/bin` wrappers. Entry points are `haunt`, `haunt-mcp`, `haunt-hook`, `haunt-hook-claude`.
- Host doctor accepts `haunt-mcp` / `haunt-hook` only. `lore-mcp` in `mcp.json` is not a haunt command.
- User-facing home is `~/.haunt` / `HAUNT_HOME`. Dropped `LORE_*` / `ENGRAM_*` env aliases and the `~/.lore` fallback.
- Install from git: `pip install git+https://github.com/moderatesoup/haunt.git`. `pip install haunt` on PyPI is not this project (mikepqr's stow).
- Deleted stale `WIRING_REVIEW.md` (PR #2, six-tool era).
