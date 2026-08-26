# haunt

Local-first **verbatim** memory for AI agents. One SQLite file per namespace. No cloud, no distillation, no reader-LLM.

[Apache-2.0 License](LICENSE)

## 60-second start

```bash
# Option A: pip install from git (no clone needed)
# PyPI name `haunt` is taken (mikepqr's stow); git install is the path.
pip install git+https://github.com/moderatesoup/haunt.git
haunt bootstrap          # creates ~/.haunt; probes sqlite-vec + downloads embed model unless HAUNT_FTS_ONLY=1
haunt dash               # opens the memory console in your browser → http://127.0.0.1:7340

# Option B: one-command bootstrap (git clone)
scripts/bootstrap.sh

# Option C: manual (git clone)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
haunt bootstrap          # creates ~/.haunt; probes sqlite-vec + downloads BAAI/bge-m3 (~2.28 GB) unless HAUNT_FTS_ONLY=1
haunt dash               # opens the memory console in your browser → http://127.0.0.1:7340
```

To wire up all supported hosts (Cursor + Claude Code) in one command:

```bash
haunt install            # hooks + MCP + rules for Cursor and Claude Code
```

This registers haunt hooks, MCP server, agent rules, and the haunt skill for
each host — even if the host is not installed yet (config dirs are pre-seeded).
Re-run `haunt install` or `haunt doctor` after adding a new editor, or if an
installer rewrote a config. `haunt doctor` checks the files install wrote,
the `haunt-mcp` wrapper (not a PATH name), sqlite-vec load (or explicit FTS-only), and embed model
presence (or explicit FTS-only) and exits 1 if any check fails.

To wire up Cursor only:

```bash
haunt cursor-install     # hooks.json + mcp.json + haunt.mdc (Cursor only)
```

> **Note:** `haunt-mcp` is a stdio server — do not run it with `--help` or directly in a terminal; it reads JSON on stdin and will hang. Use it only as an MCP server command.

### macOS: use Homebrew Python, not pyenv

pyenv-compiled CPython typically lacks `--enable-loadable-sqlite-extensions`, which means `sqlite-vec` cannot load (`enable_load_extension` is missing or disabled). `haunt bootstrap` will correctly fail loud if this happens, unless `HAUNT_FTS_ONLY=1` (or `HAUNT_EMBED_MODEL=off`) is set — then bootstrap still creates the layout and a usable FTS-only default namespace.

The working pattern on macOS:

```bash
# Install Homebrew Python (has loadable-extension support out of the box)
brew install python@3.14      # or python@3.12, python@3.13

# Create a venv from Homebrew Python — NOT from pyenv Python
/opt/homebrew/bin/python3 -m venv ~/.haunt/venv
source ~/.haunt/venv/bin/activate
pip install -e .              # or: pip install git+https://github.com/moderatesoup/haunt.git
haunt bootstrap               # should succeed — sqlite-vec loads

# ~/.haunt/bin/haunt-mcp execs the venv Python, so MCP works everywhere
```

If you **must** use pyenv, rebuild with:

```bash
PYTHON_CONFIGURE_OPTS="--enable-loadable-sqlite-extensions" pyenv install 3.12
```

### What is and isn't automatic

- **Hooks store automatically.** Once `haunt install` runs, prompts and responses are stored verbatim in Cursor and Claude Code. Tool input/output is best-effort redacted, capped per field, and can be skipped with `HAUNT_EXCLUDE_TOOLS`. New writes also carry a versioned source-provenance envelope; hook writes record their actual channel and supplied tool/call IDs.
- **Recall is NOT automatic.** Neither Cursor nor Claude Code reliably inject recall into the model context. Agents must call `memory_recall` explicitly via MCP unless a `[haunt ns=…]` block is already visible.
- **sessionStart / SessionStart** may return a worldview card — this may appear as context but is not a guaranteed recall path and is not a kernel.
- **No-hook IDEs** (Grok Bot, Codex, etc.): call `memory_observe` and `memory_recall` manually via MCP.

## Memory console (dashboard)

`haunt dash` serves a local-only memory management UI at `http://127.0.0.1:7340` and opens it in your default browser once the server is ready. Pass `--no-open` to suppress the browser (useful for CI or scripting). No React, no npm — single-file inline HTML.

The console binds **127.0.0.1** by default. At start it mints a random launch token, prints it, and requires it (`X-Haunt-Token` or `?token=`) on every `/api` route, including GET. The HTML page can still load locally; the API is gated. On loopback the page injects the token so the local UI works. `--allow-remote` / a non-loopback bind does **not** embed the token in HTML — it is printed only on `haunt dash` stdout. Requests with an untrusted `Host` (DNS rebind) are rejected. Mutation routes also check `Origin`.

`--allow-remote` binds beyond loopback and is **unsafe without the launch token** — it exposes the memory admin API on the network. Namespaces are storage isolation, not authorization. See [SECURITY.md](SECURITY.md).

Features:

- **Browse** memories with filters: tier, origin, session, time range. Paginated.
- **Search / recall** with hybrid vec+FTS. Results link to detail view. Clicking any result row opens the detail panel.
- **All-namespaces search**: select "all namespaces" in the sidebar to fan out a single query across every registered namespace. Results are namespace-grouped, with each namespace retaining its own ranking; RRF scores are not compared across namespaces. The response includes `ranking_scope="per_namespace"`, `namespace_groups`, and a compatibility `hits` list in that same deterministic group order. API: `GET /api/recall?q=&k=&tier=` (`k` applies per namespace).
- **Memory detail** with source context and correction trace: structured provenance, origin, session_id, event_id, memory_id, role, tier, event_time, valid_from/valid_to, ordered lineage, db_path (absolute), haunt_home, tool name/input/output, related memories from the same session, entity mentions. Pre-v8 rows are labeled `legacy_unstructured`; their original origin/meta are not guessed into import fields.
- **Timeline** view: events in time order with since/until day filters. Rows click through to memory detail.
- **Time-bounded search**: `as_of`, `since`, `until` date filters on recall (both per-namespace and all-namespaces).
- **Supersede** a memory from the detail panel (append-only correction record plus `valid_to=now`, optional replacement text). Keeps original data — distinct from delete.
- **Delete** a memory from the UI (with confirmation). Hard purge removes the memory, FTS index, vector embedding, graph rows, and orphan events. Recall will not return deleted content. If surviving correction history crosses the erased member, trace shows only a fresh opaque four-field tombstone; purge is the explicit exception to ordinary append-only lineage. Every target or adjacent correction session ID is rekeyed when surviving events still use it; unrelated event content/origin and clean session metadata remain intact while erased context is sanitized.
- **Procedures and worldview** visible (browse, view detail; write stays CLI/MCP).
- **Health** strip: sqlite-vec status/version, embed model+dim+availability, last write age, event count, namespace, absolute db_path. Always visible (persistent header), live-updating (15s poll).

### Desktop shortcut

`haunt bootstrap` automatically installs a desktop shortcut. You can also install or reinstall it manually:

```bash
haunt dash --install-icon
```

Writes a real shortcut: Linux `.desktop` file (`Haunt Memories` → `haunt dash`), macOS `.command` script, or Windows `.bat`. The shortcut launches `haunt dash`, which starts the server and opens the browser.

## Embeddings

Default is **`BAAI/bge-m3`** dense (1024-d, ~2.28 GB ONNX). haunt loads it locally via onnxruntime (no API key). First `haunt bootstrap` downloads the model from Hugging Face.

Namespaces lock to the embed model/dimension they were written with. Switching models later requires a full re-embed (`haunt bootstrap --reembed`). A namespace switch = full re-embed of that namespace.

For machines that cannot spare 2 GB, use the small model as an explicit opt-in:

```bash
HAUNT_EMBED_MODEL=BAAI/bge-small-en-v1.5 haunt bootstrap
```

For FTS-only (no embeddings, sqlite-vec not required):

```bash
HAUNT_FTS_ONLY=1 haunt bootstrap
```

This still creates `~/.haunt` and the default namespace. It does not download BGE-M3 and does not fail if sqlite-vec cannot load. `HAUNT_EMBED_MODEL=off` is the same FTS gate. CI uses this to avoid the 2 GB download.

### Read-only recall, residue, and offline mode

`haunt recall`, Python `recall()`/`planned_recall()`, MCP `memory_recall`, and
dashboard recall resolve a registered E3 identity and read through a guarded
zero-write SQLite snapshot. They do not migrate schema, configure WAL, rebuild
the graph, change permissions, touch the registry, or drain embedding jobs.
Their `execution` metadata reports `read_only=true`,
`maintenance_performed=false`, and any embedding jobs only as observed.

Use `haunt maintenance -n NAME --limit 64` for the explicitly mutating,
bounded embedding upgrade/job drain. It opens an existing namespace only, so a
typo cannot create a database. In `HAUNT_OFFLINE=1`, lexical FTS recall still
works while vectors are reported `not_run` with reason `offline_mode`; Haunt
does not initialize/download an embedding backend even with ambient provider
tokens.

Schema v9 has a nullable `events.recall_class` constrained to `tool|task`.
There is no historical backfill: null means legacy/unknown and is eligible.
The tool role or raw tool fields automatically stamp `tool`; incompatible supplied classes fail
before write. Hooks stamp only their actual session-start coordinate event as
`task`, and corrections preserve the target's effective class (including a
legacy raw-tool target). Ranked recall excludes raw
tool structure and `tool`/`task` residue by default. Use `--include-residue`
for an intentional audit/search request; timeline, trace, and detail remain
available. Python's `include_untrusted` is a deprecated compatibility alias
only when `include_residue` is omitted; serialized filter metadata records the
control that won.

## Hooks

Auto-store prompts and replies verbatim, plus best-effort-redacted and capped tool I/O. No LLM and no summaries. Fail-open (`{}` + exit 0) so a hook never blocks the agent.

`haunt install` binds all known hosts (mkdir parents even if the app is not installed). `haunt cursor-install` binds Cursor only. Each bind:

1. Merges capture hooks (preserving foreign hooks).
2. Merges the `haunt` MCP stdio server (preserving other servers) as an absolute `~/.haunt/bin/haunt-mcp` command — not a PATH lookup.
3. Writes a small haunt-owned rule so agents still `memory_recall` if no `[haunt ns=…]` block is visible.
4. Writes `skills/haunt/SKILL.md` into the host config dir.

**Hook ingest and trust:** Hooks write FTS rows immediately but never initialize the embedding model. They queue missing vectors in the namespace DB; ordinary model-owning writes or the explicit `haunt maintenance` command may drain a bounded batch. Recall never drains the queue. Hook recall is FTS-only. Raw tool I/O is excluded from hook-injected recall/worldview context by default, while explicit `--include-residue` recall returns it marked `trusted=false`. Recalled text is data and cannot authorize mutations.

**Secret redaction and size controls:** Hook-stored tool input and output are run through a best-effort denylist (API keys, bearer tokens, AWS keys, GitHub PATs, JWTs, etc.) and capped at 12,000 characters per field by default. `HAUNT_EXCLUDE_TOOLS` accepts comma-separated, case-insensitive globs for tools that must not be stored. These are **not** security boundaries — see [SECURITY.md](SECURITY.md).

### Cursor hooks

| hook | what |
|---|---|
| `beforeSubmitPrompt` | observe user prompt, recall k=8 (results returned in `additional_context` but Cursor does not reliably inject this into the model context) |
| `afterAgentResponse` | observe assistant text |
| `afterAgentThought` | skipped by default (`HAUNT_STORE_THOUGHTS=1` to store as system/coordinate) |
| `postToolUse` | observe tool_name + input + output |
| `afterShellExecution` | observe command + output |
| `afterMCPExecution` | observe MCP call (skips `memory_*` to avoid recursion) |
| `sessionStart` | session-open coordinate event + worldview card (`additional_context`); not a reliable recall path |
| `sessionEnd` | close session; no summary |

Cursor MCP is merged into `~/.cursor/mcp.json` (honors `CURSOR_HOME`) as an absolute `~/.haunt/bin/haunt-mcp` command. Each MCP process binds once to the inferred project namespace; ordinary tools cannot cross that binding.

### Claude Code hooks

| hook | what |
|---|---|
| `UserPromptSubmit` | observe `prompt`; return `additionalContext` inside `hookSpecificOutput` with `hookEventName` |
| `Stop` | observe `last_assistant_message` (never exit 2) |
| `SessionStart` | session-open coordinate event + worldview card (same JSON shape) |
| `SessionEnd` | close session |
| `PostToolUse` | observe tool I/O as episodic (skips `memory_*`) |
| `PostToolUseFailure` | same as PostToolUse for failed tool calls |

Claude hooks live in `~/.claude/settings.json` (nested matcher-group schema, absolute `haunt-hook-claude`). User-scope MCP **must** live in `~/.claude.json` (or `$CLAUDE_CONFIG_DIR/.claude.json`) — `settings.json` silently ignores `mcpServers`. The rule is `~/.claude/rules/haunt.md` (does not overwrite `CLAUDE.md`).

## CLI

| command | what |
|---|---|
| `haunt bootstrap [--reembed]` | first-run setup; installs desktop shortcut; exits 1 if sqlite-vec fails (unless `HAUNT_FTS_ONLY=1`) |
| `haunt init [name] [--repo PATH]` | create a namespace |
| `haunt observe TEXT ...` | store a turn / tool call verbatim |
| `haunt recall QUERY [--as-of --since --until --clock --tier --k] [--include-residue] [--json]` | read-only hybrid recall (vec + FTS5 + RRF); ranked results exclude tool/task residue unless explicitly requested; `--json` emits explanations |
| `haunt maintenance [-n NAMESPACE] [--limit 64] [--json]` | explicit mutating embedding upgrade/job-drain surface; never run by recall and never creates an unknown namespace |
| `haunt export OUTPUT [-n NAMESPACE] [--cut ISO]` | write a new mode-0600 canonical v1 namespace bundle; warns that verbatim data is sensitive |
| `haunt import BUNDLE [--timeout SECONDS] [limit options]` | strictly validate and transactionally import/replay one canonical namespace bundle |
| `haunt correct MEMORY_ID --idempotency-key KEY [--replacement --reason]` | atomically append a correction and optional verbatim replacement; omitted/null, empty, and whitespace-only replacement values are distinct; a nonempty caller key is required for safe retries |
| `haunt trace MEMORY_ID` | ordered correction chain from any surviving member, including erased-gap tombstones |
| `haunt delete MEMORY_ID [-y]` / `haunt delete --event-id EVENT_ID [-y]` | hard-delete a memory (or all memories for an event) and its provenance chain |
| `haunt timeline [--since --until --clock --json]` | events by `event_time` or `storage_time` (`ts` ingest time; `write_time` is a deprecated alias); human rows show source channel/origin and JSON preserves the complete provenance envelope |
| `haunt namespaces` | list + counts |
| `haunt namespace migrate OLD NEW [--repo PATH_OR_REMOTE] [--apply --plan-digest DIGEST]` | plan/apply a canonical rename; apply requires the matching dry-run digest |
| `haunt namespace alias SOURCE ALIAS [--repo PATH_OR_REMOTE] [--apply --plan-digest DIGEST]` | plan/apply an additional unique alias with the same digest gate |
| `haunt namespace undo MIGRATION_ID [--apply --plan-digest DIGEST]` | plan/apply an exact recorded reversal; refuses affected-state drift |
| `haunt namespace retire-alias ALIAS [--apply]` | check registry-owned references and optionally retire an alias |
| `haunt health [-n NAMESPACE]` | vec / embed / counts / db path |
| `haunt worldview [-n NAMESPACE]` | compact namespace briefing: facts, entities, and procedures with source provenance |
| `haunt procedure write NAME --body BODY` | store a named procedure |
| `haunt procedure get NAME` | retrieve a named procedure and its provenance |
| `haunt procedure list` | list all active procedures and their provenance |
| `haunt graph [--entity] [--rebuild]` | entities + relations |
| `haunt dash [--port 7340] [--install-icon] [--no-open] [--allow-remote]` | local memory console (127.0.0.1); prints a launch token required on `/api/*`; `--allow-remote` is unsafe without that token; namespaces are not auth |
| `haunt install` | bind all known hosts (Cursor, Claude Code): hooks + MCP + rules + skill |
| `haunt doctor` | check sqlite-vec (or FTS-only), haunt-mcp wrapper/python, embed (or FTS-only), and host files; rematch host files if missing; exit 1 if any check fails |
| `haunt cursor-install` | bind Cursor only: hooks.json + mcp.json + haunt.mdc + skill |

### Portable namespace bundles

`haunt export` moves durable namespace semantics—not SQLite implementation
state. The v1 bundle preserves the stable namespace ID and aliases, sessions,
events, memory IDs and validity, corrections/tombstones, provenance, entities,
mentions, and graph evidence. It excludes local paths, embeddings, FTS/vector
tables, jobs, and WAL files. Import rebuilds destination-local indexes and
queues non-empty text memories for the destination embedding model.

```bash
haunt export project.haunt.json -n project
HAUNT_HOME=/path/to/fresh-home haunt import project.haunt.json --json
```

Default exports use a stable durable high-water temporal cut, so unchanged
exports and a fresh import/re-export retain the same semantic digest. Use
`--cut` for an explicit historical instant. Import is bounded, validates before
publication, is idempotent for an exact replay, and rejects identity/alias or
record conflicts rather than merging ambiguously.

Each bundle also carries an opaque privacy-lineage head. Hard purge rotates it
atomically without retaining erased IDs or bytes, so a pre-purge bundle or a
diverged post-purge fork is rejected by that existing namespace; a fresh home
still imports the bundle exactly. Fresh publication uses a private fsynced
intent bound to the exact claimed database identities, so an interrupted import
is safely recovered on retry without deleting replaced or unrelated files.

The file contains potentially sensitive verbatim history. The digest detects
transfer corruption; it is not encryption or proof of authorship. MCP transfer
tools require `HAUNT_MCP_ADMIN=1`. Dashboard transfer uses the launch-token
authenticated export API and the trusted-Origin import API; the HTML console
intentionally has no paste/import control. See
[the v1 format contract](docs/EXPORT_FORMAT.md) and [SECURITY.md](SECURITY.md).

## MCP

haunt is its own MCP server — it runs alongside any other servers you already have (IronRecall, etc.) without interfering.

By default, one `haunt-mcp` process is bound immutably to one canonical namespace identity. New git-backed projects use the full remote identity (`host/owner/repo`), so same-leaf repositories do not collide. Existing namespaces registered to the same remote or repository path keep their current name. A proven alias of the bound identity is accepted; an alias belonging to any other identity is denied. Passing another namespace to an ordinary tool is denied, and `memory_namespaces` returns only the bound namespace. `HAUNT_MCP_ADMIN=1` enables cross-namespace access for an intentionally admin-scoped process.

Namespace labels are registry aliases for one stable identity and one existing SQLite file. `namespace migrate` changes the canonical display label while retaining the old label; `namespace alias` leaves the canonical label unchanged. Both commands are dry-run unless `--apply` is supplied. Apply requires the exact digest printed by the matching dry-run, creates and verifies a private registry-only backup, refuses drift and normalized-label/repository/database collisions, and records the plan and backup evidence. `namespace undo` uses the same dry-run/digest/backup protocol and refuses reversal after alias retirement or other affected-state drift. These operations never copy, rename, or move memory databases, and exact apply/undo retries are safe.

Haunt validates that the namespace directory and every registered database are real, single-link files inside `HAUNT_HOME/namespaces`, and records each database's device/inode identity. SQLite `-wal`, `-shm`, and `-journal` sidecars for both the registry and namespace databases must likewise be canonical, single-link regular files; missing names are claimed across SQLite open/configuration, and permission tightening never follows a sidecar symlink. Fresh files are privately initialized and atomically claimed before registry publication; mapped files are revalidated whenever opened. This is practical corruption and accidental-redirection protection for a local same-user store, not a security kernel against a malicious process running as that same OS user.

When upgrading a legacy registry that contains several labels for the same database path, Haunt preserves every label as an alias and deterministically selects the canonical label by earliest `created_at`, then normalized label, then display label.

`namespace retire-alias` checks only references Haunt owns in its registry: the canonical-label record, repository bindings, and dependent aliases. Editor, MCP-host, hook, and other external configuration cannot be verified from the registry; inspect and update those settings before applying retirement. Missing or stale external configuration is an operator caveat, not an automatic blocker.

`memory_purge` is marked destructive and is off for MCP by default. Use the confirmed `haunt delete` CLI flow, or explicitly launch a process with `HAUNT_MCP_ALLOW_PURGE=1`. Admin mode alone does not enable purge.

Every recall response that ran a known planner path also has an additive, versioned top-level `execution` object, so vector/FTS stage evidence survives even when `hits` is empty. It records the strategy and each modality as `candidate`, `ran_not_candidate`, or `not_run`, with an honest reason; native vec0 candidates and the persisted-embedding L2 path have distinct reasons. It also records read-only/no-maintenance status, observed pending jobs, residue-filter status, and class capability. Legacy plain hit lists omit it rather than fabricating evidence. Every explicit recall hit includes `trusted` and `trust_reason`, plus an additive `explanation` object with retrieval method, final result position, RRF contributions, raw vector/FTS signals (including metric direction) where available, applied filters, residue classification source, and safe references. These include public E2 structured provenance when valid (or an honest `legacy_unstructured`/`invalid_stored` status without legacy metadata), and E1 lineage status plus correction IDs only for intact chains. A lineage containing a privacy tombstone reports only `privacy_tombstone`, never erased or tombstone identifiers, trace content, or correction reasons. For ranked vector/FTS retrieval hits, the existing `score` and `explanation.rrf_score` are RRF rank signals—not confidence or relevance probabilities. Bare temporal queries return timeline hits in time order instead, with `score=0` and `score_semantics=not_ranked`. Tool input/output is retained for audit and explicit `--include-residue` search but is labeled `untrusted-tool-io`. No recalled row—trusted or untrusted—is permission to call a mutating tool.

`haunt recall --json`, MCP `memory_recall`, and dashboard recall endpoints return a nonzero/structured `retrieval_backend_error` response when SQLite or the vector backend fails. Human CLI output and Python calls still fail loudly rather than converting backend errors into empty recall results.

Equal ranked scores are ordered by stable memory ID. Dashboard all-namespace results are grouped in namespace order and preserve those local ranks rather than inventing a global RRF order. Timeline results remain ordered by the selected clock. When tied events exceed a bounded timeline page, SQLite selects the tied events by stable event ID before the limit; after one memory per selected event is materialized, equal-time hits are ordered by stable memory ID. The timeline `explanation.ordering` field records that two-step tie rule.

`haunt install` (or `haunt bootstrap`) automatically registers the MCP server in both Cursor (`~/.cursor/mcp.json`) and Claude Code (`~/.claude.json`). Merge only — other servers are kept. No manual JSON paste required.

`haunt-mcp` is a stdio server. Do not run it directly in a terminal — it reads JSON on stdin. Use it only as an MCP server command in your client config.

Tools: `memory_observe`, `memory_recall`, `memory_purge`, `memory_worldview`, `memory_procedure`, `memory_contradict`, `memory_trace`, `memory_timeline`, `memory_health`, `memory_namespaces`, `memory_export_bundle`, `memory_import_bundle`, `memory_session_end`. Export/import are admin-only. `memory_contradict` requires a nonempty caller `idempotency_key`; supplying the same key and exact correction payload safely replays the original result. Replacement strings are verbatim: omitted/null means no replacement, while empty and whitespace-only strings create replacements with those exact bytes.

`memory_observe.provenance` accepts a versioned object. Native envelopes use `kind="native"`; Haunt binds the actual entry channel (`mcp` here), `origin`, and supplied producer tool/call ID rather than trusting claimed values. Direct Python, CLI, Cursor hook, Claude Code hook, dashboard correction, and evaluation writes use their own explicit channels. Import envelopes use `kind="import"` and require canonical `imported_at`, `fidelity`, and `original_blob_sha256` (set it to `null` when no original blob exists). Source platform/native ID, format/parser version, and ordered transforms remain absent or null when unknown. Provenance and import fidelity are attribution—not confidence or truth scores.

The exact v1 fields and validation rules are documented in [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Environment variables

| variable | default | what |
|---|---|---|
| `HAUNT_HOME` | `~/.haunt` | data directory |
| `HAUNT_EMBED_MODEL` | `BAAI/bge-m3` | embedding model (set to `BAAI/bge-small-en-v1.5` for smaller; `off` for none) |
| `HAUNT_FTS_ONLY` | unset | set to `1` for FTS-only (no embeddings; sqlite-vec not required) |
| `HAUNT_OFFLINE` | unset | set to `1` to prohibit embedding backend initialization/download; FTS recall remains available |
| `HAUNT_NAMESPACE` | inferred from full git remote | override and bind the project namespace |
| `HAUNT_MCP_ADMIN` | unset | set to `1` only for an admin MCP process that may cross/list namespaces |
| `HAUNT_MCP_ALLOW_PURGE` | unset | set to `1` to expose hard purge in that MCP process; off by default |
| `HAUNT_EXCLUDE_TOOLS` | unset | comma-separated case-insensitive tool globs to skip in hook storage (for example `Read,Shell,secret_*`) |
| `HAUNT_TOOL_IO_MAX_CHARS` | `12000` | maximum stored characters for each hook tool-input/output field (clamped 256–100000) |
| `HAUNT_MODEL_CACHE` | `$HAUNT_HOME/models` | model download directory |

## Layout

```
~/.haunt/
├── registry.db
├── namespaces/<name>.db
├── bin/haunt-mcp
├── bin/haunt-hook
├── bin/haunt-hook-claude
└── models/
```

Default home is `~/.haunt` (`HAUNT_HOME`).

## What v1 does / does not

**Does:** verbatim events + memories, bi-temporal filtering, namespaces as isolated SQLite files, deterministic graph extract, hybrid recall, hard purge (delete with full provenance chain cleanup), CLI, MCP, Cursor hooks, Claude Code hooks, local memory management dashboard.

**Does not:** summarize or distill, call any LLM, talk to the network at query time, require Docker/Postgres, expose a team HTTP API, auto-recall into the model context.
