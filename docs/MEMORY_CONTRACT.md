# Haunt memory contract

**Status:** Accepted program contract

**Applies to:** the adoption sequence in [`../BACKLOG.md`](../BACKLOG.md)

This document records the minimal Memory Protocol ideas Haunt adopts and the
larger MP architecture it deliberately does not adopt. It is the product-level
source of truth for implementation and review. When a backlog item admits
multiple implementations, the implementation must still satisfy this contract.

The governing posture is simple: Haunt remains a local-first store of verbatim
events and memories, with one SQLite file per namespace, deterministic local
retrieval, and no cloud, distillation, or reader LLM (`README.md`, intro and
“What v1 does / does not”). “MP adoption” here means selected auditability,
portability, identity, explanation, and abstention semantics. It does not mean
feature parity or protocol conformance.

Normative terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** describe behavior
that future product changes are expected to preserve.

The MP inputs deliberately narrowed here are its append-only correction and
trace model, structured ImportRef/ExportBundle provenance, workspace
identity-versus-label distinction, retrieval introspection, and honest
abstention. The relevant source discussions are
`persistent_memory_architecture.md` §§2.1, 2.6, 3.2.1, 4.8, 5.7, and 9;
`non_negotiables.md` §§3, 4, 7, and 9; and
`protocol_considerations.md` §§2.4, 2.5, 6, and 7 in the local
`memory-protocol` repository. Their broader requirements do not override the
Haunt-specific decisions below.

## Existing seams this contract extends

- Events and memories are already distinct rows with session, time, role, tool,
  origin, validity, and content fields (`src/haunt/store.py`). Observe writes
  the event, memory, FTS row, vector/cache state, and graph evidence in one
  transaction (`src/haunt/store.py`).
- Current recall excludes rows whose `valid_to` is set, while explicit `as_of`
  recall uses the validity interval (`src/haunt/recall.py`).
- Correction currently mutates `valid_to` and may write a separate replacement,
  but stores no durable link between them (`src/haunt/store.py`).
- Memory detail already exposes useful provenance fields, though `origin` and
  `meta` are not a structured import contract (`src/haunt/store.py`).
- Repository identity already prefers normalized remote identity and preserves
  a matching legacy registration (`src/haunt/paths.py`). The registry still
  identifies a namespace by a single name and database path
  (`src/haunt/store.py`).
- Recall already uses reciprocal-rank fusion and retains vector/FTS component
  ranks (`src/haunt/recall.py`). The MCP description correctly says those
  scores are rank-normalized, not relevance probabilities
  (`src/haunt/mcp_server.py`).
- Purge is an intentional physical delete across canonical and derived rows
  (`src/haunt/store.py`). It is destructive, separately gated, and disabled
  over MCP by default (`src/haunt/mcp_server.py`).

These are migrations from working behavior, not permission to silently rewrite
old stores. Schema changes MUST remain additive and old rows MUST stay readable
with honest legacy labels when new structure cannot be reconstructed.

## 1. Correction is append-only; privacy erasure is the explicit override

Under normal operation, a correction **MUST** append an immutable correction
record that identifies the target memory and, when supplied, the replacement.
The record **MUST** carry its own ID, time, origin, session context, and an
explicit caller-supplied idempotency key unique within the canonical namespace.
A correction record **MUST NOT** be edited or deleted to make history look
clean.

`memories.valid_to` may remain as the efficient current/as-of projection used by
existing recall. It is not sufficient correction lineage by itself. The durable
correction record is the audit source for “what replaced this?” and “what did
this replace?” Normal current-state projection updates do not authorize mutation
of the correction history or original verbatim content.

A correction and its optional replacement **MUST** be atomic. A target has at
most one direct successor under normal operation, so lineage is a simple chain,
not a generic graph. Trace **MUST** return that chain and its source event/session
context. Legacy rows closed only by `valid_to` **MUST** be labeled as unlinked
legacy history; Haunt **MUST NOT** invent a replacement edge.

Correction replay **MUST** compare the caller key and canonical correction
payload (target memory ID plus exact replacement/reason bytes or null). The same
key and payload return the original committed response, marked deduplicated,
without writing again. The same key with a different payload is a conflict and
writes nothing. Database uniqueness and transaction boundaries **MUST** make
those semantics hold under concurrent callers. Two distinct keys racing for one
current target still produce at most one successor.

Explicit purge remains the privacy-erasure override. A confirmed/gated purge
**MAY** physically remove memories, original events, indexes, embeddings, graph
evidence, provenance, and correction material necessary to erase the selected
content. This is a deliberate exception to ordinary append-only history, not a
kind of correction. After erasure, trace and export **MUST** fail honest: they
may report that a lineage member was erased, but **MUST NOT** reconstruct or
retain the erased bytes merely to preserve audit completeness.

A purge **MUST** also erase the selected content from every namespace-database
copy Haunt itself wrote under its own backup directory, and **MUST** report the
copies it could not. Copies Haunt did not write — export bundles, operator
copies, filesystem snapshots, removable media — are outside the guarantee, and
documentation **MUST NOT** imply otherwise.

When a surviving chain needs an erased-gap marker, its public/exported
tombstone has exactly four fields: `schema_version`, a fresh random
`tombstone_id` that is not derived from erased data, `status="erased"`, and
`erased_at`. Trace position conveys the gap. A tombstone **MUST NOT** contain
erased content; memory, event, session, tool-call, import, or native-source IDs;
hashes of those IDs; blob hashes/references; origin/provenance; or correction
and erasure reasons. Tests must plant canaries in every forbidden class and
prove they are absent from all surviving logical rows and serialized surfaces.
When an erased target's user-controlled metadata is copied into an affected
shared-session metadata object, purge conservatively treats exact metadata keys
as erased context as well as values. A coincidentally identical session key may
therefore be removed even if its value differs; unrelated session fields remain.

This reconciles the user-visible distinction already documented between
supersede and delete (`README.md`, “Memory console”: Supersede vs Delete) with
a durable correction audit trail.

## 2. Provenance is structured attribution, not fact confidence

New observations **MUST** be able to carry a versioned, machine-readable source
envelope. For native memories, it identifies the actual entry-point channel,
origin, and any producer tool/call IDs that Haunt actually received. These
actual inputs are bounded and validated before any observation-side write, and
claimed fields must match them exactly. For imports, it additionally records
the source platform and native ID when known, parser/format version, import time,
fidelity, original-blob hash/reference when retained, and transform names.

Direct Python observation, procedure, and correction-replacement APIs bind both
origin and channel to `python` by default. CLI, MCP, dashboard, hooks, and
evaluation code bind their own actual entry point explicitly. Timeline and
procedure read surfaces return the same stored envelope rather than dropping or
reconstructing source fields.

Unknown values **MUST** remain absent or explicitly unknown. Haunt **MUST NOT**
guess an actor, platform, timestamp precision, source-native ID, fidelity, or
transform. Existing string `origin` and free-form `meta` data remain readable;
when they cannot be losslessly upgraded, they are
`legacy_unstructured`, not synthetic structured provenance.

Legacy SQLite values retain their dynamic type. If an old `origin`, `meta`, or
other surfaced SQLite value is a BLOB, public JSON represents it losslessly as
`{"encoding":"base64","data":"..."}` using standard base64, without trying
to decode it as text; decoding `data` yields the exact stored bytes. Ordinary
JSON-safe SQLite scalars keep their existing shape, and the read-time encoding
never changes the database. This rule applies recursively to raw public fields
and to values copied into `legacy_unstructured` or `invalid_stored`.
Malformed legacy BLOB metadata remains available through detail/trace but is
not guessed into typed procedure fields; guarded selectors treat it as no
match rather than raising.

New non-null structured provenance **MUST** use SQLite `TEXT` storage. A
non-text value found in a triggerless or corrupt database is never decoded,
even if its bytes look like valid JSON: reads label it `invalid_stored` and
idempotent replay fails closed. Public mappings **MUST NOT** rely on JSON's
lossy string-key coercion. Non-string SQLite keys use Haunt's reserved,
versioned reversible key codec; ordinary string keys keep their existing shape
unless they begin with the reserved prefix, in which case they are escaped.

Human CLI output is a bounded presentation of those already-serialized
values, never a replacement for the machine envelope. It labels BLOB and
non-finite REAL values explicitly, escapes terminal controls, and accepts every
JSON-safe scalar/container without assuming timestamp, content, role, tier,
identifier, or procedure fields are strings.
Generic dictionaries remain stable JSON and are not inferred to be SQLite
scalar envelopes. Only explicitly typed scalar presentation sites recognize a
valid BLOB/REAL envelope. Recall results use the same recursive lossless
serialization on direct Python, CLI JSON, MCP, per-namespace dashboard, and
all-namespace dashboard paths, with no `str()` fallback.

For ordered import transforms, omitted, explicit `null`, and an empty list are
distinct: not supplied, explicitly unknown, and known to have no transforms,
respectively. Validation and idempotency preserve that distinction.

An idempotency replay succeeds only when stored structured provenance is valid
and byte-for-byte equal to the newly canonicalized attribution. Legacy-null or
invalid stored attribution remains readable but cannot be safely replayed.

Import fidelity describes preservation by the import process:
`lossless`, `lossy`, `reconstructed`, or `derived`. It does not describe whether
the imported statement is true. Fidelity, recency, source kind, tool trust, and
provenance completeness **MUST NOT** be collapsed into fake fact confidence.

Haunt stores verbatim memories rather than MP-style distilled facts. This
program therefore adds no `Fact`, extractor confidence, source reputation, or
LLM self-assessment. Existing `trusted`/`trust_reason` labels continue to mean
that raw tool I/O is untrusted data and no recalled row authorizes a mutation
(`README.md`, “MCP”; `haunt.recall` trust labelling). “Trusted” is not
“true.”

## 2a. Recall selection and access are explicit, read-only semantics

Schema v9 records nullable `events.recall_class`, constrained to `tool` or
`task`. A null class means legacy/unknown; migrations **MUST NOT** backfill or
guess it from text. The `tool` role or raw tool structure **MUST** be stamped
`tool` before the
event write, and contradictory explicit classification **MUST** fail before any
event/session/job write. An actual host session-start coordinate entry point
MAY stamp its own event `task`; no ordinary prompt, procedure, correction,
import, or free text is task-classified without entry-point knowledge.

Correction replacement **MUST** preserve the target event's effective class:
the recorded class when present, otherwise `tool` when a legacy target has raw
tool structure. This prevents a correction from laundering task/tool residue
into eligible memory. Procedures and imports are unclassified unless their
real input shape or entry point carries an explicit valid class.

Ranked recall **MUST** exclude raw tool structure and explicit `tool`/`task`
classes by default. An explicit `include_residue` option **MAY** bypass that
selection rule for audit/search. The legacy `include_untrusted` option is a
deprecated alias only when the modern option is omitted; structured metadata
**MUST** identify the winning control. On a pre-v9 read-only database, raw tool
structure remains excluded, class capability is reported unavailable, and null-
like legacy rows remain eligible. Timeline, trace, and detail are reachability
surfaces, not ranked recall; timeline execution **MUST** say this filter is not
applicable, while trace and detail remain available for audit.

Recall **MUST NOT** perform maintenance. It resolves the stable namespace
identity without registry writes and opens a guarded zero-write SQLite view;
schema migration, WAL configuration, graph rebuild, permission tightening,
embedding-job drain, and model upgrade are forbidden on that path. A complete
live WAL **MAY** be read from a verified private temporary shadow when required
by SQLite portability, but source `HAUNT_HOME` files remain untouched. A
separately named maintenance operation owns any embedding upgrade/job drain and
reports it as mutating. Recall execution metadata **MUST** expose read-only
status, no-maintenance status, observed pending jobs, residue filtering/class
capability, and an honest offline/vector stage reason.

With `HAUNT_OFFLINE=1`, Haunt **MUST NOT** initialize/download a remote-capable
embedding backend or invoke its network path. FTS retrieval remains available;
the vector stage **MUST** report that it was not run rather than fabricate a
vector result.

## 3. RRF score is ranking evidence, never confidence

The value computed from `1 / (RRF_K + rank)` contributions is an RRF ordering
score (`src/haunt/recall.py`). It says how a
candidate ranked relative to other candidates in the active retrieval lists.
It is not:

- a probability that the result answers the query;
- a probability that the memory is true;
- comparable across arbitrary corpora, retrieval modes, or model versions; or
- permission to act on recalled text.

Recall explanations **MUST** expose the retrieval mode, applied filters,
component-list membership, component ranks and raw metrics when available, RRF
contributions, fused RRF score, and final rank. A compatibility field named
`score` **MAY** remain, but documentation and structured metadata **MUST** call
it an RRF ranking score and **MUST NOT** label it confidence.

Abstention is a separate decision. Before fitting, it **MUST** have a separately
versioned calibration dataset with answerable/unanswerable labels, a predeclared
fit/held-out split, and hashes for both dataset and split definition. E0's
deterministic FTS-only corpus is regression evidence only and **MUST NOT** count
as calibration fit or held-out evidence. Calibration uses raw retrieval evidence
appropriate to the active profile. Haunt **MUST NOT** manufacture confidence by
renormalizing RRF or apply one profile's threshold to another model/mode. The
pinned hybrid model ID, dimension, and retrieval configuration are part of its
E6 calibration identity and **MUST** fail loud on FTS fallback. A no-answer
result **MUST** be explicit and distinguish thresholded abstention from zero
candidates and an uncalibrated profile.

## 4. Canonical export is versioned and excludes embeddings

Haunt **MUST** provide a versioned canonical export/import representation of one
namespace's durable semantics: namespace identity/labels, sessions, events,
verbatim memories, validity, correction lineage, structured provenance, and
the durable graph evidence required to reproduce current behavior. The format
**MUST** define version negotiation, canonical ordering, integrity checks, and
transactional/idempotent import behavior.

Import **MUST** resolve finite positive budgets for input/decompressed bytes,
record count, per-record bytes, JSON depth, and collection items per record.
The format ships documented safe defaults and clamps; the parser enforces
actual streamed usage rather than trusting declared counts. Limit, parse,
validation, timeout, and resource failures **MUST** close temporary resources
and commit no logical mutations or derived jobs. Byte-identical SQLite files are
not the rollback criterion: page allocation, WAL activity, or an independently
required schema migration may change file bytes. The proof compares namespace,
alias, session, event, memory, correction, provenance, graph/index, vector, and
embedding-job state immediately before and after the rejected import.

Canonical namespace identity **MUST** include an opaque privacy-lineage head.
Every successful hard purge rotates that head atomically without retaining
erased identifiers, identifier hashes, content, or provenance. Existing import
requires an exact head match before any repair or write; fresh import preserves
the head. Fresh publication **MUST** be crash-recoverable from a durable intent
bound to bundle digest, stable namespace ID, unpredictable token, and exact
claimed file identities. Recovery may remove only those still-owned files and
must fail closed for replacements, symlinks, unrelated files, or unsafe
hardlinks.

Embeddings **MUST NOT** appear in canonical export. Neither may vector tables,
FTS tables, embedding jobs, absolute machine-local paths, WAL/SHM state, or
other rebuildable indexes/caches. Embeddings depend on local model identity and
dimension; Haunt already treats model/dimension changes as a full re-embed
concern (`README.md`, “Embeddings”). Import rebuilds FTS and graph
projections and queues or rebuilds embeddings using the destination's
configured model.

Export **MUST** include superseded surviving history and **MUST NOT** include
previously purged bytes. A fresh import followed by re-export must preserve the
canonical semantic digest and reproduce IDs, verbatim content, temporal
membership, correction trace, source provenance, and trust labels. This is a
Haunt format; compatibility with MP UIIR/ExportBundle or vendor formats is not
claimed unless separately implemented and tested.

## 5. Namespace aliases aid identity and migration; they never authorize

A canonical namespace identity names one registered store. Human/repository
labels may be aliases for that same identity so renames, moves, remote URL
normalization, and legacy names do not fragment memory. Alias resolution
**MUST** be unique, collision-checked, deterministic, and fail closed. It
**MUST NOT** create a new database as a side effect of a read.

An alias is evidence that two labels refer to the same registered store. It is
not a capability, ACL, group, scope, permission, or cross-namespace federation
rule. Knowing or presenting an alias **MUST NOT** broaden authority.

The ordinary MCP process remains immutably bound to one canonical namespace
identity. Requests may use a proven alias for that same identity, but an alias
resolving to any other identity is denied. Admin mode and purge enablement stay
explicit and independent (`haunt.mcp_server.MCPAuthority`; `README.md`, “MCP”).
The README's existing warning remains governing: namespaces are storage
isolation, not authorization (`README.md`, “Memory console”; `SECURITY.md`,
“File-per-namespace isolation”).

Alias migration **MUST** be dry-run-first. Dry-run is zero-write and returns a
deterministic plan digest bound to the exact registry state and requested
operation. Apply **MUST** require that caller-supplied digest, recompute it, and
fail closed on drift. Before apply, Haunt **MUST** create a consistent,
integrity-verified mode-0600 backup of the registry inside a private mode-0700
Haunt backup directory. It **MUST NOT** copy a namespace database as part of a
label migration.

Each applied migration **MUST** retain its audit record and the exact affected
canonical, alias, legacy-label, and repository-binding state needed for an
explicit undo keyed by migration ID. Undo follows the same dry-run, digest,
drift-check, and verified-backup protocol; it is atomic and idempotent. If an
alias was retired or any affected recorded state changed after apply, undo
**MUST** refuse rather than guess or recreate unproven state. A successful
rename retains the old label until explicit safe retirement. Aliases may help
select the right existing database; they do not turn Haunt into MP's
hierarchical scope model.

Automatic retirement checks are limited to registry-owned recorded references:
repository bindings, canonical-label records, and dependent aliases. A recorded
reference blocks retirement. Editor/host configurations outside Haunt's
registry are not a reliable authority surface; the command **SHOULD** report
them as an operator caveat, but missing, unreadable, or stale external config
**MUST NOT** become an unverifiable automatic blocker.

## 6. Preserve Haunt's simplicity

The following MP mechanisms are deliberately outside this adoption program:

- **Generic event/provenance DAG:** correction lineage is a single-successor
  chain plus direct source attribution. Haunt does not add arbitrary
  `parent_ids`, multi-parent derivations, merge/unmerge, replay as a universal
  event-sourced substrate, or a second graph beside its deterministic entity
  evidence.
- **HMAC-signed/tamper-evident log:** no signature fields, signing keys, key
  rotation, signed tips, or integrity protocol. Export digests detect transfer
  corruption; they are not authenticity claims.
- **Hierarchical scopes and authorization expansion:** no user/org/team/branch/
  worktree scope URIs, capabilities, ABAC, cross-namespace promotion, or
  federation. Existing MCP binding and dashboard controls remain the product's
  narrow access posture; aliases do not expand it.
- **Distillation and derived belief machinery:** no fact layer, summaries,
  reader LLM, extractor ladder, entity profiles, source-derived confidence,
  consolidation scheduler, or proactive procedure learning. Haunt continues to
  store and retrieve verbatim text.
- **Platform expansion:** no cloud/team/org tier, warehouse adapter, live sync,
  CRDT, multi-agent lease/handoff system, or new wire protocol.

These are rejected because they would replace Haunt's product rather than make
its existing memory more auditable and portable. A future proposal to add one
is outside this program and requires an explicit product decision that amends
this contract before implementation.

## 7. Compatibility and evidence rules

- Schema changes **MUST** use Haunt's versioned migration path and remain
  idempotent (`src/haunt/store.py`). Read/query paths **MUST NOT** create
  unknown namespaces.
- Old data **MUST** remain byte-preserved unless the user invokes explicit
  privacy purge. Missing new metadata is represented as legacy/unknown, not
  backfilled with guesses.
- Public Python, CLI, MCP, and dashboard behavior **MUST** agree on correction
  lineage, provenance, alias resolution, export semantics, ranking explanation,
  and abstention reason codes.
- Every epic **MUST** land its tests and machine-readable evidence with the
  implementation. E0 gates integration of E1, E3, and E5, which may be developed
  in parallel. E4 waits for E1/E2/E3 durable schemas; E6 waits for E0/E5 and uses
  separate predeclared calibration evidence; E7 waits for E4/E6. An epic can be
  in progress before a dependency lands but cannot be marked done first.
- A release **MUST NOT** claim complete MP conformance. The accurate claim is
  that Haunt adopts this documented subset of MP-inspired semantics.

## Deferred implementation decisions

The contract fixes behavior but intentionally leaves these representations to
their owning epics:

1. Whether correction records live in a dedicated table or a narrowly typed
   event extension, provided the history is append-only and traceable.
2. Which provenance fields are typed columns versus versioned JSON, provided
   validation and export semantics are identical.
3. The internal canonical namespace ID shape, alias-retirement UX, and whether
   database filenames ever change in a separate maintenance operation.
4. The export container/media type, minor-version support window, and treatment
   of volatile creation metadata outside the canonical semantic digest.
5. Numeric safe defaults and clamps for the mandatory import byte, record,
   depth, and collection-item budgets.
6. The abstention feature formula and default compatibility behavior for an
   uncalibrated retrieval profile; neither may reinterpret RRF as confidence.

These are not permission to weaken the normative requirements above. If an
implementation choice would change observable semantics, amend this contract
before product code lands.
