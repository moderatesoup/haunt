# Memory adoption backlog

This is the dependency graph for adopting the useful parts of
Memory Protocol (MP) in Haunt. The normative product decisions are in
[`docs/MEMORY_CONTRACT.md`](docs/MEMORY_CONTRACT.md). An epic is not complete
until its acceptance criteria and named evidence are committed together.

Haunt is not trying to become the MP reference implementation. The program
keeps Haunt local-first, verbatim, one-SQLite-file-per-namespace, and free of a
reader LLM or distillation pipeline (`README.md:3`, `README.md:239-243`).

## Status key

- **Ready**: all dependencies are complete and no implementation is active.
- **In progress**: an implementation branch exists, but acceptance evidence has
  not yet landed with this backlog; dependencies may still be pending.
- **Blocked**: no active deliverable can close while a dependency is incomplete.
- **Done**: every acceptance criterion is met and the evidence is committed.

The integration commit for an epic must change its status to **Done** and link
the committed evidence. Parallel branches may be **In progress**, but an epic
cannot become **Done** until every dependency is **Done**; branch work alone is
not completion.

## Current baseline

| Area | Existing behavior | Adoption gap |
|---|---|---|
| Evaluation | The optional LongMemEval temporal probe remains external-data-only (`scripts/score_lme_temporal.py:74-154`). E0 now provides a deterministic FTS-only `K=3` regression gate with logical-ID normalization, corpus/config hashes, exact ordered results, Porter morphology, serialized tool-I/O trust metadata, and non-vacuous positive/negative assertions (`src/haunt/frozen_retrieval_eval.py`; `tests/test_frozen_retrieval_eval.py`). | E0 is complete. Pinned hybrid evaluation and calibration remain E6 work. |
| Correction | `Store.contradict()` updates `memories.valid_to` and may add an unlinked replacement (`src/haunt/store.py:1502-1565`); current recall hides closed rows (`src/haunt/recall.py:93-123`). | An append-only correction record and traversable old-to-new lineage. |
| Provenance | Schema v8 stores a validated v1 source/import envelope on each new event while preserving legacy `origin`/`meta` bytes; detail, browse, timeline, correction trace, CLI, MCP, and dashboard expose the same structured attribution (`src/haunt/provenance.py`; `src/haunt/store.py`; `tests/test_structured_provenance.py`). | E2 implementation is in progress pending independent review. Canonical bundle transport remains E4 work. |
| Namespace identity | Repository remotes derive collision-resistant names and legacy registrations are reused (`src/haunt/paths.py:56-117`, `src/haunt/paths.py:148-178`). | Explicit aliases, rename/move migration, collision handling, and retirement rules. |
| Portability | Namespace schema migration is versioned (`src/haunt/store.py:278-321`), but no canonical export/import exists. | A versioned, deterministic, embedding-free round trip. |
| Retrieval | Recall fuses vector and FTS ranks with RRF and retains component ranks internally (`src/haunt/recall.py:31-78`, `src/haunt/recall.py:243-288`). | A stable explanation contract and a calibrated ability to return no answer. |
| Erasure | Purge physically deletes memory, indexes, graph evidence, and orphan events (`src/haunt/store.py:1169-1244`) and is explicitly gated over MCP (`src/haunt/mcp_server.py:417-461`). | Preserve this privacy override while correction becomes append-only. |

## Dependency graph

| Epic | Direct dependencies |
|---|---|
| E0 — frozen FTS-only regression | none |
| E1 — correction lineage | E0 |
| E2 — structured provenance | E1 |
| E3 — namespace aliases | E0 |
| E4 — export/import | E1, E2, E3 |
| E5 — ranking explanations | E0 |
| E6 — calibrated abstention | E0, E5 |
| E7 — end-to-end release proof | E4, E6 |

After E0, correction, aliasing, and ranking explanations can proceed in
parallel. E3 does not depend on correction/provenance because it changes the
registry identity layer, not namespace memory rows. E5 explains retrieval data
that already exists; E2 references enrich it later but do not block the v1
explanation shape. E4 waits for the durable correction, provenance, and alias
schemas it must round-trip. E7 joins the portability and retrieval branches.

## E0 — Freeze the retrieval evaluation

**Status:** Done

**Depends on:** none

**Outcome:** A small, repository-owned, deterministic FTS-only gate is the fixed
regression comparison for every later epic without requiring embeddings.

**Acceptance criteria**

- Commit a versioned corpus and evaluation configuration. Record the SHA-256 of
  the canonical corpus and the canonical configuration; individual fixture
  files do not need separate manifest entries.
- Give fixture memories stable logical IDs. Ingest them through normal Haunt
  writes, capture the generated physical memory IDs, and map returned results
  back to logical IDs before comparison. Physical IDs and storage timestamps
  are intentionally generated and excluded from the result lock.
- Fix only semantic timestamps needed by temporal cases. The evaluator must not
  use wall-clock time to interpret a query or expected result.
- Cover exact lexical lookup, Unicode tokenization, stemming/morphology,
  temporal current/as-of behavior, superseded content, namespace isolation, and
  designated out-of-corpus/no-hit queries. True semantic paraphrase coverage
  belongs to E6's pinned hybrid profile, not this FTS-only gate.
- Add at least one tool-I/O query whose exact logical result and
  `trusted=false` / `trust_reason="untrusted-tool-io"` metadata are locked. This
  valuable case is explicit remaining E0 acceptance until it lands.
- Declare `K` in the configuration (`K=3` for the current gate). Score by
  expected logical IDs, report Recall@K as Recall@3, and reject a result merely
  because it is non-empty when its IDs or order are wrong.
- Lock the exact ordered logical-ID result list for every query, including `[]`
  for designated no-hit cases, plus aggregate Recall@3 and evaluated-case count.
  The committed result lock records the corpus hash and configuration hash; it
  does not depend on a Haunt revision or generated database metadata.
- Run with `HAUNT_FTS_ONLY=1` in normal CI and assert that no vector model/stage
  ran. Changing the corpus, configuration, or expected result requires a
  reviewed version/hash and result-lock update.

**Tests/evidence**

- Two fresh-store runs generate different physical IDs but produce the same
  exact ordered logical-ID results, metadata locks, and Recall@3.
- Full baseline equality over resolved `K`, corpus/config hashes, per-query
  logical results/metadata, and Recall@3 fails on any drift. Fixtures require a
  non-empty positive expectation and at least one exact `[]` no-hit expectation
  so the equality check cannot pass vacuously.
- The gate runs under the existing `HAUNT_FTS_ONLY=1` pytest CI path. The
  committed baseline and failing assertion output are the evidence; E0 does not
  require CI artifact upload/publication unless the workflow later adds it.

**Completion evidence**

- `tests/fixtures/retrieval_eval/corpus.json` and `baseline.json` are schema v2
  and lock ten cases: nine positive and one designated no-hit query.
- The locked metrics are Recall@3 `1.0`, MRR `1.0`, false-negative rate `0.0`,
  and negative-query false-positive rate `0.0`.
- `tests/test_frozen_retrieval_eval.py` proves exact full-baseline equality,
  repeated-run determinism, caller-home isolation, Porter morphology, and that
  a public `Hit.as_dict()` trust-label regression fails the gate.

**Non-goals**

- Claiming benchmark leadership, vendoring a restricted third-party corpus, or
  using an LLM judge.
- Embeddings, hybrid evaluation, calibration, or tuning ranking/abstention.
  Those belong to E6, so E0 never blocks E1 on model availability.

## E1 — Add append-only correction lineage and trace

**Status:** Done

**Depends on:** E0

**Outcome:** A correction is an immutable record that points to what it
supersedes and, when present, its replacement; any surviving memory can explain
that lineage.

**Acceptance criteria**

- Add an additive schema migration for a correction record with its own stable
  ID, target memory ID, optional replacement memory ID, timestamp, origin,
  session ID, optional reason, and caller-supplied idempotency key. The key is
  non-empty, bounded, and unique within the canonical namespace. Normal
  correction code only inserts these records; it never edits or deletes one.
- Make correction plus optional replacement atomic. A failed replacement write
  leaves no correction record and leaves the target current, preserving the
  atomicity guarantee already tested around `Store.contradict()`.
- Keep `valid_to` as a current/as-of projection for compatibility, while making
  the correction record the durable source of lineage. Existing pre-migration
  `valid_to` rows remain readable and are honestly labeled `legacy_unlinked`
  when no link can be recovered.
- Define replay over the explicit idempotency key. The canonical payload is the
  target memory ID plus exact replacement/reason bytes or null. Same namespace,
  key, and payload returns the originally committed response with
  `deduplicated=true` and writes nothing. Reusing the key with a different
  canonical payload returns an idempotency conflict and writes nothing.
- Prevent forks under normal operation: a current target may receive at most
  one direct correction. Two different keys racing to correct the same target
  yield exactly one success and one already-superseded conflict.
- Add a trace API used by CLI/MCP/dashboard detail that returns the ordered
  correction chain, source event/session identifiers, and an explicit status
  for a lineage member removed by privacy purge. Trace must never invent the
  erased bytes.
- If purge leaves a gap needed to trace a surviving chain, replace the public
  gap with an allowlisted tombstone containing only `schema_version`, a fresh
  random `tombstone_id` not derived from erased data, `status="erased"`, and
  `erased_at`. Its position in the trace conveys the gap. It contains no erased
  content, memory/event/session/tool-call/import/native-source IDs, ID hashes,
  blob hashes/references, origin/provenance, or correction/erasure reason.
- Preserve historical recall: current recall returns only the chain tip, while
  an `as_of` before correction can return the prior memory.

**Tests/evidence**

- Migration tests open a schema-v3 database, correct a legacy memory, restart,
  and recover the same lineage.
- Positive tests cover correction with and without replacement, a three-link
  chain, and trace after restart. Replay tests cover same-key/same-payload and
  same-key/different-payload behavior before and after restart.
- Concurrency tests cover same-key/same-payload callers, same-key/different-
  payload callers, and different keys targeting one memory; each asserts exact
  committed row counts and responses.
- Failure-injection tests prove atomic rollback. A mutation test that removes
  the lineage insert must fail the trace assertion.
- Erasure tests plant unique canaries in content, every source identifier, blob
  hash, origin/provenance, and reasons; after purge, surviving database rows,
  Store/CLI trace JSON, dashboard output, and MCP output contain none of them.
  A response schema test rejects any tombstone key outside the four-field
  allowlist. E4 must add the corresponding canonical-export purge proof before
  export/import can be marked done.
- Existing temporal and purge suites remain green.

**Completion evidence (2026-08-25)**

- Schema v7 stores append-only correction rows, enforces normal-row insert
  invariants and UPDATE/DELETE guards in SQLite, and permits mutation only
  inside the connection-local privacy-purge transaction.
- Exact canonical payload, replay, fork prevention, rollback, restart,
  historical recall, legacy-unlinked trace, and CLI/MCP/dashboard behavior are
  covered by `tests/test_correction_lineage.py` and adjacent compatibility
  suites.
- Privacy tests cover first/middle/last/shared-event erasure; reused target and
  correction sessions; UTF-8 and opaque BLOB metadata; tool, event, session,
  request-payload, and metadata-key canaries; derived-index failure rollback;
  and the strict four-field tombstone across every currently shipped serialized
  surface.
- The final independent GPT-5.6 Terra merge-gate review was CLEAN. The final
  focused lineage suite passed 50 tests; the dependency-correct FTS profile
  passed 372 tests with 1 skip and 7 expected failures before CI.

**Non-goals**

- A generic provenance DAG, arbitrary many-parent derivations, merge/unmerge,
  HMAC signing, or tamper-evident replication.
- Removing the explicit privacy purge path.

## E2 — Structure source and import provenance

**Status:** Done

**Depends on:** E1

**Outcome:** Every new memory can report how it entered Haunt without pretending
that source metadata measures truth.

**Acceptance criteria**

- Define and validate a versioned provenance envelope for new observations.
  Native observations identify channel/origin and, when applicable, producer
  tool plus tool call ID. Imports additionally identify source platform,
  source-native ID, source format/parser version, import time, fidelity
  (`lossless`, `lossy`, `reconstructed`, or `derived`), original blob hash or
  explicit absence, and ordered transform names.
- Store the envelope structurally (typed columns and/or versioned JSON), not as
  a display string. Keep the existing `origin` field readable during migration.
- Represent unknown fields as absent/null and surface them as unknown. Do not
  infer a platform, actor, timestamp precision, or fidelity that the source did
  not supply.
- Do not add fact confidence. Haunt stores verbatim memories, and provenance or
  import fidelity must not be converted into a `confidence` value.
- Include structured provenance in memory detail, trace, browse output, and the
  future canonical export. Existing rows remain readable as
  `legacy_unstructured` with their original `origin`/`meta` intact.
- Reject unsupported provenance schema versions and invalid fidelity values
  before writing an event; the rejection leaves no partial session/event row.

**Tests/evidence**

- Round-trip unit tests cover each fidelity label, native tool capture, unknown
  fields, Unicode source IDs, and parser-version rejection.
- Migration tests prove old `origin` and `meta` bytes are unchanged.
- A schema/response test fails if a `confidence` field is added to a memory or
  provenance object.
- A trace test starts at a corrected imported memory and reaches both its
  correction lineage and source envelope.

**Completion evidence (2026-08-25)**

- `src/haunt/provenance.py` defines the bounded schema-v1 native/import
  envelope, canonical UTC/hash rules, four fidelity values, actual producer
  matching, actual entry-point channel binding, honest legacy/invalid labels,
  and no truth-confidence field.
- Schema v8 adds event provenance without rewriting old `origin` or `meta`;
  `Store.observe()` validates and canonicalizes it before any session, event,
  derived job, graph, or index write. Idempotency includes exact canonical
  attribution, including concurrent retry behavior, and fails closed when a
  legacy-null or invalid stored envelope cannot prove exact attribution.
- Schema-v8 insert/update triggers require non-null provenance to use SQLite
  `TEXT`. Triggerless/corrupt BLOB provenance is labeled `invalid_stored`
  without a UTF-8 guess and conflicts on replay, even when its bytes contain
  valid-looking JSON.
- Opaque legacy SQLite BLOB values are exposed losslessly through an explicit
  standard-base64 object at the recursive Store serialization boundary. This
  includes BLOB `origin`, `meta`, and invalid-stored rows without UTF-8 guesses,
  while leaving every database byte untouched and keeping ordinary scalar
  response shapes compatible. Guarded JSON selectors and tolerant stored-meta
  parsing make malformed BLOB procedure metadata an honest no-match instead of
  a read-surface failure.
- Human CLI reads use one bounded, terminal-control-safe renderer for the
  already-serialized values. Timeline, worldview, recall, graph, namespace,
  and procedure formatting no longer applies string-only operations directly
  to migrated SQLite dynamic types; JSON output remains exact and unchanged.
- Recursive public serialization uses a reserved reversible codec for
  non-string mapping keys and escapes colliding ordinary string keys. Stats,
  graph, namespace, detail, and recall payloads therefore remain strict JSON
  without collapsing dynamic SQLite types; generic dictionaries are displayed
  as stable JSON unless a caller explicitly identifies a serialized scalar.
- `tests/test_structured_provenance.py` covers every fidelity, Unicode source
  and call IDs, unknown fields, parser/version/hash rejection with zero rows,
  byte-preserving migration/restart, corrected-import trace, actual Store/CLI/
  MCP/dashboard/hook channels, strict UTF-8 input bounds and zero-write
  rejection, honest direct-Python defaults, timeline and procedure provenance
  parity, dashboard timeline JSON errors, worldview procedure attribution,
  explicit-null/omitted/empty transform semantics, corrupt stored envelopes,
  recursive no-confidence assertions, schema-v7 BLOB origin/meta migration and
  restart across Store/CLI/MCP/dashboard/procedure/worldview surfaces, strict
  JSON serialization, exact base64 recovery, exhaustive bounded human CLI
  rendering, strict/reversible dynamic mapping keys, exact recall values across
  Python/CLI/MCP/dashboard, and raw plus encoded privacy-purge canaries across
  shared sessions, all tables, and serialized surfaces.
- The dependency-correct Python 3.14/MCP 2.1 sqlite-vec full suite passes with
  566 tests, 2 environment/data skips, and 7 declared temporal xfails. The
  focused E2/E1/E0/security/authority/clock FTS compatibility group passes 158
  tests under both Python 3.10 and Python 3.12 with MCP 2.1. Those pyenv builds
  cannot load SQLite extensions, so their unrelated vector-required full-suite
  bootstrap failures are not a valid compatibility profile.
- The final independent GPT-5.6 Terra merge-gate review was CLEAN at
  `4664ab5722363d1e78cac93d927c79e87f2d8224`. Its isolated Python 3.10 and
  Python 3.12 MCP 2.1 FTS profiles each passed 149 E2/E1/E0 tests, and its
  independent corrupt-store, exact-recall, mapping-key, and encoded-purge
  mutation matrix found no remaining issue.

**Non-goals**

- Automatic truth scoring, source reputation, fact extraction, source blob
  storage policy, or import adapters for every vendor.
- Silent redaction or transformation of verbatim memory content.

## E3 — Add namespace aliases and migration

**Status:** Done

**Depends on:** E0

**Outcome:** A repository move, remote normalization change, or deliberate
namespace rename can retain one memory identity without copying data or
broadening access.

**Acceptance criteria**

- Add an additive registry migration that separates a stable canonical
  namespace record from one or more unique labels/aliases. Existing namespace
  names become canonical labels without moving their database files.
- Resolve aliases to the same registered database for CLI, hooks, dashboard,
  and MCP. Alias resolution must occur without creating a typo namespace.
- Provide a dry-run-first migration command. Dry-run is zero-write and emits a
  deterministic plan digest bound to the exact registry state and operation;
  apply requires that caller-supplied digest and fails on drift. Before apply,
  Haunt creates and verifies a mode-0600 registry-only backup under its private
  home (never a namespace database copy). Apply is atomic and idempotent,
  records old/new labels, repository identity, plan, and backup evidence, and
  refuses alias collisions or a target mapped to another database.
- Every applied migration records enough prior registry state for an explicit
  undo-by-migration-ID workflow. Undo is itself dry-run-first, digest-gated,
  backed up, atomic, and idempotent. It fails closed after alias retirement or
  any other affected alias/canonical/repository-binding drift; audit history is
  retained rather than erased.
- Preserve the immutable MCP process boundary: authorization is checked against
  the process's canonical namespace identity, not granted by possession of an
  alias string. Admin status and purge permission remain separate controls
  (`src/haunt/mcp_server.py:70-113`).
- Support a documented alias-retirement check over registry-owned recorded
  references only. It refuses retirement while a registry repository binding,
  canonical-label record, or dependent alias still needs the candidate alias.
  Host/editor configuration outside the registry is reported as an operator
  caveat to inspect; it is not an unverifiable automatic blocker.
- Prove that two clones of the same normalized remote resolve to one canonical
  namespace, while same-leaf repositories with different remotes remain
  distinct.

**Tests/evidence**

- Fresh, upgraded, rename, move, remote URL form, truncation/hash, collision,
  retirement, typo-read, concurrent-migration, plan-tamper/drift, backup
  integrity/restore, restart, and undo tests.
- Retirement tests cover every registry-owned reference class and prove a
  missing/unreadable external host configuration can only produce the caveat,
  not a false registry reference or permanent refusal.
- Security tests bind an ordinary MCP process to namespace A and prove that an
  alias for namespace B cannot be used to read or mutate B.
- Filesystem evidence shows migration does not duplicate or rename the
  namespace database unless a separately confirmed maintenance action says so.

**Completion evidence (2026-08-26)**

- E3 was integrated over E1/E2 at `fb9ab09`, preserving namespace database
  schema v8, independent registry schema v5, correction/privacy invariants,
  structured provenance, guarded SQLite opening, and stable-ID MCP authority.
- The author integration run passed 308 conflict-focused tests, 274 E0–E3 tests
  on both Python 3.10 and Python 3.12/SQLite 3.43 with MCP 2, and the exact
  dependency-correct full profile with 692 passes, 1 skip, and 7 expected
  failures.
- A fresh independent GPT-5.6 Terra review of exact `fb9ab09` was CLEAN. Its
  Python 3.10/MCP 2 and Python 3.12/SQLite 3.43/MCP 2 E0–E3 profiles each passed
  274 tests; its alias/sidecar plus integration profiles each passed 125 tests.
  Its separate Python 3.10 broad compatibility environment passed 688 tests
  with 5 optional-model skips and 7 expected failures. Compile, diff, clean-tree,
  frozen-E0, and manual E1/E2/E3 schema and surface audits were also clean.

**Non-goals**

- Treating namespaces or aliases as a general authorization/scoping system.
- User/org/team/branch scope hierarchies, cross-namespace federation, or
  automatic alias retirement based only on elapsed time.

## E4 — Ship versioned export/import round trips

**Status:** In progress

**Depends on:** E1, E2, E3

**Outcome:** A user can export one namespace, import it into a fresh Haunt home,
and retain the canonical memory semantics and audit information.

**Acceptance criteria**

- Define a canonical UTF-8 JSON export envelope with a format name, major/minor
  version, creation metadata, canonical namespace identity/labels, and ordered
  records for sessions, events, memories, correction lineage, structured
  provenance, and graph evidence needed to preserve current behavior.
- Export active and superseded records at an explicit temporal cut. Never
  export previously purged bytes. The command clearly states that an export
  contains the namespace's potentially sensitive verbatim data.
- Exclude embeddings, vector tables, FTS tables, embedding jobs, absolute local
  paths, WAL/SHM state, and other rebuildable caches. Import rebuilds FTS and
  graph materializations and queues/rebuilds embeddings under the destination's
  configured model.
- Canonically order keys and records and define volatile manifest fields so two
  exports of an unchanged namespace at the same cut have the same content
  digest.
- Make import transactional and idempotent by bundle/record identity. Reject an
  unsupported major version, malformed digest, alias collision, duplicate ID
  with different bytes, or invalid provenance before any destination mutation.
  Accept known older minor versions through explicit migrations.
- Every import resolves finite positive budgets for input/decompressed bytes,
  total records, bytes per record, JSON nesting depth, and collection items per
  record. Safe defaults and clamps are committed with the format; CLI/API output
  reports the resolved limits. Declared counts are preflight hints only—the
  streaming parser enforces actual counts/bytes/depth and aborts on overrun.
  Compressed input, if supported, is charged by decompressed bytes.
- Limit, parse, validation, timeout, or resource-exhaustion failure rolls back
  the transaction and closes any temporary resources. It may change SQLite
  allocation/WAL bytes, but it commits no new or changed namespace, alias,
  session, event, memory, correction, provenance, graph, FTS, vector, or
  embedding-job rows.
- After import into a fresh home, event/memory IDs, verbatim content, current
  and as-of recall membership, trace lineage, timestamps, trust labels,
  provenance, and namespace resolution match the source. Re-export has the same
  canonical semantic digest.

**Tests/evidence**

- Golden-bundle schema tests and a source -> export -> fresh import -> export
  comparison in FTS-only mode.
- A deterministic stub-embedding test proves excluded embeddings are queued or
  rebuilt without requiring source-model compatibility. The pinned real hybrid
  profile is exercised in E6.
- Boundary tests cover each resolved input/record/byte/depth/item limit, false
  declared counts, decompression expansion when applicable, timeout, and parser
  cleanup. They assert no committed logical mutations or jobs; byte-identical
  SQLite files are explicitly not the proof.
- Failure tests corrupt each record class and assert the same logical rollback.
  A purge test proves erased content is absent from the bundle and cannot
  reappear after import.

**Implementation evidence (2026-08-26; independent merge review pending)**

- The v1 implementation fixes a stable data-derived default cut and explicit
  historical-cut projection, including atomic correction/replacement closure
  and graph entity clocks derived only from retained evidence. Guarded
  zero-write snapshots retry concurrent observe/correct/purge drift; the
  deterministic purge race never returns the erased canary.
- Canonical namespace identity uses stable ID, canonical-first alias ordering,
  closed/acyclic `source_alias_norm` links, and remote identities without local
  paths. Existing import opens exact ID/path/device/inode under the migration
  lock; a deterministic label-retirement/reassignment race leaves namespace B
  unchanged.
- Exact import receipts recheck every durable row before reporting a no-write
  replay and conflict on later tamper or deletion. Scratch SQLite readback
  rejects affinity/type coercion and signed-64 overflow. Legitimate BLOB and
  non-finite REAL values round-trip exactly; BLOB memory content gets neither a
  fake FTS row nor an embedding job.
- The security-review remediation adds an opaque privacy-lineage head to the
  semantic namespace identity. Legacy state has a stable-ID-derived genesis;
  each successful hard purge rotates the head with cryptographic randomness in
  the erasure transaction. Existing imports require an exact match before any
  receipt/record merge, while fresh import preserves the head. Standalone and
  first/middle/last/all correction-chain tests prove raw/base64 canaries cannot
  be restored after purge/restart, independently purged forks diverge, malformed
  heads fail closed, and purge failure rolls back.
- Fresh publication now fsyncs a mode-0600 recovery intent binding the stable
  namespace ID, bundle digest/head, unpredictable token, and exact claimed
  primary/sidecar identities. Subprocess exits at intent, staging, both link
  states, registry precommit, and registry postcommit recover on retry without
  staged database/sidecar/intent residue. Replaced symlink/hardlink targets are
  rejected without deleting unrelated bytes.
- Existing import completes an exact-ID zero-write schema/head/receipt/record
  preflight before opening a maintenance-free guarded writer, then repeats the
  checks inside its write transaction. Rejected conflicts preserve durable,
  projection, job, meta, schema, and SQLite data-version state; a valid existing
  import still rebuilds destination projections.
- Manifest counts now require the exact record-table key set and bounded
  nonnegative JSON integers (booleans forbidden) for each count and total, plus
  an exact sum and equality with actual parser-charged records.
- Token-level parsing enforces actual UTF-8, duplicate-key, input/decompressed
  byte, record/count, depth/item, and timeout budgets. Injected whitespace,
  scratch-validation, and SQLite-progress timeouts clean up without a
  namespace or job; corrupt versions/digests and every durable record class
  likewise commit no logical mutation.
- CLI/Python semantics are mirrored by admin-only MCP export/import and the
  launch-token dashboard APIs. Dashboard tests cover authentication, trusted
  Origin, media type, bounded body, conflict status, safe download filename,
  and unknown-export no-create. Purge tests scan raw and base64 canaries before
  import and after re-export.
- Author evidence: focused portability passed 50/50 on Python 3.10 and 3.12;
  the broad Python 3.10 compatibility suite passed 800 with 5 optional skips
  and 7 expected failures. Ruff, compile, canonical fixture, and diff checks
  were clean. A full native Python 3.12 run was not used as evidence because
  its model fixture repeatedly downloaded outside the pinned cache.
- Independent root pre-review evidence: portability passed 50/50 on Python
  3.10; portability plus E3 alias/sidecar integration passed 182/182 on exact
  Python 3.12/SQLite 3.43/MCP2; diff check was clean. E4 remains **In progress**
  until a fresh independent GPT-5.6 Terra review approves the exact commit.
- Security-review remediation author evidence on the review-candidate tree:
  portability, including purge-head/publication-crash/count/preflight attacks,
  passed 83/83 on Python 3.10 and 3.12; the native purge, correction-lineage,
  namespace-sidecar/alias, E3, and #48-#52/#67-#68 selection passed 237/237 on
  both runtimes. The full Python 3.10 suite passed 837 with 1 optional skip and
  7 expected temporal-generalization failures. Python 3.12 reached 100% of the
  same native tests, then the embedding runtime aborted during interpreter
  teardown (`recursive_mutex lock failed`), so that full process is explicitly
  not counted green; the exact 83- and 237-test Python 3.12 matrices are the
  cross-version evidence pending fresh independent security/correctness review.

**Verification and open gaps (2026-08-27, `integration/all-work` `d1aec40`)**

The implementation is real, not scaffolded: export, import, limits, budgets,
rollback, and receipts execute against a live store. Determinism and the fresh-
home round trip were re-confirmed (see the reconciliation register's integration
state). Four gaps stand between here and merge.

- The purge-lineage claim above is overstated. It prevents resurrection only
  within the same namespace lineage. The implementation's own test is honest
  about this: `tests/test_portability.py:1305-1319` imports a pre-purge bundle
  into a fresh `HAUNT_HOME` and asserts the purged canary **is** restored
  (`assert event["tool_output"] == blob_canary`). Correct the PR and doc
  wording, not the test. (R23)
- The default limits are not co-feasible. `_StrictJSONParser`
  (`src/haunt/portability.py:679`) scans byte-at-a-time in pure Python at
  roughly 5 MB/s, so the default 64 MiB `input_bytes`
  (`src/haunt/portability.py:86`) consumes about 13 s of the default 30 s
  `timeout_seconds` (`:92`). Raising `input_bytes` toward the 256 MiB clamp
  (`:95`) without raising `timeout_seconds` yields an opaque timeout. (R24)
- `FORMAT_MINOR = 0` (`src/haunt/portability.py:61`) leaves "Accept known older
  minor versions through explicit migrations" with no exercised path. Recorded
  as an accepted caveat on that criterion, not as scheduled work. (R25)
- Import validates `recall_class` against `RECALL_CLASSES`
  (`src/haunt/portability.py:38`, `:1344`) but never validates `tier`, which it
  carries as a plain field (`:146`, `:154`) into columns with no CHECK
  constraint. With the dashboard's unescaped sinks this is stored XSS from a
  crafted bundle — see M1 in the reconciliation register. (M1)

**Non-goals**

- Byte-copying SQLite databases, syncing live stores, exporting caches, remote
  backup, encryption/key management, or claiming compatibility with MP's full
  UIIR/ExportBundle.
- Import adapters for third-party products; they may target this envelope later.

## E5 — Explain ranking without calling it confidence

**Status:** Done

**Depends on:** E0

**Outcome:** Every hit explains why it ranked where it did using stable,
machine-readable retrieval evidence.

**Acceptance criteria**

- Add a versioned explanation object to explicit recall results containing
  retrieval mode, applied filters, candidate-list membership, vector rank and
  raw distance when present, FTS rank and raw BM25 value when present, each RRF
  contribution, fused RRF score, and final rank.
- Name the fused value `rrf_score` in the explanation. Keep any compatibility
  `score` field documented as an RRF ranking score, never probability,
  relevance confidence, truth confidence, or calibrated strength.
- Explain FTS-only, vector-only, and hybrid results honestly; absent stages are
  marked not run rather than assigned zero evidence.
- Make ties deterministic with a documented stable tiebreaker. Turning
  explanations on or off must not change candidate selection or ordering.
- Include trust labels. When E2's correction/provenance references exist, add
  them without changing ranking semantics; before E2 lands, mark those optional
  references unavailable/legacy rather than inventing them. E2 is an enrichment
  dependency, not a blocker for the v1 explanation object. Trusted text never
  authorizes a tool mutation.
- Expose the same explanation semantics through Python, CLI JSON, MCP, and
  dashboard APIs; human rendering may be implementation-specific.
- The v1 execution evidence must additionally distinguish a physically
  read-only recall from explicit maintenance, report pending embedding jobs as
  observed-not-drained, report the residue-filter/classification capability,
  and give an honest offline/vector-stage reason. Ranked recall defaults to
  excluding raw tool structure and explicit task/tool residue; audit opt-in
  remains explicit and timeline/trace/detail say the filter is not applicable.

**Tests/evidence**

- Golden explanation tests recompute RRF from component ranks and assert exact
  equality with the reported fused score and final ordering.
- Mode tests cover FTS-only, vector-only, hybrid, filtered, empty-token, and tie
  cases. A mutation that labels RRF as confidence must fail a contract test.
- The frozen E0 metrics and hit order remain unchanged except for an explicitly
  reviewed deterministic-tie correction.
- Once E2 lands, integration tests assert its correction/provenance references
  appear in explanations without changing scores or order.

**Completion evidence (2026-08-26)**

- Pre-fix author/release proof reached reviewed head `46649df`; the frozen E0
  corpus and baseline remained byte-for-byte unchanged.
- The pre-fix independent GPT-5.6 Terra review was **CLEAN** at `46649df` with
  broad Py3.10 and exact
  Py3.12/SQLite 3.43/MCP2 matrices, repeated 4x8 fresh-process creation,
  native-vec checks, and compile/diff checks green.
- The exact final GPT-5.6 Terra merge-gate review was **CLEAN** at
  `681419220df2374608fd52965efc8063dc555b2a` after the Linux SIGBUS
  graph-lifecycle fix. PR #82 CI was green on Python 3.10 and 3.12 and the
  branch squash-merged as `ed806b2c0e97f04f87106672cb1ea60b27fe245e`.
- Root release gating also passed the Py3.10 E0-E5 integration selection and
  native four-target matrix; it caught and this approved head fixed the stale
  MCP read-only-opener assertion and fail-closed initialization cleanup before
  approval. Calibrated abstention remains E6 work, not E5.

**Non-goals**

- Natural-language explanations from an LLM, causal claims about why content is
  true, or exposure of embeddings themselves.
- Calibrated abstention; that is E6 and uses separate evidence.

## E6 — Add evaluation-calibrated abstention

**Status:** Blocked

**Depends on:** E0, E5

**Outcome:** Recall can return an honest no-answer result when retrieval evidence
does not support the query.

**Acceptance criteria**

- Define an evidence-strength feature set from raw retrieval signals and query
  coverage; do not threshold the RRF rank score as though it were confidence.
- Before fitting, commit a separately versioned calibration dataset with
  answerable/unanswerable labels and a predeclared immutable fit/held-out split.
  Record the canonical dataset hash and split-definition hash. E0's corpus is a
  regression lock only and must not count as fitting or held-out evidence.
- Define the pinned hybrid profile here, including embedding model ID, dimension,
  retrieval configuration, and fail-loud proof that it did not fall back to
  FTS-only. Calibrate/version separate policies for the FTS-only and pinned
  hybrid profiles.
- Include held-out hybrid answerable cases that are true semantic paraphrases
  with no required lexical/stemming overlap. Lock expected logical IDs and
  report pre-abstention Recall@5 separately from abstention metrics.
- Each calibration artifact records dataset/split hashes, feature definition,
  profile/model ID, threshold, fit-case metrics, held-out metrics, and the exact
  implementation/configuration versions needed to reproduce the features.
- Select thresholds using only the predeclared fitting split. On the held-out split,
  abstain on 100% of designated unanswerable cases while retaining at least 95%
  of answerable cases whose relevant memory ranked in the pre-abstention top 5.
  If the calibration dataset is too small to support those gates, version and
  review an expanded E6 dataset/split before fitting; do not repurpose E0.
- Return an explicit structured result with `abstained`, a stable reason code,
  calibration ID, and the strongest observed evidence. An abstention returns no
  hits; an ordinary zero-candidate query remains distinguishable from a
  thresholded no-answer.
- Fail honest when no calibration matches the active retrieval mode/model:
  report `uncalibrated` and use the documented compatibility behavior rather
  than applying another model's threshold.
- Apply identical policy semantics in Python, CLI, MCP, dashboard, and temporal
  planned recall. Provide an explicit caller override only if it is named as
  disabling abstention in the response metadata.

**Tests/evidence**

- Committed fit and held-out reports for FTS-only and the pinned hybrid profile,
  with a script that reproduces them from the predeclared dataset/split hashes.
- A test runs the pinned hybrid profile and asserts the exact model ID and
  dimension before accepting its calibration result.
- Boundary tests immediately above, at, and below every threshold; profile
  mismatch tests; and out-of-corpus adversarial queries.
- A leakage test proves held-out labels are not read while fitting, and a
  separation test proves E0 cases/results are not calibration inputs. A mutation
  that uses `rrf_score` as the threshold must fail.

**Blocker evidence (2026-08-26)**

- E6 v1 is frozen separately from E0 at dataset hash
  `8119f4508d3582bc665a5a0117940c6eeca593de56f33999563ddf188846264c`
  and split hash
  `d57b1a02e30d48087065528bfc01575d67735295003de26b1087bb348d05414d`.
  Its 40 records and 160 cases have no logical-ID, canonical record, or query
  hash overlap with E0; the sealed E0 paths have a byte-for-byte empty diff.
  Records, unlabeled queries, split membership, fit labels, and held labels are
  physically separate. The held-label file is first opened only after the
  fit-only boundary exists; the composite manifest is first opened and verified
  after held-out scoring, where it reconstructs the same frozen dataset hash.
- The FTS-only fit cohort is separable at `0.875`; held-out Recall@5,
  conditional retention, and negative abstention are all `1.0`. This proves the
  harness but does not close E6 because a separate pinned hybrid profile is
  required.
- Under hashed local `BAAI/bge-m3` ONNX inputs (dimension 1024) and actual
  sqlite-vec native cosine execution, the fit-only boundary needed for 100%
  negative abstention retains only `6/20` conditional positives (`0.30`, below
  the required `0.95`). Held-out pre-abstention Recall@5 is `15/20` (`0.75`);
  at that unchanged fit-only boundary, all 20 negatives abstain but only `6/15`
  conditional positives remain (`0.40`). Close absent-attribute negatives
  dominate real semantic positives on both strength and the diagnostic top-two
  distance margin, so no approved raw-evidence threshold can meet both gates.
- Reproduction runs with ambient fake API/Hugging Face keys. FTS uses strict
  `HAUNT_OFFLINE=1`; hybrid leaves it unset, requires an explicit verified local
  cache, denies socket/DNS/HTTP access, proves zero attempts, and fails on any
  non-native vector arm. Coverage evidence is case-folded/deduplicated and uses
  one batched SQL statement for the fixed top five; 1k/10k/100k timing evidence
  is recorded separately as a non-cross-machine gate.
- Hybrid preflight accepts only committed artifact manifest
  `haunt-bge-m3-onnx-split-f8425123-v1`: exact relative paths, sizes, and hashes
  for config, nonquantized ONNX, required external-data sidecar, tokenizer, and
  tokenizer config. Missing/extra/zero-byte/size/hash/sidecar/variant mismatches
  fail before embedding initialization; quantized and root-level variants are
  explicitly forbidden.
- Evidence and reproduction live in `src/haunt/abstention_eval.py`,
  `scripts/reproduce_abstention_eval.py`,
  `scripts/benchmark_abstention_evidence.py`, and
  `tests/fixtures/abstention_eval/v1/`. No public runtime policy, threshold
  artifact, CLI/MCP/dashboard behavior, or fallback compatibility behavior is
  shipped while the hybrid gate is unsatisfied.
- Honest unblock choices are: predeclare and review a new raw retrieval feature
  in a new E6 evidence version; explicitly amend the contract to permit a
  reader/cross-encoder; or leave hybrid abstention unshipped. Removing or
  relabeling hard negatives, padding with lexical positives, thresholding RRF,
  fitting held-out labels, or silently falling back to FTS are not valid fixes.

**Blocker review (2026-08-27)**

The harness is sound and should merge. Its integrity controls were re-verified and
real: held labels are physically separate and first opened only after the fit
boundary exists (`src/haunt/abstention_eval.py:1398-1411`), the composite
manifest is verified only at `verification_phase == "after_held_out_scoring"`
(`:508-513`), and the BGE-M3 artifact identity is enforced before embed
initialization by `verify_local_hybrid_cache` (`:659`). No label leakage was
found; model pinning is genuine.

The **conclusion** above does not follow. The dataset forces the result.

- All 40 hybrid answerable queries carry `required_lexical_overlap: false`
  (`tests/fixtures/abstention_eval/v1/queries.json`, 40 occurrences, 0 set
  `true`), and the rule is applied to **both** splits
  (`tests/fixtures/abstention_eval/v1/split.json:17`). The fixture additionally
  requires the gold row to be absent from the complete FTS candidate set
  (`tests/fixtures/abstention_eval/v1/README.md:18-21`).
- The negatives are under no such constraint and keep on-topic wording. Mean
  max-Jaccard of negatives against positive queries is 0.160-0.164 with 19-20 of
  40 at or above 0.20, while positives against *other positives* is 0.079 with 0
  of 40 at or above 0.20. The positives are more dissimilar from each other than
  the negatives are from them.
- Blocker mass is concentrated in two cases: dropping `hybrid-fit-u04` moves fit
  retention from 6/20 to 12/20; dropping `u04` and `u10` moves it to 17/20. The
  dominating pairs are `u04` against `a16`/`a18`. The 40 negatives were checked
  individually and contain zero mislabels — the problem is the positives.
- The contract was misread. The acceptance criterion at `BACKLOG.md:647-649`
  requires held-out answerables that are paraphrases with **no required**
  lexical overlap; the implementation applied **zero** overlap to every hybrid
  answerable in both splits.

**A named acceptance criterion is directly unmet.** Held-out conditional
positives number 15, so a 95% gate means 15/15, and held-out pre-abstention
Recall@5 is only 0.75
(`tests/fixtures/abstention_eval/v1/reports/hybrid-blocked.json`,
`held_out.metrics_at_fit_only_diagnostic_boundary`). The threshold criterion
at `BACKLOG.md:653-657` states that "If the calibration dataset is too small to
support those gates, version and review an expanded E6 dataset/split before
fitting" (`:656-657`). That escape hatch was triggered and ignored; fitting
proceeded anyway.

**The accurate statement of the result** is: *raw BGE-M3 cosine cannot separate
absent-attribute negatives from deliberately zero-overlap paraphrase positives
at n=15.* It is **not** "abstention is infeasible", and it must not be recorded
as a scientific finding of infeasibility.

The three honest unblock options above stand, restated against this review:

1. Predeclare a v2 feature family **and** a v2 dataset whose paraphrases span
   zero-to-moderate overlap, with enough held-out positives that 95% is not a
   single-case coin flip.
2. Explicitly amend the contract to admit a reader/cross-encoder.
3. Leave hybrid abstention unshipped and amend E7.

**Standing recommendation: option 1.** Option 2 must not be taken on the
strength of an experiment that never fairly tested the light signals. Option 3
is a defensible schedule call. This survives the strongest counter-argument —
that zero-overlap paraphrase is E6's legitimate hard case — because the
contract's own escape hatch was triggered and ignored.

E6 therefore remains **Blocked**, on the dataset and the unmet dataset-size
criterion rather than on a proven infeasibility. E7 remains blocked on E6.

**Non-goals**

- Fact confidence, truth verification, source reputation, a reader LLM, or a
  universal threshold shared across retrieval modes/models.
- Treating the E0 exact-result regression corpus as calibration evidence.
- Guaranteeing that retained results are correct; abstention controls evidence
  sufficiency, not truth.

## E7 — Produce final end-to-end and release proof

**Status:** Blocked

**Depends on:** E4, E6

**Outcome:** The complete adoption slice is proven on fresh and upgraded stores
and is ready for a normal Haunt release.

**Acceptance criteria**

- Run one scripted journey from an empty `HAUNT_HOME`: bootstrap; observe native
  and imported memories; correct twice; trace; migrate a namespace alias;
  export; import into a second empty home; recall with explanations; demonstrate
  a positive answer and calibrated abstention; purge one lineage member; export
  again; and prove erased bytes do not return.
- Run the same semantic journey against a committed pre-adoption database
  fixture. Opening/migration is idempotent, preserves existing IDs/content, and
  produces explicit legacy provenance/lineage labels where reconstruction is
  impossible.
- Pass the full test suite on supported Python versions in FTS-only mode and a
  pinned hybrid release run that asserts the loaded model/dimension. No skipped
  or xfailed adoption test may be counted as evidence.
- Publish the E0 exact-result/Recall@3 before-and-after comparison and the E6
  held-out calibration reports. Any regression or threshold gate miss blocks
  release unless its separately versioned corpus/config or calibration
  dataset/split and this contract change in an explicit review.
- Update README, CLI/MCP help, dashboard copy, schema version notes, changelog,
  privacy/export warnings, and recovery guidance to match shipped behavior.
- Audit the final diff and public surfaces against
  `docs/MEMORY_CONTRACT.md`: no generic DAG, HMAC log, scope hierarchy,
  distillation, fact confidence, or embedding export has entered the product.

**Tests/evidence**

- CI logs and machine-readable end-to-end reports name the commit, Python
  version, platform, retrieval profile, E0 corpus/config hashes, E6 calibration
  dataset/split hashes, export format version, schema version, and calibration
  ID.
- A clean install smoke test and an upgrade smoke test both execute public CLI
  and MCP surfaces, not private helper-only shortcuts.
- Release notes link each epic to its tests and evidence artifact.

**Non-goals**

- New memory tiers, hosted services, team/org authorization, cross-store sync,
  or unrelated dashboard redesign.
- Declaring conformance with the complete Memory Protocol specification.

## Program-level non-goals

- Haunt will not adopt MP's generic event DAG, HMAC-signed log, hierarchical
  scopes/capability system, distillation ladder, derived-fact confidence, reader
  LLM, warehouse tiers, federation, or multi-agent coordination surface.
- This program does not turn namespace aliases into permissions or file-per-
  namespace storage into an authorization boundary.
- This program does not weaken explicit privacy erasure, export embeddings, or
  silently synthesize provenance that the source did not provide.

## Corpus health and capture policy (C-series)

**Status:** Ready

The E-series above adopts new capabilities. The C-series repairs what Haunt
already does to a live store. These items were found on 2026-08-26 by measuring
the dogfooded corpus under `~/.haunt/namespaces/`, not by reading the contract,
and several are firing in production today.

Every count below is from a live, growing corpus. The direction is the
evidence; the last digit is not.

### Measured baseline (2026-08-26)

| Namespace | `repo_path` | Memories | Unembedded | Registered |
|---|---|---:|---:|---|
| `aronriley` | *(blank)* | 58 | 3 | 2026-08-21 |
| `default` | *(blank)* | 0 | 0 | 2026-08-21 |
| `haunt` | `github.com/moderatesoup/haunt` | 211 | 4 | 2026-08-21 |
| `memory-protocol` | *(blank)* | 151 | 0 | 2026-08-22 |
| `ironscope` | *(blank)* | 156 | 0 | 2026-08-22 |
| `github.com-moderatesoup-ironscope` | *(blank)* | 313 | 313 | 2026-08-25T17:55:08 |
| `github.com-memory-protocol-memory-protocol` | *(blank)* | 1491 | 1363 | 2026-08-25T17:55:15 |

Two repositories hold memory in two namespaces each, and the duplicate pair was
registered seven seconds apart. `haunt` is the only namespace with a populated
`repo_path` and the only one that did not fork.

Across the six non-empty namespaces, `role='tool'` rows are 79.8% of all
memory, and 8.1–13.5% of rows are exact byte-duplicates of another row
depending on the grouping cut.

### Relationship to the E-series

- C1–C3 sit beneath **E3 — namespace aliases** and are not satisfied by it.
  E3 adds identity tables and a label migration, but that migration explicitly
  never moves, copies, or renames a namespace database; its legacy backfill
  skips registry rows whose `repo_path` is blank; and it groups legacy rows by
  `db_path`, so two database files are never unified. The split stores above
  survive E3 unchanged.
- C8 is **evidence for E6 — calibrated abstention**, not a competing epic.
- Two ranking defects found in the same investigation — `vec_distance` and
  `fts_rank_raw` computed then dropped from `as_dict()`, and the dashboard
  sorting RRF scores across independent namespace pools — are already resolved
  on `codex/ranking-explanations` and are deliberately absent from this list.

### Namespace integrity

**C1 — Persist `repo_path` on every namespace registration**

Hooks and MCP build `Store(ns)` with no repository
(`src/haunt/claude_hook.py:225`, `src/haunt/cursor_hook.py:477`,
`src/haunt/mcp_server.py:169`), so hook-created namespaces register with a
blank `repo_path`. `_registered_namespace_for_repo()` matches only on
`repo_path` and skips blank rows (`src/haunt/paths.py:107`), so reuse can
never match, and `infer_namespace()` falls through and mints a second namespace
for a repository that already had one (`src/haunt/paths.py:148-172`). Only
`haunt init --repo` supplies the value today.

- Every path that can create a namespace records the repository it was inferred
  from, or records explicitly that there was none.
- Reuse matches repositories whose registry row predates the fix.
- A repository with an existing namespace never mints a second one, asserted
  across the hook, MCP, and CLI entry points.

**C2 — Refuse to derive a namespace from a non-repository directory**

`infer_namespace()` falls back to the working directory's name
(`src/haunt/paths.py:169-171`). A session started in the home directory
produced the namespace `aronriley`, holding 58 rows that are every one of them
the identical string `haunt session start` — pure ceremony, zero knowledge.
`_is_user_home()` already exists (`src/haunt/paths.py:181`) but is used only in
permission paths, never in inference.

- The home directory never becomes a namespace name.
- A non-repository working directory resolves to `default` or fails loudly; it
  never silently mints a directory-named store.
- Existing junk namespaces are reportable so an operator can retire them.

**C3 — Reconcile namespaces that are already split**

E3 hard-fails when a label already maps to another namespace, which is correct
for aliasing but leaves the four forked stores above with no path forward.

- An operator can merge two namespaces that resolve to the same repository,
  preserving rows from both sides and keeping event identity stable.
- Dry-run reports exactly what would move before anything moves.
- The operation is idempotent, and reversible or backed up before it writes.

### Embedding pipeline

**C4 — Hook-deferred embeddings have no drain**

Hook writes pass `defer_embedding=True`, and `observe()` drains its queue only
when `commit and not defer_embedding` (`src/haunt/store.py:916`), so a
hook-driven write never triggers the drain that would clear it; the queue
shrinks only when something calls `recall()`. Measured result: 1363 of 1491
rows unembedded in one namespace (91%), and 313 of 313 in another (100%) — the
latter has no `vec_memories` table at all, making it FTS-only without ever
saying so.

- A drain runs independently of read traffic.
- Embedding coverage is reportable per namespace, and a namespace with no
  vector index says so rather than silently degrading to FTS-only.
- Sequencing: land C6's policy first. Draining before that spends compute
  embedding precisely the rows C6 excludes.

**C5 — Isolate per-row failures in `process_embedding_jobs`**

One permanently failing row blocks every job queued behind it, with no attempt
ceiling and no surfaced error (`src/haunt/store.py:1105`).

- Per-row failure isolation, a max-attempt cutoff, and failures visible rather
  than silent.

### Capture policy and duplication

**C6 — Exclusion should skip embedding, not drop capture**

`HAUNT_EXCLUDE_TOOLS` returns before `observe()` is ever called
(`src/haunt/claude_hook.py:191`, `src/haunt/cursor_hook.py:354`, `:375`,
`:396`), so an excluded call leaves no event, no memory row, and no FTS entry.
That is irreversible, and it destroys the forensic record — the same record
this investigation used to find every defect on this list. The log should stay
complete; the embedder is where the cost actually is.

FTS insertion already runs unconditionally (`src/haunt/store.py:978`), so
keyword recall over un-embedded rows needs no new mechanism.

- Excluded tool I/O is still captured and still keyword-searchable.
- Policy is per-tool, not per-category: `Bash` is the flood, `Edit`/`Write`/
  `Task` are not.
- Session ceremony (`src/haunt/cursor_hook.py:446`,
  `src/haunt/claude_hook.py:157`) is excluded from embedding.
- The embed call and the `embedding_jobs` enqueue are skipped together
  (`src/haunt/store.py:955`, `:985`); skipping only one either inflates
  the backlog or embeds the rows anyway once C4 lands.
- Open question to settle before building: whether excluded rows should be
  retroactively embeddable on demand. That is the difference between reversible
  in principle and reversible in practice.

**C7 — Exact-content-hash dedup and reference-not-copy**

No `content_hash` exists. The only dedup is idempotency-key replay
(`src/haunt/store.py:447`, `:913`), which catches a redelivered hook
call but not two distinct calls producing identical bytes. This is orthogonal
to C6: the measured duplicates include `role='system'` ceremony rows that are
not tool I/O at all.

- Hash at admission on identical bytes only, never semantic similarity.
- A repeat hash writes a reference to the original rather than a second
  payload, preserving that the event genuinely recurred.
- Re-measure the duplicate fraction after C6 lands to size what remains.

*Superseded as of 2026-08-27: phase 1 shipped. Schema v10 adds
`memories.content_hash` and `idx_memories_content_hash`
(`src/haunt/store.py:1703-1711`), so "No `content_hash` exists" above no longer
holds. The measurement it enables is not reachable from the CLI (R19), and it is
never populated by an E4 import (R7).*

**C7 phase 2 (reference-not-copy) — dropped (2026-08-27)**

Re-measured across all seven namespace databases: 11,017 rows, 394 extra copies
(3.58%), wasting 713.2 KB of 38.3 MB of content (**1.86% of bytes**). Of the 394,
229 are tool exhaust (Bash 142, Read 74, and 13 in `ToolSearch`/`mcp`/`Write`/
`ScheduleWakeup` that the default exclusion does not cover), 161 are
`haunt session start` ceremony, and **4 are genuine**.

The decision stands on those two numbers — 1.86% of bytes and 4 genuine
duplicates in 11,017 rows — and not on the earlier rate comparison. Both
duplication measurements were correct; the corpus grew about 5x during
development (2,200 to 11,017 rows), so the rate fell by dilution while the
absolute count rose. Two parts of the original justification do not hold:

- "Already neutralized by C6" does not address phase 2. C6 sets
  `skip_embedding`, which keeps a row out of the vector index but still
  **stores** it (`src/haunt/store.py:5119-5123`). Phase 2 was a storage
  proposal.
- Of the 161 ceremony duplicates, only 85 sit in the junk `aronriley` namespace
  that C2 prevents; 76 are in legitimate namespaces where C2 does nothing.

### Retrieval quality and cost

**C8 — Reproducible abstention failure (evidence for E6)**

A long query about git-history tooling, run against namespace `aronriley`,
returned ten hits that were every one of them the string `haunt session start`,
with `fts_rank: null` on all ten and scores of exactly `1/(RRF_K + rank)` —
0.016393, 0.016129, 0.015873, down the floor.

The FTS half was right to return nothing: the query shares no token with the
only string in that corpus. The defect is that vector search returns k-nearest
with no distance floor, so a corpus with nothing relevant still fills k. The
consuming agent had to read the hit contents to conclude the recall was noise.

- Recorded as a fixture for E6 rather than a separate epic. E6 owns the
  abstention contract; this is the case it must handle.

**C9 — FTS5 tokenizer fragments code identifiers**

`tokenize='porter unicode61'` (`src/haunt/store.py:255`) splits `snake_case`
and `dotted.paths` and never splits `camelCase`, while the corpus is almost
entirely coding sessions.

- Keyword recall finds identifiers in all three shapes.
- Any tokenizer change re-runs the E0 frozen gate, since it moves FTS ranks.

**C10 — Unindexed validity scans**

Default recall filters `m.valid_to IS NULL` (`src/haunt/recall.py:113`) with no
supporting index, so the current slice scans every superseded row ever written.
`trace()` loads a namespace's whole correction history per call
(`src/haunt/store.py:2036`) instead of walking existing indexes.

**C11 — Recall response has no size budget**

`k` accepts up to 100, and each hit carries full untruncated `content` plus a
redundant 200-character `snippet`. `codex/ranking-explanations` adds a per-hit
`explanation` object on top, so payloads grow rather than shrink.

**C12 — Diversity and reranking**

There is no MMR or diversity step, so near-duplicates can occupy the whole
top-k, as C8 shows in the extreme. `src/haunt/recall.py:3` documents the
cross-encoder as deliberately not wired.

- Measure before adopting. Extend the frozen evaluation to hybrid retrieval
  first (E6 already owns pinned hybrid evaluation), then compare recall quality
  with and without a rerank pass on clear-top-1 and ambiguous-candidate
  queries.
- A reranker cannot repair C8: nothing can rerank a corpus holding one distinct
  string. C1–C4 land first or rerank evaluation measures noise.

**Tests/evidence**

- A namespace-integrity test creates a namespace through each entry point and
  asserts a repository never yields two namespaces.
- A migration test merges two populated namespaces and proves no row is lost,
  duplicated, or re-identified.
- Coverage reporting proves the embedding backlog is drainable and observable.
- A capture-policy test proves excluded tool I/O remains keyword-recallable.
- The dedup test asserts identical bytes collapse to a reference and that
  distinct rows never collapse.
- Any change touching FTS ranks or fusion re-runs the E0 frozen gate.

**Non-goals**

- Semantic or embedding-based deduplication at write. Its failure mode is
  silent suppression of genuinely distinct facts.
- Dropping capture as a corpus-size remedy. Size is managed at the embedder and
  in the view, not by discarding the log.
- Redefining `trusted`, which is an authority label, as a visibility switch.

## Reconciliation register (2026-08-27)

Verified against `main` `ed806b2` and `integration/all-work` `d1aec40`, then
extended by a late pass at `integration/all-work` `413f8b9` that closed 19 of
these rows, added six new ones (L11-L16), and recorded the verification
evidence below. This register supersedes every prior audit of this repository. **Audits written
against `88f607f` are 15 commits stale** — that revision is 15 commits behind
`main`, and most of what those audits report was fixed in between. The closed
table below is the list; do not re-open anything in it without new evidence.

`Checked` reads: `verified` — citation re-checked during this reconciliation, at
`d1aec40` for R1-R30 and M1-M3, at `413f8b9` for L11-L16; `operator` —
re-checked directly by the operator; `unverified` — carried from the source
ledger and **not** independently re-checked, so treat both the citation and the
finding as unconfirmed.

### Open items

| ID | Item | Severity | Evidence | Checked |
|---|---|---|---|---|
| R7 | `content_hash` is NULL after an E4 import into a fresh home. `portability.py` has zero `content_hash` references and the C7 backfill runs only in the `current < 10` migration step, which never fires on a fresh v11 database. C7 duplicate measurement silently reads zero in any imported namespace. Not data loss — recomputable. | MEDIUM | `src/haunt/portability.py` (0 hits); `_backfill_content_hashes` reached only at `src/haunt/store.py:1703-1717` | verified |
| R8 | Remote-less repositories still collide on the directory leaf name: two local-only repositories named `api` share one memory database. Misses worktrees and pre-`git remote add` repositories. | MEDIUM | fallback at `src/haunt/paths.py:1707` `safe_name(repo_root.name)` | unverified |
| R9 | Prompt injection is mitigated at the highest-volume channel only, not closed. `trusted=False` covers raw tool I/O; assistant replies and user prompts replay verbatim into the next session's `additionalContext` with no data/instruction delimiter. Laundering path: poisoned page to tool output (excluded) to assistant summary (stored trusted) to replay. | MEDIUM | `src/haunt/recall.py:138`; hook injection in `src/haunt/claude_hook.py`, `src/haunt/cursor_hook.py` | unverified |
| R1 | Inspection surfaces (timeline, health, worldview, trace, procedure get/list, and eight dashboard GET routes) open a **read-write** connection and take the writer lifecycle lock: they need write permission and serialize concurrent inspection. Entity surrogate UUIDs are not stable across a rebuild. **Not** a per-read rebuild — `_ensure_graph_evidence` is a version-gated one-shot backfill returning immediately once `graph_evidence_version == "1"`; measured steady-state mutation is zero. **Not** an undocumented invariant violation — `README.md:130-136` and `docs/MEMORY_CONTRACT.md:211-217` scope the read-only requirement to recall, and `MEMORY_CONTRACT.md:206-208` names timeline/trace/detail as a different class. Read-only media fail at WAL configuration first. GitHub issue #84. | LOW-MEDIUM (downgraded from HIGHEST) | `src/haunt/store.py:4886-4897`, `_rebuild_graph_with_configuration_lock` at `:5911` | operator |
| R12 | Dashboard async routes block the event loop (no `to_thread`/`run_in_threadpool` anywhere) and re-embed the same query once per namespace with no query-vector cache. Latency only, single user. | LOW-MEDIUM | `src/haunt/dashboard.py` recall routes; `src/haunt/recall.py` `embed_one` | unverified |
| R24 | E4 default limits are not co-feasible: `_StrictJSONParser` runs at ~5 MB/s, so the default 64 MiB input cap consumes ~13 s of the default 30 s timeout. See the E4 gap list. | LOW-MEDIUM | `src/haunt/portability.py:679`, `:86`, `:92`, `:95` | verified |
| R15 | The `execution` block is computed from unbounded hits rather than `bounded_hits`. | LOW | `src/haunt/mcp_server.py:831` region | unverified |
| R21 | No ruff or mypy configuration and no CI gate. Current state: ruff 13 findings (7 `F401`, 5 `F841`, 1 `E741`; only one in `src/` — `desktop.py:5` unused `os`); mypy 1.11.2 reports 57 errors across 11 files, about 17 from `MCPNamespaceAccess` attribute/assignment. | LOW code risk / MEDIUM process | `pyproject.toml` (no `[tool.ruff]`/`[tool.mypy]`); `.github/workflows/ci.yml` runs neither | unverified |
| R23 | E4's purge-lineage claim is overstated in the PR and doc wording. See the E4 gap list; the test itself is honest. | LOW code / MEDIUM honesty | `tests/test_portability.py:1305-1319` | verified |

9 open items above. Nothing here is a merge blocker: R6 was the only one and
it is fixed and closed below.

### LongMemEval retrieval (2026-08-27)

Operator-run and judge-free, FTS-only, `k=10`, against clean `main` `ed806b2`.
Harness `scripts/score_lme_retrieval.py`, which lands at `4973169` on
`perf/embed-batching` and is **not** in `d1aec40`. Dataset
`longmemeval_s_cleaned.json`, n=500, seeded stratified split of 350 working and
150 held-out.

| Split | n | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| working | 350 | 0.851 | 0.920 | 0.937 | 0.960 | 0.890 |
| held-out | 150 | 0.887 | 0.927 | 0.960 | 0.973 | 0.915 |

The held-out split exists to verify generalization. Do not diagnose against it,
do not tune against it, and do not report a per-type held-out cell without the
power check in L8.

| ID | Item | Severity | Evidence | Checked |
|---|---|---|---|---|
| L1 | `single-session-preference` is the weakest retrieval type, by lexical signal inversion. Working R@1 0.429 / R@10 0.714; held-out 0.444 / 0.889; every other type is 0.95 to 1.00 at R@10. The mechanism is measured, not inferred: lexical margin = (best gold-turn content-word overlap with the query) minus (best non-gold-turn overlap), and margin predicts R@1 monotonically across all six types - knowledge-update +0.334/0.945, single-session-user +0.259/0.959, single-session-assistant +0.232/0.872, multi-session +0.107/0.860, temporal-reasoning +0.098/0.817, single-session-preference -0.089/0.429. Preference is the only negative margin: the best distractor matches the query better than anything in gold, so BM25 cannot rank gold first. The cause is not benchmark-specific - a preference question names the task ("what should I serve for dinner") while the gold session holds the constraint ("my homegrown cherry tomatoes, basil, mint"); those vocabularies are disjoint by construction, and a real haystack reliably contains a session that genuinely is about the task. | MEDIUM (product) | working n=350, held-out n=150 at `ed806b2`; GitHub issue #37, which until now had no measurement behind it | operator |
| L2 | Hybrid retrieval fixes L1 at zero code cost, so the FTS-only figure understates the shipped product. Measured on the 16 working-set preference questions with embeddings fully drained (bge-m3, coverage 1.0): preference R@10 0.688 to 1.000, R@1 0.312 to 0.438, 10 improved and 0 regressed, all five completed misses recovered. Evidence, not a task: the action is to run and report hybrid, not to change code. | n/a (evidence) | working preference subset n=16, embedding coverage 1.0 | operator |
| L3 | The harness's `ranking_fusion` miss bucket is vacuous and should be replaced. In FTS-only mode there is one modality, and RRF `1.0/(RRF_K + rank)` is strictly decreasing in rank, so the fused order is identical to the BM25 order - there is no fusion to fail. The bucket fires whenever gold lands anywhere in the candidate pool, which is always. Note also that `_fts_hits` is called with `CANDIDATES` = 40, not `k` and not a probe constant. Working-set recall@40 = 1.000, so candidate generation never fails; ordering within 40 is the whole problem (R@10 0.960, R@15 0.971, R@20 0.977). Replace it with a `margin < 0` bucket, which labels 14/14 misses meaningfully. | LOW (harness correctness) | `src/haunt/recall.py:30`, `:31`, `:577`, `:614`, `:617`, `:620` | verified |
| L4 | `CANDIDATES` = 40 also caps each modality independently before fusion, so hybrid mode fuses two 40-candidate pools rather than widening the pool. Untested in FTS-only. Measure it on its own before the L2 hybrid proof run, or the two effects confound. | LOW (unmeasured) | `src/haunt/recall.py:31`, `:577`, `:592` | verified |
| L5 | `temporal-reasoning` misses are the documented implicit-time limitation, now quantified. Questions carrying a relative-time expression: n=26, R@1 0.615, R@10 0.808. Questions without: n=67, R@1 0.896, R@10 1.000. All five misses fall in the relative group. Bolting relative-time date parsing onto the query path is explicitly **not** recommended: it contradicts a stated design decision and would be dataset-tuning. | LOW (known limit, quantified) | working temporal-reasoning n=93; GitHub issues #39 and #40 | operator |
| L6 | No fuzzy fallback exists for a query anchor that is absent from the index. `tokenize='porter unicode61'` means a query term with no index entry cannot match at all. Two working-set cases turn on this: `homegrown` (df 0/503) and `buisiness`, a misspelling in the benchmark's own question (df 0/476). Recorded as a product property worth knowing; no fix proposed. | LOW (known limit) | `src/haunt/store.py:1395` | verified |
| L7 | Candidate fix, **not adopted**: stopword-filter the FTS query only, roughly 6 lines. Measured over all 350 working questions: overall R@1 0.851 to 0.866 and R@10 0.960 to 0.974; preference R@10 0.714 to 0.905; single-session-assistant R@1 0.872 to 1.000; 6 gained and 1 lost at R@10. It is underpowered - no cell reaches p<0.05 (McNemar: assistant R@1 5/0 p=0.063; preference R@10 4/0 p=0.125; overall R@10 6/1 p=0.125) - and single-session-user R@1 regresses 0.959 to 0.918. Sensitivity: a 33-word minimal IR stoplist recovers the assistant gain but none of the preference gain, so the effect lives in mid-frequency closed-class words and is list-dependent. Governing constraint if it is ever adopted: one published list, frozen a priori. Curating membership against working-set recall would be tuning and would manufacture a held-out failure. Removing `not` is separately risky for negation queries. | LOW (not adopted) | working n=350; McNemar exact, two-sided | operator |
| L8 | Held-out power warning. Held-out `single-session-preference` is n=9, where a +0.19 R@10 effect moves about 1.7 questions. Judge generalization on pooled R@10 (n=150) and on single-session-assistant R@1 (n=17), never on that cell. | n/a (method) | held-out split composition, n=150 | operator |
| L9 | Embedding ingest wastes throughput on padding: `tokenizers` pads each batch to its longest member, so a randomly ordered batch of 100 turns is almost always padded to 512, roughly 2.6x the tokens actually needed. Length-sorting before batching measured 3.34x (3.14 to 10.49 texts/s at batch=16) with vectors unchanged (min cosine 1.00000000, max absolute difference 6.6e-07). This affects every user's ingest, not only the benchmark. Fix in progress. | MEDIUM (performance, all users) | `perf/embed-batching` `6198083` | operator |
| L10 | `answer_substring_proxy` is not a retrieval signal for `single-session-preference`. It reads 0.000 at every cutoff because the benchmark's preference answers are free-form rationales that never appear in the haystack. Scoring artifact; do not treat it as a miss. | n/a (scoring artifact) | working n=350, all cutoffs | operator |

10 LongMemEval items, none of them a merge blocker.

### Late findings (2026-08-27)

Found after the reconciliation above, during integration and the cleanup pass.
IDs continue the L-series. None is a merge blocker.

| ID | Item | Severity | Evidence | Checked |
|---|---|---|---|---|
| L11 | The `test-hybrid` job was authored and exercised on macOS only and has never run on Linux. Its first CI run is a cold `actions/cache` miss that downloads bge-small (~64 MB) before the key populates. The key is a bare literal with no `restore-keys`, so if the embedding model pin ever changes that key must be bumped or the job silently reuses the wrong cached model. | LOW (operational) | `.github/workflows/ci.yml:61-86`, key at `:78` | verified |
| L12 | Purge is now O(namespace), not O(row): `VACUUM` rebuilds the whole namespace file. Acceptable for a gated, confirmed delete, but `haunt delete --event-id` calls `purge()` once per memory on that event, so the rebuild runs once per memory — that is where it surfaces first. `bytes_overwritten` in the purge report says whether the rebuild actually completed (false when a concurrent reader blocks `VACUUM`); the purge's own bytes are zeroed either way. | LOW-MEDIUM (performance) | `src/haunt/store.py:6484-6508`; per-memory loop at `src/haunt/cli.py:1003-1009` | verified |
| L13 | `drain_embedding_queue(batch_size=N)` silently clamps N to 100, via `clamp_limit`'s `LIMIT_MAX`. Undocumented kwarg and no production caller sets it — `bootstrap` calls with no arguments. `README.md` is **not** wrong: it documents `HAUNT_EMBED_DRAIN_LIMIT`, which controls `max_rows` and genuinely reaches 100,000. A one-line docstring note is the whole fix. | LOW | `src/haunt/store.py:5714-5715`, clamp at `:5510`; `LIMIT_MAX` at `src/haunt/util.py:154`; `README.md:312` | verified |
| L14 | `_load_fastembed` remains unpinned: fastembed exposes no revision knob, so `TextEmbedding(model_name=...)` takes whatever the hub serves. A separate download path from the one the R2 BGE-M3 revision pin closed. | LOW (supply chain) | `src/haunt/embed.py:367-374` | verified |
| L15 | `tests/test_correction_lineage.py::test_concurrent_corrections_do_not_fork` failed once at machine load ~97 with `NamespacePathError: SQLite zero-write read observed storage drift`. Passed 5/5 in isolation at the same load, and in the clean-load full suite. Cause not established. Not a merge blocker; re-check if it recurs. | LOW | `tests/test_correction_lineage.py:359`; one observed failure, not reproduced | operator |
| L16 | Hooks fail **open** on exception, but there is no timeout at all, so a hang is not covered: a contended SQLite lock or a slow disk holds the turn until the host's own hook timeout fires. `SECURITY.md`/`README.md` wording was corrected to "a hook *error* never blocks", which is honest, but the underlying gap remains. Found during the cleanup pass and deliberately not fixed there — behaviour change, out of scope. | LOW-MEDIUM | `SECURITY.md:117-119`; `README.md:158`; no `timeout`/`signal` in `src/haunt/claude_hook.py` or `src/haunt/cursor_hook.py` | verified |

6 late findings. 25 open in the register.

### Verification evidence (2026-08-27)

Plain facts, each re-checkable.

- Full suite on the integrated tree: 1157 passed, 1 skipped, 7 xfailed, 0
  failed, of 1165 collected.
- Retrieval regression against clean `main` `ed806b2`, judge-free LongMemEval
  FTS-only `k=10`, n=500 on the seeded 350 working / 150 held-out split: **0 of
  180 numeric fields differ**. Confirmed independently three times — at 7
  branches, at 8 branches with the reranker wired into `recall()`, and on the
  final tree after the cleanup pass.
- Cleanup pass: 493 lines removed, 223 added, across 13 files (270 net).
- Embedding ingest: 1.98x faster through the shipped `embed_texts()` path,
  vectors identical at min cosine 1.0000000000, max absolute difference
  3.257e-07.
- Test suite wall clock: 377-590s down to 102-165s.
- A hybrid n=500 LongMemEval run is **in progress and not complete**. There is
  no result yet. Do not cite one.

### Closed and disproven claims

Do not schedule work against any row here. Each was checked against current
code, not against a PR title or a prior audit: the original rows at `d1aec40`,
citing symbols rather than lines because line numbers had drifted since the
audits; the late-pass rows at `413f8b9`, citing lines at that revision.

| Claim | Verdict | Closing evidence |
|---|---|---|
| `HAUNT_HOME` shell injection in the generated wrapper | FIXED | `_sh_single_quote` (`src/haunt/bootstrap.py:16`); 5 payloads executed, 0 markers created |
| Planner temporal `k` under-fill | FIXED | `src/haunt/planner.py` paginates until `len(merged) >= limit`; the audit's exact repro now returns 2 hits, not 0 |
| Contradict is non-atomic | FIXED | `Store.contradict` (`src/haunt/store.py:7164`) runs under `BEGIN IMMEDIATE`; re-contradict returns `conflict: already_superseded`, guarded twice |
| Contradict "rewrites `valid_to`" | MISREAD | It still writes `valid_to`; that is the designed supersede (E1 keeps it as a current/as-of projection). What was fixed is atomicity and the re-contradict guard |
| Observe commits before graph extraction | FIXED | `extract_and_store(commit=False)`, single commit, rollback on exception |
| Purge erases older relation evidence | FIXED | `remove_event_evidence` scoped to that event, then `_refresh_relation` re-aggregates |
| Purge deletes unrelated singleton entities | FIXED | scoped to entities mentioned by that event only |
| Malformed editor JSON is overwritten | FIXED | `read_json_object` raises `HostConfigError`; the CLI exits 1 without writing |
| FTS-only bootstrap still requires sqlite-vec | FIXED | `if not vec.get("ok") and not fts_only():` (`src/haunt/bootstrap.py:192`); verified on an interpreter lacking `enable_load_extension` |
| MCP/CLI create typo namespaces | FIXED | `_open_mcp_store(ns, create=False)`; probe returns 404, zero database files created |
| Existing timestamps not migrated to UTC | FIXED | `_normalize_stored_clocks` (`src/haunt/store.py:1414`) runs when `schema_version < 1`; covers sessions, events, memories, entities, relations |
| Second-resolution timestamp ties | FIXED | `timespec="microseconds"` (`src/haunt/util.py:39`); `ORDER BY created_at DESC, rowid DESC` |
| `timeline --limit -1` is unbounded | FIXED | `clamp_limit` (`src/haunt/util.py:139`) floors at `LIMIT_MIN` |
| Negative or uncapped worldview caps | FIXED | `clamp_limit` in CLI and MCP; -1 to 1, 1e9 to 100 |
| `HAUNT_STORE_THOUGHTS` unwired | FIXED | handler registered and env-gated in `src/haunt/cursor_hook.py` |
| Invalid global params mislabeled per-namespace | FIXED | `_validate_recall_request` (`src/haunt/dashboard.py:926`) runs once before the namespace loop |
| Hooks reload the heavyweight model | FIXED | `defer_embedding=True` plus hook recall `use_vectors=False`; no fastembed import on the hook path |
| Inverted entity time bounds (`first_seen > last_seen`) | FIXED | `MIN`/`MAX` in `src/haunt/graph.py` |
| `lore` to `haunt` data/wiring migration missing | VACUOUS | zero `lore` references in `src/` or `docs/`; aliases removed in `8e35095`, never PyPI-released. Only residue is the `MODEL_CACHE` path in `tests/conftest.py:10`, which is R13 |
| Graph dedup race | NOT REACHABLE | `INSERT INTO events` takes the SQLite write lock before the `SELECT`/`INSERT` on the only path that reaches it |
| "10 unmerged codex branches, biggest integration risk" | FALSE | Squash-merge ghosts. `codex/recall-release-gate` is PR #82, merged, `src/` and `tests/` byte-identical to `main`. Same for `namespace-aliases` (#80), `structured-provenance` (#78), `correction-lineage` (#77), `frozen-retrieval-eval` (#74), `eval-trust-controls` (#76), `memory-roadmap` (#75) |
| Dashboard `GET /` leaks the admin token | NOT A DEFECT | `embed_launch_token_in_html()` (`src/haunt/dashboard.py:875`) gates to loopback and non-remote, is explicitly withheld under `--allow-remote`, and is documented. Residual risk only on a shared multi-user host |
| Blob content is marked truncated rather than shrunk (C11) | ALREADY HANDLED | the budget returns `None` (drops the hit); regression test in `tests/test_recall_budget.py`. Also unreachable: `content` is `TEXT NOT NULL` |
| Tiny budget is not hard-asserted (C11) | MISDESCRIBED | hard assertions exist in `tests/test_recall_budget.py` at `cap=2000`; overflow is impossible by construction because the budget measures rather than estimates |
| Drain runs only via `haunt bootstrap` | MISDESCRIBED | three drain sites: `haunt maintenance`, an inline `Store` drain, and bootstrap. The true half is that there is no timer or daemon |
| No index on `embedding_jobs.attempts` | NOT A DEFECT | the hot query is already ordered by `idx_embedding_jobs_queued`; a composite would force a sort, and rows are deleted on success |
| Entity resolution not attempted (C3) | NOT A DEFECT | deliberate scope decision, documented in `src/haunt/store.py` |
| Trigram tokenizer rejected (C9) | WELL-FOUNDED | FTS5 trigram cannot be wrapped by porter; losing stemming on a prose corpus is a net loss |
| C7 phase 2 (reference-not-copy) dropped | WELL-FOUNDED, restated | see the C7 phase 2 decision above — the grounds changed, the decision did not |
| R25 — `FORMAT_MINOR = 0` leaves the older-minor path unexercised | NOT SCHEDULED | true but not work; recorded as an accepted E4 caveat (`src/haunt/portability.py:61`) |
| R26 — `_backfill_content_hashes` runs on every store open, inside R5's unprotected loop | REFUTED | it is called once inside `if current < 10:` (`src/haunt/store.py:1703-1717`) and `_ensure_namespace_schema` early-returns when `current >= SCHEMA_VERSION` (`:1610-1615`). One-shot migration. It is unbounded within that single pass; cheap batching is welcome, heavy machinery is not |
| R27 — `idx_memories_content_hash` costs a write per INSERT to serve one diagnostic | REFUTED twice | `EXPLAIN QUERY PLAN` shows `SEARCH memories USING COVERING INDEX idx_memories_content_hash`, so the index is used. The residual worry that a fresh v11 database never gets the C7/C10 indexes is false: a fresh database stamps at `SCHEMA_VERSION` 11 and receives both `idx_memories_content_hash` (`src/haunt/store.py:1709-1711`) and `idx_memories_current` (`:1723-1726`), 20 `idx_` indexes total |
| R29 — missing `UNIQUE(norm_name, type)` on the entities index | DROPPED | latent invariant gap; the race is unreachable today via `observe` |
| R30 — camelCase identifiers unsearchable by parts | DROPPED | deliberate, tested (`tests/test_identifier_tokenization.py`), and documented. Keep as a known limit |
| Stored XSS via `session_id` in the dashboard | **FIXED, but the class is not closed** | that specific sink is escaped and ~30 `innerHTML` sinks were swept. Other stored values are still raw — see M1 |
| E4 import sets `graph_evidence_version='1'` | CORRECT BY DESIGN | `src/haunt/portability.py:1453-1455`. A restored bundle is correctly not re-derived by R1's backfill. Do not "fix" this |
| R6 — E6 integrity guard diffs the whole tree against a pinned revision (the merge blocker) | FIXED | attribution is now per commit over E6's own evidence paths, plus uncommitted edits to them (`_e6_attributed_diff`, `src/haunt/abstention_eval.py:618-645`). `E0_BASE_REVISION` is gone, so a later merge cannot break it; a shallow clone reports `history_attributable: false` instead of claiming every file; `_git` raises rather than returning a false clean when git is unrunnable (`:597-615`) |
| R2 — embedding model download unpinned, bare `except` redirects to a different repo | FIXED | `BGE_M3_REVISION` pinned (`src/haunt/embed.py:43`); the fallback catches only repo/revision-unavailable hub errors, not timeouts or 5xx (`:166-188`, `:225`), and is opt-in behind `HAUNT_EMBED_QUANT_FALLBACK`, otherwise re-raising (`:234-235`); the cache records which repo produced its bytes (`:191`) |
| R3 — closed sessions accept new events on the explicit-`session_id` path | FIXED | an ended id now routes to `_successor_session` rather than being ignored (`src/haunt/store.py:5077-5080`, `:5111`); `end_session` closes successors alongside the id it was given (`:5145-5180`) |
| R4 — reconcile takes its verified backups before acquiring the write lock | FIXED | the TARGET backup runs inside the merge's own `BEGIN IMMEDIATE` (`src/haunt/store.py:4492-4499`); SOURCE reads one frozen read-only snapshot for both its backup and the copy (`:4489`) |
| R5 — namespace fan-out has no per-namespace isolation | FIXED | `bootstrap` catches `sqlite3.Error`/`OSError`/`NamespacePathError` per namespace and continues (`src/haunt/bootstrap.py:177-183`), so `install_desktop_icon()` and `install_all_hosts()` at `:192-199` are always reached; `reembed_all_namespaces` matches (`src/haunt/store.py:7897-7900`) |
| R10 — the recall budget bounds the sum of the parts, not the serialized list | FIXED | `apply_recall_budget` measures `len(serialize(hits))` and charges the `", "` separator per kept hit (`src/haunt/budget.py:66`, `:70`, `:251`) |
| R11 — `haunt recall --json`, dashboard `/api/recall`, and MCP `memory_timeline` are uncapped | FIXED | one budget on every retrieval surface: `src/haunt/cli.py:368`, `src/haunt/dashboard.py:1093` (fan-out budgeted once, not per namespace), `:1159`, `src/haunt/mcp_server.py:525`, `:567` |
| R13 — the CI fixture defeats `HAUNT_FTS_ONLY` and guards on a dead `MODEL_CACHE` path | FIXED | ambient `HAUNT_FTS_ONLY`/`HAUNT_EMBED_MODEL` now win (`tests/conftest.py:54-73`) and the cache is resolved from `models_dir()` before `HAUNT_HOME` moves (`:12-29`), so no second absolute path can rot. That leaves sqlite-vec and embeddings without coverage, so a `test-hybrid` job runs the suite against them (`.github/workflows/ci.yml:61-86`) — see L11 |
| R14 — C3 source-side digest TOCTOU | FIXED | `_execute_reconciliation_writes` compares the digest of its own in-transaction read against the plan's and refuses a mismatch (`src/haunt/store.py:4406-4415`) |
| R16 — no reconcile concurrency test | FIXED | racing writer added, asserting every TARGET row the merge reads is covered by that merge's backup (`tests/test_namespace_reconcile.py:647-695`); verified failing against the pre-fix functions and passing after |
| R17 — Cursor `Shell` output embeds while Claude Code `Bash` output does not | FIXED | `EMBED_EXCLUDE_TOOLS_DEFAULT = "Bash,Read,Shell"` (`src/haunt/cursor_hook.py:239`) |
| R18 — `HAUNT_STORE_THOUGHTS` rows omit `skip_embedding` | FIXED | `src/haunt/cursor_hook.py:534` |
| R19 — C7/C4 metrics are invisible from the CLI | FIXED | `health_cmd` prints `memories_embedded`, `vector_index`, `duplicate_memories`, and `duplicate_content_values` (`src/haunt/cli.py:755-761`) |
| R20 — no CLI affordance to remove the drained namespace after reconcile | FIXED | `haunt namespace retire LABEL --into TARGET` (`src/haunt/cli.py:692-709`, `retire_namespace` at `src/haunt/store.py:4591`): dry-run by default, refuses while any row is still unique to LABEL, verifies a backup, then drops identity/aliases/bindings and unlinks the database |
| R22 — `rerank.py` ships with zero production callers | D1 EXECUTED | wired into `recall()` between fusion and the returned `k`, still off by default (`src/haunt/recall.py:646-655`, `:744`); default `HAUNT_RERANK_LAMBDA` corrected to `0.3`, and both variables documented (`README.md:322-323`) |
| R28 — triplicated environment parse/clamp idiom | FIXED | `util.env_int`/`util.env_flag` (`src/haunt/util.py:20`, `:31`), used from `budget.py:53`, `cursor_hook.py:265`, `:291`, `embed.py`, and `store.py:918` |
| M1 — E4 import never validates `tier`; the dashboard interpolates it raw | FIXED | `_validate_enumerated_columns` rejects unknown `events.tier`, `memories.tier`, and `entities.type` (`src/haunt/portability.py:1350-1368`); `tierCls` returns `""` off-vocabulary instead of `"t-"+t` (`src/haunt/dashboard.py:384`), and its remaining attribute uses are escaped |
| M2 — no `Content-Security-Policy` header anywhere in the dashboard | FIXED | `default-src 'none'; script-src 'nonce-...'` with a fresh per-response nonce (`src/haunt/dashboard.py:52-56`, `:919-924`). `style-src` stays `'unsafe-inline'`, documented at `:52` |
| M3 — `esc()` is the wrong escaper for inline handler attributes | FIXED | every `onclick="..."` attribute replaced by `data-act` plus one delegated `click` listener (`src/haunt/dashboard.py:806-829`), which is also what lets M2's nonce work; the three surviving `onclick` hits are JS property assignments, not attributes |
| E0's only stemming case was a literal prefix, so the gate could not observe stemming | FIXED | `porter_stemming_nonprefix` added — query "study" against the indexed "studies", neither a prefix nor a substring of the other (`tests/fixtures/retrieval_eval/corpus.json:29`, `tests/test_frozen_retrieval_eval.py:100-103`). Mutation-confirmed load-bearing: under `trigram` it is the one case in the corpus that mismatches baseline. Baseline relocked; `config_sha256` unchanged because the retrieval contract is unchanged |
| E0 scored reranked retrieval against a baseline frozen without reranking | FIXED | `frozen_retrieval_eval` pins `HAUNT_RERANK_ENABLED` out of the isolated environment it already owns (`src/haunt/frozen_retrieval_eval.py:263-265`). Gate produces the frozen result with the flag unset and with it set, so no relock |
| `memory_purge` claimed erasure while plain DELETEs left the bytes readable | FIXED | a canary planted before a purge was still readable 8 times in the raw database, twice in live `memories_fts_data` rows. Purge now runs under `PRAGMA secure_delete=ON` (`src/haunt/store.py:6230`), merges the FTS index (`:6359`), then rebuilds the file and truncates the WAL (`:6502-6506`). All three mutation-verified load-bearing: removing any one puts the canary back. `bytes_overwritten` (`:6475`) reports false rather than claiming a rebuild a concurrent reader blocked. Cost is L12 |

58 closed rows.

### Decisions (do not re-litigate)

- **D1 — reranker.** Wire `haunt.rerank` in behind `HAUNT_RERANK_ENABLED`
  (default off) **and** correct the default `HAUNT_RERANK_LAMBDA` from `0.5` to
  `0.3`. The correction is on measured evidence, confirmed by an independent
  sweep: `lambda` 0.2/0.3 give recall@k `0.9375`, MRR `0.8542`, redundancy
  `0.0417`, against the shipped `0.5` at `0.875`/`0.8542`/`0.0833`; `lambda` 0.1
  also dominates. Caveat: 8 cases over 20 records on a purpose-built fixture.
  Update `README.md:319-320` with the wiring. See R22.
- **D2 — real-namespace reconcile.** Approved to run against the real corpus
  **after** the R4 backup-ordering fix lands, not before.
- **D3 — C7 phase 2.** Dropped. See the C7 phase 2 decision above.
- **D4 — E6 unblock path.** Standing recommendation is option 1 (v2 feature
  family plus v2 dataset). See the E6 blocker review above.
- **D5 — reranker scope.** D1 stands, and the reranker is **not** the fix for
  L1. An MMR-style diversity rerank over BM25 candidates cannot repair
  vocabulary disjunction: it reorders a pool in which gold is already
  out-ranked by a genuinely on-topic distractor. Hybrid retrieval (L2) is the
  fix.
- **D6 — CoreML execution provider.** `CoreMLExecutionProvider` is available,
  but `embed.py` hardcodes `CPUExecutionProvider`
  (`src/haunt/embed.py:218`). Deliberately deferred pending a vector-agreement
  measurement, and to land as a change separate from L9.
- **D7 — ONNX thread tuning.** Not adopted. The `intra=8` 1.32x result came
  from a short-sequence-biased sample and would ship as a hardware-specific
  hardcode.

### Integration state (verified)

- `integration/all-work` `d1aec40` = `main` `ed806b2` + #79 + #81 + #83 +
  `codex/calibrated-abstention`. All four merged with **zero conflicts**;
  confirmed from the merge parents.
- The schema ladder merged sequentially and correctly: v9 `recall_class` (main)
  to v10 `content_hash` plus index plus backfill (C7) to v11
  `idx_memories_current` partial index (C10). No duplicate version numbers;
  `SCHEMA_VERSION = 11` (`src/haunt/store.py:101`).
- `portability`, `rerank`, and `abstention_eval` all co-import.
- `integration/all-work` `413f8b9` adds the E6 guard, store-correctness,
  dashboard-XSS, surface-polish, resilience, embed-batching, E0-gate,
  rerank-wiring, test-environment, and cleanup branches on top of `d1aec40`.
- The full suite on the integrated tree passes: 1157 passed, 1 skipped, 7
  xfailed, 0 failed. The four `tests/test_abstention_feasibility.py` failures
  are gone with R6.
- E4 determinism confirmed: two exports differ only in
  `creation.exported_at` (the contract's permitted volatile field); with a
  pinned `exported_at` they are byte-identical.
- E4 round trip confirmed: memory count preserved, correction lineage
  preserved, recall works after import, superseded value not resurrected. The
  only gap is R7.
- The integrated tree reproduces the working-set LongMemEval retrieval numbers
  exactly, so the 39,941 lines merged over `ed806b2` across 54 files are
  retrieval-neutral. See the LongMemEval register above.
