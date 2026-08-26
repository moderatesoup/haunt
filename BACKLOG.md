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

**Status:** In progress

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

**Implementation evidence (pending independent review)**

- `src/haunt/provenance.py` defines the bounded schema-v1 native/import
  envelope, canonical UTC/hash rules, four fidelity values, actual producer
  matching, actual entry-point channel binding, honest legacy/invalid labels,
  and no truth-confidence field.
- Schema v8 adds event provenance without rewriting old `origin` or `meta`;
  `Store.observe()` validates and canonicalizes it before any session, event,
  derived job, graph, or index write. Idempotency includes exact canonical
  attribution, including concurrent retry behavior, and fails closed when a
  legacy-null or invalid stored envelope cannot prove exact attribution.
- `tests/test_structured_provenance.py` covers every fidelity, Unicode source
  and call IDs, unknown fields, parser/version/hash rejection with zero rows,
  byte-preserving migration/restart, corrected-import trace, actual Store/CLI/
  MCP/dashboard/hook channels, strict UTF-8 input bounds and zero-write
  rejection, honest direct-Python defaults, timeline and procedure provenance
  parity, corrupt stored envelopes, recursive no-confidence assertions, and
  privacy purge canaries across all tables and serialized surfaces.
- The dependency-correct Python 3.14/MCP 2.1 full suite passes with 524 tests,
  5 environment/data skips, and 7 declared temporal xfails. The focused E2,
  E1, E0, hook, host, migration, and FTS compatibility group passes 190 tests
  under Python 3.12/MCP 2.1; that pyenv build cannot load SQLite extensions, so
  its unrelated vector-required tests are not a valid profile.

**Non-goals**

- Automatic truth scoring, source reputation, fact extraction, source blob
  storage policy, or import adapters for every vendor.
- Silent redaction or transformation of verbatim memory content.

## E3 — Add namespace aliases and migration

**Status:** Blocked

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
- Provide a migration command with dry-run and apply modes. Apply is atomic and
  idempotent, records old/new labels and repository identity, and refuses
  alias collisions or a target mapped to another database.
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
  retirement, typo-read, and concurrent-migration tests.
- Retirement tests cover every registry-owned reference class and prove a
  missing/unreadable external host configuration can only produce the caveat,
  not a false registry reference or permanent refusal.
- Security tests bind an ordinary MCP process to namespace A and prove that an
  alias for namespace B cannot be used to read or mutate B.
- Filesystem evidence shows migration does not duplicate or rename the
  namespace database unless a separately confirmed maintenance action says so.

**Non-goals**

- Treating namespaces or aliases as a general authorization/scoping system.
- User/org/team/branch scope hierarchies, cross-namespace federation, or
  automatic alias retirement based only on elapsed time.

## E4 — Ship versioned export/import round trips

**Status:** Blocked

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

**Non-goals**

- Byte-copying SQLite databases, syncing live stores, exporting caches, remote
  backup, encryption/key management, or claiming compatibility with MP's full
  UIIR/ExportBundle.
- Import adapters for third-party products; they may target this envelope later.

## E5 — Explain ranking without calling it confidence

**Status:** In progress

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

**Tests/evidence**

- Golden explanation tests recompute RRF from component ranks and assert exact
  equality with the reported fused score and final ordering.
- Mode tests cover FTS-only, vector-only, hybrid, filtered, empty-token, and tie
  cases. A mutation that labels RRF as confidence must fail a contract test.
- The frozen E0 metrics and hit order remain unchanged except for an explicitly
  reviewed deterministic-tie correction.
- Once E2 lands, integration tests assert its correction/provenance references
  appear in explanations without changing scores or order.

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
