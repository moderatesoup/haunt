# Memory adoption backlog

This is the dependency-ordered delivery plan for adopting the useful parts of
Memory Protocol (MP) in Haunt. The normative product decisions are in
[`docs/MEMORY_CONTRACT.md`](docs/MEMORY_CONTRACT.md). An epic is not complete
until its acceptance criteria and named evidence are committed together.

Haunt is not trying to become the MP reference implementation. The program
keeps Haunt local-first, verbatim, one-SQLite-file-per-namespace, and free of a
reader LLM or distillation pipeline (`README.md:3`, `README.md:239-243`).

## Status key

- **Ready**: all dependencies are complete; implementation may start.
- **Blocked**: a listed dependency is incomplete.
- **Done**: every acceptance criterion is met and the evidence is committed.

## Current baseline

| Area | Existing behavior | Adoption gap |
|---|---|---|
| Evaluation | Temporal probes exist, but the optional LongMemEval script skips without an external file and its hit test accepts any non-empty result (`scripts/score_lme_temporal.py:74-154`). | A committed, immutable corpus and non-vacuous relevance/negative-query assertions. |
| Correction | `Store.contradict()` updates `memories.valid_to` and may add an unlinked replacement (`src/haunt/store.py:1502-1565`); current recall hides closed rows (`src/haunt/recall.py:93-123`). | An append-only correction record and traversable old-to-new lineage. |
| Provenance | Events carry session, time, role, tool fields, an `origin` string, and free-form `meta` (`src/haunt/store.py:131-164`, `src/haunt/store.py:569-659`). | A validated, structured source/import envelope and a trace surface. |
| Namespace identity | Repository remotes derive collision-resistant names and legacy registrations are reused (`src/haunt/paths.py:56-117`, `src/haunt/paths.py:148-178`). | Explicit aliases, rename/move migration, collision handling, and retirement rules. |
| Portability | Namespace schema migration is versioned (`src/haunt/store.py:278-321`), but no canonical export/import exists. | A versioned, deterministic, embedding-free round trip. |
| Retrieval | Recall fuses vector and FTS ranks with RRF and retains component ranks internally (`src/haunt/recall.py:31-78`, `src/haunt/recall.py:243-288`). | A stable explanation contract and a calibrated ability to return no answer. |
| Erasure | Purge physically deletes memory, indexes, graph evidence, and orphan events (`src/haunt/store.py:1169-1244`) and is explicitly gated over MCP (`src/haunt/mcp_server.py:417-461`). | Preserve this privacy override while correction becomes append-only. |

## Dependency chain

`E0 -> E1 -> E2 -> E3 -> E4 -> E5 -> E6 -> E7`

The chain is intentionally strict. It prevents schema, migration, portability,
and ranking changes from landing without a frozen behavioral baseline, and it
keeps the final proof representative of the exact shipped sequence.

## E0 — Freeze the retrieval evaluation

**Status:** Ready

**Depends on:** none

**Outcome:** A small, repository-owned evaluation is the fixed comparison point
for every later epic.

**Acceptance criteria**

- Commit a versioned evaluation manifest, deterministic ingest fixtures, query
  cases, expected relevant memory IDs, and expected-unanswerable cases. Each
  file has a SHA-256 recorded in the manifest.
- Give every fixture a fixed memory/event identifier and UTC event, valid, and
  storage time; the evaluator must not read wall-clock time.
- Cover exact lexical lookup, paraphrase, temporal current/as-of behavior,
  superseded content, tool-I/O trust, namespace isolation, and out-of-corpus
  queries. Every category has at least one positive and one negative control.
- Score relevance by expected IDs, not merely by non-empty output. Report at
  least recall@1, recall@5, reciprocal rank, false-positive count on
  unanswerable cases, and the count of evaluated cases.
- Freeze an FTS-only profile that runs in normal CI. Freeze a hybrid profile
  with its embedding model ID and dimension; its baseline artifact may run in
  a dedicated release job, but it may not silently fall back to FTS-only.
- Write baseline results as a committed machine-readable artifact containing
  the manifest hash, Haunt revision, configuration, per-case result, and
  aggregate metrics. Updating a fixture or expected answer requires an
  explicit manifest version bump and a reviewed baseline diff.

**Tests/evidence**

- A test mutates one expected relevant ID and proves the evaluator fails.
- A test removes all returned hits and proves the positive controls fail.
- A test returns a hit for every query and proves the negative controls fail.
- CI publishes the FTS-only report; release evidence publishes the pinned
  hybrid report and verifies the declared model was actually loaded.

**Non-goals**

- Claiming benchmark leadership, vendoring a restricted third-party corpus, or
  using an LLM judge.
- Tuning ranking or abstention in this epic.

## E1 — Add append-only correction lineage and trace

**Status:** Blocked

**Depends on:** E0

**Outcome:** A correction is an immutable record that points to what it
supersedes and, when present, its replacement; any surviving memory can explain
that lineage.

**Acceptance criteria**

- Add an additive schema migration for a correction record with its own stable
  ID, target memory ID, optional replacement memory ID, timestamp, origin,
  session ID, and optional reason. Normal correction code only inserts these
  records; it never edits or deletes a correction record.
- Make correction plus optional replacement atomic. A failed replacement write
  leaves no correction record and leaves the target current, preserving the
  atomicity guarantee already tested around `Store.contradict()`.
- Keep `valid_to` as a current/as-of projection for compatibility, while making
  the correction record the durable source of lineage. Existing pre-migration
  `valid_to` rows remain readable and are honestly labeled `legacy_unlinked`
  when no link can be recovered.
- Prevent forks under normal operation: a current target may receive at most
  one direct correction. Repeating the same request is idempotent; attempting a
  different second correction fails without partial writes.
- Add a trace API used by CLI/MCP/dashboard detail that returns the ordered
  correction chain, source event/session identifiers, and an explicit status
  for a lineage member removed by privacy purge. Trace must never invent the
  erased bytes.
- Preserve historical recall: current recall returns only the chain tip, while
  an `as_of` before correction can return the prior memory.

**Tests/evidence**

- Migration tests open a schema-v3 database, correct a legacy memory, restart,
  and recover the same lineage.
- Positive tests cover correction with and without replacement, a three-link
  chain, idempotent replay, concurrent attempts, and trace after restart.
- Failure-injection tests prove atomic rollback. A mutation test that removes
  the lineage insert must fail the trace assertion.
- Existing temporal and purge suites remain green.

**Non-goals**

- A generic provenance DAG, arbitrary many-parent derivations, merge/unmerge,
  HMAC signing, or tamper-evident replication.
- Removing the explicit privacy purge path.

## E2 — Structure source and import provenance

**Status:** Blocked

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

**Non-goals**

- Automatic truth scoring, source reputation, fact extraction, source blob
  storage policy, or import adapters for every vendor.
- Silent redaction or transformation of verbatim memory content.

## E3 — Add namespace aliases and migration

**Status:** Blocked

**Depends on:** E2

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
- Support a documented alias-retirement check that refuses retirement while a
  registered repository or installed host still resolves through that alias.
- Prove that two clones of the same normalized remote resolve to one canonical
  namespace, while same-leaf repositories with different remotes remain
  distinct.

**Tests/evidence**

- Fresh, upgraded, rename, move, remote URL form, truncation/hash, collision,
  retirement, typo-read, and concurrent-migration tests.
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

**Depends on:** E3

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
- After import into a fresh home, event/memory IDs, verbatim content, current
  and as-of recall membership, trace lineage, timestamps, trust labels,
  provenance, and namespace resolution match the source. Re-export has the same
  canonical semantic digest.

**Tests/evidence**

- Golden-bundle schema tests and a source -> export -> fresh import -> export
  comparison in FTS-only mode.
- A pinned hybrid test proves excluded embeddings are regenerated and recall
  works without requiring source-model compatibility.
- Failure tests corrupt each record class and prove the destination remains
  byte-for-byte unchanged. A purge test proves erased content is absent from
  the bundle and cannot reappear after import.

**Non-goals**

- Byte-copying SQLite databases, syncing live stores, exporting caches, remote
  backup, encryption/key management, or claiming compatibility with MP's full
  UIIR/ExportBundle.
- Import adapters for third-party products; they may target this envelope later.

## E5 — Explain ranking without calling it confidence

**Status:** Blocked

**Depends on:** E4

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
- Include trust labels and correction/provenance identifiers by reference,
  without treating trusted text as authorization to mutate tools.
- Expose the same explanation semantics through Python, CLI JSON, MCP, and
  dashboard APIs; human rendering may be implementation-specific.

**Tests/evidence**

- Golden explanation tests recompute RRF from component ranks and assert exact
  equality with the reported fused score and final ordering.
- Mode tests cover FTS-only, vector-only, hybrid, filtered, empty-token, and tie
  cases. A mutation that labels RRF as confidence must fail a contract test.
- The frozen E0 metrics and hit order remain unchanged except for an explicitly
  reviewed deterministic-tie correction.

**Non-goals**

- Natural-language explanations from an LLM, causal claims about why content is
  true, or exposure of embeddings themselves.
- Calibrated abstention; that is E6 and uses separate evidence.

## E6 — Add evaluation-calibrated abstention

**Status:** Blocked

**Depends on:** E5 (and the unchanged E0 corpus)

**Outcome:** Recall can return an honest no-answer result when retrieval evidence
does not support the query.

**Acceptance criteria**

- Define an evidence-strength feature set from raw retrieval signals and query
  coverage; do not threshold the RRF rank score as though it were confidence.
- Calibrate and version separate policies for the frozen FTS-only and pinned
  hybrid profiles. Each artifact records manifest hash, feature definition,
  model/configuration ID, threshold, fitting cases, held-out cases, and metrics.
- Select thresholds using only the fitting split. On the frozen held-out split,
  abstain on 100% of designated unanswerable cases while retaining at least 95%
  of answerable cases whose relevant memory ranked in the pre-abstention top 5.
  If the corpus is too small to support those gates, expand and re-freeze E0 in
  a separately reviewed change before tuning.
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

- Committed calibration and held-out reports, with a script that reproduces
  their metrics from the frozen manifest.
- Boundary tests immediately above, at, and below every threshold; profile
  mismatch tests; and out-of-corpus adversarial queries.
- A leakage test proves held-out labels are not read while fitting. A mutation
  that uses `rrf_score` as the threshold must fail.

**Non-goals**

- Fact confidence, truth verification, source reputation, a reader LLM, or a
  universal threshold shared across retrieval modes/models.
- Guaranteeing that retained results are correct; abstention controls evidence
  sufficiency, not truth.

## E7 — Produce final end-to-end and release proof

**Status:** Blocked

**Depends on:** E6

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
- Publish the frozen-eval before/after comparison. Any regression or threshold
  gate miss blocks release unless the frozen manifest and contract are changed
  in an explicit reviewed decision.
- Update README, CLI/MCP help, dashboard copy, schema version notes, changelog,
  privacy/export warnings, and recovery guidance to match shipped behavior.
- Audit the final diff and public surfaces against
  `docs/MEMORY_CONTRACT.md`: no generic DAG, HMAC log, scope hierarchy,
  distillation, fact confidence, or embedding export has entered the product.

**Tests/evidence**

- CI logs and machine-readable end-to-end reports name the commit, Python
  version, platform, retrieval profile, eval manifest hash, export format
  version, schema version, and calibration ID.
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
