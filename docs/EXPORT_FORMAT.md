# Haunt canonical namespace export v1

This document defines `haunt.namespace-export` version 1.1, and the version 1.0
bundles it still accepts. It is a portable representation of one namespace's
durable memory semantics, not a SQLite backup and not an MP UIIR/ExportBundle
compatibility claim.

Exports contain potentially sensitive verbatim conversations, tool data,
provenance, and correction history. Protect them like the namespace database.
The format provides a corruption digest, not encryption, authentication, or a
signature.

## Container and canonical encoding

- Media type: `application/vnd.haunt.namespace-export+json;version=1`. The
  media type is major-scoped and does not move with a compatible minor: a
  minor adds fields an older reader refuses outright, not a different
  container to negotiate.
- Encoding: strict UTF-8 JSON. Duplicate object keys, invalid Unicode,
  non-finite JSON numbers, trailing bytes, and compressed input are rejected.
- Canonical bytes: object keys are sorted lexicographically, arrays retain
  their specified order, non-ASCII strings are emitted directly, and no
  insignificant whitespace is emitted.
- Record arrays are sorted by the canonical JSON encoding of their primary-key
  tuple. Namespace aliases put the one canonical label first, then sort by
  normalized label. Repository identities are unique and sorted.
- `manifest.semantic_digest` is `sha256:` plus the lowercase SHA-256 of the
  canonical encoding of exactly `format`, `version`, `temporal_cut`,
  `namespace`, and `records`.
- `creation.exported_at` is the only declared volatile field and is outside the
  semantic digest. Changing it changes the file bytes but not the represented
  namespace state.

## Versioning and the 1.0 compatibility rule

Exports are written at 1.1. Both 1.0 and 1.1 are accepted; an unknown major, an
unknown newer minor, and a negative minor are all rejected before mutation. A
future compatible minor must ship an explicit, tested migration before the
importer accepts it, because its added fields would carry meaning this reader
has no defined default for.

A minor adds record fields and never changes how a field an earlier minor
already emitted is read. A bundle is validated against the exact field set of
the minor it declares: a 1.0 bundle carrying a 1.1 field is rejected, and so is
a 1.1 bundle missing one. The declared version is inside the semantic digest,
so a 1.0 bundle keeps the digest its exporter published and its import receipt
stays replayable; re-exporting it produces a 1.1 bundle with a new digest.

A 1.0 default is what that bundle's silence means, not a value it may impose on
a destination. Importing a 1.0 bundle onto rows that already carry the added
fields is therefore an ordinary record conflict, refused before any write, not
a silent un-exclusion or a dropped successor link.

Both mixed-minor orders are refused, for the same reason and with the same
`ImportConflictError`: the two bundles disagree about a durable field of a row
whose identity they share, and the importer never resolves such a disagreement
by preferring one side. Importing 1.1 then 1.0 is refused, and so is importing
1.0 then 1.1. That is correct and fail-closed, but it has a consequence worth
stating plainly: **a destination already imported from a 1.0 bundle cannot be
repaired in place by re-importing the same namespace at 1.1.** The re-import is
refused on the first conflicting record and the destination keeps its 1.0
defaults. The refusal only fires where the bundles actually disagree: a 1.1
bundle whose added fields all hold their 1.0 defaults is accepted and inserts
nothing, because every record matches. It is a bundle carrying real exclusions
or real successor links — precisely the one worth re-importing — that is turned
away.

Remediation for a destination in that state is to import the 1.1 bundle into a
home that does not already carry the namespace and use that copy going forward;
there is no in-place upgrade path through the importer. Restoring the flags by
hand is not sufficient on its own either, because the 1.0 import already queued
the re-admitted rows and any drain since then has put them in the vector index;
a destination repaired that way needs `reembed()` afterwards, which drops and
rebuilds `vec_memories` from the `skip_embedding=0` rows only.

1.1 adds exactly two fields, both durable store columns that 1.0 dropped:

| Field | Added | 1.0 default | Why that default is safe |
|---|---|---|---|
| `sessions.succeeds_session` | schema v12 | `NULL` | A 1.0 bundle records no succession anywhere, so `NULL` — the column's own value for a session that continues nothing — is the only non-fabricating choice, and it is what a 1.0 bundle already imported as. A 1.0 bundle exported from a pre-v12 namespace may carry a stale `succeeds_session` key inside `sessions.meta`; it is imported verbatim as opaque caller metadata and deliberately not promoted to the column, because rewriting `meta` would make the stored row differ from the bundle record and break receipt replay, and leaving both would give privacy purge a second carrier to rekey. |
| `memories.skip_embedding` | schema v13 | `0` | The capture-policy decision is genuinely absent from a 1.0 bundle, not merely unread: the store infers it from "no vector, no queue row, non-blank content", and a bundle carries neither vectors nor queue rows by construction. `0` is what a 1.0 bundle already imported as, so old bundles keep meaning exactly what they meant. The harm is asymmetric — defaulting to `1` would silently drop every memory in every legacy bundle out of the vector index and out of `reembed()`, with no error and no way to tell which rows were genuinely excluded, while `0` re-admits rows the source had excluded. That cost is not just embedding work: an excluded row's content is exactly what the capture policy withheld from the vector index, and a 1.0 import queues it and lets a later `reembed()` index it, so the withheld content becomes semantically recallable at the destination — see [What a 1.0 bundle costs](#what-a-10-bundle-costs). `0` is still the right choice, because the two errors are not equally detectable: a wrongly re-admitted row announces itself in the index and in `skip_embedding`, while a wrongly excluded one is indistinguishable from a correctly excluded one and would be found only by missing it. Absence of a recorded exclusion is not evidence of one. |

## Envelope

The root object has exactly these fields:

```json
{
  "format": "haunt.namespace-export",
  "version": {"major": 1, "minor": 1},
  "temporal_cut": "2026-08-26T12:00:00.000000+00:00",
  "namespace": {
    "namespace_id": "stable-id",
    "canonical_label": "project",
    "privacy_lineage_head": "sha256:...",
    "aliases": [
      {"label": "project", "is_canonical": true, "source_alias_norm": null}
    ],
    "repository_identities": ["github.com/owner/project"]
  },
  "records": {},
  "creation": {
    "exported_at": "2026-08-26T12:00:01.000000+00:00",
    "media_type": "application/vnd.haunt.namespace-export+json;version=1",
    "volatile_fields": ["creation.exported_at"]
  },
  "manifest": {
    "semantic_digest": "sha256:...",
    "record_counts": {},
    "total_records": 0
  }
}
```

The namespace identity includes no database path, local repository path,
device/inode identity, registry migration history, or backup path. Alias source
references are normalized labels within the same included alias set; missing,
self-referential, cyclic, or canonical-label dependencies are invalid.

`privacy_lineage_head` is an opaque, privacy-safe history head. A legacy
namespace with no stored head has a deterministic genesis derived only from its
already-public stable namespace ID. Every successful hard purge rotates the
head with cryptographic randomness in the same SQLite transaction as erasure;
the value retains no erased ID, ID hash, content, or provenance. Existing
imports require an exact head match before considering records or receipts, so
a pre-purge or independently purged fork cannot restore erased rows. A fresh
home preserves the bundle head exactly.

## Durable record classes

`records` has exactly the arrays below. Every record has exactly the listed
fields, including nullable fields:

- `sessions`: `id`, `started_at`, `ended_at`, `source`, `meta`,
  `succeeds_session` (1.1)
- `events`: `id`, `idempotency_key`, `session_id`, `ts`, `event_time`, `role`,
  `content`, `tool_name`, `tool_input`, `tool_output`, `origin`, `tier`, `meta`,
  `provenance`, `recall_class`
- `memories`: `id`, `event_id`, `tier`, `content`, `valid_from`, `valid_to`,
  `created_at`, `skip_embedding` (1.1)
- `lineage_tombstones`: `schema_version`, `tombstone_id`, `status`, `erased_at`
- `corrections`: `id`, `target_memory_id`, `target_tombstone_id`,
  `replacement_memory_id`, `replacement_tombstone_id`, `corrected_at`,
  `origin`, `session_id`, `reason`, `idempotency_key`, `request_identity`,
  `request_payload`, `response_json`
- `entities`: `id`, `name`, `type`, `norm_name`, `first_seen`, `last_seen`
- `entity_mentions`: `event_id`, `entity_id`, `observed_at`
- `relation_evidence`: `event_id`, `src_entity`, `rel`, `dst_entity`,
  `observed_at`, `weight`

Structured event provenance remains the canonical stored v1 JSON TEXT defined
in [PROVENANCE.md](PROVENANCE.md). Referential closure is mandatory: every
event/session, memory/event, correction endpoint, entity mention, and relation
evidence reference must resolve inside the bundle. A non-null
`sessions.succeeds_session` must name another session in the same bundle, and
succession must be acyclic: a row may not succeed itself, and no chain of
successor links may return to a session it already passed through. The column
carries no SQLite foreign key, so import is the only gate on a dangling or
looping successor link, and the store itself cannot mint a loop — it only ever
points a freshly minted session at an existing one.
`memories.skip_embedding` must be exactly the integer `0` or `1` — the column
has no CHECK constraint, and
every reader treats it as a two-valued flag, so any other value would read as
excluded.

SQLite BLOBs and legacy non-finite REAL values cannot be represented as normal
JSON scalars. V1 preserves them exactly with these closed tagged objects:

```json
{"$haunt_sqlite":"blob","base64":"AAE="}
{"$haunt_sqlite":"real","bits":"inf"}
```

Finite REAL values use JSON numbers. Unknown tags and finite values in the
REAL tag are rejected. Legacy BLOB memory content remains a BLOB after import;
it is not converted with `str()`, indexed as FTS text, or queued for embedding.
Scratch validation reads every inserted field back and compares its exact
SQLite type and canonical value, so TEXT affinity cannot silently coerce a
JSON boolean/number and out-of-range integers cannot reach publication.

## Temporal cut

`--cut` selects state at an explicit UTC instant. Events are selected by
storage time, memories by creation time, corrections by correction time, and
graph mentions/evidence by observation time. Sessions and `valid_to` values
written after the cut are projected back to their at-cut open state.
Correction and replacement are treated as one logical commit: a cut cannot
expose a future replacement as an unrelated current memory or retain dangling
lineage.

Entity `first_seen`/`last_seen` are projected from only the retained mentions
and relation evidence. A later graph observation therefore cannot leak a
future `last_seen` into a historical bundle.

Without `--cut`, export uses a stable high-water mark over included durable
write/audit clocks. An empty namespace uses
`1970-01-01T00:00:00.000000+00:00`. This makes repeated default exports and a
default export followed by fresh import/re-export retain the same semantic
digest. Re-exporting a caller-selected cut requires selecting that same cut.

Previously purged bytes are never reconstructed. Only surviving opaque
four-field lineage tombstones needed by a surviving correction chain may
appear.

Export holds the namespace-migration coordination lock while it snapshots the
stable ID, aliases, remote identities, and database mapping. The namespace read
uses one explicit transaction over a guarded zero-write SQLite snapshot. The
guard compares durable primary/WAL state when the attempt closes; a concurrent
observe, correction, or purge invalidates and retries the whole attempt up to a
fixed bound. It never publishes a cross-statement mix, and a purge that commits
during an attempt cannot yield a successful bundle containing that canary.

## Excluded and rebuilt state

V1 excludes memory embeddings, vector tables, FTS tables, embedding jobs,
materialized graph relations, SQLite WAL/SHM/journal state, local paths, file
identity, and other caches. Import rebuilds FTS for TEXT memories and graph
relations from durable evidence. It queues non-empty TEXT memories for the
destination's configured embedding model; source model identity, dimension,
and vectors do not transfer.

A memory with `skip_embedding=1` is still indexed for FTS and is not queued,
exactly as the store admits one: the persisted exclusion withholds the vector
index only, and it survives a later full `reembed()` at the destination rather
than being re-derived from the destination's own environment.

### What a 1.0 bundle costs

A 1.0 bundle has no `skip_embedding` field, so every memory it carries imports
at `0` — including the rows whose capture policy excluded them at the source.
The cost of that is larger than "embedding work", and it is worth naming
exactly, because the rows in question are the ones the policy singled out. A
tool result withheld from the vector index because it contained a secret
imports as an ordinary memory, is queued like any other, and after the next
drain or `reembed()` is in the destination's vector index and reachable by
semantic recall. The exclusion is not merely unrecorded at the destination; it
is undone there. Nothing about this is silent in the sense of being invisible
after the fact — the row's `skip_embedding=0` and its presence in the index
both say so — but nothing warns at import time either.

This is pre-existing behavior, not a regression introduced by 1.1. The
pre-1.1 importer lands the same bundle the same way — the same row at
`skip_embedding=0` and queued — because the field it would have needed was
never on the wire to read. 1.1 does not repair an old bundle; it makes the flag
transferable from here on, so a *new* export carries the exclusion and a 1.0
bundle is the only kind still carrying this loss.

It was also not fixable at import time. The store derives the flag for existing
rows from "no vector, no queue row, non-blank content" — an inference over
exactly the two things a bundle excludes by construction. A bundle carries
neither vectors nor queue rows, so the signal the store backfills from is not
merely missing from the wire, it is unavailable in principle to any reader of
this format. Guessing `1` instead would be worse in the other direction (see
the table above), and there is no third value meaning "unknown" to fall back
to: every reader of the column tests it as a two-valued flag, so any other
value would read as excluded.

The remedy is to re-export the source at 1.1 and import that into a destination
that has not already taken the 1.0 bundle. A destination that has taken one
cannot be repaired by re-import, as described under
[the 1.0 compatibility rule](#versioning-and-the-10-compatibility-rule).

`memories.content_hash` is likewise rebuilt rather than transferred. It is a
pure function of the stored `content` the bundle already carries, so import
writes exactly the store's own `_content_hash(content)` for every TEXT memory,
at every accepted version — a 1.0 bundle gains it too. Carrying it on the wire
would only add a second value that validation would have to reconcile against
this one, and a bundle whose hash disagreed with its content would be
meaningless. Non-TEXT (legacy BLOB) content has no defined hash and stays NULL.

## Import validation, limits, and atomicity

The importer reads a path in bounded 64 KiB chunks, then uses a token-level
strict parser. V1 is intentionally a bounded in-memory parse: input is never
allowed beyond the resolved byte budget, and token consumption enforces actual
depth, collection items, record count, record bytes, UTF-8 validity, duplicate
keys, and the monotonic deadline before semantic validation. Manifest counts
are checked only after actual usage has already been charged.
Every table count and `total_records` is nevertheless an exact bounded
nonnegative JSON integer: booleans, floats, strings, negatives, missing/extra
table keys, a wrong sum, and values above the resolved record budget are
rejected.

| Budget | Default | Maximum clamp |
|---|---:|---:|
| input bytes | 64 MiB | 256 MiB |
| decompressed bytes | 64 MiB | 256 MiB |
| total records | 100,000 | 1,000,000 |
| bytes per record | 1 MiB | 8 MiB |
| JSON depth | 32 | 64 |
| collection items | 10,000 | 100,000 |
| timeout | 30 seconds | 300 seconds |

Every limit must be finite and positive. V1 has no compression, so input and
decompressed byte charges are identical; gzip/ZIP/bzip signatures are rejected.

Validation first proves the complete envelope, digest, identity, provenance,
ordering, and references in a scratch database. A fresh namespace is populated
privately and published with its stable namespace identity only after all work
succeeds. An existing namespace must have the same namespace ID, canonical
label, aliases, and remote repository identities. Existing record IDs must be
byte-semantically identical. A receipt permits an exact replay to write
nothing, but is never trusted without rechecking every durable record; deleted
or changed rows cause a conflict.

Fresh publication is restart-recoverable. Before staging can become visible,
Haunt fsyncs a private mode-0600 intent that binds an unpredictable token and
the bundle digest/head to the exact namespace ID, target name, and claimed
primary/sidecar device+inode identities. Recovery runs under the namespace
migration and SQLite configuration locks. An uncommitted intent removes only
names that still match those recorded identities; a committed intent must also
match the registry identity and staged receipt before cleanup. Replaced,
unrelated, symlinked, or unexpected-hardlink files fail closed and are never
deleted. A crash before, during, or immediately after registry publication can
therefore be retried without an exposed unmapped database or leftover staged
WAL/SHM/journal files.

Any parse, limit, timeout, validation, collision, constraint, or write failure
rolls back and closes scratch/staged resources. The guarantee is zero committed
logical namespace, registry, durable record, graph/FTS, vector, or job changes;
it is not byte-identical SQLite allocation or WAL files.

For an existing destination, the importer first opens the selected stable
namespace ID through the guarded zero-write reader and checks current schema,
privacy head, receipt, and every durable row. A rejected conflict therefore
cannot run schema migration, graph repair, or writer configuration. Only a
clean preflight opens an exact path/device/inode guarded writer that deliberately
skips maintenance, then repeats the checks inside `BEGIN IMMEDIATE` and
revalidates identity, aliases, remotes, and storage before commit. A concurrent
label retirement/reassignment cannot retarget the write.

## Public surfaces

- Python: `build_namespace_export`, `canonical_export_bytes`,
  `export_namespace_path`, `import_namespace_bytes`, and
  `import_namespace_path` in `haunt.portability`
- CLI: `haunt export OUTPUT [-n NAMESPACE] [--cut ISO]` and
  `haunt import BUNDLE [limit options]`
- MCP: admin-only `memory_export_bundle` and `memory_import_bundle`; both
  require a process launched with `HAUNT_MCP_ADMIN=1`
- Dashboard: authenticated `GET /api/namespace/{name}/export` and
  `POST /api/import`. Import additionally requires a trusted `Origin`, an
  accepted JSON media type, and the launch token already required by every API
  route. Transfer is deliberately API-only in v1; the HTML console does not
  add an accidental paste/import control for sensitive bundles.

The dashboard token and MCP admin flag grant administrative transfer ability;
they are not per-namespace authorization. Export never creates an unknown
namespace. Import may create the exact bundle namespace after validation.
