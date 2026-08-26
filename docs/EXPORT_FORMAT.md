# Haunt canonical namespace export v1

This document defines `haunt.namespace-export` version 1.0. It is a portable
representation of one namespace's durable memory semantics, not a SQLite
backup and not an MP UIIR/ExportBundle compatibility claim.

Exports contain potentially sensitive verbatim conversations, tool data,
provenance, and correction history. Protect them like the namespace database.
The format provides a corruption digest, not encryption, authentication, or a
signature.

## Container and canonical encoding

- Media type: `application/vnd.haunt.namespace-export+json;version=1`
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

Version 1.0 is the only accepted version today. An unknown major or minor is
rejected before mutation. A future compatible minor must ship an explicit,
tested migration before the importer accepts it.

## Envelope

The root object has exactly these fields:

```json
{
  "format": "haunt.namespace-export",
  "version": {"major": 1, "minor": 0},
  "temporal_cut": "2026-08-26T12:00:00.000000+00:00",
  "namespace": {
    "namespace_id": "stable-id",
    "canonical_label": "project",
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

## Durable record classes

`records` has exactly the arrays below. Every record has exactly the listed
fields, including nullable fields:

- `sessions`: `id`, `started_at`, `ended_at`, `source`, `meta`
- `events`: `id`, `idempotency_key`, `session_id`, `ts`, `event_time`, `role`,
  `content`, `tool_name`, `tool_input`, `tool_output`, `origin`, `tier`, `meta`,
  `provenance`, `recall_class`
- `memories`: `id`, `event_id`, `tier`, `content`, `valid_from`, `valid_to`,
  `created_at`
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
evidence reference must resolve inside the bundle.

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

## Import validation, limits, and atomicity

The importer reads a path in bounded 64 KiB chunks, then uses a token-level
strict parser. V1 is intentionally a bounded in-memory parse: input is never
allowed beyond the resolved byte budget, and token consumption enforces actual
depth, collection items, record count, record bytes, UTF-8 validity, duplicate
keys, and the monotonic deadline before semantic validation. Manifest counts
are checked only after actual usage has already been charged.

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

Any parse, limit, timeout, validation, collision, constraint, or write failure
rolls back and closes scratch/staged resources. The guarantee is zero committed
logical namespace, registry, durable record, graph/FTS, vector, or job changes;
it is not byte-identical SQLite allocation or WAL files.

For an existing destination, the importer opens the selected stable namespace
ID with the preflighted database path/device/inode while holding the migration
lock, then revalidates identity, aliases, remotes, and storage before commit.
A concurrent label retirement/reassignment cannot retarget the write.

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
