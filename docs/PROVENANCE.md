# Source provenance envelope v1

Haunt stores source attribution on the event that produced a memory. The
`provenance` value is structured JSON and is returned as the same object by
Store detail/browse/trace/timeline, CLI observe, MCP observe/trace, and the
dashboard APIs. It describes how bytes entered Haunt; it is not evidence that
the memory is true.

Every new write has `schema_version: 1`, `kind`, and the actual nonempty
`origin` passed to `Store.observe`. Unknown optional values are omitted or null;
Haunt never guesses them.

## Native observations

```json
{
  "schema_version": 1,
  "kind": "native",
  "channel": "cursor_hook",
  "origin": "cursor",
  "producer_tool": "Shell",
  "producer_call_id": "call-123"
}
```

`channel`, `producer_tool`, and `producer_call_id` are optional. Producer
fields are accepted only when they match the tool/call inputs actually supplied
to `Store.observe`; a call ID requires a tool. Cursor and Claude hooks record
their known channel and supplied host call ID automatically.

## Imports

```json
{
  "schema_version": 1,
  "kind": "import",
  "origin": "archive-importer",
  "source_platform": "example-chat",
  "source_native_id": "message-123",
  "source_format": "vendor-json",
  "parser_version": "2.4.1",
  "imported_at": "2026-08-25T12:00:00.000000+00:00",
  "fidelity": "lossless",
  "original_blob_sha256": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "transforms": ["decode:utf-8", "normalize:newlines"]
}
```

Imports require canonicalizable `imported_at`, one of `lossless`, `lossy`,
`reconstructed`, or `derived`, and `original_blob_sha256`. The hash is lowercase
`sha256:<64 hex>` or explicit `null` when no original blob was retained.
Platform/native ID, format/parser version, and transforms are optional because
unknown is not the same as empty or inferred. When present, transform order is
preserved.

Unsupported versions, fields, fidelity values, types, sizes, timestamps, and
hashes fail before any session, event, memory, embedding job, graph, or index
write. An idempotency-key replay must carry the exact same canonical provenance
or it conflicts.

## Legacy and invalid rows

The v8 migration adds the provenance column without rewriting existing
`origin` or `meta` bytes. A pre-v8 null provenance is returned as:

```json
{
  "schema_version": 1,
  "kind": "legacy_unstructured",
  "origin": "original-origin",
  "meta": "original meta bytes"
}
```

Haunt does not synthesize import fields from that legacy data. A structurally
invalid or unsupported non-null stored envelope is labeled `invalid_stored`
instead of being presented as valid or leaking unvalidated fields.
