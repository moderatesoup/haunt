# Source provenance envelope v1

Haunt stores source attribution on the event that produced a memory. The
`provenance` value is structured JSON and is returned as the same object by
Store detail/browse/trace/timeline and procedure get/list, CLI observe/timeline/
procedure output, MCP observe/trace/timeline/procedure output, and the dashboard
APIs. It describes how bytes entered Haunt; it is not evidence that the memory
is true.

Every new write has `schema_version: 1`, `kind`, and the actual nonempty
`channel` and `origin` passed to `Store.observe`. Direct Python calls default to
`channel="python"`; the CLI and MCP server bind `cli` and `mcp`; Cursor and
Claude Code hooks bind their host-specific channels. A correction replacement
inherits the channel of the correction entry point. A caller-supplied channel
must match that actual input. Unknown optional values are omitted or null;
Haunt never guesses them.

The direct Python `Store.observe`, top-level `observe`, `procedure_write`, and
correction-replacement defaults all bind both channel and origin to `python`.
Public integrations override both explicitly; a storage default never claims
that a direct Python caller came from the CLI.

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

The stored `channel` is mandatory. `producer_tool` and `producer_call_id` are
optional, but are accepted only when they match the tool/call inputs actually
supplied to `Store.observe`; a call ID requires a tool. Cursor and Claude hooks
record their known channel and supplied host call ID automatically. Actual
origin, channel, tool name, and call ID must be nonempty strings no larger than
2,048 UTF-8 bytes. They are validated before embedding or session work begins.

## Imports

```json
{
  "schema_version": 1,
  "kind": "import",
  "channel": "archive_import",
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
or it conflicts. A retry against a legacy null or invalid stored envelope fails
closed because Haunt cannot prove that the attempted attribution is identical.
The old row remains readable and unchanged.

`haunt timeline --json` returns the same event objects as Store, MCP, and the
dashboard, including valid, `legacy_unstructured`, or `invalid_stored`
provenance. Human timeline rows show `source=<channel>/<origin>`, using
`unknown` rather than inventing a missing legacy or invalid channel. Procedure
get/list results likewise retain the provenance of their sourced memory across
Store, MCP, CLI, dashboard, and worldview output.

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
