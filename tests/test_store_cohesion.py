"""store.py may not grow new top-level definitions without a named decision.

An audit of three candidate extraction boundaries --
privacy erasure, reconcile, namespace label administration -- rejected all
three; the blocking coupling for each is recorded in BACKLOG.md under D8. That
decision only stays honest if the file does not quietly accumulate unrelated
work in the meantime.

This pins the set of top-level definition names. Adding one fails until the
name is added below **in the same commit**, which is the moment to say which
cluster it joins:

  connection layer | schema init/migration | registry | namespace label admin
  | reconcile | privacy erasure/purge | Store and its factories

Deliberately NOT a line-count ceiling. Over 06ebd234..10963523 the file grew
950 lines while `class Store` SHRANK by 488: f299353 de-classed the erasure
path into module functions taking an explicit conn, so the identical erasure
could run against a loose backup file. A ceiling would have fired on the
refactor that reduced coupling, and stayed silent on a genuinely unrelated
40-line feature. Names track the thing being gated; lines do not.

Renaming or removing a definition fails this too, which is intended: both are
decisions worth seeing in a diff.

No line count appears above on purpose. An earlier draft quoted one and it went
stale two commits later, inside the same PR -- in the file whose whole argument
is that counts in prose drift while names do not.
"""

from __future__ import annotations

import ast
from pathlib import Path

STORE = Path(__file__).resolve().parents[1] / "src" / "haunt" / "store.py"

# Top-level def/class names in store.py, as of the D8 cohesion decision.
PINNED: frozenset[str] = frozenset({
    "AliasRetirementError",
    "NamespaceCollisionError",
    "NamespaceMigrationError",
    "ObserveResult",
    "ReadOnlyStore",
    "Store",
    "UnknownNamespaceError",
    "_FreshNamespaceClaim",
    "_NamespaceOwnedElsewhere",
    "_SidecarGuardedConnection",
    "_TableDiff",
    "_VerifiedRegistryBackup",
    "_apply_namespace_reconciliation",
    "_backfill_content_hashes",
    "_backfill_skip_embedding",
    "_backup_namespace_database",
    "_backup_registry",
    "_backup_rewrite_refusal",
    "_bind_repository",
    "_canonical_json",
    "_change_namespace_label",
    "_claim_fresh_namespace_db",
    "_claim_fresh_namespace_db_with_configuration_lock",
    "_configure_connection",
    "_connect",
    "_connect_with_configuration_lock",
    "_content_hash",
    "_correction_request_identity",
    "_correction_request_payload",
    "_create_purge_safe_session",
    "_diff_reconcile_table",
    "_embed_drain_limit",
    "_embed_max_attempts",
    "_ensure_correction_append_only_triggers",
    "_ensure_correction_invariant_triggers",
    "_ensure_namespace_schema",
    "_ensure_provenance_type_triggers",
    "_erase_memory_content",
    "_erase_memory_from_backup",
    "_erasure_context_values",
    "_execute_reconciliation_writes",
    "_fetch_rows_by_pk",
    "_finish_plan",
    "_foreign_repository_owner",
    "_fresh_namespace_claim_hook",
    "_held_file_sha256",
    "_held_sqlite_integrity",
    "_identity_row",
    "_init_namespace_schema",
    "_init_registry_once",
    "_insert_dict_row",
    "_json_safe_row",
    "_legacy_namespace_change_source",
    "_list_namespace_rows_readonly_once",
    "_mapped_namespace_open_hook",
    "_merged_window",
    "_namespace_migration_lock",
    "_namespace_state",
    "_normalize_clock_value",
    "_normalize_stored_clocks",
    "_opaque_erasure_values",
    "_open_backup_copy",
    "_open_mapped_namespace_db",
    "_open_mapped_namespace_db_readonly",
    "_open_mapped_namespace_db_unmaintained",
    "_open_mapped_namespace_db_with_configuration_lock",
    "_open_namespace_identity_unmaintained",
    "_open_readonly_connection",
    "_open_zero_write_sqlite_snapshot",
    "_orderable_instant",
    "_plan_namespace_label_read_only",
    "_plan_namespace_reconciliation",
    "_plan_namespace_undo_read_only",
    "_preflight_registry_storage_read_only",
    "_private_backup_root",
    "_provenance_erasure_values",
    "_prune_erased_only_lineage",
    "_publish_namespace_with_configuration_lock",
    "_purge_backup_copies",
    "_purge_safe_session_context",
    "_raw_connect",
    "_read_relative_file",
    "_readonly_registry",
    "_reconcile_content_state_digest",
    "_reconcile_requeue_embedding",
    "_reconcile_sort_key",
    "_reconcile_unsafe_reasons",
    "_record_unerased_backup",
    "_register_namespace_once",
    "_register_namespace_once_with_configuration_lock",
    "_registration_candidates",
    "_registry",
    "_registry_backup_hook",
    "_registry_state",
    "_relative_regular_file",
    "_remove_namespace_database",
    "_repository_context",
    "_residual_erasure_markers",
    "_resolve_namespace_id_once",
    "_resolve_namespace_identity_once",
    "_resolve_window_merges",
    "_resolved_recall_class",
    "_restore_namespace_state",
    "_retire_namespace",
    "_rotate_privacy_lineage_head",
    "_rows_equal",
    "_sanitize_correction_replacement_event",
    "_schema_version",
    "_session_event_extents",
    "_session_meta_json",
    "_sidecar_identities",
    "_sqlite_configuration_lock",
    "_sqlite_configuration_lock_held",
    "_sqlite_sidecar_open_hook",
    "_sqlite_sidecar_pragma_hook",
    "_sqlite_sidecar_verified_hook",
    "_state_digest",
    "_stored_privacy_lineage_head",
    "_sweep_backup_copies",
    "_table_column_signatures",
    "_table_columns",
    "_table_exists",
    "_undo_namespace_migration",
    "_unlink_relative_identity",
    "_validate_all_registered_namespace_dbs_read_only",
    "_validate_selected_readonly_identity",
    "_validate_unmapped_namespace_target",
    "_validated_privacy_lineage_head",
    "_vec_loaded",
    "_verify_backup_erasure",
    "_verify_new_sqlite_fd",
    "_verify_private_backup_root",
    "_window_instant",
    "change_namespace_label",
    "ensure_vec_table",
    "get_store",
    "init_registry",
    "is_concurrent_registry_change",
    "iter_stores",
    "list_namespace_rows",
    "list_namespace_rows_readonly",
    "list_namespaces",
    "namespace_exists",
    "namespace_exists_readonly",
    "namespace_privacy_lineage_head",
    "observe",
    "open_existing",
    "open_existing_readonly",
    "open_namespace_identity",
    "open_namespace_identity_readonly",
    "privacy_lineage_genesis",
    "reconcile_namespaces",
    "reembed_all_namespaces",
    "register_namespace",
    "register_namespace_context",
    "resolve_namespace_id",
    "resolve_namespace_identity",
    "retire_namespace",
    "retire_namespace_alias",
    "touch_namespace",
    "undo_namespace_migration",
    "verbatim_text",
})


def _top_level_names() -> set[str]:
    tree = ast.parse(STORE.read_text(encoding="utf-8"), filename=str(STORE))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_store_top_level_definitions_are_pinned() -> None:
    current = _top_level_names()
    added = sorted(current - PINNED)
    removed = sorted(PINNED - current)
    message = []
    if added:
        message.append(
            "new top-level definitions in store.py: "
            + ", ".join(added)
            + "\n  Name the cluster each one joins, record it in BACKLOG.md D8 if it"
            + "\n  is a new boundary, and add it to PINNED in this same commit."
        )
    if removed:
        message.append(
            "top-level definitions gone from store.py: "
            + ", ".join(removed)
            + "\n  Remove them from PINNED in the commit that removed them."
        )
    assert not message, "\n".join(message)
