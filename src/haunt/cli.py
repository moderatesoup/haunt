"""haunt CLI — human-readable stdout, JSON diagnostics on stderr."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import NoReturn, Optional

import typer

from haunt import __version__
from haunt.bootstrap import bootstrap, format_report
from haunt.budget import apply_recall_budget
from haunt.embed import state as embed_state
from haunt.paths import (
    NamespacePathError,
    haunt_home,
    infer_namespace_context,
    resolve_namespace,
)
from haunt.planner import planned_recall
from haunt.portability import (
    ExportError,
    ImportBundleError,
    export_namespace_path,
    import_namespace_path,
    resolve_import_limits,
)
from haunt.recall import BACKEND_ERROR_CODE, execution_metadata, is_retrieval_backend_error
from haunt.store import (
    AliasRetirementError,
    NamespaceCollisionError,
    NamespaceMigrationError,
    Store,
    UnknownNamespaceError,
    change_namespace_label,
    list_namespaces,
    namespace_exists_readonly,
    open_existing,
    open_existing_readonly,
    reconcile_namespaces,
    register_namespace_context,
    retire_namespace,
    retire_namespace_alias,
    undo_namespace_migration,
)
from haunt.temporal import TemporalParseError
from haunt.util import (
    clamp_limit,
    dumps,
    env_flag,
    format_iso,
    human_display,
    snippet,
)

app = typer.Typer(
    add_completion=False,
    help="haunt — local-first verbatim memory for AI agents",
)


def _ns(name: Optional[str]) -> str:
    return resolve_namespace(name)


def _die(exc: BaseException, *, code: int = 2) -> NoReturn:
    """`error: <exc>` on stderr, then exit. 2 = bad request, 1 = failed action."""
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code) from exc


def _existing(ns: str) -> Store:
    """Open an existing namespace or exit 2. Never creates a DB."""
    try:
        return open_existing(ns)
    except UnknownNamespaceError as exc:
        _die(exc)


def _existing_readonly(ns: str):
    """Open recall's stable alias target without writer maintenance."""
    try:
        return open_existing_readonly(ns)
    except UnknownNamespaceError as exc:
        _die(exc)


def _recall_json_error(
    exc: Exception, *, namespace: str | None, query: str
) -> NoReturn:
    """Keep --json machine-readable even when recall rejects its input."""
    typer.echo(
        dumps(
            {
                "ok": False,
                "code": (
                    BACKEND_ERROR_CODE
                    if is_retrieval_backend_error(exc)
                    else "invalid_recall_request"
                ),
                "error": str(exc),
                "namespace": namespace,
                "query": query,
            }
        )
    )
    raise typer.Exit(2)


@app.command("export")
def export_cmd(
    output: Path = typer.Argument(..., help="New canonical JSON bundle path"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    cut: Optional[str] = typer.Option(
        None,
        "--cut",
        help="Explicit UTC temporal cut (default: stable durable high-water mark)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output stable JSON report"),
) -> None:
    """Export durable namespace semantics without embeddings or local paths."""
    ns = _ns(namespace)
    typer.echo(
        "warning: export contains potentially sensitive verbatim namespace data",
        err=True,
    )
    try:
        report = export_namespace_path(ns, output, cut=cut)
    except (ExportError, NamespacePathError, OSError, ValueError) as exc:
        _die(exc)
    if json_out:
        typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    typer.echo(f"namespace       {report['namespace']}")
    typer.echo(f"namespace id    {report['namespace_id']}")
    typer.echo(f"temporal cut    {report['temporal_cut']}")
    typer.echo(f"semantic digest {report['semantic_digest']}")
    typer.echo(f"bytes            {report['bytes']}")
    typer.echo(f"output           {report['path']}")


@app.command("import")
def import_cmd(
    bundle: Path = typer.Argument(..., help="Canonical Haunt namespace bundle"),
    timeout: float = typer.Option(30.0, "--timeout", help="Finite import timeout seconds"),
    input_bytes: Optional[int] = typer.Option(None, "--input-bytes"),
    decompressed_bytes: Optional[int] = typer.Option(None, "--decompressed-bytes"),
    records: Optional[int] = typer.Option(None, "--records"),
    record_bytes: Optional[int] = typer.Option(None, "--record-bytes"),
    json_depth: Optional[int] = typer.Option(None, "--json-depth"),
    collection_items: Optional[int] = typer.Option(None, "--collection-items"),
    json_out: bool = typer.Option(False, "--json", help="Output stable JSON report"),
) -> None:
    """Validate fully, then transactionally import one canonical namespace."""
    try:
        limits = resolve_import_limits(
            input_bytes=input_bytes,
            decompressed_bytes=decompressed_bytes,
            records=records,
            record_bytes=record_bytes,
            json_depth=json_depth,
            collection_items=collection_items,
        )
        report = import_namespace_path(
            bundle, limits=limits, timeout_seconds=timeout
        )
    except (ImportBundleError, NamespacePathError, OSError, ValueError) as exc:
        _die(exc)
    if json_out:
        typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    typer.echo(f"namespace       {report['namespace']}")
    typer.echo(f"namespace id    {report['namespace_id']}")
    typer.echo(f"semantic digest {report['semantic_digest']}")
    typer.echo(f"created          {report['created_namespace']}")
    typer.echo(f"deduplicated     {report['deduplicated']}")
    typer.echo(f"inserted         {sum(report['inserted'].values())}")
    typer.echo(
        "limits           "
        + " ".join(f"{key}={value}" for key, value in report["limits"].items())
    )


@app.command("bootstrap")
def bootstrap_cmd(
    reembed: bool = typer.Option(
        False,
        "--reembed",
        help="Rebuild embeddings in every namespace for the loaded model (required after a dim change).",
    ),
) -> None:
    """Create ~/.haunt, probe sqlite-vec (unless FTS-only), download the embed model, init default."""
    from haunt.bootstrap import BootstrapError

    try:
        report = bootstrap(reembed=reembed)
    except BootstrapError as exc:
        typer.echo(exc.message, err=True)
        raise typer.Exit(1)
    typer.echo(format_report(report))


@app.command("init")
def init_cmd(
    name: Optional[str] = typer.Argument(
        None, help="Namespace name (default: inferred)"
    ),
    repo: Optional[Path] = typer.Option(
        None, "--repo", help="Repo path recorded in the registry"
    ),
) -> None:
    """Create a namespace (one SQLite file)."""
    if name:
        # An explicit name is a deliberate override, not an inference, so
        # (like HAUNT_NAMESPACE) it never auto-binds a repository unless
        # --repo also says so explicitly.
        ns = name
        repo_path = str(repo) if repo else None
    else:
        ns, inferred_repo_path = infer_namespace_context(repo)
        repo_path = str(repo) if repo else inferred_repo_path
    try:
        # Registration reports the label it published: an inferred name can
        # fork when another repository already owns it. A name the user typed
        # cannot -- `haunt init team --repo A` then `--repo B` is a request to
        # share `team`, and typing a name must produce that name. Without this
        # flag `--repo` would silently reroute the second one to
        # `team-<digest>`, because it is the only path that hands registration
        # a chosen label together with a repository to assert ownership with.
        ns, db = register_namespace_context(
            ns, repo_path=repo_path, explicit_label=name is not None
        )
        with Store(ns) as st:
            stats = st.stats()
    except (NamespaceCollisionError, NamespacePathError) as exc:
        _die(exc)
    typer.echo(f"namespace  {ns}")
    typer.echo(f"db         {db}")
    typer.echo(f"events     {stats['events']}")


@app.command("observe")
def observe_cmd(
    text: str = typer.Argument(
        "", help="Verbatim content (empty ok for tool-only events)"
    ),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    tier: str = typer.Option("episodic", "--tier"),
    session: Optional[str] = typer.Option(None, "--session"),
    role: str = typer.Option("user", "--role"),
    tool_name: Optional[str] = typer.Option(None, "--tool-name"),
    tool_input: Optional[str] = typer.Option(None, "--tool-input"),
    tool_output: Optional[str] = typer.Option(None, "--tool-output"),
    producer_call_id: Optional[str] = typer.Option(None, "--producer-call-id"),
    event_time: Optional[str] = typer.Option(None, "--event-time"),
    idempotency_key: Optional[str] = typer.Option(None, "--idempotency-key"),
    origin: str = typer.Option("cli", "--origin"),
    provenance_json: Optional[str] = typer.Option(
        None,
        "--provenance-json",
        help="Versioned source provenance envelope as JSON",
    ),
    recall_class: Optional[str] = typer.Option(
        None,
        "--recall-class",
        help="Residue class: tool | task. Raw tool fields always stamp tool.",
    ),
) -> None:
    """Store a chat turn or tool call as-is. No summarization."""
    ns = _ns(namespace)
    try:
        provenance = (
            json.loads(provenance_json) if provenance_json is not None else None
        )
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("provenance must be a JSON object")
        with Store(ns) as st:
            result = st.observe(
                text,
                role=role,
                tier=tier,
                session_id=session,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                producer_call_id=producer_call_id,
                event_time=event_time,
                idempotency_key=idempotency_key,
                origin=origin,
                channel="cli",
                provenance=provenance,
                recall_class=recall_class,
            )
    except json.JSONDecodeError as exc:
        typer.echo(f"error: invalid provenance JSON: {exc}", err=True)
        raise typer.Exit(2) from exc
    except ValueError as exc:
        _die(exc)
    ents = ",".join(result.entities[:8]) if result.entities else "-"
    typer.echo(
        f"ok  event={result.event_id}  memory={result.memory_id}  "
        f"ns={result.namespace}  tier={result.tier}  "
        f"session={result.session_id}  embedded={int(result.embedded)}  "
        f"recall_class={result.recall_class or '-'}  entities={ents}"
    )
    typer.echo(f"provenance {dumps(result.provenance)}")


@app.command("recall")
def recall_cmd(
    query: str = typer.Argument(..., help="Search query (verbatim or paraphrase)"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    as_of: Optional[str] = typer.Option(None, "--as-of"),
    since: Optional[str] = typer.Option(None, "--since"),
    until: Optional[str] = typer.Option(None, "--until"),
    clock: Optional[str] = typer.Option(
        None,
        "--clock",
        help="event_time | storage_time (default event_time). "
        "storage_time is ingest time (events.ts), not source time. "
        "write_time is a deprecated alias for storage_time.",
    ),
    tier: Optional[str] = typer.Option(None, "--tier"),
    k: int = typer.Option(8, "--k"),
    include_residue: bool = typer.Option(
        False,
        "--include-residue",
        help="Include raw tool and explicitly classified task/tool residue.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit serialized recall hits, including additive ranking explanations.",
    ),
) -> None:
    """Recall memories. Ranked hits show an RRF signal; timeline hits show time order.

    Natural-language time phrases are compiled at query time. Non-temporal
    queries take the existing recall path unchanged.
    """
    try:
        ns = _ns(namespace)
    except ValueError as exc:
        if json_out:
            _recall_json_error(exc, namespace=namespace, query=query)
        _die(exc)

    try:
        # _existing retains the established human diagnostic. JSON callers
        # need the raw exception so stdout remains a single JSON document.
        opener = open_existing_readonly if json_out else _existing_readonly
        with opener(ns) as st:
            hits = planned_recall(
                query,
                namespace=ns,
                as_of=as_of,
                since=since,
                until=until,
                clock=clock,
                tier=tier,
                k=k,
                store=st,
                include_residue=include_residue,
            )
    except (TemporalParseError, UnknownNamespaceError, ValueError, sqlite3.Error) as exc:
        if json_out:
            _recall_json_error(exc, namespace=ns, query=query)
        _die(exc)
    except Exception as exc:
        if json_out and is_retrieval_backend_error(exc):
            _recall_json_error(exc, namespace=ns, query=query)
        else:
            raise
    if json_out:
        bounded_hits, recall_budget = apply_recall_budget(
            [hit.as_dict() for hit in hits], k=k
        )
        payload = {
            "namespace": ns,
            "query": query,
            "hits": bounded_hits,
            "recall_budget": recall_budget,
        }
        execution = execution_metadata(hits)
        if execution is not None:
            payload["execution"] = execution
        typer.echo(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return
    if not hits:
        typer.echo("no hits")
        return
    typer.echo(f"{'#':<3} {'signal':<12} {'tier':<12} {'id':<36} snippet")
    for i, h in enumerate(hits, 1):
        tier_text = human_display(
            h.tier, limit=40, collapse_whitespace=True, sqlite_scalar=True
        )
        memory_id = human_display(
            h.memory_id, limit=64, collapse_whitespace=True, sqlite_scalar=True
        )
        signal = (
            f"rrf={h.score:.4f}"
            if h.vec_rank is not None or h.fts_rank is not None
            else "time-order"
        )
        typer.echo(
            f"{i:<3} {signal:<12} {tier_text:<12} {memory_id:<36} "
            f"{snippet(h.content, 140)}"
        )


@app.command("maintenance")
def maintenance_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    limit: int = typer.Option(64, "--limit"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Explicitly upgrade embeddings and drain queued embedding jobs.

    Recall is read-only and never invokes this operation.
    """
    try:
        ns = _ns(namespace)
        limit = clamp_limit(limit, default=64)
    except ValueError as exc:
        _die(exc)
    try:
        # This is the explicit mutating surface, but still must not create a
        # typo namespace.  Do the rejection through E3's no-write registry
        # snapshot before opening the mutating store: ``open_existing`` uses
        # the normal writer registry opener for a known namespace.
        if not namespace_exists_readonly(ns):
            raise UnknownNamespaceError(ns)
        with open_existing(ns) as st:
            upgraded = st.ensure_current_embeddings()
            drained = st.process_embedding_jobs(limit=limit)
        payload = {
            "namespace": ns,
            "maintenance_performed": True,
            "offline": env_flag("HAUNT_OFFLINE"),
            "embedding_upgrade": upgraded,
            "embedding_jobs": drained,
        }
    except (UnknownNamespaceError, ValueError) as exc:
        _die(exc)
    if json_out:
        typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    else:
        typer.echo(dumps(payload))


@app.command("timeline")
def timeline_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    session: Optional[str] = typer.Option(None, "--session"),
    since: Optional[str] = typer.Option(None, "--since"),
    until: Optional[str] = typer.Option(None, "--until"),
    clock: Optional[str] = typer.Option(
        None,
        "--clock",
        help="event_time | storage_time (default event_time). "
        "storage_time is ingest time (events.ts), not source time. "
        "write_time is a deprecated alias for storage_time.",
    ),
    limit: int = typer.Option(50, "--limit"),
    json_out: bool = typer.Option(False, "--json", help="Output stable JSON"),
) -> None:
    """Events in clock order (newest first). Default clock is event_time."""
    ns = _ns(namespace)
    limit = clamp_limit(limit, default=50)
    try:
        with open_existing(ns) as st:
            rows = st.events(
                session_id=session, since=since, until=until, clock=clock, limit=limit
            )
    except (UnknownNamespaceError, ValueError) as exc:
        if json_out:
            typer.echo(
                dumps({"ok": False, "error": str(exc), "namespace": ns}),
                err=True,
            )
            raise typer.Exit(2) from exc
        _die(exc)
    if json_out:
        typer.echo(dumps({"namespace": ns, "events": rows}))
        return
    if not rows:
        typer.echo("no events")
        return
    for r in rows:
        body = "" if r["content"] is None else snippet(r["content"], 120)
        if r["tool_name"] is not None and r["tool_name"] != "":
            tool = human_display(
                r["tool_name"],
                limit=48,
                collapse_whitespace=True,
                sqlite_scalar=True,
            )
            body = snippet(f"[tool:{tool}] {body}", 120)
        provenance = r["provenance"]
        source_channel = provenance.get("channel")
        source_origin = provenance.get("origin")
        source = (
            f"{human_display(source_channel, limit=48, collapse_whitespace=True, sqlite_scalar=True) if source_channel is not None and source_channel != '' else 'unknown'}/"
            f"{human_display(source_origin, limit=80, collapse_whitespace=True, sqlite_scalar=True) if source_origin is not None and source_origin != '' else 'unknown'}"
        )
        role = human_display(
            r["role"], limit=40, collapse_whitespace=True, sqlite_scalar=True
        )
        tier = human_display(
            r["tier"], limit=40, collapse_whitespace=True, sqlite_scalar=True
        )
        event_id = human_display(
            r["id"], limit=64, collapse_whitespace=True, sqlite_scalar=True
        )
        typer.echo(
            f"{format_iso(r['event_time'])}  {role:<10} {tier:<12} "
            f"{event_id}  source={source}  {body}"
        )


@app.command("namespaces")
def namespaces_cmd() -> None:
    """List namespaces with counts."""
    try:
        rows = list_namespaces()
    except NamespacePathError as exc:
        _die(exc)
    if not rows:
        typer.echo("no namespaces (run: haunt bootstrap)")
        return
    typer.echo(f"{'name':<24} {'events':>7} {'mem':>7} {'sess':>6} {'ents':>6}  db")
    for r in rows:
        err = r.get("error")
        name = human_display(
            r.get("name"), limit=48, collapse_whitespace=True, sqlite_scalar=True
        )
        db_path = human_display(
            r.get("db_path"),
            limit=240,
            collapse_whitespace=True,
            sqlite_scalar=True,
        )
        if err:
            error = human_display(err, limit=240, collapse_whitespace=True)
            typer.echo(f"{name:<24}  error: {error}  {db_path}")
        else:
            typer.echo(
                f"{name:<24} {human_display(r.get('events'), limit=16):>7} "
                f"{human_display(r.get('memories'), limit=16):>7} "
                f"{human_display(r.get('sessions'), limit=16):>6} "
                f"{human_display(r.get('entities'), limit=16):>6}  {db_path}"
            )


namespace_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Manage canonical namespace labels and aliases without moving databases.",
)
app.add_typer(namespace_app, name="namespace")


def _namespace_change(
    old_label: str,
    new_label: str,
    *,
    repository: str | None,
    action: str,
    apply: bool,
    plan_digest: str | None,
) -> None:
    try:
        report = change_namespace_label(
            old_label,
            new_label,
            repository=repository,
            action=action,
            apply=apply,
            plan_digest=plan_digest,
        )
    except (UnknownNamespaceError, NamespaceCollisionError, NamespaceMigrationError, ValueError) as exc:
        _die(exc)
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


@namespace_app.command("migrate")
def namespace_migrate_cmd(
    old_label: str = typer.Argument(..., help="Existing canonical label or alias"),
    new_label: str = typer.Argument(..., help="New canonical label"),
    repository: Optional[str] = typer.Option(
        None, "--repo", "--repository", help="Local repo path or git remote URL to bind"
    ),
    apply: bool = typer.Option(False, "--apply", help="Apply atomically (default is dry-run)"),
    plan_digest: Optional[str] = typer.Option(
        None, "--plan-digest", help="Digest printed by the matching dry-run"
    ),
) -> None:
    """Rename a canonical label, retaining the old label as an alias."""
    _namespace_change(
        old_label, new_label, repository=repository, action="rename", apply=apply,
        plan_digest=plan_digest,
    )


@namespace_app.command("alias")
def namespace_alias_cmd(
    source_label: str = typer.Argument(..., help="Existing canonical label or alias"),
    alias: str = typer.Argument(..., help="Additional label for the same namespace"),
    repository: Optional[str] = typer.Option(
        None, "--repo", "--repository", help="Local repo path or git remote URL to bind"
    ),
    apply: bool = typer.Option(False, "--apply", help="Apply atomically (default is dry-run)"),
    plan_digest: Optional[str] = typer.Option(
        None, "--plan-digest", help="Digest printed by the matching dry-run"
    ),
) -> None:
    """Add a unique alias to an existing canonical namespace."""
    _namespace_change(
        source_label, alias, repository=repository, action="alias", apply=apply,
        plan_digest=plan_digest,
    )


@namespace_app.command("reconcile")
def namespace_reconcile_cmd(
    source: str = typer.Argument(
        ..., help="Existing namespace to copy FROM (opened read-only, never modified)"
    ),
    target: str = typer.Argument(
        ..., help="Existing namespace to copy INTO (receives SOURCE's missing rows)"
    ),
    apply: bool = typer.Option(False, "--apply", help="Apply atomically (default is dry-run)"),
    plan_digest: Optional[str] = typer.Option(
        None, "--plan-digest", help="Digest printed by the matching dry-run"
    ),
) -> None:
    """Heal a namespace that was already split (backlog C3).

    Copies every row SOURCE has that TARGET does not into TARGET's database,
    preserving ids, timestamps, and correction/provenance lineage exactly.
    SOURCE is never written to. Both databases are backed up before TARGET is
    touched. Refuses -- writing nothing -- if any row's id collides with
    different content, if idempotency keys collide, or if either namespace is
    not at the current schema version. The single exception to "verbatim" is
    a session window: a session either side has a row or an event for is
    widened to hold every event either side has for it -- never narrowed,
    and never closed while TARGET still holds it open. The dry-run's
    `window_merges` lists every such window with the values it will take,
    and `unresolvable_windows` names any left untouched because a stored
    timestamp in them cannot be ordered. Embeddings are dropped and re-queued
    rather than copied; run this again to pick up rows added since. This does
    not touch the registry: both labels remain independently resolvable.
    """
    try:
        report = reconcile_namespaces(
            source, target, apply=apply, plan_digest=plan_digest
        )
    except (
        UnknownNamespaceError,
        NamespaceCollisionError,
        NamespaceMigrationError,
        ValueError,
    ) as exc:
        _die(exc)
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


@namespace_app.command("undo")
def namespace_undo_cmd(
    migration_id: str = typer.Argument(..., help="Applied migration identifier"),
    apply: bool = typer.Option(False, "--apply", help="Apply the reversal"),
    plan_digest: Optional[str] = typer.Option(
        None, "--plan-digest", help="Digest printed by the matching undo dry-run"
    ),
) -> None:
    """Reverse a recorded alias/rename after an exact-state dry-run."""
    try:
        report = undo_namespace_migration(
            migration_id, apply=apply, plan_digest=plan_digest
        )
    except (NamespaceMigrationError, UnknownNamespaceError, ValueError) as exc:
        _die(exc)
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


@namespace_app.command("retire-alias")
def namespace_retire_alias_cmd(
    label: str = typer.Argument(..., help="Noncanonical alias to check or retire"),
    apply: bool = typer.Option(False, "--apply", help="Retire when checks pass"),
) -> None:
    """Check registry references; external host config must be inspected manually."""
    try:
        report = retire_namespace_alias(label, apply=apply)
    except (UnknownNamespaceError, AliasRetirementError) as exc:
        _die(exc)
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


@namespace_app.command("retire")
def namespace_retire_cmd(
    label: str = typer.Argument(..., help="Drained namespace to check or retire"),
    into: str = typer.Option(
        ..., "--into", help="Namespace that must already hold every row of LABEL"
    ),
    apply: bool = typer.Option(False, "--apply", help="Retire when checks pass"),
) -> None:
    """Deregister and remove a namespace reconcile has already drained.

    Refuses while any row is still unique to LABEL, so run `namespace
    reconcile LABEL INTO --apply` first. Removes every label, alias, and
    repository binding for LABEL, then the database file itself -- left in
    place it would be re-adopted the next time anything registers that
    label. The database is backed up under HAUNT_HOME/backups first.
    """
    try:
        report = retire_namespace(label, into=into, apply=apply)
    except (
        UnknownNamespaceError,
        AliasRetirementError,
        NamespaceCollisionError,
        NamespaceMigrationError,
    ) as exc:
        _die(exc)
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


@app.command("health")
def health_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
) -> None:
    """Store, vec, and embed status."""
    from haunt.paths import registry_path

    es = embed_state()
    typer.echo(f"haunt         v{__version__}")
    typer.echo(f"HAUNT_HOME    {haunt_home()}")
    typer.echo(f"registry      {registry_path()}  exists={registry_path().exists()}")
    typer.echo(
        f"embed         loaded={es.model_id} dim={es.dim} requested={es.requested} "
        f"available={es.available} fallback={es.fallback} backend={getattr(es, 'backend', '?')}"
    )
    if es.error:
        typer.echo(f"embed error   {es.error}")
    ns = _ns(namespace)
    with _existing(ns) as st:
        vec_ok = st.vec_ok()
        vec_ver = st.vec_version()
        typer.echo(
            f"sqlite-vec    {'ok ' + str(vec_ver) if vec_ok else 'off (FTS-only)'}"
        )
        s = st.stats()
        typer.echo(f"namespace     {s['namespace']}")
        typer.echo(
            f"db            {s['db_path']}  bytes={s['db_size_bytes']}  wal={s['wal']}"
        )
        typer.echo(
            f"counts        events={s['events']} memories={s['memories']} "
            f"sessions={s['sessions']} entities={s['entities']} relations={s['relations']}"
        )
        typer.echo(f"tiers         {s['tiers']}")
        typer.echo(
            f"embedding     embedded={s['memories_embedded']} "
            f"pending={s['embedding_pending']} exhausted={s['embedding_exhausted']} "
            f"index={s['vector_index']}"
        )
        typer.echo(
            f"duplicates    memories={s['duplicate_memories']} "
            f"content={s['duplicate_content_values']}"
        )
        typer.echo(f"last write    {s['last_write']}")


@app.command("graph")
def graph_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    entity: Optional[str] = typer.Option(None, "--entity"),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Wipe entities/relations and re-extract from stored events (events/memories kept).",
    ),
) -> None:
    """Entities and typed relations (deterministic extract)."""
    ns = _ns(namespace)
    with _existing(ns) as st:
        if rebuild:
            report = st.rebuild_graph()
            typer.echo(
                f"rebuilt graph  ns={ns}  events={report['events']} memories={report['memories']}  "
                f"entities={report['entities_before']}→{report['entities']}  "
                f"relations={report['relations_before']}→{report['relations']}"
            )
        g = st.graph(entity)
        names = {e["id"]: e for e in g["entities"]}
        typer.echo(f"entities ({len(g['entities'])})")
        for e in g["entities"][:40]:
            name_text = human_display(
                e.get("name"), limit=64, collapse_whitespace=True, sqlite_scalar=True
            )
            type_text = human_display(
                e.get("type"), limit=40, collapse_whitespace=True, sqlite_scalar=True
            )
            last_seen = human_display(
                e.get("last_seen"),
                limit=80,
                collapse_whitespace=True,
                sqlite_scalar=True,
            )
            typer.echo(f"  {name_text:<32} {type_text:<12} last={last_seen}")
        typer.echo(f"relations ({len(g['relations'])})")
        for r in g["relations"][:50]:
            src_name = names.get(r["src_entity"], {}).get("name")
            dst_name = names.get(r["dst_entity"], {}).get("name")
            src_value = src_name if src_name is not None else r["src_entity"]
            dst_value = dst_name if dst_name is not None else r["dst_entity"]
            src = human_display(
                src_value, limit=64, collapse_whitespace=True, sqlite_scalar=True
            )
            dst = human_display(
                dst_value, limit=64, collapse_whitespace=True, sqlite_scalar=True
            )
            if src_name is None:
                src = src[:8]
            if dst_name is None:
                dst = dst[:8]
            rel = human_display(
                r.get("rel"), limit=48, collapse_whitespace=True, sqlite_scalar=True
            )
            weight = human_display(
                r.get("weight"),
                limit=24,
                collapse_whitespace=True,
                sqlite_scalar=True,
            )
            typer.echo(f"  {src}  --{rel}-->  {dst}  w={weight}")


@app.command("worldview")
def worldview_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    facts_cap: int = typer.Option(12, "--facts-cap"),
    names_cap: int = typer.Option(12, "--names-cap"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Compact namespace briefing: facts, entities, procedures, counts."""
    ns = _ns(namespace)
    facts_cap = clamp_limit(facts_cap, default=12)
    names_cap = clamp_limit(names_cap, default=12)
    with _existing(ns) as st:
        wv = st.worldview(facts_cap=facts_cap, names_cap=names_cap)
    if json_out:
        typer.echo(json.dumps(wv, ensure_ascii=False, allow_nan=False, indent=2))
        return
    typer.echo(
        f"namespace  {human_display(wv['namespace'], limit=80, collapse_whitespace=True, sqlite_scalar=True)}"
    )
    typer.echo(
        f"counts     events={wv['counts']['events']}  memories={wv['counts']['memories']}  sessions={wv['counts']['sessions']}"
    )
    typer.echo("")
    typer.echo(f"facts ({len(wv['facts'])})")
    for f in wv["facts"]:
        fact_id = human_display(
            f.get("id"), limit=40, collapse_whitespace=True, sqlite_scalar=True
        )
        typer.echo(f"  {fact_id[:8]}  {snippet(f.get('content'), 120)}")
    typer.echo("")
    typer.echo(f"names ({len(wv['names'])})")
    for n in wv["names"]:
        name = human_display(
            n.get("name"), limit=64, collapse_whitespace=True, sqlite_scalar=True
        )
        kind = human_display(
            n.get("type"), limit=40, collapse_whitespace=True, sqlite_scalar=True
        )
        mentions = human_display(n.get("mentions"), limit=24)
        typer.echo(f"  {name:<28} {kind:<12} mentions={mentions}")
    typer.echo("")
    typer.echo(f"procedures ({len(wv['procedures'])})")
    for p in wv["procedures"]:
        name = human_display(p.get("name"), limit=64, collapse_whitespace=True)
        proc_id = human_display(
            p.get("id"), limit=40, collapse_whitespace=True, sqlite_scalar=True
        )
        trigger_value = p.get("trigger")
        trigger = (
            f"  when: {human_display(trigger_value, limit=120, collapse_whitespace=True)}"
            if trigger_value is not None and trigger_value != ""
            else ""
        )
        typer.echo(f"  {name:<28} {proc_id[:8]}{trigger}")


procedure_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Named how-to procedures (verbatim steps).",
)
app.add_typer(procedure_app, name="procedure")


@procedure_app.command("write")
def procedure_write_cmd(
    name: str = typer.Argument(..., help="Procedure name"),
    body: str = typer.Option(..., "--body", "-b", help="Verbatim step-by-step body"),
    when: Optional[str] = typer.Option(
        None, "--when", "-w", help="Trigger description"
    ),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
) -> None:
    """Store a named procedure."""
    ns = _ns(namespace)
    with Store(ns) as st:
        r = st.procedure_write(
            name, body, trigger=when or "", origin="cli", channel="cli"
        )
    typer.echo(f"ok  procedure={name}  memory={r.memory_id}  ns={r.namespace}")


@procedure_app.command("get")
def procedure_get_cmd(
    name: str = typer.Argument(..., help="Procedure name"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
) -> None:
    """Retrieve a named procedure with source provenance."""
    ns = _ns(namespace)
    with _existing(ns) as st:
        proc = st.procedure_get(name)
    if not proc:
        typer.echo(f"not found: {name}")
        raise typer.Exit(1)
    typer.echo(
        f"name     {human_display(proc['name'], limit=160, collapse_whitespace=True)}"
    )
    if proc.get("trigger") is not None and proc.get("trigger") != "":
        typer.echo(
            f"trigger  {human_display(proc['trigger'], limit=240, collapse_whitespace=True)}"
        )
    typer.echo(
        f"id       {human_display(proc['id'], limit=160, collapse_whitespace=True, sqlite_scalar=True)}"
    )
    typer.echo(
        f"created  {human_display(proc['created_at'], limit=80, collapse_whitespace=True, sqlite_scalar=True)}"
    )
    typer.echo(f"provenance {human_display(proc['provenance'], limit=4096)}")
    typer.echo("---")
    typer.echo(
        human_display(
            proc["body"], limit=8192, preserve_layout=True, sqlite_scalar=True
        )
    )


@procedure_app.command("list")
def procedure_list_cmd(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
) -> None:
    """List all active procedures with source provenance."""
    ns = _ns(namespace)
    with _existing(ns) as st:
        procs = st.procedure_list()
    if not procs:
        typer.echo("no procedures")
        return
    typer.echo(f"{'name':<28} {'trigger':<32} id")
    for p in procs:
        name = human_display(p.get("name"), limit=64, collapse_whitespace=True)
        trigger = human_display(
            p.get("trigger", ""), limit=96, collapse_whitespace=True
        )
        proc_id = human_display(
            p.get("id"), limit=64, collapse_whitespace=True, sqlite_scalar=True
        )
        typer.echo(f"{name:<28} {trigger:<32} {proc_id[:12]}")
        typer.echo(f"  provenance {human_display(p['provenance'], limit=4096)}")


def _warn_unerased_backups(names: list[str]) -> None:
    """Report backups the purge could not sweep. Silent when there are none."""
    if not names:
        return
    typer.echo(
        "warning: erased content remains in "
        f"{len(names)} backup(s) under HAUNT_HOME/backups: " + ", ".join(names),
        err=True,
    )


@app.command("delete")
def delete_cmd(
    memory_id: Optional[str] = typer.Argument(
        None, help="Memory ID to permanently delete (omit when using --event-id)"
    ),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    event_id: Optional[str] = typer.Option(
        None, "--event-id", help="Delete all memories for this event"
    ),
) -> None:
    """Hard-delete a memory and its provenance chain (FTS, vec, graph, orphan events)."""
    if memory_id and event_id:
        typer.echo("error: pass memory_id or --event-id, not both", err=True)
        raise typer.Exit(2)
    if not memory_id and not event_id:
        typer.echo("error: memory_id or --event-id is required", err=True)
        raise typer.Exit(2)
    ns = _ns(namespace)
    if event_id:
        with _existing(ns) as st:
            rows = st.conn.execute(
                "SELECT id FROM memories WHERE event_id=?", (event_id,)
            ).fetchall()
            if not rows:
                typer.echo(f"no memories for event {event_id}")
                raise typer.Exit(1)
            if not yes:
                typer.confirm(
                    f"Permanently delete {len(rows)} memories for event {event_id}?",
                    abort=True,
                )
            # Each purge's whole-file rebuild takes a cross-process lock and
            # costs the size of the namespace, so one per memory would block
            # every writer in every namespace N times over. Defer it and pay
            # once; the erasure is only complete after the rebuild below.
            unerased: set[str] = set()
            for r in rows:
                result = st.purge(r["id"], rebuild=False)
                unerased.update(result.get("backups_unerased") or ())
                typer.echo(
                    f"purged  fts={result.get('fts_deleted')}  "
                    f"vec={result.get('vec_deleted')}  rels={result.get('relations_deleted')}  "
                    f"event={result.get('event_deleted')}"
                )
            typer.echo(f"bytes_overwritten={st.overwrite_erased_pages()}")
            _warn_unerased_backups(sorted(unerased))
        return
    with _existing(ns) as st:
        if not yes:
            typer.confirm(
                f"Permanently delete memory {memory_id}? This removes the memory, "
                "FTS index, vector embedding, graph data, and orphaned events.",
                abort=True,
            )
        result = st.purge(memory_id)
    if not result.get("ok"):
        typer.echo(f"error: {result.get('error', 'unknown')}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"ok  purged  fts={result['fts_deleted']}  vec={result['vec_deleted']}  "
        f"rels={result['relations_deleted']}  ents={result['entities_deleted']}  "
        f"event_deleted={result['event_deleted']}  "
        f"bytes_overwritten={result['bytes_overwritten']}"
    )
    _warn_unerased_backups(result["backups_unerased"])


@app.command("correct")
def correct_cmd(
    memory_id: str = typer.Argument(..., help="Memory ID to supersede"),
    replacement: Optional[str] = typer.Option(
        None,
        "--replacement",
        help="Verbatim replacement; omit for none (empty/whitespace are intentional)",
    ),
    reason: Optional[str] = typer.Option(None, "--reason"),
    idempotency_key: str = typer.Option(
        ...,
        "--idempotency-key",
        help="Required stable caller key for safe retries",
    ),
    session_id: Optional[str] = typer.Option(None, "--session"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    origin: str = typer.Option("cli", "--origin"),
) -> None:
    """Append a correction, optionally with a verbatim replacement."""
    ns = _ns(namespace)
    try:
        with _existing(ns) as st:
            result = st.contradict(
                memory_id,
                replacement=replacement,
                reason=reason,
                idempotency_key=idempotency_key,
                session_id=session_id,
                origin=origin,
                channel="cli",
            )
    except ValueError as exc:
        _die(exc)
    typer.echo(dumps({"namespace": ns, **result}))
    if not result.get("ok"):
        raise typer.Exit(1)


@app.command("trace")
def trace_cmd(
    memory_id: str = typer.Argument(..., help="Any surviving memory in the chain"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
) -> None:
    """Print an ordered correction trace as JSON."""
    ns = _ns(namespace)
    with _existing(ns) as st:
        result = st.trace(memory_id)
    typer.echo(dumps(result))
    if not result.get("ok"):
        raise typer.Exit(1)


@app.command("install")
def install_cmd(
    allow_alt_home: bool = typer.Option(
        False,
        "--allow-alt-home",
        help=(
            "Bind hosts even though HAUNT_HOME is not the default ~/.haunt. "
            "Global editor config will name this home; if it goes away, every "
            "hook stops capturing silently."
        ),
    ),
) -> None:
    """Bind haunt to all known hosts (Cursor, Claude Code). Idempotent."""
    from haunt.bootstrap import bind_launchers
    from haunt.hosts import AlternateHomeRefused, HostConfigError, install_all_hosts

    home, hook_cmd, mcp_cmd = bind_launchers()
    try:
        reports = install_all_hosts(
            str(home), hook_cmd, mcp_cmd, force=allow_alt_home
        )
    except AlternateHomeRefused as exc:
        # Hard error, unlike bootstrap: binding hosts is the entire point of
        # this command, so doing nothing quietly would be the same silence the
        # guard exists to prevent.
        _die(exc, code=2)
    except HostConfigError as exc:
        _die(exc, code=1)
    for r in reports:
        status = "seeded" if r.seeded else "merged"
        typer.echo(f"[{r.host}]  {status}")
        typer.echo(f"  hooks   {r.hooks_path}")
        typer.echo(f"  mcp     {r.mcp_path}")
        typer.echo(f"  rule    {r.rule_path}")
        typer.echo(f"  skill   {r.skill_path}")
        typer.echo(f"  events  {', '.join(r.events)}")
    typer.echo(f"home      {home}  (HAUNT_HOME)")
    typer.echo("Re-run after adding another editor: haunt install")


@app.command("cursor-install")
def cursor_install_cmd() -> None:
    """Bind haunt to Cursor: hooks.json + mcp.json + haunt.mdc + skill."""
    from haunt.cursor_hook import install_cursor_hooks

    from haunt.hosts import AlternateHomeRefused, HostConfigError

    try:
        report = install_cursor_hooks()
    except AlternateHomeRefused as exc:
        _die(exc, code=2)
    except HostConfigError as exc:
        _die(exc, code=1)
    typer.echo(f"hooks     {report['hooks_json']}")
    typer.echo(f"mcp       {report.get('mcp_json', '-')}")
    typer.echo(f"launcher  {report['launcher']}")
    typer.echo(f"events    {', '.join(report['events'])}")
    if report.get("rule"):
        typer.echo(f"rule      {report['rule']}")
    if report.get("skill"):
        typer.echo(f"skill     {report['skill']}")
    typer.echo("merged existing hooks/MCP; other entries were kept")
    typer.echo(f"home      {report['haunt_home']}  (HAUNT_HOME)")


@app.command("doctor")
def doctor_cmd() -> None:
    """Check sqlite-vec, haunt-mcp, embed, host files, and namespace collisions.

    Exit 1 if any check fails. A namespace already shared by two repositories
    is reported as an advisory and does not affect the exit code: healing it
    is `haunt namespace reconcile`'s operator-invoked job.
    """
    from haunt.bootstrap import bind_launchers
    from haunt.doctor import diagnose, format_doctor
    from haunt.hosts import host_install_refusal, install_all_hosts

    home, hook_cmd, mcp_cmd = bind_launchers()
    from haunt.paths import repair_private_modes

    repair_private_modes(home)
    report = diagnose(str(home), hook_cmd, mcp_cmd)
    typer.echo(format_doctor(report))

    if not report.ok and report.host_file_issues:
        # doctor's repair step is a host install, so it is guarded exactly like
        # one -- and it is the likelier of the two to be run by accident from a
        # smoke-test shell. Decided before the "Re-merging" line so the output
        # never claims a write that did not happen.
        refusal = host_install_refusal(str(home))
        if refusal is not None:
            typer.echo("")
            typer.echo("NOT re-merging hosts.")
            for line in str(refusal).splitlines():
                typer.echo("  " + line)
            raise typer.Exit(1)
        typer.echo("")
        typer.echo("Re-merging all hosts...")
        from haunt.hosts import HostConfigError

        try:
            install_all_hosts(str(home), hook_cmd, mcp_cmd)
        except HostConfigError as exc:
            _die(exc, code=1)
        report = diagnose(str(home), hook_cmd, mcp_cmd)
        typer.echo(format_doctor(report))
        if report.ok:
            typer.echo("Re-merged. All checks ok.")
            return

    if not report.ok:
        raise typer.Exit(1)


@app.command("dash")
def dash_cmd(
    port: int = typer.Option(7340, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
    install_icon: bool = typer.Option(
        False, "--install-icon", help="Write a desktop shortcut and exit"
    ),
    no_open: bool = typer.Option(
        False, "--no-open", help="Do not open the browser automatically"
    ),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help=(
            "Allow binding beyond loopback. Unsafe without the launch token: "
            "exposes the memory API on the network. Namespaces are not authorization."
        ),
    ),
) -> None:
    """Start the local memory console (127.0.0.1), or install a desktop shortcut."""
    if install_icon:
        from haunt.desktop import install_desktop_icon

        result = install_desktop_icon()
        if result.get("written"):
            typer.echo(f"desktop icon  {result['path']}")
        else:
            typer.echo(
                f"desktop icon  skipped ({result.get('reason', 'unsupported platform')})"
            )
        return

    from haunt.dashboard import check_dashboard_bind, run_dashboard

    try:
        check_dashboard_bind(host, allow_remote=allow_remote)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    typer.echo(f"haunt dash  http://{host}:{port}  home={haunt_home()}")
    run_dashboard(
        host=host, port=port, open_browser=not no_open, allow_remote=allow_remote
    )


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", help="Print version and exit", is_eager=True
    ),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None and not version:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
