"""First-run setup: dirs, launcher, sqlite-vec probe, embed model, default ns."""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
from pathlib import Path

from haunt.embed import bge_m3_source, fts_only, warmup
from haunt.paths import (
    bin_dir,
    ensure_layout,
    haunt_home,
    models_dir,
    repair_private_modes,
)
from haunt.store import Store, init_registry, register_namespace, list_namespace_rows, reembed_all_namespaces
from haunt.util import diag, dumps, sh_single_quote


def _write_sh_wrapper(dest: Path, sibling_name: str, module: str) -> Path:
    """Space-free /bin/sh launcher. Do not Path.resolve() the venv python."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    python = str(Path(sys.executable).absolute())
    sibling = Path(python).parent / sibling_name
    home = sh_single_quote(str(haunt_home()))
    # Do not use export HAUNT_HOME="${HAUNT_HOME:-...}". The default word
    # inside double quotes still runs command substitution.
    prefix = (
        "#!/bin/sh\n"
        "if [ -z \"${HAUNT_HOME}\" ]; then\n"
        f"  HAUNT_HOME={home}\n"
        "  export HAUNT_HOME\n"
        "fi\n"
    )
    if sibling.is_file():
        body = prefix + f"exec {sh_single_quote(str(sibling))} \"$@\"\n"
    else:
        body = prefix + f"exec {sh_single_quote(python)} -m {module} \"$@\"\n"
    dest.write_text(body, encoding="utf-8")
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def write_hook_launcher() -> Path:
    """Space-free launcher at ~/.haunt/bin/haunt-hook."""
    return _write_sh_wrapper(bin_dir() / "haunt-hook", "haunt-hook", "haunt.cursor_hook")


def write_claude_hook_launcher() -> Path:
    """Space-free launcher at ~/.haunt/bin/haunt-hook-claude."""
    return _write_sh_wrapper(
        bin_dir() / "haunt-hook-claude", "haunt-hook-claude", "haunt.claude_hook"
    )


def write_cli_launcher() -> Path:
    """Space-free launcher at ~/.haunt/bin/haunt.

    Without this the desktop shortcut has no canonical target and falls back to
    `shutil.which("haunt")`, which resolves at icon-write time and freezes. On a
    machine with a pyenv shim ahead of the haunt venv on PATH that pins the
    launcher to an interpreter whose sqlite3 was built without extension
    loading, so the icon opens a console that cannot do vector work.
    """
    return _write_sh_wrapper(bin_dir() / "haunt", "haunt", "haunt.cli")


def write_launcher() -> Path:
    """Space-free absolute launchers at ~/.haunt/bin/haunt{,-mcp,-hook,-hook-claude}."""
    dest = _write_sh_wrapper(bin_dir() / "haunt-mcp", "haunt-mcp", "haunt.mcp_server")
    write_hook_launcher()
    write_claude_hook_launcher()
    write_cli_launcher()
    return dest


def bind_launchers() -> tuple[Path, str, str]:
    """Write wrappers and return (haunt_home, hook_cmd, mcp_cmd).

    MCP command is the haunt-mcp wrapper under HAUNT_HOME/bin — never a PATH name.
    """
    home = ensure_layout()
    write_launcher()
    return home, str(bin_dir() / "haunt-hook"), str(bin_dir() / "haunt-mcp")


def probe_sqlite_vec() -> dict[str, str | bool]:
    import sqlite3

    import sqlite_vec

    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        ver = conn.execute("SELECT vec_version()").fetchone()[0]
        return {"ok": True, "version": str(ver)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        conn.close()


class BootstrapError(SystemExit):
    """Raised when bootstrap hits a fatal problem (e.g. sqlite-vec missing)."""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(1)


def _drain_worth_reporting(drained: dict, *, deliberately_off: bool) -> bool:
    """True when a drain result earns a per-namespace line in the report.

    Pass `deliberately_off` as `EmbedState.backend == "off"`, not as
    `not available`: an FTS-only or offline namespace queues embedding jobs
    that can never run, so its permanent backlog is reported as nothing,
    while a genuine embed-backend load failure (backend="none") still is.
    Non-zero `remaining` alone is therefore not enough to return True.
    """
    if (
        drained.get("available") is False
        and deliberately_off
        and not drained.get("processed")
        and not drained.get("failed")
        and not drained.get("exhausted")
    ):
        return False
    return bool(
        drained.get("processed")
        or drained.get("failed")
        or drained.get("remaining")
        or drained.get("exhausted")
    )


def bootstrap(default_namespace: str = "default", reembed: bool = False) -> dict:
    home = ensure_layout()
    repair_private_modes(home)
    launcher = write_launcher()
    vec = probe_sqlite_vec()
    if not vec.get("ok") and not fts_only():
        hint = (
            "sqlite-vec failed to load: " + str(vec.get("error", "unknown")) + "\n"
            "\n"
            "haunt requires sqlite-vec for vector storage. Common fixes:\n"
            "  - Use the system Python or Homebrew Python (not pyenv) which\n"
            "    ships with loadable-extension support.\n"
            "  - Ensure 'pip install sqlite-vec' succeeded.\n"
            "  - On macOS with pyenv: rebuild with\n"
            "    PYTHON_CONFIGURE_OPTS=\"--enable-loadable-sqlite-extensions\" pyenv install\n"
            "  - Or set HAUNT_FTS_ONLY=1 (or HAUNT_EMBED_MODEL=off) to bootstrap FTS-only.\n"
        )
        raise BootstrapError(hint)
    try:
        init_registry()
    except Exception as exc:
        raise BootstrapError(
            "failed to init registry after sqlite-vec probe: " + str(exc)
        ) from exc
    embed_state = warmup()
    db = register_namespace(default_namespace, repo_path=None)
    # touch schema
    with Store(default_namespace) as st:
        st.set_meta("bootstrapped", "1")
        if embed_state.available:
            st.set_meta("embed_model", embed_state.model_id)
            st.set_meta("embed_dim", str(embed_state.dim))
    reembed_report: list = []
    if reembed:
        reembed_report = reembed_all_namespaces()
    else:
        from haunt.store import Store as _S
        for row in list_namespace_rows():
            try:
                with _S(row["name"], create=False) as st:
                    changed = st.ensure_current_embeddings()
                    # Hook writes always defer embedding, so nothing else
                    # drains this queue out of band. Bounded per namespace
                    # per run by HAUNT_EMBED_DRAIN_LIMIT.
                    drained = st.drain_embedding_queue()
            except (sqlite3.Error, OSError, ValueError) as exc:
                # Isolated per namespace: aborting here would strand every
                # namespace listed after this one on an undrained queue, and
                # skip install_desktop_icon()/install_all_hosts() below.
                #
                # ValueError, not the individual classes: NamespacePathError,
                # UnknownNamespaceError, NamespaceCollisionError,
                # NamespaceMigrationError and AliasRetirementError are all
                # ValueError subclasses, and naming only some of them let a
                # sibling through. `namespace retire --apply` deregistering a
                # namespace between list_namespace_rows() and Store() raises
                # UnknownNamespaceError here, which is exactly the abort this
                # guard exists to prevent.
                reembed_report.append(
                    {"namespace": row["name"], "auto": True, "error": str(exc)}
                )
                continue
            entry = dict(changed) if changed else {}
            if changed or _drain_worth_reporting(
                drained, deliberately_off=(embed_state.backend == "off")
            ):
                entry["drain"] = drained
                entry["namespace"] = row["name"]
                entry["auto"] = True
                reembed_report.append(entry)
    from haunt.hosts import host_install_refusal, install_all_hosts

    # One decision for both global-config writes below, so the report tells a
    # single coherent story. bootstrap() skips rather than aborts: every other
    # step above (layout, launchers, registry, embeddings) is still useful and
    # still correct for an alternate home. The skip is reported loudly by
    # format_report() -- silence here is the failure mode this guard exists for.
    refusal = host_install_refusal(str(home))

    from haunt.desktop import install_desktop_icon
    icon_result = (
        install_desktop_icon()
        if refusal is None
        else {"written": False, "reason": "haunt home is not the default home"}
    )

    hook_cmd = str(bin_dir() / "haunt-hook")
    mcp_cmd = str(bin_dir() / "haunt-mcp")
    host_reports = (
        install_all_hosts(str(home), hook_cmd, mcp_cmd) if refusal is None else []
    )

    hook_launcher = bin_dir() / "haunt-hook"
    report = {
        "haunt_home": str(home),
        "launcher": str(launcher.resolve()),
        "hook_launcher": str(hook_launcher.resolve()),
        "models_dir": str(models_dir()),
        "desktop_icon": icon_result.get("path") if icon_result.get("written") else None,
        "desktop_icon_reason": (
            None if icon_result.get("written") else icon_result.get("reason")
        ),
        "sqlite_vec": vec,
        "embed": {
            "requested": embed_state.requested,
            "loaded": embed_state.model_id,
            "dim": embed_state.dim,
            "available": embed_state.available,
            "fallback": embed_state.fallback,
            "error": embed_state.error,
            # None for a non-ONNX backend, or a cache predating the marker.
            "source": bge_m3_source() if embed_state.backend == "onnx" else None,
        },
        "hosts_skipped": str(refusal) if refusal is not None else None,
        "default_namespace": default_namespace,
        "default_db": str(db),
        "python": sys.executable,
        "fts_only": not embed_state.available,
        "reembed": reembed_report,
        "backend": getattr(embed_state, "backend", None),
        "download_bytes": getattr(embed_state, "download_bytes", None),
        "hosts": [
            {
                "host": hr.host,
                "hooks_path": hr.hooks_path,
                "mcp_path": hr.mcp_path,
                "rule_path": hr.rule_path,
                "skill_path": hr.skill_path,
                "events": hr.events,
                "seeded": hr.seeded,
            }
            for hr in host_reports
        ],
    }
    repair_private_modes(home)
    diag("bootstrap", **{k: report[k] for k in ("haunt_home", "launcher")})
    return report


def _format_sqlite_vec_line(report: dict) -> str:
    vec = report.get("sqlite_vec") or {}
    if vec.get("ok"):
        return "ok " + str(vec.get("version", ""))
    err = str(vec.get("error", "unknown"))
    if report.get("fts_only"):
        return "skipped (FTS-only) " + err
    return "FAIL " + err


def format_report(report: dict) -> str:
    home = report.get('haunt_home', '')
    icon = report.get('desktop_icon')
    # Never claim "unsupported platform" for a skip that had another cause.
    icon_line = icon or (
        "skipped (" + str(report.get("desktop_icon_reason") or "unsupported platform") + ")"
    )
    lines = [
        f"haunt home    {home}",
        f"launcher      {report['launcher']}",
        f"hook          {report.get('hook_launcher', '')}",
        f"desktop icon  {icon_line}",
        f"python        {report['python']}",
        f"sqlite-vec    {_format_sqlite_vec_line(report)}",
        f"embed         loaded={report['embed']['loaded']} dim={report['embed']['dim']} requested={report['embed']['requested']}"
        + (" (fallback)" if report["embed"]["fallback"] else ""),
    ]
    if report["embed"].get("error"):
        lines.append(f"embed error   {report['embed']['error']}")
    source = report["embed"].get("source") or {}
    if source:
        lines.append(
            f"embed source  {source.get('repo_id')}@{source.get('revision')}"
        )
    if report.get("backend"):
        lines.append(f"embed backend {report['backend']}")
    if report.get("download_bytes"):
        lines.append(f"embed bytes   {report['download_bytes']}")
    lines.append(f"namespace     {report['default_namespace']} → {report['default_db']}")
    for row in report.get("reembed") or []:
        line = f"reembed       ns={row.get('namespace')}"
        if row.get("error"):
            # The other namespaces still ran; say which one did not.
            lines.append(line + f" FAILED {row['error']}")
            continue
        # Full-reembed entries (auto, stale model/dim, or --reembed) carry
        # updated/total/dim/model. C4-only drain entries (nothing stale,
        # just a backlog to clear) may carry only `drain` -- render each
        # part only when present so a drain-only row doesn't print
        # "updated=None/None".
        if "updated" in row:
            line += (
                f" updated={row.get('updated')}/{row.get('total')}"
                f" dim={row.get('dim')} model={row.get('model')}"
            )
        drain = row.get("drain")
        if drain:
            status = (
                "fully drained"
                if not drain.get("stopped_early")
                else f"stopped early ({drain.get('stop_reason')})"
            )
            line += (
                f" drain processed={drain.get('processed')} failed={drain.get('failed')}"
                f" remaining={drain.get('remaining')} exhausted={drain.get('exhausted')}"
                f" [{status}]"
            )
        lines.append(line)
    embed = report.get("embed", {})
    if embed.get("available") and not embed.get("fallback"):
        model_id = embed.get("loaded", "")
        if "bge-m3" in model_id.lower():
            lines.append("")
            lines.append(
                "note        The quality embed model BAAI/bge-m3 (~2.28 GB) is active."
            )
            lines.append(
                "            First observe/recall may take a moment if the model"
            )
            lines.append(
                "            was just downloaded. For a smaller model (~67 MB), set:"
            )
            lines.append(
                "            HAUNT_EMBED_MODEL=BAAI/bge-small-en-v1.5"
            )
    elif embed.get("available") and embed.get("fallback"):
        lines.append("")
        lines.append(
            "note        Running with fallback model. For best quality, ensure"
        )
        lines.append(
            "            BAAI/bge-m3 can download (~2.28 GB)."
        )
    skipped = report.get("hosts_skipped")
    if skipped:
        lines.append("")
        lines.append("HOST BIND SKIPPED — editor hooks were NOT installed or updated.")
        for line in str(skipped).splitlines():
            lines.append("  " + line)
    for h in report.get("hosts") or []:
        status = "seeded" if h.get("seeded") else "merged"
        lines.append(
            f"host bind     {h['host']}: {status}  "
            f"hooks={h.get('hooks_path', '-')}  mcp={h.get('mcp_path', '-')}  "
            f"rule={h.get('rule_path', '-')}  skill={h.get('skill_path', '-')}"
        )
    if os.environ.get("HAUNT_JSON"):
        return dumps(report)
    return "\n".join(lines)
