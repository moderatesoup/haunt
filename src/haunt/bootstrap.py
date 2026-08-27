"""First-run setup: dirs, launcher, sqlite-vec probe, embed model, default ns."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

from haunt.embed import fts_only, warmup
from haunt.paths import bin_dir, ensure_layout, haunt_home, models_dir, repair_private_modes
from haunt.store import Store, init_registry, register_namespace, list_namespace_rows, reembed_all_namespaces
from haunt.util import diag, dumps


def _sh_single_quote(value: str) -> str:
    """Quote a string for /bin/sh so command substitution cannot run."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def _write_sh_wrapper(dest: Path, sibling_name: str, module: str) -> Path:
    """Space-free /bin/sh launcher. Do not Path.resolve() the venv python."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    python = str(Path(sys.executable).absolute())
    sibling = Path(python).parent / sibling_name
    home = _sh_single_quote(str(haunt_home()))
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
        body = prefix + f"exec {_sh_single_quote(str(sibling))} \"$@\"\n"
    else:
        body = prefix + f"exec {_sh_single_quote(python)} -m {module} \"$@\"\n"
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


def write_launcher() -> Path:
    """Space-free absolute launchers at ~/.haunt/bin/haunt-{mcp,hook,hook-claude}."""
    dest = _write_sh_wrapper(bin_dir() / "haunt-mcp", "haunt-mcp", "haunt.mcp_server")
    write_hook_launcher()
    write_claude_hook_launcher()
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
    """True when Store.drain_embedding_queue() found or touched anything
    worth telling the operator about.

    Keeps reembed_report (and format_report's rendering of it) from
    growing one no-op entry per namespace on every `haunt bootstrap` call
    -- most namespaces most of the time have an empty embedding_jobs queue,
    and drain_embedding_queue always makes at least one
    process_embedding_jobs() call (so `batches` alone is not a useful
    "was there ever a backlog" signal).

    C-series follow-up: one deliberate exception. A permanently FTS-only
    namespace -- HAUNT_FTS_ONLY=1 or HAUNT_EMBED_MODEL=off, both
    first-class supported modes; the project's own CI runs FTS-only --
    still queues embedding_jobs rows on every write (see observe()), but
    process_embedding_jobs() returns `available=False` before it ever
    touches a row's `attempts` (its `if not es.available` branch). That
    means `remaining` never shrinks and drain_embedding_queue() reports
    stop_reason="blocked" on *every single* `haunt bootstrap` call,
    forever -- not a one-time event, since nothing about that namespace
    is ever going to change on its own. Before the C4 out-of-band drain
    existed, a namespace in this shape produced no report line at all;
    this restores that silence for that specific shape rather than
    printing "stopped early (blocked)" -- wording that reads as a fault --
    for a namespace that is deliberately, permanently configured this way
    and not actionable by the operator. The top-level report already says
    the embedding backend is off once per run (`sqlite-vec skipped
    (FTS-only)` / the `embed` block); repeating it per namespace on every
    run adds no information, only alarm. Anyone who wants the raw
    per-namespace backlog count can still get it from Store.stats()'s
    `embedding_pending` / `embedding_exhausted` fields (also what the
    dashboard's per-namespace JSON exposes) -- it is only this alarming,
    repeats-forever bootstrap report line that is silenced, not the
    underlying data.

    `deliberately_off` is the caller's answer to "is `available=False`
    actually this deliberate, permanent shape, or something else".
    embed.py::_load() has THREE branches that all report
    available=False: fts_only(), offline(), and a bare `except Exception`
    catch-all for a genuine load failure (missing dependency, corrupted
    model cache, persistent network failure). Only the first two are
    "normal, permanent, and not actionable" -- both set
    EmbedState.backend="off" for exactly this reason. The catch-all sets
    EmbedState.backend="none" and is none of those things: it is an
    operator-fixable fault that can start or stop happening between runs,
    and silencing it here would hide a real backlog stuck for a reason
    the operator can actually do something about. The caller
    (bootstrap()) passes `embed_state.backend == "off"` -- reading the
    very EmbedState that already decided `available`, rather than
    re-deriving fts_only()/offline() a second time, which could disagree
    with the state that actually produced this `drained` result if
    anything about the environment changed in between. Threading this in
    as an explicit parameter (instead of calling fts_only()/offline()
    directly in here) also keeps this function a pure, hermetic check of
    its inputs -- see the tests, none of which need to control process
    env to exercise every branch.

    This also checks the `available` flag itself rather than
    `stop_reason == "blocked"`, and only silences when `processed`,
    `failed`, and `exhausted` are *all* zero too -- so it stays narrowly
    scoped to "nothing happened because there is no backend at all, and
    there is no other signal (like pre-existing exhausted rows) either".
    A block that happens while a backend IS available -- available=True --
    never matches this branch and always falls through to the normal
    reporting rule below; that is a real signal and must survive. Keying
    off `available` rather than the derived stop_reason string also means
    this stays correct even if a future change adds some other way to
    reach stop_reason="blocked" -- the only thing silenced here is "no
    backend, and it is deliberately, permanently configured that way,
    with nothing else to say" -- never a block that happens with a
    working backend, and never a genuine embedding-backend load failure.
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
            with _S(row["name"], create=False) as st:
                changed = st.ensure_current_embeddings()
                # C4: hook writes always defer embedding (defer_embedding=True),
                # so a hook-driven write never reaches the drain gated on
                # `commit and not defer_embedding` inside Store.observe().
                # Before this, the only other caller of process_embedding_jobs
                # was recall() -- so a namespace that is written to but rarely
                # searched grew an unbounded backlog. This out-of-band call
                # drains it, bounded by HAUNT_EMBED_DRAIN_LIMIT per namespace
                # per bootstrap run so a huge backlog cannot block `haunt
                # bootstrap` indefinitely -- see Store.drain_embedding_queue.
                drained = st.drain_embedding_queue()
                entry = dict(changed) if changed else {}
                # embed_state is the same EmbedState warmup() already
                # produced above -- backend=="off" only for _load()'s
                # fts_only()/offline() branches, never for the genuine
                # load-failure catch-all -- see _drain_worth_reporting.
                if changed or _drain_worth_reporting(
                    drained, deliberately_off=(embed_state.backend == "off")
                ):
                    entry["drain"] = drained
                    entry["namespace"] = row["name"]
                    entry["auto"] = True
                    reembed_report.append(entry)
    from haunt.desktop import install_desktop_icon
    icon_result = install_desktop_icon()

    from haunt.hosts import install_all_hosts

    hook_cmd = str(bin_dir() / "haunt-hook")
    mcp_cmd = str(bin_dir() / "haunt-mcp")
    host_reports = install_all_hosts(str(home), hook_cmd, mcp_cmd)

    hook_launcher = bin_dir() / "haunt-hook"
    report = {
        "haunt_home": str(home),
        "launcher": str(launcher.resolve()),
        "hook_launcher": str(hook_launcher.resolve()),
        "models_dir": str(models_dir()),
        "desktop_icon": icon_result.get("path") if icon_result.get("written") else None,
        "sqlite_vec": vec,
        "embed": {
            "requested": embed_state.requested,
            "loaded": embed_state.model_id,
            "dim": embed_state.dim,
            "available": embed_state.available,
            "fallback": embed_state.fallback,
            "error": embed_state.error,
        },
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
    lines = [
        f"haunt home    {home}",
        f"launcher      {report['launcher']}",
        f"hook          {report.get('hook_launcher', '')}",
        f"desktop icon  {icon or 'skipped (unsupported platform)'}",
        f"python        {report['python']}",
        f"sqlite-vec    {_format_sqlite_vec_line(report)}",
        f"embed         loaded={report['embed']['loaded']} dim={report['embed']['dim']} requested={report['embed']['requested']}"
        + (" (fallback)" if report["embed"]["fallback"] else ""),
    ]
    if report["embed"].get("error"):
        lines.append(f"embed error   {report['embed']['error']}")
    if report.get("backend"):
        lines.append(f"embed backend {report['backend']}")
    if report.get("download_bytes"):
        lines.append(f"embed bytes   {report['download_bytes']}")
    lines.append(f"namespace     {report['default_namespace']} → {report['default_db']}")
    for row in report.get("reembed") or []:
        # Full-reembed entries (auto, stale model/dim, or --reembed) carry
        # updated/total/dim/model. C4-only drain entries (nothing stale,
        # just a backlog to clear) may carry only `drain` -- render each
        # part only when present so a drain-only row doesn't print
        # "updated=None/None".
        line = f"reembed       ns={row.get('namespace')}"
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
