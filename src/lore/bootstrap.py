"""First-run setup: dirs, launcher, sqlite-vec probe, embed model, default ns."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

from lore.embed import warmup
from lore.paths import bin_dir, ensure_layout, lore_home, models_dir
from lore.store import Store, init_registry, register_namespace, list_namespace_rows, reembed_all_namespaces
from lore.util import diag, dumps


def _write_sh_wrapper(dest: Path, sibling_name: str, module: str) -> Path:
    """Space-free /bin/sh launcher. Do not Path.resolve() the venv python."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    python = str(Path(sys.executable).absolute())
    sibling = Path(python).parent / sibling_name
    home = lore_home()
    if sibling.is_file():
        body = (
            "#!/bin/sh\n"
            f'export LORE_HOME="${{LORE_HOME:-{home}}}"\n'
            f'exec "{sibling}" "$@"\n'
        )
    else:
        body = (
            "#!/bin/sh\n"
            f'export LORE_HOME="${{LORE_HOME:-{home}}}"\n'
            f'exec "{python}" -m {module} "$@"\n'
        )
    dest.write_text(body, encoding="utf-8")
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def write_hook_launcher() -> Path:
    """Space-free launcher at ~/.lore/bin/engram-hook (plus lore-hook alias)."""
    _write_sh_wrapper(bin_dir() / "lore-hook", "lore-hook", "lore.cursor_hook")
    return _write_sh_wrapper(bin_dir() / "engram-hook", "engram-hook", "lore.cursor_hook")


def write_launcher() -> Path:
    """Space-free absolute launchers at ~/.lore/bin/{lore,engram}-{mcp,hook}."""
    dest = _write_sh_wrapper(bin_dir() / "lore-mcp", "lore-mcp", "lore.mcp_server")
    _write_sh_wrapper(bin_dir() / "engram-mcp", "engram-mcp", "lore.mcp_server")
    write_hook_launcher()
    return dest


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


def bootstrap(default_namespace: str = "default", reembed: bool = False) -> dict:
    home = ensure_layout()
    init_registry()
    launcher = write_launcher()
    vec = probe_sqlite_vec()
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
        # rebuild any namespace whose stored dim/model does not match
        from lore.store import Store as _S
        for row in list_namespace_rows():
            with _S(row["name"], create=False) as st:
                changed = st.ensure_current_embeddings()
                if changed:
                    changed["namespace"] = row["name"]
                    changed["auto"] = True
                    reembed_report.append(changed)
    hook_launcher = bin_dir() / "engram-hook"
    report = {
        "lore_home": str(home),
        "launcher": str(launcher.resolve()),
        "hook_launcher": str(hook_launcher.resolve()),
        "models_dir": str(models_dir()),
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
    }
    diag("bootstrap", **{k: report[k] for k in ("lore_home", "launcher")})
    return report


def format_report(report: dict) -> str:
    lines = [
        f"lore home     {report['lore_home']}",
        f"launcher      {report['launcher']}",
        f"hook          {report.get('hook_launcher', '')}",
        f"python        {report['python']}",
        f"sqlite-vec    {'ok ' + str(report['sqlite_vec'].get('version', '')) if report['sqlite_vec'].get('ok') else 'FAIL ' + str(report['sqlite_vec'].get('error'))}",
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
        lines.append(
            f"reembed       ns={row.get('namespace')} updated={row.get('updated')}/"
            f"{row.get('total')} dim={row.get('dim')} model={row.get('model')}"
        )
    if os.environ.get("LORE_JSON"):
        return dumps(report)
    return "\n".join(lines)
