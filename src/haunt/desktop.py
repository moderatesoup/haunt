"""Desktop shortcut creation. Best-effort, no extra dependencies."""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from haunt.paths import bin_dir
from haunt.util import desktop_exec_quote, sh_single_quote


def _find_haunt_cmd() -> str:
    """Resolve the haunt CLI for a shortcut, preferring the canonical wrapper.

    ~/.haunt/bin/haunt is written by bootstrap and re-exec's the interpreter
    haunt is actually installed into, so it keeps working across PATH changes
    and venv rebuilds. shutil.which resolves once, at icon-write time, and then
    freezes -- on a machine with a pyenv shim ahead of the haunt venv that
    pinned the shortcut to an interpreter whose sqlite3 cannot load extensions,
    so the console opened but no vector search worked. Keep which() as the
    fallback for installs that never ran bootstrap.
    """
    canonical = bin_dir() / "haunt"
    if canonical.is_file():
        return str(canonical)
    which = shutil.which("haunt")
    if which:
        return which
    local_bin = Path(sys.executable).parent / "haunt"
    if local_bin.is_file():
        return str(local_bin)
    return "haunt"


def install_desktop_icon(
    home: Path | None = None, *, force: bool = False
) -> dict[str, Any]:
    """Write a desktop shortcut for 'haunt dash'.

    Linux: writes a .desktop file to ~/.local/share/applications/.
    macOS: best-effort .command wrapper on ~/Desktop.
    Windows: best-effort .bat on Desktop.

    Skipped when `home` is None (so the real user home is the target) and
    HAUNT_HOME is not the default home: a bootstrap run from a temp home is
    not entitled to drop files in the operator's actual Desktop. An explicit
    `home` is already a redirected target, so it is never refused.

    Returns dict with 'written' bool, 'path', and 'reason' on skip.
    """
    if home is None:
        from haunt.hosts import ALT_HOME_ENV, host_install_refusal
        from haunt.paths import haunt_home

        refusal = host_install_refusal(haunt_home(), force=force)
        if refusal is not None:
            return {
                "written": False,
                "reason": (
                    f"haunt home {refusal.haunt_home} is not the default "
                    f"{refusal.default_home}; set {ALT_HOME_ENV}=1 to write anyway"
                ),
            }
    platform = sys.platform
    haunt_cmd = _find_haunt_cmd()

    if platform.startswith("linux"):
        return _install_linux(haunt_cmd, home)
    elif platform == "darwin":
        return _install_macos(haunt_cmd, home)
    elif platform == "win32":
        return _install_windows(haunt_cmd, home)
    else:
        return {"written": False, "reason": f"unsupported platform: {platform}"}


def _install_linux(haunt_cmd: str, home: Path | None = None) -> dict[str, Any]:
    apps_dir = (home or Path.home()) / ".local" / "share" / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)
    desktop_path = apps_dir / "haunt-memories.desktop"

    desktop_entry = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Haunt Memories\n"
        "Comment=Local memory console for AI agents\n"
        f"Exec={desktop_exec_quote(haunt_cmd)} dash\n"
        "Terminal=true\n"
        "Categories=Development;Utility;\n"
        "StartupNotify=false\n"
    )
    desktop_path.write_text(desktop_entry, encoding="utf-8")
    desktop_path.chmod(
        desktop_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )

    try:
        subprocess.run(
            ["update-desktop-database", str(apps_dir)],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        pass

    return {"written": True, "path": str(desktop_path), "platform": "linux"}


def _install_macos(haunt_cmd: str, home: Path | None = None) -> dict[str, Any]:
    desktop = (home or Path.home()) / "Desktop"
    if not desktop.exists():
        desktop = (home or Path.home()) / ".local" / "share" / "applications"
        desktop.mkdir(parents=True, exist_ok=True)

    script_path = desktop / "Haunt Memories.command"
    script_path.write_text(
        f"#!/bin/bash\n"
        f"exec {sh_single_quote(haunt_cmd)} dash\n",
        encoding="utf-8",
    )
    script_path.chmod(
        script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    return {"written": True, "path": str(script_path), "platform": "macos"}


def _install_windows(haunt_cmd: str, home: Path | None = None) -> dict[str, Any]:
    desktop = (home or Path.home()) / "Desktop"
    if not desktop.exists():
        desktop = (home or Path.home()) / ".local" / "share" / "applications"
        desktop.mkdir(parents=True, exist_ok=True)

    bat_path = desktop / "Haunt Memories.bat"
    bat_path.write_text(
        '@echo off\n"' + haunt_cmd.replace("%", "%%") + '" dash\n',
        encoding="utf-8",
    )
    return {"written": True, "path": str(bat_path), "platform": "windows"}
