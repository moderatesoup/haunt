"""Desktop shortcut creation. Best-effort, no extra dependencies."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


def _find_haunt_cmd() -> str:
    """Find the haunt CLI command path."""
    which = shutil.which("haunt")
    if which:
        return which
    local_bin = Path(sys.executable).parent / "haunt"
    if local_bin.is_file():
        return str(local_bin)
    return "haunt"


def install_desktop_icon(home: Path | None = None) -> dict[str, Any]:
    """Write a desktop shortcut for 'haunt dash'.

    Linux: writes a .desktop file to ~/.local/share/applications/.
    macOS: best-effort .command wrapper on ~/Desktop.
    Windows: best-effort .bat on Desktop.

    Returns dict with 'written' bool, 'path', and 'reason' on skip.
    """
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
        f"Exec={haunt_cmd} dash\n"
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
        f'exec "{haunt_cmd}" dash\n',
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
        f'@echo off\n"{haunt_cmd}" dash\n',
        encoding="utf-8",
    )
    return {"written": True, "path": str(bat_path), "platform": "windows"}
