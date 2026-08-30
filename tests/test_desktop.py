"""Tests for desktop icon installation — uses temp HOME, no real desktop."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_install_icon_writes_file_linux(tmp_path, monkeypatch):
    """--install-icon writes a .desktop file in a temp HOME."""
    monkeypatch.setattr(sys, "platform", "linux")
    from haunt.desktop import install_desktop_icon

    result = install_desktop_icon(home=tmp_path)
    assert result["written"] is True
    path = Path(result["path"])
    assert path.exists()
    assert path.suffix == ".desktop"
    content = path.read_text()
    assert "Haunt Memories" in content
    assert "haunt" in content
    assert "dash" in content


def test_install_icon_writes_file_macos(tmp_path, monkeypatch):
    """--install-icon writes a .command file on macOS."""
    monkeypatch.setattr(sys, "platform", "darwin")
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    from haunt.desktop import install_desktop_icon

    result = install_desktop_icon(home=tmp_path)
    assert result["written"] is True
    path = Path(result["path"])
    assert path.exists()
    assert path.suffix == ".command"
    content = path.read_text()
    assert "haunt" in content
    assert "dash" in content


def test_install_icon_writes_file_windows(tmp_path, monkeypatch):
    """--install-icon writes a .bat file for Windows platform."""
    from haunt.desktop import _install_windows

    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    result = _install_windows("haunt", home=tmp_path)
    assert result["written"] is True
    path = Path(result["path"])
    assert path.exists()
    assert path.suffix == ".bat"
    content = path.read_text()
    assert "haunt" in content
    assert "dash" in content


def test_install_icon_unsupported_platform(tmp_path, monkeypatch):
    """Unsupported platform returns written=False."""
    monkeypatch.setattr(sys, "platform", "aix")
    from haunt.desktop import install_desktop_icon

    result = install_desktop_icon(home=tmp_path)
    assert result["written"] is False
    assert "unsupported" in result.get("reason", "")


def test_shortcut_invokes_haunt_dash_linux(tmp_path, monkeypatch):
    """Linux shortcut Exec line must call 'haunt dash' so browser-open rides the CLI."""
    monkeypatch.setattr(sys, "platform", "linux")
    from haunt.desktop import install_desktop_icon

    result = install_desktop_icon(home=tmp_path)
    assert result["written"] is True
    content = Path(result["path"]).read_text()
    exec_lines = [l for l in content.splitlines() if l.startswith("Exec=")]
    assert len(exec_lines) == 1
    assert exec_lines[0].endswith("dash"), (
        f"Exec must end with 'dash' (got: {exec_lines[0]!r}); "
        "browser-open is handled by 'haunt dash' itself"
    )


def test_shortcut_invokes_haunt_dash_macos(tmp_path, monkeypatch):
    """macOS .command must call 'haunt dash' so browser-open rides the CLI."""
    monkeypatch.setattr(sys, "platform", "darwin")
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    from haunt.desktop import install_desktop_icon

    result = install_desktop_icon(home=tmp_path)
    assert result["written"] is True
    content = Path(result["path"]).read_text()
    assert "dash" in content
    assert "--no-open" not in content, (
        "Desktop shortcut must NOT pass --no-open; browser should open"
    )


# --- launcher target and quoting ------------------------------------------
# A shortcut is written once and then read by a shell for years. It has to name
# an interpreter that still works after a PATH change, and it has to survive a
# path containing a space, a dollar sign, or a backtick.


def _canonical_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    from haunt.paths import bin_dir

    bin_dir().mkdir(parents=True, exist_ok=True)
    return bin_dir()


def test_bootstrap_plants_a_canonical_cli_wrapper(tmp_path, monkeypatch):
    """Without ~/.haunt/bin/haunt the shortcut has no stable target."""
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    from haunt.bootstrap import write_launcher
    from haunt.paths import bin_dir

    write_launcher()
    wrapper = bin_dir() / "haunt"
    assert wrapper.is_file(), "write_launcher must plant the canonical CLI wrapper"
    assert os.access(wrapper, os.X_OK)


def test_shortcut_prefers_the_canonical_wrapper_over_path(tmp_path, monkeypatch):
    """PATH resolution freezes at icon-write time; the wrapper does not."""
    bindir = _canonical_home(tmp_path, monkeypatch)
    wrapper = bindir / "haunt"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setattr(
        "haunt.desktop.shutil.which",
        lambda _name: "/somewhere/else/on/PATH/haunt",
    )
    from haunt.desktop import _find_haunt_cmd

    assert _find_haunt_cmd() == str(wrapper)


def test_macos_shortcut_cannot_run_command_substitution(tmp_path, monkeypatch):
    """A path holding $(...) must reach exec as literal text, not run.

    The shortcut used to interpolate into double quotes, where bash still
    performs command substitution before exec sees the word.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    canary = tmp_path / "CANARY"
    hostile = f"/nonexistent/$(touch {canary})/haunt"
    monkeypatch.setattr("haunt.desktop._find_haunt_cmd", lambda: hostile)
    from haunt.desktop import install_desktop_icon

    result = install_desktop_icon(home=tmp_path)
    script = Path(result["path"])
    # Execute it for real: exec fails on a nonexistent path, which is fine --
    # what matters is whether the substitution ran on the way there.
    subprocess.run(["/bin/bash", str(script)], capture_output=True)
    assert not canary.exists(), (
        f"command substitution executed from the shortcut: {script.read_text()!r}"
    )


def test_linux_exec_line_survives_a_path_with_a_space(tmp_path, monkeypatch):
    """Exec= is whitespace-split, so an unquoted path silently breaks.

    This repo's own checkout lives under 'GitHub Repos', so the space case is
    the normal case, not the exotic one.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    spaced = "/Users/someone/GitHub Repos/haunt/bin/haunt"
    monkeypatch.setattr("haunt.desktop._find_haunt_cmd", lambda: spaced)
    from haunt.desktop import install_desktop_icon

    result = install_desktop_icon(home=tmp_path)
    exec_line = next(
        line
        for line in Path(result["path"]).read_text().splitlines()
        if line.startswith("Exec=")
    )
    value = exec_line[len("Exec=") :]
    assert value.startswith('"') and '" dash' in value, exec_line
    # Unescape per the Desktop Entry Spec and confirm the path round-trips.
    argument = value[1 : value.index('" dash')].replace("\\\\", "\\")
    assert argument == spaced, argument
