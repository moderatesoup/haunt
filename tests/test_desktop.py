"""Tests for desktop icon installation — uses temp HOME, no real desktop."""

from __future__ import annotations

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
