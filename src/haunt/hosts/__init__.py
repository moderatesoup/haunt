"""Host adapters: bind haunt hooks + MCP + rules into editor/agent hosts.

Each adapter exposes:
    install(haunt_home, hook_cmd, mcp_cmd) -> HostReport
    doctor(haunt_home, hook_cmd, mcp_cmd)  -> HostStatus
    HOST_NAME: str                         (e.g. "cursor", "claude-code")

install_all_hosts() iterates every known adapter and binds them all.
A later host (Codex, …) is another module added to _adapters().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HAUNT_MCP_LEAVES = frozenset({"haunt-mcp"})
HAUNT_HOOK_LEAVES = frozenset({"haunt-hook"})
# Must stay in lockstep with #44 / contrib host rules. Doctor fails if install
# plants the pre-rewrite essay that omitted the wired temporal path.
RULE_MARKERS = ("memory_recall", "haunt", "compile() runs automatically on memory_recall")


def command_leaf(command: str) -> str:
    return command.replace("\\", "/").rstrip("/").split("/")[-1]


def mcp_command_issues(command: str, expected: str | None = None) -> list[str]:
    """Specific errors when a host MCP command is not the haunt-mcp wrapper."""
    issues: list[str] = []
    leaf = command_leaf(command)
    if leaf not in HAUNT_MCP_LEAVES:
        issues.append(f"MCP command is not haunt-mcp: {command}")
        return issues
    path = Path(command)
    if not path.is_absolute():
        issues.append(f"MCP command is not an absolute haunt-mcp wrapper: {command}")
        return issues
    if expected and command != expected:
        try:
            same = path.is_file() and Path(expected).is_file() and path.samefile(expected)
        except OSError:
            same = False
        if not same:
            issues.append(f"MCP command is not the haunt-mcp wrapper: {command}")
    return issues


def hook_command_issues(
    command: str,
    expected: str | None = None,
    leaves: frozenset[str] | None = None,
) -> list[str]:
    """Specific errors when a host hook command is not the expected wrapper.

    Same honesty bar as MCP (leaf + absolute + samefile against expected),
    plus a missing-file FAIL. MCP existence lives on the wrapper probe;
    hooks have no equivalent, so doctor must check the planted path here.
    """
    issues: list[str] = []
    if leaves is not None:
        allowed = leaves
    elif expected:
        allowed = frozenset({command_leaf(expected)})
    else:
        allowed = HAUNT_HOOK_LEAVES
    leaf = command_leaf(command)
    if leaf not in allowed:
        label = command_leaf(expected) if expected else "haunt-hook"
        issues.append(f"hook command is not {label}: {command}")
        return issues
    path = Path(command)
    if not path.is_absolute():
        issues.append(f"hook command is not an absolute haunt-hook wrapper: {command}")
        return issues
    if not path.is_file():
        issues.append(f"hook command not found: {command}")
        return issues
    if expected and command != expected:
        try:
            same = Path(expected).is_file() and path.samefile(expected)
        except OSError:
            same = False
        if not same:
            issues.append(f"hook command is not the haunt-hook wrapper: {command}")
    return issues


def rule_issue(path: Path, label: str) -> str | None:
    if not path.is_file():
        return f"{label} not found"
    text = path.read_text(encoding="utf-8")
    missing = [m for m in RULE_MARKERS if m not in text]
    if missing:
        return f"{label} missing expected text: {', '.join(missing)}"
    return None


@dataclass
class HostReport:
    """What a single host bind wrote / merged."""

    host: str
    hooks_path: str | None = None
    mcp_path: str | None = None
    rule_path: str | None = None
    skill_path: str | None = None
    events: list[str] = field(default_factory=list)
    seeded: bool = False  # True if the config dir was created fresh
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HostStatus:
    """Doctor report for one host."""

    host: str
    hooks_present: bool = False
    mcp_present: bool = False
    rule_present: bool = False
    skill_present: bool = False
    hooks_path: str | None = None
    mcp_path: str | None = None
    rule_path: str | None = None
    skill_path: str | None = None
    issues: list[str] = field(default_factory=list)


def _adapters() -> list:
    from haunt.hosts import claude, cursor

    return [cursor, claude]


def install_all_hosts(haunt_home: str, hook_cmd: str, mcp_cmd: str) -> list[HostReport]:
    """Bind every known host. Returns a report per host.

    Each adapter may compute host-specific launchers from haunt_home/bin/.
    hook_cmd is the default (Cursor) hook; adapters that need a different
    launcher (e.g. haunt-hook-claude) derive it from haunt_home.
    """
    reports: list[HostReport] = []
    for adapter in _adapters():
        report = adapter.install(haunt_home, hook_cmd, mcp_cmd)
        reports.append(report)
    return reports


def doctor_all_hosts(haunt_home: str, hook_cmd: str, mcp_cmd: str) -> list[HostStatus]:
    """Check every known host for correct bindings."""
    statuses: list[HostStatus] = []
    for adapter in _adapters():
        status = adapter.doctor(haunt_home, hook_cmd, mcp_cmd)
        statuses.append(status)
    return statuses
