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
from typing import Any


@dataclass
class HostReport:
    """What a single host bind wrote / merged."""

    host: str
    hooks_path: str | None = None
    mcp_path: str | None = None
    rule_path: str | None = None
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
    hooks_path: str | None = None
    mcp_path: str | None = None
    rule_path: str | None = None
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
