"""Host adapters: bind haunt hooks + MCP + rules into editor/agent hosts.

Each adapter exposes:
    install(haunt_home, hook_cmd, mcp_cmd) -> HostReport
    doctor(haunt_home, hook_cmd, mcp_cmd)  -> HostStatus
    HOST_NAME: str                         (e.g. "cursor", "claude-code")

install_all_hosts() iterates every known adapter and binds them all.
A later host (Codex, …) is another module added to _adapters().
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class HostConfigError(ValueError):
    """Existing host JSON is malformed. Refuse to overwrite it."""


ALT_HOME_ENV = "HAUNT_ALLOW_ALT_HOME_HOST_INSTALL"


class AlternateHomeRefused(RuntimeError):
    """Host config install refused: this haunt home is not the default one.

    Editor host config (Claude Code settings.json, Cursor hooks.json) is
    global and shared by every session on the machine. Binding it to a
    HAUNT_HOME that is not ~/.haunt plants that path in the user's real
    editor config; when the alternate home goes away -- a smoke test's
    temp dir, a deleted worktree -- every hook command points at nothing
    and memory capture stops with no error anywhere.
    """

    def __init__(self, haunt_home: str, default_home: str) -> None:
        self.haunt_home = haunt_home
        self.default_home = default_home
        super().__init__(
            "refusing to write global host config for a non-default haunt home\n"
            f"  haunt home    {haunt_home}\n"
            f"  default home  {default_home}\n"
            "  Host config is global: this path would replace the hook command in\n"
            "  the real Claude Code / Cursor config. If it later disappears, every\n"
            "  hook silently stops capturing.\n"
            f"  To install anyway, set {ALT_HOME_ENV}=1 (exactly 1)."
        )


def default_haunt_home() -> Path:
    """The one haunt home that may be written into global host config."""
    return (Path.home() / ".haunt").resolve()


def _resolved(path: str | Path) -> Path:
    """Absolute, symlink-free, ~-expanded. `.smoke-home` cannot alias past this."""
    return Path(path).expanduser().resolve()


def alt_home_install_allowed() -> bool:
    """True only for an exact ALT_HOME_ENV=1. Nothing else counts as consent."""
    return os.environ.get(ALT_HOME_ENV, "").strip() == "1"


def host_install_refusal(
    haunt_home: str | Path, *, force: bool = False
) -> AlternateHomeRefused | None:
    """The refusal for binding global host config to `haunt_home`, or None.

    Checks the home about to be *written*, not the ambient HAUNT_HOME: the
    argument is what lands in settings.json, so it is the thing to judge.
    Both sides are fully resolved, so a symlink pointing at a temp dir is
    refused and a symlink pointing at ~/.haunt is allowed.
    """
    if force or alt_home_install_allowed():
        return None
    resolved = _resolved(haunt_home)
    default = default_haunt_home()
    if resolved == default:
        return None
    return AlternateHomeRefused(str(resolved), str(default))


def check_host_install_allowed(haunt_home: str | Path, *, force: bool = False) -> None:
    """Raise AlternateHomeRefused unless this home may touch global host config."""
    refusal = host_install_refusal(haunt_home, force=force)
    if refusal is not None:
        raise refusal


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


def hook_command_defect(command: str) -> str | None:
    """Why a planted hook command cannot run, or None when it can.

    Existence *and* executability, because both fail the same way: the host
    fires the hook, nothing runs, nothing is reported, capture is off. Only
    absolute commands are judged -- a bare name is resolved by the host's own
    PATH and is a different (already reported) kind of wrong.
    """
    path = Path(command)
    if not path.is_absolute():
        return None
    try:
        if not path.exists():
            return "not found"
        if not path.is_file():
            return "not a regular file"
        if not os.access(path, os.X_OK):
            return "not executable"
    except OSError as exc:
        return f"cannot be checked: {exc}"
    return None


@dataclass(frozen=True)
class DanglingHook:
    """One host event whose haunt hook command cannot run.

    This is the shape of the silent failure: the host config still lists the
    hook, so nothing looks missing, but the command behind it is gone.
    """

    host: str
    event: str
    command: str
    reason: str

    def __str__(self) -> str:
        return f"{self.host}  {self.event}  {self.command}  ({self.reason})"


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
    defect = hook_command_defect(command)
    if defect is not None:
        issues.append(f"hook command {defect}: {command}")
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
    # Every (event, command) whose hook cannot run -- one entry per event,
    # never deduplicated by command: the operator needs to see that all six
    # Claude Code events are dead, not that one path is bad.
    dangling_hooks: list[DanglingHook] = field(default_factory=list)


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Load a JSON object, or None if the file does not exist.

    Malformed or non-object JSON raises HostConfigError. Never invent {}.
    """
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HostConfigError(
            f"{path} is malformed JSON; leaving it unchanged"
        ) from exc
    if not isinstance(loaded, dict):
        raise HostConfigError(
            f"{path} is not a JSON object; leaving it unchanged"
        )
    return loaded


def write_json_atomic(path: Path, data: dict[str, Any], *, backup: bool = True) -> None:
    """Backup the previous file (if any), then replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        bak = path.with_name(path.name + ".bak")
        bak.write_bytes(path.read_bytes())
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _adapters() -> list:
    from haunt.hosts import claude, cursor

    return [cursor, claude]


def install_all_hosts(
    haunt_home: str, hook_cmd: str, mcp_cmd: str, *, force: bool = False
) -> list[HostReport]:
    """Bind every known host. Returns a report per host.

    Each adapter may compute host-specific launchers from haunt_home/bin/.
    hook_cmd is the default (Cursor) hook; adapters that need a different
    launcher (e.g. haunt-hook-claude) derive it from haunt_home.

    Raises AlternateHomeRefused before touching anything when haunt_home is
    not the default home and neither `force` nor ALT_HOME_ENV says otherwise.
    Checked here *and* inside each adapter.install(), so a caller that reaches
    an adapter directly (haunt cursor-install does) is guarded too.
    """
    check_host_install_allowed(haunt_home, force=force)
    reports: list[HostReport] = []
    for adapter in _adapters():
        report = adapter.install(haunt_home, hook_cmd, mcp_cmd, force=force)
        reports.append(report)
    return reports


def doctor_all_hosts(haunt_home: str, hook_cmd: str, mcp_cmd: str) -> list[HostStatus]:
    """Check every known host for correct bindings."""
    statuses: list[HostStatus] = []
    for adapter in _adapters():
        status = adapter.doctor(haunt_home, hook_cmd, mcp_cmd)
        statuses.append(status)
    return statuses
