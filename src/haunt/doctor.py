"""Read-only first-run checks. Every advertised check is executed and reported."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from haunt.bootstrap import probe_sqlite_vec
from haunt.embed import _env_model, _local_bge_m3_ready, _wants_bge_m3, fts_only
from haunt.hosts import HostStatus, doctor_all_hosts

REQUIRED_CHECKS = (
    "sqlite-vec",
    "haunt-mcp",
    "mcp-python",
    "embed",
    "cursor.hooks",
    "cursor.mcp",
    "cursor.rule",
    "cursor.skill",
    "claude-code.hooks",
    "claude-code.mcp",
    "claude-code.rule",
    "claude-code.skill",
)

_EXEC_RE = re.compile(r'^exec\s+"([^"]+)"')
_MCP_PROBE = (
    "import haunt, sqlite3, sqlite_vec\n"
    "c = sqlite3.connect(':memory:')\n"
    "c.enable_load_extension(True)\n"
    "sqlite_vec.load(c)\n"
    "print(c.execute('select vec_version()').fetchone()[0])\n"
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    path: str | None = None


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)
    hosts: list[HostStatus] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        names = {c.name for c in self.checks}
        if any(req not in names for req in REQUIRED_CHECKS):
            return False
        return all(c.ok for c in self.checks)

    @property
    def issues(self) -> list[str]:
        missing = [req for req in REQUIRED_CHECKS if req not in {c.name for c in self.checks}]
        skipped = [f"{name}: check was not run" for name in missing]
        failed = [f"{c.name}: {c.detail}" for c in self.checks if not c.ok]
        return skipped + failed

    @property
    def host_file_issues(self) -> bool:
        return any(
            (not c.ok) and c.name.startswith(("cursor.", "claude-code."))
            for c in self.checks
        )


def python_for_mcp_command(command: str) -> tuple[str | None, str | None]:
    """Resolve the Python that an MCP command will exec. Never searches PATH."""
    path = Path(command)
    if not path.is_file():
        return None, f"haunt-mcp wrapper not found: {command}"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, f"cannot read MCP command {command}: {exc}"

    exec_target: str | None = None
    for line in text.splitlines():
        match = _EXEC_RE.match(line.strip())
        if match:
            exec_target = match.group(1)
            break

    if exec_target is None:
        return _python_from_shebang(text, command)

    target = Path(exec_target)
    if not target.is_file():
        return None, f"MCP exec target not found: {exec_target}"
    if "python" in target.name:
        return str(target), None
    try:
        first = target.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    except OSError as exc:
        return None, f"cannot read MCP exec target {exec_target}: {exc}"
    if first:
        resolved, _err = _python_from_shebang(first[0], exec_target)
        if resolved:
            return resolved, None
    for sibling in ("python3", "python"):
        candidate = target.parent / sibling
        if candidate.is_file():
            return str(candidate), None
    return None, f"could not resolve python from MCP command {command}"


def _python_from_shebang(text: str, origin: str) -> tuple[str | None, str | None]:
    first = text.splitlines()[0] if text else ""
    if not first.startswith("#!"):
        return None, f"could not resolve python from MCP command {origin}"
    parts = first[2:].strip().split()
    if not parts:
        return None, f"could not resolve python from MCP command {origin}"
    if "python" in Path(parts[0]).name:
        return parts[0], None
    if parts[0].endswith("env") and len(parts) >= 2 and "python" in parts[1]:
        sibling = Path(origin).parent
        for name in ("python3", "python"):
            candidate = sibling / name
            if candidate.is_file():
                return str(candidate), None
    return None, f"could not resolve python from MCP command {origin}"


def _check_sqlite_vec() -> Check:
    vec = probe_sqlite_vec()
    if vec.get("ok"):
        return Check("sqlite-vec", True, f"ok {vec.get('version', '')}".strip())
    return Check(
        "sqlite-vec",
        False,
        "sqlite-vec failed to load: " + str(vec.get("error", "unknown")),
    )


def _check_wrapper(mcp_cmd: str) -> Check:
    path = Path(mcp_cmd)
    if not path.is_file():
        return Check("haunt-mcp", False, f"haunt-mcp wrapper not found: {mcp_cmd}", mcp_cmd)
    if not os.access(path, os.X_OK):
        return Check("haunt-mcp", False, f"haunt-mcp wrapper is not executable: {mcp_cmd}", mcp_cmd)
    return Check("haunt-mcp", True, "wrapper present", mcp_cmd)


def _check_mcp_python(mcp_cmd: str) -> Check:
    python, err = python_for_mcp_command(mcp_cmd)
    if err:
        return Check("mcp-python", False, err, mcp_cmd)
    try:
        proc = subprocess.run(
            [python, "-c", _MCP_PROBE],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check(
            "mcp-python",
            False,
            f"mcp command python cannot import haunt or sqlite-vec: {exc}",
            python,
        )
    if proc.returncode != 0:
        err_text = (proc.stderr or proc.stdout or "import failed").strip()
        last = err_text.splitlines()[-1] if err_text else "import failed"
        return Check(
            "mcp-python",
            False,
            f"mcp command python cannot import haunt or sqlite-vec: {last}",
            python,
        )
    ver = proc.stdout.strip() or "?"
    return Check(
        "mcp-python",
        True,
        f"import haunt + sqlite-vec ok via {python} (vec {ver})",
        python,
    )


def _check_embed() -> Check:
    if fts_only():
        return Check("embed", True, "FTS-only (explicit opt-in)")
    requested = _env_model()
    if _wants_bge_m3(requested):
        if _local_bge_m3_ready():
            return Check("embed", True, f"local {requested}")
        return Check(
            "embed",
            False,
            f"{requested} not present under models dir (run haunt bootstrap)",
        )
    return Check(
        "embed",
        False,
        f"embed model {requested} not present locally (run haunt bootstrap)",
    )


def _issue_matching(status: HostStatus, *needles: str) -> str | None:
    for issue in status.issues:
        low = issue.lower()
        if any(n.lower() in low for n in needles):
            return issue
    return None


def _host_checks(status: HostStatus) -> list[Check]:
    prefix = status.host
    hooks_issue = _issue_matching(
        status, "hooks.json", "settings.json", "hook missing", "hook entries"
    )
    mcp_issue = _issue_matching(
        status,
        "mcp.json",
        ".claude.json",
        "mcp command",
        "mcp server",
        "settings.json (silently",
    )
    rule_issue_text = _issue_matching(status, "rule", "haunt.mdc", "haunt.md")
    skill_issue_text = _issue_matching(status, "skill")
    leftover = [
        i
        for i in status.issues
        if i
        not in {hooks_issue, mcp_issue, rule_issue_text, skill_issue_text}
    ]
    if leftover and mcp_issue is None:
        mcp_issue = leftover[0]
        leftover = leftover[1:]
    checks = [
        Check(
            f"{prefix}.hooks",
            status.hooks_present and hooks_issue is None,
            hooks_issue or ("present" if status.hooks_present else "MISSING"),
            status.hooks_path,
        ),
        Check(
            f"{prefix}.mcp",
            status.mcp_present and mcp_issue is None,
            mcp_issue or ("present" if status.mcp_present else "MISSING"),
            status.mcp_path,
        ),
        Check(
            f"{prefix}.rule",
            status.rule_present and rule_issue_text is None,
            rule_issue_text or ("present" if status.rule_present else "MISSING"),
            status.rule_path,
        ),
        Check(
            f"{prefix}.skill",
            status.skill_present and skill_issue_text is None,
            skill_issue_text or ("present" if status.skill_present else "MISSING"),
            status.skill_path,
        ),
    ]
    for extra in leftover:
        checks.append(Check(f"{prefix}.extra", False, extra))
    return checks


def diagnose(haunt_home: str, hook_cmd: str, mcp_cmd: str) -> DoctorReport:
    """Run every advertised check. Does not write files or rematch hosts."""
    report = DoctorReport()
    report.checks.append(_check_sqlite_vec())
    report.checks.append(_check_wrapper(mcp_cmd))
    report.checks.append(_check_mcp_python(mcp_cmd))
    report.checks.append(_check_embed())

    report.hosts = doctor_all_hosts(haunt_home, hook_cmd, mcp_cmd)
    seen_hosts = {s.host for s in report.hosts}
    for required_host in ("cursor", "claude-code"):
        if required_host not in seen_hosts:
            report.checks.append(
                Check(f"{required_host}.hooks", False, "check was not run")
            )
            report.checks.append(
                Check(f"{required_host}.mcp", False, "check was not run")
            )
            report.checks.append(
                Check(f"{required_host}.rule", False, "check was not run")
            )
            report.checks.append(
                Check(f"{required_host}.skill", False, "check was not run")
            )
    for status in report.hosts:
        report.checks.extend(_host_checks(status))

    present = {c.name for c in report.checks}
    for name in REQUIRED_CHECKS:
        if name not in present:
            report.checks.append(Check(name, False, "check was not run"))
    return report


def format_doctor(report: DoctorReport) -> str:
    lines: list[str] = ["[runtime]"]
    runtime = {"sqlite-vec", "haunt-mcp", "mcp-python", "embed"}
    for check in report.checks:
        if check.name not in runtime:
            continue
        flag = "ok" if check.ok else "FAIL"
        lines.append(f"  {check.name:<12} {flag}  {check.detail}")

    by_host: dict[str, list[Check]] = {}
    extras: list[Check] = []
    for check in report.checks:
        if check.name in runtime:
            continue
        if "." in check.name:
            host, _kind = check.name.split(".", 1)
            by_host.setdefault(host, []).append(check)
        else:
            extras.append(check)

    for host, checks in by_host.items():
        host_ok = all(c.ok for c in checks)
        lines.append(f"[{host}]  {'ok' if host_ok else 'ISSUES'}")
        for check in checks:
            kind = check.name.split(".", 1)[1]
            flag = "present" if check.ok else "MISSING"
            path = f"  {check.path}" if check.path else ""
            if check.ok:
                lines.append(f"  {kind:<8} {flag}{path}")
            else:
                lines.append(f"  {kind:<8} FAIL  {check.detail}{path}")

    for check in extras:
        flag = "ok" if check.ok else "FAIL"
        lines.append(f"  {check.name:<12} {flag}  {check.detail}")

    if not report.ok:
        lines.append("")
        for issue in report.issues:
            lines.append(f"  ! {issue}")
    return "\n".join(lines)
