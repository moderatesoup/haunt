"""Install + doctor first-run checks. Isolated temp dirs only — no model, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from haunt.doctor import REQUIRED_CHECKS, diagnose
from haunt.hosts.claude import HOOK_EVENTS as CLAUDE_EVENTS
from haunt.hosts.cursor import HOOK_EVENTS as CURSOR_EVENTS

# Live-tool fact from #44. Install must plant this, not the pre-rewrite essay.
AUTO_COMPILE_PHRASE = "compile() runs automatically on memory_recall"


@pytest.fixture
def onboard_env(tmp_path, monkeypatch):
    """Isolated HOME / HAUNT_HOME / Cursor + Claude dirs. FTS-only."""
    haunt_home = tmp_path / "haunthome"
    cursor_home = tmp_path / "cursor"
    claude_dir = tmp_path / "claude-config"
    monkeypatch.setenv("HAUNT_HOME", str(haunt_home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("CURSOR_HOME", str(cursor_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    monkeypatch.delenv("CURSOR_HOOKS_JSON", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield {
        "haunt_home": haunt_home,
        "cursor_home": cursor_home,
        "claude_dir": claude_dir,
        "hook_cmd": str(haunt_home / "bin" / "haunt-hook"),
        "mcp_cmd": str(haunt_home / "bin" / "haunt-mcp"),
        "claude_hook": str(haunt_home / "bin" / "haunt-hook-claude"),
    }
    embed.reset()


def _install(env) -> None:
    from haunt.cli import app

    result = CliRunner().invoke(app, ["install"])
    assert result.exit_code == 0, result.stdout + result.stderr


def _diagnose(env):
    return diagnose(str(env["haunt_home"]), env["hook_cmd"], env["mcp_cmd"])


def test_install_writes_host_files_doctor_expects(onboard_env):
    env = onboard_env
    _install(env)

    wrapper = Path(env["mcp_cmd"])
    assert wrapper.is_file()
    body = wrapper.read_text(encoding="utf-8")
    assert "haunt.mcp_server" in body or "haunt-mcp" in body
    assert "PATH" not in body.split("exec", 1)[-1]

    cursor_hooks = json.loads((env["cursor_home"] / "hooks.json").read_text())
    for event in CURSOR_EVENTS:
        commands = [
            item.get("command", "")
            for item in cursor_hooks["hooks"][event]
            if isinstance(item, dict)
        ]
        assert any(str(c).endswith("haunt-hook") for c in commands), event

    cursor_mcp = json.loads((env["cursor_home"] / "mcp.json").read_text())
    haunt = cursor_mcp["mcpServers"]["haunt"]
    assert haunt["command"] == env["mcp_cmd"]
    assert Path(haunt["command"]).is_absolute()
    assert Path(haunt["command"]).name == "haunt-mcp"

    cursor_rule = (env["cursor_home"] / "rules" / "haunt.mdc").read_text(encoding="utf-8")
    assert "memory_recall" in cursor_rule
    assert "alwaysApply: true" in cursor_rule
    assert AUTO_COMPILE_PHRASE in cursor_rule, (
        "planted Cursor rule must state the #44 temporal fact, not the old essay"
    )
    cursor_skill = (env["cursor_home"] / "skills" / "haunt" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "memory_recall" in cursor_skill
    assert "verbatim" in cursor_skill
    assert AUTO_COMPILE_PHRASE in cursor_skill, (
        "planted Cursor skill must state the #44 temporal fact, not the old essay"
    )

    settings = json.loads((env["claude_dir"] / "settings.json").read_text())
    for event in CLAUDE_EVENTS:
        group = settings["hooks"][event][0]
        hook = group["hooks"][0]
        assert hook["type"] == "command"
        assert hook["command"] == env["claude_hook"]
    assert "mcpServers" not in settings

    claude_mcp = json.loads((env["claude_dir"] / ".claude.json").read_text())
    haunt = claude_mcp["mcpServers"]["haunt"]
    assert haunt["command"] == env["mcp_cmd"]
    assert haunt["type"] == "stdio"
    claude_rule = (env["claude_dir"] / "rules" / "haunt.md").read_text(encoding="utf-8")
    assert "memory_recall" in claude_rule
    assert AUTO_COMPILE_PHRASE in claude_rule
    claude_skill = (env["claude_dir"] / "skills" / "haunt" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "memory_recall" in claude_skill
    assert AUTO_COMPILE_PHRASE in claude_skill


def test_install_plants_auto_compile_phrase(onboard_env):
    """Regression lock for #44: isolated install must plant the live temporal fact."""
    env = onboard_env
    _install(env)
    planted = [
        env["cursor_home"] / "rules" / "haunt.mdc",
        env["cursor_home"] / "skills" / "haunt" / "SKILL.md",
        env["claude_dir"] / "rules" / "haunt.md",
        env["claude_dir"] / "skills" / "haunt" / "SKILL.md",
    ]
    for path in planted:
        text = path.read_text(encoding="utf-8")
        assert AUTO_COMPILE_PHRASE in text, f"missing {AUTO_COMPILE_PHRASE!r} in {path}"


def test_doctor_ok_after_install_runs_every_check(onboard_env):
    env = onboard_env
    _install(env)
    report = _diagnose(env)
    names = [c.name for c in report.checks]
    for required in REQUIRED_CHECKS:
        assert required in names, f"doctor skipped {required}"
    assert report.ok, report.issues
    embed = next(c for c in report.checks if c.name == "embed")
    assert "FTS-only" in embed.detail


def _plant_cursor_hook_command(env, command: str) -> None:
    path = env["cursor_home"] / "hooks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for entries in data.get("hooks", {}).values():
        if not isinstance(entries, list):
            continue
        for item in entries:
            if isinstance(item, dict) and str(item.get("command", "")).endswith(
                "haunt-hook"
            ):
                item["command"] = command
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _plant_claude_hook_command(env, command: str) -> None:
    path = env["claude_dir"] / "settings.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for groups in data.get("hooks", {}).values():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            for item in group.get("hooks") or []:
                if isinstance(item, dict) and "haunt-hook" in str(
                    item.get("command", "")
                ):
                    item["command"] = command
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_doctor_fails_if_hook_command_is_missing_leaf_named_path(onboard_env):
    """#57: leaf-named missing haunt-hook must FAIL diagnose(), not paint ok.

    hook_cmd equals the planted missing path so samefile-when-different is
    skipped. Revert the path-exists check in hook_command_issues and this fails.
    MCP checks must stay ok — do not weaken MCP doctor honesty.
    """
    env = onboard_env
    _install(env)
    missing_hook = "/tmp/does-not-exist/haunt-hook"
    missing_claude = "/tmp/does-not-exist/haunt-hook-claude"
    assert not Path(missing_hook).exists()
    assert not Path(missing_claude).exists()
    _plant_cursor_hook_command(env, missing_hook)
    _plant_claude_hook_command(env, missing_claude)

    report = diagnose(str(env["haunt_home"]), missing_hook, env["mcp_cmd"])
    by_name = {c.name: c for c in report.checks}
    assert by_name["cursor.hooks"].ok is False
    assert by_name["claude-code.hooks"].ok is False
    assert report.ok is False
    blob = " ".join(report.issues)
    assert "not found" in blob
    assert "does-not-exist" in blob
    assert by_name["cursor.mcp"].ok is True
    assert by_name["claude-code.mcp"].ok is True


def test_doctor_fails_if_hooks_json_deleted(onboard_env):
    env = onboard_env
    _install(env)
    (env["cursor_home"] / "hooks.json").unlink()
    report = _diagnose(env)
    assert report.ok is False
    assert any("hooks.json" in i for i in report.issues)


def test_install_plants_haunt_named_binaries_only(onboard_env):
    env = onboard_env
    _install(env)
    bin_dir = env["haunt_home"] / "bin"
    names = {p.name for p in bin_dir.iterdir()}
    leftover = {n for n in names if n.startswith(("lore", "engram"))}
    assert leftover == set(), leftover
    assert {"haunt-mcp", "haunt-hook", "haunt-hook-claude"} <= names
    cursor_mcp = json.loads((env["cursor_home"] / "mcp.json").read_text(encoding="utf-8"))
    assert Path(cursor_mcp["mcpServers"]["haunt"]["command"]).name == "haunt-mcp"


def test_doctor_fails_if_mcp_is_lore_mcp(onboard_env):
    env = onboard_env
    _install(env)
    mcp_path = env["cursor_home"] / "mcp.json"
    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    data["mcpServers"]["haunt"]["command"] = str(env["haunt_home"] / "bin" / "lore-mcp")
    mcp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    report = _diagnose(env)
    assert report.ok is False
    blob = " ".join(report.issues)
    assert "lore-mcp" in blob
    assert "not haunt-mcp" in blob


def test_doctor_fails_if_mcp_points_at_bin_false(onboard_env):
    env = onboard_env
    _install(env)
    mcp_path = env["cursor_home"] / "mcp.json"
    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    data["mcpServers"]["haunt"]["command"] = "/bin/false"
    mcp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    report = _diagnose(env)
    assert report.ok is False
    blob = " ".join(report.issues)
    assert "/bin/false" in blob
    assert "not haunt-mcp" in blob


def test_doctor_fails_if_skill_deleted(onboard_env):
    env = onboard_env
    _install(env)
    (env["cursor_home"] / "skills" / "haunt" / "SKILL.md").unlink()
    report = _diagnose(env)
    assert report.ok is False
    assert any("skill" in i.lower() for i in report.issues)


def test_doctor_fails_if_wrapper_python_cannot_import_haunt(onboard_env, tmp_path):
    env = onboard_env
    _install(env)
    fake_py = tmp_path / "dead-python"
    fake_py.write_text(
        "#!/bin/sh\n"
        "echo 'ModuleNotFoundError: No module named \\'haunt\\'' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)
    wrapper = Path(env["mcp_cmd"])
    wrapper.write_text(
        "#!/bin/sh\n"
        f'exec "{fake_py}" -m haunt.mcp_server "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    report = _diagnose(env)
    assert report.ok is False
    blob = " ".join(report.issues)
    assert "cannot import haunt or sqlite-vec" in blob


def test_bootstrap_vec_fail_writes_no_namespace(tmp_path, monkeypatch):
    from unittest.mock import patch

    from haunt import embed
    from haunt.bootstrap import BootstrapError, bootstrap
    from haunt.paths import namespace_db_path, registry_path

    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "fresh-home"))
    monkeypatch.delenv("HAUNT_FTS_ONLY", raising=False)
    monkeypatch.delenv("HAUNT_EMBED_MODEL", raising=False)
    embed.reset()
    with patch(
        "haunt.bootstrap.probe_sqlite_vec",
        return_value={"ok": False, "error": "mocked: extension load failed"},
    ):
        with pytest.raises(BootstrapError) as exc:
            bootstrap()
        assert "sqlite-vec" in exc.value.message.lower()

    assert not registry_path().exists()
    assert not namespace_db_path("default").exists()
    embed.reset()


def test_cli_doctor_sqlite_vec_not_fatal_when_fts_only(onboard_env):
    """#64: HAUNT_FTS_ONLY=1 must not fail doctor on a failed vec probe."""
    from unittest.mock import patch

    from haunt.cli import app

    _install(onboard_env)
    with patch(
        "haunt.doctor.probe_sqlite_vec",
        return_value={"ok": False, "error": "mocked: extension load failed"},
    ):
        result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "sqlite-vec" in result.stdout
    assert "FTS-only" in result.stdout
    sqlite_lines = [
        line for line in result.stdout.splitlines() if "sqlite-vec" in line
    ]
    assert sqlite_lines, result.stdout
    assert all("FAIL" not in line for line in sqlite_lines), result.stdout
