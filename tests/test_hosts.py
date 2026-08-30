"""Host bind tests: Cursor + Claude Code. Isolated temp dirs only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from haunt.hosts import doctor_all_hosts, hook_command_issues, install_all_hosts
from haunt.hosts.claude import HOOK_EVENTS as CLAUDE_EVENTS
from haunt.hosts.claude import HOST_NAME as CLAUDE_HOST
from haunt.hosts.cursor import HOOK_EVENTS as CURSOR_EVENTS
from haunt.hosts.cursor import HOST_NAME as CURSOR_HOST

MISSING_HOOK = "/tmp/does-not-exist/haunt-hook"
MISSING_CLAUDE_HOOK = "/tmp/does-not-exist/haunt-hook-claude"


@pytest.fixture
def host_env(tmp_path, monkeypatch, fake_home):
    """Isolated HAUNT_HOME + host dirs. FTS-only — never download BGE-M3."""
    haunt_home = fake_home / ".haunt"
    cursor_home = fake_home / ".cursor"
    claude_dir = fake_home / ".claude"
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


def _install(env):
    return install_all_hosts(str(env["haunt_home"]), env["hook_cmd"], env["mcp_cmd"])


def _doctor(env):
    return doctor_all_hosts(str(env["haunt_home"]), env["hook_cmd"], env["mcp_cmd"])


def _count_haunt_cursor_hooks(hooks: dict, event: str) -> int:
    entries = hooks.get(event) or []
    n = 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        name = str(item.get("command", "")).replace("\\", "/").split("/")[-1]
        if name == "haunt-hook":
            n += 1
    return n


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


def _count_haunt_claude_hooks(hooks: dict, event: str) -> int:
    n = 0
    for group in hooks.get(event) or []:
        if not isinstance(group, dict):
            continue
        for h in group.get("hooks") or []:
            if not isinstance(h, dict):
                continue
            name = str(h.get("command", "")).replace("\\", "/").split("/")[-1]
            if name in {"haunt-hook", "haunt-hook-claude"}:
                n += 1
    return n


def test_install_seeds_when_host_dirs_missing(host_env):
    env = host_env
    assert not env["cursor_home"].exists()
    assert not env["claude_dir"].exists()

    reports = _install(env)
    by_host = {r.host: r for r in reports}
    assert by_host[CURSOR_HOST].seeded is True
    assert by_host[CLAUDE_HOST].seeded is True

    cursor_hooks = json.loads((env["cursor_home"] / "hooks.json").read_text())
    for event in CURSOR_EVENTS:
        assert _count_haunt_cursor_hooks(cursor_hooks["hooks"], event) == 1

    cursor_mcp = json.loads((env["cursor_home"] / "mcp.json").read_text())
    assert cursor_mcp["mcpServers"]["haunt"]["command"] == env["mcp_cmd"]
    assert "type" not in cursor_mcp["mcpServers"]["haunt"]
    assert (env["cursor_home"] / "rules" / "haunt.mdc").is_file()
    cursor_skill = env["cursor_home"] / "skills" / "haunt" / "SKILL.md"
    assert cursor_skill.is_file()
    assert "memory_recall" in cursor_skill.read_text(encoding="utf-8")
    assert "verbatim" in cursor_skill.read_text(encoding="utf-8")

    settings = json.loads((env["claude_dir"] / "settings.json").read_text())
    for event in CLAUDE_EVENTS:
        assert _count_haunt_claude_hooks(settings["hooks"], event) == 1
        group = settings["hooks"][event][0]
        hook = group["hooks"][0]
        assert hook["type"] == "command"
        assert hook["command"] == env["claude_hook"]

    claude_mcp = json.loads((env["claude_dir"] / ".claude.json").read_text())
    haunt = claude_mcp["mcpServers"]["haunt"]
    assert haunt["command"] == env["mcp_cmd"]
    assert haunt["type"] == "stdio"
    assert (env["claude_dir"] / "rules" / "haunt.md").is_file()
    claude_skill = env["claude_dir"] / "skills" / "haunt" / "SKILL.md"
    assert claude_skill.is_file()
    assert "memory_recall" in claude_skill.read_text(encoding="utf-8")
    assert "mcpServers" not in settings


def test_foreign_hooks_and_mcp_survive(host_env):
    env = host_env
    env["cursor_home"].mkdir(parents=True)
    (env["cursor_home"] / "hooks.json").write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "afterFileEdit": [{"command": "./hooks/format.sh"}],
                    "beforeSubmitPrompt": [{"command": "./hooks/audit.sh"}],
                },
            }
        ),
        encoding="utf-8",
    )
    (env["cursor_home"] / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "ironrecall": {"command": "/opt/ironrecall/mcp"},
                }
            }
        ),
        encoding="utf-8",
    )

    env["claude_dir"].mkdir(parents=True)
    (env["claude_dir"] / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash"]},
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/usr/local/bin/other-hook",
                                }
                            ]
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (env["claude_dir"] / ".claude.json").write_text(
        json.dumps(
            {
                "theme": "dark",
                "mcpServers": {
                    "filesystem": {"command": "npx", "args": ["-y", "fs"]},
                },
            }
        ),
        encoding="utf-8",
    )
    claude_md = env["claude_dir"] / "CLAUDE.md"
    claude_md.write_text("# user CLAUDE.md — do not overwrite\n", encoding="utf-8")

    reports = _install(env)
    by_host = {r.host: r for r in reports}
    assert by_host[CURSOR_HOST].seeded is False
    assert by_host[CLAUDE_HOST].seeded is False

    cursor_hooks = json.loads((env["cursor_home"] / "hooks.json").read_text())
    assert cursor_hooks["hooks"]["afterFileEdit"] == [{"command": "./hooks/format.sh"}]
    prompts = cursor_hooks["hooks"]["beforeSubmitPrompt"]
    assert any("audit.sh" in c["command"] for c in prompts)
    assert any("haunt-hook" in c["command"] for c in prompts)

    cursor_mcp = json.loads((env["cursor_home"] / "mcp.json").read_text())
    assert cursor_mcp["mcpServers"]["ironrecall"]["command"] == "/opt/ironrecall/mcp"
    assert cursor_mcp["mcpServers"]["haunt"]["command"] == env["mcp_cmd"]

    settings = json.loads((env["claude_dir"] / "settings.json").read_text())
    assert settings["permissions"] == {"allow": ["Bash"]}
    groups = settings["hooks"]["UserPromptSubmit"]
    commands = [h["command"] for g in groups for h in g["hooks"]]
    assert "/usr/local/bin/other-hook" in commands
    assert env["claude_hook"] in commands

    claude_mcp = json.loads((env["claude_dir"] / ".claude.json").read_text())
    assert claude_mcp["theme"] == "dark"
    assert claude_mcp["mcpServers"]["filesystem"]["command"] == "npx"
    assert claude_mcp["mcpServers"]["haunt"]["command"] == env["mcp_cmd"]
    assert claude_md.read_text(encoding="utf-8") == "# user CLAUDE.md — do not overwrite\n"


def test_second_install_idempotent(host_env):
    env = host_env
    _install(env)
    first_cursor = (env["cursor_home"] / "hooks.json").read_text()
    first_mcp = (env["cursor_home"] / "mcp.json").read_text()
    first_settings = (env["claude_dir"] / "settings.json").read_text()
    first_dot = (env["claude_dir"] / ".claude.json").read_text()

    reports = _install(env)
    assert all(r.seeded is False for r in reports)

    cursor_hooks = json.loads((env["cursor_home"] / "hooks.json").read_text())
    for event in CURSOR_EVENTS:
        assert _count_haunt_cursor_hooks(cursor_hooks["hooks"], event) == 1

    settings = json.loads((env["claude_dir"] / "settings.json").read_text())
    for event in CLAUDE_EVENTS:
        assert _count_haunt_claude_hooks(settings["hooks"], event) == 1

    assert (env["cursor_home"] / "hooks.json").read_text() == first_cursor
    assert (env["cursor_home"] / "mcp.json").read_text() == first_mcp
    assert (env["claude_dir"] / "settings.json").read_text() == first_settings
    assert (env["claude_dir"] / ".claude.json").read_text() == first_dot


def test_claude_mcp_in_dotfile_not_settings(host_env):
    env = host_env
    _install(env)
    settings = json.loads((env["claude_dir"] / "settings.json").read_text())
    dot = json.loads((env["claude_dir"] / ".claude.json").read_text())
    assert "mcpServers" not in settings
    assert "haunt" in dot["mcpServers"]
    assert dot["mcpServers"]["haunt"]["command"] == env["mcp_cmd"]


def test_mutation_fails_if_claude_mcp_only_in_settings(host_env):
    """Doctor must NOT treat settings.json mcpServers as a valid Claude MCP bind.

    Claude Code silently ignores mcpServers in settings.json. If this test
    ever passes while mcp_present is True, the adapter is wrong.
    """
    env = host_env
    env["claude_dir"].mkdir(parents=True)
    (env["claude_dir"] / "settings.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "haunt": {
                        "command": env["mcp_cmd"],
                        "type": "stdio",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    statuses = {s.host: s for s in _doctor(env)}
    claude = statuses[CLAUDE_HOST]
    assert claude.mcp_present is False, (
        "mcp_present must be False when haunt MCP exists only in settings.json"
    )
    assert any("settings.json" in i for i in claude.issues)
    assert any(".claude.json" in i for i in claude.issues)


def test_cursor_mcp_in_mcp_json(host_env):
    env = host_env
    _install(env)
    mcp_path = env["cursor_home"] / "mcp.json"
    data = json.loads(mcp_path.read_text())
    assert data["mcpServers"]["haunt"]["command"] == env["mcp_cmd"]
    statuses = {s.host: s for s in _doctor(env)}
    assert statuses[CURSOR_HOST].mcp_present is True
    assert statuses[CURSOR_HOST].mcp_path == str(mcp_path)


def test_doctor_remerges_when_missing(host_env):
    env = host_env
    _install(env)
    (env["cursor_home"] / "hooks.json").unlink()
    (env["claude_dir"] / ".claude.json").unlink()

    statuses = {s.host: s for s in _doctor(env)}
    assert statuses[CURSOR_HOST].issues
    assert statuses[CLAUDE_HOST].issues

    from haunt.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Re-merging" in result.stdout
    assert (env["cursor_home"] / "hooks.json").is_file()
    assert (env["claude_dir"] / ".claude.json").is_file()
    assert "haunt" in json.loads((env["claude_dir"] / ".claude.json").read_text())["mcpServers"]


def test_cli_install_prints_seeded_vs_merged(host_env):
    from haunt.cli import app

    runner = CliRunner()
    first = runner.invoke(app, ["install"])
    assert first.exit_code == 0
    assert "[cursor]  seeded" in first.stdout
    assert "[claude-code]  seeded" in first.stdout

    second = runner.invoke(app, ["install"])
    assert second.exit_code == 0
    assert "[cursor]  merged" in second.stdout
    assert "[claude-code]  merged" in second.stdout


def test_cursor_install_alias_writes_mcp(host_env):
    from haunt.cursor_hook import install_cursor_hooks

    report = install_cursor_hooks()
    mcp_path = Path(report["mcp_json"])
    assert mcp_path.name == "mcp.json"
    data = json.loads(mcp_path.read_text())
    assert data["mcpServers"]["haunt"]["command"] == host_env["mcp_cmd"]
    # Cursor-only: Claude files should not be required by this alias.
    # (autouse isolation may have an empty claude-config dir; do not bind it.)
    assert not (host_env["claude_dir"] / "settings.json").exists()


def test_hook_command_issues_missing_path_fails_when_expected_matches():
    """#57 sabotage: revert path.is_file() in hook_command_issues and this fails.

    Planted command == expected, so samefile-when-different is skipped.
    Leaf name alone used to paint this ok.
    """
    assert not Path(MISSING_HOOK).exists()
    issues = hook_command_issues(MISSING_HOOK, expected=MISSING_HOOK)
    assert issues, "leaf-named missing haunt-hook must be a doctor issue"
    assert any("not found" in i for i in issues)
    assert all("MCP" not in i for i in issues)


def test_hook_command_issues_real_wrapper_passes(host_env):
    from haunt.bootstrap import bind_launchers

    bind_launchers()
    issues = hook_command_issues(host_env["hook_cmd"], expected=host_env["hook_cmd"])
    assert issues == []
    claude_issues = hook_command_issues(
        host_env["claude_hook"], expected=host_env["claude_hook"]
    )
    assert claude_issues == []


def test_doctor_fails_if_hook_command_is_missing_leaf_named_path(host_env):
    """Plant /tmp/does-not-exist/haunt-hook. cursor.hooks and claude-code.hooks FAIL.

    hook_cmd is the same missing path so samefile-when-different is skipped.
    Revert the path-exists check and this test fails.
    """
    env = host_env
    _install(env)
    assert not Path(MISSING_HOOK).exists()
    assert not Path(MISSING_CLAUDE_HOOK).exists()
    _plant_cursor_hook_command(env, MISSING_HOOK)
    _plant_claude_hook_command(env, MISSING_CLAUDE_HOOK)

    statuses = {
        s.host: s
        for s in doctor_all_hosts(str(env["haunt_home"]), MISSING_HOOK, env["mcp_cmd"])
    }
    cursor = statuses[CURSOR_HOST]
    claude = statuses[CLAUDE_HOST]
    assert cursor.hooks_present is False
    assert claude.hooks_present is False
    assert any("not found" in i for i in cursor.issues)
    assert any("not found" in i for i in claude.issues)
    assert cursor.mcp_present is True
    assert claude.mcp_present is True


def test_doctor_ok_for_real_expected_hook_wrapper(host_env):
    env = host_env
    from haunt.bootstrap import bind_launchers

    bind_launchers()
    _install(env)
    statuses = {s.host: s for s in _doctor(env)}
    assert statuses[CURSOR_HOST].hooks_present is True
    assert statuses[CLAUDE_HOST].hooks_present is True
    assert statuses[CURSOR_HOST].mcp_present is True
    assert statuses[CLAUDE_HOST].mcp_present is True
