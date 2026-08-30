"""Two guards for the global-host-config incident. Fully sandboxed homes only.

The incident: a smoke test ran with HAUNT_HOME pointed at a worktree's
`.smoke-home`. install_all_hosts() wrote that temporary path into the real
global ~/.claude/settings.json for all six hook events. The worktree was
deleted later; every hook command then pointed at nothing, and memory capture
stopped silently for three days. Nothing in `doctor` said so.

Guard 1 refuses to write global host config from a non-default HAUNT_HOME.
Guard 2 makes `doctor` name every host event whose hook command cannot run.

Every test here redirects HOME as well as HAUNT_HOME, CURSOR_HOME and
CLAUDE_CONFIG_DIR, and asserts the HOME redirect took effect before writing
anything: these are the exact code paths that damaged the owner's machine.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from haunt.doctor import diagnose, format_doctor
from haunt.hosts import (
    ALT_HOME_ENV,
    AlternateHomeRefused,
    default_haunt_home,
    doctor_all_hosts,
    host_install_refusal,
    install_all_hosts,
)
from haunt.hosts.claude import HOOK_EVENTS as CLAUDE_EVENTS
from haunt.hosts.claude import HOST_NAME as CLAUDE_HOST
from haunt.hosts.cursor import HOOK_EVENTS as CURSOR_EVENTS
from haunt.hosts.cursor import HOST_NAME as CURSOR_HOST


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A fake HOME plus redirected host dirs, with the guard switched ON.

    HOME is redirected so `default_haunt_home()` is a temp path: without it
    "the default home still installs" could only be tested by writing to the
    operator's actual ~/.haunt and ~/.claude, which is the incident.
    """
    fake_home = tmp_path / "home"
    (fake_home / ".haunt").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    # Fail loudly rather than fall through to the real home.
    assert Path.home().resolve() == fake_home.resolve()
    assert default_haunt_home() == (fake_home / ".haunt").resolve()

    cursor_home = fake_home / ".cursor"
    claude_dir = fake_home / ".claude"
    monkeypatch.setenv("CURSOR_HOME", str(cursor_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("CURSOR_HOOKS_JSON", raising=False)
    # conftest no longer disables the guard suite-wide; belt and braces.
    monkeypatch.delenv(ALT_HOME_ENV, raising=False)

    return {
        "tmp": tmp_path,
        "fake_home": fake_home,
        "default_home": fake_home / ".haunt",
        "alt_home": tmp_path / "worktree" / ".smoke-home",
        "cursor_home": cursor_home,
        "claude_dir": claude_dir,
    }


def _use_home(monkeypatch, home: Path) -> dict[str, str]:
    """Point HAUNT_HOME at `home`, lay it out, and return its commands."""
    from haunt.bootstrap import bind_launchers

    monkeypatch.setenv("HAUNT_HOME", str(home))
    _home, hook_cmd, mcp_cmd = bind_launchers()
    return {
        "home": str(home),
        "hook_cmd": hook_cmd,
        "mcp_cmd": mcp_cmd,
        "claude_hook": str(Path(hook_cmd).parent / "haunt-hook-claude"),
    }


def _host_files(sandbox) -> list[Path]:
    return [
        sandbox["cursor_home"] / "hooks.json",
        sandbox["cursor_home"] / "mcp.json",
        sandbox["claude_dir"] / "settings.json",
        sandbox["claude_dir"] / ".claude.json",
    ]


def _assert_no_host_config(sandbox) -> None:
    for path in _host_files(sandbox):
        assert not path.exists(), f"guard let {path} be written"


# ---------------------------------------------------------------------------
# Guard 1 — refuse to bind global host config from a non-default HAUNT_HOME
# ---------------------------------------------------------------------------


def test_alternate_home_does_not_write_host_config(sandbox, monkeypatch):
    """The incident, blocked. Revert check_host_install_allowed() and this fails."""
    env = _use_home(monkeypatch, sandbox["alt_home"])

    with pytest.raises(AlternateHomeRefused) as excinfo:
        install_all_hosts(env["home"], env["hook_cmd"], env["mcp_cmd"])

    message = str(excinfo.value)
    assert str(sandbox["alt_home"].resolve()) in message
    assert str(sandbox["default_home"].resolve()) in message
    assert ALT_HOME_ENV in message
    _assert_no_host_config(sandbox)


def test_default_home_still_writes_host_config(sandbox, monkeypatch):
    """The normal path is untouched: a plain bind from ~/.haunt still installs."""
    env = _use_home(monkeypatch, sandbox["default_home"])

    reports = install_all_hosts(env["home"], env["hook_cmd"], env["mcp_cmd"])

    assert {r.host for r in reports} == {CURSOR_HOST, CLAUDE_HOST}
    for path in _host_files(sandbox):
        assert path.exists(), f"default home failed to write {path}"

    settings = json.loads((sandbox["claude_dir"] / "settings.json").read_text())
    assert set(settings["hooks"]) >= set(CLAUDE_EVENTS)
    hooks = json.loads((sandbox["cursor_home"] / "hooks.json").read_text())
    assert set(hooks["hooks"]) >= set(CURSOR_EVENTS)


def test_symlinked_alternate_home_is_still_refused(sandbox, monkeypatch):
    """A non-default home reached through a symlink is refused, and only that.

    Both directions matter and both are checked here. Revert the .resolve()
    in host_install_refusal() to a plain .absolute() and the second half
    fails: a link that genuinely points at ~/.haunt stops being recognised as
    the default home, and a legitimate install starts being refused.
    """
    real_alt = sandbox["tmp"] / "elsewhere" / ".smoke-home"
    real_alt.mkdir(parents=True)
    link = sandbox["fake_home"] / ".haunt-link"
    link.symlink_to(real_alt, target_is_directory=True)

    env = _use_home(monkeypatch, link)
    with pytest.raises(AlternateHomeRefused):
        install_all_hosts(env["home"], env["hook_cmd"], env["mcp_cmd"])
    _assert_no_host_config(sandbox)

    # A symlink that resolves to the real default home is not an alternate home.
    default_link = sandbox["tmp"] / "default-link"
    default_link.symlink_to(sandbox["default_home"], target_is_directory=True)
    assert host_install_refusal(default_link) is None


def test_force_flag_and_env_var_re_enable_the_install(sandbox, monkeypatch):
    """A legitimate alternate-home bind is still possible, two explicit ways."""
    env = _use_home(monkeypatch, sandbox["alt_home"])

    reports = install_all_hosts(
        env["home"], env["hook_cmd"], env["mcp_cmd"], force=True
    )
    assert {r.host for r in reports} == {CURSOR_HOST, CLAUDE_HOST}
    for path in _host_files(sandbox):
        assert path.exists()

    monkeypatch.setenv(ALT_HOME_ENV, "1")
    assert host_install_refusal(sandbox["alt_home"]) is None


@pytest.mark.parametrize("value", ["", " ", "0", "true", "yes", "on", "TRUE", "2"])
def test_only_an_exact_one_counts_as_consent(sandbox, monkeypatch, value):
    """The escape hatch must be impossible to trip by accident.

    Revert the `== "1"` in alt_home_install_allowed() to a truthiness test and
    the "0"/"" cases here fail.
    """
    monkeypatch.setenv(ALT_HOME_ENV, value)
    env = _use_home(monkeypatch, sandbox["alt_home"])

    with pytest.raises(AlternateHomeRefused):
        install_all_hosts(env["home"], env["hook_cmd"], env["mcp_cmd"])
    _assert_no_host_config(sandbox)


def test_cursor_install_alias_is_guarded_too(sandbox, monkeypatch):
    """`haunt cursor-install` reaches the adapter directly; it is guarded there."""
    _use_home(monkeypatch, sandbox["alt_home"])
    from haunt.cursor_hook import install_cursor_hooks

    with pytest.raises(AlternateHomeRefused):
        install_cursor_hooks()
    _assert_no_host_config(sandbox)


def test_bootstrap_skips_hosts_loudly_and_still_does_its_other_work(
    sandbox, monkeypatch
):
    """bootstrap() skips rather than aborts, and says so in the report."""
    from haunt.bootstrap import bootstrap, format_report

    monkeypatch.setenv("HAUNT_HOME", str(sandbox["alt_home"]))
    from haunt import embed

    embed.reset()
    report = bootstrap("default")
    embed.reset()

    assert report["hosts"] == []
    assert report["hosts_skipped"]
    assert str(sandbox["alt_home"].resolve()) in report["hosts_skipped"]
    _assert_no_host_config(sandbox)

    # The rest of bootstrap still ran.
    assert Path(report["launcher"]).is_file()
    assert Path(report["default_db"]).exists()

    text = format_report(report)
    assert "HOST BIND SKIPPED" in text
    assert ALT_HOME_ENV in text
    # And it must not claim the desktop icon was skipped for the wrong reason.
    assert "unsupported platform" not in text


def test_bootstrap_from_the_default_home_still_binds_hosts(sandbox, monkeypatch):
    """Guard 1 must not break the normal path: plain bootstrap still installs."""
    from haunt.bootstrap import bootstrap

    monkeypatch.setenv("HAUNT_HOME", str(sandbox["default_home"]))
    from haunt import embed

    embed.reset()
    report = bootstrap("default")
    embed.reset()

    assert report["hosts_skipped"] is None
    assert {h["host"] for h in report["hosts"]} == {CURSOR_HOST, CLAUDE_HOST}
    for path in _host_files(sandbox):
        assert path.exists()


def test_desktop_icon_is_skipped_from_an_alternate_home(sandbox, monkeypatch):
    """A temp-home run does not drop a shortcut in the operator's real home."""
    from haunt.desktop import install_desktop_icon

    monkeypatch.setenv("HAUNT_HOME", str(sandbox["alt_home"]))
    result = install_desktop_icon()
    assert result["written"] is False
    assert ALT_HOME_ENV in result["reason"]
    assert not any(sandbox["fake_home"].glob("**/Haunt Memories.*"))

    # Forced, and with an explicit (already redirected) home, it still writes.
    assert install_desktop_icon(force=True)["written"] is True
    assert install_desktop_icon(sandbox["tmp"] / "explicit")["written"] is True


def test_haunt_install_exits_nonzero_instead_of_doing_nothing(sandbox, monkeypatch):
    """`haunt install` binds hosts or fails; it never succeeds silently."""
    from typer.testing import CliRunner

    from haunt.cli import app

    monkeypatch.setenv("HAUNT_HOME", str(sandbox["alt_home"]))
    result = CliRunner().invoke(app, ["install"])
    assert result.exit_code != 0
    assert ALT_HOME_ENV in (result.stdout + result.stderr)
    _assert_no_host_config(sandbox)

    forced = CliRunner().invoke(app, ["install", "--allow-alt-home"])
    assert forced.exit_code == 0, forced.stdout
    for path in _host_files(sandbox):
        assert path.exists()


def test_doctor_refuses_its_own_repair_merge_from_an_alternate_home(
    sandbox, monkeypatch
):
    """doctor's auto re-merge is a host install, so the guard covers it too.

    doctor is the likelier of the two commands to be run from a smoke-test
    shell, and its repair step writes the same global files bootstrap does.
    """
    from typer.testing import CliRunner

    from haunt.cli import app

    monkeypatch.setenv("HAUNT_HOME", str(sandbox["alt_home"]))
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "NOT re-merging hosts" in result.stdout
    assert "Re-merging all hosts" not in result.stdout
    _assert_no_host_config(sandbox)


# ---------------------------------------------------------------------------
# Guard 2 — doctor names every host event whose hook command cannot run
# ---------------------------------------------------------------------------


def _plant_cursor_command(sandbox, command: str) -> None:
    path = sandbox["cursor_home"] / "hooks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for entries in data.get("hooks", {}).values():
        for item in entries:
            if isinstance(item, dict) and "haunt-hook" in str(item.get("command", "")):
                item["command"] = command
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _plant_claude_command(sandbox, command: str) -> None:
    path = sandbox["claude_dir"] / "settings.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for groups in data.get("hooks", {}).values():
        for group in groups:
            for item in group.get("hooks") or []:
                if isinstance(item, dict) and "haunt-hook" in str(
                    item.get("command", "")
                ):
                    item["command"] = command
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_doctor_names_host_event_and_path_for_a_deleted_hook(sandbox, monkeypatch):
    """A hook command pointing at a deleted file is named per host and event.

    Revert the per-event DanglingHook append in claude.py/cursor.py (or
    hook_command_defect) and this fails: the old report deduplicated by
    command, so one line stood in for every dead event and named none of them.
    """
    env = _use_home(monkeypatch, sandbox["default_home"])
    install_all_hosts(env["home"], env["hook_cmd"], env["mcp_cmd"])

    gone = sandbox["tmp"] / "deleted-worktree" / ".smoke-home" / "bin"
    gone.mkdir(parents=True)
    cursor_gone = gone / "haunt-hook"
    claude_gone = gone / "haunt-hook-claude"
    cursor_gone.write_text("#!/bin/sh\n")
    claude_gone.write_text("#!/bin/sh\n")
    _plant_cursor_command(sandbox, str(cursor_gone))
    _plant_claude_command(sandbox, str(claude_gone))
    shutil.rmtree(sandbox["tmp"] / "deleted-worktree")

    report = diagnose(env["home"], env["hook_cmd"], env["mcp_cmd"])

    dangling = report.dangling_hooks
    assert dangling, "doctor did not notice a hook command that cannot run"
    for host, events, path in (
        (CURSOR_HOST, CURSOR_EVENTS, str(cursor_gone)),
        (CLAUDE_HOST, CLAUDE_EVENTS, str(claude_gone)),
    ):
        found = {d.event for d in dangling if d.host == host and d.command == path}
        assert found == set(events), f"{host}: expected {set(events)}, got {found}"
    assert {d.reason for d in dangling} == {"not found"}

    text = format_doctor(report)
    for event in tuple(CURSOR_EVENTS) + tuple(CLAUDE_EVENTS):
        assert event in text, event
    assert str(cursor_gone) in text and str(claude_gone) in text
    assert CURSOR_HOST in text and CLAUDE_HOST in text
    # Says what to do about it, and what the repair path does.
    assert "haunt install" in text
    assert "re-merges hosts" in text
    assert report.ok is False


def test_doctor_flags_a_hook_command_that_is_not_executable(sandbox, monkeypatch):
    """Present but chmod-000 is exactly as silent as deleted.

    Revert the os.access(X_OK) branch in hook_command_defect() and this fails:
    the old check was is_file() only, which paints an unrunnable hook ok.
    """
    env = _use_home(monkeypatch, sandbox["default_home"])
    install_all_hosts(env["home"], env["hook_cmd"], env["mcp_cmd"])
    Path(env["claude_hook"]).chmod(0o600)

    report = diagnose(env["home"], env["hook_cmd"], env["mcp_cmd"])

    reasons = {d.reason for d in report.dangling_hooks if d.host == CLAUDE_HOST}
    assert reasons == {"not executable"}
    assert {d.event for d in report.dangling_hooks if d.host == CLAUDE_HOST} == set(
        CLAUDE_EVENTS
    )
    assert report.ok is False
    assert "not executable" in format_doctor(report)


def test_doctor_is_quiet_when_every_hook_command_resolves(sandbox, monkeypatch):
    """No false alarms on a healthy install."""
    env = _use_home(monkeypatch, sandbox["default_home"])
    install_all_hosts(env["home"], env["hook_cmd"], env["mcp_cmd"])

    report = diagnose(env["home"], env["hook_cmd"], env["mcp_cmd"])

    assert report.dangling_hooks == []
    for status in doctor_all_hosts(env["home"], env["hook_cmd"], env["mcp_cmd"]):
        assert status.dangling_hooks == []
    text = format_doctor(report)
    assert "dangling hooks" not in text
    assert "CANNOT capture" not in text


def test_doctor_check_writes_nothing(sandbox, monkeypatch):
    """The detector only stats. diagnose() must not repair as a side effect."""
    env = _use_home(monkeypatch, sandbox["default_home"])
    install_all_hosts(env["home"], env["hook_cmd"], env["mcp_cmd"])
    _plant_claude_command(sandbox, "/nonexistent/bin/haunt-hook-claude")

    before = {p: p.read_bytes() for p in _host_files(sandbox)}
    report = diagnose(env["home"], env["hook_cmd"], env["mcp_cmd"])
    assert report.dangling_hooks

    for path, blob in before.items():
        assert path.read_bytes() == blob, f"diagnose() rewrote {path}"


def test_incident_shape_regression(sandbox, monkeypatch):
    """The reported incident end to end: temp home, guard off, home deleted.

    Install from a worktree's `.smoke-home` with the guard explicitly forced
    off, delete the worktree, then run doctor. All six Claude Code events and
    all seven Cursor events must be reported as dangling, by name.
    """
    monkeypatch.setenv(ALT_HOME_ENV, "1")
    worktree = sandbox["tmp"] / "worktrees" / "chore-cleanup"
    smoke_home = worktree / ".smoke-home"
    env = _use_home(monkeypatch, smoke_home)
    install_all_hosts(env["home"], env["hook_cmd"], env["mcp_cmd"])

    settings = json.loads((sandbox["claude_dir"] / "settings.json").read_text())
    assert set(settings["hooks"]) >= set(CLAUDE_EVENTS)

    shutil.rmtree(worktree)
    assert not smoke_home.exists()

    report = diagnose(env["home"], env["hook_cmd"], env["mcp_cmd"])

    by_host: dict[str, set[str]] = {}
    for d in report.dangling_hooks:
        by_host.setdefault(d.host, set()).add(d.event)
        assert str(smoke_home) in d.command
    assert by_host.get(CLAUDE_HOST) == set(CLAUDE_EVENTS)
    assert len(CLAUDE_EVENTS) == 6
    assert by_host.get(CURSOR_HOST) == set(CURSOR_EVENTS)
    assert report.ok is False

    text = format_doctor(report)
    for event in CLAUDE_EVENTS:
        assert event in text, event
    assert ".smoke-home" in text


# --- the config target, not just the home ----------------------------------
# host_install_refusal judges the haunt home that would be written INTO the
# config. It cannot see WHICH config: that resolves through CLAUDE_CONFIG_DIR,
# CURSOR_HOME and CURSOR_HOOKS_JSON. With HOME redirected and one of those
# still pointing at the operator's real config, the home check passes on its
# own terms and the temp path lands in the real global settings.json anyway.


@pytest.fixture
def foreign_config(tmp_path):
    """A host config root deliberately outside the redirected home."""
    foreign = tmp_path / "elsewhere" / "real-config"
    foreign.mkdir(parents=True)
    return foreign


def test_home_guard_alone_would_allow_a_foreign_config_target(sandbox, foreign_config):
    """Pins why the second guard exists: the first one has nothing to object to.

    The home about to be written IS the default home for this HOME, so the
    home guard is satisfied and stays satisfied no matter where the config
    file lives. Only the target guard can see the difference.
    """
    from haunt.hosts import host_config_target_refusal, host_install_refusal

    assert host_install_refusal(sandbox["default_home"]) is None
    assert host_config_target_refusal(foreign_config / "settings.json") is not None


@pytest.mark.parametrize("variable", ["CLAUDE_CONFIG_DIR", "CURSOR_HOME"])
def test_a_config_root_outside_the_home_is_refused(
    sandbox, foreign_config, monkeypatch, variable
):
    """The incident, reproduced with the home guard fully enabled."""
    from haunt.hosts import ForeignHostConfigRefused, claude, cursor

    monkeypatch.setenv(variable, str(foreign_config))
    installer = claude.install if variable == "CLAUDE_CONFIG_DIR" else cursor.install
    cmds = _use_home(monkeypatch, sandbox["default_home"])
    with pytest.raises(ForeignHostConfigRefused):
        installer(cmds["home"], cmds["hook_cmd"], cmds["mcp_cmd"])
    assert sorted(foreign_config.iterdir()) == [], "wrote into the foreign config root"


def test_cursor_hooks_json_override_cannot_escape_the_home(
    sandbox, foreign_config, monkeypatch
):
    """CURSOR_HOOKS_JSON overrides the file directly, past CURSOR_HOME."""
    from haunt.hosts import ForeignHostConfigRefused, cursor

    target = foreign_config / "hooks.json"
    monkeypatch.setenv("CURSOR_HOOKS_JSON", str(target))
    cmds = _use_home(monkeypatch, sandbox["default_home"])
    with pytest.raises(ForeignHostConfigRefused):
        cursor.install(cmds["home"], cmds["hook_cmd"], cmds["mcp_cmd"])
    assert not target.exists()


def test_a_symlinked_config_root_cannot_alias_back_inside_the_home(
    sandbox, foreign_config, monkeypatch
):
    """Both sides resolve, so a symlink under the home does not launder it."""
    from haunt.hosts import ForeignHostConfigRefused, claude

    link = sandbox["fake_home"] / "looks-local"
    link.symlink_to(foreign_config)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(link))
    cmds = _use_home(monkeypatch, sandbox["default_home"])
    with pytest.raises(ForeignHostConfigRefused):
        claude.install(cmds["home"], cmds["hook_cmd"], cmds["mcp_cmd"])
    assert sorted(foreign_config.iterdir()) == []


def test_the_escape_hatch_still_allows_a_deliberate_foreign_target(
    sandbox, foreign_config, monkeypatch
):
    """One consent variable, exactly 1, covers both guards."""
    from haunt.hosts import ALT_HOME_ENV, claude

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(foreign_config))
    monkeypatch.setenv(ALT_HOME_ENV, "1")
    cmds = _use_home(monkeypatch, sandbox["default_home"])
    claude.install(cmds["home"], cmds["hook_cmd"], cmds["mcp_cmd"])
    assert (foreign_config / "settings.json").is_file()


# --- a home directory with a space ------------------------------------------
# `/Users/First Last/.haunt/bin/haunt-hook` is an ordinary macOS home, not an
# exotic input. The editors run hook commands through a shell, so that path
# starts nothing and capture stops with no error -- and doctor does not catch
# it, because the file exists. Refusing at install is the loud version of a
# failure that was previously silent, but it must stay escapable: nobody can
# rename their home directory.


def test_a_hook_command_with_a_space_is_refused(sandbox, monkeypatch):
    from haunt.hosts import UnsafeHookCommandRefused, check_hook_command_safe

    monkeypatch.delenv("HAUNT_ALLOW_UNSAFE_HOOK_COMMAND", raising=False)
    with pytest.raises(UnsafeHookCommandRefused):
        check_hook_command_safe("/Users/First Last/.haunt/bin/haunt-hook")


def test_command_substitution_in_a_hook_command_is_refused(sandbox, monkeypatch):
    from haunt.hosts import UnsafeHookCommandRefused, check_hook_command_safe

    monkeypatch.delenv("HAUNT_ALLOW_UNSAFE_HOOK_COMMAND", raising=False)
    for hostile in (
        "/tmp/x$(id -un)/bin/haunt-hook",
        "/tmp/x`id -un`/bin/haunt-hook",
        "/tmp/x;touch /tmp/pwned;/bin/haunt-hook",
    ):
        with pytest.raises(UnsafeHookCommandRefused):
            check_hook_command_safe(hostile)


def test_an_ordinary_path_is_allowed(sandbox):
    from haunt.hosts import check_hook_command_safe

    check_hook_command_safe("/Users/aronriley/.haunt/bin/haunt-hook")


@pytest.mark.parametrize("value", ["", " ", "0", "true", "yes", "2", "TRUE"])
def test_only_exactly_one_overrides_the_hook_command_check(
    sandbox, monkeypatch, value
):
    from haunt.hosts import UnsafeHookCommandRefused, check_hook_command_safe

    monkeypatch.setenv("HAUNT_ALLOW_UNSAFE_HOOK_COMMAND", value)
    with pytest.raises(UnsafeHookCommandRefused):
        check_hook_command_safe("/Users/First Last/.haunt/bin/haunt-hook")


def test_the_override_is_separate_from_the_alternate_home_consent(
    sandbox, monkeypatch
):
    """Consenting to one risk must not consent to the other."""
    from haunt.hosts import (
        ALT_HOME_ENV,
        UnsafeHookCommandRefused,
        check_hook_command_safe,
    )

    monkeypatch.setenv(ALT_HOME_ENV, "1")
    monkeypatch.delenv("HAUNT_ALLOW_UNSAFE_HOOK_COMMAND", raising=False)
    with pytest.raises(UnsafeHookCommandRefused):
        check_hook_command_safe("/Users/First Last/.haunt/bin/haunt-hook")

    monkeypatch.setenv("HAUNT_ALLOW_UNSAFE_HOOK_COMMAND", "1")
    check_hook_command_safe("/Users/First Last/.haunt/bin/haunt-hook")
