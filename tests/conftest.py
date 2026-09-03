from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import pytest

from tests.dashutil import TEST_DASH_TOKEN

# Captured before any redirect, so the sentinel below watches the real home.
_REAL_HOME = Path.home()


@lru_cache(maxsize=1)
def _host_model_cache() -> Path | None:
    """The model cache this host already has, resolved from the real environment.

    Must be read before `haunt_env` repoints HAUNT_HOME, because `models_dir()`
    follows HAUNT_HOME: unpinned, every test that embeds re-downloads the model
    into a tmp directory pytest then deletes. An explicit HAUNT_MODEL_CACHE is
    taken as given; the derived default (`~/.haunt/models` unless the host set
    HAUNT_HOME) only counts when it already holds something.
    """
    from haunt.paths import models_dir

    cache = models_dir()
    if os.environ.get("HAUNT_MODEL_CACHE"):
        return cache
    if cache.is_dir() and any(cache.iterdir()):
        return cache
    return None


# Pinned now, while HOME and HAUNT_HOME are still the host's own: the autouse
# fixture below redirects HOME, and models_dir() follows it.
_host_model_cache()


@pytest.fixture(autouse=True)
def _dashboard_security_defaults():
    """Every test starts with a configured launch token and loopback bind host."""
    from haunt.dashboard import configure_dashboard_security, reset_dashboard_security

    configure_dashboard_security(token=TEST_DASH_TOKEN, bind_host="127.0.0.1")
    yield
    reset_dashboard_security()


@pytest.fixture(autouse=True)
def isolate_host_homes(tmp_path, monkeypatch):
    """Redirect every host root a haunt write can reach, unconditionally.

    Conditional redirection was the hole: honouring an ambient CURSOR_HOME or
    CLAUDE_CONFIG_DIR let a suite run with HOME redirected still write the real
    global editor config, which is how a smoke home ended up in
    ~/.claude/settings.json. Nothing here consults the ambient value.

    The config roots live INSIDE the fake home rather than beside it, so the
    haunt home under test is genuinely ~/.haunt for the redirected HOME. The
    alternate-home guard then passes on its own merits instead of being
    switched off suite-wide, which kept the branch's headline protection live
    exactly where the historical damage came from.

    The model cache was pinned at import, above, while HAUNT_HOME was still
    the host's own.
    """
    fake_home = tmp_path / "user-home"
    (fake_home / ".haunt").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("CURSOR_HOME", str(fake_home / ".cursor"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(fake_home / ".claude"))
    # Overrides _cursor_dir() on its own, so redirecting CURSOR_HOME is not
    # enough to contain it.
    monkeypatch.delenv("CURSOR_HOOKS_JSON", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(fake_home / ".local" / "share"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_home / ".cache"))
    # Assert the redirect took before any test body can write through it. A
    # silently ineffective redirect is worse than none: the suite would look
    # isolated while writing to the operator's real home.
    assert Path.home().resolve() == fake_home.resolve(), Path.home()


# Files haunt itself writes into a real host. Hashed, not stat'd: ~/Desktop is
# symlinked into iCloud Drive on the maintainer's machine and sync advances
# mtime on files nothing wrote. ~/.haunt is deliberately absent -- live
# haunt-mcp servers write to it continuously, so it can never be proven quiet.
_HOST_SENTINEL_PATHS = (
    ".cursor/hooks.json",
    ".cursor/mcp.json",
    ".claude/settings.json",
    ".claude.json",
    "Desktop/Haunt Memories.command",
    ".local/share/applications/haunt-memories.desktop",
)


def _host_sentinel() -> dict[str, str]:
    """Current contents of every real host file a haunt install would touch."""
    snapshot: dict[str, str] = {}
    for relative in _HOST_SENTINEL_PATHS:
        try:
            snapshot[relative] = (_REAL_HOME / relative).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            snapshot[relative] = ""
    return snapshot


@pytest.fixture(autouse=True)
def _no_real_host_writes(tmp_path, isolate_host_homes):
    """Fail the test that wrote outside its allocated roots.

    Backstop, not the primary defence: isolate_host_homes is what prevents the
    write, and this is what notices when prevention was incomplete -- an
    absolute path read from somewhere other than an env var, or a subprocess
    that rebuilt its own environment. Depending on isolate_host_homes puts this
    fixture inside it, so the redirects are still in force at teardown and a
    failure here cannot leave a test running against the real home.
    """
    before = _host_sentinel()
    yield
    after = _host_sentinel()
    # Attribute the change rather than merely detect it. A bare
    # changed/unchanged check is flaky wherever anything else on the machine
    # legitimately writes these files, and it is not what went wrong: the
    # incident was a sandbox path landing inside a real host file. Look for
    # exactly that.
    sandbox = str(tmp_path)
    leaked = sorted(
        relative
        for relative, content in after.items()
        if content != before[relative] and sandbox in content
    )
    assert not leaked, (
        "this test's sandbox path leaked into real host files: "
        + ", ".join(leaked)
        + f"\n  sandbox: {sandbox}"
    )


@pytest.fixture
def fake_home() -> Path:
    """The redirected HOME for this test.

    Host config roots and the haunt home under test belong UNDER this, not
    beside it: that is the real-world shape (~/.haunt, ~/.cursor, ~/.claude),
    it keeps the alternate-home guard satisfied on its own merits rather than
    switched off, and it is what the host-config target guard checks.
    """
    return Path(os.environ["HOME"])


@pytest.fixture
def haunt_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME reusing the host model cache.

    HAUNT_FTS_ONLY and HAUNT_EMBED_MODEL are left alone when the caller set
    them: CI runs the suite with HAUNT_FTS_ONLY=1, and clearing it here would
    make that run exercise the embedding path it claims to skip. With neither
    set the run is still correct, only slower.
    """
    model_cache = _host_model_cache()
    # Under the redirected HOME, so this IS ~/.haunt and the alternate-home
    # guard has nothing to refuse.
    home = Path(os.environ["HOME"]) / ".haunt"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    if model_cache is not None:
        monkeypatch.setenv("HAUNT_MODEL_CACHE", str(model_cache))
    if not os.environ.get("HAUNT_EMBED_MODEL"):
        # Smallest model that still exercises the vector path. The bge-m3
        # default is 2.1 GB.
        monkeypatch.setenv("HAUNT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    from haunt import embed
    from haunt.bootstrap import bootstrap
    from haunt.paths import ensure_layout

    embed.reset()
    ensure_layout()
    bootstrap("default")
    yield home
    embed.reset()
