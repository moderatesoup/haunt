"""C1/C2: persist repo_path on namespace registration; refuse to derive a
namespace from a non-repository directory.

C1 bug: hooks and MCP construct Store(ns) with no repository, so a
hook-created namespace registers with a blank repo_path and no
repository_bindings row. _registered_namespace_for_repo() skips blank
repo_path rows, so reuse never matches and a repository that already has a
namespace gets a second one minted for it (the real-world duplicate pairs:
ironscope / github.com-moderatesoup-ironscope, memory-protocol /
github.com-memory-protocol-memory-protocol).

C2 bug: infer_namespace() falls back to the bare working-directory name for
any non-repository directory, including the user's home directory (the
real-world aronriley namespace: 58 rows of nothing but session-start
ceremony).

The fix threads the git context infer_namespace() already computes through
to Store()/register_namespace() at every entry point
(infer_namespace_context()), and gates the bare-directory-name fallback so
it can only ever *reuse* an already-registered namespace, never mint a new
one -- this is what keeps a legitimate, pre-existing directory-derived
namespace (e.g. ironscope) resolving to itself while refusing to mint any
new junk (e.g. a fresh aronriley on someone else's machine).

Finding on C1's second acceptance bullet ("reuse matches repositories whose
registry row predates the fix", i.e. blank-repo_path rows like the real
ironscope/156-rows and memory-protocol/151-rows namespaces): this was
investigated and found unsafe to implement automatically, not merely
unimplemented. A blank-repo_path row stores nothing that ties it to a
repository -- no identity, no path, and its db_path is derived only from
its own name -- so the only available correlator is matching the row's
name against the current checkout's directory basename, which is a
coincidence-of-labels guess (two unrelated repositories routinely share a
clone-directory name) that can silently commingle two repositories' memory
under one namespace with no clean undo. See _registered_namespace_for_repo()
in src/haunt/paths.py for the full reasoning. Healing an already-split pair
like ironscope / github.com-moderatesoup-ironscope is backlog C3
(operator-invoked, dry-run-first, reversible), not C1. The tests below that
involve a blank-repo_path row assert this forward-only behavior honestly: a
repository whose only prior registration is such a row mints one fresh,
uniquely-named namespace the first time it is seen post-fix, and stabilizes
on that fresh namespace afterward (never a second fork on top of the
first) -- it does not, and by design cannot safely, reclaim the old row.

Note on test mechanics: haunt.store imports _git_repo_context directly
(`from haunt.paths import ..., _git_repo_context, ...`), so patching
haunt.paths._git_repo_context alone does not reach the copy store.py
already bound at import time. Tests that exercise the write path
(register_namespace -> _repository_context -> _git_repo_context) patch
both modules via _patch_git_context() below so inference and registration
see the same git context -- exactly like a real, unmocked git repo would.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

import haunt.paths as paths
import haunt.store as store
from haunt.cli import app
from haunt.paths import infer_namespace_context, registry_path
from haunt.store import Store, init_registry, namespace_exists, register_namespace


@pytest.fixture
def repo_env(tmp_path, monkeypatch):
    home = tmp_path / "haunt-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    monkeypatch.delenv("HAUNT_MCP_ADMIN", raising=False)
    monkeypatch.delenv("HAUNT_MCP_ALLOW_PURGE", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
    from haunt import embed

    embed.reset()
    init_registry()
    yield tmp_path
    embed.reset()
    _reset_mcp()


def _reset_mcp() -> None:
    import haunt.mcp_server as mcp

    mcp._MCP_AUTHORITY = None
    mcp._MCP_AUTHORITY_HOME = None


def _patch_git_context(monkeypatch, remote_url: str | None, repo_root: Path | None) -> None:
    """Patch _git_repo_context in both haunt.paths and haunt.store.

    store.py's `_repository_context` calls its own directly-imported
    `_git_repo_context`, independent of haunt.paths' module attribute, so
    both must be patched for inference (paths.py) and registration
    (store.py) to agree -- see module docstring.
    """

    def fake(_root):
        return remote_url, repo_root

    monkeypatch.setattr(paths, "_git_repo_context", fake)
    monkeypatch.setattr(store, "_git_repo_context", fake)


def _namespace_rows() -> list[sqlite3.Row]:
    conn = sqlite3.connect(registry_path())
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT name, repo_path FROM namespaces").fetchall()
    finally:
        conn.close()


def _binding_count(*, repository_identity: str | None = None, repo_path: str | None = None) -> int:
    conn = sqlite3.connect(registry_path())
    try:
        if repository_identity is not None:
            return conn.execute(
                "SELECT COUNT(*) FROM repository_bindings WHERE repository_identity=?",
                (repository_identity,),
            ).fetchone()[0]
        return conn.execute(
            "SELECT COUNT(*) FROM repository_bindings WHERE repo_path=?",
            (repo_path,),
        ).fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# C1: a repository whose namespace already exists never mints a second one,
# across the hook, MCP, and CLI entry points.
# ---------------------------------------------------------------------------


def test_claude_hook_reuses_existing_repo_namespace(repo_env, monkeypatch):
    from haunt.claude_hook import handle_event

    project = repo_env / "hookrepo"
    project.mkdir()
    _patch_git_context(monkeypatch, "git@github.com:acme/hookrepo.git", project)

    base_payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(project),
    }
    handle_event({**base_payload, "prompt": "first turn", "session_id": "sess-1"})
    handle_event({**base_payload, "prompt": "second turn", "session_id": "sess-2"})

    rows = _namespace_rows()
    assert [r["name"] for r in rows] == ["github.com-acme-hookrepo"]
    assert rows[0]["repo_path"] == str(project.resolve())


def test_cursor_hook_reuses_existing_repo_namespace(repo_env, monkeypatch):
    from haunt.cursor_hook import handle_event

    project = repo_env / "cursorrepo"
    project.mkdir()
    _patch_git_context(monkeypatch, "git@github.com:acme/cursorrepo.git", project)

    base_payload = {
        "hook_event_name": "beforeSubmitPrompt",
        "cwd": str(project),
        "workspace_roots": [str(project)],
    }
    handle_event({**base_payload, "prompt": "first turn", "conversation_id": "conv-1"})
    handle_event({**base_payload, "prompt": "second turn", "conversation_id": "conv-2"})

    rows = _namespace_rows()
    assert [r["name"] for r in rows] == ["github.com-acme-cursorrepo"]
    assert rows[0]["repo_path"] == str(project.resolve())


def test_mcp_reuses_existing_repo_namespace(repo_env, monkeypatch):
    import haunt.mcp_server as mcp

    project = repo_env / "mcprepo"
    project.mkdir()
    monkeypatch.chdir(project)
    _patch_git_context(monkeypatch, "git@github.com:acme/mcprepo.git", project)

    _reset_mcp()
    first = mcp._authority().bound_namespace
    assert first == "github.com-acme-mcprepo"
    import json

    observed = json.loads(mcp.memory_observe(text="first mcp canary"))
    assert observed["ok"] is True

    # A brand-new MCP process (fresh authority cache) for the same repo.
    _reset_mcp()
    second = mcp._authority().bound_namespace
    assert second == first

    rows = _namespace_rows()
    assert [r["name"] for r in rows] == ["github.com-acme-mcprepo"]
    assert _binding_count(repository_identity="github.com/acme/mcprepo") == 1


def test_cli_init_reuses_existing_repo_namespace(repo_env, monkeypatch):
    project = repo_env / "clirepo"
    project.mkdir()
    monkeypatch.chdir(project)
    _patch_git_context(monkeypatch, "git@github.com:acme/clirepo.git", project)

    runner = CliRunner()
    first = runner.invoke(app, ["init"])
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, ["init"])
    assert second.exit_code == 0, second.output

    rows = _namespace_rows()
    assert [r["name"] for r in rows] == ["github.com-acme-clirepo"]
    assert rows[0]["repo_path"] == str(project.resolve())


def test_cli_init_explicit_name_does_not_auto_bind_cwd_repo(repo_env, monkeypatch):
    """An explicit namespace name is a deliberate choice, like HAUNT_NAMESPACE
    -- it must not silently pick up whatever repo happens to be in cwd."""
    project = repo_env / "unrelated-repo"
    project.mkdir()
    monkeypatch.chdir(project)
    _patch_git_context(monkeypatch, "git@github.com:acme/unrelated-repo.git", project)

    runner = CliRunner()
    result = runner.invoke(app, ["init", "my-custom-name"])
    assert result.exit_code == 0, result.output

    rows = _namespace_rows()
    assert [r["name"] for r in rows] == ["my-custom-name"]
    assert rows[0]["repo_path"] is None


# ---------------------------------------------------------------------------
# C1: namespace creation persists the repository so a *second* inference
# matches by binding rather than by re-deriving the same label.
#
# A blank-repo_path row (every hook/MCP-created namespace, pre-fix) is
# deliberately NOT matched by inference -- see the module docstring above
# and _registered_namespace_for_repo()'s docstring in src/haunt/paths.py
# for why this was investigated and found unsafe to do automatically. The
# tests below that involve such a row assert that honestly: the row is
# left alone, a fresh namespace is minted the first time (one honest fork,
# not a guess), and repeated inference afterward stabilizes on the fresh
# namespace rather than minting a third/fourth/fifth one.
# ---------------------------------------------------------------------------


def test_blank_repo_path_row_is_not_reused_forks_once(repo_env, monkeypatch):
    """Fixed version of the former test_reuse_matches_preexisting_row_with_
    blank_repo_path. That test pre-registered a namespace whose name
    coincidentally equalled what fresh identity-formula derivation computes
    anyway (`github.com-acme-widgets` for remote acme/widgets), so it
    passed whether or not any reuse of the blank row happened -- it could
    not distinguish "found the row" from "recomputed the same string by
    coincidence" (see the mutation check for
    test_blank_repo_path_row_forks_once_then_stabilizes below, which
    exercises the same coincidence and confirms it).

    Registering under a name that does NOT match the identity formula
    (mirroring the real ironscope row: registered as "ironscope" while the
    identity formula yields "github.com-moderatesoup-ironscope") makes the
    assertion below genuinely discriminating: it only passes if inference
    left the legacy row alone and derived the fresh identity name, and it
    would fail if a future change wrongly started matching blank rows by
    some coincidence-of-labels heuristic (e.g. "reuse the only blank row in
    the registry")."""
    project = repo_env / "widgets"
    project.mkdir()
    register_namespace("legacy-widgets")  # pre-fix state: no repo_path, no binding
    rows = _namespace_rows()
    assert rows[0]["repo_path"] in (None, "")

    _patch_git_context(monkeypatch, "git@github.com:acme/widgets.git", project)
    ns, repo_path = infer_namespace_context(project)

    assert ns == "github.com-acme-widgets"
    assert repo_path == str(project.resolve())
    # The legacy row is untouched by inference alone (only Store() writes).
    names = {r["name"] for r in _namespace_rows()}
    assert names == {"legacy-widgets"}


def test_namespace_creation_persists_repository_for_binding_reuse(repo_env, monkeypatch):
    project = repo_env / "widgets2"
    project.mkdir()
    _patch_git_context(monkeypatch, "git@github.com:acme/widgets2.git", project)

    ns1, repo_path1 = infer_namespace_context(project)
    assert ns1 == "github.com-acme-widgets2"
    assert repo_path1 == str(project.resolve())
    # Before any Store() has opened this namespace, no binding exists yet.
    assert _binding_count(repository_identity="github.com/acme/widgets2") == 0

    with Store(ns1, repo_path1):
        pass

    assert _binding_count(repository_identity="github.com/acme/widgets2") == 1

    # A second, independent inference call must still resolve correctly --
    # now via the binding table rather than by coincidence of formula.
    ns2, _ = infer_namespace_context(project)
    assert ns2 == ns1
    assert len(_namespace_rows()) == 1


def test_blank_repo_path_row_forks_once_then_stabilizes(repo_env, monkeypatch):
    """Fixed version of the former
    test_repository_with_existing_namespace_never_mints_second_across_many_inferences.
    That test started from an EMPTY registry (no pre-registration at all),
    so every one of its 5 loop iterations recomputing the same
    identity-formula string proved nothing about reuse -- the identity
    formula is deterministic, so "matched an existing row" and "recomputed
    the same fresh name every time" are indistinguishable when there is
    nothing else registered to (wrongly) prefer.

    Starting from a pre-existing blank-repo_path row under a
    non-formula-matching name (as the real ironscope row was) makes this
    meaningful: it proves repeated inference is stable even though the
    first call cannot safely reuse that row (see the module docstring) --
    it forks exactly once, and the following four calls reuse that fresh
    namespace via the repository_bindings row Store() records for it
    (test_namespace_creation_persists_repository_for_binding_reuse covers
    that mechanism directly), rather than minting a third, fourth, or
    fifth namespace."""
    project = repo_env / "widgets3"
    project.mkdir()
    register_namespace("legacy-widgets3")  # pre-fix orphan; deliberately unmatched
    _patch_git_context(monkeypatch, "git@github.com:acme/widgets3.git", project)

    seen = set()
    for _ in range(5):
        ns, repo_path = infer_namespace_context(project)
        with Store(ns, repo_path):
            pass
        seen.add(ns)

    assert seen == {"github.com-acme-widgets3"}
    names = {r["name"] for r in _namespace_rows()}
    assert names == {"legacy-widgets3", "github.com-acme-widgets3"}


def test_forked_repository_does_not_switch_back_to_blank_namespace(repo_env, monkeypatch):
    """The real, already-measured registry state (see BACKLOG.md C-series
    baseline): `ironscope` (156 rows, blank repo_path) and
    `github.com-moderatesoup-ironscope` (313 rows, the fork) both already
    exist. This is the ambiguous case the module docstring and
    _registered_namespace_for_repo() describe -- the fork already happened
    and the newer namespace may hold real data that a blank row cannot
    prove it supersedes. Inference must never silently switch back to the
    older blank one just because its repo_path is empty: that would hide
    the 313 newer rows behind the 156-row namespace, which is worse than
    leaving the split in place. Healing an existing fork like this one is
    backlog C3 (operator-invoked, dry-run, reversible), not something
    inference does on its own."""
    project = repo_env / "ironscope"
    project.mkdir()
    register_namespace("ironscope")  # the old, blank-repo_path namespace
    register_namespace("github.com-moderatesoup-ironscope")  # the fork
    _patch_git_context(
        monkeypatch, "git@github.com:moderatesoup/ironscope.git", project
    )

    ns, repo_path = infer_namespace_context(project)

    assert ns == "github.com-moderatesoup-ironscope"
    names = {r["name"] for r in _namespace_rows()}
    assert names == {"ironscope", "github.com-moderatesoup-ironscope"}


# ---------------------------------------------------------------------------
# C2: the home directory never yields a namespace named after it, and a
# non-repository directory does not silently mint a directory-named store.
# ---------------------------------------------------------------------------


def test_home_directory_never_yields_namespace_named_after_it(repo_env, monkeypatch):
    fake_home = repo_env / "Users" / "aronriley"
    fake_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    ns, repo_path = infer_namespace_context(fake_home)

    assert ns != fake_home.name
    assert ns == "default"
    assert repo_path is None
    assert not namespace_exists(fake_home.name)


def test_home_directory_does_not_reuse_a_preexisting_same_name_namespace(repo_env, monkeypatch):
    """A namespace named after $HOME's basename that already exists (the
    real-world `aronriley` artifact: 58 rows of nothing but session-start
    ceremony) is the bug, not data future inference should keep targeting."""
    fake_home = repo_env / "Users" / "aronriley"
    fake_home.mkdir(parents=True)
    register_namespace("aronriley")
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    ns, repo_path = infer_namespace_context(fake_home)

    assert ns == "default"
    assert repo_path is None


def test_non_repository_directory_does_not_mint_directory_named_store(repo_env):
    scratch = repo_env / "scratch-work-dir"
    scratch.mkdir()

    ns, repo_path = infer_namespace_context(scratch)

    assert ns == "default"
    assert repo_path is None
    assert not namespace_exists("scratch-work-dir")


def test_non_repository_directory_resolution_does_not_register_anything(repo_env):
    """Merely inferring must not itself mint a row -- only Store()/
    register_namespace() does, and "default" is the sanctioned catch-all
    (the same namespace `haunt bootstrap` creates), not a new junk store."""
    scratch = repo_env / "another-scratch-dir"
    scratch.mkdir()

    infer_namespace_context(scratch)

    assert _namespace_rows() == []


# ---------------------------------------------------------------------------
# Backward compatibility: an already-registered namespace -- including one
# derived from a bare directory name before this fix -- must keep resolving
# to itself. C2 must prevent NEW junk namespaces, not re-route existing ones.
# ---------------------------------------------------------------------------


def test_existing_directory_named_namespace_still_resolves_to_itself(repo_env):
    """ironscope / memory-protocol style: a real, in-use namespace minted
    from a bare directory name before this fix must not be orphaned by it.

    Registered under a different case than the directory name itself
    (namespace labels preserve case; registry lookup is case-insensitive --
    see normalize_namespace_label()). This is what makes the assertion
    below genuinely prove the registry lookup fired: the directory's own
    basename is "ironscope" (lowercase), so a broken/absent reuse gate that
    just derives the candidate fresh would also produce lowercase
    "ironscope" -- a coincidental pass indistinguishable from success (the
    original vacuousness bug, applied to this fallback). Only a real
    registry hit can produce the registered, case-preserved "IronScope"."""
    project = repo_env / "ironscope"
    project.mkdir()
    register_namespace("IronScope")
    assert namespace_exists("ironscope")

    ns, repo_path = infer_namespace_context(project)

    assert ns == "IronScope"
    assert repo_path is None
    assert len(_namespace_rows()) == 1


def test_existing_directory_named_namespace_survives_repeated_inference(repo_env):
    """Same guard, exercised the way a long-running session actually would:
    repeated inference calls against the same pre-existing namespace. See
    test_existing_directory_named_namespace_still_resolves_to_itself for why
    the registered name's case must differ from the directory's."""
    project = repo_env / "memory-protocol"
    project.mkdir()
    register_namespace("Memory-Protocol")

    for _ in range(3):
        ns, repo_path = infer_namespace_context(project)
        assert ns == "Memory-Protocol"
        assert repo_path is None

    assert len(_namespace_rows()) == 1


# ---------------------------------------------------------------------------
# Defense in depth: an MCP admin explicitly requesting an unrelated,
# not-yet-existing namespace by name must not bind it to the *process's*
# inferred repository -- that would silently mis-attribute an operator's
# explicit choice to whatever repo the server process happens to run from.
# ---------------------------------------------------------------------------


def test_mcp_admin_explicit_namespace_does_not_inherit_process_repo(repo_env, monkeypatch):
    import json

    import haunt.mcp_server as mcp

    project = repo_env / "adminrepo"
    project.mkdir()
    monkeypatch.chdir(project)
    _patch_git_context(monkeypatch, "git@github.com:acme/adminrepo.git", project)
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    _reset_mcp()

    result = json.loads(
        mcp.memory_observe(text="admin canary", namespace="totally-unrelated")
    )
    assert result["ok"] is True

    rows = {r["name"]: r["repo_path"] for r in _namespace_rows()}
    assert rows["totally-unrelated"] is None
