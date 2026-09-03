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
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

import haunt.paths as paths
import haunt.store as store
from haunt.cli import app
from haunt.paths import (
    disambiguate_namespace_label,
    infer_namespace_context,
    namespaces_dir,
    registry_path,
)
from haunt.store import (
    NamespaceCollisionError,
    Store,
    init_registry,
    namespace_exists,
    register_namespace,
    register_namespace_context,
)


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


# ---------------------------------------------------------------------------
# Namespace collisions: two unrelated repositories must never derive the same
# label. Both derivation paths lose information -- the remote path rewrites
# "/" to "-" (so "acme/foo-bar" and "acme-foo/bar" flatten alike), and the
# remote-less path keeps only the checkout's basename (so every "api"
# directory looks identical). Registration is what makes a label a claim, so
# the guard fires on the second repository only: the first keeps the label it
# already has, which is what stops this from re-routing anyone's memory.
# ---------------------------------------------------------------------------


def test_separator_collision_between_distinct_remotes_forks(repo_env, monkeypatch):
    """`github.com/acme/foo-bar` and `github.com/acme-foo/bar` both sanitize
    to `github.com-acme-foo-bar`, well under the 80-character threshold that
    triggers namespace_for_repo_identity()'s hash. Pre-fix the second
    repository silently landed in the first's database."""
    first = repo_env / "foo-bar"
    first.mkdir()
    _patch_git_context(monkeypatch, "git@github.com:acme/foo-bar.git", first)
    first_ns, first_repo = infer_namespace_context(first)
    with Store(first_ns, first_repo):
        pass
    assert first_ns == "github.com-acme-foo-bar"

    second = repo_env / "bar"
    second.mkdir()
    _patch_git_context(monkeypatch, "git@github.com:acme-foo/bar.git", second)
    second_ns, second_repo = infer_namespace_context(second)

    assert second_ns != first_ns
    assert second_ns.startswith("github.com-acme-foo-bar-")
    assert second_repo == str(second.resolve())


def test_basename_collision_between_remoteless_repos_forks(repo_env, monkeypatch):
    """R8: with no git remote, inference falls back to the checkout's
    basename, so `clientA/api` and `clientB/api` both resolve to `api`."""
    first = repo_env / "clientA" / "api"
    first.mkdir(parents=True)
    _patch_git_context(monkeypatch, None, first)
    first_ns, first_repo = infer_namespace_context(first)
    with Store(first_ns, first_repo):
        pass
    assert first_ns == "api"

    second = repo_env / "clientB" / "api"
    second.mkdir(parents=True)
    _patch_git_context(monkeypatch, None, second)
    second_ns, second_repo = infer_namespace_context(second)

    assert second_ns != first_ns
    assert second_ns.startswith("api-")
    assert second_repo == str(second.resolve())


def test_colliding_label_is_stable_across_repeated_inference(repo_env, monkeypatch):
    """The disambiguator is a digest of the repository's own identity, not a
    counter or a random suffix, so the forked repository stabilizes on one
    label instead of minting a fresh namespace on every inference."""
    first = repo_env / "foo-bar2"
    first.mkdir()
    _patch_git_context(monkeypatch, "git@github.com:acme/foo-bar2.git", first)
    first_ns, first_repo = infer_namespace_context(first)
    with Store(first_ns, first_repo):
        pass

    second = repo_env / "bar2"
    second.mkdir()
    _patch_git_context(monkeypatch, "git@github.com:acme-foo/bar2.git", second)
    seen = set()
    for _ in range(5):
        ns, repo_path = infer_namespace_context(second)
        with Store(ns, repo_path):
            pass
        seen.add(ns)

    assert len(seen) == 1
    assert seen != {first_ns}
    assert len(_namespace_rows()) == 2


# ---------------------------------------------------------------------------
# Backward compatibility for the collision guard. Changing a derivation
# function re-routes every namespace it already produced, which is the exact
# damage `haunt namespace reconcile` exists to heal. Registered labels must
# therefore keep resolving unchanged; only a label a *second* repository
# would newly steal is allowed to move.
# ---------------------------------------------------------------------------


def test_registered_collision_prone_label_still_resolves(repo_env, monkeypatch):
    """A namespace registered under exactly what the pre-fix derivation
    produced -- a dashed repo name, so a label the guard now recognises as
    collision-prone -- must still resolve to itself, bare digest-free label
    and all, with no second namespace minted."""
    project = repo_env / "foo-bar3"
    project.mkdir()
    _patch_git_context(monkeypatch, "git@github.com:acme/foo-bar3.git", project)
    register_namespace("github.com-acme-foo-bar3", str(project.resolve()))

    for _ in range(3):
        ns, repo_path = infer_namespace_context(project)
        assert ns == "github.com-acme-foo-bar3"
        assert repo_path == str(project.resolve())

    assert len(_namespace_rows()) == 1


def test_registration_wins_over_a_label_the_formula_cannot_produce(
    repo_env, monkeypatch
):
    """The load-bearing back-compat case: a namespace whose label no
    derivation would ever mint for this repo.

    The sibling tests register the label the formula happens to produce, so
    they cannot tell a registry lookup apart from a re-derivation that
    coincidentally agrees. This one can: nothing about `acme/foo-bar4`
    yields `legacy-ns-from-before-c1`, so returning it proves resolution
    came from the binding, not the formula."""
    project = repo_env / "foo-bar4"
    project.mkdir()
    _patch_git_context(monkeypatch, "git@github.com:acme/foo-bar4.git", project)
    register_namespace("legacy-ns-from-before-c1", str(project.resolve()))

    for _ in range(3):
        ns, repo_path = infer_namespace_context(project)
        assert ns == "legacy-ns-from-before-c1"
        assert repo_path == str(project.resolve())

    assert len(_namespace_rows()) == 1


def test_registered_basename_label_still_resolves(repo_env, monkeypatch):
    """The same guarantee for R8's derivation path: a remote-less checkout
    whose basename-derived namespace already exists keeps resolving to it."""
    project = repo_env / "clientA" / "api2"
    project.mkdir(parents=True)
    _patch_git_context(monkeypatch, None, project)
    register_namespace("api2", str(project.resolve()))

    for _ in range(3):
        ns, repo_path = infer_namespace_context(project)
        assert ns == "api2"
        assert repo_path == str(project.resolve())

    assert len(_namespace_rows()) == 1


def test_blank_repo_path_row_does_not_count_as_a_collision(repo_env, monkeypatch):
    """A blank-repo_path row is the pre-C1 hook/MCP registration shape: it
    ties nothing to any repository, so it cannot be evidence that some
    *other* repository owns the label. Treating it as a collision would
    fork every legacy namespace away from the repository that has been
    using it -- the opposite of the intent -- so the repository that
    derives the name still adopts the row, exactly as before this guard.
    See _registered_namespace_for_repo()'s docstring for the blank-row rule
    this composes with."""
    project = repo_env / "widgets9"
    project.mkdir()
    register_namespace("github.com-acme-widgets9")  # blank repo_path, no binding
    _patch_git_context(monkeypatch, "git@github.com:acme/widgets9.git", project)

    ns, repo_path = infer_namespace_context(project)

    assert ns == "github.com-acme-widgets9"
    with Store(ns, repo_path):
        pass
    assert len(_namespace_rows()) == 1


# ---------------------------------------------------------------------------
# Concurrent registration. The mint-time guard above reads the registry during
# inference, so two repositories that infer before either registers both see
# the shared label free -- the first publishes it, and the second used to be
# handed the same namespace and simply have a second repository binding added
# to it. Ownership is therefore settled inside register_namespace()'s
# BEGIN IMMEDIATE transaction, the first point that can see the winner.
# ---------------------------------------------------------------------------


def _patch_git_contexts(
    monkeypatch, contexts: dict[Path, tuple[str | None, Path]]
) -> None:
    """Serve each checkout its own git context, keyed by resolved path.

    _patch_git_context() above fakes one context for the whole process, which
    cannot express two repositories running at once. Unknown paths report no
    repository, as a non-git directory does.
    """
    resolved = {path.resolve(): value for path, value in contexts.items()}

    def fake(root):
        return resolved.get(Path(root).resolve(), (None, None))

    monkeypatch.setattr(paths, "_git_repo_context", fake)
    monkeypatch.setattr(store, "_git_repo_context", fake)


def _namespaces_with_two_repository_owners() -> dict[str, list[str]]:
    """Namespace ids bound to more than one repository: the defect, measured."""
    conn = sqlite3.connect(registry_path())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT namespace_id, repository_identity, repo_path FROM repository_bindings"
        ).fetchall()
    finally:
        conn.close()
    owners: dict[str, set[str]] = {}
    for row in rows:
        owner = str(row["repository_identity"] or row["repo_path"] or "")
        if owner:
            owners.setdefault(str(row["namespace_id"]), set()).add(owner)
    return {ns: sorted(found) for ns, found in owners.items() if len(found) > 1}


def _race_infer_then_register(projects: list[Path]) -> list[tuple[str, str]]:
    """Infer for every project, then register them all at once.

    The barrier is the point of the exercise: it holds every thread until all
    of them have finished inference, which is the only interleaving that
    reproduces the defect. Letting a thread register while another is still
    inferring lets the second inference see the first registration, which is
    the already-working serial case.

    Returns each project's ``(canonical label, database path)`` as the Store
    that opened it reports them -- the namespace the repository actually
    landed in, not the one it asked for.
    """
    barrier = threading.Barrier(len(projects))
    landed: list[tuple[str, str] | None] = [None] * len(projects)
    failures: list[BaseException | None] = [None] * len(projects)

    def worker(index: int, project: Path) -> None:
        try:
            ns, repo_path = infer_namespace_context(project)
            barrier.wait(timeout=60)
            with Store(ns, repo_path) as st:
                landed[index] = (st.name, str(st.db_path))
        except BaseException as exc:  # re-raised on the calling thread below
            failures[index] = exc
            barrier.abort()

    threads = [
        threading.Thread(target=worker, args=(index, project), daemon=True)
        for index, project in enumerate(projects)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for failure in failures:
        if failure is not None:
            raise failure
    assert all(entry is not None for entry in landed), landed
    return [entry for entry in landed if entry is not None]


def test_concurrent_registration_of_a_separator_collision_forks(
    repo_env, monkeypatch
):
    """`github.com/acme/foo-bar6` and `github.com/acme-foo/bar6` both sanitize
    to `github.com-acme-foo-bar6`, and both inferences complete before either
    registers, so neither sees the other's claim."""
    first = repo_env / "foo-bar6"
    second = repo_env / "bar6"
    first.mkdir()
    second.mkdir()
    identities = ("github.com/acme/foo-bar6", "github.com/acme-foo/bar6")
    _patch_git_contexts(
        monkeypatch,
        {
            first: ("git@github.com:acme/foo-bar6.git", first),
            second: ("git@github.com:acme-foo/bar6.git", second),
        },
    )

    landed = _race_infer_then_register([first, second])

    assert _namespaces_with_two_repository_owners() == {}
    bare = "github.com-acme-foo-bar6"
    forks = [disambiguate_namespace_label(bare, ident) for ident in identities]
    names = [name for name, _ in landed]
    assert len({path for _, path in landed}) == 2
    assert set(names) in ({bare, forks[0]}, {bare, forks[1]})
    # Whichever lost, it landed on the digest of its own remote identity --
    # the label it would have inferred had it simply run second, never a
    # third label and never the other repository's fork.
    for index, name in enumerate(names):
        assert name in (bare, forks[index])
    for project, name in zip((first, second), names):
        again, again_repo = infer_namespace_context(project)
        assert again == name
        assert again_repo == str(project.resolve())


def test_concurrent_registration_of_a_basename_collision_forks(
    repo_env, monkeypatch
):
    """The remote-less shape: `/a/app` and `/b/app` both derive `app`, and the
    discriminator is each checkout's own path."""
    first = repo_env / "a" / "app"
    second = repo_env / "b" / "app"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _patch_git_contexts(monkeypatch, {first: (None, first), second: (None, second)})

    landed = _race_infer_then_register([first, second])

    assert _namespaces_with_two_repository_owners() == {}
    forks = [
        disambiguate_namespace_label("app", str(project.resolve()))
        for project in (first, second)
    ]
    names = [name for name, _ in landed]
    assert len({path for _, path in landed}) == 2
    assert set(names) in ({"app", forks[0]}, {"app", forks[1]})
    for index, name in enumerate(names):
        assert name in ("app", forks[index])
    for project, name in zip((first, second), names):
        again, again_repo = infer_namespace_context(project)
        assert again == name
        assert again_repo == str(project.resolve())


def test_concurrent_registration_from_one_repository_does_not_fork(
    repo_env, monkeypatch
):
    """The negative case the guard must not break: one repository registering
    twice at once is not a collision, so it keeps one namespace and one
    binding rather than forking away from itself."""
    project = repo_env / "solo6"
    project.mkdir()
    _patch_git_contexts(
        monkeypatch, {project: ("git@github.com:acme/solo6.git", project)}
    )

    landed = _race_infer_then_register([project, project])

    assert {name for name, _ in landed} == {"github.com-acme-solo6"}
    assert len({path for _, path in landed}) == 1
    assert [r["name"] for r in _namespace_rows()] == ["github.com-acme-solo6"]
    assert _binding_count(repository_identity="github.com/acme/solo6") == 1
    assert _namespaces_with_two_repository_owners() == {}


def test_concurrent_registration_from_one_remoteless_repository_does_not_fork(
    repo_env, monkeypatch
):
    """The same negative case for the path-discriminated shape, where the only
    thing telling two callers apart is a checkout path they share."""
    project = repo_env / "clientC" / "api6"
    project.mkdir(parents=True)
    _patch_git_contexts(monkeypatch, {project: (None, project)})

    landed = _race_infer_then_register([project, project])

    assert {name for name, _ in landed} == {"api6"}
    assert len({path for _, path in landed}) == 1
    assert [r["name"] for r in _namespace_rows()] == ["api6"]
    assert _binding_count(repo_path=str(project.resolve())) == 1
    assert _namespaces_with_two_repository_owners() == {}


def _drop_repository_bindings() -> None:
    """Reduce the registry to its pre-bindings shape: `namespaces` rows only."""
    conn = sqlite3.connect(registry_path())
    try:
        conn.execute("DELETE FROM repository_bindings")
        conn.commit()
    finally:
        conn.close()


def test_legacy_repo_path_row_without_a_binding_still_owns_its_label(
    repo_env, monkeypatch
):
    """A registry written before repository_bindings records the repository
    only on `namespaces`, and that row is still an ownership claim -- unlike
    a blank one, which names nobody. A second repository must fork off it,
    and the repository it names must still be handed its own namespace."""
    first = repo_env / "clientD" / "api7"
    second = repo_env / "clientE" / "api7"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _patch_git_contexts(monkeypatch, {first: (None, first), second: (None, second)})
    register_namespace("api7", str(first.resolve()))
    _drop_repository_bindings()

    intruder, _ = register_namespace_context("api7", str(second.resolve()))
    assert intruder == disambiguate_namespace_label("api7", str(second.resolve()))

    owner, _ = register_namespace_context("api7", str(first.resolve()))
    assert owner == "api7"
    assert _namespaces_with_two_repository_owners() == {}


# ---------------------------------------------------------------------------
# Candidate exhaustion. Forking is what keeps a raced registration off another
# repository's namespace, so it has exactly one fallback: the label inference
# would have minted for this repository had it run second. When a third
# repository already owns that too, there is nothing left to fall back to and
# registration refuses. Nothing below asserts that refusing is pleasant --
# only that it refuses, leaves no half-registration behind, and that the
# divergence it creates from the sequential outcome is the documented one.
# ---------------------------------------------------------------------------


def _namespace_db_files() -> list[str]:
    """Every namespace database file on disk, so an orphan is visible."""
    root = namespaces_dir()
    if not root.is_dir():
        return []
    return sorted(entry.name for entry in root.iterdir())


def _alias_rows() -> list[tuple[str, str]]:
    conn = sqlite3.connect(registry_path())
    try:
        return sorted(
            (str(a), str(b))
            for a, b in conn.execute(
                "SELECT normalized_label, namespace_id FROM namespace_aliases"
            ).fetchall()
        )
    finally:
        conn.close()


def _squat(label: str, project: Path) -> None:
    """Give *project* ownership of exactly *label*, whatever label that is."""
    taken, _ = register_namespace_context(label, str(project.resolve()))
    assert taken == label, f"squatter was itself forked to {taken!r}"


def test_exhausted_fork_candidates_refuse_rather_than_commingle(
    repo_env, monkeypatch
):
    """Both the bare label and the fork target are foreign-owned.

    This is the fix's only hard-failure branch. `second` cannot have the bare
    label -- `first` published it -- and cannot have its own fork either,
    because `third` holds that. Binding it to either would be exactly the
    commingling the fork exists to prevent, so registration refuses.
    """
    first = repo_env / "cA" / "api8"
    second = repo_env / "cB" / "api8"
    third = repo_env / "cC" / "squatter8"
    for project in (first, second, third):
        project.mkdir(parents=True)
    _patch_git_contexts(
        monkeypatch,
        {first: (None, first), second: (None, second), third: (None, third)},
    )

    owner, _ = register_namespace_context("api8", str(first.resolve()))
    assert owner == "api8"
    fork = disambiguate_namespace_label("api8", str(second.resolve()))
    _squat(fork, third)

    before_dbs = _namespace_db_files()
    before_names = sorted(str(row["name"]) for row in _namespace_rows())
    before_aliases = _alias_rows()

    with pytest.raises(NamespaceCollisionError) as refused:
        register_namespace_context("api8", str(second.resolve()))

    # The message names the label it could not have and who holds it, so an
    # operator can act without reading the registry by hand.
    assert fork in str(refused.value)
    assert str(third.resolve()) in str(refused.value)

    # No binding written for the refused repository...
    assert _binding_count(repo_path=str(second.resolve())) == 0
    # ...no orphan namespace database left behind by the rolled-back attempt...
    assert _namespace_db_files() == before_dbs
    # ...and no registry row or alias invented for it either.
    assert sorted(str(row["name"]) for row in _namespace_rows()) == before_names
    assert _alias_rows() == before_aliases
    # The whole point: nothing was commingled to make the refusal go away.
    assert _namespaces_with_two_repository_owners() == {}
    # Refusing is idempotent -- retrying does not eventually let it through.
    with pytest.raises(NamespaceCollisionError):
        register_namespace_context("api8", str(second.resolve()))
    assert _binding_count(repo_path=str(second.resolve())) == 0
    # The repositories that did register are untouched by the refusal.
    assert register_namespace_context("api8", str(first.resolve()))[0] == "api8"
    assert register_namespace_context(fork, str(third.resolve()))[0] == fork


def test_raced_and_sequential_losers_diverge_when_the_fork_target_is_taken(
    repo_env, monkeypatch
):
    """The precise exception to "the loser lands where it would have run second".

    The claim holds whenever the fork target is free, which is the ordinary
    case. It does not hold here. A loser that inferred *before* the winner
    published asks for the bare label and has one fallback, which `third`
    holds, so it fails closed. A loser that inferred *after* asks for the fork
    label itself and still has a fallback -- forking a second time -- so it
    succeeds on `label-digest-digest`. Both fail safe; they do not agree, and
    register_namespace_context()'s docstring says so.
    """
    first = repo_env / "dA" / "api9"
    raced = repo_env / "dB" / "api9"
    sequential = repo_env / "dC" / "api9"
    third = repo_env / "dD" / "squatter9"
    fourth = repo_env / "dE" / "squatter9b"
    for project in (first, raced, sequential, third, fourth):
        project.mkdir(parents=True)
    _patch_git_contexts(
        monkeypatch,
        {
            first: (None, first),
            raced: (None, raced),
            sequential: (None, sequential),
            third: (None, third),
            fourth: (None, fourth),
        },
    )
    register_namespace_context("api9", str(first.resolve()))
    raced_fork = disambiguate_namespace_label("api9", str(raced.resolve()))
    sequential_fork = disambiguate_namespace_label(
        "api9", str(sequential.resolve())
    )
    _squat(raced_fork, third)
    _squat(sequential_fork, fourth)

    # Raced: inference ran before `first` published, so the label asked for is
    # the bare one and the single fallback is already `third`'s.
    with pytest.raises(NamespaceCollisionError):
        register_namespace_context("api9", str(raced.resolve()))

    # Sequential: inference ran after, so it asks for its own fork label --
    # and forks off that, one level deeper than the raced caller could reach.
    inferred, inferred_repo = infer_namespace_context(sequential)
    assert inferred == sequential_fork
    landed, _ = register_namespace_context(inferred, inferred_repo)
    assert landed == disambiguate_namespace_label(
        sequential_fork, str(sequential.resolve())
    )
    assert landed != sequential_fork
    assert _namespaces_with_two_repository_owners() == {}


def test_registration_candidates_are_never_empty(repo_env):
    """What makes the publication loop total.

    The loop attempts every candidate but the last, then attempts the last
    outside the loop and converts its refusal. That is only exhaustive because
    there is always a last candidate -- there is no "ran out without failing"
    state to handle, and no unreachable assertion standing in for one.
    """
    from haunt.store import _registration_candidates

    for label, discriminator in (
        ("api", None),
        ("api", ""),
        ("api", "/checkouts/api"),
        ("x" * 80, "/checkouts/api"),
        ("", "/checkouts/api"),
    ):
        candidates = _registration_candidates(label, discriminator)
        assert candidates, (label, discriminator)
        assert candidates[0] == label


def test_a_label_that_is_its_own_fork_is_attempted_once_not_twice(
    repo_env, monkeypatch
):
    """The `forked == label` short-circuit, exercised rather than assumed.

    disambiguate_namespace_label() truncates its base to 69 characters before
    appending an 11-character suffix, so a label that is already this
    discriminator's fork and already 80 characters long forks to itself. A
    second attempt at the identical candidate could only fail identically, so
    the candidate list collapses to one entry -- and the refusal below is the
    single-candidate exhaustion path, not the two-candidate one above.
    """
    from haunt import store as store_module

    project = repo_env / "eA" / ("x" * 69)
    squatter = repo_env / "eB" / "squatter10"
    for path in (project, squatter):
        path.mkdir(parents=True)
    _patch_git_contexts(
        monkeypatch, {project: (None, project), squatter: (None, squatter)}
    )
    discriminator = str(project.resolve())
    fixed_point = disambiguate_namespace_label("x" * 69, discriminator)
    assert len(fixed_point) == 80
    assert disambiguate_namespace_label(fixed_point, discriminator) == fixed_point

    _squat(fixed_point, squatter)

    published = store_module._publish_namespace_with_configuration_lock
    attempts: list[str] = []

    def counted(label, *args, **kwargs):
        attempts.append(label)
        return published(label, *args, **kwargs)

    monkeypatch.setattr(
        store_module, "_publish_namespace_with_configuration_lock", counted
    )

    with pytest.raises(NamespaceCollisionError):
        register_namespace_context(fixed_point, discriminator)

    # Without the short-circuit this is [fixed_point, fixed_point].
    assert attempts == [fixed_point]
    assert _binding_count(repo_path=discriminator) == 0
    assert _namespaces_with_two_repository_owners() == {}


# ---------------------------------------------------------------------------
# Explicit selection. A label a human typed is not a label inference derived,
# and only one entry point hands registration both at once
# (`haunt init NAME --repo PATH`). Typing a name must produce that name --
# deliberately pointing two checkouts at one namespace is a supported
# workflow, not a collision to break up -- while everything that *derives* a
# label from a repository stays forkable.
# ---------------------------------------------------------------------------


def test_cli_init_explicit_name_with_repo_binds_rather_than_forking(
    repo_env, monkeypatch
):
    """`haunt init team --repo A` then `--repo B` leaves both on `team`."""
    first = repo_env / "fA" / "checkout"
    second = repo_env / "fB" / "checkout"
    for project in (first, second):
        project.mkdir(parents=True)
    _patch_git_contexts(
        monkeypatch, {first: (None, first), second: (None, second)}
    )
    runner = CliRunner()

    for project in (first, second):
        result = runner.invoke(
            app, ["init", "team10", "--repo", str(project)]
        )
        assert result.exit_code == 0, result.output
        assert "namespace  team10\n" in result.output

    assert [str(row["name"]) for row in _namespace_rows()] == ["team10"]
    assert _binding_count(repo_path=str(first.resolve())) == 1
    assert _binding_count(repo_path=str(second.resolve())) == 1
    # Deliberate sharing is the one place two owners on one namespace is the
    # requested outcome, so it is also the one place this is not the defect.
    shared = _namespaces_with_two_repository_owners()
    assert list(shared.values()) == [
        sorted([str(first.resolve()), str(second.resolve())])
    ]


def test_explicit_label_flag_is_the_only_thing_suppressing_the_fork(
    repo_env, monkeypatch
):
    """The control for the test above: the same label and the same shape of
    repository, derived instead of chosen. A derived label must still fork, or
    the flag would be disabling the race fix rather than narrowing it -- and a
    chosen one must still land on the name that was typed."""
    first = repo_env / "gA" / "checkout"
    second = repo_env / "gB" / "checkout"
    third = repo_env / "gC" / "checkout"
    for project in (first, second, third):
        project.mkdir(parents=True)
    _patch_git_contexts(
        monkeypatch,
        {first: (None, first), second: (None, second), third: (None, third)},
    )

    assert register_namespace_context("team11", str(first.resolve()))[0] == "team11"

    derived, _ = register_namespace_context("team11", str(second.resolve()))
    assert derived == disambiguate_namespace_label(
        "team11", str(second.resolve())
    )
    assert _namespaces_with_two_repository_owners() == {}

    chosen, _ = register_namespace_context(
        "team11", str(third.resolve()), explicit_label=True
    )
    assert chosen == "team11"
    assert _binding_count(repo_path=str(third.resolve())) == 1


def test_explicit_label_does_not_repoint_an_already_bound_repository(
    repo_env, monkeypatch
):
    """The flag narrows the ownership decision; it does not defeat the older
    one-namespace-per-repository rule. A checkout that already has a namespace
    cannot be moved onto another by naming it -- that is
    `haunt namespace reconcile`'s reversible, operator-invoked job."""
    project = repo_env / "iA" / "checkout"
    other = repo_env / "iB" / "checkout"
    for path in (project, other):
        path.mkdir(parents=True)
    _patch_git_contexts(
        monkeypatch, {project: (None, project), other: (None, other)}
    )
    register_namespace_context("mine13", str(project.resolve()))
    register_namespace_context("theirs13", str(other.resolve()))

    with pytest.raises(NamespaceCollisionError):
        register_namespace_context(
            "theirs13", str(project.resolve()), explicit_label=True
        )
    assert _binding_count(repo_path=str(project.resolve())) == 1
    assert _namespaces_with_two_repository_owners() == {}


def test_haunt_namespace_reaches_registration_without_a_repository(
    repo_env, monkeypatch
):
    """HAUNT_NAMESPACE is non-forkable because it names no repository, not
    because of the flag -- assert that, so a future caller cannot quietly
    start passing a repo_path along with it."""
    project = repo_env / "hA" / "checkout"
    project.mkdir(parents=True)
    _patch_git_contexts(monkeypatch, {project: (None, project)})
    monkeypatch.setenv("HAUNT_NAMESPACE", "chosen12")

    assert infer_namespace_context(project) == ("chosen12", None)

    from haunt.claude_hook import _hook_namespace_context
    from haunt.cursor_hook import hook_namespace_context

    payload = {"cwd": str(project)}
    assert _hook_namespace_context(payload) == ("chosen12", None)
    assert hook_namespace_context(payload) == ("chosen12", None)

    # And with no repository named, an already-owned label is never contested.
    monkeypatch.delenv("HAUNT_NAMESPACE")
    register_namespace_context("chosen12", str(project.resolve()))
    monkeypatch.setenv("HAUNT_NAMESPACE", "chosen12")
    ns, repo_path = infer_namespace_context(project)
    assert register_namespace_context(ns, repo_path)[0] == "chosen12"
