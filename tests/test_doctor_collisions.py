"""doctor reports namespaces that two repositories already share.

The collision guard forks only the registration in front of it -- at mint
time in paths.py, and again inside register_namespace()'s publishing
transaction, which is what settles ownership when two repositories infer
one label before either registers. It never re-derives an existing
namespace: that would re-route it and make the memory stored under it
invisible, which is the harm C1/C2 exists to prevent. So a
pair that collided before the guard landed is still colliding, and nothing
reported it. These tests pin that doctor now does -- as an advisory that
leaves the exit code alone, because healing a pair is `haunt namespace
reconcile`'s operator-invoked, reversible job.

Both real collision shapes are the states a pre-guard haunt actually
produced: separator collapse (github.com/acme/foo-bar against
github.com/acme-foo/bar, both minting github.com-acme-foo-bar) and
remote-less basename (any two checkouts named app). The first repository
registers normally; the second's binding is written directly, because
register_namespace() now settles ownership inside its publishing
transaction and forks rather than adding a second repository to an owned
namespace -- so it can no longer produce the state under test.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import haunt.paths as paths
import haunt.store as store
from haunt.doctor import (
    REQUIRED_CHECKS,
    Check,
    DoctorReport,
    _check_namespace_collisions,
    format_doctor,
)
from haunt.paths import normalize_namespace_label, registry_path
from haunt.store import init_registry, register_namespace
from haunt.util import new_id, now_iso


@pytest.fixture
def registry_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunt-home"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed

    embed.reset()
    init_registry()
    yield tmp_path
    embed.reset()


def _patch_remotes(monkeypatch, remotes: dict[str, str | None]) -> None:
    """Give each checkout path its own origin, for both paths.py and store.py.

    store.py binds _git_repo_context at import time, so registration reads a
    different object than inference does -- see tests/test_repo_binding.py.
    """

    def fake(root):
        resolved = str(Path(root).resolve())
        return remotes.get(resolved), Path(resolved)

    monkeypatch.setattr(paths, "_git_repo_context", fake)
    monkeypatch.setattr(store, "_git_repo_context", fake)


def _record_second_repository(label: str, *, identity: str | None, root: Path) -> None:
    """Bind a second repository to *label* the way a pre-guard haunt did.

    Written directly rather than through register_namespace(), which now
    refuses this state -- see the module docstring. doctor still has to
    report registries that already contain it.
    """
    conn = sqlite3.connect(registry_path())
    conn.row_factory = sqlite3.Row
    try:
        norm = normalize_namespace_label(label)
        namespace_id = conn.execute(
            "SELECT namespace_id FROM namespace_aliases WHERE normalized_label=?",
            (norm,),
        ).fetchone()["namespace_id"]
        now = now_iso()
        conn.execute(
            """INSERT INTO repository_bindings(
                   binding_id,namespace_id,repository_identity,repo_path,label_norm,
                   created_at,updated_at
               ) VALUES (?,?,?,?,?,?,?)""",
            (new_id(), namespace_id, identity, str(root.resolve()), norm, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def _report(check: Check, collisions) -> DoctorReport:
    """A report whose only non-passing check is the one under test."""
    checks = [
        Check(name, True, "ok") for name in REQUIRED_CHECKS if name != check.name
    ]
    return DoctorReport(checks=[*checks, check], collisions=list(collisions))


def test_collision_from_separator_collapse_is_reported(registry_env, monkeypatch):
    foo_bar = registry_env / "foo-bar"
    bar = registry_env / "bar"
    foo_bar.mkdir()
    bar.mkdir()
    _patch_remotes(
        monkeypatch,
        {
            str(foo_bar): "git@github.com:acme/foo-bar.git",
            str(bar): "git@github.com:acme-foo/bar.git",
        },
    )
    register_namespace("github.com-acme-foo-bar", str(foo_bar))
    _record_second_repository(
        "github.com-acme-foo-bar", identity="github.com/acme-foo/bar", root=bar
    )

    check, collisions = _check_namespace_collisions()

    assert [c.namespace for c in collisions] == ["github.com-acme-foo-bar"]
    assert set(collisions[0].repositories) == {
        "github.com/acme/foo-bar",
        "github.com/acme-foo/bar",
    }
    assert check.ok is False
    assert check.advisory is True


def test_collision_from_remote_less_basename_is_reported(registry_env, monkeypatch):
    first = registry_env / "a" / "app"
    second = registry_env / "b" / "app"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _patch_remotes(monkeypatch, {str(first): None, str(second): None})
    register_namespace("app", str(first))
    _record_second_repository("app", identity=None, root=second)

    check, collisions = _check_namespace_collisions()

    assert [c.namespace for c in collisions] == ["app"]
    assert set(collisions[0].repositories) == {str(first), str(second)}
    assert check.advisory is True


def test_collision_is_advisory_and_never_fails_doctor(registry_env, monkeypatch):
    first = registry_env / "a" / "app"
    second = registry_env / "b" / "app"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _patch_remotes(monkeypatch, {str(first): None, str(second): None})
    register_namespace("app", str(first))
    _record_second_repository("app", identity=None, root=second)

    report = _report(*_check_namespace_collisions())

    assert report.ok, report.issues
    assert report.issues == []
    assert report.advisories == [
        "namespaces: 1 of 1 are shared by more than one repository"
    ]
    text = format_doctor(report)
    assert "  namespaces   WARN  1 of 1 are shared by more than one repository" in text
    assert "  ~ namespaces: 1 of 1 are shared by more than one repository" in text
    assert f"      app\n        {first}\n        {second}" in text
    assert "haunt namespace reconcile SOURCE TARGET" in text
    assert "nothing was changed" in text


def test_clean_registry_reports_no_collision(registry_env, monkeypatch):
    """Guards against a vacuously green check: one repo, one namespace."""
    project = registry_env / "solo"
    project.mkdir()
    _patch_remotes(monkeypatch, {str(project): "git@github.com:acme/solo.git"})
    register_namespace("github.com-acme-solo", str(project))

    check, collisions = _check_namespace_collisions()
    report = _report(check, collisions)

    assert collisions == []
    assert check.ok is True
    assert check.advisory is False
    assert check.detail == "1 registered, none shared by more than one repository"
    assert report.advisories == []
    assert "~" not in format_doctor(report)


def test_blank_repo_path_row_is_not_a_collision(registry_env, monkeypatch):
    """A pre-C1 hook/MCP row names no repository, so it owns no label.

    The real-world pair: a blank-repo_path `ironscope` registered by a hook,
    beside the identity-derived namespace the same checkout mints today. One
    repository, two namespaces -- a split, not a shared namespace. Matching
    the blank row's name against the checkout's basename is the
    coincidence-of-labels guess _registered_namespace_for_repo() refuses, and
    reporting it would flag every legacy namespace.

    Runs from the directory holding the checkouts, where that guess resolves:
    treating `ironscope` as a repository there names the very checkout the
    other namespace already owns, which is how the false positive appears.
    """
    project = registry_env / "checkouts" / "ironscope"
    project.mkdir(parents=True)
    monkeypatch.chdir(project.parent)
    _patch_remotes(
        monkeypatch, {str(project): "git@github.com:moderatesoup/ironscope.git"}
    )
    register_namespace("ironscope")
    register_namespace("github.com-moderatesoup-ironscope", str(project))

    check, collisions = _check_namespace_collisions()

    assert collisions == []
    assert check.ok is True
    assert check.detail == "2 registered, none shared by more than one repository"


def test_missing_registry_is_not_a_collision(registry_env, monkeypatch):
    paths.registry_path().unlink()

    check, collisions = _check_namespace_collisions()

    assert collisions == []
    assert check.ok is True
    assert check.detail == "no registry yet"
