"""One bad namespace must not stop the fan-out from reaching the rest.

bootstrap()'s per-namespace loop and store.reembed_all_namespaces() both
walked every registered namespace with no isolation, so the first corrupt,
locked, or missing database aborted the walk and every namespace listed
after it silently kept a stale vector index and an undrained embedding
queue -- degrading those namespaces to FTS-only recall with nothing in the
report to say so. In bootstrap()'s case the abort also skipped everything
downstream of the loop, host installation included. store.list_namespaces()
already guards the same failure shape
(tests/test_list_namespaces_corrupt.py).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fanout_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME, FTS-only -- same pattern as
    tests/test_reembed_counts.py::_fts_only_home. Per-namespace isolation
    is orthogonal to which backend embeds, so no real model is loaded.
    """
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    yield
    embed.reset()


def _seed() -> tuple[str, list[str]]:
    """Populate three namespaces. Returns (first listed, the ones after it).

    Corrupting whatever the registry happens to list first is what makes
    the assertions independent of registry ordering: everything in the
    second element is downstream of the failure.
    """
    from haunt.store import list_namespace_rows, observe

    for name in ("fanout-a", "fanout-b", "fanout-c"):
        observe(f"fanout canary {name}", namespace=name, role="user")
    order = [row["name"] for row in list_namespace_rows()]
    assert len(order) == 3
    return order[0], order[1:]


def _corrupt(name: str) -> None:
    from haunt.paths import namespace_db_path

    namespace_db_path(name).write_text("GARBAGE")


def test_bootstrap_drains_namespaces_listed_after_a_corrupt_one(
    fanout_env, monkeypatch
):
    from haunt.bootstrap import bootstrap, format_report
    from haunt.store import Store

    victim, later = _seed()
    _corrupt(victim)

    visited: list[str] = []
    real_drain = Store.drain_embedding_queue

    def spy(self, *args, **kwargs):
        visited.append(self.name)
        return real_drain(self, *args, **kwargs)

    monkeypatch.setattr(Store, "drain_embedding_queue", spy)
    report = bootstrap("default")

    assert set(later) <= set(visited), (
        f"corrupt {victim} stranded the namespaces after it; "
        f"drained only {visited}"
    )
    rows = {row.get("namespace"): row for row in report["reembed"]}
    assert rows.get(victim, {}).get("error"), (
        f"corrupt {victim} must be reported, not swallowed: {report['reembed']!r}"
    )
    assert f"ns={victim} FAILED" in format_report(report)
    # Everything downstream of the loop still runs.
    assert report["hosts"], "host installation was skipped"


def test_bootstrap_survives_a_namespace_deregistered_mid_walk(fanout_env, monkeypatch):
    """`namespace retire --apply` can deregister a namespace between
    list_namespace_rows() and Store(), which raises UnknownNamespaceError.

    That is a plain ValueError subclass, like NamespaceCollisionError,
    NamespaceMigrationError and NamespacePathError -- naming only some of
    them in the guard let the others abort the whole walk.
    """
    import haunt.store as store_module
    from haunt.bootstrap import bootstrap
    from haunt.store import Store, UnknownNamespaceError

    victim, later = _seed()
    real_store = Store

    def retired_mid_walk(name, *args, **kwargs):
        if name == victim:
            raise UnknownNamespaceError(name)
        return real_store(name, *args, **kwargs)

    monkeypatch.setattr(store_module, "Store", retired_mid_walk)

    visited: list[str] = []
    real_drain = Store.drain_embedding_queue

    def spy(self, *args, **kwargs):
        visited.append(self.name)
        return real_drain(self, *args, **kwargs)

    monkeypatch.setattr(Store, "drain_embedding_queue", spy)
    report = bootstrap("default")

    assert set(later) <= set(visited), (
        f"retiring {victim} stranded the namespaces after it; "
        f"drained only {visited}"
    )
    rows = {row.get("namespace"): row for row in report["reembed"]}
    assert "unknown namespace" in rows.get(victim, {}).get("error", "")
    # Everything downstream of the loop still runs.
    assert report["hosts"], "host installation was skipped"


def test_reembed_all_namespaces_reports_the_corrupt_one_and_keeps_going(fanout_env):
    from haunt.store import reembed_all_namespaces

    victim, later = _seed()
    _corrupt(victim)

    rows = {row["namespace"]: row for row in reembed_all_namespaces()}
    assert rows[victim].get("error"), f"corrupt {victim} must surface an error"
    for name in later:
        assert not rows[name].get("error"), rows[name]
        assert "total" in rows[name], f"{name} never reembedded: {rows[name]!r}"


def test_healthy_namespaces_never_grow_an_error_key(fanout_env):
    """The isolation must not report failure for namespaces that worked."""
    from haunt.store import reembed_all_namespaces

    _seed()
    rows = reembed_all_namespaces()
    assert len(rows) == 3
    for row in rows:
        assert not row.get("error"), row
        assert "total" in row
