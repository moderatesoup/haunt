"""#48 MCP namespace authority and destructive-tool gating."""

from __future__ import annotations

import json
import sqlite3

import pytest


def _seed_home(home, monkeypatch, *, admin=False, allow_purge=False):
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "alpha")
    if admin:
        monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    else:
        monkeypatch.delenv("HAUNT_MCP_ADMIN", raising=False)
    if allow_purge:
        monkeypatch.setenv("HAUNT_MCP_ALLOW_PURGE", "1")
    else:
        monkeypatch.delenv("HAUNT_MCP_ALLOW_PURGE", raising=False)

    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import Store, init_registry

    embed.reset()
    ensure_layout()
    init_registry()
    with Store("alpha") as store:
        alpha = store.observe("ALPHA-ONLY-CANARY")
    with Store("beta") as store:
        beta = store.observe("BETA-SECRET-CANARY")
    return alpha, beta


def test_remote_identity_includes_host_owner_and_repo():
    from haunt.paths import namespace_for_repo_identity, repository_identity

    https = repository_identity("https://github.com/company-a/api.git")
    ssh = repository_identity("git@github.com:company-a/api.git")
    other = repository_identity("git@github.com:company-b/api.git")
    assert https == ssh == "github.com/company-a/api"
    assert other == "github.com/company-b/api"
    assert namespace_for_repo_identity(https) == "github.com-company-a-api"
    assert namespace_for_repo_identity(https) != namespace_for_repo_identity(other)


def test_long_remote_identity_keeps_hash_suffix():
    from haunt.paths import namespace_for_repo_identity

    first = "git.example.com/" + "a" * 90 + "/api"
    second = "git.example.com/" + "a" * 89 + "b/api"
    one = namespace_for_repo_identity(first)
    two = namespace_for_repo_identity(second)
    assert len(one) <= 80
    assert len(two) <= 80
    assert one != two


def test_inference_preserves_registered_legacy_remote(tmp_path, monkeypatch):
    home = tmp_path / "haunthome"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt.paths import registry_path
    from haunt.store import init_registry, register_namespace
    import haunt.paths as paths

    init_registry()
    register_namespace("legacy-api")
    conn = sqlite3.connect(registry_path())
    conn.execute(
        "UPDATE namespaces SET repo_path=? WHERE name=?",
        ("https://github.com/company-a/api", "legacy-api"),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        paths,
        "_git_repo_context",
        lambda root: ("git@github.com:company-a/api.git", project),
    )

    assert paths.infer_namespace(project) == "legacy-api"


def test_new_repo_inference_uses_canonical_remote(tmp_path, monkeypatch):
    home = tmp_path / "haunthome"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    import haunt.paths as paths

    monkeypatch.setattr(
        paths,
        "_git_repo_context",
        lambda root: ("https://git.example.com/team/api.git", project),
    )
    assert paths.infer_namespace(project) == "git.example.com-team-api"


@pytest.fixture
def authority_home(tmp_path, monkeypatch):
    alpha, beta = _seed_home(tmp_path / "ordinary", monkeypatch)
    yield tmp_path, alpha, beta
    from haunt import embed

    embed.reset()


def test_ordinary_mcp_cannot_read_or_write_other_namespace(
    authority_home, monkeypatch
):
    _, _, _ = authority_home
    from haunt.mcp_server import memory_observe, memory_recall
    from haunt.store import Store

    own = json.loads(memory_recall(query="ALPHA-ONLY-CANARY"))
    assert own.get("hits")
    assert own["namespace"] == "alpha"

    denied = json.loads(
        memory_recall(query="BETA-SECRET-CANARY", namespace="beta")
    )
    assert denied["ok"] is False
    assert "denied" in denied["error"]
    assert "BETA-SECRET-CANARY" not in json.dumps(denied)

    before = None
    with Store("beta", create=False) as store:
        before = store.stats()["events"]
    write = json.loads(
        memory_observe(text="CROSS-WRITE-CANARY", namespace="beta")
    )
    assert write["ok"] is False
    with Store("beta", create=False) as store:
        assert store.stats()["events"] == before


def test_binding_and_capabilities_are_immutable_for_process(authority_home, monkeypatch):
    import haunt.mcp_server as mcp

    seen: list[str | None] = []
    real_list = mcp.list_namespaces

    def tracked_list(*, only=None):
        seen.append(only)
        return real_list(only=only)

    monkeypatch.setattr(mcp, "list_namespaces", tracked_list)

    first = json.loads(mcp.memory_namespaces())
    assert first["bound_namespace"] == "alpha"
    assert first["admin"] is False
    assert [row["name"] for row in first["namespaces"]] == ["alpha"]
    assert seen == ["alpha"], "ordinary listing must not open every namespace"

    monkeypatch.setenv("HAUNT_NAMESPACE", "beta")
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    second = json.loads(mcp.memory_namespaces())
    assert second["bound_namespace"] == "alpha"
    assert second["admin"] is False
    denied = json.loads(mcp.memory_recall(query="secret", namespace="beta"))
    assert denied["ok"] is False


def test_mcp_purge_is_disabled_and_annotated_by_default(authority_home):
    _, alpha, _ = authority_home
    from haunt.mcp_server import memory_purge, server
    from haunt.store import Store

    result = json.loads(memory_purge(memory_id=alpha.memory_id))
    assert result["ok"] is False
    assert "disabled" in result["error"]
    with Store("alpha", create=False) as store:
        assert store.get_memory(alpha.memory_id) is not None

    tool = next(
        tool for tool in server._tool_manager.list_tools()
        if tool.name == "memory_purge"
    )
    assert tool.annotations.destructive_hint is True


def test_explicit_admin_can_cross_namespaces_but_not_purge(tmp_path, monkeypatch):
    _, beta = _seed_home(tmp_path / "admin", monkeypatch, admin=True)
    from haunt.mcp_server import memory_namespaces, memory_purge, memory_recall
    from haunt.store import Store

    recalled = json.loads(
        memory_recall(query="BETA-SECRET-CANARY", namespace="beta")
    )
    assert recalled.get("hits")
    listed = json.loads(memory_namespaces())
    assert listed["admin"] is True
    assert {row["name"] for row in listed["namespaces"]} == {"alpha", "beta"}

    denied = json.loads(memory_purge(beta.memory_id, namespace="beta"))
    assert denied["ok"] is False
    assert "disabled" in denied["error"]
    with Store("beta", create=False) as store:
        assert store.get_memory(beta.memory_id) is not None


def test_explicit_purge_capability_is_bound_to_own_namespace(tmp_path, monkeypatch):
    alpha, beta = _seed_home(
        tmp_path / "purge", monkeypatch, allow_purge=True
    )
    from haunt.mcp_server import memory_purge
    from haunt.store import Store

    cross = json.loads(memory_purge(beta.memory_id, namespace="beta"))
    assert cross["ok"] is False
    assert "denied" in cross["error"]

    own = json.loads(memory_purge(alpha.memory_id))
    assert own["ok"] is True
    with Store("alpha", create=False) as store:
        assert store.get_memory(alpha.memory_id) is None
    with Store("beta", create=False) as store:
        assert store.get_memory(beta.memory_id) is not None
