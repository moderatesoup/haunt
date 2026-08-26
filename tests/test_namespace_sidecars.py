"""Safety coverage for SQLite registry and namespace sidecar files."""

from __future__ import annotations

from pathlib import Path

import pytest

from haunt.paths import (
    NamespacePathError,
    namespace_db_path,
    registry_path,
    tighten_db_files,
)
from haunt.store import (
    NamespaceCollisionError,
    Store,
    init_registry,
    namespace_exists,
    register_namespace,
)


@pytest.fixture
def sidecar_home(tmp_path, monkeypatch):
    home = tmp_path / "haunt-home"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    init_registry()
    return home


def _victim(tmp_path: Path) -> tuple[Path, bytes, int]:
    victim = tmp_path / "external-victim"
    content = b"EXTERNAL-SIDECAR-VICTIM"
    victim.write_bytes(content)
    victim.chmod(0o640)
    return victim, content, victim.stat().st_mode & 0o777


def _redirect(path: Path, victim: Path, kind: str) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()
    if kind == "symlink":
        path.symlink_to(victim)
    elif kind == "hardlink":
        path.hardlink_to(victim)
    else:
        path.mkdir()


def _assert_victim_untouched(victim: Path, content: bytes, mode: int) -> None:
    assert victim.read_bytes() == content
    assert victim.stat().st_mode & 0o777 == mode


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory"])
def test_registry_init_rejects_ambiguous_sidecars_without_touching_victim(
    tmp_path, monkeypatch, suffix, kind
):
    home = tmp_path / f"registry-{kind}-{suffix[1:]}"
    (home / "namespaces").mkdir(parents=True)
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    victim, content, mode = _victim(tmp_path)
    attack = Path(str(registry_path()) + suffix)
    _redirect(attack, victim, kind)

    with pytest.raises(NamespacePathError, match="sidecar"):
        init_registry()
    assert not registry_path().exists()
    assert attack.exists() or attack.is_symlink()
    _assert_victim_untouched(victim, content, mode)


@pytest.mark.parametrize("phase", ["open", "pragma"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_registry_claim_rejects_exact_sidecar_swap_before_sqlite_use(
    tmp_path, monkeypatch, phase, kind
):
    home = tmp_path / f"registry-hook-{phase}-{kind}"
    (home / "namespaces").mkdir(parents=True)
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    victim, content, mode = _victim(tmp_path)
    attack = Path(str(home / "registry.db") + "-wal")
    import haunt.store as store_module

    fired = False

    def replace_claim(path: Path) -> None:
        nonlocal fired
        if path != home / "registry.db" or fired:
            return
        fired = True
        _redirect(attack, victim, kind)

    hook = (
        "_sqlite_sidecar_open_hook"
        if phase == "open"
        else "_sqlite_sidecar_pragma_hook"
    )
    monkeypatch.setattr(store_module, hook, replace_claim)
    with pytest.raises(NamespacePathError, match="sidecar physical identity changed"):
        init_registry()
    assert fired
    assert attack.exists() or attack.is_symlink()
    _assert_victim_untouched(victim, content, mode)


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory"])
def test_mapped_store_rejects_ambiguous_sidecars_without_touching_victim(
    sidecar_home, tmp_path, suffix, kind
):
    db = register_namespace("sidecar-owner")
    victim, content, mode = _victim(tmp_path)
    attack = Path(str(db) + suffix)
    _redirect(attack, victim, kind)

    with pytest.raises(NamespacePathError, match="sidecar"):
        Store("sidecar-owner", create=False)
    assert attack.exists() or attack.is_symlink()
    _assert_victim_untouched(victim, content, mode)


@pytest.mark.parametrize("phase", ["open", "pragma"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_sidecar_swap_at_exact_open_boundary_fails_before_victim_use(
    sidecar_home, tmp_path, monkeypatch, phase, kind
):
    db = register_namespace("hook-owner")
    victim, content, mode = _victim(tmp_path)
    attack = Path(str(db) + "-wal")
    import haunt.store as store_module

    fired = False

    def replace_claim(path: Path) -> None:
        nonlocal fired
        if path != db or fired:
            return
        fired = True
        _redirect(attack, victim, kind)

    hook = (
        "_sqlite_sidecar_open_hook"
        if phase == "open"
        else "_sqlite_sidecar_pragma_hook"
    )
    monkeypatch.setattr(store_module, hook, replace_claim)
    with pytest.raises(NamespacePathError, match="sidecar physical identity changed"):
        Store("hook-owner", create=False)
    assert fired
    assert attack.exists() or attack.is_symlink()
    _assert_victim_untouched(victim, content, mode)


@pytest.mark.parametrize("surface", ["registry", "namespace"])
@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_hardlink_swap_after_validation_fails_before_first_pragma(
    tmp_path, monkeypatch, surface, suffix
):
    home = tmp_path / f"verified-hook-{surface}-{suffix[1:]}"
    (home / "namespaces").mkdir(parents=True)
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    victim, content, mode = _victim(tmp_path)
    victim_inode = int(victim.stat().st_ino)
    import haunt.store as store_module

    if surface == "registry":
        target = registry_path()
        action = init_registry
    else:
        init_registry()
        target = register_namespace("verified-sidecar-owner")

        def action():
            return Store("verified-sidecar-owner", create=False)
    attack = Path(str(target) + suffix)
    fired = False

    def replace_after_validation(path: Path) -> None:
        nonlocal fired
        if path != target or fired:
            return
        fired = True
        assert store_module._sqlite_configuration_lock_held()
        _redirect(attack, victim, "hardlink")

    monkeypatch.setattr(
        store_module, "_sqlite_sidecar_verified_hook", replace_after_validation
    )
    with pytest.raises(NamespacePathError, match="sidecar physical identity changed"):
        action()
    assert fired
    assert attack.exists() and not attack.is_symlink()
    assert int(victim.stat().st_ino) == victim_inode
    _assert_victim_untouched(victim, content, mode)


def test_live_wal_shm_and_rollback_journal_reopen_safely(sidecar_home):
    db = register_namespace("recovery")
    with Store("recovery", create=False) as writer:
        writer.observe("LIVE-WAL-CANARY")
        wal = Path(str(db) + "-wal")
        shm = Path(str(db) + "-shm")
        assert wal.is_file()
        assert shm.is_file()
        with Store("recovery", create=False) as reader:
            assert reader.stats()["events"] == 1

    journal = Path(str(db) + "-journal")
    journal.touch(mode=0o600)
    with Store("recovery", create=False) as reopened:
        assert reopened.stats()["events"] == 1
    assert journal.is_file()
    assert journal.stat().st_nlink == 1


def test_live_registry_wal_and_shm_are_accepted(sidecar_home):
    import sqlite3

    db = register_namespace("registry-live-wal")
    conn = sqlite3.connect(registry_path())
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "UPDATE namespaces SET updated_at=updated_at WHERE name=?",
        ("registry-live-wal",),
    )
    conn.commit()
    assert Path(str(registry_path()) + "-wal").is_file()
    assert Path(str(registry_path()) + "-shm").is_file()
    assert namespace_db_path("registry-live-wal") == db
    init_registry()
    conn.close()


def test_tighten_never_follows_a_sidecar_symlink(sidecar_home, tmp_path):
    db = register_namespace("tighten-owner")
    victim, content, mode = _victim(tmp_path)
    attack = Path(str(db) + "-shm")
    attack.symlink_to(victim)

    with pytest.raises(NamespacePathError, match="sidecar"):
        tighten_db_files(db)
    _assert_victim_untouched(victim, content, mode)


def test_fresh_claim_leaves_ambiguous_replacement_and_removes_only_owned_files(
    sidecar_home, tmp_path, monkeypatch
):
    victim, content, mode = _victim(tmp_path)
    import haunt.store as store_module

    replaced: list[Path] = []

    def replace_hidden_wal(path: Path) -> None:
        if not path.name.startswith(".haunt-claim-") or replaced:
            return
        attack = Path(str(path) + "-wal")
        _redirect(attack, victim, "symlink")
        replaced.append(attack)

    monkeypatch.setattr(
        store_module, "_sqlite_sidecar_open_hook", replace_hidden_wal
    )
    with pytest.raises(NamespacePathError, match="sidecar physical identity changed"):
        Store("fresh-sidecar-failure")
    assert not namespace_exists("fresh-sidecar-failure")
    assert len(replaced) == 1
    assert replaced[0].is_symlink()
    assert not Path(str(replaced[0])[: -len("-wal")]).exists()
    _assert_victim_untouched(victim, content, mode)


def test_unmapped_crash_sidecar_is_not_deleted_or_replaced(sidecar_home):
    target = sidecar_home / "namespaces" / "crash-leftover.db"
    leftover = Path(str(target) + "-journal")
    leftover.write_bytes(b"CRASH-LEFTOVER")
    before = leftover.read_bytes()

    with pytest.raises(NamespaceCollisionError, match="unmapped SQLite sidecar"):
        Store("crash-leftover")
    assert leftover.read_bytes() == before
    assert not target.exists()
