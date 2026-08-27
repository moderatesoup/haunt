"""CLI tests for haunt delete — memory_id vs --event-id, no dummy positional."""

from __future__ import annotations

from typer.testing import CliRunner

from haunt.cli import app
from haunt.store import Store, observe

runner = CliRunner()


def test_delete_event_id_without_positional_missing_event(haunt_env):
    """haunt delete --event-id <eid> --yes must not require a dummy memory_id.

    Old required-argument behavior: Missing argument 'memory_id', exit=2.
    """
    result = runner.invoke(app, ["delete", "--event-id", "fake-event-123", "--yes", "-n", "default"])
    combined = f"{result.stdout}{result.stderr}{result.output}"
    assert "Missing argument" not in combined
    assert result.exit_code == 1
    assert "no memories for event fake-event-123" in result.stdout


def test_delete_by_event_id_purges_without_positional(haunt_env):
    r = observe("DELETE-EVENT-ID-CANARY unique phrase", namespace="default")
    result = runner.invoke(
        app, ["delete", "--event-id", r.event_id, "--yes", "-n", "default"]
    )
    assert result.exit_code == 0, result.output
    assert "purged" in result.stdout
    with Store("default") as st:
        gone = st.conn.execute(
            "SELECT id FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert gone is None


def test_delete_by_memory_id_still_works(haunt_env):
    r = observe("DELETE-MEMORY-ID-CANARY unique phrase", namespace="default")
    result = runner.invoke(app, ["delete", r.memory_id, "--yes", "-n", "default"])
    assert result.exit_code == 0, result.output
    assert "purged" in result.stdout
    with Store("default") as st:
        gone = st.conn.execute(
            "SELECT id FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert gone is None


def test_delete_rejects_both_memory_id_and_event_id(haunt_env):
    result = runner.invoke(
        app, ["delete", "real-memory-id", "--event-id", "real-event-id", "--yes"]
    )
    combined = f"{result.stdout}{result.stderr}{result.output}"
    assert result.exit_code == 2
    assert "not both" in combined


def test_delete_requires_memory_id_or_event_id(haunt_env):
    result = runner.invoke(app, ["delete", "--yes"])
    combined = f"{result.stdout}{result.stderr}{result.output}"
    assert result.exit_code == 2
    assert "memory_id or --event-id is required" in combined


def _canary_counts(db_path, needle: bytes) -> int:
    """Raw occurrences of needle across the namespace file and its sidecars."""
    from pathlib import Path

    paths = (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm"))
    return sum(p.read_bytes().count(needle) for p in paths if p.exists())


def test_delete_by_event_id_rebuilds_once_and_still_erases(haunt_env, monkeypatch):
    """One event's memories must cost one whole-file rebuild, not one each.

    Every purge ends in a VACUUM and a truncating checkpoint under a
    cross-process lock, so rebuilding per memory blocks every writer in every
    namespace once per memory. Deferring the rebuild must leave the erasure
    guarantee intact by the end of the command.
    """
    from haunt import store as store_module

    canaries = [f"eventfanoutcanary{i}zq" for i in range(3)]
    r = observe(f"first {canaries[0]} secret", namespace="default")
    with Store("default") as st:
        # observe() materializes one memory per event; the several-memories
        # case purge handles is arranged directly.
        for i, canary in enumerate(canaries[1:], start=1):
            extra_id = f"fanout-memory-{i}"
            st.conn.execute(
                """
                INSERT INTO memories(
                    id, event_id, tier, content, embedding, valid_from, valid_to,
                    created_at
                )
                SELECT ?, event_id, tier, ?, embedding, valid_from, valid_to,
                       created_at
                FROM memories WHERE id=?
                """,
                (extra_id, f"extra {canary} secret", r.memory_id),
            )
            st.conn.execute(
                "INSERT INTO memories_fts(id, content) VALUES (?, ?)",
                (extra_id, f"extra {canary} secret"),
            )
        st.conn.commit()
        # Churn: index merges and page splits leave older copies of a row
        # behind, which only the rebuild removes.
        for i in range(40):
            st.observe(f"filler note {i} about deployments and databases")
        db_path = st.db_path

    rebuilds = 0
    real_rebuild = store_module.Store.overwrite_erased_pages

    def counting_rebuild(self):
        nonlocal rebuilds
        rebuilds += 1
        return real_rebuild(self)

    monkeypatch.setattr(store_module.Store, "overwrite_erased_pages", counting_rebuild)

    result = runner.invoke(
        app, ["delete", "--event-id", r.event_id, "--yes", "-n", "default"]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.count("purged") == len(canaries)
    assert rebuilds == 1, f"one rebuild per command, not per memory (got {rebuilds})"
    assert "bytes_overwritten=True" in result.stdout

    for canary in canaries:
        assert _canary_counts(db_path, canary.encode()) == 0, (
            f"{canary} survives on disk after the command"
        )
