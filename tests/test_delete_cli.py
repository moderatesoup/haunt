"""CLI tests for haunt delete — memory_id vs --event-id, no dummy positional."""

from __future__ import annotations

from typer.testing import CliRunner

from haunt.cli import app
from haunt.store import Store, observe

runner = CliRunner()


def test_delete_event_id_without_positional_missing_event(lore_env):
    """haunt delete --event-id <eid> --yes must not require a dummy memory_id.

    Old required-argument behavior: Missing argument 'memory_id', exit=2.
    """
    result = runner.invoke(app, ["delete", "--event-id", "fake-event-123", "--yes"])
    combined = f"{result.stdout}{result.stderr}{result.output}"
    assert "Missing argument" not in combined
    assert result.exit_code == 1
    assert "no memories for event fake-event-123" in result.stdout


def test_delete_by_event_id_purges_without_positional(lore_env):
    r = observe("DELETE-EVENT-ID-CANARY unique phrase", namespace="default")
    result = runner.invoke(app, ["delete", "--event-id", r.event_id, "--yes"])
    assert result.exit_code == 0, result.output
    assert "purged" in result.stdout
    with Store("default") as st:
        gone = st.conn.execute(
            "SELECT id FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert gone is None


def test_delete_by_memory_id_still_works(lore_env):
    r = observe("DELETE-MEMORY-ID-CANARY unique phrase", namespace="default")
    result = runner.invoke(app, ["delete", r.memory_id, "--yes"])
    assert result.exit_code == 0, result.output
    assert "purged" in result.stdout
    with Store("default") as st:
        gone = st.conn.execute(
            "SELECT id FROM memories WHERE id=?", (r.memory_id,)
        ).fetchone()
        assert gone is None


def test_delete_rejects_both_memory_id_and_event_id(lore_env):
    result = runner.invoke(
        app, ["delete", "real-memory-id", "--event-id", "real-event-id", "--yes"]
    )
    combined = f"{result.stdout}{result.stderr}{result.output}"
    assert result.exit_code == 2
    assert "not both" in combined


def test_delete_requires_memory_id_or_event_id(lore_env):
    result = runner.invoke(app, ["delete", "--yes"])
    combined = f"{result.stdout}{result.stderr}{result.output}"
    assert result.exit_code == 2
    assert "memory_id or --event-id is required" in combined
