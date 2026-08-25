"""CI gate for the checked-in, deterministic FTS retrieval evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from haunt.frozen_retrieval_eval import DEFAULT_BASELINE, evaluate


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_frozen_retrieval_baseline():
    """Fail meaningful recall/planner regressions without any model dependency."""
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    actual = evaluate().as_dict()

    # Exact ranked cases catch failures hidden by a superficially stable macro.
    assert actual == baseline

    # Floors document the intended quality bar as well as the historic lock.
    metrics = actual["metrics"]
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["false_positive_rate"] == 0.0


def test_evaluate_preserves_caller_home_and_is_repeatable(tmp_path, monkeypatch):
    """The public evaluator never mutates the caller's home or environment."""
    caller_home = tmp_path / "caller-home"
    monkeypatch.setenv("HAUNT_HOME", str(caller_home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    from haunt import embed
    from haunt.paths import namespace_db_path, registry_path
    from haunt.store import Store

    embed.reset()
    with Store("caller") as store:
        store.observe("caller memory must survive evaluation", defer_embedding=True)
    before = _snapshot(caller_home)
    assert namespace_db_path("caller").is_file()
    assert registry_path().is_file()

    # Evaluate must supply its own FTS-only environment, then restore these
    # intentionally non-evaluation values exactly.
    monkeypatch.delenv("HAUNT_FTS_ONLY")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "caller-selected-model")
    first = evaluate().as_dict()
    second = evaluate().as_dict()

    assert first == second
    assert os.environ["HAUNT_HOME"] == str(caller_home)
    assert "HAUNT_FTS_ONLY" not in os.environ
    assert os.environ["HAUNT_EMBED_MODEL"] == "caller-selected-model"
    assert _snapshot(caller_home) == before
    embed.reset()
