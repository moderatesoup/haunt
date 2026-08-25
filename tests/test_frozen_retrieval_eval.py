"""CI gate for the checked-in, deterministic FTS retrieval evaluation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from haunt import frozen_retrieval_eval
from haunt.frozen_retrieval_eval import DEFAULT_BASELINE, evaluate, load_corpus
from haunt.recall import Hit


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_matches_baseline(actual: dict, baseline: dict) -> None:
    """Keep all result fields, including locked trust metadata, exact."""
    assert actual == baseline


def test_frozen_retrieval_baseline():
    """Fail meaningful recall/planner regressions without any model dependency."""
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    actual = evaluate().as_dict()

    # Exact ranked cases catch failures hidden by a superficially stable macro.
    _assert_matches_baseline(actual, baseline)

    # Floors document the intended quality bar as well as the historic lock.
    metrics = actual["metrics"]
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["false_positive_rate"] == 0.0


def test_frozen_retrieval_metadata_lock_rejects_serialized_trust_drift(monkeypatch):
    """A trust-label regression cannot hide behind unchanged logical IDs."""
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    original_as_dict = Hit.as_dict

    def drift_tool_trust(self):
        serialized = original_as_dict(self)
        if serialized["trusted"] is False:
            serialized = {
                **serialized,
                "trusted": True,
                "trust_reason": "ordinary-memory",
            }
        return serialized

    monkeypatch.setattr(Hit, "as_dict", drift_tool_trust)
    actual = evaluate().as_dict()

    assert actual["cases"]["tool_io_trust_metadata"]["returned"] == [
        "tool_io_capture"
    ]
    assert actual["cases"]["tool_io_trust_metadata"]["returned_metadata"] == [
        {
            "id": "tool_io_capture",
            "trusted": True,
            "trust_reason": "ordinary-memory",
        }
    ]
    with pytest.raises(AssertionError):
        _assert_matches_baseline(actual, baseline)


def test_porter_stemming_case_uses_a_morphological_query():
    """The fixture queries a Porter variant absent from the indexed text."""
    corpus = load_corpus()
    record = next(
        record for record in corpus["records"] if record["id"] == "porter_stemming"
    )
    case = next(case for case in corpus["cases"] if case["id"] == "porter_stemming")

    assert re.search(
        rf"\b{re.escape(case['query'])}\b", record["content"], re.IGNORECASE
    ) is None
    assert evaluate().as_dict()["cases"]["porter_stemming"]["returned"] == [
        "porter_stemming"
    ]


def test_evaluate_preserves_caller_home(tmp_path, monkeypatch):
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
    evaluate()

    assert os.environ["HAUNT_HOME"] == str(caller_home)
    assert "HAUNT_FTS_ONLY" not in os.environ
    assert os.environ["HAUNT_EMBED_MODEL"] == "caller-selected-model"
    assert _snapshot(caller_home) == before
    embed.reset()


def test_two_evaluations_are_identical():
    """Fresh temporary homes must produce an identical deterministic lock."""
    assert evaluate().as_dict() == evaluate().as_dict()
