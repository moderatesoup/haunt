"""C12 measurement harness: tests for haunt.rerank_eval.

This is the pass/fail gate for the harness itself (structure, isolation,
determinism), plus assertions that lock in what was actually measured on
the checked-in fixture (tests/fixtures/rerank_eval/corpus.json) so a future
change to rerank.mmr_rerank or this corpus cannot silently flip the
reported conclusion without a visible test failure: clear_top1 queries are
completely unaffected by reranking; ambiguous queries see a real recall@k
gain and a large redundancy drop, at the cost of a small, bounded amount of
MRR (see test_reranking_improves_ambiguous_recall_with_a_small_mrr_tradeoff_on_the_fixture
for why that tradeoff is expected, not a bug).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from haunt import rerank_eval
from haunt.rerank_eval import DEFAULT_CORPUS, evaluate, load_corpus


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_load_corpus_round_trips():
    corpus = load_corpus()
    assert corpus["schema_version"] == rerank_eval.SCHEMA_VERSION
    assert corpus["records"]
    assert corpus["cases"]


def test_load_corpus_rejects_bad_schema_version(tmp_path):
    bad = tmp_path / "corpus.json"
    bad.write_text(json.dumps({"schema_version": 999, "records": [], "cases": []}))
    with pytest.raises(ValueError):
        load_corpus(bad)


def test_load_corpus_rejects_unknown_case_class(tmp_path):
    corpus = json.loads(DEFAULT_CORPUS.read_text(encoding="utf-8"))
    corpus["cases"][0]["class"] = "not_a_real_class"
    bad = tmp_path / "corpus.json"
    bad.write_text(json.dumps(corpus))
    with pytest.raises(ValueError):
        load_corpus(bad)


def test_evaluate_is_deterministic():
    first = evaluate().as_dict()
    second = evaluate().as_dict()
    assert first == second


def test_evaluate_preserves_caller_home(tmp_path, monkeypatch):
    """The public evaluator never mutates the caller's home or environment,
    mirroring frozen_retrieval_eval.py's own isolation test."""
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

    monkeypatch.delenv("HAUNT_FTS_ONLY")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "caller-selected-model")
    evaluate()

    assert os.environ["HAUNT_HOME"] == str(caller_home)
    assert "HAUNT_FTS_ONLY" not in os.environ
    assert os.environ["HAUNT_EMBED_MODEL"] == "caller-selected-model"
    assert _snapshot(caller_home) == before
    embed.reset()


def test_report_covers_both_arms_and_both_query_classes():
    report = evaluate().as_dict()["report"]
    assert set(report.keys()) == {"baseline", "reranked"}
    for arm in report.values():
        assert set(arm.keys()) == {"overall", "clear_top1", "ambiguous"}
        for bucket in arm.values():
            assert "recall_at_k" in bucket
            assert "mrr" in bucket


def test_every_case_is_classified_clear_top1_or_ambiguous():
    cases = evaluate().as_dict()["cases"]
    assert {c["class"] for c in cases.values()} == {"clear_top1", "ambiguous"}


# ---------------------------------------------------------------------------
# What was actually measured on the checked-in fixture. These lock the
# reported conclusion, not an implementation detail: see the deliverable
# report for the exact numbers.
# ---------------------------------------------------------------------------


def test_clear_top1_cases_are_unaffected_by_rerank_on_the_fixture():
    report = evaluate().as_dict()["report"]
    assert report["baseline"]["clear_top1"] == report["reranked"]["clear_top1"]
    assert report["reranked"]["clear_top1"]["recall_at_k"] == 1.0
    assert report["reranked"]["clear_top1"]["mrr"] == 1.0


def test_reranking_improves_ambiguous_recall_with_a_small_mrr_tradeoff_on_the_fixture():
    """The honest, measured shape of the effect at the default lambda: MMR
    trades a little MRR for a large recall@k gain and a large redundancy
    drop. Diversity reordering can push the *first* relevant hit (what MRR
    scores) down a rank even while pulling a *second* relevant hit into the
    window (what recall@k scores) -- textbook MMR behavior, not a bug. See
    the deliverable report for the full lambda sweep this is drawn from.
    """
    report = evaluate().as_dict()["report"]
    baseline = report["baseline"]["ambiguous"]
    reranked = report["reranked"]["ambiguous"]
    assert reranked["recall_at_k"] > baseline["recall_at_k"]
    # MRR is not free: bound how much it may give up, rather than pretending
    # it never regresses.
    assert baseline["mrr"] - reranked["mrr"] <= 0.05


def test_reranking_reduces_redundancy_on_ambiguous_cases_on_the_fixture():
    report = evaluate().as_dict()["report"]
    baseline = report["baseline"]["ambiguous"]["mean_redundancy_rate"]
    reranked = report["reranked"]["ambiguous"]["mean_redundancy_rate"]
    assert reranked < baseline
