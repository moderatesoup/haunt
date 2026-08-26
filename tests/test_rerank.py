"""C12: off-by-default lexical MMR rerank over haunt.recall.recall() output.

Covers the acceptance bar from BACKLOG.md C12 and the task that implemented
it: feature-off is byte-identical to current recall() behavior;
HAUNT_RERANK_ENABLED / HAUNT_RERANK_LAMBDA parse and clamp like their
siblings (HAUNT_TOOL_IO_MAX_CHARS's _tool_io_cap idiom in cursor_hook.py);
the reranker changes hit ordering only when explicitly enabled; and
trusted/trust_reason survive reordering unchanged, because mmr_rerank never
rebuilds a Hit -- it only reorders and truncates the objects recall() (or a
caller here) already produced.
"""

from __future__ import annotations

import pytest

from haunt.recall import Hit, RecallResult
from haunt.rerank import (
    RERANK_ENABLED_ENV,
    RERANK_LAMBDA_DEFAULT,
    RERANK_LAMBDA_ENV,
    apply,
    mmr_rerank,
    recall_with_rerank,
    rerank_enabled,
    rerank_lambda,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _hit(
    n: int,
    *,
    content: str,
    score: float,
    trusted: bool = True,
    memory_id: str | None = None,
) -> Hit:
    """A minimally-populated Hit, mirroring tests/test_recall_budget.py's
    _hit() helper so both files agree on how to build one by hand."""
    return Hit(
        memory_id=memory_id or f"mem-{n:04d}",
        event_id=f"evt-{n:04d}",
        score=score,
        tier="episodic",
        content=content,
        role="tool" if not trusted else "user",
        event_time="2026-08-20T12:00:00.000000+00:00",
        valid_from="2026-08-20T12:00:00.000000+00:00",
        valid_to=None,
        tool_name="Bash" if not trusted else None,
        final_rank=n + 1,
        raw_tool_structure=not trusted,
    )


@pytest.fixture(autouse=True)
def _no_rerank_env(monkeypatch):
    """Every test starts from the documented off default."""
    monkeypatch.delenv(RERANK_ENABLED_ENV, raising=False)
    monkeypatch.delenv(RERANK_LAMBDA_ENV, raising=False)


@pytest.fixture
def rerank_store_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunt"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.delenv("HAUNT_NAMESPACE", raising=False)
    from haunt import embed

    embed.reset()
    yield tmp_path / "haunt"
    embed.reset()


# ---------------------------------------------------------------------------
# Env var parsing/clamping
# ---------------------------------------------------------------------------


def test_rerank_disabled_by_default():
    assert rerank_enabled() is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("YES", True),
        ("yes", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("garbage", False),
        ("  ", False),
    ],
)
def test_rerank_enabled_env_var_parses(monkeypatch, raw, expected):
    monkeypatch.setenv(RERANK_ENABLED_ENV, raw)
    assert rerank_enabled() is expected


def test_rerank_lambda_env_var_parses_and_clamps(monkeypatch):
    monkeypatch.delenv(RERANK_LAMBDA_ENV, raising=False)
    assert rerank_lambda() == RERANK_LAMBDA_DEFAULT

    monkeypatch.setenv(RERANK_LAMBDA_ENV, "not-a-number")
    assert rerank_lambda() == RERANK_LAMBDA_DEFAULT

    monkeypatch.setenv(RERANK_LAMBDA_ENV, "-5")
    assert rerank_lambda() == 0.0

    monkeypatch.setenv(RERANK_LAMBDA_ENV, "5")
    assert rerank_lambda() == 1.0

    monkeypatch.setenv(RERANK_LAMBDA_ENV, "0.3")
    assert rerank_lambda() == pytest.approx(0.3)

    monkeypatch.setenv(RERANK_LAMBDA_ENV, "0")
    assert rerank_lambda() == 0.0

    monkeypatch.setenv(RERANK_LAMBDA_ENV, "1")
    assert rerank_lambda() == 1.0


# ---------------------------------------------------------------------------
# apply(): the off/on seam
# ---------------------------------------------------------------------------


def test_apply_is_a_noop_when_disabled():
    hits = [_hit(0, content="a b c", score=0.9), _hit(1, content="a b c", score=0.5)]
    assert rerank_enabled() is False
    result = apply(hits, k=2)
    assert result is hits  # identical object, not merely equal


def test_apply_returns_equivalent_list_for_a_non_list_sequence_when_disabled():
    hits = (_hit(0, content="a b c", score=0.9),)
    result = apply(hits, k=1)
    assert result == list(hits)


def test_apply_changes_ordering_only_when_enabled(monkeypatch):
    # Three near-duplicates outrank one distinct, lower-relevance hit -- the
    # exact crowding shape the C12 backlog entry names.
    dup_content = "alpha bravo charlie delta"
    hits = [
        _hit(0, content=dup_content, score=0.05, memory_id="dup-1"),
        _hit(1, content=dup_content, score=0.04, memory_id="dup-2"),
        _hit(2, content=dup_content, score=0.03, memory_id="dup-3"),
        _hit(3, content="zulu yankee xray whiskey", score=0.02, memory_id="distinct"),
    ]

    # Disabled: apply() is a pure passthrough, unaware of k, exactly like a
    # caller who never routed through it -- so it returns all 4, not a
    # truncated 3. Truncating to k is recall()'s own job when this feature
    # is off (see recall_with_rerank(), which only ever asks apply() to
    # truncate when the feature is on).
    disabled = apply(hits, k=3)
    assert disabled is hits
    assert [h.memory_id for h in disabled] == ["dup-1", "dup-2", "dup-3", "distinct"]

    monkeypatch.setenv(RERANK_ENABLED_ENV, "1")
    enabled = apply(hits, k=3)
    assert len(enabled) == 3
    assert [h.memory_id for h in enabled] != [h.memory_id for h in disabled][:3]
    assert "distinct" in [h.memory_id for h in enabled]


def test_apply_when_enabled_preserves_and_annotates_execution_evidence(monkeypatch):
    hits = RecallResult(
        [
            _hit(0, content="alpha bravo", score=0.05, memory_id="a"),
            _hit(1, content="alpha bravo", score=0.04, memory_id="b"),
        ],
        modalities={"vector": {"state": "not_run", "reason": "x"}},
    )
    original_execution = hits.execution
    monkeypatch.setenv(RERANK_ENABLED_ENV, "1")

    result = apply(hits, k=2)

    assert isinstance(result, RecallResult)
    # Original modalities evidence is untouched, not dropped or rewritten.
    assert result.execution["modalities"] == original_execution["modalities"]
    assert result.execution["rerank"]["enabled"] is True
    assert result.execution["rerank"]["method"] == "lexical_mmr"
    # apply() must not mutate the caller's original RecallResult in place.
    assert "rerank" not in original_execution


# ---------------------------------------------------------------------------
# mmr_rerank(): pure algorithm
# ---------------------------------------------------------------------------


def test_mmr_rerank_respects_k():
    hits = [_hit(i, content=f"unique-{i}", score=1.0 / (1 + i)) for i in range(6)]
    result = mmr_rerank(hits, k=3, lambda_=0.5)
    assert len(result) == 3
    assert len({h.memory_id for h in result}) == 3
    assert {h.memory_id for h in result} <= {h.memory_id for h in hits}


def test_mmr_rerank_k_larger_than_pool_returns_everything():
    hits = [_hit(i, content=f"unique-{i}", score=1.0 / (1 + i)) for i in range(2)]
    result = mmr_rerank(hits, k=10, lambda_=0.5)
    assert len(result) == 2


@pytest.mark.parametrize("k", [0, -1])
def test_mmr_rerank_non_positive_k_returns_empty(k):
    hits = [_hit(0, content="a", score=1.0)]
    assert mmr_rerank(hits, k=k, lambda_=0.5) == []


def test_mmr_rerank_empty_input_returns_empty():
    assert mmr_rerank([], k=3, lambda_=0.5) == []


def test_mmr_rerank_never_mutates_input_hits():
    hits = [
        _hit(0, content="alpha bravo", score=0.05, memory_id="a"),
        _hit(1, content="alpha bravo", score=0.04, memory_id="b"),
        _hit(2, content="zulu yankee", score=0.03, memory_id="c"),
    ]
    before = [dict(h.__dict__) for h in hits]
    mmr_rerank(hits, k=2, lambda_=0.5)
    after = [dict(h.__dict__) for h in hits]
    assert before == after


def test_mmr_rerank_is_deterministic():
    hits = [
        _hit(0, content="alpha bravo charlie", score=0.05, memory_id="a"),
        _hit(1, content="alpha bravo charlie", score=0.05, memory_id="b"),
        _hit(2, content="delta echo foxtrot", score=0.05, memory_id="c"),
    ]
    repeated_runs = [
        [h.memory_id for h in mmr_rerank(hits, k=3, lambda_=0.5)] for _ in range(5)
    ]
    assert len(set(map(tuple, repeated_runs))) == 1

    # Also invariant to the input list's own order: MMR re-derives selection
    # order from (score, content, memory_id), not from input position.
    shuffled = [hits[2], hits[0], hits[1]]
    assert (
        [h.memory_id for h in mmr_rerank(shuffled, k=3, lambda_=0.5)]
        == repeated_runs[0]
    )


def test_mmr_rerank_diversity_promotes_distinct_item_over_redundant_duplicate():
    """Hand-verified: with three identical-content hits ranked above one
    lexically distinct, lower-score hit, plain top-3 truncation returns only
    the three duplicates. MMR at the default lambda promotes the distinct
    hit to 2nd, ahead of the third duplicate, which falls out of the top 3.
    """
    dup_content = "alpha bravo charlie delta"
    h1 = _hit(0, content=dup_content, score=0.05, memory_id="dup-1")
    h2 = _hit(1, content=dup_content, score=0.04, memory_id="dup-2")
    h3 = _hit(2, content=dup_content, score=0.03, memory_id="dup-3")
    h4 = _hit(3, content="zulu yankee xray whiskey", score=0.02, memory_id="distinct")

    baseline = [h.memory_id for h in [h1, h2, h3, h4][:3]]
    assert baseline == ["dup-1", "dup-2", "dup-3"]

    reranked = mmr_rerank([h1, h2, h3, h4], k=3, lambda_=RERANK_LAMBDA_DEFAULT)
    assert [h.memory_id for h in reranked] == ["dup-1", "distinct", "dup-2"]


def test_mmr_rerank_lambda_one_is_pure_relevance_order():
    """lambda_=1.0 should reduce to "ignore diversity" and keep the original
    score-descending order, since relevance alone then drives every choice.
    """
    hits = [
        _hit(0, content="same same same", score=0.05, memory_id="a"),
        _hit(1, content="same same same", score=0.04, memory_id="b"),
        _hit(2, content="same same same", score=0.03, memory_id="c"),
    ]
    result = mmr_rerank(hits, k=3, lambda_=1.0)
    assert [h.memory_id for h in result] == ["a", "b", "c"]


def test_mmr_rerank_preserves_trust_labelling():
    trusted_dup_1 = _hit(0, content="alpha bravo", score=0.05, memory_id="t1", trusted=True)
    trusted_dup_2 = _hit(1, content="alpha bravo", score=0.045, memory_id="t2", trusted=True)
    untrusted_distinct = _hit(
        2, content="zulu yankee", score=0.04, memory_id="u1", trusted=False
    )
    before = {
        h.memory_id: (h.trusted, h.trust_reason)
        for h in (trusted_dup_1, trusted_dup_2, untrusted_distinct)
    }

    result = mmr_rerank(
        [trusted_dup_1, trusted_dup_2, untrusted_distinct], k=3, lambda_=0.5
    )

    after = {h.memory_id: (h.trusted, h.trust_reason) for h in result}
    assert after == before


# ---------------------------------------------------------------------------
# recall_with_rerank(): the recall.recall() wrapper
# ---------------------------------------------------------------------------


def test_recall_with_rerank_byte_identical_to_recall_when_disabled(rerank_store_env):
    from haunt.recall import recall
    from haunt.store import Store

    with Store("default") as store:
        store.observe("alpha bravo charlie note one", defer_embedding=True)
        store.observe("alpha bravo charlie note two", defer_embedding=True)
        store.observe("delta echo foxtrot distinct note", defer_embedding=True)

    assert rerank_enabled() is False
    direct = recall("alpha bravo charlie", namespace="default", k=3, use_vectors=False)
    wrapped = recall_with_rerank(
        "alpha bravo charlie", namespace="default", k=3, use_vectors=False
    )

    assert [h.as_dict() for h in wrapped] == [h.as_dict() for h in direct]


def test_recall_with_rerank_widens_pool_when_enabled(rerank_store_env, monkeypatch):
    from haunt.recall import recall
    from haunt.store import Store

    dup = "alpha bravo charlie delta echo"
    with Store("default") as store:
        for i in range(4):
            store.observe(f"{dup} restatement {i}", defer_embedding=True)
        # Shares exactly one query token ("alpha") with the dup cluster, so
        # it is a real (if weaker) FTS candidate -- MMR can only ever
        # promote a hit recall() actually returned as a candidate, never one
        # that shares nothing with the query at all.
        store.observe(
            "alpha is also mentioned in a wholly distinct golf hotel fact",
            defer_embedding=True,
        )

    baseline = recall("alpha bravo charlie delta echo", namespace="default", k=3, use_vectors=False)
    baseline_ids = {h.memory_id for h in baseline}

    monkeypatch.setenv(RERANK_ENABLED_ENV, "1")
    monkeypatch.setenv(RERANK_LAMBDA_ENV, "0.5")
    reranked = recall_with_rerank(
        "alpha bravo charlie delta echo", namespace="default", k=3, use_vectors=False
    )

    assert len(reranked) == 3
    # Widening the pool before applying MMR is the whole point: enabling the
    # feature must be able to surface a candidate outside the un-widened
    # top-k, not just reorder exactly the same three hits.
    assert {h.memory_id for h in reranked} != baseline_ids


def test_mmr_diversity_penalty_is_scaled_by_one_minus_lambda():
    """The (1 - lambda_) factor on the diversity term must be load-bearing.

    An adversarial review found that dropping that factor -- computing
    `lambda_*rel - sim` instead of `lambda_*rel - (1-lambda_)*sim` --
    survives every other test in this file, because their fixtures apply
    the (wrongly unscaled) penalty uniformly across candidates, so no
    relative ordering ever flips.

    This fixture separates them. At lambda_=0.9 the diversity weight is
    only 0.1, so a strong near-duplicate of the already-selected top hit
    should still beat a much weaker but unrelated hit. Unscaled, the full
    similarity penalty swamps the relevance gap and the weak hit wins --
    which is the mutant's behavior, not the formula's.
    """
    hits = [
        _hit(0, content="alpha bravo charlie delta", score=1.0, memory_id="top"),
        _hit(1, content="alpha bravo charlie delta", score=0.9, memory_id="dup"),
        _hit(2, content="zulu yankee xray whiskey", score=0.05, memory_id="far"),
    ]

    result = mmr_rerank(hits, k=2, lambda_=0.9)

    assert [h.memory_id for h in result] == ["top", "dup"], (
        "with lambda_=0.9 the diversity penalty is scaled to 0.1 and must not "
        "outweigh a 0.85 relevance gap; picking 'far' means the (1 - lambda_) "
        "factor was dropped"
    )
