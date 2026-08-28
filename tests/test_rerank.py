"""C12: off-by-default lexical MMR rerank, wired into haunt.recall.recall().

Covers the acceptance bar from BACKLOG.md C12 and the task that implemented
it: feature-off leaves recall() output, its ranking provenance and its row
reads exactly as they were before this stage existed; HAUNT_RERANK_ENABLED /
HAUNT_RERANK_LAMBDA parse and clamp like their siblings
(HAUNT_TOOL_IO_MAX_CHARS's _tool_io_cap idiom in cursor_hook.py); the
reranker changes hit ordering only when explicitly enabled; the order it
reports is the order it returned; the response budget truncates what the
reranker chose rather than the other way round; and trusted/trust_reason
survive reordering unchanged, because mmr_rerank never rebuilds a Hit -- it
only reorders and truncates the objects recall() already produced.
"""

from __future__ import annotations

import pytest

from haunt.recall import Hit, RecallResult
from haunt.rerank import (
    RERANK_ENABLED_ENV,
    RERANK_LAMBDA_DEFAULT,
    RERANK_LAMBDA_ENV,
    RERANK_METHOD,
    apply,
    candidate_pool,
    mmr_rerank,
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


@pytest.fixture
def rerank_mcp_env(rerank_store_env, monkeypatch):
    """rerank_store_env plus the registry/authority mcp_server needs."""
    from haunt import mcp_server
    from haunt.paths import ensure_layout
    from haunt.store import init_registry, register_namespace

    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    monkeypatch.delenv("HAUNT_RECALL_MAX_CHARS", raising=False)
    mcp_server._MCP_AUTHORITY = None
    ensure_layout()
    init_registry()
    register_namespace("default")
    yield rerank_store_env
    mcp_server._MCP_AUTHORITY = None


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
    # is off (recall() slices candidate_pool(k) == k before calling here).
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
# recall(): the wired stage
# ---------------------------------------------------------------------------

CROWDED_QUERY = "alpha bravo charlie delta echo"
RRF_ORDERING = {"primary": "rrf_score_desc", "ties": "memory_id_asc"}


def _seed_crowded_corpus(filler: str = "") -> None:
    """Four near-duplicates plus one distinct row sharing a single query token.

    MMR can only trade one fused candidate for another, so the distinct row
    has to be a real (if weak) FTS candidate rather than an unrelated one.
    """
    from haunt.store import Store

    with Store("default") as store:
        for i in range(4):
            store.observe(
                f"{CROWDED_QUERY} restatement {i} {filler}", defer_embedding=True
            )
        store.observe(
            f"alpha is also mentioned in a wholly distinct golf hotel fact {filler}",
            defer_embedding=True,
        )


def test_disabled_recall_is_unchanged_and_reports_fusion_ordering(rerank_store_env):
    """Default-off is inert: same order, same provenance, no rerank evidence."""
    from haunt.recall import recall

    _seed_crowded_corpus()
    assert rerank_enabled() is False

    hits = recall(CROWDED_QUERY, namespace="default", k=3, use_vectors=False)

    assert list(hits) == sorted(hits, key=lambda h: (-h.score, h.memory_id))
    assert all(hit.rerank_stage is None for hit in hits)
    assert "rerank" not in hits.execution
    for position, hit in enumerate(hits, start=1):
        explanation = hit.as_dict()["explanation"]
        assert explanation["final_rank"] == position
        assert explanation["ordering"] == RRF_ORDERING
        # The fusion rank IS final_rank here, so repeating it would be noise.
        assert "rrf_rank" not in explanation


def test_disabled_recall_reads_no_more_rows_than_it_returns(
    rerank_store_env, monkeypatch
):
    """Off must cost nothing: no MMR call, and no widened pool materialized."""
    from haunt import rerank as rerank_module
    from haunt.recall import recall
    from haunt.store import Store

    _seed_crowded_corpus()

    def _explode(*args, **kwargs):
        raise AssertionError("mmr_rerank ran with HAUNT_RERANK_ENABLED unset")

    monkeypatch.setattr(rerank_module, "mmr_rerank", _explode)

    statements: list[str] = []
    with Store("default") as store:
        store.conn.set_trace_callback(statements.append)
        hits = recall(
            CROWDED_QUERY, namespace="default", k=3, use_vectors=False, store=store
        )
        store.conn.set_trace_callback(None)

    assert candidate_pool(3) == 3
    assert len(hits) == 3
    # One per-hit row read per returned hit. Enabled, the same k=3 answer
    # would cost RERANK_POOL of these reads instead.
    row_read = "SELECT m.id, m.event_id, m.tier, m.content"
    assert sum(row_read in sql for sql in statements) == 3


def test_enabled_recall_widens_the_pool_and_promotes_a_distinct_hit(
    rerank_store_env, monkeypatch
):
    from haunt.recall import recall

    _seed_crowded_corpus()
    baseline = recall(CROWDED_QUERY, namespace="default", k=3, use_vectors=False)
    baseline_ids = {hit.memory_id for hit in baseline}

    monkeypatch.setenv(RERANK_ENABLED_ENV, "1")
    reranked = recall(CROWDED_QUERY, namespace="default", k=3, use_vectors=False)

    assert len(reranked) == 3
    # Widening the fused slice before MMR is the whole point: enabling the
    # feature must be able to surface a candidate outside the un-widened
    # top-k, not merely reorder the same three hits.
    assert {hit.memory_id for hit in reranked} != baseline_ids


def test_enabled_recall_reports_the_order_it_actually_returned(
    rerank_store_env, monkeypatch
):
    """E5: the reported ordering must be the stage that produced it.

    Before this stage was wired, final_rank and explanation.ordering kept
    describing RRF fusion for a list MMR had already reordered.
    """
    from haunt.recall import recall

    _seed_crowded_corpus()
    monkeypatch.setenv(RERANK_ENABLED_ENV, "1")

    hits = recall(CROWDED_QUERY, namespace="default", k=3, use_vectors=False)
    explanations = [hit.as_dict()["explanation"] for hit in hits]

    assert [item["final_rank"] for item in explanations] == [1, 2, 3]
    assert all(
        item["ordering"]
        == {
            "primary": f"{RERANK_METHOD}_desc",
            "ties": "memory_id_asc",
            "stage": RERANK_METHOD,
            "reordered_from": "rrf_score_desc",
        }
        for item in explanations
    )
    # RRF evidence is kept, not overwritten -- and at least one hit really
    # moved, so rrf_rank is load-bearing rather than a copy of final_rank.
    assert all(item["rrf_score"] is not None for item in explanations)
    assert any(
        item["rrf_rank"] != item["final_rank"] for item in explanations
    )
    assert hits.execution["rerank"] == {
        "enabled": True,
        "method": RERANK_METHOD,
        "lambda": RERANK_LAMBDA_DEFAULT,
        "pool": 5,
        "selected": 3,
    }


def test_enabled_planned_recall_does_not_resort_away_the_rerank(
    rerank_store_env, monkeypatch
):
    """planner.run_recall re-sorts merged hits; that must not undo MMR."""
    from haunt.planner import planned_recall
    from haunt.recall import recall

    _seed_crowded_corpus()
    monkeypatch.setenv(RERANK_ENABLED_ENV, "1")

    direct = recall(CROWDED_QUERY, namespace="default", k=3, use_vectors=False)
    planned = planned_recall(CROWDED_QUERY, namespace="default", k=3)

    assert [hit.memory_id for hit in planned] == [hit.memory_id for hit in direct]


def test_response_budget_truncates_what_the_reranker_chose(
    rerank_mcp_env, monkeypatch
):
    """Ordering constraint: rerank reorders, the budget truncates -- in that
    order. Budget-first would drop the hits MMR promoted before it ever ran.
    """
    import json

    from haunt import budget as budget_module
    from haunt import mcp_server

    _seed_crowded_corpus(filler="padding " * 120)
    fused_ids = [
        hit["memory_id"]
        for hit in json.loads(
            mcp_server.memory_recall(query=CROWDED_QUERY, k=3)
        )["hits"]
    ]

    monkeypatch.setenv(RERANK_ENABLED_ENV, "1")
    unbudgeted = json.loads(mcp_server.memory_recall(query=CROWDED_QUERY, k=3))
    reranked = unbudgeted["hits"]
    reranked_ids = [hit["memory_id"] for hit in reranked]
    assert unbudgeted["recall_budget"]["applied"] is False
    assert reranked_ids[:2] != fused_ids[:2]

    # Size the budget to admit exactly the first two hits, so it is forced
    # to drop a suffix of whatever list reached it.
    slim = [
        {key: value for key, value in hit.items() if key != "snippet"}
        for hit in reranked
    ]
    monkeypatch.setenv(
        "HAUNT_RECALL_MAX_CHARS", str(len(budget_module.serialize(slim[:2])))
    )

    payload = json.loads(mcp_server.memory_recall(query=CROWDED_QUERY, k=3))

    assert payload["recall_budget"]["hits_dropped"] == 1
    assert [hit["memory_id"] for hit in payload["hits"]] == reranked_ids[:2]


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


# ---------------------------------------------------------------------------
# rerank_eval: the flag must not leak into its baseline arm
# ---------------------------------------------------------------------------


def test_rerank_eval_baseline_arm_survives_the_flag_being_set(monkeypatch):
    """Both arms come from one recall() call, and recall() honours the flag,
    so a set flag would quietly turn the baseline arm into a second reranked
    arm -- and the comparison this harness exists for into a no-op.
    """
    from haunt import rerank_eval

    clean = rerank_eval.evaluate().as_dict()
    monkeypatch.setenv(RERANK_ENABLED_ENV, "1")

    assert rerank_eval.evaluate().as_dict() == clean


def test_readme_documents_rerank_env_vars():
    """C-series adversarial review defect (LOW): HAUNT_RERANK_ENABLED and
    HAUNT_RERANK_LAMBDA appeared nowhere in README.md, unlike every other
    env var this branch adds. Guards against that regressing, and -- now
    that recall() is the wiring point -- against the table still telling a
    reader these vars change nothing, which was true only while the
    reranker had no caller.
    """
    from pathlib import Path

    readme = Path("README.md").read_text(encoding="utf-8")
    assert RERANK_ENABLED_ENV in readme
    assert RERANK_LAMBDA_ENV in readme
    assert "no call site" not in readme.lower()
    assert "currently changes nothing" not in readme.lower()
