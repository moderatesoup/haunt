"""L30: rows the eligibility filter discards must not spend the KNN budget.

`vec0` answers a KNN by returning the `k` nearest rows. Haunt's validity
(`m.valid_to IS NULL`) and residue (`recall_class`, raw tool structure)
predicates are applied afterwards, in the join. So a superseded row or a raw
tool-I/O row that sits nearer to the query than the live answer occupies a
candidate slot and is then thrown away.

Before the fix, 45 such rows were enough to make the vector leg return
**nothing** on a corpus that plainly contained the answer -- and report
`no_vector_candidates`, which is false: the index returned a full 40. The
failure is invisible from outside because FTS usually still finds the row, so
hybrid retrieval quietly becomes lexical-only. It is exactly the
preference/decision shape (BACKLOG L1) where FTS is weakest.

Pre-existing, not a consequence of tail spans: the same shape is at
`06ebd23:src/haunt/recall.py:505`.
"""

from __future__ import annotations

import os

import pytest

# A live answer that is genuinely relevant but shares little wording with the
# query, so it is not the single nearest vector. Hidden rows are near-verbatim
# restatements of the query and therefore all rank ahead of it.
QUERY = "quarterly reconciliation ledger anomaly Ravenna branch"
LIVE = (
    "The books for the Ravenna office did not balance in Q3; the discrepancy "
    "traced to a duplicated journal entry."
)

requires_model = pytest.mark.skipif(
    bool(os.environ.get("HAUNT_FTS_ONLY"))
    or os.environ.get("HAUNT_EMBED_MODEL") == "off",
    reason="the KNN budget is a vector-path property",
)


def _vector_leg(store, query: str, limit: int):
    """Memory ids the vector leg alone contributes, ignoring FTS.

    The predicate comes from `recall._filters`, not a hand-written one: the
    residue exclusion lives there, and a hand-rolled `m.valid_to IS NULL`
    would make raw tool rows *eligible* and so measure nothing.
    """
    from haunt.embed import embed_one
    from haunt.recall import _filters, _vec_hits

    where, params = _filters(
        None,
        None,
        None,
        None,
        include_residue=False,
        recall_class_available=getattr(store, "recall_class_available", False),
    )
    return [mid for mid, _rank, _dist, _metric in _vec_hits(
        store, embed_one(query), where, list(params), limit
    )]


def _bury(store, count: int, kind: str) -> None:
    for i in range(count):
        near = f"{QUERY} note {i}"
        if kind == "superseded":
            row = store.observe(near, role="user", tier="episodic")
            store.contradict(row.memory_id, idempotency_key=f"bury-{i}")
        else:
            store.observe(
                "",
                role="tool",
                tier="episodic",
                tool_name="Bash",
                tool_input=near,
                tool_output=near,
            )


@requires_model
@pytest.mark.parametrize("kind", ["superseded", "residue"])
@pytest.mark.parametrize("hidden", [45, 80])
def test_hidden_rows_do_not_exhaust_the_vector_candidate_budget(
    haunt_env, kind, hidden
):
    """The live answer stays reachable however many hidden rows outrank it."""
    from haunt import embed
    from haunt.store import Store

    if not embed.available():
        pytest.skip("no embedding backend")

    with Store(f"knn-{kind}-{hidden}") as store:
        live = store.observe(LIVE, role="user", tier="episodic")
        _bury(store, hidden, kind)
        found = _vector_leg(store, QUERY, 40)

    assert live.memory_id in found, (
        f"{hidden} {kind} rows nearer than the live answer pushed it out of "
        "the vector leg entirely; the filter discarded the whole budget"
    )


@requires_model
def test_the_vector_stage_does_not_claim_no_candidates_when_it_had_them(
    haunt_env,
):
    """The reported reason must not say the index was empty when it was full.

    Contract section 3 requires a no-answer result to distinguish thresholded
    abstention from genuinely zero candidates. Reporting
    `no_vector_candidates` after the index returned 40 rows fails that.
    """
    from haunt import embed
    from haunt.recall import recall
    from haunt.store import Store

    if not embed.available():
        pytest.skip("no embedding backend")

    with Store("knn-reason") as store:
        store.observe(LIVE, role="user", tier="episodic")
        _bury(store, 60, "superseded")
        hits = recall(QUERY, store=store, k=5, use_vectors=True)

    stage = hits.execution["modalities"]["vector"]
    assert stage["reason"] != "no_vector_candidates"
    assert stage["state"] == "candidate"


@requires_model
def test_an_exhausted_index_still_terminates(haunt_env):
    """Escalation must stop when the index has no more rows to give.

    The loop widens `k` when too few rows survive the filter. If it did not
    notice that the index returned fewer rows than asked, a corpus that is
    entirely ineligible would escalate to the ceiling on every single recall.
    """
    from haunt import embed
    from haunt.recall import _vec_hits
    from haunt.store import Store

    if not embed.available():
        pytest.skip("no embedding backend")

    with Store("knn-exhausted") as store:
        # Every row superseded: nothing is eligible, and there are far fewer
        # rows than the ceiling.
        for i in range(5):
            row = store.observe(f"{QUERY} {i}", role="user", tier="episodic")
            store.contradict(row.memory_id, idempotency_key=f"all-{i}")
        found = _vector_leg(store, QUERY, 40)

    assert found == []


def test_the_growth_factor_reaches_the_ceiling_in_few_steps():
    """A pure-arithmetic guard on the escalation schedule.

    Runs without a model so the bound is checked even on the FTS-only CI job.
    """
    from haunt.recall import CANDIDATES, VEC_KNN_GROWTH, VEC_KNN_MAX

    steps, k = 0, CANDIDATES
    while k < VEC_KNN_MAX:
        k = min(k * VEC_KNN_GROWTH, VEC_KNN_MAX)
        steps += 1
    assert steps <= 4, (
        f"escalation takes {steps} round trips to reach the ceiling; that is "
        "too many queries on a recall that finds nothing eligible"
    )
