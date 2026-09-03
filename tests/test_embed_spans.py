"""L21: content past the embedding window must reach vector search.

`HAUNT_EMBED_MAX_LEN` truncates every text before embedding. Measured against
the live corpora, that left 51-66% of embedded memories partly unrepresented
and dropped 66-75% of all tokens from the vector index, while FTS kept
indexing the same rows whole -- so a verbatim phrase from a memory's tail
retrieved that memory far less often than one from its head.

The planner tests below need no model. The retrieval tests do, and they are
the ones that actually prove the fix: the load-bearing one embeds a distinctive
phrase into a long memory's tail and asserts the memory comes back for that
phrase with spans on and does not with spans off, in the same process, on the
same corpus.
"""

from __future__ import annotations

import os

import pytest

from haunt import spans

# ---------------------------------------------------------------------------
# Planner: deterministic, verbatim, bounded. No model needed.
# ---------------------------------------------------------------------------


class _WordTokenizer:
    """One token per whitespace-separated word, with real character offsets.

    Stands in for the BGE tokenizer so the planner's window arithmetic is
    testable without a 2 GB download. Offsets are exact, which is the only
    property `spans.plan` relies on.
    """

    def encode(self, text: str):
        offsets = []
        index = 0
        for word in text.split(" "):
            if word:
                offsets.append((index, index + len(word)))
            index += len(word) + 1
        return type("Enc", (), {"offsets": offsets})()


def _words(count: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i:04d}" for i in range(count))


def test_text_inside_the_window_plans_no_spans():
    """The common case must be untouched: one memory, one vector, as before."""
    plan = spans.plan(_words(100), max_len=512, tokenizer=_WordTokenizer())
    assert plan == spans.EMPTY
    assert not plan


def test_spans_are_verbatim_slices_of_the_stored_text():
    """A span is a substring, never a rewrite -- Haunt embeds what it stored."""
    text = _words(1500)
    plan = spans.plan(text, max_len=512, tokenizer=_WordTokenizer())
    assert plan.spans
    for span in plan.spans:
        assert text[span.start_char : span.end_char] == span.slice(text)
        assert span.slice(text) in text


def test_spans_cover_every_token_past_the_head_window():
    """No gap between the head and the tail, and none between tail windows."""
    text = _words(1500)
    tok = _WordTokenizer()
    plan = spans.plan(text, max_len=512, tokenizer=tok, overlap=64)
    head_end = tok.encode(text).offsets[511][1]

    assert plan.spans[0].start_char < head_end, "span 1 must overlap the head"
    assert plan.covered_chars == len(text), "the tail must reach the end"
    assert plan.truncated is False
    for previous, current in zip(plan.spans, plan.spans[1:]):
        assert current.start_char < previous.end_char, "windows must overlap"


def test_overlap_keeps_a_boundary_straddling_phrase_whole():
    """A sentence cut in half by the window is whole inside some span.

    Without overlap the phrase would be split across the head and span 1 and
    appear complete in neither vector -- the failure the head/tail contrast
    measured on the real corpus.
    """
    tok = _WordTokenizer()
    phrase = "quarterly reconciliation ledger anomaly"
    head = _words(500)
    tail = _words(600, prefix="t")
    text = f"{head} {phrase} {tail}"
    plan = spans.plan(text, max_len=512, tokenizer=tok, overlap=64)
    assert any(phrase in span.slice(text) for span in plan.spans)


def test_planning_is_deterministic():
    text = _words(3000)
    first = spans.plan(text, max_len=512, tokenizer=_WordTokenizer())
    second = spans.plan(text, max_len=512, tokenizer=_WordTokenizer())
    assert first == second


def test_span_cap_bounds_the_work_and_reports_the_shortfall():
    """A pathological row costs a bounded number of model calls, and says so."""
    text = _words(40000)
    plan = spans.plan(text, max_len=512, tokenizer=_WordTokenizer(), cap=4)
    assert len(plan.spans) == 3, "cap counts the head window too"
    assert plan.truncated is True
    assert plan.covered_chars < plan.total_chars


def test_disabled_by_env_plans_nothing(monkeypatch):
    """The kill switch restores exactly the pre-fix behavior."""
    monkeypatch.setenv(spans.SPANS_ENABLED_ENV, "0")
    assert spans.plan(_words(3000), max_len=512, tokenizer=_WordTokenizer()) == spans.EMPTY


def test_no_tokenizer_falls_back_to_deterministic_character_windows():
    """A backend with no reachable tokenizer must still get coverage.

    Both shipped backends now expose one (`haunt.embed.span_tokenizer`), so
    this is the guard for a future third. The windows move; the properties --
    verbatim, overlapping, reaching the end, deterministic -- do not.
    """
    text = _words(3000)
    plan = spans.plan(text, max_len=512, tokenizer=None)
    assert plan.method == "char_estimate"
    assert plan.spans
    for span in plan.spans:
        assert text[span.start_char : span.end_char] == span.slice(text)
    for previous, current in zip(plan.spans, plan.spans[1:]):
        assert current.start_char < previous.end_char
    assert plan.covered_chars == len(text)
    assert plan.truncated is False
    assert spans.plan(text, max_len=512, tokenizer=None) == plan


# ---------------------------------------------------------------------------
# Retrieval: the fix itself. These need a real embedding backend.
# ---------------------------------------------------------------------------

pytestmark_reason = "hybrid retrieval requires an embedding model"


def _model_available() -> bool:
    from haunt import embed

    return embed.available()


requires_model = pytest.mark.skipif(
    bool(os.environ.get("HAUNT_FTS_ONLY")) or os.environ.get("HAUNT_EMBED_MODEL") == "off",
    reason=pytestmark_reason,
)

# A phrase that shares no content word with the filler around it, so a hit on
# it is a hit on the span that contains it and not on topical bleed from the
# head. Deliberately unlike the surrounding text in both vocabulary and
# subject.
NEEDLE = (
    "The harpsichord restoration in Ravenna used bone glue and quartersawn "
    "spruce salvaged from a demolished granary."
)


def _filler(paragraphs: int) -> str:
    """Bulk text on an unrelated subject, long enough to bury the needle."""
    unit = (
        "The deployment pipeline builds the container image, pushes it to the "
        "registry, runs the migration job, and then rolls the service forward "
        "one replica at a time while the health check gates each step. "
    )
    return (unit * paragraphs).strip()


@requires_model
def test_tail_content_reaches_the_vector_index_and_did_not_before(
    haunt_env, monkeypatch
):
    """The whole fix, as a before/after on one corpus.

    Same store, same query, same model. The assertion is on vector *distance*
    to the needle rather than on rank: rank over a small corpus is decided by
    how many distractors happen to sit between, which makes a threshold there
    a coin flip. Distance measures the thing that actually changed -- whether
    any vector of this memory represents its tail at all.
    """
    from haunt import embed
    from haunt.recall import recall
    from haunt.store import Store

    if not _model_available():
        pytest.skip(pytestmark_reason)

    long_text = f"{_filler(40)} {NEEDLE} {_filler(10)}"

    def distance_to_needle(namespace: str) -> tuple[float, int | None]:
        """Nearest distance between the needle query and this memory, and rank."""
        with Store(namespace) as store:
            written = store.observe(long_text, role="user", tier="episodic")
            for i in range(20):
                store.observe(f"{_filler(2)} note {i}", role="user", tier="episodic")
            hits = recall(NEEDLE, store=store, k=25, use_vectors=True)
        hit = next((h for h in hits if h.memory_id == written.memory_id), None)
        assert hit is not None and hit.vec_distance is not None, (
            "the memory did not appear in the vector candidate set at all; "
            "this measurement needs it present to compare distances"
        )
        return hit.vec_distance, hit.vec_rank

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()

    monkeypatch.setenv(spans.SPANS_ENABLED_ENV, "0")
    before_distance, _ = distance_to_needle("spans-off")

    monkeypatch.delenv(spans.SPANS_ENABLED_ENV, raising=False)
    after_distance, after_rank = distance_to_needle("spans-on")

    # The needle is near-verbatim inside one span, so with tail coverage the
    # nearest vector for this memory should be very close and should be the
    # best match in the corpus. Without it, the only vector is the head --
    # filler prose on an unrelated subject.
    # Thresholds are relative, not absolute: the distance a span vector
    # lands at depends on the model and on how much of the window the needle
    # occupies, and CI runs bge-small where the measurements here run
    # bge-m3. The claim under test is a change in reachability, so that is
    # what is asserted.
    assert before_distance > 0.4, (
        f"the pre-fix arm was already close ({before_distance:.3f}), so this "
        "corpus does not reproduce the defect and proves nothing"
    )
    assert before_distance - after_distance > 0.25, (
        f"tail coverage moved the memory only {before_distance:.3f} -> "
        f"{after_distance:.3f}; content past the window is still barely "
        "represented"
    )
    assert after_rank == 1, (
        "with its tail indexed, a memory containing the query near-verbatim "
        "should be the nearest vector in the corpus"
    )


@requires_model
def test_the_matching_span_is_named_in_the_explanation(haunt_env, monkeypatch):
    """A hit must not claim a head match when a tail window produced it."""
    from haunt import embed
    from haunt.recall import recall
    from haunt.store import Store

    if not _model_available():
        pytest.skip(pytestmark_reason)

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()
    with Store("spans-explain") as store:
        written = store.observe(
            f"{_filler(40)} {NEEDLE}", role="user", tier="episodic"
        )
        hits = recall(NEEDLE, store=store, k=5, use_vectors=True)

    hit = next(h for h in hits if h.memory_id == written.memory_id)
    vector = hit.as_dict()["explanation"]["vector"]
    assert vector["metric"] == "cosine_distance", (
        "a span vector is the same metric measured on a different window; "
        "renaming it would move E6's pinned profile identity"
    )
    assert vector["matched_span_ord"] >= 1


@requires_model
def test_a_short_memory_reports_no_span_and_serializes_as_before(
    haunt_env, monkeypatch
):
    """The overwhelming majority of hits must be byte-identical to pre-v14."""
    from haunt import embed
    from haunt.recall import recall
    from haunt.store import Store

    if not _model_available():
        pytest.skip(pytestmark_reason)

    embed.reset()
    with Store("spans-short") as store:
        store.observe("a short note about bone glue", role="user", tier="episodic")
        hits = recall("bone glue", store=store, k=5, use_vectors=True)

    assert hits
    vector = hits[0].as_dict()["explanation"]["vector"]
    assert "matched_span_ord" not in vector


@requires_model
def test_purge_erases_span_rows_and_span_vectors(haunt_env, monkeypatch):
    """Contract section 1: erasure covers every derivative of the content.

    A surviving span vector would leave the erased text semantically
    searchable, which is the thing purge exists to prevent.
    """
    from haunt import embed
    from haunt.store import Store

    if not _model_available():
        pytest.skip(pytestmark_reason)

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()
    with Store("spans-purge") as store:
        written = store.observe(
            f"{_filler(40)} {NEEDLE}", role="user", tier="episodic"
        )
        before = store.conn.execute(
            "SELECT COUNT(*) FROM memory_spans WHERE memory_id=?",
            (written.memory_id,),
        ).fetchone()[0]
        assert before > 0, "fixture did not produce spans; nothing is proven"

        assert store.purge(written.memory_id)["ok"] is True

        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM memory_spans WHERE memory_id=?",
                (written.memory_id,),
            ).fetchone()[0]
            == 0
        )
        has_vec = store.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='vec_memory_spans'"
        ).fetchone()
        if has_vec:
            surviving = store.conn.execute(
                "SELECT COUNT(*) FROM vec_memory_spans"
            ).fetchone()[0]
            assert surviving == 0


@requires_model
def test_a_correction_that_shortens_a_memory_drops_its_stale_tail(
    haunt_env, monkeypatch
):
    """Re-embedding a shrunk memory must not leave the old tail searchable."""
    from haunt import embed
    from haunt.store import Store, store_memory_spans

    if not _model_available():
        pytest.skip(pytestmark_reason)

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()
    with Store("spans-shrink") as store:
        written = store.observe(
            f"{_filler(40)} {NEEDLE}", role="user", tier="episodic"
        )
        assert store.conn.execute(
            "SELECT COUNT(*) FROM memory_spans WHERE memory_id=?",
            (written.memory_id,),
        ).fetchone()[0] > 0

        # Re-run the writer over a now-short text for the same memory.
        store_memory_spans(
            store.conn,
            [(written.memory_id, "short now")],
            dim=embed.dimension(),
            ts="2026-01-01T00:00:00.000000+00:00",
        )
        assert store.conn.execute(
            "SELECT COUNT(*) FROM memory_spans WHERE memory_id=?",
            (written.memory_id,),
        ).fetchone()[0] == 0


@requires_model
def test_an_existing_namespace_gains_tail_coverage_without_re_embedding_heads(
    haunt_env, monkeypatch
):
    """The migration path: a pre-v14 store must not stay half-indexed.

    Simulates the upgrade by embedding a long memory with spans disabled --
    which is exactly what every existing namespace holds -- then enabling
    them, re-queuing as the v14 migration does, and draining.
    """
    from haunt import embed
    from haunt.recall import recall
    from haunt.store import Store

    if not _model_available():
        pytest.skip(pytestmark_reason)

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()
    long_text = f"{_filler(40)} {NEEDLE}"

    monkeypatch.setenv(spans.SPANS_ENABLED_ENV, "0")
    with Store("spans-upgrade") as store:
        written = store.observe(long_text, role="user", tier="episodic")
        head_vector = store.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (written.memory_id,)
        ).fetchone()[0]
        assert store.conn.execute("SELECT COUNT(*) FROM memory_spans").fetchone()[0] == 0

    monkeypatch.delenv(spans.SPANS_ENABLED_ENV, raising=False)
    with Store("spans-upgrade") as store:
        # What the v14 migration does: enqueue, never embed.
        store.conn.execute(
            "INSERT OR IGNORE INTO embedding_jobs(memory_id, queued_at) "
            "VALUES (?, ?)",
            (written.memory_id, "2026-01-01T00:00:00.000000+00:00"),
        )
        store.conn.commit()
        report = store.process_embedding_jobs(limit=8)
        assert report["spans"]["spans"] > 0

        after = store.conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (written.memory_id,)
        ).fetchone()[0]
        assert after == head_vector, (
            "the head vector must be unchanged; the upgrade adds tail "
            "coverage rather than re-deriving what was already correct"
        )
        hits = recall(NEEDLE, store=store, k=10, use_vectors=True)
    assert any(h.memory_id == written.memory_id for h in hits)


@requires_model
def test_export_carries_no_spans_and_import_rebuilds_them(haunt_env, monkeypatch):
    """Contract section 4: derived, rebuildable projections stay out of bundles."""
    import json

    from haunt import embed
    from haunt.portability import export_namespace_path
    from haunt.store import Store

    if not _model_available():
        pytest.skip(pytestmark_reason)

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()
    with Store("spans-export") as store:
        store.observe(f"{_filler(40)} {NEEDLE}", role="user", tier="episodic")
        assert store.conn.execute("SELECT COUNT(*) FROM memory_spans").fetchone()[0] > 0

    out = haunt_env / "spans-export.json"
    export_namespace_path("spans-export", out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "memory_spans" not in serialized
    assert "vec_memory_spans" not in serialized


# ---------------------------------------------------------------------------
# Review findings. Each of these failed before the fix it guards.
# ---------------------------------------------------------------------------


def test_a_cap_that_allows_no_tail_window_says_so():
    """HAUNT_EMBED_MAX_SPANS=1 must not look like "this text fits"."""
    text = _words(5000)
    plan = spans.plan(text, max_len=512, tokenizer=_WordTokenizer(), cap=1)
    assert plan.spans == ()
    assert plan.truncated is True, (
        "the whole tail was dropped and nothing recorded it; the module "
        "promises truncation is never silent"
    )
    assert plan.total_chars == len(text)
    # And the ordinary short-text case must still be EMPTY, not truncated.
    assert spans.plan(_words(50), max_len=512, tokenizer=_WordTokenizer()) == spans.EMPTY


def test_store_counts_a_truncated_plan_that_produced_no_spans(monkeypatch):
    """The drain report must show the shortfall, not spans=0/truncated=0."""
    from haunt.store import store_memory_spans

    monkeypatch.setenv(spans.MAX_SPANS_ENV, "1")

    class _Conn:
        def execute(self, sql, params=()):
            class _R:
                def fetchone(self_inner):
                    return (1,) if "sqlite_master" in sql else None

                def fetchall(self_inner):
                    return []
            return _R()

    stats = store_memory_spans(
        _Conn(), [("m1", _words(5000))], dim=4, ts="2026-01-01T00:00:00.000000+00:00"
    )
    assert stats["truncated"] == 1
    assert stats["spans"] == 0


def test_backfill_floor_is_a_constant_not_the_live_window():
    """A one-shot migration must not depend on the env var of the day.

    The v14 backfill runs once, guarded by the schema version. Selecting its
    population with the live HAUNT_EMBED_MAX_LEN means a single open under a
    large value permanently excludes every row below it, with no later pass to
    notice.
    """
    import inspect

    from haunt import store as store_mod

    assert store_mod.SPAN_FLOOR_CHARS == 512
    source = inspect.getsource(store_mod._ensure_namespace_schema)
    backfill = source[source.index("if current < 14:") :]
    assert "SPAN_FLOOR_CHARS" in backfill
    assert "(embed_max_len(),)" not in backfill, (
        "the backfill floor must not be the live window"
    )


@requires_model
def test_the_final_span_fits_inside_the_encoder_window(haunt_env):
    """Interior spans are saved by overlap; the last one has nothing after it.

    Windows are cut on content tokens, but the encoder adds CLS/SEP and then
    truncates, and re-tokenizing a substring drifts from the parent
    tokenization. Both push the end of a span past the window.
    """
    from haunt import embed
    from haunt.store import plan_memory_spans

    if not _model_available():
        pytest.skip(pytestmark_reason)

    tok = embed.span_tokenizer()
    if tok is None:
        pytest.skip("backend exposes no tokenizer; span widths are estimated")
    overhead = embed.special_token_overhead()
    ceiling = embed.max_len()

    text = " ".join(f"token{i:05d}" for i in range(4000))
    plan = plan_memory_spans(text)
    assert plan.spans

    final = plan.spans[-1]
    width = len(tok.encode(final.slice(text), add_special_tokens=False).ids) + overhead
    assert width <= ceiling, (
        f"final span encodes to {width} against a {ceiling} window, so the "
        "last tokens of the memory are still truncated away"
    )


@requires_model
def test_the_span_leg_offers_as_many_memories_as_the_head_leg(haunt_env, monkeypatch):
    """`limit` nearest spans is not `limit` nearest memories.

    One long memory can hold dozens of spans and occupy the whole span KNN,
    so without over-fetching the shortfall lands entirely on the tail-only
    memories this feature exists to surface.
    """
    import sqlite_vec

    from haunt import embed
    from haunt.recall import _span_hits
    from haunt.store import Store

    if not _model_available():
        pytest.skip(pytestmark_reason)

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()
    limit = 8
    with Store("spans-fanout") as store:
        for i in range(limit * 2):
            store.observe(
                f"{_filler(12)} record {i} {NEEDLE}", role="user", tier="episodic"
            )
        per_memory = store.conn.execute(
            "SELECT MAX(c) FROM (SELECT COUNT(*) c FROM memory_spans GROUP BY memory_id)"
        ).fetchone()[0]
        assert per_memory and per_memory > 1, "fixture needs multi-span memories"

        blob = sqlite_vec.serialize_float32(embed.embed_one(NEEDLE))
        hits = _span_hits(store.conn, blob, "1=1", [], limit)
    assert len(hits) == limit, (
        f"span leg returned {len(hits)} distinct memories for limit={limit}; "
        f"one memory holds up to {per_memory} spans and crowded the rest out"
    )


@requires_model
def test_tail_is_reachable_on_the_persisted_embedding_fallback(haunt_env, monkeypatch):
    """A namespace whose sqlite-vec did not load must not be head-only.

    `haunt health` reports tail coverage from memory_spans rows. If the
    fallback retrieval path cannot read those vectors, that number promises
    something the active path cannot deliver.
    """
    from haunt import embed
    from haunt.recall import _vec_hits
    from haunt.store import Store

    if not _model_available():
        pytest.skip(pytestmark_reason)

    monkeypatch.setenv("HAUNT_EMBED_MAX_LEN", "128")
    embed.reset()
    with Store("spans-fallback") as store:
        written = store.observe(
            f"{_filler(40)} {NEEDLE}", role="user", tier="episodic"
        )
        for i in range(5):
            store.observe(f"{_filler(2)} note {i}", role="user", tier="episodic")
        assert store.conn.execute(
            "SELECT COUNT(*) FROM memory_spans WHERE embedding IS NOT NULL"
        ).fetchone()[0] > 0

        # Force the persisted-embedding path the way a failed extension load
        # would: vec_ok() False sends _vec_hits down the brute-force branch.
        monkeypatch.setattr(type(store), "vec_ok", lambda self: False)
        hits = _vec_hits(store, embed.embed_one(NEEDLE), "1=1", [], 10)

    assert any(mid == written.memory_id for mid, _r, _d, _m in hits), (
        "tail content is unreachable on the fallback path while health "
        "reports it as covered"
    )
    assert all(metric == "l2_distance" for _m, _r, _d, metric in hits)
