"""C11: recall responses have no size budget.

A recall response had no ceiling. `k` accepts up to 100 and every hit
carried the full untruncated `content` plus a redundant 200-char `snippet`
of that same content, plus (since the ranking-explanation work) a
per-hit `explanation` object -- so a k=100 payload could be hundreds of KB
to over a megabyte, injected straight into agent context with no way to
page it.

recall.py (Hit.as_dict()) stays a library call returning complete data.
The budget instead lives at the places that actually inject recall output
into agent context:

  * haunt.budget -- the serialized-size budget every machine surface shares
    (apply_recall_budget, recall_payload_cap, _truncate_hit_content), wired
    into MCP memory_recall/memory_timeline, `haunt recall --json`, and the
    dashboard recall endpoints.
  * haunt.cursor_hook.format_recall_block -- the `[haunt ns=...]` text
    block both Cursor and Claude Code hooks inject as additional_context
    on every prompt (claude_hook.py imports the same function, so one fix
    covers both hosts) (_recall_block_cap).

Both parse their env var through util.env_int (parse -> fallback on
garbage -> clamp), and both mark truncation/dropping explicitly rather
than silently shortening a field that claims to be the complete verbatim
record -- see _cap_tool_io's inline marker, which this mirrors.

This file is the pass/fail gate for that budget: a large k=100 corpus
stays under the configured budget, truncation is always visibly marked,
a caller can tell hits were dropped, ranking/ordering is untouched, the
default budget is a no-op for small ordinary responses, and both env vars
parse/clamp like their siblings.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from haunt import budget
from haunt.recall import Hit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _hit(n: int, *, content: str, trusted: bool = True, score: float | None = None) -> Hit:
    """A minimally-populated Hit with a distinct id and rank-ordered score.

    Higher n -> lower score, matching how recall() already hands Hit lists
    to callers in descending-rank order.
    """
    return Hit(
        memory_id=f"mem-{n:04d}",
        event_id=f"evt-{n:04d}",
        score=score if score is not None else 1.0 / (1 + n),
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


def _small_hits(count: int) -> list[Hit]:
    return [_hit(i, content=f"short note about topic {i}") for i in range(count)]


def _render_truncated(hit: dict[str, Any], content: str, keep: int) -> dict[str, Any]:
    """Test-local mirror of budget._rendered_hit's marker format --
    kept independent of that internal helper on purpose, so a test using
    this to probe "what would keep=N measure?" exercises the module's
    observable behavior (what actually round-trips through `serialize`), not
    the presence of one particular private function name. Any correct
    implementation of the truncation marker, estimate-based or
    measurement-based, produces this exact shape.
    """
    omitted = len(content) - keep
    out = dict(hit)
    out["content"] = f"{content[:keep]}\n… [truncated by haunt: {omitted} chars omitted]"
    out["content_truncated"] = True
    out["content_omitted_chars"] = omitted
    return out


def _big_content(n_chars: int, tag: str = "X") -> str:
    body = f"{tag} " * ((n_chars // (len(tag) + 1)) + 1)
    return body[:n_chars]


def _mixed_realistic_hits(count: int, *, big_fraction: float = 0.8) -> list[Hit]:
    """Approximates a dogfooded corpus: mostly raw tool I/O near the
    cursor_hook 12,000-char per-field cap, some short conversational rows.
    See cursor_hook._embed_excluded's comment for the ~80% tool-row figure.
    """
    n_big = int(count * big_fraction)
    hits = []
    for i in range(count):
        if i < n_big:
            hits.append(_hit(i, content=_big_content(11_800, tag=f"line{i}"), trusted=False))
        else:
            hits.append(_hit(i, content=f"short conversational note {i}"))
    return hits


# ---------------------------------------------------------------------------
# DEFECT 1 test-gap generators: every hit above varies only `content` size.
# vec_rank/fts_rank/filter_context/references/vector_stage/fts_stage stay at
# dataclass defaults, so explanation overhead is always small and constant --
# nothing in the suite exercised a hit whose *non-content* overhead alone
# (explanation.references: an unbounded correction_lineage.correction_ids
# list, or a multi-KB validated provenance envelope) could blow the budget
# with hardly any content to show for it. These generators vary that
# dimension instead.
# ---------------------------------------------------------------------------


def _correction_lineage(n: int) -> dict[str, Any]:
    """The real recall_references_many() shape (store.py) for an intact
    n-entry correction chain: {"status": "linked", "correction_ids": [...]}.
    """
    return {
        "status": "linked",
        "correction_ids": [f"cccccccc-cccc-4ccc-8ccc-{i:012d}" for i in range(n)],
    }


def _realistic_provenance_envelope(*, transforms: int = 0) -> dict[str, Any]:
    """A genuine validate_provenance()-accepted 'import' envelope, sized
    like a real high-fidelity import (near-max text fields) -- built by
    the actual production validator, not a fabricated shape. Zero
    transforms by default already serializes to several KB on its own
    (measured 6,448 bytes); pass `transforms` for an even larger one.
    """
    from haunt.provenance import validate_provenance

    return validate_provenance(
        {
            "schema_version": 1,
            "kind": "import",
            "channel": "import",
            "source_platform": "p" * 2000,
            "source_native_id": "n" * 2000,
            "source_format": "f" * 2000,
            "parser_version": "v" * 100,
            "imported_at": "2026-01-01T00:00:00+00:00",
            "fidelity": "lossless",
            "original_blob_sha256": "sha256:" + "a" * 64,
            "transforms": [("t" * 250) for _ in range(transforms)],
        },
        origin="external-system",
        channel="import",
    )


def _hit_with_references(
    n: int,
    *,
    content: str,
    correction_count: int = 0,
    provenance: dict[str, Any] | None = None,
    trusted: bool = True,
    score: float | None = None,
) -> Hit:
    """Like _hit(), but also varies explanation.references overhead: a
    correction lineage of `correction_count` entries and/or a validated
    provenance envelope. This is the one generator in the suite that
    varies NON-content size.
    """
    lineage = (
        _correction_lineage(correction_count) if correction_count else {"status": "standalone"}
    )
    return Hit(
        memory_id=f"mem-{n:04d}",
        event_id=f"evt-{n:04d}",
        score=score if score is not None else 1.0 / (1 + n),
        tier="episodic",
        content=content,
        role="tool" if not trusted else "user",
        event_time="2026-08-20T12:00:00.000000+00:00",
        valid_from="2026-08-20T12:00:00.000000+00:00",
        valid_to=None,
        tool_name="Bash" if not trusted else None,
        final_rank=n + 1,
        raw_tool_structure=not trusted,
        references={
            "correction_lineage": lineage,
            "correction_lineage_status": lineage["status"],
            "provenance": provenance,
            "provenance_status": "import" if provenance else "legacy_unstructured",
        },
    )


@pytest.fixture
def recall_budget_env(tmp_path, monkeypatch):
    """Isolated HAUNT_HOME, FTS-only, MCP admin, no embedding model."""
    home = tmp_path / "haunthome"
    monkeypatch.setenv("HAUNT_HOME", str(home))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "default")
    monkeypatch.setenv("HAUNT_MCP_ADMIN", "1")
    monkeypatch.delenv("HAUNT_RECALL_MAX_CHARS", raising=False)
    monkeypatch.delenv("HAUNT_RECALL_BLOCK_MAX_CHARS", raising=False)
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry, register_namespace

    embed.reset()
    ensure_layout()
    init_registry()
    register_namespace("default")
    yield home
    embed.reset()


# ---------------------------------------------------------------------------
# budget.apply_recall_budget: unit-level coverage
# ---------------------------------------------------------------------------


def test_default_budget_is_a_no_op_for_a_small_ordinary_response(recall_budget_env):
    """No behavior change for the common case: identical hits, nothing
    dropped, snippet still present, budget metadata says so plainly."""
    hits = _small_hits(5)
    hit_dicts = [h.as_dict() for h in hits]
    bounded, meta = budget.apply_recall_budget(hit_dicts, k=8)

    assert bounded == hit_dicts
    assert all("snippet" in h for h in bounded)
    assert meta["applied"] is False
    assert meta["hits_dropped"] == 0
    assert meta["hits_returned"] == 5
    assert meta["hits_available"] == 5
    assert meta["snippet_dropped"] is False
    assert meta["content_truncated_memory_ids"] == []
    assert meta["max_chars"] == budget.RECALL_PAYLOAD_MAX_CHARS_DEFAULT


def test_large_corpus_at_k100_stays_under_the_configured_budget(recall_budget_env):
    """The literal C11 failure mode: k=100 against a realistic, mostly raw
    tool-I/O corpus (~1.1MB unbounded) is bounded to the configured cap."""
    hits = _mixed_realistic_hits(100, big_fraction=0.8)
    hit_dicts = [h.as_dict() for h in hits]
    unbounded_total = len(budget.serialize(hit_dicts))
    assert unbounded_total > 500_000  # sanity: this really is the C11 failure case

    bounded, meta = budget.apply_recall_budget(hit_dicts, k=100)

    bounded_total = len(budget.serialize(bounded))
    assert bounded_total <= meta["max_chars"]
    assert meta["applied"] is True
    assert meta["hits_available"] == 100
    assert len(bounded) < 100
    assert meta["hits_returned"] == len(bounded)
    assert meta["hits_dropped"] == 100 - len(bounded)


def test_cap_bounds_the_serialized_list_not_the_sum_of_its_hits(
    recall_budget_env, monkeypatch
):
    """Hits cross the boundary as a JSON list, and json.dumps spends two
    chars per hit on that list's own brackets and separators. Summing
    per-hit sizes ignored them, so a response could pack exactly to the cap
    and still serialize 2n chars past it -- ~200 at k=100, and ~15% over at
    the 2,000 clamp floor. The cap here is tuned to the exact fill the old
    accounting called full, so the old metric cannot pass this.
    """
    hit_dicts = [h.as_dict() for h in _small_hits(40)]
    slim = [{k: v for k, v in h.items() if k != "snippet"} for h in hit_dicts]
    packed = 15
    cap = sum(len(budget.serialize(hit)) for hit in slim[:packed])
    monkeypatch.setenv("HAUNT_RECALL_MAX_CHARS", str(cap))

    bounded, meta = budget.apply_recall_budget(hit_dicts, k=40)

    assert meta["max_chars"] == cap
    assert len(budget.serialize(bounded)) <= cap
    # Summing per hit reported exactly `packed` hits as fitting. They do
    # not, once the list they are emitted in is counted.
    assert meta["hits_returned"] < packed


@pytest.mark.parametrize("big_fraction", [0.0, 0.5, 1.0])
def test_large_corpus_stays_under_budget_across_content_mixes(recall_budget_env, big_fraction):
    """Same bound holds whether the bloat comes from many small hits'
    fixed explanation overhead, a realistic mix, or all-huge tool I/O."""
    hits = _mixed_realistic_hits(100, big_fraction=big_fraction)
    hit_dicts = [h.as_dict() for h in hits]
    bounded, meta = budget.apply_recall_budget(hit_dicts, k=100)
    bounded_total = len(budget.serialize(bounded))
    assert bounded_total <= meta["max_chars"]
    assert len(bounded) >= 1  # never silently empty when hits exist


def test_caller_can_tell_hits_were_dropped_for_budget_not_corpus_size(recall_budget_env):
    """hits_available vs hits_returned/hits_dropped disambiguates 'the
    corpus only had N matches' from 'the budget cut off the rest'."""
    hits = _mixed_realistic_hits(30, big_fraction=1.0)
    hit_dicts = [h.as_dict() for h in hits]
    bounded, meta = budget.apply_recall_budget(hit_dicts, k=30)

    assert meta["hits_available"] == 30
    assert meta["hits_returned"] < meta["hits_available"]
    assert meta["hits_dropped"] == meta["hits_available"] - meta["hits_returned"]
    assert meta["hits_dropped"] > 0
    assert meta["k_requested"] == 30


def test_truncation_is_explicit_and_never_silent(recall_budget_env):
    """A single hit whose content alone exceeds the whole budget is
    truncated, not dropped -- and the truncation can never be mistaken
    for a complete verbatim record."""
    huge = _hit(0, content="MEGA-" + ("z" * 60_000))
    hit_dicts = [huge.as_dict()]
    original_content = hit_dicts[0]["content"]

    bounded, meta = budget.apply_recall_budget(hit_dicts, k=1)

    assert len(bounded) == 1  # never silently empty for a nonempty result
    hit = bounded[0]
    assert hit["content_truncated"] is True
    assert hit["content_omitted_chars"] > 0
    assert hit["content"] != original_content
    assert hit["content"].startswith("MEGA-")
    assert "truncated by haunt" in hit["content"]
    assert str(hit["content_omitted_chars"]) in hit["content"]
    assert meta["content_truncated_memory_ids"] == ["mem-0000"]
    assert meta["hits_dropped"] == 0
    # The whole serialized list, not just the content slice, must respect
    # the budget -- fixed per-hit overhead (explanation, memory_id, ...) and
    # the list's own brackets and separators all count.
    assert len(budget.serialize(bounded)) <= meta["max_chars"]


def test_truncation_is_never_faked_when_it_cannot_help_the_hit_fit(recall_budget_env):
    """DEFECT 1 (fixed): sibling of test_truncation_is_explicit_and_never_silent
    above. That test covers content-driven overage, where truncating
    content succeeds and one partial hit is returned. This covers
    reference-driven overage -- a 51-char content, tiny by itself, next
    to a 600-entry correction_lineage (the exact adversarial-review
    repro: 25,458 serialized chars against the 24,000 default cap) --
    where truncating content CANNOT succeed, because the overage isn't in
    content at all. The same invariant from the test above
    (len(serialize(hits)) <= meta["max_chars"]) must still hold, but only
    because the hit is dropped, not because it was fake-truncated while
    the response stayed over budget and recall_budget claimed success
    anyway -- which is what this function actually did before the fix
    (applied=True, hits_dropped=0, and 51 real chars of content destroyed
    for zero size benefit).
    """
    huge_lineage = _hit_with_references(
        0,
        content="normal-length memory content, nothing unusual here.",
        correction_count=600,
    )
    hit_dicts = [huge_lineage.as_dict()]
    unbounded_size = len(budget.serialize(hit_dicts))
    # Sanity: this really does reproduce the defect's premise (fixed
    # overhead alone, not content, blows the default budget).
    assert unbounded_size > budget.RECALL_PAYLOAD_MAX_CHARS_DEFAULT
    assert len(hit_dicts[0]["content"]) < 100

    bounded, meta = budget.apply_recall_budget(hit_dicts, k=1)

    assert len(budget.serialize(bounded)) <= meta["max_chars"]
    assert bounded == []
    assert meta["hits_available"] == 1
    assert meta["hits_returned"] == 0
    assert meta["hits_dropped"] == 1
    assert meta["applied"] is True
    # Never falsely marked truncated: nothing was cut, so nothing is
    # reported as cut.
    assert meta["content_truncated_memory_ids"] == []


def _realistic_full_explanation_hit(n: int, *, content: str) -> Hit:
    """A hit with every explanation-contributing field populated the way
    a real ranked recall result actually has them (vec_rank, fts_rank,
    filter_context, vector_stage, fts_stage) -- not left at dataclass
    defaults like _hit()'s -- and UUID-style ids, matching what a real
    memory_id/event_id actually look like (_hit()'s short "mem-0000"
    ids underestimate this). This alone, with no lineage or provenance
    at all, already costs ~1,815 chars of fixed overhead once `snippet`
    is stripped (the shape apply_recall_budget's truncation step
    actually sees) -- small next to the 24,000 default, but enough to
    push the *estimated* room for content negative at the 2,000 clamp
    floor (2000 - 1815 - the 200-char marker reserve < 0), even though
    the hit can still genuinely be truncated to fit.
    """
    return Hit(
        memory_id=f"cccccccc-cccc-4ccc-8ccc-{n:012d}",
        event_id=f"dddddddd-dddd-4ddd-8ddd-{n:012d}",
        score=1.0 / (1 + n),
        tier="episodic",
        content=content,
        role="tool",
        event_time="2026-08-20T12:00:00.000000+00:00",
        valid_from="2026-08-20T12:00:00.000000+00:00",
        valid_to=None,
        tool_name="Bash",
        vec_rank=1,
        fts_rank=1,
        vec_distance=0.123456,
        vec_metric="cosine_distance",
        fts_rank_raw=-12.345,
        filter_context={
            "validity": "current",
            "as_of": None,
            "clock": "event_time",
            "since": None,
            "until": None,
            "tier": None,
            "include_residue": False,
            "include_untrusted": None,
            "residue_filter_source": "default",
            "residue_filter": "applied",
            "recall_class_capability": "available",
            "maintenance_performed": False,
        },
        final_rank=1,
        vector_stage={"state": "candidate", "reason": "returned_vector_candidate"},
        fts_stage={"state": "candidate", "reason": "returned_fts_candidates"},
        references={
            "correction_lineage": None,
            "correction_lineage_status": "none",
            "provenance": None,
            "provenance_status": "legacy_unstructured",
        },
        recall_class="tool",
        classification_source="raw_tool_structure",
        raw_tool_structure=True,
    )


def test_truncation_finds_the_largest_fitting_keep_near_the_clamp_floor(
    recall_budget_env, monkeypatch
):
    """Regression for a boundary case caught while reviewing this feature
    against a REALISTIC (not pathological) hit shape: a hit with a full,
    ordinary explanation object (vec_rank/fts_rank/filter_context/
    vector_stage/fts_stage, no huge lineage or provenance at all) already
    costs real, substantial fixed overhead close to the 2,000 clamp
    floor, leaving very little room for content.

    An earlier, estimate-based version of this function computed a
    fixed-arithmetic guess for how much content to keep and, right at
    this margin, could go two different kinds of wrong: give up entirely
    the moment the estimate went negative (dropping a hit truncation
    could have saved), or under-keep by assuming keep=0 when more
    content genuinely still fit. Measurement-only binary search has no
    estimate to go wrong in either direction: it always finds the true
    largest `keep` whose real serialized size fits, whatever that
    happens to be -- not dropped, and not needlessly reduced to keep=0
    when the margin is tight but nonzero.
    """
    monkeypatch.setenv("HAUNT_RECALL_MAX_CHARS", "2000")
    content = "C" * 5000
    hit = _realistic_full_explanation_hit(0, content=content)
    hit_dicts = [hit.as_dict()]
    unbounded_size = len(budget.serialize(hit_dicts))
    assert unbounded_size > 2000  # sanity: this hit really doesn't fit as-is
    # apply_recall_budget always strips the redundant `snippet` field
    # (step 2) before content truncation is even considered (step 3/4) --
    # so this is the exact shape _truncate_hit_content actually measures
    # against, and what the "one more char" boundary probe below must
    # match, or it would be checking a differently-sized dict than the
    # one the algorithm under test really operates on.
    slim_hit_dict = {k: v for k, v in hit_dicts[0].items() if k != "snippet"}

    bounded, meta = budget.apply_recall_budget(hit_dicts, k=1)

    assert len(bounded) == 1  # truncation succeeded -- the hit was not dropped
    assert bounded[0]["content_truncated"] is True
    assert meta["hits_dropped"] == 0
    assert meta["hits_returned"] == 1
    assert meta["content_truncated_memory_ids"] == [
        "cccccccc-cccc-4ccc-8ccc-000000000000"
    ]
    assert len(budget.serialize(bounded)) <= meta["max_chars"] == 2000
    # It's the LARGEST fitting keep, not merely *a* fitting one (which a
    # weaker "stop at the first candidate that fits" search could also
    # satisfy, or which an estimate could land on by coincidence): keeping
    # one more content char must no longer fit. This pins down the actual
    # binary-search postcondition, not just "some truncation happened".
    kept_chars = 5000 - bounded[0]["content_omitted_chars"]
    assert kept_chars >= 0
    one_more = _render_truncated(dict(slim_hit_dict), content, kept_chars + 1)
    assert len(budget.serialize([one_more])) > 2000


def test_truncation_keeps_escape_heavy_content_instead_of_dropping_it(
    recall_budget_env, monkeypatch
):
    """THE round-3 adversarial-review defect this redesign exists to make
    unrepresentable: content made entirely of JSON-escape-expanding
    characters (a quote and a backslash, alternating -- each one costs 2
    serialized chars once json.dumps escapes it, not 1) broke the old
    estimator's implicit "one kept char ~= one serialized char"
    assumption so badly that its single arithmetic guess for `keep`
    landed on a value whose real serialized size overshot the budget --
    and the old code gave up right there, dropping the hit, even though
    keep=0 (or many smaller values) fit with hundreds of chars of slack
    to spare. Measurement-only binary search has no such assumption: it
    verifies every candidate by actually building and measuring it, so
    it always converges on the true largest fitting `keep` regardless of
    how much any given slice happens to expand under escaping.
    """
    monkeypatch.setenv("HAUNT_RECALL_MAX_CHARS", "2000")
    content = '"\\' * 4200  # 8,400 raw chars; every one doubles once escaped
    hit = _hit(0, content=content)
    hit_dicts = [hit.as_dict()]
    unbounded_size = len(budget.serialize(hit_dicts))
    assert unbounded_size > 2000  # sanity: doesn't fit whole
    # apply_recall_budget always strips the redundant `snippet` field
    # (step 2) before content truncation is even considered (step 3/4) --
    # this is the exact shape _truncate_hit_content actually measures
    # against, and what both probes below must match.
    slim_hit_dict = {k: v for k, v in hit_dicts[0].items() if k != "snippet"}

    # Sanity: keep=0 really does fit with real slack at this budget -- the
    # exact "over-correction" premise: an estimate that fails once and
    # gives up would drop this hit even though the most conservative
    # truncation comfortably succeeds.
    floor_size = len(
        budget.serialize([_render_truncated(dict(slim_hit_dict), content, 0)])
    )
    assert floor_size < 2000 - 100, "test premise: keep=0 must fit with real slack"

    bounded, meta = budget.apply_recall_budget(hit_dicts, k=1)

    assert len(bounded) == 1  # KEPT, not dropped -- the whole point of this test
    assert bounded[0]["content_truncated"] is True
    assert meta["hits_dropped"] == 0
    assert meta["hits_returned"] == 1
    assert meta["content_truncated_memory_ids"] == ["mem-0000"]
    assert len(budget.serialize(bounded)) <= meta["max_chars"] == 2000
    # And it's the largest fitting keep, strictly better than the keep=0
    # an estimate stuck permanently at "the conservative floor" would
    # have produced: one more content char must no longer fit.
    kept_chars = len(content) - bounded[0]["content_omitted_chars"]
    assert kept_chars > 0
    one_more = _render_truncated(dict(slim_hit_dict), content, kept_chars + 1)
    assert len(budget.serialize([one_more])) > 2000


def test_truncated_hit_never_returns_fewer_than_one_hit_for_nonempty_result(recall_budget_env):
    """Even at the minimum clamp, a nonempty result never comes back
    empty -- the caller always gets something, visibly marked partial."""
    huge = _hit(0, content="Z" * 60_000)
    bounded, meta = budget.apply_recall_budget(
        [huge.as_dict()], k=1
    )
    assert len(bounded) == 1
    assert bounded[0]["content_truncated"] is True


def test_non_string_content_is_dropped_when_it_cannot_be_shrunk(recall_budget_env):
    """DEFECT 1 (fixed): a sqlite-blob envelope (json_safe_sqlite's shape
    for BLOB content) cannot be sliced -- there is no way to shrink it.
    Before the fix, a hit like this that didn't fit was kept unchanged,
    marked content_truncated=True, content_omitted_chars=None, while
    still sitting tens of thousands of chars over the cap: "truncated" in
    name only, with zero actual size benefit. Now it is dropped instead
    of shipped over budget wearing a fabricated marker.
    """
    hit_dict = _hit(0, content="placeholder").as_dict()
    # Simulate what json_safe_sqlite produces for BLOB content -- a dict,
    # not a str -- without needing a real BLOB row end-to-end.
    hit_dict["content"] = {"encoding": "base64", "data": "eeee" * 20_000}
    unbounded_size = len(budget.serialize(hit_dict))
    assert unbounded_size > budget.RECALL_PAYLOAD_MAX_CHARS_DEFAULT  # sanity

    bounded, meta = budget.apply_recall_budget([hit_dict], k=1)

    assert bounded == []
    assert meta["hits_available"] == 1
    assert meta["hits_returned"] == 0
    assert meta["hits_dropped"] == 1
    assert meta["applied"] is True
    assert meta["content_truncated_memory_ids"] == []  # never fabricated


def test_redundant_snippet_is_dropped_before_any_hit_is_dropped_or_mangled(recall_budget_env):
    """Step 2 of the degrade ladder: removing the pure-duplicate snippet
    field is preferred over losing or truncating an actual hit."""
    # 16 hits of this shape serialize to ~24,818 chars with snippet (just
    # over the 24,000 default) and ~22,542 without it (comfortably under)
    # -- sized so removing the redundant snippets is enough on its own to
    # fit the default budget, without dropping or truncating a hit.
    hits = [_hit(i, content=f"conversational row {i} " * 6) for i in range(16)]
    hit_dicts = [h.as_dict() for h in hits]
    bounded, meta = budget.apply_recall_budget(hit_dicts, k=16)

    assert meta["snippet_dropped"] is True
    assert meta["hits_dropped"] == 0
    assert meta["content_truncated_memory_ids"] == []
    assert len(bounded) == 16
    assert all("snippet" not in h for h in bounded)
    # Every other field is untouched.
    for original, kept in zip(hit_dicts, bounded):
        without_snippet = {k: v for k, v in original.items() if k != "snippet"}
        assert kept == without_snippet


def test_ranking_and_order_are_never_changed_by_the_budget(recall_budget_env):
    """The kept hits are an exact, byte-identical prefix of the already-
    ranked input -- the budget can drop a suffix, never reorder or
    cherry-pick by size."""
    hits = _mixed_realistic_hits(50, big_fraction=1.0)
    hit_dicts = [h.as_dict() for h in hits]
    bounded, meta = budget.apply_recall_budget(hit_dicts, k=50)

    assert meta["hits_dropped"] > 0  # otherwise this test proves nothing
    without_snippets = [
        {k: v for k, v in h.items() if k != "snippet"} for h in hit_dicts
    ]
    assert bounded == without_snippets[: len(bounded)]
    assert [h["memory_id"] for h in bounded] == [
        f"mem-{i:04d}" for i in range(len(bounded))
    ]


def test_trust_labelling_survives_budgeting_including_on_a_truncated_hit(recall_budget_env):
    """Requirement: never weaken trusted/trust_reason. Check it holds
    through every degrade path, including the truncated-content one --
    and check the budget actually ran (meta["applied"]), not just that
    trust survived whatever apply_recall_budget happened to do. Without
    the applied checks below, this test previously passed vacuously even
    under a mutation that replaced the whole budget with a no-op
    passthrough (applied always False): trust labels trivially "survive"
    a function that does nothing to the hits at all.
    """
    untrusted_huge = _hit(0, content="U" * 60_000, trusted=False)
    bounded, meta = budget.apply_recall_budget([untrusted_huge.as_dict()], k=1)
    assert meta["applied"] is True
    assert bounded[0]["trusted"] is False
    assert bounded[0]["trust_reason"] == "untrusted-tool-io"

    mixed = [_hit(0, content="trusted note", trusted=True)] + [
        _hit(i, content=_big_content(11_800), trusted=False) for i in range(1, 40)
    ]
    hit_dicts = [h.as_dict() for h in mixed]
    bounded, meta = budget.apply_recall_budget(hit_dicts, k=40)
    assert meta["applied"] is True
    by_id = {h["memory_id"]: h for h in bounded}
    assert by_id["mem-0000"]["trusted"] is True
    for h in bounded[1:]:
        assert h["trusted"] is False
        assert h["trust_reason"] == "untrusted-tool-io"


@pytest.mark.parametrize(
    "cap_env,expect_kept",
    [
        ("2000", False),
        (None, False),
        ("200000", True),
    ],
    ids=["min_clamp_2000", "default_24000", "max_clamp_200000"],
)
def test_unfittable_provenance_and_lineage_overhead_across_cap_tiers(
    recall_budget_env, monkeypatch, cap_env, expect_kept
):
    """DEFECT 1 (fixed), second adversarial-review reproduction: a genuine
    validate_provenance()-accepted envelope plus a 500-entry correction
    lineage gives ~27.9K chars of fixed per-hit overhead -- over budget at
    both the 2,000 clamp floor and the 24,000 default; only the 200,000
    ceiling is large enough to admit the hit whole and untouched, with
    applied=False (no budgeting needed at all). Sweeps all three
    documented clamp tiers in one parametrization.
    """
    if cap_env is not None:
        monkeypatch.setenv("HAUNT_RECALL_MAX_CHARS", cap_env)
    else:
        monkeypatch.delenv("HAUNT_RECALL_MAX_CHARS", raising=False)

    envelope = _realistic_provenance_envelope(transforms=0)
    hit = _hit_with_references(
        0,
        content="short content, not the issue here",
        correction_count=500,
        provenance=envelope,
    )
    hit_dicts = [hit.as_dict()]

    bounded, meta = budget.apply_recall_budget(hit_dicts, k=1)

    bounded_total = len(budget.serialize(bounded))
    assert bounded_total <= meta["max_chars"]
    if expect_kept:
        assert meta["hits_returned"] == 1
        assert meta["hits_dropped"] == 0
        assert meta["applied"] is False  # fits whole; no budgeting needed
        assert bounded[0] == hit_dicts[0]  # completely untouched
    else:
        assert meta["hits_returned"] == 0
        assert meta["hits_dropped"] == 1
        assert meta["applied"] is True
        assert meta["content_truncated_memory_ids"] == []


def test_unfittable_top_ranked_hit_drops_lower_ranked_hits_too_not_promoted(
    recall_budget_env,
):
    """Resolves the tension the review called out explicitly: this
    function's existing invariant is that a hit later in rank order is
    never substituted in over an earlier one just because it is smaller
    (see apply_recall_budget's docstring step 3). If the top-ranked hit
    cannot be made to fit even truncated, the correct behavior is NOT to
    fall through and let a smaller, lower-ranked hit take its place --
    that would let hit size dictate selection, exactly what this function
    must never do. So every hit behind an unfittable top-ranked hit is
    dropped too, even though the smaller hits would trivially fit the
    budget on their own. This is what makes "a nonempty result never
    returns zero hits" yield to honest reporting: the alternative
    (skipping ahead to smaller hits) would silently change what recall
    selected, which is worse.
    """
    unfittable_top = _hit_with_references(
        0, content="top ranked but unfittable", correction_count=600
    )
    small_a = _hit(1, content="small note A")
    small_b = _hit(2, content="small note B")
    hit_dicts = [h.as_dict() for h in (unfittable_top, small_a, small_b)]

    bounded, meta = budget.apply_recall_budget(hit_dicts, k=3)

    assert bounded == []
    assert meta["hits_available"] == 3
    assert meta["hits_returned"] == 0
    assert meta["hits_dropped"] == 3
    # Sanity: small_a/small_b would trivially fit on their own -- this
    # proves the result is "rank order is never bent", not "nothing
    # could possibly have fit".
    small_only_total = len(
        budget.serialize([h.as_dict() for h in (small_a, small_b)])
    )
    assert small_only_total < meta["max_chars"]


@pytest.mark.parametrize("cap", [2000, 6000, 24000, 60000, 200000])
def test_reference_overhead_sweep_stays_under_cap_at_every_clamp_tier(
    recall_budget_env, monkeypatch, cap
):
    """Sweep across the clamp floor, default, ceiling, and two
    intermediate values, against hits whose bulk is in explanation
    overhead rather than content -- lineage-heavy, provenance-heavy, and
    both combined, plus one ordinary small hit. Whatever the mix, the
    returned hits list never serializes to more than the configured cap,
    and dropped/returned always accounts for every available hit.
    """
    monkeypatch.setenv("HAUNT_RECALL_MAX_CHARS", str(cap))

    hits = [
        _hit_with_references(0, content="lineage-heavy hit", correction_count=600),
        _hit_with_references(
            1,
            content="provenance-heavy hit",
            provenance=_realistic_provenance_envelope(transforms=0),
        ),
        _hit_with_references(
            2,
            content="both-heavy hit",
            correction_count=500,
            provenance=_realistic_provenance_envelope(transforms=0),
        ),
        _hit(3, content="an ordinary small hit"),
    ]
    hit_dicts = [h.as_dict() for h in hits]

    bounded, meta = budget.apply_recall_budget(hit_dicts, k=len(hits))

    assert meta["max_chars"] == cap
    bounded_total = len(budget.serialize(bounded))
    # The real invariant this whole function exists to uphold: never over
    # budget, regardless of cap tier or where the bloat comes from -- even
    # (especially) when applied is False or hits_dropped is 0, the exact
    # metadata shape Defect 1 used to pair with an over-budget response.
    assert bounded_total <= cap
    assert meta["hits_returned"] + meta["hits_dropped"] == meta["hits_available"]
    assert meta["hits_returned"] == len(bounded)


def test_empty_hits_list_is_a_no_op(recall_budget_env):
    from haunt import mcp_server

    bounded, meta = budget.apply_recall_budget([], k=8)
    assert bounded == []
    assert meta["applied"] is False
    assert meta["hits_available"] == 0
    assert meta["hits_returned"] == 0


# ---------------------------------------------------------------------------
# mcp_server.memory_recall: through the real MCP tool boundary
# ---------------------------------------------------------------------------


def test_memory_recall_tool_bounds_k100_via_monkeypatched_planned_recall(
    recall_budget_env, monkeypatch
):
    """The public memory_recall(k=100) tool call, not just the internal
    helper: wires _apply_recall_budget into the real serialized payload."""
    from haunt import mcp_server

    hits = _mixed_realistic_hits(100, big_fraction=0.8)
    monkeypatch.setattr(mcp_server, "planned_recall", lambda *a, **kw: hits)
    mcp_server._MCP_AUTHORITY = None

    raw = mcp_server.memory_recall(query="anything", namespace="default", k=100)
    payload = json.loads(raw)

    assert "recall_budget" in payload
    recall_budget = payload["recall_budget"]
    assert recall_budget["applied"] is True
    assert recall_budget["hits_dropped"] > 0
    assert len(payload["hits"]) == recall_budget["hits_returned"]
    hits_total = len(json.dumps(payload["hits"], ensure_ascii=False))
    assert hits_total <= recall_budget["max_chars"]
    # Ordering untouched: an exact-prefix, in-order subset of the 100 ranked ids.
    expected_prefix = [f"mem-{i:04d}" for i in range(len(payload["hits"]))]
    assert [h["memory_id"] for h in payload["hits"]] == expected_prefix


def test_memory_recall_tool_unchanged_for_small_response(recall_budget_env, monkeypatch):
    """Through the real tool call: a small result carries the same hits
    as before, plus only the additive recall_budget key."""
    from haunt import mcp_server

    hits = _small_hits(3)
    monkeypatch.setattr(mcp_server, "planned_recall", lambda *a, **kw: hits)
    mcp_server._MCP_AUTHORITY = None

    raw = mcp_server.memory_recall(query="anything", namespace="default", k=8)
    payload = json.loads(raw)

    assert payload["hits"] == [h.as_dict() for h in hits]
    assert payload["recall_budget"]["applied"] is False


def test_memory_recall_end_to_end_real_store_bounds_large_fts_corpus(
    recall_budget_env, monkeypatch
):
    """No mocking of recall internals: real Store.observe() rows (tool
    I/O concatenated into memories.content the same way cursor_hook /
    claude_hook produce it), real FTS recall, real memory_recall()."""
    from haunt import mcp_server
    from haunt.store import Store

    mcp_server._MCP_AUTHORITY = None
    filler = _big_content(3_000, tag="BUDGETCORPUSTOKEN")
    with Store("default") as store:
        for i in range(45):
            store.observe(
                "",
                role="tool",
                tier="episodic",
                tool_name="Bash",
                tool_input="run",
                tool_output=f"{filler} row-{i}",
                defer_embedding=True,
            )

    # These are raw tool-I/O rows, excluded from default (non-residue)
    # recall by design (see recall.py's _filters) -- include_residue=True
    # is the documented audit/search escape hatch, appropriate here since
    # the point of this test is exercising large tool-I/O hits specifically.
    raw = mcp_server.memory_recall(
        query="BUDGETCORPUSTOKEN", namespace="default", k=100, include_residue=True
    )
    payload = json.loads(raw)
    recall_budget = payload["recall_budget"]

    assert recall_budget["hits_available"] > 0
    assert recall_budget["applied"] is True
    assert recall_budget["hits_dropped"] > 0
    hits_total = len(json.dumps(payload["hits"], ensure_ascii=False))
    assert hits_total <= recall_budget["max_chars"]


def test_memory_recall_tool_drops_hit_whose_reference_overhead_alone_exceeds_budget(
    recall_budget_env, monkeypatch
):
    """DEFECT 1 (fixed), through the real memory_recall() tool boundary,
    not just the internal helper -- the exact adversarial-review
    reproduction: 51 chars of real content next to a 600-entry
    correction lineage, default budget. Before the fix this returned
    recall_budget={"applied": True, "hits_dropped": 0, ...} while
    shipping a hit 25,458 chars against a 24,000 cap (full response
    25,976 chars) and destroying its only 51 chars of real content for
    zero size benefit. Now: the hit is dropped and recall_budget says so
    truthfully.
    """
    from haunt import mcp_server

    hit = _hit_with_references(
        0,
        content="normal-length memory content, nothing unusual here.",
        correction_count=600,
    )
    monkeypatch.setattr(mcp_server, "planned_recall", lambda *a, **kw: [hit])
    mcp_server._MCP_AUTHORITY = None

    raw = mcp_server.memory_recall(query="anything", namespace="default", k=1)
    payload = json.loads(raw)

    recall_budget = payload["recall_budget"]
    hits_total = len(json.dumps(payload["hits"], ensure_ascii=False))
    assert hits_total <= recall_budget["max_chars"]
    assert payload["hits"] == []
    assert recall_budget["hits_available"] == 1
    assert recall_budget["hits_returned"] == 0
    assert recall_budget["hits_dropped"] == 1
    assert recall_budget["applied"] is True
    assert recall_budget["content_truncated_memory_ids"] == []


# ---------------------------------------------------------------------------
# The other machine surfaces: one budget, not one per surface
# ---------------------------------------------------------------------------


def _seed_big_tool_io_corpus(rows: int = 45) -> str:
    """Real Store.observe() tool-I/O rows, the shape hooks actually write."""
    from haunt.store import Store

    filler = _big_content(3_000, tag="SURFACEBUDGETTOKEN")
    with Store("default") as store:
        for i in range(rows):
            store.observe(
                "",
                role="tool",
                tier="episodic",
                tool_name="Bash",
                tool_input="run",
                tool_output=f"{filler} row-{i}",
                defer_embedding=True,
            )
    return "SURFACEBUDGETTOKEN"


def test_cli_recall_json_is_bounded_like_the_mcp_tool(recall_budget_env):
    """`haunt recall --json` feeds an agent the same way memory_recall does,
    so it gets the same cap and the same honest recall_budget."""
    from typer.testing import CliRunner

    from haunt import cli

    query = _seed_big_tool_io_corpus()
    result = CliRunner().invoke(
        cli.app,
        ["recall", query, "-n", "default", "--json", "--k", "100", "--include-residue"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    recall_budget = payload["recall_budget"]

    assert recall_budget["applied"] is True
    assert recall_budget["hits_dropped"] > 0
    assert len(json.dumps(payload["hits"], ensure_ascii=False)) <= recall_budget["max_chars"]


def test_dashboard_recall_endpoints_are_bounded_including_the_fan_out(
    recall_budget_env,
):
    """The all-namespace endpoint answers from every registered namespace at
    once, so it needs the cap more than the single-namespace one does. Its
    groups must agree with the bounded `hits` rather than keep the dropped
    rows."""
    from tests.dashutil import make_dash_client

    query = _seed_big_tool_io_corpus()
    client = make_dash_client()
    params = {"q": query, "k": 100, "include_residue": "1"}

    one = client.get("/api/namespace/default/recall", params=params).json()
    assert one["recall_budget"]["applied"] is True
    assert one["recall_budget"]["hits_dropped"] > 0
    assert len(json.dumps(one["hits"], ensure_ascii=False)) <= one["recall_budget"]["max_chars"]

    every = client.get("/api/recall", params=params).json()
    assert every["recall_budget"]["applied"] is True
    assert len(json.dumps(every["hits"], ensure_ascii=False)) <= every["recall_budget"]["max_chars"]
    grouped = [hit for group in every["namespace_groups"] for hit in group["hits"]]
    assert grouped == every["hits"]


def test_memory_timeline_is_bounded_at_the_mcp_boundary(recall_budget_env):
    """An event row carries the same uncapped tool I/O a hit does, and
    limit accepts up to 100 of them."""
    from haunt import mcp_server

    _seed_big_tool_io_corpus()
    mcp_server._MCP_AUTHORITY = None

    payload = json.loads(mcp_server.memory_timeline(namespace="default", limit=100))
    recall_budget = payload["recall_budget"]

    assert recall_budget["applied"] is True
    assert recall_budget["hits_dropped"] > 0
    assert len(json.dumps(payload["events"], ensure_ascii=False)) <= recall_budget["max_chars"]


# ---------------------------------------------------------------------------
# HAUNT_RECALL_MAX_CHARS parsing (util.env_int, like HAUNT_TOOL_IO_MAX_CHARS)
# ---------------------------------------------------------------------------


def test_recall_payload_cap_env_var_parses_and_clamps(recall_budget_env, monkeypatch):
    from haunt.budget import (
        RECALL_PAYLOAD_MAX_CHARS_DEFAULT,
        RECALL_PAYLOAD_MAX_CHARS_MAX,
        RECALL_PAYLOAD_MAX_CHARS_MIN,
        recall_payload_cap,
    )

    monkeypatch.delenv("HAUNT_RECALL_MAX_CHARS", raising=False)
    assert recall_payload_cap() == RECALL_PAYLOAD_MAX_CHARS_DEFAULT

    monkeypatch.setenv("HAUNT_RECALL_MAX_CHARS", "not-a-number")
    assert recall_payload_cap() == RECALL_PAYLOAD_MAX_CHARS_DEFAULT

    monkeypatch.setenv("HAUNT_RECALL_MAX_CHARS", "0")
    assert recall_payload_cap() == RECALL_PAYLOAD_MAX_CHARS_MIN

    monkeypatch.setenv("HAUNT_RECALL_MAX_CHARS", "-5")
    assert recall_payload_cap() == RECALL_PAYLOAD_MAX_CHARS_MIN

    monkeypatch.setenv("HAUNT_RECALL_MAX_CHARS", "5000")
    assert recall_payload_cap() == 5000

    monkeypatch.setenv("HAUNT_RECALL_MAX_CHARS", "99999999")
    assert recall_payload_cap() == RECALL_PAYLOAD_MAX_CHARS_MAX


# ---------------------------------------------------------------------------
# format_recall_block: the [haunt ns=...] hook-injected block
# ---------------------------------------------------------------------------


def test_format_recall_block_unchanged_for_typical_hook_usage(recall_budget_env):
    """Both hooks call recall() with a fixed k=8 -- the default block
    budget must be a no-op for that, byte-for-byte the pre-C11 output."""
    from haunt.cursor_hook import format_recall_block
    from haunt.util import snippet

    hits = _small_hits(8)
    block = format_recall_block(hits, "myns")

    expected_lines = [f"[haunt ns=myns]"] + [
        f"{i}  rrf={h.score:.4f}  {h.tier}  {h.memory_id}  {snippet(h.content, 160)}"
        for i, h in enumerate(hits, 1)
    ]
    assert block == "\n".join(expected_lines)
    assert "truncated by haunt" not in block


def test_format_recall_block_claude_hook_shares_the_same_budgeted_function():
    from haunt.claude_hook import format_recall_block as claude_format
    from haunt.cursor_hook import format_recall_block as cursor_format

    assert claude_format is cursor_format


def test_format_recall_block_drops_tail_and_marks_it_when_over_budget(
    recall_budget_env, monkeypatch
):
    from haunt.cursor_hook import _recall_block_cap, format_recall_block

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", "1000")
    hits = [_hit(i, content=_big_content(150, tag=f"tok{i}")) for i in range(40)]
    block = format_recall_block(hits, "myns")

    assert "truncated by haunt" in block
    hit_lines = [ln for ln in block.splitlines() if ln[:1].isdigit()]
    assert 0 < len(hit_lines) < 40
    # DEFECT 2 (fixed): the real invariant is the block never exceeds its
    # own configured cap -- full stop. The previous version of this
    # assertion, `len(block) <= 1000 + 120  # cap plus the marker line
    # itself`, quietly padded the bound to fit the marker's own
    # uncounted cost instead of catching it: a test padded to accommodate
    # a bug is worse than no test, because it looks like coverage.
    assert len(block) <= _recall_block_cap()
    # The lines present are an exact, in-order prefix of the unbounded block.
    monkeypatch.delenv("HAUNT_RECALL_BLOCK_MAX_CHARS", raising=False)
    unbounded = format_recall_block(hits, "myns")
    unbounded_lines = unbounded.splitlines()
    assert unbounded_lines[: 1 + len(hit_lines)] == block.splitlines()[: 1 + len(hit_lines)]


@pytest.mark.parametrize("cap", [500, 501, 600, 700, 800, 1000, 1500, 2000, 4000])
@pytest.mark.parametrize("n_hits", [5, 10, 20, 40, 80])
def test_format_recall_block_marker_cost_never_pushes_block_over_cap(
    monkeypatch, cap, n_hits
):
    """DEFECT 2 (fixed): the trailing drop-marker line used to be
    appended unconditionally, uncounted against its own budget -- e.g.
    cap=600 with 20 hits measured a 644-char block, 44 over. This is the
    same shape of sweep the adversarial review used to find that (caps
    500-4000, hit counts 5-80); it is now a permanent regression test
    asserting the real invariant, len(block) <= cap, at every point.
    """
    from haunt.cursor_hook import format_recall_block

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", str(cap))
    hits = [_hit(i, content=_big_content(150, tag=f"tok{i}")) for i in range(n_hits)]
    block = format_recall_block(hits, "myns")
    assert len(block) <= cap


def test_format_recall_block_never_renders_untrusted_hits_regardless_of_budget(
    recall_budget_env, monkeypatch
):
    """Trust filtering happens before budgeting -- raw tool I/O must never
    appear in the auto-injected block even under budget pressure."""
    from haunt.cursor_hook import format_recall_block

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", "500")
    hits = [_hit(0, content="trusted note", trusted=True)] + [
        _hit(i, content=_big_content(300), trusted=False) for i in range(1, 20)
    ]
    block = format_recall_block(hits, "myns")
    assert "tool:" not in block
    for h in hits[1:]:
        assert h.memory_id not in block


def test_format_recall_block_degenerate_header_never_exceeds_cap_and_says_so(
    monkeypatch,
):
    """ROUND-3 DEFECT (fixed): a pathological namespace far longer than
    the configured cap used to bypass budgeting in two different ways.

    format_recall_block([], "n"*500) at a 500-char cap: the "(no
    memories)" early return did no budget accounting at all -- it came
    back 525 chars, 25 over cap, unconditionally.

    format_recall_block([one_hit], "n"*500): the packing loop's
    `available` went negative (header alone already exceeds cap), so the
    loop kept zero hit lines and the unconditional block[:cap] backstop
    sliced straight through the header -- landing exactly at cap by
    coincidence, but eating the drop-count marker along with it: a real
    hit was dropped with zero indication anything was omitted, the exact
    "never silent" failure this whole feature exists to prevent.

    Both real call sites (cursor_hook, claude_hook) clamp namespace to 80
    chars via safe_name() before calling this function, so this is
    format_recall_block enforcing its own documented contract for a
    degenerate configuration, not a path either real caller can trigger
    today.
    """
    from haunt.cursor_hook import format_recall_block

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", "500")
    ns = "n" * 500

    empty_block = format_recall_block([], ns)
    assert len(empty_block) <= 500
    assert "truncated by haunt" in empty_block

    one_hit_block = format_recall_block([_hit(0, content="short note")], ns)
    assert len(one_hit_block) <= 500
    assert "truncated by haunt" in one_hit_block


def test_format_recall_block_no_memories_path_respects_cap_even_when_header_fits_alone(
    monkeypatch,
):
    """DEFECT 1 (fixed), isolated from the header-itself-too-long case the
    test above covers: a namespace chosen so "[haunt ns=...]" alone fits
    the cap (491 of 500 chars), but "[haunt ns=...]\\n(no memories)"
    (491 + 14 = 505) does not. This is the one narrow band where the "(no
    memories)" early return's own missing budget check is the only thing
    that can be wrong -- namespaces long enough to blow the header alone
    (like the "n"*500 case above) hit a different guard first and never
    actually exercise this specific early return's own accounting. A
    mutation that strips the cap check from just this one return path
    would pass every other test in this file and only fail here.
    """
    from haunt.cursor_hook import format_recall_block

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", "500")
    ns = "n" * 480
    header_len = len(f"[haunt ns={ns}]")
    assert header_len <= 500, "test premise: header alone must fit"
    assert header_len + len("\n(no memories)") > 500, "test premise: body must overflow"

    block = format_recall_block([], ns)
    assert len(block) <= 500
    assert "truncated by haunt" in block


@pytest.mark.parametrize("cap", [500, 501, 600, 700, 800, 1000, 1500, 2000, 4000])
@pytest.mark.parametrize("n_hits", [5, 10, 20, 40, 80])
def test_format_recall_block_marker_present_and_intact_whenever_hits_are_dropped(
    monkeypatch, cap, n_hits
):
    """Distinct from test_format_recall_block_marker_cost_never_pushes_
    block_over_cap above: that test (and this suite generally, before
    this one) only ever checked the OUTER contract, len(block) <= cap.
    An unconditional block[:cap] backstop independently satisfies that
    contract even when the code that is supposed to reserve room for the
    drop-count marker is wrong or missing entirely -- every test in this
    file stayed green under exactly that mutation, because nothing
    checked that the marker itself, not just the block's length, survived
    intact. This asserts the actual claimed mechanism instead: whenever
    format_recall_block reports an omission by dropping trailing hit
    lines, the exact marker naming the correct dropped count must
    actually be present, complete, and un-sliced in the returned block.
    """
    from haunt.cursor_hook import _drop_marker, format_recall_block

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", str(cap))
    hits = [_hit(i, content=_big_content(150, tag=f"tok{i}")) for i in range(n_hits)]
    block = format_recall_block(hits, "myns")

    assert len(block) <= cap
    hit_lines = [ln for ln in block.splitlines() if ln[:1].isdigit()]
    dropped = n_hits - len(hit_lines)
    if dropped > 0:
        assert _drop_marker(dropped, cap) in block


def test_recall_block_cap_env_var_parses_and_clamps(monkeypatch):
    from haunt.cursor_hook import (
        RECALL_BLOCK_MAX_CHARS_DEFAULT,
        _recall_block_cap,
    )

    monkeypatch.delenv("HAUNT_RECALL_BLOCK_MAX_CHARS", raising=False)
    assert _recall_block_cap() == RECALL_BLOCK_MAX_CHARS_DEFAULT

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", "not-a-number")
    assert _recall_block_cap() == RECALL_BLOCK_MAX_CHARS_DEFAULT

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", "0")
    assert _recall_block_cap() == 500

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", "-100")
    assert _recall_block_cap() == 500

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", "2500")
    assert _recall_block_cap() == 2500

    monkeypatch.setenv("HAUNT_RECALL_BLOCK_MAX_CHARS", "99999999")
    assert _recall_block_cap() == 100_000
