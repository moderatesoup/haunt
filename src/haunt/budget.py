"""Serialized-size budget for retrieval payloads leaving haunt.

Every machine surface that hands hits to a caller shares this one budget --
MCP `memory_recall` and `memory_timeline`, `haunt recall --json`, and the
dashboard recall endpoints -- so a cap configured once holds everywhere
rather than only on the surface it was first written for.
"""

from __future__ import annotations

import json
from typing import Any

from haunt.util import env_int


def serialize(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


# C11: retrieval responses had no size ceiling. `k` accepts up to 100 and
# every hit already carries full untruncated `content`, a redundant 200-char
# `snippet` of that same content, and (since the ranking-explanation work)
# a per-hit `explanation` object -- so a k=100 payload can be hundreds of
# KB, injected straight into agent context that has no way to page it. The
# dashboard's all-namespace recall multiplies that by every registered
# namespace.
#
# recall()/planned_recall() are library calls returning complete data --
# this is the presentation boundary with the real context budget, so the
# cap belongs here, at serialization, not in retrieval.
RECALL_PAYLOAD_MAX_CHARS_DEFAULT = 24_000
RECALL_PAYLOAD_MAX_CHARS_MIN = 2_000
RECALL_PAYLOAD_MAX_CHARS_MAX = 200_000


def recall_payload_cap() -> int:
    """HAUNT_RECALL_MAX_CHARS, clamped (util.env_int) so a bad value can't
    disable the budget (too small to hold even one hit's fixed overhead) or
    blow past what agent context can reasonably absorb.

    This bounds the *serialized hits list*, not the whole response envelope
    -- namespace/query/trust_policy/execution are small and do not grow with
    corpus size, so budgeting them adds complexity without addressing the
    actual failure mode.

    Default 24,000 chars (~6k tokens): generous for the common case (a
    handful of short conversational hits, or the hook's own fixed k=8
    lookups) while keeping a single tool call's result from being able to
    dominate a conversation's context budget the way an uncapped k=100 of
    ~12-16KB tool-I/O hits could.
    """
    return env_int(
        "HAUNT_RECALL_MAX_CHARS",
        default=RECALL_PAYLOAD_MAX_CHARS_DEFAULT,
        lo=RECALL_PAYLOAD_MAX_CHARS_MIN,
        hi=RECALL_PAYLOAD_MAX_CHARS_MAX,
    )


# json.dumps frames a list as "[" + ", ".join(hits) + "]". Summing per-hit
# sizes misses that framing -- 2n chars, ~200 at k=100 -- and reported a
# response as fitting a cap it actually exceeded. The cap is a promise about
# what crosses the boundary, so it is measured on the list.
_HITS_FRAME_CHARS = len("[]")
_HIT_SEPARATOR_CHARS = len(", ")


def _hits_size(hits: list[dict[str, Any]]) -> int:
    return len(serialize(hits))


def _rendered_hit(hit: dict[str, Any], content: str, keep: int) -> dict[str, Any]:
    """`hit` with `content` replaced by its first `keep` chars plus an
    inline "chars omitted" marker and two structured sibling keys. Pure
    data construction, no size reasoning -- the one place that assembles
    what a truncated hit looks like, shared by every `keep` candidate
    _truncate_hit_content measures.
    """
    omitted = len(content) - keep
    out = dict(hit)
    out["content"] = f"{content[:keep]}\n… [truncated by haunt: {omitted} chars omitted]"
    out["content_truncated"] = True
    out["content_omitted_chars"] = omitted
    return out


def _truncate_hit_content(hit: dict[str, Any], budget: int) -> dict[str, Any] | None:
    """Try to shrink one hit's content so the whole hit dict fits in
    `budget` serialized chars, always marked when it does.

    Last-resort path only: used when a single hit does not fit the recall
    budget even alone (e.g. k=1 against one huge raw-tool-I/O hit, or the
    budget was configured very small). Never silent -- haunt's whole
    premise is verbatim fidelity, so a shortened value must never still
    look like the complete record. Follows _cap_tool_io's precedent of an
    explicit inline marker in the text itself, plus structured sibling
    fields so a machine reader is not forced to parse marker text out of
    content to detect truncation.

    MEASUREMENT ONLY -- no estimation. Three prior rounds each patched a
    different way `_truncate_hit_content` tried to *predict* the
    JSON-serialized size of a hit instead of measuring it: ignoring
    `explanation.references` overhead, overestimating a fixed RESERVE and
    giving up on hits truncation could have saved, and (the shape that
    finally forced this rewrite) assuming one kept content char costs one
    serialized char -- false whenever content is escape-heavy (quotes,
    backslashes, control chars all expand under json.dumps), which made
    a single over-budget estimate abandon the hit entirely instead of
    trying a smaller `keep`. Patching the arithmetic a fourth time would
    just add a fourth escape hatch, so there is no arithmetic left here
    to be wrong: every candidate below is checked by actually building it
    and calling `serialize` on it, exactly like the caller's own size check.

    `budget` bounds the *entire* hit dict, not just content -- a hit still
    carries memory_id/tier/timestamps and the whole explanation object
    (rrf_contributions, references, filters, ...) alongside content, and
    that fixed scaffolding is often itself well over a thousand chars.

    Returns None -- never a hit whose serialized size still exceeds
    `budget` -- when truncating content cannot make this hit fit:

      * content is not a string (e.g. a sqlite-blob envelope from
        json_safe_sqlite), so there is nothing to slice; or
      * even with content emptied out completely (keep=0, i.e. only the
        marker and its two sibling keys remain), the hit's non-content
        scaffolding (memory_id, tier, timestamps, and the whole
        `explanation` object -- rrf_contributions, filters, and
        `references`, which can itself carry an unbounded
        correction_lineage.correction_ids list or a multi-KB validated
        provenance envelope) still measures over `budget`, by an actual
        measurement of that exact keep=0 rendering, not an estimate of
        it.

    Either way the overage lives entirely outside `content`, so cutting
    `content` would destroy real, verbatim data for zero size benefit --
    exactly what haunt must never do. Callers must drop a hit this
    returns None for rather than ship it over budget with a truncation
    marker that didn't actually help.
    """
    content = hit.get("content")
    if budget <= 0:
        return None
    if not isinstance(content, str):
        # Non-string content (e.g. a sqlite-blob envelope from
        # json_safe_sqlite) cannot be sliced -- there is no way to shrink
        # it, so it cannot help this hit fit. Marking it "truncated"
        # without changing a single byte would be exactly the lie this
        # function must not tell.
        return None
    if len(serialize(hit)) <= budget:
        # Defensive only: the caller only reaches this function when the
        # hit's real measured size didn't fit, so this shouldn't occur in
        # practice. Measuring first costs one cheap check and makes this
        # function correct standalone, not just under its one caller's
        # current control flow.
        return hit

    # keep=0: content entirely emptied, only the marker and its two
    # sibling keys remain. The smallest this hit's content can ever make
    # it. If even this measures over budget, the overage lives entirely
    # in fixed scaffolding no amount of content truncation can touch --
    # drop the hit rather than ship a marker that saved nothing.
    if len(serialize(_rendered_hit(hit, content, 0))) > budget:
        return None

    # A fitting `keep` exists in [0, len(content)] -- keep=0 was just
    # measured as fitting -- so binary search over [0, len(content)] finds
    # a large fitting `keep` in O(log len(content)) real measurements,
    # about 17 for a 100KB hit.
    #
    # Size is *very nearly* monotonic non-decreasing in `keep`: each extra
    # raw char only ever adds to the JSON-escaped output (a plain char
    # costs 1, `"`/`\` cost 2, control chars up to 6 -- never 0 or
    # negative). It is not strictly monotonic, though, and the earlier
    # version of this comment claimed a proof it does not have: `omitted`
    # appears twice in the payload, once inside the marker text and again
    # as the content_omitted_chars integer, so crossing a power-of-ten
    # boundary downward can shrink the total by 2 while the newly kept
    # char adds only 1. An adversarial review reproduced those one-index
    # "notches" directly.
    #
    # Safety does not rest on monotonicity. Every candidate is built and
    # measured, and the value returned is always one that was itself
    # measured to fit -- either the keep=0 floor or a passing midpoint --
    # never a value inferred from the ordering. So a notch can at worst
    # cost a byte or two of kept content; it cannot produce an
    # over-budget result. Brute-force comparison across ~150 adversarial
    # cases (escape-heavy, control chars, astral emoji, lone surrogates)
    # found the search agreeing with the true optimum every time.
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(serialize(_rendered_hit(hit, content, mid))) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return _rendered_hit(hit, content, lo)


def apply_recall_budget(
    hit_dicts: list[dict[str, Any]], *, k: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bound the serialized size of a hits list crossing a machine surface.

    recall()/planned_recall() already selected and ranked these rows; this
    function only decides how much TEXT of that fixed, ordered list is
    allowed to cross into agent context. It never reorders hits, never
    changes which rows were selected, and never fabricates a hit -- it can
    only shorten what one hit emits (always marked) or drop a suffix of
    the already-ranked list (also always marked).

    Degrade order, least to most destructive:
      1. No-op if the untouched payload already fits. This is the common
         case (small corpora, the hook's fixed k=8 lookups, ordinary
         k=8 default calls) and must stay byte-for-byte unchanged.
      2. Drop the redundant `snippet` field. It is a 200-char derivative
         of `content`, which is always still present in full when
         `snippet` is -- pure waste once it is `content`, not `snippet`,
         that is crossing the boundary.
      3. Keep hits whole, in rank order, until the next one would overflow
         the budget; drop the remaining suffix. Chosen over truncating
         every hit a little: this system's premise is verbatim fidelity,
         so a caller is better served by fewer *complete*, trustworthy
         hits than by many partially-cut ones it cannot fully rely on.
         Every hit that IS returned here is untouched (post step 2).
      4. Only if the very first hit alone cannot fit whole (k=1 against
         one huge hit, or a very small configured budget): try truncating
         that single hit's `content` (see _truncate_hit_content). This
         step only ever shortens `content` -- a hit's non-content
         scaffolding (memory_id, tier, timestamps, and the whole
         `explanation` object: rrf_contributions, filters, and
         `references`, which by itself can carry an unbounded
         correction_lineage.correction_ids list or a multi-KB validated
         provenance envelope) is never touched, because silently
         shortening haunt's own retrieval/trust/correction metadata would
         misrepresent it, which is worse than dropping the hit.

         If truncating content makes the hit fit, that one hit is
         returned, marked partial via content_truncated /
         content_omitted_chars, and every hit after it (by rank) is
         still dropped, not also truncated -- one marked-partial hit,
         never many.

         If truncating content CANNOT make it fit -- because the overage
         lives entirely in that untouched fixed scaffolding, e.g. a
         600-entry correction_lineage or a multi-KB provenance envelope,
         so the hit is still oversized even with content emptied out --
         the hit is dropped instead of being shipped over budget wearing
         a truncation marker that saved nothing. This is the one place
         this function can return fewer hits than one for a nonempty
         input: `hits_returned` can be 0 even though `hits_available` is
         not. That is a deliberate, narrow exception to "a nonempty
         result never returns zero hits": between that guarantee and
         `recall_budget` honestly describing what crossed the boundary,
         this function always keeps the second promise. Returning an
         oversized hit with `applied: True` and `hits_dropped: 0` would
         be a silent lie about the one invariant this function exists to
         uphold (the serialized result never exceeds `cap`); an honestly
         empty `hits` with `hits_available: 1`, `hits_returned: 0`,
         `hits_dropped: 1` tells the caller exactly what happened -- the
         budget, not the corpus, produced zero hits -- so they can retry
         with a larger HAUNT_RECALL_MAX_CHARS if they specifically need
         that one hit.
    """
    cap = recall_payload_cap()
    meta: dict[str, Any] = {
        "version": 1,
        "max_chars": cap,
        "k_requested": k,
        "hits_available": len(hit_dicts),
        "hits_returned": len(hit_dicts),
        "hits_dropped": 0,
        "snippet_dropped": False,
        "content_truncated_memory_ids": [],
        "applied": False,
    }
    if not hit_dicts:
        return hit_dicts, meta
    baseline_total = _hits_size(hit_dicts)
    if baseline_total <= cap:
        return hit_dicts, meta

    # Step 2: strip the redundant snippet (content, right next to it, is
    # never removed by this step -- only the derivative copy of it).
    slim = [
        {key: value for key, value in hit.items() if key != "snippet"}
        if "snippet" in hit
        else hit
        for hit in hit_dicts
    ]
    slim_total = _hits_size(slim)
    if slim_total < baseline_total:
        meta["snippet_dropped"] = True
        meta["applied"] = True
    if slim_total <= cap:
        return slim, meta

    # Step 3 + 4: strict prefix of the rank-ordered list. A hit later in
    # rank order is never substituted in over an earlier one just because
    # it happens to be smaller -- that would let hit size influence which
    # rows are effectively selected, which is exactly what this function
    # must not do.
    kept: list[dict[str, Any]] = []
    used = _HITS_FRAME_CHARS
    for hit in slim:
        # A hit appended to a non-empty list also costs the ", " before it;
        # the enclosing "[]" is already charged into `used`.
        size = len(serialize(hit)) + (_HIT_SEPARATOR_CHARS if kept else 0)
        if used + size <= cap:
            kept.append(hit)
            used += size
            continue
        if not kept:
            truncated = _truncate_hit_content(hit, max(0, cap - used))
            if truncated is not None:
                kept.append(truncated)
                meta["content_truncated_memory_ids"].append(hit.get("memory_id"))
        # Whether or not the first hit could be salvaged by truncation,
        # every hit from here on (by rank) is dropped, never truncated in
        # its place: only ever one marked-partial hit, and a hit is never
        # promoted ahead of a higher-ranked one just because it is
        # smaller (see the docstring above). If `truncated` came back
        # None, `kept` is still empty here, so this call returns zero
        # hits for a nonempty `hit_dicts` -- see the docstring's step 4
        # for why that is the correct, deliberate outcome, not a bug.
        break

    meta["hits_returned"] = len(kept)
    meta["hits_dropped"] = len(slim) - len(kept)
    meta["applied"] = True
    return kept, meta
