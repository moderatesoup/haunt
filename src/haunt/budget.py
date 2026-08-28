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


# Which keys hold a row's shrinkable bulk, listed most expendable first.
#
# A recall hit's bulk is `content`. An *event* row's is not: hooks write
# tool rows with content="" and the payload in tool_input/tool_output, each
# capped at HAUNT_TOOL_IO_MAX_CHARS (12,000 by default) -- so one hook-captured
# Bash event can carry 24,000 chars while its `content` is empty. Shrinking
# only `content` there saves nothing, leaves the row over budget, and made
# memory_timeline return zero events for a namespace that had them.
#
# Order is the sacrifice order: tool_output (the bulkiest, and reconstructible
# by re-running the tool) goes before tool_input (the command that produced
# it), and verbatim conversational `content` is cut last.
HIT_TEXT_FIELDS = ("content",)
EVENT_TEXT_FIELDS = ("tool_output", "tool_input", "content")


def _row_identity(row: dict[str, Any]) -> Any:
    """The id a caller can use to re-fetch this row in full.

    Recall hits key on `memory_id`; event rows key on `id`. Reading only
    `memory_id` filled the budget's truncated-id list with None, so a caller
    could see that something was cut but not which row.
    """
    identity = row.get("memory_id")
    return identity if identity is not None else row.get("id")


def _rendered_field(row: dict[str, Any], field: str, keep: int) -> dict[str, Any]:
    """`row` with `field` replaced by its first `keep` chars plus an
    inline "chars omitted" marker and two structured sibling keys
    (`<field>_truncated`, `<field>_omitted_chars`). Pure data
    construction, no size reasoning -- the one place that assembles what a
    truncated row looks like, shared by every `keep` candidate
    _truncate_row_text measures.
    """
    value = row[field]
    omitted = len(value) - keep
    out = dict(row)
    out[field] = f"{value[:keep]}\n… [truncated by haunt: {omitted} chars omitted]"
    out[f"{field}_truncated"] = True
    out[f"{field}_omitted_chars"] = omitted
    return out


def _truncate_row_text(
    row: dict[str, Any], budget: int, fields: tuple[str, ...]
) -> dict[str, Any] | None:
    """Try to shrink one row's text fields so the whole dict fits in
    `budget` serialized chars, always marked when it does.

    Last-resort path only: used when a single row does not fit the recall
    budget even alone. Never silent -- a shortened value must never still look
    like the complete record, so each cut field carries an inline marker and
    gains structured `<field>_truncated`/`<field>_omitted_chars` siblings.

    `fields` is in sacrifice order, most expendable first. Each field is cut
    only as far as it has to be: the first field is kept as long as the
    remaining fields at full length still leave the row over budget, and a
    later field is touched only if emptying every earlier one was not enough.
    A row whose bulk is in an earlier field therefore keeps the later ones
    byte-for-byte.

    MEASUREMENT ONLY -- no estimation. Every candidate is built and passed to
    `serialize`, exactly like the caller's own size check. Predicting a row's
    JSON size is not possible here: one kept char can cost up to six
    serialized ones.

    `budget` bounds the *entire* row dict. A hit still carries
    memory_id/tier/timestamps and the whole explanation object alongside
    content, often itself well over a thousand chars.

    Returns None -- never a row still over `budget` -- when truncation cannot
    help, because the overage then lives entirely outside `fields`:

      * no listed field holds a non-empty string (e.g. a sqlite-blob envelope
        from json_safe_sqlite), so there is nothing to slice; or
      * even with every listed field emptied out completely (keep=0, markers
        and siblings only, measured rather than estimated), the row's
        remaining scaffolding still exceeds `budget`.

    Cutting text there would destroy verbatim data for zero size benefit.
    Callers must drop such a row rather than ship it over budget wearing a
    truncation marker that did not help.
    """
    if budget <= 0:
        return None
    # A non-string field (e.g. a sqlite-blob envelope from json_safe_sqlite)
    # cannot be sliced, so it cannot help this row fit; marking it
    # "truncated" without changing a single byte would be exactly the lie
    # this function must not tell. An already-empty field is skipped for the
    # same reason -- its marker would add chars, not remove them.
    shrinkable = [
        field for field in fields if isinstance(row.get(field), str) and row.get(field)
    ]
    if not shrinkable:
        return None
    if len(serialize(row)) <= budget:
        # Defensive only: the caller only reaches this function when the
        # row's real measured size didn't fit, so this shouldn't occur in
        # practice. Measuring first costs one cheap check and makes this
        # function correct standalone, not just under its one caller's
        # current control flow.
        return row

    # keep=0 everywhere: every shrinkable field emptied, only markers and
    # their sibling keys left. The smallest this row's text can ever make
    # it. If even this measures over budget, the overage lives entirely in
    # fixed scaffolding no amount of truncation can touch -- drop the row
    # rather than ship markers that saved nothing.
    floor = row
    for field in shrinkable:
        floor = _rendered_field(floor, field, 0)
    if len(serialize(floor)) > budget:
        return None

    # A fitting assignment exists -- `floor` was just measured as fitting --
    # so walk the fields in sacrifice order, binary-searching each one's
    # largest fitting `keep` while every later field is still at full length.
    # O(len(fields) * log len(value)) real measurements, about 17 per field
    # for a 100KB value.
    #
    # Size is *nearly* monotonic non-decreasing in `keep` -- each extra raw
    # char only ever adds to the escaped output -- but not strictly:
    # `omitted` appears both in the marker text and as
    # <field>_omitted_chars, so crossing a power-of-ten boundary downward
    # can shrink the total by 2 while the newly kept char adds 1.
    #
    # Safety does not rest on monotonicity. Every candidate is built and
    # measured, and the returned value is always one measured to fit --
    # the all-fields floor or a passing midpoint -- never inferred from the
    # ordering. A notch can cost a byte or two of kept text; it cannot
    # produce an over-budget result.
    out = row
    for field in shrinkable:
        if len(serialize(out)) <= budget:
            break
        lo, hi = 0, len(out[field])
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(serialize(_rendered_field(out, field, mid))) <= budget:
                lo = mid
            else:
                hi = mid - 1
        out = _rendered_field(out, field, lo)
    return out


def apply_recall_budget(
    hit_dicts: list[dict[str, Any]],
    *,
    k: int,
    text_fields: tuple[str, ...] = HIT_TEXT_FIELDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bound the serialized size of a hits list crossing a machine surface.

    recall()/planned_recall() already selected and ranked these rows; this
    only decides how much TEXT of that fixed, ordered list crosses into agent
    context. It never reorders, never changes which rows were selected, and
    never fabricates a hit.

    Degrade order, least to most destructive:
      1. No-op if the untouched payload already fits. The common case; stays
         byte-for-byte unchanged.
      2. Drop the redundant `snippet` field -- a 200-char derivative of
         `content`, which is always still present in full beside it.
      3. Keep hits whole, in rank order, until the next would overflow; drop
         the remaining suffix. Fewer complete, trustworthy hits beat many
         partially-cut ones. Every hit returned here is untouched.
      4. Only if the first hit alone cannot fit whole: try truncating that one
         hit's `text_fields` (see _truncate_row_text). Everything else --
         memory_id, tier, timestamps, the whole `explanation` object -- is
         never touched, because silently shortening haunt's own
         retrieval/trust/correction metadata misrepresents it.

         If that makes the hit fit, it is returned marked partial via
         <field>_truncated/<field>_omitted_chars on each cut field, and every
         hit after it is dropped, not also truncated -- one marked-partial
         hit, never many.

         If it cannot (the overage lives entirely in that untouched
         scaffolding, e.g. a 600-entry correction_lineage), the hit is dropped
         instead. This is the one place a nonempty input can return zero hits:
         `hits_returned` 0 with `hits_available` 1. Deliberate -- an oversized
         hit shipped with `applied: True, hits_dropped: 0` would lie about the
         single invariant this function exists to uphold, while an honestly
         empty `hits` tells the caller the budget, not the corpus, produced
         it, so they can retry with a larger HAUNT_RECALL_MAX_CHARS.

    `text_fields` names the keys holding this kind of row's bulk, most
    expendable first. It defaults to a recall hit's (`content`); event rows
    carry theirs in tool_output/tool_input and pass EVENT_TEXT_FIELDS, without
    which step 4 empties an already-empty `content` and drops the row.

    `content_truncated_memory_ids` names every row step 4 shortened, by
    `memory_id` for a hit and `id` for an event row, whichever the row has.
    Which field was cut is marked on the row itself.
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
            truncated = _truncate_row_text(hit, max(0, cap - used), text_fields)
            if truncated is not None:
                kept.append(truncated)
                meta["content_truncated_memory_ids"].append(_row_identity(hit))
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
