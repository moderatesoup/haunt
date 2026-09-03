"""Deterministic tail segmentation for memories longer than the embed window.

`HAUNT_EMBED_MAX_LEN` truncates every text before it is embedded
(`haunt.embed.OnnxEmbedder.__init__` calls `enable_truncation`). A memory
longer than that window is stored and FTS-indexed whole, but only its first
`max_len` tokens ever reach a vector. On the corpora this was measured
against, that silently removed roughly two thirds of the tokens from vector
search -- see the L21 register entry in BACKLOG.md.

This module plans the *remainder*. Span 0 is the head window: it is already
embedded, its vector already lives in `memories.embedding` / `vec_memories`,
and nothing here touches it. `plan()` returns spans 1..n covering the text
from just before the head boundary to the end, so an existing namespace gains
tail coverage without re-embedding a single vector it already has.

Two properties everything else depends on:

- **Deterministic.** The same text, window and overlap yield byte-identical
  spans on any machine, so a re-run rebuilds exactly what it replaced and a
  test can assert offsets rather than approximate them.
- **Verbatim.** A span is a character range of the stored content, never a
  rewrite of it. Haunt embeds what it stored (`docs/MEMORY_CONTRACT.md` -- no
  distillation), and a span that is a plain substring keeps that true.

Windows overlap by `overlap` tokens so a phrase lying across a boundary is
whole inside at least one span. Overlap is also what makes span 1 start
*before* the head boundary: a sentence cut in half by truncation is otherwise
in neither vector.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from haunt.util import env_int

# Tokens of overlap between neighbouring windows. 64 is an eighth of the
# default 512-token window: enough to hold a long sentence whole across a
# boundary, small enough that it does not meaningfully inflate span count.
DEFAULT_OVERLAP = 64
# Ceiling on spans per memory, counting the head. A pathological row (an
# unbounded tool dump that escaped HAUNT_TOOL_IO_MAX_CHARS) must not turn one
# observe into hundreds of model calls. At the default 512-token window and
# 64-token overlap the stride is 448, so 32 windows reach about 14,400
# tokens -- past the longest row measured in the live corpora (12,436) with
# headroom. The cap is counted in spans, not tokens, because model calls are
# the resource being bounded; that means its reach shrinks with a smaller
# HAUNT_EMBED_MAX_LEN. Truncation past the cap is recorded, never silent.
DEFAULT_MAX_SPANS = 32
# Do not emit a trailing sliver. A window holding fewer than this many new
# tokens adds a near-duplicate of its predecessor's overlap and little else.
MIN_TAIL_TOKENS = 16
# Fallback only: characters per token when no offset-capable tokenizer is
# reachable (the fastembed backend exposes none). Deliberately conservative --
# undershooting produces more, smaller spans, which costs throughput. It never
# produces a gap, because windows still overlap.
FALLBACK_CHARS_PER_TOKEN = 3.0

SPANS_ENABLED_ENV = "HAUNT_EMBED_SPANS"
OVERLAP_ENV = "HAUNT_EMBED_SPAN_OVERLAP"
MAX_SPANS_ENV = "HAUNT_EMBED_MAX_SPANS"


@dataclass(frozen=True)
class Span:
    """One tail window: `text[start_char:end_char]`, `ord` >= 1."""

    ord: int
    start_char: int
    end_char: int

    def slice(self, text: str) -> str:
        return text[self.start_char : self.end_char]


@dataclass(frozen=True)
class SpanPlan:
    """Spans 1..n for one memory, plus whether the cap cut the text short."""

    spans: tuple[Span, ...]
    covered_chars: int
    total_chars: int
    truncated: bool
    method: str

    def __bool__(self) -> bool:
        return bool(self.spans)


EMPTY = SpanPlan(spans=(), covered_chars=0, total_chars=0, truncated=False, method="none")


def enabled() -> bool:
    """Tail spans are on unless explicitly disabled.

    Off is the pre-fix behavior: the head window is embedded and the rest of
    a long memory stays out of vector search.
    """
    raw = (os.environ.get(SPANS_ENABLED_ENV) or "").strip().lower()
    return raw not in {"0", "off", "false", "no"}


def overlap_tokens(max_len: int) -> int:
    """Overlap, clamped below half the window so windows always advance."""
    raw = env_int(OVERLAP_ENV, default=DEFAULT_OVERLAP, lo=0, hi=4096)
    return max(0, min(raw, max(0, max_len // 2)))


def max_spans() -> int:
    return env_int(MAX_SPANS_ENV, default=DEFAULT_MAX_SPANS, lo=1, hi=512)


def _offsets(tokenizer: object, text: str) -> list[tuple[int, int]] | None:
    """Character offsets for every token, or None if this tokenizer cannot.

    The encoder Haunt embeds with has truncation enabled, so it can never
    report an offset past the window. Callers pass a separate un-truncated
    tokenizer; anything that does not answer with usable offsets falls back.
    """
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None
    try:
        encoding = encode(text)
        raw = list(getattr(encoding, "offsets", ()) or ())
    except Exception:
        return None
    offsets = [
        (int(start), int(end))
        for start, end in raw
        if int(end) > int(start)
    ]
    return offsets or None


def _windows(count: int, max_len: int, overlap: int, cap: int) -> list[tuple[int, int]]:
    """Token index ranges for spans 1..n, or [] when the head covers it all.

    The head is tokens [0, max_len). Span 1 backs up by `overlap` so the
    boundary itself is inside a window, and each later span advances by
    `max_len - overlap`.

    Coverage runs to the last token or the cap stops it -- never anything in
    between. A trailing sliver is not dropped: the previous window's overlap
    is behind it, not over it, so dropping it would leave the final tokens of
    a memory in no vector at all, which is the exact defect this module
    exists to remove. Instead the last window is anchored to the end, which
    keeps it inside `max_len` and costs nothing extra.
    """
    if count <= max_len:
        return []
    stride = max(1, max_len - overlap)
    out: list[tuple[int, int]] = []
    start = max(0, max_len - overlap)
    while start < count and len(out) < cap - 1:
        end = min(count, start + max_len)
        out.append((start, end))
        if end >= count:
            break
        next_start = start + stride
        # The window after this one would add only a sliver of new tokens.
        # Anchor it to the end instead of emitting a near-duplicate.
        if count - end < MIN_TAIL_TOKENS and len(out) < cap - 1:
            out.append((max(0, count - max_len), count))
            break
        start = next_start
    return out


def _encoded_len(tokenizer: object, text: str) -> int | None:
    """Token count the encoder will see for `text`, or None if unknowable."""
    try:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)  # type: ignore[attr-defined]
    except Exception:
        try:
            return len(getattr(tokenizer.encode(text), "ids", ()))  # type: ignore[attr-defined]
        except Exception:
            return None


def _fit_last_span(
    spans_out: list[Span],
    offsets: list[tuple[int, int]],
    ranges: list[tuple[int, int]],
    text: str,
    tokenizer: object,
    ceiling: int,
) -> None:
    """Shrink the final span from the front until the encoder will take it whole.

    Subtracting the special-token overhead is necessary but not sufficient:
    re-tokenizing a substring is not guaranteed to reproduce the parent
    tokenization, and the drift observed on BGE-M3 is about a token. Interior
    spans absorb that in the next window's overlap. The final span has no next
    window, so an overshoot there loses the last tokens of the memory outright
    -- the exact defect this module exists to remove.

    The front is what moves, never the end: the beginning of the final window
    is already inside its predecessor's overlap, the end is not covered by
    anything.
    """
    if not spans_out:
        return
    last = spans_out[-1]
    lo, hi = ranges[-1]
    while lo < hi - 1:
        length = _encoded_len(tokenizer, text[last.start_char : last.end_char])
        if length is None or length <= ceiling:
            return
        lo += 1
        last = Span(ord=last.ord, start_char=offsets[lo][0], end_char=last.end_char)
        spans_out[-1] = last


def plan(
    text: str,
    *,
    max_len: int,
    tokenizer: object | None = None,
    overlap: int | None = None,
    cap: int | None = None,
    special_overhead: int = 0,
) -> SpanPlan:
    """Plan the tail spans of `text`. Returns EMPTY when the head covers it.

    `tokenizer` must be an un-truncated tokenizer exposing `encode(text)` with
    `.offsets`. Without one, windows are cut on an estimated character width;
    the spans stay deterministic and overlapping, only the boundaries move.

    `special_overhead` is how many tokens the encoder adds to every input
    (CLS/SEP). Windows are planned that much narrower so a span arrives inside
    the window it was planned for instead of losing its tail to truncation.
    """
    if not text or max_len <= 0 or not enabled():
        return EMPTY
    ceiling = max_len
    max_len = max(1, max_len - max(0, special_overhead))
    step = overlap_tokens(max_len) if overlap is None else max(0, min(overlap, max_len // 2))
    limit = max_spans() if cap is None else max(1, cap)
    total = len(text)

    offsets = _offsets(tokenizer, text) if tokenizer is not None else None
    if offsets is not None:
        ranges = _windows(len(offsets), max_len, step, limit)
        planned = [
            Span(
                ord=i,
                start_char=offsets[lo][0],
                end_char=offsets[hi - 1][1],
            )
            for i, (lo, hi) in enumerate(ranges, start=1)
        ]
        if planned and tokenizer is not None:
            _fit_last_span(
                planned, offsets, ranges, text, tokenizer,
                max(1, ceiling - max(0, special_overhead)),
            )
        spans = tuple(planned)
        method = "token_offsets"
        # No ranges with a text longer than the window means the cap allowed
        # no tail window at all -- the most complete truncation there is, and
        # the one a `bool(ranges)` guard would silently report as none.
        if not ranges:
            truncated = len(offsets) > max_len
        else:
            truncated = ranges[-1][1] < len(offsets)
    else:
        # Character mirror of _windows. Every threshold is scaled into
        # characters here -- comparing a character remainder against a token
        # constant would silently change the rule at a different rate for
        # every language.
        width = max(1, int(max_len * FALLBACK_CHARS_PER_TOKEN))
        stride = max(1, int((max_len - step) * FALLBACK_CHARS_PER_TOKEN))
        min_tail = max(1, int(MIN_TAIL_TOKENS * FALLBACK_CHARS_PER_TOKEN))
        if total <= width:
            return EMPTY
        spans_out: list[Span] = []
        start = max(0, width - int(step * FALLBACK_CHARS_PER_TOKEN))
        while start < total and len(spans_out) < limit - 1:
            end = min(total, start + width)
            spans_out.append(
                Span(ord=len(spans_out) + 1, start_char=start, end_char=end)
            )
            if end >= total:
                break
            if total - end < min_tail and len(spans_out) < limit - 1:
                spans_out.append(
                    Span(
                        ord=len(spans_out) + 1,
                        start_char=max(0, total - width),
                        end_char=total,
                    )
                )
                break
            start += stride
        spans = tuple(spans_out)
        method = "char_estimate"
        truncated = (spans[-1].end_char < total) if spans else total > width

    if not spans:
        # A cap of 1 leaves no room for a tail window, so the text past the
        # head is dropped. Returning EMPTY here would be a lie -- EMPTY means
        # "the head covers this text" -- so say what happened instead.
        if truncated:
            return SpanPlan(
                spans=(),
                covered_chars=0,
                total_chars=total,
                truncated=True,
                method=method,
            )
        return EMPTY
    return SpanPlan(
        spans=spans,
        covered_chars=spans[-1].end_char,
        total_chars=total,
        truncated=truncated,
        method=method,
    )
