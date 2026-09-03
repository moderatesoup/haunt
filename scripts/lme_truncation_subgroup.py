#!/usr/bin/env python3
"""Declare the LongMemEval subgroup the embedding truncation cap can reach.

    python3 scripts/lme_truncation_subgroup.py \
        --path /path/to/longmemeval_s_cleaned.json \
        --out tests/fixtures/lme_truncation/exposed_ids.json

`HAUNT_EMBED_MAX_LEN` truncates every text at a fixed token count before it is
embedded (`src/haunt/embed.py`). Raising that cap can only change a vector rank
for a question whose *answer-bearing gold turn* is longer than the cap: below
it, the same tokens are embedded either way and the two arms are the same run.

Scoring every question and reading off the long ones afterwards would be
choosing a subgroup with the results in hand. This script draws the subgroup
from the dataset and the tokenizer alone -- no retrieval, no scores -- so it
can be committed and digested before either arm is run. It reuses
`score_lme_retrieval._split`, so the working/held-out boundary is the same one
the scorer enforces.

Deterministic: same dataset plus same tokenizer yields the same file, and the
scorer records the file's sha256 in its report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_lme_retrieval as scorer  # noqa: E402

SCHEMA_VERSION = 1
# The shipped default in src/haunt/embed.py:_max_len(). The subgroup is
# everything the *current* ceiling cuts, so this is the number to draw against.
BASELINE_MAX_LEN = 512
TOKENIZER = Path.home() / ".haunt/models/BAAI-bge-m3/onnx/tokenizer.json"
BATCH = 128


def _token_lengths(tok: Any, texts: list[str]) -> list[int]:
    out: list[int] = []
    for i in range(0, len(texts), BATCH):
        out.extend(len(e.ids) for e in tok.encode_batch(texts[i : i + BATCH]))
    return out


def _answer_bearing(row: dict[str, Any]) -> list[str]:
    """Gold-session turns that literally contain the answer string.

    Same normalization the scorer's own `_gold_signals` uses, so "answer
    bearing" means the same thing in the subgroup and in the report.
    """
    gold = {str(s) for s in row.get("answer_session_ids") or []}
    answer = scorer._norm(row.get("answer"))
    if not answer:
        return []
    return [
        text
        for sid, session in zip(
            row.get("haystack_session_ids") or [],
            row.get("haystack_sessions") or [],
        )
        if str(sid) in gold
        for turn in session
        if answer in scorer._norm(text := str(turn.get("content") or ""))
    ]


def build(path: Path, tokenizer: Path, max_len: int) -> dict[str, Any]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer))
    rows = scorer._rows(json.loads(path.read_text(encoding="utf-8")))
    split = scorer._split(rows)
    member = {qid: name for name in scorer.SETS for qid in split[name]}

    exposed: dict[str, list[str]] = {name: [] for name in scorer.SETS}
    control: dict[str, list[str]] = {name: [] for name in scorer.SETS}
    longest = 0
    for row in rows:
        qid = str(row.get("question_id") or "")
        name = member.get(qid)
        if name is None:
            continue
        lengths = _token_lengths(tok, _answer_bearing(row))
        peak = max(lengths, default=0)
        longest = max(longest, peak)
        (exposed if peak > max_len else control)[name].append(qid)

    return {
        "schema_version": SCHEMA_VERSION,
        "criterion": (
            "the longest answer-bearing gold turn exceeds "
            f"{max_len} BGE-M3 tokens"
        ),
        "baseline_max_len": max_len,
        "longest_answer_bearing_turn_tokens": longest,
        "dataset": {
            "path": str(path),
            "sha256": scorer._digest(path),
            "questions": len(rows),
        },
        "tokenizer": str(tokenizer),
        "split_seed": scorer.SPLIT_SEED,
        "counts": {
            name: {
                "exposed": len(exposed[name]),
                "control": len(control[name]),
            }
            for name in scorer.SETS
        },
        "exposed": {name: sorted(exposed[name]) for name in scorer.SETS},
        "control": {name: sorted(control[name]) for name in scorer.SETS},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=str(TOKENIZER))
    p.add_argument("--max-len", type=int, default=BASELINE_MAX_LEN)
    args = p.parse_args(argv)

    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        print(f"skip: no LongMemEval dataset at --path {args.path}")
        return 0

    payload = build(path, Path(args.tokenizer).expanduser(), args.max_len)
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    for name in scorer.SETS:
        counts = payload["counts"][name]
        print(f"{name}: exposed {counts['exposed']} control {counts['control']}")
    print(f"longest answer-bearing turn {payload['longest_answer_bearing_turn_tokens']} tokens")
    print(f"{out}  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
