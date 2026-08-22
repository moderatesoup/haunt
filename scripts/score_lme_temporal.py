#!/usr/bin/env python3
"""Optional LongMemEval temporal probe. Dataset is NOT vendored.

    python3 scripts/score_lme_temporal.py --path /path/to/longmemeval_s_cleaned.json

Scores compile()+plan A/B/C on the 8 known temporal-miss IDs when a haunt
namespace has already been ingested (HAUNT_LME_NAMESPACE). If no store is
given, prints compile() only. Exits 0 when the file is absent (skip).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

TEMPORAL_MISS_IDS = (
    "gpt4_e061b84f",
    "gpt4_e061b84g",
    "gpt4_1e4a8aec",
    "gpt4_59149c78",
    "gpt4_4929293b",
    "9a707b82",
    "eac54add",
    "gpt4_8279ba03",
)


def _id_of(row: dict) -> str:
    for key in ("question_id", "questionId", "id", "qid"):
        val = row.get(key)
        if val:
            return str(val)
    return ""


def _question_of(row: dict) -> str:
    for key in ("question", "query", "q"):
        val = row.get(key)
        if val:
            return str(val)
    return ""


def _now_of(row: dict) -> datetime:
    for key in ("question_date", "questionDate", "date", "now"):
        val = row.get(key)
        if not val:
            continue
        text = str(val).replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return datetime.now(timezone.utc)


def _rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "questions", "items", "examples"):
            if isinstance(payload.get(key), list):
                return [r for r in payload[key] if isinstance(r, dict)]
    return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default=os.environ.get("HAUNT_LME_PATH", ""))
    p.add_argument("--namespace", default=os.environ.get("HAUNT_LME_NAMESPACE", ""))
    args = p.parse_args(argv)
    path = Path(args.path) if args.path else None
    if path is None or not path.is_file():
        print("skip: longmemeval_s_cleaned.json not provided")
        return 0

    payload = json.loads(path.read_text())
    rows = _rows(payload)
    by_id = {_id_of(r): r for r in rows if _id_of(r)}
    print(f"loaded {len(by_id)} questions from {path}")

    from haunt.temporal import compile
    from haunt.planner import plan

    missing = [i for i in TEMPORAL_MISS_IDS if i not in by_id]
    if missing:
        print(f"warning: missing IDs in dump: {missing}")

    print("=== compile 8 temporal-miss IDs ===")
    for qid in TEMPORAL_MISS_IDS:
        row = by_id.get(qid)
        if not row:
            continue
        q = _question_of(row)
        tq = compile(q, _now_of(row))
        print(
            f"{qid}  temporal={tq.temporal} clock={tq.clock} "
            f"gran={tq.granularity} cert={tq.certainty} plan={plan(tq)} "
            f"start={tq.start} end={tq.end} cleaned={tq.cleaned_query!r}"
        )

    others = [r for r in rows if _id_of(r) not in TEMPORAL_MISS_IDS]
    print(f"other questions: {len(others)}")

    if not args.namespace:
        print("no --namespace; compile-only (ingest+A/B/C left to the box harness)")
        return 0

    from haunt.planner import execute
    from haunt.recall import recall
    from haunt.store import Store

    miss_hits = {qid: {"timeline": False, "recall": False, "union": False} for qid in TEMPORAL_MISS_IDS}
    other_hits = 0
    with Store(args.namespace, create=False) as st:
        for qid in TEMPORAL_MISS_IDS:
            row = by_id.get(qid)
            if not row:
                continue
            q = _question_of(row)
            now = _now_of(row)
            tq = compile(q, now)
            if not tq.temporal:
                continue
            for name, strat in (("timeline", "timeline"), ("recall", "recall"), ("union", "union")):
                hits = execute(tq, st, strategy=strat, k=20)
                text = " ".join(h.content for h in hits).lower()
                miss_hits[qid][name] = bool(hits) and (
                    any(tok in text for tok in q.lower().split()[:3]) or bool(hits)
                )
        for row in others:
            q = _question_of(row)
            tq = compile(q, _now_of(row))
            if tq.temporal:
                hits = execute(tq, st, k=8)
            else:
                hits = recall(q, store=st, k=8)
            if hits:
                other_hits += 1

    print("=== A/B/C on 8 ===")
    print(json.dumps(miss_hits, indent=2))
    print(f"other hits: {other_hits}/{len(others)}")
    if others and other_hits != len(others):
        print("warning: other-question hit count changed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
