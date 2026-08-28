#!/usr/bin/env python3
"""Judge-free LongMemEval retrieval scoring. Dataset is NOT vendored.

    HAUNT_HOME=/tmp/haunt-lme python3 scripts/score_lme_retrieval.py \
        --path /path/to/longmemeval_s_cleaned.json --k 10 --fts-only \
        --set working --out lme_retrieval_report.json

Each question's haystack is ingested into its own namespace through
Store.observe, then scored with recall(). The primary metric is
gold-evidence retrieval: a question counts as hit@N when one of the first
N hits came from a session listed in answer_session_ids. No LLM judges
the answer text.

The 500 questions are split, seeded and stratified by question_type, into
a working set and a held-out set. Diagnosis belongs to the working set;
held-out exists to show that a later fix generalized. The two are never
merged into one number, --set defaults to working, and held-out
per-question diagnostics stay suppressed unless --held-out-detail asks.

Only the public store API is used, so the same file scores an older tree
and a newer one and the before/after comparison stays valid. The report
carries the tree SHA, the dataset digest and the embedding coverage that
distinguishes a real hybrid run from an FTS-only run wearing that label.

HAUNT_HOME is forced away from the operator's real store; the run writes
only into an isolated tree. Hybrid runs resolve the embedding model under
HAUNT_HOME/models unless HAUNT_MODEL_CACHE points at an existing cache.

Exits 0 when the dataset file is absent, naming the path it looked for.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 2
CUTOFFS = (1, 3, 5, 10)
SETS = ("working", "held-out")
SPLIT_SEED = 20240517
WORKING_FRACTION = 0.7
# Deep probe for missed questions only: gold inside this pool but outside k
# separates a ranking failure from a candidate-generation failure.
PROBE_K = 100
# Below this share of the question's content words present in the gold text,
# a lexical backend has essentially nothing to match on.
LEXICAL_GAP = 0.15
# A gold turn longer than this buries its evidence in one indexed unit.
CHUNK_CHARS = 2000
MIN_COVERAGE = 0.99
# Per-question cost outside the timed section: namespace open/close,
# the mapping cross-check and the report rewrite.
OVERHEAD_S = 0.15
# Marks how many haystack sessions are committed. set_meta() commits, so the
# marker lands in the same transaction as the observes it covers and an
# interrupted run resumes at a session boundary rather than mid-session.
SESSIONS_DONE = "lme_sessions_done"
COMMIT_EVERY = 8
# clamp_limit caps process_embedding_jobs at 100 rows per call.
EMBED_BATCH = 100
LME_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\D+(\d{2}):(\d{2})")
WORD = re.compile(r"[a-z0-9']+")
WS = re.compile(r"\s+")
STOPWORDS = frozenset(
    """a about after all also am an and any are as at be been before being but by
    can did do does doing done for from had has have how i if in into is it its me
    my no not of on or our out over so some than that the their them then there
    these they this those to up was we were what when where which who why will with
    would you your""".split()
)
MECHANISMS = (
    "needs_abstention",
    "ranking_fusion",
    "lexical_gap",
    "temporal_reasoning",
    "multi_session_aggregation",
    "chunking_segmentation",
    "unclassified",
)
# Enough to recompute the held-out metrics and to resume, without the
# diagnostic payload that would invite tuning against the held-out set.
HELD_OUT_PUBLIC = (
    "question_id",
    "question_type",
    "first_gold_rank",
    "answer_substring_rank",
    "miss_mechanism",
    "miss_mechanism_secondary",
)


def _isolated_home() -> Path:
    """Guarantee this harness can never write to the operator's real store."""
    real = (Path.home() / ".haunt").resolve()
    raw = os.environ.get("HAUNT_HOME")
    if raw:
        home = Path(raw).expanduser().resolve()
        if home == real or real in home.parents:
            raise SystemExit(f"refusing to run inside the real store: {home}")
    else:
        home = Path(tempfile.gettempdir()).resolve() / "haunt-lme-eval"
        os.environ["HAUNT_HOME"] = str(home)
    home.mkdir(parents=True, exist_ok=True)
    return home


def _git(root: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _tree() -> dict[str, Any]:
    """Identify the tree under test so a before/after comparison is anchored."""
    import haunt

    root = Path(haunt.__file__).resolve().parents[2]
    return {
        "root": str(root),
        "sha": _git(root, "rev-parse", "HEAD") or "unknown",
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(_git(root, "status", "--porcelain")),
    }


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _event_time(value: str) -> str:
    """LongMemEval stamps look like '2023/05/20 (Sat) 02:21'; store as UTC."""
    m = LME_DATE.search(str(value or ""))
    if not m:
        raise ValueError(f"unparseable haystack date: {value!r}")
    y, mo, d, h, mi = (int(g) for g in m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).isoformat(
        timespec="microseconds"
    )


def _norm(text: str) -> str:
    return WS.sub(" ", str(text or "")).strip().casefold()


def _content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(str(text or "").casefold()) if w not in STOPWORDS}


def _rows(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "questions", "items", "examples"):
            if isinstance(payload.get(key), list):
                return [r for r in payload[key] if isinstance(r, dict)]
    return []


def _sessions(row: dict[str, Any]) -> Iterator[tuple[str, str, list[dict[str, Any]]]]:
    ids = row.get("haystack_session_ids") or []
    dates = row.get("haystack_dates") or []
    turns = row.get("haystack_sessions") or []
    if not (len(ids) == len(dates) == len(turns)):
        raise ValueError(f"{row.get('question_id')}: haystack arrays are not parallel")
    for sid, date, session in zip(ids, dates, turns):
        yield str(sid), _event_time(date), list(session)


def _split(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Seeded, type-stratified working/held-out assignment.

    Each set is ordered round-robin across question_type, so --limit N on a
    smoke run samples every type instead of the dataset's type-sorted prefix.
    """
    by_type: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row.get("question_id") or "")
        if qid:
            by_type.setdefault(str(row.get("question_type") or ""), []).append(qid)
    picked: dict[str, dict[str, list[str]]] = {}
    for kind, qids in by_type.items():
        ordered = sorted(qids)
        rng = random.Random(f"{SPLIT_SEED}:{kind}")
        rng.shuffle(ordered)
        cut = round(len(ordered) * WORKING_FRACTION)
        picked[kind] = {"working": ordered[:cut], "held-out": ordered[cut:]}
    out: dict[str, list[str]] = {name: [] for name in SETS}
    for name in SETS:
        kinds = sorted(picked)
        for i in range(max((len(picked[k][name]) for k in kinds), default=0)):
            for kind in kinds:
                bucket = picked[kind][name]
                if i < len(bucket):
                    out[name].append(bucket[i])
    return out


def _ingest(store: Any, row: dict[str, Any]) -> tuple[int, dict[str, str]]:
    """Write every turn through observe(). Returns (turns, event -> session)."""
    done = int(store.get_meta(SESSIONS_DONE) or 0)
    observed: dict[str, str] = {}
    turns = 0
    pending = 0
    for index, (sid, event_time, session) in enumerate(_sessions(row)):
        if index < done:
            continue
        for turn in session:
            result = store.observe(
                str(turn.get("content") or ""),
                role=str(turn.get("role") or "user"),
                tier="episodic",
                session_id=sid,
                event_time=event_time,
                origin="longmemeval",
                channel="score_lme_retrieval",
                # Embedding runs as one batched drain after ingest; one model
                # call per turn would dominate the wall clock.
                defer_embedding=True,
                commit=False,
            )
            observed[result.event_id] = result.session_id
            turns += 1
        pending += 1
        if pending >= COMMIT_EVERY:
            store.set_meta(SESSIONS_DONE, str(index + 1))
            pending = 0
    store.set_meta(SESSIONS_DONE, str(len(row.get("haystack_sessions") or [])))
    return turns, observed


def _drain_embeddings(store: Any) -> int:
    processed = 0
    while True:
        stats = store.process_embedding_jobs(limit=EMBED_BATCH)
        if not stats.get("queued"):
            return processed
        if stats.get("available") is False:
            raise SystemExit(
                "embedding backend unavailable; rerun with --fts-only or point "
                "HAUNT_MODEL_CACHE at a cache that already holds the model"
            )
        done = int(stats.get("processed") or 0)
        if not done:
            # Rows are failing rather than draining; embedding_coverage in the
            # report is what makes the resulting gap visible.
            return processed
        processed += done


def _coverage(store: Any) -> dict[str, int]:
    """Embedding coverage. Older trees' stats() lacks the C4 fields."""
    stats = store.stats()
    embedded = stats.get("memories_embedded")
    if embedded is None:
        embedded = store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        ).fetchone()[0]
    pending = stats.get("embedding_pending")
    if pending is None:
        pending = stats.get("embedding_jobs")
    return {
        "memories": int(stats.get("memories") or 0),
        "embedded": int(embedded or 0),
        "queued": int(pending or 0),
    }


def _session_of(store: Any, event_ids: list[str]) -> dict[str, str]:
    if not event_ids:
        return {}
    marks = ",".join("?" * len(event_ids))
    rows = store.conn.execute(
        f"SELECT id, session_id FROM events WHERE id IN ({marks})", event_ids
    ).fetchall()
    return {str(r["id"]): str(r["session_id"]) for r in rows}


def _check_mapping(
    store: Any, row: dict[str, Any], observed: dict[str, str]
) -> dict[str, int]:
    """Cross-check the scorer's join against what the write path reported.

    ``observed`` comes from ObserveResult; the query below is the same read
    the scorer uses to attribute a hit. A disagreement means gold matching
    is unsound, so it is counted rather than assumed away.
    """
    stored = {
        str(r["id"]): str(r["session_id"])
        for r in store.conn.execute("SELECT id, session_id FROM events").fetchall()
    }
    expected = {str(s) for s in row.get("haystack_session_ids") or []}
    gold = {str(s) for s in row.get("answer_session_ids") or []}
    return {
        "event_session_mismatches": sum(
            1 for eid, sid in observed.items() if stored.get(eid) != sid
        ),
        "sessions_missing": len(expected - set(stored.values())),
        "sessions_unexpected": len(set(stored.values()) - expected),
        "gold_sessions_absent": len(gold - set(stored.values())),
    }


def _gold_signals(row: dict[str, Any]) -> dict[str, Any]:
    """Raw per-question signals a human needs to bucket a miss by mechanism."""
    gold = {str(s) for s in row.get("answer_session_ids") or []}
    turns = [
        str(turn.get("content") or "")
        for sid, session in zip(
            row.get("haystack_session_ids") or [], row.get("haystack_sessions") or []
        )
        if str(sid) in gold
        for turn in session
    ]
    text = "\n".join(turns)
    question = _content_words(row.get("question"))
    answer = _norm(row.get("answer"))
    bearing = [len(t) for t in turns if answer and answer in _norm(t)]
    return {
        "n_gold_sessions": len(gold),
        "gold_turns": len(turns),
        "gold_lexical_overlap": round(
            len(question & _content_words(text)) / len(question), 4
        )
        if question
        else 0.0,
        "gold_max_turn_chars": max((len(t) for t in turns), default=0),
        "answer_in_gold_text": bool(bearing),
        "answer_turn_chars": max(bearing, default=0),
        # LongMemEval marks its abstention variants with an _abs suffix.
        "abstention": str(row.get("question_id") or "").endswith("_abs"),
    }


def _mechanism(
    signals: dict[str, Any], record: dict[str, Any], *, allow_ranking: bool = True
) -> str:
    """Bucket a miss by mechanism. Order is precedence, most specific first.

    ``allow_ranking`` off yields the secondary bucket: gold sitting inside the
    deep pool proves a ranking failure and would otherwise mask every
    content-derived mechanism behind it.
    """
    if signals["abstention"]:
        return "needs_abstention"
    if allow_ranking and record["deep_gold_rank"] is not None:
        return "ranking_fusion"
    if signals["gold_lexical_overlap"] < LEXICAL_GAP:
        return "lexical_gap"
    if record["question_type"] == "temporal-reasoning":
        return "temporal_reasoning"
    if signals["n_gold_sessions"] > 1 or record["question_type"] == "multi-session":
        return "multi_session_aggregation"
    if signals["answer_in_gold_text"] and signals["answer_turn_chars"] > CHUNK_CHARS:
        return "chunking_segmentation"
    return "unclassified"


def _score(
    row: dict[str, Any],
    hits: list[Any],
    sessions: dict[str, str],
    signals: dict[str, Any],
) -> dict[str, Any]:
    gold = {str(s) for s in row.get("answer_session_ids") or []}
    retrieved = [sessions.get(hit.event_id, "") for hit in hits]
    gold_ranks = [i for i, sid in enumerate(retrieved, start=1) if sid in gold]
    answer = _norm(row.get("answer"))
    substring_rank = next(
        (
            i
            for i, hit in enumerate(hits, start=1)
            if answer and answer in _norm(hit.content)
        ),
        None,
    )
    return {
        "question_id": str(row.get("question_id") or ""),
        "question_type": str(row.get("question_type") or ""),
        "gold_sessions": sorted(gold),
        "retrieved_sessions": retrieved,
        "gold_ranks": gold_ranks,
        "first_gold_rank": gold_ranks[0] if gold_ranks else None,
        "gold_retrieved": bool(gold_ranks),
        "answer_substring_rank": substring_rank,
        "signals": signals,
    }


def _metrics(records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    n = len(records)
    if not n:
        return {"n": 0}
    out: dict[str, Any] = {"n": n}
    for cut in CUTOFFS:
        if cut > k:
            continue
        rank_hits = sum(
            1
            for r in records
            if r.get("first_gold_rank") is not None and r["first_gold_rank"] <= cut
        )
        out[f"recall_at_{cut}"] = round(rank_hits / n, 4)
    out["mrr"] = round(
        sum(1 / r["first_gold_rank"] for r in records if r.get("first_gold_rank")) / n,
        4,
    )
    return out


def _substring_metrics(records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    n = len(records)
    if not n:
        return {"n": 0}
    out: dict[str, Any] = {"n": n}
    for cut in CUTOFFS:
        if cut > k:
            continue
        found = sum(
            1
            for r in records
            if r.get("answer_substring_rank") is not None
            and r["answer_substring_rank"] <= cut
        )
        out[f"present_at_{cut}"] = round(found / n, 4)
    return out


def _by_type(records: list[dict[str, Any]], k: int, fn: Any) -> dict[str, Any]:
    kinds = sorted({r.get("question_type", "") for r in records})
    return {
        kind: fn([r for r in records if r.get("question_type") == kind], k)
        for kind in kinds
    }


def _mechanism_counts(misses: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = {
        name: sum(1 for r in misses if r.get(key) == name) for name in MECHANISMS
    }
    return {name: n for name, n in counts.items() if n}


def _mechanisms(records: list[dict[str, Any]]) -> dict[str, Any]:
    misses = [r for r in records if r.get("miss_mechanism")]
    return {
        "misses": len(misses),
        "counts": _mechanism_counts(misses, "miss_mechanism"),
        "secondary_counts": _mechanism_counts(misses, "miss_mechanism_secondary"),
        "by_type": {
            kind: _mechanism_counts(
                [r for r in misses if r.get("question_type") == kind],
                "miss_mechanism",
            )
            for kind in sorted({r.get("question_type", "") for r in misses})
        },
    }


def _set_block(records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    return {
        "overall": _metrics(records, k),
        "by_type": _by_type(records, k, _metrics),
        "miss_mechanisms": _mechanisms(records),
        # Weaker proxy: kept separate because it conflates retrieval with
        # surface phrasing and misses answers the gold session paraphrases.
        "answer_substring_proxy": {
            "note": "weaker signal than gold-evidence recall; not a headline metric",
            "overall": _substring_metrics(records, k),
            "by_type": _by_type(records, k, _substring_metrics),
        },
    }


def _report(
    records: dict[str, list[dict[str, Any]]],
    *,
    provenance: dict[str, Any],
    split: dict[str, list[str]],
    timing: dict[str, float],
    checks: dict[str, int],
    k: int,
    held_out_detail: bool,
) -> dict[str, Any]:
    held_out = records["held-out"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": provenance,
        "split": {
            "seed": SPLIT_SEED,
            "working_fraction": WORKING_FRACTION,
            "stratified_by": "question_type",
            "sizes": {name: len(split[name]) for name in SETS},
            "note": "diagnose on working; held-out only checks that a fix generalized",
        },
        "timing": {key: round(value, 2) for key, value in timing.items()},
        "mapping_checks": checks,
        "sets": {name: _set_block(records[name], k) for name in SETS},
        "questions": records["working"],
        "held_out_questions": held_out
        if held_out_detail
        else [{key: r.get(key) for key in HELD_OUT_PUBLIC} for r in held_out],
        "held_out_detail_included": held_out_detail,
    }


def _write(out: Path, report: dict[str, Any]) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(out)


def _resume(out: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    loaded: dict[str, list[dict[str, Any]]] = {name: [] for name in SETS}
    if not out.is_file():
        return loaded, {}
    payload = json.loads(out.read_text(encoding="utf-8"))
    loaded["working"] = [
        r for r in payload.get("questions") or [] if isinstance(r, dict)
    ]
    loaded["held-out"] = [
        r for r in payload.get("held_out_questions") or [] if isinstance(r, dict)
    ]
    checks = {
        key: int(value) for key, value in (payload.get("mapping_checks") or {}).items()
    }
    return loaded, checks


def _print_set(name: str, block: dict[str, Any], k: int) -> None:
    overall = block["overall"]
    if not overall["n"]:
        return
    print(f"\n[{name}] n={overall['n']}  gold-evidence retrieval (primary)")
    cuts = "  ".join(
        f"@{c}={overall[f'recall_at_{c}']:.3f}"
        for c in CUTOFFS
        if f"recall_at_{c}" in overall
    )
    print(f"  overall  {cuts}  mrr={overall['mrr']:.3f}")
    for kind, stats in block["by_type"].items():
        line = "  ".join(
            f"@{c}={stats[f'recall_at_{c}']:.3f}"
            for c in CUTOFFS
            if f"recall_at_{c}" in stats
        )
        print(f"  {kind:<26} n={stats['n']:<4} {line}")
    proxy = block["answer_substring_proxy"]["overall"]
    line = "  ".join(
        f"@{c}={proxy[f'present_at_{c}']:.3f}"
        for c in CUTOFFS
        if f"present_at_{c}" in proxy
    )
    print(f"  answer-substring proxy (weaker signal)  {line}")
    mech = block["miss_mechanisms"]
    if mech["misses"]:
        print(f"  misses {mech['misses']} by mechanism {mech['counts']}")
        print(f"  {'':>6} beneath the ranking failure {mech['secondary_counts']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default=os.environ.get("HAUNT_LME_PATH", ""))
    p.add_argument("--limit", type=int, default=0, help="score the first N of the set")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default="lme_retrieval_report.json")
    p.add_argument("--fts-only", action="store_true", help="no vectors, no model")
    p.add_argument("--set", dest="which", choices=(*SETS, "both"), default="working")
    p.add_argument("--held-out-detail", action="store_true", help="unsuppress detail")
    p.add_argument("--resume", action="store_true", help="skip questions in --out")
    args = p.parse_args(argv)

    path = Path(args.path) if args.path else None
    if path is None or not path.is_file():
        print(f"skip: no LongMemEval dataset at --path {args.path or '(unset)'}")
        return 0
    if args.k < max(CUTOFFS):
        print(f"warning: k={args.k} omits cutoffs above {args.k}", file=sys.stderr)

    home = _isolated_home()
    if args.fts_only:
        os.environ["HAUNT_FTS_ONLY"] = "1"
        os.environ["HAUNT_EMBED_MODEL"] = "off"

    from haunt.embed import state as embed_state
    from haunt.recall import recall
    from haunt.store import Store

    out = Path(args.out).expanduser().resolve()
    records, checks = (
        _resume(out) if args.resume else ({name: [] for name in SETS}, {})
    )
    scored = {r.get("question_id") for group in records.values() for r in group}

    t_load = time.perf_counter()
    rows = _rows(json.loads(path.read_text(encoding="utf-8")))
    load_s = time.perf_counter() - t_load
    by_id = {str(r.get("question_id") or ""): r for r in rows}
    split = _split(rows)
    wanted = SETS if args.which == "both" else (args.which,)
    queue: list[tuple[str, str]] = []
    for name in wanted:
        chosen = split[name][: args.limit] if args.limit > 0 else split[name]
        queue.extend((name, qid) for qid in chosen)

    state = embed_state()
    provenance = {
        "tree": _tree(),
        "dataset": {
            "path": str(path),
            "sha256": _digest(path),
            "bytes": path.stat().st_size,
            "questions": len(rows),
        },
        "profile": "fts-only" if args.fts_only else "hybrid",
        "embed_model": state.model_id,
        "embed_dim": state.dim,
        "k": args.k,
        "probe_k": PROBE_K,
        "haunt_home": str(home),
        "sets_run": list(wanted),
        "limit": args.limit,
        "embedding_coverage": {"memories": 0, "embedded": 0, "queued": 0},
    }
    print(
        f"loaded {len(rows)} questions in {load_s:.1f}s; queued {len(queue)} "
        f"({'+'.join(wanted)}) k={args.k} "
        f"{'fts-only' if args.fts_only else 'hybrid ' + state.model_id} home={home}",
        file=sys.stderr,
    )

    # Namespaces are per profile as well as per question: an FTS-only store
    # has no vectors, and reusing it under a model would force a re-embed.
    prefix = "lme-fts" if args.fts_only else "lme-vec"
    timing = {"load_s": load_s, "ingest_s": 0.0, "embed_s": 0.0, "query_s": 0.0}
    coverage = {"memories": 0, "embedded": 0, "queued": 0}
    turns_total = 0
    # Projection base: only questions this run actually ingested. A namespace
    # left behind by an earlier run costs nothing and would flatter the rate.
    fresh = {"n": 0, "seconds": 0.0}
    started = time.perf_counter()

    for index, (name, qid) in enumerate(queue, start=1):
        row = by_id.get(qid)
        if row is None or qid in scored:
            continue
        signals = _gold_signals(row)
        with Store(f"{prefix}-{qid}") as store:
            t0 = time.perf_counter()
            turns, observed = _ingest(store, row)
            t1 = time.perf_counter()
            embedded = 0 if args.fts_only else _drain_embeddings(store)
            t2 = time.perf_counter()
            hits = recall(
                str(row.get("question") or ""),
                store=store,
                k=args.k,
                use_vectors=not args.fts_only,
            )
            sessions = _session_of(store, [h.event_id for h in hits])
            record = _score(row, hits, sessions, signals)
            record["deep_gold_rank"] = None
            if not record["gold_retrieved"]:
                # Only misses pay for the deep probe.
                deep = recall(
                    str(row.get("question") or ""),
                    store=store,
                    k=PROBE_K,
                    use_vectors=not args.fts_only,
                )
                deep_sessions = _session_of(store, [h.event_id for h in deep])
                gold = set(record["gold_sessions"])
                record["deep_gold_rank"] = next(
                    (
                        i
                        for i, h in enumerate(deep, start=1)
                        if deep_sessions.get(h.event_id, "") in gold
                    ),
                    None,
                )
            t3 = time.perf_counter()
            for key, value in _check_mapping(store, row, observed).items():
                checks[key] = checks.get(key, 0) + value
            for key, value in _coverage(store).items():
                coverage[key] += value
        missed = not record["gold_retrieved"]
        record["miss_mechanism"] = _mechanism(signals, record) if missed else None
        record["miss_mechanism_secondary"] = (
            _mechanism(signals, record, allow_ranking=False) if missed else None
        )
        record.update(
            {
                "split": name,
                "n_sessions": len(row.get("haystack_sessions") or []),
                "n_turns": sum(len(s) for s in row.get("haystack_sessions") or []),
                # Zero when a previous run already committed this namespace.
                "n_turns_written": turns,
                "n_embedded": embedded,
                "ingest_s": round(t1 - t0, 2),
                "embed_s": round(t2 - t1, 2),
                "query_s": round(t3 - t2, 3),
            }
        )
        records[name].append(record)
        turns_total += turns
        if turns:
            fresh["n"] += 1
            fresh["seconds"] += t3 - t0
        timing["ingest_s"] += t1 - t0
        timing["embed_s"] += t2 - t1
        timing["query_s"] += t3 - t2
        timing["total_s"] = time.perf_counter() - started
        provenance["embedding_coverage"] = dict(coverage)
        _write(
            out,
            _report(
                records,
                provenance=provenance,
                split=split,
                timing=timing,
                checks=checks,
                k=args.k,
                held_out_detail=args.held_out_detail,
            ),
        )
        print(
            f"[{index}/{len(queue)}] {name} {qid} {record['question_type']} "
            f"turns={record['n_turns']} written={turns} "
            f"ingest={record['ingest_s']}s embed={record['embed_s']}s "
            f"query={record['query_s']}s gold_rank={record['first_gold_rank']} "
            f"miss={record['miss_mechanism']}",
            file=sys.stderr,
        )
        # The haystacks are ~99% of a 265MB parse; drop each one once it is
        # stored so a 500-question run does not hold the whole dataset.
        row["haystack_sessions"] = None

    timing["total_s"] = time.perf_counter() - started
    ratio = coverage["embedded"] / coverage["memories"] if coverage["memories"] else 0.0
    degraded = not args.fts_only and (ratio < MIN_COVERAGE or coverage["queued"] > 0)
    provenance["embedding_coverage"] = {
        **coverage,
        "ratio": round(ratio, 4),
        "degraded": degraded,
    }
    provenance["seconds_per_fresh_question"] = (
        round(fresh["seconds"] / fresh["n"] + OVERHEAD_S, 2) if fresh["n"] else None
    )
    report = _report(
        records,
        provenance=provenance,
        split=split,
        timing=timing,
        checks=checks,
        k=args.k,
        held_out_detail=args.held_out_detail,
    )
    _write(out, report)

    print(f"\ntree {provenance['tree']['sha'][:12]} "
          f"{'(dirty)' if provenance['tree']['dirty'] else ''} "
          f"profile {provenance['profile']} k {args.k} turns written {turns_total}")
    for name in SETS:
        _print_set(name, report["sets"][name], args.k)
    print(f"\nmapping checks {checks}")
    print(
        f"embedding coverage {coverage['embedded']}/{coverage['memories']} "
        f"({ratio:.4f}) queued {coverage['queued']}"
    )
    if degraded:
        message = (
            f"WARNING: hybrid run embedded only {ratio:.2%} of memories with "
            f"{coverage['queued']} still queued -- these numbers are closer to "
            "FTS-only than to hybrid"
        )
        print(message)
        print(message, file=sys.stderr)
    print(
        f"wall clock {timing['total_s']:.1f}s (ingest {timing['ingest_s']:.1f}s, "
        f"embed {timing['embed_s']:.1f}s, query {timing['query_s']:.1f}s)"
    )
    if fresh["n"]:
        per_q = fresh["seconds"] / fresh["n"] + OVERHEAD_S
        print(
            f"projected wall clock {per_q * 500 / 60:.1f} min for n=500 "
            f"at {per_q:.1f}s/question over {fresh['n']} freshly ingested"
        )
    print(f"report {out}")
    if any(checks.values()):
        print("error: hit-to-session mapping failed its checks", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
