#!/usr/bin/env python3
"""Measure the evaluation-only coverage-query overhead at fixed corpus scales.

This isolates the new diagnostic query from the existing E5 FTS candidate
query. Timings are observational machine evidence; the statement-count bound
is the deterministic regression gate.
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from haunt.abstention_eval import _coverage_many

QUERY = "alpha beta gamma"
MATCH = '"alpha" OR "beta" OR "gamma"'
SCALES = (1_000, 10_000, 100_000)


@dataclass
class _StoreView:
    conn: sqlite3.Connection


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _fts_top_five(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row["id"])
        for row in conn.execute(
            "SELECT id FROM memories_fts WHERE memories_fts MATCH ? "
            "ORDER BY rank, id LIMIT 5",
            (MATCH,),
        ).fetchall()
    ]


def benchmark(*, repeats: int = 31) -> dict:
    if repeats < 5:
        raise ValueError("repeats must be at least 5")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE VIRTUAL TABLE memories_fts USING fts5("
        "id UNINDEXED, content, tokenize='porter unicode61')"
    )
    targets = [
        (f"target-{index}", f"alpha beta gamma target {index}") for index in range(5)
    ]
    conn.executemany("INSERT INTO memories_fts(id,content) VALUES (?,?)", targets)
    inserted = 5
    measurements: list[dict] = []
    view = _StoreView(conn)
    for scale in SCALES:
        new_rows = [
            (f"filler-{index}", f"unrelated filler corpus row {index}")
            for index in range(inserted, scale)
        ]
        conn.executemany("INSERT INTO memories_fts(id,content) VALUES (?,?)", new_rows)
        conn.commit()
        inserted = scale
        expected = _fts_top_five(conn)
        _coverage_many(view, QUERY, expected)  # warm both paths
        baseline_ms: list[float] = []
        combined_ms: list[float] = []
        statement_counts: set[int] = set()
        for _ in range(repeats):
            start = time.perf_counter_ns()
            ids = _fts_top_five(conn)
            baseline_ms.append((time.perf_counter_ns() - start) / 1_000_000)

            start = time.perf_counter_ns()
            ids = _fts_top_five(conn)
            _, diagnostics = _coverage_many(view, QUERY, ids)
            combined_ms.append((time.perf_counter_ns() - start) / 1_000_000)
            statement_counts.add(int(diagnostics["sql_statement_count"]))
        baseline_p95 = _percentile(baseline_ms, 0.95)
        combined_p95 = _percentile(combined_ms, 0.95)
        measurements.append(
            {
                "corpus_rows": scale,
                "repeats": repeats,
                "top_five": expected,
                "coverage_sql_statement_counts": sorted(statement_counts),
                "e5_fts_candidate_query_ms": {
                    "median": statistics.median(baseline_ms),
                    "p95": baseline_p95,
                },
                "fts_plus_e6_diagnostic_ms": {
                    "median": statistics.median(combined_ms),
                    "p95": combined_p95,
                },
                "diagnostic_p95_overhead_ms": combined_p95 - baseline_p95,
                "non_gating_observation_below_10ms_p95_overhead": (
                    combined_p95 - baseline_p95 < 10.0
                ),
            }
        )
    conn.close()
    return {
        "schema_version": 1,
        "report_id": "haunt-e6-evidence-latency-v1",
        "method": (
            "in-memory sqlite FTS5; existing top-five candidate query compared "
            "with that query plus one batched diagnostic coverage statement"
        ),
        "deterministic_gate": {
            "coverage_sql_statements_per_top_five": 1,
            "corpus_sizes": list(SCALES),
        },
        "timing_is_observational_not_a_cross_machine_gate": True,
        "environment": {
            "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "measurements": measurements,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = (
        json.dumps(benchmark(repeats=args.repeats), sort_keys=True, indent=2) + "\n"
    )
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
