#!/usr/bin/env python3
"""Reproduce E6 FTS-only or pinned-hybrid feasibility evidence.

A scientifically valid ``status=blocked`` report is a successful reproduction,
so this command exits zero unless setup or evaluation itself fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from haunt.abstention_eval import DEFAULT_FIXTURE_DIR, evaluate_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("fts", "hybrid"), required=True)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument(
        "--model-cache",
        type=Path,
        help="Required for hybrid; must contain verified local BAAI-bge-m3 ONNX files.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_profile(
        args.profile,
        fixture_dir=args.fixture_dir,
        model_cache=args.model_cache,
    )
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
