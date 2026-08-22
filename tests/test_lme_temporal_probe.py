"""Optional LongMemEval probe. Skipped unless the dump is present."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CANDIDATES = [
    Path(os.environ["HAUNT_LME_PATH"]) if os.environ.get("HAUNT_LME_PATH") else None,
    Path("/workspace/longmemeval_s_cleaned.json"),
    Path("/data/longmemeval_s_cleaned.json"),
]


def _lme_path() -> Path | None:
    for p in CANDIDATES:
        if p is not None and p.is_file():
            return p
    return None


@pytest.mark.skipif(_lme_path() is None, reason="longmemeval_s_cleaned.json not present")
def test_lme_temporal_probe_script():
    import runpy

    path = _lme_path()
    assert path is not None
    ns = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "score_lme_temporal.py")
    )
    assert ns["main"](["--path", str(path)]) == 0
