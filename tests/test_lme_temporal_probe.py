"""Optional LongMemEval probe. Skipped unless the dump is present."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DATASET = "longmemeval_s_cleaned.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
# The two absolute paths are the container layout this probe was written
# against; a checkout on any other machine only ever matches the repo-local
# drop points or an explicit HAUNT_LME_PATH.
CANDIDATES = [
    *([Path(os.environ["HAUNT_LME_PATH"])] if os.environ.get("HAUNT_LME_PATH") else []),
    REPO_ROOT / DATASET,
    REPO_ROOT / "data" / DATASET,
    Path("/workspace") / DATASET,
    Path("/data") / DATASET,
]


def _lme_path() -> Path | None:
    for p in CANDIDATES:
        if p.is_file():
            return p
    return None


def _skip_reason() -> str:
    """Name every path the probe looked at, so ``pytest -rs`` shows the gap.

    A reason that only says the dump is absent reads as one missing file on a
    machine where none of the checked locations could ever exist.
    """
    checked = ", ".join(str(p) for p in CANDIDATES)
    return (
        f"no LongMemEval {DATASET}: set HAUNT_LME_PATH, or place the dump at one "
        f"of the paths checked: {checked}"
    )


@pytest.mark.skipif(_lme_path() is None, reason=_skip_reason())
def test_lme_temporal_probe_script():
    import runpy

    path = _lme_path()
    assert path is not None
    ns = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "score_lme_temporal.py")
    )
    assert ns["main"](["--path", str(path)]) == 0


def test_the_skip_names_every_path_it_checked():
    """The probe's own skip is the only report that it ran nothing."""
    mark = next(
        m for m in test_lme_temporal_probe_script.pytestmark if m.name == "skipif"
    )
    reason = mark.kwargs["reason"]
    assert "HAUNT_LME_PATH" in reason
    for candidate in CANDIDATES:
        assert str(candidate) in reason
