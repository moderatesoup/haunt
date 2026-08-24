"""Lock product name: haunt only. No lore/engram aliases, no fake PyPI install."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
BANNED_SCRIPTS = (
    "lore",
    "lore-mcp",
    "lore-hook",
    "engram",
    "engram-mcp",
    "engram-hook",
)
SCRIPT_ASSIGN = re.compile(
    r"^(lore|lore-mcp|lore-hook|engram|engram-mcp|engram-hook)\s*="
)


def _project_scripts(text: str) -> dict[str, str]:
    in_section = False
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == "[project.scripts]"
            continue
        if in_section and "=" in stripped and not stripped.startswith("#"):
            key, val = stripped.split("=", 1)
            out[key.strip()] = val.strip()
    return out


def test_project_scripts_have_no_lore_or_engram():
    text = PYPROJECT.read_text(encoding="utf-8")
    leftover_lines = [
        f"{i}:{line.strip()}"
        for i, line in enumerate(text.splitlines(), 1)
        if SCRIPT_ASSIGN.match(line.strip())
    ]
    assert leftover_lines == [], leftover_lines
    scripts = _project_scripts(text)
    leftover = [name for name in BANNED_SCRIPTS if name in scripts]
    assert leftover == [], leftover
    for required in ("haunt", "haunt-mcp", "haunt-hook", "haunt-hook-claude"):
        assert required in scripts


def test_mutation_lore_script_assignment_fails():
    text = PYPROJECT.read_text(encoding="utf-8")
    mutated = text.replace("[project.scripts]\n", "[project.scripts]\nlore = \"haunt.cli:main\"\n")
    leftover = [
        line.strip()
        for line in mutated.splitlines()
        if SCRIPT_ASSIGN.match(line.strip())
    ]
    assert leftover, "mutation must insert a lore = assignment"
    with pytest.raises(AssertionError):
        scripts = _project_scripts(mutated)
        leftover_keys = [name for name in BANNED_SCRIPTS if name in scripts]
        assert leftover_keys == []


def test_readme_has_no_bare_pip_install_haunt():
    text = README.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), 1):
        assert line.strip() != "pip install haunt", f"README line {i} is a bare pip install haunt"
    assert "pip install git+https://github.com/moderatesoup/haunt.git" in text


def test_wiring_review_must_not_exist():
    assert not (ROOT / "WIRING_REVIEW.md").exists()


def test_user_facing_text_has_no_legacy_product_aliases():
    paths = [
        ROOT / "README.md",
        ROOT / "contrib" / "skills" / "haunt" / "SKILL.md",
        ROOT / "contrib" / "cursor" / "haunt.mdc",
        ROOT / "src" / "haunt" / "cli.py",
        ROOT / "src" / "haunt" / "hosts" / "cursor.py",
        ROOT / "src" / "haunt" / "hosts" / "claude.py",
    ]
    needles = (
        "LORE_HOME",
        "ENGRAM_HOME",
        "lore and engram are aliases",
        "CLI aliases `lore`",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        for needle in needles:
            assert needle.lower() not in lower, f"{path} still mentions {needle!r}"
