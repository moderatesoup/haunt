"""Host skill file written by `haunt install`. Copied from contrib when present."""

from __future__ import annotations

from pathlib import Path

SKILL_MARKERS = ("memory_recall", "verbatim")

# Used when contrib/ is not on disk (wheel install, isolated tests).
_SKILL_FALLBACK = """\
# haunt — local-first agent memory

Local-first verbatim memory. MCP server name is `haunt`.

**Hooks store; agents must recall unless context was injected.**

If no `[haunt ns=…]` block is visible, call `memory_recall` with the
user's exact wording before acting. Never summarize. Never distill.
Store verbatim or don't store.
"""


def skill_text() -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "contrib" / "skills" / "haunt" / "SKILL.md"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return _SKILL_FALLBACK


def install_host_skill(host_dir: Path) -> Path:
    dest = host_dir / "skills" / "haunt" / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(skill_text(), encoding="utf-8")
    return dest


def skill_issue(path: Path) -> str | None:
    if not path.is_file():
        return "haunt skill not found"
    text = path.read_text(encoding="utf-8")
    missing = [m for m in SKILL_MARKERS if m not in text]
    if missing:
        return f"haunt skill missing expected text: {', '.join(missing)}"
    return None
