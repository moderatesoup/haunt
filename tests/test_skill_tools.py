"""Skill/rule text must stay in lockstep with live MCP tools. Not a model judge."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MCP_PATH = ROOT / "src" / "haunt" / "mcp_server.py"
PLANNER_PATH = ROOT / "src" / "haunt" / "planner.py"
SKILL_PATH = ROOT / "contrib" / "skills" / "haunt" / "SKILL.md"
MDC_PATH = ROOT / "contrib" / "cursor" / "haunt.mdc"

# Stable phrase the skill must use when compile() is on the recall path.
AUTO_COMPILE_PHRASE = "compile() runs automatically on memory_recall"

# Parameter / wildcard tokens that look like memory_* but are not MCP tools.
_NON_TOOL_MEMORY_TOKENS = frozenset({"memory_id"})

_TOOL_RE = re.compile(r"\bmemory_[a-z][a-z0-9_]*\b")


def live_memory_tools(source: str) -> set[str]:
    """Parse mcp_server.py for @server.tool functions named memory_*."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("memory_"):
            continue
        if any(_is_server_tool(dec) for dec in node.decorator_list):
            names.add(node.name)
    return names


def _is_server_tool(dec: ast.AST) -> bool:
    call = dec if isinstance(dec, ast.Call) else None
    attr = call.func if call is not None else dec
    return (
        isinstance(attr, ast.Attribute)
        and attr.attr == "tool"
        and isinstance(attr.value, ast.Name)
        and attr.value.id == "server"
    )


def mentioned_tools(text: str) -> set[str]:
    return set(_TOOL_RE.findall(text)) - _NON_TOOL_MEMORY_TOKENS


def assert_skill_covers_live_tools(skill_text: str, live: set[str]) -> None:
    mentioned = mentioned_tools(skill_text)
    missing = live - mentioned
    extra = mentioned - live
    assert not missing, f"SKILL.md missing live tools: {sorted(missing)}"
    assert not extra, f"SKILL.md names tools that are not live: {sorted(extra)}"


def _fn(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _calls_name(fn: ast.FunctionDef, name: str) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == name:
                return True
    return False


def recall_auto_compiles(mcp_src: str, planner_src: str) -> bool:
    """True when memory_recall → planned_recall → compile()."""
    mcp = ast.parse(mcp_src)
    planner = ast.parse(planner_src)
    return _calls_name(_fn(mcp, "memory_recall"), "planned_recall") and _calls_name(
        _fn(planner, "planned_recall"), "compile"
    )


def test_parser_finds_live_memory_tools():
    live = live_memory_tools(MCP_PATH.read_text(encoding="utf-8"))
    assert live, "parser found no memory_* tools in mcp_server.py"
    assert "memory_recall" in live


def test_skill_lists_every_live_mcp_tool():
    live = live_memory_tools(MCP_PATH.read_text(encoding="utf-8"))
    skill = SKILL_PATH.read_text(encoding="utf-8")
    mdc = MDC_PATH.read_text(encoding="utf-8")
    assert_skill_covers_live_tools(skill, live)
    extra_mdc = mentioned_tools(mdc) - live
    assert not extra_mdc, f"haunt.mdc names tools that are not live: {sorted(extra_mdc)}"


def test_skill_documents_auto_compile_or_manual_window():
    mcp_src = MCP_PATH.read_text(encoding="utf-8")
    planner_src = PLANNER_PATH.read_text(encoding="utf-8")
    skill = SKILL_PATH.read_text(encoding="utf-8")
    if recall_auto_compiles(mcp_src, planner_src):
        assert AUTO_COMPILE_PHRASE in skill, (
            f"SKILL.md must contain {AUTO_COMPILE_PHRASE!r} because "
            "compile() is auto-invoked on memory_recall"
        )
        assert "does not auto-compile" not in skill.lower()
    else:
        assert AUTO_COMPILE_PHRASE not in skill
        assert "since" in skill and "until" in skill


def test_mutation_missing_live_tool_fails():
    live = live_memory_tools(MCP_PATH.read_text(encoding="utf-8"))
    victim = sorted(live)[0]
    mutated = SKILL_PATH.read_text(encoding="utf-8").replace(victim, "MEMORY_TOOL_REMOVED")
    assert victim not in mutated
    with pytest.raises(AssertionError, match="missing live tools"):
        assert_skill_covers_live_tools(mutated, live)


def test_embedded_cursor_rule_matches_contrib_mdc():
    from haunt.hosts.cursor import _HAUNT_MDC

    assert _HAUNT_MDC == MDC_PATH.read_text(encoding="utf-8")


def test_claude_rule_has_no_dead_tools():
    from haunt.hosts.claude import _HAUNT_CLAUDE_RULE

    live = live_memory_tools(MCP_PATH.read_text(encoding="utf-8"))
    extra = mentioned_tools(_HAUNT_CLAUDE_RULE) - live
    assert not extra, f"Claude haunt.md names tools that are not live: {sorted(extra)}"
