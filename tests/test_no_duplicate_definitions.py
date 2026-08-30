"""No module, class, or function may bind the same name twice.

A second `_table_columns` returning a set once shadowed the first for every
caller in store.py -- including the two defined above it, because Python
resolves module globals at call time -- and escaped the entire suite. Six of
its eight call sites were type-agnostic, the two order-sensitive ones stayed
self-consistent within a process, and `_canonical_json` sorted the resulting
dict keys before hashing, so the plan digest never moved. The damage was a
latent determinism defect: the emitted INSERT column order changed run to run.

Nothing about that was specific to that name, so this pins the shape instead of
the instance. `ruff --select F811` does not catch it: ruff's default
dummy-variable-rgx exempts every underscore-prefixed name, which is every
private helper in this codebase, and no rule selection models duplicate
module-level constant assignment at all.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = ("src", "tests", "scripts", "contrib")

# (relative path, scope, name) triples deliberately exempted. Empty, and the
# scan is clean -- an entry here needs a comment saying why the shadowing is
# intentional.
ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset()

# Decorators that legitimately rebind a name in the same scope.
_OVERLOAD = {"overload", "typing.overload"}
_PROPERTY_SUFFIXES = ("setter", "deleter", "getter", "register")


@dataclass(frozen=True)
class Duplicate:
    path: str
    scope: str
    name: str
    first: int
    second: int
    kind: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.path, self.scope, self.name)

    def render(self) -> str:
        return (
            f"{self.path}: {self.kind} {self.name!r} in {self.scope} "
            f"defined at line {self.first} and again at line {self.second}"
        )


def _decorator_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for decorator in getattr(node, "decorator_list", []) or []:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
            value = target.value
            if isinstance(value, ast.Name):
                names.add(f"{value.id}.{target.attr}")
    return names


def _is_exempt(node: ast.AST) -> bool:
    decorators = _decorator_names(node)
    if decorators & _OVERLOAD:
        return True
    return any(name in _PROPERTY_SUFFIXES for name in decorators)


def _definition_like(node: ast.AST) -> bool:
    """An assignment whose value is a literal table, not a computed rebinding.

    `_TABLE_FIELDS = {...}` twice is the same shadowing hazard as two defs;
    `count = count + 1` is not.
    """
    return isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.Constant))


def _scan_body(
    body: list[ast.stmt], path: str, scope: str, branching: bool
) -> list[Duplicate]:
    """One scope. `branching` suppresses alternatives that cannot both bind."""
    found: list[Duplicate] = []
    seen: dict[str, tuple[int, str]] = {}

    def record(name: str, line: int, kind: str) -> None:
        if branching:
            return
        previous = seen.get(name)
        if previous is not None:
            found.append(Duplicate(path, scope, name, previous[0], line, kind))
        seen[name] = (line, kind)

    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_exempt(node):
                record(node.name, node.lineno, "function")
            found += _scan_body(node.body, path, f"{scope}.{node.name}", branching)
        elif isinstance(node, ast.ClassDef):
            record(node.name, node.lineno, "class")
            found += _scan_body(node.body, path, f"{scope}.{node.name}", branching)
        elif isinstance(node, ast.Assign) and _definition_like(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    record(target.id, node.lineno, "binding")
        elif (
            # `NAME: type = {...}` is an AnnAssign, not an Assign. Most module
            # tables in this codebase are written that way -- _RECONCILE_TABLES
            # among them -- so omitting this branch left the exact constant
            # shadowing the gate exists to catch entirely uncovered.
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
            and _definition_like(node.value)
        ):
            record(node.target.id, node.lineno, "binding")
        elif isinstance(node, (ast.If, ast.Try)):
            # A TYPE_CHECKING stub, a try/except ImportError, or a platform
            # branch defines alternatives, only one of which ever binds.
            for sub in ast.iter_child_nodes(node):
                if isinstance(sub, ast.stmt):
                    found += _scan_body([sub], path, scope, True)
            for attribute in ("body", "orelse", "finalbody", "handlers"):
                for sub in getattr(node, attribute, []) or []:
                    subbody = getattr(sub, "body", [sub])
                    found += _scan_body(subbody, path, scope, True)
    return found


def scan(repo: Path = REPO) -> list[Duplicate]:
    found: list[Duplicate] = []
    for root in ROOTS:
        directory = repo / root
        if not directory.is_dir():
            continue
        for file in sorted(directory.rglob("*.py")):
            relative = str(file.relative_to(repo))
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=relative)
            found += _scan_body(tree.body, relative, "<module>", False)
    return found


def test_no_shadowed_definitions() -> None:
    hits = [hit for hit in scan() if hit.key not in ALLOWLIST]
    assert not hits, "duplicate definitions shadow earlier ones:\n" + "\n".join(
        hit.render() for hit in hits
    )


if __name__ == "__main__":  # standalone, no pytest required
    problems = [hit for hit in scan() if hit.key not in ALLOWLIST]
    for problem in problems:
        print(problem.render())
    sys.exit(1 if problems else 0)
