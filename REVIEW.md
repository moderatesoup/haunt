# Adversarial Naming Review — haunt

**Reviewer:** automated scan  
**Date:** 2026-08-21  
**Branches scanned:** `cursor/rebrand-haunt-01d0` (PR #2), `main`, `cursor/memory-tools-mvp-6af2` (PR #1)  
**Search terms:** `engram`, `lore`, `ironrecall`, `iron-recall`, `memory-protocol`, `IRONRECALL`, `LORE_`, `ENGRAM_`, `~/.lore`, `~/.engram`

---

## Verdict

**`cursor/rebrand-haunt-01d0` is clean.** All remaining `engram`/`lore` references are in allowed categories (deprecated CLI aliases, backward-compat env-var fallbacks, historical rename notes). No user-facing output presents "engram" or "lore" as the current product name.

**`cursor/memory-tools-mvp-6af2` (PR #1) predates the rebrand and has extensive old-name leftovers.** These will resolve naturally when PR #1 is rebased onto the rebrand branch after PR #2 merges.

**`main` is the pre-rebrand state** (package `src/lore/`, name "engram"). PR #2 is the fix.

No `ironrecall`, `iron-recall`, `memory-protocol`, `IRONRECALL`, or `~/.engram` references found anywhere.

---

## Rebrand branch (`cursor/rebrand-haunt-01d0`) — Detailed findings

### PASS: User-facing product identity

| Surface | Value | Status |
|---------|-------|--------|
| README title | `# haunt` | ✓ |
| pyproject.toml name | `haunt` | ✓ |
| pyproject.toml description | `haunt — local-first verbatim memory…` | ✓ |
| CLI `--help` | `haunt — local-first verbatim memory for AI agents` | ✓ |
| MCP server name | `"haunt"` | ✓ |
| MCP instructions | `"haunt is local-first verbatim agent memory."` | ✓ |
| Hook stdout recall blocks | `[haunt ns=…]` | ✓ |
| Hook stdout timeline blocks | `[haunt recent ns=…]` | ✓ |
| Session start message | `"You have persistent local memory via haunt…"` | ✓ |
| `.mdc` rule title | `# haunt` | ✓ |
| `contrib/cursor/hooks.json` | `~/.haunt/bin/haunt-hook` | ✓ |
| Bootstrap report CLI output | `haunt home …` | ✓ |
| Primary launcher | `~/.haunt/bin/haunt-hook` | ✓ |
| Primary MCP launcher | `~/.haunt/bin/haunt-mcp` | ✓ |

### ALLOWED: Deprecated CLI aliases (intentional backward-compat)

| File | Line | Content |
|------|------|---------|
| `pyproject.toml` | 27–32 | `lore`, `lore-mcp`, `lore-hook`, `engram`, `engram-mcp`, `engram-hook` console scripts |
| `src/haunt/bootstrap.py` | 41–42 | Writes `lore-hook`, `engram-hook` launchers |
| `src/haunt/bootstrap.py` | 49–50 | Writes `lore-mcp`, `engram-mcp` launchers |
| `src/haunt/cursor_hook.py` | 321 | `name in {"haunt-hook", "engram-hook", "lore-hook"}` (recognizes all variants) |

### ALLOWED: Legacy env-var fallbacks (documented in README as "accepted as legacy aliases")

| File | Line | Env var |
|------|------|---------|
| `src/haunt/paths.py` | 16 | `LORE_HOME` |
| `src/haunt/paths.py` | 17 | `ENGRAM_HOME` |
| `src/haunt/paths.py` | 22–24 | `~/.lore` directory fallback |
| `src/haunt/paths.py` | 33 | `LORE_MODEL_CACHE` |
| `src/haunt/paths.py` | 66 | `LORE_NAMESPACE` |
| `src/haunt/paths.py` | 67 | `ENGRAM_NAMESPACE` |
| `src/haunt/embed.py` | 65 | `LORE_EMBED_MODEL` |
| `src/haunt/embed.py` | 74 | `LORE_FTS_ONLY` |
| `src/haunt/embed.py` | 86 | `LORE_EMBED_MAX_LEN` |
| `src/haunt/cursor_hook.py` | 30 | `ENGRAM_STORE_THOUGHTS`, `LORE_STORE_THOUGHTS` |
| `src/haunt/cursor_hook.py` | 117–118 | `LORE_NAMESPACE`, `ENGRAM_NAMESPACE` |
| `src/haunt/bootstrap.py` | 148 | `LORE_JSON` |

### ALLOWED: Historical rename note / documentation of deprecated aliases

| File | Line | Content |
|------|------|---------|
| `README.md` | 5 | "Renamed from *engram* (2025) to avoid collisions…" |
| `README.md` | 21, 82 | Documents that `lore` and `engram` still work as deprecated aliases |
| `README.md` | 103, 118–121 | Lists `engram-mcp`, `lore-mcp` etc. as "(legacy alias)" |
| `README.md` | 125 | Mentions `LORE_HOME / ENGRAM_HOME accepted as legacy aliases` |
| `contrib/cursor/haunt.mdc` | 18 | "`LORE_HOME` / `ENGRAM_HOME` as legacy aliases" |
| `src/haunt/cli.py` | 240 | Prints `(HAUNT_HOME / LORE_HOME / ENGRAM_HOME)` — showing accepted vars |

### ALLOWED: Internal backward-compat aliases (not user-facing)

| File | Line | Content |
|------|------|---------|
| `src/haunt/__init__.py` | 9, 14 | `lore_home = haunt_home` alias + export |
| `src/haunt/paths.py` | 28–29 | `lore_home = haunt_home` internal alias |
| `src/haunt/cursor_hook.py` | 369 | `"lore_home"` key in internal dict (consumed by CLI output as fallback) |
| `src/haunt/bootstrap.py` | 100 | `"lore_home"` key in bootstrap report dict |
| `src/haunt/bootstrap.py` | 126 | `report.get('lore_home', '')` fallback in format_report |

### NOTE: Test infrastructure (not user-facing, not a violation)

| File | Line | Content |
|------|------|---------|
| `tests/conftest.py` | 8 | `Path("/workspace/lore/.model-cache")` — workspace CI cache path |
| `tests/conftest.py` | 12 | `def lore_env(…)` — fixture name |
| `tests/test_lore.py` | filename | Test file still named `test_lore.py` |
| `tests/test_lore.py` | 106–107 | References `/workspace/lore/.model-cache` |
| `tests/test_cursor_hook.py` | 25–26 | `monkeypatch.delenv("ENGRAM_*")` — testing legacy env vars work |

These are internal test names/paths. Not user-facing. No action needed.

---

## Memory-tools branch (`cursor/memory-tools-mvp-6af2`, PR #1) — Pre-rebrand leftovers

This branch has not been rebased onto the rebrand. The following are user-facing leftovers that will need fixing when PR #1 is rebased:

| File | Line | Issue |
|------|------|-------|
| `README.md` | 1 | Title: `# engram` |
| `README.md` | 5 | "Public name is **engram**" |
| `README.md` | 14–18 | All quickstart examples use `engram …` |
| `README.md` | 93–110 | MCP config uses `"lore"` server name, `~/.lore/bin/lore-mcp` |
| `pyproject.toml` | 2 | `name = "lore"` |
| `pyproject.toml` | 4 | `description = "engram — local-first…"` |
| `src/lore/mcp_server.py` | 16 | `name="lore"` |
| `src/lore/mcp_server.py` | 19 | `"engram (lore) is local-first…"` |
| `contrib/cursor/hooks.json` | all | Uses `~/.lore/bin/engram-hook` as launcher |
| `src/lore/cursor_hook.py` | 345 | Function `_is_engram_command` (missing `"haunt-hook"`) |
| `src/lore/cursor_hook.py` | 407 | Docstring: "Write ~/.lore/bin/engram-hook…" |

**Resolution:** Merge PR #2 (rebrand) first, then rebase PR #1 onto the new main. The rebase will require conflict resolution in `cursor_hook.py`, `mcp_server.py`, `bootstrap.py`, and `README.md`.

---

## Conclusion

No action required on the rebrand branch. The naming is consistent and correct. All old-name references serve a documented backward-compatibility purpose or are internal test infrastructure.
