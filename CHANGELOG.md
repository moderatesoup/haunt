# Changelog

## 0.2.0 — 2026-08-24

Naming cleanup. Product name is **haunt** only.

- Removed `lore` / `engram` console scripts and `~/.haunt/bin` wrappers. Entry points are `haunt`, `haunt-mcp`, `haunt-hook`, `haunt-hook-claude`.
- Host doctor accepts `haunt-mcp` / `haunt-hook` only. `lore-mcp` in `mcp.json` is not a haunt command.
- User-facing home is `~/.haunt` / `HAUNT_HOME`. Dropped `LORE_*` / `ENGRAM_*` env aliases and the `~/.lore` fallback.
- Install from git: `pip install git+https://github.com/moderatesoup/haunt.git`. `pip install haunt` on PyPI is not this project (mikepqr's stow).
- Deleted stale `WIRING_REVIEW.md` (PR #2, six-tool era).
