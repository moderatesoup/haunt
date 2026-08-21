#!/usr/bin/env bash
set -euo pipefail

# Idempotent bootstrap for haunt.
# Creates a venv, installs haunt, runs first-run setup.
# No Docker, no cloud, no root required.
#
# macOS: if you hit "enable_load_extension" errors, your Python was
# likely compiled by pyenv without --enable-loadable-sqlite-extensions.
# Use Homebrew Python instead:
#   /opt/homebrew/bin/python3 -m venv .venv
#   source .venv/bin/activate && pip install -e . && haunt bootstrap

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

echo "==> haunt bootstrap"
echo "    repo: ${REPO_ROOT}"

# ── venv ────────────────────────────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "==> Creating virtualenv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
else
    echo "==> Virtualenv already exists at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ── install ─────────────────────────────────────────────────────────────
echo "==> Installing haunt (editable)"
pip install -e "${REPO_ROOT}" --quiet

# ── first-run setup ─────────────────────────────────────────────────────
echo "==> Running haunt bootstrap"
if ! haunt bootstrap; then
    echo ""
    echo "FATAL: haunt bootstrap failed."
    echo "  sqlite-vec must load successfully. See output above."
    echo ""
    echo "  macOS fix: use Homebrew Python, not pyenv:"
    echo "    /opt/homebrew/bin/python3 -m venv ${VENV_DIR}"
    echo "    source ${VENV_DIR}/bin/activate"
    echo "    pip install -e ${REPO_ROOT} && haunt bootstrap"
    exit 1
fi

# ── HAUNT_HOME and db_path ──────────────────────────────────────────────
HAUNT_HOME="${HAUNT_HOME:-${HOME}/.haunt}"
echo ""
echo "    HAUNT_HOME  ${HAUNT_HOME}"
echo "    db_path     ${HAUNT_HOME}/namespaces/default.db"
echo "    registry    ${HAUNT_HOME}/registry.db"

# ── desktop icon ────────────────────────────────────────────────────────
echo "==> Installing desktop shortcut"
haunt dash --install-icon || true

echo ""
echo "── next steps ──────────────────────────────────────────────────────"
echo ""
echo "  1. Activate the venv:"
echo "       source ${VENV_DIR}/bin/activate"
echo ""
echo "  2. Open the memory console:"
echo "       haunt dash"
echo "       → http://127.0.0.1:7340"
echo ""
echo "  3. Install Cursor hooks (optional):"
echo "       haunt cursor-install"
echo ""
echo "  4. Add haunt as an MCP server (alongside any others you have):"
echo "       {\"mcpServers\":{\"haunt\":{\"command\":\"${HAUNT_HOME}/bin/haunt-mcp\"}}}"
echo ""
echo "     haunt-mcp is a stdio server — do not run it directly."
echo ""
echo "  Done. Run 'haunt --help' to see all commands."
