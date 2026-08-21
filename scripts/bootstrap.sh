#!/usr/bin/env bash
set -euo pipefail

# Idempotent bootstrap for haunt.
# Creates a venv, installs haunt, runs first-run setup.
# No Docker, no cloud, no root required.

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
haunt bootstrap

echo ""
echo "── next steps ──────────────────────────────────────────────────────"
echo ""
echo "  1. Activate the venv:"
echo "       source ${VENV_DIR}/bin/activate"
echo ""
echo "  2. Install Cursor hooks (optional):"
echo "       haunt cursor-install"
echo ""
echo "  3. Point your MCP client at the launcher:"
echo "       ~/.haunt/bin/haunt-mcp"
echo ""
echo "  Done. Run 'haunt --help' to see all commands."
