"""Shared dashboard TestClient: honest 127.0.0.1 + launch token."""

from __future__ import annotations

from starlette.testclient import TestClient

TEST_DASH_TOKEN = "test-haunt-dash-token"


def make_dash_client(
    app=None,
    *,
    token: str | None = TEST_DASH_TOKEN,
    host: str = "127.0.0.1",
):
    """TestClient that sends a trusted Host and the launch token by default."""
    from haunt.dashboard import app as dash_app

    headers = {"X-Haunt-Token": token} if token else {}
    return TestClient(app or dash_app, base_url=f"http://{host}:7340", headers=headers)
