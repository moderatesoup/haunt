"""The BGE-M3 download is pinned, and the third-party fallback is opt-in.

_download_bge_m3 used to call snapshot_download with no revision= -- so it
fetched whatever the repo's main branch pointed at that day -- inside a bare
`except Exception` that redirected to a different publisher's repo on ANY
failure. A timeout or a 503 was enough to silently swap the ONNX graph that
onnxruntime then executes.
"""

from __future__ import annotations

import httpx
import pytest

from haunt import embed


@pytest.fixture
def model_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunthome"))
    monkeypatch.delenv("HAUNT_OFFLINE", raising=False)
    monkeypatch.delenv("HAUNT_EMBED_QUANT_FALLBACK", raising=False)
    return tmp_path / "BAAI-bge-m3"


def _fake_hub(monkeypatch, effect) -> list[dict]:
    """Replace snapshot_download, recording the kwargs of every call."""
    import huggingface_hub

    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        return effect(kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return calls


def _land_official(root) -> None:
    """The split (non-quantized) layout _find_onnx accepts via its sidecar."""
    (root / "onnx").mkdir(parents=True, exist_ok=True)
    (root / "onnx" / "model.onnx").write_bytes(b"graph")
    (root / "onnx" / "model.onnx_data").write_bytes(b"weights")
    (root / "onnx" / "tokenizer.json").write_text("{}")


def _land_quant(root) -> None:
    (root / "onnx").mkdir(parents=True, exist_ok=True)
    (root / "onnx" / "model_quantized.onnx").write_bytes(b"graph")
    (root / "tokenizer.json").write_text("{}")


def _repo_gone() -> Exception:
    """Hub's not-found errors take a required response= on current versions."""
    unavailable = embed._repo_unavailable_errors()
    assert unavailable, "huggingface_hub exposes no repo-unavailable errors"
    cls = unavailable[0]
    response = httpx.Response(
        404, request=httpx.Request("GET", "https://huggingface.co/BAAI/bge-m3")
    )
    try:
        return cls("repo gone", response=response)
    except TypeError:
        return cls("repo gone")


def test_official_download_is_revision_pinned_and_recorded(model_root, monkeypatch):
    calls = _fake_hub(monkeypatch, lambda _kw: _land_official(model_root))

    assert embed._download_bge_m3(model_root) == model_root
    assert [c["repo_id"] for c in calls] == [embed.BGE_M3_ID]
    assert calls[0]["revision"] == embed.BGE_M3_REVISION
    assert embed.bge_m3_source(model_root) == {
        "repo_id": embed.BGE_M3_ID,
        "revision": embed.BGE_M3_REVISION,
    }


@pytest.mark.parametrize(
    "exc",
    [TimeoutError("read timed out"), OSError("connection reset"), RuntimeError("boom")],
)
def test_transient_failure_never_redirects_to_another_publisher(
    model_root, monkeypatch, exc
):
    """Opt-in set, so only the error's kind can stop the redirect here."""
    monkeypatch.setenv("HAUNT_EMBED_QUANT_FALLBACK", "1")

    def boom(_kw):
        raise exc

    calls = _fake_hub(monkeypatch, boom)
    with pytest.raises(type(exc)):
        embed._download_bge_m3(model_root)
    assert [c["repo_id"] for c in calls] == [embed.BGE_M3_ID]


def test_repo_gone_without_opt_in_does_not_switch_publishers(model_root, monkeypatch):
    gone = _repo_gone()

    def boom(_kw):
        raise gone

    calls = _fake_hub(monkeypatch, boom)
    with pytest.raises(type(gone)):
        embed._download_bge_m3(model_root)
    assert [c["repo_id"] for c in calls] == [embed.BGE_M3_ID]
    assert embed.bge_m3_source(model_root) is None


def test_opt_in_fallback_is_pinned_and_names_the_repo_it_used(model_root, monkeypatch):
    monkeypatch.setenv("HAUNT_EMBED_QUANT_FALLBACK", "1")
    gone = _repo_gone()

    def effect(kwargs):
        if kwargs["repo_id"] == embed.BGE_M3_ID:
            raise gone
        _land_quant(model_root)

    calls = _fake_hub(monkeypatch, effect)

    assert embed._download_bge_m3(model_root) == model_root
    assert [c["repo_id"] for c in calls] == [embed.BGE_M3_ID, embed.BGE_M3_QUANT_REPO]
    assert calls[1]["revision"] == embed.BGE_M3_QUANT_REVISION
    assert embed.bge_m3_source(model_root) == {
        "repo_id": embed.BGE_M3_QUANT_REPO,
        "revision": embed.BGE_M3_QUANT_REVISION,
    }


def test_an_incomplete_official_snapshot_is_not_a_silent_redirect(
    model_root, monkeypatch
):
    """snapshot_download returning with nothing usable is still the official
    repo failing -- it must raise unless the redirect was asked for.
    """
    calls = _fake_hub(monkeypatch, lambda _kw: None)
    with pytest.raises(RuntimeError, match="missing after download"):
        embed._download_bge_m3(model_root)
    assert [c["repo_id"] for c in calls] == [embed.BGE_M3_ID]
