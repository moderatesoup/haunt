"""The cached BGE-M3 bytes are checked against the committed manifest.

_download_bge_m3 pins repo and revision, but that pin only binds the download
path. A model dropped straight into the cache never passes through it, and the
haunt-model-source.json marker beside it is written by whoever wrote the
cache -- so the recorded identity attests nothing on its own. These tests hold
the load path to verifying the bytes before onnxruntime is handed the graph,
and to saying so out loud on the paths where it cannot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from haunt import abstention_eval, embed
from haunt.abstention_eval import canonical_hash, verify_local_hybrid_cache
from haunt.paths import models_dir

FIXTURE = Path(__file__).parent / "fixtures" / "abstention_eval" / "v1"


def _fake_cache(root: Path) -> Path:
    """Enough layout for _local_bge_m3_ready, with none of a real model."""
    onnx = root / embed.BGE_M3_DIRNAME / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "model.onnx").write_bytes(b"graph")
    (onnx / "model.onnx_data").write_bytes(b"weights")
    (onnx / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root / embed.BGE_M3_DIRNAME


def _fake_quant_cache(root: Path) -> Path:
    """The third-party quantized layout, which no committed manifest covers."""
    model = root / embed.BGE_M3_DIRNAME
    (model / "onnx").mkdir(parents=True)
    (model / "onnx" / "model_quantized.onnx").write_bytes(b"quantized-graph")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


def _must_not_run(message: str):
    def boom(*_args, **_kwargs):
        raise AssertionError(message)

    return boom


def _clean_env(monkeypatch, root: Path) -> None:
    monkeypatch.setenv("HAUNT_MODEL_CACHE", str(root))
    monkeypatch.delenv("HAUNT_EMBED_SKIP_MODEL_VERIFY", raising=False)
    monkeypatch.delenv("HAUNT_EMBED_QUANT_FALLBACK", raising=False)


def test_load_verifies_the_cache_before_building_the_session(tmp_path, monkeypatch):
    _clean_env(monkeypatch, tmp_path)
    _fake_cache(tmp_path)
    calls: list[tuple[Path, dict]] = []

    def refuse(cache_root, **kwargs):
        calls.append((cache_root, kwargs))
        raise RuntimeError("hybrid cache artifact hash mismatch: planted model")

    monkeypatch.setattr(abstention_eval, "verify_local_hybrid_cache", refuse)
    monkeypatch.setattr(
        embed,
        "OnnxEmbedder",
        _must_not_run("an unverified graph reached onnxruntime"),
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        embed._load_onnx_bge_m3()
    assert calls == [(models_dir(), {"hash_max_bytes": embed.VERIFY_HASH_MAX_BYTES})]


def test_a_verified_cache_loads(tmp_path, monkeypatch):
    _clean_env(monkeypatch, tmp_path)
    root = _fake_cache(tmp_path)
    monkeypatch.setattr(
        abstention_eval,
        "verify_local_hybrid_cache",
        lambda *_a, **_k: {"matched_manifest_id": "test", "files": []},
    )
    monkeypatch.setattr(embed, "OnnxEmbedder", lambda ready: f"model:{ready}")
    model, nbytes = embed._load_onnx_bge_m3()
    assert model == f"model:{root}"
    assert nbytes > 0


@pytest.mark.parametrize(
    ("flag", "cache"),
    [
        ("HAUNT_EMBED_SKIP_MODEL_VERIFY", _fake_cache),
        ("HAUNT_EMBED_QUANT_FALLBACK", _fake_quant_cache),
    ],
)
def test_skipping_verification_takes_an_env_opt_in_and_is_recorded(
    tmp_path, monkeypatch, capsys, flag, cache
):
    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv(flag, "1")
    root = cache(tmp_path)
    monkeypatch.setattr(
        abstention_eval,
        "verify_local_hybrid_cache",
        _must_not_run("verification ran under an explicit skip"),
    )
    embed._verify_bge_m3_cache(root)
    assert "embed_m3_unverified" in capsys.readouterr().err


def test_planted_quantized_files_do_not_skip_verification(tmp_path, monkeypatch):
    """The quantized layout is unmanifested, so it must not be self-certifying."""
    _clean_env(monkeypatch, tmp_path)
    root = _fake_quant_cache(tmp_path)
    calls: list[Path] = []

    def record(cache_root, **_kwargs):
        calls.append(cache_root)
        return {"matched_manifest_id": "test", "files": []}

    monkeypatch.setattr(abstention_eval, "verify_local_hybrid_cache", record)
    embed._verify_bge_m3_cache(root)
    assert calls == [models_dir()]


def test_an_uninstalled_manifest_is_reported_not_assumed(
    tmp_path, monkeypatch, capsys
):
    """The manifest ships with the repo, not the wheel; absence is not consent."""
    _clean_env(monkeypatch, tmp_path)
    root = _fake_cache(tmp_path)

    def absent(*_a, **_k):
        raise FileNotFoundError("no hybrid-model-manifest.json")

    monkeypatch.setattr(abstention_eval, "verify_local_hybrid_cache", absent)
    embed._verify_bge_m3_cache(root)
    err = capsys.readouterr().err
    assert "embed_m3_unverified" in err
    assert "manifest not installed" in err


def _synthetic_fixture(
    tmp_path: Path, cache: Path, contents: dict[str, bytes]
) -> tuple[Path, str]:
    """A manifest over throwaway files, keyed to the real one's structure."""
    manifest = json.loads((FIXTURE / "hybrid-model-manifest.json").read_text("utf-8"))
    manifest["files"] = [
        {
            "relative_path": relative,
            "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for relative, body in contents.items()
    ]
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "hybrid-model-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for relative, body in contents.items():
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return fixture, canonical_hash(manifest)


GRAPH = b"graph-bytes"
SIDECAR = b"external-tensor-bytes-past-the-cap"
CONTENTS = {
    "BAAI-bge-m3/onnx/model.onnx": GRAPH,
    "BAAI-bge-m3/onnx/model.onnx_data": SIDECAR,
}


@pytest.fixture()
def capped_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    fixture, digest = _synthetic_fixture(tmp_path, cache, CONTENTS)
    monkeypatch.setattr(abstention_eval, "HYBRID_ARTIFACT_MANIFEST_SHA256", digest)
    return cache, fixture


def test_hash_cap_checks_oversized_files_by_size_and_the_rest_by_content(
    capped_cache,
):
    cache, fixture = capped_cache
    report = verify_local_hybrid_cache(cache, fixture_dir=fixture, hash_max_bytes=16)
    assert report["files"] == [
        {
            "relative_path": "BAAI-bge-m3/onnx/model.onnx",
            "size": len(GRAPH),
            "sha256": hashlib.sha256(GRAPH).hexdigest(),
        },
        {
            "relative_path": "BAAI-bge-m3/onnx/model.onnx_data",
            "size": len(SIDECAR),
            "sha256": None,
        },
    ]


def test_hash_cap_still_catches_a_substituted_graph(capped_cache):
    cache, fixture = capped_cache
    (cache / "BAAI-bge-m3/onnx/model.onnx").write_bytes(b"other-graph")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        verify_local_hybrid_cache(cache, fixture_dir=fixture, hash_max_bytes=16)


def test_hash_cap_catches_a_resized_sidecar_but_not_a_restuffed_one(capped_cache):
    """The residual gap, stated as a test: size is all the cap can promise."""
    cache, fixture = capped_cache
    sidecar = cache / "BAAI-bge-m3/onnx/model.onnx_data"
    sidecar.write_bytes(b"X" * len(SIDECAR))
    verify_local_hybrid_cache(cache, fixture_dir=fixture, hash_max_bytes=16)

    sidecar.write_bytes(b"X" * (len(SIDECAR) + 1))
    with pytest.raises(RuntimeError, match="size_mismatch"):
        verify_local_hybrid_cache(cache, fixture_dir=fixture, hash_max_bytes=16)


def test_uncapped_verification_still_hashes_every_file(capped_cache):
    cache, fixture = capped_cache
    (cache / "BAAI-bge-m3/onnx/model.onnx_data").write_bytes(b"X" * len(SIDECAR))
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        verify_local_hybrid_cache(cache, fixture_dir=fixture)
